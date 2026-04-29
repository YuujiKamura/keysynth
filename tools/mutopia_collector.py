#!/usr/bin/env python3
"""mutopia_collector.py — Stage B Agent Collector v0.

Scrapes the Mutopia Project catalog for guitar + piano pieces under
permissive licenses, downloads selected MIDI files into
``bench-out/songs/``, and appends entries to ``manifest.json`` so
``keysynth_db rebuild`` re-imports the catalog.

Layer 1 of the data-collection plan: AI agent (this script + the
operator running it) does the legwork that a human did before.

License policy
--------------
Mutopia anchors used:

  publicdomain   → Public Domain                 — accepted
  cca            → CC-BY 4.0                     — accepted
  ccasa          → CC-BY-SA 4.0                  — REJECTED (copyleft)
  ccbync*        → CC-BY-NC* (any non-commercial) — REJECTED

BSD-3 isn't used by Mutopia for music; the allowlist is
``{publicdomain, cca}``.

Idempotence
-----------
Re-running the script:

* skips entries whose stable id (file stem of the .mid URL) is already
  in ``manifest.json`` (no duplicate ids, never overwrites a downloaded
  .mid),
* re-uses cached HTML listings under ``bench-out/.mutopia-cache/``,
* leaves the manifest untouched if no new entries clear all filters.

Usage
-----
::

    python tools/mutopia_collector.py \\
        --target-guitar 25 --target-piano 25 \\
        [--manifest bench-out/songs/manifest.json] \\
        [--out-dir bench-out/songs] \\
        [--cache-dir bench-out/.mutopia-cache] \\
        [--dry-run]

Stdlib only — no third-party deps.
"""

from __future__ import annotations

import argparse
import dataclasses
import html
import html.parser
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

MUTOPIA_BASE = "https://www.mutopiaproject.org"
LICENSE_ALLOW = {"publicdomain", "cca"}
PAGE_SIZE = 10
USER_AGENT = "keysynth-mutopia-collector/0.1 (+https://github.com/YuujiKamura/keysynth)"


# ─── Data types ────────────────────────────────────────────────────────


@dataclasses.dataclass
class CatalogEntry:
    """One parsed result-table block from the Mutopia listing."""

    title: str
    composer_raw: str       # "by D. Aguado (1784–1849)"
    opus: str               # "Op. 3 No. 1" (often blank)
    instrument: str         # "Guitar" / "Piano"
    year: str               # "1830" (composition year)
    era: str                # "Romantic" / "Baroque" / ...
    license_anchor: str     # "publicdomain" / "cca" / "ccasa" / ...
    license_name: str       # "Public Domain" / "Creative Commons Attribution 4.0"
    mid_url: Optional[str]  # absolute https URL or None if multi-mid
    info_url: Optional[str]

    @property
    def file_stem(self) -> Optional[str]:
        if not self.mid_url:
            return None
        tail = self.mid_url.rsplit("/", 1)[-1]
        return re.sub(r"\.mid$", "", tail)

    @property
    def composer_name_with_years(self) -> str:
        """Strip the leading 'by ' and replace en-dashes with ASCII hyphens
        so derive_era()'s digit-extraction lands on a clean range."""
        raw = self.composer_raw
        if raw.lower().startswith("by "):
            raw = raw[3:]
        # Mutopia uses en-dash (U+2013) between birth/death years.
        # derive_era() splits on non-digit chars so en-dash works, but
        # ASCII hyphen normalises the manifest for grep-friendliness.
        return raw.replace("–", "-").replace("—", "-").strip()


# ─── HTML parsing ──────────────────────────────────────────────────────


class _ResultTableParser(html.parser.HTMLParser):
    """Walks the catalog page and emits one CatalogEntry per
    ``<table class="...result-table">`` block. Tracks raw cell text plus
    every link inside the block; ``finalise_block`` reconstitutes the
    fields from those primitives."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.entries: list[CatalogEntry] = []
        # current state ----------------------------------------------------
        self._in_block = False
        self._in_cell = False
        self._cell_text: list[str] = []
        self._cells: list[str] = []
        self._block_links: list[tuple[str, str]] = []  # (href, text)
        self._link_href: Optional[str] = None
        self._link_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag == "table":
            cls = dict(attrs).get("class", "") or ""
            if "result-table" in cls:
                self._open_block()
                return
        if not self._in_block:
            return
        if tag == "td":
            self._in_cell = True
            self._cell_text = []
        elif tag == "a":
            self._link_href = dict(attrs).get("href")
            self._link_text = []

    def handle_endtag(self, tag: str) -> None:
        if not self._in_block:
            return
        if tag == "td" and self._in_cell:
            self._in_cell = False
            self._cells.append("".join(self._cell_text).strip())
        elif tag == "a" and self._link_href is not None:
            self._block_links.append((self._link_href, "".join(self._link_text).strip()))
            self._link_href = None
            self._link_text = []
        elif tag == "table":
            self._close_block()

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_text.append(data)
        if self._in_block and self._link_href is not None:
            self._link_text.append(data)

    # internal -----------------------------------------------------------

    def _open_block(self) -> None:
        self._in_block = True
        self._cells = []
        self._block_links = []

    def _close_block(self) -> None:
        if self._cells:
            entry = self._build_entry()
            if entry is not None:
                self.entries.append(entry)
        self._in_block = False
        self._cells = []
        self._block_links = []

    def _build_entry(self) -> Optional[CatalogEntry]:
        # Expected layout (mostly):
        #   cells[0] = title
        #   cells[1] = "by Composer (years)"
        #   cells[2] = opus
        #   cells[3] = filler
        #   cells[4] = "for Instrument"
        #   cells[5] = year
        #   cells[6] = era
        #   cells[7] = filler
        #   cells[8] = publisher
        #   cells[9] = license link text
        #   cells[10] = "More Information" link text
        #   cells[11] = entry-date
        #   cells[12+] = download row(s)
        if len(self._cells) < 12:
            return None
        title = self._cells[0]
        composer = self._cells[1]
        opus = self._cells[2]
        instrument = re.sub(r"^for ", "", self._cells[4]).strip()
        year = self._cells[5]
        era = self._cells[6]
        license_name = self._cells[9]

        license_anchor = ""
        info_url: Optional[str] = None
        for href, _text in self._block_links:
            if href and "legal.html#" in href:
                license_anchor = href.split("#", 1)[1].lower()
            elif href and href.startswith("piece-info.cgi"):
                info_url = MUTOPIA_BASE + "/cgibin/" + href

        # Pick the .mid link. Skip multi-mid zip entries (only zipped form).
        mid_candidates = [
            href
            for href, _text in self._block_links
            if href and href.lower().endswith(".mid")
        ]
        mid_url = mid_candidates[0] if len(mid_candidates) == 1 else None

        if not (title and composer and instrument):
            return None
        return CatalogEntry(
            title=title,
            composer_raw=composer,
            opus=opus,
            instrument=instrument,
            year=year,
            era=era,
            license_anchor=license_anchor,
            license_name=license_name,
            mid_url=mid_url,
            info_url=info_url,
        )


def parse_catalog_html(text: str) -> list[CatalogEntry]:
    p = _ResultTableParser()
    p.feed(text)
    p.close()
    return p.entries


def has_next_page(text: str) -> bool:
    return 'startat=' in text and 'Next 10' in text


# ─── Catalog crawl ─────────────────────────────────────────────────────


def fetch_url(url: str, *, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — public site, https only
        return resp.read()


def cached_fetch_text(url: str, cache_path: Path) -> str:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_text(encoding="utf-8", errors="replace")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    body = fetch_url(url)
    text = body.decode("utf-8", errors="replace")
    cache_path.write_text(text, encoding="utf-8")
    return text


def crawl_instrument(
    instrument: str,
    cache_dir: Path,
    *,
    max_pages: int = 200,
    polite_delay: float = 0.4,
) -> list[CatalogEntry]:
    """Walk all `make-table.cgi?Instrument=<X>&startat=<n>` pages until
    the listing runs out, returning every parsed entry. Pages are cached
    under ``<cache_dir>/<instrument>-pNN.html`` so repeated runs don't
    hammer the server."""
    out: list[CatalogEntry] = []
    seen_first_entry: Optional[str] = None
    for page in range(max_pages):
        startat = page * PAGE_SIZE
        url = (
            f"{MUTOPIA_BASE}/cgibin/make-table.cgi"
            f"?Instrument={instrument}&startat={startat}"
        )
        cache_path = cache_dir / f"{instrument.lower()}-p{page:03d}.html"
        try:
            text = cached_fetch_text(url, cache_path)
        except urllib.error.URLError as e:
            print(f"  warn: page {page} fetch failed: {e}", file=sys.stderr)
            break
        entries = parse_catalog_html(text)
        if not entries:
            break
        # Mutopia returns the same first page if startat goes past the
        # end. Detect by comparing the first entry on this page to the
        # first entry on page 0.
        first = entries[0].title + "|" + entries[0].composer_raw
        if page == 0:
            seen_first_entry = first
        elif first == seen_first_entry:
            break
        out.extend(entries)
        if not has_next_page(text):
            break
        if polite_delay and not cache_path.stat().st_size == 0:
            # We just hit the network (or read from cache) — small sleep
            # only when we actually fetched live; the cache hit case
            # touches no network so polite_delay is harmless either way.
            time.sleep(polite_delay)
    return out


# ─── Selection / scoring ────────────────────────────────────────────────


def is_acceptable(
    entry: CatalogEntry,
    existing_ids: set[str],
    existing_urls: set[str],
) -> bool:
    if entry.license_anchor not in LICENSE_ALLOW:
        return False
    if not entry.mid_url:
        return False
    if entry.mid_url in existing_urls:
        return False
    stem = entry.file_stem
    if stem is None or stem in existing_ids:
        return False
    if not entry.composer_raw:
        return False
    return True


def composer_surname_key(composer_raw: str) -> str:
    """Crude surname key for diversification — lowercase, strip leading
    'by ', drop the parenthetical year range, take last whitespace
    token."""
    raw = composer_raw
    if raw.lower().startswith("by "):
        raw = raw[3:]
    raw = raw.split("(", 1)[0].strip()
    if not raw:
        return ""
    parts = re.split(r"[\s.]+", raw)
    parts = [p for p in parts if p]
    if not parts:
        return ""
    return parts[-1].lower()


def diversify(entries: Iterable[CatalogEntry], target: int) -> list[CatalogEntry]:
    """Round-robin by composer surname so the catalog spreads across
    composers rather than dumping 25 Aguado études in a row."""
    by_composer: dict[str, list[CatalogEntry]] = {}
    order: list[str] = []
    for e in entries:
        key = composer_surname_key(e.composer_raw)
        if key not in by_composer:
            by_composer[key] = []
            order.append(key)
        by_composer[key].append(e)

    picked: list[CatalogEntry] = []
    rounds = 0
    while len(picked) < target:
        progress = False
        for key in order:
            bucket = by_composer.get(key)
            if not bucket:
                continue
            picked.append(bucket.pop(0))
            progress = True
            if len(picked) >= target:
                break
        rounds += 1
        if not progress or rounds > 10_000:
            break
    return picked


# ─── Manifest writing ──────────────────────────────────────────────────


def suggested_voice_for(instrument: str) -> Optional[str]:
    inst = instrument.lower()
    if "guitar" in inst:
        return "guitar-stk"
    if "piano" in inst:
        return "piano-modal"
    return None


def manifest_instrument(instrument: str) -> str:
    inst = instrument.lower()
    if "guitar" in inst:
        return "guitar"
    if "piano" in inst:
        return "piano"
    return inst


def derive_tags(entry: CatalogEntry) -> list[str]:
    tags: list[str] = []
    era = (entry.era or "").lower()
    if era:
        tags.append(era)
    inst = manifest_instrument(entry.instrument)
    if inst:
        tags.append(inst)
    title = (entry.title or "").lower()
    for pattern, tag in [
        (r"\betude|\bétude|\bestudio|study", "etude"),
        (r"prelude|prélude", "prelude"),
        (r"sonata|sonate", "sonata"),
        (r"fugue|fuga", "fugue"),
        (r"invention", "invention"),
        (r"nocturne", "nocturne"),
        (r"mazurka", "mazurka"),
        (r"waltz|valse", "waltz"),
        (r"march", "march"),
        (r"minuet|menuet", "minuet"),
        (r"gavotte", "gavotte"),
        (r"sarabande", "sarabande"),
        (r"gigue", "gigue"),
        (r"allemande", "allemande"),
        (r"chaconne", "chaconne"),
        (r"variation", "variations"),
        (r"impromptu", "impromptu"),
        (r"capriccio|caprice", "caprice"),
        (r"fantasy|fantasia|fantaisie", "fantasy"),
        (r"ragtime|rag\b", "ragtime"),
    ]:
        if re.search(pattern, title):
            tags.append(tag)
    if entry.license_anchor == "publicdomain":
        tags.append("license:public-domain")
    elif entry.license_anchor == "cca":
        tags.append("license:cc-by")
    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def license_label(anchor: str, raw: str) -> str:
    if anchor == "publicdomain":
        return "Public Domain"
    if anchor == "cca":
        return "CC-BY 4.0"
    return raw or anchor


def make_context(entry: CatalogEntry) -> str:
    inst = manifest_instrument(entry.instrument)
    era = (entry.era or "").strip() or "?"
    surname = composer_surname_key(entry.composer_raw).capitalize() or "?"
    title = entry.title.strip()
    voice = suggested_voice_for(entry.instrument) or "?"
    bits: list[str] = []
    bits.append(f"{era}-period {surname} {inst} piece")
    if entry.opus and entry.opus.strip():
        bits.append(f"({entry.opus.strip()})")
    if entry.year and entry.year.strip():
        bits.append(f"composed {entry.year.strip()}")
    bits.append(f"— exercises the {voice} voice on real repertoire from the Mutopia catalog.")
    return " ".join(bits).replace("  ", " ").strip()


def make_manifest_entry(entry: CatalogEntry) -> dict:
    stem = entry.file_stem
    assert stem is not None
    return {
        "file": f"{stem}.mid",
        "title": entry.title.strip(),
        "composer": entry.composer_name_with_years,
        "instrument": manifest_instrument(entry.instrument),
        "license": license_label(entry.license_anchor, entry.license_name),
        "source": "Mutopia Project",
        "source_url": entry.mid_url,
        "suggested_voice": suggested_voice_for(entry.instrument),
        "context": make_context(entry),
        "tags": derive_tags(entry),
    }


# ─── Manifest IO ───────────────────────────────────────────────────────


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, data: dict) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def existing_ids(manifest: dict) -> set[str]:
    out: set[str] = set()
    for e in manifest.get("entries", []):
        f = e.get("file")
        if not f:
            continue
        out.add(re.sub(r"\.mid$", "", f))
    return out


def existing_urls(manifest: dict) -> set[str]:
    out: set[str] = set()
    for e in manifest.get("entries", []):
        u = e.get("source_url")
        if u:
            out.add(u)
    return out


# ─── Download ──────────────────────────────────────────────────────────


def download_mid(url: str, dest: Path, *, polite_delay: float = 0.3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        body = fetch_url(url)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"  fail: {url} → {e}", file=sys.stderr)
        return False
    if not body or len(body) < 8:
        print(f"  fail: {url} → empty body", file=sys.stderr)
        return False
    if body[:4] != b"MThd":
        print(f"  fail: {url} → not an SMF (no MThd header)", file=sys.stderr)
        return False
    dest.write_bytes(body)
    if polite_delay:
        time.sleep(polite_delay)
    return True


# ─── Orchestration ─────────────────────────────────────────────────────


def collect(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)

    manifest = load_manifest(manifest_path)
    taken_ids = existing_ids(manifest)
    taken_urls = existing_urls(manifest)
    print(f"manifest: {len(manifest.get('entries', []))} existing entries", file=sys.stderr)

    targets: list[tuple[str, int]] = []
    if args.target_guitar > 0:
        targets.append(("Guitar", args.target_guitar))
    if args.target_piano > 0:
        targets.append(("Piano", args.target_piano))
    if not targets:
        print("nothing to do (target counts are zero)", file=sys.stderr)
        return 0

    new_entries: list[dict] = []

    for instrument, target in targets:
        print(f"crawl: {instrument} (target {target})", file=sys.stderr)
        all_entries = crawl_instrument(instrument, cache_dir)
        print(f"  parsed {len(all_entries)} catalog entries", file=sys.stderr)
        eligible = [
            e for e in all_entries if is_acceptable(e, taken_ids, taken_urls)
        ]
        print(f"  eligible after license/license-collision filter: {len(eligible)}",
              file=sys.stderr)
        chosen = diversify(eligible, target)
        if len(chosen) < target:
            print(
                f"  warn: only {len(chosen)} pieces clear filter for {instrument}",
                file=sys.stderr,
            )
        for entry in chosen:
            stem = entry.file_stem
            assert stem is not None
            dest = out_dir / f"{stem}.mid"
            if args.dry_run:
                print(f"  dry: would fetch {entry.mid_url} → {dest}", file=sys.stderr)
                taken_ids.add(stem)
                taken_urls.add(entry.mid_url or "")
                new_entries.append(make_manifest_entry(entry))
                continue
            ok = download_mid(entry.mid_url or "", dest)
            if not ok:
                continue
            taken_ids.add(stem)
            taken_urls.add(entry.mid_url or "")
            new_entries.append(make_manifest_entry(entry))
            print(f"  ok:  {stem}.mid", file=sys.stderr)

    if not new_entries:
        print("no new entries collected", file=sys.stderr)
        return 0

    manifest.setdefault("entries", []).extend(new_entries)
    manifest["acquisition_date"] = time.strftime("%Y-%m-%d")
    if args.dry_run:
        print(f"dry-run: would add {len(new_entries)} entries to {manifest_path}",
              file=sys.stderr)
    else:
        save_manifest(manifest_path, manifest)
        print(f"manifest: +{len(new_entries)} entries → {manifest_path}",
              file=sys.stderr)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--manifest", default="bench-out/songs/manifest.json")
    p.add_argument("--out-dir", default="bench-out/songs")
    p.add_argument("--cache-dir", default="bench-out/.mutopia-cache")
    p.add_argument("--target-guitar", type=int, default=25)
    p.add_argument("--target-piano", type=int, default=25)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args(sys.argv[1:])
    try:
        return collect(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
