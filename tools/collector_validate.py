#!/usr/bin/env python3
"""Stage C — post-collection hardening for the song catalog.

Reads a Mutopia/etc song-catalog manifest (default
``bench-out/songs/manifest.json`` — the schema produced by
``tools/mutopia_collector.py`` and future siblings under
``tools/imslp_collector.py`` / ``tools/wikifonia_collector.py``)
and applies four classes of post-collection checks before the
catalog can ship to ``main``:

  1. License whitelist enforcement.
     Strict three-way whitelist: Public Domain / CC0 / CC-BY only.
     CC-BY-SA, CC-BY-NC*, GPL*, copyleft, and unrecognized strings
     are rejected. Mutopia accepts CC-BY-SA upstream; we are stricter
     than upstream because audio renders of CC-BY-SA scores would
     virally relicense bench-out artifacts.

  2. ID-collision detection.
     The catalog's primary key is the ``file`` field (it doubles as
     the on-disk basename). Two entries colliding on ``file`` would
     clobber each other on disk. We detect duplicates and propose
     ``<stem>-2.mid`` / ``<stem>-3.mid`` ... renames.

  3. Playable-MIDI smoke check (--render-check).
     Per surviving entry, invoke ``cargo run --release --bin
     render_midi -- --in <mid> --engine square --out <tmp>.wav``
     under a per-entry timeout. Any non-zero exit / timeout / empty
     output WAV rejects the entry as unplayable. Cargo build cache
     is reused across entries so only the first invocation pays the
     compile tax (we pre-build once before the loop).

  4. Failure log dump.
     Writes ``tools/.collector-failures.json`` with one record per
     rejected entry plus a timestamped run summary so daily cron
     runs can be diffed.

Usage::

    python tools/collector_validate.py
    python tools/collector_validate.py --manifest bench-out/songs/manifest.json
    python tools/collector_validate.py --render-check
    python tools/collector_validate.py --render-check --apply
    python tools/collector_validate.py --self-test

Exit codes:
    0  — all entries passed (or self-test ok)
    1  — one or more entries rejected (failures dumped)
    2  — usage / IO error before validation could run
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "bench-out" / "songs" / "manifest.json"
FAILURES_LOG = REPO_ROOT / "tools" / ".collector-failures.json"
VALIDATED_LOG = REPO_ROOT / "tools" / ".collector-validated.json"

# Per-entry render timeout (seconds). 90 s comfortably covers a 3-minute
# Mutopia piece on a release build; anything slower is either a stuck
# voice or a malformed SMF and deserves rejection.
RENDER_TIMEOUT_S = 90
# Initial cargo build timeout — cold cache compile of render_midi can
# legitimately take a few minutes on CI.
BUILD_TIMEOUT_S = 600

LICENSE_WHITELIST_LABELS = {"Public Domain", "CC0", "CC-BY"}


# ---------------------------------------------------------------------------
# License classifier
# ---------------------------------------------------------------------------

# Tokens that, if present after [\s_-] strip + uppercase, immediately
# disqualify the entry. Order matters only for the reported label —
# the *match* is decisive regardless of which token fired first.
_FORBIDDEN_TOKENS: Tuple[Tuple[str, str], ...] = (
    ("NONCOMMERCIAL", "CC-BY-NC"),
    ("BYNC", "CC-BY-NC"),
    ("CCNC", "CC-BY-NC"),
    ("BYSA", "CC-BY-SA"),
    ("SHAREALIKE", "CC-BY-SA"),
    ("CCSA", "CC-BY-SA"),
    ("AGPL", "AGPL"),
    ("LGPL", "LGPL"),
    ("GPL", "GPL"),
    ("COPYLEFT", "copyleft"),
    ("PROPRIETARY", "proprietary"),
    ("ALLRIGHTSRESERVED", "all-rights-reserved"),
)


def classify_license(raw: str) -> Tuple[bool, str]:
    """Return ``(accepted, label_or_reason)``.

    Accepted labels are drawn from ``LICENSE_WHITELIST_LABELS``; reasons
    on rejection are short human-readable strings used in the failure
    log.
    """
    if not raw or not raw.strip():
        return False, "empty license"
    s = raw.strip()
    norm = re.sub(r"[\s_\-./]", "", s).upper()

    for tok, label in _FORBIDDEN_TOKENS:
        if tok in norm:
            return False, f"forbidden:{label}"

    # Public Domain family (PD, PDM, "Public Domain", CC0).
    if "CC0" in norm:
        return True, "CC0"
    if norm in {"PD", "PDM"} or "PUBLICDOMAIN" in norm:
        return True, "Public Domain"
    # Mutopia sometimes records "Creative Commons Public Domain Mark".
    if "PUBLICDOMAINMARK" in norm:
        return True, "Public Domain"

    # CC-BY family — we already excluded SA/NC variants above, so any
    # remaining "CC" + "BY" or "CREATIVECOMMONSATTRIBUTION" combo is
    # vanilla attribution.
    if "CCBY" in norm:
        return True, "CC-BY"
    if "CREATIVECOMMONSATTRIBUTION" in norm:
        return True, "CC-BY"

    return False, f"unrecognized:{s!r}"


# ---------------------------------------------------------------------------
# Manifest model
# ---------------------------------------------------------------------------


@dataclass
class Entry:
    """One catalog entry. Mirrors the fields produced by Stage B but
    only the keys we validate are typed; the rest is preserved
    verbatim for round-tripping."""

    file: str
    license: str
    raw: dict
    source_url: str = ""
    title: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Entry":
        return cls(
            file=str(d.get("file", "")),
            license=str(d.get("license", "")),
            raw=d,
            source_url=str(d.get("source_url", "")),
            title=str(d.get("title", "")),
        )


@dataclass
class Failure:
    file: str
    reason: str
    detail: str = ""
    source_url: str = ""
    title: str = ""
    suggested_rename: str = ""

    def as_dict(self) -> dict:
        out = {"file": self.file, "reason": self.reason}
        if self.detail:
            out["detail"] = self.detail
        if self.source_url:
            out["source_url"] = self.source_url
        if self.title:
            out["title"] = self.title
        if self.suggested_rename:
            out["suggested_rename"] = self.suggested_rename
        return out


@dataclass
class Report:
    accepted: List[Entry] = field(default_factory=list)
    rejected: List[Failure] = field(default_factory=list)
    rename_plan: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rejected


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _check_license(entry: Entry) -> Optional[Failure]:
    accepted, label = classify_license(entry.license)
    if accepted:
        return None
    return Failure(
        file=entry.file,
        reason="license-rejected",
        detail=f"declared={entry.license!r} classified={label}",
        source_url=entry.source_url,
        title=entry.title,
    )


def _detect_collisions(entries: Iterable[Entry]) -> Tuple[List[Entry], List[Failure], List[Tuple[str, str]]]:
    """Return (kept, dupe_failures, rename_plan).

    First occurrence of each ``file`` is kept; subsequent occurrences
    become dupe failures with a suggested ``<stem>-N.mid`` rename. The
    rename plan is consumed by ``--apply``.
    """
    seen: dict = {}
    kept: List[Entry] = []
    failures: List[Failure] = []
    rename_plan: List[Tuple[str, str]] = []
    for e in entries:
        if e.file in seen:
            seen[e.file] += 1
            n = seen[e.file]
            stem, dot, ext = e.file.rpartition(".")
            suggested = f"{stem}-{n}.{ext}" if dot else f"{e.file}-{n}"
            failures.append(
                Failure(
                    file=e.file,
                    reason="id-collision",
                    detail=f"duplicate of earlier entry; rename to {suggested}",
                    source_url=e.source_url,
                    title=e.title,
                    suggested_rename=suggested,
                )
            )
            rename_plan.append((e.file, suggested))
        else:
            seen[e.file] = 1
            kept.append(e)
    return kept, failures, rename_plan


def _check_midi_exists(entry: Entry, songs_dir: Path) -> Optional[Failure]:
    p = songs_dir / entry.file
    if not p.is_file():
        return Failure(
            file=entry.file,
            reason="missing-file",
            detail=f"expected {p} not found on disk",
            source_url=entry.source_url,
            title=entry.title,
        )
    if p.stat().st_size < 64:
        return Failure(
            file=entry.file,
            reason="empty-file",
            detail=f"file present but only {p.stat().st_size} bytes",
            source_url=entry.source_url,
            title=entry.title,
        )
    return None


# ---------------------------------------------------------------------------
# Render check (Rust)
# ---------------------------------------------------------------------------


def _ensure_cargo_built() -> Tuple[bool, str]:
    """Pre-build render_midi once so per-entry runs are fast.

    Returns (ok, detail). On non-cargo systems we surface a clean
    skip rather than a hard fail — the caller decides whether the
    --render-check flag was load-bearing.
    """
    if shutil.which("cargo") is None:
        return False, "cargo not on PATH"
    try:
        proc = subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--bin",
                "render_midi",
                "--features",
                "native",
            ],
            cwd=REPO_ROOT,
            timeout=BUILD_TIMEOUT_S,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return False, f"cargo build timed out after {BUILD_TIMEOUT_S}s"
    except OSError as e:
        return False, f"cargo build OSError: {e}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1000:]
        return False, f"cargo build exit={proc.returncode}: {tail.strip()}"
    return True, "ok"


def _render_check(entry: Entry, songs_dir: Path) -> Optional[Failure]:
    """Run a short render through render_midi; reject on any failure mode."""
    src = songs_dir / entry.file
    with tempfile.TemporaryDirectory(prefix="ks-collector-") as td:
        out_wav = Path(td) / "probe.wav"
        try:
            proc = subprocess.run(
                [
                    "cargo",
                    "run",
                    "--release",
                    "--bin",
                    "render_midi",
                    "--features",
                    "native",
                    "--quiet",
                    "--",
                    "--in",
                    str(src),
                    "--engine",
                    "square",
                    "--out",
                    str(out_wav),
                ],
                cwd=REPO_ROOT,
                timeout=RENDER_TIMEOUT_S,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return Failure(
                file=entry.file,
                reason="render-timeout",
                detail=f"render_midi exceeded {RENDER_TIMEOUT_S}s",
                source_url=entry.source_url,
                title=entry.title,
            )
        except OSError as e:
            return Failure(
                file=entry.file,
                reason="render-spawn-error",
                detail=f"OSError: {e}",
                source_url=entry.source_url,
                title=entry.title,
            )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-400:]
            return Failure(
                file=entry.file,
                reason="render-failed",
                detail=f"exit={proc.returncode}: {tail.strip()}",
                source_url=entry.source_url,
                title=entry.title,
            )
        if not out_wav.exists() or out_wav.stat().st_size < 1024:
            return Failure(
                file=entry.file,
                reason="render-empty",
                detail=f"output WAV missing or <1KiB ({out_wav})",
                source_url=entry.source_url,
                title=entry.title,
            )
    return None


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def validate_manifest(
    manifest_path: Path,
    *,
    render_check: bool = False,
    songs_dir_override: Optional[Path] = None,
) -> Report:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    raw_entries = manifest.get("entries") or []
    if not isinstance(raw_entries, list):
        raise ValueError(f"manifest.entries must be a list, got {type(raw_entries).__name__}")
    songs_dir = (
        songs_dir_override
        if songs_dir_override is not None
        else (REPO_ROOT / manifest.get("directory", "bench-out/songs"))
    )

    entries = [Entry.from_dict(e) for e in raw_entries]
    report = Report()

    kept, dupes, rename_plan = _detect_collisions(entries)
    report.rejected.extend(dupes)
    report.rename_plan.extend(rename_plan)

    licensed: List[Entry] = []
    for e in kept:
        f = _check_license(e)
        if f is not None:
            report.rejected.append(f)
            continue
        licensed.append(e)

    on_disk: List[Entry] = []
    for e in licensed:
        f = _check_midi_exists(e, songs_dir)
        if f is not None:
            report.rejected.append(f)
            continue
        on_disk.append(e)

    if render_check and on_disk:
        ok, detail = _ensure_cargo_built()
        if not ok:
            for e in on_disk:
                report.rejected.append(
                    Failure(
                        file=e.file,
                        reason="render-precondition",
                        detail=detail,
                        source_url=e.source_url,
                        title=e.title,
                    )
                )
            return report
        playable: List[Entry] = []
        for e in on_disk:
            f = _render_check(e, songs_dir)
            if f is not None:
                report.rejected.append(f)
                continue
            playable.append(e)
        report.accepted.extend(playable)
    else:
        report.accepted.extend(on_disk)

    return report


def dump_artifacts(
    report: Report,
    manifest_path: Path,
    *,
    failures_path: Path = FAILURES_LOG,
    validated_path: Path = VALIDATED_LOG,
) -> None:
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "manifest": (
            manifest_path.relative_to(REPO_ROOT).as_posix()
            if manifest_path.is_relative_to(REPO_ROOT)
            else manifest_path.as_posix()
        ),
        "run_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds"),
        "summary": {
            "accepted": len(report.accepted),
            "rejected": len(report.rejected),
            "renames_proposed": len(report.rename_plan),
        },
        "rejected": [f.as_dict() for f in report.rejected],
        "rename_plan": [
            {"from": a, "to": b} for a, b in report.rename_plan
        ],
    }
    with failures_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    accepted_payload = {
        "schema_version": 1,
        "manifest": payload["manifest"],
        "run_at_utc": payload["run_at_utc"],
        "accepted_files": [e.file for e in report.accepted],
    }
    with validated_path.open("w", encoding="utf-8") as f:
        json.dump(accepted_payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def apply_in_place(report: Report, manifest_path: Path) -> None:
    """Rewrite the manifest to drop rejected entries and apply renames.

    On-disk MIDI files are renamed alongside the manifest update so the
    catalog stays internally consistent. Backed up to
    ``<manifest>.bak`` before mutation.
    """
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    raw_entries = manifest.get("entries") or []
    rejected_files = {fail.file for fail in report.rejected if fail.reason != "id-collision"}
    rename_map = dict(report.rename_plan)

    songs_dir = REPO_ROOT / manifest.get("directory", "bench-out/songs")
    new_entries = []
    seen_files = set()
    for raw in raw_entries:
        f = str(raw.get("file", ""))
        if f in rejected_files:
            continue
        if f in seen_files and f in rename_map:
            new_name = rename_map[f]
            old_p = songs_dir / f
            new_p = songs_dir / new_name
            if old_p.is_file() and not new_p.exists():
                old_p.rename(new_p)
            raw = dict(raw)
            raw["file"] = new_name
            f = new_name
        seen_files.add(f)
        new_entries.append(raw)
    manifest["entries"] = new_entries

    backup = manifest_path.with_suffix(manifest_path.suffix + ".bak")
    shutil.copy2(manifest_path, backup)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _run_self_test() -> int:
    """Validate the classifier + collision detector against fixtures.

    Does not invoke cargo; the render-check path is exercised by the
    daily cron job, not by `--self-test`.
    """
    cases = [
        ("Public Domain", True, "Public Domain"),
        ("PD", True, "Public Domain"),
        ("CC0", True, "CC0"),
        ("CC0 1.0 Universal", True, "CC0"),
        ("CC-BY 4.0", True, "CC-BY"),
        ("Creative Commons Attribution 3.0", True, "CC-BY"),
        ("CC-BY-SA 3.0", False, "forbidden:CC-BY-SA"),
        ("CC-BY-SA", False, "forbidden:CC-BY-SA"),
        ("CC-BY-NC", False, "forbidden:CC-BY-NC"),
        ("CC-BY-NC-SA 4.0", False, "forbidden:CC-BY-NC"),
        ("GPL-3.0", False, "forbidden:GPL"),
        ("LGPL", False, "forbidden:LGPL"),
        ("Proprietary", False, "forbidden:proprietary"),
        ("All Rights Reserved", False, "forbidden:all-rights-reserved"),
        ("", False, "empty license"),
        ("Whatever-License-Foo", False, ""),  # last: only check rejection, not exact label
    ]
    failed = 0
    for raw, want_ok, want_label in cases:
        got_ok, got_label = classify_license(raw)
        if got_ok != want_ok:
            print(f"FAIL classify_license({raw!r}): want ok={want_ok} got ok={got_ok} label={got_label!r}", file=sys.stderr)
            failed += 1
        elif want_label and got_ok and got_label != want_label:
            print(f"FAIL classify_license({raw!r}): want label={want_label!r} got {got_label!r}", file=sys.stderr)
            failed += 1
        elif want_label and not got_ok and not got_label.startswith(want_label.split(":")[0]):
            print(f"FAIL classify_license({raw!r}): want reason starting {want_label!r} got {got_label!r}", file=sys.stderr)
            failed += 1

    fixture = [
        Entry.from_dict({"file": "a.mid", "license": "Public Domain"}),
        Entry.from_dict({"file": "b.mid", "license": "CC0"}),
        Entry.from_dict({"file": "a.mid", "license": "Public Domain"}),
        Entry.from_dict({"file": "a.mid", "license": "Public Domain"}),
    ]
    kept, dupes, rename_plan = _detect_collisions(fixture)
    if [e.file for e in kept] != ["a.mid", "b.mid"]:
        print(f"FAIL collision-kept: {[e.file for e in kept]}", file=sys.stderr)
        failed += 1
    if [d.suggested_rename for d in dupes] != ["a-2.mid", "a-3.mid"]:
        print(f"FAIL collision-renames: {[d.suggested_rename for d in dupes]}", file=sys.stderr)
        failed += 1
    if [b for _, b in rename_plan] != ["a-2.mid", "a-3.mid"]:
        print(f"FAIL collision-rename-plan: {rename_plan}", file=sys.stderr)
        failed += 1

    if failed:
        print(f"self-test: {failed} failure(s)", file=sys.stderr)
        return 1
    print("self-test: ok")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _format_human_summary(report: Report, manifest_path: Path) -> str:
    lines = [
        f"collector_validate: manifest={manifest_path}",
        f"  accepted={len(report.accepted)}  rejected={len(report.rejected)}  renames_proposed={len(report.rename_plan)}",
    ]
    by_reason: dict = {}
    for f in report.rejected:
        by_reason.setdefault(f.reason, 0)
        by_reason[f.reason] += 1
    if by_reason:
        lines.append("  reasons:")
        for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {reason}: {n}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="collector_validate",
        description="Stage C hardening pass for the song catalog.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"manifest path (default: {DEFAULT_MANIFEST.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--render-check",
        action="store_true",
        help="invoke `cargo run --bin render_midi` per surviving entry",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="rewrite manifest in place (drops rejected, applies renames). Backs up to <manifest>.bak.",
    )
    parser.add_argument(
        "--failures-path",
        type=Path,
        default=FAILURES_LOG,
        help=f"failure log JSON path (default: {FAILURES_LOG.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run unit-style tests against the classifier + collision detector and exit",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return _run_self_test()

    try:
        report = validate_manifest(args.manifest, render_check=args.render_check)
    except FileNotFoundError as e:
        print(f"collector_validate: {e}", file=sys.stderr)
        return 2
    except (ValueError, json.JSONDecodeError) as e:
        print(f"collector_validate: bad manifest: {e}", file=sys.stderr)
        return 2

    dump_artifacts(report, args.manifest, failures_path=args.failures_path)

    if args.apply and (report.rejected or report.rename_plan):
        try:
            apply_in_place(report, args.manifest)
        except OSError as e:
            print(f"collector_validate: --apply failed: {e}", file=sys.stderr)
            return 2

    print(_format_human_summary(report, args.manifest))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
