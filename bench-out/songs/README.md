# Permissive-license MIDI catalog (`bench-out/songs/`)

Collection of permissive-license MIDI files (Public Domain or CC-BY 4.0)
for hand-auditioning the keysynth voice families on real musical
material. Two curation paths feed the same `manifest.json`:

1. **Hand-curated seed** (the original 13 entries, PR #63) — picked for
   pedagogical relevance to specific voice families (sustain stress,
   tremolo, broken chords, etc.). Per-piece `context` field documents
   the testing rationale.
2. **Agent-collected** (Stage B, `tools/mutopia_collector.py`) — an
   AI agent walks the Mutopia Project catalogue, filters by license
   (allowlist: Public Domain, CC-BY 4.0; excludes CC-BY-SA / CC-BY-NC),
   round-robins by composer surname for diversity, and downloads up to
   N pieces per instrument. Idempotent (skips ids and source URLs that
   already appear in the manifest). Layer 1 of the data-collection
   plan: AI agent does the legwork that a human did before.

This catalog is **independent** of the in-tree `render_song::piece_*()`
hardcoded pieces (arpeggio / twinkle / bach_invention /
assistant_uptempo / etc.) — those continue to work exactly as before
through `cargo run --release --bin render_song -- --piece <name>`.
The two paths are complementary: render_song's pieces are
deterministic Rust-defined fixtures; this catalog is real public-domain
repertoire.

## Contents

`manifest.json` is the source of truth. Counts and per-piece metadata
(`composer`, `license`, `source_url`, `suggested_voice`, `tags`,
`context`) live there. Run `keysynth_db query --instrument <i>` for the
current listing, e.g.:

```bash
# Adjust the binary path to your cargo target dir — `target/release/` by
# default, or wherever `CARGO_TARGET_DIR` points. The `.exe` suffix only
# applies on Windows.
target/release/keysynth_db query --instrument guitar
target/release/keysynth_db query --composer Bach
target/release/keysynth_db query --era Romantic --instrument piano
```

The hand-curated seed (PR #63) is identifiable by short, descriptive
file names (`aguado_op3_no1.mid`, `tarrega_recuerdos.mid`,
`mozart_kv309_1.mid`, ...). Agent-collected entries keep their
upstream Mutopia file stems so the canonical URL ↔ on-disk filename
mapping is one-to-one.

## License

Every entry is **Public Domain**. CC-BY-NC, proprietary, and unknown
licenses are explicitly excluded. The Mutopia Project catalogue
restricts contributions to Public Domain / CC0 / CC-BY / CC-BY-SA;
each file's per-piece license metadata sits on its catalogue page
(linked from the `source_url` field).

## Re-download (existing entries)

Every URL is HTTPS / direct-link / curl-able with no authentication.
To re-fetch the whole catalog from scratch using stdlib only:

```bash
python - <<'PY'
import json, urllib.request, pathlib
m = json.load(open('bench-out/songs/manifest.json', encoding='utf-8'))
for e in m['entries']:
    out = pathlib.Path('bench-out/songs') / e['file']
    if out.exists() and out.stat().st_size > 0:
        print('skip', e['file']); continue
    out.write_bytes(urllib.request.urlopen(e['source_url']).read())
    print('fetch', e['file'])
PY
```

If a Mutopia URL returns 404, the piece directory has been
restructured upstream. Open the catalogue page
(`https://www.mutopiaproject.org/cgibin/make-table.cgi?Instrument=Guitar`
or `?Instrument=Piano`), find the piece, copy the `.mid` URL into
the entry's `source_url` in `manifest.json`, and re-run the loop.

## Add new entries (Stage B Agent Collector)

`tools/mutopia_collector.py` (stdlib only, no third-party deps) walks
the Mutopia catalogue, filters by license, and appends N more entries
per instrument. Idempotent — re-running with the same target counts is
a no-op once the manifest is full enough.

```bash
python tools/mutopia_collector.py \
    --target-guitar 25 --target-piano 25 \
    [--dry-run]            # preview picks without downloading

# Then refresh the materialized SQLite catalog so query / Steel REPL
# pick up the new rows:
target/release/keysynth_db rebuild
target/release/ksrepl -e '(query-songs :composer "Bach")'
```

License allowlist: `Public Domain`, `CC-BY 4.0`. CC-BY-SA, CC-BY-NC,
and unknown licenses are rejected. The script picks composer-diverse
entries via round-robin so a target of 25 spreads across ~15+
composers rather than dumping a single op-set.

## Rendering

Pick a voice from the per-entry `suggested_voice` field (or override
with any other engine string supported by `render_midi`):

```bash
# Single render
cargo run --release --bin render_midi --features native -- \
    --in bench-out/songs/tarrega_recuerdos.mid \
    --engine guitar-stk \
    --out bench-out/songs/tarrega_recuerdos_stk.wav

# Render a piano piece through piano-modal
cargo run --release --bin render_midi --features native -- \
    --in bench-out/songs/satie_danse.mid \
    --engine piano-modal \
    --out bench-out/songs/satie_danse_piano.wav
```

`*.wav` outputs are **gitignored** (the `**/*.wav` rule applies). Only
the source `*.mid` files + `manifest.json` + this README are committed.

## Engine map at a glance

| `suggested_voice` value | Engine string for `render_midi --engine ...` | Voice family                                 |
|-------------------------|----------------------------------------------|-----------------------------------------------|
| `guitar-stk`            | `guitar-stk`                                 | `voices_live/guitar_stk` (STK port)          |
| `guitar`                | `guitar`                                     | `voices_live/guitar` (acoustic phys-mod)     |
| `piano-modal`           | `piano-modal`                                | `Engine::PianoModal` (modal piano + LUT)     |
| `piano`                 | `piano`                                      | `Engine::Piano`                               |
| `sfz-piano`             | `sfz-piano` + `--sfz PATH`                   | SFZ Salamander Grand                          |

For chord-progression demos and rhythm patterns, use the
`kssong --midi-out` + `render_midi` pipeline instead — those are
generated, not part of this curated catalog.

## Acquisition

- **Source**: Mutopia Project (https://www.mutopiaproject.org/)
- **About**: https://www.mutopiaproject.org/about.html
- **Date**: 2026-04-28
- **Files**: 13 SMF format-1 (per-file `MThd` header verified at
  fetch time; PPQ varies, typically 384 or 480; tracks 1-3
  depending on arrangement).

## Out of scope

- **Audio renders (`*.wav`)** — generated on demand via `render_midi`,
  gitignored. The MIDI catalog is the SSOT; renders are derived state.
- **kssong-generated chord-progression demos** (e.g. `bright_*`,
  `modern_*`, `rhythm_*`) — those are CC0 in-tree generated material
  and ship through the kssong DSL, not this catalog.
- **Existing `render_song::piece_*()` Rust-defined pieces** — kept as-is
  in `src/bin/render_song.rs`, accessed via the render_song binary, not
  duplicated here.
- **GuitarSet / Slakh2100 corpus material** — handled by the dedicated
  `bench-out/REF/guitar/` corpus (issue #44), not this catalog.
