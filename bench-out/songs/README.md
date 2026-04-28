# Public-domain MIDI catalog (`bench-out/songs/`)

Static collection of permissive-license MIDI files for hand-auditioning
the keysynth voice families on real musical material. Curated rather
than auto-generated: every entry is verified Public Domain via the
Mutopia Project catalogue, with the canonical FTP URL recorded in
`manifest.json`.

This catalog is **independent** of the in-tree `render_song::piece_*()`
hardcoded pieces (arpeggio / twinkle / bach_invention /
assistant_uptempo / etc.) — those continue to work exactly as before
through `cargo run --release --bin render_song -- --piece <name>`.
The two paths are complementary: render_song's pieces are
deterministic Rust-defined fixtures; this catalog is real public-domain
repertoire.

## Contents (13 entries)

### Guitar (10 pieces)

| File                                | Composer                  | Date        |
|-------------------------------------|---------------------------|-------------|
| `aguado_op3_no1.mid`                | D. Aguado                 | 1820s       |
| `bach_bwv999_prelude.mid`           | J.S. Bach (BWV 999)       | c. 1720     |
| `carcassi_op1_no1.mid`              | M. Carcassi               | 1820s       |
| `giuliani_op50_no1_papillon.mid`    | M. Giuliani (Le Papillon) | 1822        |
| `sor_op1_no1.mid`                   | F. Sor                    | 1810s       |
| `tarrega_adelita.mid`               | F. Tárrega (Mazurka)      | 1899        |
| `tarrega_capricho_arabe.mid`        | F. Tárrega                | 1892        |
| `tarrega_recuerdos.mid`             | F. Tárrega (Recuerdos de la Alhambra) | 1896 |
| `trad_redapplerag.mid`              | Traditional (American)    | early 20th C |
| `trad_soldiersjoy.mid`              | Traditional (Anglo-American) | 18th C    |

### Piano (3 pieces)

| File                                | Composer                  | Date  |
|-------------------------------------|---------------------------|-------|
| `mozart_kv309_1.mid`                | W.A. Mozart (KV 309 mvt 1) | 1777 |
| `satie_danse.mid`                   | E. Satie (6 Croquis)       | 1913 |
| `albeniz_op71_rumores.mid`          | I. Albéniz (Rumores de la Caleta) | 1886 |

Per-piece `composer`, `license`, `source_url`, `suggested_voice`, and
`context` (one-line rationale for the choice) are in
[`manifest.json`](./manifest.json).

## License

Every entry is **Public Domain**. CC-BY-NC, proprietary, and unknown
licenses are explicitly excluded. The Mutopia Project catalogue
restricts contributions to Public Domain / CC0 / CC-BY / CC-BY-SA;
each file's per-piece license metadata sits on its catalogue page
(linked from the `source_url` field).

## Re-download

Every URL is HTTPS / direct-link / curl-able with no authentication.
To re-fetch the whole catalog from scratch:

```bash
mkdir -p bench-out/songs
while IFS= read -r line; do
    file=$(echo "$line" | jq -r .file)
    url=$(echo "$line"  | jq -r .source_url)
    out="bench-out/songs/${file}"
    [[ -s "$out" ]] && { echo "skip $file"; continue; }
    curl -fsSL -o "$out" "$url" && echo "fetch $file"
done < <(jq -c '.entries[]' bench-out/songs/manifest.json)
```

If a Mutopia URL returns 404, the piece directory has been
restructured upstream. Open the catalogue page
(`https://www.mutopiaproject.org/cgibin/make-table.cgi?Instrument=Guitar`
or `?Instrument=Piano`), find the piece, copy the `.mid` URL into
the entry's `source_url` in `manifest.json`, and re-run the loop.

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
