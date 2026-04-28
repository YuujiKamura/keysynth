# Guitar reference samples

CC0-licensed steel-string acoustic guitar single-note WAVs used by
`tests/guitar_e2e.rs` to evaluate the residual between Engine::Ks
(legacy default) and the Engine::Live + voices_live/guitar physical
model introduced in issue #44.

## License

Creative Commons CC0 1.0 Universal (public domain). Verbatim
attribution and source URLs are in `LICENSE.txt`.

## Why these WAVs aren't committed

The 7 WAVs total ~2 MB — small enough to embed alongside the existing
piano references — but issue #44's brief explicitly asked for the
"download URL + fetch script" pattern so the audit trail stays
externally verifiable. Run `tools/fetch-guitar-refs.sh` to populate
this directory; the script is idempotent and skips files already
present with non-zero size.

## Contents (after fetch)

| File                          | MIDI | Pitch  | Used by gate                      |
|-------------------------------|------|--------|-----------------------------------|
| `MartinGM2_040__E2_1.wav`     | 40   | E2     | gate 2 (Fletcher B, low E)        |
| `MartinGM2_046_Bb2_1.wav`     | 46   | Bb2    | gate 2 nearest to A2 (one above)  |
| `MartinGM2_052__E3_1.wav`     | 52   | E3     | gate 2 nearest to D3 (two above)  |
| `MartinGM2_055__G3_1.wav`     | 55   | G3     | gate 2 (open G)                   |
| `MartinGM2_058_Bb3_1.wav`     | 58   | Bb3    | gate 2 nearest to B3              |
| `MartinGM2_061_Db4_1.wav`     | 61   | Db4    | gate 1 (Engine::Ks A/B residual)  |
| `MartinGM2_064__E4_1.wav`     | 64   | E4     | gate 2 + gate 3 decay envelope    |

The Discord SFZ GM Bank sources its acoustic-steel patch from a
2017 Martin HD-28 Vintage Series instrument. `LICENSE.txt` carries
the full provenance plus the upstream README's hard rule that
"only CC0, CC-BY, and equivalent licences are allowed" in that
sample bank.
