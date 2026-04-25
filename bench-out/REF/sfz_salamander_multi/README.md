# SFZ Salamander multi-note teacher signal

Five reference renders of the SFZ Salamander Grand Piano V3, used as a
supervised-learning teacher signal for the piano-realism work tracked
under issue #3 (`piano-5am` LUT calibration).

## Source instrument

- **Salamander Grand Piano V3** (44.1 kHz, 16-bit edition)
- Manifest: `Salamander/SalamanderGrandPianoV3_44.1khz16bit/SalamanderGrandPianoV3.sfz`
- Copyright (c) Alexander Holm, distributed under **CC-BY 3.0**.
- The full sample library is `.gitignore`d in this repo (GB-scale, plus
  the licence permits redistribution but the binary noise is not worth
  carrying in git history). Only the small individually-rendered WAVs
  in this directory are checked in, with attribution.

## Files

Each WAV is mono, 16-bit, 44.1 kHz, 6.000 s total (4 s held + 2 s
release tail), velocity 100, peak-normalised to 0.9.

| File          | MIDI | Note | Frequency  | Frames | sha256                                                             |
| ------------- | ---- | ---- | ---------- | ------ | ------------------------------------------------------------------ |
| `note_36.wav` |   36 | C2   |   65.41 Hz | 264600 | `ba4e83cbf1dcda68314ef57cea92927f2c53d5f67e4285696a07c8b06e51fb5a` |
| `note_48.wav` |   48 | C3   |  130.81 Hz | 264600 | `1f713e3aae38f7b7db619ed1d8b6569d1bc83205aa75cc19aac8357b618d1033` |
| `note_60.wav` |   60 | C4   |  261.63 Hz | 264600 | `f95ca633b22fef5f27263796896a4c66901097dcc27fafbb8e083b53029459aa` |
| `note_72.wav` |   72 | C5   |  523.25 Hz | 264600 | `158253481a5db3a2eeead94852cade02af80549343a4a6c42f61b2e2e48718ca` |
| `note_84.wav` |   84 | C6   | 1046.50 Hz | 264600 | `c5dd7fd04e37938f87206d66363514de13636900133c4a12b4a106b526b72445` |

Each file is 529 244 bytes (44 byte WAV header + 264 600 frames * 2 bytes).

## Reproduce

With the Salamander library unpacked at the path above, build the
`bench` binary and render all five notes:

```sh
cargo build --release --bin bench

for note in 36 48 60 72 84; do
    cargo run --release --bin bench -- \
        --engine sfz-piano \
        --sfz Salamander/SalamanderGrandPianoV3_44.1khz16bit/SalamanderGrandPianoV3.sfz \
        --note "$note" \
        --duration 6 --hold 4 \
        --velocity 100 \
        --only keysynth \
        --out bench-out/REF/sfz_salamander_multi
    mv "bench-out/REF/sfz_salamander_multi/keysynth_sfz-piano_n${note}.wav" \
       "bench-out/REF/sfz_salamander_multi/note_${note}.wav"
done
```

Notes on the invocation:

- `--engine sfz-piano` routes the keysynth-side render through
  `SfzPlayer` (`src/sfz/`). This is the path that consumes the SFZ
  manifest.
- `--only keysynth` skips the SoundFont (`render_soundfont`) leg, which
  is what produces the SFZ render. The bench harness names SF2 outputs
  `soundfont_n<note>.wav` and SFZ outputs `keysynth_sfz-piano_n<note>.wav`,
  so we rename to `note_<NN>.wav` for stable downstream filenames.
- Mono mix: `SfzPlayer` produces stereo; `bench` averages L/R to mono
  to match the rest of the harness.
- Output is peak-normalised to 0.9 inside `bench`; relative loudness
  between notes is therefore not preserved (each WAV is independently
  normalised). Downstream consumers that need true relative levels
  should re-render without normalisation, or compare per-partial
  spectra rather than absolute amplitude.

## Attribution

If you redistribute these WAVs (e.g. as part of a release artefact),
preserve the attribution:

> Salamander Grand Piano V3 (c) Alexander Holm, CC-BY 3.0.
> https://archive.org/details/SalamanderGrandPianoV3
