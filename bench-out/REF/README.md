# Reference WAV (locked-in eval baseline)

This directory holds the canonical reference WAV used as the deterministic
regression baseline for the rendering pipeline. It is **NOT** a piano
reference and **NOT** a perceptual-quality target — it is a synthetic
square-wave render whose only job is to catch regressions in the engine
dispatch, voice rendering, normalisation, and WAV writer paths.

The proprietary SFZ Salamander SoundFont is **not** required to regenerate
this reference, so anyone with the repo can rebuild it from source and
verify byte/hash equivalence (modulo Rust/codegen drift).

## Files

- `square_c4_4s.wav` — Engine::Square at MIDI 60 (C4, 261.63 Hz),
  4 s total, 3 s held + 1 s release tail, mono 16-bit 44.1 kHz.
  - sha256: `9e730eb8600ea5edc2b8c758a44dca0d9953b1ba6c8fc639bff5eb1e344ceb97`
  - size:   352844 bytes

## Regenerate

The bench binary names its output `keysynth_<engine_slug>_n<note>.wav`,
so we render and rename:

```bash
cargo run --release --bin bench -- \
    --engine square --note 60 --duration 4 --hold 3 \
    --only keysynth --out bench-out/REF
mv bench-out/REF/keysynth_square_n60.wav bench-out/REF/square_c4_4s.wav
sha256sum bench-out/REF/square_c4_4s.wav
```

`--only keysynth` is required because the default `both` mode would also
try to render a SoundFont side, which needs `--sf2 PATH`. Pinning the
keysynth-only path is exactly what we want for a SoundFont-free baseline.

## Why square, not piano

The piano engines (piano, piano-thick, piano-lite, sf-piano, sfz-piano)
all carry tuning knobs that are *intentionally* tweaked over time —
their renders should drift as we chase the SFZ Salamander C4 target.
Pinning a piano render would create false-positive regressions every
time we tune decay/coupling/hammer.

`Engine::Square` has no such tuning surface: it's a fixed oscillator +
fixed envelope. Any change to its output means the rendering pipeline
itself changed (engine dispatch, render_add, normalise, voice lifecycle,
WAV writer, fractional-delay defaults that leak into shared paths,
etc.) — which is exactly the regression class this baseline catches.
