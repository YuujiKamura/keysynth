# Changelog

## Unreleased

(Commits since `5eaf527` "cap voice pool at 32" baseline. Hash + commit
subject + revert-risk tag.)

### Sound-character changes

- `f85088d` piano-lite: register Engine::PianoLite (3-string + 12-mode lite, no sym) `[REVERT-CANDIDATE]`
- `abad21d` piano-thick: decay/coupling tuning to match SFZ Salamander C4 reference `[REVERT-CANDIDATE]`
- `2cb3083` piano-lite: decay tuning to match SFZ Salamander C4 reference `[REVERT-CANDIDATE]`
- `d0ad787` piano: shared sympathetic string bank (24 strings, soundboard-driven) `[REVERT-CANDIDATE]`
- `d81bf97` piano: bidirectional soundboard coupling (replaces additive click components) `[REVERT-CANDIDATE]`
- `8f0df23` piano attack + engine UX polish `[REVERT-CANDIDATE]`
- `af266f7` synth(piano): composite 3.4796 -> 3.4587 + add Fix A/B infrastructure `[REVERT-CANDIDATE]`
- `c249296` synth(piano): drive composite metric 4.04 -> 3.48 (-13.8%) vs SF2 ref `[REVERT-CANDIDATE]`
- `3252a6a` synth: drop piano LSD vs SF2 reference 60.55 -> 56.39 dB `[REVERT-CANDIDATE]`

### Architectural

- `8c2bdd9` refactor: ReleaseEnvelope trait helper + comprehensive test coverage (140 tests) `[KEEP]`
- `52552c9` sf-piano: GM 128 program picker (live switch via GUI + MIDI PC + CLI) `[KEEP]`
- `df05bc0` keysynth: sf-piano engine + body IR reverb (synthetic + WAV-loadable) `[KEEP]`
- `2e34772` keysynth: extract DSP into lib for bench binary reuse `[KEEP]`

### Bug fixes

- `e965523` keysynth: fix GUI/MIDI master race, drop RT-thread eprintln, idiom cleanups `[KEEP]`
- `914b4ed` keysynth: fix allpass sign, koto pluck-position, piano detune, fractional-delay tuning `[KEEP]`

### Eval / measurement

- `188f3d7` analyse: v0.2 metric stack (B, MR-STFT, T60 vector, onset, centroid) + metrics.json `[KEEP]`
- `31f0923` docs: add reference/ ground-truth workflow + analyse v0.2 README section `[KEEP]`
- `6076f7e` analysis: add B estimator, MR-STFT L1, onset L2, T60 vector + Bank weighting, centroid MSE `[KEEP]`
- `be6c753` analyse: STFT + per-harmonic decay + LSD distance for falsifiable WAV comparison `[KEEP]`
- `742004c` gitignore: block acquired sound data + compressed bundles `[KEEP]`
- `3ad116f` bench: SoundFont reference comparison harness `[KEEP]`

## CPU budget baseline (issue #2)

Measured by `cargo run --release --bin cpubench` (32 voices × 4 s @ 44.1 kHz,
no audio I/O, no reverb, no sym bank). Numbers are wall / audio = CPU%, so
<100 % means real-time-capable on this machine.

| engine        | CPU%   |
|---------------|--------|
| square        |  0.5 % |
| ks            |  0.8 % |
| ks-rich       |  1.3 % |
| sub           |  1.2 % |
| fm            |  2.1 % |
| piano         |  4.4 % |
| piano-thick   |  5.5 % |
| piano-lite    |  2.9 % |
| koto          |  1.2 % |

Headroom is comfortable across the board — 32-voice piano-thick polyphony
uses ~5.5 % of one core. No voice-cap reduction or soundboard SIMD is
needed to ship; the issue #2 worry that "sustained piano polyphony eats a
core" turns out not to bite at the current implementation. Re-measure if
the soundboard gets denser (more modes) or if the sympathetic bank starts
running per-engine instead of stream-shared.
