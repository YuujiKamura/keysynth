# keysynth

Real-time MIDI keyboard → speaker synth. Single-binary Rust app with an egui
dashboard, written as a Phase-1 physical-modelling experiment after the Python
prototype hit GIL / GC / API issues.

## Engines

| `--engine` | Approach | Character |
|---|---|---|
| `square` | NES-style pulse + linear AR envelope | Reference tone |
| `ks` | Karplus-Strong (1 string, 2-tap LP) | Thin plucked string |
| `ks-rich` | KS × 3 detune + 1-pole allpass dispersion | Stiffer, chorused |
| `piano` | KS × 3 + asymmetric ms-scale hammer + freq-dep decay | Piano-leaning |
| `koto` | KS × 1 + sharp plectrum at 1/4 string-length | Plucked Japanese string |
| `sub` | Saw → SVF lowpass (TPT) with cutoff env + ADSR | 80s analog synth |
| `fm` | 2-op (sine carrier × sine modulator), ratio 14:1 | DX7-ish bell |

Engine selection is live-switchable from the GUI.

## Usage

```bash
cargo run --release
cargo run --release -- --engine ks-rich --master 1.0
cargo run --release -- --list
```

Defaults to MPK mini 3 (or first available MIDI input port) and the system
default audio output device.

### CLI

```
keysynth [--engine ENGINE] [--port NAME] [--master FLOAT]
keysynth --list

ENGINE: square|ks|ks-rich|sub|fm|piano|koto
```

## MIDI

- **Note on / off**: standard, including `note_on velocity=0` running-status off
- **CC 7 / CC 70**: master volume (CC 7 absolute, CC 70 relative encoder
  matching the MPK mini 3 K1 factory preset, ±0.05 per tick)
- All other CCs are logged + visualised on the dashboard so you can identify
  which physical control sends which CC number on your specific controller

## Architecture

- `midir` callback thread receives MIDI, mutates `Arc<Mutex<Vec<Voice>>>`
  (voice pool) and `Arc<Mutex<LiveParams>>` (master / engine).
- `cpal` audio callback runs on its own thread, reads the same shared state,
  mixes voices into the output buffer, retires voices whose envelopes have
  decayed to silence.
- `eframe`/`egui` runs the GUI on the main thread. Both audio and MIDI
  handles are owned by the App struct so they live for the GUI's lifetime.

## Status

- Working: real-time playback, polyphony, note_off release, MPK relative-
  encoder master, GUI engine swap, dashboard log
- Acknowledged limits (Phase 1 scope):
  - No body / soundboard convolution (the missing 50% of piano identity)
  - No sympathetic resonance / 3-string bridge coupling
  - Naive saw oscillator aliases at high notes (Sub engine)
  - Audio latency = whatever WASAPI shared mode delivers (typically 20–40 ms
    on Windows). Not yet tuned for low-latency exclusive mode.

## bench: SoundFont reference comparison

A second binary, `bench`, renders the same MIDI note through both `keysynth`
and a SoundFont (`.sf2`) so you can A/B against a "ground truth" reference.
Without this, parameter tuning is blind — there's no way to tell whether your
allpass coefficient or hammer-felt curve is moving you closer to a real
instrument or further away.

```bash
# Render C4 through the piano engine + a SoundFont's grand piano.
# Outputs two mono WAVs side by side under bench-out/.
cargo run --release --bin bench -- --sf2 path/to/piano.sf2 \
    --engine piano --note 60 --duration 3 --hold 1

# Compare output:
ls bench-out/
# keysynth_piano_n60.wav   <- our DWS attempt
# soundfont_n60.wav        <- reference

# Drag both into Audacity / Reaper / Sonic Visualiser to compare
# waveform + spectrogram. Iterate engine parameters, re-render, A/B.

# Skip the SoundFont half (e.g. just dump a keysynth WAV for inspection):
cargo run --release --bin bench -- --only keysynth --engine ks-rich --note 72
```

### SoundFont sources

Free, redistributable piano-flavoured SoundFonts:

- **GeneralUser GS** (~30 MB, CC0-equivalent permissive license) — solid GM
  bank, decent grand piano: <https://schristiancollins.com/generaluser.php>
- **Salamander Grand Piano** (~100 MB+ in `.sfz` form; SF2 conversions exist
  online, CC-BY) — Yamaha C5 multi-velocity samples; the popular reference for
  "what a real grand piano sounds like" in open-source DAW work
- **FluidR3_GM** — public domain GM SoundFont; ships with most Linux distros

Drop a `.sf2` anywhere on disk and pass the path via `--sf2`. Files are
gitignored.

## analyse: quantitative WAV comparison

Ear-based "does this sound like piano?" is unfalsifiable. The `analyse`
binary turns each A/B into a stack of measurable scalars from the piano-
synthesis literature, plus per-harmonic decay tables and spectrogram PNGs.

### Workflow

```bash
# 0. (one time) acquire ground truth: see reference/README.md
#    -> places reference/iowa_piano_ff_c4.wav

# 1. Render the candidate.
cargo run --release --bin bench -- --sf2 GeneralUser-GS.sf2 \
    --engine piano --note 60 --duration 3 --hold 1 \
    --only keysynth --out bench-out
mv bench-out/keysynth_piano_n60.wav bench-out/candidate.wav

# 2. Compare against the real piano recording.
cargo run --release --bin analyse -- \
    --reference reference/iowa_piano_ff_c4.wav \
    --candidate bench-out/candidate.wav \
    --note 60 --out bench-out/report/

# 3. Read the metric stack.
cat bench-out/report/summary.txt
```

### Metric stack (issue #1)

| Metric | Captures | Lower = closer |
|---|---|---|
| `mrstft_l1` | Multi-resolution STFT L1 (windows 512/1024/2048, lin+log) | ✓ |
| `b_residual` | Inharmonicity coefficient B (Fletcher 1962, Rauhala 2007 estimator) | ✓ |
| `t60_vector_loss` | Per-partial T60 with Bank Taylor weighting | ✓ |
| `onset_l2_80ms` | Hammer transient envelope L2, first 80 ms | ✓ |
| `centroid_traj_mse` | Spectral brightness evolution shape | ✓ |
| `lsd_db` | Single-window log-spectral distance (legacy, kept for back-compat) | ✓ |

### Output dir

- `summary.txt` — human-readable single page with all metrics
- `metrics.json` — machine-consumable, same data
- `harmonics.json` — per-partial freq, T60, initial dB, fit R²; plus deltas
- `centroid.csv` — spectral centroid Hz per frame, both files
- `spectrogram_reference.png` / `spectrogram_candidate.png`

### Why these metrics, not LSD alone

Single-window LSD is energy-sensitive and known to correlate poorly with
perceived audio quality (Kilgour et al. 2018, FAD paper). The literature
(Helsinki/Aalto group, Modartt, recent DDSP work) uses physically-meaningful
per-partial metrics: B coefficient drives inharmonicity perception, T60
vector drives decay character, MR-STFT captures temporal evolution at
multiple time scales. A model that only optimises LSD can land in local
minima that satisfy the metric but sound nothing like a piano.

See [issue #1](https://github.com/YuujiKamura/keysynth/issues/1) for the
literature synthesis and the deviation log of the original LSD-only loop.

## License

MIT (or whatever the user prefers later).
