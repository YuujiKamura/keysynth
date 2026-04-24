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

## License

MIT (or whatever the user prefers later).
