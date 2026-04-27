# Brief B (Codex): KiraEngine + custom Sound impl in src/audio_kira.rs

## Context

The user is replacing the hand-rolled AudioWorkletProcessor + chunk-
render path in `src/bin/web.rs` with Kira (`kira = "0.12"`). The
work is split between two parallel agents:

- **Gemini (other deck)** — main-thread glue layer in
  `src/bin/web.rs` + `Cargo.toml`. Calls into the API you build.
- **You (Codex)** — `src/audio_kira.rs` implementation.

DO NOT TOUCH `src/bin/web.rs` or `Cargo.toml`. Gemini owns those.
You own ONLY `src/audio_kira.rs`. If you need a Cargo dep added,
note it in the PR description; do not edit `Cargo.toml`.

## Working tree

- Worktree: `C:\Users\yuuji\AppData\Local\Temp\keysynth-web`
  (Unix: `/tmp/keysynth-web`)
- Branch: `claude/web-kira-experiment` (already checked out, based
  on `origin/main`)
- `src/audio_kira.rs` exists as a skeleton. Replace the
  placeholder body with your implementation. Top-of-file doc
  comment lists the public API contract; that's the integration
  point with Gemini.
- `Cargo.toml` already has `kira = { version = "0.12",
  default-features = false, features = ["cpal"], optional = true }`
  gated under the `web` feature. You can use it freely.

## Goal

Implement `KiraEngine` in `src/audio_kira.rs`. Public API:

```rust
pub struct KiraEngine {
    // private: AudioManager, voice handle map, current engine,
    //          current modal preset, sample rate, …
}

impl KiraEngine {
    /// Initialise Kira's AudioManager with default settings.
    /// Fails loudly if Kira can't acquire an AudioContext (e.g.
    /// because no user gesture preceded the call) — return a
    /// `Result<Self, String>` with the error stringified for UI
    /// display.
    pub fn new() -> Result<Self, String>;

    /// Trigger a note. Constructs a `keysynth::Voice` via
    /// `make_voice(engine, sr, freq, velocity)` and plays it
    /// through Kira. Stash the resulting handle in an internal
    /// `HashMap<u8, SoundHandle>` keyed by MIDI note so
    /// `note_off` can find it.
    pub fn note_on(&mut self, engine: keysynth::synth::Engine,
                   note: u8, velocity: u8);

    /// Release a held note. Looks up the handle for `note`,
    /// triggers Kira's stop with a short fade, drops from the
    /// map.
    pub fn note_off(&mut self, note: u8);

    /// Update the active engine. Doesn't retroactively affect
    /// already-playing voices; only the next `note_on` uses it.
    pub fn set_engine(&mut self, engine: keysynth::synth::Engine);

    /// Apply a `ModalPreset` — writes to the global MODAL_PARAMS
    /// cell + MODAL_PHYSICS atomic that `make_voice(PianoModal,
    /// …)` reads on construction. (Note: the main-thread wasm
    /// has its OWN MODAL_PARAMS cell since wasm32 single-instance
    /// — the existing `preset.apply()` from `keysynth::synth`
    /// already writes to that cell. Calling it here just makes
    /// the API ergonomic for the UI side.)
    pub fn set_modal_preset(&mut self,
                            preset: keysynth::synth::ModalPreset);

    /// Sample rate the AudioManager negotiated with the platform.
    /// Used for downstream frequency math (`midi_to_freq`).
    pub fn sample_rate(&self) -> u32;
}
```

Internally:

1. **AudioManager** — `kira::AudioManager::<DefaultBackend>::new(
   AudioManagerSettings::default())` on `KiraEngine::new()`.
2. **VoiceSound** — implement `kira::sound::Sound` +
   `kira::sound::SoundData`. Wraps a `Box<dyn keysynth::synth::
   VoiceImpl + Send>` and renders into Kira's `Frame` buffer
   each `process()` call. Mono → stereo (`frame.left =
   frame.right = sample`).
3. **Bus chain** — for now, run the live audio bus chain INLINE in
   the VoiceSound's `process()` per voice (sympathetic + reverb +
   MixMode). Wrong-ish for sympathetic (should be shared) but
   simplest. Document the limitation in a doc comment; promote to
   shared state in a follow-up.
   - Or, if Kira's `Effect` trait is clean, add a sub-track with
     a custom `BusEffect` implementation. Either is acceptable;
     pick whichever lets you ship sound first.
4. **Map** — `HashMap<u8, SoundHandle>` keyed on MIDI note.
   `note_on` evicts the previous handle for that note (stop +
   replace). `note_off` removes + stops.

## Bus chain DSP (mirror of native `src/main.rs::audio_callback`)

After voice render, in the same `process()` (or in the BusEffect):
- Sympathetic-string bank drive on piano-family engines: `COUPLING
  = 0.0002`, `MIX = 0.3`. Tick the bank silently for non-piano so
  state doesn't go stale. Use `keysynth::sympathetic::
  SympatheticBank::new_piano(sr)`.
- NaN/Inf + denormal flush: `if !s.is_finite() { *s = 0.0; }
  else if s.abs() < 1e-30 { *s = 0.0; }`.
- Body IR reverb: `keysynth::reverb::Reverb::new(reverb::
  synthetic_body_ir(sr_u))`. Apply `process(samples, wet)` with
  default wet=0.3.
- MixMode: default ParallelComp, master=1.0. Same constants as
  `src/main.rs::audio_callback`:
  - Plain: `(s * master).tanh()`
  - ParallelComp: ALPHA=0.7, BETA=0.6, ATTACK=0.5, RELEASE=0.0001
  - Limiter: ATTACK=0.5, RELEASE=0.0001

## What "play it through Kira" means concretely

Skim https://docs.rs/kira/0.12/kira/sound/index.html for the
current trait shape. Implement `SoundData::into_sound(self) ->
Box<dyn Sound>` on a `VoiceSoundData` builder; implement `Sound`
on `VoiceSound`.

Likely process signature (verify against actual 0.12 docs):
```rust
fn process(&mut self, out: &mut [Frame], info: &Info)
```

Pre-allocate the mono scratch buffer once at construction; do NOT
allocate inside `process()`.

## Verification

1. `cargo check --no-default-features --features web --target
   wasm32-unknown-unknown --bin keysynth-web` — clean. The bin's
   build is what tests your module's compile.
2. `cargo check --bin keysynth` (native) — clean. `audio_kira` is
   gated to `#[cfg(feature = "web")]` so native shouldn't see it.
3. `cargo fmt --check` — clean.
4. Wait for Gemini to land their `src/bin/web.rs` rewrite (will
   call into your API). Pull, rebuild, verify both compile
   together.
5. `trunk serve web/index.html --port 8090 --no-autoreload`. Wait
   for ✅ success. Browser at http://localhost:8090/, ▶ Start,
   click "Square" voice slot, press `Z`. Should produce a clean
   square tone with no clicks under sustained playback.

## Hard constraints

- **No silent fallbacks.** If Kira's AudioManager fails to
  initialise, return `Err(message)` from `KiraEngine::new()`. The
  UI splash gate already shows errors via `audio_err`. Don't
  retry, don't paper over, don't fake-succeed with silence. The
  user has explicitly said this rule: silent failure < no sound.
- **No allocations on `process()`.** Pre-allocate scratch buffers
  in the `Sound` constructor.
- **Native build doesn't break.** `audio_kira` module is gated to
  `feature = "web"` (see `src/lib.rs`); preserve the gate.
- **Use the same DSP code keysynth ships.** Don't reimplement
  modal piano, sympathetic, reverb. Use `keysynth::voices`,
  `keysynth::sympathetic::SympatheticBank`, `keysynth::reverb::
  Reverb`. Same Rust source = same audio as native.

## Out of scope

- SF2 / `Engine::SfPiano` — separate problem (rustysynth doesn't
  fit Kira's Sound shape cleanly without a multi-voice wrapper).
  Stub `Engine::SfPiano` to log "not yet supported" and skip.
- Master/reverb/mix UI sliders affecting Kira's track. Just
  default values for first cut.
- Modal preset propagation when slots clicked while a Modal voice
  is held. The new voice picks up the new params; the old one
  finishes its release at old params. Acceptable.

## When stuck

- Kira docs: https://docs.rs/kira/0.12
- Kira github: https://github.com/tesselode/kira
- bevy_kira_audio for AudioContext / web wiring patterns:
  https://github.com/NiklasEi/bevy_kira_audio

If Kira's CpalBackend on wasm32 has an unfixable issue (e.g.
requires a feature flag we can't enable, or fails at runtime in a
way I can't repro headlessly), STOP and report it in your commit
message rather than hacking around it. The point of this
experiment is to find out empirically whether Kira works on web —
honest "no" is more useful than fake "yes".

## Deliverable

A commit on `claude/web-kira-experiment` that:
- Implements the public API documented above.
- Passes `cargo check` for both targets.
- Passes `cargo fmt --check`.
- Has a clear commit message describing what landed and what
  defaults the bus chain ships with.

Push when done. Gemini will pull and integrate. Joint PR opens
once both halves land.
