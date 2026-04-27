//! Kira-backed audio engine for the web build.
//!
//! This module owns:
//!   - the `kira::AudioManager<DefaultBackend>` (page-lifetime)
//!   - a custom `Sound` impl wrapping a keysynth `Voice`
//!   - an optional bus `Effect` (sympathetic + reverb + MixMode tanh)
//!   - a `HashMap<u8, SoundHandle>` so `note_off(midi_note)` can find
//!     the right active sound to release
//!
//! Public API expected by `src/bin/web.rs`:
//!
//! ```ignore
//! pub struct KiraEngine { /* … */ }
//!
//! impl KiraEngine {
//!     pub fn new() -> Result<Self, String>;
//!     pub fn note_on(&mut self, engine: Engine, note: u8, velocity: u8);
//!     pub fn note_off(&mut self, note: u8);
//!     pub fn set_engine(&mut self, engine: Engine);
//!     pub fn set_modal_preset(&mut self, preset: ModalPreset);
//!     pub fn sample_rate(&self) -> u32;
//! }
//! ```
//!
//! The Codex dispatch fills in the body. This skeleton exists so the
//! Gemini side (web.rs glue) can compile against the public type
//! while Codex iterates on the implementation.

#![cfg(feature = "web")]

// IMPLEMENTATION GOES HERE — see .dispatch/kira-codex-brief.md
