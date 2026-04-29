//! keysynth library: DSP / voice / engine code shared between the
//! interactive `keysynth` binary and offline harnesses like `bench`.
//!
//! `synth` holds the engines, voices, ADSR, hammer/pluck excitations, and
//! MIDI math. `ui` is the egui dashboard, kept here so both bins (or
//! future tools) can re-skin or embed it.
//!
//! Native-only modules (offline analysis, SFZ sampler, IR-WAV reverb
//! loader, midir-driven dashboard) are gated behind the `native` Cargo
//! feature so the wasm32 build for the GitHub Pages demo doesn't try to
//! pull in midir / hound / rustfft / image / rustysynth / etc.

pub mod calibration;
pub mod gm;
pub mod live_abi;
pub mod resample;
pub mod reverb;
pub mod song;
pub mod soundboard;
pub mod sympathetic;
pub mod synth;
pub mod voice_lib;
pub mod voices;

// Native-only modules: pull in midir / rustysynth / rustfft / hound /
// game-music-emu / midly / symphonia which are gated out of the
// wasm32 build via the `web` Cargo feature.
#[cfg(feature = "native")]
pub mod ab_test;
#[cfg(feature = "native")]
pub mod analysis;
#[cfg(feature = "native")]
pub mod audio_audit;
#[cfg(feature = "native")]
pub mod chiptune_import;
#[cfg(feature = "native")]
pub mod cp;
#[cfg(feature = "native")]
pub mod drums;
#[cfg(feature = "native")]
pub mod gui_cp;
#[cfg(feature = "native")]
pub mod jukebox_core;
#[cfg(feature = "native")]
pub mod dsp;
#[cfg(feature = "native")]
pub mod extract;
#[cfg(feature = "native")]
pub mod library_db;
#[cfg(feature = "native")]
pub mod live_reload;
#[cfg(feature = "native")]
pub mod play_log;
#[cfg(feature = "native")]
pub mod preview_cache;
#[cfg(feature = "native")]
pub mod score;
#[cfg(feature = "native")]
pub mod scripting;
#[cfg(feature = "native")]
pub mod sequencer;
#[cfg(feature = "native")]
pub mod sfz;
#[cfg(feature = "native")]
pub mod ui;
#[cfg(feature = "native")]
pub mod voice_collector;

// Re-export the dashboard-facing types at the crate root so older callers
// that imported them via `crate::{LiveParams, DashState, Engine}` keep
// working without churn. New code should prefer `keysynth::synth::...`.
pub use synth::{DashState, Engine, LiveParams};
