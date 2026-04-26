//! keysynth library: DSP / voice / engine code shared between the
//! interactive `keysynth` binary and offline harnesses like `bench`.
//!
//! `synth` holds the engines, voices, ADSR, hammer/pluck excitations, and
//! MIDI math. `ui` is the egui dashboard, kept here so both bins (or
//! future tools) can re-skin or embed it.

pub mod analysis;
pub mod chiptune_import;
pub mod drums;
pub mod dsp;
pub mod extract;
pub mod gm;
pub mod resample;
pub mod reverb;
pub mod score;
pub mod sequencer;
pub mod sfz;
pub mod soundboard;
pub mod sympathetic;
pub mod synth;
pub mod ui;
pub mod voice_lib;
pub mod voices;

// Re-export the dashboard-facing types at the crate root so older callers
// that imported them via `crate::{LiveParams, DashState, Engine}` keep
// working without churn. New code should prefer `keysynth::synth::...`.
pub use synth::{DashState, Engine, LiveParams};
