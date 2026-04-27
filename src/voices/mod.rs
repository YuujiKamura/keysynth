//! Per-voice DSP impls split out of `synth.rs` (issue #2). Each module
//! holds a single `VoiceImpl` and any voice-local helpers; shared
//! primitives (`KsString`, `ReleaseEnvelope`, hammer/pluck excitations,
//! `Adsr`) stay in `synth` and are imported here.

pub mod fm;
pub mod hammer_stulov;
pub mod koto;
pub mod ks;
pub mod ks_rich;
pub mod longitudinal;
pub mod piano;
pub mod piano_5am;
pub mod piano_modal;
pub mod placeholder;
pub mod square;
pub mod sub;
