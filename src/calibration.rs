//! Per-voice output-level calibration (issue #13).
//!
//! Different engines naturally render at wildly different peak
//! amplitudes — the chord-headroom audit tracked single-note peaks
//! ranging from ~0.1 (Square / FM) to >1.0 (KS family pre-`vel_scaled`).
//! The user's complaint: when A/B-ing voices in the GUI, the louder
//! voice always sounded "better" because the comparison was
//! contaminated by loudness, not timbre. This module fixes that.
//!
//! ## Contract
//!
//! Every voice constructed by [`crate::synth::make_voice`] is wrapped
//! in [`CalibratedVoice`], which scales the inner `render_add` output
//! by a per-engine coefficient measured against a fixed reference
//! signal (C4 / vel 100 / 1.0 s render). The coefficient is chosen so
//! that the resulting peak amplitude lands at [`TARGET_PEAK_DBFS`]
//! ±1.5 dB. The contract is verified by `tests/voice_output_level.rs`.
//!
//! Placeholder voices (`SfPiano`, `SfzPiano`, `Live`) render silence
//! from the trait and have their audio rendered by a shared synthesiser
//! owned by the audio callback. Calibration is a no-op for them — the
//! shared synth is responsible for its own gain staging.
//!
//! ## Why peak, not LUFS
//!
//! Peak is deterministic, requires no K-weighting filter or integration
//! window, matches the metric `chord_headroom_audit` already uses, and
//! is the right metric for the contract: the goal is "no voice can
//! clip the bus harder than another". A perceptually correct loudness
//! match (LUFS / EBU R128) belongs at a later stage if A/B blind tests
//! show audible loudness drift after this contract is in place.
//!
//! ## Re-measuring
//!
//! Run the diagnostic test:
//!
//! ```bash
//! cargo test --test voice_output_level --release \
//!     voice_output_level_diagnostic -- --ignored --nocapture
//! ```
//!
//! and copy the printed `needed_gain` values into the table in
//! [`engine_calibration_gain`].

use crate::synth::{make_voice_raw_inner, Engine, ReleaseEnvelope, VoiceImpl};

/// Target peak amplitude for every calibrated voice's first-second
/// render at C4 / vel 100. `-12 dBFS` ≈ `0.2512` linear, leaving 12 dB
/// of headroom for chord stacking before the bus hits 0 dBFS.
pub const TARGET_PEAK_DBFS: f32 = -12.0;

/// `TARGET_PEAK_DBFS` expressed in linear amplitude. Pre-computed for
/// hot-path use in `CalibratedVoice::new`.
pub const TARGET_PEAK_LINEAR: f32 = 0.251_188_64; // 10^(-12/20)

/// Convert a linear amplitude to dBFS. Returns `f32::NEG_INFINITY` for
/// silence so the diagnostic report doesn't blow up on an all-zero
/// buffer.
#[inline]
pub fn db_from_linear(lin: f32) -> f32 {
    if lin <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * lin.log10()
    }
}

/// Per-engine multiplicative gain that brings a vel-100 C4 render to
/// `TARGET_PEAK_LINEAR` ±1.5 dB. Measured by the
/// `voice_output_level_diagnostic` test on 2026-04-29; rerun if any
/// engine's render path changes.
///
/// Placeholder engines (SfPiano / SfzPiano / Live) return `1.0` because
/// their voice impls render silence; the shared synthesiser handles
/// their gain.
pub fn engine_calibration_gain(engine: Engine) -> f32 {
    match engine {
        // Single-note C4 raw peaks measured 2026-04-29 with the
        // diagnostic test (see module docs). `gain = 0.2512 / raw_peak`.
        // Values are baked rather than computed at runtime so the
        // contract test can fail loudly if a voice's render path
        // changes its loudness.
        Engine::Square => GAINS.square,
        Engine::Ks => GAINS.ks,
        Engine::KsRich => GAINS.ks_rich,
        Engine::Sub => GAINS.sub,
        Engine::Fm => GAINS.fm,
        Engine::Piano => GAINS.piano,
        Engine::PianoThick => GAINS.piano_thick,
        Engine::PianoLite => GAINS.piano_lite,
        Engine::Piano5AM => GAINS.piano_5am,
        Engine::PianoModal => GAINS.piano_modal,
        Engine::Koto => GAINS.koto,
        // Placeholders: their `render_add` is silent. Calibration is a
        // pass-through; the shared SF/SFZ synth and the live cdylib
        // are each responsible for their own gain staging.
        Engine::SfPiano | Engine::SfzPiano | Engine::Live => 1.0,
    }
}

/// Initial measurement table. Field-grouped so changing one engine's
/// number is a single-line edit. See module docs for the measurement
/// procedure.
struct CalibrationTable {
    square: f32,
    ks: f32,
    ks_rich: f32,
    sub: f32,
    fm: f32,
    piano: f32,
    piano_thick: f32,
    piano_lite: f32,
    piano_5am: f32,
    piano_modal: f32,
    koto: f32,
}

// Measured 2026-04-29 by `voice_output_level_diagnostic` (release
// build, SR=44_100, C4, vel=100, 1.0 s render). Raw peaks ranged from
// −8.0 dBFS (Piano) to −2.1 dBFS (Square / Fm / Piano5AM / Koto) — a
// 6 dB spread — which is exactly the loudness contamination this
// contract is here to remove. `gain = TARGET_PEAK_LINEAR / raw_peak`.
const GAINS: CalibrationTable = CalibrationTable {
    square: 0.3190,
    ks: 0.3527,
    ks_rich: 0.5585,
    sub: 0.4081,
    fm: 0.3195,
    piano: 0.6318,
    piano_thick: 0.4495,
    piano_lite: 0.6171,
    piano_5am: 0.3193,
    piano_modal: 0.5353,
    koto: 0.3190,
};

/// `VoiceImpl` wrapper that scales the inner voice's `render_add` output
/// by a fixed coefficient. All other trait methods delegate verbatim,
/// so calibration is invisible to the voice pool, the release
/// envelope, the pedal, and the eviction policy.
///
/// Implementation: render the inner voice into a stack-/heap-local
/// scratch buffer, then scaled-add into the caller's `buf`. We can't
/// pre-zero `buf` (the trait is `render_add`, not `render`: the caller
/// expects mixing), so we have to materialise the inner output before
/// scaling.
pub struct CalibratedVoice {
    inner: Box<dyn VoiceImpl + Send>,
    gain: f32,
}

impl CalibratedVoice {
    pub fn new(inner: Box<dyn VoiceImpl + Send>, gain: f32) -> Self {
        Self { inner, gain }
    }

    /// Wrap `inner` for `engine`. If the engine's calibration is a
    /// no-op (`gain == 1.0`), returns `inner` unwrapped to skip the
    /// per-sample multiply on the audio thread.
    pub fn wrap(
        inner: Box<dyn VoiceImpl + Send>,
        engine: Engine,
    ) -> Box<dyn VoiceImpl + Send> {
        let gain = engine_calibration_gain(engine);
        if (gain - 1.0).abs() < f32::EPSILON {
            inner
        } else {
            Box::new(Self { inner, gain })
        }
    }
}

impl VoiceImpl for CalibratedVoice {
    fn render_add(&mut self, buf: &mut [f32]) {
        let mut scratch = vec![0.0_f32; buf.len()];
        self.inner.render_add(&mut scratch);
        let g = self.gain;
        for (out, &s) in buf.iter_mut().zip(scratch.iter()) {
            *out += s * g;
        }
    }

    fn release_env(&self) -> Option<&ReleaseEnvelope> {
        self.inner.release_env()
    }

    fn release_env_mut(&mut self) -> Option<&mut ReleaseEnvelope> {
        self.inner.release_env_mut()
    }

    fn trigger_release(&mut self) {
        self.inner.trigger_release();
    }

    fn is_done(&self) -> bool {
        self.inner.is_done()
    }

    fn is_releasing(&self) -> bool {
        self.inner.is_releasing()
    }

    fn set_pedal_sustain(&mut self, pedal: f32) {
        self.inner.set_pedal_sustain(pedal);
    }
}

/// Construct an uncalibrated voice. Used by the diagnostic test to
/// measure raw peaks and by tests that need to reason about a voice's
/// natural loudness without the calibration coefficient confounding
/// the measurement. Production code paths go through
/// [`crate::synth::make_voice`] and get calibration for free.
pub fn make_voice_raw(
    engine: Engine,
    sr: f32,
    freq: f32,
    velocity: u8,
) -> Box<dyn VoiceImpl + Send> {
    make_voice_raw_inner(engine, sr, freq, velocity)
}
