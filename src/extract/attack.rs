//! Hammer attack envelope characterisation. (Stub — filled by the
//! issue #3 P4 agent.)

#![allow(dead_code)]

/// Summary of the first ~100 ms of a recorded note.
#[derive(Clone, Debug)]
pub struct AttackEnvelope {
    /// Time from note onset to peak amplitude (seconds).
    pub time_to_peak_s: f32,
    /// Peak RMS level in dB (relative to full-scale).
    pub peak_db: f32,
    /// Slope (dB/sec) of the linear fit over the 30..80 ms window.
    pub post_peak_slope_db_s: f32,
    /// Sub-sample of the RMS envelope (every `step_samples`-th sample of
    /// the per-sample RMS, useful for visual A/B vs candidate).
    pub rms_envelope_db: Vec<f32>,
}

/// Extract the first `window_ms` of attack characterisation from a mono
/// signal. Default window is 100 ms; rms uses ~5 ms hop.
///
/// TODO(#3): implement. See module docstring for required test contracts.
pub fn extract_attack(_signal: &[f32], _sr: f32, _window_ms: f32) -> AttackEnvelope {
    todo!("issue #3 P4: hammer attack envelope")
}
