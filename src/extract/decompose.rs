//! Partial decomposition: extract the first N harmonic peak frequencies
//! and per-partial initial amplitudes from a recorded note. (Stub —
//! filled by the issue #3 P1 agent.)

#![allow(dead_code)]

/// One extracted partial.
#[derive(Clone, Debug)]
pub struct Partial {
    /// Index in the harmonic series (1 = fundamental).
    pub n: usize,
    /// Measured frequency in Hz.
    pub freq_hz: f32,
    /// Initial amplitude in dB (relative to the strongest partial = 0 dB).
    pub init_db: f32,
}

/// Decompose a mono signal into its first `max_partials` partials around
/// `f0_hz`. Returns peaks ordered by harmonic index.
///
/// TODO(#3): implement. See module docstring for required test contracts.
pub fn decompose(_signal: &[f32], _sr: f32, _f0_hz: f32, _max_partials: usize) -> Vec<Partial> {
    todo!("issue #3 P1: partial decomposition extractor")
}
