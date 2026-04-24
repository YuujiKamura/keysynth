//! Per-partial T60 extraction (time to -60 dB). (Stub — filled by the
//! issue #3 P3 agent.)

#![allow(dead_code)]

use super::decompose::Partial;

/// Per-partial T60 vector. Index n in the returned vec is partial n+1
/// (i.e. `vec[0]` = fundamental T60, `vec[1]` = 2nd partial T60, ...).
#[derive(Clone, Debug)]
pub struct T60Vector {
    pub seconds: Vec<f32>,
}

/// Extract T60 (seconds to -60 dB) per partial. Implementation must
/// reproduce the existing `analyse` h1-h8 values for
/// `bench-out/REF_sfz_C4.wav` (h1 ≈ 18.14 s, h2 ≈ 11.21 s, h3 ≈ 9.58 s,
/// h4 ≈ 7.38 s, h5 ≈ 6.98 s, h6 ≈ 7.93 s, h7 ≈ 8.98 s, h8 ≈ 8.66 s)
/// within stated tolerance.
///
/// TODO(#3): implement. See module docstring for required test contracts.
pub fn extract_t60(_signal: &[f32], _sr: f32, _partials: &[Partial]) -> T60Vector {
    todo!("issue #3 P3: per-partial T60 extraction")
}
