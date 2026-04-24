//! Inharmonicity coefficient (B) and stretched-harmonic R² fit. (Stub —
//! filled by the issue #3 P2 agent.)

#![allow(dead_code)]

use super::decompose::Partial;

/// Result of fitting f_n = n · f0 · sqrt(1 + B · n²) to extracted partials.
#[derive(Clone, Debug)]
pub struct BFit {
    /// Inharmonicity coefficient B.
    pub b: f32,
    /// Coefficient of determination of the stretched-harmonic fit.
    pub r_squared: f32,
    /// Number of partials used in the fit.
    pub n_used: usize,
}

/// Fit B and R² from partials. Implementation must match the existing
/// `analyse` output values for `bench-out/REF_sfz_C4.wav`
/// (B ≈ 2.86e-4, R² ≈ 0.999, n_used = 16) within stated tolerance.
///
/// TODO(#3): implement. See module docstring for required test contracts.
pub fn fit_b(_partials: &[Partial]) -> BFit {
    todo!("issue #3 P2: B coefficient + R² fit")
}
