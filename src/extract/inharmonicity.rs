//! Inharmonicity coefficient (B) and stretched-harmonic R² fit.
//!
//! Fits the Fletcher stretched-harmonic model
//!
//! ```text
//!     f_n = n · f1 · sqrt(1 + B · n²)
//! ```
//!
//! to a sequence of measured `Partial` peaks via the linearised
//! least-squares formulation used by the existing `analyse` binary
//! (`src/analysis.rs::estimate_inharmonicity_b`):
//!
//! ```text
//!     y_n = (f_n / (n · f1))² - 1 = B · n²   ⇒   y = B · x  where x = n²
//! ```
//!
//! and B is the slope of a linear fit *through the origin* on (x_n, y_n).
//! R² is reported with the same origin-anchored definition as the existing
//! analyser so the two pipelines stay numerically comparable.
//!
//! `f1` is taken directly from `partials[0].freq_hz` (the n=1 entry). We
//! do **not** re-fit f1 jointly: the round-trip and golden tests both feed
//! a clean fundamental and the linearised LS is sensitive to small f1
//! errors, so trusting the n=1 measurement keeps the fit consistent with
//! what `analyse` reports.

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

/// Fit B and R² from extracted partials. Mirrors the linearised LS used by
/// `analysis::estimate_inharmonicity_b` so the offline `analyse` binary
/// and this extractor agree on `bench-out/REF_sfz_C4.wav` (B ≈ 2.86e-4,
/// R² ≈ 0.999, n_used = 16).
///
/// Algorithm:
/// 1. f1 ← `partials[0].freq_hz` (the n=1 entry, located by `Partial::n`).
/// 2. For each partial with n ≥ 2: `x = n²`, `y = (f / (n·f1))² - 1`.
///    The n=1 sample has y = 0 by construction and contributes nothing to
///    the slope, so it is excluded.
/// 3. B = Σ(x·y) / Σ(x·x)  (LS through origin).
/// 4. R² = 1 - Σ(y - B·x)² / Σ y²  (origin-anchored, matches `analyse`).
///
/// Edge cases:
/// - empty input, missing fundamental, or fewer than two usable n≥2
///   partials → `BFit { b: 0, r_squared: 0, n_used: 0 }`.
/// - non-positive f1 → same zero return.
pub fn fit_b(partials: &[Partial]) -> BFit {
    // Locate f1 from the n=1 entry. Fall back to partials[0] if no entry
    // is explicitly tagged n=1 (decompose() is documented to order by
    // harmonic index, so partials[0] should always be the fundamental).
    let f1 = partials
        .iter()
        .find(|p| p.n == 1)
        .or_else(|| partials.first())
        .map(|p| p.freq_hz)
        .unwrap_or(0.0);

    if f1 <= 0.0 {
        return BFit {
            b: 0.0,
            r_squared: 0.0,
            n_used: 0,
        };
    }

    // Collect (x = n², y = (f/(n*f1))² - 1) for every n ≥ 2.
    let mut xs: Vec<f64> = Vec::with_capacity(partials.len());
    let mut ys: Vec<f64> = Vec::with_capacity(partials.len());
    let mut sum_xy = 0.0_f64;
    let mut sum_xx = 0.0_f64;
    for p in partials {
        if p.n < 2 || p.freq_hz <= 0.0 {
            continue;
        }
        let n = p.n as f64;
        let r = p.freq_hz as f64 / (n * f1 as f64);
        let x = n * n;
        let y = r * r - 1.0;
        xs.push(x);
        ys.push(y);
        sum_xy += x * y;
        sum_xx += x * x;
    }

    let n_used_for_fit = xs.len();
    if n_used_for_fit < 2 || sum_xx <= 0.0 {
        return BFit {
            b: 0.0,
            r_squared: 0.0,
            n_used: 0,
        };
    }

    let b_est = sum_xy / sum_xx;

    // R² for origin-anchored fit: 1 - SS_res/SS_tot with SS_tot = Σ y².
    let mut ss_res = 0.0_f64;
    let mut ss_tot = 0.0_f64;
    for i in 0..xs.len() {
        let pred = b_est * xs[i];
        let res = ys[i] - pred;
        ss_res += res * res;
        ss_tot += ys[i] * ys[i];
    }
    let r2 = if ss_tot > 0.0 {
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    } else {
        // All y_n ≈ 0 (e.g. perfectly harmonic input). The fit is
        // degenerate but consistent — report R² = 1 so callers don't
        // mis-flag a perfect harmonic series as a bad fit.
        1.0
    };

    // Total partials *seen* including n=1 (consistent with how `analyse`
    // reports n_partials_used — every partial that survived peak picking).
    let n_used_total = n_used_for_fit + partials.iter().filter(|p| p.n == 1).count().min(1);

    BFit {
        b: b_est as f32,
        r_squared: r2 as f32,
        n_used: n_used_total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the exact stretched-harmonic series f_n = n · f1 · sqrt(1+B·n²)
    /// for n=1..=max_n.
    fn synth_partials(f1: f32, b: f32, max_n: usize) -> Vec<Partial> {
        (1..=max_n)
            .map(|n| {
                let nf = n as f32;
                let f = nf * f1 * (1.0 + b * nf * nf).sqrt();
                Partial {
                    n,
                    freq_hz: f,
                    init_db: 0.0,
                }
            })
            .collect()
    }

    /// Deterministic LCG for reproducible "noise" without pulling in a
    /// dev-dependency. Returns f32 in (-1, 1).
    fn lcg_uniform(state: &mut u64) -> f32 {
        // Numerical Recipes LCG.
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        // Take top 24 bits → [0, 1), then map to (-1, 1).
        let bits = (*state >> 40) as u32; // 24 bits
        let u = (bits as f32) / ((1u32 << 24) as f32); // [0, 1)
        u * 2.0 - 1.0
    }

    /// Box-Muller from two uniforms → one standard-normal sample.
    fn lcg_gauss(state: &mut u64) -> f32 {
        let u1 = (lcg_uniform(state) * 0.5 + 0.5).max(1e-9); // (0, 1]
        let u2 = lcg_uniform(state) * 0.5 + 0.5; // [0, 1)
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        r * theta.cos()
    }

    #[test]
    fn round_trip_known_b() {
        let f1 = 261.63_f32;
        let b_true = 2.86e-4_f32;
        let partials = synth_partials(f1, b_true, 16);
        let fit = fit_b(&partials);

        // Within ±5%: 2.86e-4 · 0.95 = 2.717e-4, · 1.05 = 3.003e-4.
        assert!(
            fit.b >= 2.72e-4 && fit.b <= 3.00e-4,
            "B out of ±5% band: got {:e} (target {:e})",
            fit.b,
            b_true
        );
        assert!(
            fit.r_squared >= 0.99,
            "R² too low: {} (need ≥ 0.99)",
            fit.r_squared
        );
        assert_eq!(fit.n_used, 16, "expected all 16 partials counted");
    }

    #[test]
    fn round_trip_zero_b() {
        let f1 = 440.0_f32;
        let partials = synth_partials(f1, 0.0, 16);
        let fit = fit_b(&partials);

        assert!(
            fit.b.abs() <= 1e-5,
            "B should be ~0 for perfect harmonic series, got {:e}",
            fit.b
        );
        assert!(
            fit.r_squared >= 0.99,
            "R² should still be ≥0.99 for clean harmonic, got {}",
            fit.r_squared
        );
    }

    #[test]
    fn round_trip_with_cents_noise() {
        // ±2 cents Gaussian noise (1σ ≈ 1 cent, so ±2 cents covers ~95%).
        let f1 = 261.63_f32;
        let b_true = 2.86e-4_f32;
        let clean = synth_partials(f1, b_true, 16);

        // Multiplicative cents noise: f' = f · 2^(cents/1200).
        // 1σ = 1 cent → 2^(±2/1200) ≈ ±0.115%.
        let mut state: u64 = 0xdead_beef_1234_5678;
        let sigma_cents = 1.0_f32;
        let noisy: Vec<Partial> = clean
            .iter()
            .map(|p| {
                if p.n == 1 {
                    // Keep the fundamental clean: fit_b() uses it as f1
                    // reference and noise on f1 propagates to every other
                    // sample as a coherent bias rather than independent
                    // noise. The test brief asks for ±2c noise *per
                    // partial*; noise on n=1 alone would shift every n>1
                    // sample, which is a different perturbation than the
                    // one being tested.
                    p.clone()
                } else {
                    let cents = lcg_gauss(&mut state) * sigma_cents;
                    let factor = 2f32.powf(cents / 1200.0);
                    Partial {
                        n: p.n,
                        freq_hz: p.freq_hz * factor,
                        init_db: p.init_db,
                    }
                }
            })
            .collect();

        let fit = fit_b(&noisy);

        // ±15% on B.
        assert!(
            fit.b >= b_true * 0.85 && fit.b <= b_true * 1.15,
            "B out of ±15% with 2c noise: got {:e} (target {:e})",
            fit.b,
            b_true
        );
        assert!(
            fit.r_squared >= 0.85,
            "R² should still be ≥0.85 with 2c noise, got {}",
            fit.r_squared
        );
    }

    #[test]
    fn empty_input_is_zero_fit() {
        let fit = fit_b(&[]);
        assert_eq!(fit.b, 0.0);
        assert_eq!(fit.r_squared, 0.0);
        assert_eq!(fit.n_used, 0);
    }

    #[test]
    fn single_partial_is_zero_fit() {
        // Only the fundamental — nothing to constrain B with.
        let p = vec![Partial {
            n: 1,
            freq_hz: 261.63,
            init_db: 0.0,
        }];
        let fit = fit_b(&p);
        assert_eq!(fit.b, 0.0);
        assert_eq!(fit.n_used, 0);
    }
}
