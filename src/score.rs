//! Perceptually-weighted multi-metric loss between a reference and a
//! candidate WAV (issue #3 P3 skeleton).
//!
//! Combines five raw distance terms — multi-resolution STFT L1, per-partial
//! T60 L2, onset-envelope L2 over the first 80 ms, centroid-trajectory MSE,
//! and the absolute residual on the inharmonicity coefficient B — each
//! scaled by an independent weight so the heterogeneous units (dB, seconds,
//! Hz², dimensionless B) can sum into a single scalar `total`.
//!
//! Implementation reuses every analysis primitive shipped with the crate:
//!   * `analysis::mr_stft_l1`
//!   * `analysis::onset_envelope_l2`
//!   * `analysis::stft` + `analysis::centroid_trajectory_mse`
//!   * `extract::decompose::decompose` (partials, per side)
//!   * `extract::inharmonicity::fit_b` (B coefficient, per side)
//!   * `extract::t60::extract_t60` (per-partial T60 vector, per side)
//!
//! The weights have **not** been ear-validated yet; the
//! [`LossWeights::default`] values are calibrated only to bring each raw
//! term into a comparable [0, ~10] band on the SFZ Salamander C4 baseline.
//! Treat them as a starting point for the upcoming `tuneloop` AI search.

#![allow(clippy::too_many_arguments)]

use std::path::Path;

use crate::analysis::{
    centroid_trajectory_mse, mr_stft_l1, onset_envelope_l2, spectral_centroid_per_frame, stft,
};
use crate::extract::decompose::decompose;
use crate::extract::inharmonicity::fit_b;
use crate::extract::t60::extract_t60;

/// Number of partials to extract on each side before fitting B / T60.
/// Matches the `extract_dump` debug helper for parity with pinned goldens.
const MAX_PARTIALS: usize = 16;

/// Onset window in milliseconds (matches the issue #3 brief).
const ONSET_WINDOW_MS: u32 = 80;

/// Sentinel returned by `extract::t60::extract_t60` when a partial has no
/// usable decay. Mirrored here so we can skip those entries when summing.
const T60_SENTINEL: f32 = -1.0;

// ---- Public types ---------------------------------------------------------

/// Per-component weights applied to the raw distance values.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LossWeights {
    /// MR-STFT L1 (existing in `analysis::mr_stft_l1`).
    pub mr_stft: f32,
    /// T60 vector L2 in seconds.
    pub t60: f32,
    /// Onset envelope L2 across the first 80 ms.
    pub onset: f32,
    /// Centroid trajectory MSE.
    pub centroid: f32,
    /// `|B_cand - B_ref|`.
    pub b_residual: f32,
}

impl Default for LossWeights {
    /// Default weights: equal contribution at typical scales (no
    /// ear-validated calibration yet — that's a follow-up). The loss is
    /// dimensionally heterogeneous, so each weight scales its raw value
    /// into a comparable [0, ~10] band:
    fn default() -> Self {
        Self {
            mr_stft: 1.0,    // raw MR-STFT L1 ~ 5-15
            t60: 0.1,        // raw T60 L2 in seconds, ~ 5-25
            onset: 5.0,      // raw onset L2 ~ 0.1-1.0
            centroid: 1e-6,  // raw centroid MSE ~ 1e5-1e6
            b_residual: 1e4, // raw |dB| ~ 1e-4-3e-4
        }
    }
}

/// Per-component breakdown of a single loss evaluation. `*_raw` is the
/// unweighted value, `*_term` is `raw * weight`, `total` is the sum of all
/// `*_term` fields.
#[derive(Clone, Debug, serde::Serialize)]
pub struct LossBreakdown {
    pub mr_stft_raw: f32,
    pub mr_stft_term: f32,
    pub t60_raw: f32,
    pub t60_term: f32,
    pub onset_raw: f32,
    pub onset_term: f32,
    pub centroid_raw: f32,
    pub centroid_term: f32,
    pub b_residual_raw: f32,
    pub b_residual_term: f32,
    /// Sum of every `*_term`.
    pub total: f32,
}

// ---- Path-based entry point -----------------------------------------------

/// Read both WAVs and compute the weighted loss. Mono-mixes multi-channel
/// inputs by averaging channels (matching `extract_dump`'s behaviour).
pub fn loss_paths(
    reference_wav: &Path,
    candidate_wav: &Path,
    f0_hz: f32,
    weights: &LossWeights,
) -> Result<LossBreakdown, String> {
    let (ref_sig, ref_sr) = read_wav_mono(reference_wav)
        .map_err(|e| format!("read reference {}: {}", reference_wav.display(), e))?;
    let (cand_sig, cand_sr) = read_wav_mono(candidate_wav)
        .map_err(|e| format!("read candidate {}: {}", candidate_wav.display(), e))?;
    if (ref_sr - cand_sr).abs() > 0.5 {
        return Err(format!(
            "sample-rate mismatch: reference {} Hz vs candidate {} Hz",
            ref_sr, cand_sr
        ));
    }
    Ok(loss_signals(&ref_sig, &cand_sig, ref_sr, f0_hz, weights))
}

// ---- Sample-based core ----------------------------------------------------

/// Same contract as [`loss_paths`] but on raw `&[f32]` mono buffers — used
/// by unit tests so they don't need on-disk WAVs. If the two slices differ
/// in length the longer is truncated to match the shorter, with a single
/// stderr warning.
pub fn loss_signals(
    reference: &[f32],
    candidate: &[f32],
    sr: f32,
    f0_hz: f32,
    weights: &LossWeights,
) -> LossBreakdown {
    let n = reference.len().min(candidate.len());
    if reference.len() != candidate.len() {
        eprintln!(
            "score: warning: length mismatch (ref={}, cand={}), truncating to {} samples",
            reference.len(),
            candidate.len(),
            n
        );
    }
    let r = &reference[..n];
    let c = &candidate[..n];
    let sr_u = sr.round() as u32;

    // ---- MR-STFT L1 -------------------------------------------------------
    let mr_stft_raw = mr_stft_l1(r, c, sr_u);

    // ---- Onset envelope L2 ------------------------------------------------
    let onset_raw = onset_envelope_l2(r, c, sr_u, ONSET_WINDOW_MS);

    // ---- Centroid trajectory MSE -----------------------------------------
    // `centroid_trajectory_mse` asserts equal frame counts. The two STFTs
    // are computed from the same-length truncated signals so the assertion
    // holds; we still guard against the empty case to match the rest of
    // the pipeline's "any term that can't be computed becomes 0".
    let centroid_raw = if n == 0 {
        0.0
    } else {
        let s_ref = stft(r, sr_u);
        let s_cand = stft(c, sr_u);
        if s_ref.n_frames() == 0 || s_cand.n_frames() == 0 {
            0.0
        } else if s_ref.n_frames() != s_cand.n_frames() {
            // Defensive: should not happen for equal-length inputs but
            // fall back to per-frame trimming rather than panic.
            let m = s_ref.n_frames().min(s_cand.n_frames());
            let ca = spectral_centroid_per_frame(&s_ref);
            let cb = spectral_centroid_per_frame(&s_cand);
            let mut acc = 0.0_f64;
            for i in 0..m {
                let d = (ca[i] - cb[i]) as f64;
                acc += d * d;
            }
            (acc / m as f64) as f32
        } else {
            centroid_trajectory_mse(&s_ref, &s_cand)
        }
    };

    // ---- Partials → B residual + T60 vector ------------------------------
    let ref_partials = decompose(r, sr, f0_hz, MAX_PARTIALS);
    let cand_partials = decompose(c, sr, f0_hz, MAX_PARTIALS);

    let b_ref = fit_b(&ref_partials).b;
    let b_cand = fit_b(&cand_partials).b;
    let b_residual_raw = (b_cand - b_ref).abs();

    let t60_ref = extract_t60(r, sr, &ref_partials);
    let t60_cand = extract_t60(c, sr, &cand_partials);
    let t60_raw = t60_l2(&t60_ref.seconds, &t60_cand.seconds);

    // ---- Weight & total ---------------------------------------------------
    let mr_stft_term = mr_stft_raw * weights.mr_stft;
    let t60_term = t60_raw * weights.t60;
    let onset_term = onset_raw * weights.onset;
    let centroid_term = centroid_raw * weights.centroid;
    let b_residual_term = b_residual_raw * weights.b_residual;
    let total = mr_stft_term + t60_term + onset_term + centroid_term + b_residual_term;

    LossBreakdown {
        mr_stft_raw,
        mr_stft_term,
        t60_raw,
        t60_term,
        onset_raw,
        onset_term,
        centroid_raw,
        centroid_term,
        b_residual_raw,
        b_residual_term,
        total,
    }
}

// ---- Helpers --------------------------------------------------------------

/// Plain L2 norm of the per-partial T60 difference, treating the
/// `extract::t60` sentinel `-1.0` as "skip this partial". The two vectors
/// are expected to be aligned (same partials list went into both
/// `extract_t60` calls); we still guard against length mismatch by walking
/// the shorter prefix.
fn t60_l2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut acc = 0.0_f64;
    for i in 0..n {
        let av = a[i];
        let bv = b[i];
        if av == T60_SENTINEL || bv == T60_SENTINEL {
            continue;
        }
        if !av.is_finite() || !bv.is_finite() {
            continue;
        }
        let d = (av - bv) as f64;
        acc += d * d;
    }
    (acc.sqrt()) as f32
}

/// Mono WAV reader matching `bin/extract_dump.rs::read_wav_mono`. Returns
/// `(samples, sample_rate_hz)`.
fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, f32), String> {
    let mut r = hound::WavReader::open(path).map_err(|e| e.to_string())?;
    let spec = r.spec();
    let sr = spec.sample_rate as f32;
    let channels = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().filter_map(Result::ok).collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / max)
                .collect()
        }
    };
    if channels <= 1 {
        Ok((samples, sr))
    } else {
        let mut mono = Vec::with_capacity(samples.len() / channels);
        for frame in samples.chunks_exact(channels) {
            let sum: f32 = frame.iter().sum();
            mono.push(sum / channels as f32);
        }
        Ok((mono, sr))
    }
}

// ---- Tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Exponentially-decaying sine — gives `decompose` something to find
    /// and `extract_t60` a usable decay so every component of the loss has
    /// a well-defined value.
    fn synth_decaying_sine(sr: f32, dur_sec: f32, freq: f32, t60: f32) -> Vec<f32> {
        let n = (dur_sec * sr) as usize;
        let omega = 2.0 * PI * freq / sr;
        let k = 3.0_f32 * (10.0_f32).ln() / t60;
        (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                let env = (-k * t).exp();
                0.5 * env * (omega * i as f32).sin()
            })
            .collect()
    }

    #[test]
    fn identity_loss_is_small() {
        let sr = 44100.0_f32;
        let f0 = 261.63_f32;
        let sig = synth_decaying_sine(sr, 2.0, f0, 1.5);
        let w = LossWeights::default();
        let breakdown = loss_signals(&sig, &sig, sr, f0, &w);
        // For a signal vs itself every raw component should be 0 modulo
        // numerical noise from successive STFT calls.
        assert!(
            breakdown.total < 1.0,
            "identity total too high: {} (breakdown {:?})",
            breakdown.total,
            breakdown
        );
        // Spot-check the dominant components individually.
        assert!(breakdown.mr_stft_raw < 1e-3);
        assert!(breakdown.onset_raw < 1e-9);
        assert!(breakdown.b_residual_raw < 1e-9);
    }

    #[test]
    fn pitch_shifted_loss_dominates_identity() {
        let sr = 44100.0_f32;
        let f0 = 261.63_f32;
        let sig_ref = synth_decaying_sine(sr, 2.0, f0, 1.5);
        // +50 cents pitch shift on the candidate.
        let f0_shifted = f0 * 2f32.powf(50.0 / 1200.0);
        let sig_cand = synth_decaying_sine(sr, 2.0, f0_shifted, 1.5);

        let w = LossWeights::default();
        let identity = loss_signals(&sig_ref, &sig_ref, sr, f0, &w);
        let shifted = loss_signals(&sig_ref, &sig_cand, sr, f0, &w);
        assert!(
            shifted.total >= 5.0 * identity.total.max(1e-6),
            "shifted total {} should be >= 5x identity total {}",
            shifted.total,
            identity.total
        );
    }

    #[test]
    fn weight_scaling_is_linear() {
        let sr = 44100.0_f32;
        let f0 = 261.63_f32;
        let sig_ref = synth_decaying_sine(sr, 1.0, f0, 1.0);
        let f0_shifted = f0 * 2f32.powf(50.0 / 1200.0);
        let sig_cand = synth_decaying_sine(sr, 1.0, f0_shifted, 1.0);

        let w_unit = LossWeights {
            mr_stft: 1.0,
            t60: 0.0,
            onset: 0.0,
            centroid: 0.0,
            b_residual: 0.0,
        };
        let w_double = LossWeights {
            mr_stft: 2.0,
            t60: 0.0,
            onset: 0.0,
            centroid: 0.0,
            b_residual: 0.0,
        };
        let unit = loss_signals(&sig_ref, &sig_cand, sr, f0, &w_unit);
        let double = loss_signals(&sig_ref, &sig_cand, sr, f0, &w_double);
        // total == mr_stft_term in both, so the doubled total must be
        // exactly 2x within a tight tolerance.
        let expected = 2.0 * unit.total;
        assert!(
            (double.total - expected).abs() <= 1e-6 * expected.max(1.0),
            "linearity failure: 2x weight gave {} vs expected {}",
            double.total,
            expected
        );
    }
}
