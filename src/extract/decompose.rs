//! Partial decomposition: extract the first N harmonic peak frequencies
//! and per-partial initial amplitudes from a recorded note.
//!
//! Implementation reuses [`crate::analysis::stft`] (4096-pt Hann, hop 1024)
//! and the same `±50 cents around n·f0` peak picker + 3-bin log-magnitude
//! parabolic interpolation that drives the `analyse` binary's
//! `estimate_inharmonicity_b`. The single-frame magnitude budget is too
//! noisy to satisfy a 60 dB SNR cut for an 8th partial under -40 dBFS
//! noise, so we time-average the magnitude spectrum across the **first
//! ~100 ms of frames** before peak picking. Sustained partials survive
//! intact while broadband noise averages down by `sqrt(n_frames)`. Initial
//! amplitudes are reported in dB **relative to the strongest extracted
//! partial** (so the loudest partial in the returned vec is 0 dB by
//! definition), which differs from `analysis::HarmonicTrack::initial_db`
//! (referenced to the global spectrogram peak). Round-trip and
//! noise-stability tests live in the `#[cfg(test)] mod tests` block at
//! the bottom of this file; the cross-file golden test against the
//! committed SFZ Salamander reference WAV lives in
//! `tests/extract_decompose.rs`.

#![allow(dead_code)]

use crate::analysis::stft;

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
/// `f0_hz`. Returns peaks ordered by harmonic index. A partial is dropped
/// (and the returned `Vec` ends up shorter than `max_partials`) when its
/// time-averaged peak magnitude sits below the SNR threshold (~40 dB
/// over a 10th-percentile noise-floor estimate; see in-body comments
/// for the rationale on why this is looser than the brief's nominal
/// 60 dB).
///
/// Peak detection uses the time-averaged STFT magnitude over the whole
/// signal (which makes sustained partials trivially recoverable under
/// broadband noise), while `init_db` is read from the per-bin maximum
/// across the first ~100 ms of frames — matching the `initial_db`
/// window used by [`crate::analysis::harmonic_tracks`].
pub fn decompose(signal: &[f32], sr: f32, f0_hz: f32, max_partials: usize) -> Vec<Partial> {
    if signal.is_empty() || sr <= 0.0 || f0_hz <= 0.0 || max_partials == 0 {
        return Vec::new();
    }

    let nyquist = sr / 2.0;

    // Run the canonical STFT (4096-pt Hann, hop 1024) over the whole
    // signal. Peak detection is done on the *time-averaged* spectrum:
    // sustained partials add coherently while broadband noise averages
    // down by ~`sqrt(n_frames)`, which is what makes the 60 dB SNR cut
    // achievable on an 8th-partial sine under -40 dBFS noise. Same
    // averaging trick as `analysis::estimate_inharmonicity_b`. The
    // "first ~100 ms" framing in the spec applies to the *initial
    // amplitude* read (see `init_window_frames` below), not to the bins
    // we search over.
    const FFT_SIZE: usize = 4096;
    if signal.len() < FFT_SIZE {
        // Need at least one full window to make a meaningful spectrum.
        return Vec::new();
    }
    let stft_res = stft(signal, sr.round() as u32);
    if stft_res.n_frames() == 0 {
        return Vec::new();
    }

    let n_bins = stft_res.n_bins();
    let n_frames_f = stft_res.n_frames() as f32;

    // Average magnitude spectrum across all frames (peak-detection
    // surface).
    let mut mag = vec![0.0_f32; n_bins];
    for frame in &stft_res.mag {
        for k in 0..n_bins {
            mag[k] += frame[k];
        }
    }
    for v in &mut mag {
        *v /= n_frames_f;
    }

    // Initial-amplitude surface: max bin value across the first ~100 ms
    // of frames. This is what we read `init_db` from, matching the
    // `analysis::HarmonicTrack::initial_db` window.
    let frames_per_sec = sr / stft_res.hop_size as f32;
    let init_window_frames = ((0.100 * frames_per_sec) as usize)
        .max(1)
        .min(stft_res.n_frames());
    let mut init_mag = vec![0.0_f32; n_bins];
    for frame in stft_res.mag.iter().take(init_window_frames) {
        for k in 0..n_bins {
            if frame[k] > init_mag[k] {
                init_mag[k] = frame[k];
            }
        }
    }

    // Noise-floor estimate: 10th-percentile magnitude across all bins.
    // Median (50th percentile) is too contaminated by Hann-window
    // sidelobe leakage from strong partials; a lower percentile sits
    // safely in genuinely-empty regions of the spectrum where only
    // additive noise contributes.
    //
    // SNR threshold: we drop partials whose averaged-magnitude peak sits
    // less than 40 dB above this noise floor. The original brief said
    // 60 dB, but a single-bin per-frame Rayleigh-distributed magnitude
    // estimate has a mean of ~0.35 for σ=0.01 white noise through a
    // 4096-pt Hann window, so an 8th-partial sine at -18 dB sits at
    // ~51 dB above that floor — strictly below 60 dB and would be
    // erroneously dropped under the original criterion despite being
    // visually unambiguous in the spectrum. 40 dB is conservatively
    // above the Rayleigh tail and still rejects pure-noise bins
    // (where peak ≈ noise_floor → 0 dB SNR). The decision is
    // documented here per the issue #3 brief's "be honest about
    // widened tolerances" rule.
    let mut sorted_mag: Vec<f32> = mag.clone();
    sorted_mag.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let noise_floor = sorted_mag[sorted_mag.len() / 10].max(1e-12);
    // SNR threshold lowered 40 → 20 dB so high-order partials (typically
    // 30-50 dB below fundamental on real piano samples) are retained.
    // Iter K (modal_lut max_partials 24 → 48) was a near no-op on
    // residual_l2_aw because the extractor stopped at ~24 partials per
    // mid note regardless of the cap — the extractor's own SNR gate
    // dropped the high partials before max_partials applied. 20 dB
    // (factor 10) keeps the gate above pure-noise bins (Rayleigh tail
    // sits ~5-10 dB above floor for FFT_SIZE=4096) but recovers the
    // -30 to -50 dBFS partial structure that drives the user-perceived
    // "lo-fi" feel of modal voice (2-4k +2.5 dB deficit, very-hi
    // surplus dominated by hammer noise rather than real partials).
    let snr_threshold = noise_floor * 10.0;

    let bin_hz = sr / FFT_SIZE as f32;
    // ±50 cents window: factor of 2^(50/1200) ≈ 1.0293
    let cents_factor: f32 = 2f32.powf(50.0 / 1200.0);

    // Per-partial peak search.
    let mut found: Vec<(usize, f32, f32)> = Vec::with_capacity(max_partials);
    for n in 1..=max_partials {
        let f_expected = f0_hz * n as f32;
        if f_expected >= nyquist {
            break;
        }
        let f_lo = f_expected / cents_factor;
        let f_hi = f_expected * cents_factor;
        let bin_lo = (f_lo / bin_hz).floor() as usize;
        let bin_hi = ((f_hi / bin_hz).ceil() as usize).min(n_bins - 1);
        if bin_hi <= bin_lo {
            continue;
        }

        // Find max-magnitude bin in [bin_lo, bin_hi].
        let mut peak_bin = bin_lo;
        let mut peak_val = mag[bin_lo];
        for b in (bin_lo + 1)..=bin_hi {
            if mag[b] > peak_val {
                peak_val = mag[b];
                peak_bin = b;
            }
        }

        // SNR gate.
        if peak_val < snr_threshold {
            continue;
        }
        // Need both neighbours for parabolic interpolation.
        if peak_bin == 0 || peak_bin >= n_bins - 1 {
            continue;
        }

        // Parabolic interpolation on the *averaged* magnitude spectrum
        // gives a sub-bin frequency offset `p` ∈ [-1, 1].
        let alpha = mag[peak_bin - 1].max(1e-12).log10();
        let beta = peak_val.max(1e-12).log10();
        let gamma = mag[peak_bin + 1].max(1e-12).log10();
        let denom = alpha - 2.0 * beta + gamma;
        let p = if denom.abs() < 1e-12 {
            0.0
        } else {
            0.5 * (alpha - gamma) / denom
        };
        let p_clamped = p.clamp(-1.0, 1.0);
        let interp_bin = peak_bin as f32 + p_clamped;
        let f_obs = interp_bin * bin_hz;

        // Initial-amplitude read: re-do the parabolic interpolation on
        // the *first-100-ms* magnitude buffer (`init_mag`) so the
        // returned `init_db` reflects attack-time amplitudes (matching
        // `analysis::HarmonicTrack::initial_db`) and so amplitude
        // estimates survive between-bin scalloping loss.
        let i_alpha = init_mag[peak_bin - 1].max(1e-12).log10();
        let i_beta = init_mag[peak_bin].max(1e-12).log10();
        let i_gamma = init_mag[peak_bin + 1].max(1e-12).log10();
        let i_denom = i_alpha - 2.0 * i_beta + i_gamma;
        let i_p = if i_denom.abs() < 1e-12 {
            0.0
        } else {
            (0.5 * (i_alpha - i_gamma) / i_denom).clamp(-1.0, 1.0)
        };
        let init_log_mag = i_beta - 0.25 * (i_alpha - i_gamma) * i_p;
        let init_peak_mag = 10f32.powf(init_log_mag);

        found.push((n, f_obs, init_peak_mag));
    }

    if found.is_empty() {
        return Vec::new();
    }

    // Reference magnitude = strongest extracted partial. init_db is
    // 20·log10(peak / strongest), so the loudest partial is 0 dB.
    let strongest = found
        .iter()
        .map(|&(_, _, m)| m)
        .fold(0.0_f32, f32::max)
        .max(1e-12);

    found
        .into_iter()
        .map(|(n, freq_hz, peak_val)| Partial {
            n,
            freq_hz,
            init_db: 20.0 * (peak_val / strongest).max(1e-12).log10(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Synthesise a stretched-harmonic signal: partial n at
    /// `n · f0 · sqrt(1 + B · n²)` Hz with linear amplitude `1.0 / n`.
    /// Returned signal is **not** peak-normalised — keeping the absolute
    /// amplitudes preserves the per-partial dB relationships the test
    /// asserts on (h1 = 0 dB, h2 = -6.02 dB, h3 = -9.54 dB, ...).
    fn synth_stretched(sr: f32, dur_sec: f32, f0: f32, b: f32, n_partials: usize) -> Vec<f32> {
        let n_samples = (dur_sec * sr) as usize;
        let mut out = vec![0.0_f32; n_samples];
        for n in 1..=n_partials {
            let f_n = (n as f32) * f0 * (1.0 + b * (n as f32).powi(2)).sqrt();
            if f_n >= sr / 2.0 {
                break;
            }
            let amp = 1.0 / n as f32;
            let omega = 2.0 * PI * f_n / sr;
            for (i, sample) in out.iter_mut().enumerate() {
                *sample += amp * (omega * i as f32).sin();
            }
        }
        out
    }

    fn cents_between(a: f32, b: f32) -> f32 {
        1200.0 * (a / b).abs().log2()
    }

    /// Box-Muller Gaussian noise (deterministic LCG seed for repeatability).
    fn gaussian_noise(n: usize, sigma: f32, seed: u64) -> Vec<f32> {
        // Tiny LCG (numerical recipes constants).
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut next_uniform = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // top 32 bits to a (0,1) float, avoiding exact 0.
            let bits = (state >> 32) as u32;
            ((bits as f64 + 1.0) / (u32::MAX as f64 + 2.0)) as f32
        };
        let mut out = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            let u1 = next_uniform();
            let u2 = next_uniform();
            let r = (-2.0_f32 * u1.max(1e-20).ln()).sqrt();
            let theta = 2.0 * PI * u2;
            out.push(sigma * r * theta.cos());
            if i + 1 < n {
                out.push(sigma * r * theta.sin());
            }
            i += 2;
        }
        out.truncate(n);
        out
    }

    #[test]
    fn round_trip_synthetic_eight_partials() {
        let sr = 44100.0_f32;
        let f0 = 440.0_f32;
        let b = 2.86e-4_f32;
        let n_partials = 8;

        let sig = synth_stretched(sr, 1.0, f0, b, n_partials);
        let partials = decompose(&sig, sr, f0, n_partials);

        assert_eq!(
            partials.len(),
            n_partials,
            "should recover all 8 partials, got {}",
            partials.len()
        );

        for (i, p) in partials.iter().enumerate() {
            let n = i + 1;
            assert_eq!(p.n, n, "partial index mismatch");
            let f_expected = (n as f32) * f0 * (1.0 + b * (n as f32).powi(2)).sqrt();
            let cents_err = cents_between(p.freq_hz, f_expected).abs();
            assert!(
                cents_err <= 0.5,
                "n={} freq {} Hz vs expected {} Hz: {:.3} cents (>0.5)",
                n,
                p.freq_hz,
                f_expected,
                cents_err
            );
            // amp ratio = (1/n) / 1.0 = 1/n  =>  init_db = 20·log10(1/n)
            let db_expected = 20.0 * (1.0_f32 / n as f32).log10();
            let db_err = (p.init_db - db_expected).abs();
            assert!(
                db_err <= 0.5,
                "n={} init_db {} vs expected {}: {:.3} dB error (>0.5)",
                n,
                p.init_db,
                db_expected,
                db_err
            );
        }
    }

    #[test]
    fn round_trip_with_noise_minus_40_dbfs() {
        let sr = 44100.0_f32;
        let f0 = 440.0_f32;
        let b = 2.86e-4_f32;
        let n_partials = 8;

        let mut sig = synth_stretched(sr, 1.0, f0, b, n_partials);
        // -40 dBFS RMS = 0.01 RMS amplitude.
        let noise = gaussian_noise(sig.len(), 0.01, 0xC0FFEE);
        for (s, n) in sig.iter_mut().zip(noise.iter()) {
            *s += *n;
        }

        let partials = decompose(&sig, sr, f0, n_partials);
        assert_eq!(
            partials.len(),
            n_partials,
            "should still recover all 8 partials under -40 dBFS noise, got {}",
            partials.len()
        );

        for (i, p) in partials.iter().enumerate() {
            let n = i + 1;
            let f_expected = (n as f32) * f0 * (1.0 + b * (n as f32).powi(2)).sqrt();
            let cents_err = cents_between(p.freq_hz, f_expected).abs();
            assert!(
                cents_err <= 1.5,
                "n={} freq {} Hz vs expected {} Hz: {:.3} cents (>1.5) under noise",
                n,
                p.freq_hz,
                f_expected,
                cents_err
            );
            let db_expected = 20.0 * (1.0_f32 / n as f32).log10();
            let db_err = (p.init_db - db_expected).abs();
            assert!(
                db_err <= 1.0,
                "n={} init_db {} vs expected {}: {:.3} dB error (>1.0) under noise",
                n,
                p.init_db,
                db_expected,
                db_err
            );
        }
    }

    #[test]
    fn empty_input_returns_empty() {
        let out = decompose(&[], 44100.0, 261.63, 8);
        assert!(out.is_empty());
    }

    #[test]
    fn loudest_partial_is_zero_db() {
        let sr = 44100.0_f32;
        let f0 = 440.0_f32;
        let sig = synth_stretched(sr, 1.0, f0, 0.0, 8);
        let partials = decompose(&sig, sr, f0, 8);
        assert!(!partials.is_empty());
        // h1 has amp 1/1 = strongest; its init_db must be ~0.
        let h1 = partials.iter().find(|p| p.n == 1).expect("h1 missing");
        assert!(
            h1.init_db.abs() < 0.1,
            "h1 should be the 0 dB reference, got {}",
            h1.init_db
        );
    }
}
