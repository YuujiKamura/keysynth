//! Audio bytes audit: numerical metrics over PCM f32 samples.
//!
//! The point of this module is to let an AI agent (which cannot listen)
//! decide "is this WAV qualitatively different from the previous render?"
//! using nothing but numbers. Every function here returns a scalar or a
//! short fixed-shape vector that's safe to JSON-serialize and diff.
//!
//! Coverage:
//! - level: [`peak_dbfs`], [`rms_dbfs`], [`crest_factor_db`]
//! - structure: [`silence_intervals`], [`rms_envelope`]
//! - spectrum: [`spectral_envelope`], [`dominant_frequencies`]
//! - similarity (two signals): [`cross_correlate`], [`mr_stft_distance`],
//!   [`spectral_envelope_diff_db`]
//!
//! Heavy spectral primitives (STFT, multi-resolution STFT distance) are
//! reused from [`crate::analysis`] so we don't fork the FFT path.

use std::f32::consts::PI;

use rustfft::{num_complex::Complex32, FftPlanner};
use serde::Serialize;

use crate::analysis::{mr_stft_l1, stft};

// ---- Level -----------------------------------------------------------------

/// Peak amplitude in dBFS (full-scale = 0 dB). `-inf` for true silence.
pub fn peak_dbfs(samples: &[f32]) -> f32 {
    let peak = samples
        .iter()
        .copied()
        .fold(0.0_f32, |acc, v| acc.max(v.abs()));
    if peak <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * peak.log10()
    }
}

/// Root-mean-square level in dBFS. `-inf` for empty / all-zero input.
pub fn rms_dbfs(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mut acc = 0.0_f64;
    for &s in samples {
        acc += (s as f64) * (s as f64);
    }
    let rms = (acc / samples.len() as f64).sqrt();
    if rms <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * (rms as f32).log10()
    }
}

/// Crest factor in dB: `peak_dbfs - rms_dbfs`. High = transient-rich
/// (drum hits, plucks), low = steady tone or heavily compressed.
pub fn crest_factor_db(samples: &[f32]) -> f32 {
    let p = peak_dbfs(samples);
    let r = rms_dbfs(samples);
    if !p.is_finite() || !r.is_finite() {
        return f32::NAN;
    }
    p - r
}

// ---- Structure -------------------------------------------------------------

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct SilenceInterval {
    pub start_sec: f32,
    pub end_sec: f32,
    pub duration_sec: f32,
}

/// Detect contiguous silent regions: spans where `|sample| < 10^(threshold/20)`
/// for at least `min_ms` milliseconds. Useful to spot dropouts, premature
/// note-offs, or unintended trailing silence in a render.
pub fn silence_intervals(
    samples: &[f32],
    sr: u32,
    threshold_dbfs: f32,
    min_ms: u32,
) -> Vec<SilenceInterval> {
    if samples.is_empty() || sr == 0 {
        return Vec::new();
    }
    let lin_threshold = 10f32.powf(threshold_dbfs / 20.0);
    let min_samples = ((min_ms as f32 / 1000.0) * sr as f32) as usize;
    let mut out = Vec::new();
    let mut run_start: Option<usize> = None;
    for (i, &s) in samples.iter().enumerate() {
        if s.abs() < lin_threshold {
            if run_start.is_none() {
                run_start = Some(i);
            }
        } else if let Some(start) = run_start.take() {
            if i - start >= min_samples {
                out.push(make_interval(start, i, sr));
            }
        }
    }
    if let Some(start) = run_start {
        if samples.len() - start >= min_samples {
            out.push(make_interval(start, samples.len(), sr));
        }
    }
    out
}

fn make_interval(start: usize, end: usize, sr: u32) -> SilenceInterval {
    let s = start as f32 / sr as f32;
    let e = end as f32 / sr as f32;
    SilenceInterval {
        start_sec: s,
        end_sec: e,
        duration_sec: e - s,
    }
}

/// Time-domain RMS envelope sampled every `window_ms` ms (non-overlapping).
/// Use this to compare dynamics shape between two renders without caring
/// about absolute level.
pub fn rms_envelope(samples: &[f32], sr: u32, window_ms: u32) -> Vec<f32> {
    if samples.is_empty() || sr == 0 || window_ms == 0 {
        return Vec::new();
    }
    let win = ((window_ms as f32 / 1000.0) * sr as f32) as usize;
    if win == 0 {
        return Vec::new();
    }
    let n_frames = (samples.len() + win - 1) / win;
    let mut out = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let start = f * win;
        let end = (start + win).min(samples.len());
        let n = end - start;
        if n == 0 {
            break;
        }
        let mut acc = 0.0_f64;
        for &s in &samples[start..end] {
            acc += (s as f64) * (s as f64);
        }
        out.push((acc / n as f64).sqrt() as f32);
    }
    out
}

// ---- Spectrum --------------------------------------------------------------

#[derive(Serialize, Debug, Clone)]
pub struct SpectralEnvelope {
    pub n_bands: usize,
    pub sr: u32,
    /// Linear average magnitude in each band (DC..Nyquist split into
    /// `n_bands` equal-Hz slices over a single full-signal FFT).
    pub bands: Vec<f32>,
    /// Lower edge of each band in Hz.
    pub band_edges_hz: Vec<f32>,
}

/// Single-shot magnitude spectrum of the whole signal, then averaged into
/// `n_bands` equal-width bins from 0 Hz to Nyquist. Cheap to compute and
/// stable to compare across renders of the same length.
pub fn spectral_envelope(samples: &[f32], sr: u32, n_bands: usize) -> SpectralEnvelope {
    if samples.is_empty() || sr == 0 || n_bands == 0 {
        return SpectralEnvelope {
            n_bands,
            sr,
            bands: vec![0.0; n_bands],
            band_edges_hz: (0..n_bands).map(|i| i as f32).collect(),
        };
    }

    let n = samples.len().next_power_of_two().min(1 << 16).max(1024);
    let mut buf: Vec<Complex32> = (0..n)
        .map(|i| {
            let s = if i < samples.len() { samples[i] } else { 0.0 };
            // Hann window over the active region.
            let w = if !samples.is_empty() {
                let active = samples.len().min(n) as f32;
                if (i as f32) < active && active > 1.0 {
                    0.5 - 0.5 * (2.0 * PI * i as f32 / (active - 1.0)).cos()
                } else {
                    0.0
                }
            } else {
                0.0
            };
            Complex32::new(s * w, 0.0)
        })
        .collect();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buf);

    let half = n / 2 + 1;
    let bin_hz = sr as f32 / n as f32;
    let nyquist = sr as f32 / 2.0;
    let band_hz = nyquist / n_bands as f32;

    let mut bands = vec![0.0_f64; n_bands];
    let mut counts = vec![0_usize; n_bands];
    for k in 0..half {
        let f = k as f32 * bin_hz;
        let band = ((f / band_hz) as usize).min(n_bands - 1);
        bands[band] += buf[k].norm() as f64;
        counts[band] += 1;
    }
    let bands_f32: Vec<f32> = bands
        .iter()
        .zip(&counts)
        .map(|(&s, &c)| if c == 0 { 0.0 } else { (s / c as f64) as f32 })
        .collect();
    let edges: Vec<f32> = (0..n_bands).map(|i| i as f32 * band_hz).collect();

    SpectralEnvelope {
        n_bands,
        sr,
        bands: bands_f32,
        band_edges_hz: edges,
    }
}

#[derive(Serialize, Debug, Clone)]
pub struct DominantFrequency {
    pub freq_hz: f32,
    pub magnitude: f32,
    pub magnitude_db: f32,
}

/// Top-N peak frequencies of the whole-signal magnitude spectrum, with
/// parabolic interpolation for sub-bin resolution. Peaks are picked by
/// local maxima in the magnitude spectrum and are spaced apart by at
/// least 1 FFT bin.
pub fn dominant_frequencies(samples: &[f32], sr: u32, top_n: usize) -> Vec<DominantFrequency> {
    if samples.is_empty() || sr == 0 || top_n == 0 {
        return Vec::new();
    }
    let n = samples.len().next_power_of_two().min(1 << 16).max(1024);
    let active = samples.len().min(n);
    let mut buf: Vec<Complex32> = (0..n)
        .map(|i| {
            if i < active {
                let w = if active > 1 {
                    0.5 - 0.5 * (2.0 * PI * i as f32 / (active as f32 - 1.0)).cos()
                } else {
                    1.0
                };
                Complex32::new(samples[i] * w, 0.0)
            } else {
                Complex32::new(0.0, 0.0)
            }
        })
        .collect();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buf);

    let half = n / 2 + 1;
    let bin_hz = sr as f32 / n as f32;
    let mags: Vec<f32> = buf[..half].iter().map(|c| c.norm()).collect();

    // Find local maxima (strict on the inside, allow plateau-tolerant at edges).
    let mut peaks: Vec<(usize, f32)> = Vec::new();
    for k in 1..half - 1 {
        if mags[k] > mags[k - 1] && mags[k] > mags[k + 1] {
            peaks.push((k, mags[k]));
        }
    }
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    peaks.truncate(top_n);

    let global_peak = mags.iter().copied().fold(0.0_f32, f32::max).max(1e-12);
    peaks
        .into_iter()
        .map(|(k, m)| {
            // Parabolic interpolation in log-magnitude.
            let alpha = mags[k - 1].max(1e-12).log10();
            let beta = m.max(1e-12).log10();
            let gamma = mags[k + 1].max(1e-12).log10();
            let denom = alpha - 2.0 * beta + gamma;
            let p = if denom.abs() < 1e-12 {
                0.0
            } else {
                0.5 * (alpha - gamma) / denom
            };
            let freq = (k as f32 + p.clamp(-1.0, 1.0)) * bin_hz;
            DominantFrequency {
                freq_hz: freq,
                magnitude: m,
                magnitude_db: 20.0 * (m / global_peak).max(1e-12).log10(),
            }
        })
        .collect()
}

// ---- Similarity (two signals) ---------------------------------------------

#[derive(Serialize, Debug, Clone)]
pub struct CrossCorrelation {
    /// Best-match peak of the (mean-removed, length-normalised) cross-corr
    /// in the range [-1, 1]. 1 = identical shape, 0 = uncorrelated, -1 = inverted.
    pub peak: f32,
    /// Lag of the peak in samples (positive: `b` is delayed relative to `a`).
    pub lag_samples: i32,
    /// Same lag, in seconds (requires `sr`; pass 0 to skip).
    pub lag_sec: f32,
}

/// FFT-based normalised cross-correlation. R[k] = sum_i a[i]*b[i+k] /
/// (||a||*||b||). Positive lag k means `b` is `a` delayed by k samples
/// (b[i+k] re-aligns with a[i]). Returns the lag at which |R| is largest.
///
/// `sr = 0` skips the lag-in-seconds calculation.
pub fn cross_correlate(a: &[f32], b: &[f32], sr: u32) -> CrossCorrelation {
    if a.is_empty() || b.is_empty() {
        return CrossCorrelation {
            peak: 0.0,
            lag_samples: 0,
            lag_sec: 0.0,
        };
    }
    // Mean-remove so DC offset doesn't dominate the correlation.
    let mean_a: f32 = a.iter().sum::<f32>() / a.len() as f32;
    let mean_b: f32 = b.iter().sum::<f32>() / b.len() as f32;
    let za: Vec<f32> = a.iter().map(|x| x - mean_a).collect();
    let zb: Vec<f32> = b.iter().map(|x| x - mean_b).collect();

    let norm_a = (za.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-12);
    let norm_b = (zb.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-12);

    let na = za.len();
    let nb = zb.len();
    // Linear (non-cyclic) cross-correlation needs N >= na + nb - 1.
    let n = (na + nb - 1).next_power_of_two();
    let mut buf_a: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(if i < na { za[i] } else { 0.0 }, 0.0))
        .collect();
    let mut buf_b: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(if i < nb { zb[i] } else { 0.0 }, 0.0))
        .collect();

    let mut planner = FftPlanner::<f32>::new();
    let fwd = planner.plan_fft_forward(n);
    let inv = planner.plan_fft_inverse(n);
    fwd.process(&mut buf_a);
    fwd.process(&mut buf_b);

    // R = ifft(A .* conj(B)); element-wise multiply with B conjugated.
    let mut prod: Vec<Complex32> = buf_a
        .iter()
        .zip(&buf_b)
        .map(|(av, bv)| av * bv.conj())
        .collect();
    inv.process(&mut prod);

    // ifft(A * conj(B))[idx] = R[-idx] (cyclic). With zero-padding, the
    // valid lag mapping is:
    //   idx = 0                  -> lag 0
    //   idx in [1, na - 1]       -> lag = -idx (a delayed)
    //   idx in [n - nb + 1, n)   -> lag = n - idx (b delayed)
    let scale = 1.0 / (n as f32 * norm_a * norm_b);
    let mut best_peak = 0.0_f32;
    let mut best_lag: i32 = 0;
    let mut best_abs = -1.0_f32;
    let mut consider = |idx: usize, lag: i32| {
        let val = prod[idx].re * scale;
        if val.abs() > best_abs {
            best_abs = val.abs();
            best_peak = val;
            best_lag = lag;
        }
    };
    consider(0, 0);
    if na > 1 {
        for idx in 1..na {
            consider(idx, -(idx as i32));
        }
    }
    if nb > 1 {
        for idx in (n - nb + 1)..n {
            let lag = n as i32 - idx as i32;
            consider(idx, lag);
        }
    }

    let lag_sec = if sr > 0 {
        best_lag as f32 / sr as f32
    } else {
        0.0
    };
    CrossCorrelation {
        peak: best_peak,
        lag_samples: best_lag,
        lag_sec,
    }
}

/// Multi-resolution STFT distance — wrapper around [`crate::analysis::mr_stft_l1`].
/// Higher = more dissimilar in timbre/spectrum. Identical inputs return ~0.
pub fn mr_stft_distance(a: &[f32], b: &[f32], sr: u32) -> f32 {
    mr_stft_l1(a, b, sr)
}

/// Per-band log-magnitude distance between two signals' spectral envelopes.
/// Returns RMS dB difference across the bands. Both inputs are envelope'd
/// independently so it's tolerant of small length / phase differences.
pub fn spectral_envelope_diff_db(a: &[f32], b: &[f32], sr: u32, n_bands: usize) -> f32 {
    let ea = spectral_envelope(a, sr, n_bands);
    let eb = spectral_envelope(b, sr, n_bands);
    if ea.bands.is_empty() || eb.bands.is_empty() {
        return 0.0;
    }
    let n = ea.bands.len().min(eb.bands.len());
    let mut sum_sq = 0.0_f64;
    let mut count = 0_usize;
    for i in 0..n {
        let la = 20.0 * (ea.bands[i].max(1e-12) as f64).log10();
        let lb = 20.0 * (eb.bands[i].max(1e-12) as f64).log10();
        let d = la - lb;
        sum_sq += d * d;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        ((sum_sq / count as f64).sqrt()) as f32
    }
}

// ---- Aggregate report ------------------------------------------------------

#[derive(Serialize, Debug, Clone)]
pub struct AuditReport {
    pub sr: u32,
    pub n_samples: usize,
    pub duration_sec: f32,
    pub peak_dbfs: f32,
    pub rms_dbfs: f32,
    pub crest_factor_db: f32,
    pub silence_intervals: Vec<SilenceInterval>,
    pub rms_envelope: Vec<f32>,
    pub rms_envelope_window_ms: u32,
    pub spectral_envelope: SpectralEnvelope,
    pub dominant_frequencies: Vec<DominantFrequency>,
}

/// One-shot audit covering every single-signal metric in this module.
pub fn audit(
    samples: &[f32],
    sr: u32,
    silence_threshold_dbfs: f32,
    silence_min_ms: u32,
    rms_window_ms: u32,
    n_envelope_bands: usize,
    top_n_freqs: usize,
) -> AuditReport {
    AuditReport {
        sr,
        n_samples: samples.len(),
        duration_sec: if sr > 0 {
            samples.len() as f32 / sr as f32
        } else {
            0.0
        },
        peak_dbfs: peak_dbfs(samples),
        rms_dbfs: rms_dbfs(samples),
        crest_factor_db: crest_factor_db(samples),
        silence_intervals: silence_intervals(samples, sr, silence_threshold_dbfs, silence_min_ms),
        rms_envelope: rms_envelope(samples, sr, rms_window_ms),
        rms_envelope_window_ms: rms_window_ms,
        spectral_envelope: spectral_envelope(samples, sr, n_envelope_bands),
        dominant_frequencies: dominant_frequencies(samples, sr, top_n_freqs),
    }
}

#[derive(Serialize, Debug, Clone)]
pub struct DiffReport {
    pub a_n_samples: usize,
    pub b_n_samples: usize,
    pub sr: u32,
    pub cross_correlation: CrossCorrelation,
    pub mr_stft_distance: f32,
    pub spectral_envelope_diff_db: f32,
    pub peak_dbfs_delta: f32,
    pub rms_dbfs_delta: f32,
}

/// Two-signal diff. `peak_dbfs_delta` and `rms_dbfs_delta` are `b - a`
/// in dB — positive means `b` is louder than `a`.
pub fn diff(a: &[f32], b: &[f32], sr: u32, n_envelope_bands: usize) -> DiffReport {
    let xc = cross_correlate(a, b, sr);
    let mrstft = mr_stft_distance(a, b, sr);
    let env_db = spectral_envelope_diff_db(a, b, sr, n_envelope_bands);
    let pa = peak_dbfs(a);
    let pb = peak_dbfs(b);
    let ra = rms_dbfs(a);
    let rb = rms_dbfs(b);
    let peak_delta = if pa.is_finite() && pb.is_finite() {
        pb - pa
    } else {
        f32::NAN
    };
    let rms_delta = if ra.is_finite() && rb.is_finite() {
        rb - ra
    } else {
        f32::NAN
    };
    DiffReport {
        a_n_samples: a.len(),
        b_n_samples: b.len(),
        sr,
        cross_correlation: xc,
        mr_stft_distance: mrstft,
        spectral_envelope_diff_db: env_db,
        peak_dbfs_delta: peak_delta,
        rms_dbfs_delta: rms_delta,
    }
}

// Quiet the unused-import lint when building without the `stft` consumer below.
#[allow(dead_code)]
fn _ensure_stft_used(samples: &[f32], sr: u32) {
    let _ = stft(samples, sr);
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn synth_sine(sr: u32, dur_sec: f32, freq: f32, amp: f32) -> Vec<f32> {
        let n = (dur_sec * sr as f32) as usize;
        let omega = 2.0 * PI * freq / sr as f32;
        (0..n).map(|i| amp * (omega * i as f32).sin()).collect()
    }

    #[test]
    fn peak_dbfs_full_scale_is_zero() {
        let mut sig = vec![0.0_f32; 100];
        sig[10] = 1.0;
        sig[20] = -1.0;
        let p = peak_dbfs(&sig);
        assert!((p - 0.0).abs() < 1e-5, "peak should be 0 dBFS, got {p}");
    }

    #[test]
    fn peak_dbfs_silence_is_neg_inf() {
        let sig = vec![0.0_f32; 100];
        assert_eq!(peak_dbfs(&sig), f32::NEG_INFINITY);
    }

    #[test]
    fn rms_dbfs_full_scale_sine_is_minus_3() {
        let sig = synth_sine(44100, 0.5, 440.0, 1.0);
        let r = rms_dbfs(&sig);
        // sin RMS = 1/sqrt(2) -> -3.0103 dB
        assert!((r + 3.0103).abs() < 0.05, "expected ~-3 dB, got {r}");
    }

    #[test]
    fn crest_factor_sine_is_3db() {
        let sig = synth_sine(44100, 0.5, 440.0, 0.5);
        let c = crest_factor_db(&sig);
        assert!((c - 3.0103).abs() < 0.1, "sine crest ~3 dB, got {c}");
    }

    #[test]
    fn silence_intervals_finds_gap() {
        let sr = 44100;
        let mut sig = synth_sine(sr, 0.2, 440.0, 0.5);
        let silence = vec![0.0_f32; (0.1 * sr as f32) as usize];
        sig.extend_from_slice(&silence);
        sig.extend(synth_sine(sr, 0.2, 440.0, 0.5));
        let ivs = silence_intervals(&sig, sr, -60.0, 50);
        assert!(!ivs.is_empty(), "should find at least one silence");
        let total: f32 = ivs.iter().map(|i| i.duration_sec).sum();
        assert!(
            total > 0.05 && total < 0.15,
            "silence total ~0.1s, got {total}"
        );
    }

    #[test]
    fn silence_intervals_empty_for_loud_signal() {
        let sr = 44100;
        let sig = synth_sine(sr, 0.5, 440.0, 0.9);
        let ivs = silence_intervals(&sig, sr, -60.0, 20);
        assert!(ivs.is_empty(), "no silence expected, got {:?}", ivs);
    }

    #[test]
    fn rms_envelope_constant_for_steady_sine() {
        let sr = 44100;
        let sig = synth_sine(sr, 0.5, 440.0, 0.5);
        let env = rms_envelope(&sig, sr, 50);
        assert!(env.len() >= 5);
        let mid = env[env.len() / 2];
        for &v in env.iter().take(env.len() - 1).skip(1) {
            assert!(
                (v - mid).abs() < 0.05,
                "steady sine envelope should be flat, got {v} vs {mid}"
            );
        }
    }

    #[test]
    fn dominant_frequencies_finds_440() {
        let sr = 44100;
        let sig = synth_sine(sr, 0.3, 440.0, 0.5);
        let peaks = dominant_frequencies(&sig, sr, 3);
        assert!(!peaks.is_empty());
        // Top peak should be near 440 Hz.
        let top = &peaks[0];
        assert!(
            (top.freq_hz - 440.0).abs() < 10.0,
            "top peak should be near 440 Hz, got {}",
            top.freq_hz
        );
    }

    #[test]
    fn cross_correlate_identity_is_one_at_zero_lag() {
        let sr = 44100;
        let sig = synth_sine(sr, 0.1, 440.0, 0.5);
        let xc = cross_correlate(&sig, &sig, sr);
        assert!(
            (xc.peak - 1.0).abs() < 1e-3,
            "self-corr peak should be ~1, got {}",
            xc.peak
        );
        assert_eq!(xc.lag_samples, 0, "self-corr lag should be 0");
    }

    #[test]
    fn cross_correlate_detects_known_lag() {
        let sr = 1000;
        let a = synth_sine(sr, 0.1, 50.0, 0.5);
        // Shift b right by 7 samples: prepend 7 zeros, drop tail.
        let mut b = vec![0.0_f32; 7];
        b.extend_from_slice(&a[..a.len() - 7]);
        let xc = cross_correlate(&a, &b, sr);
        assert_eq!(xc.lag_samples, 7, "expected lag 7, got {}", xc.lag_samples);
    }

    #[test]
    fn mr_stft_distance_zero_for_identity() {
        let sr = 44100;
        let sig = synth_sine(sr, 0.2, 440.0, 0.5);
        let d = mr_stft_distance(&sig, &sig, sr);
        assert!(d < 1e-4, "identity mr_stft should be ~0, got {d}");
    }

    #[test]
    fn spectral_envelope_diff_zero_for_identity() {
        let sr = 44100;
        let sig = synth_sine(sr, 0.3, 440.0, 0.5);
        let d = spectral_envelope_diff_db(&sig, &sig, sr, 16);
        assert!(d < 1e-3, "identity envelope diff should be ~0, got {d}");
    }

    #[test]
    fn spectral_envelope_diff_nonzero_for_different_freqs() {
        let sr = 44100;
        let a = synth_sine(sr, 0.3, 220.0, 0.5);
        let b = synth_sine(sr, 0.3, 4000.0, 0.5);
        let d = spectral_envelope_diff_db(&a, &b, sr, 32);
        assert!(d > 5.0, "expected sizeable diff, got {d}");
    }

    #[test]
    fn audit_report_roundtrips_to_json() {
        let sr = 44100;
        let sig = synth_sine(sr, 0.2, 440.0, 0.5);
        let report = audit(&sig, sr, -60.0, 20, 50, 16, 4);
        let s = serde_json::to_string(&report).unwrap();
        assert!(s.contains("peak_dbfs"));
        assert!(s.contains("dominant_frequencies"));
    }

    #[test]
    fn diff_report_roundtrips_to_json() {
        let sr = 44100;
        let a = synth_sine(sr, 0.2, 440.0, 0.5);
        let b = synth_sine(sr, 0.2, 880.0, 0.5);
        let r = diff(&a, &b, sr, 16);
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains("mr_stft_distance"));
        assert!(s.contains("cross_correlation"));
    }
}
