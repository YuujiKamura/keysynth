//! Hammer attack envelope characterisation.
//!
//! Computes a per-hop RMS envelope over the first ~100 ms of a recorded
//! note, finds the peak (time-to-peak / peak level), and fits a linear
//! dB-vs-time slope just past the peak so callers can compare a candidate
//! render's hammer attack & immediate post-strike decay to a reference.
//!
//! ## Algorithm
//! - Sliding RMS, **window = 5 ms (220 samples @ 44.1 kHz), hop = 1 ms
//!   (44 samples)**. 5 ms is a tradeoff: long enough to average a few
//!   cycles of even a low piano note (~165 samples for A0 @ 27.5 Hz is
//!   over the limit, but for keysynth's typical mid/treble targets like
//!   C4 = 261.6 Hz, 5 ms covers ~1.3 cycles), short enough that the
//!   peak-time isn't smeared by more than ~2.5 ms.
//! - Each RMS sample → dB FS via `20·log10(rms.max(1e-10))`.
//! - `time_to_peak_s` = `(first hop within 0.5 dB of the global max) ·
//!   0.001`. The "first within EPS" rule keeps a steady plateau (or a
//!   silence-then-tone signal) from reporting an arbitrary mid-plateau
//!   hop just because windowed-RMS wobbles by a fraction of a dB
//!   between hop positions.
//! - `peak_db` = dB at that chosen hop.
//! - `post_peak_slope_db_s`: ordinary least-squares fit of dB vs t over
//!   `[time_to_peak + 30 ms, time_to_peak + 80 ms]`. **Fallback**: if
//!   the primary window doesn't fit inside the analysis window
//!   (`window_ms`), we fall back to `[time_to_peak + 20 ms, window_ms]`,
//!   and finally to `[time_to_peak + 0, window_ms]`. If even that yields
//!   fewer than 2 samples, slope is reported as 0.0.

#![allow(dead_code)]

/// Summary of the first ~100 ms of a recorded note.
#[derive(Clone, Debug)]
pub struct AttackEnvelope {
    /// Time from note onset to peak amplitude (seconds).
    pub time_to_peak_s: f32,
    /// Peak RMS level in dB (relative to full-scale).
    pub peak_db: f32,
    /// Slope (dB/sec) of the linear fit over the 30..80 ms window
    /// (with documented fallbacks if the window doesn't fit).
    pub post_peak_slope_db_s: f32,
    /// Per-hop RMS envelope in dB FS over the analysed window
    /// (hop = 1 ms). Useful for plotting / external analysis.
    pub rms_envelope_db: Vec<f32>,
}

/// Extract the first `window_ms` of attack characterisation from a mono
/// signal. Default window is 100 ms; RMS uses a 5 ms window with a 1 ms
/// hop. If the input signal is shorter than `window_ms`, only the
/// available portion is analysed.
pub fn extract_attack(signal: &[f32], sr: f32, window_ms: f32) -> AttackEnvelope {
    // Guard: empty input → return zeros so callers don't crash.
    if signal.is_empty() || sr <= 0.0 || window_ms <= 0.0 {
        return AttackEnvelope {
            time_to_peak_s: 0.0,
            peak_db: -200.0,
            post_peak_slope_db_s: 0.0,
            rms_envelope_db: Vec::new(),
        };
    }

    let rms_window = ((sr * 0.005).round() as usize).max(1); // 5 ms
    let hop = ((sr * 0.001).round() as usize).max(1); // 1 ms
    let total_samples = ((sr * window_ms / 1000.0).round() as usize).min(signal.len());

    // Number of hops we can take such that the window centred-ish at the
    // hop start still has data. We allow partial windows at the tail by
    // clamping the slice end to `signal.len()`.
    let mut env_db: Vec<f32> = Vec::new();
    let mut idx = 0usize;
    while idx < total_samples {
        let end = (idx + rms_window).min(signal.len());
        if end <= idx {
            break;
        }
        let slice = &signal[idx..end];
        let mut acc = 0.0f64;
        for &s in slice {
            acc += (s as f64) * (s as f64);
        }
        let rms = ((acc / slice.len() as f64).sqrt()) as f32;
        let db = 20.0 * rms.max(1e-10).log10();
        env_db.push(db);
        idx += hop;
    }

    if env_db.is_empty() {
        return AttackEnvelope {
            time_to_peak_s: 0.0,
            peak_db: -200.0,
            post_peak_slope_db_s: 0.0,
            rms_envelope_db: Vec::new(),
        };
    }

    // Find peak. We use a "first arrival within EPS_DB of the global
    // max" rule rather than a strict argmax: a constant-amplitude sine
    // has a per-window RMS that wobbles by a fraction of a dB depending
    // on which fraction of a cycle falls in the window, and a strict
    // argmax picks an arbitrary hop in the middle of the plateau. We
    // want hammer-like signals to report `time_to_peak` at the *onset*
    // of the steady region, not somewhere along it.
    const EPS_DB: f32 = 0.5;
    let global_max = env_db.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let peak_idx = env_db
        .iter()
        .position(|&v| v >= global_max - EPS_DB)
        .unwrap_or(0);
    let peak_db = env_db[peak_idx];
    let time_to_peak_s = (peak_idx as f32) * 0.001; // hop = 1 ms

    // Slope fit: try primary [peak+30ms, peak+80ms], fallbacks documented above.
    let post_peak_slope_db_s = fit_slope(&env_db, peak_idx, window_ms);

    AttackEnvelope {
        time_to_peak_s,
        peak_db,
        post_peak_slope_db_s,
        rms_envelope_db: env_db,
    }
}

/// Linear regression of `env_db[lo..=hi]` (interpreted as dB at t = i·1 ms)
/// over a chosen post-peak window. Returns dB/sec.
fn fit_slope(env_db: &[f32], peak_idx: usize, window_ms: f32) -> f32 {
    let len = env_db.len();
    let last_idx = len.saturating_sub(1);
    let window_idx_max = (window_ms.round() as isize - 1).max(0) as usize;
    let upper_bound = last_idx.min(window_idx_max);

    // Primary: [peak + 30, peak + 80]
    let cand_pri = (peak_idx + 30, peak_idx + 80);
    // Fallback 1: [peak + 20, window_ms]
    let cand_fb1 = (peak_idx + 20, upper_bound);
    // Fallback 2: [peak, window_ms]
    let cand_fb2 = (peak_idx, upper_bound);

    for (lo, hi) in [cand_pri, cand_fb1, cand_fb2] {
        if hi <= lo {
            continue;
        }
        let lo_c = lo.min(last_idx);
        let hi_c = hi.min(last_idx);
        if hi_c <= lo_c + 1 {
            continue;
        }
        return ols_slope_db_per_sec(&env_db[lo_c..=hi_c]);
    }
    0.0
}

/// OLS slope of dB-vs-time, where samples are spaced 1 ms apart.
/// Returns slope in dB/sec.
fn ols_slope_db_per_sec(y: &[f32]) -> f32 {
    let n = y.len();
    if n < 2 {
        return 0.0;
    }
    // x in milliseconds: 0, 1, 2, ..., n-1
    let nf = n as f64;
    let sx: f64 = (0..n).map(|i| i as f64).sum();
    let sy: f64 = y.iter().map(|&v| v as f64).sum();
    let sxx: f64 = (0..n).map(|i| (i as f64) * (i as f64)).sum();
    let sxy: f64 = (0..n).map(|i| (i as f64) * (y[i] as f64)).sum();
    let denom = nf * sxx - sx * sx;
    if denom.abs() < 1e-12 {
        return 0.0;
    }
    let slope_per_ms = (nf * sxy - sx * sy) / denom; // dB / ms
    (slope_per_ms * 1000.0) as f32 // dB / sec
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    /// Render a synthetic AR-enveloped 440 Hz sine: 5 ms linear attack,
    /// exponential decay with τ = 0.300 s.
    fn render_ar_sine(secs: f32, sr: f32, freq: f32) -> Vec<f32> {
        let n = (secs * sr) as usize;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sr;
            let attack = (t / 0.005).clamp(0.0, 1.0);
            let decay = (-t / 0.300).exp();
            let env = attack * decay;
            let s = env * (TAU * freq * t).sin();
            out.push(s);
        }
        out
    }

    #[test]
    fn ar_transient_recovers_attack() {
        let sr = 44_100.0;
        let signal = render_ar_sine(0.200, sr, 440.0);
        let env = extract_attack(&signal, sr, 100.0);

        // Envelope sanity: 100 ms window with 1 ms hops → ~100-101 samples
        // (inclusive of both endpoints).
        assert!(
            env.rms_envelope_db.len() >= 95 && env.rms_envelope_db.len() <= 101,
            "unexpected env len {}",
            env.rms_envelope_db.len()
        );

        // Peak should land near the 5 ms attack target. Allow slack
        // because the 5 ms RMS window smears the peak forward a bit.
        assert!(
            env.time_to_peak_s >= 0.004 && env.time_to_peak_s <= 0.012,
            "time_to_peak_s = {} out of [0.004, 0.012]",
            env.time_to_peak_s
        );

        // Peak dB: 0 dB is the instantaneous peak sample; RMS over 5 ms
        // averages over the rising portion, so we lose a few dB.
        assert!(
            env.peak_db >= -4.0 && env.peak_db <= 0.0,
            "peak_db = {} out of [-4.0, 0.0]",
            env.peak_db
        );

        // Decay slope: τ = 0.3 s ⇒ dB/sec = -20·log10(e)/τ ≈ -28.95.
        // Allow a wide band because the 50 ms post-peak window is short.
        assert!(
            env.post_peak_slope_db_s < 0.0,
            "expected negative slope, got {}",
            env.post_peak_slope_db_s
        );
        assert!(
            env.post_peak_slope_db_s >= -50.0 && env.post_peak_slope_db_s <= -10.0,
            "post_peak_slope_db_s = {} out of [-50, -10]",
            env.post_peak_slope_db_s
        );
    }

    #[test]
    fn plateau_has_near_zero_slope() {
        let sr = 44_100.0;
        let n = (0.100 * sr) as usize;
        let mut signal = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sr;
            signal.push(0.5 * (TAU * 440.0 * t).sin());
        }
        let env = extract_attack(&signal, sr, 100.0);

        // Sine of amplitude 0.5 has RMS 0.5/sqrt(2) ≈ 0.3536 → -9.03 dB
        // for a *long* window. Our 5 ms window at 440 Hz contains ~2.2
        // cycles which still averages ~ -9 dB. Allow a generous band.
        assert!(
            env.peak_db >= -10.0 && env.peak_db <= -8.0,
            "peak_db = {} out of [-10, -8] (expected near -9 dB for 0.5 sine)",
            env.peak_db
        );

        // First non-zero RMS hop should be at ~0 ms; allow up to a few
        // hops of slack because the very first window may still be
        // building up to steady-state.
        assert!(
            env.time_to_peak_s <= 0.020,
            "time_to_peak_s = {} should be early for plateau",
            env.time_to_peak_s
        );

        // No decay → slope ≈ 0.
        assert!(
            env.post_peak_slope_db_s.abs() <= 2.0,
            "post_peak_slope_db_s = {} not near zero for plateau",
            env.post_peak_slope_db_s
        );
    }

    #[test]
    fn silence_then_transient_finds_late_peak() {
        let sr = 44_100.0;
        let n_silence = (0.050 * sr) as usize;
        let n_tone = (0.050 * sr) as usize;
        let mut signal = vec![0.0_f32; n_silence];
        for i in 0..n_tone {
            let t = i as f32 / sr;
            signal.push(1.0 * (TAU * 440.0 * t).sin());
        }
        let env = extract_attack(&signal, sr, 100.0);

        assert!(
            env.time_to_peak_s >= 0.048 && env.time_to_peak_s <= 0.058,
            "time_to_peak_s = {} out of [0.048, 0.058]",
            env.time_to_peak_s
        );
    }

    #[test]
    fn ols_slope_basic() {
        // y = -10·t (in ms), so over 1 s slope = -10000 dB/s.
        let y: Vec<f32> = (0..50).map(|i| -10.0 * (i as f32)).collect();
        let s = ols_slope_db_per_sec(&y);
        assert!((s - -10_000.0).abs() < 1e-3, "got {}", s);
    }

    #[test]
    fn empty_signal_returns_safe_default() {
        let env = extract_attack(&[], 44_100.0, 100.0);
        assert!(env.rms_envelope_db.is_empty());
        assert_eq!(env.time_to_peak_s, 0.0);
        assert_eq!(env.post_peak_slope_db_s, 0.0);
    }
}
