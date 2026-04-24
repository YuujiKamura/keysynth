//! Per-partial T60 extraction (time to -60 dB).
//!
//! Pipeline (issue #3 P3):
//!   1. STFT the whole signal with window=2048, hop=512, Hann.
//!   2. For each partial, max-pool magnitude in a ±10-cent bin window
//!      around `freq_hz` per frame.
//!   3. Convert to dB relative to that partial's peak frame.
//!   4. Least-squares fit of dB-vs-time on the segment from the peak
//!      frame to the first frame whose dB falls below -45 dB (or fall
//!      back to the 0.1..3.0 s window if no -45 crossing exists).
//!   5. T60 = -60 / slope_dB_per_sec. If the slope is non-negative
//!      (no decay) or the fit's R² is below 0.5, return the sentinel
//!      -1.0 for that partial.

#![allow(dead_code)]

use std::f32::consts::PI;

use rustfft::{num_complex::Complex32, FftPlanner};

use super::decompose::Partial;

/// Per-partial T60 vector. Index n in the returned vec is partial n+1
/// (i.e. `vec[0]` = fundamental T60, `vec[1]` = 2nd partial T60, ...).
#[derive(Clone, Debug)]
pub struct T60Vector {
    pub seconds: Vec<f32>,
}

const FFT_SIZE: usize = 2048;
const HOP: usize = 512;
/// Sentinel returned when the partial does not exhibit a usable decay.
const NO_DECAY: f32 = -1.0;

/// Extract T60 (seconds to -60 dB) per partial. See module docstring for
/// the full contract; the implementation mirrors `analysis::harmonic_tracks`
/// at high level but pins the STFT geometry, the bin window (±10 cents),
/// and the regression segment to the spec required for `extract_t60`.
pub fn extract_t60(signal: &[f32], sr: f32, partials: &[Partial]) -> T60Vector {
    if signal.is_empty() || sr <= 0.0 || partials.is_empty() {
        return T60Vector {
            seconds: vec![NO_DECAY; partials.len()],
        };
    }

    let stft = stft_mag(signal, sr);
    let n_frames = stft.len();
    if n_frames == 0 {
        return T60Vector {
            seconds: vec![NO_DECAY; partials.len()],
        };
    }
    let n_bins = FFT_SIZE / 2 + 1;
    let bin_hz = sr / FFT_SIZE as f32;
    let nyquist = sr / 2.0;
    let frame_dt = HOP as f32 / sr; // seconds per frame
                                    // Cents factor of 10 cents = 2^(10/1200) ~= 1.005793.
    let cents_factor: f32 = 2f32.powf(10.0 / 1200.0);

    let mut seconds = Vec::with_capacity(partials.len());

    for p in partials {
        let f = p.freq_hz;
        if !(f.is_finite() && f > 0.0) || f >= nyquist {
            seconds.push(NO_DECAY);
            continue;
        }

        // Bin window covering ±10 cents around f.
        let f_lo = f / cents_factor;
        let f_hi = f * cents_factor;
        let mut bin_lo = (f_lo / bin_hz).floor() as isize;
        let mut bin_hi = (f_hi / bin_hz).ceil() as isize;
        if bin_lo < 0 {
            bin_lo = 0;
        }
        if bin_hi >= n_bins as isize {
            bin_hi = n_bins as isize - 1;
        }
        // ±10 cents at low frequencies can collapse to a single bin; make
        // sure we always have *some* coverage (bin containing f at minimum).
        if bin_hi < bin_lo {
            let center_bin = ((f / bin_hz).round() as isize).clamp(0, n_bins as isize - 1);
            bin_lo = center_bin;
            bin_hi = center_bin;
        }
        let bin_lo = bin_lo as usize;
        let bin_hi = bin_hi as usize;

        // Max-pool magnitude across the bin window per frame.
        let mut env = Vec::with_capacity(n_frames);
        for frame in &stft {
            let mut peak = frame[bin_lo];
            for b in (bin_lo + 1)..=bin_hi {
                if frame[b] > peak {
                    peak = frame[b];
                }
            }
            env.push(peak);
        }

        // Locate the partial's max magnitude across all frames; convert to dB
        // relative to that peak.
        let max_mag: f32 = env.iter().copied().fold(0.0_f32, f32::max);
        if max_mag <= 0.0 {
            seconds.push(NO_DECAY);
            continue;
        }
        let env_db: Vec<f32> = env
            .iter()
            .map(|&m| 20.0 * (m / max_mag).max(1e-9).log10())
            .collect();

        // Peak frame (first frame at 0 dB; use argmax of env to be robust to
        // ties from clipping).
        let peak_frame = env_db
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
                if v > acc.1 {
                    (i, v)
                } else {
                    acc
                }
            })
            .0;

        // Determine the regression segment.
        // Primary: peak_frame .. first frame after peak whose dB <= -45.
        // Fallback: frames whose time is in [0.1 s, 3.0 s].
        let mut end_frame: Option<usize> = None;
        for (i, &v) in env_db.iter().enumerate().skip(peak_frame + 1) {
            if v <= -45.0 {
                end_frame = Some(i);
                break;
            }
        }

        let (xs, ys) = if let Some(end) = end_frame {
            // Use peak_frame..=end.
            let mut xs = Vec::with_capacity(end - peak_frame + 1);
            let mut ys = Vec::with_capacity(end - peak_frame + 1);
            for i in peak_frame..=end {
                xs.push(i as f32 * frame_dt);
                ys.push(env_db[i]);
            }
            (xs, ys)
        } else {
            // Fallback window: 0.1 .. 3.0 s.
            let lo_idx = ((0.1_f32 / frame_dt).ceil() as usize).min(n_frames);
            let hi_idx = ((3.0_f32 / frame_dt).floor() as usize).min(n_frames.saturating_sub(1));
            if hi_idx <= lo_idx + 2 {
                seconds.push(NO_DECAY);
                continue;
            }
            let mut xs = Vec::with_capacity(hi_idx - lo_idx + 1);
            let mut ys = Vec::with_capacity(hi_idx - lo_idx + 1);
            for i in lo_idx..=hi_idx {
                xs.push(i as f32 * frame_dt);
                ys.push(env_db[i]);
            }
            (xs, ys)
        };

        if xs.len() < 4 {
            seconds.push(NO_DECAY);
            continue;
        }

        let (slope, r2) = least_squares_slope_r2(&xs, &ys);

        // slope is dB/sec. T60 = -60 / slope. Reject non-decaying or
        // garbage fits.
        if !slope.is_finite() || slope >= 0.0 || r2 < 0.5 {
            seconds.push(NO_DECAY);
            continue;
        }
        let t60 = -60.0 / slope;
        if !t60.is_finite() || t60 <= 0.0 {
            seconds.push(NO_DECAY);
            continue;
        }
        seconds.push(t60);
    }

    T60Vector { seconds }
}

/// STFT magnitude with window=`FFT_SIZE`, hop=`HOP`, Hann window.
/// Returns Vec<frame_mag>, where each frame is `FFT_SIZE/2 + 1` bins long.
fn stft_mag(signal: &[f32], _sr: f32) -> Vec<Vec<f32>> {
    let window: Vec<f32> = (0..FFT_SIZE)
        .map(|n| 0.5 - 0.5 * (2.0 * PI * n as f32 / (FFT_SIZE - 1) as f32).cos())
        .collect();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    let n_samples = signal.len();
    if n_samples == 0 {
        return Vec::new();
    }
    let n_frames = (n_samples + HOP - 1) / HOP;
    let n_bins = FFT_SIZE / 2 + 1;

    let mut buf = vec![Complex32::new(0.0, 0.0); FFT_SIZE];
    let mut out = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let start = f * HOP;
        for n in 0..FFT_SIZE {
            let s = if start + n < n_samples {
                signal[start + n]
            } else {
                0.0
            };
            buf[n] = Complex32::new(s * window[n], 0.0);
        }
        fft.process(&mut buf);
        let mut mag = Vec::with_capacity(n_bins);
        for k in 0..n_bins {
            mag.push(buf[k].norm());
        }
        out.push(mag);
    }
    out
}

/// Least-squares slope of `y = a + b*x`. Returns `(b, r2)`. Returns
/// `(NaN, 0)` for degenerate input.
fn least_squares_slope_r2(xs: &[f32], ys: &[f32]) -> (f32, f32) {
    let n = xs.len();
    if n < 2 || n != ys.len() {
        return (f32::NAN, 0.0);
    }
    let mean_x: f64 = xs.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let mean_y: f64 = ys.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let mut sxx = 0.0_f64;
    let mut sxy = 0.0_f64;
    let mut syy = 0.0_f64;
    for i in 0..n {
        let dx = xs[i] as f64 - mean_x;
        let dy = ys[i] as f64 - mean_y;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    if sxx <= 0.0 {
        return (f32::NAN, 0.0);
    }
    let b = sxy / sxx;
    let r2 = if syy <= 0.0 {
        0.0
    } else {
        let r = sxy / (sxx.sqrt() * syy.sqrt());
        (r * r).min(1.0).max(0.0)
    };
    (b as f32, r2 as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Render a single exponentially-decaying sinusoid:
    ///   x(t) = sin(2π f t) · exp(-3 ln(10) t / T60_target)
    /// (-60 dB drop in `t60` seconds since 20·log10(exp(-3 ln10)) = -60.)
    fn render_decaying_sine(sr: f32, dur_s: f32, freq: f32, t60: f32) -> Vec<f32> {
        let n = (dur_s * sr) as usize;
        let omega = 2.0 * PI * freq / sr;
        let k = 3.0_f32 * (10.0_f32).ln() / t60;
        (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                let env = (-k * t).exp();
                env * (omega * i as f32).sin()
            })
            .collect()
    }

    #[test]
    fn round_trip_single_partial_4s() {
        let sr = 44100.0_f32;
        let f = 261.63_f32;
        let t60_target = 4.0_f32;
        let signal = render_decaying_sine(sr, 5.0, f, t60_target);

        let partial = Partial {
            n: 1,
            freq_hz: f,
            init_db: 0.0,
        };
        let out = extract_t60(&signal, sr, std::slice::from_ref(&partial));
        assert_eq!(out.seconds.len(), 1);
        let t60 = out.seconds[0];
        assert!(t60 > 0.0, "expected positive T60, got {t60}");
        let lo = t60_target * 0.95;
        let hi = t60_target * 1.05;
        assert!(
            t60 >= lo && t60 <= hi,
            "T60 {t60} outside ±5% window [{lo}, {hi}]"
        );
    }

    #[test]
    fn multi_partial_recovery() {
        let sr = 44100.0_f32;
        let dur = 6.0_f32;
        let n = (dur * sr) as usize;
        let mut sig = vec![0.0_f32; n];

        let parts: [(f32, f32, f32); 3] =
            [(261.63, 18.0, 1.0), (523.10, 11.0, 0.5), (785.60, 9.0, 0.3)];

        for &(freq, t60, amp) in &parts {
            let omega = 2.0 * PI * freq / sr;
            let k = 3.0_f32 * (10.0_f32).ln() / t60;
            for i in 0..n {
                let t = i as f32 / sr;
                let env = (-k * t).exp();
                sig[i] += amp * env * (omega * i as f32).sin();
            }
        }

        let partials: Vec<Partial> = parts
            .iter()
            .enumerate()
            .map(|(i, &(freq, _, _))| Partial {
                n: i + 1,
                freq_hz: freq,
                init_db: 0.0,
            })
            .collect();

        let out = extract_t60(&sig, sr, &partials);
        assert_eq!(out.seconds.len(), 3);
        for ((i, target), &got) in parts
            .iter()
            .map(|p| p.1)
            .enumerate()
            .zip(out.seconds.iter())
        {
            assert!(got > 0.0, "partial {i}: T60 sentinel/negative ({got})");
            let lo = target * 0.9;
            let hi = target * 1.1;
            assert!(
                got >= lo && got <= hi,
                "partial {i}: recovered T60 {got} outside ±10% of {target}"
            );
        }
    }

    #[test]
    fn no_decay_returns_sentinel() {
        let sr = 44100.0_f32;
        let dur = 5.0_f32;
        let n = (dur * sr) as usize;
        let f = 440.0_f32;
        let omega = 2.0 * PI * f / sr;
        let signal: Vec<f32> = (0..n).map(|i| 0.5 * (omega * i as f32).sin()).collect();

        let partial = Partial {
            n: 1,
            freq_hz: f,
            init_db: 0.0,
        };
        let out = extract_t60(&signal, sr, std::slice::from_ref(&partial));
        assert_eq!(out.seconds.len(), 1);
        let t60 = out.seconds[0];
        assert!(
            t60 < 0.0,
            "sustained sine should yield negative sentinel, got {t60}"
        );
    }
}
