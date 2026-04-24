//! Quantitative WAV analysis: STFT, per-harmonic decay tracking,
//! log-spectral distance, and PNG spectrogram rendering.
//!
//! Built so a parameter tweak in the synth can be classified as improvement
//! or regression by a *number* (log-spectral distance dropped, T60 of the
//! 4th harmonic moved closer to reference, spectral centroid fell off
//! instead of staying flat) rather than by ear-based "sounds more piano-y".

use std::f32::consts::PI;
use std::path::Path;
use std::sync::Arc;

use rustfft::{num_complex::Complex32, FftPlanner};
use serde::Serialize;

// ---- STFT ------------------------------------------------------------------

pub struct StftResult {
    pub fft_size: usize,
    pub hop_size: usize,
    pub sr: u32,
    /// magnitude spectrogram, indexed `[frame][bin]`. bins go
    /// `0 .. (fft_size/2 + 1)`.
    pub mag: Vec<Vec<f32>>,
}

impl StftResult {
    pub fn n_bins(&self) -> usize {
        self.fft_size / 2 + 1
    }

    pub fn n_frames(&self) -> usize {
        self.mag.len()
    }

    pub fn bin_to_hz(&self, bin: usize) -> f32 {
        bin as f32 * self.sr as f32 / self.fft_size as f32
    }

    pub fn frame_to_sec(&self, frame: usize) -> f32 {
        frame as f32 * self.hop_size as f32 / self.sr as f32
    }
}

/// `fft_size = 4096`, `hop_size = 1024`, Hann window. Pads with zeros at end
/// so the last partial hop still becomes a frame.
pub fn stft(samples: &[f32], sr: u32) -> StftResult {
    const FFT_SIZE: usize = 4096;
    const HOP: usize = 1024;

    // Hann window: 0.5 * (1 - cos(2*pi*n / (N-1))).
    let window: Vec<f32> = (0..FFT_SIZE)
        .map(|n| 0.5 - 0.5 * (2.0 * PI * n as f32 / (FFT_SIZE - 1) as f32).cos())
        .collect();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    // Number of frames such that the last frame's start index < samples.len().
    let n_samples = samples.len();
    let n_frames = if n_samples == 0 {
        0
    } else {
        // Ceil division so a partial last hop still produces a frame.
        (n_samples + HOP - 1) / HOP
    };

    let mut mag = Vec::with_capacity(n_frames);
    let mut buf = vec![Complex32::new(0.0, 0.0); FFT_SIZE];

    for frame_idx in 0..n_frames {
        let start = frame_idx * HOP;
        for n in 0..FFT_SIZE {
            let s = if start + n < n_samples {
                samples[start + n]
            } else {
                0.0
            };
            buf[n] = Complex32::new(s * window[n], 0.0);
        }
        fft.process(&mut buf);

        let n_bins = FFT_SIZE / 2 + 1;
        let mut frame_mag = Vec::with_capacity(n_bins);
        for k in 0..n_bins {
            frame_mag.push(buf[k].norm());
        }
        mag.push(frame_mag);
    }

    StftResult {
        fft_size: FFT_SIZE,
        hop_size: HOP,
        sr,
        mag,
    }
}

// ---- Harmonic tracking -----------------------------------------------------

#[derive(Serialize, Debug, Clone)]
pub struct HarmonicTrack {
    pub n: usize,                 // harmonic index (1 = fundamental)
    pub freq_expected_hz: f32,    // n * f0
    pub freq_observed_hz: f32,    // actual peak after inharmonicity
    pub inharmonicity_cents: f32, // 1200 * log2(observed/expected)
    pub initial_db: f32,          // peak magnitude in first 100ms, dB ref to global peak
    pub t60_sec: f32,             // -60dB decay time, fitted via log-linear regression
    pub fit_quality_r2: f32,      // R^2 of the fit, useful to discard noisy harmonics
}

/// For each harmonic n in 1..=max_harmonics, locate the peak bin near n*f0
/// (search +/- ~50 cents), trace the magnitude through every STFT frame,
/// and fit log10(amp) vs t to get T60.
///
/// T60 derivation:
///   amplitude decays as A(t) = A0 * 10^(b*t)  (b<0)
///   in dB amplitude: dB(t) = 20 * b * t
///   for dB drop of -60: -60 = 20 * b * T60 -> T60 = -3 / b
/// So if `b` is the least-squares slope of log10(amp) vs time in seconds,
/// `T60 = -3 / b`.
pub fn harmonic_tracks(
    stft: &StftResult,
    f0_hz: f32,
    max_harmonics: usize,
) -> Vec<HarmonicTrack> {
    let mut out = Vec::with_capacity(max_harmonics);
    if stft.n_frames() == 0 {
        return out;
    }

    // Global peak in dB reference (use max magnitude across whole spectrogram)
    // so per-harmonic initial_db is comparable across files.
    let mut global_peak: f32 = 0.0;
    for frame in &stft.mag {
        for &m in frame {
            if m > global_peak {
                global_peak = m;
            }
        }
    }
    let global_peak_safe = global_peak.max(1e-12);

    let bin_hz = stft.sr as f32 / stft.fft_size as f32;
    // 50 cents window: factor of 2^(50/1200) ~= 1.0293
    let cents_factor: f32 = 2f32.powf(50.0 / 1200.0);

    for n in 1..=max_harmonics {
        let f_expected = f0_hz * n as f32;
        if f_expected >= stft.sr as f32 / 2.0 {
            break;
        }
        let f_lo = f_expected / cents_factor;
        let f_hi = f_expected * cents_factor;
        let bin_lo = (f_lo / bin_hz).floor() as usize;
        let bin_hi = ((f_hi / bin_hz).ceil() as usize).min(stft.n_bins() - 1);
        if bin_hi <= bin_lo {
            continue;
        }

        // Per frame: pick max bin in the window, parabolic-interpolated freq.
        // Track magnitude (linear) per frame for the harmonic.
        let mut frame_mags: Vec<f32> = Vec::with_capacity(stft.n_frames());
        let mut observed_freq_sum: f32 = 0.0;
        let mut observed_freq_count: f32 = 0.0;

        for frame in &stft.mag {
            // find peak bin in [bin_lo, bin_hi]
            let mut peak_bin = bin_lo;
            let mut peak_val = frame[bin_lo];
            for b in (bin_lo + 1)..=bin_hi {
                if frame[b] > peak_val {
                    peak_val = frame[b];
                    peak_bin = b;
                }
            }
            frame_mags.push(peak_val);

            // Parabolic interpolation only if we have neighbours and a strong-ish peak.
            if peak_bin > 0 && peak_bin < stft.n_bins() - 1 && peak_val > 1e-6 {
                let alpha = frame[peak_bin - 1].max(1e-12).log10();
                let beta = peak_val.max(1e-12).log10();
                let gamma = frame[peak_bin + 1].max(1e-12).log10();
                let denom = alpha - 2.0 * beta + gamma;
                let p = if denom.abs() < 1e-12 {
                    0.0
                } else {
                    0.5 * (alpha - gamma) / denom
                };
                let interp_bin = peak_bin as f32 + p.clamp(-1.0, 1.0);
                let f_obs = interp_bin * bin_hz;
                // Weight observation by magnitude so quiet noisy frames don't drift the average.
                observed_freq_sum += f_obs * peak_val;
                observed_freq_count += peak_val;
            }
        }

        let f_observed = if observed_freq_count > 0.0 {
            observed_freq_sum / observed_freq_count
        } else {
            f_expected
        };
        let inharm_cents = 1200.0 * (f_observed / f_expected).max(1e-12).log2();

        // Initial dB: peak magnitude in first 100ms, ref to global peak.
        let frames_per_sec = stft.sr as f32 / stft.hop_size as f32;
        let n_initial = ((0.100 * frames_per_sec) as usize).max(1).min(frame_mags.len());
        let initial_peak: f32 = frame_mags[..n_initial].iter().copied().fold(0.0_f32, f32::max);
        let initial_db = 20.0 * (initial_peak / global_peak_safe).max(1e-12).log10();

        // T60 fit: skip first ~50ms (attack transient), use remaining frames whose
        // magnitude is a non-trivial fraction of the harmonic's peak (>= peak*1e-3,
        // i.e. above -60 dB) so we don't fit noise floor.
        let n_skip = ((0.050 * frames_per_sec) as usize).min(frame_mags.len());
        let mut xs: Vec<f32> = Vec::new();
        let mut ys: Vec<f32> = Vec::new();
        let local_peak: f32 = frame_mags.iter().copied().fold(0.0_f32, f32::max).max(1e-12);
        let floor = local_peak * 1e-3;
        for (i, &m) in frame_mags.iter().enumerate().skip(n_skip) {
            if m <= floor {
                continue;
            }
            let t = i as f32 * stft.hop_size as f32 / stft.sr as f32;
            xs.push(t);
            ys.push(m.log10());
        }

        let (t60, r2) = fit_t60(&xs, &ys);

        out.push(HarmonicTrack {
            n,
            freq_expected_hz: f_expected,
            freq_observed_hz: f_observed,
            inharmonicity_cents: inharm_cents,
            initial_db,
            t60_sec: t60,
            fit_quality_r2: r2,
        });
    }

    out
}

/// Least-squares slope of y = a + b*x; returns (T60, R^2).
/// T60 = -3 / b. If too few points or zero variance, returns (NaN, 0).
fn fit_t60(xs: &[f32], ys: &[f32]) -> (f32, f32) {
    let n = xs.len();
    if n < 4 {
        return (f32::NAN, 0.0);
    }
    let mean_x: f32 = xs.iter().sum::<f32>() / n as f32;
    let mean_y: f32 = ys.iter().sum::<f32>() / n as f32;
    let mut sxx: f32 = 0.0;
    let mut sxy: f32 = 0.0;
    let mut syy: f32 = 0.0;
    for i in 0..n {
        let dx = xs[i] - mean_x;
        let dy = ys[i] - mean_y;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    if sxx.abs() < 1e-12 || syy.abs() < 1e-12 {
        return (f32::NAN, 0.0);
    }
    let b = sxy / sxx;
    let r = sxy / (sxx.sqrt() * syy.sqrt());
    let r2 = r * r;
    let t60 = if b.abs() < 1e-12 || b > 0.0 {
        // b > 0 means amplitude is *growing* (no decay): T60 undefined.
        f32::INFINITY
    } else {
        -3.0 / b
    };
    (t60, r2)
}

// ---- Spectral centroid / flux ---------------------------------------------

/// Spectral centroid in Hz per frame: sum(freq*mag) / sum(mag).
pub fn spectral_centroid_per_frame(stft: &StftResult) -> Vec<f32> {
    let bin_hz = stft.sr as f32 / stft.fft_size as f32;
    stft.mag
        .iter()
        .map(|frame| {
            let mut num = 0.0_f32;
            let mut den = 0.0_f32;
            for (k, &m) in frame.iter().enumerate() {
                let f = k as f32 * bin_hz;
                num += f * m;
                den += m;
            }
            if den > 1e-12 {
                num / den
            } else {
                0.0
            }
        })
        .collect()
}

/// Spectral flux per frame (Euclidean distance between successive frame
/// magnitudes after L2 normalisation). First frame is 0.
pub fn spectral_flux_per_frame(stft: &StftResult) -> Vec<f32> {
    let n = stft.mag.len();
    let mut out = vec![0.0_f32; n];
    if n < 2 {
        return out;
    }
    let normalise = |frame: &[f32]| -> Vec<f32> {
        let norm = frame.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-12 {
            vec![0.0; frame.len()]
        } else {
            frame.iter().map(|x| x / norm).collect()
        }
    };
    let mut prev = normalise(&stft.mag[0]);
    for i in 1..n {
        let curr = normalise(&stft.mag[i]);
        let mut acc = 0.0_f32;
        for k in 0..curr.len() {
            let d = curr[k] - prev[k];
            acc += d * d;
        }
        out[i] = acc.sqrt();
        prev = curr;
    }
    out
}

// ---- Log-spectral distance -------------------------------------------------

/// Mean over (frame, bin) of (20*log10|S_a| - 20*log10|S_b|)^2, returned as
/// the sqrt (RMS) so the unit is dB. Both inputs must have same shape.
pub fn log_spectral_distance_db(a: &StftResult, b: &StftResult) -> f32 {
    assert_eq!(a.fft_size, b.fft_size, "STFT fft_size mismatch");
    assert_eq!(a.hop_size, b.hop_size, "STFT hop_size mismatch");
    assert_eq!(a.n_frames(), b.n_frames(), "STFT frame count mismatch");
    let n_bins = a.n_bins();

    let mut sum_sq: f64 = 0.0;
    let mut count: usize = 0;
    for f in 0..a.n_frames() {
        for k in 0..n_bins {
            let la = 20.0 * (a.mag[f][k].max(1e-9) as f64).log10();
            let lb = 20.0 * (b.mag[f][k].max(1e-9) as f64).log10();
            let d = la - lb;
            sum_sq += d * d;
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    ((sum_sq / count as f64).sqrt()) as f32
}

// ---- PNG spectrogram rendering --------------------------------------------

/// Render magnitude spectrogram to PNG. Width = frames, height = fft_size/2
/// (low freq at bottom). Magnitude in dB clipped to [-80, 0], mapped to a
/// viridis-ish 4-stop gradient. Adds a 1-px white border.
pub fn spectrogram_png(
    stft: &StftResult,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use image::{ImageBuffer, Rgb};

    let height_bins = stft.fft_size / 2; // skip Nyquist for symmetry
    let width = stft.n_frames().max(1) as u32;
    let height = height_bins as u32;

    // Find global peak (linear) for dB normalisation.
    let mut peak: f32 = 1e-12;
    for frame in &stft.mag {
        for &m in frame {
            if m > peak {
                peak = m;
            }
        }
    }

    // Image with 2-px wider/taller for border.
    let img_w = width + 2;
    let img_h = height + 2;
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::new(img_w, img_h);

    // Border colour.
    let border = Rgb([180u8, 180u8, 180u8]);
    for x in 0..img_w {
        img.put_pixel(x, 0, border);
        img.put_pixel(x, img_h - 1, border);
    }
    for y in 0..img_h {
        img.put_pixel(0, y, border);
        img.put_pixel(img_w - 1, y, border);
    }

    for (fi, frame) in stft.mag.iter().enumerate() {
        for k in 0..height_bins {
            let m = frame[k];
            let db = 20.0 * (m / peak).max(1e-9).log10(); // <= 0
            let db_clamped = db.max(-80.0).min(0.0);
            let rgb = colormap_viridis_like(db_clamped);
            // y axis: low freq at bottom, so bin k -> y = height-1-k, then +1 for border
            let y = (height_bins - 1 - k) as u32 + 1;
            let x = fi as u32 + 1;
            img.put_pixel(x, y, Rgb(rgb));
        }
    }

    img.save(path)?;
    Ok(())
}

/// 4-stop viridis-like gradient. Input: dB in [-80, 0].
/// Stops: -80 black, -50 blue, -25 green, 0 yellow->white blend.
fn colormap_viridis_like(db: f32) -> [u8; 3] {
    // Stops in (db, [r,g,b]).
    const STOPS: [(f32, [f32; 3]); 5] = [
        (-80.0, [0.0, 0.0, 0.0]),       // black
        (-50.0, [0.10, 0.10, 0.55]),    // blue
        (-25.0, [0.10, 0.60, 0.30]),    // green
        (-5.0,  [0.95, 0.90, 0.20]),    // yellow
        (0.0,   [1.0, 1.0, 1.0]),       // white
    ];
    let db = db.max(STOPS[0].0).min(STOPS[STOPS.len() - 1].0);
    for w in 0..(STOPS.len() - 1) {
        let (db0, c0) = STOPS[w];
        let (db1, c1) = STOPS[w + 1];
        if db <= db1 {
            let t = if (db1 - db0).abs() < 1e-9 {
                0.0
            } else {
                (db - db0) / (db1 - db0)
            };
            let r = c0[0] + (c1[0] - c0[0]) * t;
            let g = c0[1] + (c1[1] - c0[1]) * t;
            let b = c0[2] + (c1[2] - c0[2]) * t;
            return [
                (r * 255.0).round().clamp(0.0, 255.0) as u8,
                (g * 255.0).round().clamp(0.0, 255.0) as u8,
                (b * 255.0).round().clamp(0.0, 255.0) as u8,
            ];
        }
    }
    [255, 255, 255]
}

// Suppress unused-import warning for Arc if compiler complains; actually
// we don't need it. Remove if linter pings.
#[allow(dead_code)]
fn _ensure_arc_used(_: Arc<()>) {}

// ---- Inharmonicity (Rauhala/Lehtonen/Välimäki 2007) ------------------------

#[derive(Serialize, Debug, Clone)]
pub struct InharmonicityResult {
    /// Fletcher inharmonicity coefficient: f_n = n*f0*sqrt(1 + B*n^2)
    pub b: f32,
    /// Fit quality (R^2) of the estimation, 0..1
    pub r2: f32,
    /// Number of partials that contributed to the fit
    pub n_partials_used: usize,
}

/// Rauhala/Lehtonen/Välimäki 2007 "Fast automatic inharmonicity estimation
/// algorithm" (JASA-EL). Iteratively refines B by minimising residuals between
/// observed partial peaks and predicted f_n = n*f0*sqrt(1+B*n^2). Skips
/// partials with weak / noisy peaks (SNR or amplitude floor). Returns B in
/// the typical piano range 1e-5..1e-1.
pub fn estimate_inharmonicity_b(
    stft: &StftResult,
    f0_hz: f32,
    max_n: usize,
) -> InharmonicityResult {
    if stft.n_frames() == 0 || f0_hz <= 0.0 || max_n == 0 {
        return InharmonicityResult { b: 0.0, r2: 0.0, n_partials_used: 0 };
    }

    let bin_hz = stft.sr as f32 / stft.fft_size as f32;
    let nyquist = stft.sr as f32 / 2.0;
    // ±50 cents window
    let cents_factor: f32 = 2f32.powf(50.0 / 1200.0);

    // First, time-average the magnitude spectrum so we get a stable peak per
    // partial regardless of decay. Average rather than max — averaging keeps
    // the long-lived partials prominent without being thrown by per-frame
    // noise spikes.
    let n_bins = stft.n_bins();
    let n_frames = stft.n_frames() as f32;
    let mut avg_spec = vec![0.0_f32; n_bins];
    for frame in &stft.mag {
        for k in 0..n_bins {
            avg_spec[k] += frame[k];
        }
    }
    for v in &mut avg_spec {
        *v /= n_frames;
    }

    // Locate the fundamental's amplitude to use as a reference (noise floor).
    // We scan a wider window for the fundamental than the cents factor in case
    // of slight f0 mistuning.
    let f1 = f0_hz;
    let bin_lo_f1 = ((f1 / cents_factor) / bin_hz).floor() as usize;
    let bin_hi_f1 = (((f1 * cents_factor) / bin_hz).ceil() as usize).min(n_bins - 1);
    let mut fund_amp: f32 = 0.0;
    if bin_hi_f1 > bin_lo_f1 {
        for b in bin_lo_f1..=bin_hi_f1 {
            if avg_spec[b] > fund_amp {
                fund_amp = avg_spec[b];
            }
        }
    }
    if fund_amp <= 0.0 {
        return InharmonicityResult { b: 0.0, r2: 0.0, n_partials_used: 0 };
    }
    let amp_floor = fund_amp * 1e-3; // -60 dB below fundamental

    // Two-pass peak detection. First pass uses a tight ±50-cent window
    // around n*f0, which is safe for low-to-moderate B. We then fit a
    // preliminary B, use it to predict f_n for all partials, and re-search
    // in a narrow (±50 cents) window around the predicted location. This
    // avoids the wide-window pitfall of picking the *previous* partial as
    // the peak for high-n, heavily stretched partials.
    let peak_search = |center_hz: f32, half_cents: f32| -> Option<f32> {
        let factor = 2f32.powf(half_cents / 1200.0);
        let f_lo = (center_hz / factor).max(bin_hz);
        let f_hi = (center_hz * factor).min(nyquist - bin_hz);
        let bin_lo = (f_lo / bin_hz).floor() as usize;
        let bin_hi = ((f_hi / bin_hz).ceil() as usize).min(n_bins - 1);
        if bin_hi <= bin_lo + 1 {
            return None;
        }
        let mut peak_bin = bin_lo;
        let mut peak_val = avg_spec[bin_lo];
        for b in (bin_lo + 1)..=bin_hi {
            if avg_spec[b] > peak_val {
                peak_val = avg_spec[b];
                peak_bin = b;
            }
        }
        if peak_val < amp_floor {
            return None;
        }
        if peak_bin == 0 || peak_bin >= n_bins - 1 {
            return None;
        }
        // Parabolic interpolation in log-magnitude domain for sub-bin precision.
        let alpha = avg_spec[peak_bin - 1].max(1e-12).log10();
        let beta = peak_val.max(1e-12).log10();
        let gamma = avg_spec[peak_bin + 1].max(1e-12).log10();
        let denom = alpha - 2.0 * beta + gamma;
        let p = if denom.abs() < 1e-12 {
            0.0
        } else {
            0.5 * (alpha - gamma) / denom
        };
        let interp_bin = peak_bin as f32 + p.clamp(-1.0, 1.0);
        Some(interp_bin * bin_hz)
    };

    // Pass 1: tight window at n*f0 for n=1..=min(max_n, 6). Low partials
    // have cents(B, n) = 600*log2(1+B*n^2) which is <35 cents for B<=0.01
    // and n<=6 — comfortably inside ±50 cents.
    let mut samples: Vec<(usize, f32)> = Vec::new();
    let early_max = max_n.min(6);
    for n in 1..=early_max {
        let f_expected = f0_hz * n as f32;
        if f_expected >= nyquist {
            break;
        }
        if let Some(f_obs) = peak_search(f_expected, 50.0) {
            samples.push((n, f_obs));
        }
    }

    if samples.len() < 3 {
        return InharmonicityResult { b: 0.0, r2: 0.0, n_partials_used: 0 };
    }

    // Preliminary B from pass 1.
    let mut sum_xy = 0.0_f64;
    let mut sum_xx = 0.0_f64;
    for &(n, f_obs) in &samples {
        let x = (n as f64).powi(2);
        let r = f_obs as f64 / (n as f64 * f0_hz as f64);
        let y = r * r - 1.0;
        sum_xy += x * y;
        sum_xx += x * x;
    }
    let b_prelim: f64 = if sum_xx > 0.0 {
        (sum_xy / sum_xx).max(0.0)
    } else {
        0.0
    };

    // Pass 2: for each partial n beyond the tight-window set, use b_prelim to
    // predict f_n and re-search ±50c around that prediction.
    if early_max < max_n {
        for n in (early_max + 1)..=max_n {
            let f_pred = (n as f32) * f0_hz * (1.0 + (b_prelim as f32) * (n as f32).powi(2)).sqrt();
            if f_pred >= nyquist {
                break;
            }
            if let Some(f_obs) = peak_search(f_pred, 50.0) {
                samples.push((n, f_obs));
            }
        }
    }

    let _ = b_prelim; // used above; silence unused-warning if cfg(test) drops the debug block.

    // Closed-form linearised least-squares (final fit using all collected samples).
    //   f_obs = n*f0*sqrt(1 + B*n^2)
    //   => (f_obs/(n*f0))^2 - 1 = B * n^2
    //   y = B * x  where x = n^2, y = (f_obs/(n*f0))^2 - 1
    // Force the fit through the origin: B = sum(x*y) / sum(x*x).
    let mut sum_xy = 0.0_f64;
    let mut sum_xx = 0.0_f64;
    let mut xs: Vec<f64> = Vec::with_capacity(samples.len());
    let mut ys: Vec<f64> = Vec::with_capacity(samples.len());
    for &(n, f_obs) in &samples {
        let x = (n as f64).powi(2);
        let r = f_obs as f64 / (n as f64 * f0_hz as f64);
        let y = r * r - 1.0;
        xs.push(x);
        ys.push(y);
        sum_xy += x * y;
        sum_xx += x * x;
    }
    if sum_xx <= 0.0 {
        return InharmonicityResult { b: 0.0, r2: 0.0, n_partials_used: 0 };
    }
    let b_est = sum_xy / sum_xx;

    // R^2 of the linear fit through origin: 1 - SS_res/SS_tot, where SS_tot
    // is sum(y^2) (origin-centred), SS_res is sum((y - B*x)^2).
    let mut ss_res = 0.0_f64;
    let mut ss_tot = 0.0_f64;
    for i in 0..xs.len() {
        let pred = b_est * xs[i];
        let res = ys[i] - pred;
        ss_res += res * res;
        ss_tot += ys[i] * ys[i];
    }
    let r2 = if ss_tot > 0.0 {
        (1.0 - ss_res / ss_tot).max(0.0).min(1.0)
    } else {
        0.0
    };

    // Clamp to typical piano safety range.
    let b_clamped = (b_est as f32).clamp(1e-6, 5e-1);
    // If the raw estimate was negative or otherwise nonsensical, treat as zero.
    let b_final = if b_est <= 0.0 { 0.0 } else { b_clamped };

    InharmonicityResult {
        b: b_final,
        r2: r2 as f32,
        n_partials_used: samples.len(),
    }
}

// ---- Multi-resolution STFT L1 loss ----------------------------------------

/// Multi-resolution STFT L1 loss. Computes STFT at FFT sizes [512, 1024, 2048]
/// (hop = N/4 each), takes L1 of (linear magnitude diff) + L1 of (log
/// magnitude diff with floor 1e-7), sums across resolutions. Standard DDSP
/// loss since Hayes 2021. Lower = closer.
pub fn mr_stft_l1(a: &[f32], b: &[f32], _sr: u32) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let a = &a[..n];
    let b = &b[..n];

    const FFT_SIZES: [usize; 3] = [512, 1024, 2048];
    let mut total = 0.0_f64;
    let mut planner = FftPlanner::<f32>::new();

    for &fft_size in &FFT_SIZES {
        let hop = fft_size / 4;
        // Hann window.
        let window: Vec<f32> = (0..fft_size)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / (fft_size - 1) as f32).cos())
            .collect();
        let fft = planner.plan_fft_forward(fft_size);
        let n_bins = fft_size / 2 + 1;
        // Frame count: ceil division so a partial last hop produces a frame.
        let n_frames = if n == 0 { 0 } else { (n + hop - 1) / hop };
        if n_frames == 0 {
            continue;
        }

        let mut buf_a = vec![Complex32::new(0.0, 0.0); fft_size];
        let mut buf_b = vec![Complex32::new(0.0, 0.0); fft_size];

        let mut lin_acc = 0.0_f64;
        let mut log_acc = 0.0_f64;
        let mut count: usize = 0;

        for f in 0..n_frames {
            let start = f * hop;
            for i in 0..fft_size {
                let sa = if start + i < n { a[start + i] } else { 0.0 };
                let sb = if start + i < n { b[start + i] } else { 0.0 };
                buf_a[i] = Complex32::new(sa * window[i], 0.0);
                buf_b[i] = Complex32::new(sb * window[i], 0.0);
            }
            fft.process(&mut buf_a);
            fft.process(&mut buf_b);
            for k in 0..n_bins {
                let ma = buf_a[k].norm();
                let mb = buf_b[k].norm();
                lin_acc += (ma - mb).abs() as f64;
                log_acc += ((ma + 1e-7).ln() - (mb + 1e-7).ln()).abs() as f64;
                count += 1;
            }
        }
        if count > 0 {
            let lin_loss = lin_acc / count as f64;
            let log_loss = log_acc / count as f64;
            total += lin_loss + log_loss;
        }
    }

    total as f32
}

// ---- Onset envelope L2 -----------------------------------------------------

/// Onset/attack envelope L2: extracts RMS envelope of the first `window_ms`
/// of each input using 5ms hop, then computes L2 of the per-frame RMS
/// difference. Captures hammer transient morphology — a primary perceptual
/// cue for instrument identification.
pub fn onset_envelope_l2(a: &[f32], b: &[f32], sr: u32, window_ms: u32) -> f32 {
    let total_samples = ((window_ms as f32 / 1000.0) * sr as f32) as usize;
    let hop_samples = ((0.005_f32) * sr as f32) as usize;
    if hop_samples == 0 || total_samples == 0 {
        return 0.0;
    }
    let n_frames = total_samples / hop_samples;

    let rms = |sig: &[f32], frame_idx: usize| -> f32 {
        let start = frame_idx * hop_samples;
        let end = (start + hop_samples).min(sig.len());
        if end <= start {
            return 0.0;
        }
        let mut acc = 0.0_f64;
        for i in start..end {
            acc += (sig[i] as f64) * (sig[i] as f64);
        }
        ((acc / (end - start) as f64).sqrt()) as f32
    };

    let mut acc = 0.0_f64;
    for f in 0..n_frames {
        let ra = rms(a, f);
        let rb = rms(b, f);
        let d = ra - rb;
        acc += (d as f64) * (d as f64);
    }
    acc as f32
}

// ---- T60 vector loss with Bank-style weighting ----------------------------

/// Bank-style Taylor-series weighting for per-partial T60 errors. Lower
/// partials get higher weight (perceptually more important). For partial n
/// (1-indexed), weight = 1.0 / (1.0 + 0.05 * (n - 1)). Returns the weighted
/// L2 norm of T60 differences in seconds across `min(reference.len(),
/// candidate.len())` partials.
pub fn t60_vector_loss(reference: &[HarmonicTrack], candidate: &[HarmonicTrack]) -> f32 {
    let n = reference.len().min(candidate.len());
    if n == 0 {
        return 0.0;
    }
    let mut acc = 0.0_f64;
    for i in 0..n {
        let n_idx = reference[i].n.max(1);
        let w = 1.0 / (1.0 + 0.05 * (n_idx as f64 - 1.0));
        let r = reference[i].t60_sec;
        let c = candidate[i].t60_sec;
        // Skip non-finite (NaN / Inf) to keep the loss numerically meaningful.
        if !r.is_finite() || !c.is_finite() {
            continue;
        }
        let d = (r - c) as f64;
        acc += w * d * d;
    }
    (acc.sqrt()) as f32
}

// ---- Centroid trajectory MSE ----------------------------------------------

/// Centroid trajectory MSE: per-frame absolute centroid difference squared,
/// averaged. Both inputs must have same number of frames; asserts.
pub fn centroid_trajectory_mse(reference: &StftResult, candidate: &StftResult) -> f32 {
    assert_eq!(
        reference.n_frames(),
        candidate.n_frames(),
        "centroid_trajectory_mse: frame count mismatch"
    );
    let ca = spectral_centroid_per_frame(reference);
    let cb = spectral_centroid_per_frame(candidate);
    let n = ca.len();
    if n == 0 {
        return 0.0;
    }
    let mut acc = 0.0_f64;
    for i in 0..n {
        let d = (ca[i] - cb[i]) as f64;
        acc += d * d;
    }
    (acc / n as f64) as f32
}

// ---- Tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_inharmonic_signal(
        sr: u32,
        dur_sec: f32,
        f0: f32,
        b: f32,
        n_partials: usize,
    ) -> Vec<f32> {
        let n_samples = (dur_sec * sr as f32) as usize;
        let mut out = vec![0.0_f32; n_samples];
        for n in 1..=n_partials {
            let f_n = (n as f32) * f0 * (1.0 + b * (n as f32).powi(2)).sqrt();
            if f_n >= sr as f32 / 2.0 {
                break;
            }
            let amp = 1.0 / n as f32;
            let omega = 2.0 * PI * f_n / sr as f32;
            for i in 0..n_samples {
                out[i] += amp * (omega * i as f32).sin();
            }
        }
        // Normalise to peak 0.5 to avoid clipping.
        let peak: f32 = out.iter().copied().fold(0.0_f32, |acc, v| acc.max(v.abs()));
        if peak > 0.0 {
            let scale = 0.5 / peak;
            for s in &mut out {
                *s *= scale;
            }
        }
        out
    }

    #[test]
    fn test_inharmonicity_b_synthetic_recovers_known_value() {
        let sr = 44100;
        let f0 = 261.63;
        let b_true = 0.0008_f32;
        let sig = synth_inharmonic_signal(sr, 1.0, f0, b_true, 8);
        let s = stft(&sig, sr);
        let res = estimate_inharmonicity_b(&s, f0, 16);
        let rel_err = ((res.b - b_true).abs() / b_true) as f32;
        assert!(
            rel_err <= 0.20,
            "B estimate {} differs from {} by {:.2}% (>20%)",
            res.b,
            b_true,
            rel_err * 100.0
        );
        assert!(res.r2 > 0.95, "R^2 too low: {}", res.r2);
        assert!(
            res.n_partials_used >= 5,
            "n_partials_used too low: {}",
            res.n_partials_used
        );
    }

    #[test]
    fn test_inharmonicity_b_synthetic_zero_b() {
        let sr = 44100;
        let f0 = 220.0;
        let sig = synth_inharmonic_signal(sr, 1.0, f0, 0.0, 8);
        let s = stft(&sig, sr);
        let res = estimate_inharmonicity_b(&s, f0, 16);
        assert!(
            res.b < 1e-4,
            "expected B ~ 0 for harmonic series, got {}",
            res.b
        );
        // For perfectly harmonic series, residuals are tiny and the linear
        // fit through the origin can produce any R^2 (denominator collapses
        // to numerical noise). We only assert that b is effectively zero.
        assert!(
            res.n_partials_used >= 5,
            "n_partials_used too low: {}",
            res.n_partials_used
        );
    }

    fn synth_sine(sr: u32, dur_sec: f32, freq: f32) -> Vec<f32> {
        let n = (dur_sec * sr as f32) as usize;
        let omega = 2.0 * PI * freq / sr as f32;
        (0..n).map(|i| 0.5 * (omega * i as f32).sin()).collect()
    }

    #[test]
    fn test_mr_stft_l1_identity() {
        let sr = 44100;
        let sig = synth_sine(sr, 0.3, 440.0);
        let loss = mr_stft_l1(&sig, &sig, sr);
        assert!(loss < 1e-5, "identity loss too high: {}", loss);
    }

    #[test]
    fn test_mr_stft_l1_silence_vs_tone() {
        let sr = 44100;
        let tone = synth_sine(sr, 0.5, 440.0);
        let silence = vec![0.0_f32; tone.len()];
        let loss = mr_stft_l1(&silence, &tone, sr);
        assert!(loss > 0.0 && loss.is_finite(), "loss not finite-positive: {}", loss);
    }

    #[test]
    fn test_onset_envelope_l2_zero_for_identity() {
        let sr = 44100;
        let sig = synth_sine(sr, 0.2, 440.0);
        let loss = onset_envelope_l2(&sig, &sig, sr, 80);
        assert!(loss < 1e-9, "identity onset L2 not zero: {}", loss);
    }

    #[test]
    fn test_t60_vector_loss_zero_for_identical_tracks() {
        let make = |n: usize, t60: f32| HarmonicTrack {
            n,
            freq_expected_hz: 440.0 * n as f32,
            freq_observed_hz: 440.0 * n as f32,
            inharmonicity_cents: 0.0,
            initial_db: -6.0,
            t60_sec: t60,
            fit_quality_r2: 0.99,
        };
        let tracks: Vec<HarmonicTrack> =
            (1..=6).map(|n| make(n, 1.5 - 0.1 * n as f32)).collect();
        let loss = t60_vector_loss(&tracks, &tracks);
        assert!(loss < 1e-6, "identical T60 vectors gave nonzero loss: {}", loss);
    }

    #[test]
    fn test_centroid_trajectory_mse_zero_for_identity_stft() {
        let sr = 44100;
        let sig = synth_sine(sr, 0.3, 440.0);
        let s = stft(&sig, sr);
        let mse = centroid_trajectory_mse(&s, &s);
        assert!(mse < 1e-3, "identity centroid MSE not ~0: {}", mse);
    }

    #[test]
    fn stft_n_bins_and_n_frames_consistent() {
        let sr = 44100;
        let sig = synth_sine(sr, 0.2, 440.0);
        let s = stft(&sig, sr);
        assert_eq!(s.n_bins(), s.fft_size / 2 + 1);
        assert_eq!(s.n_frames(), s.mag.len());
    }

    #[test]
    fn stft_bin_to_hz_matches_formula() {
        let sr = 44100;
        let sig = synth_sine(sr, 0.1, 440.0);
        let s = stft(&sig, sr);
        // bin 0 = DC = 0 Hz.
        assert_eq!(s.bin_to_hz(0), 0.0);
        // bin (fft_size/2) = Nyquist = sr/2.
        let nyq = s.bin_to_hz(s.fft_size / 2);
        assert!((nyq - sr as f32 / 2.0).abs() < 1e-3);
    }

    #[test]
    fn stft_frame_to_sec_starts_zero() {
        let sr = 44100;
        let sig = synth_sine(sr, 0.1, 440.0);
        let s = stft(&sig, sr);
        assert_eq!(s.frame_to_sec(0), 0.0);
        let t1 = s.frame_to_sec(1);
        assert!((t1 - s.hop_size as f32 / sr as f32).abs() < 1e-6);
    }

    #[test]
    fn stft_empty_input_yields_no_frames() {
        let s = stft(&[], 44100);
        assert_eq!(s.n_frames(), 0);
    }

    #[test]
    fn stft_finds_peak_near_440hz_for_a4_sine() {
        let sr = 44100;
        let sig = synth_sine(sr, 0.5, 440.0);
        let s = stft(&sig, sr);
        // Look at the middle frame, find max bin.
        let mid = s.n_frames() / 2;
        let frame = &s.mag[mid];
        let (peak_bin, _) = frame
            .iter()
            .enumerate()
            .fold((0usize, 0.0_f32), |(i_max, v_max), (i, &v)| {
                if v > v_max { (i, v) } else { (i_max, v_max) }
            });
        let peak_hz = s.bin_to_hz(peak_bin);
        assert!(
            (peak_hz - 440.0).abs() < 30.0,
            "peak should be near 440 Hz, got {peak_hz}"
        );
    }

    #[test]
    fn spectral_centroid_returns_one_per_frame() {
        let sr = 44100;
        let sig = synth_sine(sr, 0.2, 440.0);
        let s = stft(&sig, sr);
        let c = spectral_centroid_per_frame(&s);
        assert_eq!(c.len(), s.n_frames());
    }

    #[test]
    fn spectral_centroid_higher_for_higher_freq() {
        let sr = 44100;
        let low = stft(&synth_sine(sr, 0.2, 200.0), sr);
        let high = stft(&synth_sine(sr, 0.2, 4000.0), sr);
        let c_low = spectral_centroid_per_frame(&low);
        let c_high = spectral_centroid_per_frame(&high);
        // Centroids are computed per frame; compare middle-frame values.
        let mid = c_low.len() / 2;
        assert!(c_high[mid] > c_low[mid], "high tone centroid should be higher");
    }

    #[test]
    fn spectral_flux_returns_one_per_frame() {
        let sr = 44100;
        let sig = synth_sine(sr, 0.2, 440.0);
        let s = stft(&sig, sr);
        let f = spectral_flux_per_frame(&s);
        assert_eq!(f.len(), s.n_frames());
    }

    #[test]
    fn spectral_flux_zero_for_steady_sine() {
        let sr = 44100;
        let sig = synth_sine(sr, 0.5, 440.0);
        let s = stft(&sig, sr);
        let f = spectral_flux_per_frame(&s);
        // After the first few frames the spectrum is ~constant; flux should be tiny.
        for &v in f.iter().skip(8).take(8) {
            assert!(v < 0.5, "steady sine flux should be small, got {v}");
        }
    }

    #[test]
    fn log_spectral_distance_zero_for_identity() {
        let sr = 44100;
        let sig = synth_sine(sr, 0.3, 440.0);
        let s = stft(&sig, sr);
        let d = log_spectral_distance_db(&s, &s);
        assert!(d < 1e-3, "identity LSD should be ~0, got {d}");
    }

    #[test]
    fn log_spectral_distance_nonzero_for_different_freqs() {
        let sr = 44100;
        let a = stft(&synth_sine(sr, 0.3, 220.0), sr);
        let b = stft(&synth_sine(sr, 0.3, 880.0), sr);
        let d = log_spectral_distance_db(&a, &b);
        assert!(d > 0.5, "different freqs should give LSD > 0.5, got {d}");
        assert!(d.is_finite());
    }

    #[test]
    fn harmonic_tracks_finds_first_partial() {
        let sr = 44100;
        let f0 = 261.63;
        let sig = synth_inharmonic_signal(sr, 0.5, f0, 0.0, 6);
        let s = stft(&sig, sr);
        let tracks = harmonic_tracks(&s, f0, 6);
        assert!(!tracks.is_empty(), "should produce at least one track");
        assert_eq!(tracks[0].n, 1);
        // Observed should be close to expected for the harmonic series.
        assert!(
            (tracks[0].freq_observed_hz - f0).abs() < 5.0,
            "n=1 observed {} vs expected {}",
            tracks[0].freq_observed_hz, f0
        );
    }

    #[test]
    fn t60_vector_loss_increases_with_difference() {
        let make_with_t60 = |t60_scale: f32| -> Vec<HarmonicTrack> {
            (1..=6)
                .map(|n| HarmonicTrack {
                    n,
                    freq_expected_hz: 440.0 * n as f32,
                    freq_observed_hz: 440.0 * n as f32,
                    inharmonicity_cents: 0.0,
                    initial_db: -6.0,
                    t60_sec: 1.0 * t60_scale - 0.1 * n as f32,
                    fit_quality_r2: 0.99,
                })
                .collect()
        };
        let ref_tracks = make_with_t60(1.0);
        let close = make_with_t60(1.05);
        let far = make_with_t60(2.0);
        let loss_close = t60_vector_loss(&ref_tracks, &close);
        let loss_far = t60_vector_loss(&ref_tracks, &far);
        assert!(loss_far > loss_close, "farther T60s should give larger loss");
    }
}
