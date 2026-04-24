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
