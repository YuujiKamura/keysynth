//! Pitch / time-stretch primitives for issue #7 Stage 5.
//!
//! Three independent functions:
//!
//! * [`varispeed`] – coupled pitch+tempo (linear-interpolation resample).
//!   Like turning a tape speed knob: factor>1 → shorter & higher,
//!   factor<1 → longer & lower.
//!
//! * [`time_stretch`] – pitch-preserving time stretch using WSOLA
//!   (Waveform-Similarity Overlap-Add). 30 ms Hann window, 8 ms synthesis
//!   hop, cross-correlation search for the best-aligned analysis frame.
//!
//! * [`pitch_shift`] – tempo-preserving pitch shift, implemented as
//!   `varispeed(2^(s/12))` followed by `time_stretch(2^(-s/12))`.
//!
//! The module is intentionally self-contained: no shared mixer or jukebox
//! dependencies, so it can be wired into any voice / offline tool without
//! touching the realtime engine paths.

/// Resample `input` by `factor` using linear interpolation.
///
/// `factor == 1.0` returns a copy of the input. `factor > 1.0` produces a
/// shorter, higher-pitched signal (pitch and tempo move together, like a
/// tape speed change). `factor < 1.0` produces a longer, lower-pitched
/// signal. Non-finite or non-positive factors fall back to identity.
pub fn varispeed(input: &[f32], factor: f32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    if !factor.is_finite() || factor <= 0.0 || (factor - 1.0).abs() < 1e-9 {
        return input.to_vec();
    }

    // Output length scales by 1/factor (factor=2 → output half as long).
    let out_len = ((input.len() as f64) / factor as f64).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    let last = input.len() - 1;

    for i in 0..out_len {
        let src = i as f64 * factor as f64;
        let idx = src.floor() as usize;
        if idx >= last {
            out.push(input[last]);
        } else {
            let frac = (src - idx as f64) as f32;
            let a = input[idx];
            let b = input[idx + 1];
            out.push(a + (b - a) * frac);
        }
    }
    out
}

/// WSOLA time-stretch: `factor == 2.0` doubles the duration while preserving
/// pitch; `factor == 0.5` halves it. `sr` is sample rate (Hz). The window /
/// hop are sized in milliseconds so behaviour is sample-rate independent.
pub fn time_stretch(input: &[f32], sr: u32, factor: f32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    if !factor.is_finite() || factor <= 0.0 || (factor - 1.0).abs() < 1e-9 {
        return input.to_vec();
    }

    // 30 ms window, 8 ms synthesis hop. These are typical WSOLA values:
    // long enough to fit ~2 cycles of an 80 Hz tone, short enough to avoid
    // smearing transients.
    let window: usize = ((sr as f32) * 0.030).round() as usize;
    let hop_syn: usize = ((sr as f32) * 0.008).round() as usize;
    let window = window.max(64);
    let hop_syn = hop_syn.max(8).min(window / 2);
    // Search range for similarity alignment: ±half a synthesis hop.
    let search: isize = (hop_syn as isize) / 2;

    // factor=2 means output is 2x longer (tempo halved). To stretch we
    // walk the input *slower* than we write the output, so analysis hop
    // = synthesis hop / factor.
    let hop_ana_f = hop_syn as f32 / factor;

    // Pre-compute Hann window.
    let win: Vec<f32> = (0..window)
        .map(|n| {
            let x = (n as f32) / ((window - 1) as f32);
            0.5 - 0.5 * (2.0 * std::f32::consts::PI * x).cos()
        })
        .collect();

    let out_len = ((input.len() as f32) * factor).round() as usize + window;
    let mut out = vec![0.0f32; out_len];
    let mut norm = vec![0.0f32; out_len];

    // First frame is copied straight from the start of the input.
    for n in 0..window.min(input.len()) {
        out[n] += input[n] * win[n];
        norm[n] += win[n];
    }

    // `prev_tail` is the trailing `hop_syn` samples of the previously
    // written analysis frame. We pick the next analysis frame whose head
    // best matches `prev_tail` (cross-correlation maximum).
    let mut prev_tail: Vec<f32> = if window > hop_syn && input.len() >= window {
        input[(window - hop_syn)..window].to_vec()
    } else {
        vec![0.0; hop_syn]
    };

    let mut syn_pos = hop_syn;
    let mut frame_idx: usize = 1;

    loop {
        // Nominal start of the next analysis frame.
        let ana_center = (frame_idx as f32 * hop_ana_f).round() as isize;
        if ana_center as usize >= input.len() {
            break;
        }

        // Search ±`search` samples around `ana_center` for the offset
        // whose first `hop_syn` samples maximally correlate with
        // `prev_tail`.
        let mut best_off: isize = 0;
        let mut best_score = f32::NEG_INFINITY;
        for off in -search..=search {
            let start = ana_center + off;
            if start < 0 {
                continue;
            }
            let start_us = start as usize;
            if start_us + hop_syn > input.len() {
                continue;
            }
            let mut s = 0.0f32;
            for k in 0..hop_syn {
                s += prev_tail[k] * input[start_us + k];
            }
            if s > best_score {
                best_score = s;
                best_off = off;
            }
        }

        let ana_start = (ana_center + best_off).max(0) as usize;
        if ana_start >= input.len() {
            break;
        }

        // Overlap-add the windowed analysis frame at syn_pos.
        let avail = input.len().saturating_sub(ana_start).min(window);
        for n in 0..avail {
            let dst = syn_pos + n;
            if dst >= out.len() {
                break;
            }
            out[dst] += input[ana_start + n] * win[n];
            norm[dst] += win[n];
        }

        // Update prev_tail to the trailing `hop_syn` samples of the
        // analysis frame we just emitted.
        if avail >= window {
            prev_tail.copy_from_slice(&input[(ana_start + window - hop_syn)..(ana_start + window)]);
        } else if avail >= hop_syn {
            prev_tail.copy_from_slice(&input[(ana_start + avail - hop_syn)..(ana_start + avail)]);
        }

        syn_pos += hop_syn;
        frame_idx += 1;
        if syn_pos >= out_len {
            break;
        }
    }

    // Normalise by the accumulated window envelope so overlapping Hann
    // tapers reconstruct unity gain.
    let target_len = ((input.len() as f32) * factor).round() as usize;
    let target_len = target_len.min(out.len());
    let mut result = Vec::with_capacity(target_len);
    for i in 0..target_len {
        let n = norm[i];
        if n > 1e-6 {
            result.push(out[i] / n);
        } else {
            result.push(0.0);
        }
    }
    result
}

/// Shift pitch by `semitones` while preserving duration. Implemented as
/// `varispeed(r)` followed by `time_stretch(1/r)` where `r = 2^(s/12)`.
pub fn pitch_shift(input: &[f32], sr: u32, semitones: f32) -> Vec<f32> {
    if input.is_empty() || semitones.abs() < 1e-6 {
        return input.to_vec();
    }
    let r = 2.0f32.powf(semitones / 12.0);
    // varispeed(r): output length = input/r, pitch * r, tempo * r.
    let stage1 = varispeed(input, r);
    // time_stretch(r): output length = stage1 * r = input. Pitch
    // unchanged by the stretch, so final pitch is r and length is input.
    let stretched = time_stretch(&stage1, sr, r);

    // Trim / pad to the original length so callers can assume length
    // invariance.
    let mut out = stretched;
    if out.len() > input.len() {
        out.truncate(input.len());
    } else if out.len() < input.len() {
        out.resize(input.len(), 0.0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustfft::{num_complex::Complex32, FftPlanner};

    /// Find the bin index of the spectral peak in `signal` (real-valued)
    /// and return its frequency in Hz. Uses an FFT length that is a power
    /// of two ≤ signal length, with a Hann window to reduce leakage.
    fn peak_freq(signal: &[f32], sr: u32) -> f32 {
        // Largest power-of-two length that fits.
        let mut n = 1usize;
        while n * 2 <= signal.len() {
            n *= 2;
        }
        assert!(n >= 1024, "signal too short for FFT analysis: {}", n);

        // Skip the leading 10% to dodge the WSOLA boundary frame, which
        // can carry a brief amplitude transient.
        let skip = signal.len() / 10;
        let take = (signal.len() - skip).min(n);
        let mut buf: Vec<Complex32> = (0..take)
            .map(|i| {
                let x = (i as f32) / ((take - 1).max(1) as f32);
                let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * x).cos();
                Complex32::new(signal[skip + i] * w, 0.0)
            })
            .collect();
        if buf.len() < n {
            buf.resize(n, Complex32::new(0.0, 0.0));
        }
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut buf);

        // Search the positive-frequency half.
        let half = n / 2;
        let mut peak_bin = 0usize;
        let mut peak_mag = 0.0f32;
        for k in 1..half {
            let m = buf[k].norm_sqr();
            if m > peak_mag {
                peak_mag = m;
                peak_bin = k;
            }
        }
        peak_bin as f32 * sr as f32 / n as f32
    }

    fn sine(freq: f32, sr: u32, n: usize) -> Vec<f32> {
        let dt = 1.0 / sr as f32;
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 * dt).sin())
            .collect()
    }

    #[test]
    fn varispeed_identity() {
        let sr = 48_000u32;
        let s = sine(1000.0, sr, 4096);
        let out = varispeed(&s, 1.0);
        assert_eq!(out.len(), s.len());
    }

    #[test]
    fn varispeed_doubles_pitch() {
        let sr = 48_000u32;
        let s = sine(1000.0, sr, 1 << 15); // 32768 samples
        let out = varispeed(&s, 2.0);
        // Length halved.
        assert!((out.len() as isize - (s.len() as isize) / 2).abs() <= 1);
        let f = peak_freq(&out, sr);
        assert!(
            (f - 2000.0).abs() < 30.0,
            "varispeed(2.0): expected ~2000 Hz, got {}",
            f
        );
    }

    #[test]
    fn time_stretch_doubles_length_keeps_pitch() {
        let sr = 48_000u32;
        let s = sine(1000.0, sr, 1 << 15);
        let out = time_stretch(&s, sr, 2.0);
        // Output should be ~2x input length (allow 5% tolerance for
        // boundary handling).
        let ratio = out.len() as f32 / s.len() as f32;
        assert!(
            (ratio - 2.0).abs() < 0.1,
            "time_stretch(2.0) length ratio {} (expected ~2.0)",
            ratio
        );
        let f = peak_freq(&out, sr);
        assert!(
            (f - 1000.0).abs() < 30.0,
            "time_stretch(2.0): pitch should stay ~1000 Hz, got {}",
            f
        );
    }

    #[test]
    fn time_stretch_halves_length_keeps_pitch() {
        let sr = 48_000u32;
        let s = sine(1000.0, sr, 1 << 15);
        let out = time_stretch(&s, sr, 0.5);
        let ratio = out.len() as f32 / s.len() as f32;
        assert!(
            (ratio - 0.5).abs() < 0.1,
            "time_stretch(0.5) length ratio {} (expected ~0.5)",
            ratio
        );
        let f = peak_freq(&out, sr);
        assert!(
            (f - 1000.0).abs() < 30.0,
            "time_stretch(0.5): pitch should stay ~1000 Hz, got {}",
            f
        );
    }

    #[test]
    fn pitch_shift_octave_up_preserves_length() {
        let sr = 48_000u32;
        let s = sine(1000.0, sr, 1 << 15);
        let out = pitch_shift(&s, sr, 12.0);
        assert_eq!(out.len(), s.len(), "pitch_shift must preserve input length");
        let f = peak_freq(&out, sr);
        assert!(
            (f - 2000.0).abs() < 60.0,
            "pitch_shift(+12): expected ~2000 Hz, got {}",
            f
        );
    }

    #[test]
    fn pitch_shift_zero_is_identity_length() {
        let sr = 48_000u32;
        let s = sine(440.0, sr, 4096);
        let out = pitch_shift(&s, sr, 0.0);
        assert_eq!(out.len(), s.len());
    }
}
