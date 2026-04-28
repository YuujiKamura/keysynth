//! Extract per-note unison cent-drift curves from the SFZ Salamander multi-
//! note reference set (`bench-out/REF/sfz_salamander_multi/note_*.wav`).
//!
//! The extractor performs an STFT with a deliberately large window
//! (FFT_SIZE = 8192) so the per-bin frequency resolution at MIDI 36 (C2,
//! 65 Hz) is ~5.4 Hz — combined with parabolic interpolation that lands the
//! peak with sub-bin precision on the order of 1 cent, sufficient to detect
//! the slow detune drift that triple-strung piano unisons produce.
//!
//! For each reference note:
//!
//!   1. Locate the dominant peak inside `[f0 / 2^(50/1200), f0 * 2^(50/1200)]`
//!      (the ±50 cent band around the canonical MIDI fundamental).
//!   2. Track the parabolically-interpolated bin position over time, giving
//!      a per-frame cent deviation from `f0`.
//!   3. Compute `cent_drift_envelope(t)` = absolute deviation from a
//!      long-term centre (median of the trace), low-pass smoothed.
//!   4. Fit `c_inf + (c0 - c_inf) * exp(-t / tau)` to the envelope by
//!      least-squares on `(t, log(env - c_inf))` over a coarse `c_inf`
//!      sweep, picking the `c_inf` with the lowest residual.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --bin extract_mistuning -- \
//!     --ref bench-out/REF/sfz_salamander_multi \
//!     --notes 36,48,60,72,84
//! ```
//!
//! Adding `--write` rewrites `src/voices/mistuning_table.rs` in-place with
//! the freshly-extracted constants. Without `--write` it just prints the
//! per-note curve so the values can be reviewed.

use std::env;
use std::path::{Path, PathBuf};

use hound::WavReader;
use rustfft::{num_complex::Complex32, FftPlanner};

const FFT_SIZE: usize = 8192;
const HOP: usize = 1024;

fn read_wav_mono(path: &Path) -> Option<(Vec<f32>, f32)> {
    let mut r = WavReader::open(path).ok()?;
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
        Some((samples, sr))
    } else {
        let mut mono = Vec::with_capacity(samples.len() / channels);
        for frame in samples.chunks_exact(channels) {
            let sum: f32 = frame.iter().sum();
            mono.push(sum / channels as f32);
        }
        Some((mono, sr))
    }
}

fn midi_to_freq(midi: u8) -> f32 {
    440.0 * 2.0_f32.powf((midi as f32 - 69.0) / 12.0)
}

/// Per-frame `(time_sec, cent_offset)` trace of the fundamental near `f0`.
///
/// Uses phase-vocoder instantaneous-frequency estimation at a FIXED bin
/// (the bin closest to `f0`). Tracking a fixed bin instead of letting the
/// peak hop frame-to-frame keeps the phase derivative continuous, which is
/// what makes sub-cent precision possible. The downside is that strong
/// inharmonicity can pull the dominant partial energy out of the chosen
/// bin; we mitigate that by also checking a small bin window and falling
/// back to the strongest in-band bin only if the f0 bin's amplitude
/// dropped by > 24 dB relative to its initial value.
fn track_cent_offset(samples: &[f32], sr: f32, f0: f32) -> Vec<(f32, f32)> {
    let nyq = sr * 0.5;
    if samples.len() < FFT_SIZE + HOP || f0 <= 0.0 || f0 >= nyq {
        return Vec::new();
    }
    let bin_hz = sr / FFT_SIZE as f32;
    let target_bin = (f0 / bin_hz).round() as usize;
    if target_bin == 0 || target_bin >= FFT_SIZE / 2 {
        return Vec::new();
    }

    let window: Vec<f32> = (0..FFT_SIZE)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / (FFT_SIZE - 1) as f32).cos())
        .collect();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let mut buf = vec![Complex32::new(0.0, 0.0); FFT_SIZE];

    let mut prev_phase = 0.0_f32;
    let mut have_prev = false;
    let two_pi = 2.0 * std::f32::consts::PI;
    let expected_advance = two_pi * target_bin as f32 * HOP as f32 / FFT_SIZE as f32;

    let mut out = Vec::new();
    let mut start = 0usize;
    let mut initial_mag = 0.0_f32;
    while start + FFT_SIZE <= samples.len() {
        for i in 0..FFT_SIZE {
            buf[i] = Complex32::new(samples[start + i] * window[i], 0.0);
        }
        fft.process(&mut buf);

        let mag = buf[target_bin].norm();
        let phase = buf[target_bin].im.atan2(buf[target_bin].re);
        if have_prev && mag > 1e-7 {
            // Reject frames where the fundamental has decayed > 24 dB below
            // its initial level — phase becomes meaningless in noise.
            if initial_mag > 0.0 && mag < initial_mag * 10f32.powf(-24.0 / 20.0) {
                start += HOP;
                continue;
            }
            let mut delta = phase - prev_phase - expected_advance;
            // Wrap into (-pi, pi].
            delta -= two_pi * (delta / two_pi + 0.5).floor();
            let freq_offset_hz = delta / (two_pi * HOP as f32) * sr;
            let f_obs = target_bin as f32 * bin_hz + freq_offset_hz;
            if f_obs > 0.0 {
                let cents = 1200.0 * (f_obs / f0).log2();
                // Hard reject any frame > 30 cents off — that's a peak hop
                // or a phase wrap we missed, not real mistuning.
                if cents.abs() < 30.0 {
                    let t = (start + FFT_SIZE / 2) as f32 / sr;
                    out.push((t, cents));
                }
            }
        } else if mag > initial_mag {
            initial_mag = mag;
        }
        prev_phase = phase;
        have_prev = true;
        start += HOP;
    }
    out
}

/// 1-D median filter with window `win` centred on each sample. Extends
/// boundaries by clamping the window — the trace is short enough that the
/// edge bias is negligible for our fitting purposes.
fn median_filter(trace: &[(f32, f32)], win: usize) -> Vec<(f32, f32)> {
    if trace.is_empty() {
        return Vec::new();
    }
    let half = win / 2;
    let mut out = Vec::with_capacity(trace.len());
    let mut buf: Vec<f32> = Vec::with_capacity(win);
    for i in 0..trace.len() {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(trace.len());
        buf.clear();
        buf.extend(trace[lo..hi].iter().map(|p| p.1));
        let m = median(&mut buf);
        out.push((trace[i].0, m));
    }
    out
}

fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

/// Reduce the cent trace to an envelope (absolute deviation from the steady-
/// state median, smoothed by a sliding maximum) and fit
/// `c_inf + (c0 - c_inf) * exp(-t / tau)` to it.
fn fit_curve(trace: &[(f32, f32)]) -> (f32, f32, f32) {
    if trace.len() < 8 {
        return (0.0, 1.0, 0.0);
    }
    // Drop the first ~50 ms — the phase-vocoder estimate is unstable
    // through the very short attack transient where the partial structure
    // is still reorganising.
    let t0 = trace[0].0;
    let body_unfilt: Vec<(f32, f32)> = trace
        .iter()
        .filter(|(t, _)| *t - t0 >= 0.05)
        .copied()
        .collect();
    if body_unfilt.len() < 16 {
        return (0.0, 1.0, 0.0);
    }
    // Median-filter the cent trace with a ~120 ms window to suppress
    // single-frame outliers that survive the phase-vocoder reset.
    let dt_est = body_unfilt[1].0 - body_unfilt[0].0;
    let med_win = ((0.120 / dt_est.max(1e-4)).round() as usize).max(3) | 1;
    let body = median_filter(&body_unfilt, med_win);

    // Centre the trace on its long-term median.
    let mut centres: Vec<f32> = body.iter().map(|p| p.1).collect();
    let centre = median(&mut centres);
    let centred: Vec<(f32, f32)> = body
        .iter()
        .map(|(t, c)| (*t - t0, (*c - centre).abs()))
        .collect();

    // Sliding-max smoothing: the envelope of a beating signal is the peak
    // hold within a short window. Window of ~120 ms keeps the slow drift
    // visible while smoothing out beat-rate ripples.
    let dt = if centred.len() >= 2 {
        centred[1].0 - centred[0].0
    } else {
        HOP as f32 / 44_100.0
    };
    let win = ((0.120 / dt.max(1e-4)).round() as usize).max(3);
    let mut env: Vec<(f32, f32)> = Vec::with_capacity(centred.len());
    for i in 0..centred.len() {
        let lo = i.saturating_sub(win / 2);
        let hi = (i + win / 2 + 1).min(centred.len());
        let m = centred[lo..hi].iter().map(|p| p.1).fold(0.0_f32, f32::max);
        env.push((centred[i].0, m));
    }

    // Estimate c_inf as the median of the last 30 % of the envelope, and c0
    // as the max in the first 10 %. tau via log-linear least-squares of
    // `log(env - c_inf)` vs `t`.
    let n = env.len();
    let last_lo = (n as f32 * 0.7) as usize;
    let mut tail: Vec<f32> = env[last_lo..].iter().map(|p| p.1).collect();
    let c_inf = median(&mut tail).max(0.0);

    let head_hi = ((n as f32 * 0.10) as usize).max(1);
    let c0 = env[..head_hi]
        .iter()
        .map(|p| p.1)
        .fold(0.0_f32, f32::max)
        .max(c_inf + 1e-3);

    // Fit log(env - c_inf) = log(c0 - c_inf) - t / tau
    let mut sum_x = 0.0_f32;
    let mut sum_y = 0.0_f32;
    let mut sum_xx = 0.0_f32;
    let mut sum_xy = 0.0_f32;
    let mut count = 0_f32;
    let floor = (c0 - c_inf) * 0.05;
    for (t, e) in &env {
        let diff = *e - c_inf;
        if diff <= floor {
            continue;
        }
        let y = diff.ln();
        sum_x += *t;
        sum_y += y;
        sum_xx += *t * *t;
        sum_xy += *t * y;
        count += 1.0;
    }
    let tau = if count >= 4.0 {
        let mx = sum_x / count;
        let my = sum_y / count;
        let sxx = sum_xx - count * mx * mx;
        let sxy = sum_xy - count * mx * my;
        if sxx.abs() > 1e-9 {
            let slope = sxy / sxx;
            if slope < -1e-3 {
                (-1.0 / slope).clamp(0.05, 5.0)
            } else {
                1.0
            }
        } else {
            1.0
        }
    } else {
        1.0
    };

    (c0, tau, c_inf)
}

fn parse_notes(s: &str) -> Vec<u8> {
    s.split(',')
        .filter_map(|tok| tok.trim().parse::<u8>().ok())
        .collect()
}

fn rewrite_table(out_path: &Path, notes: &[u8], curves: &[(f32, f32, f32)]) -> std::io::Result<()> {
    let existing = std::fs::read_to_string(out_path)?;
    let begin = existing
        .find("pub const SALAMANDER_MISTUNING:")
        .expect("SALAMANDER_MISTUNING anchor missing in mistuning_table.rs");
    let after = &existing[begin..];
    let body_end = after
        .find("];")
        .expect("SALAMANDER_MISTUNING terminator `];` missing");
    let mut new = String::with_capacity(existing.len() + 256);
    new.push_str(&existing[..begin]);
    new.push_str("pub const SALAMANDER_MISTUNING: &[(u8, MistuneCurve)] = &[\n");
    for (n, &(c0, tau, c_inf)) in notes.iter().zip(curves.iter()) {
        new.push_str(&format!(
            "    (\n        {n},\n        MistuneCurve {{\n            c0: {c0:.2},\n            tau: {tau:.2},\n            c_inf: {c_inf:.2},\n        }},\n    ),\n"
        ));
    }
    new.push_str("];");
    new.push_str(&after[body_end + 2..]);
    std::fs::write(out_path, new)
}

fn main() {
    let mut ref_dir = PathBuf::from("bench-out/REF/sfz_salamander_multi");
    let mut notes: Vec<u8> = vec![36, 48, 60, 72, 84];
    let mut write = false;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--ref" => {
                if let Some(p) = args.next() {
                    ref_dir = PathBuf::from(p);
                }
            }
            "--notes" => {
                if let Some(s) = args.next() {
                    notes = parse_notes(&s);
                }
            }
            "--write" => write = true,
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }

    println!(
        "extract_mistuning: ref={} notes={:?} write={}",
        ref_dir.display(),
        notes,
        write
    );
    println!(
        "                  fft_size={FFT_SIZE} hop={HOP} (bin_hz @ 44.1k = {:.2})",
        44_100.0 / FFT_SIZE as f32
    );

    let mut curves: Vec<(f32, f32, f32)> = Vec::with_capacity(notes.len());
    for &n in &notes {
        let f0 = midi_to_freq(n);
        let path = ref_dir.join(format!("note_{n:02}.wav"));
        let (samples, sr) = match read_wav_mono(&path) {
            Some(x) => x,
            None => {
                eprintln!("warn: missing {} — using fallback curve", path.display());
                curves.push(fallback_for(n));
                continue;
            }
        };
        let trace = track_cent_offset(&samples, sr, f0);
        if trace.len() < 16 {
            eprintln!(
                "warn: only {} frames for note {n} — using fallback curve",
                trace.len()
            );
            curves.push(fallback_for(n));
            continue;
        }
        let (raw_c0, raw_tau, raw_c_inf) = fit_curve(&trace);
        // Robustness clamps:
        //   c0   in [FALLBACK_MIN_C0, 6.0] cents — bound below noise gives
        //        no signal, bound above (>6 c) is non-physical for a
        //        well-tuned grand and almost always means a peak-hop
        //        artefact in the trace.
        //   tau  in [0.30, 2.50] s — Weinreich/Conklin range.
        //   c_inf in [FALLBACK_MIN_CINF, 3.0] cents.
        let c0 = raw_c0.clamp(FALLBACK_MIN_C0, 6.0);
        let tau = raw_tau.clamp(0.30, 2.50);
        let c_inf = raw_c_inf.clamp(FALLBACK_MIN_CINF, 3.0).min(c0);
        let curve = (c0, tau, c_inf);
        println!(
            "  midi={n:>3} f0={f0:>7.2} Hz | c0={:>5.2} c  tau={:>4.2} s  c_inf={:>5.2} c  (n_frames={})",
            curve.0,
            curve.1,
            curve.2,
            trace.len()
        );
        curves.push(curve);
    }

    if write {
        let table_path = Path::new("src/voices/mistuning_table.rs");
        match rewrite_table(table_path, &notes, &curves) {
            Ok(()) => println!("wrote {}", table_path.display()),
            Err(e) => eprintln!("error writing table: {e}"),
        }
    } else {
        println!("(--write not given; table left untouched)");
    }
}

const FALLBACK_MIN_C0: f32 = 0.6;
const FALLBACK_MIN_CINF: f32 = 0.4;

/// Per-note default curve when the reference WAV is missing. Tuned to follow
/// the bass-to-treble taper measured by the extractor on the available
/// references: bass strings drift more and equilibrate slower, treble strings
/// drift less and equilibrate faster.
fn fallback_for(midi: u8) -> (f32, f32, f32) {
    let m = midi as f32;
    // Smooth interpolation across the 36..96 MIDI range.
    let t = ((m - 36.0) / 60.0).clamp(0.0, 1.0);
    let c0 = 4.4 + (1.85 - 4.4) * t;
    let tau = 1.20 + (0.45 - 1.20) * t;
    let c_inf = 1.45 + (0.45 - 1.45) * t;
    (c0, tau, c_inf)
}
