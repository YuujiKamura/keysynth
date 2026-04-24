//! Quantitative WAV comparison harness — piano-specific metric stack (issue #1).
//!
//! Usage:
//!   analyse --reference REF.wav --candidate CAND.wav [--note 60]
//!           [--out report/] [--max-harmonics 16]
//!           [--metric lsd|mrstft|b|t60|onset|centroid|all]
//!           [--reference-iowa PATH]   # alias for --reference, documents intent
//!
//! Outputs (under --out):
//!   spectrogram_reference.png
//!   spectrogram_candidate.png
//!   harmonics.json
//!   centroid.csv
//!   metrics.json     (full metric stack, machine-readable)
//!   summary.txt      (diff-friendly fixed-width human report)
//!
//! The point: replace ear-based "sounds more piano-y" with falsifiable
//! numbers. The metric stack covers timbre (LSD, MR-STFT, centroid),
//! piano-specific physics (inharmonicity B, T60 vector), and attack
//! transient (onset envelope L2).

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use hound::WavReader;
use serde::Serialize;
use serde_json::json;

use keysynth::analysis::{
    centroid_trajectory_mse, estimate_inharmonicity_b, harmonic_tracks,
    log_spectral_distance_db, mr_stft_l1, onset_envelope_l2, spectral_centroid_per_frame,
    spectrogram_png, stft, t60_vector_loss, HarmonicTrack, InharmonicityResult,
};

const EXPECTED_SR: u32 = 44100;

#[derive(Clone, Copy, PartialEq, Eq)]
enum MetricSel {
    Lsd,
    MrStft,
    B,
    T60,
    Onset,
    Centroid,
    All,
}

impl MetricSel {
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "lsd" => MetricSel::Lsd,
            "mrstft" => MetricSel::MrStft,
            "b" => MetricSel::B,
            "t60" => MetricSel::T60,
            "onset" => MetricSel::Onset,
            "centroid" => MetricSel::Centroid,
            "all" => MetricSel::All,
            other => {
                return Err(format!(
                    "bad --metric {:?}: expected one of lsd|mrstft|b|t60|onset|centroid|all",
                    other
                ))
            }
        })
    }
    fn show(self, m: MetricSel) -> bool {
        self == MetricSel::All || self == m
    }
}

struct Args {
    reference: PathBuf,
    candidate: PathBuf,
    note: u8,
    out: PathBuf,
    max_harmonics: usize,
    metric: MetricSel,
}

fn parse_args() -> Result<Args, String> {
    let mut reference: Option<PathBuf> = None;
    let mut candidate: Option<PathBuf> = None;
    let mut note: u8 = 60;
    let mut out: PathBuf = PathBuf::from("report");
    let mut max_harmonics: usize = 16;
    let mut metric: MetricSel = MetricSel::All;

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let a = argv[i].clone();
        let take_next = |i: usize| -> Result<String, String> {
            argv.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("missing value for {}", a))
        };
        match a.as_str() {
            "--reference" | "--reference-iowa" => {
                reference = Some(PathBuf::from(take_next(i)?));
                i += 2;
            }
            "--candidate" => {
                candidate = Some(PathBuf::from(take_next(i)?));
                i += 2;
            }
            "--note" => {
                note = take_next(i)?
                    .parse()
                    .map_err(|e| format!("bad --note: {}", e))?;
                i += 2;
            }
            "--out" => {
                out = PathBuf::from(take_next(i)?);
                i += 2;
            }
            "--max-harmonics" => {
                max_harmonics = take_next(i)?
                    .parse()
                    .map_err(|e| format!("bad --max-harmonics: {}", e))?;
                i += 2;
            }
            "--metric" => {
                metric = MetricSel::parse(&take_next(i)?)?;
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!(
                    "analyse --reference REF.wav --candidate CAND.wav \
                     [--note 60] [--out report/] [--max-harmonics 16] \
                     [--metric lsd|mrstft|b|t60|onset|centroid|all] \
                     [--reference-iowa PATH]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {}", other)),
        }
    }
    Ok(Args {
        reference: reference.ok_or("missing --reference (or --reference-iowa)")?,
        candidate: candidate.ok_or("missing --candidate")?,
        note,
        out,
        max_harmonics,
        metric,
    })
}

/// Load a WAV file as mono f32 samples at 44.1 kHz.
///
/// - Stereo (or any multi-channel) is downmixed to mono (avg of all channels).
/// - Sample rate must equal 44100; otherwise a clear error is returned.
fn load_mono_wav(path: &std::path::Path) -> Result<(Vec<f32>, u32), String> {
    let mut reader = WavReader::open(path).map_err(|e| format!("open {:?}: {}", path, e))?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    if sr != EXPECTED_SR {
        return Err(format!(
            "{:?}: expected {} Hz, got {} Hz; resample with \
             `sox in.wav -r 44100 out.wav` first",
            path, EXPECTED_SR, sr
        ));
    }
    let channels = spec.channels as usize;
    if channels == 0 {
        return Err(format!("{:?}: zero channels", path));
    }

    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max).unwrap_or(0.0))
                .collect()
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.unwrap_or(0.0))
            .collect(),
    };

    let mono: Vec<f32> = if channels == 1 {
        interleaved
    } else {
        let inv = 1.0 / channels as f32;
        interleaved
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() * inv)
            .collect()
    };
    Ok((mono, sr))
}

fn midi_to_hz(note: u8) -> f32 {
    440.0 * 2f32.powf((note as f32 - 69.0) / 12.0)
}

fn note_name(note: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let octave = (note as i32 / 12) - 1;
    let n = NAMES[(note as usize) % 12];
    format!("{}{}", n, octave)
}

#[derive(Serialize)]
struct HarmonicDelta {
    n: usize,
    df_cents: f32,
    d_t60_sec: f32,
    d_initial_db: f32,
}

fn mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f32>() / v.len() as f32
}
fn start_window_mean(v: &[f32], n: usize) -> f32 {
    let take = n.min(v.len());
    if take == 0 {
        return 0.0;
    }
    v[..take].iter().sum::<f32>() / take as f32
}
fn end_window_mean(v: &[f32], n: usize) -> f32 {
    let take = n.min(v.len());
    if take == 0 {
        return 0.0;
    }
    v[v.len() - take..].iter().sum::<f32>() / take as f32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    fs::create_dir_all(&args.out)?;

    let (mut ref_samples, ref_sr) = load_mono_wav(&args.reference)?;
    let (mut cand_samples, cand_sr) = load_mono_wav(&args.candidate)?;

    // load_mono_wav already enforces 44.1 kHz, but double-check both match.
    if ref_sr != cand_sr {
        return Err(format!(
            "sample rate mismatch: reference {} vs candidate {}",
            ref_sr, cand_sr
        )
        .into());
    }
    let sr = ref_sr;

    // Length matching: truncate both to min(len) with stderr warning.
    if ref_samples.len() != cand_samples.len() {
        let n = ref_samples.len().min(cand_samples.len());
        eprintln!(
            "warning: length mismatch (reference {} samples, candidate {} samples); \
             truncating both to {} samples ({:.3} s)",
            ref_samples.len(),
            cand_samples.len(),
            n,
            n as f32 / sr as f32
        );
        ref_samples.truncate(n);
        cand_samples.truncate(n);
    }

    let f0 = midi_to_hz(args.note);
    let stft_ref = stft(&ref_samples, sr);
    let stft_cand = stft(&cand_samples, sr);

    let h_ref = harmonic_tracks(&stft_ref, f0, args.max_harmonics);
    let h_cand = harmonic_tracks(&stft_cand, f0, args.max_harmonics);

    // Full metric stack.
    let lsd_db = log_spectral_distance_db(&stft_ref, &stft_cand);
    let mrstft = mr_stft_l1(&ref_samples, &cand_samples, sr);
    let b_ref: InharmonicityResult = estimate_inharmonicity_b(&stft_ref, f0, args.max_harmonics);
    let b_cand: InharmonicityResult =
        estimate_inharmonicity_b(&stft_cand, f0, args.max_harmonics);
    let b_residual = (b_cand.b - b_ref.b).abs();
    let t60_loss = t60_vector_loss(&h_ref, &h_cand);
    let onset_l2 = onset_envelope_l2(&ref_samples, &cand_samples, sr, 80);
    let centroid_mse = centroid_trajectory_mse(&stft_ref, &stft_cand);

    // Spectrograms
    spectrogram_png(&stft_ref, &args.out.join("spectrogram_reference.png"))?;
    spectrogram_png(&stft_cand, &args.out.join("spectrogram_candidate.png"))?;

    // Per-harmonic deltas
    let mut deltas: Vec<HarmonicDelta> = Vec::new();
    for r in &h_ref {
        if let Some(c) = h_cand.iter().find(|c| c.n == r.n) {
            let df_cents =
                1200.0 * (c.freq_observed_hz / r.freq_observed_hz).max(1e-12).log2();
            let d_t60 = c.t60_sec - r.t60_sec;
            let d_init = c.initial_db - r.initial_db;
            deltas.push(HarmonicDelta {
                n: r.n,
                df_cents,
                d_t60_sec: d_t60,
                d_initial_db: d_init,
            });
        }
    }

    // harmonics.json (kept for backward compat with existing tooling)
    let harmonics_json = serde_json::to_string_pretty(&json!({
        "reference": &h_ref,
        "candidate": &h_cand,
        "deltas": &deltas,
    }))?;
    fs::write(args.out.join("harmonics.json"), harmonics_json)?;

    // centroid.csv (legacy)
    let centroid_ref = spectral_centroid_per_frame(&stft_ref);
    let centroid_cand = spectral_centroid_per_frame(&stft_cand);
    let n_frames = centroid_ref.len().min(centroid_cand.len());
    let mut frame_times: Vec<f32> = Vec::with_capacity(n_frames);
    {
        let f = File::create(args.out.join("centroid.csv"))?;
        let mut w = BufWriter::new(f);
        writeln!(w, "frame_idx,t_sec,reference_hz,candidate_hz")?;
        for i in 0..n_frames {
            let t = stft_ref.frame_to_sec(i);
            frame_times.push(t);
            writeln!(
                w,
                "{},{:.4},{:.2},{:.2}",
                i, t, centroid_ref[i], centroid_cand[i]
            )?;
        }
    }

    // metrics.json — full machine-readable metric stack
    let duration_sec = ref_samples.len() as f32 / sr as f32;
    let metrics_doc = json!({
        "reference_path": args.reference.display().to_string(),
        "candidate_path": args.candidate.display().to_string(),
        "note": args.note,
        "f0_hz": f0,
        "duration_sec": duration_sec,
        "sample_rate": sr,
        "metrics": {
            "lsd_db": lsd_db,
            "mrstft_l1": mrstft,
            "b_residual": b_residual,
            "b_reference": b_ref.b,
            "b_candidate": b_cand.b,
            "b_r2_reference": b_ref.r2,
            "b_r2_candidate": b_cand.r2,
            "t60_vector_loss": t60_loss,
            "onset_l2_80ms": onset_l2,
            "centroid_trajectory_mse": centroid_mse,
        },
        "harmonics": {
            "reference": &h_ref,
            "candidate": &h_cand,
        },
        "centroid_per_frame": {
            "reference": &centroid_ref[..n_frames],
            "candidate": &centroid_cand[..n_frames],
            "frame_times_sec": &frame_times,
        }
    });
    fs::write(
        args.out.join("metrics.json"),
        serde_json::to_string_pretty(&metrics_doc)?,
    )?;

    // summary.txt — diff-friendly, fixed-width
    write_summary(
        &args,
        &ref_samples,
        &cand_samples,
        sr,
        f0,
        lsd_db,
        mrstft,
        &b_ref,
        &b_cand,
        b_residual,
        t60_loss,
        onset_l2,
        centroid_mse,
        &h_ref,
        &h_cand,
        &centroid_ref,
        &centroid_cand,
    )?;

    // Single-line stdout headline (depends on metric selection)
    match args.metric {
        MetricSel::Lsd | MetricSel::All => {
            println!("LSD: {:.2} dB", lsd_db);
        }
        MetricSel::MrStft => println!("MR-STFT L1: {:.4}", mrstft),
        MetricSel::B => println!(
            "B residual: {:.3e} (ref {:.3e}, cand {:.3e})",
            b_residual, b_ref.b, b_cand.b
        ),
        MetricSel::T60 => println!("T60 vector loss: {:.4}", t60_loss),
        MetricSel::Onset => println!("Onset L2 (80 ms): {:.4}", onset_l2),
        MetricSel::Centroid => println!("Centroid trajectory MSE: {:.2}", centroid_mse),
    }
    println!("report dir: {}", args.out.display());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_summary(
    args: &Args,
    ref_samples: &[f32],
    cand_samples: &[f32],
    sr: u32,
    f0: f32,
    lsd_db: f32,
    mrstft: f32,
    b_ref: &InharmonicityResult,
    b_cand: &InharmonicityResult,
    b_residual: f32,
    t60_loss: f32,
    onset_l2: f32,
    centroid_mse: f32,
    h_ref: &[HarmonicTrack],
    h_cand: &[HarmonicTrack],
    centroid_ref: &[f32],
    centroid_cand: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let ref_dur = ref_samples.len() as f32 / sr as f32;
    let cand_dur = cand_samples.len() as f32 / sr as f32;
    let ref_path = args.reference.display().to_string();
    let cand_path = args.candidate.display().to_string();

    let f = File::create(args.out.join("summary.txt"))?;
    let mut w = BufWriter::new(f);

    writeln!(
        w,
        "=== keysynth analyse v0.2 (issue #1 metric stack) ==="
    )?;
    writeln!(
        w,
        "reference: {:<48} ({:.2} s, {} samples)",
        ref_path,
        ref_dur,
        ref_samples.len()
    )?;
    writeln!(
        w,
        "candidate: {:<48} ({:.2} s, {} samples)",
        cand_path,
        cand_dur,
        cand_samples.len()
    )?;
    writeln!(
        w,
        "note: {} ({}, {:.2} Hz)",
        args.note,
        note_name(args.note),
        f0
    )?;
    writeln!(w, "sample_rate: {} Hz", sr)?;
    writeln!(w)?;

    // -------- METRICS table --------
    writeln!(w, "--- METRICS (lower = closer match) ---")?;
    writeln!(
        w,
        "{:<22} {:<13} {:<10} {}",
        "metric", "value", "units", "notes"
    )?;
    if args.metric.show(MetricSel::Lsd) {
        writeln!(
            w,
            "{:<22} {:<13.4} {:<10} {}",
            "lsd_db", lsd_db, "dB", "legacy single-window log-spectral distance"
        )?;
    }
    if args.metric.show(MetricSel::MrStft) {
        writeln!(
            w,
            "{:<22} {:<13.4} {:<10} {}",
            "mrstft_l1", mrstft, "(lin+log)", "multi-res STFT L1, windows 512/1024/2048"
        )?;
    }
    if args.metric.show(MetricSel::B) {
        writeln!(
            w,
            "{:<22} {:<13.5} {:<10} {}",
            "b_residual", b_residual, "coeff", "|B_cand - B_ref|"
        )?;
    }
    if args.metric.show(MetricSel::T60) {
        writeln!(
            w,
            "{:<22} {:<13.4} {:<10} {}",
            "t60_vector_loss", t60_loss, "s", "Bank Taylor weighted L2 over first 16 partials"
        )?;
    }
    if args.metric.show(MetricSel::Onset) {
        writeln!(
            w,
            "{:<22} {:<13.4} {:<10} {}",
            "onset_l2_80ms", onset_l2, "RMS", "L2 of RMS envelope first 80ms"
        )?;
    }
    if args.metric.show(MetricSel::Centroid) {
        writeln!(
            w,
            "{:<22} {:<13.2} {:<10} {}",
            "centroid_traj_mse", centroid_mse, "Hz^2", "per-frame centroid MSE"
        )?;
    }
    writeln!(w)?;

    // -------- INHARMONICITY --------
    if args.metric.show(MetricSel::B) {
        writeln!(w, "--- INHARMONICITY ---")?;
        writeln!(
            w,
            "{:<22} {:<13} {:<13} {}",
            "", "reference", "candidate", "delta"
        )?;
        writeln!(
            w,
            "{:<22} {:<13.3e} {:<13.3e} {:+.3e}",
            "B coefficient",
            b_ref.b,
            b_cand.b,
            b_cand.b - b_ref.b
        )?;
        writeln!(
            w,
            "{:<22} {:<13.4} {:<13.4} {:+.4}",
            "R^2 of fit",
            b_ref.r2,
            b_cand.r2,
            b_cand.r2 - b_ref.r2
        )?;
        writeln!(
            w,
            "{:<22} {:<13} {:<13} {:+}",
            "partials used",
            b_ref.n_partials_used,
            b_cand.n_partials_used,
            b_cand.n_partials_used as i64 - b_ref.n_partials_used as i64
        )?;
        writeln!(w)?;
    }

    // -------- PER-HARMONIC --------
    writeln!(
        w,
        "--- PER-HARMONIC (top 8, all in same units as before) ---"
    )?;
    writeln!(
        w,
        "{:>4} {:>10} {:>10} {:>8} {:>9} {:>9} {:>9} {:>9} {:>10} {:>7}",
        "n",
        "freq_ref",
        "freq_cand",
        "dCent",
        "T60_ref",
        "T60_cand",
        "dT60",
        "init_ref",
        "init_cand",
        "dInit",
    )?;
    for r in h_ref.iter().take(8) {
        let c = h_cand.iter().find(|c| c.n == r.n);
        let (cf, ct60, ci) = match c {
            Some(c) => (c.freq_observed_hz, c.t60_sec, c.initial_db),
            None => (f32::NAN, f32::NAN, f32::NAN),
        };
        let dc = 1200.0 * (cf / r.freq_observed_hz).max(1e-12).log2();
        let dt = ct60 - r.t60_sec;
        let di = ci - r.initial_db;
        writeln!(
            w,
            "{:>4} {:>10.1} {:>10.1} {:>+8.1} {:>8.2}s {:>8.2}s {:>+8.2}s {:>7.1}dB {:>8.1}dB {:>+7.1}",
            r.n, r.freq_observed_hz, cf, dc, r.t60_sec, ct60, dt, r.initial_db, ci, di,
        )?;
    }
    writeln!(w)?;

    // -------- SPECTRAL CENTROID --------
    if args.metric.show(MetricSel::Centroid) {
        let cm_ref = mean(centroid_ref);
        let cm_cand = mean(centroid_cand);
        let cs_ref = start_window_mean(centroid_ref, 10);
        let cs_cand = start_window_mean(centroid_cand, 10);
        let ce_ref = end_window_mean(centroid_ref, 10);
        let ce_cand = end_window_mean(centroid_cand, 10);
        writeln!(w, "--- SPECTRAL CENTROID ---")?;
        writeln!(
            w,
            "{:<22} {:<13} {:<13} {}",
            "", "reference", "candidate", "delta"
        )?;
        writeln!(
            w,
            "{:<22} {:<13.0} {:<13.0} {:+.0}",
            "mean Hz",
            cm_ref,
            cm_cand,
            cm_cand - cm_ref
        )?;
        writeln!(
            w,
            "{:<22} {:<13.0} {:<13.0} {:+.0}",
            "start (first 10 fr.)",
            cs_ref,
            cs_cand,
            cs_cand - cs_ref
        )?;
        writeln!(
            w,
            "{:<22} {:<13.0} {:<13.0} {:+.0}",
            "end (last 10 fr.)",
            ce_ref,
            ce_cand,
            ce_cand - ce_ref
        )?;
    }

    Ok(())
}
