//! Quantitative WAV comparison harness.
//!
//! Usage:
//!   analyse --reference REF.wav --candidate CAND.wav [--note 60]
//!           [--out report/] [--max-harmonics 16]
//!
//! Outputs (under --out):
//!   spectrogram_reference.png
//!   spectrogram_candidate.png
//!   harmonics.json
//!   centroid.csv
//!   summary.txt
//!
//! The point: replace ear-based "sounds more piano-y" with falsifiable
//! numbers. The headline single number is LOG-SPECTRAL DISTANCE in dB.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use hound::WavReader;
use serde::Serialize;

use keysynth::analysis::{
    harmonic_tracks, log_spectral_distance_db, spectral_centroid_per_frame, spectrogram_png,
    stft, HarmonicTrack,
};

struct Args {
    reference: PathBuf,
    candidate: PathBuf,
    note: u8,
    out: PathBuf,
    max_harmonics: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut reference: Option<PathBuf> = None;
    let mut candidate: Option<PathBuf> = None;
    let mut note: u8 = 60;
    let mut out: PathBuf = PathBuf::from("report");
    let mut max_harmonics: usize = 16;

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let a = &argv[i];
        let next = || -> Result<&String, String> {
            argv.get(i + 1)
                .ok_or_else(|| format!("missing value for {}", a))
        };
        match a.as_str() {
            "--reference" => {
                reference = Some(PathBuf::from(next()?));
                i += 2;
            }
            "--candidate" => {
                candidate = Some(PathBuf::from(next()?));
                i += 2;
            }
            "--note" => {
                note = next()?.parse().map_err(|e| format!("bad --note: {}", e))?;
                i += 2;
            }
            "--out" => {
                out = PathBuf::from(next()?);
                i += 2;
            }
            "--max-harmonics" => {
                max_harmonics = next()?
                    .parse()
                    .map_err(|e| format!("bad --max-harmonics: {}", e))?;
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!(
                    "analyse --reference REF.wav --candidate CAND.wav \
                     [--note 60] [--out report/] [--max-harmonics 16]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {}", other)),
        }
    }
    Ok(Args {
        reference: reference.ok_or("missing --reference")?,
        candidate: candidate.ok_or("missing --candidate")?,
        note,
        out,
        max_harmonics,
    })
}

fn load_mono_wav(path: &std::path::Path) -> Result<(Vec<f32>, u32), String> {
    let mut reader = WavReader::open(path).map_err(|e| format!("open {:?}: {}", path, e))?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err(format!(
            "{:?} has {} channels; mono only",
            path, spec.channels
        ));
    }
    let sr = spec.sample_rate;
    let samples: Vec<f32> = match spec.sample_format {
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
    Ok((samples, sr))
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

#[derive(Serialize)]
struct Report<'a> {
    reference: &'a [HarmonicTrack],
    candidate: &'a [HarmonicTrack],
    deltas: Vec<HarmonicDelta>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    fs::create_dir_all(&args.out)?;

    let (ref_samples, ref_sr) = load_mono_wav(&args.reference)?;
    let (cand_samples, cand_sr) = load_mono_wav(&args.candidate)?;

    if ref_sr != cand_sr {
        return Err(format!(
            "sample rate mismatch: reference {} vs candidate {}",
            ref_sr, cand_sr
        )
        .into());
    }

    // Pad/truncate to identical length so STFT shapes line up for LSD.
    let n = ref_samples.len().max(cand_samples.len());
    let mut ref_pad = ref_samples.clone();
    let mut cand_pad = cand_samples.clone();
    ref_pad.resize(n, 0.0);
    cand_pad.resize(n, 0.0);

    let f0 = midi_to_hz(args.note);
    let stft_ref = stft(&ref_pad, ref_sr);
    let stft_cand = stft(&cand_pad, cand_sr);

    let h_ref = harmonic_tracks(&stft_ref, f0, args.max_harmonics);
    let h_cand = harmonic_tracks(&stft_cand, f0, args.max_harmonics);
    let lsd = log_spectral_distance_db(&stft_ref, &stft_cand);

    // Spectrograms
    spectrogram_png(&stft_ref, &args.out.join("spectrogram_reference.png"))?;
    spectrogram_png(&stft_cand, &args.out.join("spectrogram_candidate.png"))?;

    // Deltas
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

    // harmonics.json
    let report = Report {
        reference: &h_ref,
        candidate: &h_cand,
        deltas,
    };
    let json = serde_json::to_string_pretty(&serde_json::json!({
        "reference": report.reference,
        "candidate": report.candidate,
        "deltas": report.deltas,
    }))?;
    fs::write(args.out.join("harmonics.json"), json)?;

    // centroid.csv
    let centroid_ref = spectral_centroid_per_frame(&stft_ref);
    let centroid_cand = spectral_centroid_per_frame(&stft_cand);
    let n_frames = centroid_ref.len().min(centroid_cand.len());
    {
        let f = File::create(args.out.join("centroid.csv"))?;
        let mut w = BufWriter::new(f);
        writeln!(w, "frame_idx,t_sec,reference_hz,candidate_hz")?;
        for i in 0..n_frames {
            let t = stft_ref.frame_to_sec(i);
            writeln!(w, "{},{:.4},{:.2},{:.2}", i, t, centroid_ref[i], centroid_cand[i])?;
        }
    }

    // summary.txt
    let ref_dur = ref_samples.len() as f32 / ref_sr as f32;
    let cand_dur = cand_samples.len() as f32 / cand_sr as f32;
    let ref_name = args
        .reference
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let cand_name = args
        .candidate
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();

    let centroid_ref_mean = mean(&centroid_ref);
    let centroid_cand_mean = mean(&centroid_cand);
    let centroid_ref_end = end_window_mean(&centroid_ref, 10);
    let centroid_cand_end = end_window_mean(&centroid_cand, 10);
    let centroid_ref_start = start_window_mean(&centroid_ref, 10);
    let centroid_cand_start = start_window_mean(&centroid_cand, 10);

    let f = File::create(args.out.join("summary.txt"))?;
    let mut w = BufWriter::new(f);
    writeln!(w, "=== keysynth analyse v0.1 ===")?;
    writeln!(
        w,
        "reference: {:<32} ({:.2} s, {} samples)",
        ref_name,
        ref_dur,
        ref_samples.len()
    )?;
    writeln!(
        w,
        "candidate: {:<32} ({:.2} s, {} samples)",
        cand_name,
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
    writeln!(w, "----")?;
    writeln!(w, "LOG-SPECTRAL DISTANCE: {:.2} dB    <-- single number to optimize", lsd)?;
    writeln!(w, "----")?;
    writeln!(w, "PER-HARMONIC (top 8):")?;
    writeln!(
        w,
        "  {:>2}  {:>9}  {:>9}  {:>7}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>6}",
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
            "  {:>2}  {:>8.1}  {:>8.1}  {:>+6.1}  {:>7.2}s  {:>7.2}s  {:>+7.2}s  {:>6.1}dB  {:>6.1}dB  {:>+5.1}",
            r.n,
            r.freq_observed_hz,
            cf,
            dc,
            r.t60_sec,
            ct60,
            dt,
            r.initial_db,
            ci,
            di,
        )?;
    }
    writeln!(w, "----")?;
    writeln!(w, "SPECTRAL CENTROID:")?;
    writeln!(
        w,
        "  reference: mean={:>5.0} Hz, start={:>5.0} Hz, end={:>5.0} Hz (delta {:+.0} Hz)",
        centroid_ref_mean,
        centroid_ref_start,
        centroid_ref_end,
        centroid_ref_end - centroid_ref_start
    )?;
    writeln!(
        w,
        "  candidate: mean={:>5.0} Hz, start={:>5.0} Hz, end={:>5.0} Hz (delta {:+.0} Hz)",
        centroid_cand_mean,
        centroid_cand_start,
        centroid_cand_end,
        centroid_cand_end - centroid_cand_start
    )?;

    println!("LOG-SPECTRAL DISTANCE: {:.2} dB", lsd);
    println!("report dir: {}", args.out.display());
    Ok(())
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
