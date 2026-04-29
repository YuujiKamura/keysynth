//! wav_audit — single-WAV numerical audit. Emits one JSON document with
//! peak/RMS/crest, silence intervals, RMS envelope, spectral envelope,
//! and dominant frequencies. Built so an AI agent (which cannot listen)
//! can decide "is this render qualitatively wrong?" from numbers alone.
//!
//! Usage:
//!   wav_audit <path.wav>
//!     [--silence-threshold-db -60]
//!     [--silence-min-ms 50]
//!     [--rms-window-ms 50]
//!     [--bands 32]
//!     [--top-freqs 8]
//!     [--pretty]

use std::path::PathBuf;
use std::process::ExitCode;

use hound::WavReader;
use keysynth::audio_audit::audit;

struct Args {
    path: PathBuf,
    silence_threshold_db: f32,
    silence_min_ms: u32,
    rms_window_ms: u32,
    bands: usize,
    top_freqs: usize,
    pretty: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut path: Option<PathBuf> = None;
    let mut silence_threshold_db = -60.0_f32;
    let mut silence_min_ms = 50_u32;
    let mut rms_window_ms = 50_u32;
    let mut bands = 32_usize;
    let mut top_freqs = 8_usize;
    let mut pretty = false;

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let a = argv[i].clone();
        let take = |i: usize| -> Result<String, String> {
            argv.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("missing value for {a}"))
        };
        match a.as_str() {
            "--silence-threshold-db" => {
                silence_threshold_db = take(i)?.parse().map_err(|e| format!("--silence-threshold-db: {e}"))?;
                i += 2;
            }
            "--silence-min-ms" => {
                silence_min_ms = take(i)?.parse().map_err(|e| format!("--silence-min-ms: {e}"))?;
                i += 2;
            }
            "--rms-window-ms" => {
                rms_window_ms = take(i)?.parse().map_err(|e| format!("--rms-window-ms: {e}"))?;
                i += 2;
            }
            "--bands" => {
                bands = take(i)?.parse().map_err(|e| format!("--bands: {e}"))?;
                i += 2;
            }
            "--top-freqs" => {
                top_freqs = take(i)?.parse().map_err(|e| format!("--top-freqs: {e}"))?;
                i += 2;
            }
            "--pretty" => {
                pretty = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!("{}", USAGE);
                std::process::exit(0);
            }
            other if !other.starts_with("--") && path.is_none() => {
                path = Some(PathBuf::from(other));
                i += 1;
            }
            other => return Err(format!("unknown arg {other:?}")),
        }
    }
    Ok(Args {
        path: path.ok_or("missing <path.wav>")?,
        silence_threshold_db,
        silence_min_ms,
        rms_window_ms,
        bands,
        top_freqs,
        pretty,
    })
}

const USAGE: &str = "wav_audit <path.wav> [--silence-threshold-db N] \
[--silence-min-ms N] [--rms-window-ms N] [--bands N] [--top-freqs N] [--pretty]";

fn load_wav(path: &std::path::Path) -> Result<(Vec<f32>, u32), String> {
    let mut reader = WavReader::open(path).map_err(|e| format!("open {:?}: {}", path, e))?;
    let spec = reader.spec();
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
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
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
    Ok((mono, spec.sample_rate))
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n\nusage: {USAGE}");
            return ExitCode::from(2);
        }
    };
    let (samples, sr) = match load_wav(&args.path) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };
    let report = audit(
        &samples,
        sr,
        args.silence_threshold_db,
        args.silence_min_ms,
        args.rms_window_ms,
        args.bands,
        args.top_freqs,
    );
    let out = if args.pretty {
        serde_json::to_string_pretty(&report)
    } else {
        serde_json::to_string(&report)
    };
    match out {
        Ok(s) => {
            println!("{s}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("serialise error: {e}");
            ExitCode::from(1)
        }
    }
}
