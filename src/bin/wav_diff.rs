//! wav_diff — two-WAV numerical diff. Cross-correlation peak+lag,
//! multi-resolution STFT distance, spectral envelope ΔdB, peak/RMS
//! deltas. Use to gate ship: if `mr_stft_distance` jumps vs the last
//! known-good render, something changed in timbre.
//!
//! Usage:
//!   wav_diff <a.wav> <b.wav> [--bands 32] [--pretty]

use std::path::PathBuf;
use std::process::ExitCode;

use hound::WavReader;
use keysynth::audio_audit::diff;

struct Args {
    a: PathBuf,
    b: PathBuf,
    bands: usize,
    pretty: bool,
}

const USAGE: &str = "wav_diff <a.wav> <b.wav> [--bands N] [--pretty]";

fn parse_args() -> Result<Args, String> {
    let mut positional: Vec<PathBuf> = Vec::new();
    let mut bands = 32_usize;
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
            "--bands" => {
                bands = take(i)?.parse().map_err(|e| format!("--bands: {e}"))?;
                i += 2;
            }
            "--pretty" => {
                pretty = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!("{USAGE}");
                std::process::exit(0);
            }
            other if !other.starts_with("--") => {
                positional.push(PathBuf::from(other));
                i += 1;
            }
            other => return Err(format!("unknown arg {other:?}")),
        }
    }
    if positional.len() != 2 {
        return Err(format!(
            "expected exactly 2 positional WAV paths, got {}",
            positional.len()
        ));
    }
    let mut it = positional.into_iter();
    Ok(Args {
        a: it.next().unwrap(),
        b: it.next().unwrap(),
        bands,
        pretty,
    })
}

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
    let (a, sra) = match load_wav(&args.a) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("error reading a: {e}");
            return ExitCode::from(1);
        }
    };
    let (b, srb) = match load_wav(&args.b) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("error reading b: {e}");
            return ExitCode::from(1);
        }
    };
    if sra != srb {
        eprintln!(
            "error: sample rate mismatch ({} Hz vs {} Hz). Resample to a common rate first.",
            sra, srb
        );
        return ExitCode::from(1);
    }
    let report = diff(&a, &b, sra, args.bands);
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
