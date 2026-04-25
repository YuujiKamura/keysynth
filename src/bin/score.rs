//! Compute the perceptually-weighted multi-metric loss between a
//! reference WAV and a candidate WAV (issue #3 P3 skeleton).
//!
//! Usage:
//! ```text
//!   score --reference REF.wav --candidate CAND.wav [--note 60]
//!         [--weights WEIGHTS.json] [--json]
//! ```
//!
//! Defaults: reference =
//! `bench-out/REF/sfz_salamander_grand_v3_C4.wav`, MIDI note = 60 (C4).

use std::path::PathBuf;
use std::process::ExitCode;

use keysynth::score::{loss_paths, LossBreakdown, LossWeights};
use keysynth::synth::midi_to_freq;

#[derive(Debug)]
struct Args {
    reference: PathBuf,
    candidate: PathBuf,
    note: u8,
    weights_path: Option<PathBuf>,
    json: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut reference = PathBuf::from("bench-out/REF/sfz_salamander_grand_v3_C4.wav");
    let mut candidate: Option<PathBuf> = None;
    let mut note: u8 = 60;
    let mut weights_path: Option<PathBuf> = None;
    let mut json = false;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--reference" => {
                reference = PathBuf::from(it.next().ok_or("--reference needs a path")?);
            }
            "--candidate" => {
                candidate = Some(PathBuf::from(it.next().ok_or("--candidate needs a path")?));
            }
            "--note" => {
                note = it
                    .next()
                    .ok_or("--note needs a value")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| format!("--note: {e}"))?;
            }
            "--weights" => {
                weights_path = Some(PathBuf::from(it.next().ok_or("--weights needs a path")?));
            }
            "--json" => json = true,
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    let candidate = candidate.ok_or_else(|| "--candidate is required".to_string())?;
    Ok(Args {
        reference,
        candidate,
        note,
        weights_path,
        json,
    })
}

fn print_usage() {
    eprintln!(
        "score --reference REF.wav --candidate CAND.wav \
         [--note 60] [--weights WEIGHTS.json] [--json]"
    );
}

fn load_weights(path: &std::path::Path) -> Result<LossWeights, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read weights {}: {e}", path.display()))?;
    serde_json::from_slice::<LossWeights>(&bytes)
        .map_err(|e| format!("parse weights {}: {e}", path.display()))
}

fn print_human(
    reference: &std::path::Path,
    candidate: &std::path::Path,
    w: &LossWeights,
    b: &LossBreakdown,
) {
    println!(
        "=== score: {} vs {} ===",
        reference.display(),
        candidate.display()
    );
    println!("  metric         raw          weight       term");
    println!(
        "  mr_stft        {:<12.4} {:<12.4e} {:<.4}",
        b.mr_stft_raw, w.mr_stft, b.mr_stft_term
    );
    println!(
        "  t60            {:<12.4} {:<12.4e} {:<.4}",
        b.t60_raw, w.t60, b.t60_term
    );
    println!(
        "  onset          {:<12.4} {:<12.4e} {:<.4}",
        b.onset_raw, w.onset, b.onset_term
    );
    println!(
        "  centroid       {:<12.2} {:<12.4e} {:<.4}",
        b.centroid_raw, w.centroid, b.centroid_term
    );
    println!(
        "  b_residual     {:<12.4e} {:<12.4e} {:<.4}",
        b.b_residual_raw, w.b_residual, b.b_residual_term
    );
    println!("  --");
    println!("  total          {:.4}", b.total);
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("score: {e}");
            print_usage();
            return ExitCode::from(2);
        }
    };

    let weights = match args.weights_path.as_deref() {
        Some(p) => match load_weights(p) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("score: {e}");
                return ExitCode::from(2);
            }
        },
        None => LossWeights::default(),
    };

    let f0 = midi_to_freq(args.note);
    let breakdown = match loss_paths(&args.reference, &args.candidate, f0, &weights) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("score: {e}");
            return ExitCode::from(1);
        }
    };

    if args.json {
        match serde_json::to_string_pretty(&breakdown) {
            Ok(s) => println!("{s}"),
            Err(e) => {
                eprintln!("score: serialise breakdown: {e}");
                return ExitCode::from(1);
            }
        }
    } else {
        print_human(&args.reference, &args.candidate, &weights, &breakdown);
    }
    ExitCode::SUCCESS
}
