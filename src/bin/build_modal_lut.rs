//! Build a per-note modal LUT from SFZ Salamander reference WAVs.
//!
//! For each `note_NN.wav` file in `--input-dir`, this binary loads the
//! mono signal, runs the issue #3 extractors (`decompose`, `extract_t60`,
//! `extract_attack`), and emits a JSON LUT consumed by
//! `voices::piano_modal::ModalPianoVoice` for the supervised piano-realism
//! path (issue #3 A2).
//!
//! Usage:
//!   build_modal_lut [--input-dir DIR] [--output PATH] [--max-partials N]
//!
//! Defaults:
//!   --input-dir     bench-out/REF/sfz_salamander_multi
//!   --output        bench-out/REF/sfz_salamander_multi/modal_lut.json
//!   --max-partials  16

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use hound::WavReader;
use keysynth::extract::attack::extract_attack;
use keysynth::extract::decompose::decompose;
use keysynth::extract::t60::extract_t60;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct LutMode {
    freq_hz: f32,
    t60_sec: f32,
    init_db: f32,
}

#[derive(Serialize, Deserialize, Debug)]
struct LutAttack {
    time_to_peak_s: f32,
    peak_db: f32,
    post_peak_slope_db_s: f32,
}

#[derive(Serialize, Deserialize, Debug)]
struct LutEntry {
    midi_note: u8,
    f0_hz: f32,
    modes: Vec<LutMode>,
    attack: LutAttack,
}

#[derive(Serialize, Deserialize, Debug)]
struct ModalLut {
    schema_version: u32,
    source: String,
    rendered: String,
    max_partials: usize,
    lut: Vec<LutEntry>,
}

fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, f32), String> {
    let mut r = WavReader::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
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
        Ok((samples, sr))
    } else {
        let mut mono = Vec::with_capacity(samples.len() / channels);
        for frame in samples.chunks_exact(channels) {
            let sum: f32 = frame.iter().sum();
            mono.push(sum / channels as f32);
        }
        Ok((mono, sr))
    }
}

fn midi_to_f0(note: u8) -> f32 {
    440.0 * 2f32.powf((note as f32 - 69.0) / 12.0)
}

/// Parse `note_NN.wav` filename → MIDI note. Returns None for non-matches.
fn parse_note_filename(name: &str) -> Option<u8> {
    let stem = name.strip_suffix(".wav")?;
    let n = stem.strip_prefix("note_")?;
    n.parse::<u8>().ok()
}

struct CliArgs {
    input_dir: PathBuf,
    output: PathBuf,
    max_partials: usize,
}

fn parse_args() -> Result<CliArgs, String> {
    let mut input_dir = PathBuf::from("bench-out/REF/sfz_salamander_multi");
    let mut output: Option<PathBuf> = None;
    let mut max_partials: usize = 16;

    let argv: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--input-dir" => {
                i += 1;
                input_dir = PathBuf::from(argv.get(i).ok_or("--input-dir requires a value")?);
            }
            "--output" => {
                i += 1;
                output = Some(PathBuf::from(
                    argv.get(i).ok_or("--output requires a value")?,
                ));
            }
            "--max-partials" => {
                i += 1;
                max_partials = argv
                    .get(i)
                    .ok_or("--max-partials requires a value")?
                    .parse()
                    .map_err(|e| format!("--max-partials parse: {e}"))?;
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
        i += 1;
    }

    let output = output.unwrap_or_else(|| input_dir.join("modal_lut.json"));
    Ok(CliArgs {
        input_dir,
        output,
        max_partials,
    })
}

fn print_help() {
    eprintln!(
        "build_modal_lut [--input-dir DIR] [--output PATH] [--max-partials N]\n\n\
         Defaults:\n\
           --input-dir     bench-out/REF/sfz_salamander_multi\n\
           --output        <input-dir>/modal_lut.json\n\
           --max-partials  16"
    );
}

fn run(args: &CliArgs) -> Result<ModalLut, String> {
    if !args.input_dir.is_dir() {
        return Err(format!(
            "input directory does not exist: {}",
            args.input_dir.display()
        ));
    }

    // Discover note_NN.wav files, sort by MIDI note.
    let mut notes: Vec<(u8, PathBuf)> = Vec::new();
    for entry in
        fs::read_dir(&args.input_dir).map_err(|e| format!("read_dir {:?}: {e}", args.input_dir))?
    {
        let entry = entry.map_err(|e| format!("read_dir entry: {e}"))?;
        let path = entry.path();
        let Some(fname) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if let Some(n) = parse_note_filename(fname) {
            notes.push((n, path));
        }
    }
    notes.sort_by_key(|(n, _)| *n);

    if notes.is_empty() {
        return Err(format!(
            "no note_NN.wav files in {}",
            args.input_dir.display()
        ));
    }

    let mut lut: Vec<LutEntry> = Vec::with_capacity(notes.len());
    for (midi_note, path) in &notes {
        let (sig, sr) = read_wav_mono(path)?;
        let f0 = midi_to_f0(*midi_note);
        eprintln!(
            "  note {:>3} f0={:>7.2} Hz  ({}, {:.2} s)",
            midi_note,
            f0,
            path.display(),
            sig.len() as f32 / sr
        );

        let partials = decompose(&sig, sr, f0, args.max_partials);
        let t60 = extract_t60(&sig, sr, &partials);
        let att = extract_attack(&sig, sr, 100.0);

        let mut modes: Vec<LutMode> = Vec::with_capacity(partials.len());
        for (i, p) in partials.iter().enumerate() {
            let t = t60.seconds.get(i).copied().unwrap_or(-1.0);
            modes.push(LutMode {
                freq_hz: p.freq_hz,
                t60_sec: t,
                init_db: p.init_db,
            });
        }

        lut.push(LutEntry {
            midi_note: *midi_note,
            f0_hz: f0,
            modes,
            attack: LutAttack {
                time_to_peak_s: att.time_to_peak_s,
                peak_db: att.peak_db,
                post_peak_slope_db_s: att.post_peak_slope_db_s,
            },
        });
    }

    let rendered_template = format!(
        "{}/note_NN.wav",
        args.input_dir.display().to_string().replace('\\', "/")
    );

    Ok(ModalLut {
        schema_version: 1,
        source: "SFZ Salamander Grand Piano V3 (CC-BY 3.0)".to_string(),
        rendered: rendered_template,
        max_partials: args.max_partials,
        lut,
    })
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("build_modal_lut: error: {e}");
            print_help();
            std::process::exit(2);
        }
    };

    eprintln!(
        "build_modal_lut: input={} output={} max_partials={}",
        args.input_dir.display(),
        args.output.display(),
        args.max_partials
    );

    let table = match run(&args) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("build_modal_lut: failed: {e}");
            std::process::exit(1);
        }
    };

    let json = match serde_json::to_string_pretty(&table) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("build_modal_lut: serialise: {e}");
            std::process::exit(1);
        }
    };

    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = fs::create_dir_all(parent) {
                eprintln!(
                    "build_modal_lut: create_dir_all {:?}: {e}",
                    parent.display()
                );
                std::process::exit(1);
            }
        }
    }
    if let Err(e) = fs::write(&args.output, &json) {
        eprintln!("build_modal_lut: write {:?}: {e}", args.output.display());
        std::process::exit(1);
    }

    eprintln!(
        "build_modal_lut: wrote {} entries to {}",
        table.lut.len(),
        args.output.display()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_filename_recognises_note_nn_wav() {
        assert_eq!(parse_note_filename("note_36.wav"), Some(36));
        assert_eq!(parse_note_filename("note_60.wav"), Some(60));
        assert_eq!(parse_note_filename("note_84.wav"), Some(84));
        assert_eq!(parse_note_filename("note_60.txt"), None);
        assert_eq!(parse_note_filename("foo.wav"), None);
        assert_eq!(parse_note_filename("note_xx.wav"), None);
    }

    #[test]
    fn midi_to_f0_matches_a4_and_c4() {
        assert!((midi_to_f0(69) - 440.0).abs() < 1e-3);
        assert!((midi_to_f0(60) - 261.625_57).abs() < 1e-2);
        assert!((midi_to_f0(36) - 65.406_4).abs() < 1e-2);
    }
}
