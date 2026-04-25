//! Modal piano voice C4 pilot (issue #3 alt path).
//!
//! Reads the SFZ Salamander C4 reference, runs the issue #3 extractors
//! to derive (freq, T60, init_amp) for each of the first ~16 modes, and
//! renders a candidate WAV by driving a `ModalPianoVoice` with those
//! modes plus a default 3 ms half-sine hammer impulse. The output is
//! saved alongside the existing `piano-thick`/`piano-lite` renders so
//! `analyse` can compare it directly.
//!
//! Usage:
//!     cargo run --release --bin render_modal --
//!         --reference bench-out/REF/sfz_salamander_grand_v3_C4.wav
//!         --note 60
//!         --duration 6
//!         --hold 4
//!         --out bench-out/keysynth_piano-modal_n60.wav

use std::env;
use std::path::{Path, PathBuf};

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use keysynth::extract::attack::extract_attack;
use keysynth::extract::decompose::{decompose, Partial};
use keysynth::extract::inharmonicity::fit_b;
use keysynth::extract::t60::{extract_t60, T60Vector};
use keysynth::synth::VoiceImpl;
use keysynth::voices::piano_modal::{ModalPianoVoice, Mode};

fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, f32), String> {
    let mut r =
        WavReader::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let spec = r.spec();
    let sr = spec.sample_rate as f32;
    let channels = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => r.samples::<f32>().filter_map(Result::ok).collect(),
        SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / max)
                .collect()
        }
    };
    let mono = if channels <= 1 {
        samples
    } else {
        samples
            .chunks(channels)
            .map(|c| c.iter().copied().sum::<f32>() / channels as f32)
            .collect()
    };
    Ok((mono, sr))
}

fn write_wav_mono(path: &Path, samples: &[f32], sr: f32) -> Result<(), String> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).map_err(|e| format!("create_dir_all {}: {e}", p.display()))?;
    }
    let spec = WavSpec {
        channels: 1,
        sample_rate: sr as u32,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut w = WavWriter::create(path, spec).map_err(|e| format!("WavWriter::create: {e}"))?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let i = (clamped * i16::MAX as f32) as i16;
        w.write_sample(i)
            .map_err(|e| format!("write_sample: {e}"))?;
    }
    w.finalize().map_err(|e| format!("finalize: {e}"))?;
    Ok(())
}

fn midi_to_freq(note: u8) -> f32 {
    440.0_f32 * 2.0_f32.powf((note as f32 - 69.0) / 12.0)
}

struct Args {
    reference: PathBuf,
    note: u8,
    duration: f32,
    hold: f32,
    out: PathBuf,
    velocity: u8,
    max_partials: usize,
    /// Default T60 used for partials whose extractor returned the -1
    /// sentinel (i.e. could not measure decay over the 5 s ref).
    default_t60: f32,
    /// `extractor`: drive the modal LUT from `decompose` + `extract_t60`
    /// only (current pilot, exposes extractor bugs on h1/h2/h5).
    /// `analyse`: ignore extractor T60 / init_dB and use the values that
    /// the existing `analyse` binary reports for the SFZ Salamander C4
    /// ref. Useful to isolate "is the modal voice correct?" from "are
    /// the extractors accurate enough?".
    source: LutSource,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum LutSource {
    Extractor,
    Analyse,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            reference: PathBuf::from("bench-out/REF/sfz_salamander_grand_v3_C4.wav"),
            note: 60,
            duration: 6.0,
            hold: 4.0,
            out: PathBuf::from("bench-out/keysynth_piano-modal_n60.wav"),
            velocity: 100,
            max_partials: 8,
            default_t60: 12.0,
            source: LutSource::Extractor,
        }
    }
}

/// Pinned (freq_hz, T60_sec, init_db) for the SFZ Salamander Grand
/// Piano V3 C4 reference, as reported by the existing `analyse` binary
/// on the committed `bench-out/REF/sfz_salamander_grand_v3_C4.wav`.
/// Only used when --source=analyse.
const ANALYSE_LUT_C4: [(f32, f32, f32); 8] = [
    (261.6, 18.14, -2.7),
    (523.1, 11.21, 0.0),
    (785.6, 9.58, -18.5),
    (1048.8, 7.38, -18.4),
    (1311.7, 6.98, -14.3),
    (1577.3, 7.93, -26.1),
    (1843.7, 8.98, -17.4),
    (2111.5, 8.66, -32.9),
];

fn parse_args() -> Result<Args, String> {
    let mut out = Args::default();
    let mut iter = env::args().skip(1);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--reference" => {
                out.reference = PathBuf::from(iter.next().ok_or("--reference needs a path")?)
            }
            "--note" => {
                out.note = iter
                    .next()
                    .ok_or("--note needs a u8")?
                    .parse()
                    .map_err(|e| format!("bad --note: {e}"))?
            }
            "--duration" => {
                out.duration = iter
                    .next()
                    .ok_or("--duration needs a float")?
                    .parse()
                    .map_err(|e| format!("bad --duration: {e}"))?
            }
            "--hold" => {
                out.hold = iter
                    .next()
                    .ok_or("--hold needs a float")?
                    .parse()
                    .map_err(|e| format!("bad --hold: {e}"))?
            }
            "--out" => out.out = PathBuf::from(iter.next().ok_or("--out needs a path")?),
            "--velocity" => {
                out.velocity = iter
                    .next()
                    .ok_or("--velocity needs a u8")?
                    .parse()
                    .map_err(|e| format!("bad --velocity: {e}"))?
            }
            "--max-partials" => {
                out.max_partials = iter
                    .next()
                    .ok_or("--max-partials needs a usize")?
                    .parse()
                    .map_err(|e| format!("bad --max-partials: {e}"))?
            }
            "--default-t60" => {
                out.default_t60 = iter
                    .next()
                    .ok_or("--default-t60 needs a float")?
                    .parse()
                    .map_err(|e| format!("bad --default-t60: {e}"))?
            }
            "--source" => {
                let v = iter.next().ok_or("--source needs extractor|analyse")?;
                out.source = match v.as_str() {
                    "extractor" => LutSource::Extractor,
                    "analyse" => LutSource::Analyse,
                    other => {
                        return Err(format!("--source must be extractor|analyse, got {other}"))
                    }
                };
            }
            "--help" | "-h" => {
                eprintln!(
                    "render_modal: extract a modal LUT from a reference WAV and render \
                     it back via ModalPianoVoice.\n\n\
                     options (defaults in parens):\n  \
                     --reference PATH (bench-out/REF/sfz_salamander_grand_v3_C4.wav)\n  \
                     --note N (60)\n  \
                     --duration SEC (6)\n  \
                     --hold SEC (4)\n  \
                     --out PATH (bench-out/keysynth_piano-modal_n60.wav)\n  \
                     --velocity V (100)\n  \
                     --max-partials N (8)   how many partials to extract from the ref\n  \
                     --default-t60 SEC (12) substitute when the per-partial T60 sentinel fires"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    Ok(out)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("render_modal: {e}");
            std::process::exit(2);
        }
    };

    let (sig, sr) = match read_wav_mono(&args.reference) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("render_modal: {e}");
            std::process::exit(2);
        }
    };
    let f0 = midi_to_freq(args.note);
    println!(
        "render_modal: ref={} ({:.0} Hz, {:.2}s) note={} (f0={:.2} Hz)",
        args.reference.display(),
        sr,
        sig.len() as f32 / sr,
        args.note,
        f0
    );

    // 1. Extract partials.
    let partials: Vec<Partial> = decompose(&sig, sr, f0, args.max_partials);
    println!("\n--- decompose: {} partials ---", partials.len());
    for p in &partials {
        println!(
            "  n={:>2}  freq={:>9.3} Hz  init_db={:>+7.2}",
            p.n, p.freq_hz, p.init_db
        );
    }

    // 2. Extract per-partial T60.
    let t60: T60Vector = extract_t60(&sig, sr, &partials);

    // 3. Sanity: B coefficient (informational only).
    let bfit = fit_b(&partials);
    println!(
        "\n--- fit_b ---\n  B = {:.4e}  R² = {:.4}  n_used = {}",
        bfit.b, bfit.r_squared, bfit.n_used
    );

    // 4. Build modes. init_amp = 10^(init_db / 20).
    let mut modes: Vec<Mode> = Vec::with_capacity(partials.len());
    match args.source {
        LutSource::Extractor => {
            println!(
                "\n--- modes [source=extractor, T60 sentinels → --default-t60 = {} s] ---",
                args.default_t60
            );
            for (i, p) in partials.iter().enumerate() {
                let raw_t60 = t60.seconds.get(i).copied().unwrap_or(args.default_t60);
                let used_t60 = if raw_t60 > 0.0 {
                    raw_t60
                } else {
                    args.default_t60
                };
                let init_amp = 10f32.powf(p.init_db / 20.0);
                modes.push(Mode {
                    freq_hz: p.freq_hz,
                    t60_sec: used_t60,
                    init_amp,
                });
                let flag = if raw_t60 < 0.0 { " (sentinel)" } else { "" };
                println!(
                    "  n={:>2}  f={:>9.3} Hz  T60={:>6.2} s{}  init_amp={:.4}",
                    p.n, p.freq_hz, used_t60, flag, init_amp
                );
            }
        }
        LutSource::Analyse => {
            println!("\n--- modes [source=analyse, hardcoded SFZ Salamander C4 LUT] ---");
            for (i, &(f, t60_s, init_db)) in ANALYSE_LUT_C4.iter().enumerate() {
                let init_amp = 10f32.powf(init_db / 20.0);
                modes.push(Mode {
                    freq_hz: f,
                    t60_sec: t60_s,
                    init_amp,
                });
                println!(
                    "  n={:>2}  f={:>9.3} Hz  T60={:>6.2} s  init_db={:>+6.2}  init_amp={:.4}",
                    i + 1,
                    f,
                    t60_s,
                    init_db,
                    init_amp
                );
            }
        }
    }

    // 5. Attack envelope (informational — we use a default 3 ms half-sine
    //    in the pilot; documenting the SFZ ref's measured attack here so
    //    callers can see how far off it is).
    let att = extract_attack(&sig, sr, 100.0);
    println!(
        "\n--- attack (ref) ---\n  time_to_peak={:.4}s  peak_db={:.2}  slope={:.2} dB/s",
        att.time_to_peak_s, att.peak_db, att.post_peak_slope_db_s
    );

    // 6. Render. Hold for `hold` seconds (full-amplitude resonator
    //    decay); after note_off, multiplicative release env takes over
    //    for the remaining `duration - hold` seconds.
    let total_samples = (args.duration * sr) as usize;
    let hold_samples = (args.hold * sr) as usize;
    let mut voice = ModalPianoVoice::new_default_excitation(sr, &modes, args.velocity);
    let mut out = vec![0.0_f32; total_samples];

    let frames_per_buf = 1024;
    let mut idx = 0;
    while idx < hold_samples {
        let end = (idx + frames_per_buf).min(hold_samples);
        let mut buf = vec![0.0_f32; end - idx];
        voice.render_add(&mut buf);
        out[idx..end].copy_from_slice(&buf);
        idx = end;
    }
    voice.trigger_release();
    while idx < total_samples {
        let end = (idx + frames_per_buf).min(total_samples);
        let mut buf = vec![0.0_f32; end - idx];
        voice.render_add(&mut buf);
        out[idx..end].copy_from_slice(&buf);
        idx = end;
    }

    // 7. Normalise to -3 dBFS so the render is comparable to the SFZ ref
    //    in level. The exact level isn't a fitted parameter; analyse
    //    metrics that care (MR-STFT, LSD) are amplitude-sensitive but
    //    the SFZ render is itself peak-normalised in `bench`, so doing
    //    the same here keeps the comparison fair.
    let peak = out.iter().copied().fold(0.0_f32, |a, b| a.max(b.abs()));
    if peak > 1e-9 {
        let target = 10f32.powf(-3.0 / 20.0);
        let scale = target / peak;
        for s in out.iter_mut() {
            *s *= scale;
        }
    }

    if let Err(e) = write_wav_mono(&args.out, &out, sr) {
        eprintln!("render_modal: {e}");
        std::process::exit(2);
    }
    println!(
        "\nrender_modal: wrote {} ({:.2}s, peak {:.3})",
        args.out.display(),
        out.len() as f32 / sr,
        out.iter().copied().fold(0.0_f32, |a, b| a.max(b.abs()))
    );
}
