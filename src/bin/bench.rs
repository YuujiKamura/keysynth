//! Reference comparison harness: render the same MIDI note through both
//! `keysynth` (one of our engines) and a SoundFont (the "ground truth"
//! sample-based reference), then write both as WAVs side by side so you can
//! diff them in a DAW or spectrogram tool.
//!
//! Why: until you can A/B against a real instrument, parameter tuning is
//! blind. The closest free piano reference is the Salamander Grand or
//! GeneralUser GS SoundFont -- both render the same MIDI note as actual
//! recorded samples (or carefully synthesised approximations), so they're a
//! reasonable stand-in for "what a real grand piano sounds like".
//!
//! Usage:
//!   bench --sf2 path/to.sf2 [--note 60] [--engine piano] [--out bench-out]
//!         [--reverb 0..1] [--ir PATH]
//!
//! Output:
//!   <out>/keysynth_<engine>_n<note>.wav
//!   <out>/soundfont_n<note>.wav
//!
//! Both are mono 16-bit 44100 Hz WAV. Drag into Audacity / Reaper / iZotope
//! RX / Sonic Visualiser to compare waveform + spectrogram.
//!
//! Reverb: when --reverb > 0 the keysynth-side render gets the body-IR
//! convolution applied before normalisation, so you can A/B dry vs body
//! within the same engine. The SoundFont side is left untouched (its own
//! samples already include real-world room/cabinet character).
//!
//! sf-piano engine: passing --engine sf-piano makes the keysynth-side
//! render use the SAME shared rustysynth instance the runtime uses. This
//! lets you verify that sf-piano matches the reference soundfont render
//! exactly (modulo reverb).

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::Arc;

use hound::{SampleFormat, WavSpec, WavWriter};
use rustysynth::{SoundFont, Synthesizer, SynthesizerSettings};

use keysynth::reverb::{self, Reverb};
use keysynth::sfz::SfzPlayer;
use keysynth::synth::{make_voice, midi_to_freq, Engine};

const SR: u32 = 44100;

struct BenchArgs {
    sf2_path: PathBuf,
    note: u8,
    velocity: u8,
    duration_sec: f32,
    hold_sec: f32,
    engine: Engine,
    out_dir: PathBuf,
    sf2_program: u8,
    sf2_bank: u8,
    only: OnlyTarget,
    reverb_wet: f32,
    ir_path: Option<PathBuf>,
    sfz_path: Option<PathBuf>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OnlyTarget {
    Both,
    Keysynth,
    Soundfont,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("bench: error: {e}");
            print_help();
            std::process::exit(2);
        }
    };

    std::fs::create_dir_all(&args.out_dir)?;
    let total_samples = (args.duration_sec * SR as f32) as usize;
    let release_at = ((args.hold_sec * SR as f32) as usize).min(total_samples);

    eprintln!(
        "bench: note={} ({:.2} Hz) vel={} hold={:.2}s total={:.2}s sr={} \
         engine={:?} reverb={:.2}",
        args.note,
        midi_to_freq(args.note),
        args.velocity,
        args.hold_sec,
        args.duration_sec,
        SR,
        args.engine,
        args.reverb_wet,
    );

    if args.only != OnlyTarget::Soundfont {
        let path = render_keysynth(&args, total_samples, release_at)?;
        eprintln!("  keysynth -> {}", path.display());
    }
    if args.only != OnlyTarget::Keysynth {
        let path = render_soundfont(&args, total_samples, release_at)?;
        eprintln!("  soundfont -> {}", path.display());
    }

    Ok(())
}

fn render_keysynth(
    args: &BenchArgs,
    total_samples: usize,
    release_at: usize,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut buf = vec![0.0_f32; total_samples];

    if args.engine == Engine::SfzPiano {
        // sfz-piano renders via the SfzPlayer; mono-mix its stereo output
        // exactly like the live audio callback does.
        let sfz_path = args
            .sfz_path
            .as_ref()
            .ok_or("engine 'sfz-piano' requires --sfz PATH (no SFZ manifest provided)")?;
        let mut player = SfzPlayer::load(sfz_path, SR as f32)?;
        player.note_on(0, args.note, args.velocity);
        let mut left = vec![0.0_f32; total_samples];
        let mut right = vec![0.0_f32; total_samples];
        if release_at > 0 {
            player.render(&mut left[..release_at], &mut right[..release_at]);
        }
        player.note_off(0, args.note);
        if release_at < total_samples {
            player.render(&mut left[release_at..], &mut right[release_at..]);
        }
        for i in 0..total_samples {
            buf[i] = (left[i] + right[i]) * 0.5;
        }
    } else if args.engine == Engine::SfPiano {
        // sf-piano routes through rustysynth -- the placeholder voice
        // produces no audio on its own, so we drive the synth here and
        // mix its stereo output to mono, exactly like the live audio
        // callback does.
        if args.sf2_path.as_os_str().is_empty() {
            return Err("engine 'sf-piano' requires --sf2 PATH (no SoundFont provided)".into());
        }
        let mut file = BufReader::new(
            File::open(&args.sf2_path)
                .map_err(|e| format!("opening SoundFont {:?}: {e}", args.sf2_path))?,
        );
        let sf = Arc::new(SoundFont::new(&mut file)?);
        let settings = SynthesizerSettings::new(SR as i32);
        let mut synth = Synthesizer::new(&sf, &settings)?;
        if args.sf2_bank > 0 {
            synth.process_midi_message(0, 0xB0, 0, args.sf2_bank as i32);
        }
        synth.process_midi_message(0, 0xC0, args.sf2_program as i32, 0);
        synth.note_on(0, args.note as i32, args.velocity as i32);

        let mut left = vec![0.0_f32; total_samples];
        let mut right = vec![0.0_f32; total_samples];
        if release_at > 0 {
            synth.render(&mut left[..release_at], &mut right[..release_at]);
        }
        synth.note_off(0, args.note as i32);
        if release_at < total_samples {
            synth.render(&mut left[release_at..], &mut right[release_at..]);
        }
        for i in 0..total_samples {
            buf[i] = (left[i] + right[i]) * 0.5;
        }
    } else {
        let freq = midi_to_freq(args.note);
        let mut voice = make_voice(args.engine, SR as f32, freq, args.velocity);
        if release_at > 0 {
            voice.render_add(&mut buf[..release_at]);
        }
        voice.trigger_release();
        if release_at < total_samples {
            voice.render_add(&mut buf[release_at..]);
        }
    }

    // Apply body-IR convolution if requested. Ordering matches the live
    // audio callback: reverb -> normalise (live runs tanh after; bench
    // normalises so the WAV stays at sane peak).
    if args.reverb_wet > 0.0 {
        let ir = match &args.ir_path {
            Some(p) => reverb::load_ir_wav(p)?,
            None => reverb::synthetic_body_ir(SR),
        };
        let mut rev = Reverb::new(ir);
        rev.process(&mut buf, args.reverb_wet);
    }

    normalise(&mut buf);

    let path = args.out_dir.join(format!(
        "keysynth_{}_n{}.wav",
        engine_slug(args.engine),
        args.note
    ));
    write_wav_mono(&path, &buf)?;
    Ok(path)
}

fn render_soundfont(
    args: &BenchArgs,
    total_samples: usize,
    release_at: usize,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut file = BufReader::new(
        File::open(&args.sf2_path)
            .map_err(|e| format!("opening SoundFont {:?}: {e}", args.sf2_path))?,
    );
    let sf = Arc::new(SoundFont::new(&mut file)?);

    let settings = SynthesizerSettings::new(SR as i32);
    let mut synth = Synthesizer::new(&sf, &settings)?;

    // Bank + program select. Most piano-flavoured SoundFonts put the grand
    // piano on bank 0, program 0 (GM Acoustic Grand). For GS/XG drum banks
    // you'd flip --sf2-bank to 128.
    if args.sf2_bank > 0 {
        // Bank Select MSB (CC 0)
        synth.process_midi_message(0, 0xB0, 0, args.sf2_bank as i32);
    }
    // Program Change
    synth.process_midi_message(0, 0xC0, args.sf2_program as i32, 0);

    synth.note_on(0, args.note as i32, args.velocity as i32);

    let mut left = vec![0.0_f32; total_samples];
    let mut right = vec![0.0_f32; total_samples];

    if release_at > 0 {
        synth.render(&mut left[..release_at], &mut right[..release_at]);
    }
    synth.note_off(0, args.note as i32);
    if release_at < total_samples {
        synth.render(&mut left[release_at..], &mut right[release_at..]);
    }

    // Mix to mono so output shape matches keysynth's mono WAV exactly.
    let mut mono: Vec<f32> = left
        .iter()
        .zip(right.iter())
        .map(|(l, r)| (l + r) * 0.5)
        .collect();
    normalise(&mut mono);

    let path = args.out_dir.join(format!("soundfont_n{}.wav", args.note));
    write_wav_mono(&path, &mono)?;
    Ok(path)
}

fn normalise(buf: &mut [f32]) {
    let peak = buf.iter().fold(0.0_f32, |a, &x| a.max(x.abs()));
    if peak > 1e-6 {
        let scale = 0.9 / peak;
        for s in buf {
            *s *= scale;
        }
    }
}

fn write_wav_mono(path: &PathBuf, samples: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: SR,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let pcm = (clamped * i16::MAX as f32) as i16;
        writer.write_sample(pcm)?;
    }
    writer.finalize()?;
    Ok(())
}

fn engine_slug(e: Engine) -> &'static str {
    match e {
        Engine::Square => "square",
        Engine::Ks => "ks",
        Engine::KsRich => "ks-rich",
        Engine::Sub => "sub",
        Engine::Fm => "fm",
        Engine::Piano => "piano",
        Engine::Koto => "koto",
        Engine::SfPiano => "sf-piano",
        Engine::SfzPiano => "sfz-piano",
        Engine::PianoThick => "piano-thick",
        Engine::PianoLite => "piano-lite",
    }
}

fn parse_args() -> Result<BenchArgs, String> {
    let mut sf2: Option<PathBuf> = None;
    let mut note: u8 = 60;
    let mut velocity: u8 = 100;
    let mut duration: f32 = 3.0;
    let mut hold: f32 = 1.0;
    let mut engine = Engine::Piano;
    let mut out = PathBuf::from("bench-out");
    let mut program: u8 = 0;
    let mut bank: u8 = 0;
    let mut only = OnlyTarget::Both;
    let mut reverb_wet: f32 = 0.0;
    let mut ir_path: Option<PathBuf> = None;
    let mut sfz_path: Option<PathBuf> = None;

    let mut iter = std::env::args().skip(1);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--sf2" => {
                sf2 = Some(PathBuf::from(iter.next().ok_or("--sf2 needs a path")?));
            }
            "--note" => {
                note = iter
                    .next()
                    .ok_or("--note needs an integer")?
                    .parse()
                    .map_err(|e| format!("bad --note: {e}"))?;
            }
            "--velocity" => {
                velocity = iter
                    .next()
                    .ok_or("--velocity needs an integer")?
                    .parse()
                    .map_err(|e| format!("bad --velocity: {e}"))?;
            }
            "--duration" => {
                duration = iter
                    .next()
                    .ok_or("--duration needs a float")?
                    .parse()
                    .map_err(|e| format!("bad --duration: {e}"))?;
            }
            "--hold" => {
                hold = iter
                    .next()
                    .ok_or("--hold needs a float")?
                    .parse()
                    .map_err(|e| format!("bad --hold: {e}"))?;
            }
            "--engine" => {
                let v = iter.next().ok_or("--engine needs a value")?;
                engine = match v.as_str() {
                    "square" => Engine::Square,
                    "ks" => Engine::Ks,
                    "ks-rich" => Engine::KsRich,
                    "sub" => Engine::Sub,
                    "fm" => Engine::Fm,
                    "piano" => Engine::Piano,
                    "koto" => Engine::Koto,
                    "sf-piano" => Engine::SfPiano,
                    "sfz-piano" => Engine::SfzPiano,
                    "piano-thick" => Engine::PianoThick,
                    "piano-lite" => Engine::PianoLite,
                    other => return Err(format!("unknown engine: {other}")),
                };
            }
            "--out" => {
                out = PathBuf::from(iter.next().ok_or("--out needs a path")?);
            }
            "--sf2-program" => {
                program = iter
                    .next()
                    .ok_or("--sf2-program needs an integer")?
                    .parse()
                    .map_err(|e| format!("bad --sf2-program: {e}"))?;
            }
            "--sf2-bank" => {
                bank = iter
                    .next()
                    .ok_or("--sf2-bank needs an integer")?
                    .parse()
                    .map_err(|e| format!("bad --sf2-bank: {e}"))?;
            }
            "--only" => {
                let v = iter.next().ok_or("--only needs keysynth|soundfont|both")?;
                only = match v.as_str() {
                    "keysynth" => OnlyTarget::Keysynth,
                    "soundfont" => OnlyTarget::Soundfont,
                    "both" => OnlyTarget::Both,
                    other => {
                        return Err(format!(
                            "--only must be keysynth|soundfont|both, got {other}"
                        ))
                    }
                };
            }
            "--reverb" => {
                reverb_wet = iter
                    .next()
                    .ok_or("--reverb needs a float 0..1")?
                    .parse::<f32>()
                    .map_err(|e| format!("bad --reverb: {e}"))?
                    .clamp(0.0, 1.0);
            }
            "--ir" => {
                ir_path = Some(PathBuf::from(iter.next().ok_or("--ir needs a path")?));
            }
            "--sfz" => {
                sfz_path = Some(PathBuf::from(iter.next().ok_or("--sfz needs a path")?));
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }

    // sf-piano on the keysynth side also needs the SoundFont, so it counts
    // as a "needs --sf2" case even when --only keysynth. sfz-piano does NOT
    // need an .sf2 — it has its own --sfz, and --only keysynth + sfz-piano
    // is a perfectly fine config.
    let needs_sf2 =
        (only != OnlyTarget::Keysynth && engine != Engine::SfzPiano) || engine == Engine::SfPiano;
    let sf2_path = match (sf2, needs_sf2) {
        (Some(p), _) => p,
        (None, false) => PathBuf::new(),
        (None, true) => return Err(
            "--sf2 PATH is required (omit only when --only keysynth with --engine != sf-piano, \
             or when using --engine sfz-piano with --only keysynth)"
                .into(),
        ),
    };

    if hold > duration {
        return Err(format!(
            "--hold ({hold}) cannot exceed --duration ({duration})"
        ));
    }

    Ok(BenchArgs {
        sf2_path,
        note,
        velocity,
        duration_sec: duration,
        hold_sec: hold,
        engine,
        out_dir: out,
        sf2_program: program,
        sf2_bank: bank,
        only,
        reverb_wet,
        ir_path,
        sfz_path,
    })
}

fn print_help() {
    println!(
        "bench - render the same MIDI note via keysynth and a SoundFont, side by side\n\n\
         USAGE:\n  \
            bench --sf2 PATH [options]\n  \
            bench --only keysynth [options]   (skip SoundFont; not allowed with --engine sf-piano)\n\n\
         OPTIONS:\n  \
            --sf2 PATH         SoundFont (.sf2) file. Required unless --only keysynth\n  \
                               AND --engine != sf-piano.\n  \
            --note N           MIDI note 0..127 (default: 60 = C4)\n  \
            --velocity V       MIDI velocity 1..127 (default: 100)\n  \
            --duration SEC     Total render length (default: 3.0)\n  \
            --hold SEC         How long to hold before note_off (default: 1.0)\n  \
            --engine ENGINE    square|ks|ks-rich|sub|fm|piano|koto|sf-piano (default: piano)\n  \
            --out DIR          Output dir (default: bench-out/)\n  \
            --sf2-program N    GM program number 0..127 (default: 0 = Acoustic Grand)\n  \
            --sf2-bank N       Bank number 0..128 (default: 0)\n  \
            --only TARGET      keysynth|soundfont|both (default: both)\n  \
            --reverb FLOAT     Body-IR wet 0..1 applied to keysynth side (default: 0)\n  \
            --ir PATH          WAV impulse response (default: built-in synthetic body IR)\n  \
            -h, --help         Show this help"
    );
}
