//! CPU-cost micro-benchmark for sustained polyphony per engine.
//!
//! Renders N voices of one engine for D seconds at sr=44.1 kHz, measures
//! wall-clock vs real-time, prints a CPU% (= wall / audio) table. This is the
//! "can we play 32 voices live" sanity check tracked in issue #2 (per-voice
//! CPU budget).
//!
//! Not a substitute for a real profiler — it just measures rendering cost
//! per engine in a tight loop with no audio I/O. Background syscalls,
//! Mutex locking, the audio callback's bus mixing, and the body-IR reverb
//! are NOT included; treat the numbers as a lower bound on per-engine cost.
//!
//! Usage:
//!     cargo run --release --bin cpubench
//!     cargo run --release --bin cpubench -- --voices 32 --duration 4
//!     cargo run --release --bin cpubench -- --engine piano --voices 32

use std::time::Instant;

use keysynth::synth::{make_voice, midi_to_freq, Engine};

const DEFAULT_VOICES: usize = 32;
const DEFAULT_DURATION_SEC: f32 = 4.0;
const SR: f32 = 44_100.0;
const FRAMES_PER_BUF: usize = 1024;

#[derive(Clone, Copy)]
struct EngineSpec {
    engine: Engine,
    name: &'static str,
}

const ALL_ENGINES: &[EngineSpec] = &[
    EngineSpec {
        engine: Engine::Square,
        name: "square",
    },
    EngineSpec {
        engine: Engine::Ks,
        name: "ks",
    },
    EngineSpec {
        engine: Engine::KsRich,
        name: "ks-rich",
    },
    EngineSpec {
        engine: Engine::Sub,
        name: "sub",
    },
    EngineSpec {
        engine: Engine::Fm,
        name: "fm",
    },
    EngineSpec {
        engine: Engine::Piano,
        name: "piano",
    },
    EngineSpec {
        engine: Engine::PianoThick,
        name: "piano-thick",
    },
    EngineSpec {
        engine: Engine::PianoLite,
        name: "piano-lite",
    },
    EngineSpec {
        engine: Engine::Koto,
        name: "koto",
    },
];

struct Args {
    voices: usize,
    duration_sec: f32,
    only: Option<String>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            voices: DEFAULT_VOICES,
            duration_sec: DEFAULT_DURATION_SEC,
            only: None,
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let mut out = Args::default();
    let mut iter = std::env::args().skip(1);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--voices" => {
                out.voices = iter
                    .next()
                    .ok_or("--voices needs an integer")?
                    .parse()
                    .map_err(|e| format!("bad --voices: {e}"))?;
            }
            "--duration" => {
                out.duration_sec = iter
                    .next()
                    .ok_or("--duration needs a float")?
                    .parse()
                    .map_err(|e| format!("bad --duration: {e}"))?;
            }
            "--engine" => {
                out.only = Some(iter.next().ok_or("--engine needs a name")?);
            }
            "--help" | "-h" => {
                eprintln!(
                    "cpubench --voices N --duration SEC [--engine NAME]\n\
                     defaults: voices=32 duration=4.0\n\
                     engines: square|ks|ks-rich|sub|fm|piano|piano-thick|piano-lite|koto"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    Ok(out)
}

/// Render `voice_count` simultaneous voices of `engine` for `duration_sec`
/// seconds. Returns wall-clock elapsed in seconds. Voices are spread across
/// MIDI 48..(48 + voice_count) so each gets a unique frequency (avoiding
/// any KS-string buffer aliasing that could collapse to one cache line).
fn render_polyphony(engine: Engine, voice_count: usize, duration_sec: f32) -> f64 {
    let total_frames = (duration_sec * SR) as usize;
    let mut voices: Vec<Box<dyn keysynth::synth::VoiceImpl + Send>> = (0..voice_count)
        .map(|i| {
            let note = 48 + (i as u8 % 36);
            make_voice(engine, SR, midi_to_freq(note), 100)
        })
        .collect();

    let mut buf = vec![0.0_f32; FRAMES_PER_BUF];
    let mut rendered = 0;
    let start = Instant::now();
    while rendered < total_frames {
        for s in buf.iter_mut() {
            *s = 0.0;
        }
        for v in voices.iter_mut() {
            v.render_add(buf.as_mut_slice());
        }
        rendered += FRAMES_PER_BUF;
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64()
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("cpubench: {e}");
            std::process::exit(2);
        }
    };

    println!(
        "cpubench: voices={} duration={:.1}s sr={:.0} frames_per_buf={}",
        args.voices, args.duration_sec, SR, FRAMES_PER_BUF
    );
    println!("          (CPU% = wall / audio. <100% = faster than real-time.)");
    println!();
    println!(
        "{:<13} {:>10} {:>10} {:>10}",
        "engine", "audio(s)", "wall(s)", "CPU%"
    );
    println!("{}", "-".repeat(48));

    for spec in ALL_ENGINES {
        if let Some(only) = &args.only {
            if spec.name != only {
                continue;
            }
        }
        let wall = render_polyphony(spec.engine, args.voices, args.duration_sec);
        let cpu_pct = (wall / args.duration_sec as f64) * 100.0;
        println!(
            "{:<13} {:>10.2} {:>10.3} {:>9.1}%",
            spec.name, args.duration_sec, wall, cpu_pct
        );
    }
}
