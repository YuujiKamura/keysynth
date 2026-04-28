use std::path::PathBuf;

use midly::num::{u15, u24, u28, u4, u7};
use midly::{Format, Header, MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind};

use keysynth::song::{parse_progression_with_key, Chord, Key, Voicing};
use keysynth::synth::{midi_to_freq, Engine, VoiceImpl};
use keysynth::voices::guitar::GuitarVoice;

#[allow(dead_code, deprecated)]
#[path = "render_midi.rs"]
mod render_midi_impl;

const PPQ: u16 = 480;
const BEATS_PER_BAR: u32 = 4;
const DEFAULT_BPM: u32 = 120;
const DEFAULT_VELOCITY: u8 = 90;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VoiceChoice {
    Engine(Engine),
    Guitar,
}

impl VoiceChoice {
    fn parse(input: &str) -> Result<Self, String> {
        match input {
            "square" => Ok(Self::Engine(Engine::Square)),
            "ks" => Ok(Self::Engine(Engine::Ks)),
            "ks-rich" => Ok(Self::Engine(Engine::KsRich)),
            "sub" => Ok(Self::Engine(Engine::Sub)),
            "fm" => Ok(Self::Engine(Engine::Fm)),
            "piano" => Ok(Self::Engine(Engine::Piano)),
            "piano-thick" => Ok(Self::Engine(Engine::PianoThick)),
            "piano-lite" => Ok(Self::Engine(Engine::PianoLite)),
            "piano-5am" => Ok(Self::Engine(Engine::Piano5AM)),
            "piano-modal" => Ok(Self::Engine(Engine::PianoModal)),
            "koto" => Ok(Self::Engine(Engine::Koto)),
            "sf-piano" => Ok(Self::Engine(Engine::SfPiano)),
            "sfz-piano" => Ok(Self::Engine(Engine::SfzPiano)),
            "guitar" => Ok(Self::Guitar),
            other => Err(format!(
                "unknown --voice: {other} (piano|piano-modal|piano-thick|piano-lite|piano-5am|guitar|square|ks|ks-rich|fm|sub|koto|sf-piano|sfz-piano)"
            )),
        }
    }
}

#[derive(Debug)]
struct PlayArgs {
    progression: String,
    voice: VoiceChoice,
    voicing: Voicing,
    bpm: u32,
    bars: Option<usize>,
    key: Option<Key>,
    out_path: PathBuf,
    midi_out_path: Option<PathBuf>,
    sfz_path: Option<PathBuf>,
    modal_lut_path: Option<PathBuf>,
}

fn print_help() {
    eprintln!(
        "kssong - one-line chord progression renderer\n\n\
         usage:\n  \
         kssong play \"C - G - Am - F\" --voice piano --bpm 120 --out out.wav\n\n\
         play options:\n  \
         --voice NAME      piano|piano-modal|piano-thick|piano-lite|piano-5am|guitar|square|ks|ks-rich|fm|sub|koto|sf-piano|sfz-piano\n  \
         --voicing MODE    close|piano|guitar|open (default close)\n  \
         --bpm N           tempo in beats per minute (default 120)\n  \
         --bars N          total bars to render; repeats/truncates progression to fit\n  \
         --key KEY         base key for roman numerals (example: C, F#, Bb)\n  \
         --sfz PATH        required when --voice sfz-piano\n  \
         --modal-lut PATH  optional modal LUT override for piano-modal\n  \
         --out PATH        output WAV path (required)\n  \
         --midi-out PATH   also dump the generated SMF to PATH (so the same\n  \
                           progression can be re-rendered through any\n  \
                           render_midi engine, including --engine guitar-stk)"
    );
}

fn parse_args() -> Result<PlayArgs, String> {
    let mut iter = std::env::args().skip(1);
    let Some(subcommand) = iter.next() else {
        print_help();
        return Err("missing subcommand".to_string());
    };

    if matches!(subcommand.as_str(), "--help" | "-h") {
        print_help();
        std::process::exit(0);
    }

    if subcommand != "play" {
        return Err(format!("unknown subcommand: {subcommand}"));
    }

    let progression = iter
        .next()
        .ok_or_else(|| "play requires a chord progression string".to_string())?;
    let mut voice = VoiceChoice::Engine(Engine::Piano);
    let mut voicing = Voicing::Close;
    let mut bpm = DEFAULT_BPM;
    let mut bars = None;
    let mut key = None;
    let mut out_path: Option<PathBuf> = None;
    let mut midi_out_path: Option<PathBuf> = None;
    let mut sfz_path = None;
    let mut modal_lut_path = None;

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--voice" | "--engine" => {
                let value = iter.next().ok_or("--voice needs a value")?;
                voice = VoiceChoice::parse(&value)?;
            }
            "--voicing" => {
                let value = iter.next().ok_or("--voicing needs a value")?;
                voicing = Voicing::parse(&value).map_err(|e| format!("bad --voicing: {e}"))?;
            }
            "--bpm" => {
                bpm = iter
                    .next()
                    .ok_or("--bpm needs a value")?
                    .parse()
                    .map_err(|e| format!("bad --bpm: {e}"))?;
                if bpm == 0 {
                    return Err("--bpm must be > 0".to_string());
                }
            }
            "--bars" => {
                let value: usize = iter
                    .next()
                    .ok_or("--bars needs a value")?
                    .parse()
                    .map_err(|e| format!("bad --bars: {e}"))?;
                if value == 0 {
                    return Err("--bars must be > 0".to_string());
                }
                bars = Some(value);
            }
            "--key" => {
                let value = iter.next().ok_or("--key needs a value")?;
                key = Some(Key::parse(&value).map_err(|e| format!("bad --key: {e}"))?);
            }
            "--out" => {
                out_path = Some(PathBuf::from(iter.next().ok_or("--out needs a value")?));
            }
            "--midi-out" => {
                midi_out_path = Some(PathBuf::from(
                    iter.next().ok_or("--midi-out needs a value")?,
                ));
            }
            "--sfz" => {
                sfz_path = Some(PathBuf::from(iter.next().ok_or("--sfz needs a value")?));
            }
            "--modal-lut" => {
                modal_lut_path = Some(PathBuf::from(
                    iter.next().ok_or("--modal-lut needs a value")?,
                ));
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }

    Ok(PlayArgs {
        progression,
        voice,
        voicing,
        bpm,
        bars,
        key,
        out_path: out_path.ok_or("--out is required")?,
        midi_out_path,
        sfz_path,
        modal_lut_path,
    })
}

fn expand_progression(chords: &[Chord], total_bars: usize) -> Vec<Chord> {
    (0..total_bars)
        .map(|bar| chords[bar % chords.len()].clone())
        .collect()
}

fn build_smf_bytes(chords: &[Chord], bpm: u32, voicing: Voicing) -> Result<Vec<u8>, String> {
    #[derive(Clone, Copy)]
    enum MidiKind {
        NoteOff(u8),
        NoteOn(u8),
    }

    let bar_ticks = BEATS_PER_BAR * PPQ as u32;
    let mut timeline: Vec<(u32, MidiKind)> = Vec::new();
    for (bar_idx, chord) in chords.iter().enumerate() {
        let start_tick = bar_idx as u32 * bar_ticks;
        let end_tick = start_tick + bar_ticks;
        for note in chord.voice(voicing) {
            timeline.push((end_tick, MidiKind::NoteOff(note)));
            timeline.push((start_tick, MidiKind::NoteOn(note)));
        }
    }
    timeline.sort_by_key(|(tick, kind)| {
        let order = match kind {
            MidiKind::NoteOff(_) => 0_u8,
            MidiKind::NoteOn(_) => 1_u8,
        };
        (*tick, order)
    });

    let tempo_us_per_quarter = 60_000_000_u32 / bpm;
    let header = Header {
        format: Format::Parallel,
        timing: Timing::Metrical(u15::new(PPQ)),
    };

    let tempo_track = vec![
        TrackEvent {
            delta: u28::new(0),
            kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::new(tempo_us_per_quarter))),
        },
        TrackEvent {
            delta: u28::new(0),
            kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
        },
    ];

    let mut note_track: Vec<TrackEvent<'static>> = Vec::with_capacity(timeline.len() + 1);
    let mut last_tick = 0_u32;
    for (tick, kind) in timeline {
        let delta = tick.saturating_sub(last_tick);
        last_tick = tick;
        let message = match kind {
            MidiKind::NoteOn(note) => MidiMessage::NoteOn {
                key: u7::new(note),
                vel: u7::new(DEFAULT_VELOCITY),
            },
            MidiKind::NoteOff(note) => MidiMessage::NoteOff {
                key: u7::new(note),
                vel: u7::new(0),
            },
        };
        note_track.push(TrackEvent {
            delta: u28::new(delta),
            kind: TrackEventKind::Midi {
                channel: u4::new(0),
                message,
            },
        });
    }
    note_track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });

    let smf = Smf {
        header,
        tracks: vec![tempo_track, note_track],
    };

    let mut bytes = Vec::new();
    smf.write_std(&mut bytes)
        .map_err(|e| format!("midly write: {e}"))?;
    Ok(bytes)
}

fn render_guitar(events: &[render_midi_impl::NoteEvent]) -> (Vec<f32>, Vec<f32>) {
    let max_end = events
        .iter()
        .map(|event| event.start_sec + event.duration_sec)
        .fold(0.0_f32, f32::max);
    let total_sec = max_end + render_midi_impl::RELEASE_TAIL_SEC;
    let total_samples = (total_sec * render_midi_impl::SR as f32) as usize;
    let mut left = vec![0.0_f32; total_samples];
    let mut right = vec![0.0_f32; total_samples];

    const RELEASE_BUDGET_SEC: f32 = 1.5;
    let release_budget_samples = (RELEASE_BUDGET_SEC * render_midi_impl::SR as f32) as usize;

    for event in events {
        let start_sample = (event.start_sec * render_midi_impl::SR as f32) as usize;
        let release_sample =
            ((event.start_sec + event.duration_sec) * render_midi_impl::SR as f32) as usize;
        if start_sample >= total_samples {
            continue;
        }
        let release_at = release_sample.saturating_sub(start_sample);
        let voice_total = (release_at + release_budget_samples).min(total_samples - start_sample);
        let mut voice_buf = vec![0.0_f32; voice_total];

        let freq = midi_to_freq(event.midi_note);
        let mut voice: Box<dyn VoiceImpl + Send> = Box::new(GuitarVoice::new(
            render_midi_impl::SR as f32,
            freq,
            event.velocity,
        ));
        if release_at > 0 {
            voice.render_add(&mut voice_buf[..release_at.min(voice_total)]);
        }
        voice.trigger_release();
        if release_at < voice_total {
            voice.render_add(&mut voice_buf[release_at..voice_total]);
        }

        let (left_gain, right_gain) = constant_power_pan(pan_for_note(event.midi_note));
        for (idx, sample) in voice_buf.iter().enumerate() {
            left[start_sample + idx] += *sample * left_gain;
            right[start_sample + idx] += *sample * right_gain;
        }
    }

    (left, right)
}

fn constant_power_pan(pan: f32) -> (f32, f32) {
    let p = pan.clamp(-1.0, 1.0);
    let theta = (p + 1.0) * std::f32::consts::FRAC_PI_4;
    (theta.cos(), theta.sin())
}

fn pan_for_note(midi_note: u8) -> f32 {
    const CENTRE: f32 = 60.0;
    const SPAN: f32 = 36.0;
    ((midi_note as f32 - CENTRE) / SPAN).clamp(-1.0, 1.0)
}

fn render_to_wav(args: &PlayArgs, midi_bytes: &[u8]) -> Result<(), String> {
    let events = render_midi_impl::parse_smf(midi_bytes, 1.0)?;
    let (left, right) = match args.voice {
        VoiceChoice::Engine(Engine::SfzPiano) => {
            let sfz_path = args
                .sfz_path
                .as_deref()
                .ok_or("--sfz PATH is required when --voice sfz-piano")?;
            render_midi_impl::render_sfz(sfz_path, &events)?
        }
        VoiceChoice::Engine(Engine::SfPiano) => {
            return Err(
                "sf-piano is live-only in keysynth; use sfz-piano or a synthesized engine"
                    .to_string(),
            );
        }
        VoiceChoice::Engine(engine) => {
            render_midi_impl::render_keysynth(engine, &events, args.modal_lut_path.as_deref())
        }
        VoiceChoice::Guitar => render_guitar(&events),
    };
    render_midi_impl::write_wav_stereo(&args.out_path, &left, &right)
}

fn run_play(args: PlayArgs) -> Result<(), String> {
    let chords = parse_progression_with_key(&args.progression, args.key)
        .map_err(|e| format!("progression parse: {e}"))?;
    let total_bars = args.bars.unwrap_or(chords.len());
    let expanded = expand_progression(&chords, total_bars);
    let midi_bytes = build_smf_bytes(&expanded, args.bpm, args.voicing)?;
    if let Some(path) = args.midi_out_path.as_ref() {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create_dir_all {}: {e}", parent.display()))?;
        }
        std::fs::write(path, &midi_bytes)
            .map_err(|e| format!("write midi-out {}: {e}", path.display()))?;
        eprintln!(
            "kssong: wrote MIDI {} ({} bytes)",
            path.display(),
            midi_bytes.len()
        );
    }
    render_to_wav(&args, &midi_bytes)?;
    eprintln!(
        "kssong: wrote {} (bars={} bpm={} voicing={})",
        args.out_path.display(),
        total_bars,
        args.bpm,
        args.voicing.as_str()
    );
    Ok(())
}

fn main() {
    #[cfg(target_arch = "x86_64")]
    #[allow(deprecated)]
    unsafe {
        use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
        let csr = _mm_getcsr();
        _mm_setcsr(csr | 0x8040);
    }

    let args = match parse_args() {
        Ok(args) => args,
        Err(err) => {
            eprintln!("kssong: {err}");
            std::process::exit(2);
        }
    };

    if let Err(err) = run_play(args) {
        eprintln!("kssong: {err}");
        std::process::exit(2);
    }
}
