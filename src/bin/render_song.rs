//! Render short musical pieces through any engine for chord-progression
//! / melody / bass-line context evaluation. Single chords (`render_chord`)
//! prove polyphonic stacking works, but real-world piano realism is
//! judged on multi-note phrases where ring-out overlaps next-chord
//! attack and bass-line sustain underpins melody.
//!
//! Usage:
//!     render_song --engine ENGINE --piece NAME [--out PATH]
//!         [--sfz PATH] [--modal-lut PATH] [--velocity V]
//!
//! Built-in pieces (--piece NAME):
//!   c_progression  : I–IV–V–I in C major, 8 s, rich chord voicings
//!   minor_cadence  : i–VI–III–VII / i in A minor, 8 s
//!   arpeggio       : Cmaj7 arpeggio across 2 octaves, 6 s
//!   twinkle        : "Twinkle Twinkle Little Star" first phrase, 8 s
//!                     (treble melody + sustained bass)
//!   bach_invention : 1-bar two-voice imitation à la Bach Invention 1, 6 s

use std::env;
use std::path::PathBuf;

use hound::{SampleFormat, WavSpec, WavWriter};

use keysynth::sfz::SfzPlayer;
use keysynth::synth::{make_voice, midi_to_freq, Engine, ModalLut, VoiceImpl, MODAL_LUT};

const SR: u32 = 44100;

#[derive(Clone, Copy, Debug)]
struct NoteEvent {
    start_sec: f32,
    midi_note: u8,
    duration_sec: f32,
    velocity: u8,
}

fn piece_c_progression() -> Vec<NoteEvent> {
    // I - IV - V - I in C major. Each chord 1.8 s with 0.2 s
    // overlap so previous-chord ring fades under the next.
    // Voicings use root + 3rd + 5th + octave for richer stacking.
    let mut v = Vec::new();
    let chords: &[(f32, &[u8])] = &[
        (0.0, &[36, 60, 64, 67, 72]), // C major (C2 + C4 E4 G4 + C5)
        (1.6, &[41, 60, 65, 69, 72]), // F major (F2 + C4 F4 A4 + C5)
        (3.2, &[43, 62, 67, 71, 74]), // G major (G2 + D4 G4 B4 + D5)
        (4.8, &[36, 60, 64, 67, 72]), // back to C
    ];
    for (t, notes) in chords {
        for &n in *notes {
            v.push(NoteEvent {
                start_sec: *t,
                midi_note: n,
                duration_sec: 1.8,
                velocity: 95,
            });
        }
    }
    v
}

fn piece_minor_cadence() -> Vec<NoteEvent> {
    // i – VI – III – VII / i in A minor. 8 s.
    let mut v = Vec::new();
    let chords: &[(f32, &[u8])] = &[
        (0.0, &[33, 57, 60, 64, 69]),  // A minor
        (1.6, &[41, 57, 60, 65, 69]),  // F major (VI of Am)
        (3.2, &[40, 55, 59, 64, 67]),  // C major (III of Am)
        (4.8, &[43, 59, 62, 67, 71]),  // G major (VII)
        (6.4, &[33, 57, 60, 64, 69]),  // back to Am
    ];
    for (t, notes) in chords {
        for &n in *notes {
            v.push(NoteEvent {
                start_sec: *t,
                midi_note: n,
                duration_sec: 1.6,
                velocity: 90,
            });
        }
    }
    v
}

fn piece_arpeggio() -> Vec<NoteEvent> {
    // Cmaj7 arpeggio across 2 octaves: C E G B C E G B C ... 6 s.
    let mut v = Vec::new();
    let pattern = [60, 64, 67, 71, 72, 76, 79, 83, 84];
    let step = 0.30;
    let dur = 0.8;
    for (i, &n) in pattern.iter().enumerate() {
        v.push(NoteEvent {
            start_sec: i as f32 * step,
            midi_note: n,
            duration_sec: dur,
            velocity: 100,
        });
    }
    // Held bass
    v.push(NoteEvent {
        start_sec: 0.0,
        midi_note: 36,
        duration_sec: 5.0,
        velocity: 80,
    });
    v
}

fn piece_twinkle() -> Vec<NoteEvent> {
    // "Twinkle Twinkle Little Star" first phrase. Treble melody + held
    // bass. Standard 4-bar version, ~8 s.
    let melody: &[(f32, u8, f32)] = &[
        (0.0, 60, 0.45), // C — twin-
        (0.5, 60, 0.45), // C — kle
        (1.0, 67, 0.45), // G — twin-
        (1.5, 67, 0.45), // G — kle
        (2.0, 69, 0.45), // A — lit-
        (2.5, 69, 0.45), // A — tle
        (3.0, 67, 0.95), // G — star
        (4.0, 65, 0.45), // F — how
        (4.5, 65, 0.45), // F — I
        (5.0, 64, 0.45), // E — won-
        (5.5, 64, 0.45), // E — der
        (6.0, 62, 0.45), // D — what
        (6.5, 62, 0.45), // D — you
        (7.0, 60, 0.95), // C — are
    ];
    let mut v: Vec<NoteEvent> = melody
        .iter()
        .map(|&(t, n, d)| NoteEvent {
            start_sec: t,
            midi_note: n,
            duration_sec: d,
            velocity: 100,
        })
        .collect();
    // Bass: C2 (mostly) with F2 and G2 changes following the V/IV harmonic
    // implication of the melody. Each chord ~2 s.
    let bass: &[(f32, u8, f32)] = &[
        (0.0, 36, 1.95), // C
        (2.0, 41, 1.95), // F
        (4.0, 41, 1.95), // F (tonal stay)
        (6.0, 43, 1.0),  // G
        (7.0, 36, 1.0),  // C
    ];
    v.extend(bass.iter().map(|&(t, n, d)| NoteEvent {
        start_sec: t,
        midi_note: n,
        duration_sec: d,
        velocity: 75,
    }));
    v
}

fn piece_bach_invention() -> Vec<NoteEvent> {
    // 1-bar two-voice imitative figure inspired by Bach Invention 1.
    // Soprano: C-D-E-F-D-E-C / Bass enters one beat later with same
    // shape down a fifth. 6 s.
    let s_pitches = [60, 62, 64, 65, 62, 64, 60, 67, 65, 64, 62, 60];
    let s_step = 0.25;
    let mut v: Vec<NoteEvent> = s_pitches
        .iter()
        .enumerate()
        .map(|(i, &n)| NoteEvent {
            start_sec: i as f32 * s_step,
            midi_note: n,
            duration_sec: s_step * 1.1,
            velocity: 95,
        })
        .collect();
    let b_pitches = [48, 50, 52, 53, 50, 52, 48, 55, 53, 52, 50, 48];
    let b_offset = 0.25;
    for (i, &n) in b_pitches.iter().enumerate() {
        v.push(NoteEvent {
            start_sec: b_offset + i as f32 * s_step,
            midi_note: n,
            duration_sec: s_step * 1.1,
            velocity: 85,
        });
    }
    v
}

fn pick_piece(name: &str) -> Result<Vec<NoteEvent>, String> {
    match name {
        "c_progression" => Ok(piece_c_progression()),
        "minor_cadence" => Ok(piece_minor_cadence()),
        "arpeggio" => Ok(piece_arpeggio()),
        "twinkle" => Ok(piece_twinkle()),
        "bach_invention" => Ok(piece_bach_invention()),
        other => Err(format!(
            "unknown piece: {other} \
             (c_progression|minor_cadence|arpeggio|twinkle|bach_invention)"
        )),
    }
}

struct Args {
    engine: Engine,
    piece: String,
    out_path: PathBuf,
    sfz_path: Option<PathBuf>,
    modal_lut_path: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut engine: Option<Engine> = None;
    let mut piece: Option<String> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut sfz_path: Option<PathBuf> = None;
    let mut modal_lut_path: Option<PathBuf> = None;

    let mut iter = env::args().skip(1);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--engine" => {
                let v = iter.next().ok_or("--engine needs a value")?;
                engine = Some(match v.as_str() {
                    "square" => Engine::Square,
                    "ks" => Engine::Ks,
                    "ks-rich" => Engine::KsRich,
                    "sub" => Engine::Sub,
                    "fm" => Engine::Fm,
                    "piano" => Engine::Piano,
                    "koto" => Engine::Koto,
                    "sfz-piano" => Engine::SfzPiano,
                    "piano-thick" => Engine::PianoThick,
                    "piano-lite" => Engine::PianoLite,
                    "piano-5am" => Engine::Piano5AM,
                    "piano-modal" => Engine::PianoModal,
                    other => return Err(format!("unknown engine: {other}")),
                });
            }
            "--piece" => piece = Some(iter.next().ok_or("--piece needs a name")?),
            "--out" => out_path = Some(PathBuf::from(iter.next().ok_or("--out needs a path")?)),
            "--sfz" => sfz_path = Some(PathBuf::from(iter.next().ok_or("--sfz needs a path")?)),
            "--modal-lut" => {
                modal_lut_path = Some(PathBuf::from(
                    iter.next().ok_or("--modal-lut needs a path")?,
                ))
            }
            "--help" | "-h" => {
                eprintln!(
                    "render_song — render a short musical piece through any engine.\n\n\
                     options:\n  \
                     --engine ENGINE   square|ks|ks-rich|sub|fm|piano|piano-thick|piano-lite|\
                                       piano-5am|piano-modal|sfz-piano|koto\n  \
                     --piece NAME      c_progression|minor_cadence|arpeggio|twinkle|bach_invention\n  \
                     --sfz PATH        SFZ manifest (required for sfz-piano)\n  \
                     --modal-lut PATH  modal LUT JSON (auto-discovered if omitted)\n  \
                     --out PATH        output WAV path (required)"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    Ok(Args {
        engine: engine.ok_or("--engine is required")?,
        piece: piece.ok_or("--piece is required")?,
        out_path: out_path.ok_or("--out is required")?,
        sfz_path,
        modal_lut_path,
    })
}

fn write_wav(path: &std::path::Path, samples: &[f32]) -> Result<(), String> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).map_err(|e| format!("create_dir_all {}: {e}", p.display()))?;
    }
    let spec = WavSpec {
        channels: 1,
        sample_rate: SR,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut w = WavWriter::create(path, spec).map_err(|e| format!("WavWriter::create: {e}"))?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let i = (clamped * i16::MAX as f32) as i16;
        w.write_sample(i).map_err(|e| format!("write_sample: {e}"))?;
    }
    w.finalize().map_err(|e| format!("finalize: {e}"))?;
    Ok(())
}

fn render_keysynth_piece(args: &Args, events: &[NoteEvent]) -> Result<Vec<f32>, String> {
    if args.engine == Engine::PianoModal {
        let (lut, source) = ModalLut::auto_load(args.modal_lut_path.as_deref());
        eprintln!("render_song: modal LUT source = {source}");
        let _ = MODAL_LUT.set(lut);
    }
    // Total length = max(start + duration) + 2 s release tail.
    let max_end = events
        .iter()
        .map(|e| e.start_sec + e.duration_sec)
        .fold(0.0_f32, f32::max);
    let total_sec = max_end + 2.0;
    let total_samples = (total_sec * SR as f32) as usize;
    let mut mono = vec![0.0_f32; total_samples];

    // Per-event voice. Each event occupies a window
    // [start_sample, end_sample]; render note_on..note_off then let
    // the release tail trail off through the remaining samples.
    for ev in events {
        let start_sample = (ev.start_sec * SR as f32) as usize;
        let release_sample = ((ev.start_sec + ev.duration_sec) * SR as f32) as usize;
        if start_sample >= total_samples {
            continue;
        }
        let release_at = release_sample.saturating_sub(start_sample);
        let voice_total = total_samples - start_sample;
        let mut voice_buf = vec![0.0_f32; voice_total];

        let freq = midi_to_freq(ev.midi_note);
        let mut voice = make_voice(args.engine, SR as f32, freq, ev.velocity);
        if release_at > 0 {
            voice.render_add(&mut voice_buf[..release_at.min(voice_total)]);
        }
        voice.trigger_release();
        if release_at < voice_total {
            voice.render_add(&mut voice_buf[release_at..voice_total]);
        }
        for (i, s) in voice_buf.iter().enumerate() {
            mono[start_sample + i] += *s;
        }
    }
    Ok(mono)
}

fn render_sfz_piece(args: &Args, events: &[NoteEvent]) -> Result<Vec<f32>, String> {
    let sfz_path = args
        .sfz_path
        .as_ref()
        .ok_or("--sfz PATH required for engine sfz-piano")?;
    let mut player = SfzPlayer::load(sfz_path, SR as f32)
        .map_err(|e| format!("SfzPlayer::load: {e}"))?;
    let max_end = events
        .iter()
        .map(|e| e.start_sec + e.duration_sec)
        .fold(0.0_f32, f32::max);
    let total_sec = max_end + 2.0;
    let total_samples = (total_sec * SR as f32) as usize;

    // Build a flat timeline of (sample_index, event_kind) ordered.
    #[derive(Clone, Copy)]
    enum Kind {
        On(u8, u8),
        Off(u8),
    }
    let mut timeline: Vec<(usize, Kind)> = Vec::new();
    for ev in events {
        let on_at = (ev.start_sec * SR as f32) as usize;
        let off_at = ((ev.start_sec + ev.duration_sec) * SR as f32) as usize;
        timeline.push((on_at.min(total_samples), Kind::On(ev.midi_note, ev.velocity)));
        timeline.push((off_at.min(total_samples), Kind::Off(ev.midi_note)));
    }
    timeline.sort_by_key(|(t, _)| *t);

    let mut left = vec![0.0_f32; total_samples];
    let mut right = vec![0.0_f32; total_samples];
    let mut cursor = 0usize;
    for (t, kind) in &timeline {
        if *t > cursor {
            let span = (*t).saturating_sub(cursor);
            if span > 0 {
                player.render(
                    &mut left[cursor..cursor + span],
                    &mut right[cursor..cursor + span],
                );
                cursor += span;
            }
        }
        match kind {
            Kind::On(n, v) => player.note_on(0, *n, *v),
            Kind::Off(n) => player.note_off(0, *n),
        }
    }
    if cursor < total_samples {
        player.render(&mut left[cursor..total_samples], &mut right[cursor..total_samples]);
    }
    let mut mono = vec![0.0_f32; total_samples];
    for i in 0..total_samples {
        mono[i] = (left[i] + right[i]) * 0.5;
    }
    Ok(mono)
}

fn peak_normalise(samples: &mut [f32], target_dbfs: f32) {
    let peak = samples.iter().copied().fold(0.0_f32, |a, b| a.max(b.abs()));
    if peak > 1e-9 {
        let target = 10f32.powf(target_dbfs / 20.0);
        let scale = target / peak;
        for s in samples.iter_mut() {
            *s *= scale;
        }
    }
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("render_song: {e}");
            std::process::exit(2);
        }
    };
    let events = match pick_piece(&args.piece) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("render_song: {e}");
            std::process::exit(2);
        }
    };

    eprintln!(
        "render_song: piece='{}' engine={:?} events={}",
        args.piece,
        args.engine,
        events.len()
    );

    let mut mono = match args.engine {
        Engine::SfzPiano => render_sfz_piece(&args, &events),
        _ => render_keysynth_piece(&args, &events),
    }
    .unwrap_or_else(|e| {
        eprintln!("render_song: {e}");
        std::process::exit(2);
    });

    let raw_peak = mono.iter().copied().fold(0.0_f32, |a, b| a.max(b.abs()));
    peak_normalise(&mut mono, -3.0);
    eprintln!("render_song: raw peak {:.3} → normalised -3 dBFS", raw_peak);

    if let Err(e) = write_wav(&args.out_path, &mono) {
        eprintln!("render_song: {e}");
        std::process::exit(2);
    }
    eprintln!("render_song: wrote {}", args.out_path.display());
}
