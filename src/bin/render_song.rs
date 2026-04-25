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

fn piece_bach_prelude_c() -> Vec<NoteEvent> {
    // Bach Prelude in C, BWV 846 (Well-Tempered Clavier I). First 12
    // measures, sixteenth-note arpeggio pattern at ~75 BPM. Each
    // measure repeats an 8-note bass+chord-arpeggio figure twice
    // (16 sixteenths total per measure). Public domain (Bach d. 1750).
    //
    // Harmonic outline of the opening 12 bars:
    //   m1  C maj           m7  G7 / B
    //   m2  D min7 / C      m8  C maj
    //   m3  G7 / B          m9  A min / C
    //   m4  C maj           m10 D / A   (secondary V of V)
    //   m5  A min / C       m11 D7 / A
    //   m6  D7 / C          m12 G maj   (half cadence)
    let step = 0.20; // sixteenth at ~75 BPM
    let dur = 0.8;
    let mut v = Vec::new();
    let measures: &[(u8, u8, [u8; 4])] = &[
        // (bass1, bass2, [arpeggio_4_notes])
        (36, 52, [55, 60, 64, 67]), // m1: C2 E3 | G3 C4 E4 G4  Cmaj
        (36, 50, [57, 62, 65, 69]), // m2: C2 D3 | A3 D4 F4 A4  Dm7/C
        (35, 50, [55, 62, 65, 67]), // m3: B1 D3 | G3 D4 F4 G4  G7/B
        (36, 52, [55, 60, 64, 67]), // m4: C2 E3 | G3 C4 E4 G4  Cmaj
        (36, 52, [57, 60, 64, 69]), // m5: C2 E3 | A3 C4 E4 A4  Am/C
        (36, 50, [54, 57, 60, 62]), // m6: C2 D3 | F#3 A3 C4 D4 D7/C
        (35, 50, [55, 59, 62, 67]), // m7: B1 D3 | G3 B3 D4 G4  G7/B
        (36, 52, [55, 60, 64, 67]), // m8: C2 E3 | G3 C4 E4 G4  Cmaj
        (33, 48, [55, 60, 64, 69]), // m9: A1 C3 | G3 C4 E4 A4  Am/C variant
        (33, 49, [54, 57, 62, 69]), // m10: A1 C#3| F#3 A3 D4 A4 D/A
        (33, 48, [55, 57, 60, 65]), // m11: A1 C3 | G3 A3 C4 F4 D7/A
        (31, 50, [55, 59, 62, 67]), // m12: G1 D3 | G3 B3 D4 G4 G major
    ];
    for (mi, (b1, b2, arp)) in measures.iter().enumerate() {
        let m_start = mi as f32 * 16.0 * step;
        // Bass held for whole measure (slightly under so adjacent
        // measures don't blur into each other through the modal damper).
        v.push(NoteEvent {
            start_sec: m_start,
            midi_note: *b1,
            duration_sec: 16.0 * step * 0.95,
            velocity: 75,
        });
        v.push(NoteEvent {
            start_sec: m_start,
            midi_note: *b2,
            duration_sec: 16.0 * step * 0.95,
            velocity: 75,
        });
        // Two repeats of the 4-note arpeggio per measure.
        for rep in 0..2 {
            let r_start = m_start + rep as f32 * 4.0 * step;
            for (i, &n) in arp.iter().enumerate() {
                v.push(NoteEvent {
                    start_sec: r_start + (i as f32 + 1.0) * step,
                    midi_note: n,
                    duration_sec: dur,
                    velocity: 90,
                });
            }
        }
    }
    v
}

fn piece_fur_elise() -> Vec<NoteEvent> {
    // Beethoven, Bagatelle No. 25 in A minor "Für Elise", WoO 59.
    // Opening 4-bar theme. Public domain (Beethoven d. 1827).
    // 3/8 time, ~75 BPM → eighth = 0.27 s. Triplets = 0.135 s.
    let dt = 0.27_f32;
    let melody: &[(f32, u8, f32)] = &[
        (0.0 * dt, 76, dt),       // E5
        (1.0 * dt, 75, dt),       // D#5
        (2.0 * dt, 76, dt),       // E5
        (3.0 * dt, 75, dt),       // D#5
        (4.0 * dt, 76, dt),       // E5
        (5.0 * dt, 71, dt),       // B4
        (6.0 * dt, 74, dt),       // D5
        (7.0 * dt, 72, dt),       // C5
        (8.0 * dt, 69, 2.0 * dt), // A4 (longer)
        (10.5 * dt, 60, dt),      // C4
        (11.5 * dt, 64, dt),      // E4
        (12.5 * dt, 69, dt),      // A4
        (13.5 * dt, 71, 2.0 * dt),// B4
        (16.0 * dt, 64, dt),      // E4
        (17.0 * dt, 68, dt),      // G#4
        (18.0 * dt, 71, dt),      // B4
        (19.0 * dt, 72, 2.0 * dt),// C5
        (21.5 * dt, 64, dt),      // E4 lead-in
        (22.5 * dt, 76, dt),      // E5 (theme repeat anchor)
        (23.5 * dt, 75, dt),      // D#5
    ];
    let mut v: Vec<NoteEvent> = melody
        .iter()
        .map(|&(t, n, d)| NoteEvent {
            start_sec: t,
            midi_note: n,
            duration_sec: d * 0.92,
            velocity: 95,
        })
        .collect();
    // Sparse left-hand: A2 octave on first beat of each pair of measures.
    let bass: &[(f32, u8)] = &[
        (8.0 * dt, 33),  // A2 under "A4" landing
        (16.0 * dt, 28), // E2 under "E4 G#4 B4" run
        (22.0 * dt, 33), // A2 reprise
    ];
    v.extend(bass.iter().map(|&(t, n)| NoteEvent {
        start_sec: t,
        midi_note: n,
        duration_sec: 8.0 * dt * 0.95,
        velocity: 70,
    }));
    v
}

fn piece_gymnopedie() -> Vec<NoteEvent> {
    // Erik Satie, Gymnopédie No. 1 (1888). Opening 4 measures.
    // 3/4 time at ~70 BPM → quarter = 0.857 s. Public domain
    // (Satie d. 1925, work first published 1888 — well past
    // any rights term in any jurisdiction).
    let q = 0.857_f32;
    let mut v = Vec::new();
    // Left hand: bar 1 = D2 (beat 1), F#3+A3+C#4 (beat 2 chord),
    // Pattern alternates between G2 and D2 bass with chord on beat 2/3.
    // Simplified for the opening 4-bar phrase:
    let bass: &[(f32, u8)] = &[
        (0.0 * q, 38), // D2
        (3.0 * q, 43), // G2
        (6.0 * q, 38), // D2
        (9.0 * q, 43), // G2
    ];
    let chords: &[(f32, [u8; 3])] = &[
        (1.0 * q, [54, 57, 61]), // F#3 A3 C#4 (Dmaj7)
        (2.0 * q, [54, 57, 61]),
        (4.0 * q, [55, 59, 62]), // G3 B3 D4 (Gmaj triad)
        (5.0 * q, [55, 59, 62]),
        (7.0 * q, [54, 57, 61]),
        (8.0 * q, [54, 57, 61]),
        (10.0 * q, [55, 59, 62]),
        (11.0 * q, [55, 59, 62]),
    ];
    // Melody: slow, sparse, mostly stepwise.
    let melody: &[(f32, u8, f32)] = &[
        (1.0 * q, 78, 2.0 * q),  // F#5
        (3.0 * q, 79, q),        // G5
        (4.0 * q, 78, q),        // F#5
        (5.0 * q, 76, q),        // E5
        (6.0 * q, 73, 2.0 * q),  // C#5
        (8.0 * q, 75, q),        // D#5/Eb5
        (9.0 * q, 73, q),        // C#5
        (10.0 * q, 72, 2.0 * q), // C5 -> resolution
    ];
    for &(t, n) in bass {
        v.push(NoteEvent {
            start_sec: t,
            midi_note: n,
            duration_sec: q * 0.95,
            velocity: 70,
        });
    }
    for &(t, ns) in chords {
        for &n in &ns {
            v.push(NoteEvent {
                start_sec: t,
                midi_note: n,
                duration_sec: q * 0.95,
                velocity: 65,
            });
        }
    }
    for &(t, n, d) in melody {
        v.push(NoteEvent {
            start_sec: t,
            midi_note: n,
            duration_sec: d * 0.92,
            velocity: 90,
        });
    }
    v
}

fn piece_canon_d() -> Vec<NoteEvent> {
    // Pachelbel, Canon in D (~1680). The 8-bar ground-bass progression
    // repeated, with one canonic upper voice playing the most
    // recognisable counterpoint figure. Public domain (Pachelbel
    // d. 1706). 8 bars, ~32 s at quarter = 1.0 s.
    let q = 1.0_f32; // quarter note
    let mut v = Vec::new();
    // Bass line (whole notes, 4 q each), 2 cycles of 8 bars = 16 bars
    // total ≈ 64 s. Trim to 12 bars for the test (still long enough
    // to show the modal/SFZ contrast over a sustained progression).
    let bass_pattern = [
        38, // D2
        33, // A1
        35, // B1
        30, // F#1
        31, // G1
        38, // D2  (octave A1 below would be too low for most synths)
        31, // G1
        33, // A1
    ];
    // Triadic chord on each bar (3 notes, half-note pulses on beats 1 and 3).
    let chord_pattern: [[u8; 3]; 8] = [
        [50, 54, 57], // D maj    (D3 F#3 A3)
        [45, 49, 52], // A maj    (A2 C#3 E3)
        [47, 50, 54], // B min    (B2 D3 F#3)
        [42, 45, 49], // F# min   (F#2 A2 C#3)
        [43, 47, 50], // G maj    (G2 B2 D3)
        [50, 54, 57], // D maj
        [43, 47, 50], // G maj
        [45, 49, 52], // A maj
    ];
    // First-voice canon melody (the famous descending one), simplified.
    let melody_pattern: &[(u8, f32)] = &[
        (74, 2.0 * q), // D5  (held)
        (73, 2.0 * q), // C#5
        (74, 1.0 * q), // D5
        (76, 1.0 * q), // E5
        (78, 1.0 * q), // F#5
        (76, 1.0 * q), // E5
        (74, 1.0 * q), // D5
        (73, 1.0 * q), // C#5
        (71, 1.0 * q), // B4
        (74, 1.0 * q), // D5
        (73, 1.0 * q), // C#5
        (76, 1.0 * q), // E5
        (74, 1.0 * q), // D5
        (73, 1.0 * q), // C#5
        (71, 1.0 * q), // B4
        (69, 1.0 * q), // A4
        (66, 1.0 * q), // F#4
        (69, 1.0 * q), // A4
        (71, 1.0 * q), // B4
        (74, 1.0 * q), // D5
        (73, 1.0 * q), // C#5
        (71, 1.0 * q), // B4
        (74, 1.0 * q), // D5  (resolution)
        (76, 2.0 * q), // E5
    ];
    // Lay out 8 bars (4 beats each = 4q per bar = 32 q total).
    for bar in 0..bass_pattern.len() {
        let bar_start = bar as f32 * 4.0 * q;
        // bass note held for full bar
        v.push(NoteEvent {
            start_sec: bar_start,
            midi_note: bass_pattern[bar],
            duration_sec: 4.0 * q * 0.95,
            velocity: 75,
        });
        // chord on beats 1 and 3 (half notes)
        for beat in [0.0, 2.0] {
            for &n in &chord_pattern[bar] {
                v.push(NoteEvent {
                    start_sec: bar_start + beat * q,
                    midi_note: n,
                    duration_sec: 2.0 * q * 0.95,
                    velocity: 65,
                });
            }
        }
    }
    // Melody starts at bar 3 (after 2 bars of bass+chord intro), runs
    // through bar 8.
    let melody_start = 2.0 * 4.0 * q;
    let mut t = melody_start;
    for &(n, d) in melody_pattern {
        v.push(NoteEvent {
            start_sec: t,
            midi_note: n,
            duration_sec: d * 0.92,
            velocity: 92,
        });
        t += d;
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
        "bach_prelude_c" => Ok(piece_bach_prelude_c()),
        "fur_elise" => Ok(piece_fur_elise()),
        "gymnopedie" => Ok(piece_gymnopedie()),
        "canon_d" => Ok(piece_canon_d()),
        other => Err(format!(
            "unknown piece: {other} \
             (c_progression|minor_cadence|arpeggio|twinkle|bach_invention|\
              bach_prelude_c|fur_elise|gymnopedie|canon_d)"
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

fn write_wav_stereo(
    path: &std::path::Path,
    left: &[f32],
    right: &[f32],
) -> Result<(), String> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).map_err(|e| format!("create_dir_all {}: {e}", p.display()))?;
    }
    let spec = WavSpec {
        channels: 2,
        sample_rate: SR,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut w = WavWriter::create(path, spec).map_err(|e| format!("WavWriter::create: {e}"))?;
    let n = left.len().min(right.len());
    for i in 0..n {
        let l = (left[i].clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        let r = (right[i].clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        w.write_sample(l)
            .map_err(|e| format!("write_sample(L): {e}"))?;
        w.write_sample(r)
            .map_err(|e| format!("write_sample(R): {e}"))?;
    }
    w.finalize().map_err(|e| format!("finalize: {e}"))?;
    Ok(())
}

/// Constant-power pan curve. `pan` in [-1, +1]. Returns (left_gain, right_gain)
/// with `left_gain² + right_gain² == 1.0`. pan = -1 → fully left, +1 → fully right.
fn constant_power_pan(pan: f32) -> (f32, f32) {
    let p = pan.clamp(-1.0, 1.0);
    let theta = (p + 1.0) * std::f32::consts::FRAC_PI_4; // [0, π/2]
    (theta.cos(), theta.sin())
}

/// Per-MIDI-note pan, mimicking a grand piano keyboard's spatial layout
/// from the audience's perspective. MIDI 60 (C4) sits centre, the bass
/// pans left, the treble pans right. Linear in semitones across MIDI
/// 24..96 (≈ piano range), clamped beyond.
fn pan_for_note(midi_note: u8) -> f32 {
    const CENTRE: f32 = 60.0;
    const SPAN: f32 = 36.0;
    ((midi_note as f32 - CENTRE) / SPAN).clamp(-1.0, 1.0)
}

fn render_keysynth_piece(args: &Args, events: &[NoteEvent]) -> Result<(Vec<f32>, Vec<f32>), String> {
    if args.engine == Engine::PianoModal {
        let (lut, source) = ModalLut::auto_load(args.modal_lut_path.as_deref());
        eprintln!("render_song: modal LUT source = {source}");
        let _ = MODAL_LUT.set(lut);
        let res_dir = std::path::PathBuf::from("bench-out/RESIDUAL");
        if res_dir.is_dir() {
            if let Ok(rl) = keysynth::voices::piano_modal::ResidualLut::from_dir(&res_dir) {
                eprintln!(
                    "render_song: residual LUT source = {} ({} entries)",
                    rl.source,
                    rl.entries.len(),
                );
                let _ = keysynth::voices::piano_modal::RESIDUAL_LUT.set(rl);
            }
        }
    }
    // Total length = max(start + duration) + 2 s release tail.
    let max_end = events
        .iter()
        .map(|e| e.start_sec + e.duration_sec)
        .fold(0.0_f32, f32::max);
    let total_sec = max_end + 2.0;
    let total_samples = (total_sec * SR as f32) as usize;
    let mut left = vec![0.0_f32; total_samples];
    let mut right = vec![0.0_f32; total_samples];

    // Per-event voice. Each event renders mono (the voice itself
    // doesn't know about stereo); we pan into L/R based on the note's
    // position on the keyboard so the chord opens up spatially the way
    // a recorded grand piano does.
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
        let (lg, rg) = constant_power_pan(pan_for_note(ev.midi_note));
        for (i, s) in voice_buf.iter().enumerate() {
            left[start_sample + i] += *s * lg;
            right[start_sample + i] += *s * rg;
        }
    }
    Ok((left, right))
}

fn render_sfz_piece(args: &Args, events: &[NoteEvent]) -> Result<(Vec<f32>, Vec<f32>), String> {
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
    // Keep the SFZ player's native stereo image — that's what makes it
    // sound "spacious" vs the per-voice-pan modal stereo we synthesise.
    Ok((left, right))
}

fn peak_normalise_stereo(left: &mut [f32], right: &mut [f32], target_dbfs: f32) {
    let peak_l = left.iter().copied().fold(0.0_f32, |a, b| a.max(b.abs()));
    let peak_r = right.iter().copied().fold(0.0_f32, |a, b| a.max(b.abs()));
    let peak = peak_l.max(peak_r);
    if peak > 1e-9 {
        let target = 10f32.powf(target_dbfs / 20.0);
        let scale = target / peak;
        for s in left.iter_mut() {
            *s *= scale;
        }
        for s in right.iter_mut() {
            *s *= scale;
        }
    }
}

fn main() {
    // Flush-to-zero / denormals-as-zero on the offline render path.
    // Without it, the modal voice's high-Q biquad bank generates state
    // values around 1e-30 to 1e-38 during long sustains; x86 SSE
    // denormal arithmetic is 100-1000× slower per op, which made
    // longer pieces (bach_prelude_c at 120 events × 144 sub-modes)
    // appear to hang. main.rs::audio_callback already sets these on
    // the live audio thread; this mirrors that for offline renders.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
        let csr = _mm_getcsr();
        _mm_setcsr(csr | 0x8040);
    }

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

    let (mut left, mut right) = match args.engine {
        Engine::SfzPiano => render_sfz_piece(&args, &events),
        _ => render_keysynth_piece(&args, &events),
    }
    .unwrap_or_else(|e| {
        eprintln!("render_song: {e}");
        std::process::exit(2);
    });

    let raw_peak_l = left.iter().copied().fold(0.0_f32, |a, b| a.max(b.abs()));
    let raw_peak_r = right.iter().copied().fold(0.0_f32, |a, b| a.max(b.abs()));
    peak_normalise_stereo(&mut left, &mut right, -3.0);
    eprintln!(
        "render_song: raw peak L={:.3} R={:.3} → normalised -3 dBFS",
        raw_peak_l, raw_peak_r
    );

    if let Err(e) = write_wav_stereo(&args.out_path, &left, &right) {
        eprintln!("render_song: {e}");
        std::process::exit(2);
    }
    eprintln!("render_song: wrote {} (stereo)", args.out_path.display());
}
