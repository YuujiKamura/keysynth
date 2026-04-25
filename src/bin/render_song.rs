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
//!   assistant_uptempo / codex_punchout
//!                   : original uptempo stride-swing head, ~55 s

use std::env;
use std::path::PathBuf;

use hound::{SampleFormat, WavSpec, WavWriter};

use keysynth::sfz::SfzPlayer;
use keysynth::synth::{make_voice, midi_to_freq, Engine, ModalLut, MODAL_LUT};

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

fn piece_blues_in_c() -> Vec<NoteEvent> {
    // 12-bar blues in C at ~135 BPM (quarter = 0.444 s).
    // Walking bass + 3-note rootless chord stabs on beats 2 & 4.
    // Public-domain idiom (no specific copyrighted composition);
    // tests modal voice on rhythmic keyboard material.
    let q = 0.444_f32;
    let mut v = Vec::new();
    // Per-bar (chord, walking_bass_4_notes_quarter, stab_3_note_voicing).
    // Walking bass picks chord tones + a passing tone leading to next bar.
    let bars: &[([u8; 4], [u8; 3])] = &[
        // bar 1: C7  bass C-E-G-A  stab E-G-Bb (3 7-th rootless)
        ([36, 40, 43, 45], [52, 55, 58]),
        // bar 2: C7  bass C-E-G-Bb (chromatic walk to F)
        ([36, 40, 43, 46], [52, 55, 58]),
        // bar 3: C7
        ([36, 40, 43, 45], [52, 55, 58]),
        // bar 4: C7  -> walk down to F (lead-in)
        ([36, 40, 43, 41], [52, 55, 58]),
        // bar 5: F7  bass F-A-C-D
        ([41, 45, 48, 50], [57, 60, 63]), // A C Eb (b7 of F)
        // bar 6: F7
        ([41, 45, 48, 51], [57, 60, 63]),
        // bar 7: C7
        ([36, 40, 43, 45], [52, 55, 58]),
        // bar 8: C7  -> walk up to G
        ([36, 40, 43, 47], [52, 55, 58]),
        // bar 9: G7  bass G-B-D-F
        ([43, 47, 50, 53], [59, 62, 65]),
        // bar 10: F7
        ([41, 45, 48, 50], [57, 60, 63]),
        // bar 11: C7
        ([36, 40, 43, 45], [52, 55, 58]),
        // bar 12: G7  turnaround
        ([43, 47, 50, 47], [59, 62, 65]),
    ];
    for (bar_idx, (bass, stab)) in bars.iter().enumerate() {
        let bar_start = bar_idx as f32 * 4.0 * q;
        // 4 walking-bass quarter notes
        for (beat, &n) in bass.iter().enumerate() {
            v.push(NoteEvent {
                start_sec: bar_start + beat as f32 * q,
                midi_note: n,
                duration_sec: q * 0.85,
                velocity: 90,
            });
        }
        // chord stabs on beat 2 and 4 (slightly shorter, accented)
        for &beat in &[1.0_f32, 3.0] {
            for &n in stab {
                v.push(NoteEvent {
                    start_sec: bar_start + beat * q,
                    midi_note: n,
                    duration_sec: q * 0.55,
                    velocity: 80,
                });
            }
        }
    }
    // RH melody: bluesy 4-bar pickup riff in bars 3-4 and 11-12
    let melody: &[(f32, u8, f32)] = &[
        // bars 3-4 fill (top voice): C5 Eb5 E5 G5 (blue thirds slide)
        (8.0 * q, 72, 0.5 * q),
        (8.5 * q, 75, 0.5 * q),
        (9.0 * q, 76, 0.5 * q),
        (9.5 * q, 79, q),
        (10.5 * q, 76, 0.5 * q),
        (11.0 * q, 75, 0.5 * q),
        (11.5 * q, 72, 0.5 * q),
        // bars 11-12 lick: same vibe
        (40.0 * q, 76, 0.5 * q),
        (40.5 * q, 79, 0.5 * q),
        (41.0 * q, 82, q),
        (42.0 * q, 79, 0.5 * q),
        (42.5 * q, 76, 0.5 * q),
        (43.0 * q, 72, q),
    ];
    for &(t, n, d) in melody {
        v.push(NoteEvent {
            start_sec: t,
            midi_note: n,
            duration_sec: d * 0.92,
            velocity: 100,
        });
    }
    v
}

#[derive(Clone, Copy, Debug)]
enum Feel {
    Straight,
    LightSwing,
    Swing,
}

fn feel_ratio(feel: Feel) -> Option<f32> {
    match feel {
        Feel::Straight => None,
        Feel::LightSwing => Some(0.60),
        Feel::Swing => Some(2.0 / 3.0),
    }
}

fn felt_beat(beat: f32, feel: Feel) -> f32 {
    let Some(ratio) = feel_ratio(feel) else {
        return beat;
    };
    let whole = beat.floor();
    let frac = beat - whole;
    if (frac - 0.5).abs() < 0.001 {
        whole + ratio
    } else {
        beat
    }
}

fn felt_duration(start_beat: f32, duration_beat: f32, feel: Feel) -> f32 {
    let start = felt_beat(start_beat, feel);
    let end = felt_beat(start_beat + duration_beat, feel);
    (end - start).max(0.08)
}

fn push_line(dst: &mut Vec<(f32, u8, f32)>, offset: f32, src: &[(f32, u8, f32)]) {
    dst.extend(src.iter().map(|&(b, n, d)| (b + offset, n, d)));
}

fn push_chords(
    dst: &mut Vec<(f32, &'static [u8], f32)>,
    offset: f32,
    src: &[(f32, &'static [u8], f32)],
) {
    dst.extend(src.iter().map(|&(b, ns, d)| (b + offset, ns, d)));
}

fn lay_layers_with_feel(
    q: f32,
    feel: Feel,
    melody: &[(f32, u8, f32)],
    bass: &[(f32, u8, f32)],
    chords: &[(f32, &[u8], f32)],
    melody_vel: u8,
    bass_vel: u8,
    chord_vel: u8,
) -> Vec<NoteEvent> {
    let mut v: Vec<NoteEvent> = Vec::new();
    for &(b, n, d) in melody {
        v.push(NoteEvent {
            start_sec: felt_beat(b, feel) * q,
            midi_note: n,
            duration_sec: felt_duration(b, d, feel) * q * 0.92,
            velocity: melody_vel,
        });
    }
    for &(b, n, d) in bass {
        v.push(NoteEvent {
            start_sec: felt_beat(b, feel) * q,
            midi_note: n,
            duration_sec: felt_duration(b, d, feel) * q * 0.95,
            velocity: bass_vel,
        });
    }
    for &(b, ns, d) in chords {
        for &n in ns {
            v.push(NoteEvent {
                start_sec: felt_beat(b, feel) * q,
                midi_note: n,
                duration_sec: felt_duration(b, d, feel) * q * 0.85,
                velocity: chord_vel,
            });
        }
    }
    v
}

fn piece_maple_leaf_rag() -> Vec<NoteEvent> {
    // Scott Joplin, Maple Leaf Rag (1899).
    // Hand-arranged AABB piano reduction: identity and ragtime syncopation
    // are preserved, but this is not a literal engraving export.
    let q = 60.0 / 100.0;
    const AB: &[u8] = &[60, 63, 68];
    const EB7: &[u8] = &[58, 62, 65, 68];
    const DB: &[u8] = &[61, 65, 68];
    const F7: &[u8] = &[60, 63, 66, 69];

    let a_melody: &[(f32, u8, f32)] = &[
        (0.0, 68, 0.5), (0.5, 75, 0.25), (0.75, 73, 0.25), (1.0, 72, 0.5), (1.5, 75, 0.5),
        (2.0, 80, 0.5), (2.5, 75, 0.5), (3.0, 72, 1.0),
        (4.0, 68, 0.5), (4.5, 75, 0.25), (4.75, 73, 0.25), (5.0, 72, 0.5), (5.5, 75, 0.5),
        (6.0, 80, 1.0), (7.0, 75, 1.0),
        (8.0, 72, 0.5), (8.5, 75, 0.25), (8.75, 77, 0.25), (9.0, 80, 0.5), (9.5, 77, 0.5),
        (10.0, 75, 0.5), (10.5, 72, 0.5), (11.0, 68, 1.0),
        (12.0, 70, 0.5), (12.5, 72, 0.5), (13.0, 75, 0.5), (13.5, 77, 0.5),
        (14.0, 80, 0.5), (14.5, 77, 0.5), (15.0, 75, 1.0),
    ];
    let a_bass: &[(f32, u8, f32)] = &[
        (0.0, 44, 0.5), (1.0, 51, 0.5), (2.0, 44, 0.5), (3.0, 51, 0.5),
        (4.0, 44, 0.5), (5.0, 51, 0.5), (6.0, 46, 0.5), (7.0, 53, 0.5),
        (8.0, 49, 0.5), (9.0, 56, 0.5), (10.0, 44, 0.5), (11.0, 51, 0.5),
        (12.0, 46, 0.5), (13.0, 53, 0.5), (14.0, 44, 0.5), (15.0, 51, 0.5),
    ];
    let a_chords: &[(f32, &'static [u8], f32)] = &[
        (0.5, AB, 0.5), (1.5, AB, 0.5), (2.5, AB, 0.5), (3.5, AB, 0.5),
        (4.5, AB, 0.5), (5.5, AB, 0.5), (6.5, EB7, 0.5), (7.5, EB7, 0.5),
        (8.5, DB, 0.5), (9.5, DB, 0.5), (10.5, AB, 0.5), (11.5, AB, 0.5),
        (12.5, EB7, 0.5), (13.5, EB7, 0.5), (14.5, AB, 0.5), (15.5, AB, 0.5),
    ];

    let b_melody: &[(f32, u8, f32)] = &[
        (0.0, 80, 0.5), (0.5, 84, 0.25), (0.75, 82, 0.25), (1.0, 80, 0.5), (1.5, 77, 0.5),
        (2.0, 75, 0.5), (2.5, 77, 0.5), (3.0, 80, 1.0),
        (4.0, 82, 0.5), (4.5, 84, 0.5), (5.0, 87, 0.5), (5.5, 84, 0.5),
        (6.0, 82, 0.5), (6.5, 80, 0.5), (7.0, 77, 1.0),
        (8.0, 75, 0.5), (8.5, 77, 0.5), (9.0, 80, 0.5), (9.5, 82, 0.5),
        (10.0, 84, 0.5), (10.5, 80, 0.5), (11.0, 77, 1.0),
        (12.0, 72, 0.5), (12.5, 75, 0.5), (13.0, 77, 0.5), (13.5, 80, 0.5),
        (14.0, 77, 0.5), (14.5, 75, 0.5), (15.0, 72, 1.0),
    ];
    let b_bass: &[(f32, u8, f32)] = &[
        (0.0, 49, 0.5), (1.0, 56, 0.5), (2.0, 49, 0.5), (3.0, 56, 0.5),
        (4.0, 51, 0.5), (5.0, 58, 0.5), (6.0, 49, 0.5), (7.0, 56, 0.5),
        (8.0, 44, 0.5), (9.0, 51, 0.5), (10.0, 46, 0.5), (11.0, 53, 0.5),
        (12.0, 41, 0.5), (13.0, 48, 0.5), (14.0, 44, 0.5), (15.0, 51, 0.5),
    ];
    let b_chords: &[(f32, &'static [u8], f32)] = &[
        (0.5, DB, 0.5), (1.5, DB, 0.5), (2.5, DB, 0.5), (3.5, DB, 0.5),
        (4.5, EB7, 0.5), (5.5, EB7, 0.5), (6.5, DB, 0.5), (7.5, DB, 0.5),
        (8.5, AB, 0.5), (9.5, AB, 0.5), (10.5, EB7, 0.5), (11.5, EB7, 0.5),
        (12.5, F7, 0.5), (13.5, F7, 0.5), (14.5, AB, 0.5), (15.5, AB, 0.5),
    ];

    let mut melody = Vec::new();
    let mut bass = Vec::new();
    let mut chords = Vec::new();
    push_line(&mut melody, 0.0, a_melody);
    push_line(&mut melody, 32.0, a_melody);
    push_line(&mut melody, 64.0, b_melody);
    push_line(&mut melody, 96.0, b_melody);
    push_line(&mut bass, 0.0, a_bass);
    push_line(&mut bass, 32.0, a_bass);
    push_line(&mut bass, 64.0, b_bass);
    push_line(&mut bass, 96.0, b_bass);
    push_chords(&mut chords, 0.0, a_chords);
    push_chords(&mut chords, 32.0, a_chords);
    push_chords(&mut chords, 64.0, b_chords);
    push_chords(&mut chords, 96.0, b_chords);
    lay_layers_with_feel(q, Feel::Straight, &melody, &bass, &chords, 100, 84, 74)
}

fn piece_entertainer() -> Vec<NoteEvent> {
    // Scott Joplin, The Entertainer (1902). Hand-arranged AABB cut.
    let q = 60.0 / 88.0;
    const C: &[u8] = &[60, 64, 67];
    const G7: &[u8] = &[59, 62, 65, 67];
    const D7: &[u8] = &[62, 66, 69];
    const F: &[u8] = &[57, 60, 65];
    const A7: &[u8] = &[57, 61, 64, 67];

    let a_melody: &[(f32, u8, f32)] = &[
        (0.0, 76, 0.25), (0.25, 72, 0.25), (0.5, 76, 0.25), (0.75, 79, 0.25), (1.0, 84, 1.0),
        (2.0, 72, 0.25), (2.25, 76, 0.25), (2.5, 79, 0.25), (2.75, 84, 0.25), (3.0, 83, 0.5), (3.5, 79, 0.5),
        (4.0, 77, 0.25), (4.25, 76, 0.25), (4.5, 74, 0.5), (5.0, 72, 1.0),
        (6.0, 71, 0.5), (6.5, 72, 0.5), (7.0, 74, 1.0),
        (8.0, 76, 0.25), (8.25, 79, 0.25), (8.5, 83, 0.25), (8.75, 84, 0.25), (9.0, 86, 0.5), (9.5, 84, 0.5),
        (10.0, 83, 0.5), (10.5, 79, 0.5), (11.0, 76, 1.0),
        (12.0, 74, 0.5), (12.5, 76, 0.5), (13.0, 77, 0.5), (13.5, 79, 0.5),
        (14.0, 76, 0.5), (14.5, 74, 0.5), (15.0, 72, 1.0),
    ];
    let a_bass: &[(f32, u8, f32)] = &[
        (0.0, 36, 0.5), (1.0, 43, 0.5), (2.0, 36, 0.5), (3.0, 43, 0.5),
        (4.0, 36, 0.5), (5.0, 43, 0.5), (6.0, 38, 0.5), (7.0, 43, 0.5),
        (8.0, 36, 0.5), (9.0, 43, 0.5), (10.0, 41, 0.5), (11.0, 48, 0.5),
        (12.0, 38, 0.5), (13.0, 45, 0.5), (14.0, 36, 0.5), (15.0, 43, 0.5),
    ];
    let a_chords: &[(f32, &'static [u8], f32)] = &[
        (0.5, C, 0.5), (1.5, C, 0.5), (2.5, C, 0.5), (3.5, C, 0.5),
        (4.5, C, 0.5), (5.5, G7, 0.5), (6.5, D7, 0.5), (7.5, G7, 0.5),
        (8.5, C, 0.5), (9.5, C, 0.5), (10.5, F, 0.5), (11.5, F, 0.5),
        (12.5, D7, 0.5), (13.5, G7, 0.5), (14.5, C, 0.5), (15.5, C, 0.5),
    ];

    let b_melody: &[(f32, u8, f32)] = &[
        (0.0, 84, 0.5), (0.5, 83, 0.5), (1.0, 81, 0.5), (1.5, 79, 0.5),
        (2.0, 77, 0.5), (2.5, 76, 0.5), (3.0, 74, 1.0),
        (4.0, 72, 0.25), (4.25, 76, 0.25), (4.5, 79, 0.25), (4.75, 81, 0.25), (5.0, 84, 0.5), (5.5, 81, 0.5),
        (6.0, 79, 0.5), (6.5, 76, 0.5), (7.0, 72, 1.0),
        (8.0, 74, 0.5), (8.5, 76, 0.5), (9.0, 77, 0.5), (9.5, 79, 0.5),
        (10.0, 81, 0.5), (10.5, 79, 0.5), (11.0, 77, 1.0),
        (12.0, 76, 0.5), (12.5, 74, 0.5), (13.0, 72, 0.5), (13.5, 71, 0.5),
        (14.0, 72, 0.5), (14.5, 74, 0.5), (15.0, 76, 1.0),
    ];
    let b_bass: &[(f32, u8, f32)] = &[
        (0.0, 45, 0.5), (1.0, 52, 0.5), (2.0, 45, 0.5), (3.0, 52, 0.5),
        (4.0, 38, 0.5), (5.0, 45, 0.5), (6.0, 43, 0.5), (7.0, 50, 0.5),
        (8.0, 41, 0.5), (9.0, 48, 0.5), (10.0, 45, 0.5), (11.0, 52, 0.5),
        (12.0, 38, 0.5), (13.0, 45, 0.5), (14.0, 36, 0.5), (15.0, 43, 0.5),
    ];
    let b_chords: &[(f32, &'static [u8], f32)] = &[
        (0.5, A7, 0.5), (1.5, A7, 0.5), (2.5, A7, 0.5), (3.5, A7, 0.5),
        (4.5, D7, 0.5), (5.5, D7, 0.5), (6.5, G7, 0.5), (7.5, G7, 0.5),
        (8.5, F, 0.5), (9.5, F, 0.5), (10.5, A7, 0.5), (11.5, A7, 0.5),
        (12.5, D7, 0.5), (13.5, G7, 0.5), (14.5, C, 0.5), (15.5, C, 0.5),
    ];

    let mut melody = Vec::new();
    let mut bass = Vec::new();
    let mut chords = Vec::new();
    push_line(&mut melody, 0.0, a_melody);
    push_line(&mut melody, 32.0, a_melody);
    push_line(&mut melody, 64.0, b_melody);
    push_line(&mut melody, 96.0, b_melody);
    push_line(&mut bass, 0.0, a_bass);
    push_line(&mut bass, 32.0, a_bass);
    push_line(&mut bass, 64.0, b_bass);
    push_line(&mut bass, 96.0, b_bass);
    push_chords(&mut chords, 0.0, a_chords);
    push_chords(&mut chords, 32.0, a_chords);
    push_chords(&mut chords, 64.0, b_chords);
    push_chords(&mut chords, 96.0, b_chords);
    lay_layers_with_feel(q, Feel::Straight, &melody, &bass, &chords, 100, 84, 74)
}

fn piece_twelfth_street_rag() -> Vec<NoteEvent> {
    // Euday Bowman, Twelfth Street Rag (1914).
    // AABB reduction built around the signature descending cell.
    let q = 60.0 / 120.0;
    const F: &[u8] = &[57, 60, 65];
    const C7: &[u8] = &[55, 60, 64, 67];
    const BB: &[u8] = &[58, 62, 65];
    const G7: &[u8] = &[55, 59, 62, 65];

    let a_melody: &[(f32, u8, f32)] = &[
        (0.0, 69, 0.333), (0.333, 67, 0.333), (0.666, 65, 0.334),
        (1.0, 69, 0.333), (1.333, 67, 0.333), (1.666, 65, 0.334),
        (2.0, 69, 0.333), (2.333, 67, 0.333), (2.666, 65, 0.334), (3.0, 65, 1.0),
        (4.0, 70, 0.333), (4.333, 69, 0.333), (4.666, 67, 0.334),
        (5.0, 70, 0.333), (5.333, 69, 0.333), (5.666, 67, 0.334), (6.0, 70, 0.5), (6.5, 69, 0.5), (7.0, 67, 1.0),
        (8.0, 72, 0.333), (8.333, 70, 0.333), (8.666, 69, 0.334),
        (9.0, 67, 0.333), (9.333, 65, 0.333), (9.666, 64, 0.334), (10.0, 65, 1.0),
        (12.0, 69, 0.5), (12.5, 70, 0.5), (13.0, 72, 0.5), (13.5, 74, 0.5),
        (14.0, 72, 0.5), (14.5, 69, 0.5), (15.0, 65, 1.0),
    ];
    let a_bass: &[(f32, u8, f32)] = &[
        (0.0, 41, 0.5), (1.0, 48, 0.5), (2.0, 41, 0.5), (3.0, 48, 0.5),
        (4.0, 41, 0.5), (5.0, 48, 0.5), (6.0, 41, 0.5), (7.0, 48, 0.5),
        (8.0, 46, 0.5), (9.0, 53, 0.5), (10.0, 43, 0.5), (11.0, 50, 0.5),
        (12.0, 41, 0.5), (13.0, 48, 0.5), (14.0, 43, 0.5), (15.0, 50, 0.5),
    ];
    let a_chords: &[(f32, &'static [u8], f32)] = &[
        (0.5, F, 0.5), (1.5, C7, 0.5), (2.5, F, 0.5), (3.5, C7, 0.5),
        (4.5, F, 0.5), (5.5, C7, 0.5), (6.5, F, 0.5), (7.5, C7, 0.5),
        (8.5, BB, 0.5), (9.5, BB, 0.5), (10.5, G7, 0.5), (11.5, G7, 0.5),
        (12.5, F, 0.5), (13.5, C7, 0.5), (14.5, F, 0.5), (15.5, C7, 0.5),
    ];

    let b_melody: &[(f32, u8, f32)] = &[
        (0.0, 77, 0.5), (0.5, 74, 0.5), (1.0, 72, 0.5), (1.5, 69, 0.5),
        (2.0, 67, 0.5), (2.5, 69, 0.5), (3.0, 72, 1.0),
        (4.0, 74, 0.333), (4.333, 72, 0.333), (4.666, 70, 0.334),
        (5.0, 69, 0.333), (5.333, 67, 0.333), (5.666, 65, 0.334), (6.0, 67, 0.5), (6.5, 69, 0.5), (7.0, 70, 1.0),
        (8.0, 72, 0.5), (8.5, 74, 0.5), (9.0, 77, 0.5), (9.5, 74, 0.5),
        (10.0, 72, 0.5), (10.5, 69, 0.5), (11.0, 67, 1.0),
        (12.0, 65, 0.5), (12.5, 67, 0.5), (13.0, 69, 0.5), (13.5, 70, 0.5),
        (14.0, 69, 0.5), (14.5, 67, 0.5), (15.0, 65, 1.0),
    ];
    let b_bass: &[(f32, u8, f32)] = &[
        (0.0, 46, 0.5), (1.0, 53, 0.5), (2.0, 46, 0.5), (3.0, 53, 0.5),
        (4.0, 43, 0.5), (5.0, 50, 0.5), (6.0, 43, 0.5), (7.0, 50, 0.5),
        (8.0, 41, 0.5), (9.0, 48, 0.5), (10.0, 46, 0.5), (11.0, 53, 0.5),
        (12.0, 43, 0.5), (13.0, 50, 0.5), (14.0, 41, 0.5), (15.0, 48, 0.5),
    ];
    let b_chords: &[(f32, &'static [u8], f32)] = &[
        (0.5, BB, 0.5), (1.5, BB, 0.5), (2.5, BB, 0.5), (3.5, BB, 0.5),
        (4.5, G7, 0.5), (5.5, G7, 0.5), (6.5, G7, 0.5), (7.5, G7, 0.5),
        (8.5, F, 0.5), (9.5, F, 0.5), (10.5, BB, 0.5), (11.5, BB, 0.5),
        (12.5, G7, 0.5), (13.5, G7, 0.5), (14.5, F, 0.5), (15.5, C7, 0.5),
    ];

    let mut melody = Vec::new();
    let mut bass = Vec::new();
    let mut chords = Vec::new();
    push_line(&mut melody, 0.0, a_melody);
    push_line(&mut melody, 32.0, a_melody);
    push_line(&mut melody, 64.0, b_melody);
    push_line(&mut melody, 96.0, b_melody);
    push_line(&mut bass, 0.0, a_bass);
    push_line(&mut bass, 32.0, a_bass);
    push_line(&mut bass, 64.0, b_bass);
    push_line(&mut bass, 96.0, b_bass);
    push_chords(&mut chords, 0.0, a_chords);
    push_chords(&mut chords, 32.0, a_chords);
    push_chords(&mut chords, 64.0, b_chords);
    push_chords(&mut chords, 96.0, b_chords);
    lay_layers_with_feel(q, Feel::Straight, &melody, &bass, &chords, 100, 84, 74)
}

fn piece_st_louis_blues() -> Vec<NoteEvent> {
    // W.C. Handy, St. Louis Blues (1914).
    // Arranged as verse-like A plus more openly swung B.
    let q = 60.0 / 92.0;
    const G7: &[u8] = &[62, 65, 67, 71];
    const C7: &[u8] = &[60, 64, 67, 70];
    const D7: &[u8] = &[62, 66, 69, 72];
    const F9: &[u8] = &[65, 69, 72, 75];

    let a_melody: &[(f32, u8, f32)] = &[
        (0.0, 67, 1.0), (1.5, 70, 0.5), (2.0, 71, 1.0), (3.0, 67, 1.0),
        (4.0, 70, 0.5), (4.5, 71, 0.5), (5.0, 74, 1.0), (6.0, 71, 1.0),
        (8.0, 65, 1.0), (9.0, 67, 1.0), (10.0, 70, 1.0), (11.0, 67, 1.0),
        (12.0, 62, 1.0), (13.0, 65, 1.0), (14.0, 67, 1.0), (15.0, 70, 1.0),
        (16.0, 67, 0.5), (16.5, 70, 0.5), (17.0, 71, 1.0), (18.0, 67, 1.0),
        (20.0, 65, 0.5), (20.5, 67, 0.5), (21.0, 70, 1.0), (22.0, 74, 1.0),
        (24.0, 72, 1.0), (25.0, 70, 1.0), (26.0, 67, 1.0), (27.0, 65, 1.0),
        (28.0, 62, 1.0), (29.0, 65, 1.0), (30.0, 67, 1.0), (31.0, 62, 1.0),
    ];
    let a_bass: &[(f32, u8, f32)] = &[
        (0.0, 43, 1.0), (2.0, 50, 1.0), (4.0, 43, 1.0), (6.0, 50, 1.0),
        (8.0, 48, 1.0), (10.0, 55, 1.0), (12.0, 43, 1.0), (14.0, 50, 1.0),
        (16.0, 45, 1.0), (18.0, 52, 1.0), (20.0, 48, 1.0), (22.0, 55, 1.0),
        (24.0, 43, 1.0), (26.0, 50, 1.0), (28.0, 50, 1.0), (30.0, 57, 1.0),
    ];
    let a_chords: &[(f32, &'static [u8], f32)] = &[
        (1.0, G7, 1.0), (3.0, G7, 1.0), (5.0, G7, 1.0), (7.0, G7, 1.0),
        (9.0, C7, 1.0), (11.0, C7, 1.0), (13.0, G7, 1.0), (15.0, G7, 1.0),
        (17.0, D7, 1.0), (19.0, D7, 1.0), (21.0, C7, 1.0), (23.0, C7, 1.0),
        (25.0, G7, 1.0), (27.0, G7, 1.0), (29.0, D7, 1.0), (31.0, D7, 1.0),
    ];

    let b_melody: &[(f32, u8, f32)] = &[
        (0.0, 74, 0.5), (0.5, 70, 0.5), (1.0, 67, 1.0), (2.0, 70, 0.5), (2.5, 74, 0.5), (3.0, 77, 1.0),
        (4.0, 74, 0.5), (4.5, 72, 0.5), (5.0, 70, 1.0), (6.0, 67, 0.5), (6.5, 65, 0.5), (7.0, 62, 1.0),
        (8.0, 67, 0.5), (8.5, 70, 0.5), (9.0, 72, 1.0), (10.0, 70, 0.5), (10.5, 67, 0.5), (11.0, 65, 1.0),
        (12.0, 62, 0.5), (12.5, 65, 0.5), (13.0, 67, 1.0), (14.0, 70, 0.5), (14.5, 72, 0.5), (15.0, 74, 1.0),
        (16.0, 77, 0.5), (16.5, 74, 0.5), (17.0, 70, 1.0), (18.0, 67, 0.5), (18.5, 65, 0.5), (19.0, 62, 1.0),
        (20.0, 65, 0.5), (20.5, 67, 0.5), (21.0, 70, 1.0), (22.0, 67, 0.5), (22.5, 65, 0.5), (23.0, 62, 1.0),
        (24.0, 67, 0.5), (24.5, 70, 0.5), (25.0, 72, 1.0), (26.0, 74, 0.5), (26.5, 72, 0.5), (27.0, 70, 1.0),
        (28.0, 67, 0.5), (28.5, 65, 0.5), (29.0, 62, 1.0), (30.0, 65, 0.5), (30.5, 67, 0.5), (31.0, 62, 1.0),
    ];
    let b_bass: &[(f32, u8, f32)] = &[
        (0.0, 43, 1.0), (1.0, 45, 1.0), (2.0, 46, 1.0), (3.0, 48, 1.0),
        (4.0, 48, 1.0), (5.0, 50, 1.0), (6.0, 51, 1.0), (7.0, 53, 1.0),
        (8.0, 43, 1.0), (9.0, 45, 1.0), (10.0, 46, 1.0), (11.0, 48, 1.0),
        (12.0, 50, 1.0), (13.0, 51, 1.0), (14.0, 53, 1.0), (15.0, 55, 1.0),
        (16.0, 48, 1.0), (17.0, 50, 1.0), (18.0, 51, 1.0), (19.0, 53, 1.0),
        (20.0, 43, 1.0), (21.0, 45, 1.0), (22.0, 46, 1.0), (23.0, 48, 1.0),
        (24.0, 50, 1.0), (25.0, 51, 1.0), (26.0, 53, 1.0), (27.0, 55, 1.0),
        (28.0, 43, 1.0), (29.0, 46, 1.0), (30.0, 50, 1.0), (31.0, 55, 1.0),
    ];
    let b_chords: &[(f32, &'static [u8], f32)] = &[
        (0.0, G7, 1.0), (2.0, G7, 1.0), (4.0, C7, 1.0), (6.0, C7, 1.0),
        (8.0, G7, 1.0), (10.0, G7, 1.0), (12.0, D7, 1.0), (14.0, D7, 1.0),
        (16.0, C7, 1.0), (18.0, C7, 1.0), (20.0, G7, 1.0), (22.0, G7, 1.0),
        (24.0, F9, 1.0), (26.0, C7, 1.0), (28.0, G7, 1.0), (30.0, D7, 1.0),
    ];

    let mut melody = Vec::new();
    let mut bass = Vec::new();
    let mut chords = Vec::new();
    push_line(&mut melody, 0.0, a_melody);
    push_line(&mut melody, 64.0, b_melody);
    push_line(&mut bass, 0.0, a_bass);
    push_line(&mut bass, 64.0, b_bass);
    push_chords(&mut chords, 0.0, a_chords);
    push_chords(&mut chords, 64.0, b_chords);
    lay_layers_with_feel(q, Feel::LightSwing, &melody, &bass, &chords, 100, 82, 72)
}

fn piece_king_porter_stomp() -> Vec<NoteEvent> {
    // Jelly Roll Morton, King Porter Stomp (1924).
    // This one leans hardest into swing and riff repetition.
    let q = 60.0 / 126.0;
    const AB: &[u8] = &[63, 68, 72];
    const EB7: &[u8] = &[63, 67, 70, 73];
    const DB: &[u8] = &[61, 65, 68, 73];
    const EDIM: &[u8] = &[64, 67, 70];

    let a_melody: &[(f32, u8, f32)] = &[
        (0.0, 75, 0.5), (0.5, 80, 0.5), (1.0, 79, 0.5), (1.5, 75, 0.5),
        (2.0, 72, 0.5), (2.5, 75, 0.5), (3.0, 80, 1.0),
        (4.0, 78, 0.5), (4.5, 79, 0.5), (5.0, 80, 0.5), (5.5, 82, 0.5), (6.0, 80, 0.5), (6.5, 75, 0.5), (7.0, 72, 1.0),
        (8.0, 75, 0.5), (8.5, 77, 0.5), (9.0, 80, 0.5), (9.5, 82, 0.5), (10.0, 84, 0.5), (10.5, 82, 0.5), (11.0, 80, 1.0),
        (12.0, 79, 0.5), (12.5, 80, 0.5), (13.0, 82, 0.5), (13.5, 80, 0.5), (14.0, 79, 0.5), (14.5, 75, 0.5), (15.0, 72, 1.0),
    ];
    let a_bass: &[(f32, u8, f32)] = &[
        (0.0, 44, 1.0), (2.0, 51, 1.0), (4.0, 44, 1.0), (6.0, 51, 1.0),
        (8.0, 44, 1.0), (10.0, 49, 1.0), (12.0, 44, 1.0), (14.0, 51, 1.0),
        (16.0, 44, 1.0), (18.0, 51, 1.0), (20.0, 44, 1.0), (22.0, 51, 1.0),
        (24.0, 46, 1.0), (26.0, 53, 1.0), (28.0, 44, 1.0), (30.0, 51, 1.0),
    ];
    let a_chords: &[(f32, &'static [u8], f32)] = &[
        (1.0, AB, 1.0), (3.0, AB, 1.0), (5.0, AB, 1.0), (7.0, EB7, 1.0),
        (9.0, AB, 1.0), (11.0, DB, 1.0), (13.0, AB, 1.0), (15.0, EB7, 1.0),
        (17.0, AB, 1.0), (19.0, AB, 1.0), (21.0, DB, 1.0), (23.0, DB, 1.0),
        (25.0, EDIM, 1.0), (27.0, EB7, 1.0), (29.0, AB, 1.0), (31.0, EB7, 1.0),
    ];

    let b_melody: &[(f32, u8, f32)] = &[
        (0.0, 84, 0.5), (0.5, 82, 0.5), (1.0, 80, 0.5), (1.5, 77, 0.5),
        (2.0, 75, 0.5), (2.5, 77, 0.5), (3.0, 80, 1.0),
        (4.0, 82, 0.5), (4.5, 80, 0.5), (5.0, 79, 0.5), (5.5, 77, 0.5), (6.0, 75, 0.5), (6.5, 72, 0.5), (7.0, 75, 1.0),
        (8.0, 77, 0.5), (8.5, 79, 0.5), (9.0, 80, 0.5), (9.5, 82, 0.5), (10.0, 84, 0.5), (10.5, 87, 0.5), (11.0, 84, 1.0),
        (12.0, 82, 0.5), (12.5, 80, 0.5), (13.0, 79, 0.5), (13.5, 77, 0.5), (14.0, 75, 0.5), (14.5, 72, 0.5), (15.0, 68, 1.0),
    ];
    let b_bass: &[(f32, u8, f32)] = &[
        (0.0, 49, 1.0), (2.0, 56, 1.0), (4.0, 49, 1.0), (6.0, 56, 1.0),
        (8.0, 44, 1.0), (10.0, 51, 1.0), (12.0, 46, 1.0), (14.0, 53, 1.0),
        (16.0, 49, 1.0), (18.0, 56, 1.0), (20.0, 44, 1.0), (22.0, 51, 1.0),
        (24.0, 46, 1.0), (26.0, 53, 1.0), (28.0, 44, 1.0), (30.0, 51, 1.0),
    ];
    let b_chords: &[(f32, &'static [u8], f32)] = &[
        (1.0, DB, 1.0), (3.0, DB, 1.0), (5.0, AB, 1.0), (7.0, AB, 1.0),
        (9.0, EB7, 1.0), (11.0, EB7, 1.0), (13.0, EDIM, 1.0), (15.0, AB, 1.0),
        (17.0, DB, 1.0), (19.0, DB, 1.0), (21.0, AB, 1.0), (23.0, AB, 1.0),
        (25.0, EB7, 1.0), (27.0, EB7, 1.0), (29.0, AB, 1.0), (31.0, AB, 1.0),
    ];

    let mut melody = Vec::new();
    let mut bass = Vec::new();
    let mut chords = Vec::new();
    push_line(&mut melody, 0.0, a_melody);
    push_line(&mut melody, 64.0, a_melody);
    push_line(&mut melody, 128.0, b_melody);
    push_line(&mut melody, 192.0, b_melody);
    push_line(&mut bass, 0.0, a_bass);
    push_line(&mut bass, 64.0, a_bass);
    push_line(&mut bass, 128.0, b_bass);
    push_line(&mut bass, 192.0, b_bass);
    push_chords(&mut chords, 0.0, a_chords);
    push_chords(&mut chords, 64.0, a_chords);
    push_chords(&mut chords, 128.0, b_chords);
    push_chords(&mut chords, 192.0, b_chords);
    lay_layers_with_feel(q, Feel::Swing, &melody, &bass, &chords, 100, 82, 72)
}

fn piece_assistant_internal() -> Vec<NoteEvent> {
    // 即興 — F minor の slow ballad / late-night bar tune. AABA × 32
    // bars at quarter = 95 BPM. No transcription; this is what the
    // assistant put down without external reference. May or may not
    // sound like jazz to a listener — it is the assistant's honest
    // attempt at "what would I write".
    let q = 60.0 / 95.0;
    let mut v: Vec<NoteEvent> = Vec::new();

    let push_note = |v: &mut Vec<NoteEvent>, beat: f32, n: u8, dur: f32, vel: u8| {
        v.push(NoteEvent {
            start_sec: beat * q,
            midi_note: n,
            duration_sec: dur * q * 0.92,
            velocity: vel,
        });
    };

    // ---------------- A section (8 bars, repeated twice + once after B) ----
    // Chord plan: Fm7 | Fm7 | B♭m7 | B♭m7 | D♭maj7 | D♭maj7 | C7 | Fm7
    // F minor = 53 (F3), A♭3 = 56, C4 = 60, E♭4 = 63, F4 = 65, A♭4 = 68,
    // C5 = 72, D♭5 = 73, E♭5 = 75, F5 = 77, G5 = 79.
    // Bass octave = 2 (F2 = 41).
    let a_melody: &[(f32, u8, f32)] = &[
        // Bar 1 — call: rest, then ascending arpeggio of F minor
        (0.5, 65, 0.5), (1.0, 68, 0.5), (1.5, 72, 0.5),
        (2.0, 75, 1.0),                                // E♭5 sustain
        (3.0, 73, 1.0),                                // D♭5 (♭6) lean
        // Bar 3 — descend through B♭m7 chord
        (4.0, 72, 0.5), (4.5, 70, 0.5),                // C5, B♭4
        (5.0, 68, 0.5), (5.5, 66, 0.5),                // A♭4, G♭4 (blue note, ♭5 of C)
        (6.0, 65, 1.5),                                // F4 hold
        // Bar 5-6 — D♭maj7 area, lift to F5
        (8.0, 77, 1.0),                                // F5
        (9.0, 75, 0.5), (9.5, 73, 0.5),                // E♭5 D♭5
        (10.0, 72, 0.5), (10.5, 70, 0.5),              // C5 B♭4
        (11.0, 68, 1.0),                               // A♭4
        // Bar 7-8 — C7 → resolution to F minor
        (12.0, 71, 0.5), (12.5, 70, 0.5),              // B4, B♭4 (chromatic into V)
        (13.0, 72, 0.5), (13.5, 68, 0.5),              // C5, A♭4
        (14.0, 65, 2.0),                               // F4 final
    ];
    let a_bass: &[(f32, u8, f32)] = &[
        // Walking quarters across the 8 bars
        (0.0, 41, 0.95),                               // F2
        (1.0, 44, 0.95),                               // A♭2
        (2.0, 48, 0.95),                               // C3
        (3.0, 51, 0.95),                               // E♭3
        (4.0, 46, 0.95),                               // B♭2
        (5.0, 49, 0.95),                               // D♭3
        (6.0, 53, 0.95),                               // F3
        (7.0, 50, 0.95),                               // D3 (chromatic walk)
        (8.0, 49, 0.95),                               // D♭2 → 49 = D♭3
        (9.0, 53, 0.95),                               // F3
        (10.0, 56, 0.95),                              // A♭3
        (11.0, 53, 0.95),                              // F3
        (12.0, 36, 0.95),                              // C2 (V root)
        (13.0, 40, 0.95),                              // E2
        (14.0, 41, 1.95),                              // F2 hold cadence
    ];
    let a_chords: &[(f32, [u8; 3], f32)] = &[
        // Comp on beats 2 & 4. Sparse — leave space for the melody.
        (1.5, [56, 63, 68], 0.5),                      // Fm7 (A♭3 E♭4 A♭4)
        (3.5, [56, 63, 68], 0.5),
        (5.5, [58, 65, 70], 0.5),                      // B♭m7 (B♭3 F4 B♭4 partial)
        (7.5, [58, 65, 70], 0.5),
        (9.5, [60, 65, 72], 0.5),                      // D♭maj7-ish (C4 F4 C5)
        (11.5, [60, 65, 72], 0.5),
        (13.5, [55, 64, 70], 0.5),                     // C7 (G3 E4 B♭4)
    ];

    // ---------------- B section (bridge, 8 bars) ----
    // Chord plan: A♭maj7 | A♭maj7 | G♭maj7 | G♭maj7 | Fm7 | Fm7 | C7 | C7sus
    let b_melody: &[(f32, u8, f32)] = &[
        // Lift to A♭ major colour — brighter
        (0.0, 79, 1.0),                                // G5 (7 of A♭)
        (1.0, 77, 0.5), (1.5, 75, 0.5),                // F5 E♭5
        (2.0, 75, 1.0),                                // E♭5 hold
        (3.0, 72, 1.0),                                // C5
        // G♭maj7 — sit on the ♭5 colour
        (4.0, 73, 1.0),                                // D♭5
        (5.0, 70, 1.0),                                // B♭4
        (6.0, 73, 0.5), (6.5, 75, 0.5),                // D♭5 E♭5
        (7.0, 70, 1.0),                                // B♭4
        // Bar 5-6 — back to F minor figure, repeated motif
        (8.0, 68, 0.5), (8.5, 67, 0.5),                // A♭4 G4
        (9.0, 68, 1.0),                                // A♭4
        (10.0, 70, 0.5), (10.5, 68, 0.5),              // B♭4 A♭4
        (11.0, 65, 1.0),                               // F4
        // Bar 7-8 — C7 buildup back to A
        (12.0, 67, 0.5), (12.5, 70, 0.5),              // G4 B♭4
        (13.0, 72, 1.0),                               // C5
        (14.0, 70, 0.5), (14.5, 67, 0.5),              // B♭4 G4
        (15.0, 64, 1.0),                               // E4 (leading tone, suspended)
    ];
    let b_bass: &[(f32, u8, f32)] = &[
        (0.0, 44, 0.95),                               // A♭2
        (1.0, 48, 0.95),                               // C3
        (2.0, 51, 0.95),                               // E♭3
        (3.0, 48, 0.95),                               // C3
        (4.0, 42, 0.95),                               // G♭2
        (5.0, 46, 0.95),                               // B♭2
        (6.0, 49, 0.95),                               // D♭3
        (7.0, 46, 0.95),                               // B♭2
        (8.0, 41, 0.95),                               // F2
        (9.0, 44, 0.95),                               // A♭2
        (10.0, 48, 0.95),                              // C3
        (11.0, 51, 0.95),                              // E♭3
        (12.0, 36, 0.95),                              // C2
        (13.0, 40, 0.95),                              // E2
        (14.0, 43, 0.95),                              // G2
        (15.0, 46, 0.95),                              // B♭2 (leading back to A)
    ];
    let b_chords: &[(f32, [u8; 3], f32)] = &[
        (1.5, [56, 63, 67], 0.5),                      // A♭maj7 (A♭3 E♭4 G4)
        (3.5, [56, 63, 67], 0.5),
        (5.5, [54, 61, 66], 0.5),                      // G♭maj7 (G♭3 D♭4 G♭4 partial)
        (7.5, [54, 61, 66], 0.5),
        (9.5, [56, 63, 68], 0.5),                      // Fm7
        (11.5, [56, 63, 68], 0.5),
        (13.5, [55, 64, 70], 0.5),                     // C7
        (15.5, [55, 65, 70], 0.5),                     // C7sus (anticipation)
    ];

    // Lay out: A (0..16) + A (16..32) + B (32..48) + A (48..64).
    let push_at = |v: &mut Vec<NoteEvent>, off: f32,
                   melody: &[(f32, u8, f32)],
                   bass: &[(f32, u8, f32)],
                   chords: &[(f32, [u8; 3], f32)]| {
        for &(b, n, d) in melody {
            push_note(v, off + b, n, d, 95);
        }
        for &(b, n, d) in bass {
            push_note(v, off + b, n, d, 78);
        }
        for &(b, ns, d) in chords {
            for &n in &ns {
                push_note(v, off + b, n, d, 65);
            }
        }
    };
    push_at(&mut v, 0.0, a_melody, a_bass, a_chords);
    push_at(&mut v, 16.0, a_melody, a_bass, a_chords);
    push_at(&mut v, 32.0, b_melody, b_bass, b_chords);
    push_at(&mut v, 48.0, a_melody, a_bass, a_chords);
    v
}

fn piece_assistant_uptempo() -> Vec<NoteEvent> {
    // "ズンタズンタ" 即興 — F major upright stride, ~140 BPM, AABA × 32
    // bars. ~55 s. Bass plays root-on-1/3 + triadic chord stab on 2/4
    // (the canonical oompah / stride pattern). Right hand hook is
    // pentatonic-y with a IV-lift in the second phrase. Bridge moves
    // to B♭ (IV) for contrast then circles back. No transcription —
    // assistant's own choices for melody, bass walk, chord placement.
    let q = 60.0 / 140.0;
    let mut v: Vec<NoteEvent> = Vec::new();

    let push_note = |v: &mut Vec<NoteEvent>, beat: f32, n: u8, dur: f32, vel: u8| {
        v.push(NoteEvent {
            start_sec: beat * q,
            midi_note: n,
            duration_sec: dur * q * 0.85,
            velocity: vel,
        });
    };

    // Stride helper: lay out 4 beats per bar = bass(1) chord(2) bass(3) chord(4)
    fn push_stride_bar(
        v: &mut Vec<NoteEvent>,
        push_note: &impl Fn(&mut Vec<NoteEvent>, f32, u8, f32, u8),
        bar_start: f32,
        root: u8,
        fifth: u8,
        chord: [u8; 3],
    ) {
        push_note(v, bar_start + 0.0, root, 0.9, 88);
        for &n in &chord {
            push_note(v, bar_start + 1.0, n, 0.85, 70);
        }
        push_note(v, bar_start + 2.0, fifth, 0.9, 84);
        for &n in &chord {
            push_note(v, bar_start + 3.0, n, 0.85, 70);
        }
    }

    // Chord voicings (mid-register stabs, 3 notes each).
    // F major: F3 A3 C4 = [53, 57, 60]
    // B♭ major (IV): F3 B♭3 D4 = [53, 58, 62]
    // C7 (V): G3 B♭3 E4 = [55, 58, 64]
    // D7 (V/V): F#3 A3 C4 = [54, 57, 60]
    // G7: F3 G3 B3 = [53, 55, 59]
    // A7 (V/ii): E3 G3 C#4 = [52, 55, 61]
    // Dm7 (vi): A3 C4 F4 = [57, 60, 65]
    // Gm7 (vi of B♭): F3 B♭3 D4 = [53, 58, 62] (overlap, OK)

    // Bass walks: root MIDI for each bar (F2=41, B♭2=46, C3=48, D2=38, G2=43, A2=45, Dm=38, Gm=43)
    // Fifth note is bass+7 semitones (or octave +7).
    // Layout per bar in struct:
    type BarPlan = (u8, u8, [u8; 3]); // (bass_root_midi, bass_fifth_midi, chord_voicing)
    const F: BarPlan = (41, 48, [53, 57, 60]);
    const BB: BarPlan = (46, 53, [53, 58, 62]);
    const C7: BarPlan = (36, 43, [55, 58, 64]);
    const D7: BarPlan = (38, 45, [54, 57, 60]);
    const G7: BarPlan = (43, 50, [53, 55, 59]);
    const A7: BarPlan = (45, 40, [52, 55, 61]); // root A2, "fifth" pivots to E2 walk
    const DM: BarPlan = (38, 45, [57, 60, 65]); // D minor 7
    const GM: BarPlan = (43, 50, [53, 58, 62]); // G minor 7

    // A section chord plan (8 bars):
    //   F | F | B♭ | F | D7 | G7 | C7 | F
    let a_chords: &[BarPlan] = &[F, F, BB, F, D7, G7, C7, F];

    // B section (bridge, 8 bars):
    //   B♭ | B♭ | Gm | C7 | A7 | Dm | G7 | C7
    let b_chords: &[BarPlan] = &[BB, BB, GM, C7, A7, DM, G7, C7];

    // A section melody (16 beats = 4 bars × 4 beats), repeated for second half
    // F4=65, A4=69, C5=72, D5=74, F5=77, G5=79, B♭5=82
    let a_melody_h1: &[(f32, u8, f32)] = &[
        // Bar 1 — pickup-style ascending hook
        (0.5, 65, 0.5),  (1.0, 69, 0.5),  (1.5, 72, 0.5),  (2.0, 77, 1.0),  (3.0, 74, 1.0),
        // Bar 2 — descend
        (4.0, 72, 0.5), (4.5, 69, 0.5), (5.0, 65, 1.0), (6.0, 67, 0.5), (6.5, 69, 0.5),
        (7.0, 70, 1.0), // B♭4 leading into IV
        // Bar 3 — over B♭, lift
        (8.0, 70, 0.5), (8.5, 74, 0.5), (9.0, 77, 0.5), (9.5, 74, 0.5),
        (10.0, 70, 1.0), (11.0, 65, 1.0),
        // Bar 4 — back to F, simple
        (12.0, 69, 0.5), (12.5, 72, 0.5), (13.0, 69, 0.5), (13.5, 65, 0.5),
        (14.0, 65, 2.0),
    ];
    let a_melody_h2: &[(f32, u8, f32)] = &[
        // Bar 5 — restate but bigger
        (16.0, 65, 0.5), (16.5, 69, 0.5), (17.0, 72, 0.5), (17.5, 77, 0.5),
        (18.0, 79, 1.0), (19.0, 77, 1.0),
        // Bar 6 — D7 chromatic colour
        (20.0, 74, 0.5), (20.5, 73, 0.5), (21.0, 72, 0.5), (21.5, 69, 0.5),
        (22.0, 66, 1.0), (23.0, 69, 1.0),
        // Bar 7 — G7 → C7 step
        (24.0, 71, 0.5), (24.5, 72, 0.5), (25.0, 74, 0.5), (25.5, 76, 0.5),
        (26.0, 77, 1.0), (27.0, 74, 1.0),
        // Bar 8 — cadence to F
        (28.0, 72, 0.5), (28.5, 69, 0.5), (29.0, 65, 0.5), (29.5, 64, 0.5),
        (30.0, 65, 2.0),
    ];

    // B section melody (bridge, 16 beats from offset 32)
    let b_melody: &[(f32, u8, f32)] = &[
        // Bar 9 — B♭, brighter top
        (32.0, 77, 0.5), (32.5, 79, 0.5), (33.0, 82, 1.0),
        (34.0, 79, 0.5), (34.5, 77, 0.5), (35.0, 74, 1.0),
        // Bar 10 — B♭ continued
        (36.0, 73, 0.5), (36.5, 74, 0.5), (37.0, 77, 0.5), (37.5, 74, 0.5),
        (38.0, 70, 2.0),
        // Bar 11 — Gm
        (40.0, 70, 0.5), (40.5, 72, 0.5), (41.0, 74, 0.5), (41.5, 75, 0.5),
        (42.0, 74, 1.0), (43.0, 70, 1.0),
        // Bar 12 — C7 with anticipation
        (44.0, 67, 0.5), (44.5, 70, 0.5), (45.0, 72, 0.5), (45.5, 76, 0.5),
        (46.0, 77, 2.0),
        // Bar 13 — A7 (V/ii) — reach high
        (48.0, 69, 0.5), (48.5, 72, 0.5), (49.0, 76, 0.5), (49.5, 77, 0.5),
        (50.0, 81, 1.0), (51.0, 77, 1.0),
        // Bar 14 — Dm
        (52.0, 74, 0.5), (52.5, 72, 0.5), (53.0, 69, 0.5), (53.5, 65, 0.5),
        (54.0, 62, 2.0),
        // Bar 15 — G7
        (56.0, 65, 0.5), (56.5, 67, 0.5), (57.0, 71, 0.5), (57.5, 74, 0.5),
        (58.0, 77, 1.0), (59.0, 74, 1.0),
        // Bar 16 — C7 turnaround
        (60.0, 72, 0.5), (60.5, 76, 0.5), (61.0, 79, 0.5), (61.5, 76, 0.5),
        (62.0, 70, 2.0),
    ];

    // Lay out form: A | A | B | A (bars 0..8, 8..16, 16..24, 24..32)
    // Each section is 8 bars × 4 beats = 32 beats.
    // We have a_chords (8 bars) and a_melody_h1 (4 bars) + a_melody_h2 (4 bars).
    let push_a_section = |v: &mut Vec<NoteEvent>, off: f32| {
        for (i, plan) in a_chords.iter().enumerate() {
            push_stride_bar(v, &push_note, off + (i as f32) * 4.0, plan.0, plan.1, plan.2);
        }
        for &(b, n, d) in a_melody_h1 {
            push_note(v, off + b, n, d, 100);
        }
        for &(b, n, d) in a_melody_h2 {
            push_note(v, off + b, n, d, 100);
        }
    };
    let push_b_section = |v: &mut Vec<NoteEvent>, off: f32| {
        for (i, plan) in b_chords.iter().enumerate() {
            push_stride_bar(v, &push_note, off + (i as f32) * 4.0, plan.0, plan.1, plan.2);
        }
        for &(b, n, d) in b_melody {
            push_note(v, off + (b - 32.0), n, d, 100);
        }
    };

    push_a_section(&mut v, 0.0);     // A (bars 1-8)
    push_a_section(&mut v, 32.0);    // A (bars 9-16)
    push_b_section(&mut v, 64.0);    // B (bars 17-24)
    push_a_section(&mut v, 96.0);    // A (bars 25-32)
    v
}

fn piece_gemini_nocturne() -> Vec<NoteEvent> {
    // Gemini original: Nocturne in C# Minor.
    // Slow, atmospheric ballad, ~53 seconds at 72 BPM.
    // ABA structure (8 + 4 + 4 bars).
    let q = 60.0 / 72.0;

    // Voicings
    const CSM: &[u8] = &[49, 52, 56]; // C#3 E3 G#3
    const FSM: &[u8] = &[54, 57, 61]; // F#3 A3 C#4
    const GS7: &[u8] = &[56, 60, 63, 66]; // G#3 B#3 D#4 F#4 (B#3=C4=60)
    const AMAJ: &[u8] = &[57, 61, 64]; // A3 C#4 E4
    const EMAJ: &[u8] = &[52, 56, 59, 64]; // E3 G#3 B3 E4
    const DSDIM: &[u8] = &[51, 54, 57, 60]; // D#dim7: D#3 F#3 A3 C4

    let mut melody = Vec::new();
    let mut bass = Vec::new();
    let mut chords = Vec::new();

    // --- Section A (Bars 1-8) ---
    // Chord: C#m | F#m | G#7 | C#m | A | E | D#dim7 | G#7
    let a_bass: &[(f32, u8, f32)] = &[
        (0.0, 25, 4.0), (4.0, 30, 4.0), (8.0, 32, 4.0), (12.0, 37, 4.0),
        (16.0, 21, 4.0), (20.0, 28, 4.0), (24.0, 27, 4.0), (28.0, 32, 4.0),
    ];
    let a_chords: &[(f32, &'static [u8], f32)] = &[
        (1.0, CSM, 1.0), (2.0, CSM, 1.0), (3.0, CSM, 1.0),
        (5.0, FSM, 1.0), (6.0, FSM, 1.0), (7.0, FSM, 1.0),
        (9.0, GS7, 1.0), (10.0, GS7, 1.0), (11.0, GS7, 1.0),
        (13.0, CSM, 1.0), (14.0, CSM, 1.0), (15.0, CSM, 1.0),
        (17.0, AMAJ, 1.0), (18.0, AMAJ, 1.0), (19.0, AMAJ, 1.0),
        (21.0, EMAJ, 1.0), (22.0, EMAJ, 1.0), (23.0, EMAJ, 1.0),
        (25.0, DSDIM, 1.0), (26.0, DSDIM, 1.0), (27.0, DSDIM, 1.0),
        (29.0, GS7, 1.0), (30.0, GS7, 1.0), (31.0, GS7, 1.0),
    ];
    let a_melody: &[(f32, u8, f32)] = &[
        (0.5, 61, 1.0), (1.5, 63, 0.5), (2.0, 64, 1.5),
        (4.5, 66, 1.0), (5.5, 68, 0.5), (6.0, 69, 1.5),
        (8.5, 70, 0.5), (9.0, 71, 0.5), (9.5, 72, 1.0), (11.0, 68, 1.0),
        (12.5, 64, 0.5), (13.0, 63, 0.5), (13.5, 61, 2.0),
        // Phase 2 of A
        (16.5, 69, 1.0), (17.5, 73, 0.5), (18.0, 72, 1.5),
        (20.5, 71, 1.0), (21.5, 75, 0.5), (22.0, 76, 1.5),
        (24.5, 77, 0.5), (25.0, 75, 0.5), (25.5, 73, 1.0), (27.0, 72, 1.0),
        (28.5, 68, 0.5), (29.0, 67, 0.5), (29.5, 68, 2.0),
    ];

    // --- Section B (Bars 9-12) ---
    // Modulation to E Major area: E | B7 | A | G#7
    let b_bass: &[(f32, u8, f32)] = &[
        (32.0, 28, 4.0), (36.0, 35, 4.0), (40.0, 33, 4.0), (44.0, 32, 4.0),
    ];
    let b_chords: &[(f32, &'static [u8], f32)] = &[
        (33.0, EMAJ, 1.0), (34.0, EMAJ, 1.0), (35.0, EMAJ, 1.0),
        (37.0, GS7, 1.0), (38.0, GS7, 1.0), (39.0, GS7, 1.0), // Using G#7 as B7-ish proxy or V of iii
        (41.0, AMAJ, 1.0), (42.0, AMAJ, 1.0), (43.0, AMAJ, 1.0),
        (45.0, GS7, 1.0), (46.0, GS7, 1.0), (47.0, GS7, 1.0),
    ];
    let b_melody: &[(f32, u8, f32)] = &[
        (32.5, 76, 1.5), (34.5, 75, 0.5), (35.0, 73, 1.0),
        (36.5, 71, 1.5), (38.5, 70, 0.5), (39.0, 68, 1.0),
        (40.5, 69, 1.0), (41.5, 71, 0.5), (42.0, 73, 1.5),
        (44.5, 72, 1.0), (45.5, 70, 0.5), (46.0, 68, 1.5),
    ];

    // --- Section A reprise (Bars 13-16) ---
    // Final cadence: C#m | F#m | G#7 | C#m
    let a2_bass: &[(f32, u8, f32)] = &[
        (48.0, 25, 4.0), (52.0, 30, 4.0), (56.0, 32, 4.0), (60.0, 37, 4.0),
    ];
    let a2_chords: &[(f32, &'static [u8], f32)] = &[
        (49.0, CSM, 1.0), (50.0, CSM, 1.0), (51.0, CSM, 1.0),
        (53.0, FSM, 1.0), (54.0, FSM, 1.0), (55.0, FSM, 1.0),
        (57.0, GS7, 1.0), (58.0, GS7, 1.0), (59.0, GS7, 1.0),
        (61.0, CSM, 3.0),
    ];
    let a2_melody: &[(f32, u8, f32)] = &[
        (48.5, 73, 1.5), (50.5, 75, 0.5), (51.0, 76, 1.0),
        (52.5, 78, 1.0), (53.5, 80, 0.5), (54.0, 81, 1.5),
        (56.5, 80, 0.5), (57.0, 78, 0.5), (57.5, 76, 0.5), (58.0, 75, 0.5),
        (60.0, 73, 4.0),
    ];

    push_line(&mut melody, 0.0, a_melody);
    push_line(&mut melody, 0.0, b_melody);
    push_line(&mut melody, 0.0, a2_melody);
    push_line(&mut bass, 0.0, a_bass);
    push_line(&mut bass, 0.0, b_bass);
    push_line(&mut bass, 0.0, a2_bass);
    push_chords(&mut chords, 0.0, a_chords);
    push_chords(&mut chords, 0.0, b_chords);
    push_chords(&mut chords, 0.0, a2_chords);

    lay_layers_with_feel(q, Feel::Straight, &melody, &bass, &chords, 92, 75, 65)
}

fn piece_gemini_boogie() -> Vec<NoteEvent> {
    // Gemini original: Boogie-Woogie Stride in G Major.
    // High-energy, danceable, ~40 seconds at 144 BPM.
    // Structure: A (8) + A (8) + B (4) + A-Final (8).
    let q = 60.0 / 144.0;

    // Voicings
    const G_MAJ: &[u8] = &[55, 59, 62]; // G3 B3 D4
    const C_MAJ: &[u8] = &[52, 55, 60, 64]; // C3 G3 C4 E4
    const D7: &[u8] = &[54, 57, 60, 62]; // D3 F#3 A3 C4 D4
    const E7: &[u8] = &[52, 56, 59, 62]; // E3 G#3 B3 D4

    let mut melody = Vec::new();
    let mut bass = Vec::new();
    let mut chords = Vec::new();

    // Section A Bass & Chords (0.0 to 32.0 beats)
    let a_bass_chords: &[(f32, u8, &'static [u8])] = &[
        (0.0, 31, G_MAJ), (4.0, 31, G_MAJ), (8.0, 36, C_MAJ), (12.0, 31, G_MAJ),
        (16.0, 38, D7), (20.0, 36, C_MAJ), (24.0, 31, G_MAJ), (28.0, 38, D7),
    ];
    for &(off, root, chord) in a_bass_chords {
        bass.push((off + 0.0, root, 0.9));
        bass.push((off + 2.0, root + 7, 0.9));
        chords.push((off + 1.0, chord, 0.6));
        chords.push((off + 3.0, chord, 0.6));
    }

    // A Melody: Rhythmic, syncopated
    let a_mel: &[(f32, u8, f32)] = &[
        (0.0, 67, 0.5), (0.5, 71, 0.5), (1.0, 74, 1.0), (2.5, 74, 0.5), (3.0, 76, 1.0),
        (4.0, 74, 0.5), (4.5, 71, 0.5), (5.0, 67, 1.0), (6.5, 67, 0.5), (7.0, 64, 1.0),
        (8.0, 64, 0.5), (8.5, 67, 0.5), (9.0, 70, 0.5), (9.5, 72, 1.0), (11.0, 70, 0.5), (11.5, 67, 0.5),
        (12.0, 67, 2.0),
        (16.0, 78, 0.5), (17.0, 78, 0.5), (18.0, 76, 0.5), (19.0, 74, 1.0),
        (20.0, 72, 0.5), (21.0, 72, 0.5), (22.0, 70, 0.5), (23.0, 67, 1.0),
        (24.0, 67, 0.5), (24.5, 71, 0.5), (25.0, 74, 0.5), (25.5, 76, 0.5), (26.0, 79, 1.0), (27.0, 82, 1.0),
        (28.0, 83, 0.5), (29.0, 79, 0.5), (30.0, 74, 1.0), (31.0, 67, 1.0),
    ];
    push_line(&mut melody, 0.0, a_mel);
    push_line(&mut melody, 32.0, a_mel); // Repeat A

    // Repeat Bass/Chords for A repeat
    for &(off, root, chord) in a_bass_chords {
        let b_off = off + 32.0;
        bass.push((b_off + 0.0, root, 0.9));
        bass.push((b_off + 2.0, root + 7, 0.9));
        chords.push((b_off + 1.0, chord, 0.6));
        chords.push((b_off + 3.0, chord, 0.6));
    }

    // --- Section B (Bridge, 4 bars) ---
    // E7 | A7 (D7) | D7 | D7
    let b_bass_chords: &[(f32, u8, &'static [u8])] = &[
        (64.0, 28, E7), (68.0, 33, D7), (72.0, 38, D7), (76.0, 38, D7),
    ];
    for &(off, root, chord) in b_bass_chords {
        bass.push((off + 0.0, root, 0.9));
        bass.push((off + 2.0, root + 7, 0.9));
        chords.push((off + 1.0, chord, 0.6));
        chords.push((off + 3.0, chord, 0.6));
    }
    let b_mel: &[(f32, u8, f32)] = &[
        (64.0, 76, 0.5), (64.5, 76, 0.5), (65.0, 80, 0.5), (65.5, 83, 1.0),
        (68.0, 81, 0.5), (68.5, 78, 0.5), (69.0, 74, 1.0),
        (72.0, 74, 0.5), (72.5, 78, 0.5), (73.0, 81, 0.5), (73.5, 84, 0.5), (74.0, 86, 2.0),
    ];
    push_line(&mut melody, 0.0, b_mel);

    // Final A return
    for &(off, root, chord) in a_bass_chords {
        let f_off = off + 80.0;
        bass.push((f_off + 0.0, root, 0.9));
        bass.push((f_off + 2.0, root + 7, 0.9));
        chords.push((f_off + 1.0, chord, 0.6));
        chords.push((f_off + 3.0, chord, 0.6));
    }
    push_line(&mut melody, 80.0, a_mel);

    lay_layers_with_feel(q, Feel::LightSwing, &melody, &bass, &chords, 105, 95, 85)
}
fn piece_codex_railspike() -> Vec<NoteEvent> {
    // 「railspike」 — Codex がこの名前を選んだあと usage limit に当たって
    // 中身は書けなかったので、本体は assistant の引き取り。train-clack
    // boogie-woogie in C: classic 8-note bass figure on each bar
    // (root-3-5-6-♭7-6-5-3) is the train. 12-bar blues progression
    // I-I-I-I7 / IV7-IV7-I-I / V7-IV7-I-V7. Two choruses, ~144 BPM.
    // Treble layers blues-scale shouts on the I/IV/V landings.
    let q = 60.0 / 144.0;
    let mut v: Vec<NoteEvent> = Vec::new();

    let push_note = |v: &mut Vec<NoteEvent>, beat: f32, n: u8, dur: f32, vel: u8| {
        v.push(NoteEvent {
            start_sec: beat * q,
            midi_note: n,
            duration_sec: dur * q * 0.85,
            velocity: vel,
        });
    };

    // Boogie bass figure: 8 eighth notes per bar = root, 3rd, 5th,
    // 6th, ♭7, 6th, 5th, 3rd. Dominant 7th flavour throughout — the
    // ♭7 on beat 3 is the engine of the figure.
    fn push_boogie_bar(
        v: &mut Vec<NoteEvent>,
        push_note: &impl Fn(&mut Vec<NoteEvent>, f32, u8, f32, u8),
        bar_start: f32,
        chord_root_midi: u8,
    ) {
        // Intervals in semitones from root for the 8 eighths:
        // 0 (root), 4 (3), 7 (5), 9 (6), 10 (♭7), 9, 7, 4
        const INTERVALS: [u8; 8] = [0, 4, 7, 9, 10, 9, 7, 4];
        for (i, semi) in INTERVALS.iter().enumerate() {
            let beat = bar_start + (i as f32) * 0.5;
            push_note(v, beat, chord_root_midi + semi, 0.5, 90);
        }
    }

    // 12-bar blues in C: I-I-I-I7 / IV-IV-I-I / V-IV-I-V (turnaround)
    // C2=36, F2=41, G2=43.
    let progression: [u8; 12] = [36, 36, 36, 36, 41, 41, 36, 36, 43, 41, 36, 43];

    // Treble shouts — short syncopated riffs on each chord landing
    // C blues scale: C E♭ F G♭ G B♭ C → 60, 63, 65, 66, 67, 70, 72
    // F blues scale: F A♭ B♭ B C E♭ F → 65, 68, 70, 71, 72, 75, 77
    // G blues scale: G B♭ C D♭ D F G → 67, 70, 72, 73, 74, 77, 79

    // Chorus 1 treble (sparse — let the bass drive the train)
    let treble_chorus_1: &[(f32, u8, f32)] = &[
        // Bar 1 — 8th-note pickup riff into beat 3
        (1.5, 67, 0.5), (2.0, 70, 0.5), (2.5, 72, 1.0),
        // Bar 2 — answer
        (5.5, 70, 0.5), (6.0, 67, 0.5), (6.5, 65, 1.5),
        // Bar 5 — IV chord shout
        (16.0, 72, 0.5), (16.5, 75, 0.5), (17.0, 77, 1.0),
        (18.0, 75, 0.5), (18.5, 72, 0.5), (19.0, 70, 1.0),
        // Bar 7 — back to I
        (24.0, 67, 0.5), (24.5, 70, 0.5), (25.0, 72, 1.0),
        (26.0, 70, 0.5), (26.5, 67, 0.5),
        // Bar 9 — V chord climax
        (32.0, 74, 0.5), (32.5, 77, 0.5), (33.0, 79, 1.0),
        (34.0, 77, 0.5), (34.5, 74, 0.5),
        // Bar 10 — IV (resolving)
        (36.0, 75, 0.5), (36.5, 72, 0.5), (37.0, 70, 1.0),
        // Bar 11 — I cadence
        (40.0, 72, 0.5), (40.5, 70, 0.5), (41.0, 67, 1.0),
        // Bar 12 — turnaround pickup to chorus 2
        (44.0, 67, 0.5), (44.5, 70, 0.5), (45.0, 72, 0.5), (45.5, 74, 0.5),
    ];

    // Chorus 2 treble — busier, bigger
    let treble_chorus_2: &[(f32, u8, f32)] = &[
        // Bar 13 — call up high
        (48.0, 75, 0.5), (48.5, 77, 0.5), (49.0, 79, 0.5), (49.5, 82, 0.5),
        (50.0, 84, 1.0),
        // Bar 14 — descend through C blues
        (52.0, 82, 0.5), (52.5, 79, 0.5), (53.0, 77, 0.5), (53.5, 75, 0.5),
        (54.0, 72, 1.0),
        // Bar 15 — repeat with variation
        (56.0, 75, 0.5), (56.5, 78, 0.5), (57.0, 79, 1.0),
        (58.0, 77, 0.5), (58.5, 75, 0.5), (59.0, 72, 1.0),
        // Bar 17 — IV
        (64.0, 72, 0.5), (64.5, 75, 0.5), (65.0, 77, 0.5), (65.5, 80, 0.5),
        (66.0, 81, 1.0),
        // Bar 18
        (68.0, 80, 0.5), (68.5, 77, 0.5), (69.0, 75, 1.0),
        // Bar 19 — back to I
        (72.0, 79, 0.5), (72.5, 75, 0.5), (73.0, 72, 0.5), (73.5, 70, 0.5),
        (74.0, 67, 1.0),
        // Bar 21 — V climax
        (80.0, 79, 0.5), (80.5, 82, 0.5), (81.0, 84, 1.0),
        // Bar 22 — IV resolution
        (84.0, 82, 0.5), (84.5, 79, 0.5), (85.0, 77, 1.0),
        // Bar 23 — I final
        (88.0, 75, 0.5), (88.5, 72, 0.5), (89.0, 67, 0.5), (89.5, 60, 0.5),
        // Bar 24 — V hold
        (92.0, 67, 1.0), (93.0, 70, 1.0), (94.0, 72, 2.0),
    ];

    // Lay out bass: 24 bars (chorus 1: 0..12, chorus 2: 12..24)
    for chorus in 0..2 {
        for bar_idx in 0..12 {
            let bar_start = (chorus as f32) * 48.0 + (bar_idx as f32) * 4.0;
            push_boogie_bar(&mut v, &push_note, bar_start, progression[bar_idx]);
        }
    }
    for &(b, n, d) in treble_chorus_1 {
        push_note(&mut v, b, n, d, 100);
    }
    for &(b, n, d) in treble_chorus_2 {
        push_note(&mut v, b, n, d, 105);
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
        "blues_in_c" => Ok(piece_blues_in_c()),
        "maple_leaf_rag" => Ok(piece_maple_leaf_rag()),
        "entertainer" => Ok(piece_entertainer()),
        "twelfth_street_rag" => Ok(piece_twelfth_street_rag()),
        "st_louis_blues" => Ok(piece_st_louis_blues()),
        "king_porter_stomp" => Ok(piece_king_porter_stomp()),
        "assistant_internal" => Ok(piece_assistant_internal()),
        "assistant_uptempo" => Ok(piece_assistant_uptempo()),
        "codex_punchout" => Ok(piece_assistant_uptempo()),
        "codex_railspike" => Ok(piece_codex_railspike()),
        "gemini_nocturne" => Ok(piece_gemini_nocturne()),
        "gemini_boogie" => Ok(piece_gemini_boogie()),
        other => Err(format!(
            "unknown piece: {other} \
             (c_progression|minor_cadence|arpeggio|twinkle|bach_invention|\
              bach_prelude_c|fur_elise|gymnopedie|canon_d|blues_in_c|\
              maple_leaf_rag|entertainer|twelfth_street_rag|st_louis_blues|king_porter_stomp|\
              assistant_internal|assistant_uptempo|codex_punchout|codex_railspike|gemini_nocturne|gemini_boogie)"
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
                     --piece NAME      c_progression|minor_cadence|arpeggio|twinkle|bach_invention|gemini_nocturne|gemini_boogie\n  \
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
