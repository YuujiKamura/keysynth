//! Import a chiptune-lab JSON song (the format shared by
//! `ai-chiptune-demo/songs/*.json` and `listener-lab/chunks/*.json`)
//! into keysynth's NoteEvent + DrumEvent representation.
//!
//! Schema (only the parts we use):
//! ```jsonc
//! {
//!   "bpm": 150,
//!   "bars": 4,                 // optional; we infer from track length
//!   "tracks": {
//!     "lead":  ["A4", 2, "E4", 1, "rest", 1, ...],   // [note, sixteenth_steps] pairs
//!     "harm":  [...],
//!     "bass":  [...],
//!     "drums": "K.H.S.HK..K.S.HH"                    // one char per 16th step
//!   }
//! }
//! ```
//!
//! Each track event is 16 sixteenth-steps per bar. `step_sec = 60/bpm/4`.

use serde::Deserialize;
use serde_json::Value;

use crate::drums::{parse_drum_pattern, DrumEvent};

#[derive(Debug, Deserialize)]
pub struct ChiptuneSong {
    pub name: Option<String>,
    pub bpm: f32,
    pub bars: Option<u32>,
    pub key: Option<String>,
    pub tracks: ChiptuneTracks,
}

#[derive(Debug, Deserialize)]
pub struct ChiptuneTracks {
    #[serde(default)]
    pub lead: Vec<Value>,
    #[serde(default)]
    pub harm: Vec<Value>,
    #[serde(default)]
    pub bass: Vec<Value>,
    #[serde(default)]
    pub drums: String,
}

/// Note name "A4" → MIDI 69. Returns None for "rest" / unparseable.
fn note_name_to_midi(s: &str) -> Option<u8> {
    if s.eq_ignore_ascii_case("rest") {
        return None;
    }
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let pitch = match bytes[0].to_ascii_uppercase() {
        b'C' => 0,
        b'D' => 2,
        b'E' => 4,
        b'F' => 5,
        b'G' => 7,
        b'A' => 9,
        b'B' => 11,
        _ => return None,
    };
    let mut idx = 1;
    let mut accidental = 0i32;
    if idx < bytes.len() {
        match bytes[idx] {
            b'#' => {
                accidental = 1;
                idx += 1;
            }
            b'b' => {
                accidental = -1;
                idx += 1;
            }
            _ => {}
        }
    }
    if idx >= bytes.len() {
        return None;
    }
    let octave: i32 = std::str::from_utf8(&bytes[idx..]).ok()?.parse().ok()?;
    let midi = (pitch as i32) + accidental + 12 * (octave + 1);
    if (0..=127).contains(&midi) {
        Some(midi as u8)
    } else {
        None
    }
}

/// One pitched note event in seconds.
#[derive(Clone, Copy, Debug)]
pub struct ImportedNote {
    pub start_sec: f32,
    pub midi_note: u8,
    pub duration_sec: f32,
    pub velocity: u8,
}

fn iter_track(events: &[Value], step_sec: f32, velocity: u8, gate: f32) -> Vec<ImportedNote> {
    let mut out = Vec::new();
    let mut t = 0.0_f32;
    let mut i = 0;
    while i + 1 < events.len() {
        let name = events[i].as_str().unwrap_or("rest");
        let steps = events[i + 1].as_f64().unwrap_or(0.0) as f32;
        let dur = steps * step_sec;
        if let Some(midi) = note_name_to_midi(name) {
            out.push(ImportedNote {
                start_sec: t,
                midi_note: midi,
                duration_sec: (dur * gate).max(0.005),
                velocity,
            });
        }
        t += dur;
        i += 2;
    }
    out
}

/// Parse a song JSON and extract (lead+harm+bass merged notes, drum events).
pub fn parse_song_json(
    json_text: &str,
) -> Result<(Vec<ImportedNote>, Vec<DrumEvent>, ChiptuneSong), String> {
    let song: ChiptuneSong = serde_json::from_str(json_text).map_err(|e| format!("parse: {e}"))?;
    let step_sec = 60.0 / song.bpm.max(1.0) / 4.0;

    let mut notes: Vec<ImportedNote> = Vec::new();
    notes.extend(iter_track(&song.tracks.lead, step_sec, 100, 0.92));
    notes.extend(iter_track(&song.tracks.harm, step_sec, 75, 0.92));
    notes.extend(iter_track(&song.tracks.bass, step_sec, 85, 0.95));
    notes.sort_by(|a, b| {
        a.start_sec
            .partial_cmp(&b.start_sec)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let drums = parse_drum_pattern(&song.tracks.drums, 0.0, step_sec, 100);

    Ok((notes, drums, song))
}
