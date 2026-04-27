//! Step-sequencer pattern grid (issue #7 Stage 6).
//!
//! `Pattern` is a tracker-style grid: a fixed `bars * steps_per_bar`
//! row count with one boolean per step per drum track (Kick / Snare /
//! Hi-Hat). It is independent of the piano voices and the Stage 1
//! mixer; the GUI driver lives in `src/bin/seq_grid.rs`.
//!
//! Two import paths exist:
//! 1. Direct cell toggling from the egui grid.
//! 2. Chiptune-format string ("K.H.S.HK..." style) via
//!    [`Pattern::from_chiptune_string`], compatible with the syntax
//!    already used by [`crate::drums::parse_drum_pattern`].
//!
//! Output: [`Pattern::to_drum_events`] returns
//! `Vec<crate::drums::DrumEvent>` ready for the existing drum
//! synthesiser, so the same render path used by `render_chiptune`
//! works for sequencer playback / wav export.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::drums::DrumEvent;

/// Three drum tracks, mirroring the `K`/`S`/`H` characters used by
/// `drums.rs`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum TrackKind {
    Kick,
    Snare,
    HiHat,
}

impl TrackKind {
    /// Iteration order used by the grid UI and (de)serialisation.
    pub const ALL: [TrackKind; 3] = [TrackKind::Kick, TrackKind::Snare, TrackKind::HiHat];

    /// Single-character label matching `drums::DrumEvent::kind`.
    pub fn as_char(self) -> char {
        match self {
            TrackKind::Kick => 'K',
            TrackKind::Snare => 'S',
            TrackKind::HiHat => 'H',
        }
    }

    /// Inverse of [`as_char`]: parse a chiptune-format character into
    /// a `TrackKind`. Anything not in `K|S|H` (including `.`) returns
    /// `None`.
    pub fn from_char(c: char) -> Option<TrackKind> {
        match c {
            'K' => Some(TrackKind::Kick),
            'S' => Some(TrackKind::Snare),
            'H' => Some(TrackKind::HiHat),
            _ => None,
        }
    }

    /// Short two-letter label for the GUI row header.
    pub fn label(self) -> &'static str {
        match self {
            TrackKind::Kick => "Kick",
            TrackKind::Snare => "Snare",
            TrackKind::HiHat => "HiHat",
        }
    }
}

/// A drum-grid pattern. `tracks[k][i]` = true means drum `k` fires at
/// step `i`. Step count = `bars * steps_per_bar`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pattern {
    pub bpm: f32,
    pub bars: u32,
    pub steps_per_bar: u32,
    /// One row per track. Missing rows are treated as all-rests on
    /// load (so an old JSON without HiHat still parses).
    pub tracks: HashMap<TrackKind, Vec<bool>>,
}

impl Pattern {
    /// Empty grid with the given dimensions; every cell is rest.
    pub fn empty(bpm: f32, bars: u32, steps_per_bar: u32) -> Self {
        let total = (bars * steps_per_bar) as usize;
        let mut tracks: HashMap<TrackKind, Vec<bool>> = HashMap::new();
        for k in TrackKind::ALL {
            tracks.insert(k, vec![false; total]);
        }
        Self {
            bpm,
            bars,
            steps_per_bar,
            tracks,
        }
    }

    /// Total step count = `bars * steps_per_bar`.
    pub fn total_steps(&self) -> usize {
        (self.bars * self.steps_per_bar) as usize
    }

    /// Seconds per step: a step is one sub-beat at the given BPM.
    /// Convention: `steps_per_bar = 16` ⇒ sixteenth-notes ⇒ 4 steps
    /// per quarter, so step = 60 / bpm / 4. Generalises for arbitrary
    /// `steps_per_bar` by treating one bar as 4 quarters (4/4 time).
    pub fn step_sec(&self) -> f32 {
        // bar_sec = 4 quarters * (60 / bpm). step_sec = bar_sec / steps_per_bar.
        let bar_sec = 4.0 * 60.0 / self.bpm;
        bar_sec / self.steps_per_bar as f32
    }

    /// Toggle (Kick / Snare / HiHat) at step `i`. No-op if the track
    /// is missing or `i` is out of range.
    pub fn toggle(&mut self, kind: TrackKind, i: usize) {
        if let Some(row) = self.tracks.get_mut(&kind) {
            if let Some(cell) = row.get_mut(i) {
                *cell = !*cell;
            }
        }
    }

    /// Read cell value, defaulting to `false` for missing rows /
    /// out-of-range indices (treated as rest).
    pub fn cell(&self, kind: TrackKind, i: usize) -> bool {
        self.tracks
            .get(&kind)
            .and_then(|row| row.get(i).copied())
            .unwrap_or(false)
    }

    /// Materialise the grid as drum events, one per active cell. The
    /// time origin is t = 0 at step 0; the caller can offset the
    /// returned events if they need a count-in.
    pub fn to_drum_events(&self) -> Vec<DrumEvent> {
        let mut out = Vec::new();
        let step = self.step_sec();
        let total = self.total_steps();
        for kind in TrackKind::ALL {
            let Some(row) = self.tracks.get(&kind) else {
                continue;
            };
            for (i, &on) in row.iter().enumerate() {
                if !on || i >= total {
                    continue;
                }
                out.push(DrumEvent {
                    start_sec: i as f32 * step,
                    kind: kind.as_char(),
                    velocity: 100,
                });
            }
        }
        // Sort by start time, then kick < snare < hihat for determinism.
        out.sort_by(|a, b| {
            a.start_sec
                .partial_cmp(&b.start_sec)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.kind.cmp(&b.kind))
        });
        out
    }

    /// Export pattern as a chiptune-format string. One character per
    /// step, bars separated by a `|`. Multiple drums on the same step
    /// fall back to the K > S > H precedence (chiptune syntax is
    /// monophonic per character).
    pub fn to_chiptune_string(&self) -> String {
        let total = self.total_steps();
        let spb = self.steps_per_bar as usize;
        let mut out = String::with_capacity(total + self.bars as usize);
        for i in 0..total {
            if i > 0 && spb > 0 && i % spb == 0 {
                out.push('|');
            }
            // K > S > H precedence so the round-trip is well-defined
            // when only one drum fires per step (the round-trip test
            // exercises exactly this case).
            let c = if self.cell(TrackKind::Kick, i) {
                'K'
            } else if self.cell(TrackKind::Snare, i) {
                'S'
            } else if self.cell(TrackKind::HiHat, i) {
                'H'
            } else {
                '.'
            };
            out.push(c);
        }
        out
    }

    /// Parse a chiptune-format string. Bars may be separated by `|`
    /// (ignored). Other whitespace is also skipped. The total number
    /// of step characters must equal `bars * 16`; otherwise the
    /// pattern is padded with rests / truncated to fit.
    ///
    /// `steps_per_bar` is fixed to 16 (the chiptune-lab convention).
    pub fn from_chiptune_string(s: &str, bpm: f32, bars: u32) -> Self {
        let steps_per_bar = 16_u32;
        let mut p = Pattern::empty(bpm, bars, steps_per_bar);
        let total = p.total_steps();
        let mut idx = 0_usize;
        for c in s.chars() {
            if c == '|' || c.is_whitespace() {
                continue;
            }
            if idx >= total {
                break;
            }
            if let Some(kind) = TrackKind::from_char(c) {
                if let Some(row) = p.tracks.get_mut(&kind) {
                    row[idx] = true;
                }
            }
            idx += 1;
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_kick_64_steps_emits_64_events_with_correct_spacing() {
        // 4 bars × 16 steps = 64 steps, all kick.
        let bpm = 120.0_f32;
        let bars = 4_u32;
        let mut p = Pattern::empty(bpm, bars, 16);
        for cell in p.tracks.get_mut(&TrackKind::Kick).unwrap() {
            *cell = true;
        }
        let evs = p.to_drum_events();
        assert_eq!(evs.len(), 64);
        // step at 120 bpm, 16 steps/bar = 0.5 bar = 2.0 sec / 16 = 0.125 sec.
        let expected_step = 2.0_f32 / 16.0;
        assert!((p.step_sec() - expected_step).abs() < 1e-6);
        for (i, ev) in evs.iter().enumerate() {
            assert_eq!(ev.kind, 'K');
            let want = i as f32 * expected_step;
            assert!(
                (ev.start_sec - want).abs() < 1e-4,
                "step {i}: got {} want {}",
                ev.start_sec,
                want,
            );
        }
    }

    #[test]
    fn chiptune_string_roundtrip() {
        // Single drum per step ⇒ string round-trip is loss-free.
        let s = "K...S...K...S...H.H.H.H.H.H.H.H.";
        let p = Pattern::from_chiptune_string(s, 100.0, 2);
        let back = p.to_chiptune_string();
        // The round-trip inserts a `|` between bars; strip it for
        // comparison since the input had none.
        let stripped: String = back.chars().filter(|c| *c != '|').collect();
        assert_eq!(stripped, s);

        // Re-parse the round-tripped (bar-separated) string and
        // compare cell-by-cell to ensure `|` is ignored on parse.
        let p2 = Pattern::from_chiptune_string(&back, 100.0, 2);
        for kind in TrackKind::ALL {
            for i in 0..p.total_steps() {
                assert_eq!(
                    p.cell(kind, i),
                    p2.cell(kind, i),
                    "mismatch at {kind:?}[{i}]",
                );
            }
        }
    }

    #[test]
    fn json_roundtrip_preserves_grid() {
        let mut p = Pattern::empty(140.0, 1, 16);
        p.toggle(TrackKind::Kick, 0);
        p.toggle(TrackKind::Snare, 4);
        p.toggle(TrackKind::HiHat, 2);
        let json = serde_json::to_string(&p).expect("serialize");
        let p2: Pattern = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(p.bpm, p2.bpm);
        assert_eq!(p.bars, p2.bars);
        assert_eq!(p.steps_per_bar, p2.steps_per_bar);
        for kind in TrackKind::ALL {
            for i in 0..p.total_steps() {
                assert_eq!(p.cell(kind, i), p2.cell(kind, i));
            }
        }
    }
}
