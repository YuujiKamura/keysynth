//! Sample-accurate MIDI event scheduler.
//!
//! Phase 1 SCHED component of the Jukebox Live migration. Replaces the
//! pre-render path (`render_midi` subprocess + WAV cache) with an
//! event source that the cpal output callback can drain in block-sized
//! chunks. See `JUKEBOX_LIVE_MIGRATION.md`.
//!
//! Design notes:
//! - SMF is parsed once at construction time. Tempo events are folded
//!   into per-event absolute sample offsets, so `advance` does no
//!   per-event time math.
//! - `events` is an immutable `Vec` after construction; `cursor` and
//!   `next_event_idx` are atomics. No `Mutex` is taken on the audio
//!   path.
//! - `advance` allocates a single small `Vec` (capacity 16) for the
//!   block's events. `panic!` / `unwrap` are forbidden outside
//!   `from_smf`.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use midly::{MetaMessage, MidiMessage, Smf, Timing, TrackEventKind};

/// One MIDI channel event (note on / off / panic).
///
/// `AllNotesOff` is synthesised by `stop()` and injected at the head of
/// the next `advance` block on every channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MidiEvent {
    pub channel: u8,
    pub kind: MidiEventKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MidiEventKind {
    NoteOn { note: u8, velocity: u8 },
    NoteOff { note: u8 },
    AllNotesOff,
}

/// Sample-accurate MIDI event source for the audio callback.
pub struct MidiSequencer {
    /// (absolute sample offset, event), sorted ascending by offset.
    /// Immutable after construction.
    events: Vec<(u64, MidiEvent)>,
    /// Current playback position in samples since the start of the SMF.
    cursor: AtomicU64,
    /// Smallest index `i` such that `events[i].0 >= cursor`. Maintained
    /// monotonically across `advance` calls; reset by `stop()` / `seek()`.
    next_event_idx: AtomicUsize,
    /// Total length of the rendered piece (last event + 1 s pad).
    total_samples: u64,
    #[allow(dead_code)]
    sample_rate: u32,
    /// `true` while `play()` has been called and `pause()` / `stop()`
    /// have not. `advance` is a no-op when `false`.
    playing: AtomicBool,
    /// Set by `stop()`; the next `advance` emits AllNotesOff on
    /// channels 0..16 before any other events.
    stop_pending: AtomicBool,
}

impl MidiSequencer {
    /// Parse a Standard MIDI File at `path` and produce a sequencer
    /// keyed in samples at `sample_rate`. Tempo meta events are folded
    /// into the absolute sample offset of each note event so the audio
    /// callback never has to interpret tempo.
    pub fn from_smf(path: &Path, sample_rate: u32) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let smf = Smf::parse(&bytes).map_err(|e| format!("midly parse: {e}"))?;
        let ppq: u32 = match smf.header.timing {
            Timing::Metrical(t) => t.as_int() as u32,
            Timing::Timecode(_, _) => {
                return Err("SMPTE timecode timing not supported".to_string())
            }
        };
        if ppq == 0 {
            return Err("PPQ is zero".to_string());
        }

        // Merged stream of (absolute_tick, raw event) across all tracks.
        // Note: we keep tempo meta and channel note events; everything
        // else is dropped. `velocity == 0` NoteOn is canonicalised to
        // NoteOff per the MIDI spec.
        #[derive(Clone, Copy)]
        enum Raw {
            On { ch: u8, note: u8, vel: u8 },
            Off { ch: u8, note: u8 },
            Tempo { us_per_q: u32 },
        }
        let mut raw: Vec<(u64, Raw)> = Vec::new();
        for track in smf.tracks.iter() {
            let mut tick: u64 = 0;
            for ev in track {
                tick += ev.delta.as_int() as u64;
                match ev.kind {
                    TrackEventKind::Midi { channel, message } => match message {
                        MidiMessage::NoteOn { key, vel } => {
                            let v = vel.as_int();
                            if v > 0 {
                                raw.push((
                                    tick,
                                    Raw::On {
                                        ch: channel.as_int(),
                                        note: key.as_int(),
                                        vel: v,
                                    },
                                ));
                            } else {
                                raw.push((
                                    tick,
                                    Raw::Off {
                                        ch: channel.as_int(),
                                        note: key.as_int(),
                                    },
                                ));
                            }
                        }
                        MidiMessage::NoteOff { key, .. } => {
                            raw.push((
                                tick,
                                Raw::Off {
                                    ch: channel.as_int(),
                                    note: key.as_int(),
                                },
                            ));
                        }
                        _ => {}
                    },
                    TrackEventKind::Meta(MetaMessage::Tempo(us_per_q)) => {
                        raw.push((
                            tick,
                            Raw::Tempo {
                                us_per_q: us_per_q.as_int(),
                            },
                        ));
                    }
                    _ => {}
                }
            }
        }
        // Stable sort by absolute tick so tempo events at the same
        // tick as a note are processed first (they sort equal but we
        // walk in input order — close enough; tempo at exactly the
        // same tick as a note still applies because we update
        // `us_per_q` before computing the note's sample offset only
        // when the tempo event is encountered first in the merged
        // stream, which is good enough for typical SMF authoring).
        raw.sort_by_key(|(t, _)| *t);

        // Walk the merged stream, maintain current tempo, convert
        // (tick → microseconds → samples). For each note event, push
        // a (sample_offset, MidiEvent) pair into `events`.
        let mut us_per_q: u64 = 500_000; // default 120 BPM
        let mut last_tick: u64 = 0;
        let mut now_us: f64 = 0.0;
        let mut events: Vec<(u64, MidiEvent)> = Vec::with_capacity(raw.len());
        let sr = sample_rate as f64;
        for (tick, ev) in &raw {
            if *tick > last_tick {
                let delta_ticks = (*tick - last_tick) as f64;
                // sec_per_tick = us_per_q / (ppq * 1e6); accumulate in us.
                let delta_us = delta_ticks * (us_per_q as f64) / (ppq as f64);
                now_us += delta_us;
                last_tick = *tick;
            }
            match *ev {
                Raw::Tempo { us_per_q: t } => {
                    us_per_q = t as u64;
                }
                Raw::On { ch, note, vel } => {
                    let sample = (now_us * sr / 1_000_000.0) as u64;
                    events.push((
                        sample,
                        MidiEvent {
                            channel: ch,
                            kind: MidiEventKind::NoteOn { note, velocity: vel },
                        },
                    ));
                }
                Raw::Off { ch, note } => {
                    let sample = (now_us * sr / 1_000_000.0) as u64;
                    events.push((
                        sample,
                        MidiEvent {
                            channel: ch,
                            kind: MidiEventKind::NoteOff { note },
                        },
                    ));
                }
            }
        }
        // Stable sort by sample offset (within a tick the tempo /
        // note ordering is preserved because we only push notes here).
        events.sort_by_key(|(s, _)| *s);
        let last_sample = events.last().map(|(s, _)| *s).unwrap_or(0);
        // Pad 1 second so the caller can detect "track ended" by
        // `cursor >= total_samples` while voices ring out.
        let total_samples = last_sample.saturating_add(sample_rate as u64);

        Ok(Self {
            events,
            cursor: AtomicU64::new(0),
            next_event_idx: AtomicUsize::new(0),
            total_samples,
            sample_rate,
            playing: AtomicBool::new(false),
            stop_pending: AtomicBool::new(false),
        })
    }

    /// Build a sequencer from a pre-baked event list (used by tests
    /// and any future caller that wants to drive the scheduler from
    /// something other than an SMF). Events are sorted on entry.
    pub fn from_events(
        mut events: Vec<(u64, MidiEvent)>,
        sample_rate: u32,
        total_samples: u64,
    ) -> Self {
        events.sort_by_key(|(s, _)| *s);
        Self {
            events,
            cursor: AtomicU64::new(0),
            next_event_idx: AtomicUsize::new(0),
            total_samples,
            sample_rate,
            playing: AtomicBool::new(false),
            stop_pending: AtomicBool::new(false),
        }
    }

    pub fn play(&self) {
        self.playing.store(true, Ordering::Release);
    }

    pub fn pause(&self) {
        self.playing.store(false, Ordering::Release);
    }

    /// Stop playback, rewind cursor to 0, and arm a one-shot
    /// AllNotesOff burst (16 channels) to be emitted at sample-offset
    /// 0 on the next `advance` call. Voices held by the caller
    /// (Player layer) should also drop their state on stop, but the
    /// burst gives a clean fallback.
    pub fn stop(&self) {
        self.playing.store(false, Ordering::Release);
        self.cursor.store(0, Ordering::Release);
        self.next_event_idx.store(0, Ordering::Release);
        self.stop_pending.store(true, Ordering::Release);
    }

    /// Move the cursor to `samples`. Note state is the caller's
    /// responsibility — typical Player usage is `seq.seek(s);
    /// pool.all_notes_off();`. We re-base `next_event_idx` here so
    /// the next `advance` skips events earlier than `samples`.
    pub fn seek(&self, samples: u64) {
        let clamped = samples.min(self.total_samples);
        self.cursor.store(clamped, Ordering::Release);
        // Re-binary-search for the first event >= clamped.
        let idx = self
            .events
            .binary_search_by(|probe| probe.0.cmp(&clamped))
            .unwrap_or_else(|i| i);
        self.next_event_idx.store(idx, Ordering::Release);
    }

    pub fn cursor(&self) -> u64 {
        self.cursor.load(Ordering::Acquire)
    }

    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    pub fn is_playing(&self) -> bool {
        self.playing.load(Ordering::Acquire)
    }

    /// Advance the cursor by `num_samples` and return every event that
    /// fires during this block, paired with its block-local sample
    /// offset (`absolute_sample - cursor_at_block_start`).
    ///
    /// Audio callback contract:
    /// - allocates one small `Vec` (capacity 16) — no Mutex, no panic.
    /// - paused → cursor unchanged, returns empty Vec.
    /// - `stop()` was called → first event in the returned Vec is
    ///   AllNotesOff on every channel 0..16 (offset 0), followed by
    ///   any events that fire from the new cursor onward.
    pub fn advance(&self, num_samples: u32) -> Vec<(u32, MidiEvent)> {
        let mut out: Vec<(u32, MidiEvent)> = Vec::with_capacity(16);

        // Drain a pending stop before checking `playing`: stop()
        // unsets playing, and we still want the AllNotesOff burst on
        // the next callback after stop, even if play() has not been
        // called again.
        if self
            .stop_pending
            .swap(false, Ordering::AcqRel)
        {
            for ch in 0u8..16 {
                out.push((
                    0,
                    MidiEvent {
                        channel: ch,
                        kind: MidiEventKind::AllNotesOff,
                    },
                ));
            }
        }

        if !self.playing.load(Ordering::Acquire) {
            return out;
        }

        let n = num_samples as u64;
        if n == 0 {
            return out;
        }
        let cursor_start = self.cursor.load(Ordering::Acquire);
        let cursor_end = cursor_start.saturating_add(n);
        // Store the new cursor before draining events so a concurrent
        // `cursor()` reader sees a consistent advance.
        self.cursor.store(cursor_end, Ordering::Release);

        let mut idx = self.next_event_idx.load(Ordering::Acquire);
        let len = self.events.len();
        while idx < len {
            let (abs, ev) = self.events[idx];
            if abs >= cursor_end {
                break;
            }
            // Clamp to the start of the block in case the cursor was
            // set past an event by a prior seek that landed mid-tick.
            let local = abs.saturating_sub(cursor_start);
            // num_samples fits in u32 by contract; clamp just in case.
            let local32 = if local > u32::MAX as u64 {
                u32::MAX
            } else {
                local as u32
            };
            out.push((local32, ev));
            idx += 1;
        }
        self.next_event_idx.store(idx, Ordering::Release);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_smf() -> PathBuf {
        // bench-out/songs/alice.mid is a small format-1 MIDI file
        // checked into the repo. We pick it because it has tempo
        // events and multi-track voicing, exercising the parse path.
        let p = PathBuf::from("bench-out/songs/alice.mid");
        assert!(
            p.exists(),
            "test fixture missing: {} (run from repo root)",
            p.display()
        );
        p
    }

    #[test]
    fn from_smf_loads_simple_track() {
        let seq = MidiSequencer::from_smf(&fixture_smf(), 44100)
            .expect("alice.mid should parse");
        assert!(seq.total_samples() > 0, "total_samples must be > 0");
        // Should have at least one note event.
        assert!(!seq.events.is_empty(), "events vec should not be empty");
        // Cursor starts at 0, not playing.
        assert_eq!(seq.cursor(), 0);
        assert!(!seq.is_playing());
    }

    #[test]
    fn advance_advances_cursor() {
        // Empty event list, just verify cursor math.
        let seq = MidiSequencer::from_events(Vec::new(), 44100, 44100 * 10);
        seq.play();
        // 100 ms at 44.1 kHz = 4410 samples.
        let evs = seq.advance(4410);
        assert!(evs.is_empty());
        assert_eq!(seq.cursor(), 4410);
        // Another 100 ms → 8820.
        let _ = seq.advance(4410);
        assert_eq!(seq.cursor(), 8820);
    }

    #[test]
    fn advance_yields_events_in_window() {
        // Hand-rolled events: NoteOn at 0, NoteOff at 22050 (0.5 s),
        // NoteOn at 44100 (1.0 s).
        let events = vec![
            (
                0u64,
                MidiEvent {
                    channel: 0,
                    kind: MidiEventKind::NoteOn { note: 60, velocity: 100 },
                },
            ),
            (
                22050u64,
                MidiEvent {
                    channel: 0,
                    kind: MidiEventKind::NoteOff { note: 60 },
                },
            ),
            (
                44100u64,
                MidiEvent {
                    channel: 0,
                    kind: MidiEventKind::NoteOn { note: 64, velocity: 100 },
                },
            ),
        ];
        let seq = MidiSequencer::from_events(events, 44100, 88200);
        seq.play();
        // Advance 1 second. Should pop the first two events; the
        // 44100-th sample is exactly cursor_end (exclusive), so it
        // should NOT pop yet.
        let evs = seq.advance(44100);
        assert_eq!(evs.len(), 2, "got events: {evs:?}");
        assert_eq!(evs[0].0, 0);
        assert!(matches!(
            evs[0].1.kind,
            MidiEventKind::NoteOn { note: 60, .. }
        ));
        assert_eq!(evs[1].0, 22050);
        assert!(matches!(evs[1].1.kind, MidiEventKind::NoteOff { note: 60 }));
        // Next 1-sample advance picks up the 44100-th event.
        let evs2 = seq.advance(1);
        assert_eq!(evs2.len(), 1);
        assert_eq!(evs2[0].0, 0);
        assert!(matches!(
            evs2[0].1.kind,
            MidiEventKind::NoteOn { note: 64, .. }
        ));
    }

    #[test]
    fn pause_freezes_cursor() {
        let events = vec![(
            1000u64,
            MidiEvent {
                channel: 0,
                kind: MidiEventKind::NoteOn { note: 60, velocity: 100 },
            },
        )];
        let seq = MidiSequencer::from_events(events, 44100, 44100);
        // Not playing yet → advance is a no-op.
        let evs = seq.advance(4410);
        assert!(evs.is_empty());
        assert_eq!(seq.cursor(), 0, "cursor must not move while paused");
        // pause() after a play → still freezes.
        seq.play();
        seq.pause();
        let evs = seq.advance(4410);
        assert!(evs.is_empty());
        assert_eq!(seq.cursor(), 0);
    }

    #[test]
    fn stop_resets_and_emits_all_notes_off() {
        let events = vec![(
            1000u64,
            MidiEvent {
                channel: 0,
                kind: MidiEventKind::NoteOn { note: 60, velocity: 100 },
            },
        )];
        let seq = MidiSequencer::from_events(events, 44100, 44100);
        seq.play();
        let _ = seq.advance(2000); // consume the note-on, cursor → 2000
        seq.stop();
        assert_eq!(seq.cursor(), 0, "stop must rewind cursor");
        // Re-arm playback so advance actually runs.
        seq.play();
        let evs = seq.advance(512);
        // First 16 entries should be AllNotesOff, one per channel.
        let off_count = evs
            .iter()
            .filter(|(_, e)| matches!(e.kind, MidiEventKind::AllNotesOff))
            .count();
        assert_eq!(
            off_count, 16,
            "expected 16 AllNotesOff events after stop, got {evs:?}"
        );
        let channels: std::collections::HashSet<u8> = evs
            .iter()
            .filter_map(|(_, e)| match e.kind {
                MidiEventKind::AllNotesOff => Some(e.channel),
                _ => None,
            })
            .collect();
        for ch in 0u8..16 {
            assert!(
                channels.contains(&ch),
                "AllNotesOff missing for channel {ch}"
            );
        }
    }

    #[test]
    fn seek_repositions_event_cursor() {
        let events = vec![
            (
                100u64,
                MidiEvent {
                    channel: 0,
                    kind: MidiEventKind::NoteOn { note: 60, velocity: 100 },
                },
            ),
            (
                500u64,
                MidiEvent {
                    channel: 0,
                    kind: MidiEventKind::NoteOff { note: 60 },
                },
            ),
        ];
        let seq = MidiSequencer::from_events(events, 44100, 44100);
        seq.seek(400);
        assert_eq!(seq.cursor(), 400);
        seq.play();
        // From 400, a 200-sample advance hits the off at 500 only.
        let evs = seq.advance(200);
        assert_eq!(evs.len(), 1);
        assert!(matches!(evs[0].1.kind, MidiEventKind::NoteOff { note: 60 }));
    }
}
