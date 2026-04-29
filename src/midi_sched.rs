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
//! - `events` is an immutable `Vec` after construction; `cursor`,
//!   `next_event_idx` and `seek_target` are atomics. No `Mutex` is
//!   taken on the audio path.
//! - `advance` does NOT allocate; the caller passes a reusable `Vec`
//!   that we `clear()` then `push` into. `panic!` / `unwrap` are
//!   forbidden outside `from_smf`.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use midly::{MetaMessage, MidiMessage, Smf, Timing, TrackEventKind};

/// Sentinel value for `seek_target` meaning "no pending seek". We use
/// `u64::MAX` because real seek targets are clamped to `total_samples`
/// which is bounded by the MIDI file length (always far below MAX).
/// A real-world SMF would need to be roughly 10^14 years long at
/// 44.1 kHz to collide with this sentinel, so the only collision risk
/// is `total_samples == u64::MAX`, which we never construct (`from_smf`
/// derives `total_samples` from the last event sample plus a 1 s pad).
const SEEK_NONE: u64 = u64::MAX;

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
    // TODO(midi-sched): add ControlChange { number, value } to support
    // sustain pedal (CC64), pitch bend etc. (deferred from initial
    // review — out of Phase 1 scope).
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
    /// Pending seek target in samples, or `SEEK_NONE` if no seek is
    /// pending. `seek()` only writes here; the actual `cursor` /
    /// `next_event_idx` rebase is performed atomically at the head of
    /// the next `advance()` so an audio-thread reader never observes
    /// half-applied seek state. A pending seek also injects an
    /// AllNotesOff burst at offset 0 of that block (the caller / Player
    /// is responsible for resetting voice state).
    seek_target: AtomicU64,
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
        impl Raw {
            /// Tie-break order at identical ticks: Tempo applies first
            /// (so notes use the new BPM), then NoteOff (release before
            /// retrigger to avoid stuck voices), then NoteOn.
            fn priority(&self) -> u8 {
                match self {
                    Raw::Tempo { .. } => 0,
                    Raw::Off { .. } => 1,
                    Raw::On { .. } => 2,
                }
            }
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
        // Sort by (absolute tick, then explicit priority). Tempo wins
        // at a tied tick so the note that follows uses the new BPM;
        // NoteOff wins over NoteOn at the same tick to avoid a voice
        // pile-up if a sequencer re-triggers the same key. `sort_by`
        // is stable, so events of equal priority keep author order.
        raw.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.priority().cmp(&b.1.priority())));

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
        // Stable sort by sample offset. The pre-sort above already
        // arranged Tempo > NoteOff > NoteOn at identical ticks, and
        // ties in sample offset (after tick→sample conversion)
        // preserve that order because `sort_by_key` is stable.
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
            seek_target: AtomicU64::new(SEEK_NONE),
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
            seek_target: AtomicU64::new(SEEK_NONE),
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
        // A pending seek would otherwise overwrite the rewind on the
        // next advance(), so cancel it.
        self.seek_target.store(SEEK_NONE, Ordering::Release);
        self.stop_pending.store(true, Ordering::Release);
    }

    /// Request a seek to `samples`. The actual `cursor` and
    /// `next_event_idx` rebase happens at the head of the next
    /// `advance()`, atomically with respect to the audio thread —
    /// callers (control thread) only post intent here so an
    /// `advance()` reader can never observe a half-applied seek
    /// (cursor updated, idx not yet, or vice-versa).
    ///
    /// A pending seek also injects an `AllNotesOff` burst at offset 0
    /// of the block in which it lands. Note state outside the
    /// scheduler is the caller's responsibility (typical Player usage
    /// is `seq.seek(s); pool.all_notes_off();`).
    pub fn seek(&self, samples: u64) {
        let clamped = samples.min(self.total_samples);
        // SEEK_NONE collides with the saturating max only if
        // total_samples == u64::MAX, which we never produce; the
        // saturating_add of `sample_rate as u64` to a SMF-derived
        // last_sample is bounded by the file length.
        debug_assert!(clamped != SEEK_NONE);
        self.seek_target.store(clamped, Ordering::Release);
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

    /// Advance the cursor by `num_samples` and append every event
    /// that fires during this block to `out`, paired with its
    /// block-local sample offset (`absolute_sample -
    /// cursor_at_block_start`).
    ///
    /// `out` is **cleared at entry** and reused — the caller (Player
    /// layer) is expected to keep a single buffer and pass it on
    /// every callback so the audio thread does not allocate. As long
    /// as the buffer's `capacity()` is not exceeded, no allocation
    /// occurs (push into spare capacity is amortised O(1)).
    ///
    /// Audio callback contract:
    /// - no allocation on the steady-state path (provided
    ///   `out.capacity()` is large enough), no Mutex, no panic.
    /// - paused → cursor unchanged, `out` left empty.
    /// - `stop()` was called → `out` starts with AllNotesOff on every
    ///   channel 0..16 (offset 0), followed by any events that fire
    ///   from the new cursor onward.
    /// - a pending `seek()` is applied atomically at the head of this
    ///   block (cursor + next_event_idx are rebased before we drain
    ///   events) and prepends an AllNotesOff burst to `out` so any
    ///   notes the caller did not already release get cut.
    pub fn advance(&self, num_samples: u32, out: &mut Vec<(u32, MidiEvent)>) {
        out.clear();

        // Drain a pending stop before checking `playing`: stop()
        // unsets playing, and we still want the AllNotesOff burst on
        // the next callback after stop, even if play() has not been
        // called again.
        if self.stop_pending.swap(false, Ordering::AcqRel) {
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

        // Apply any pending seek atomically with respect to other
        // audio-thread state. swap() consumes the request so a
        // concurrent seek() that lands after this point will be
        // serviced on the next advance(). Because cursor and
        // next_event_idx are rebased here (not in seek()), an
        // audio-thread reader can never observe a half-applied seek.
        let pending_seek = self.seek_target.swap(SEEK_NONE, Ordering::AcqRel);
        if pending_seek != SEEK_NONE {
            self.cursor.store(pending_seek, Ordering::Release);
            let idx = self
                .events
                .binary_search_by(|probe| probe.0.cmp(&pending_seek))
                .unwrap_or_else(|i| i);
            self.next_event_idx.store(idx, Ordering::Release);
            // Cut any voices the caller did not release — same burst
            // as stop(), at offset 0 of this block.
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
            return;
        }

        let n = num_samples as u64;
        if n == 0 {
            return;
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
        let mut out = Vec::with_capacity(16);
        // 100 ms at 44.1 kHz = 4410 samples.
        seq.advance(4410, &mut out);
        assert!(out.is_empty());
        assert_eq!(seq.cursor(), 4410);
        // Another 100 ms → 8820.
        seq.advance(4410, &mut out);
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
        let mut out = Vec::with_capacity(16);
        // Advance 1 second. Should pop the first two events; the
        // 44100-th sample is exactly cursor_end (exclusive), so it
        // should NOT pop yet.
        seq.advance(44100, &mut out);
        assert_eq!(out.len(), 2, "got events: {out:?}");
        assert_eq!(out[0].0, 0);
        assert!(matches!(
            out[0].1.kind,
            MidiEventKind::NoteOn { note: 60, .. }
        ));
        assert_eq!(out[1].0, 22050);
        assert!(matches!(out[1].1.kind, MidiEventKind::NoteOff { note: 60 }));
        // Next 1-sample advance picks up the 44100-th event.
        seq.advance(1, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, 0);
        assert!(matches!(
            out[0].1.kind,
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
        let mut out = Vec::with_capacity(16);
        // Not playing yet → advance is a no-op.
        seq.advance(4410, &mut out);
        assert!(out.is_empty());
        assert_eq!(seq.cursor(), 0, "cursor must not move while paused");
        // pause() after a play → still freezes.
        seq.play();
        seq.pause();
        seq.advance(4410, &mut out);
        assert!(out.is_empty());
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
        let mut out = Vec::with_capacity(32);
        seq.advance(2000, &mut out); // consume the note-on, cursor → 2000
        seq.stop();
        assert_eq!(seq.cursor(), 0, "stop must rewind cursor");
        // Re-arm playback so advance actually runs.
        seq.play();
        seq.advance(512, &mut out);
        // First 16 entries should be AllNotesOff, one per channel.
        let off_count = out
            .iter()
            .filter(|(_, e)| matches!(e.kind, MidiEventKind::AllNotesOff))
            .count();
        assert_eq!(
            off_count, 16,
            "expected 16 AllNotesOff events after stop, got {out:?}"
        );
        let channels: std::collections::HashSet<u8> = out
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
        // seek() is now deferred until the next advance(); this test
        // validates the post-advance state.
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
        // seek() does not touch cursor directly; it stores intent.
        assert_eq!(seq.cursor(), 0, "cursor must not move until advance()");
        seq.play();
        // First advance applies the seek and emits the AllNotesOff
        // burst; from 400, a 200-sample advance hits the off at 500.
        let mut out = Vec::with_capacity(32);
        seq.advance(200, &mut out);
        assert_eq!(seq.cursor(), 600);
        // 16 AllNotesOff (one per channel) + 1 NoteOff.
        let off_count = out
            .iter()
            .filter(|(_, e)| matches!(e.kind, MidiEventKind::AllNotesOff))
            .count();
        assert_eq!(off_count, 16, "expected 16 AllNotesOff after seek");
        let note_offs: Vec<_> = out
            .iter()
            .filter(|(_, e)| matches!(e.kind, MidiEventKind::NoteOff { .. }))
            .collect();
        assert_eq!(note_offs.len(), 1);
        assert!(matches!(
            note_offs[0].1.kind,
            MidiEventKind::NoteOff { note: 60 }
        ));
    }

    #[test]
    fn seek_during_advance_does_not_burst() {
        // Race scenario the old (cursor-then-idx) seek() exposed: an
        // advance() running between the cursor store and the idx
        // store would replay every event from the old idx up to the
        // new cursor at local_offset = 0. With the deferred-seek
        // design the worst case is "advance happened before seek
        // landed" → seek is serviced on the *next* advance.
        //
        // Determinism check: cursor=0 → advance(N) consumes the early
        // events → seek(target) → next advance starts with the
        // 16-channel AllNotesOff burst at offset 0 and only emits
        // events whose absolute sample is >= target.
        let mut events: Vec<(u64, MidiEvent)> = Vec::new();
        // Pack 50 NoteOns evenly across [0, 5000).
        for i in 0..50u64 {
            events.push((
                i * 100,
                MidiEvent {
                    channel: 0,
                    kind: MidiEventKind::NoteOn {
                        note: 60 + (i as u8 % 12),
                        velocity: 100,
                    },
                },
            ));
        }
        let seq = MidiSequencer::from_events(events, 44100, 10_000);
        seq.play();
        let mut out = Vec::with_capacity(64);

        // First block consumes events at 0..1000 (10 events).
        seq.advance(1000, &mut out);
        let pre_seek_count = out
            .iter()
            .filter(|(_, e)| matches!(e.kind, MidiEventKind::NoteOn { .. }))
            .count();
        assert_eq!(pre_seek_count, 10, "first block should hold 10 NoteOns");

        // Seek to 4500.
        seq.seek(4500);
        // cursor() still reads the old position until advance().
        assert_eq!(seq.cursor(), 1000);

        // Second block of 1000 samples after seek lands at [4500,
        // 5500). Only the events at 4500..5000 (i.e. ticks 45..49,
        // 5 NoteOns) should fire. NONE of the 10..44 events should
        // appear at local_offset 0 — which is exactly the regression
        // the old seek would have produced.
        seq.advance(1000, &mut out);
        let off_count = out
            .iter()
            .filter(|(_, e)| matches!(e.kind, MidiEventKind::AllNotesOff))
            .count();
        assert_eq!(off_count, 16, "seek must inject AllNotesOff burst");
        let note_ons: Vec<_> = out
            .iter()
            .filter(|(_, e)| matches!(e.kind, MidiEventKind::NoteOn { .. }))
            .collect();
        assert_eq!(
            note_ons.len(),
            5,
            "post-seek block should hold exactly the 5 NoteOns at \
             4500..5000, got {note_ons:?}"
        );
        // local_offsets must reflect the post-seek cursor (4500),
        // not zero. The first NoteOn is at 4500 → local 0; that is
        // legitimate. The other four MUST have local > 0.
        let zero_local = note_ons.iter().filter(|(o, _)| *o == 0).count();
        assert_eq!(
            zero_local, 1,
            "only the event exactly at the seek target should land \
             at local_offset 0; others must be spread, got {note_ons:?}"
        );
        assert_eq!(seq.cursor(), 5500);
    }

    #[test]
    fn advance_reuses_caller_vec() {
        let events = vec![(
            100u64,
            MidiEvent {
                channel: 0,
                kind: MidiEventKind::NoteOn { note: 60, velocity: 100 },
            },
        )];
        let seq = MidiSequencer::from_events(events, 44100, 44100);
        seq.play();

        // Pre-fill out with stale data; advance() must clear() it
        // before pushing.
        let mut out: Vec<(u32, MidiEvent)> = Vec::with_capacity(16);
        for _ in 0..5 {
            out.push((
                999,
                MidiEvent {
                    channel: 9,
                    kind: MidiEventKind::AllNotesOff,
                },
            ));
        }
        let cap_before = out.capacity();

        seq.advance(200, &mut out);
        // Stale entries must be gone.
        assert!(
            out.iter().all(|(o, _)| *o != 999),
            "advance() must clear stale entries, got {out:?}"
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, 100);

        // Second call on the same Vec — must not retain previous
        // events and must not have grown capacity (single small push
        // fits in 16-cap buffer).
        seq.advance(200, &mut out);
        assert!(out.is_empty(), "no events in 200..400 window");
        let cap_after = out.capacity();
        assert_eq!(
            cap_before, cap_after,
            "capacity must not grow for steady-state push counts"
        );
    }
}
