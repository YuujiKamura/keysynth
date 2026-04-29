//! Polyphonic voice pool with per-channel routing and live factory flip.
//!
//! This module is the audio-callback-side substrate for the Jukebox Live
//! Migration (see `JUKEBOX_LIVE_MIGRATION.md`). It owns up to `voice_cap`
//! simultaneously sounding voices and routes incoming MIDI events to
//! per-channel `VoiceFactory` slots so the UI can hot-swap timbres
//! between notes without interrupting voices that are already ringing.
//!
//! Invariants (audio-thread safe):
//!   - `process_block` does **not** allocate. Voices are summed in place
//!     into the caller-provided buffer; finished voices are removed via
//!     `Vec::retain` (in-place, no realloc on shrink).
//!   - `dispatch(NoteOn)` may allocate inside `make_voice`; the audio
//!     thread tolerates this because note-on rate is bounded by the MIDI
//!     event stream (≪ block rate).
//!   - No `panic!` / `unwrap` / `expect`. Poisoned locks are recovered
//!     with `into_inner` so a single misbehaving caller can't take down
//!     the audio thread.
//!   - `set_channel_factory` is the only writer of the channel routing
//!     table; everything else takes a read lock.
//!
//! TODO(integrate): unify `MidiEvent`/`MidiEventKind` with
//! `crate::midi_sched` once the SCHED branch lands. The duplicated
//! definitions below are a deliberate placeholder so this file builds
//! and tests in isolation; the hub will replace them with a `pub use`
//! re-export at integration time.

use std::sync::{Arc, Mutex, RwLock};

use crate::synth::VoiceImpl;
use crate::voice_lib::DecayModel;

// ---------------------------------------------------------------------------
// MidiEvent (placeholder; see TODO(integrate) above)
// ---------------------------------------------------------------------------

/// One MIDI event tagged with its destination channel. The pool is the
/// authority over how each channel maps to a `VoiceFactory`; the
/// scheduler only emits these structs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidiEvent {
    pub channel: u8,
    pub kind: MidiEventKind,
}

/// Subset of the MIDI 1.0 wire protocol that the pool understands.
/// Velocity is 0..=127 (raw MIDI byte); the pool normalises it to
/// 0.0..=1.0 before handing it to the factory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MidiEventKind {
    NoteOn { note: u8, velocity: u8 },
    NoteOff { note: u8 },
    /// Trigger release on every voice routed to this channel.
    AllNotesOff,
}

// ---------------------------------------------------------------------------
// VoiceFactory
// ---------------------------------------------------------------------------

/// Stamps fresh `VoiceImpl`s on demand. One factory instance is shared
/// across many notes (typically one per loaded cdylib / engine), so the
/// trait is `Send + Sync` and consumed via `Arc<dyn VoiceFactory>`.
pub trait VoiceFactory: Send + Sync {
    /// Build one voice for the given pitch / velocity. `velocity` is
    /// pre-normalised to 0.0..=1.0 by the pool.
    fn make_voice(
        &self,
        sample_rate: f32,
        freq: f32,
        velocity: f32,
    ) -> Box<dyn VoiceImpl + Send>;

    /// Human-readable factory id (used by the UI dropdown and by
    /// `channel_factory_name`).
    fn name(&self) -> &str;

    /// How NoteOff should be handled for voices this factory produces.
    /// `Damper` calls `trigger_release()`; `Natural` lets the voice ring
    /// out via its own physics.
    fn decay_model(&self) -> DecayModel;
}

// ---------------------------------------------------------------------------
// ActiveVoice
// ---------------------------------------------------------------------------

/// A voice currently sounding inside the pool. `decay_model` is captured
/// at note-on time from the factory so a later `set_channel_factory` does
/// not change how this voice responds to its eventual NoteOff.
struct ActiveVoice {
    channel: u8,
    note: u8,
    voice: Box<dyn VoiceImpl + Send>,
    decay_model: DecayModel,
    /// Sample-time at note-on. Currently always 0 (the pool has no
    /// global clock yet); reserved for the SCHED-driven sample-accurate
    /// version where eviction wants the truly-oldest voice.
    #[allow(dead_code)]
    started_at: u64,
}

// ---------------------------------------------------------------------------
// VoicePool
// ---------------------------------------------------------------------------

const NUM_CHANNELS: usize = 16;

/// Polyphonic voice pool. See module docs for invariants.
pub struct VoicePool {
    sample_rate: f32,
    voice_cap: usize,
    voices: Mutex<Vec<ActiveVoice>>,
    channel_factories: RwLock<[Arc<dyn VoiceFactory>; NUM_CHANNELS]>,
}

impl VoicePool {
    /// Build a pool of capacity `voice_cap`, with every channel pointing
    /// at `default` initially. The UI can override individual channels
    /// later via `set_channel_factory`.
    pub fn new(sample_rate: f32, voice_cap: usize, default: Arc<dyn VoiceFactory>) -> Self {
        // `array::from_fn` keeps the array on the stack and avoids needing
        // `Clone` on the trait object itself.
        let factories: [Arc<dyn VoiceFactory>; NUM_CHANNELS] =
            std::array::from_fn(|_| Arc::clone(&default));
        Self {
            sample_rate,
            voice_cap,
            voices: Mutex::new(Vec::with_capacity(voice_cap)),
            channel_factories: RwLock::new(factories),
        }
    }

    /// Hot-swap the factory bound to `channel`. Voices already sounding
    /// keep their original engine; only NoteOns received after this call
    /// pick up the new factory. Out-of-range channels (>= 16) are
    /// silently ignored — MIDI status bytes are masked to 0..=15 by the
    /// scheduler but defensive-coding against bad callers is cheap.
    pub fn set_channel_factory(&self, channel: u8, factory: Arc<dyn VoiceFactory>) {
        let idx = channel as usize;
        if idx >= NUM_CHANNELS {
            return;
        }
        let mut guard = match self.channel_factories.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard[idx] = factory;
    }

    /// Name of the factory currently bound to `channel`. Returns an
    /// empty string if `channel` is out of range so callers don't have
    /// to handle a `Result` for the common UI-label case.
    pub fn channel_factory_name(&self, channel: u8) -> String {
        let idx = channel as usize;
        if idx >= NUM_CHANNELS {
            return String::new();
        }
        let guard = match self.channel_factories.read() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard[idx].name().to_string()
    }

    /// Route one MIDI event to the pool. Safe to call from the audio
    /// thread; only blocks on its own `Mutex` (no contention in
    /// practice — the UI uses `set_channel_factory`, not `dispatch`).
    pub fn dispatch(&self, ev: &MidiEvent) {
        match ev.kind {
            MidiEventKind::NoteOn { note, velocity } => {
                self.dispatch_note_on(ev.channel, note, velocity);
            }
            MidiEventKind::NoteOff { note } => {
                self.dispatch_note_off(ev.channel, note);
            }
            MidiEventKind::AllNotesOff => {
                self.dispatch_all_notes_off(ev.channel);
            }
        }
    }

    fn dispatch_note_on(&self, channel: u8, note: u8, velocity: u8) {
        if channel as usize >= NUM_CHANNELS {
            return;
        }
        // A pool with zero capacity can never hold a voice; bail out
        // before we touch the channel-factory map or the voices Mutex
        // (the eviction branch below would otherwise call
        // `voices.remove(0)` on an empty Vec and panic).
        if self.voice_cap == 0 {
            return;
        }
        // Pull factory + decay model out under the read lock first so we
        // never hold the channel lock and the voices lock at the same
        // time (lock-ordering hygiene; matters if a future caller takes
        // them in the opposite order).
        let (factory, decay_model) = {
            let guard = match self.channel_factories.read() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            let f = Arc::clone(&guard[channel as usize]);
            let dm = f.decay_model();
            (f, dm)
        };

        let freq = crate::synth::midi_to_freq(note);
        let vel_norm = (velocity as f32) / 127.0;
        let voice = factory.make_voice(self.sample_rate, freq, vel_norm);

        let mut voices = match self.voices.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };

        // Eviction: at cap, drop a releasing voice first (chord_pad
        // pattern — those are already on their way out and stealing
        // them is inaudible), else the oldest sustained voice.
        if voices.len() >= self.voice_cap {
            let releasing_idx = voices.iter().position(|v| v.voice.is_releasing());
            let victim = releasing_idx.unwrap_or(0);
            voices.remove(victim);
        }

        voices.push(ActiveVoice {
            channel,
            note,
            voice,
            decay_model,
            started_at: 0,
        });
    }

    fn dispatch_note_off(&self, channel: u8, note: u8) {
        let mut voices = match self.voices.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        for v in voices.iter_mut() {
            if v.channel == channel
                && v.note == note
                && !v.voice.is_releasing()
            {
                match v.decay_model {
                    DecayModel::Damper => {
                        v.voice.trigger_release();
                    }
                    DecayModel::Natural => {
                        // Plucked-string semantics: NoteOff has no
                        // physical analogue. Let the voice ring out
                        // and rely on `process_block`'s retain step
                        // to cull it once `is_done()` flips.
                    }
                }
                // Only release the first matching voice — repeated
                // (channel, note) pairs are legitimate (re-triggered
                // notes before the previous one finished), and a
                // single NoteOff should only release one of them.
                break;
            }
        }
    }

    fn dispatch_all_notes_off(&self, channel: u8) {
        let mut voices = match self.voices.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        for v in voices.iter_mut() {
            if v.channel == channel {
                v.voice.trigger_release();
            }
        }
    }

    /// Sum every active voice into `out` (additive — caller is
    /// responsible for zeroing first if they want a clean slate).
    /// Voices that report `is_done()` after rendering are removed.
    /// **Allocation-free.**
    pub fn process_block(&self, out: &mut [f32]) {
        let mut voices = match self.voices.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        for v in voices.iter_mut() {
            v.voice.render_add(out);
        }
        voices.retain(|v| !v.voice.is_done());
    }

    /// Number of voices currently in the pool (before the next
    /// `process_block` retain). Cheap; takes the voices lock briefly.
    pub fn active_count(&self) -> usize {
        let voices = match self.voices.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        voices.len()
    }

    /// Trigger release on every voice in the pool, regardless of
    /// channel. Used by `LiveMidiPlayer::stop` / `seek`.
    pub fn all_notes_off(&self) {
        let mut voices = match self.voices.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        for v in voices.iter_mut() {
            v.voice.trigger_release();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Test voice with a tunable lifetime. `note_off_done` controls
    /// whether `trigger_release` immediately marks the voice done (so
    /// `process_block` retires it on the next pass) — this lets the
    /// `note_off_releases_voice` test verify the full pipeline without
    /// running a real envelope for hundreds of samples.
    struct MockVoice {
        released: bool,
        done_after_release: bool,
        /// Counter to verify `render_add` actually ran.
        render_calls: Arc<AtomicUsize>,
    }

    impl VoiceImpl for MockVoice {
        fn render_add(&mut self, buf: &mut [f32]) {
            self.render_calls.fetch_add(1, Ordering::Relaxed);
            for s in buf.iter_mut() {
                *s += 0.0;
            }
        }
        fn trigger_release(&mut self) {
            self.released = true;
        }
        fn is_done(&self) -> bool {
            self.released && self.done_after_release
        }
        fn is_releasing(&self) -> bool {
            self.released
        }
    }

    struct MockFactory {
        name: String,
        decay: DecayModel,
        done_after_release: bool,
        render_calls: Arc<AtomicUsize>,
    }

    impl MockFactory {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                decay: DecayModel::Damper,
                done_after_release: true,
                render_calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl VoiceFactory for MockFactory {
        fn make_voice(
            &self,
            _sample_rate: f32,
            _freq: f32,
            _velocity: f32,
        ) -> Box<dyn VoiceImpl + Send> {
            Box::new(MockVoice {
                released: false,
                done_after_release: self.done_after_release,
                render_calls: Arc::clone(&self.render_calls),
            })
        }
        fn name(&self) -> &str {
            &self.name
        }
        fn decay_model(&self) -> DecayModel {
            self.decay
        }
    }

    fn note_on(channel: u8, note: u8, velocity: u8) -> MidiEvent {
        MidiEvent {
            channel,
            kind: MidiEventKind::NoteOn { note, velocity },
        }
    }

    fn note_off(channel: u8, note: u8) -> MidiEvent {
        MidiEvent {
            channel,
            kind: MidiEventKind::NoteOff { note },
        }
    }

    #[test]
    fn note_on_allocates_voice() {
        let f: Arc<dyn VoiceFactory> = Arc::new(MockFactory::new("a"));
        let pool = VoicePool::new(44_100.0, 32, f);
        assert_eq!(pool.active_count(), 0);
        pool.dispatch(&note_on(0, 60, 100));
        assert_eq!(pool.active_count(), 1);
    }

    #[test]
    fn note_off_releases_voice() {
        let f: Arc<dyn VoiceFactory> = Arc::new(MockFactory::new("a"));
        let pool = VoicePool::new(44_100.0, 32, f);
        pool.dispatch(&note_on(0, 60, 100));
        assert_eq!(pool.active_count(), 1);
        pool.dispatch(&note_off(0, 60));
        // NoteOff alone doesn't shrink the pool — only `process_block`
        // calls `retain`. After one render pass the MockVoice (which
        // reports is_done == true once released) should be culled.
        let mut buf = vec![0.0_f32; 64];
        pool.process_block(&mut buf);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn voice_cap_steals_oldest() {
        let f: Arc<dyn VoiceFactory> = Arc::new(MockFactory::new("a"));
        let pool = VoicePool::new(44_100.0, 2, f);
        pool.dispatch(&note_on(0, 60, 100));
        pool.dispatch(&note_on(0, 62, 100));
        pool.dispatch(&note_on(0, 64, 100));
        assert_eq!(pool.active_count(), 2);
        // The oldest voice (note 60) should be gone; 62 and 64 remain.
        let voices = pool.voices.lock().unwrap_or_else(|p| p.into_inner());
        let notes: Vec<u8> = voices.iter().map(|v| v.note).collect();
        assert!(!notes.contains(&60), "note 60 should have been stolen");
        assert!(notes.contains(&62));
        assert!(notes.contains(&64));
    }

    #[test]
    fn set_channel_factory_swaps_for_next_note() {
        let fa: Arc<dyn VoiceFactory> = Arc::new(MockFactory::new("factory_a"));
        let fb: Arc<dyn VoiceFactory> = Arc::new(MockFactory::new("factory_b"));
        let pool = VoicePool::new(44_100.0, 32, Arc::clone(&fa));
        assert_eq!(pool.channel_factory_name(0), "factory_a");

        // Note 1: on factory_a.
        pool.dispatch(&note_on(0, 60, 100));
        assert_eq!(pool.active_count(), 1);

        // Hot-swap.
        pool.set_channel_factory(0, Arc::clone(&fb));
        assert_eq!(pool.channel_factory_name(0), "factory_b");

        // The first voice is still alive (live flip does not interrupt).
        assert_eq!(pool.active_count(), 1);

        // Note 2: should come from factory_b. The channel slot's name
        // is the contract the UI relies on; use it as the swap-took-
        // effect signal. To independently verify the new voice came
        // from factory_b (and not, say, a stale Arc), we keep concrete
        // handles to both factories and check render-call counters
        // before / after a process_block that targets only the new
        // voice.
        let fb_concrete = Arc::new(MockFactory::new("factory_b_probe"));
        let fb_trait: Arc<dyn VoiceFactory> = Arc::clone(&fb_concrete) as Arc<dyn VoiceFactory>;
        pool.set_channel_factory(0, fb_trait);
        assert_eq!(pool.channel_factory_name(0), "factory_b_probe");

        let before = fb_concrete.render_calls.load(Ordering::Relaxed);
        pool.dispatch(&note_on(0, 64, 100));
        let mut buf = vec![0.0_f32; 16];
        pool.process_block(&mut buf);
        let after = fb_concrete.render_calls.load(Ordering::Relaxed);
        assert!(
            after > before,
            "factory_b_probe should have produced the new voice and rendered at least once"
        );
        assert_eq!(pool.channel_factory_name(0), "factory_b_probe");
    }

    #[test]
    fn voice_cap_zero_does_not_panic() {
        // A degenerate pool with zero capacity should silently swallow
        // NoteOn dispatches rather than panic on `voices.remove(0)`.
        let f: Arc<dyn VoiceFactory> = Arc::new(MockFactory::new("a"));
        let pool = VoicePool::new(44_100.0, 0, f);
        assert_eq!(pool.active_count(), 0);
        pool.dispatch(&note_on(0, 60, 100));
        pool.dispatch(&note_on(0, 62, 100));
        assert_eq!(pool.active_count(), 0);
        // process_block on an empty pool must also be a no-op.
        let mut buf = vec![0.0_f32; 16];
        pool.process_block(&mut buf);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn process_block_is_additive_and_culls_done() {
        let f: Arc<dyn VoiceFactory> = Arc::new(MockFactory::new("a"));
        let pool = VoicePool::new(44_100.0, 8, f);
        pool.dispatch(&note_on(0, 60, 100));
        pool.dispatch(&note_on(0, 62, 100));
        let mut buf = vec![1.0_f32; 32]; // pre-existing content
        pool.process_block(&mut buf);
        // MockVoice adds 0; pre-existing content must survive (additive).
        assert!(buf.iter().all(|s| *s == 1.0));
        // No voice is done yet.
        assert_eq!(pool.active_count(), 2);
        pool.all_notes_off();
        pool.process_block(&mut buf);
        assert_eq!(pool.active_count(), 0);
    }
}
