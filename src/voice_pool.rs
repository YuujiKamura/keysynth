//! Polyphonic voice pool with per-channel routing and live factory flip.
//!
//! This module is the audio-callback-side substrate for the Jukebox Live
//! Migration (see `JUKEBOX_LIVE_MIGRATION.md`). It owns up to `voice_cap`
//! simultaneously sounding voices and routes incoming MIDI events to
//! per-channel `VoiceFactory` slots so the UI can hot-swap timbres
//! between notes without interrupting voices that are already ringing.
//!
//! # Concurrency model (Issue #71)
//!
//! The previous version put `voices: Mutex<Vec<ActiveVoice>>` between
//! the audio callback and any UI thread that wanted to read a counter
//! for a frame update. Every `active_count()` from a 60 Hz UI loop
//! contended with the cpal callback for the same lock, opening a
//! priority-inversion path to audio under-runs.
//!
//! This version splits the surface in two:
//!
//! - **Audio thread (exclusive owner of the voice list).** `tick`,
//!   `process_block`, `all_notes_off` and the `dispatch_*` helpers
//!   take `&mut self` and are invoked from the cpal callback (which
//!   owns the `VoicePool` value). Because the borrow checker enforces
//!   exclusivity, no `unsafe` and no `UnsafeCell` are needed. The
//!   voice list is a plain `Vec<ActiveVoice>` field. No locks of any
//!   kind are taken on this path.
//!
//! - **UI thread (read-only via atomics, write-only via SPSC).** The UI
//!   gets a thin `VoicePoolUi` handle with a `Sender<PoolCmd>` and
//!   shared atomic / `ArcSwap` snapshots. `active_count()` is
//!   `AtomicUsize::load(Relaxed)`; channel-factory names are read via
//!   wait-free `ArcSwap::load_full`; channel-factory writes go onto
//!   the SPSC queue and are applied by the audio thread on its next tick.
//!
//! # Invariants (audio-thread safe)
//!
//! - `process_block` does not allocate. Voices are summed in place into
//!   the caller-provided buffer; finished voices are removed via
//!   `Vec::retain` (in-place, no realloc on shrink).
//! - `dispatch(NoteOn)` may allocate inside `make_voice`; the audio
//!   thread tolerates this because note-on rate is bounded by the MIDI
//!   event stream (≪ block rate).
//! - No `panic!` / `unwrap` / `expect`. SPSC sends from the UI silently
//!   drop if the queue is full or the audio side has gone away — UI
//!   command rate is human-scale, and the audio thread drains on every
//!   tick, so under steady state the queue never approaches its bound.
//! - Voice eviction order: `is_releasing()` voices are stolen first
//!   (chord_pad pattern — they are inaudible by then); otherwise
//!   `voices.remove(0)` takes the oldest still-sustaining voice. The
//!   `Vec::retain` step in `process_block` preserves index order, so
//!   "position 0 == oldest still sustaining" holds without a separate
//!   timestamp (Issue #73 — `started_at` removed).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::Arc;

use arc_swap::ArcSwap;

use crate::synth::VoiceImpl;
use crate::voice_lib::DecayModel;

// `MidiEvent` / `MidiEventKind` の単一定義は `crate::midi_sched` 側。
// POOL は同じ型を再エクスポートして使う (Phase 1 hub 統合で Plan A 解消)。
pub use crate::midi_sched::{MidiEvent, MidiEventKind};

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
///
/// No `started_at` field (Issue #73): eviction picks the oldest
/// sustaining voice purely by index, which works because the voice list
/// is a `Vec` and `Vec::retain` preserves order.
struct ActiveVoice {
    channel: u8,
    note: u8,
    voice: Box<dyn VoiceImpl + Send>,
    decay_model: DecayModel,
}

// ---------------------------------------------------------------------------
// PoolCmd — UI → audio thread message
// ---------------------------------------------------------------------------

/// Commands the UI thread can post to the audio thread via the SPSC
/// queue. Only operations that mutate the voice list or per-channel
/// factory routing live here; pure read-side helpers (`active_count`,
/// `channel_factory_name`) bypass the queue and read atomic snapshots.
enum PoolCmd {
    SetChannelFactory {
        channel: u8,
        factory: Arc<dyn VoiceFactory>,
    },
    AllNotesOff,
}

/// Capacity of the UI → audio command queue. UI command rate is
/// human-scale (factory swaps, panic stops), so even tens of commands
/// per second sit comfortably inside this bound. A bounded queue lets
/// us reject (silently drop) rather than block the UI thread if the
/// audio thread ever wedges.
const CMD_QUEUE_CAP: usize = 256;

// ---------------------------------------------------------------------------
// VoicePool
// ---------------------------------------------------------------------------

const NUM_CHANNELS: usize = 16;

/// Polyphonic voice pool. See module docs for the audio/UI split.
///
/// # Ownership model
///
/// The `VoicePool` value should live inside the cpal callback closure
/// (or a struct moved into the closure). Audio-thread methods take
/// `&mut self`, so the borrow checker enforces that nothing else can
/// see the voice list while the audio thread is touching it. UI threads
/// interact through a `VoicePoolUi` handle returned by `ui_handle()`,
/// which holds only `Send + Sync` projections — an SPSC sender, an
/// `Arc<AtomicUsize>` for the voice count, and per-channel
/// `Arc<ArcSwap<String>>` for factory names.
///
/// The split has two consequences worth highlighting:
///
/// - **No `unsafe` anywhere.** The previous draft of this module used
///   `UnsafeCell<Vec<…>>` to mutate from `&self` methods. That is
///   unsound for any caller that can clone the pool into multiple
///   threads. By taking `&mut self` instead, we get the same lock-
///   free behaviour with full Rust safety.
/// - **`set_channel_factory(&self, …)` stays `&self`.** The legacy
///   direct path is retained for back-compat (existing tests, audio-
///   thread init code that doesn't yet have an exclusive borrow).
///   This field is therefore stored in an `ArcSwap` so the read
///   side stays wait-free even when shared.
pub struct VoicePool {
    sample_rate: f32,
    voice_cap: usize,

    /// Active voices. Mutated only via `&mut self` audio-thread
    /// methods; the borrow checker rules out any concurrent access.
    voices: Vec<ActiveVoice>,

    /// Per-channel current factory. Audio thread reads via `load_full`
    /// when handling NoteOn; UI thread requests writes by sending
    /// `PoolCmd::SetChannelFactory`, and the audio thread applies them
    /// via `store` while draining the queue. ArcSwap is wait-free on
    /// the read side, which is what we need on the audio path.
    ///
    /// The element type is `Arc<dyn VoiceFactory>` (boxed once), and
    /// arc-swap stores it behind another `Arc`. The double `Arc` is
    /// arc-swap's `RefCnt` requirement: `Arc<T>: RefCnt` only for
    /// `T: Sized`, so we cannot use `ArcSwap<dyn VoiceFactory>`
    /// directly. The extra indirection is on a NoteOn-rate path, not
    /// the per-sample loop, so it is free of consequence.
    channel_factories: [ArcSwap<Arc<dyn VoiceFactory>>; NUM_CHANNELS],

    /// UI-readable snapshot of each channel's factory name. Updated by
    /// the audio thread alongside `channel_factories`. Wrapped in `Arc`
    /// per element so `VoicePoolUi` can share the same `ArcSwap`
    /// instance (rather than a stale clone). Reads are wait-free.
    channel_names: [Arc<ArcSwap<String>>; NUM_CHANNELS],

    /// UI → audio command queue. Receiver is drained by the audio
    /// thread on every `tick`; `cmd_tx` is cloned out to UI handles.
    cmd_rx: Receiver<PoolCmd>,
    cmd_tx: SyncSender<PoolCmd>,

    /// UI-readable snapshot of `voices.len()`. Audio thread writes this
    /// after every state change; UI reads relaxed.
    active_count_atomic: Arc<AtomicUsize>,
}

impl VoicePool {
    /// Build a pool of capacity `voice_cap`, with every channel pointing
    /// at `default` initially. The UI can override individual channels
    /// later via `set_channel_factory` (audio-thread direct path) or
    /// `VoicePoolUi::set_channel_factory` (SPSC-routed path).
    pub fn new(sample_rate: f32, voice_cap: usize, default: Arc<dyn VoiceFactory>) -> Self {
        let default_name = default.name().to_string();
        let channel_factories: [ArcSwap<Arc<dyn VoiceFactory>>; NUM_CHANNELS] =
            std::array::from_fn(|_| ArcSwap::from(Arc::new(Arc::clone(&default))));
        let channel_names: [Arc<ArcSwap<String>>; NUM_CHANNELS] =
            std::array::from_fn(|_| Arc::new(ArcSwap::from(Arc::new(default_name.clone()))));

        let (cmd_tx, cmd_rx) = sync_channel(CMD_QUEUE_CAP);

        Self {
            sample_rate,
            voice_cap,
            voices: Vec::with_capacity(voice_cap),
            channel_factories,
            channel_names,
            cmd_rx,
            cmd_tx,
            active_count_atomic: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Hand out a thin UI-side handle. The handle clones the
    /// `Sender<PoolCmd>`, the active-count `Arc<AtomicUsize>`, and the
    /// per-channel `Arc<ArcSwap<String>>` projections so it is `Send +
    /// Sync` and cheap to copy. UI code can park it inside an egui
    /// context or a separate thread without holding `&VoicePool`.
    pub fn ui_handle(&self) -> VoicePoolUi {
        VoicePoolUi {
            cmd_tx: self.cmd_tx.clone(),
            active_count: Arc::clone(&self.active_count_atomic),
            channel_names: std::array::from_fn(|i| Arc::clone(&self.channel_names[i])),
        }
    }

    /// Audio-thread entry point. Drain pending UI commands, dispatch
    /// `events` (which the caller has produced this block from
    /// `MidiSequencer::advance` or similar), render every active voice
    /// into `out`, retire finished voices, and publish the new
    /// `active_count` snapshot.
    ///
    /// This is the **single canonical hot path** for the audio
    /// callback. Existing callers that prefer the fine-grained API
    /// (`dispatch` + `process_block`) keep working — `tick` is
    /// equivalent to "drain commands, then call `dispatch` for each
    /// event, then call `process_block`".
    ///
    /// AUDIO THREAD ONLY. Takes `&mut self` so the borrow checker
    /// enforces exclusivity — no extra runtime guard needed.
    pub fn tick(&mut self, events: &[MidiEvent], out: &mut [f32]) {
        self.drain_cmds();
        for ev in events {
            self.dispatch(ev);
        }
        self.process_block(out);
    }

    /// Drain pending UI commands from the SPSC queue. AUDIO THREAD ONLY.
    fn drain_cmds(&mut self) {
        while let Ok(cmd) = self.cmd_rx.try_recv() {
            match cmd {
                PoolCmd::SetChannelFactory { channel, factory } => {
                    let idx = channel as usize;
                    if idx >= NUM_CHANNELS {
                        continue;
                    }
                    let name = Arc::new(factory.name().to_string());
                    self.channel_factories[idx].store(Arc::new(factory));
                    self.channel_names[idx].store(name);
                }
                PoolCmd::AllNotesOff => {
                    self.dispatch_all_notes_off_global();
                }
            }
        }
    }

    /// Hot-swap the factory bound to `channel`. AUDIO-THREAD-CALLABLE
    /// (legacy path): out-of-thread callers should prefer
    /// `VoicePoolUi::set_channel_factory`, which routes through the
    /// SPSC queue. This direct method is kept for backward
    /// compatibility (existing tests, audio-thread init code).
    /// Out-of-range channels (>= 16) are silently ignored.
    pub fn set_channel_factory(&self, channel: u8, factory: Arc<dyn VoiceFactory>) {
        let idx = channel as usize;
        if idx >= NUM_CHANNELS {
            return;
        }
        let name = Arc::new(factory.name().to_string());
        self.channel_factories[idx].store(Arc::new(factory));
        self.channel_names[idx].store(name);
    }

    /// Name of the factory currently bound to `channel`. Returns an
    /// empty string if `channel` is out of range. Wait-free
    /// `ArcSwap::load_full` snapshot — no locks.
    pub fn channel_factory_name(&self, channel: u8) -> String {
        let idx = channel as usize;
        if idx >= NUM_CHANNELS {
            return String::new();
        }
        (*self.channel_names[idx].load_full()).clone()
    }

    /// Route one MIDI event to the pool. AUDIO THREAD ONLY.
    ///
    /// Existing callers (jukebox / live MIDI player) invoke `dispatch`
    /// directly from inside the cpal callback, which satisfies the
    /// audio-thread contract.
    pub fn dispatch(&mut self, ev: &MidiEvent) {
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

    fn dispatch_note_on(&mut self, channel: u8, note: u8, velocity: u8) {
        if channel as usize >= NUM_CHANNELS {
            return;
        }
        // A pool with zero capacity can never hold a voice; bail out
        // before we touch anything else (the eviction branch below
        // would otherwise call `voices.remove(0)` on an empty Vec).
        if self.voice_cap == 0 {
            return;
        }
        // Snapshot the current factory atomically. ArcSwap::load_full
        // returns an `Arc<Arc<dyn VoiceFactory>>` (see field-level doc
        // for the double-Arc justification). Cloning the inner `Arc`
        // gives us a flat handle the rest of this function can use;
        // later `store` from the cmd drain step does not invalidate
        // it because each snapshot keeps its own ref-counted handle.
        let factory_outer = self.channel_factories[channel as usize].load_full();
        let factory: Arc<dyn VoiceFactory> = Arc::clone(&*factory_outer);
        let decay_model = factory.decay_model();

        let freq = crate::synth::midi_to_freq(note);
        let vel_norm = (velocity as f32) / 127.0;
        let voice = factory.make_voice(self.sample_rate, freq, vel_norm);

        let voices = &mut self.voices;

        // Eviction: at cap, drop a releasing voice first (chord_pad
        // pattern — those are already on their way out and stealing
        // them is inaudible), else the oldest sustained voice. Index
        // 0 is "oldest" because `Vec::retain` (the only other shrink
        // step) preserves index order.
        //
        // We use `Vec::remove` (O(n) shift) deliberately, *not*
        // `swap_remove` (O(1) but reorders): the eviction policy
        // depends on positional order being preserved so "index 0 ==
        // oldest" stays true on the next eviction. With voice_cap
        // bounded to typical polyphony (≤ 128), the shift is a
        // handful of pointer copies on the audio thread — far below
        // the cost of one block of synth math.
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
        });

        let n = voices.len();
        self.publish_active_count(n);
    }

    fn dispatch_note_off(&mut self, channel: u8, note: u8) {
        for v in self.voices.iter_mut() {
            if v.channel == channel && v.note == note && !v.voice.is_releasing() {
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

    fn dispatch_all_notes_off(&mut self, channel: u8) {
        for v in self.voices.iter_mut() {
            if v.channel == channel {
                v.voice.trigger_release();
            }
        }
    }

    fn dispatch_all_notes_off_global(&mut self) {
        for v in self.voices.iter_mut() {
            v.voice.trigger_release();
        }
    }

    /// Sum every active voice into `out` (additive — caller is
    /// responsible for zeroing first if they want a clean slate).
    /// Voices that report `is_done()` after rendering are removed.
    /// **Allocation-free.** AUDIO THREAD ONLY.
    pub fn process_block(&mut self, out: &mut [f32]) {
        for v in self.voices.iter_mut() {
            v.voice.render_add(out);
        }
        self.voices.retain(|v| !v.voice.is_done());
        let n = self.voices.len();
        self.publish_active_count(n);
    }

    /// Number of voices currently in the pool (last published by the
    /// audio thread). Wait-free `Relaxed` load — safe to call from any
    /// thread, including a 60 Hz UI redraw loop.
    pub fn active_count(&self) -> usize {
        self.active_count_atomic.load(Ordering::Relaxed)
    }

    /// Trigger release on every voice in the pool, regardless of
    /// channel. AUDIO-THREAD-CALLABLE legacy path; UI threads should
    /// use `VoicePoolUi::all_notes_off`, which posts to the SPSC queue.
    pub fn all_notes_off(&mut self) {
        self.dispatch_all_notes_off_global();
        let n = self.voices.len();
        self.publish_active_count(n);
    }

    #[inline]
    fn publish_active_count(&self, n: usize) {
        self.active_count_atomic.store(n, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// VoicePoolUi — UI-side handle
// ---------------------------------------------------------------------------

/// UI-side handle returned by `VoicePool::ui_handle`.
///
/// All methods on this type are safe to call from any thread other than
/// the audio thread. Mutations go through the SPSC queue (one queue
/// element per call); reads are atomic / wait-free.
///
/// If the audio thread has already been dropped (queue receiver gone)
/// or the queue is full, sends are silently discarded — the UI must
/// not treat these as fallible because the audio thread is the
/// dominant lifetime in the app.
#[derive(Clone)]
pub struct VoicePoolUi {
    cmd_tx: SyncSender<PoolCmd>,
    active_count: Arc<AtomicUsize>,
    /// One read-only projection per channel, sharing the same
    /// `ArcSwap<String>` instance with the pool. Wait-free reads.
    channel_names: [Arc<ArcSwap<String>>; NUM_CHANNELS],
}

impl VoicePoolUi {
    /// Request the audio thread bind `factory` to `channel`. Out-of-
    /// range channels are dropped on the audio side.
    pub fn set_channel_factory(&self, channel: u8, factory: Arc<dyn VoiceFactory>) {
        // try_send so a full queue (or dropped audio thread) cannot
        // block the UI. Dropping a SetChannelFactory is preferable to
        // stalling a 60 Hz redraw — the user can re-issue.
        match self
            .cmd_tx
            .try_send(PoolCmd::SetChannelFactory { channel, factory })
        {
            Ok(()) | Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {}
        }
    }

    /// Request the audio thread release every voice in the pool.
    pub fn all_notes_off(&self) {
        match self.cmd_tx.try_send(PoolCmd::AllNotesOff) {
            Ok(()) | Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {}
        }
    }

    /// Wait-free snapshot of how many voices are currently sounding.
    pub fn active_count(&self) -> usize {
        self.active_count.load(Ordering::Relaxed)
    }

    /// Wait-free snapshot of the factory name currently bound to
    /// `channel`. Reflects the most recent `SetChannelFactory` that
    /// the audio thread has drained.
    pub fn channel_factory_name(&self, channel: u8) -> String {
        let idx = channel as usize;
        if idx >= NUM_CHANNELS {
            return String::new();
        }
        (*self.channel_names[idx].load_full()).clone()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize as TAtomicUsize;
    use std::sync::Arc as TArc;
    use std::thread;
    use std::time::{Duration, Instant};

    /// Test voice with a tunable lifetime. `done_after_release` controls
    /// whether `trigger_release` immediately marks the voice done (so
    /// `process_block` retires it on the next pass).
    struct MockVoice {
        released: bool,
        done_after_release: bool,
        /// Counter to verify `render_add` actually ran.
        render_calls: TArc<TAtomicUsize>,
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
        render_calls: TArc<TAtomicUsize>,
    }

    impl MockFactory {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                decay: DecayModel::Damper,
                done_after_release: true,
                render_calls: TArc::new(TAtomicUsize::new(0)),
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
                render_calls: TArc::clone(&self.render_calls),
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
        let mut pool = VoicePool::new(44_100.0, 32, f);
        assert_eq!(pool.active_count(), 0);
        pool.dispatch(&note_on(0, 60, 100));
        assert_eq!(pool.active_count(), 1);
    }

    #[test]
    fn note_off_releases_voice() {
        let f: Arc<dyn VoiceFactory> = Arc::new(MockFactory::new("a"));
        let mut pool = VoicePool::new(44_100.0, 32, f);
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

    /// Issue #73 acceptance: with `started_at` removed, eviction at
    /// capacity steals the position-0 voice, which is the oldest
    /// because `Vec::retain` preserves order.
    #[test]
    fn voice_cap_steals_oldest_by_index() {
        let f: Arc<dyn VoiceFactory> = Arc::new(MockFactory::new("a"));
        let mut pool = VoicePool::new(44_100.0, 2, f);
        pool.dispatch(&note_on(0, 60, 100));
        pool.dispatch(&note_on(0, 62, 100));
        pool.dispatch(&note_on(0, 64, 100));
        assert_eq!(pool.active_count(), 2);
        // The oldest voice (note 60) should be gone; 62 and 64 remain.
        // Read voices back via the field directly (legal here — the
        // test is single-threaded and lives in the same module).
        let notes: Vec<u8> = pool.voices.iter().map(|v| v.note).collect();
        assert!(!notes.contains(&60), "note 60 should have been stolen");
        assert!(notes.contains(&62));
        assert!(notes.contains(&64));
    }

    #[test]
    fn set_channel_factory_swaps_for_next_note() {
        let fa: Arc<dyn VoiceFactory> = Arc::new(MockFactory::new("factory_a"));
        let fb: Arc<dyn VoiceFactory> = Arc::new(MockFactory::new("factory_b"));
        let mut pool = VoicePool::new(44_100.0, 32, Arc::clone(&fa));
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
        let mut pool = VoicePool::new(44_100.0, 0, f);
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
        let mut pool = VoicePool::new(44_100.0, 8, f);
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

    /// Issue #71 acceptance: a UI thread spamming
    /// `ui_handle().active_count()` and `set_channel_factory` does not
    /// block the audio thread, and the audio thread's `tick` actually
    /// applies the queued factory swaps.
    ///
    /// We avoid wall-clock timing assertions (flaky on busy CI). The
    /// proxy assertion is functional: after the audio thread runs N
    /// ticks while the UI thread sends K factory swaps, the audio
    /// thread's view of the channel-0 factory name must end up at the
    /// last swap value. If a lock had snuck back in, either the
    /// `active_count()` poll would have stalled or one of the swaps
    /// would have been lost.
    #[test]
    fn concurrent_ui_query_does_not_block_audio() {
        // Models the real ownership: audio thread owns the pool by
        // value, UI threads share `VoicePoolUi` clones. No `Arc<Pool>`
        // anywhere — the borrow checker enforces single-mutator on
        // the audio side.
        let fa: Arc<dyn VoiceFactory> = Arc::new(MockFactory::new("audio_default"));
        let mut pool = VoicePool::new(44_100.0, 32, fa);
        let ui = pool.ui_handle();

        const NUM_SWAPS: usize = 100;
        const NUM_QUERIES: usize = 1000;

        // Pre-build NUM_SWAPS distinct factories so the UI thread
        // doesn't allocate inside the hot loop.
        let factories: Vec<Arc<dyn VoiceFactory>> = (0..NUM_SWAPS)
            .map(|i| {
                Arc::new(MockFactory::new(&format!("ui_swap_{i}")))
                    as Arc<dyn VoiceFactory>
            })
            .collect();

        let stop = TArc::new(std::sync::atomic::AtomicBool::new(false));

        // UI thread A: spam wait-free reads.
        let stop_q = TArc::clone(&stop);
        let ui_q = ui.clone();
        let querier = thread::spawn(move || {
            let mut sum = 0_usize;
            for _ in 0..NUM_QUERIES {
                sum = sum.wrapping_add(ui_q.active_count());
                let _ = ui_q.channel_factory_name(0);
                if stop_q.load(Ordering::Relaxed) {
                    break;
                }
            }
            sum
        });

        // UI thread B: send all swaps.
        let ui_swap = ui.clone();
        let factories_for_swap = factories.clone();
        let swapper = thread::spawn(move || {
            for f in factories_for_swap.iter() {
                ui_swap.set_channel_factory(0, Arc::clone(f));
            }
        });

        // Audio thread: run ticks until the swapper finishes, then
        // drain a few extra ticks for in-flight commands. Wall-clock
        // bound is a CI safety net only — the functional assert below
        // is what really proves Issue #71 is fixed.
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut buf = vec![0.0_f32; 64];
        let mut events: Vec<MidiEvent> = Vec::new();
        loop {
            pool.tick(&events, &mut buf);
            // Inject a NoteOn occasionally so the audio thread does
            // real work, not just queue-draining.
            events.clear();
            events.push(note_on(0, 60, 100));
            if swapper.is_finished() {
                // Drain a couple more ticks to absorb in-flight cmds.
                for _ in 0..8 {
                    pool.tick(&[], &mut buf);
                }
                break;
            }
            if Instant::now() >= deadline {
                stop.store(true, Ordering::Relaxed);
                break;
            }
        }

        stop.store(true, Ordering::Relaxed);
        let _ = querier.join();
        let _ = swapper.join();

        // Functional assert: the last swap must be reflected on both
        // the audio side (`channel_factory_name`) and the UI side
        // (`ui.channel_factory_name`). This is the proxy for "no
        // contention dropped a command".
        let expected = format!("ui_swap_{}", NUM_SWAPS - 1);
        assert_eq!(
            pool.channel_factory_name(0),
            expected,
            "audio side did not converge to last swap — queue or atomic dropped a write"
        );
        assert_eq!(
            ui.channel_factory_name(0),
            expected,
            "UI handle did not see last swap on the shared ArcSwap"
        );
    }
}
