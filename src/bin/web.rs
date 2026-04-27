//! WASM / web entry point for the GitHub Pages demo build.
//!
//! Trims the live native synth down to: pure modelling engines (square / ks
//! / piano / piano-modal etc.), an on-screen 2-octave keyboard driven by
//! mouse + PC keyboard, and an optional Web MIDI bridge. No SF2, no SFZ,
//! no live `midir` -- those are gated out at the Cargo feature level so
//! this binary only sees the cross-target DSP code.
//!
//! Audio is built lazily on first user gesture (click the "Start audio"
//! button) so the browser autoplay policy doesn't reject `AudioContext`
//! creation.
//!
//! All wasm-only code lives in `imp` and is gated to `target_arch =
//! "wasm32"`. On non-wasm targets `main()` is a no-op stub so
//! `cargo check --all-features --bins` (or any host-side build that
//! happens to enable the `web` feature) still resolves a `main` symbol.

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!(
        "keysynth-web is a wasm32-only binary. Build it via `trunk build` from \
         the repo root; this stub exists so host-side `cargo check` passes."
    );
}

#[cfg(target_arch = "wasm32")]
fn main() {
    imp::start();
}

#[cfg(target_arch = "wasm32")]
mod imp {

    use std::cell::RefCell;
    use std::io::Cursor;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};

    use eframe::egui;
    use rustysynth::{SoundFont, Synthesizer, SynthesizerSettings};
    use wasm_bindgen::closure::Closure;
    use wasm_bindgen::{JsCast, JsValue};

    use std::collections::VecDeque;

    use keysynth::gm::{GM_FAMILIES, GM_INSTRUMENTS};
    use keysynth::reverb::{self, Reverb};
    use keysynth::sympathetic::SympatheticBank;
    use keysynth::synth::{
        make_voice, midi_to_freq, Engine, LiveParams, MixMode, ModalLut, Voice, MODAL_LUT,
    };

    /// Path of the GM SoundFont served alongside the wasm bundle. Trunk
    /// copies `web/GeneralUser-GS.sf2` to the dist root via the
    /// `rel="copy-file"` link in `web/index.html`, so the relative URL
    /// works regardless of the Pages subpath (`/keysynth/` on prod,
    /// `/` under `trunk serve`).
    const SF2_URL: &str = "GeneralUser-GS.sf2";

    /// Process-wide shared `rustysynth::Synthesizer` populated by an
    /// async fetch that runs after the AudioContext is alive (the
    /// Synthesizer takes the actual `ctx.sample_rate()` for its
    /// interpolation, and we need the AudioContext sample rate before
    /// kicking off the load). `None` before fetch + parse complete;
    /// `Some` for the rest of the page lifetime once the bytes arrive.
    /// `Mutex` mirrors the native `SharedSynth` pattern even though
    /// wasm32 is single-threaded — the lock is uncontended and lets the
    /// MIDI / render paths share the same reference shape as `main.rs`.
    type SharedSynth = Arc<Mutex<Option<Synthesizer>>>;

    /// Lifecycle of the SF2 asset fetch + parse, surfaced in the UI so
    /// the user can tell whether picking `SfPiano` will actually make
    /// sound or silently no-op while the bytes are still en route.
    /// Held as `Rc<RefCell<…>>` because the async fetch closure mutates
    /// it from outside any `&mut self` borrow.
    #[derive(Clone, Debug)]
    enum SfStatus {
        /// Pre-`build_audio`. No fetch issued yet.
        Idle,
        /// Fetch (or fetch + parse) in flight.
        Loading,
        /// Synthesizer ready in `WebApp::synth`.
        Ready,
        /// Fetch or parse failed. Carries the error text so the UI can
        /// surface it to the user.
        Failed(String),
    }

    /// Engines exposed to the web UI. Three columns:
    ///
    ///   1. The `Engine` variant.
    ///   2. A short user-facing label rendered in the dropdown / current
    ///      selection. Friendlier than the CLI flag (`piano-modal` →
    ///      `Piano (modal, SFZ-derived)`) so first-time visitors can
    ///      pick a starting point without reading the source.
    ///   3. A one-line description rendered under the dropdown to
    ///      explain what the selected engine actually does.
    ///
    /// The order matters: the piano family lands at the top because
    /// that's the demo's flagship and the web build defaults into the
    /// first entry. Within each family we lead with the variant we
    /// actively recommend (`PianoModal` for piano, `KsRich` for
    /// plucked, etc.).
    ///
    /// SfzPiano deliberately omitted — it needs the multi-GB Salamander
    /// Grand V3 SFZ library on disk that we can't ship to Pages.
    /// `SfPiano` ships because the GM SoundFont is small enough (~32 MB)
    /// to ship as a static asset alongside the wasm and fetch on
    /// demand (see `fetch_and_load_sf2`); the wasm itself stays small
    /// (~5 MB) so first-paint isn't blocked by the SF2 download.
    const ENGINES_FOR_WEB: &[(Engine, &str, &str)] = &[
        // ── Piano family ────────────────────────────────────────────
        (
            Engine::SfPiano,
            "Piano (SF2 sampled, GeneralUser GS)",
            "Real recorded piano samples played by rustysynth from the \
             bundled GeneralUser-GS SoundFont (program 0 = Acoustic \
             Grand). Closest match to the native build's `--engine \
             sf-piano`; the modal/KS engines below are pure DSP for \
             comparison.",
        ),
        (
            Engine::PianoModal,
            "Piano (modal, SFZ-derived)",
            "Parallel-bandpass projection: per-note partials + T60 + \
             init_amp lifted from SFZ Salamander Grand recordings, \
             32-mode bass extrapolation, 3-string ±0.7 cent detune, \
             coloured-noise hammer impulse. Most reference-faithful.",
        ),
        (
            Engine::PianoLite,
            "Piano (light soundboard)",
            "3-string KS + 12-mode 'lite' soundboard, no shared \
             sympathetic bank. Tuned against SFZ Salamander C4 T60 \
             vector — closest per-partial decay match across the \
             physical-model variants.",
        ),
        (
            Engine::PianoThick,
            "Piano (thick soundboard)",
            "7-string KS + 12-mode lite soundboard, fuller body \
             radiation than `light`. Heavier but flabbier; nice for \
             held chords.",
        ),
        (
            Engine::Piano,
            "Piano (KS-string base)",
            "Original 3-string KS + asymmetric ms-scale hammer + \
             frequency-dependent decay + sympathetic bank. The voice \
             that all the variants above started from.",
        ),
        (
            Engine::Piano5AM,
            "Piano (5 AM snapshot)",
            "Frozen snapshot of the perceptually-balanced state from \
             commit 8f0df23 (3 strings, no soundboard, no sym bank). \
             Kept reproducible so the morning's tone can be A/B'd \
             against later builds.",
        ),
        // ── Plucked strings ─────────────────────────────────────────
        (
            Engine::Koto,
            "Koto",
            "Single-string KS with a sharp narrow plectrum injected \
             at ~1/4 string length. Long sustain, minimal stiffness.",
        ),
        (
            Engine::KsRich,
            "Karplus-Strong (rich)",
            "3-string unison detune + 1-pole allpass dispersion in the \
             feedback loop. Stiffer, chorused — closer to a piano-ish \
             sustain than the basic KS.",
        ),
        (
            Engine::Ks,
            "Karplus-Strong (basic)",
            "Single string, white-noise-excited delay loop with a \
             2-tap lowpass. Phase-1 physical model — thin plucked \
             string.",
        ),
        // ── Classic synths ──────────────────────────────────────────
        (
            Engine::Sub,
            "Subtractive (analog)",
            "Sawtooth → state-variable lowpass with cutoff envelope + \
             ADSR amp. Classic 1980s analog-synth voice.",
        ),
        (
            Engine::Fm,
            "FM (DX7-ish bell)",
            "2-operator FM (sine carrier × sine modulator, ratio 14:1) \
             with its own ADSR on the modulation index. Bell / \
             e-piano flavour.",
        ),
        (
            Engine::Square,
            "Square wave (NES)",
            "NES-style pulse + linear AR envelope. The cheapest \
             reference tone — useful as a sanity check that the \
             output path is alive.",
        ),
    ];

    // PC keyboard → MIDI semitone offset within the current octave.
    // Standard tracker layout: bottom row = white keys, q-row = upper octave.
    const PC_KEYMAP: &[(egui::Key, i32)] = &[
        // Lower octave (white + black row)
        (egui::Key::Z, 0),  // C
        (egui::Key::S, 1),  // C#
        (egui::Key::X, 2),  // D
        (egui::Key::D, 3),  // D#
        (egui::Key::C, 4),  // E
        (egui::Key::V, 5),  // F
        (egui::Key::G, 6),  // F#
        (egui::Key::B, 7),  // G
        (egui::Key::H, 8),  // G#
        (egui::Key::N, 9),  // A
        (egui::Key::J, 10), // A#
        (egui::Key::M, 11), // B
        // Upper octave
        (egui::Key::Q, 12), // C
        (egui::Key::Num2, 13),
        (egui::Key::W, 14),
        (egui::Key::Num3, 15),
        (egui::Key::E, 16),
        (egui::Key::R, 17),
        (egui::Key::Num5, 18),
        (egui::Key::T, 19),
        (egui::Key::Num6, 20),
        (egui::Key::Y, 21),
        (egui::Key::Num7, 22),
        (egui::Key::U, 23),
        (egui::Key::I, 24),
    ];

    // On-screen keyboard span in semitones (2 octaves + 1 = 25 keys).
    const KEYBOARD_SPAN: u8 = 25;

    // ---------------------------------------------------------------------
    // Audio output via AudioWorklet
    // ---------------------------------------------------------------------
    //
    // Earlier revisions used cpal's webaudio backend, which schedules a
    // pair of AudioBufferSourceNodes and re-arms them via `setTimeout`
    // polled on the main JS thread. That polling competes with egui paint
    // for main-thread time, so a busy frame slips the next-buffer
    // schedule and the output picks up clicks at every gap. The fix is
    // to move the *output* side into an AudioWorkletProcessor, whose
    // `process()` is driven from the audio thread at deterministic rate.
    //
    // Architecture:
    //   - Main thread (Rust) renders mono samples in chunks and posts
    //     them to the worklet via `port.postMessage`.
    //   - Worklet (JS) holds a ring buffer; on each `process()` call
    //     it copies samples out of the ring into the output channels.
    //   - When the ring drops below half-full, the worklet posts a
    //     `'need'` request back to the main thread, which renders one
    //     more chunk and posts it. Closed-loop, no polling.
    //
    // No SharedArrayBuffer required, so this works on plain GitHub
    // Pages without COOP/COEP service-worker shims. There's a
    // postMessage hop per chunk, but with a generous ring (default
    // 16384 frames ≈ 372 ms at 44.1 kHz) the main thread can stall
    // for 100+ ms without the worklet ever running dry.
    const WORKLET_PROCESSOR_JS: &str = r#"
class KeysynthProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.cap = opts.ringCapacity || 16384;
    this.ring = new Float32Array(this.cap);
    this.read = 0;
    this.write = 0;
    this.size = 0;
    this.requesting = false;
    this.lowWatermark = Math.floor(this.cap / 2);
    this.port.onmessage = (e) => {
      // Reset the request flag at the very top so a malformed or
      // unexpected message can't permanently hang the request loop.
      this.requesting = false;
      const msg = e.data;
      if (!msg || msg.type !== 'fill' || !msg.samples) return;
      const s = msg.samples;
      const n = s.length;
      // Drop overflow rather than overwrite — caller respects watermark.
      const room = this.cap - this.size;
      const copy = n < room ? n : room;
      if (copy <= 0) return;
      // typed-array `.set()` is a memcpy; faster than a per-sample
      // loop and cheaper for v8's GC.
      if (this.write + copy <= this.cap) {
        this.ring.set(s.subarray(0, copy), this.write);
      } else {
        const first = this.cap - this.write;
        this.ring.set(s.subarray(0, first), this.write);
        this.ring.set(s.subarray(first, copy), 0);
      }
      this.write = (this.write + copy) % this.cap;
      this.size += copy;
    };
    // Kick the main thread for the first fill so process() doesn't
    // start emitting silence before the first chunk arrives.
    this.port.postMessage({ type: 'need', want: this.cap });
    this.requesting = true;
  }
  process(inputs, outputs, parameters) {
    const channels = outputs[0];
    const out0 = channels[0];
    const n = out0.length;
    if (this.size >= n) {
      // Fast path: fully drain n samples in one go via subarray + set.
      if (this.read + n <= this.cap) {
        out0.set(this.ring.subarray(this.read, this.read + n));
      } else {
        const first = this.cap - this.read;
        out0.set(this.ring.subarray(this.read, this.cap));
        out0.set(this.ring.subarray(0, n - first), first);
      }
      this.read = (this.read + n) % this.cap;
      this.size -= n;
    } else {
      const have = this.size;
      if (have > 0) {
        if (this.read + have <= this.cap) {
          out0.set(this.ring.subarray(this.read, this.read + have));
        } else {
          const first = this.cap - this.read;
          out0.set(this.ring.subarray(this.read, this.cap));
          out0.set(this.ring.subarray(0, have - first), first);
        }
        this.read = (this.read + have) % this.cap;
      }
      out0.fill(0, have);
      this.size = 0;
    }
    for (let c = 1; c < channels.length; c++) channels[c].set(out0);
    if (this.size < this.lowWatermark && !this.requesting) {
      this.requesting = true;
      this.port.postMessage({ type: 'need', want: this.cap - this.size });
    }
    return true;
  }
}
registerProcessor('keysynth-processor', KeysynthProcessor);
"#;

    /// Live audio output handle. Holds the AudioContext + worklet node
    /// + the closure that handles fill requests so JS-side references
    /// stay alive for the page lifetime.
    struct AudioHandle {
        _ctx: web_sys::AudioContext,
        _node: web_sys::AudioWorkletNode,
        _on_message: Closure<dyn FnMut(web_sys::MessageEvent)>,
    }

    // ---------------------------------------------------------------------
    // Web MIDI bridge
    // ---------------------------------------------------------------------
    //
    // Browser fires MIDI events on the JS event loop; we translate each
    // raw 3-byte status message into a `MidiMsg` and push it onto a shared
    // inbox `Rc<RefCell<Vec<MidiMsg>>>`. The egui update loop drains the
    // inbox once per frame and turns the messages into `note_on` /
    // `note_off` calls against the existing voice pool. Single-threaded
    // wasm32 → no Send/Sync requirement on the inbox, hence Rc/RefCell
    // instead of Arc/Mutex.

    #[derive(Clone, Copy, Debug)]
    enum MidiMsg {
        NoteOn { note: u8, velocity: u8 },
        NoteOff { note: u8 },
    }

    type MidiInbox = Rc<RefCell<Vec<MidiMsg>>>;

    /// Shared registry of every per-port `onmidimessage` closure. Initial
    /// inputs (resolved from `requestMIDIAccess`) and hot-plugged inputs
    /// (added by the `onstatechange` callback) both push into the same
    /// `Vec` so we don't have to leak per-event closures on hot-plug.
    /// Lives behind `Rc<RefCell<…>>` because the onstatechange closure
    /// owns a clone separate from the one stored on `WebApp::midi_handles`.
    type MessageClosures = Rc<RefCell<Vec<Closure<dyn FnMut(web_sys::MidiMessageEvent)>>>>;

    /// Closures + access object handed to the browser. The browser only
    /// stores function references; if we drop the underlying Rust
    /// closures the references go dangling and MIDI silently stops.
    /// Park everything here so `WebApp::midi_handles` keeps it alive
    /// for the page lifetime — and so a future "Disconnect MIDI"
    /// operation has a single place to drop from.
    struct MidiHandles {
        _access: web_sys::MidiAccess,
        _closures: MessageClosures,
        _on_state_change: Closure<dyn FnMut(web_sys::MidiConnectionEvent)>,
    }

    struct WebApp {
        voices: Arc<Mutex<Vec<Voice>>>,
        live: Arc<Mutex<LiveParams>>,
        sample_rate: u32,
        /// `AudioHandle` is moved here by the async worklet handshake
        /// (see `build_audio`) so the splash gate can flip from "Start"
        /// to the running UI as soon as the AudioContext is alive.
        audio: Rc<RefCell<Option<AudioHandle>>>,
        /// Set true synchronously by `start_audio()` and cleared by the
        /// async worklet handshake on success/failure. Stops a frantic
        /// double-click on the splash button from spawning two parallel
        /// `AudioContext` + `addModule` flows — the second one would
        /// fight the first for the voice pool's render closures and
        /// race on which `AudioHandle` lands in `audio`.
        audio_starting: Rc<std::cell::Cell<bool>>,
        /// Async-populated error slot. `Rc<RefCell<…>>` because the
        /// `spawn_local` block writes after `start_audio()` has
        /// returned.
        audio_err: Rc<RefCell<Option<String>>>,
        /// Notes currently held by mouse/touch. PC keyboard tracking is via
        /// the separate `pc_held` set below — egui's `key_pressed` returns
        /// true on every browser KeyDown including auto-repeat, so we
        /// track our own edge state instead.
        mouse_down_note: Option<u8>,
        /// MIDI notes the PC keyboard is currently holding down, packed
        /// as a 128-bit bitset (one bit per MIDI note 0..=127). Edge
        /// transitions (bit set → note_on, bit cleared → note_off) drive
        /// the voice pool so a held key produces ONE voice, not a stream
        /// of re-triggered voices on each browser auto-repeat event.
        /// Bitset over `HashSet<u8>` avoids per-frame heap traffic in
        /// the egui input loop.
        pc_held: u128,
        /// Whether the page had focus on the previous frame. Browsers
        /// sometimes drop the `keyup` event when the user Alt-Tabs /
        /// switches tabs / triggers IME mid-key — leaving the egui key
        /// state stuck as "pressed" forever and our `pc_held` bitset
        /// holding phantom notes. On a true→false focus edge we
        /// release every pc-held note; on the next focus regain the
        /// user starts from a clean slate.
        prev_focused: bool,
        /// Base MIDI note for the leftmost on-screen key. Defaults to C3 (48).
        base_note: u8,
        /// Status string for Web MIDI (e.g. "not requested" / "connected: ..."
        /// / error). `Rc<RefCell<String>>` because the async MIDI handshake
        /// (which lives outside any `&mut self` borrow) needs to write into
        /// it and the egui update loop needs to read it.
        midi_status: Rc<RefCell<String>>,
        /// Inbox shared with browser MIDI callbacks. Producer = JS event
        /// handlers (`onmidimessage`), consumer = `WebApp::update` once
        /// per frame.
        midi_inbox: MidiInbox,
        /// Tracks whether a MIDI handshake is currently in-flight or has
        /// succeeded. Held in `Rc<Cell<bool>>` so the async resolution
        /// path can clear it on error (letting the user re-click "Connect
        /// MIDI" without reloading the page) without needing a `&mut self`
        /// reference. Set true synchronously in `request_midi()`, cleared
        /// in the async error branch, kept true on success.
        midi_requested: Rc<std::cell::Cell<bool>>,
        /// Slot the async handshake populates with the resolved
        /// `MidiHandles`. Stored in `Rc<RefCell<Option<…>>>` rather than
        /// leaked via `Box::leak` so the resources have one named home,
        /// hot-plugged closures get pushed into the existing handles
        /// instead of leaking on every device reconnect, and a future
        /// "Disconnect MIDI" path has somewhere to drop from.
        midi_handles: Rc<RefCell<Option<MidiHandles>>>,
        /// Shared `rustysynth::Synthesizer` for `Engine::SfPiano`. Filled
        /// in by an async fetch task that runs after `build_audio`
        /// completes (Synthesizer needs `ctx.sample_rate()`). `None`
        /// until the SF2 download + parse completes; SfPiano is silent
        /// while loading. Other engines work normally throughout.
        synth: SharedSynth,
        /// Lifecycle of the SF2 fetch + parse, displayed in the UI so
        /// `Engine::SfPiano` selection isn't a silent no-op while
        /// loading.
        synth_status: Rc<RefCell<SfStatus>>,
        /// Rolling log of recent MIDI events drawn in the right side
        /// panel. Mirrors the native build's `DashState::recent` ring
        /// (newest at the back, capped at 64). `Rc<RefCell<…>>` because
        /// `note_on` / `note_off` are called through `&mut self` while
        /// the side panel UI also wants to read it via egui's borrow
        /// closure — using a separate Rc avoids re-entering `&mut
        /// self` during update().
        recent_events: Rc<RefCell<VecDeque<String>>>,
    }

    impl Default for WebApp {
        fn default() -> Self {
            // Initialise the modal LUT once. On wasm32 this pulls bytes from
            // include_bytes! — the file lookup never touches a real
            // filesystem.
            let (lut, source) = ModalLut::auto_load(None);
            let _ = MODAL_LUT.set(lut);
            web_sys::console::log_1(&format!("keysynth-web: modal LUT source = {source}").into());

            Self {
                voices: Arc::new(Mutex::new(Vec::with_capacity(32))),
                live: Arc::new(Mutex::new(LiveParams {
                    // Match native defaults (main.rs::Args::default). 1.5
                    // was too low for `Engine::PianoModal`, whose 3-detune
                    // resonator bank produces per-sample peaks ~10x smaller
                    // than KS / square / sub voices; under master=1.5 +
                    // ParallelComp the PianoModal bus sat below audibility
                    // while every other engine still saturated tanh.
                    master: 3.0,
                    engine: Engine::PianoModal,
                    reverb_wet: 0.25,
                    sf_program: 0,
                    sf_bank: 0,
                    // Plain tanh matches native default and avoids the
                    // ParallelComp limiter swallowing PianoModal's already
                    // low-amplitude bus into silence on the first hit.
                    mix_mode: MixMode::Plain,
                })),
                sample_rate: 0,
                audio: Rc::new(RefCell::new(None)),
                audio_starting: Rc::new(std::cell::Cell::new(false)),
                audio_err: Rc::new(RefCell::new(None)),
                mouse_down_note: None,
                pc_held: 0,
                prev_focused: true,
                base_note: 48, // C3
                midi_status: Rc::new(RefCell::new("not requested".to_string())),
                midi_inbox: Rc::new(RefCell::new(Vec::new())),
                midi_requested: Rc::new(std::cell::Cell::new(false)),
                midi_handles: Rc::new(RefCell::new(None)),
                synth: Arc::new(Mutex::new(None)),
                synth_status: Rc::new(RefCell::new(SfStatus::Idle)),
                recent_events: Rc::new(RefCell::new(VecDeque::with_capacity(64))),
            }
        }
    }

    impl WebApp {
        fn start_audio(&mut self) {
            // Reject re-entry while a previous startup is still
            // resolving. `audio.is_some()` only flips after the async
            // worklet handshake completes, so a frantic double-click
            // would otherwise spawn two parallel AudioContext + module
            // loads racing on which `AudioHandle` lands in `audio`.
            if self.audio.borrow().is_some() || self.audio_starting.get() {
                return;
            }
            self.audio_starting.set(true);
            // Clear any prior error so retries don't keep stale text.
            *self.audio_err.borrow_mut() = None;
            match build_audio(
                self.voices.clone(),
                self.live.clone(),
                self.synth.clone(),
                self.audio.clone(),
                self.audio_err.clone(),
                self.audio_starting.clone(),
            ) {
                Ok(sr) => {
                    self.sample_rate = sr;
                    web_sys::console::log_1(
                        &format!("keysynth-web: audio started @ {sr} Hz").into(),
                    );
                    // Same user-gesture context — request MIDI access in
                    // the same click so the user doesn't have to hunt
                    // for a second button. If the browser doesn't
                    // support Web MIDI or the user denies the prompt,
                    // `request_midi` writes the error into
                    // `midi_status` and resets `midi_requested` so the
                    // dedicated Retry button still appears below.
                    self.request_midi();
                    // Kick off the SF2 fetch + parse async. The page is
                    // already responsive at this point (modelling
                    // engines work immediately); SfPiano stays silent
                    // until the bytes arrive and `synth_status` flips
                    // to `Ready`.
                    self.start_sf2_load();
                }
                Err(e) => {
                    web_sys::console::error_1(
                        &format!("keysynth-web: audio start failed: {e}").into(),
                    );
                    *self.audio_err.borrow_mut() = Some(e);
                    // `build_audio` already cleared `starting` on its
                    // synchronous error path, but doing it here too is
                    // a defence-in-depth so a future refactor that
                    // forgets the inner reset still recovers.
                    self.audio_starting.set(false);
                }
            }
        }

        /// Drain the MIDI inbox. Called once per egui frame so MIDI
        /// events line up with the rest of the per-frame edge handling
        /// (PC keyboard, on-screen mouse). Each drained message turns
        /// into a `note_on` / `note_off` against the shared voice pool.
        fn drain_midi_inbox(&mut self) {
            // Take ownership of the queued messages in one swap so the
            // inbox borrow doesn't overlap with self.note_on / note_off
            // (which lock self.voices and self.live).
            let drained: Vec<MidiMsg> = {
                let mut q = self.midi_inbox.borrow_mut();
                std::mem::take(&mut *q)
            };
            for msg in drained {
                match msg {
                    MidiMsg::NoteOn { note, velocity } => self.note_on(note, velocity),
                    MidiMsg::NoteOff { note } => self.note_off(note),
                }
            }
        }

        /// Spawn the async SF2 fetch + parse. Caller (`start_audio`)
        /// only invokes this AFTER `build_audio` succeeded so we know
        /// `self.sample_rate` is valid. The fetch happens in the
        /// background — the page is fully interactive while the bytes
        /// download and rustysynth parses them. Engine selection and
        /// modelling-voice playback work throughout; only `SfPiano`
        /// stays silent until the synth lands in `self.synth`.
        fn start_sf2_load(&self) {
            if !matches!(
                *self.synth_status.borrow(),
                SfStatus::Idle | SfStatus::Failed(_)
            ) {
                return;
            }
            let synth = self.synth.clone();
            let status = self.synth_status.clone();
            let sr = self.sample_rate;
            *status.borrow_mut() = SfStatus::Loading;
            wasm_bindgen_futures::spawn_local(async move {
                let outcome = fetch_and_load_sf2(synth, sr).await;
                match outcome {
                    Ok(bytes_len) => {
                        web_sys::console::log_1(
                            &format!(
                                "keysynth-web: SF2 ready ({bytes_len} bytes fetched, sr={sr} Hz)"
                            )
                            .into(),
                        );
                        *status.borrow_mut() = SfStatus::Ready;
                    }
                    Err(e) => {
                        web_sys::console::warn_1(
                            &format!("keysynth-web: SF2 load failed: {e}").into(),
                        );
                        *status.borrow_mut() = SfStatus::Failed(e);
                    }
                }
            });
        }

        /// Kick off the Web MIDI handshake. `navigator.requestMIDIAccess()`
        /// returns a Promise we await via `wasm_bindgen_futures::spawn_local`;
        /// once it resolves we attach `onmidimessage` handlers to every
        /// input port and an `onstatechange` handler so devices plugged in
        /// AFTER the request still get wired up.
        ///
        /// The captured `inbox` and `status` are `Rc` clones — they're the
        /// only state the async block needs from `self`, and they live as
        /// long as the page does. The resulting `MidiHandles` is leaked
        /// (`Box::leak`) so the browser-side closures stay valid forever;
        /// `WebApp` is itself page-lifetime so this isn't a real leak in
        /// practice and it spares us a channel dance to ship the handles
        /// back through a `&mut self` boundary.
        fn request_midi(&mut self) {
            if self.midi_requested.get() {
                return;
            }
            self.midi_requested.set(true);
            *self.midi_status.borrow_mut() = "requesting...".to_string();
            let inbox = self.midi_inbox.clone();
            let status = self.midi_status.clone();
            let requested_flag = self.midi_requested.clone();
            let handles_slot = self.midi_handles.clone();
            wasm_bindgen_futures::spawn_local(async move {
                match request_midi_access(inbox).await {
                    Ok(handles) => {
                        let n = handles._closures.borrow().len();
                        *status.borrow_mut() = format!("{n} input(s) connected");
                        // Store on `WebApp` so the closures live as long
                        // as the page does and have a named owner. No
                        // `Box::leak` — drop happens with `WebApp` on
                        // page teardown.
                        *handles_slot.borrow_mut() = Some(handles);
                        // Leave `requested_flag` true on success so the
                        // Connect button stays out of the way.
                    }
                    Err(e) => {
                        *status.borrow_mut() = format!("error: {e}");
                        // Reset on failure so the user can press Connect
                        // again — permission-denial is recoverable on a
                        // second click that re-prompts the browser.
                        requested_flag.set(false);
                    }
                }
            });
        }

        fn note_on(&mut self, note: u8, velocity: u8) {
            if self.audio.borrow().is_none() || self.sample_rate == 0 {
                return;
            }
            let engine = self.live.lock().unwrap().engine;
            self.push_event(format!("noteOn  ch=0 note={note:>3} vel={velocity:>3}"));

            // SfPiano routes the note straight into the rustysynth
            // shared synth — the modelling-engine voice pool is bypassed
            // entirely (the synth holds its own internal voice state and
            // mixing into the bus happens in render_mono_chunk). For
            // every other engine we fall through to the modelling path.
            if engine == Engine::SfPiano {
                if let Some(s) = self.synth.lock().unwrap().as_mut() {
                    s.note_on(0, note as i32, velocity as i32);
                }
                return;
            }

            let freq = midi_to_freq(note);
            let inner = make_voice(engine, self.sample_rate as f32, freq, velocity);
            let v = Voice {
                key: (0, note),
                inner,
            };
            let mut pool = self.voices.lock().unwrap();
            if let Some(slot) = pool.iter_mut().find(|x| x.key == (0, note)) {
                *slot = v;
            } else {
                const MAX_VOICES: usize = 24;
                if pool.len() >= MAX_VOICES {
                    let evict = pool
                        .iter()
                        .position(|x| x.inner.is_done() || x.inner.is_releasing())
                        .unwrap_or(0);
                    pool.remove(evict);
                }
                pool.push(v);
            }
        }

        fn note_off(&mut self, note: u8) {
            self.push_event(format!("noteOff ch=0 note={note:>3}"));
            // Always forward note_off to the synth so SF2 keys release
            // even after the user has switched away from SfPiano while
            // a note was held — leaving rustysynth's internal envelope
            // pinned would leak voices for the AudioContext lifetime.
            if let Some(s) = self.synth.lock().unwrap().as_mut() {
                s.note_off(0, note as i32);
            }
            let mut pool = self.voices.lock().unwrap();
            if let Some(slot) = pool.iter_mut().find(|x| x.key == (0, note)) {
                slot.inner.trigger_release();
            }
        }

        /// Force-release every voice the app currently thinks is held.
        ///
        /// Called automatically on a focus-lost edge (browsers don't
        /// always deliver `keyup` when the page loses focus mid-press,
        /// leaving egui's key state and our `pc_held` bitset stuck on
        /// phantom notes) and from a manual "Panic" button. Three
        /// passes:
        ///
        ///   1. Iterate the bits set in `pc_held`, fire `note_off` for
        ///      each → matching pool voices enter their release tail.
        ///   2. Clear `pc_held` and `mouse_down_note` so subsequent
        ///      input edges are clean.
        ///   3. As a belt-and-braces, walk the entire voice pool and
        ///      `trigger_release` on every voice — even ones we've
        ///      lost track of (stuck MIDI keys after a cable yank,
        ///      etc.) get released here.
        fn panic_release(&mut self) {
            // Pass 1: drain pc_held bits.
            let mut bits = self.pc_held;
            while bits != 0 {
                let note = bits.trailing_zeros() as u8;
                self.note_off(note);
                bits &= bits - 1;
            }
            self.pc_held = 0;
            if let Some(n) = self.mouse_down_note.take() {
                self.note_off(n);
            }
            // Pass 3: anything still sounding gets release.
            let mut pool = self.voices.lock().unwrap();
            for v in pool.iter_mut() {
                v.inner.trigger_release();
            }
        }

        /// Process PC keyboard input. Two failure modes to dodge:
        ///
        ///   1. Browsers fire OS auto-repeat KeyDown events while a key is
        ///      held, and `egui::InputState::key_pressed` returns true on
        ///      each one — a held `z` plays as a fast trill instead of a
        ///      sustained tone.
        ///   2. A `keys_down` snapshot misses any tap that begins and ends
        ///      between two UI frames (low-FPS spike or short stab) — the
        ///      note never sounds at all.
        ///
        /// Stay event-based (`key_pressed` / `key_released`) so transient
        /// taps survive, but filter via our own `pc_held` set: `note_on`
        /// only fires when the note isn't already in the held-set, which
        /// suppresses auto-repeat without dropping any genuine event.
        fn handle_pc_keyboard(&mut self, ctx: &egui::Context) {
            // Bounded by KEYBOARD_SPAN (=25) so a stack-sized smallvec
            // would also work, but Vec::new() with no pushes doesn't
            // allocate, and most frames push 0-1 entries.
            let mut to_on: Vec<u8> = Vec::new();
            let mut to_off: Vec<u8> = Vec::new();
            ctx.input(|i| {
                for &(key, semi) in PC_KEYMAP {
                    let note = (self.base_note as i32 + semi).clamp(0, 127) as u8;
                    let bit = 1u128 << note;
                    if i.key_released(key) && (self.pc_held & bit) != 0 {
                        self.pc_held &= !bit;
                        to_off.push(note);
                    }
                    // `(self.pc_held & bit) == 0` — only fire note_on if
                    // the bit isn't already set. Browser KeyDown
                    // auto-repeats while a key is held arrive as
                    // `key_pressed = true` on every frame, but the bit
                    // is already set from the first press so we skip.
                    if i.key_pressed(key) && (self.pc_held & bit) == 0 {
                        self.pc_held |= bit;
                        to_on.push(note);
                    }
                }
            });
            for note in to_off {
                self.note_off(note);
            }
            for note in to_on {
                self.note_on(note, 100);
            }
        }

        /// Push a one-line summary of a MIDI event into the rolling
        /// `recent_events` ring used by the right side panel. Caps the
        /// ring at 64 entries (newest at the back) so a sustained MIDI
        /// stream doesn't grow unbounded across the page lifetime.
        fn push_event(&self, line: String) {
            let mut events = self.recent_events.borrow_mut();
            if events.len() >= 64 {
                events.pop_front();
            }
            events.push_back(line);
        }

        /// Render the GM 128 patch picker on the left. Disabled (greyed)
        /// unless `Engine::SfPiano` is the currently selected engine —
        /// the patch number only matters for the SoundFont path.
        fn draw_instruments_panel(&self, ctx: &egui::Context) {
            let (engine, mut sf_program, mut sf_bank) = {
                let lp = self.live.lock().unwrap();
                (lp.engine, lp.sf_program, lp.sf_bank)
            };
            let sf_active = engine == Engine::SfPiano;
            let mut changed = false;
            egui::SidePanel::left("instruments")
                .default_width(280.0)
                .show(ctx, |ui| {
                    ui.heading("GM 128 patches");
                    ui.add_space(2.0);
                    // Drum-kit toggle. GeneralUser-GS maps drum kits to
                    // bank 128; program then selects the kit (0 =
                    // Standard, 8 = Room, 16 = Power, ...). When drum
                    // mode is on, melodic patch clicks below first flip
                    // bank back to 0.
                    ui.horizontal(|ui| {
                        let drum_on = sf_bank == 128;
                        let label = if drum_on {
                            "Drum Kit (bank 128) [ON]"
                        } else {
                            "Drum Kit (bank 128) [off]"
                        };
                        if ui.selectable_label(drum_on, label).clicked() {
                            sf_bank = if drum_on { 0 } else { 128 };
                            changed = true;
                        }
                    });
                    ui.label(
                        egui::RichText::new(if sf_active {
                            "Click an instrument to switch live."
                        } else {
                            "Switch engine to SF2 piano to use these."
                        })
                        .small()
                        .color(egui::Color32::from_rgb(170, 170, 170)),
                    );
                    ui.separator();
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.add_enabled_ui(sf_active, |ui| {
                                for family in GM_FAMILIES.iter() {
                                    // Open Pianos by default; other
                                    // families collapse so the panel
                                    // stays scannable on narrow
                                    // viewports.
                                    let default_open = *family == "Pianos";
                                    egui::CollapsingHeader::new(*family)
                                        .default_open(default_open)
                                        .show(ui, |ui| {
                                            for (prog, name, fam) in GM_INSTRUMENTS.iter() {
                                                if fam != family {
                                                    continue;
                                                }
                                                let selected =
                                                    sf_program == *prog && sf_bank != 128;
                                                let label_text = format!("[{:>3}] {}", prog, name);
                                                let rich = if selected {
                                                    egui::RichText::new(label_text)
                                                        .monospace()
                                                        .strong()
                                                        .color(egui::Color32::from_rgb(
                                                            255, 220, 120,
                                                        ))
                                                } else {
                                                    egui::RichText::new(label_text).monospace()
                                                };
                                                if ui.selectable_label(selected, rich).clicked() {
                                                    sf_program = *prog;
                                                    // Picking a melodic
                                                    // patch implicitly
                                                    // leaves drum mode.
                                                    sf_bank = 0;
                                                    changed = true;
                                                }
                                            }
                                        });
                                }
                            });
                        });
                });

            // Apply changes outside the panel closure: persist the new
            // program/bank in `LiveParams` for the audio render to
            // observe, and immediately push the GM Bank Select + Program
            // Change pair into the shared synth so the next keypress
            // sounds the new patch instead of waiting on note-driven
            // path. No-op when the synth hasn't loaded yet.
            if changed {
                {
                    let mut lp = self.live.lock().unwrap();
                    lp.sf_program = sf_program;
                    lp.sf_bank = sf_bank;
                }
                if let Some(s) = self.synth.lock().unwrap().as_mut() {
                    // CC 0 = Bank Select MSB. Per GM/GS spec, bank
                    // change only takes effect on the next program
                    // change so we always emit the pair together.
                    s.process_midi_message(0, 0xB0, 0, sf_bank as i32);
                    s.process_midi_message(0, 0xC0, sf_program as i32, 0);
                }
                self.push_event(format!("patch program={sf_program} bank={sf_bank}"));
            }
        }

        fn draw_on_screen_keyboard(&mut self, ui: &mut egui::Ui) {
            // 2-octave (+1) keyboard. Lay out one wide rect per semitone with
            // black-key inserts drawn on top so a click selects the right
            // pitch class. Geometry is rough but legible.
            let span = KEYBOARD_SPAN as i32;
            let total_w = ui.available_width().min(720.0);
            let key_h = 100.0;
            let white_w = total_w / 15.0; // 15 white keys in a 2-octave span
            let black_w = white_w * 0.6;
            let black_h = key_h * 0.6;

            // Allocate the full rect first so child painters share it.
            let (rect, response) = ui.allocate_exact_size(
                egui::vec2(white_w * 15.0, key_h),
                egui::Sense::click_and_drag(),
            );
            let painter = ui.painter_at(rect);

            // Pass 1: white keys.
            let mut white_idx = 0usize;
            let mut white_rects: Vec<(u8, egui::Rect)> = Vec::with_capacity(15);
            for semi in 0..span {
                if !is_black(semi as u8) {
                    let x0 = rect.left() + white_idx as f32 * white_w;
                    let r = egui::Rect::from_min_size(
                        egui::pos2(x0, rect.top()),
                        egui::vec2(white_w, key_h),
                    );
                    let note = (self.base_note as i32 + semi).clamp(0, 127) as u8;
                    let active = self.is_note_sounding(note);
                    let fill = if active {
                        egui::Color32::from_rgb(160, 200, 240)
                    } else {
                        egui::Color32::WHITE
                    };
                    painter.rect_filled(r, 2.0, fill);
                    painter.rect_stroke(r, 2.0, egui::Stroke::new(1.0, egui::Color32::BLACK));
                    white_rects.push((note, r));
                    white_idx += 1;
                }
            }

            // Pass 2: black keys overlaid on top.
            let mut black_rects: Vec<(u8, egui::Rect)> = Vec::with_capacity(10);
            let mut white_idx = 0usize;
            for semi in 0..span {
                if is_black(semi as u8) {
                    // Black key sits at the right edge of the previous white key.
                    let prev_white_x =
                        rect.left() + (white_idx as f32 - 0.5) * white_w + white_w - black_w * 0.5;
                    let r = egui::Rect::from_min_size(
                        egui::pos2(prev_white_x, rect.top()),
                        egui::vec2(black_w, black_h),
                    );
                    let note = (self.base_note as i32 + semi).clamp(0, 127) as u8;
                    let active = self.is_note_sounding(note);
                    let fill = if active {
                        egui::Color32::from_rgb(80, 100, 140)
                    } else {
                        egui::Color32::BLACK
                    };
                    painter.rect_filled(r, 1.0, fill);
                    painter.rect_stroke(r, 1.0, egui::Stroke::new(1.0, egui::Color32::DARK_GRAY));
                    black_rects.push((note, r));
                } else {
                    white_idx += 1;
                }
            }

            // Hit-test pointer against rects: black keys win over white when
            // overlapping. Mouse drag = sustained note; release = note off.
            let pointer_pos = response.interact_pointer_pos();
            let mouse_down = response.is_pointer_button_down_on() && pointer_pos.is_some();

            let hovered_note = pointer_pos.and_then(|p| {
                // black first
                for (n, r) in &black_rects {
                    if r.contains(p) {
                        return Some(*n);
                    }
                }
                for (n, r) in &white_rects {
                    if r.contains(p) {
                        return Some(*n);
                    }
                }
                None
            });

            match (mouse_down, hovered_note, self.mouse_down_note) {
                (true, Some(new), None) => {
                    self.note_on(new, 100);
                    self.mouse_down_note = Some(new);
                }
                (true, Some(new), Some(old)) if new != old => {
                    self.note_off(old);
                    self.note_on(new, 100);
                    self.mouse_down_note = Some(new);
                }
                (false, _, Some(old)) => {
                    self.note_off(old);
                    self.mouse_down_note = None;
                }
                _ => {}
            }
        }

        fn is_note_sounding(&self, note: u8) -> bool {
            let pool = self.voices.lock().unwrap();
            pool.iter()
                .any(|v| v.key.1 == note && !v.inner.is_releasing())
        }
    }

    fn is_black(semi: u8) -> bool {
        matches!(semi % 12, 1 | 3 | 6 | 8 | 10)
    }

    impl eframe::App for WebApp {
        fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
            // Repaint at 30 fps rather than the implicit 60 fps from a
            // bare `request_repaint()`. The cpal webaudio backend on
            // wasm32 polls `setTimeout` for the next AudioBufferSource
            // schedule on the same JS event loop egui paints from, so
            // a busy 60 fps repaint loop directly steals scheduling
            // slack from the audio thread → audible clicks at buffer
            // boundaries. 30 fps is more than enough for the on-screen
            // keyboard's "active" highlight while leaving roughly half
            // the main-thread budget for cpal's polling.
            ctx.request_repaint_after(std::time::Duration::from_millis(33));

            // Focus-loss panic. Browsers don't reliably fire `keyup`
            // when the page loses focus (Alt+Tab, tab switch, IME
            // popup, OS-level screensaver, etc.) — so a key the user
            // released while focus was elsewhere stays "down" in
            // egui's input state forever and our `pc_held` bitset
            // holds a phantom bit for it. Detect the focus true→false
            // edge and force-release everything we think is held.
            let focused_now = ctx.input(|i| i.focused);
            if self.prev_focused && !focused_now {
                self.panic_release();
            }
            self.prev_focused = focused_now;

            if self.audio.borrow().is_some() {
                self.handle_pc_keyboard(ctx);
            }
            // Drain the MIDI inbox every frame regardless of audio state.
            // If audio isn't started yet `note_on` early-returns, so the
            // queued events get consumed silently — the alternative
            // (gating drain on audio_started) lets the queue grow
            // unbounded between page load and the first click on
            // "Start audio", and on resumption the user hears a stale
            // burst of every key they pressed during setup.
            self.drain_midi_inbox();

            // Splash gate: until the user has started audio there's
            // nothing useful to interact with — engine selector,
            // master gain, keyboard all want a running stream. Show
            // a single fullscreen overlay with a big centered Start
            // button so the required first click is impossible to
            // miss. Once audio is running, fall through to the normal
            // 3-panel layout below.
            //
            // `start_audio()` also fires `request_midi()` in the same
            // gesture so the same click satisfies both browser
            // permissions (autoplay → AudioContext, Web MIDI →
            // requestMIDIAccess). On retry / error the smaller "🎹
            // Retry MIDI" button lives inside the running UI.
            if self.audio.borrow().is_none() {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(60.0);
                        ui.heading(egui::RichText::new("keysynth (web)").size(28.0).strong());
                        ui.add_space(12.0);
                        ui.label(
                            egui::RichText::new(
                                "Pure-Rust real-time modelling synth.\n\
                                 Click below to start audio + connect any USB-MIDI keyboard.",
                            )
                            .size(15.0)
                            .color(egui::Color32::from_gray(190)),
                        );
                        ui.add_space(40.0);
                        let resp = ui.add(
                            egui::Button::new(
                                egui::RichText::new("▶ Start")
                                    .size(28.0)
                                    .color(egui::Color32::WHITE)
                                    .strong(),
                            )
                            .fill(egui::Color32::from_rgb(40, 120, 200))
                            .min_size(egui::vec2(220.0, 56.0)),
                        );
                        if resp.clicked() {
                            self.start_audio();
                        }
                        if let Some(err) = self.audio_err.borrow().as_ref() {
                            ui.add_space(20.0);
                            ui.colored_label(
                                egui::Color32::from_rgb(220, 90, 90),
                                format!("audio error: {err}"),
                            );
                            ui.label(
                                egui::RichText::new("Click again to retry")
                                    .size(13.0)
                                    .color(egui::Color32::from_gray(170)),
                            );
                        }
                        ui.add_space(40.0);
                        ui.label(
                            egui::RichText::new(
                                "No MIDI keyboard? Use the on-screen piano (mouse) or PC keys\n\
                                 zsxdcvgbhnjm = lower octave, qwertyui = upper octave",
                            )
                            .size(12.0)
                            .color(egui::Color32::from_gray(150)),
                        );
                    });
                });
                return;
            }

            egui::TopBottomPanel::top("top").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("keysynth (web)");
                    ui.separator();
                    // Audio is already running here (the splash gate
                    // above early-returns until it is), so we just show
                    // the sample rate.
                    ui.label(format!("audio @ {} Hz", self.sample_rate));
                    ui.separator();
                    // Status label is always shown so the previous
                    // outcome (success / error / not requested) stays
                    // visible even after `midi_requested` resets to
                    // false on a permission-denied retry path.
                    let status_owned = self.midi_status.borrow().clone();
                    let is_error = status_owned.starts_with("error");
                    if is_error {
                        ui.colored_label(
                            egui::Color32::from_rgb(220, 90, 90),
                            format!("MIDI: {status_owned}"),
                        );
                    } else {
                        ui.label(format!("MIDI: {status_owned}"));
                    }
                    if !self.midi_requested.get() {
                        // Only reachable if Start failed to grant MIDI
                        // (denied / Web MIDI unsupported). Recover via
                        // an explicit retry click.
                        let resp = ui.add(
                            egui::Button::new(
                                egui::RichText::new("🎹 Retry MIDI")
                                    .size(15.0)
                                    .color(egui::Color32::WHITE)
                                    .strong(),
                            )
                            .fill(egui::Color32::from_rgb(180, 90, 40))
                            .min_size(egui::vec2(140.0, 24.0)),
                        );
                        if resp.clicked() {
                            self.request_midi();
                        }
                    }
                });
                // Hint row only shows when the user is in a recoverable
                // MIDI failure state — once MIDI is connected (or even
                // mid-handshake) the hint disappears so the chrome
                // doesn't waste vertical space.
                if !self.midi_requested.get() {
                    ui.horizontal(|ui| {
                        ui.add_space(4.0);
                        ui.colored_label(
                            egui::Color32::from_gray(170),
                            "Click \"🎹 Retry MIDI\" to request USB-MIDI keyboard access again.",
                        );
                    });
                }
            });

            egui::TopBottomPanel::top("controls").show(ctx, |ui| {
                let mut live = self.live.lock().unwrap();
                let (current_label, current_desc) = ENGINES_FOR_WEB
                    .iter()
                    .find(|(e, _, _)| *e == live.engine)
                    .map(|(_, l, d)| (*l, *d))
                    .unwrap_or(("?", ""));
                ui.horizontal(|ui| {
                    ui.label("engine:");
                    // Drop the fixed-width hint so narrow viewports
                    // (mobile, split panes) aren't forced to overflow.
                    // egui auto-sizes to the longest label.
                    egui::ComboBox::from_id_salt("engine")
                        .selected_text(current_label)
                        .show_ui(ui, |ui| {
                            for (eng, label, _desc) in ENGINES_FOR_WEB {
                                ui.selectable_value(&mut live.engine, *eng, *label);
                            }
                        });

                    // Surface the SF2 fetch lifecycle so SfPiano isn't
                    // a silent no-op while the bytes are still en
                    // route. Idle = pre-audio-start (don't draw), Ready
                    // = quiet OK label, Loading / Failed = visible
                    // status with colour.
                    let status = self.synth_status.borrow().clone();
                    match status {
                        SfStatus::Idle => {}
                        SfStatus::Loading => {
                            ui.separator();
                            ui.colored_label(
                                egui::Color32::from_rgb(220, 180, 90),
                                "SF2: loading…",
                            );
                        }
                        SfStatus::Ready => {
                            ui.separator();
                            ui.colored_label(
                                egui::Color32::from_gray(150),
                                "SF2: ready",
                            );
                        }
                        SfStatus::Failed(msg) => {
                            ui.separator();
                            ui.colored_label(
                                egui::Color32::from_rgb(220, 90, 90),
                                format!("SF2 failed: {msg}"),
                            );
                        }
                    }

                    ui.separator();
                    ui.label("master:");
                    ui.add(egui::Slider::new(&mut live.master, 0.0..=4.0).step_by(0.05));

                    ui.separator();
                    ui.label("reverb:");
                    ui.add(egui::Slider::new(&mut live.reverb_wet, 0.0..=1.0).step_by(0.01));

                    ui.separator();
                    ui.label("mix:");
                    egui::ComboBox::from_id_salt("mix")
                        .selected_text(live.mix_mode.as_label())
                        .show_ui(ui, |ui| {
                            for &mm in MixMode::ALL {
                                ui.selectable_value(&mut live.mix_mode, mm, mm.as_label());
                            }
                        });
                });
                drop(live);

                // Manual panic / all-notes-off button. Always-visible
                // recovery for any input state corruption — focus-loss
                // missed `keyup` (handled automatically above too), a
                // MIDI controller cable yank, anything we couldn't
                // anticipate. One click silences everything currently
                // held.
                ui.horizontal(|ui| {
                    let panic_btn = ui.add(
                        egui::Button::new(
                            egui::RichText::new("⏹ Panic / all notes off")
                                .color(egui::Color32::WHITE)
                                .strong(),
                        )
                        .fill(egui::Color32::from_rgb(170, 60, 60))
                        .min_size(egui::vec2(180.0, 22.0)),
                    );
                    if panic_btn.clicked() {
                        self.panic_release();
                    }
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(
                            "(Press if keys get stuck. Also fires automatically when the page loses focus.)",
                        )
                        .small()
                        .color(egui::Color32::from_gray(160)),
                    );
                });

                // Description of the currently-selected engine,
                // rendered on its own row as a wrapping `Label` so
                // long sentences break cleanly on narrow viewports.
                // (`horizontal_wrapped` + `add_space` would only indent
                // the first wrapped line and break alignment on every
                // subsequent line — Gemini medium thread on this
                // diff.) Plain dim text, no faked column indent — the
                // description is a description of the row above, not a
                // column under the dropdown.
                if !current_desc.is_empty() {
                    ui.add_space(2.0);
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(current_desc)
                                .color(egui::Color32::from_gray(160))
                                .small(),
                        )
                        .wrap(),
                    );
                }

                ui.horizontal(|ui| {
                    ui.label("octave:");
                    if ui.button("◀").clicked() && self.base_note >= 12 {
                        self.base_note -= 12;
                    }
                    // Cast to i32 BEFORE the subtraction. base_note is u8 and
                    // can step down to 0 (C-1 in MIDI octave numbering),
                    // where `0u8 / 12 - 1` panics in debug / wraps to 255 in
                    // release.
                    ui.label(format!("C{}", self.base_note as i32 / 12 - 1));
                    if ui.button("▶").clicked() && self.base_note <= 96 {
                        self.base_note += 12;
                    }
                    ui.separator();
                    ui.label("PC keyboard: zsxdcvgbhnjm = lower octave, qwertyui = upper");
                });
            });

            // Left side panel: GM 128 patch picker for `Engine::SfPiano`.
            // Mirrors the native build's `instruments` SidePanel — same
            // collapsible-by-family layout, same drum-kit (bank 128)
            // toggle, same "switch engine to sf-piano to use these"
            // hint for the disabled state. Click an instrument to flip
            // `live.sf_program` / `live.sf_bank` and push the GM Bank
            // Select + Program Change pair into rustysynth so the next
            // keypress sounds the new patch.
            self.draw_instruments_panel(ctx);

            // Right side panel: rolling MIDI log. `recent_events` is
            // populated by note_on / note_off (covers both PC keyboard
            // and Web MIDI sources since both funnel through the same
            // pair). Mirrors `dash_snapshot.recent` on native.
            egui::SidePanel::right("log")
                .default_width(220.0)
                .show(ctx, |ui| {
                    ui.heading("MIDI log");
                    ui.separator();
                    egui::ScrollArea::vertical()
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            for line in self.recent_events.borrow().iter() {
                                ui.monospace(line);
                            }
                        });
                });

            egui::CentralPanel::default().show(ctx, |ui| {
                ui.add_space(12.0);
                self.draw_on_screen_keyboard(ui);
                ui.add_space(20.0);
                ui.collapsing("about", |ui| {
                    ui.label(
                        "keysynth — pure-Rust real-time modelling synth. The web build \
                     fetches GeneralUser-GS.sf2 for SfPiano, drops the multi-GB SFZ \
                     Salamander library; modelling engines (Karplus-Strong, modal \
                     piano, FM, etc.) run unchanged from the native build.",
                    );
                    ui.label("source: https://github.com/YuujiKamura/keysynth");
                });
            });
        }
    }

    // ===========================================================================
    // Audio thread setup
    // ===========================================================================

    /// Synchronous wrapper that kicks off the async worklet handshake.
    /// Returns the AudioContext sample rate immediately (the worklet
    /// loads concurrently — no audible difference because `process()`
    /// only fires after both `addModule` and the first `port` round
    /// trip have resolved). The handle gets stashed into
    /// `WebApp::audio` once the async block populates a shared slot.
    fn build_audio(
        voices: Arc<Mutex<Vec<Voice>>>,
        live: Arc<Mutex<LiveParams>>,
        synth: SharedSynth,
        slot: Rc<RefCell<Option<AudioHandle>>>,
        err_slot: Rc<RefCell<Option<String>>>,
        starting: Rc<std::cell::Cell<bool>>,
    ) -> Result<u32, String> {
        // Hint the browser at 48 kHz so the AudioContext matches what
        // the native cpal default tends to be on the same hardware.
        // Without this hint Chrome on Linux typically picks 44.1 kHz,
        // and the OS audio layer then resamples 44.1 → 48 on the way
        // to the speaker — that resampling step (PulseAudio /
        // PipeWire / CoreAudio depending on host) is the audible
        // lo-fi gap users notice vs the native build, NOT the DSP
        // path itself which is identical Rust code on both sides.
        // Browsers may ignore the hint (Safari, some Android Chrome)
        // and pick a different rate; that's fine, `ctx.sample_rate()`
        // below reflects what we actually got and the rest of the
        // pipeline (DSP, worklet ring) parameterises off it cleanly.
        let ctx_opts = web_sys::AudioContextOptions::new();
        ctx_opts.set_sample_rate(48_000.0);
        let ctx = web_sys::AudioContext::new_with_context_options(&ctx_opts)
            .or_else(|_| {
                // Fall back to the browser default if 48 kHz wasn't
                // accepted (rare, but Safari / older browsers can
                // refuse non-native rates).
                let fallback = web_sys::AudioContextOptions::new();
                web_sys::AudioContext::new_with_context_options(&fallback)
            })
            .map_err(|e| {
                starting.set(false);
                format!("AudioContext::new: {e:?}")
            })?;
        let sr = ctx.sample_rate() as u32;
        web_sys::console::log_1(&format!("keysynth-web: AudioContext sampleRate = {sr} Hz").into());

        // Inline the worklet processor source as a Blob URL so we don't
        // need to ship a separate JS file via Trunk. The browser caches
        // the URL for the AudioContext's lifetime.
        let parts = js_sys::Array::new();
        parts.push(&JsValue::from_str(WORKLET_PROCESSOR_JS));
        let blob_opts = web_sys::BlobPropertyBag::new();
        blob_opts.set_type("text/javascript");
        let blob = web_sys::Blob::new_with_str_sequence_and_options(&parts, &blob_opts)
            .map_err(|e| format!("Blob::new: {e:?}"))?;
        let url = web_sys::Url::create_object_url_with_blob(&blob)
            .map_err(|e| format!("Url::createObjectURL: {e:?}"))?;

        // Per-stream DSP state lives on the heap inside this Rc so the
        // port `onmessage` closure can hold it without hitting
        // `&mut self` lifetime issues.
        let render_state = Rc::new(RefCell::new(RenderState {
            reverb: Reverb::new(reverb::synthetic_body_ir(sr)),
            sympathetic: SympatheticBank::new_piano(sr as f32),
            mono: Vec::new(),
            mono_compressed: Vec::new(),
            limiter_gain: 1.0,
            sf_left: Vec::new(),
            sf_right: Vec::new(),
        }));

        let worklet = ctx
            .audio_worklet()
            .map_err(|e| format!("audio_worklet(): {e:?}"))?;
        let module_promise = worklet
            .add_module(&url)
            .map_err(|e| format!("addModule: {e:?}"))?;

        let ctx_clone = ctx.clone();
        let voices_cb = voices.clone();
        let live_cb = live.clone();
        let synth_cb = synth.clone();
        let render_state_cb = render_state.clone();
        let slot_cb = slot.clone();
        let err_cb = err_slot.clone();
        let starting_cb = starting.clone();

        wasm_bindgen_futures::spawn_local(async move {
            if let Err(e) = wasm_bindgen_futures::JsFuture::from(module_promise).await {
                *err_cb.borrow_mut() = Some(format!("worklet load failed: {e:?}"));
                starting_cb.set(false);
                return;
            }
            // Free the blob URL once the module is parsed.
            let _ = web_sys::Url::revoke_object_url(&url);

            let node_opts = web_sys::AudioWorkletNodeOptions::new();
            // Stereo output (mirrored from mono inside the worklet so
            // both ears get the same signal).
            node_opts.set_output_channel_count(&js_sys::Array::of1(&JsValue::from_f64(2.0)));
            // Pass a generous ring capacity via processor options so the
            // worklet can buffer ~370 ms at 44.1 kHz, absorbing main-
            // thread stalls without underrunning.
            let proc_opts = js_sys::Object::new();
            let _ = js_sys::Reflect::set(
                &proc_opts,
                &JsValue::from_str("ringCapacity"),
                &JsValue::from_f64(16384.0),
            );
            node_opts.set_processor_options(Some(&proc_opts));

            let node = match web_sys::AudioWorkletNode::new_with_options(
                &ctx_clone,
                "keysynth-processor",
                &node_opts,
            ) {
                Ok(n) => n,
                Err(e) => {
                    *err_cb.borrow_mut() = Some(format!("AudioWorkletNode::new: {e:?}"));
                    starting_cb.set(false);
                    return;
                }
            };
            if let Err(e) = node.connect_with_audio_node(ctx_clone.destination().as_ref()) {
                *err_cb.borrow_mut() = Some(format!("node.connect: {e:?}"));
                starting_cb.set(false);
                return;
            }

            // Wire the port: every `'need'` request triggers a render
            // pass on the main thread that produces `want` mono samples
            // and posts them straight back. The worklet's ring buffer
            // absorbs the postMessage round-trip latency.
            let port = match node.port() {
                Ok(p) => p,
                Err(e) => {
                    *err_cb.borrow_mut() = Some(format!("port(): {e:?}"));
                    starting_cb.set(false);
                    return;
                }
            };
            let port_for_cb = port.clone();
            let voices_h = voices_cb;
            let live_h = live_cb;
            let synth_h = synth_cb;
            let render_h = render_state_cb;
            let on_message = Closure::<dyn FnMut(web_sys::MessageEvent)>::new(
                move |ev: web_sys::MessageEvent| {
                    let msg = ev.data();
                    let kind = js_sys::Reflect::get(&msg, &JsValue::from_str("type"))
                        .ok()
                        .and_then(|v| v.as_string());
                    if kind.as_deref() != Some("need") {
                        return;
                    }
                    let want = js_sys::Reflect::get(&msg, &JsValue::from_str("want"))
                        .ok()
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1024.0) as usize;
                    // Cap one render pass at 4096 frames so a huge
                    // initial fill request doesn't stall the main
                    // thread for tens of ms; the worklet will just ask
                    // again next process() until full.
                    let chunk = want.min(4096);
                    let (master, wet, engine, mix_mode) = {
                        let lp = live_h.lock().unwrap();
                        (lp.master, lp.reverb_wet, lp.engine, lp.mix_mode)
                    };
                    // Render directly into `state.mono` so the per-
                    // request `Vec<f32>` allocation goes away —
                    // `state.mono` is reused across requests and only
                    // resizes when the chunk size changes. The
                    // Float32Array copy below is the unavoidable
                    // wasm→JS heap hop.
                    let arr = {
                        let mut state = render_h.borrow_mut();
                        render_mono_chunk(
                            &voices_h, &synth_h, master, wet, engine, mix_mode, &mut state, chunk,
                        );
                        let a = js_sys::Float32Array::new_with_length(chunk as u32);
                        a.copy_from(&state.mono);
                        a
                    };
                    let out_msg = js_sys::Object::new();
                    let _ = js_sys::Reflect::set(
                        &out_msg,
                        &JsValue::from_str("type"),
                        &JsValue::from_str("fill"),
                    );
                    let _ = js_sys::Reflect::set(&out_msg, &JsValue::from_str("samples"), &arr);
                    let _ = port_for_cb.post_message(&out_msg);
                },
            );
            port.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
            // Some browsers leave the AudioContext suspended even after
            // a user gesture; explicitly resume and treat a rejection
            // as an audio start failure rather than silently marking
            // the handle as ready (which would hide the splash gate
            // and leave the user with a silent app, no recovery path
            // short of a reload — Codex P1).
            match ctx_clone.resume() {
                Ok(p) => {
                    if let Err(e) = wasm_bindgen_futures::JsFuture::from(p).await {
                        *err_cb.borrow_mut() = Some(format!("AudioContext.resume rejected: {e:?}"));
                        starting_cb.set(false);
                        return;
                    }
                }
                Err(e) => {
                    *err_cb.borrow_mut() = Some(format!("AudioContext.resume threw: {e:?}"));
                    starting_cb.set(false);
                    return;
                }
            }
            *slot_cb.borrow_mut() = Some(AudioHandle {
                _ctx: ctx_clone,
                _node: node,
                _on_message: on_message,
            });
            // Clear the in-flight flag after the handle is published.
            // Subsequent `start_audio()` calls early-return on
            // `audio.is_some()` so the flag is just for the
            // not-yet-resolved window.
            starting_cb.set(false);
        });

        Ok(sr)
    }

    /// Per-stream DSP state owned by the audio render closure. Lives
    /// behind `Rc<RefCell<…>>` so the JS-thread `onmessage` Closure can
    /// hold it without `&mut self` plumbing.
    struct RenderState {
        reverb: Reverb,
        sympathetic: SympatheticBank,
        mono: Vec<f32>,
        mono_compressed: Vec<f32>,
        limiter_gain: f32,
        /// Scratch buffers for the rustysynth `render(left, right)` call
        /// when `Engine::SfPiano` is active. Held in `RenderState` so
        /// they're reused across audio blocks instead of allocated
        /// per-callback (the audio thread can't tolerate hidden heap
        /// traffic). Resized lazily to match the current chunk size.
        sf_left: Vec<f32>,
        sf_right: Vec<f32>,
    }

    /// Render `frames` mono samples directly into `state.mono`. Same
    /// DSP graph as the prior `audio_render` (voice mix → sym bank →
    /// denormal flush → reverb → mix-mode tanh) minus the per-channel
    /// mirroring that the worklet does on the JS side. Caller reads
    /// the result from `state.mono` after this returns; no per-
    /// request `Vec` allocation.
    fn render_mono_chunk(
        voices: &Arc<Mutex<Vec<Voice>>>,
        synth: &SharedSynth,
        master: f32,
        reverb_wet: f32,
        engine: Engine,
        mix_mode: MixMode,
        state: &mut RenderState,
        frames: usize,
    ) {
        let mono = &mut state.mono;
        if mono.len() != frames {
            mono.resize(frames, 0.0);
        } else {
            mono.fill(0.0);
        }

        // Mix the rustysynth SF2 voice into the bus when active. Same
        // downmix-to-mono shape as native `audio_callback` (sf_left +
        // sf_right) * 0.5. Always render so the synth's internal voice
        // envelopes keep advancing even on engine switches; the
        // contribution is only added when SfPiano is the selected
        // engine so non-SF voices don't get a phantom piano bus.
        {
            let mut guard = synth.lock().unwrap();
            if let Some(s) = guard.as_mut() {
                if state.sf_left.len() != frames {
                    state.sf_left.resize(frames, 0.0);
                    state.sf_right.resize(frames, 0.0);
                } else {
                    state.sf_left.fill(0.0);
                    state.sf_right.fill(0.0);
                }
                s.render(state.sf_left.as_mut_slice(), state.sf_right.as_mut_slice());
                if engine == Engine::SfPiano {
                    for i in 0..frames {
                        mono[i] += (state.sf_left[i] + state.sf_right[i]) * 0.5;
                    }
                }
            }
        }

        {
            let mut pool = voices.lock().unwrap();
            for v in pool.iter_mut() {
                v.inner.render_add(mono.as_mut_slice());
            }
            pool.retain(|v| !v.inner.is_done());
        }

        if engine.is_piano_family() {
            const COUPLING: f32 = 0.0002;
            const MIX: f32 = 0.3;
            for sample in mono.iter_mut() {
                let drive = *sample;
                let sym_out = state.sympathetic.process(drive, COUPLING);
                *sample += sym_out * MIX;
            }
        } else {
            for _ in 0..mono.len() {
                let _ = state.sympathetic.process(0.0, 0.0);
            }
        }

        // NaN/Inf guard + denormal flush.
        for s in mono.iter_mut() {
            if !s.is_finite() {
                *s = 0.0;
            } else if s.abs() < 1e-30 {
                *s = 0.0;
            }
        }

        if reverb_wet > 0.0 {
            state.reverb.process(mono.as_mut_slice(), reverb_wet);
        }

        match mix_mode {
            MixMode::Plain => {
                for sample in mono.iter_mut() {
                    *sample = (*sample * master).tanh();
                }
            }
            MixMode::Limiter => {
                const ATTACK: f32 = 0.5;
                const RELEASE: f32 = 0.0001;
                for sample in mono.iter_mut() {
                    let abs_s = sample.abs();
                    if abs_s > state.limiter_gain {
                        state.limiter_gain += (abs_s - state.limiter_gain) * ATTACK;
                    } else {
                        state.limiter_gain += (abs_s - state.limiter_gain) * RELEASE;
                    }
                    let gr = if state.limiter_gain > 1.0 {
                        1.0 / state.limiter_gain
                    } else {
                        1.0
                    };
                    *sample = (*sample * gr * master).tanh();
                }
            }
            MixMode::ParallelComp => {
                const ALPHA: f32 = 0.7;
                const BETA: f32 = 0.6;
                const ATTACK: f32 = 0.5;
                const RELEASE: f32 = 0.0001;
                let mono_compressed = &mut state.mono_compressed;
                if mono_compressed.len() != mono.len() {
                    mono_compressed.resize(mono.len(), 0.0);
                }
                for (i, sample) in mono.iter().enumerate() {
                    let abs_s = sample.abs();
                    if abs_s > state.limiter_gain {
                        state.limiter_gain += (abs_s - state.limiter_gain) * ATTACK;
                    } else {
                        state.limiter_gain += (abs_s - state.limiter_gain) * RELEASE;
                    }
                    let gr = if state.limiter_gain > 1.0 {
                        1.0 / state.limiter_gain
                    } else {
                        1.0
                    };
                    mono_compressed[i] = sample * gr;
                }
                for (i, sample) in mono.iter_mut().enumerate() {
                    let combined = (*sample * ALPHA + mono_compressed[i] * BETA) * master;
                    *sample = combined.tanh();
                }
            }
        }
        // Caller reads from `state.mono`; no copy back needed since we
        // rendered in place.
    }

    // ===========================================================================
    // wasm-bindgen entry. The outer real `main` (above the `mod imp` wrapper)
    // dispatches into `imp::start`; we install the panic hook and hand control
    // to eframe's WebRunner from there.
    // ===========================================================================

    pub fn start() {
        console_error_panic_hook::set_once();

        let runner = eframe::WebRunner::new();
        wasm_bindgen_futures::spawn_local(async move {
            let canvas = pick_canvas("keysynth-canvas");
            let result = runner
                .start(
                    canvas,
                    eframe::WebOptions::default(),
                    Box::new(|_cc| Ok(Box::new(WebApp::default()))),
                )
                .await;
            if let Err(e) = result {
                web_sys::console::error_1(&format!("eframe start failed: {e:?}").into());
            }
        });
    }

    // ---------------------------------------------------------------------
    // Web MIDI handshake
    // ---------------------------------------------------------------------

    /// Fetch the SF2 bytes from `SF2_URL` (a static asset trunk copies
    /// next to the wasm), parse with rustysynth at `sr` Hz, set the
    /// default GM program, and install the resulting `Synthesizer`
    /// into the shared `synth` slot. All async + non-blocking, so the
    /// page remains responsive across the multi-second download.
    /// Returns the byte length on success so the caller can log a
    /// confirmation; on failure returns a stringified error.
    async fn fetch_and_load_sf2(synth: SharedSynth, sr: u32) -> Result<usize, String> {
        let window = web_sys::window().ok_or_else(|| "no window".to_string())?;
        let resp_value = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(SF2_URL))
            .await
            .map_err(|e| format!("fetch({SF2_URL}): {e:?}"))?;
        let resp: web_sys::Response = resp_value
            .dyn_into()
            .map_err(|_| "fetch result was not a Response".to_string())?;
        if !resp.ok() {
            return Err(format!("HTTP {}", resp.status()));
        }
        let buffer_promise = resp
            .array_buffer()
            .map_err(|e| format!("response.array_buffer: {e:?}"))?;
        let buffer = wasm_bindgen_futures::JsFuture::from(buffer_promise)
            .await
            .map_err(|e| format!("array_buffer await: {e:?}"))?;
        let array = js_sys::Uint8Array::new(&buffer);
        let bytes: Vec<u8> = array.to_vec();
        let n = bytes.len();
        let sf = SoundFont::new(&mut Cursor::new(&bytes[..]))
            .map_err(|e| format!("SoundFont parse: {e:?}"))?;
        let mut settings = SynthesizerSettings::new(sr as i32);
        settings.maximum_polyphony = 256;
        let mut s = Synthesizer::new(&Arc::new(sf), &settings)
            .map_err(|e| format!("Synthesizer::new: {e:?}"))?;
        // Default to GM program 0 (Acoustic Grand Piano), bank 0. Same
        // initial state the native `--engine sf-piano` lands in when
        // `--sf2-program`/`--sf2-bank` are left at their defaults.
        s.process_midi_message(0, 0xC0, 0, 0);
        *synth.lock().unwrap() = Some(s);
        Ok(n)
    }

    /// Request MIDI access from the browser, attach an `onmidimessage`
    /// handler to every input port, and set up `onstatechange` on the
    /// `MidiAccess` so devices plugged in later also start producing
    /// events. Returns a `MidiHandles` blob the caller must keep alive
    /// for the lifetime of the page (the browser only stores function
    /// references; if Rust drops the underlying `Closure`s the
    /// references go dangling and MIDI silently stops).
    async fn request_midi_access(inbox: MidiInbox) -> Result<MidiHandles, String> {
        let window = web_sys::window().ok_or_else(|| "no window".to_string())?;
        let nav: web_sys::Navigator = window.navigator();
        // `requestMIDIAccess` is technically optional on the Navigator
        // interface (Firefox without the flag returns undefined). Reach
        // it via `Reflect::get` so we get a proper error message instead
        // of a static-link compile dependency on the typed binding.
        let request_fn = js_sys::Reflect::get(&nav, &JsValue::from_str("requestMIDIAccess"))
            .map_err(|_| "navigator.requestMIDIAccess unavailable".to_string())?;
        if request_fn.is_undefined() || request_fn.is_null() {
            return Err("Web MIDI not supported in this browser".to_string());
        }
        let request_fn: js_sys::Function = request_fn
            .dyn_into()
            .map_err(|_| "requestMIDIAccess not callable".to_string())?;
        let promise: js_sys::Promise = request_fn
            .call0(&nav)
            .map_err(|e| format!("requestMIDIAccess threw: {e:?}"))?
            .dyn_into()
            .map_err(|_| "requestMIDIAccess didn't return a Promise".to_string())?;
        let access_jsv = wasm_bindgen_futures::JsFuture::from(promise)
            .await
            .map_err(|e| format!("MIDI access denied: {e:?}"))?;
        let access: web_sys::MidiAccess = access_jsv
            .dyn_into()
            .map_err(|_| "MIDI access result wasn't a MIDIAccess".to_string())?;

        // Shared registry every onmidimessage closure pushes into —
        // both the initial walk below and the hot-plug `onstatechange`
        // closure further down. Keeping one named owner avoids the
        // per-reconnect leak the original `Box::leak` path produced.
        let closures: MessageClosures = Rc::new(RefCell::new(Vec::new()));

        // Walk every current input and attach a message handler. Use
        // `inputs.values()` rather than `entries()` so we get
        // `MidiInput` directly without destructuring `[key, value]`.
        let inputs = access.inputs();
        let values: js_sys::Iterator = inputs.values();
        loop {
            let next = values
                .next()
                .map_err(|e| format!("MIDIInputMap.values iter failed: {e:?}"))?;
            if next.done() {
                break;
            }
            let port: web_sys::MidiInput = match next.value().dyn_into() {
                Ok(p) => p,
                Err(_) => continue,
            };
            attach_midi_handler(&port, inbox.clone(), &closures);
        }

        // Hot-plug: handle devices that connect after the access grant.
        // Capture clones of the inbox and the shared closure registry so
        // a new `Connected` event can attach a handler and store its
        // closure inside the same `MidiHandles` we hand back below.
        let inbox_state = inbox.clone();
        let closures_state = closures.clone();
        let on_state_change = Closure::<dyn FnMut(web_sys::MidiConnectionEvent)>::new(
            move |ev: web_sys::MidiConnectionEvent| {
                let Some(port) = ev.port() else { return };
                if port.type_() != web_sys::MidiPortType::Input {
                    return;
                }
                if port.state() != web_sys::MidiPortDeviceState::Connected {
                    return;
                }
                let Ok(input) = port.dyn_into::<web_sys::MidiInput>() else {
                    return;
                };
                attach_midi_handler(&input, inbox_state.clone(), &closures_state);
            },
        );
        access.set_onstatechange(Some(on_state_change.as_ref().unchecked_ref()));

        Ok(MidiHandles {
            _access: access,
            _closures: closures,
            _on_state_change: on_state_change,
        })
    }

    /// Wire one MIDI input port: set its `onmidimessage` to a closure
    /// that parses the 3-byte status message and pushes a `MidiMsg` onto
    /// the inbox. Pushes the created `Closure` into the shared
    /// `MessageClosures` registry so the caller keeps it alive — letting
    /// it drop here would dangle the browser's stored function reference.
    fn attach_midi_handler(
        port: &web_sys::MidiInput,
        inbox: MidiInbox,
        closures: &MessageClosures,
    ) {
        // Diagnostic: log the port name once at attach time so we can
        // confirm the handler actually got bound to a real input. Cheap
        // — fires only when a port is added.
        web_sys::console::log_1(
            &format!(
                "keysynth-web: attach_midi_handler port='{}' state={:?}",
                port.name().unwrap_or_default(),
                port.state()
            )
            .into(),
        );
        let on_message = Closure::<dyn FnMut(web_sys::MidiMessageEvent)>::new(
            move |ev: web_sys::MidiMessageEvent| {
                let Ok(data) = ev.data() else { return };
                if data.len() < 2 {
                    return;
                }
                let status = data[0];
                let kind = status & 0xF0;
                let note = data[1];
                // Velocity is data[2] for note_on/off; absent on some
                // status types but we only branch on 0x80 / 0x90 below.
                let velocity = if data.len() >= 3 { data[2] } else { 0 };
                // Diagnostic: log raw bytes ONLY for note_on / note_off.
                // Logging every inbound message swamps the browser
                // console with MIDI clock (24 msg/quarter ≈ 48 msg/s
                // at 120 BPM) and active-sensing chatter from
                // sequencer-style controllers, jankifying the UI even
                // though the audio thread is unaffected. Keep the
                // signal-of-interest visible without spam. Strip once
                // we've confirmed MIDI is wired end-to-end on the live
                // demo across enough device models.
                if matches!(kind, 0x80 | 0x90) {
                    web_sys::console::log_1(
                        &format!("keysynth-web: midi raw={:02X?}", data.as_slice()).into(),
                    );
                }
                let msg = match kind {
                    0x90 if velocity > 0 => MidiMsg::NoteOn { note, velocity },
                    0x80 | 0x90 => MidiMsg::NoteOff { note },
                    _ => return,
                };
                inbox.borrow_mut().push(msg);
            },
        );
        port.set_onmidimessage(Some(on_message.as_ref().unchecked_ref()));
        closures.borrow_mut().push(on_message);
    }

    fn pick_canvas(id: &str) -> web_sys::HtmlCanvasElement {
        let document = web_sys::window()
            .expect("no window")
            .document()
            .expect("no document");
        document
            .get_element_by_id(id)
            .unwrap_or_else(|| panic!("missing <canvas id=\"{id}\">"))
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("element is not a canvas")
    }
} // mod imp
