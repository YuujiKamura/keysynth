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
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};

    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::StreamConfig;
    use eframe::egui;
    use wasm_bindgen::closure::Closure;
    use wasm_bindgen::{JsCast, JsValue};

    use keysynth::reverb::{self, Reverb};
    use keysynth::sympathetic::SympatheticBank;
    use keysynth::synth::{
        make_voice, midi_to_freq, Engine, LiveParams, MixMode, ModalLut, Voice, MODAL_LUT,
    };

    // Engines exposed to the web UI. SfPiano / SfzPiano deliberately omitted —
    // they need a real SoundFont / SFZ library on disk that we don't ship to
    // GitHub Pages.
    const ENGINES_FOR_WEB: &[(Engine, &str)] = &[
        (Engine::PianoModal, "piano-modal"),
        (Engine::PianoLite, "piano-lite"),
        (Engine::PianoThick, "piano-thick"),
        (Engine::Piano, "piano"),
        (Engine::Piano5AM, "piano-5am"),
        (Engine::Koto, "koto"),
        (Engine::KsRich, "ks-rich"),
        (Engine::Ks, "ks"),
        (Engine::Sub, "sub"),
        (Engine::Fm, "fm"),
        (Engine::Square, "square"),
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

    // Hold the cpal stream so it isn't dropped (which kills audio). Stream is
    // not Send/Sync on wasm32 single-threaded, but `WebApp` only ever lives on
    // the main JS thread, so plain ownership is fine.
    struct AudioHandle {
        _stream: cpal::Stream,
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
        audio: Option<AudioHandle>,
        audio_err: Option<String>,
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
                audio: None,
                audio_err: None,
                mouse_down_note: None,
                pc_held: 0,
                base_note: 48, // C3
                midi_status: Rc::new(RefCell::new("not requested".to_string())),
                midi_inbox: Rc::new(RefCell::new(Vec::new())),
                midi_requested: Rc::new(std::cell::Cell::new(false)),
                midi_handles: Rc::new(RefCell::new(None)),
            }
        }
    }

    impl WebApp {
        fn start_audio(&mut self) {
            if self.audio.is_some() {
                return;
            }
            match build_audio(self.voices.clone(), self.live.clone()) {
                Ok((handle, sr)) => {
                    self.sample_rate = sr;
                    self.audio = Some(handle);
                    self.audio_err = None;
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
                }
                Err(e) => {
                    web_sys::console::error_1(
                        &format!("keysynth-web: audio start failed: {e}").into(),
                    );
                    self.audio_err = Some(e);
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
            if self.audio.is_none() || self.sample_rate == 0 {
                return;
            }
            let engine = self.live.lock().unwrap().engine;
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
            let mut pool = self.voices.lock().unwrap();
            if let Some(slot) = pool.iter_mut().find(|x| x.key == (0, note)) {
                slot.inner.trigger_release();
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
            // Continuous repaint so the on-screen keyboard reflects PC-key
            // and audio state immediately.
            ctx.request_repaint();

            if self.audio.is_some() {
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

            egui::TopBottomPanel::top("top").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("keysynth (web)");
                    ui.separator();
                    if self.audio.is_some() {
                        ui.label(format!("audio @ {} Hz", self.sample_rate));
                    } else if let Some(err) = &self.audio_err {
                        ui.colored_label(egui::Color32::RED, format!("audio error: {err}"));
                        if ui.button("retry").clicked() {
                            self.start_audio();
                        }
                    } else {
                        // Big, coloured action button so first-time
                        // visitors can't miss it. Browser autoplay
                        // policy refuses to start the AudioContext
                        // until they click.
                        let resp = ui.add(
                            egui::Button::new(
                                egui::RichText::new("▶ Start audio")
                                    .size(18.0)
                                    .color(egui::Color32::WHITE)
                                    .strong(),
                            )
                            .fill(egui::Color32::from_rgb(40, 120, 200))
                            .min_size(egui::vec2(140.0, 28.0)),
                        );
                        if resp.clicked() {
                            self.start_audio();
                        }
                    }
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
                        // Same prominent treatment as Start audio so
                        // it's clear MIDI input also needs an explicit
                        // user gesture (browser permission prompt).
                        // After an error this stays visible alongside
                        // the red error label so the user can retry.
                        let label = if is_error {
                            "🎹 Retry MIDI"
                        } else {
                            "🎹 Connect MIDI keyboard"
                        };
                        let resp = ui.add(
                            egui::Button::new(
                                egui::RichText::new(label)
                                    .size(16.0)
                                    .color(egui::Color32::WHITE)
                                    .strong(),
                            )
                            .fill(egui::Color32::from_rgb(180, 90, 40))
                            .min_size(egui::vec2(220.0, 28.0)),
                        );
                        if resp.clicked() {
                            // Browser autoplay-style gating:
                            // requestMIDIAccess also wants a user
                            // gesture to surface the permission prompt.
                            self.request_midi();
                        }
                    }
                });
                // One-line hint below the action row so first-time
                // visitors know which buttons do what without having
                // to read the source.
                if self.audio.is_none() || !self.midi_requested.get() {
                    ui.horizontal(|ui| {
                        ui.add_space(4.0);
                        let mut hints: Vec<&str> = Vec::new();
                        if self.audio.is_none() {
                            hints.push(
                                "「▶ Start audio」を押すと音が出ます（USB-MIDI 鍵盤も同時に有効化されます）",
                            );
                        }
                        if !self.midi_requested.get() && self.audio.is_some() {
                            hints.push(
                                "「🎹 Retry MIDI」で USB-MIDI 鍵盤入力を再要求",
                            );
                        }
                        ui.colored_label(egui::Color32::from_gray(170), hints.join("　／　"));
                    });
                }
            });

            egui::TopBottomPanel::top("controls").show(ctx, |ui| {
                let mut live = self.live.lock().unwrap();
                ui.horizontal(|ui| {
                    ui.label("engine:");
                    let current_label = ENGINES_FOR_WEB
                        .iter()
                        .find(|(e, _)| *e == live.engine)
                        .map(|(_, l)| *l)
                        .unwrap_or("?");
                    egui::ComboBox::from_id_salt("engine")
                        .selected_text(current_label)
                        .show_ui(ui, |ui| {
                            for (eng, label) in ENGINES_FOR_WEB {
                                ui.selectable_value(&mut live.engine, *eng, *label);
                            }
                        });

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
                    ui.label("PC keyboard: zsxdcvgbhnjm = lower octave, qweryt... = upper");
                });
            });

            egui::CentralPanel::default().show(ctx, |ui| {
                ui.add_space(12.0);
                self.draw_on_screen_keyboard(ui);
                ui.add_space(20.0);
                ui.collapsing("about", |ui| {
                    ui.label(
                        "keysynth — pure-Rust real-time modelling synth. The web build \
                     drops SF2/SFZ sample players and live MIDI input; the modelling \
                     engines (Karplus-Strong, modal piano, FM, etc.) run unchanged.",
                    );
                    ui.label("source: https://github.com/YuujiKamura/keysynth");
                });
            });
        }
    }

    // ===========================================================================
    // Audio thread setup
    // ===========================================================================

    fn build_audio(
        voices: Arc<Mutex<Vec<Voice>>>,
        live: Arc<Mutex<LiveParams>>,
    ) -> Result<(AudioHandle, u32), String> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "no default output device".to_string())?;
        let supported = device
            .default_output_config()
            .map_err(|e| format!("default output config: {e}"))?;
        let sr = supported.sample_rate().0;
        let channels = supported.channels();
        let stream_cfg: StreamConfig = supported.into();

        let ir = reverb::synthetic_body_ir(sr);
        let mut reverb = Reverb::new(ir);
        let mut sympathetic = SympatheticBank::new_piano(sr as f32);
        let mut mono: Vec<f32> = Vec::new();
        let mut mono_compressed: Vec<f32> = Vec::new();
        let mut limiter_gain: f32 = 1.0;

        let voices_cb = voices.clone();
        let live_cb = live.clone();

        let stream = device
            .build_output_stream::<f32, _, _>(
                &stream_cfg,
                move |out: &mut [f32], _info| {
                    let (master, wet, engine, mix_mode) = {
                        let lp = live_cb.lock().unwrap();
                        (lp.master, lp.reverb_wet, lp.engine, lp.mix_mode)
                    };
                    audio_render(
                        out,
                        channels,
                        &voices_cb,
                        master,
                        wet,
                        &mut mono,
                        &mut mono_compressed,
                        &mut reverb,
                        &mut sympathetic,
                        &mut limiter_gain,
                        engine,
                        mix_mode,
                    );
                },
                move |err| {
                    web_sys::console::error_1(&format!("keysynth-web: stream error: {err}").into());
                },
                None,
            )
            .map_err(|e| format!("build_output_stream: {e}"))?;

        stream.play().map_err(|e| format!("stream.play: {e}"))?;
        Ok((AudioHandle { _stream: stream }, sr))
    }

    #[allow(clippy::too_many_arguments)]
    fn audio_render(
        out: &mut [f32],
        channels: u16,
        voices: &Arc<Mutex<Vec<Voice>>>,
        master: f32,
        reverb_wet: f32,
        mono: &mut Vec<f32>,
        mono_compressed: &mut Vec<f32>,
        reverb: &mut Reverb,
        sympathetic: &mut SympatheticBank,
        limiter_gain: &mut f32,
        engine: Engine,
        mix_mode: MixMode,
    ) {
        let frames = out.len() / channels as usize;
        if mono.len() != frames {
            mono.resize(frames, 0.0);
        } else {
            mono.fill(0.0);
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
                let sym_out = sympathetic.process(drive, COUPLING);
                *sample += sym_out * MIX;
            }
        } else {
            for _ in 0..mono.len() {
                let _ = sympathetic.process(0.0, 0.0);
            }
        }

        // NaN/Inf guard + denormal flush. wasm32 has no equivalent of
        // x86 SSE FZ/DAZ, so high-Q resonator state (PianoModal's 96
        // bandpass biquads decay exponentially toward zero) eventually
        // crosses ~1e-38 into f32 denormal range. Native flushes those
        // via MXCSR; on wasm we have to do it by hand or the per-sample
        // arithmetic falls off a cliff (100×–1000× slower depending on
        // the runtime), busting the audio deadline and producing
        // intermittent or full silence specifically on PianoModal.
        for s in mono.iter_mut() {
            if !s.is_finite() {
                *s = 0.0;
            } else if s.abs() < 1e-30 {
                *s = 0.0;
            }
        }

        // Reverb is direct convolution against a ~6600-sample IR
        // (synthetic body @ 44.1 kHz). At 1024-frame buffers that's
        // ~6.7 M MACs per audio callback. On wasm32 (no SIMD) this
        // alone can dominate the per-callback budget. Skip when fully
        // dry so users with limited CPU headroom still get clean
        // PianoModal output; flip back on by raising the slider.
        if reverb_wet > 0.0 {
            reverb.process(mono.as_mut_slice(), reverb_wet);
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
                    if abs_s > *limiter_gain {
                        *limiter_gain += (abs_s - *limiter_gain) * ATTACK;
                    } else {
                        *limiter_gain += (abs_s - *limiter_gain) * RELEASE;
                    }
                    let gr = if *limiter_gain > 1.0 {
                        1.0 / *limiter_gain
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
                if mono_compressed.len() != mono.len() {
                    mono_compressed.resize(mono.len(), 0.0);
                }
                for (i, sample) in mono.iter().enumerate() {
                    let abs_s = sample.abs();
                    if abs_s > *limiter_gain {
                        *limiter_gain += (abs_s - *limiter_gain) * ATTACK;
                    } else {
                        *limiter_gain += (abs_s - *limiter_gain) * RELEASE;
                    }
                    let gr = if *limiter_gain > 1.0 {
                        1.0 / *limiter_gain
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

        for (frame_idx, frame) in out.chunks_mut(channels as usize).enumerate() {
            let s = mono[frame_idx];
            for slot in frame.iter_mut() {
                *slot = s;
            }
        }
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
