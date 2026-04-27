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

    use std::sync::{Arc, Mutex};

    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::StreamConfig;
    use eframe::egui;
    use wasm_bindgen::JsCast;

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
        /// / error).
        midi_status: String,
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
                midi_status: "not requested".to_string(),
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
                }
                Err(e) => {
                    web_sys::console::error_1(
                        &format!("keysynth-web: audio start failed: {e}").into(),
                    );
                    self.audio_err = Some(e);
                }
            }
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
                    } else if ui.button("Start audio").clicked() {
                        self.start_audio();
                    }
                    ui.separator();
                    ui.label(format!("MIDI: {}", self.midi_status));
                });
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

        for s in mono.iter_mut() {
            if !s.is_finite() {
                *s = 0.0;
            }
        }

        reverb.process(mono.as_mut_slice(), reverb_wet);

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
