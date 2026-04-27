//! WASM / web entry point for the GitHub Pages demo build.
//!
//! Minimalist UI version: focuses on instrument selection, volume, and
//! the on-screen keyboard.

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

    use eframe::egui;
    use wasm_bindgen::closure::Closure;
    use wasm_bindgen::{JsCast, JsValue};

    use keysynth::audio_kira::KiraEngine;
    use keysynth::synth::{Engine, LiveParams, MixMode, ModalLut, ModalPreset, MODAL_LUT};

    #[derive(Clone, Copy, Debug)]
    enum MidiMsg {
        NoteOn { note: u8, velocity: u8 },
        NoteOff { note: u8 },
    }

    type MidiInbox = Rc<RefCell<Vec<MidiMsg>>>;

    type MessageClosures = Rc<RefCell<Vec<Closure<dyn FnMut(web_sys::MidiMessageEvent)>>>>;

    struct MidiHandles {
        _access: web_sys::MidiAccess,
        _closures: MessageClosures,
        _on_state_change: Closure<dyn FnMut(web_sys::MidiConnectionEvent)>,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum VoiceCategory {
        Piano,
        Synth,
    }

    impl VoiceCategory {
        const ALL: &'static [VoiceCategory] = &[VoiceCategory::Piano, VoiceCategory::Synth];

        fn label(self) -> &'static str {
            match self {
                VoiceCategory::Piano => "Piano",
                VoiceCategory::Synth => "Other (synth)",
            }
        }
    }

    struct WebVoiceSlot {
        label: &'static str,
        category: VoiceCategory,
        engine: Engine,
        modal_preset: Option<ModalPreset>,
    }

    const WEB_VOICE_SLOTS: &[WebVoiceSlot] = &[
        WebVoiceSlot {
            label: "Modal (default)",
            category: VoiceCategory::Piano,
            engine: Engine::PianoModal,
            modal_preset: Some(ModalPreset::Default),
        },
        WebVoiceSlot {
            label: "Modal (Round-16)",
            category: VoiceCategory::Piano,
            engine: Engine::PianoModal,
            modal_preset: Some(ModalPreset::Round16),
        },
        WebVoiceSlot {
            label: "Modal (Physics)",
            category: VoiceCategory::Piano,
            engine: Engine::PianoModal,
            modal_preset: Some(ModalPreset::Physics),
        },
        WebVoiceSlot {
            label: "Modal (Bright)",
            category: VoiceCategory::Piano,
            engine: Engine::PianoModal,
            modal_preset: Some(ModalPreset::Bright),
        },
        WebVoiceSlot {
            label: "KS Piano",
            category: VoiceCategory::Piano,
            engine: Engine::Piano,
            modal_preset: None,
        },
        WebVoiceSlot {
            label: "KS Piano (thick)",
            category: VoiceCategory::Piano,
            engine: Engine::PianoThick,
            modal_preset: None,
        },
        WebVoiceSlot {
            label: "KS Piano (lite)",
            category: VoiceCategory::Piano,
            engine: Engine::PianoLite,
            modal_preset: None,
        },
        WebVoiceSlot {
            label: "KS Piano (5AM)",
            category: VoiceCategory::Piano,
            engine: Engine::Piano5AM,
            modal_preset: None,
        },
        WebVoiceSlot {
            label: "Square",
            category: VoiceCategory::Synth,
            engine: Engine::Square,
            modal_preset: None,
        },
        WebVoiceSlot {
            label: "KS pluck",
            category: VoiceCategory::Synth,
            engine: Engine::Ks,
            modal_preset: None,
        },
        WebVoiceSlot {
            label: "KS rich",
            category: VoiceCategory::Synth,
            engine: Engine::KsRich,
            modal_preset: None,
        },
        WebVoiceSlot {
            label: "Sub (subtractive)",
            category: VoiceCategory::Synth,
            engine: Engine::Sub,
            modal_preset: None,
        },
        WebVoiceSlot {
            label: "FM bell",
            category: VoiceCategory::Synth,
            engine: Engine::Fm,
            modal_preset: None,
        },
        WebVoiceSlot {
            label: "Koto",
            category: VoiceCategory::Synth,
            engine: Engine::Koto,
            modal_preset: None,
        },
    ];

    const PC_KEYMAP: &[(egui::Key, i32)] = &[
        (egui::Key::Z, 0),
        (egui::Key::S, 1),
        (egui::Key::X, 2),
        (egui::Key::D, 3),
        (egui::Key::C, 4),
        (egui::Key::V, 5),
        (egui::Key::G, 6),
        (egui::Key::B, 7),
        (egui::Key::H, 8),
        (egui::Key::N, 9),
        (egui::Key::J, 10),
        (egui::Key::M, 11),
        (egui::Key::Q, 12),
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

    const KEYBOARD_SPAN: u8 = 25;

    struct WebApp {
        kira: Option<KiraEngine>,
        live: Arc<Mutex<LiveParams>>,
        notes_held: [bool; 128],
        audio_err: Rc<RefCell<Option<String>>>,
        mouse_down_note: Option<u8>,
        pc_held: u128,
        prev_focused: bool,
        base_note: u8,
        midi_inbox: MidiInbox,
        midi_requested: Rc<std::cell::Cell<bool>>,
        midi_handles: Rc<RefCell<Option<MidiHandles>>>,
        current_voice_idx: Rc<std::cell::Cell<usize>>,
    }

    impl Default for WebApp {
        fn default() -> Self {
            let (lut, source) = ModalLut::auto_load(None);
            let _ = MODAL_LUT.set(lut);
            web_sys::console::log_1(&format!("keysynth-web: modal LUT source = {source}").into());

            Self {
                kira: None,
                live: Arc::new(Mutex::new(LiveParams {
                    master: 1.0,
                    engine: Engine::PianoModal,
                    reverb_wet: 0.3,
                    sf_program: 0,
                    sf_bank: 0,
                    mix_mode: MixMode::ParallelComp,
                })),
                notes_held: [false; 128],
                audio_err: Rc::new(RefCell::new(None)),
                mouse_down_note: None,
                pc_held: 0,
                prev_focused: true,
                base_note: 48,
                midi_inbox: Rc::new(RefCell::new(Vec::new())),
                midi_requested: Rc::new(std::cell::Cell::new(false)),
                midi_handles: Rc::new(RefCell::new(None)),
                current_voice_idx: Rc::new(std::cell::Cell::new(0)),
            }
        }
    }

    impl WebApp {
        fn start_audio(&mut self) {
            if self.kira.is_some() {
                return;
            }
            *self.audio_err.borrow_mut() = None;
            match KiraEngine::new() {
                Ok(kira) => {
                    self.kira = Some(kira);
                    self.request_midi();
                }
                Err(e) => {
                    *self.audio_err.borrow_mut() = Some(e);
                }
            }
        }

        fn drain_midi_inbox(&mut self) {
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

        fn request_midi(&mut self) {
            if self.midi_requested.get() {
                return;
            }
            self.midi_requested.set(true);
            let inbox = self.midi_inbox.clone();
            let requested_flag = self.midi_requested.clone();
            let handles_slot = self.midi_handles.clone();
            wasm_bindgen_futures::spawn_local(async move {
                if let Ok(handles) = request_midi_access(inbox).await {
                    *handles_slot.borrow_mut() = Some(handles);
                } else {
                    requested_flag.set(false);
                }
            });
        }

        fn note_on(&mut self, note: u8, velocity: u8) {
            let Some(kira) = self.kira.as_mut() else {
                return;
            };
            let engine = self.live.lock().unwrap().engine;
            if engine == Engine::SfPiano {
                return;
            }
            kira.note_on(engine, note, velocity);
            self.notes_held[note as usize] = true;
        }

        fn note_off(&mut self, note: u8) {
            let Some(kira) = self.kira.as_mut() else {
                return;
            };
            kira.note_off(note);
            self.notes_held[note as usize] = false;
        }

        fn panic_release(&mut self) {
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
            for note in 0..128 {
                if self.notes_held[note] {
                    self.note_off(note as u8);
                }
            }
        }

        fn handle_pc_keyboard(&mut self, ctx: &egui::Context) {
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

        fn apply_voice_slot(&mut self, idx: usize, slot: &WebVoiceSlot) {
            {
                let mut lp = self.live.lock().unwrap();
                lp.engine = slot.engine;
            }
            if let Some(preset) = slot.modal_preset {
                preset.apply();
                if let Some(kira) = self.kira.as_mut() {
                    kira.set_modal_preset(preset);
                }
            }
            if let Some(kira) = self.kira.as_mut() {
                kira.set_engine(slot.engine);
            }
            self.current_voice_idx.set(idx);
        }

        fn draw_voice_browser(&mut self, ui: &mut egui::Ui) {
            let mut clicked_slot: Option<usize> = None;
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for cat in VoiceCategory::ALL {
                        egui::CollapsingHeader::new(cat.label())
                            .default_open(true)
                            .show(ui, |ui| {
                                let active_idx = self.current_voice_idx.get();
                                for (idx, slot) in WEB_VOICE_SLOTS.iter().enumerate() {
                                    if slot.category != *cat {
                                        continue;
                                    }
                                    let is_active = idx == active_idx;
                                    ui.horizontal(|ui| {
                                        ui.label(if is_active { "●" } else { "○" });
                                        let rich = if is_active {
                                            egui::RichText::new(slot.label)
                                                .monospace()
                                                .strong()
                                                .color(egui::Color32::from_rgb(255, 220, 120))
                                        } else {
                                            egui::RichText::new(slot.label).monospace()
                                        };
                                        if ui.selectable_label(is_active, rich).clicked() {
                                            clicked_slot = Some(idx);
                                        }
                                    });
                                }
                            });
                    }
                });
            if let Some(idx) = clicked_slot {
                self.apply_voice_slot(idx, &WEB_VOICE_SLOTS[idx]);
            }
        }

        fn draw_on_screen_keyboard(&mut self, ui: &mut egui::Ui) {
            let span = KEYBOARD_SPAN as i32;
            let total_w = ui.available_width().min(720.0);
            let key_h = 100.0;
            let white_w = total_w / 15.0;
            let black_w = white_w * 0.6;
            let black_h = key_h * 0.6;

            let (rect, response) = ui.allocate_exact_size(
                egui::vec2(white_w * 15.0, key_h),
                egui::Sense::click_and_drag(),
            );
            let painter = ui.painter_at(rect);

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
                    let active = self.notes_held[note as usize];
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

            let mut black_rects: Vec<(u8, egui::Rect)> = Vec::with_capacity(10);
            white_idx = 0;
            for semi in 0..span {
                if is_black(semi as u8) {
                    let prev_white_x =
                        rect.left() + (white_idx as f32 - 0.5) * white_w + white_w - black_w * 0.5;
                    let r = egui::Rect::from_min_size(
                        egui::pos2(prev_white_x, rect.top()),
                        egui::vec2(black_w, black_h),
                    );
                    let note = (self.base_note as i32 + semi).clamp(0, 127) as u8;
                    let active = self.notes_held[note as usize];
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

            let pointer_pos = response.interact_pointer_pos();
            let mouse_down = response.is_pointer_button_down_on() && pointer_pos.is_some();
            let hovered_note = pointer_pos.and_then(|p| {
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
    }

    fn is_black(semi: u8) -> bool {
        matches!(semi % 12, 1 | 3 | 6 | 8 | 10)
    }

    impl eframe::App for WebApp {
        fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
            ctx.request_repaint_after(std::time::Duration::from_millis(33));
            let focused_now = ctx.input(|i| i.focused);
            if self.prev_focused && !focused_now {
                self.panic_release();
            }
            self.prev_focused = focused_now;

            if self.kira.is_some() {
                self.handle_pc_keyboard(ctx);
            }
            self.drain_midi_inbox();

            if self.kira.is_none() {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(60.0);
                        ui.heading(egui::RichText::new("keysynth (web)").size(28.0).strong());
                        ui.add_space(12.0);
                        ui.label("Pure-Rust real-time modelling synth.");
                        ui.add_space(40.0);
                        if ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new("▶ Start")
                                        .size(28.0)
                                        .color(egui::Color32::WHITE)
                                        .strong(),
                                )
                                .fill(egui::Color32::from_rgb(40, 120, 200))
                                .min_size(egui::vec2(220.0, 56.0)),
                            )
                            .clicked()
                        {
                            self.start_audio();
                        }
                        if let Some(err) = self.audio_err.borrow().as_ref() {
                            ui.colored_label(
                                egui::Color32::from_rgb(220, 90, 90),
                                format!("audio error: {err}"),
                            );
                        }
                    });
                });
                return;
            }

            egui::SidePanel::left("voice_browser")
                .default_width(220.0)
                .show(ctx, |ui| {
                    ui.heading("Voices");
                    ui.separator();
                    self.draw_voice_browser(ui);
                });

            egui::CentralPanel::default().show(ctx, |ui| {
                let mut live = self.live.lock().unwrap();
                ui.horizontal(|ui| {
                    ui.label("volume:");
                    ui.add(egui::Slider::new(&mut live.master, 0.0..=4.0).step_by(0.05));
                    ui.separator();
                    ui.label("reverb:");
                    ui.add(egui::Slider::new(&mut live.reverb_wet, 0.0..=1.0).step_by(0.01));
                });
                ui.horizontal(|ui| {
                    ui.label("mix:");
                    for m in MixMode::ALL {
                        let selected = live.mix_mode == *m;
                        if ui.selectable_label(selected, m.as_label()).clicked() {
                            live.mix_mode = *m;
                        }
                    }
                });
                drop(live);
                ui.add_space(20.0);
                self.draw_on_screen_keyboard(ui);
            });
        }
    }

    pub fn start() {
        console_error_panic_hook::set_once();
        let runner = eframe::WebRunner::new();
        wasm_bindgen_futures::spawn_local(async move {
            let canvas = pick_canvas("keysynth-canvas");
            let _ = runner
                .start(
                    canvas,
                    eframe::WebOptions::default(),
                    Box::new(|_cc| Ok(Box::new(WebApp::default()))),
                )
                .await;
        });
    }

    async fn request_midi_access(inbox: MidiInbox) -> Result<MidiHandles, String> {
        let window = web_sys::window().ok_or("no window")?;
        let nav = window.navigator();
        let request_fn = js_sys::Reflect::get(&nav, &JsValue::from_str("requestMIDIAccess"))
            .map_err(|_| "no requestMIDIAccess")?;
        if request_fn.is_undefined() {
            return Err("not supported".into());
        }
        let request_fn: js_sys::Function = request_fn.dyn_into().map_err(|_| "not callable")?;
        let promise: js_sys::Promise = request_fn
            .call0(&nav)
            .map_err(|_| "threw")?
            .dyn_into()
            .map_err(|_| "no promise")?;
        let access_jsv = wasm_bindgen_futures::JsFuture::from(promise)
            .await
            .map_err(|_| "denied")?;
        let access: web_sys::MidiAccess = access_jsv.dyn_into().map_err(|_| "wrong type")?;
        let closures: MessageClosures = Rc::new(RefCell::new(Vec::new()));
        let inputs = access.inputs();
        let values: js_sys::Iterator = inputs.values();
        loop {
            let next = values.next().map_err(|_| "iter failed")?;
            if next.done() {
                break;
            }
            if let Ok(port) = next.value().dyn_into::<web_sys::MidiInput>() {
                attach_midi_handler(&port, inbox.clone(), &closures);
            }
        }
        let inbox_state = inbox.clone();
        let closures_state = closures.clone();
        let on_state_change = Closure::<dyn FnMut(web_sys::MidiConnectionEvent)>::new(
            move |ev: web_sys::MidiConnectionEvent| {
                if let Some(port) = ev.port() {
                    if port.type_() == web_sys::MidiPortType::Input
                        && port.state() == web_sys::MidiPortDeviceState::Connected
                    {
                        if let Ok(input) = port.dyn_into::<web_sys::MidiInput>() {
                            attach_midi_handler(&input, inbox_state.clone(), &closures_state);
                        }
                    }
                }
            },
        );
        access.set_onstatechange(Some(on_state_change.as_ref().unchecked_ref()));
        Ok(MidiHandles {
            _access: access,
            _closures: closures,
            _on_state_change: on_state_change,
        })
    }

    fn attach_midi_handler(
        port: &web_sys::MidiInput,
        inbox: MidiInbox,
        closures: &MessageClosures,
    ) {
        let on_message = Closure::<dyn FnMut(web_sys::MidiMessageEvent)>::new(
            move |ev: web_sys::MidiMessageEvent| {
                if let Ok(data) = ev.data() {
                    if data.len() < 2 {
                        return;
                    }
                    let status = data[0];
                    let kind = status & 0xF0;
                    let note = data[1];
                    let velocity = if data.len() >= 3 { data[2] } else { 0 };
                    let msg = match kind {
                        0x90 if velocity > 0 => MidiMsg::NoteOn { note, velocity },
                        0x80 | 0x90 => MidiMsg::NoteOff { note },
                        _ => return,
                    };
                    inbox.borrow_mut().push(msg);
                }
            },
        );
        port.set_onmidimessage(Some(on_message.as_ref().unchecked_ref()));
        closures.borrow_mut().push(on_message);
    }

    fn pick_canvas(id: &str) -> web_sys::HtmlCanvasElement {
        web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .get_element_by_id(id)
            .unwrap()
            .dyn_into()
            .unwrap()
    }
}
