//! egui dashboard for keysynth.
//!
//! Goals:
//!   - Visual feedback for every incoming MIDI message
//!   - Knob indicators that follow the MPK mini 3 K1-K8 in real time
//!   - Pad / key highlighting
//!   - Log of recent CC / note events so the user can identify which physical
//!     control sends which CC number
//!
//! Phase 1 is layout-agnostic (just shows whatever CCs/notes arrive). Phase 2
//! will rearrange into an MPK-mini-3 panel-mirror layout once the user has
//! confirmed which CCs the knobs actually send.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use cpal::Stream;
use eframe::egui;
use midir::MidiInputConnection;

use crate::{DashState, Engine, LiveParams};

/// Bundle passed from `main` into `run_app`. Holds the long-lived audio /
/// MIDI handles so they aren't dropped while the GUI is open.
pub struct AppContext {
    pub stream: Stream,
    pub midi_conn: MidiInputConnection<()>,
    pub live: Arc<Mutex<LiveParams>>,
    pub dash: Arc<Mutex<DashState>>,
    pub port_name: String,
    pub out_name: String,
    pub sr_hz: u32,
}

pub fn run_app(ctx: AppContext) -> Result<(), Box<dyn std::error::Error>> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 640.0])
            .with_title("keysynth"),
        ..Default::default()
    };

    eframe::run_native(
        "keysynth",
        native_options,
        Box::new(|_cc| Ok(Box::new(KeysynthApp::new(ctx)))),
    )
    .map_err(|e| -> Box<dyn std::error::Error> { format!("eframe error: {e}").into() })?;

    Ok(())
}

struct KeysynthApp {
    ctx: AppContext,
}

impl KeysynthApp {
    fn new(ctx: AppContext) -> Self {
        Self { ctx }
    }
}

impl eframe::App for KeysynthApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Repaint at ~60 fps so live indicators feel responsive.
        ctx.request_repaint_after(Duration::from_millis(16));

        // Snapshot shared state under brief locks; release before drawing.
        let (mut master, mut engine, dash_snapshot) = {
            let lp = self.ctx.live.lock().unwrap();
            let m = lp.master;
            let e = lp.engine;
            drop(lp);
            let d = self.ctx.dash.lock().unwrap();
            let snap = DashSnapshot {
                cc_raw: d.cc_raw.clone(),
                cc_count: d.cc_count.clone(),
                active_notes: d.active_notes.clone(),
                // Only clone the tail we actually display (newest 60, pre-reversed).
                recent: d.recent.iter().rev().take(60).cloned().collect(),
            };
            (m, e, snap)
        };

        let master_before = master;
        let engine_before = engine;

        egui::TopBottomPanel::top("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("keysynth").strong().size(18.0));
                ui.separator();
                ui.label(format!("MIDI: {}", self.ctx.port_name));
                ui.separator();
                ui.label(format!("Audio: {}", self.ctx.out_name));
                ui.separator();
                ui.label(format!("{} Hz", self.ctx.sr_hz));
            });
            ui.horizontal(|ui| {
                ui.label("master:");
                ui.add(
                    egui::Slider::new(&mut master, 0.0..=3.0)
                        .step_by(0.05)
                        .text("(K1/CC70/CC7)"),
                );
            });
            ui.horizontal_wrapped(|ui| {
                ui.label("engine:");
                for (label, e) in ENGINE_CHOICES {
                    let selected = engine == *e;
                    if ui.selectable_label(selected, *label).clicked() {
                        engine = *e;
                    }
                }
            });
        });

        egui::SidePanel::right("log").default_width(280.0).show(ctx, |ui| {
            ui.heading("MIDI log");
            ui.separator();
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    // Snapshot is already newest-first and capped at 60.
                    for line in dash_snapshot.recent.iter() {
                        ui.monospace(line);
                    }
                });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // ----- CC controls section ----------------------------------
            ui.heading("CC controls (knobs / sliders / pedals)");
            ui.label(
                "Whatever CC numbers arrive show up here. Twist a knob and \
                 watch its bar fill. Numbers shown are the raw MIDI value 0..127.",
            );
            ui.add_space(6.0);

            if dash_snapshot.cc_raw.is_empty() {
                ui.colored_label(
                    egui::Color32::from_rgb(180, 180, 180),
                    "(no CC messages received yet -- twist a knob!)",
                );
            } else {
                let mut ccs: Vec<(u8, u8, u64)> = dash_snapshot
                    .cc_raw
                    .iter()
                    .map(|(k, v)| (*k, *v, *dash_snapshot.cc_count.get(k).unwrap_or(&0)))
                    .collect();
                ccs.sort_by_key(|(k, _, _)| *k);

                egui::Grid::new("cc-grid")
                    .num_columns(4)
                    .striped(true)
                    .spacing([12.0, 4.0])
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("CC#").strong());
                        ui.label(egui::RichText::new("value").strong());
                        ui.label(egui::RichText::new("level").strong());
                        ui.label(egui::RichText::new("count").strong());
                        ui.end_row();

                        for (cc, val, count) in ccs {
                            ui.monospace(format!("CC{cc:>3}"));
                            ui.monospace(format!("{val:>3}"));
                            let frac = val as f32 / 127.0;
                            ui.add(
                                egui::ProgressBar::new(frac)
                                    .desired_width(220.0),
                            );
                            ui.monospace(format!("{count}"));
                            ui.end_row();
                        }
                    });
            }

            ui.add_space(16.0);
            ui.separator();

            // ----- Notes / pads section ---------------------------------
            ui.heading("Notes held");
            ui.label(
                "Press a key or pad. Note numbers light up when held. Keys \
                 typically land in 36-84; pads commonly map to 36-51.",
            );
            ui.add_space(6.0);

            // Render a 0-95 note grid as small cells; held notes light up.
            let cell_size = egui::vec2(28.0, 22.0);
            let held: std::collections::HashSet<u8> = dash_snapshot
                .active_notes
                .iter()
                .map(|(_, n)| *n)
                .collect();

            ui.horizontal_wrapped(|ui| {
                for note in 24u8..96u8 {
                    let label = format!("{note}");
                    let active = held.contains(&note);
                    let (rect, _resp) = ui.allocate_exact_size(
                        cell_size,
                        egui::Sense::hover(),
                    );
                    let painter = ui.painter();
                    let bg = if active {
                        egui::Color32::from_rgb(255, 180, 60)
                    } else {
                        egui::Color32::from_rgb(40, 40, 50)
                    };
                    painter.rect_filled(rect, 3.0, bg);
                    painter.text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        label,
                        egui::FontId::monospace(11.0),
                        if active {
                            egui::Color32::BLACK
                        } else {
                            egui::Color32::from_rgb(160, 160, 170)
                        },
                    );
                }
            });

            if held.is_empty() {
                ui.add_space(4.0);
                ui.colored_label(
                    egui::Color32::from_rgb(180, 180, 180),
                    "(no notes held)",
                );
            } else {
                ui.add_space(4.0);
                let list: Vec<String> = held.iter().map(|n| n.to_string()).collect();
                ui.monospace(format!("held: {}", list.join(", ")));
            }
        });

        // Write back any GUI-driven changes. Skip the lock entirely if
        // nothing changed so we don't fight the MIDI thread for it.
        //
        // For `master` we apply the GUI movement as a DELTA on the current
        // value rather than a blind absolute write. The MIDI callback may
        // have nudged `master` (CC70 encoder) between our snapshot at frame
        // start and this write-back; an absolute set would silently discard
        // that concurrent change. Delta-apply preserves both edits.
        //
        // For `engine` last-writer-wins is fine: it's a discrete choice, not
        // an accumulator, so racing two writers just picks one.
        let gui_master_delta = master - master_before;
        let master_changed = gui_master_delta.abs() > 1e-6;
        let engine_changed = engine != engine_before;
        if master_changed || engine_changed {
            let mut lp = self.ctx.live.lock().unwrap();
            if master_changed {
                lp.master = (lp.master + gui_master_delta).clamp(0.0, 3.0);
            }
            if engine_changed {
                lp.engine = engine;
            }
        }
    }
}

/// Snapshot of `DashState` taken under one lock; allows drawing without
/// holding the lock for the whole `update` call.
struct DashSnapshot {
    cc_raw: std::collections::HashMap<u8, u8>,
    cc_count: std::collections::HashMap<u8, u64>,
    active_notes: std::collections::HashSet<(u8, u8)>,
    recent: Vec<String>,
}

const ENGINE_CHOICES: &[(&str, Engine)] = &[
    ("square", Engine::Square),
    ("ks", Engine::Ks),
    ("ks-rich", Engine::KsRich),
    ("piano", Engine::Piano),
    ("koto", Engine::Koto),
    ("sub", Engine::Sub),
    ("fm", Engine::Fm),
];
