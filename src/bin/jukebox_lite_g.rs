//! jukebox_lite_g — Gemini variant of the music browser.
//!
//! Spotify-style sidebar + table built on the shared [`jukebox_core`].
//! The variant only owns UI-local state (sort column / direction). Audio,
//! library scan, history, and the CP server all live in `JukeboxCore`.
//!
//! ## Why this rewrite exists
//!
//! The original 1263-line bin re-implemented every backend service in the
//! file: cpal mixer, library_db loading, play_log, CP server, MIDI render
//! pump. Two of those open-coded paths broke in the same way:
//!
//! 1. `eframe::App::update` only requested a repaint **while playing**.
//!    Construction → first frame depended on a paint event that never
//!    arrived because no input had occurred yet, so eframe never flipped
//!    `WS_VISIBLE` and the OS window stayed hidden.
//! 2. The CP server callbacks pushed commands but did not call
//!    `egui::Context::request_repaint`, so a `dispatch load_track` from a
//!    verifier sat in the queue until something else woke the GUI.
//!
//! Both invariants now live in `JukeboxRunner` / `JukeboxControl::spawn`,
//! so this file deliberately does not call either repaint API.

use std::path::PathBuf;

use eframe::egui;

use keysynth::jukebox_core::{
    AudioSnapshot, Format, JukeboxApp, JukeboxCore, JukeboxRunner, Tile, Track,
};
use keysynth::library_db::Song;

// ---------------------------------------------------------------------------
// Sort column (UI-only state)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SortColumn {
    Title,
    Composer,
    Era,
    Format,
}

fn tile_icon(tile: Tile) -> &'static str {
    match tile {
        Tile::All => "\u{2630}",
        Tile::Favorites => "\u{2605}",
        Tile::Recent => "\u{21BB}",
        Tile::Classical => "\u{1F3B9}",
        Tile::Game => "\u{1F3AE}",
        Tile::Folk => "\u{1F3B8}",
    }
}

fn format_label(fmt: Format) -> &'static str {
    match fmt {
        Format::Wav => "WAV",
        Format::Mp3 => "MP3",
        Format::Mid => "MID",
    }
}

// ---------------------------------------------------------------------------
// JukeboxLiteG — UI-only state
// ---------------------------------------------------------------------------

struct JukeboxLiteG {
    sort_col: SortColumn,
    sort_asc: bool,
}

impl Default for JukeboxLiteG {
    fn default() -> Self {
        Self {
            sort_col: SortColumn::Composer,
            sort_asc: true,
        }
    }
}

impl JukeboxLiteG {
    /// Sort `tracks` according to the user-selected column. Done as a
    /// pure operation on the borrowed slice so we never mutate the
    /// caller-owned `visible` vector.
    fn sorted(&self, tracks: &[Track], core: &JukeboxCore) -> Vec<Track> {
        let mut out = tracks.to_vec();
        let asc = self.sort_asc;
        let key_str = |t: &Track, k: fn(&Song) -> &str| -> String {
            core.library
                .song(&t.label)
                .map(k)
                .unwrap_or("")
                .to_ascii_lowercase()
        };
        match self.sort_col {
            SortColumn::Title => out.sort_by(|a, b| {
                let ka = key_str(a, |s| s.title.as_str());
                let kb = key_str(b, |s| s.title.as_str());
                let ka = if ka.is_empty() { a.label.to_ascii_lowercase() } else { ka };
                let kb = if kb.is_empty() { b.label.to_ascii_lowercase() } else { kb };
                if asc { ka.cmp(&kb) } else { kb.cmp(&ka) }
            }),
            SortColumn::Composer => out.sort_by(|a, b| {
                let ka = key_str(a, |s| s.composer.as_str());
                let kb = key_str(b, |s| s.composer.as_str());
                if asc { ka.cmp(&kb) } else { kb.cmp(&ka) }
            }),
            SortColumn::Era => out.sort_by(|a, b| {
                let ka = core
                    .library
                    .song(&a.label)
                    .and_then(|s| s.era.clone())
                    .unwrap_or_default();
                let kb = core
                    .library
                    .song(&b.label)
                    .and_then(|s| s.era.clone())
                    .unwrap_or_default();
                if asc { ka.cmp(&kb) } else { kb.cmp(&ka) }
            }),
            SortColumn::Format => out.sort_by(|a, b| {
                let ka = format_label(a.format);
                let kb = format_label(b.format);
                if asc { ka.cmp(kb) } else { kb.cmp(ka) }
            }),
        }
        out
    }

    fn header_cell(&mut self, ui: &mut egui::Ui, text: &str, width: f32, col: SortColumn) {
        let selected = self.sort_col == col;
        let mut label = text.to_string();
        if selected {
            label.push(' ');
            label.push(if self.sort_asc { '\u{25B4}' } else { '\u{25BE}' });
        }
        let button = egui::Button::new(egui::RichText::new(label).small().strong())
            .frame(false)
            .min_size(egui::vec2(width, 20.0));
        if ui.add(button).clicked() {
            if self.sort_col == col {
                self.sort_asc = !self.sort_asc;
            } else {
                self.sort_col = col;
                self.sort_asc = true;
            }
        }
    }

    fn render_sidebar(&mut self, ctx: &egui::Context, core: &mut JukeboxCore) {
        egui::SidePanel::left("sidebar")
            .resizable(false)
            .default_width(180.0)
            .show(ctx, |ui| {
                ui.add_space(16.0);
                ui.vertical_centered(|ui| {
                    let text = egui::RichText::new("\u{1F3B5} JUKEBOX G")
                        .size(18.0)
                        .strong()
                        .color(ui.visuals().selection.bg_fill);
                    ui.label(text);
                });
                ui.add_space(20.0);

                for &tile in Tile::all() {
                    let selected = core.selection.tile == tile;
                    let text = format!("{}  {}", tile_icon(tile), tile.label());
                    if ui.selectable_label(selected, text).clicked() {
                        core.select_tile(tile);
                    }
                    ui.add_space(4.0);
                }

                ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
                    ui.add_space(10.0);
                    let stats = format!(
                        "{} tracks / {} voices",
                        core.library.tracks.len(),
                        core.library.db_voice_count
                    );
                    ui.label(egui::RichText::new(stats).small().color(egui::Color32::GRAY));
                    if ui.button("\u{21BB} Rescan").clicked() {
                        core.rescan();
                    }
                    ui.separator();
                });
            });
    }

    fn render_search(&mut self, ctx: &egui::Context, core: &mut JukeboxCore, visible_count: usize) {
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.add_space(8.0);
                ui.label(egui::RichText::new("\u{1F50D}").color(egui::Color32::GRAY));
                ui.add(
                    egui::TextEdit::singleline(&mut core.selection.search)
                        .hint_text("Search library...")
                        .desired_width(240.0),
                );
                if !core.selection.search.is_empty() && ui.button("\u{2715}").clicked() {
                    core.selection.search.clear();
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(format!("{} results", visible_count)).small(),
                    );
                });
            });
            ui.add_space(8.0);
        });
    }

    fn render_player(&mut self, ctx: &egui::Context, core: &mut JukeboxCore) {
        egui::TopBottomPanel::bottom("player")
            .exact_height(80.0)
            .show(ctx, |ui| {
                let snap: AudioSnapshot = core.audio.snapshot();
                let label = snap.label.clone().unwrap_or_default();
                let title = core
                    .library
                    .song(&label)
                    .map(|s| s.title.clone())
                    .unwrap_or_else(|| {
                        if label.is_empty() {
                            "—".into()
                        } else {
                            label.clone()
                        }
                    });
                let composer = core
                    .library
                    .song(&label)
                    .map(|s| s.composer.clone())
                    .unwrap_or_else(|| "—".into());

                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.add_space(20.0);
                    if snap.is_playing {
                        if ui
                            .button(egui::RichText::new("\u{23F8}").size(24.0))
                            .clicked()
                        {
                            core.audio.stop();
                        }
                    } else {
                        if ui
                            .button(egui::RichText::new("\u{25B6}").size(24.0))
                            .clicked()
                        {
                            if !label.is_empty() {
                                core.audio.resume();
                            } else if let Some(sel) = core.selection.selected_label.clone() {
                                core.play_label(&sel);
                            }
                        }
                    }
                    if ui
                        .button(egui::RichText::new("\u{25A0}").size(24.0))
                        .clicked()
                    {
                        core.audio.stop();
                    }

                    ui.add_space(20.0);
                    ui.vertical(|ui| {
                        ui.label(egui::RichText::new(title).strong());
                        ui.label(
                            egui::RichText::new(composer)
                                .small()
                                .color(egui::Color32::GRAY),
                        );
                    });

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(20.0);
                        let mut vol = core
                            .audio
                            .mixer
                            .state
                            .lock()
                            .map(|s| s.volume)
                            .unwrap_or(0.85);
                        if ui
                            .add(
                                egui::Slider::new(&mut vol, 0.0..=1.5)
                                    .show_value(false)
                                    .text("\u{1F50A}"),
                            )
                            .changed()
                        {
                            if let Ok(mut st) = core.audio.mixer.state.lock() {
                                st.volume = vol;
                            }
                        }
                    });
                });

                ui.add_space(8.0);
                let progress = if snap.total_frames > 0 {
                    snap.cursor_frames as f32 / snap.total_frames as f32
                } else {
                    0.0
                };
                ui.add(egui::ProgressBar::new(progress.clamp(0.0, 1.0)).desired_height(4.0));
            });
    }

    fn render_table(
        &mut self,
        ctx: &egui::Context,
        core: &mut JukeboxCore,
        rows: &[Track],
    ) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 0.0;

                    ui.horizontal(|ui| {
                        let w = ui.available_width();
                        let cols = [w * 0.4, w * 0.3, w * 0.15, w * 0.1];
                        self.header_cell(ui, "TITLE", cols[0], SortColumn::Title);
                        self.header_cell(ui, "COMPOSER", cols[1], SortColumn::Composer);
                        self.header_cell(ui, "ERA", cols[2], SortColumn::Era);
                        self.header_cell(ui, "FMT", cols[3], SortColumn::Format);
                    });
                    ui.separator();

                    let selected_label = core.selection.selected_label.clone();
                    let mut click_label: Option<String> = None;
                    for (idx, track) in rows.iter().enumerate() {
                        let song = core.library.song(&track.label);
                        let is_selected = selected_label.as_deref() == Some(track.label.as_str());

                        let bg = if is_selected {
                            ui.visuals().selection.bg_fill
                        } else if idx % 2 == 0 {
                            ui.visuals().faint_bg_color
                        } else {
                            ui.visuals().window_fill()
                        };

                        let frame = egui::Frame::none()
                            .fill(bg)
                            .inner_margin(egui::Margin::symmetric(8.0, 6.0));

                        let response = frame
                            .show(ui, |ui| {
                                let w = ui.available_width();
                                let cols = [w * 0.4, w * 0.3, w * 0.15, w * 0.1];
                                ui.horizontal(|ui| {
                                    let title =
                                        song.map(|s| s.title.as_str()).unwrap_or(&track.label);
                                    let text_color = if is_selected {
                                        ui.visuals().selection.stroke.color
                                    } else {
                                        ui.visuals().text_color()
                                    };
                                    ui.add_sized(
                                        [cols[0], 20.0],
                                        egui::Label::new(
                                            egui::RichText::new(title).strong().color(text_color),
                                        ),
                                    );
                                    let composer =
                                        song.map(|s| s.composer.as_str()).unwrap_or("—");
                                    ui.add_sized(
                                        [cols[1], 20.0],
                                        egui::Label::new(composer),
                                    );
                                    let era = song.and_then(|s| s.era.as_deref()).unwrap_or("");
                                    ui.add_sized([cols[2], 20.0], egui::Label::new(era));
                                    ui.add_sized(
                                        [cols[3], 20.0],
                                        egui::Label::new(format_label(track.format)),
                                    );
                                });
                            })
                            .response;

                        let response = ui.interact(
                            response.rect,
                            ui.id().with(("row", idx)),
                            egui::Sense::click(),
                        );
                        if response.clicked() {
                            click_label = Some(track.label.clone());
                        }
                        if response.hovered() && !is_selected {
                            ui.painter().rect_filled(
                                response.rect,
                                0.0,
                                ui.visuals().widgets.hovered.bg_fill.gamma_multiply(0.2),
                            );
                        }
                    }
                    if let Some(l) = click_label {
                        core.play_label(&l);
                    }
                });
        });
    }
}

impl JukeboxApp for JukeboxLiteG {
    fn render(&mut self, ctx: &egui::Context, core: &mut JukeboxCore, visible: &[Track]) {
        let rows = self.sorted(visible, core);
        self.render_sidebar(ctx, core);
        self.render_search(ctx, core, rows.len());
        self.render_player(ctx, core);
        self.render_table(ctx, core, &rows);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dirs = vec![
        PathBuf::from("bench-out/songs"),
        PathBuf::from("bench-out/CHIPTUNE"),
    ];
    let app = JukeboxLiteG::default();
    eframe::run_native(
        "JUKEBOX G",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1100.0, 700.0])
                .with_title("JUKEBOX G")
                .with_visible(true),
            ..Default::default()
        },
        Box::new(|cc| {
            keysynth::ui::setup_japanese_fonts(&cc.egui_ctx);
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            let core = match JukeboxCore::new(dirs, "jukebox_lite_g", cc.egui_ctx.clone()) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("jukebox_lite_g: core init failed: {e}");
                    return Err(format!("core init: {e}").into());
                }
            };
            Ok(Box::new(JukeboxRunner::new(core, app)))
        }),
    )
    .map_err(|e| format!("eframe: {e}").into())
}
