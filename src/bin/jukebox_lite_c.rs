//! jukebox_lite_c — Codex variant: foobar2000-style dense list browser.
//!
//! All audio / library / play-log / CP-server scaffolding lives in
//! [`keysynth::jukebox_core`]; this binary only owns UI state (sort
//! column / direction) and renders a tight column table.

use std::path::PathBuf;

use eframe::egui;
use keysynth::jukebox_core::{
    Format, JukeboxApp, JukeboxCore, JukeboxRunner, Tile, Track,
};
use keysynth::library_db::Song;

const ROW_H: f32 = 22.0;
const TILE_BAR_H: f32 = 22.0;
const TRANSPORT_H: f32 = 28.0;
const PROGRESS_H: f32 = 4.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SortColumn {
    Title,
    Composer,
    Era,
    Format,
}

impl SortColumn {
    fn label(self) -> &'static str {
        match self {
            SortColumn::Title => "title",
            SortColumn::Composer => "composer",
            SortColumn::Era => "era",
            SortColumn::Format => "format",
        }
    }
}

struct JukeboxLiteC {
    sort_col: SortColumn,
    sort_asc: bool,
}

impl Default for JukeboxLiteC {
    fn default() -> Self {
        Self {
            sort_col: SortColumn::Composer,
            sort_asc: true,
        }
    }
}

impl JukeboxLiteC {
    fn toggle_sort(&mut self, col: SortColumn) {
        if self.sort_col == col {
            self.sort_asc = !self.sort_asc;
        } else {
            self.sort_col = col;
            self.sort_asc = true;
        }
    }

    fn sort_tracks(&self, core: &JukeboxCore, tracks: &mut Vec<Track>) {
        let lib = &core.library;
        tracks.sort_by(|a, b| {
            let sa = lib.song(&a.label);
            let sb = lib.song(&b.label);
            let ord = match self.sort_col {
                SortColumn::Title => song_title(sa, a).cmp(&song_title(sb, b)),
                SortColumn::Composer => song_composer_key(sa).cmp(&song_composer_key(sb)),
                SortColumn::Era => song_era(sa).cmp(&song_era(sb)),
                SortColumn::Format => format_label(a.format).cmp(format_label(b.format)),
            };
            let tie = a.label.cmp(&b.label);
            let combined = ord.then(tie);
            if self.sort_asc {
                combined
            } else {
                combined.reverse()
            }
        });
    }
}

fn song_title(song: Option<&Song>, track: &Track) -> String {
    song.map(|s| s.title.clone())
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| track.label.clone())
        .to_ascii_lowercase()
}

fn song_composer_key(song: Option<&Song>) -> String {
    song.map(|s| s.composer_key.clone())
        .unwrap_or_else(|| "~".to_string())
}

fn song_era(song: Option<&Song>) -> String {
    song.and_then(|s| s.era.clone())
        .unwrap_or_else(|| "~".to_string())
}

fn format_label(f: Format) -> &'static str {
    match f {
        Format::Wav => "WAV",
        Format::Mp3 => "MP3",
        Format::Mid => "MID",
    }
}

fn fmt_duration(frames: u64, sample_rate: u32) -> String {
    if sample_rate == 0 || frames == 0 {
        return String::from("--:--");
    }
    let total = frames as f64 / sample_rate as f64;
    let mins = (total / 60.0).floor() as u64;
    let secs = (total - (mins as f64) * 60.0).floor() as u64;
    format!("{:02}:{:02}", mins, secs)
}

impl JukeboxApp for JukeboxLiteC {
    fn render(&mut self, ctx: &egui::Context, core: &mut JukeboxCore, visible: &[Track]) {
        // Build sorted snapshot for this paint.
        let mut sorted: Vec<Track> = visible.to_vec();
        self.sort_tracks(core, &mut sorted);

        let snap = core.audio.snapshot();
        let sample_rate = core.audio.sample_rate_for_progress;
        let active_tile = core.selection.tile;
        let selected_label = core.selection.selected_label.clone();

        // Top: search bar + tile tabs.
        egui::TopBottomPanel::top("ui_c_top")
            .exact_height(TILE_BAR_H * 2.0 + 6.0)
            .show(ctx, |ui| {
                ui.add_space(2.0);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("filter").small().weak());
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut core.selection.search)
                            .desired_width(260.0)
                            .hint_text("title / composer / label")
                            .font(egui::TextStyle::Small),
                    );
                    if resp.changed() {
                        ctx.request_repaint();
                    }
                    ui.separator();
                    ui.label(
                        egui::RichText::new(format!(
                            "{} / {} tracks",
                            sorted.len(),
                            core.library.tracks.len()
                        ))
                        .small()
                        .weak(),
                    );
                });
                ui.add_space(2.0);
                ui.horizontal(|ui| {
                    for &tile in Tile::all() {
                        let active = tile == active_tile;
                        let txt = egui::RichText::new(tile.label()).small();
                        let txt = if active { txt.strong() } else { txt.weak() };
                        if ui
                            .selectable_label(active, txt)
                            .on_hover_text(tile.label())
                            .clicked()
                        {
                            core.select_tile(tile);
                        }
                    }
                });
            });

        // Bottom: transport + progress.
        egui::TopBottomPanel::bottom("ui_c_bottom")
            .exact_height(TRANSPORT_H + PROGRESS_H + 8.0)
            .show(ctx, |ui| {
                let now_label = snap
                    .label
                    .clone()
                    .or_else(|| selected_label.clone())
                    .unwrap_or_else(|| String::from("(no track)"));
                let composer_text = core
                    .library
                    .song(&now_label)
                    .map(|s| s.composer.clone())
                    .filter(|c| !c.is_empty());
                let now_text = match composer_text {
                    Some(c) => format!("{}  -  {}", now_label, c),
                    None => now_label.clone(),
                };
                ui.add_space(2.0);
                ui.horizontal(|ui| {
                    let playing = snap.is_playing;
                    let play_label = if playing { "II" } else { ">" };
                    if ui
                        .add_sized([28.0, 22.0], egui::Button::new(play_label))
                        .clicked()
                    {
                        if playing {
                            core.audio.stop();
                        } else if let Some(label) = selected_label.clone() {
                            core.play_label(&label);
                        }
                    }
                    if ui
                        .add_sized([28.0, 22.0], egui::Button::new("[]"))
                        .clicked()
                    {
                        core.stop();
                    }
                    ui.separator();
                    ui.label(egui::RichText::new(now_text).small());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let dur = fmt_duration(snap.total_frames, sample_rate);
                        let pos = fmt_duration(snap.cursor_frames, sample_rate);
                        ui.label(
                            egui::RichText::new(format!("{} / {}", pos, dur))
                                .small()
                                .monospace(),
                        );
                    });
                });
                let frac = if snap.total_frames > 0 {
                    (snap.cursor_frames as f32 / snap.total_frames as f32).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let (rect, _) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), PROGRESS_H),
                    egui::Sense::hover(),
                );
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 0.0, egui::Color32::from_gray(30));
                if frac > 0.0 {
                    let mut filled = rect;
                    filled.set_width(rect.width() * frac);
                    painter.rect_filled(filled, 0.0, egui::Color32::from_rgb(110, 150, 200));
                }
            });

        // Central: dense table.
        egui::CentralPanel::default().show(ctx, |ui| {
            // Header row.
            let cols: [(SortColumn, f32); 4] = [
                (SortColumn::Title, 320.0),
                (SortColumn::Composer, 220.0),
                (SortColumn::Era, 110.0),
                (SortColumn::Format, 70.0),
            ];
            ui.horizontal(|ui| {
                ui.set_min_height(ROW_H);
                for (col, w) in cols.iter().copied() {
                    let arrow = if col == self.sort_col {
                        if self.sort_asc {
                            " ^"
                        } else {
                            " v"
                        }
                    } else {
                        "  "
                    };
                    let text = format!("{}{}", col.label(), arrow);
                    let resp = ui.add_sized(
                        [w, ROW_H],
                        egui::Button::new(egui::RichText::new(text).small().strong())
                            .frame(false),
                    );
                    if resp.clicked() {
                        self.toggle_sort(col);
                    }
                }
                ui.label(
                    egui::RichText::new("voice")
                        .small()
                        .strong()
                        .color(egui::Color32::from_gray(160)),
                );
            });
            ui.separator();

            let row_count = sorted.len();
            let mut to_play: Option<String> = None;
            let mut to_select: Option<String> = None;
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show_rows(ui, ROW_H, row_count, |ui, range| {
                    for idx in range {
                        let track = &sorted[idx];
                        let song = core.library.song(&track.label);
                        let title = song
                            .map(|s| s.title.clone())
                            .filter(|t| !t.is_empty())
                            .unwrap_or_else(|| track.label.clone());
                        let composer =
                            song.map(|s| s.composer.clone()).unwrap_or_default();
                        let era = song
                            .and_then(|s| s.era.clone())
                            .unwrap_or_default();
                        let voice = song
                            .and_then(|s| s.suggested_voice.clone())
                            .unwrap_or_else(|| match track.format {
                                Format::Wav | Format::Mp3 => String::from("-"),
                                Format::Mid => String::from("?"),
                            });
                        let is_selected =
                            selected_label.as_deref() == Some(track.label.as_str());

                        let row_resp = ui
                            .horizontal(|ui| {
                                ui.set_min_height(ROW_H);
                                let title_color = if is_selected {
                                    egui::Color32::from_rgb(220, 230, 250)
                                } else {
                                    egui::Color32::from_gray(210)
                                };
                                ui.add_sized(
                                    [320.0, ROW_H],
                                    egui::Label::new(
                                        egui::RichText::new(title)
                                            .small()
                                            .color(title_color),
                                    )
                                    .truncate(),
                                );
                                ui.add_sized(
                                    [220.0, ROW_H],
                                    egui::Label::new(
                                        egui::RichText::new(composer)
                                            .small()
                                            .color(egui::Color32::from_gray(170)),
                                    )
                                    .truncate(),
                                );
                                ui.add_sized(
                                    [110.0, ROW_H],
                                    egui::Label::new(
                                        egui::RichText::new(era)
                                            .small()
                                            .color(egui::Color32::from_gray(150)),
                                    )
                                    .truncate(),
                                );
                                ui.add_sized(
                                    [70.0, ROW_H],
                                    egui::Label::new(
                                        egui::RichText::new(format_label(track.format))
                                            .small()
                                            .monospace()
                                            .color(egui::Color32::from_gray(160)),
                                    ),
                                );
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(voice)
                                            .small()
                                            .color(egui::Color32::from_gray(140)),
                                    )
                                    .truncate(),
                                );
                            })
                            .response
                            .interact(egui::Sense::click());

                        if is_selected {
                            let painter = ui.painter();
                            painter.rect_filled(
                                row_resp.rect,
                                0.0,
                                egui::Color32::from_rgba_unmultiplied(80, 110, 160, 50),
                            );
                        }

                        if row_resp.clicked() {
                            to_select = Some(track.label.clone());
                        }
                        if row_resp.double_clicked() {
                            to_play = Some(track.label.clone());
                        }
                    }
                });

            if let Some(label) = to_select {
                core.selection.selected_label = Some(label);
            }
            if let Some(label) = to_play {
                core.play_label(&label);
            }
        });
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dirs = vec![
        PathBuf::from("bench-out/songs"),
        PathBuf::from("bench-out/CHIPTUNE"),
        PathBuf::from("bench-out/iterH"),
    ];
    let app = JukeboxLiteC::default();
    eframe::run_native(
        "keysynth jukebox lite c",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1380.0, 840.0])
                .with_title("keysynth jukebox lite c")
                .with_visible(true),
            ..Default::default()
        },
        Box::new(|cc| {
            keysynth::ui::setup_japanese_fonts(&cc.egui_ctx);
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            let core = JukeboxCore::new(dirs, "jukebox_lite_c", cc.egui_ctx.clone())
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
            Ok(Box::new(JukeboxRunner::new(core, app)))
        }),
    )
    .map_err(|e| format!("eframe: {e}").into())
}
