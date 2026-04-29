//! `jukebox_lite` — Claude variant of the jukebox UI, rebuilt on
//! [`keysynth::jukebox_core`].
//!
//! All shared concerns (cpal mixer, library scan, play_log, CP server,
//! first-paint backstop, render-queue draining) live in the core. This
//! file is *only* the YouTube-style card grid layout:
//!
//! ```text
//!  ┌─────────────────────────────────────────────────────────────┐
//!  │ TopBottomPanel top   — search + tile pills                  │
//!  ├─────────────────────────────────────────────────────────────┤
//!  │                                                             │
//!  │ CentralPanel         — ScrollArea + grid of cards           │
//!  │                        each card = title / composer / play  │
//!  │                                                             │
//!  ├─────────────────────────────────────────────────────────────┤
//!  │ TopBottomPanel bot.  — now playing + progress + volume      │
//!  └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! No left side panel — that's the variant signature.

use std::path::PathBuf;
use std::time::Duration;

use eframe::egui;

use keysynth::jukebox_core::{
    AudioSnapshot, Format, JukeboxApp, JukeboxCore, JukeboxRunner, Tile, Track,
};
use keysynth::library_db::Song;

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------

const CARD_MIN_W: f32 = 240.0;
const CARD_HEIGHT: f32 = 110.0;
const CARD_GAP: f32 = 10.0;

// ---------------------------------------------------------------------------
// JukeboxLite — UI-only state
// ---------------------------------------------------------------------------

/// Per-frame UI state. Anything persistent (selected track, playing
/// label, library, history) lives in [`JukeboxCore`]; this struct only
/// owns the bits that don't survive into the CP snapshot — namely the
/// search-box echo we keep in sync with `core.selection.search`.
#[derive(Default)]
struct JukeboxLite {
    /// Local mirror of the search box, kept in lockstep with
    /// `core.selection.search`. We need a separate `String` because
    /// `egui::TextEdit` borrows `&mut String` for the full pass and
    /// we can't borrow `core` mutably twice.
    search_buf: String,
    search_synced: bool,
}

impl JukeboxApp for JukeboxLite {
    fn render(&mut self, ctx: &egui::Context, core: &mut JukeboxCore, visible: &[Track]) {
        // First entry: pull the CP-driven search value into our buffer.
        if !self.search_synced {
            self.search_buf = core.selection.search.clone();
            self.search_synced = true;
        }

        self.draw_top_bar(ctx, core, visible.len());
        self.draw_now_playing(ctx, core);
        self.draw_card_grid(ctx, core, visible);

        // While audio is rolling, ask for a faster refresh so the
        // progress bar advances smoothly. The Runner already enforces
        // a 500ms backstop for first-paint; this is purely cosmetic.
        if core.audio.is_playing() || core.audio.pending_render.is_some() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }
}

impl JukeboxLite {
    // -----------------------------------------------------------------
    // Top bar — search box + tile pills
    // -----------------------------------------------------------------
    fn draw_top_bar(&mut self, ctx: &egui::Context, core: &mut JukeboxCore, result_count: usize) {
        egui::TopBottomPanel::top("jukebox_lite_top")
            .exact_height(58.0)
            .show(ctx, |ui| {
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.add_space(8.0);
                    ui.label(egui::RichText::new("\u{1F50D}").size(16.0));
                    let search = egui::TextEdit::singleline(&mut self.search_buf)
                        .hint_text("search title / composer / id…")
                        .desired_width(280.0);
                    let resp = ui.add(search);
                    if resp.changed() {
                        core.selection.search = self.search_buf.clone();
                    }
                    if !self.search_buf.is_empty() {
                        if ui.small_button("\u{2715}").clicked() {
                            self.search_buf.clear();
                            core.selection.search.clear();
                        }
                    }
                    ui.add_space(12.0);
                    ui.separator();
                    ui.add_space(6.0);

                    // Tile pills
                    for &tile in Tile::all() {
                        let active = core.selection.tile == tile;
                        let label = tile.label();
                        let txt = if active {
                            egui::RichText::new(label)
                                .strong()
                                .color(egui::Color32::WHITE)
                        } else {
                            egui::RichText::new(label).color(egui::Color32::LIGHT_GRAY)
                        };
                        let fill = if active {
                            egui::Color32::from_rgb(80, 130, 200)
                        } else {
                            egui::Color32::from_rgb(50, 50, 58)
                        };
                        let btn = egui::Button::new(txt).fill(fill).rounding(14.0);
                        if ui.add(btn).clicked() {
                            core.select_tile(tile);
                        }
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(8.0);
                        ui.label(
                            egui::RichText::new(format!("{} results", result_count))
                                .color(egui::Color32::GRAY)
                                .small(),
                        );
                        if let Some(p) = core.audio.pending_render.as_ref() {
                            ui.add_space(8.0);
                            ui.spinner();
                            ui.label(
                                egui::RichText::new(format!("rendering {}…", p.track_label))
                                    .color(egui::Color32::LIGHT_BLUE)
                                    .small(),
                            );
                        }
                    });
                });
                ui.add_space(2.0);
            });
    }

    // -----------------------------------------------------------------
    // Now playing bar (bottom)
    // -----------------------------------------------------------------
    fn draw_now_playing(&mut self, ctx: &egui::Context, core: &mut JukeboxCore) {
        let snap: AudioSnapshot = core.audio.snapshot();
        let song: Option<Song> = snap
            .label
            .as_ref()
            .and_then(|l| core.library.song(l).cloned());

        egui::TopBottomPanel::bottom("jukebox_lite_now_playing")
            .exact_height(72.0)
            .show(ctx, |ui| {
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.add_space(10.0);
                    let icon = if snap.is_playing {
                        "\u{25B6}"
                    } else if snap.label.is_some() {
                        "\u{23F8}"
                    } else {
                        "\u{1F3B5}"
                    };
                    ui.label(egui::RichText::new(icon).size(22.0));
                    ui.add_space(8.0);

                    ui.vertical(|ui| {
                        let title = match (snap.label.as_ref(), song.as_ref()) {
                            (Some(_), Some(s)) => s.title.clone(),
                            (Some(l), None) => l.clone(),
                            (None, _) => "Nothing playing".to_string(),
                        };
                        ui.label(egui::RichText::new(title).strong().size(15.0));
                        let sub = song
                            .as_ref()
                            .map(|s| compact_composer(&s.composer))
                            .unwrap_or_else(|| "—".to_string());
                        ui.label(
                            egui::RichText::new(sub)
                                .color(egui::Color32::LIGHT_GRAY)
                                .small(),
                        );
                    });

                    ui.add_space(20.0);

                    // Progress bar (fills remaining width before volume).
                    let total = snap.total_frames.max(1);
                    let cursor = snap.cursor_frames.min(total);
                    let fraction = cursor as f32 / total as f32;
                    let sr = core.audio.sample_rate_for_progress.max(1);
                    let cur_secs = cursor as f64 / sr as f64;
                    let tot_secs = total as f64 / sr as f64;
                    ui.vertical(|ui| {
                        ui.add(
                            egui::ProgressBar::new(fraction)
                                .desired_width(360.0)
                                .show_percentage(),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "{} / {}",
                                fmt_secs(cur_secs),
                                fmt_secs(tot_secs),
                            ))
                            .small()
                            .color(egui::Color32::GRAY),
                        );
                    });

                    ui.add_space(18.0);

                    // Stop / resume
                    if snap.is_playing {
                        if ui.button("\u{23F8} pause").clicked() {
                            core.audio.stop();
                        }
                    } else if snap.label.is_some() {
                        if ui.button("\u{25B6} resume").clicked() {
                            core.audio.resume();
                        }
                    }
                    if ui.button("\u{23F9} stop").clicked() {
                        core.stop();
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(10.0);
                        let mut vol = match core.audio.mixer.state.lock() {
                            Ok(g) => g.volume,
                            Err(_) => 0.85,
                        };
                        let resp = ui.add(
                            egui::Slider::new(&mut vol, 0.0..=1.5)
                                .text("vol")
                                .show_value(false),
                        );
                        if resp.changed() {
                            if let Ok(mut g) = core.audio.mixer.state.lock() {
                                g.volume = vol;
                            }
                        }
                    });
                });
            });
    }

    // -----------------------------------------------------------------
    // Central card grid
    // -----------------------------------------------------------------
    fn draw_card_grid(
        &mut self,
        ctx: &egui::Context,
        core: &mut JukeboxCore,
        visible: &[Track],
    ) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.heading(core.selection.tile.label());
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new(format!("{} tracks", visible.len()))
                        .color(egui::Color32::GRAY)
                        .small(),
                );
            });
            ui.separator();

            if visible.is_empty() {
                ui.add_space(40.0);
                ui.vertical_centered(|ui| {
                    ui.label(
                        egui::RichText::new("Nothing here yet.")
                            .heading()
                            .color(egui::Color32::GRAY),
                    );
                    ui.label("Pick a different tile or clear the search box.");
                });
                return;
            }

            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    let avail = ui.available_width();
                    let cols = ((avail / CARD_MIN_W).floor() as usize).max(1);
                    let card_w = (avail / cols as f32) - CARD_GAP;
                    let row_count = (visible.len() + cols - 1) / cols;
                    let mut to_play: Option<String> = None;

                    for r in 0..row_count {
                        ui.horizontal(|ui| {
                            for c in 0..cols {
                                let i = r * cols + c;
                                if i >= visible.len() {
                                    ui.add_space(card_w + CARD_GAP);
                                    continue;
                                }
                                let t = &visible[i];
                                let song = core.library.song(&t.label).cloned();
                                let is_selected = core
                                    .selection
                                    .selected_label
                                    .as_ref()
                                    .map(|l| l == &t.label)
                                    .unwrap_or(false);
                                let is_fav = core.history.favorites.contains(&t.label);
                                if draw_card(ui, card_w, t, song.as_ref(), is_selected, is_fav) {
                                    to_play = Some(t.label.clone());
                                }
                                ui.add_space(CARD_GAP);
                            }
                        });
                        ui.add_space(CARD_GAP);
                    }

                    if let Some(label) = to_play {
                        core.play_label(&label);
                    }
                });
        });
    }
}

// ---------------------------------------------------------------------------
// Card rendering (free function — needs no `&mut self`)
// ---------------------------------------------------------------------------

/// Returns `true` when the user clicked the play button on this card.
fn draw_card(
    ui: &mut egui::Ui,
    width: f32,
    track: &Track,
    song: Option<&Song>,
    selected: bool,
    fav: bool,
) -> bool {
    let frame_color = if selected {
        egui::Color32::from_rgb(60, 100, 170)
    } else {
        egui::Color32::from_rgb(36, 36, 42)
    };
    let stroke_color = if selected {
        egui::Color32::from_rgb(150, 200, 255)
    } else {
        egui::Color32::from_rgb(80, 80, 90)
    };
    let mut clicked = false;
    let frame = egui::Frame::none()
        .fill(frame_color)
        .stroke(egui::Stroke::new(1.0, stroke_color))
        .rounding(8.0)
        .inner_margin(egui::Margin::symmetric(10.0, 8.0));
    frame.show(ui, |ui| {
        ui.set_width(width);
        ui.set_min_height(CARD_HEIGHT);
        ui.vertical(|ui| {
            // Top row: era badge / format / fav star
            ui.horizontal(|ui| {
                if let Some(s) = song {
                    if let Some(era) = s.era.as_deref() {
                        let badge = egui::RichText::new(format!(" {} ", era))
                            .background_color(era_color(era))
                            .color(egui::Color32::BLACK)
                            .small();
                        ui.label(badge);
                    }
                    if !s.instrument.is_empty() {
                        ui.label(
                            egui::RichText::new(format!("· {}", s.instrument))
                                .small()
                                .color(egui::Color32::LIGHT_GRAY),
                        );
                    }
                } else {
                    ui.label(
                        egui::RichText::new(format!("[{}]", format_label(track.format)))
                            .small()
                            .color(egui::Color32::DARK_GRAY),
                    );
                }
                if fav {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new("\u{2605}")
                                .color(egui::Color32::from_rgb(255, 200, 80)),
                        );
                    });
                }
            });

            // Title (1 strong line)
            let title = song.map(|s| s.title.as_str()).unwrap_or(&track.label);
            ui.label(
                egui::RichText::new(title)
                    .strong()
                    .size(15.0)
                    .color(egui::Color32::WHITE),
            );

            // Composer (subtle)
            let composer = song
                .map(|s| compact_composer(&s.composer))
                .unwrap_or_else(|| "—".to_string());
            ui.label(
                egui::RichText::new(composer)
                    .color(egui::Color32::LIGHT_GRAY)
                    .small(),
            );

            // Bottom row: play button.
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                let play =
                    egui::Button::new(egui::RichText::new("\u{25B6} play").size(14.0).strong())
                        .fill(egui::Color32::from_rgb(80, 130, 200))
                        .rounding(4.0);
                if ui.add(play).clicked() {
                    clicked = true;
                }
            });
        });
    });
    clicked
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn era_color(era: &str) -> egui::Color32 {
    match era {
        "Baroque" => egui::Color32::from_rgb(255, 220, 150),
        "Classical" => egui::Color32::from_rgb(200, 230, 255),
        "Romantic" => egui::Color32::from_rgb(255, 190, 200),
        "Modern" => egui::Color32::from_rgb(200, 255, 220),
        "Traditional" => egui::Color32::from_rgb(220, 220, 180),
        _ => egui::Color32::from_rgb(220, 220, 220),
    }
}

fn format_label(f: Format) -> &'static str {
    match f {
        Format::Wav => "WAV",
        Format::Mp3 => "MP3",
        Format::Mid => "MIDI",
    }
}

/// "Wolfgang Amadeus Mozart" → "W.A. Mozart". One name → unchanged.
fn compact_composer(name: &str) -> String {
    let parts: Vec<&str> = name.split_whitespace().collect();
    if parts.len() <= 1 {
        return name.to_string();
    }
    let mut out = String::new();
    for p in &parts[..parts.len() - 1] {
        if let Some(c) = p.chars().next() {
            out.push(c);
            out.push('.');
        }
    }
    if !out.is_empty() {
        out.push(' ');
    }
    out.push_str(parts[parts.len() - 1]);
    out
}

fn fmt_secs(s: f64) -> String {
    if !s.is_finite() || s < 0.0 {
        return "0:00".to_string();
    }
    let total = s as u64;
    let m = total / 60;
    let r = total % 60;
    format!("{}:{:02}", m, r)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dirs = vec![
        PathBuf::from("bench-out/songs"),
        PathBuf::from("bench-out/CHIPTUNE"),
        PathBuf::from("bench-out/iterH"),
    ];
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 760.0])
            .with_title("keysynth jukebox lite")
            .with_visible(true),
        ..Default::default()
    };
    eframe::run_native(
        "keysynth jukebox lite",
        options,
        Box::new(|cc| {
            keysynth::ui::setup_japanese_fonts(&cc.egui_ctx);
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            let core = match JukeboxCore::new(dirs, "jukebox_lite", cc.egui_ctx.clone()) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("jukebox_lite: core init failed: {e}");
                    return Err(format!("core: {e}").into());
                }
            };
            let app = JukeboxLite::default();
            Ok(Box::new(JukeboxRunner::new(core, app)))
        }),
    )
    .map_err(|e| -> Box<dyn std::error::Error> { format!("eframe: {e}").into() })?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_composer_initials_on_multiword() {
        assert_eq!(
            compact_composer("Wolfgang Amadeus Mozart"),
            "W.A. Mozart"
        );
        assert_eq!(compact_composer("Bach"), "Bach");
        assert_eq!(compact_composer(""), "");
    }

    #[test]
    fn fmt_secs_basic() {
        assert_eq!(fmt_secs(0.0), "0:00");
        assert_eq!(fmt_secs(5.4), "0:05");
        assert_eq!(fmt_secs(65.0), "1:05");
        assert_eq!(fmt_secs(-1.0), "0:00");
    }

    #[test]
    fn format_label_covers_all_variants() {
        assert_eq!(format_label(Format::Wav), "WAV");
        assert_eq!(format_label(Format::Mp3), "MP3");
        assert_eq!(format_label(Format::Mid), "MIDI");
    }
}
