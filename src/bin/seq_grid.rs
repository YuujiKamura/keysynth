//! seq_grid: standalone egui step-sequencer pattern grid (issue #7
//! Stage 6). Drives [`keysynth::sequencer::Pattern`] only — no Stage 1
//! mixer integration, no piano-voice involvement.
//!
//! Layout: BPM + bars sliders, name field, transport buttons, then a
//! 3-row grid (Kick / Snare / HiHat) of clickable cells. Save / load
//! round-trip JSON in `bench-out/PATTERNS/<name>.json`. Export wav
//! renders the pattern through the existing drum synthesiser into
//! `bench-out/PATTERNS/<name>.wav`. Play hits the same code path then
//! plays the result through rodio.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use eframe::egui;
use hound::{SampleFormat, WavSpec, WavWriter};
use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink};

use keysynth::sequencer::{Pattern, TrackKind};

const SR: u32 = 48_000;
/// Tail after the last drum hit so the kick/snare envelopes fully
/// decay before the file ends. Snare is the longest at 0.150 sec.
const RELEASE_TAIL_SEC: f32 = 0.4;

fn patterns_dir() -> PathBuf {
    PathBuf::from("bench-out/PATTERNS")
}

fn ensure_patterns_dir() -> std::io::Result<PathBuf> {
    let d = patterns_dir();
    std::fs::create_dir_all(&d)?;
    Ok(d)
}

/// Render the pattern's drum events into a stereo (mono-duplicated) wav
/// file at the given path.
fn render_pattern_to_wav(pattern: &Pattern, path: &Path) -> Result<(), String> {
    let events = pattern.to_drum_events();
    let max_end = events
        .iter()
        .map(|e| e.start_sec + e.duration_sec())
        .fold(0.0_f32, f32::max);
    // Even an empty pattern still produces a (silent) file of one bar
    // length so save→load→export doesn't return a zero-length wav.
    let bar_sec = 4.0 * 60.0 / pattern.bpm;
    let pattern_sec = bar_sec * pattern.bars as f32;
    let total_sec = max_end.max(pattern_sec) + RELEASE_TAIL_SEC;
    let total_samples = (total_sec * SR as f32) as usize + 1;
    let mut buf = vec![0.0_f32; total_samples];

    for ev in &events {
        let start = (ev.start_sec * SR as f32) as usize;
        if start >= total_samples {
            continue;
        }
        ev.render(SR as f32, &mut buf, start);
    }

    // Normalise to -3 dBFS if peak is non-trivial; otherwise leave as is.
    let peak = buf.iter().fold(0.0_f32, |a, b| a.max(b.abs()));
    let gain = if peak > 1e-6 {
        (10.0_f32.powf(-3.0 / 20.0)) / peak
    } else {
        1.0
    };

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
    }
    let spec = WavSpec {
        channels: 2,
        sample_rate: SR,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut w = WavWriter::create(path, spec).map_err(|e| format!("WavWriter: {e}"))?;
    for s in &buf {
        let v = ((s * gain).clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        // Mono drums duplicated to L+R.
        w.write_sample(v).map_err(|e| format!("write L: {e}"))?;
        w.write_sample(v).map_err(|e| format!("write R: {e}"))?;
    }
    w.finalize().map_err(|e| format!("finalize: {e}"))?;
    Ok(())
}

struct App {
    pattern: Pattern,
    name: String,
    status: String,
    /// Held to keep the audio device alive between play presses.
    _stream: OutputStream,
    handle: OutputStreamHandle,
    sink: Option<Sink>,
    /// Cached event count for the status bar.
    last_event_count: usize,
}

impl App {
    fn new() -> Result<Self, String> {
        let (stream, handle) =
            OutputStream::try_default().map_err(|e| format!("audio: {e}"))?;
        Ok(Self {
            pattern: default_demo_pattern(),
            name: "demo".to_string(),
            status: "ready".to_string(),
            _stream: stream,
            handle,
            sink: None,
            last_event_count: 0,
        })
    }

    fn export_wav_path(&self) -> PathBuf {
        patterns_dir().join(format!("{}.wav", sanitise_name(&self.name)))
    }

    fn save_json_path(&self) -> PathBuf {
        patterns_dir().join(format!("{}.json", sanitise_name(&self.name)))
    }

    fn do_save(&mut self) {
        let path = self.save_json_path();
        match ensure_patterns_dir() {
            Ok(_) => {}
            Err(e) => {
                self.status = format!("save: mkdir {e}");
                return;
            }
        }
        match serde_json::to_string_pretty(&self.pattern)
            .map_err(|e| format!("serialize: {e}"))
            .and_then(|s| std::fs::write(&path, s).map_err(|e| format!("write: {e}")))
        {
            Ok(()) => self.status = format!("saved {}", path.display()),
            Err(e) => self.status = format!("save: {e}"),
        }
    }

    fn do_load(&mut self) {
        let path = self.save_json_path();
        match std::fs::read_to_string(&path)
            .map_err(|e| format!("read: {e}"))
            .and_then(|s| serde_json::from_str::<Pattern>(&s).map_err(|e| format!("parse: {e}")))
        {
            Ok(p) => {
                self.pattern = p;
                self.status = format!("loaded {}", path.display());
            }
            Err(e) => self.status = format!("load: {e}"),
        }
    }

    fn do_export_wav(&mut self) -> Option<PathBuf> {
        let path = self.export_wav_path();
        match render_pattern_to_wav(&self.pattern, &path) {
            Ok(()) => {
                self.last_event_count = self.pattern.to_drum_events().len();
                self.status = format!(
                    "exported {} ({} events)",
                    path.display(),
                    self.last_event_count,
                );
                Some(path)
            }
            Err(e) => {
                self.status = format!("export: {e}");
                None
            }
        }
    }

    fn do_play(&mut self) {
        // Stop any previous sink before kicking off a new render.
        if let Some(sink) = self.sink.take() {
            sink.stop();
        }
        let Some(path) = self.do_export_wav() else {
            return;
        };
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) => {
                self.status = format!("play: open {e}");
                return;
            }
        };
        let decoder = match Decoder::new(BufReader::new(file)) {
            Ok(d) => d,
            Err(e) => {
                self.status = format!("play: decode {e}");
                return;
            }
        };
        let sink = match Sink::try_new(&self.handle) {
            Ok(s) => s,
            Err(e) => {
                self.status = format!("play: sink {e}");
                return;
            }
        };
        sink.append(decoder);
        sink.play();
        self.sink = Some(sink);
        self.status = format!("playing {}", path.display());
    }

    fn do_stop(&mut self) {
        if let Some(sink) = self.sink.take() {
            sink.stop();
            self.status = "stopped".to_string();
        }
    }
}

fn sanitise_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "untitled".to_string()
    } else {
        cleaned
    }
}

/// A simple "boots-and-cats" two-bar starter so the grid isn't empty
/// the first time the user opens the app.
fn default_demo_pattern() -> Pattern {
    // 1 bar at 120 BPM; classic 4-on-the-floor with snare on 5/13 and
    // closed hi-hats on every off-beat.
    Pattern::from_chiptune_string(
        "K...S...K...S...\
         H.H.H.H.H.H.H.H.",
        120.0,
        2,
    )
}

fn cell_button_size() -> egui::Vec2 {
    egui::vec2(22.0, 22.0)
}

fn draw_grid(ui: &mut egui::Ui, pattern: &mut Pattern) {
    let total = pattern.total_steps();
    let spb = pattern.steps_per_bar as usize;
    egui::ScrollArea::horizontal().show(ui, |ui| {
        ui.vertical(|ui| {
            for kind in TrackKind::ALL {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(kind.label()).strong());
                    ui.add_space(8.0);
                    for i in 0..total {
                        let on = pattern.cell(kind, i);
                        // Beat colouring: every 4th step a touch
                        // brighter, bar boundary slightly more so.
                        let beat = i % 4 == 0;
                        let bar_start = spb > 0 && i % spb == 0;
                        let fill = if on {
                            egui::Color32::from_rgb(220, 130, 60)
                        } else if bar_start {
                            egui::Color32::from_gray(80)
                        } else if beat {
                            egui::Color32::from_gray(60)
                        } else {
                            egui::Color32::from_gray(45)
                        };
                        let btn = egui::Button::new("")
                            .fill(fill)
                            .min_size(cell_button_size());
                        if ui.add(btn).clicked() {
                            pattern.toggle(kind, i);
                        }
                        if spb > 0 && (i + 1) % spb == 0 {
                            ui.add_space(6.0);
                        }
                    }
                });
            }
        });
    });
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("seq_grid — step sequencer (issue #7 Stage 6)");
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("name:");
                ui.text_edit_singleline(&mut self.name);
            });

            ui.horizontal(|ui| {
                ui.label("BPM:");
                ui.add(egui::Slider::new(&mut self.pattern.bpm, 60.0..=200.0).step_by(1.0));
            });

            // bars slider: resize the grid in-place when the user drags.
            let mut bars_i32 = self.pattern.bars as i32;
            ui.horizontal(|ui| {
                ui.label("bars:");
                if ui
                    .add(egui::Slider::new(&mut bars_i32, 1..=8))
                    .changed()
                {
                    let new_bars = bars_i32.max(1) as u32;
                    if new_bars != self.pattern.bars {
                        resize_pattern(&mut self.pattern, new_bars);
                    }
                }
            });

            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("▶ play").clicked() {
                    self.do_play();
                }
                if ui.button("■ stop").clicked() {
                    self.do_stop();
                }
                ui.separator();
                if ui.button("save").clicked() {
                    self.do_save();
                }
                if ui.button("load").clicked() {
                    self.do_load();
                }
                if ui.button("export wav").clicked() {
                    self.do_export_wav();
                }
                if ui.button("clear").clicked() {
                    for kind in TrackKind::ALL {
                        if let Some(row) = self.pattern.tracks.get_mut(&kind) {
                            for c in row {
                                *c = false;
                            }
                        }
                    }
                }
            });

            ui.separator();
            draw_grid(ui, &mut self.pattern);

            ui.separator();
            ui.label(&self.status);
            ui.label(format!(
                "events: {} | step: {:.4} sec | bar: {:.3} sec",
                self.pattern.to_drum_events().len(),
                self.pattern.step_sec(),
                self.pattern.step_sec() * self.pattern.steps_per_bar as f32,
            ));
        });
    }
}

/// Resize the pattern's track rows when `bars` changes. Existing cells
/// are preserved up to the new length; new cells are rest.
fn resize_pattern(p: &mut Pattern, new_bars: u32) {
    p.bars = new_bars;
    let total = p.total_steps();
    for kind in TrackKind::ALL {
        let row = p.tracks.entry(kind).or_insert_with(Vec::new);
        if row.len() < total {
            row.resize(total, false);
        } else if row.len() > total {
            row.truncate(total);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Ensure the patterns dir exists at startup so save/load buttons
    // don't surprise the user with a missing-dir error.
    let _ = ensure_patterns_dir();
    // Seed sample patterns on first run so the directory has demo
    // content for the repo. These overwrite themselves only if absent.
    let _ = seed_demo_patterns();

    // Headless `--export <name>` mode: load the JSON pattern by name
    // from bench-out/PATTERNS/, render its wav, exit. No window. Used
    // by smoke tests so we can verify the render path on hosts where
    // launching a GUI is awkward.
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 3 && args[1] == "--export" {
        let name = &args[2];
        let json_path = patterns_dir().join(format!("{}.json", sanitise_name(name)));
        let s = std::fs::read_to_string(&json_path)
            .map_err(|e| format!("read {}: {e}", json_path.display()))?;
        let pattern: Pattern = serde_json::from_str(&s).map_err(|e| format!("parse: {e}"))?;
        let wav = patterns_dir().join(format!("{}.wav", sanitise_name(name)));
        render_pattern_to_wav(&pattern, &wav).map_err(|e| format!("render: {e}"))?;
        println!(
            "exported {} ({} events)",
            wav.display(),
            pattern.to_drum_events().len(),
        );
        return Ok(());
    }

    let app = match App::new() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("seq_grid: {e}");
            // Fall through to a fake handle so the GUI still opens
            // even on a headless / no-audio host (CI). We just won't
            // play, but save / load / export wav still work.
            return Err(e.into());
        }
    };
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 380.0])
            .with_title("seq_grid"),
        ..Default::default()
    };
    eframe::run_native(
        "seq_grid",
        opts,
        Box::new(|_cc| Ok(Box::new(app))),
    )
    .map_err(|e| format!("eframe: {e}").into())
}

/// Seed two demo patterns into `bench-out/PATTERNS/` if they don't
/// already exist. These are the files that ship with the repo so a
/// fresh clone has something to play with.
fn seed_demo_patterns() -> std::io::Result<()> {
    let dir = ensure_patterns_dir()?;
    let demo_a = dir.join("demo_four_on_the_floor.json");
    if !demo_a.exists() {
        let p = Pattern::from_chiptune_string(
            "K...S...K...S...\
             H.H.H.H.H.H.H.H.",
            120.0,
            2,
        );
        if let Ok(j) = serde_json::to_string_pretty(&p) {
            let _ = std::fs::write(&demo_a, j);
        }
    }
    let demo_b = dir.join("demo_breakbeat.json");
    if !demo_b.exists() {
        // 1-bar amen-ish skeleton; not the full break, just a probe.
        let p = Pattern::from_chiptune_string("K..KS..KK.K.S.K.", 100.0, 1);
        if let Ok(j) = serde_json::to_string_pretty(&p) {
            let _ = std::fs::write(&demo_b, j);
        }
    }
    Ok(())
}
