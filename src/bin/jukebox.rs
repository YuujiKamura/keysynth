//! Minimal WAV jukebox: scan bench-out/{iterH,songs}/ for *.wav, list
//! as selectable rows, click to play through default audio output via
//! rodio. For quick A/B listening of multiple engine renders without
//! launching a media player per file.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use eframe::egui;
use rodio::cpal::traits::{DeviceTrait, HostTrait};
use rodio::{cpal, Decoder, OutputStream, OutputStreamHandle, Sink};

#[derive(Clone, Debug)]
struct Track {
    path: PathBuf,
    /// Display name = filename without extension.
    label: String,
    /// Group label inferred from filename prefix (everything up to the
    /// last "_<engine>" suffix; "_sfz" / "_modal" / etc. drop into a
    /// separate column so the same piece across engines lines up).
    piece: String,
    engine: String,
    size_kb: u64,
}

const ENGINE_SUFFIXES: &[&str] = &[
    "sfz", "modal", "square", "ks", "ks-rich", "sub", "fm", "piano", "piano-thick",
    "piano-lite", "piano-5am", "koto",
];

fn parse_track(path: &Path) -> Option<Track> {
    let stem = path.file_stem()?.to_str()?.to_string();
    let size_kb = std::fs::metadata(path).ok()?.len() / 1024;
    // Try splitting "<piece>_<engine>" by recognised engine suffix.
    let (piece, engine) = ENGINE_SUFFIXES
        .iter()
        .find_map(|s| {
            let suf = format!("_{s}");
            stem.strip_suffix(&suf).map(|p| (p.to_string(), s.to_string()))
        })
        .unwrap_or_else(|| (stem.clone(), "—".to_string()));
    Some(Track {
        path: path.to_path_buf(),
        label: stem.clone(),
        piece,
        engine,
        size_kb,
    })
}

fn scan_dirs(dirs: &[&Path]) -> Vec<Track> {
    let mut out: Vec<Track> = Vec::new();
    for d in dirs {
        let read = match std::fs::read_dir(d) {
            Ok(r) => r,
            Err(_) => continue,
        };
        for ent in read.flatten() {
            let p = ent.path();
            if p.extension().and_then(|s| s.to_str()) == Some("wav") {
                if let Some(t) = parse_track(&p) {
                    out.push(t);
                }
            }
        }
    }
    // Sort by piece then engine for grouped display.
    out.sort_by(|a, b| {
        a.piece
            .cmp(&b.piece)
            .then_with(|| a.engine.cmp(&b.engine))
    });
    out
}

struct Jukebox {
    tracks: Vec<Track>,
    selected: Option<usize>,
    filter: String,
    /// Held to keep the audio device alive. When the user picks a new
    /// device we rebuild this; rodio's OutputStream owns the cpal
    /// stream and dropping it disconnects the device.
    _stream: OutputStream,
    handle: OutputStreamHandle,
    sink: Option<Sink>,
    refresh_dirs: Vec<PathBuf>,
    /// Available output devices' display names. Re-polled when the user
    /// clicks "rescan dev" so newly-plugged devices appear.
    devices: Vec<String>,
    /// Currently bound device name (the one whose stream we hold).
    /// "(default)" means OutputStream::try_default() at startup —
    /// captures whatever the OS default was at that moment.
    current_device: String,
}

fn list_output_devices() -> Vec<String> {
    let host = cpal::default_host();
    let mut out = vec!["(default)".to_string()];
    if let Ok(iter) = host.output_devices() {
        for d in iter {
            if let Ok(name) = d.name() {
                out.push(name);
            }
        }
    }
    out
}

fn open_stream(name: &str) -> Result<(OutputStream, OutputStreamHandle), String> {
    if name == "(default)" {
        return OutputStream::try_default().map_err(|e| format!("default: {e}"));
    }
    let host = cpal::default_host();
    let iter = host.output_devices().map_err(|e| format!("enumerate: {e}"))?;
    for d in iter {
        if d.name().ok().as_deref() == Some(name) {
            return OutputStream::try_from_device(&d)
                .map_err(|e| format!("device {name}: {e}"));
        }
    }
    Err(format!("device not found: {name}"))
}

impl Jukebox {
    fn new(dirs: Vec<PathBuf>) -> Result<Self, String> {
        let (stream, handle) =
            open_stream("(default)").map_err(|e| format!("audio output: {e}"))?;
        let dir_refs: Vec<&Path> = dirs.iter().map(|p| p.as_path()).collect();
        let tracks = scan_dirs(&dir_refs);
        let devices = list_output_devices();
        Ok(Self {
            tracks,
            selected: None,
            filter: String::new(),
            _stream: stream,
            handle,
            sink: None,
            refresh_dirs: dirs,
            devices,
            current_device: "(default)".to_string(),
        })
    }

    fn rebind_device(&mut self, name: &str) {
        // Kill any in-flight playback first; we're tearing down the
        // stream the sink belongs to.
        if let Some(s) = self.sink.take() {
            s.stop();
            drop(s);
        }
        match open_stream(name) {
            Ok((stream, handle)) => {
                self._stream = stream;
                self.handle = handle;
                self.current_device = name.to_string();
            }
            Err(e) => {
                eprintln!("jukebox: rebind {name}: {e}");
            }
        }
    }

    fn rescan_devices(&mut self) {
        self.devices = list_output_devices();
    }

    fn rescan(&mut self) {
        let dir_refs: Vec<&Path> = self.refresh_dirs.iter().map(|p| p.as_path()).collect();
        self.tracks = scan_dirs(&dir_refs);
    }

    fn play(&mut self, idx: usize) {
        // Stop any currently-playing sink first. stop() detaches the
        // running source; dropping the Sink afterwards releases the
        // audio thread so the next track starts cleanly.
        self.stop();
        let track = match self.tracks.get(idx) {
            Some(t) => t,
            None => return,
        };
        let file = match File::open(&track.path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("jukebox: open {}: {e}", track.path.display());
                return;
            }
        };
        let decoder = match Decoder::new(BufReader::new(file)) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("jukebox: decode {}: {e}", track.path.display());
                return;
            }
        };
        let sink = match Sink::try_new(&self.handle) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("jukebox: sink: {e}");
                return;
            }
        };
        sink.append(decoder);
        self.sink = Some(sink);
        self.selected = Some(idx);
    }

    fn stop(&mut self) {
        if let Some(s) = self.sink.take() {
            // Sink::stop detaches the queued source. Drop also releases
            // the spawned audio thread for this sink. Both together
            // guarantee playback stops within the audio device's
            // buffer-flush latency (typically ~20-50 ms).
            s.stop();
            drop(s);
        }
        self.selected = None;
    }
}

impl eframe::App for Jukebox {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(std::time::Duration::from_millis(100));

        // Auto-clear selected when sink finishes naturally.
        if let Some(s) = &self.sink {
            if s.empty() {
                self.sink = None;
            }
        }
        let playing_now = self.sink.is_some();

        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("keysynth jukebox");
                ui.separator();
                if ui.button(if playing_now { "■ stop" } else { "—" }).clicked() {
                    self.stop();
                }
                ui.separator();
                if ui.button("rescan files").clicked() {
                    self.rescan();
                }
                ui.separator();
                ui.label(format!("{} tracks", self.tracks.len()));
                ui.separator();
                ui.label("filter:");
                ui.text_edit_singleline(&mut self.filter);
            });
            ui.horizontal(|ui| {
                ui.label("output:");
                let current = self.current_device.clone();
                let mut new_pick: Option<String> = None;
                egui::ComboBox::from_id_salt("audio-device")
                    .selected_text(&current)
                    .width(360.0)
                    .show_ui(ui, |ui| {
                        for dev in &self.devices {
                            if ui
                                .selectable_label(*dev == current, dev)
                                .clicked()
                            {
                                new_pick = Some(dev.clone());
                            }
                        }
                    });
                if let Some(name) = new_pick {
                    if name != current {
                        self.rebind_device(&name);
                    }
                }
                if ui.button("rescan dev").clicked() {
                    self.rescan_devices();
                }
                if ui
                    .button("→ default")
                    .on_hover_text(
                        "Re-bind to current OS default device. Use this when you've \
                         changed the default in Windows Sound (e.g. unplugged headphones).",
                    )
                    .clicked()
                {
                    self.rebind_device("(default)");
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let needle = self.filter.to_lowercase();
            // Group tracks by piece. Each piece row gets a selectable
            // button per engine so the same sequence rendered through
            // different engines lines up for one-click A/B switching.
            let mut grouped: std::collections::BTreeMap<String, Vec<usize>> =
                std::collections::BTreeMap::new();
            for (i, t) in self.tracks.iter().enumerate() {
                if !needle.is_empty() && !t.piece.to_lowercase().contains(&needle) {
                    continue;
                }
                grouped.entry(t.piece.clone()).or_default().push(i);
            }

            egui::ScrollArea::vertical().show(ui, |ui| {
                let mut to_play: Option<usize> = None;
                for (piece, indices) in &grouped {
                    ui.horizontal_wrapped(|ui| {
                        let active_in_group = indices
                            .iter()
                            .any(|i| self.selected == Some(*i) && playing_now);
                        let piece_text = if active_in_group {
                            egui::RichText::new(piece)
                                .color(egui::Color32::from_rgb(255, 200, 80))
                                .strong()
                                .monospace()
                        } else {
                            egui::RichText::new(piece).monospace()
                        };
                        ui.add_sized([260.0, 22.0], egui::Label::new(piece_text));
                        ui.separator();
                        for &i in indices {
                            let t = &self.tracks[i];
                            let active = self.selected == Some(i) && playing_now;
                            let label = if active {
                                format!("▶ {}", t.engine)
                            } else {
                                t.engine.clone()
                            };
                            if ui
                                .selectable_label(active, label)
                                .on_hover_text(format!(
                                    "{} ({} KB)",
                                    t.path.display(),
                                    t.size_kb
                                ))
                                .clicked()
                            {
                                to_play = Some(i);
                            }
                        }
                    });
                    ui.add_space(2.0);
                }
                if let Some(i) = to_play {
                    self.play(i);
                }
            });
        });
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dirs = vec![
        PathBuf::from("bench-out/iterH"),
        PathBuf::from("bench-out/songs"),
        // ai-chiptune-demo / listener-lab JSONs rendered via
        // `render_chiptune --engine ENGINE --in path/to/song.json
        //  --out bench-out/CHIPTUNE/<name>_<engine>.wav`
        PathBuf::from("bench-out/CHIPTUNE"),
    ];
    let app = Jukebox::new(dirs)?;
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([720.0, 600.0])
            .with_title("keysynth jukebox"),
        ..Default::default()
    };
    eframe::run_native(
        "keysynth jukebox",
        options,
        Box::new(|_cc| Ok(Box::new(app))),
    )
    .map_err(|e| -> Box<dyn std::error::Error> { format!("eframe: {e}").into() })?;
    Ok(())
}
