//! egui dashboard for keysynth.
//!
//! Layout:
//!   - Left side panel: voice browser (Piano / Guitar / Synth / Samples /
//!     Custom categories, each containing selectable `VoiceSlot`s — Tier
//!     1+2 piano variants, the modeled-guitar pair (STK port + KS+IR
//!     legacy), the lightweight synth set, sample-based SFZ / SF2 pianos,
//!     and user-saved presets). Each slot carries an editorial tier
//!     (`Recommend::Best/Stable/Experimental`) shown as a small label
//!     after the slot name; entries within a category are sorted by tier
//!     so the recommended pick floats to the top. Selecting a slot
//!     hot-swaps engine + ModalParams + asset in one operation; the
//!     next note_on picks up the new voice. For named live slots
//!     (Guitar STK / Guitar KS) the apply path also asks the reloader
//!     to make that slot active, building the `voices_live/<subdir>/`
//!     crate on demand if no factory has been loaded yet.
//!   - Top panel: master / reverb / mix mode (always-relevant globals).
//!   - Central panel: ModalParams sliders (when piano-modal active),
//!     SF program picker (when sf-piano active), CC controls, held
//!     notes grid.
//!   - Right side panel: live MIDI event log.

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cpal::Stream;
use eframe::egui;
use midir::MidiInputConnection;
use rustysynth::{SoundFont, Synthesizer, SynthesizerSettings};

use crate::cp::Server as CpServer;
use crate::gm::{GM_FAMILIES, GM_INSTRUMENTS};
use crate::live_reload::{Reloader, Status as LiveStatus};
use crate::sfz::SfzPlayer;
use crate::synth::{
    make_voice, midi_to_freq, modal_params, set_modal_params, set_modal_physics, DashState, Engine,
    LiveParams, MixMode, ModalParams, Voice, VoiceImpl,
};
use crate::voice_lib::{Category, Recommend, VoiceLibrary, VoiceSlot};

/// Bundle passed from `main` into `run_app`. Holds the long-lived audio /
/// MIDI handles so they aren't dropped while the GUI is open.
pub struct AppContext {
    pub stream: Stream,
    /// MIDI input connection. `None` when no MIDI input was found (or
    /// `--port`-less startup on a machine with no devices); the egui PC
    /// keyboard path still works without it. Stored so the connection
    /// (and its callback closures) live as long as the GUI window.
    pub midi_conn: Option<MidiInputConnection<()>>,
    pub live: Arc<Mutex<LiveParams>>,
    pub dash: Arc<Mutex<DashState>>,
    pub voices: Arc<Mutex<Vec<Voice>>>,
    pub synth: Arc<Mutex<Option<Synthesizer>>>,
    pub sfz: Arc<Mutex<Option<SfzPlayer>>>,
    pub port_name: String,
    pub out_name: String,
    pub sr_hz: u32,
    /// Auto-discovered (or `--sfz`-supplied) SFZ path from `main`.
    /// Used to seed the matching voice-browser entry as already-loaded
    /// so no re-decode happens on first selection.
    pub startup_sfz_path: Option<PathBuf>,
    pub startup_sf2_path: Option<PathBuf>,
    pub startup_engine: Engine,
    /// Voice hot-reload handle. `None` if `main()` couldn't find a
    /// `voices_live/` crate to watch (release tarball without it,
    /// running outside the repo, etc.). When None, the "Live (hot edit)"
    /// browser entry is hidden and `Engine::Live` falls back to silence.
    pub live_reloader: Option<Reloader>,
    /// Voice Control Protocol server handle. `None` unless `--cp` was
    /// passed. When `Some`, the side panel renders the CP status
    /// (endpoint + connection count); when `None`, the panel hides the
    /// CP block entirely so the unmodified GUI experience is preserved.
    pub cp_server: Option<CpServer>,
}

pub fn run_app(ctx: AppContext) -> Result<(), Box<dyn std::error::Error>> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 720.0])
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
    library: VoiceLibrary,
    /// Last asset (SFZ / SF2) load result for the user to read in the
    /// GUI. Cleared by the next click; kept short so the status bar
    /// doesn't grow.
    last_asset_msg: Option<(egui::Color32, String)>,
    /// Slot index pending an inline rename (TextEdit instead of label).
    pending_rename: Option<usize>,
    rename_buf: String,
    /// Path of the SFZ that's currently decoded into `ctx.sfz`. Used
    /// to skip a re-decode when the user re-selects the same Salamander
    /// entry; otherwise every click would block the GUI for ~5 s.
    loaded_sfz_path: Option<PathBuf>,
    /// (path, program, bank) of the currently-loaded SF2 in `ctx.synth`.
    /// A program/bank change to the same file is cheap (one MIDI
    /// message); a path change requires a re-load.
    loaded_sf2: Option<(PathBuf, u8, u8)>,
}

impl KeysynthApp {
    fn new(ctx: AppContext) -> Self {
        // Plugin discovery root: `Reloader::spawn` is called with the
        // `voices_live/` directory, so `reloader.crate_root` already
        // is that root. Fall back to the hard-coded `voices_live`
        // relative path when no reloader exists (release tarball,
        // --no-cp invocation) so the GUI still catalogs whatever
        // plugin manifests are reachable from the cwd.
        let voices_live_root: Option<PathBuf> = ctx
            .live_reloader
            .as_ref()
            .map(|r| r.crate_root.clone())
            .or_else(|| {
                let p = PathBuf::from("voices_live");
                p.is_dir().then_some(p)
            });

        let library = VoiceLibrary::load(
            ctx.startup_sfz_path.as_deref(),
            ctx.startup_sf2_path.as_deref(),
            ctx.startup_engine,
            voices_live_root.as_deref(),
        );
        // Audit line for the GUI organize PR: surfaces the catalog
        // count at startup so the discovery test and the manual smoke
        // verify share a single observable. Filter to `builtin == true`
        // so the count isn't inflated by user-saved Custom presets.
        let builtin_count = library.slots.iter().filter(|s| s.builtin).count();
        eprintln!("voice_lib: {builtin_count} builtins loaded");
        // Seed loaded-asset trackers from main's auto-discovery so the
        // first click on the matching browser entry is a no-op (no
        // 5-second SFZ decode for an already-loaded sample set).
        let loaded_sfz_path = ctx.startup_sfz_path.clone();
        let loaded_sf2 = ctx.startup_sf2_path.clone().map(|p| (p, 0u8, 0u8));
        Self {
            ctx,
            library,
            last_asset_msg: None,
            pending_rename: None,
            rename_buf: String::new(),
            loaded_sfz_path,
            loaded_sf2,
        }
    }

    fn set_msg_ok(&mut self, msg: String) {
        eprintln!("keysynth: {msg}");
        self.last_asset_msg = Some((egui::Color32::from_rgb(120, 220, 120), msg));
    }

    fn set_msg_err(&mut self, msg: String) {
        eprintln!("keysynth: {msg}");
        self.last_asset_msg = Some((egui::Color32::from_rgb(255, 120, 120), msg));
    }

    /// Atomically replace the SFZ player. Blocks the GUI for the
    /// decode (Salamander V3 ≈ 5 s) — audio keeps running on the
    /// previous player until the lock is taken and the swap is one
    /// mutex acquire.
    fn load_sfz_path(&mut self, path: &std::path::Path) {
        let started = std::time::Instant::now();
        match SfzPlayer::load(path, self.ctx.sr_hz as f32) {
            Ok(player) => {
                let regions = player.regions_len();
                let samples = player.samples_len();
                *self.ctx.sfz.lock().unwrap() = Some(player);
                self.loaded_sfz_path = Some(path.to_path_buf());
                self.set_msg_ok(format!(
                    "loaded SFZ '{}' ({} regions, {} samples, {:.1}s)",
                    path.display(),
                    regions,
                    samples,
                    started.elapsed().as_secs_f32(),
                ));
            }
            Err(e) => {
                self.set_msg_err(format!("SFZ load failed for {}: {e}", path.display()));
            }
        }
    }

    fn load_sf2_path(&mut self, path: &std::path::Path, program: u8, bank: u8) {
        let result: Result<Synthesizer, String> = (|| {
            let mut file =
                BufReader::new(File::open(path).map_err(|e| format!("opening SoundFont: {e}"))?);
            let sf = std::sync::Arc::new(
                SoundFont::new(&mut file).map_err(|e| format!("parsing SoundFont: {e}"))?,
            );
            let mut settings = SynthesizerSettings::new(self.ctx.sr_hz as i32);
            settings.maximum_polyphony = 256;
            let mut synth = Synthesizer::new(&sf, &settings)
                .map_err(|e| format!("building Synthesizer: {e}"))?;
            if bank > 0 {
                synth.process_midi_message(0, 0xB0, 0, bank as i32);
            }
            synth.process_midi_message(0, 0xC0, program as i32, 0);
            Ok(synth)
        })();
        match result {
            Ok(synth) => {
                *self.ctx.synth.lock().unwrap() = Some(synth);
                self.loaded_sf2 = Some((path.to_path_buf(), program, bank));
                self.set_msg_ok(format!(
                    "loaded SoundFont '{}' (program={program} bank={bank})",
                    path.display(),
                ));
            }
            Err(e) => {
                self.set_msg_err(format!("SF2 load failed for {}: {e}", path.display()));
            }
        }
    }

    /// Apply a `VoiceSlot` selection: engine swap + modal params +
    /// asset hot-load (skipping the load if the same asset is already
    /// in the shared slot).
    fn apply_slot(&mut self, idx: usize) {
        let Some(slot) = self.library.slots.get(idx).cloned() else {
            return;
        };
        self.library.active = idx;

        // 1. Engine swap (cheap, just a write to LiveParams).
        {
            let mut lp = self.ctx.live.lock().unwrap();
            lp.engine = slot.engine;
            // Note-off semantics travel with the slot: piano variants
            // keep `Damper` (default), guitar plugins write `Natural`.
            // The MIDI callback's note-off arm reads this every key-up
            // and skips `trigger_release()` for plucked voices so they
            // ride out their natural loop-filter decay instead of
            // getting cut short by an unconditional damper pull.
            lp.decay_model = slot.decay_model;
            if let (Some(p), Some(b)) = (slot.sf_program, slot.sf_bank) {
                lp.sf_program = p;
                lp.sf_bank = b;
            }
        }

        // 2. ModalParams + physics flag (only meaningful for PianoModal).
        if slot.engine == Engine::PianoModal {
            if let Some(p) = slot.params {
                set_modal_params(p);
            }
            set_modal_physics(slot.modal_physics);
        }

        // 3. Asset hot-load — skip if path matches the currently-loaded.
        match (slot.engine, slot.asset_path.as_ref()) {
            (Engine::SfzPiano, Some(p)) => {
                let already = self
                    .loaded_sfz_path
                    .as_ref()
                    .map(|cur| cur == p)
                    .unwrap_or(false);
                if !already {
                    self.load_sfz_path(p);
                } else {
                    self.set_msg_ok(format!("selected SFZ '{}' (already loaded)", p.display(),));
                }
            }
            (Engine::SfPiano, Some(p)) => {
                let prog = slot.sf_program.unwrap_or(0);
                let bank = slot.sf_bank.unwrap_or(0);
                let needs_reload = self
                    .loaded_sf2
                    .as_ref()
                    .map(|(cur, _, _)| cur != p)
                    .unwrap_or(true);
                if needs_reload {
                    self.load_sf2_path(p, prog, bank);
                } else {
                    // Same file, just update program/bank via MIDI.
                    if let Some(synth) = self.ctx.synth.lock().unwrap().as_mut() {
                        if bank > 0 {
                            synth.process_midi_message(0, 0xB0, 0, bank as i32);
                        }
                        synth.process_midi_message(0, 0xC0, prog as i32, 0);
                    }
                    self.loaded_sf2 = Some((p.clone(), prog, bank));
                    self.set_msg_ok(format!(
                        "switched SF2 to program={prog} bank={bank} ({})",
                        p.display(),
                    ));
                }
            }
            _ => {}
        }

        // 4. Named live-slot bridge. Engine::Live with a `live_slot_name`
        //    means the user picked one of the modeled-guitar entries (or
        //    any future named cdylib voice). Try `set_active(name)` first
        //    — that's a constant-time pointer swap if the slot has
        //    already been built. If the slot is unknown to the reloader
        //    we kick off a background `build_into_slot` so the cargo
        //    compile doesn't freeze the UI thread; the existing live
        //    status panel surfaces "building..." → "loaded in N ms" /
        //    "ERR <msg>" feedback verbatim. Slots without a
        //    `live_slot_name` (the legacy "Live (hot edit)" entry) fall
        //    through unchanged — the reloader keeps watching whatever
        //    `crate_root` it was spawned against.
        if slot.engine == Engine::Live {
            if let (Some(reloader), Some(slot_name)) =
                (self.ctx.live_reloader.clone(), slot.live_slot_name.clone())
            {
                if reloader.set_active(&slot_name).is_ok() {
                    self.set_msg_ok(format!("live slot '{slot_name}' active"));
                } else if let Some(subdir) = slot.live_crate_subdir.clone() {
                    // `reloader.crate_root` is the `voices_live/` root
                    // (Reloader::spawn is given that path), so each
                    // plugin crate lives at `<root>/<subdir>/`.
                    let crate_root = reloader.crate_root.join(&subdir);
                    self.set_msg_ok(format!(
                        "building '{slot_name}' from {} ...",
                        crate_root.display()
                    ));
                    let reloader_bg = reloader.clone();
                    let slot_for_bg = slot_name.clone();
                    std::thread::spawn(move || {
                        if reloader_bg
                            .build_into_slot(&crate_root, &slot_for_bg)
                            .is_ok()
                        {
                            let _ = reloader_bg.set_active(&slot_for_bg);
                        }
                    });
                } else {
                    self.set_msg_err(format!(
                        "live slot '{slot_name}' has no crate subdir; cannot build"
                    ));
                }
            }
        }
    }

    fn dialog_load_sfz(&mut self) {
        let picked = rfd::FileDialog::new()
            .add_filter("SFZ", &["sfz"])
            .set_title("Load SFZ instrument")
            .pick_file();
        let Some(path) = picked else { return };
        let label = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("SFZ")
            .to_string();
        let slot = VoiceSlot::sfz(&label, path.clone(), false);
        self.library.add_and_select(slot);
        // Force load even if a file with the same path was previously
        // discovered (user explicitly picked it).
        self.load_sfz_path(&path);
    }

    fn dialog_load_sf2(&mut self) {
        let picked = rfd::FileDialog::new()
            .add_filter("SoundFont", &["sf2"])
            .set_title("Load SoundFont (.sf2)")
            .pick_file();
        let Some(path) = picked else { return };
        let label = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("SF2")
            .to_string();
        let slot = VoiceSlot::sf2(&label, path.clone(), 0, 0, false);
        self.library.add_and_select(slot);
        self.load_sf2_path(&path, 0, 0);
    }

    /// Save the current ModalParams + physics flag as a new Custom slot.
    fn save_current_as_custom(&mut self) {
        let n = self
            .library
            .slots
            .iter()
            .filter(|s| s.category == Category::Custom)
            .count();
        let label = format!("Custom {}", n + 1);
        let physics = crate::synth::modal_physics_enabled();
        let slot = VoiceSlot::custom(&label, modal_params(), physics);
        self.library.add_and_select(slot);
        self.pending_rename = Some(self.library.slots.len() - 1);
        self.rename_buf = label;
    }
}

/// QWERTY → MIDI mapping (FL-Studio-style two-octave layout).
const QWERTY_TO_MIDI: &[(egui::Key, u8)] = &[
    (egui::Key::Z, 60),
    (egui::Key::S, 61),
    (egui::Key::X, 62),
    (egui::Key::D, 63),
    (egui::Key::C, 64),
    (egui::Key::V, 65),
    (egui::Key::G, 66),
    (egui::Key::B, 67),
    (egui::Key::H, 68),
    (egui::Key::N, 69),
    (egui::Key::J, 70),
    (egui::Key::M, 71),
    (egui::Key::Q, 72),
    (egui::Key::Num2, 73),
    (egui::Key::W, 74),
    (egui::Key::Num3, 75),
    (egui::Key::E, 76),
    (egui::Key::R, 77),
    (egui::Key::Num5, 78),
    (egui::Key::T, 79),
    (egui::Key::Num6, 80),
    (egui::Key::Y, 81),
    (egui::Key::Num7, 82),
    (egui::Key::U, 83),
    (egui::Key::I, 84),
];

const QWERTY_MIDI_CHANNEL: u8 = 15;
const QWERTY_VELOCITY: u8 = 100;

impl eframe::App for KeysynthApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(Duration::from_millis(16));

        // QWERTY → voice pool injection.
        let (engine, sr_hz) = {
            let lp = self.ctx.live.lock().unwrap();
            (lp.engine, self.ctx.sr_hz as f32)
        };
        let mut presses: Vec<u8> = Vec::new();
        let mut releases: Vec<u8> = Vec::new();
        ctx.input(|i| {
            for event in &i.events {
                if let egui::Event::Key {
                    key,
                    pressed,
                    repeat,
                    ..
                } = event
                {
                    if *repeat {
                        continue;
                    }
                    if let Some((_, n)) = QWERTY_TO_MIDI.iter().find(|(k, _)| k == key) {
                        if *pressed {
                            presses.push(*n);
                        } else {
                            releases.push(*n);
                        }
                    }
                }
            }
        });
        if !presses.is_empty() || !releases.is_empty() {
            if engine == Engine::SfzPiano {
                if let Some(player) = self.ctx.sfz.lock().unwrap().as_mut() {
                    for n in &presses {
                        player.note_on(QWERTY_MIDI_CHANNEL, *n, QWERTY_VELOCITY);
                    }
                    for n in &releases {
                        player.note_off(QWERTY_MIDI_CHANNEL, *n);
                    }
                }
            } else if engine == Engine::SfPiano {
                if let Some(synth) = self.ctx.synth.lock().unwrap().as_mut() {
                    for n in &presses {
                        synth.note_on(
                            QWERTY_MIDI_CHANNEL as i32,
                            *n as i32,
                            QWERTY_VELOCITY as i32,
                        );
                    }
                    for n in &releases {
                        synth.note_off(QWERTY_MIDI_CHANNEL as i32, *n as i32);
                    }
                }
            }
            let mut pool = self.ctx.voices.lock().unwrap();
            let mut dash = self.ctx.dash.lock().unwrap();
            for note in presses {
                let freq = midi_to_freq(note);
                let inner: Box<dyn VoiceImpl> = make_voice(engine, sr_hz, freq, QWERTY_VELOCITY);
                let v = Voice {
                    key: (QWERTY_MIDI_CHANNEL, note),
                    inner,
                };
                if let Some(slot) = pool
                    .iter_mut()
                    .find(|x| x.key == (QWERTY_MIDI_CHANNEL, note))
                {
                    *slot = v;
                } else {
                    const MAX_VOICES: usize = 32;
                    if pool.len() >= MAX_VOICES {
                        let evict_idx = pool
                            .iter()
                            .position(|x| x.inner.is_done() || x.inner.is_releasing())
                            .unwrap_or(0);
                        pool.remove(evict_idx);
                    }
                    pool.push(v);
                }
                dash.active_notes.insert((QWERTY_MIDI_CHANNEL, note));
                dash.push_event(format!(
                    "qwerty   ch{QWERTY_MIDI_CHANNEL} n{note} v{QWERTY_VELOCITY}"
                ));
            }
            for note in releases {
                if let Some(slot) = pool
                    .iter_mut()
                    .find(|x| x.key == (QWERTY_MIDI_CHANNEL, note))
                {
                    slot.inner.trigger_release();
                }
                dash.active_notes.remove(&(QWERTY_MIDI_CHANNEL, note));
                dash.push_event(format!("qwerty-off ch{QWERTY_MIDI_CHANNEL} n{note}"));
            }
        }

        // Snapshot shared state under brief locks; release before drawing.
        let (
            mut master,
            engine,
            mut reverb_wet,
            mut sf_program,
            mut sf_bank,
            mut mix_mode,
            dash_snapshot,
        ) = {
            let lp = self.ctx.live.lock().unwrap();
            let m = lp.master;
            let e = lp.engine;
            let r = lp.reverb_wet;
            let p = lp.sf_program;
            let b = lp.sf_bank;
            let mm = lp.mix_mode;
            drop(lp);
            let d = self.ctx.dash.lock().unwrap();
            let snap = DashSnapshot {
                cc_raw: d.cc_raw.clone(),
                cc_count: d.cc_count.clone(),
                active_notes: d.active_notes.clone(),
                recent: d.recent.iter().rev().take(60).cloned().collect(),
            };
            (m, e, r, p, b, mm, snap)
        };

        let master_before = master;
        let reverb_before = reverb_wet;
        let sf_program_before = sf_program;
        let sf_bank_before = sf_bank;
        let mix_mode_before = mix_mode;

        // ─── Top status panel ─────────────────────────────────────────
        egui::TopBottomPanel::top("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("keysynth").strong().size(18.0));
                ui.separator();
                ui.label(format!("MIDI: {}", self.ctx.port_name));
                ui.separator();
                ui.label(format!("Audio: {}", self.ctx.out_name));
                ui.separator();
                ui.label(format!("{} Hz", self.ctx.sr_hz));
                ui.separator();
                if let Some(slot) = self.library.active_slot() {
                    ui.label(
                        egui::RichText::new(format!("voice: {}", slot.label))
                            .strong()
                            .color(egui::Color32::from_rgb(255, 220, 120)),
                    );
                }
            });
            ui.horizontal(|ui| {
                ui.label("master:");
                ui.add(
                    egui::Slider::new(&mut master, 0.0..=10.0)
                        .step_by(0.1)
                        .text("(K1/CC70/CC7)"),
                );
                ui.separator();
                ui.label("reverb:");
                ui.add(
                    egui::Slider::new(&mut reverb_wet, 0.0..=1.0)
                        .step_by(0.01)
                        .text("(body IR wet)"),
                );
            });
            ui.horizontal_wrapped(|ui| {
                ui.label("mix:");
                for m in MixMode::ALL {
                    let selected = mix_mode == *m;
                    if ui.selectable_label(selected, m.as_label()).clicked() {
                        mix_mode = *m;
                    }
                }
            });
            if let Some((color, text)) = &self.last_asset_msg {
                ui.colored_label(*color, text);
            }
        });

        // ─── Left side panel: voice browser ───────────────────────────
        let mut clicked_slot: Option<usize> = None;
        let mut want_load_sfz = false;
        let mut want_load_sf2 = false;
        let mut want_save_custom = false;
        let mut want_remove: Option<usize> = None;
        let mut want_rename: Option<usize> = None;

        egui::SidePanel::left("voice_browser")
            .default_width(280.0)
            .min_width(220.0)
            .show(ctx, |ui| {
                ui.heading("Voices");
                ui.label(
                    egui::RichText::new(
                        "Click to switch instrument live. \
                         + buttons add new entries.",
                    )
                    .small()
                    .color(egui::Color32::from_rgb(170, 170, 170)),
                );
                ui.separator();

                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for category in Category::ALL {
                            // Collect indices belonging to this category and
                            // sort them Best→Stable→Experimental. Stable
                            // sort by index keeps the declared order from
                            // `voice_lib::builtins()` as the tie-breaker so
                            // e.g. `Piano` still renders above `Piano Modal`
                            // even though both are Best.
                            let mut idxs: Vec<usize> = self
                                .library
                                .slots
                                .iter()
                                .enumerate()
                                .filter(|(_, s)| s.category == *category)
                                .map(|(i, _)| i)
                                .collect();
                            idxs.sort_by_key(|i| self.library.slots[*i].recommend.rank());

                            egui::CollapsingHeader::new(category.label())
                                .default_open(true)
                                .show(ui, |ui| {
                                    for idx in idxs {
                                        let slot = &self.library.slots[idx];
                                        let is_active = self.library.active == idx;
                                        let in_rename = self.pending_rename == Some(idx);
                                        ui.horizontal(|ui| {
                                            ui.label(if is_active { "●" } else { "○" });
                                            if in_rename {
                                                let resp = ui.add(
                                                    egui::TextEdit::singleline(
                                                        &mut self.rename_buf,
                                                    )
                                                    .desired_width(180.0),
                                                );
                                                if resp.lost_focus()
                                                    && ui.input(|i| i.key_pressed(egui::Key::Enter))
                                                {
                                                    want_rename = Some(idx);
                                                }
                                            } else {
                                                let label =
                                                    egui::RichText::new(&slot.label).monospace();
                                                let label = if is_active {
                                                    label.strong().color(egui::Color32::from_rgb(
                                                        255, 220, 120,
                                                    ))
                                                } else {
                                                    label
                                                };
                                                let resp = ui.selectable_label(is_active, label);
                                                if resp.clicked() {
                                                    clicked_slot = Some(idx);
                                                }
                                                // Editorial tier as a small
                                                // colored label trailing the
                                                // slot name. Color cues:
                                                // green=Best, gray=Stable,
                                                // amber=Experimental — same
                                                // palette the live status
                                                // panel uses below.
                                                let (rec_color, rec_text) = match slot.recommend {
                                                    Recommend::Best => (
                                                        egui::Color32::from_rgb(120, 220, 120),
                                                        slot.recommend.label(),
                                                    ),
                                                    Recommend::Stable => (
                                                        egui::Color32::from_rgb(170, 170, 170),
                                                        slot.recommend.label(),
                                                    ),
                                                    Recommend::Experimental => (
                                                        egui::Color32::from_rgb(220, 200, 100),
                                                        slot.recommend.label(),
                                                    ),
                                                };
                                                ui.label(
                                                    egui::RichText::new(rec_text)
                                                        .small()
                                                        .color(rec_color),
                                                );
                                                if !slot.builtin {
                                                    resp.context_menu(|ui| {
                                                        if ui.button("Rename").clicked() {
                                                            self.pending_rename = Some(idx);
                                                            self.rename_buf = slot.label.clone();
                                                            ui.close_menu();
                                                        }
                                                        if ui.button("Remove").clicked() {
                                                            want_remove = Some(idx);
                                                            ui.close_menu();
                                                        }
                                                    });
                                                }
                                            }
                                        });
                                        // One-line description under the
                                        // label, indented and dimmed so it
                                        // doesn't compete with the click
                                        // target. Empty for legacy / user
                                        // slots — rendering skips those.
                                        if !slot.description.is_empty() {
                                            ui.horizontal(|ui| {
                                                ui.add_space(16.0);
                                                ui.label(
                                                    egui::RichText::new(&slot.description)
                                                        .small()
                                                        .color(egui::Color32::from_rgb(
                                                            150, 150, 150,
                                                        )),
                                                );
                                            });
                                        }
                                    }
                                    // Per-category action buttons.
                                    ui.add_space(2.0);
                                    match category {
                                        Category::Samples => {
                                            ui.horizontal(|ui| {
                                                if ui.small_button("+ Load SFZ...").clicked() {
                                                    want_load_sfz = true;
                                                }
                                                if ui.small_button("+ Load SF2...").clicked() {
                                                    want_load_sf2 = true;
                                                }
                                            });
                                        }
                                        Category::Custom => {
                                            if ui.small_button("+ Save current...").clicked() {
                                                want_save_custom = true;
                                            }
                                        }
                                        Category::Piano | Category::Guitar | Category::Synth => {}
                                    }
                                });
                        }
                    });

                // ─── Live-reload status (bottom of voice browser) ────
                //
                // Surfaces dll path, last reload timestamp, build errors
                // verbatim. The brief flags silent reload failure as the
                // single worst-case UX outcome — errors render in red,
                // monospaced, and stick around until the next successful
                // reload. The Ctrl+R / "Rebuild now" button lets the
                // user kick a build manually if the watcher missed an
                // event (rare on Windows under some editors).
                if let Some(reloader) = self.ctx.live_reloader.as_ref() {
                    ui.add_space(8.0);
                    ui.separator();
                    ui.label(
                        egui::RichText::new("Live voice")
                            .strong()
                            .color(egui::Color32::from_rgb(220, 220, 255)),
                    );
                    let status = reloader.status_snapshot();
                    let (color, text) = match &status {
                        LiveStatus::Idle => {
                            (egui::Color32::from_rgb(180, 180, 180), "idle".to_string())
                        }
                        LiveStatus::Building { reason, .. } => (
                            egui::Color32::from_rgb(220, 200, 100),
                            format!("building... ({reason})"),
                        ),
                        LiveStatus::Ok { duration, .. } => (
                            egui::Color32::from_rgb(120, 220, 120),
                            format!("loaded in {} ms", duration.as_millis()),
                        ),
                        LiveStatus::Err { message, .. } => (
                            egui::Color32::from_rgb(255, 110, 110),
                            format!("ERR\n{message}"),
                        ),
                    };
                    ui.colored_label(color, &text);
                    if let Some((dll, _)) = reloader.current_meta() {
                        ui.label(
                            egui::RichText::new(format!(
                                "dll: {}",
                                dll.file_name().and_then(|s| s.to_str()).unwrap_or("<?>")
                            ))
                            .small()
                            .monospace(),
                        );
                    }
                    ui.label(
                        egui::RichText::new(format!("src: {}", reloader.crate_root.display()))
                            .small(),
                    );
                    if ui.small_button("Rebuild now").clicked() {
                        reloader.request_rebuild("manual");
                    }
                }

                // ─── CP server status ─────────────────────────────────
                //
                // Only renders when --cp was passed at startup. Surfaces
                // the bound endpoint + live connection count so the user
                // can confirm ksctl invocations are reaching the right
                // server (esp. when KEYSYNTH_CP overrides the path).
                if let Some(cp) = self.ctx.cp_server.as_ref() {
                    ui.add_space(8.0);
                    ui.separator();
                    ui.label(
                        egui::RichText::new("CP server")
                            .strong()
                            .color(egui::Color32::from_rgb(220, 220, 255)),
                    );
                    let color = if cp.is_ready() {
                        egui::Color32::from_rgb(120, 220, 120)
                    } else {
                        egui::Color32::from_rgb(220, 200, 100)
                    };
                    let text = if cp.is_ready() {
                        format!(
                            "running ({} connection{})",
                            cp.connection_count(),
                            if cp.connection_count() == 1 { "" } else { "s" }
                        )
                    } else {
                        "binding...".to_string()
                    };
                    ui.colored_label(color, &text);
                    ui.label(
                        egui::RichText::new(format!("endpoint: {}", cp.endpoint))
                            .small()
                            .monospace(),
                    );
                    ui.label(egui::RichText::new(format!("sr: {} Hz", cp.sr_hz)).small());
                }
            });

        // Ctrl+R also triggers a rebuild — handy when keeping focus
        // on the keyboard while iterating on a voice.
        if let Some(reloader) = self.ctx.live_reloader.as_ref() {
            ctx.input(|i| {
                if i.modifiers.command_only() && i.key_pressed(egui::Key::R) {
                    reloader.request_rebuild("Ctrl+R");
                }
            });
        }

        // ─── Right side panel: MIDI log ───────────────────────────────
        egui::SidePanel::right("log")
            .default_width(280.0)
            .show(ctx, |ui| {
                ui.heading("MIDI log");
                ui.separator();
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for line in dash_snapshot.recent.iter() {
                            ui.monospace(line);
                        }
                    });
            });

        // ─── Central panel: contextual editors + CC + notes ───────────
        egui::CentralPanel::default().show(ctx, |ui| {
            // Modal sliders only when the active engine cares.
            if engine == Engine::PianoModal {
                ui.heading("Modal params");
                let mut p = modal_params();
                let p_before = p;
                ui.horizontal_wrapped(|ui| {
                    ui.add(
                        egui::Slider::new(&mut p.detune_cents, 0.0..=3.0)
                            .step_by(0.05)
                            .text("detune ¢"),
                    );
                    ui.add(
                        egui::Slider::new(&mut p.pol_h_weight, 0.0..=0.5)
                            .step_by(0.01)
                            .text("H-pol"),
                    );
                    ui.add(
                        egui::Slider::new(&mut p.t60_cap_sec, 2.0..=30.0)
                            .step_by(0.5)
                            .text("T60 cap (s)"),
                    );
                });
                ui.horizontal_wrapped(|ui| {
                    ui.add(
                        egui::Slider::new(&mut p.stage_b_gain, 0.0..=0.30)
                            .step_by(0.005)
                            .text("noise tail"),
                    );
                    ui.add(
                        egui::Slider::new(&mut p.output_gain, 20.0..=200.0)
                            .step_by(1.0)
                            .text("modal gain"),
                    );
                    ui.add(
                        egui::Slider::new(&mut p.residual_amp, 0.0..=1.0)
                            .step_by(0.05)
                            .text("residual"),
                    );
                    if ui.button("reset").clicked() {
                        p = ModalParams::default();
                    }
                });
                if p != p_before {
                    set_modal_params(p);
                }
                ui.add_space(8.0);
                ui.separator();
            }

            // GM 128 program picker only when sf-piano is active.
            if engine == Engine::SfPiano {
                ui.collapsing("GM 128 patches", |ui| {
                    ui.horizontal(|ui| {
                        let drum_on = sf_bank == 128;
                        let label = if drum_on {
                            "Drum Kit (bank 128) [ON]"
                        } else {
                            "Drum Kit (bank 128) [off]"
                        };
                        if ui.selectable_label(drum_on, label).clicked() {
                            sf_bank = if drum_on { 0 } else { 128 };
                        }
                    });
                    egui::ScrollArea::vertical()
                        .max_height(220.0)
                        .show(ui, |ui| {
                            for family in GM_FAMILIES.iter() {
                                let default_open = *family == "Pianos";
                                egui::CollapsingHeader::new(*family)
                                    .default_open(default_open)
                                    .show(ui, |ui| {
                                        for (prog, name, fam) in GM_INSTRUMENTS.iter() {
                                            if fam != family {
                                                continue;
                                            }
                                            let selected = sf_program == *prog && sf_bank != 128;
                                            let label_text = format!("[{:>3}] {}", prog, name);
                                            let rich = if selected {
                                                egui::RichText::new(label_text)
                                                    .monospace()
                                                    .strong()
                                                    .color(egui::Color32::from_rgb(255, 220, 120))
                                            } else {
                                                egui::RichText::new(label_text).monospace()
                                            };
                                            if ui.selectable_label(selected, rich).clicked() {
                                                sf_program = *prog;
                                                sf_bank = 0;
                                            }
                                        }
                                    });
                            }
                        });
                });
                ui.add_space(8.0);
                ui.separator();
            }

            ui.heading("CC controls");
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
                            ui.add(egui::ProgressBar::new(frac).desired_width(220.0));
                            ui.monospace(format!("{count}"));
                            ui.end_row();
                        }
                    });
            }

            ui.add_space(12.0);
            ui.separator();
            ui.heading("Notes held");
            let cell_size = egui::vec2(28.0, 22.0);
            let held: std::collections::HashSet<u8> =
                dash_snapshot.active_notes.iter().map(|(_, n)| *n).collect();
            ui.horizontal_wrapped(|ui| {
                for note in 24u8..96u8 {
                    let label = format!("{note}");
                    let active = held.contains(&note);
                    let (rect, _resp) = ui.allocate_exact_size(cell_size, egui::Sense::hover());
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
        });

        // ─── Apply deferred actions ───────────────────────────────────
        if let Some(idx) = clicked_slot {
            self.apply_slot(idx);
        }
        if want_load_sfz {
            self.dialog_load_sfz();
        }
        if want_load_sf2 {
            self.dialog_load_sf2();
        }
        if want_save_custom {
            self.save_current_as_custom();
        }
        if let Some(idx) = want_remove {
            self.library.remove(idx);
            if self.pending_rename == Some(idx) {
                self.pending_rename = None;
            }
        }
        if let Some(idx) = want_rename {
            let new_label = self.rename_buf.trim().to_string();
            if !new_label.is_empty() {
                self.library.rename(idx, new_label);
            }
            self.pending_rename = None;
        }

        // ─── Write back GUI-driven changes to LiveParams ──────────────
        let gui_master_delta = master - master_before;
        let master_changed = gui_master_delta.abs() > 1e-6;
        let gui_reverb_delta = reverb_wet - reverb_before;
        let reverb_changed = gui_reverb_delta.abs() > 1e-6;
        let sf_program_changed = sf_program != sf_program_before;
        let sf_bank_changed = sf_bank != sf_bank_before;
        let mix_mode_changed = mix_mode != mix_mode_before;
        if master_changed
            || reverb_changed
            || sf_program_changed
            || sf_bank_changed
            || mix_mode_changed
        {
            let mut lp = self.ctx.live.lock().unwrap();
            if master_changed {
                lp.master = (lp.master + gui_master_delta).clamp(0.0, 10.0);
            }
            if reverb_changed {
                lp.reverb_wet = (lp.reverb_wet + gui_reverb_delta).clamp(0.0, 1.0);
            }
            if sf_program_changed {
                lp.sf_program = sf_program.min(127);
            }
            if sf_bank_changed {
                lp.sf_bank = sf_bank;
            }
            if mix_mode_changed {
                lp.mix_mode = mix_mode;
            }
        }
        // GM grid edits also need to update the loaded SF2 program live
        // (the audio callback consults lp.sf_program but the rustysynth
        // instance doesn't re-read it on its own — we have to push a
        // Program Change message into the synth too).
        if (sf_program_changed || sf_bank_changed) && engine == Engine::SfPiano {
            if let Some(synth) = self.ctx.synth.lock().unwrap().as_mut() {
                if sf_bank > 0 {
                    synth.process_midi_message(0, 0xB0, 0, sf_bank as i32);
                }
                synth.process_midi_message(0, 0xC0, sf_program as i32, 0);
            }
            if let Some((path, _, _)) = self.loaded_sf2.clone() {
                self.loaded_sf2 = Some((path, sf_program, sf_bank));
            }
        }
    }
}

struct DashSnapshot {
    cc_raw: std::collections::HashMap<u8, u8>,
    cc_count: std::collections::HashMap<u8, u64>,
    active_notes: std::collections::HashSet<(u8, u8)>,
    recent: Vec<String>,
}
