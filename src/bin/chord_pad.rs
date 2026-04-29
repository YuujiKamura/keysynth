//! Real-time chord-trigger surface for the STK guitar voice.
//!
//! Two input paths, both routing the same `NoteCmd` channel into the
//! audio thread:
//!
//! 1. **PC keyboard** — egui captures key events. The home-row keys
//!    A S D F G H J trigger the seven diatonic triads of the selected
//!    key (I ii iii IV V vi viiø). Z X C V B N M trigger the same
//!    triads one octave down. Holding spawns the chord; releasing
//!    triggers release on every voice in the chord's group.
//!
//! 2. **USB MIDI** — midir lists every input port at startup; user
//!    picks one from the dropdown. Note-on / note-off messages spawn
//!    single-note voices (one voice per pressed key, like a real
//!    polyphonic instrument).
//!
//! Voice = `voices_live/guitar_stk` cdylib loaded via
//! `keysynth::live_reload::build_and_load`, dispatched through
//! `LiveFactory::make_voice`. The whole binary builds around STK
//! because that is the only voice in the tree currently usable as a
//! standalone guitar timbre; other engines are reachable via
//! `--live-crate-root <PATH>` (or `--engine-builtin <NAME>` for the
//! in-tree `Engine` variants).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use eframe::egui;
use midir::{Ignore, MidiInput, MidiInputConnection};

use keysynth::gui_cp;
use keysynth::live_reload::{build_and_load, LiveFactory};
use keysynth::synth::{midi_to_freq, VoiceImpl};
use keysynth::voice_lib::DecayModel;
use serde::Serialize;
use serde_json::json;

/// CP snapshot published by the chord_pad GUI. Mirrors the header bar
/// state plus a count of currently-active voices so a verifier can
/// confirm note-on / note-off plumbing without listening to audio.
#[derive(Clone, Debug, Default, Serialize)]
struct CpChordPadSnapshot {
    frame_id: u64,
    key_pc: u8,
    octave: i32,
    velocity: u8,
    active_voices: usize,
    midi_active_port: Option<String>,
    midi_error: Option<String>,
    midi_port_count: usize,
    /// Group ids of every chord currently being held (PC keyboard or
    /// mouse). Lets a verifier assert "the GUI saw this chord trigger"
    /// without inspecting the audio stream.
    held_groups: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Note dispatch
// ---------------------------------------------------------------------------

/// One audio-thread command. The audio thread owns the `Vec<ActiveVoice>`;
/// the GUI / MIDI threads push commands through a single mpsc channel.
#[derive(Debug)]
enum NoteCmd {
    /// Spawn `notes.len()` voices, all tagged with `group`. Used for
    /// chord triggers (group = chord-button id) and for MIDI note-on
    /// (group = 0xff_00 | midi_note, one voice per group).
    On {
        group: u32,
        notes: Vec<u8>,
        velocity: u8,
    },
    /// Trigger release on every voice tagged with `group`.
    Off { group: u32 },
}

struct ActiveVoice {
    group: u32,
    midi_note: u8,
    voice: Box<dyn VoiceImpl + Send>,
}

const SR: u32 = 44_100;

/// Hard cap on simultaneous voices. STK voices are heavy (~2 kB state +
/// delay-line buffer); 64 is more than enough for chord-pad use cases
/// (chords + sustained pedal-style holds).
const VOICE_CAP: usize = 64;

// ---------------------------------------------------------------------------
// Diatonic chord generation
// ---------------------------------------------------------------------------

/// Major-scale degree → triad quality. Matches the diatonic functions
/// of every major key (I major, ii minor, iii minor, IV major,
/// V major, vi minor, vii diminished).
const MAJOR_TRIAD_INTERVALS: [(i8, i8, i8); 7] = [
    (0, 4, 7),    // I  major
    (2, 5, 9),    // ii minor (root D in C, intervals 2/5/9 from key root)
    (4, 7, 11),   // iii minor
    (5, 9, 12),   // IV major
    (7, 11, 14),  // V major
    (9, 12, 16),  // vi minor
    (11, 14, 17), // vii dim
];

const ROMAN: [&str; 7] = ["I", "ii", "iii", "IV", "V", "vi", "viiø"];

/// MIDI note number for the I-chord root in `key` (C..B as 0..11) at
/// octave 4 (C4 = 60).
fn key_root_midi(key_pc: u8, octave: i32) -> u8 {
    (12 * (octave + 1) + key_pc as i32).clamp(0, 127) as u8
}

fn diatonic_chord_notes(key_pc: u8, octave: i32, degree: usize) -> Vec<u8> {
    let root = key_root_midi(key_pc, octave) as i32;
    let (a, b, c) = MAJOR_TRIAD_INTERVALS[degree];
    [a, b, c]
        .iter()
        .map(|d| (root + *d as i32).clamp(0, 127) as u8)
        .collect()
}

// ---------------------------------------------------------------------------
// Audio thread
// ---------------------------------------------------------------------------

/// Wraps everything the audio callback needs. Lives behind a Mutex so
/// the GUI thread can read voice counts; lock contention is minimal
/// because the GUI only peeks once per frame.
struct AudioState {
    factory: Arc<LiveFactory>,
    voices: Vec<ActiveVoice>,
    rx: Receiver<NoteCmd>,
    /// Master gain. Defaults to 0.6 to leave headroom for ~6 stacked
    /// voices before clipping.
    master: f32,
    /// Note-off semantics for the loaded voice. Read once at boot from
    /// `voices_live/<crate>/Cargo.toml`'s `[package.metadata.keysynth-
    /// voice].decay_model`. The audio thread branches on this in the
    /// `NoteCmd::Off` arm: damper voices get `trigger_release()`, plucked
    /// voices ride out their natural decay until `is_done()` retires
    /// them. Fixed for the chord_pad lifetime because the binary loads
    /// exactly one cdylib per run.
    decay_model: DecayModel,
}

impl AudioState {
    fn drain_commands(&mut self) {
        while let Ok(cmd) = self.rx.try_recv() {
            match cmd {
                NoteCmd::On {
                    group,
                    notes,
                    velocity,
                } => {
                    for note in notes {
                        if self.voices.len() >= VOICE_CAP {
                            // Steal the oldest voice. Simple FIFO eviction;
                            // good enough for a hand-played chord pad.
                            self.voices.remove(0);
                        }
                        let freq = midi_to_freq(note);
                        let voice = self.factory.make_voice(SR as f32, freq, velocity);
                        self.voices.push(ActiveVoice {
                            group,
                            midi_note: note,
                            voice,
                        });
                    }
                }
                NoteCmd::Off { group } => match self.decay_model {
                    DecayModel::Damper => {
                        for v in self.voices.iter_mut() {
                            if v.group == group {
                                v.voice.trigger_release();
                            }
                        }
                    }
                    DecayModel::Natural => {
                        // Plucked-string voice: note-off has no
                        // physical analogue. Don't call trigger_release —
                        // let the loop filter do its work and let
                        // `render_mono`'s `retain(|v| !is_done())` cull
                        // the voice once its envelope hits silence.
                    }
                },
            }
        }
    }

    /// Render `frames` mono samples and return them. Caller fans out to
    /// L/R for stereo output.
    fn render_mono(&mut self, frames: usize) -> Vec<f32> {
        self.drain_commands();
        let mut out = vec![0.0_f32; frames];
        let mut tmp = vec![0.0_f32; frames];
        for v in self.voices.iter_mut() {
            tmp.fill(0.0);
            v.voice.render_add(&mut tmp);
            for (o, t) in out.iter_mut().zip(tmp.iter()) {
                *o += *t;
            }
        }
        // Evict voices that have rendered themselves to silence.
        self.voices.retain(|v| !v.voice.is_done());
        // Soft saturate so chord stacks don't pop on transients.
        for s in out.iter_mut() {
            *s = (*s * self.master).tanh();
        }
        out
    }
}

// ---------------------------------------------------------------------------
// MIDI input
// ---------------------------------------------------------------------------

/// MIDI ports come and go (USB hot-plug); we read the list once at
/// startup and let the user re-enumerate via the rescan button.
fn list_midi_ports() -> Vec<String> {
    let Ok(midi_in) = MidiInput::new("keysynth-chord-pad-list") else {
        return vec![];
    };
    midi_in
        .ports()
        .iter()
        .filter_map(|p| midi_in.port_name(p).ok())
        .collect()
}

fn connect_midi(port_name: &str, tx: Sender<NoteCmd>) -> Result<MidiInputConnection<()>, String> {
    let mut midi_in =
        MidiInput::new("keysynth-chord-pad").map_err(|e| format!("MidiInput::new: {e}"))?;
    midi_in.ignore(Ignore::None);
    let port = midi_in
        .ports()
        .into_iter()
        .find(|p| {
            midi_in
                .port_name(p)
                .map(|n| n == port_name)
                .unwrap_or(false)
        })
        .ok_or_else(|| format!("midi port not found: {port_name:?}"))?;
    let conn = midi_in
        .connect(
            &port,
            "keysynth-chord-pad-conn",
            move |_ts, msg, _ctx| {
                if msg.is_empty() {
                    return;
                }
                let status = msg[0] & 0xF0;
                match status {
                    0x90 if msg.len() >= 3 => {
                        let note = msg[1];
                        let vel = msg[2];
                        if vel == 0 {
                            // Running-status note-off.
                            let _ = tx.send(NoteCmd::Off {
                                group: midi_group(note),
                            });
                        } else {
                            let _ = tx.send(NoteCmd::On {
                                group: midi_group(note),
                                notes: vec![note],
                                velocity: vel,
                            });
                        }
                    }
                    0x80 if msg.len() >= 3 => {
                        let _ = tx.send(NoteCmd::Off {
                            group: midi_group(msg[1]),
                        });
                    }
                    _ => {}
                }
            },
            (),
        )
        .map_err(|e| format!("midi connect: {e}"))?;
    Ok(conn)
}

/// Distinct group id namespace for MIDI notes (high bit set so chord
/// buttons in 0..1024 never collide with MIDI 0xff_00..0xff_7f).
fn midi_group(note: u8) -> u32 {
    0xff_00 | note as u32
}

// ---------------------------------------------------------------------------
// Egui app
// ---------------------------------------------------------------------------

const KEY_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// The seven diatonic chord buttons map to A-S-D-F-G-H-J (top row of
/// chord pad). The lower row Z-X-C-V-B-N-M doubles each chord one
/// octave down so the user can jump bass voicings without changing the
/// octave selector.
const ROW_HIGH: [egui::Key; 7] = [
    egui::Key::A,
    egui::Key::S,
    egui::Key::D,
    egui::Key::F,
    egui::Key::G,
    egui::Key::H,
    egui::Key::J,
];
const ROW_LOW: [egui::Key; 7] = [
    egui::Key::Z,
    egui::Key::X,
    egui::Key::C,
    egui::Key::V,
    egui::Key::B,
    egui::Key::N,
    egui::Key::M,
];

struct ChordPadApp {
    audio: Arc<Mutex<AudioState>>,
    tx: Sender<NoteCmd>,
    /// Key picker, 0 = C ... 11 = B.
    key_pc: u8,
    /// Octave for the high-row chords. Low row is octave - 1.
    octave: i32,
    velocity: u8,
    /// Tracks which keyboard keys are currently down so we don't
    /// re-spawn voices on every egui frame.
    held: HashMap<egui::Key, u32>,
    /// Tracks which chord-button group ids are currently held by mouse
    /// (separate from `held` so we don't have to invent synthetic
    /// `egui::Key` variants for mouse-only holds).
    mouse_held: std::collections::HashSet<u32>,
    midi_ports: Vec<String>,
    midi_active_port: Option<String>,
    midi_conn: Option<MidiInputConnection<()>>,
    midi_error: Option<String>,
    _stream: cpal::Stream,
    /// CP snapshot bus. Read-only from the verifier's side: chord_pad
    /// doesn't accept commands yet (chord triggers are tied to PC
    /// keyboard / MIDI input by design), but exposing state lets a
    /// verifier confirm the GUI is responsive and observing input.
    cp_state: gui_cp::State<CpChordPadSnapshot, ()>,
    _cp_handle: Option<gui_cp::Handle>,
    cp_frame_id: u64,
}

impl ChordPadApp {
    fn new(audio: Arc<Mutex<AudioState>>, tx: Sender<NoteCmd>, stream: cpal::Stream) -> Self {
        let midi_ports = list_midi_ports();
        let cp_state: gui_cp::State<CpChordPadSnapshot, ()> = gui_cp::State::new();
        let cp_handle = match spawn_chord_pad_cp(cp_state.clone()) {
            Ok(h) => Some(h),
            Err(e) => {
                eprintln!("chord_pad: CP server failed to start: {e}");
                None
            }
        };
        Self {
            audio,
            tx,
            key_pc: 0,    // C major
            octave: 4,    // C4 root
            velocity: 96, // mf
            held: HashMap::new(),
            mouse_held: std::collections::HashSet::new(),
            midi_ports,
            midi_active_port: None,
            midi_conn: None,
            midi_error: None,
            _stream: stream,
            cp_state,
            _cp_handle: cp_handle,
            cp_frame_id: 0,
        }
    }

    fn build_cp_snapshot(&self) -> CpChordPadSnapshot {
        let active_voices = self.audio.lock().map(|a| a.voices.len()).unwrap_or(0);
        let mut held_groups: Vec<u32> = self.held.values().copied().collect();
        held_groups.extend(self.mouse_held.iter().copied());
        held_groups.sort_unstable();
        held_groups.dedup();
        CpChordPadSnapshot {
            frame_id: self.cp_frame_id,
            key_pc: self.key_pc,
            octave: self.octave,
            velocity: self.velocity,
            active_voices,
            midi_active_port: self.midi_active_port.clone(),
            midi_error: self.midi_error.clone(),
            midi_port_count: self.midi_ports.len(),
            held_groups,
        }
    }

    fn chord_group_id(degree: usize, low_row: bool) -> u32 {
        // Reserve [0..14] for the chord buttons (0..6 = high row,
        // 7..13 = low row). MIDI groups live above 0xff00 so there is
        // no collision.
        degree as u32 + if low_row { 7 } else { 0 }
    }

    fn fire_chord(&mut self, degree: usize, low_row: bool) {
        let octave = if low_row {
            self.octave - 1
        } else {
            self.octave
        };
        let notes = diatonic_chord_notes(self.key_pc, octave, degree);
        let group = Self::chord_group_id(degree, low_row);
        let _ = self.tx.send(NoteCmd::On {
            group,
            notes,
            velocity: self.velocity,
        });
    }

    fn release_chord(&mut self, degree: usize, low_row: bool) {
        let _ = self.tx.send(NoteCmd::Off {
            group: Self::chord_group_id(degree, low_row),
        });
    }

    fn handle_keyboard(&mut self, ctx: &egui::Context) {
        // Read the input snapshot once so we don't double-fire if the
        // user holds the key across multiple frames.
        let input = ctx.input(|i| i.clone());
        for (degree, &k) in ROW_HIGH.iter().enumerate() {
            let down = input.key_down(k);
            let was_held = self.held.contains_key(&k);
            if down && !was_held {
                self.fire_chord(degree, false);
                self.held.insert(k, Self::chord_group_id(degree, false));
            } else if !down && was_held {
                self.release_chord(degree, false);
                self.held.remove(&k);
            }
        }
        for (degree, &k) in ROW_LOW.iter().enumerate() {
            let down = input.key_down(k);
            let was_held = self.held.contains_key(&k);
            if down && !was_held {
                self.fire_chord(degree, true);
                self.held.insert(k, Self::chord_group_id(degree, true));
            } else if !down && was_held {
                self.release_chord(degree, true);
                self.held.remove(&k);
            }
        }
    }
}

impl eframe::App for ChordPadApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Repaint every frame so PC keyboard polling stays responsive.
        ctx.request_repaint();

        self.cp_frame_id = self.cp_frame_id.wrapping_add(1);
        let snap = self.build_cp_snapshot();
        self.cp_state.publish(snap);

        self.handle_keyboard(ctx);

        egui::TopBottomPanel::top("settings").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("key");
                egui::ComboBox::from_id_salt("key_combo")
                    .selected_text(KEY_NAMES[self.key_pc as usize])
                    .show_ui(ui, |ui| {
                        for (i, name) in KEY_NAMES.iter().enumerate() {
                            ui.selectable_value(&mut self.key_pc, i as u8, *name);
                        }
                    });
                ui.label("octave");
                ui.add(egui::DragValue::new(&mut self.octave).range(2..=6));
                ui.label("velocity");
                ui.add(egui::DragValue::new(&mut self.velocity).range(1..=127));
                ui.separator();
                if let Ok(audio) = self.audio.lock() {
                    ui.label(format!("voices: {}", audio.voices.len()));
                }
            });
            ui.horizontal(|ui| {
                ui.label("MIDI in:");
                let label = self
                    .midi_active_port
                    .clone()
                    .unwrap_or_else(|| "(none)".to_string());
                egui::ComboBox::from_id_salt("midi_combo")
                    .selected_text(label)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_label(self.midi_active_port.is_none(), "(none)")
                            .clicked()
                        {
                            self.midi_active_port = None;
                            self.midi_conn = None;
                            self.midi_error = None;
                        }
                        for port in &self.midi_ports.clone() {
                            if ui
                                .selectable_label(
                                    self.midi_active_port.as_deref() == Some(port),
                                    port,
                                )
                                .clicked()
                            {
                                match connect_midi(port, self.tx.clone()) {
                                    Ok(conn) => {
                                        self.midi_conn = Some(conn);
                                        self.midi_active_port = Some(port.clone());
                                        self.midi_error = None;
                                    }
                                    Err(e) => {
                                        self.midi_error = Some(e);
                                    }
                                }
                            }
                        }
                    });
                if ui.button("rescan").clicked() {
                    self.midi_ports = list_midi_ports();
                }
                if let Some(err) = &self.midi_error {
                    ui.colored_label(egui::Color32::LIGHT_RED, err);
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(format!(
                "Chord pad — STK guitar in key of {}",
                KEY_NAMES[self.key_pc as usize]
            ));
            ui.label(
                "Hold A S D F G H J for the seven diatonic triads in the \
                 selected major key (I ii iii IV V vi viiø). Z X C V B N M \
                 plays the same triads one octave down. Click and hold a \
                 button to play with the mouse. Plug a USB MIDI keyboard \
                 and pick the port above for note-by-note play.",
            );
            ui.separator();
            // Pre-compute button labels so the borrow on `self` (for
            // key_pc / octave) ends before we iterate and call
            // fire_chord / release_chord (mutable borrow).
            let mut button_specs: Vec<(usize, bool, String)> = Vec::with_capacity(14);
            for (i, k) in ROW_HIGH.iter().enumerate() {
                let notes = diatonic_chord_notes(self.key_pc, self.octave, i);
                let label = chord_button_label(ROMAN[i], &notes, &format!("{k:?}"));
                button_specs.push((i, false, label));
            }
            for (i, k) in ROW_LOW.iter().enumerate() {
                let notes = diatonic_chord_notes(self.key_pc, self.octave - 1, i);
                let label = chord_button_label(ROMAN[i], &notes, &format!("{k:?}"));
                button_specs.push((i, true, label));
            }

            let mut row_iter = button_specs.chunks(7);
            for low_row in [false, true] {
                let row = row_iter.next().expect("two rows");
                ui.horizontal(|ui| {
                    for (degree, _is_low, label) in row {
                        let group = ChordPadApp::chord_group_id(*degree, low_row);
                        let resp = ui
                            .add_sized(egui::vec2(110.0, 90.0), egui::Button::new(label.as_str()));
                        let pressed = resp.is_pointer_button_down_on();
                        let was_held = self.mouse_held.contains(&group);
                        if pressed && !was_held {
                            self.fire_chord(*degree, low_row);
                            self.mouse_held.insert(group);
                        } else if !pressed && was_held {
                            self.release_chord(*degree, low_row);
                            self.mouse_held.remove(&group);
                        }
                    }
                });
                if !low_row {
                    ui.add_space(8.0);
                }
            }
        });
    }
}

fn chord_button_label(roman: &str, notes: &[u8], kbd: &str) -> String {
    let mut name = String::new();
    for (j, n) in notes.iter().enumerate() {
        if j > 0 {
            name.push(' ');
        }
        name.push_str(&format!(
            "{}{}",
            KEY_NAMES[(*n as usize) % 12],
            *n as i32 / 12 - 1
        ));
    }
    format!("{name}\n{roman}\n[{kbd}]")
}

// ---------------------------------------------------------------------------
// CLI + boot
// ---------------------------------------------------------------------------

struct CliArgs {
    live_crate_root: PathBuf,
}

fn parse_args() -> CliArgs {
    let mut live_crate_root = PathBuf::from("voices_live/guitar_stk");
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--live-crate-root" => {
                if let Some(v) = iter.next() {
                    live_crate_root = PathBuf::from(v);
                }
            }
            "--help" | "-h" => {
                eprintln!(
                    "chord_pad — real-time chord-trigger surface\n\n\
                     usage:\n  \
                     chord_pad [--live-crate-root PATH]\n\n\
                     options:\n  \
                     --live-crate-root PATH   voices_live/<crate> to load \
                     (default: voices_live/guitar_stk)\n  \
                     --help                   this help"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("chord_pad: unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }
    CliArgs { live_crate_root }
}

/// Read `<crate_root>/Cargo.toml` and pull the
/// `[package.metadata.keysynth-voice].decay_model` field. Falls back to
/// `DecayModel::default()` (= `Damper`) on any error or missing key —
/// same forgiving policy as `voice_lib::discover_plugin_voices`, and
/// the right behaviour for the legacy `voices_live/piano` plugin which
/// doesn't ship the field. We deliberately keep this lookup minimal
/// (no full manifest parse, no shared schema) so chord_pad stays a
/// single-binary tool that doesn't depend on the GUI catalog code.
fn load_decay_model(crate_root: &Path) -> DecayModel {
    let manifest = crate_root.join("Cargo.toml");
    let Ok(text) = std::fs::read_to_string(&manifest) else {
        eprintln!(
            "chord_pad: cannot read {} — defaulting to decay_model=Damper",
            manifest.display(),
        );
        return DecayModel::default();
    };
    let parsed: toml::Value = match toml::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            eprintln!(
                "chord_pad: {} parse error ({e}) — defaulting to decay_model=Damper",
                manifest.display(),
            );
            return DecayModel::default();
        }
    };
    let raw = parsed
        .get("package")
        .and_then(|p| p.get("metadata"))
        .and_then(|m| m.get("keysynth-voice"))
        .and_then(|kv| kv.get("decay_model"))
        .and_then(|v| v.as_str());
    match raw {
        Some("damper") => DecayModel::Damper,
        Some("natural") => DecayModel::Natural,
        Some(other) => {
            eprintln!(
                "chord_pad: {} decay_model='{other}' unknown — defaulting to Damper",
                manifest.display(),
            );
            DecayModel::default()
        }
        None => DecayModel::default(),
    }
}

/// Spawn the chord_pad CP server. Read-only for now: exposes
/// `get_state` (returns `CpChordPadSnapshot`) plus the auto-registered
/// `ping`. Driving chord triggers over CP isn't part of this stage —
/// the binary's whole point is humans playing chords through PC
/// keyboard / MIDI in real time.
fn spawn_chord_pad_cp(
    state: gui_cp::State<CpChordPadSnapshot, ()>,
) -> std::io::Result<gui_cp::Handle> {
    let endpoint = gui_cp::resolve_endpoint("chord_pad", None);
    let st_get = state;
    gui_cp::Builder::new("chord_pad", &endpoint)
        .register("get_state", move |_p| match st_get.read_snapshot() {
            Some(s) => gui_cp::encode_result(s),
            None => Ok(json!({"frame_id": 0, "warming_up": true})),
        })
        .serve()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();

    // Read note-off semantics from the crate manifest BEFORE the cargo
    // build so the user sees `decay_model=natural` in the boot log even
    // if the build itself fails.
    let decay_model = load_decay_model(&args.live_crate_root);
    eprintln!(
        "chord_pad: voice decay_model = {:?} (from {}/Cargo.toml)",
        decay_model,
        args.live_crate_root.display(),
    );

    // Build + load the live voice cdylib.
    eprintln!(
        "chord_pad: building live voice from {} (cargo build, may take a moment)",
        args.live_crate_root.display(),
    );
    let factory = Arc::new(
        build_and_load(&args.live_crate_root)
            .map_err(|e| format!("build_and_load({}): {e}", args.live_crate_root.display()))?,
    );
    eprintln!(
        "chord_pad: live voice loaded from {}",
        factory.dll_path.display(),
    );

    // Audio output setup.
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or("no default output device")?;
    let supported = device.default_output_config()?;
    let sample_format = supported.sample_format();
    let channels = supported.channels() as usize;
    let stream_cfg: StreamConfig = supported.into();
    eprintln!(
        "chord_pad: audio out = {} ({} ch @ {} Hz, {:?})",
        device.name().unwrap_or_else(|_| "<unknown>".into()),
        channels,
        stream_cfg.sample_rate.0,
        sample_format,
    );

    let (tx, rx) = channel::<NoteCmd>();
    let audio = Arc::new(Mutex::new(AudioState {
        factory,
        voices: Vec::with_capacity(VOICE_CAP),
        rx,
        master: 0.6,
        decay_model,
    }));

    // Build the cpal stream. We support f32 and i16 output formats —
    // these are what every consumer audio device on Windows/macOS/Linux
    // exposes through the default-output config.
    let audio_for_stream = audio.clone();
    let err_fn = |e| eprintln!("chord_pad: cpal stream error: {e}");
    let stream = match sample_format {
        SampleFormat::F32 => device.build_output_stream(
            &stream_cfg,
            move |data: &mut [f32], _| {
                let frames = data.len() / channels;
                let mono = audio_for_stream.lock().unwrap().render_mono(frames);
                for (i, frame) in data.chunks_mut(channels).enumerate() {
                    let s = mono[i];
                    for ch in frame.iter_mut() {
                        *ch = s;
                    }
                }
            },
            err_fn,
            None,
        )?,
        SampleFormat::I16 => device.build_output_stream(
            &stream_cfg,
            move |data: &mut [i16], _| {
                let frames = data.len() / channels;
                let mono = audio_for_stream.lock().unwrap().render_mono(frames);
                for (i, frame) in data.chunks_mut(channels).enumerate() {
                    let s = (mono[i].clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                    for ch in frame.iter_mut() {
                        *ch = s;
                    }
                }
            },
            err_fn,
            None,
        )?,
        other => {
            return Err(format!("unsupported sample format: {other:?}").into());
        }
    };
    stream.play()?;

    // Egui app.
    let app = ChordPadApp::new(audio, tx, stream);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([960.0, 360.0])
            .with_title("keysynth chord pad — STK guitar"),
        ..Default::default()
    };
    eprintln!("chord_pad: opening GUI window (eframe::run_native)");
    eframe::run_native(
        "keysynth chord pad",
        options,
        Box::new(|cc| {
            keysynth::ui::setup_japanese_fonts(&cc.egui_ctx);
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| -> Box<dyn std::error::Error> { format!("eframe: {e}").into() })?;
    eprintln!("chord_pad: GUI window closed, exiting");
    Ok(())
}
