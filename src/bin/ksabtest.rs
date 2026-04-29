//! Stage F: blind A/B harness GUI (issue #67).
//!
//! Pick two voices, pick a MIDI piece, the harness renders both through
//! `render_midi` (cached via `preview_cache` so a re-pick is instant on
//! the second pass), randomly assigns each voice to physical Slot A /
//! Slot B, and lets the listener click between them without seeing which
//! voice they're auditioning. After the listener picks "Slot A is more
//! piano-like" / "Slot B" / "indistinguishable", the assignment is
//! revealed and the verdict is logged to `bench-out/ab_test.db`.
//!
//! Cumulative stats per canonical voice pair drive the right-hand panel:
//! across all trials for, say, (piano-modal, piano-thick), how often did
//! each voice get picked as the more realistic one? When two upright
//! voices tie at ~50% across N >> 1 trials they have hit the
//! "A/B-blind-indistinguishable" bar of issue #3 piano milestone #3.
//!
//! Architectural note: this binary is intentionally independent of
//! `jukebox`. Jukebox is a multi-track mixer with VU meters and a play
//! log; layering blind A/B logic on top of its slot machinery would
//! tangle two unrelated state spaces. The harness reuses the
//! `preview_cache` render driver and `keysynth::ab_test` data layer, so
//! the duplication is contained to GUI plumbing only.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use eframe::egui;
use hound::{SampleFormat as WavSampleFormat, WavReader};

use keysynth::ab_test::{aggregate, canonical_pair, AbTestDb, PairStats, Trial, Verdict};
use keysynth::preview_cache::{Cache, CacheKey, RenderParams};

// ---------------------------------------------------------------------------
// Voices that `render_midi` can drive without extra CLI arguments.
//
// Deliberately omits:
//   * `sfz-piano` — needs `--sfz <path>` so it isn't a one-click voice.
//   * `live` / `guitar` / `guitar-stk` — those compile a `voices_live/*`
//     cdylib at render time, which makes the first render minutes long.
//     Safe to add later once we want to A/B against them; for piano-#3
//     evaluation the modeled-piano family is the focus.
// ---------------------------------------------------------------------------
const ENGINES: &[&str] = &[
    "piano-modal",
    "piano-thick",
    "piano-lite",
    "piano",
    "piano-5am",
    "ks-rich",
    "ks",
    "fm",
    "sub",
    "square",
    "koto",
];

// ---------------------------------------------------------------------------
// Tiny single-track mixer. Both rendered WAVs are decoded into RAM up
// front; the audio callback reads from whichever `active` slot the GUI
// thread last set. Switching between A and B = swap the Arc and reset
// the cursor — no file I/O on the audio thread.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct TrackBuf {
    /// Interleaved stereo f32. Mono input is duplicated to L=R at load.
    samples: Vec<f32>,
    /// Native sample rate of the rendered file (44100 for `render_midi`).
    /// Played back at the cpal device rate; mismatch causes a tiny pitch
    /// shift but it affects A and B identically so the blind comparison
    /// stays fair. Stored only for the diagnostic readout.
    sample_rate: u32,
}

#[derive(Default)]
struct Mixer {
    active: Option<Arc<TrackBuf>>,
    /// Frame cursor (one frame = L+R sample pair).
    cursor_frames: usize,
    playing: bool,
}

fn load_wav_to_buf(path: &Path) -> Result<TrackBuf, String> {
    let mut reader = WavReader::open(path).map_err(|e| format!("hound open {}: {e}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels;
    let bits = spec.bits_per_sample;
    let mut interleaved: Vec<f32> = Vec::new();
    match spec.sample_format {
        WavSampleFormat::Int => {
            let max = ((1u32 << (bits as u32 - 1)) - 1) as f32;
            for s in reader.samples::<i32>() {
                let v = s.map_err(|e| format!("hound sample read: {e}"))?;
                interleaved.push(v as f32 / max);
            }
        }
        WavSampleFormat::Float => {
            for s in reader.samples::<f32>() {
                let v = s.map_err(|e| format!("hound sample read: {e}"))?;
                interleaved.push(v);
            }
        }
    }
    // Normalise to interleaved stereo regardless of source channel count.
    let stereo: Vec<f32> = match channels {
        1 => {
            let mut out = Vec::with_capacity(interleaved.len() * 2);
            for s in interleaved {
                out.push(s);
                out.push(s);
            }
            out
        }
        2 => interleaved,
        n => {
            // Multichannel: take channels 0/1, drop the rest.
            let frames = interleaved.len() / n as usize;
            let mut out = Vec::with_capacity(frames * 2);
            for f in 0..frames {
                let base = f * n as usize;
                out.push(interleaved[base]);
                out.push(interleaved.get(base + 1).copied().unwrap_or(interleaved[base]));
            }
            out
        }
    };
    Ok(TrackBuf {
        samples: stereo,
        sample_rate: spec.sample_rate,
    })
}

fn fill_callback(out: &mut [f32], device_channels: u16, state: &Arc<Mutex<Mixer>>) {
    for s in out.iter_mut() {
        *s = 0.0;
    }
    let mut st = match state.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    if !st.playing {
        return;
    }
    let active = match st.active.clone() {
        Some(a) => a,
        None => return,
    };
    let total_frames = active.samples.len() / 2;
    let frames_out = out.len() / device_channels.max(1) as usize;
    for f in 0..frames_out {
        if st.cursor_frames >= total_frames {
            st.playing = false;
            break;
        }
        let i = st.cursor_frames * 2;
        let l = active.samples[i];
        let r = active.samples[i + 1];
        let base = f * device_channels as usize;
        match device_channels {
            1 => out[base] = (l + r) * 0.5,
            2 => {
                out[base] = l;
                out[base + 1] = r;
            }
            n => {
                out[base] = l;
                out[base + 1] = r;
                for c in 2..n as usize {
                    out[base + c] = 0.0;
                }
            }
        }
        st.cursor_frames += 1;
    }
}

struct AudioOut {
    state: Arc<Mutex<Mixer>>,
    _stream: cpal::Stream,
    sample_rate: u32,
    device_name: String,
}

fn build_audio() -> Result<AudioOut, String> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| "no default output device".to_string())?;
    let device_name = device.name().unwrap_or_else(|_| "(default)".to_string());
    let supported = device
        .default_output_config()
        .map_err(|e| format!("default_output_config: {e}"))?;
    let sample_format = supported.sample_format();
    let sample_rate = supported.sample_rate().0;
    let channels = supported.channels();
    let cfg: StreamConfig = supported.into();
    let state = Arc::new(Mutex::new(Mixer::default()));
    let err_fn = |err| eprintln!("ksabtest audio stream error: {err}");
    let stream = match sample_format {
        SampleFormat::F32 => {
            let st = state.clone();
            device
                .build_output_stream(
                    &cfg,
                    move |out: &mut [f32], _| fill_callback(out, channels, &st),
                    err_fn,
                    None,
                )
                .map_err(|e| format!("build f32 stream: {e}"))?
        }
        SampleFormat::I16 => {
            let st = state.clone();
            let mut scratch: Vec<f32> = Vec::new();
            device
                .build_output_stream(
                    &cfg,
                    move |out: &mut [i16], _| {
                        if scratch.len() != out.len() {
                            scratch.resize(out.len(), 0.0);
                        }
                        for s in scratch.iter_mut() {
                            *s = 0.0;
                        }
                        fill_callback(&mut scratch, channels, &st);
                        for (dst, src) in out.iter_mut().zip(scratch.iter()) {
                            let c = src.clamp(-1.0, 1.0);
                            *dst = (c * i16::MAX as f32) as i16;
                        }
                    },
                    err_fn,
                    None,
                )
                .map_err(|e| format!("build i16 stream: {e}"))?
        }
        SampleFormat::U16 => {
            let st = state.clone();
            let mut scratch: Vec<f32> = Vec::new();
            device
                .build_output_stream(
                    &cfg,
                    move |out: &mut [u16], _| {
                        if scratch.len() != out.len() {
                            scratch.resize(out.len(), 0.0);
                        }
                        for s in scratch.iter_mut() {
                            *s = 0.0;
                        }
                        fill_callback(&mut scratch, channels, &st);
                        for (dst, src) in out.iter_mut().zip(scratch.iter()) {
                            let c = src.clamp(-1.0, 1.0);
                            let unsigned = ((c + 1.0) * 0.5 * u16::MAX as f32) as u16;
                            *dst = unsigned;
                        }
                    },
                    err_fn,
                    None,
                )
                .map_err(|e| format!("build u16 stream: {e}"))?
        }
        other => return Err(format!("unsupported sample format: {other:?}")),
    };
    stream.play().map_err(|e| format!("stream.play: {e}"))?;
    Ok(AudioOut {
        state,
        _stream: stream,
        sample_rate,
        device_name,
    })
}

// ---------------------------------------------------------------------------
// MIDI scanning + render path
// ---------------------------------------------------------------------------

fn scan_midi(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let read = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return out,
    };
    for ent in read.flatten() {
        let p = ent.path();
        match p.extension().and_then(|s| s.to_str()).map(|s| s.to_ascii_lowercase()) {
            Some(ext) if ext == "mid" || ext == "midi" => out.push(p),
            _ => {}
        }
    }
    out.sort();
    out
}

fn render_midi_binary_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let candidates = if cfg!(windows) {
        vec!["render_midi.exe"]
    } else {
        vec!["render_midi"]
    };
    candidates
        .into_iter()
        .map(|name| dir.join(name))
        .find(|p| p.is_file())
}

fn open_preview_cache() -> Result<Cache, String> {
    let cache_dir = PathBuf::from("bench-out/cache");
    Cache::new(&cache_dir, 1_073_741_824).map_err(|e| format!("Cache::new: {e}"))
}

fn render_one(midi: &Path, voice: &str, render_bin: &Path) -> Result<PathBuf, String> {
    let cache = open_preview_cache()?;
    let key = CacheKey {
        song_path: midi.to_path_buf(),
        voice_id: voice.to_string(),
        voice_dll: None,
        render_params: RenderParams::default(),
    };
    keysynth::preview_cache::render_to_cache(&cache, &key, voice, render_bin)
        .map_err(|e| format!("render_to_cache({voice}): {e}"))
}

/// Cheap single-bit blind coin from system time. Good enough — only one
/// bit of entropy is needed per trial and the blinding is broken on
/// reveal anyway. Avoids dragging in a `rand` dep for one flip.
fn coin_flip_a_first() -> bool {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    (nanos & 1) == 0
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

enum Phase {
    Idle,
    Rendering {
        voice_a: String,
        voice_b: String,
        midi: PathBuf,
        rx: std::sync::mpsc::Receiver<RenderResult>,
        started: std::time::Instant,
    },
    /// Both renders ready; user is auditioning before submitting verdict.
    /// `voice_in_slot_a` is the voice whose render is in `slot_a_buf`
    /// — the user does *not* see this until reveal.
    Listening {
        voice_a_canonical: String,
        voice_b_canonical: String,
        midi: PathBuf,
        slot_a_buf: Arc<TrackBuf>,
        slot_b_buf: Arc<TrackBuf>,
        voice_in_slot_a: String,
    },
    /// Verdict submitted; show the reveal until the user starts a new
    /// trial.
    Reveal {
        voice_a_canonical: String,
        voice_b_canonical: String,
        voice_in_slot_a: String,
        verdict: Verdict,
        midi: PathBuf,
        slot_a_buf: Arc<TrackBuf>,
        slot_b_buf: Arc<TrackBuf>,
    },
}

struct RenderResult {
    /// (`voice_a_input`, wav for voice_a)
    a: Result<(String, Arc<TrackBuf>), String>,
    /// (`voice_b_input`, wav for voice_b)
    b: Result<(String, Arc<TrackBuf>), String>,
}

struct App {
    audio: AudioOut,
    db: Option<AbTestDb>,
    db_path: PathBuf,

    /// Inputs the user picks before triggering a render.
    voice_a_input: String,
    voice_b_input: String,
    midi_dir: PathBuf,
    midi_files: Vec<PathBuf>,
    selected_midi: Option<usize>,

    render_bin: Option<PathBuf>,
    render_bin_warning: Option<String>,

    phase: Phase,
    last_error: Option<String>,

    /// Cached aggregation for the (voice_a, voice_b) pair currently
    /// selected in the inputs. Refreshed whenever the inputs change or
    /// a new trial is logged.
    pair_trials: Vec<Trial>,
    pair_stats: PairStats,
}

impl App {
    fn new() -> Result<Self, String> {
        let audio = build_audio().map_err(|e| format!("audio init: {e}"))?;
        let db_path = PathBuf::from("bench-out/ab_test.db");
        let db = match AbTestDb::open(&db_path) {
            Ok(mut db) => match db.migrate() {
                Ok(()) => Some(db),
                Err(e) => {
                    eprintln!("ksabtest: db migrate failed: {e} — running without persistence");
                    None
                }
            },
            Err(e) => {
                eprintln!(
                    "ksabtest: db open ({}) failed: {e} — running without persistence",
                    db_path.display()
                );
                None
            }
        };
        let midi_dir = PathBuf::from("bench-out/songs");
        let midi_files = scan_midi(&midi_dir);
        let render_bin = render_midi_binary_path();
        let render_bin_warning = if render_bin.is_none() {
            Some(format!(
                "render_midi binary not found alongside ksabtest — \
                 run `cargo build --bin render_midi --release` (or debug to match this build) \
                 and relaunch."
            ))
        } else {
            None
        };
        let mut app = App {
            audio,
            db,
            db_path,
            voice_a_input: "piano-modal".to_string(),
            voice_b_input: "piano-thick".to_string(),
            midi_dir,
            midi_files,
            selected_midi: None,
            render_bin,
            render_bin_warning,
            phase: Phase::Idle,
            last_error: None,
            pair_trials: Vec::new(),
            pair_stats: PairStats::default(),
        };
        if !app.midi_files.is_empty() {
            app.selected_midi = Some(0);
        }
        app.refresh_stats();
        Ok(app)
    }

    fn refresh_stats(&mut self) {
        let (canonical_a, canonical_b) =
            canonical_pair(&self.voice_a_input, &self.voice_b_input);
        self.pair_trials = match self.db.as_ref() {
            Some(db) => db
                .trials_for_pair(&canonical_a, &canonical_b, 200)
                .unwrap_or_else(|e| {
                    eprintln!("ksabtest: trials_for_pair: {e}");
                    Vec::new()
                }),
            None => Vec::new(),
        };
        self.pair_stats = aggregate(&self.pair_trials, &canonical_a);
    }

    fn stop_audio(&self) {
        if let Ok(mut st) = self.audio.state.lock() {
            st.playing = false;
            st.active = None;
            st.cursor_frames = 0;
        }
    }

    fn play_buf(&self, buf: &Arc<TrackBuf>) {
        if let Ok(mut st) = self.audio.state.lock() {
            st.active = Some(buf.clone());
            st.cursor_frames = 0;
            st.playing = true;
        }
    }

    fn start_render(&mut self) {
        let voice_a = self.voice_a_input.clone();
        let voice_b = self.voice_b_input.clone();
        if voice_a == voice_b {
            self.last_error = Some("voice_a and voice_b are the same — pick two different voices".to_string());
            return;
        }
        let midi = match self.selected_midi.and_then(|i| self.midi_files.get(i)).cloned() {
            Some(p) => p,
            None => {
                self.last_error = Some("no MIDI selected".to_string());
                return;
            }
        };
        let render_bin = match self.render_bin.clone() {
            Some(p) => p,
            None => {
                self.last_error = Some("render_midi binary missing — see warning above".to_string());
                return;
            }
        };
        self.last_error = None;
        self.stop_audio();

        let (tx, rx) = std::sync::mpsc::channel::<RenderResult>();
        let started = std::time::Instant::now();
        let voice_a_thread = voice_a.clone();
        let voice_b_thread = voice_b.clone();
        let midi_thread = midi.clone();
        std::thread::spawn(move || {
            let render_voice = |v: &str| -> Result<(String, Arc<TrackBuf>), String> {
                let wav = render_one(&midi_thread, v, &render_bin)?;
                let buf = load_wav_to_buf(&wav)?;
                Ok((v.to_string(), Arc::new(buf)))
            };
            let a = render_voice(&voice_a_thread);
            let b = render_voice(&voice_b_thread);
            let _ = tx.send(RenderResult { a, b });
        });

        self.phase = Phase::Rendering {
            voice_a,
            voice_b,
            midi,
            rx,
            started,
        };
    }

    fn poll_render(&mut self) {
        // Two-step ownership dance: first poll the receiver while we
        // hold a shared borrow of `self.phase`, *then* (only on a
        // successful recv) take ownership of the rendering state via
        // `mem::replace` so we can mutate `self.phase` freely.
        let recv = match &self.phase {
            Phase::Rendering { rx, .. } => rx.try_recv(),
            _ => return,
        };
        let result = match recv {
            Ok(r) => r,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.last_error = Some("render thread died before sending result".to_string());
                self.phase = Phase::Idle;
                return;
            }
        };
        let (voice_a, voice_b, midi) =
            match std::mem::replace(&mut self.phase, Phase::Idle) {
                Phase::Rendering {
                    voice_a,
                    voice_b,
                    midi,
                    ..
                } => (voice_a, voice_b, midi),
                other => {
                    // Shouldn't happen — phase changed between the recv
                    // and the take. Restore and bail.
                    self.phase = other;
                    return;
                }
            };
        let (va_id, va_buf) = match result.a {
            Ok(v) => v,
            Err(e) => {
                self.last_error = Some(format!("voice {voice_a} render failed: {e}"));
                self.phase = Phase::Idle;
                return;
            }
        };
        let (vb_id, vb_buf) = match result.b {
            Ok(v) => v,
            Err(e) => {
                self.last_error = Some(format!("voice {voice_b} render failed: {e}"));
                self.phase = Phase::Idle;
                return;
            }
        };
        // Canonicalise the pair so the slot assignment + DB row use the
        // same naming convention as the stats query.
        let (canonical_a, canonical_b) = canonical_pair(&va_id, &vb_id);
        let a_first = coin_flip_a_first();
        // `a_first` controls whether canonical_a goes into Slot A.
        let (slot_a_buf, slot_b_buf, voice_in_slot_a) = if a_first {
            // Decide which buffer matches canonical_a:
            let slot_a = if va_id == canonical_a { va_buf.clone() } else { vb_buf.clone() };
            let slot_b = if vb_id == canonical_b { vb_buf.clone() } else { va_buf.clone() };
            (slot_a, slot_b, canonical_a.clone())
        } else {
            let slot_a = if vb_id == canonical_b { vb_buf.clone() } else { va_buf.clone() };
            let slot_b = if va_id == canonical_a { va_buf.clone() } else { vb_buf.clone() };
            (slot_a, slot_b, canonical_b.clone())
        };
        self.phase = Phase::Listening {
            voice_a_canonical: canonical_a,
            voice_b_canonical: canonical_b,
            midi,
            slot_a_buf,
            slot_b_buf,
            voice_in_slot_a,
        };
    }

    fn submit_verdict(&mut self, verdict: Verdict) {
        let snapshot = match &self.phase {
            Phase::Listening {
                voice_a_canonical,
                voice_b_canonical,
                midi,
                slot_a_buf,
                slot_b_buf,
                voice_in_slot_a,
            } => (
                voice_a_canonical.clone(),
                voice_b_canonical.clone(),
                midi.clone(),
                slot_a_buf.clone(),
                slot_b_buf.clone(),
                voice_in_slot_a.clone(),
            ),
            _ => return,
        };
        let (canonical_a, canonical_b, midi, slot_a_buf, slot_b_buf, voice_in_slot_a) = snapshot;
        if let Some(db) = self.db.as_mut() {
            if let Err(e) = db.record_trial(
                &canonical_a,
                &canonical_b,
                &voice_in_slot_a,
                &midi.to_string_lossy(),
                verdict,
            ) {
                self.last_error = Some(format!("db record_trial: {e}"));
            }
        }
        self.stop_audio();
        self.phase = Phase::Reveal {
            voice_a_canonical: canonical_a,
            voice_b_canonical: canonical_b,
            voice_in_slot_a,
            verdict,
            midi,
            slot_a_buf,
            slot_b_buf,
        };
        self.refresh_stats();
    }

    fn next_trial_same_voices(&mut self) {
        // Re-randomise the slot assignment using the already-rendered
        // buffers so consecutive trials on the same voice pair / MIDI
        // don't cost another render.
        let snapshot = match &self.phase {
            Phase::Reveal {
                voice_a_canonical,
                voice_b_canonical,
                voice_in_slot_a,
                midi,
                slot_a_buf,
                slot_b_buf,
                ..
            } => (
                voice_a_canonical.clone(),
                voice_b_canonical.clone(),
                voice_in_slot_a.clone(),
                midi.clone(),
                slot_a_buf.clone(),
                slot_b_buf.clone(),
            ),
            _ => return,
        };
        let (canonical_a, canonical_b, prev_voice_in_slot_a, midi, prev_slot_a_buf, prev_slot_b_buf) =
            snapshot;
        // Map the previous slot buffers back to canonical voices.
        let (canonical_a_buf, canonical_b_buf) = if prev_voice_in_slot_a == canonical_a {
            (prev_slot_a_buf, prev_slot_b_buf)
        } else {
            (prev_slot_b_buf, prev_slot_a_buf)
        };
        let a_first = coin_flip_a_first();
        let (slot_a_buf, slot_b_buf, voice_in_slot_a) = if a_first {
            (canonical_a_buf, canonical_b_buf, canonical_a.clone())
        } else {
            (canonical_b_buf, canonical_a_buf, canonical_b.clone())
        };
        self.stop_audio();
        self.phase = Phase::Listening {
            voice_a_canonical: canonical_a,
            voice_b_canonical: canonical_b,
            midi,
            slot_a_buf,
            slot_b_buf,
            voice_in_slot_a,
        };
    }

    fn reset_to_idle(&mut self) {
        self.stop_audio();
        self.phase = Phase::Idle;
    }
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drive the rendering poll every frame so the UI flips into
        // Listening as soon as the worker thread sends.
        self.poll_render();

        // While rendering, request a repaint so the elapsed time and
        // poll fire even without user input.
        if matches!(self.phase, Phase::Rendering { .. }) {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        let mut want_render = false;
        let mut want_play_a = false;
        let mut want_play_b = false;
        let mut want_stop = false;
        let mut want_verdict: Option<Verdict> = None;
        let mut want_next_same = false;
        let mut want_reset = false;
        let mut input_changed = false;

        egui::TopBottomPanel::top("hdr").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("ksabtest — blind A/B harness");
                ui.separator();
                ui.label(format!(
                    "audio: {} @ {} Hz",
                    self.audio.device_name, self.audio.sample_rate
                ));
                ui.separator();
                ui.label(match &self.db {
                    Some(_) => format!("db: {}", self.db_path.display()),
                    None => "db: (disabled — see stderr)".to_string(),
                });
            });
            if let Some(w) = &self.render_bin_warning {
                ui.colored_label(egui::Color32::RED, w);
            }
            if let Some(e) = &self.last_error {
                ui.colored_label(egui::Color32::RED, e);
            }
        });

        egui::SidePanel::right("stats").min_width(260.0).show(ctx, |ui| {
            ui.heading("cumulative stats");
            let (canonical_a, canonical_b) =
                canonical_pair(&self.voice_a_input, &self.voice_b_input);
            ui.label(format!("pair: {canonical_a} vs {canonical_b}"));
            let s = self.pair_stats;
            ui.label(format!(
                "{}: {} wins  ·  {}: {} wins  ·  ties: {}  (n={})",
                canonical_a,
                s.a_wins,
                canonical_b,
                s.b_wins,
                s.ties,
                s.total()
            ));
            let share = s.a_share_decisive();
            ui.add(
                egui::ProgressBar::new(share)
                    .text(format!("{} share (decisive): {:.1}%", canonical_a, share * 100.0)),
            );
            ui.separator();
            ui.label("Note: 50% across many trials = blind-indistinguishable (issue #3 piano #3).");

            ui.separator();
            ui.label("recent trials:");
            egui::ScrollArea::vertical().show(ui, |ui| {
                for t in self.pair_trials.iter().take(20) {
                    let voice_in_slot_b = if t.voice_in_slot_a == t.voice_a {
                        &t.voice_b
                    } else {
                        &t.voice_a
                    };
                    let winner_voice = match t.verdict {
                        Verdict::SlotA => t.voice_in_slot_a.as_str(),
                        Verdict::SlotB => voice_in_slot_b.as_str(),
                        Verdict::Tie => "tie",
                    };
                    let midi_short = Path::new(&t.midi_path)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("?");
                    ui.label(format!("· {midi_short}  →  {winner_voice}"));
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // ---- Inputs ------------------------------------------------
            ui.heading("inputs");
            ui.horizontal(|ui| {
                ui.label("voice A:");
                let resp = egui::ComboBox::from_id_salt("vA")
                    .selected_text(&self.voice_a_input)
                    .show_ui(ui, |ui| {
                        let mut changed = false;
                        for v in ENGINES {
                            if ui
                                .selectable_value(&mut self.voice_a_input, (*v).to_string(), *v)
                                .changed()
                            {
                                changed = true;
                            }
                        }
                        changed
                    });
                if resp.inner.unwrap_or(false) {
                    input_changed = true;
                }
            });
            ui.horizontal(|ui| {
                ui.label("voice B:");
                let resp = egui::ComboBox::from_id_salt("vB")
                    .selected_text(&self.voice_b_input)
                    .show_ui(ui, |ui| {
                        let mut changed = false;
                        for v in ENGINES {
                            if ui
                                .selectable_value(&mut self.voice_b_input, (*v).to_string(), *v)
                                .changed()
                            {
                                changed = true;
                            }
                        }
                        changed
                    });
                if resp.inner.unwrap_or(false) {
                    input_changed = true;
                }
            });
            ui.horizontal(|ui| {
                ui.label(format!("MIDI dir: {}", self.midi_dir.display()));
                if ui.button("rescan").clicked() {
                    self.midi_files = scan_midi(&self.midi_dir);
                    if self.selected_midi.unwrap_or(0) >= self.midi_files.len() {
                        self.selected_midi = if self.midi_files.is_empty() { None } else { Some(0) };
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("MIDI:");
                let cur = self
                    .selected_midi
                    .and_then(|i| self.midi_files.get(i))
                    .and_then(|p| p.file_stem().and_then(|s| s.to_str()).map(str::to_string))
                    .unwrap_or_else(|| "(none)".to_string());
                egui::ComboBox::from_id_salt("midi")
                    .selected_text(cur)
                    .show_ui(ui, |ui| {
                        for (i, p) in self.midi_files.iter().enumerate() {
                            let stem = p
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("?");
                            ui.selectable_value(&mut self.selected_midi, Some(i), stem);
                        }
                    });
            });

            ui.separator();

            // ---- Phase-driven section --------------------------------
            match &self.phase {
                Phase::Idle => {
                    if ui.button("render & start trial").clicked() {
                        want_render = true;
                    }
                    ui.label("Pick two voices and a MIDI, then render. First render per (voice, MIDI) takes a few seconds; subsequent renders are cached.");
                }
                Phase::Rendering { started, .. } => {
                    let elapsed = started.elapsed().as_secs_f32();
                    ui.label(format!("rendering both voices… {:.1}s", elapsed));
                    ui.add(egui::Spinner::new());
                }
                Phase::Listening {
                    slot_a_buf,
                    slot_b_buf,
                    midi,
                    ..
                } => {
                    ui.heading("blind listening");
                    ui.label(format!("MIDI: {}", midi.display()));
                    ui.label(format!(
                        "Slot A: {:.1}s @ {} Hz   |   Slot B: {:.1}s @ {} Hz",
                        slot_a_buf.samples.len() as f32 / 2.0 / slot_a_buf.sample_rate as f32,
                        slot_a_buf.sample_rate,
                        slot_b_buf.samples.len() as f32 / 2.0 / slot_b_buf.sample_rate as f32,
                        slot_b_buf.sample_rate,
                    ));
                    ui.horizontal(|ui| {
                        if ui.button("▶ play A").clicked() {
                            want_play_a = true;
                        }
                        if ui.button("▶ play B").clicked() {
                            want_play_b = true;
                        }
                        if ui.button("■ stop").clicked() {
                            want_stop = true;
                        }
                    });
                    ui.separator();
                    ui.label("Which slot sounds more like a real piano?");
                    ui.horizontal(|ui| {
                        if ui.button("A is more real").clicked() {
                            want_verdict = Some(Verdict::SlotA);
                        }
                        if ui.button("B is more real").clicked() {
                            want_verdict = Some(Verdict::SlotB);
                        }
                        if ui.button("indistinguishable").clicked() {
                            want_verdict = Some(Verdict::Tie);
                        }
                    });
                    ui.separator();
                    if ui.button("✕ abandon trial").clicked() {
                        want_reset = true;
                    }
                }
                Phase::Reveal {
                    voice_a_canonical,
                    voice_b_canonical,
                    voice_in_slot_a,
                    verdict,
                    ..
                } => {
                    ui.heading("reveal");
                    let voice_in_slot_b = if voice_in_slot_a == voice_a_canonical {
                        voice_b_canonical.clone()
                    } else {
                        voice_a_canonical.clone()
                    };
                    ui.label(format!("Slot A was: {voice_in_slot_a}"));
                    ui.label(format!("Slot B was: {voice_in_slot_b}"));
                    let winner = match verdict {
                        Verdict::SlotA => format!("you picked: {voice_in_slot_a} (Slot A)"),
                        Verdict::SlotB => format!("you picked: {voice_in_slot_b} (Slot B)"),
                        Verdict::Tie => "you marked: indistinguishable".to_string(),
                    };
                    ui.label(winner);
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("▶ play A again").clicked() {
                            want_play_a = true;
                        }
                        if ui.button("▶ play B again").clicked() {
                            want_play_b = true;
                        }
                        if ui.button("■ stop").clicked() {
                            want_stop = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        if ui.button("next trial (same voices, re-shuffle)").clicked() {
                            want_next_same = true;
                        }
                        if ui.button("new pair / new MIDI").clicked() {
                            want_reset = true;
                        }
                    });
                }
            }
        });

        // ---- Apply intents (after the borrow on self.phase is dropped) ----
        if input_changed {
            self.refresh_stats();
        }
        if want_render {
            self.start_render();
        }
        if want_stop {
            self.stop_audio();
        }
        if want_play_a {
            // Pull a fresh clone of the buffer based on the current phase
            // because submitting verdict moves us to Reveal.
            let buf = match &self.phase {
                Phase::Listening { slot_a_buf, .. } => Some(slot_a_buf.clone()),
                Phase::Reveal { slot_a_buf, .. } => Some(slot_a_buf.clone()),
                _ => None,
            };
            if let Some(b) = buf {
                self.play_buf(&b);
            }
        }
        if want_play_b {
            let buf = match &self.phase {
                Phase::Listening { slot_b_buf, .. } => Some(slot_b_buf.clone()),
                Phase::Reveal { slot_b_buf, .. } => Some(slot_b_buf.clone()),
                _ => None,
            };
            if let Some(b) = buf {
                self.play_buf(&b);
            }
        }
        if let Some(v) = want_verdict {
            self.submit_verdict(v);
        }
        if want_next_same {
            self.next_trial_same_voices();
        }
        if want_reset {
            self.reset_to_idle();
        }
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([920.0, 620.0]),
        ..Default::default()
    };
    eframe::run_native(
        "ksabtest — blind A/B harness",
        options,
        Box::new(|_cc| match App::new() {
            Ok(app) => Ok(Box::new(app) as Box<dyn eframe::App>),
            Err(e) => {
                eprintln!("ksabtest: init failed: {e}");
                std::process::exit(2);
            }
        }),
    )
}
