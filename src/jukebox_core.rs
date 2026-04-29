//! jukebox_core — shared substrate for the `jukebox_lite{,_c,_g}` UI
//! variants.
//!
//! ## Why
//!
//! The three `jukebox_lite*` bins each grew 1200–2600 lines of
//! independently-written audio + library + CP-server scaffolding. They
//! shared the *concept* but not the code, so each variant re-invented
//! (and re-broke) the same egui first-paint discipline, the same
//! cpal mixer plumbing, and the same play_log/library_db wiring.
//!
//! `JukeboxCore` collects those fixed parts into one place so a UI
//! variant only needs to implement [`JukeboxApp::render`]. The runner
//! ([`JukeboxRunner`]) provides the `eframe::App` impl with the two
//! invariants that previously had to be re-discovered per bin:
//!
//! 1. **First-paint backstop.** `JukeboxRunner::update` unconditionally
//!    calls `ctx.request_repaint_after(500ms)` at the end of every
//!    frame. eframe needs the GUI to paint at least once before it
//!    flips `WS_VISIBLE` on the OS window; if the variant code only
//!    requests repaints "while playing", the very first frame after
//!    construction can stall and the window stays invisible. The
//!    backstop means we always come back within 500ms regardless of
//!    play state.
//!
//! 2. **CP-driven repaint.** Every CP server callback registered via
//!    [`JukeboxControl`] captures an `egui::Context` clone and calls
//!    `request_repaint()` after queuing the command. Without that, a
//!    `dispatch load_track` from a verifier would sit in the queue
//!    until the user happened to wiggle the mouse.
//!
//! ## Layering
//!
//! ```text
//!  UI variant (jukebox_lite, jukebox_lite_c, jukebox_lite_g)
//!     impl JukeboxApp
//!         ↓ &mut JukeboxCore
//!  JukeboxCore (this file)
//!     ├── JukeboxLibrary    — track scan + library_db
//!     ├── JukeboxAudio      — cpal stream + AudioState + pending_render
//!     ├── JukeboxSelection  — tile/search/sort/selected_label
//!     ├── JukeboxHistory    — play_log + favorites + recent
//!     └── JukeboxControl    — CP server (egui_ctx-aware)
//!         ↓
//!  keysynth lib (gui_cp / library_db / play_log / preview_cache)
//! ```
//!
//! ## Non-goals
//!
//! * No UI rendering. `JukeboxCore` never touches `egui::Ui`. The UI
//!   layer reads visible tracks / selection / playing state and calls
//!   into `play_track` / `stop` / `select_tile`.
//! * No music-knowledge heuristics (mood→voice mapping). That stays in
//!   the UI variant because each one wants slightly different defaults.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cp_core::RpcError;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use eframe::egui;
use hound::{SampleFormat as WavSampleFormat, WavReader};
use serde::Serialize;
use serde_json::json;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{Decoder, DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::gui_cp;
use crate::library_db::{LibraryDb, Song, SongFilter, SongSort, VoiceFilter};
use crate::play_log::{PlayEntry, PlayLogDb};

// ---------------------------------------------------------------------------
// Track + format
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Wav,
    Mp3,
    Mid,
}

impl Format {
    pub fn from_ext(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "wav" => Some(Format::Wav),
            "mp3" => Some(Format::Mp3),
            "mid" | "midi" => Some(Format::Mid),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Track {
    pub path: PathBuf,
    /// Stable id (file stem) — joins to `Song::id` and `play_log` keys.
    pub label: String,
    pub format: Format,
}

/// Walk each directory and collect playable files. Within a single
/// directory, MP3 wins over a same-stem WAV (matches the production
/// transition strategy: rendered MP3s shipped alongside legacy WAVs).
pub fn scan_dirs(dirs: &[&Path]) -> Vec<Track> {
    let mut out: Vec<Track> = Vec::new();
    for d in dirs {
        let read = match std::fs::read_dir(d) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let mut dir_tracks: Vec<Track> = Vec::new();
        for ent in read.flatten() {
            let p = ent.path();
            let ext = match p.extension().and_then(|s| s.to_str()) {
                Some(e) => e.to_ascii_lowercase(),
                None => continue,
            };
            let fmt = match Format::from_ext(&ext) {
                Some(f) => f,
                None => continue,
            };
            let stem = match p.file_stem().and_then(|s| s.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            dir_tracks.push(Track {
                path: p,
                label: stem,
                format: fmt,
            });
        }
        let mp3_stems: HashSet<String> = dir_tracks
            .iter()
            .filter(|t| t.format == Format::Mp3)
            .map(|t| t.label.clone())
            .collect();
        for t in dir_tracks {
            if t.format == Format::Wav && mp3_stems.contains(&t.label) {
                continue;
            }
            out.push(t);
        }
    }
    out.sort_by(|a, b| a.label.cmp(&b.label));
    out
}

// ---------------------------------------------------------------------------
// Decoder (WAV via hound, MP3 via symphonia)
// ---------------------------------------------------------------------------

pub struct Mp3State {
    #[allow(dead_code)]
    path: PathBuf,
    format: Box<dyn FormatReader>,
    decoder: Box<dyn Decoder>,
    track_id: u32,
    scratch: Vec<f32>,
    scratch_pos: usize,
    eof: bool,
}

pub enum DecoderState {
    Wav {
        reader: WavReader<BufReader<File>>,
        sample_format: WavSampleFormat,
        bits_per_sample: u16,
    },
    Mp3(Mp3State),
}

pub struct DecodedTrack {
    pub state: DecoderState,
    pub sample_rate: u32,
    pub channels: u16,
    pub total_samples: u64,
}

fn open_wav(path: &Path) -> Result<DecodedTrack, String> {
    let reader = WavReader::open(path).map_err(|e| format!("hound open: {e}"))?;
    let spec = reader.spec();
    let total = reader.duration() as u64 * spec.channels as u64;
    Ok(DecodedTrack {
        sample_rate: spec.sample_rate,
        channels: spec.channels,
        total_samples: total,
        state: DecoderState::Wav {
            reader,
            sample_format: spec.sample_format,
            bits_per_sample: spec.bits_per_sample,
        },
    })
}

fn open_mp3(path: &Path) -> Result<DecodedTrack, String> {
    let file = File::open(path).map_err(|e| format!("mp3 open: {e}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| format!("mp3 probe: {e}"))?;
    let format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| "mp3: no decodable track".to_string())?;
    let track_id = track.id;
    let cp = track.codec_params.clone();
    let sr = cp
        .sample_rate
        .ok_or_else(|| "mp3: missing sample rate".to_string())?;
    let ch = cp.channels.map(|c| c.count() as u16).unwrap_or(2);
    let total = cp.n_frames.map(|n| n * ch as u64).unwrap_or(0);
    let decoder = symphonia::default::get_codecs()
        .make(&cp, &DecoderOptions::default())
        .map_err(|e| format!("mp3 codec: {e}"))?;
    Ok(DecodedTrack {
        sample_rate: sr,
        channels: ch,
        total_samples: total,
        state: DecoderState::Mp3(Mp3State {
            path: path.to_path_buf(),
            format,
            decoder,
            track_id,
            scratch: Vec::new(),
            scratch_pos: 0,
            eof: false,
        }),
    })
}

pub fn open_decoded(path: &Path) -> Result<DecodedTrack, String> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "wav" => open_wav(path),
        "mp3" => open_mp3(path),
        other => Err(format!("unsupported format: {other}")),
    }
}

fn mp3_decode_next(state: &mut Mp3State) -> Result<(), String> {
    if state.eof {
        return Ok(());
    }
    loop {
        let packet = match state.format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                state.eof = true;
                return Ok(());
            }
            Err(SymphoniaError::ResetRequired) => {
                state.eof = true;
                return Ok(());
            }
            Err(e) => return Err(format!("mp3 next_packet: {e}")),
        };
        if packet.track_id() != state.track_id {
            continue;
        }
        let decoded = match state.decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::IoError(_)) => {
                state.eof = true;
                return Ok(());
            }
            Err(e) => return Err(format!("mp3 decode: {e}")),
        };
        let frames = decoded.frames();
        let spec = *decoded.spec();
        let ch = spec.channels.count();
        state.scratch.clear();
        state.scratch_pos = 0;
        state.scratch.reserve(frames * ch);
        match decoded {
            AudioBufferRef::F32(buf) => {
                for f in 0..frames {
                    for c in 0..ch {
                        state.scratch.push(buf.chan(c)[f]);
                    }
                }
            }
            AudioBufferRef::S16(buf) => {
                let inv = 1.0f32 / i16::MAX as f32;
                for f in 0..frames {
                    for c in 0..ch {
                        state.scratch.push(buf.chan(c)[f] as f32 * inv);
                    }
                }
            }
            AudioBufferRef::S32(buf) => {
                let inv = 1.0f32 / i32::MAX as f32;
                for f in 0..frames {
                    for c in 0..ch {
                        state.scratch.push(buf.chan(c)[f] as f32 * inv);
                    }
                }
            }
            AudioBufferRef::F64(buf) => {
                for f in 0..frames {
                    for c in 0..ch {
                        state.scratch.push(buf.chan(c)[f] as f32);
                    }
                }
            }
            _ => continue,
        }
        if state.scratch.is_empty() {
            continue;
        }
        return Ok(());
    }
}

fn read_one_sample(dec: &mut DecodedTrack) -> Option<f32> {
    match &mut dec.state {
        DecoderState::Wav {
            reader,
            sample_format,
            bits_per_sample,
        } => match *sample_format {
            WavSampleFormat::Int => {
                let s: i32 = reader.samples::<i32>().next()?.ok()?;
                let max = ((1u32 << (*bits_per_sample as u32 - 1)) - 1) as f32;
                Some(s as f32 / max)
            }
            WavSampleFormat::Float => {
                let s: f32 = reader.samples::<f32>().next()?.ok()?;
                Some(s)
            }
        },
        DecoderState::Mp3(state) => {
            if state.scratch_pos >= state.scratch.len() {
                if state.eof {
                    return None;
                }
                if mp3_decode_next(state).is_err() {
                    state.eof = true;
                    return None;
                }
                if state.scratch_pos >= state.scratch.len() {
                    return None;
                }
            }
            let s = state.scratch[state.scratch_pos];
            state.scratch_pos += 1;
            Some(s)
        }
    }
}

// ---------------------------------------------------------------------------
// JukeboxAudio — single-slot mixer + decoded source
// ---------------------------------------------------------------------------

/// Audio state shared between the egui thread and the cpal callback.
/// The cpal callback only ever flips `is_playing` to false on EOF and
/// advances `cursor_frames`; everything else is written from the egui
/// thread under the mutex.
pub struct AudioState {
    pub decoded: Option<DecodedTrack>,
    pub is_playing: bool,
    pub cursor_frames: u64,
    pub label: String,
    pub volume: f32,
}

impl AudioState {
    fn new() -> Self {
        Self {
            decoded: None,
            is_playing: false,
            cursor_frames: 0,
            label: String::new(),
            volume: 0.85,
        }
    }
}

pub struct MixerStream {
    pub state: Arc<Mutex<AudioState>>,
    _stream: cpal::Stream,
}

fn fill_callback(out: &mut [f32], channels: u16, state: &Arc<Mutex<AudioState>>) {
    for s in out.iter_mut() {
        *s = 0.0;
    }
    let mut st = match state.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    if !st.is_playing {
        return;
    }
    let vol = st.volume.clamp(0.0, 1.5);
    let dec = match st.decoded.as_mut() {
        Some(d) => d,
        None => {
            st.is_playing = false;
            return;
        }
    };
    let dec_ch = dec.channels;
    let frames = out.len() / channels.max(1) as usize;
    let mut cursor_advanced: u64 = 0;
    let mut hit_eof = false;
    for f in 0..frames {
        let (l, r) = match dec_ch {
            1 => {
                let s = match read_one_sample(dec) {
                    Some(v) => v,
                    None => {
                        hit_eof = true;
                        break;
                    }
                };
                (s, s)
            }
            2 => {
                let l = match read_one_sample(dec) {
                    Some(v) => v,
                    None => {
                        hit_eof = true;
                        break;
                    }
                };
                let r = read_one_sample(dec).unwrap_or(l);
                (l, r)
            }
            n => {
                let l = match read_one_sample(dec) {
                    Some(v) => v,
                    None => {
                        hit_eof = true;
                        break;
                    }
                };
                let r = read_one_sample(dec).unwrap_or(l);
                for _ in 2..n {
                    let _ = read_one_sample(dec);
                }
                (l, r)
            }
        };
        cursor_advanced += 1;
        let bus_l = (l * vol).tanh();
        let bus_r = (r * vol).tanh();
        let base = f * channels as usize;
        match channels {
            1 => out[base] = (bus_l + bus_r) * 0.5,
            2 => {
                out[base] = bus_l;
                out[base + 1] = bus_r;
            }
            n => {
                out[base] = bus_l;
                out[base + 1] = bus_r;
                for c in 2..n as usize {
                    out[base + c] = 0.0;
                }
            }
        }
    }
    st.cursor_frames = st.cursor_frames.saturating_add(cursor_advanced);
    if hit_eof {
        st.is_playing = false;
    }
}

pub fn build_mixer_stream(state: Arc<Mutex<AudioState>>) -> Result<MixerStream, String> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| "no default output device".to_string())?;
    let supported = device
        .default_output_config()
        .map_err(|e| format!("default_output_config: {e}"))?;
    let sample_format = supported.sample_format();
    let channels = supported.channels();
    let stream_cfg: StreamConfig = supported.into();
    let err_fn = |err| eprintln!("jukebox_core audio stream error: {err}");
    let stream = match sample_format {
        SampleFormat::F32 => {
            let st = state.clone();
            device
                .build_output_stream(
                    &stream_cfg,
                    move |out: &mut [f32], _| fill_callback(out, channels, &st),
                    err_fn,
                    None,
                )
                .map_err(|e| format!("build f32: {e}"))?
        }
        SampleFormat::I16 => {
            let st = state.clone();
            let mut scratch: Vec<f32> = Vec::new();
            device
                .build_output_stream(
                    &stream_cfg,
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
                .map_err(|e| format!("build i16: {e}"))?
        }
        SampleFormat::U16 => {
            let st = state.clone();
            let mut scratch: Vec<f32> = Vec::new();
            device
                .build_output_stream(
                    &stream_cfg,
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
                            *dst = (((c + 1.0) * 0.5) * u16::MAX as f32) as u16;
                        }
                    },
                    err_fn,
                    None,
                )
                .map_err(|e| format!("build u16: {e}"))?
        }
        other => return Err(format!("unsupported sample format: {other:?}")),
    };
    stream.play().map_err(|e| format!("stream.play: {e}"))?;
    Ok(MixerStream {
        state,
        _stream: stream,
    })
}

// ---------------------------------------------------------------------------
// Preview cache integration
// ---------------------------------------------------------------------------

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

fn cache_key(track_path: &Path, voice_id: &str) -> crate::preview_cache::CacheKey {
    crate::preview_cache::CacheKey {
        song_path: track_path.to_path_buf(),
        voice_id: voice_id.to_string(),
        voice_dll: None,
        render_params: crate::preview_cache::RenderParams::default(),
    }
}

fn open_preview_cache() -> Result<crate::preview_cache::Cache, String> {
    let cache_dir = PathBuf::from("bench-out/cache");
    crate::preview_cache::Cache::new(&cache_dir, 1_073_741_824)
        .map_err(|e| format!("Cache::new: {e}"))
}

pub fn lookup_in_cache(track_path: &Path, voice_id: &str) -> Result<Option<PathBuf>, String> {
    let cache = open_preview_cache()?;
    let key = cache_key(track_path, voice_id);
    cache.lookup(&key).map_err(|e| format!("lookup: {e}"))
}

pub fn render_midi_blocking(midi_path: &Path, voice_id: &str) -> Result<PathBuf, String> {
    let cache = open_preview_cache()?;
    let key = cache_key(midi_path, voice_id);
    let bin = render_midi_binary_path()
        .ok_or_else(|| "render_midi binary not found alongside jukebox binary".to_string())?;
    crate::preview_cache::render_to_cache(&cache, &key, voice_id, &bin)
        .map_err(|e| format!("render_to_cache: {e}"))
}

pub struct PendingRender {
    pub track_label: String,
    pub voice_id: String,
    pub started: Instant,
    pub rx: std::sync::mpsc::Receiver<Result<PathBuf, String>>,
    pub _handle: std::thread::JoinHandle<()>,
}

// ---------------------------------------------------------------------------
// Tile semantics
// ---------------------------------------------------------------------------

/// Sidebar tile / category. Variants share a common vocabulary so the
/// CP `select_tile` command works against any UI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tile {
    All,
    Favorites,
    Recent,
    Classical,
    Game,
    Folk,
}

impl Tile {
    pub fn label(self) -> &'static str {
        match self {
            Tile::All => "All",
            Tile::Favorites => "Favorites",
            Tile::Recent => "Recent",
            Tile::Classical => "Classical",
            Tile::Game => "Game",
            Tile::Folk => "Folk",
        }
    }
    pub fn all() -> &'static [Tile] {
        &[
            Tile::All,
            Tile::Favorites,
            Tile::Recent,
            Tile::Classical,
            Tile::Game,
            Tile::Folk,
        ]
    }
    pub fn from_str(s: &str) -> Option<Tile> {
        match s.to_ascii_lowercase().as_str() {
            "all" => Some(Tile::All),
            "favorites" | "favs" | "fav" => Some(Tile::Favorites),
            "recent" => Some(Tile::Recent),
            "classical" => Some(Tile::Classical),
            "game" => Some(Tile::Game),
            "folk" => Some(Tile::Folk),
            _ => None,
        }
    }
}

/// Decide whether `track` belongs in `tile`. Tracks without DB
/// metadata fall through to `Game` (catch-all for filename-only entries
/// like `cc0_*` chiptune dumps).
pub fn track_matches_tile(
    tile: Tile,
    track: &Track,
    song: Option<&Song>,
    favorites: &HashSet<String>,
    recent: &HashSet<String>,
) -> bool {
    match tile {
        Tile::All => true,
        Tile::Favorites => favorites.contains(&track.label),
        Tile::Recent => recent.contains(&track.label),
        Tile::Classical => song
            .and_then(|s| s.era.as_deref())
            .map(|e| matches!(e, "Baroque" | "Classical" | "Romantic" | "Modern"))
            .unwrap_or(false),
        Tile::Folk => song
            .map(|s| {
                s.instrument.to_ascii_lowercase().contains("guitar")
                    || s.era.as_deref() == Some("Traditional")
            })
            .unwrap_or(false),
        Tile::Game => {
            if song.is_none() {
                return true;
            }
            let s = song.unwrap();
            let ins = s.instrument.to_ascii_lowercase();
            ins.contains("synth") || ins.contains("game") || ins.contains("chip")
        }
    }
}

// ---------------------------------------------------------------------------
// JukeboxLibrary — track scan + library_db lookup
// ---------------------------------------------------------------------------

pub const RECENT_WINDOW: usize = 24;

pub struct JukeboxLibrary {
    /// Files discovered by `scan_dirs`.
    pub tracks: Vec<Track>,
    /// `Song` rows from `library_db`, keyed by `Song::id` (= file stem).
    pub song_index: HashMap<String, Song>,
    /// Number of voice rows in the DB, reported by the stats popover.
    pub db_voice_count: usize,
    /// Source directories, retained so `rescan` can re-walk them.
    pub source_dirs: Vec<PathBuf>,
}

impl JukeboxLibrary {
    fn load(dirs: Vec<PathBuf>) -> Self {
        let dir_refs: Vec<&Path> = dirs.iter().map(|p| p.as_path()).collect();
        let tracks = scan_dirs(&dir_refs);
        let (song_index, db_voice_count) = load_library_index();
        Self {
            tracks,
            song_index,
            db_voice_count,
            source_dirs: dirs,
        }
    }

    pub fn rescan(&mut self) {
        let dir_refs: Vec<&Path> = self.source_dirs.iter().map(|p| p.as_path()).collect();
        self.tracks = scan_dirs(&dir_refs);
        let (idx, vc) = load_library_index();
        self.song_index = idx;
        self.db_voice_count = vc;
    }

    pub fn find_track(&self, label: &str) -> Option<Track> {
        self.tracks.iter().find(|t| t.label == label).cloned()
    }

    pub fn song(&self, label: &str) -> Option<&Song> {
        self.song_index.get(label)
    }
}

fn load_library_index() -> (HashMap<String, Song>, usize) {
    let db_path = PathBuf::from("bench-out/library.db");
    let manifest = PathBuf::from("bench-out/songs/manifest.json");
    let voices_live = PathBuf::from("voices_live");
    let mut db = match LibraryDb::open(&db_path) {
        Ok(db) => db,
        Err(e) => {
            eprintln!(
                "jukebox_core: library_db open failed ({}): {e} — running without DB metadata",
                db_path.display()
            );
            return (HashMap::new(), 0);
        }
    };
    if let Err(e) = db.migrate() {
        eprintln!("jukebox_core: library_db migrate failed: {e}");
        return (HashMap::new(), 0);
    }
    if manifest.is_file() {
        if let Err(e) = db.import_songs(&manifest) {
            eprintln!("jukebox_core: library_db import_songs failed: {e}");
        }
    }
    if voices_live.is_dir() {
        if let Err(e) = db.import_voices(&voices_live) {
            eprintln!("jukebox_core: library_db import_voices failed: {e}");
        }
    }
    let songs = db
        .query_songs(&SongFilter {
            sort: SongSort::ByComposer,
            ..Default::default()
        })
        .unwrap_or_default();
    let voice_count = db
        .query_voices(&VoiceFilter::default())
        .map(|v| v.len())
        .unwrap_or(0);
    let mut idx = HashMap::new();
    for s in songs {
        idx.insert(s.id.clone(), s);
    }
    (idx, voice_count)
}

// ---------------------------------------------------------------------------
// JukeboxHistory — play_log + favorites + recent
// ---------------------------------------------------------------------------

pub struct JukeboxHistory {
    pub play_log: Option<PlayLogDb>,
    pub favorites: HashSet<String>,
    pub recent_ids: HashSet<String>,
    pub recent_entries: Vec<PlayEntry>,
}

impl JukeboxHistory {
    fn load() -> Self {
        let play_log = load_play_log();
        let (favorites, recent_ids, recent_entries) = snapshot_history(play_log.as_ref());
        Self {
            play_log,
            favorites,
            recent_ids,
            recent_entries,
        }
    }

    pub fn refresh(&mut self) {
        let (f, r, re) = snapshot_history(self.play_log.as_ref());
        self.favorites = f;
        self.recent_ids = r;
        self.recent_entries = re;
    }

    pub fn record_play(&mut self, label: &str, voice: Option<&str>) {
        if let Some(db) = self.play_log.as_mut() {
            if let Err(e) = db.record_play(label, voice, None) {
                eprintln!("jukebox_core: record_play({label}): {e}");
            }
        }
    }
}

fn load_play_log() -> Option<PlayLogDb> {
    let db_path = PathBuf::from("bench-out/play_log.db");
    let mut db = match PlayLogDb::open(&db_path) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("jukebox_core: play_log open failed: {e}");
            return None;
        }
    };
    if let Err(e) = db.migrate() {
        eprintln!("jukebox_core: play_log migrate failed: {e}");
        return None;
    }
    Some(db)
}

fn snapshot_history(
    log: Option<&PlayLogDb>,
) -> (HashSet<String>, HashSet<String>, Vec<PlayEntry>) {
    let mut favs: HashSet<String> = HashSet::new();
    let mut recent_ids: HashSet<String> = HashSet::new();
    let mut recent_entries: Vec<PlayEntry> = Vec::new();
    if let Some(db) = log {
        if let Ok(f) = db.favorites() {
            favs = f.into_iter().collect();
        }
        if let Ok(r) = db.recent_plays(RECENT_WINDOW) {
            for e in &r {
                recent_ids.insert(e.song_id.clone());
            }
            recent_entries = r;
        }
    }
    (favs, recent_ids, recent_entries)
}

// ---------------------------------------------------------------------------
// JukeboxAudio — audio state + mixer + pending render queue
// ---------------------------------------------------------------------------

pub struct JukeboxAudio {
    pub mixer: MixerStream,
    /// Sample-rate of the most recently loaded track. Used by progress
    /// bars (cursor_frames / sample_rate → seconds).
    pub sample_rate_for_progress: u32,
    pub pending_render: Option<PendingRender>,
}

impl JukeboxAudio {
    fn build() -> Result<Self, String> {
        let state = Arc::new(Mutex::new(AudioState::new()));
        let mixer = build_mixer_stream(state).map_err(|e| format!("audio: {e}"))?;
        Ok(Self {
            mixer,
            sample_rate_for_progress: 44_100,
            pending_render: None,
        })
    }

    /// Snapshot of the audio state suitable for CP / UI consumption.
    /// Releases the lock immediately — never hold the audio mutex
    /// across an egui paint.
    pub fn snapshot(&self) -> AudioSnapshot {
        let st = match self.mixer.state.lock() {
            Ok(g) => g,
            Err(_) => {
                return AudioSnapshot::default();
            }
        };
        AudioSnapshot {
            is_playing: st.is_playing,
            label: if st.label.is_empty() {
                None
            } else {
                Some(st.label.clone())
            },
            cursor_frames: st.cursor_frames,
            total_frames: st
                .decoded
                .as_ref()
                .map(|d| d.total_samples / d.channels.max(1) as u64)
                .unwrap_or(0),
        }
    }

    pub fn is_playing(&self) -> bool {
        self.mixer
            .state
            .lock()
            .map(|s| s.is_playing)
            .unwrap_or(false)
    }

    pub fn stop(&self) {
        if let Ok(mut st) = self.mixer.state.lock() {
            st.is_playing = false;
        }
    }

    /// Resume playback if a track is loaded.
    pub fn resume(&self) {
        if let Ok(mut st) = self.mixer.state.lock() {
            if st.decoded.is_some() {
                st.is_playing = true;
            }
        }
    }

    /// Replace the current decoded track and start playing it. Returns
    /// the file's sample-rate so the caller can update the progress bar
    /// reference.
    fn load(&mut self, label: String, dec: DecodedTrack) -> u32 {
        let sr = dec.sample_rate;
        if let Ok(mut st) = self.mixer.state.lock() {
            st.decoded = Some(dec);
            st.is_playing = true;
            st.cursor_frames = 0;
            st.label = label;
        }
        self.sample_rate_for_progress = sr;
        sr
    }
}

#[derive(Clone, Default)]
pub struct AudioSnapshot {
    pub is_playing: bool,
    pub label: Option<String>,
    pub cursor_frames: u64,
    pub total_frames: u64,
}

// ---------------------------------------------------------------------------
// JukeboxSelection — tile/search/sort/selected_label
// ---------------------------------------------------------------------------

pub struct JukeboxSelection {
    pub tile: Tile,
    pub search: String,
    pub selected_label: Option<String>,
}

impl JukeboxSelection {
    fn new(initial: Option<String>) -> Self {
        Self {
            tile: Tile::All,
            search: String::new(),
            selected_label: initial,
        }
    }

    /// Compute the visible tracks given current tile + search + history.
    /// Returns clones so the UI can mutate `JukeboxCore` (e.g. start
    /// playback) while iterating.
    pub fn visible(&self, library: &JukeboxLibrary, history: &JukeboxHistory) -> Vec<Track> {
        let needle = self.search.trim().to_ascii_lowercase();
        let mut out: Vec<Track> = library
            .tracks
            .iter()
            .filter(|t| {
                let song = library.song_index.get(&t.label);
                if !track_matches_tile(self.tile, t, song, &history.favorites, &history.recent_ids)
                {
                    return false;
                }
                if needle.is_empty() {
                    return true;
                }
                let title = song.map(|s| s.title.as_str()).unwrap_or("");
                let composer = song.map(|s| s.composer.as_str()).unwrap_or("");
                t.label.to_ascii_lowercase().contains(&needle)
                    || title.to_ascii_lowercase().contains(&needle)
                    || composer.to_ascii_lowercase().contains(&needle)
            })
            .cloned()
            .collect();
        if self.tile == Tile::Recent {
            let order: HashMap<&str, usize> = history
                .recent_entries
                .iter()
                .enumerate()
                .map(|(i, e)| (e.song_id.as_str(), i))
                .collect();
            out.sort_by_key(|t| order.get(t.label.as_str()).copied().unwrap_or(usize::MAX));
        } else if !library.song_index.is_empty() {
            out.sort_by(|a, b| {
                let ka = library.song_index.get(&a.label).map(|s| s.composer_key.clone());
                let kb = library.song_index.get(&b.label).map(|s| s.composer_key.clone());
                match (ka, kb) {
                    (Some(x), Some(y)) => x.cmp(&y).then_with(|| a.label.cmp(&b.label)),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => a.label.cmp(&b.label),
                }
            });
        }
        out
    }
}

// ---------------------------------------------------------------------------
// CP server — snapshot, command, and the egui-aware spawn
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, Serialize)]
pub struct CpSnapshot {
    pub frame_id: u64,
    pub track_count: usize,
    pub db_song_count: usize,
    pub db_voice_count: usize,
    pub selected_label: Option<String>,
    pub selected_tile: String,
    pub any_playing: bool,
    pub loaded_label: Option<String>,
    pub cursor_frames: u64,
    pub total_frames: u64,
    pub sample_rate: u32,
    /// Catalog rows visible after current tile + filter, in display order.
    pub visible_labels: Vec<String>,
    /// Every catalog row label, regardless of filter.
    pub all_labels: Vec<String>,
}

#[derive(Clone, Debug)]
pub enum CpCommand {
    LoadTrack { label: String },
    Play,
    Stop,
    SelectTile { tile: String },
    Rescan,
}

pub struct JukeboxControl {
    pub state: gui_cp::State<CpSnapshot, CpCommand>,
    /// Held only to keep the CP listener alive — drop = shutdown.
    _handle: Option<gui_cp::Handle>,
    pub frame_id: u64,
}

impl JukeboxControl {
    /// Start the CP server. The `egui_ctx` clone is captured by every
    /// command-pushing callback; each callback calls
    /// `egui_ctx.request_repaint()` after queuing so the egui main
    /// thread wakes up and drains the command on the next tick.
    fn spawn(app_name: &'static str, egui_ctx: egui::Context) -> Self {
        let state: gui_cp::State<CpSnapshot, CpCommand> = gui_cp::State::new();
        let endpoint = gui_cp::resolve_endpoint(app_name, None);
        let st_get = state.clone();
        let st_load = state.clone();
        let st_play = state.clone();
        let st_stop = state.clone();
        let st_tile = state.clone();
        let st_rescan = state.clone();
        let st_list = state.clone();
        let ctx_load = egui_ctx.clone();
        let ctx_play = egui_ctx.clone();
        let ctx_stop = egui_ctx.clone();
        let ctx_tile = egui_ctx.clone();
        let ctx_rescan = egui_ctx.clone();
        let builder = gui_cp::Builder::new(app_name, &endpoint)
            .register("get_state", move |_p| match st_get.read_snapshot() {
                Some(snap) => gui_cp::encode_result(snap),
                None => Ok(json!({"frame_id": 0, "warming_up": true})),
            })
            .register("load_track", move |p| {
                #[derive(serde::Deserialize)]
                struct Args {
                    label: String,
                }
                let a: Args = gui_cp::decode_params(p)?;
                st_load.push_command(CpCommand::LoadTrack {
                    label: a.label.clone(),
                });
                ctx_load.request_repaint();
                Ok(json!({"queued": "load_track", "label": a.label}))
            })
            .register("play", move |_p| {
                st_play.push_command(CpCommand::Play);
                ctx_play.request_repaint();
                Ok(json!({"queued": "play"}))
            })
            .register("stop", move |_p| {
                st_stop.push_command(CpCommand::Stop);
                ctx_stop.request_repaint();
                Ok(json!({"queued": "stop"}))
            })
            .register("select_tile", move |p| {
                #[derive(serde::Deserialize)]
                struct Args {
                    tile: String,
                }
                let a: Args = gui_cp::decode_params(p)?;
                st_tile.push_command(CpCommand::SelectTile {
                    tile: a.tile.clone(),
                });
                ctx_tile.request_repaint();
                Ok(json!({"queued": "select_tile", "tile": a.tile}))
            })
            .register("rescan", move |_p| {
                st_rescan.push_command(CpCommand::Rescan);
                ctx_rescan.request_repaint();
                Ok(json!({"queued": "rescan"}))
            })
            .register("list_tracks", move |p| {
                #[derive(serde::Deserialize, Default)]
                #[serde(default)]
                struct Args {
                    limit: Option<usize>,
                    contains: Option<String>,
                }
                let a: Args = serde_json::from_value(p)
                    .map_err(|e| RpcError::invalid_params(format!("param parse: {e}")))?;
                let snap = match st_list.read_snapshot() {
                    Some(s) => s,
                    None => return Ok(json!({"warming_up": true, "tracks": []})),
                };
                let needle = a.contains.unwrap_or_default().to_ascii_lowercase();
                let limit = a.limit.unwrap_or(usize::MAX);
                let tracks: Vec<&str> = snap
                    .all_labels
                    .iter()
                    .filter(|l| needle.is_empty() || l.to_ascii_lowercase().contains(&needle))
                    .take(limit)
                    .map(|s| s.as_str())
                    .collect();
                Ok(json!({
                    "tracks": tracks,
                    "track_count": snap.track_count,
                }))
            });
        let handle = match builder.serve() {
            Ok(h) => Some(h),
            Err(e) => {
                eprintln!("jukebox_core: CP server failed: {e}");
                None
            }
        };
        Self {
            state,
            _handle: handle,
            frame_id: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// JukeboxCore — composes the five sub-components above
// ---------------------------------------------------------------------------

/// Hook the UI variant uses to pick a voice for a given track + tile.
/// The UI variant supplies its own logic at construction time so each
/// variant can keep its preferred mood→voice mapping local.
pub type VoicePicker =
    Box<dyn Fn(Tile, Option<&Song>, &Track) -> &'static str + Send + Sync + 'static>;

/// Default voice picker. Mirrors `jukebox_lite`'s heuristic:
/// already-rendered audio uses the `"—"` sentinel; tile mood overrides
/// metadata; metadata era + instrument fall through last.
pub fn default_voice_picker() -> VoicePicker {
    Box::new(default_pick_voice)
}

fn default_pick_voice(tile: Tile, song: Option<&Song>, track: &Track) -> &'static str {
    if matches!(track.format, Format::Wav | Format::Mp3) {
        return "—";
    }
    match tile {
        Tile::Classical => return "piano-modal",
        Tile::Game => return "square",
        Tile::Folk => return "guitar-stk",
        _ => {}
    }
    if let Some(s) = song {
        let ins = s.instrument.to_ascii_lowercase();
        if ins.contains("guitar") {
            return "guitar-stk";
        }
        if ins.contains("piano") {
            return "piano-modal";
        }
        if let Some(era) = s.era.as_deref() {
            if matches!(era, "Baroque" | "Classical" | "Romantic" | "Modern") {
                return "piano-modal";
            }
            if era == "Traditional" {
                return "guitar-stk";
            }
        }
    }
    let stem = track.label.to_ascii_lowercase();
    if stem.starts_with("cc0_")
        || stem.starts_with("parodius")
        || stem.starts_with("bgm_v")
        || stem.starts_with('v')
    {
        return "square";
    }
    "guitar-stk"
}

/// Bundle of all jukebox-shared state. Owned single-source-of-truth
/// for the UI variant — every mutation flows through here.
pub struct JukeboxCore {
    pub library: JukeboxLibrary,
    pub audio: JukeboxAudio,
    pub selection: JukeboxSelection,
    pub history: JukeboxHistory,
    pub control: JukeboxControl,
    pub voice_picker: VoicePicker,
}

impl JukeboxCore {
    /// Build the core. Audio/mixer failures bubble up as `Err(String)`;
    /// library_db / play_log / CP-server failures only log (the GUI is
    /// expected to keep working with degraded metadata rather than
    /// silently fail to paint).
    pub fn new(
        dirs: Vec<PathBuf>,
        app_name: &'static str,
        egui_ctx: egui::Context,
    ) -> Result<Self, String> {
        Self::with_voice_picker(dirs, app_name, egui_ctx, default_voice_picker())
    }

    pub fn with_voice_picker(
        dirs: Vec<PathBuf>,
        app_name: &'static str,
        egui_ctx: egui::Context,
        voice_picker: VoicePicker,
    ) -> Result<Self, String> {
        let library = JukeboxLibrary::load(dirs);
        let history = JukeboxHistory::load();
        let audio = JukeboxAudio::build()?;
        let initial_label = history
            .recent_entries
            .first()
            .map(|e| e.song_id.clone())
            .or_else(|| history.favorites.iter().next().cloned())
            .or_else(|| library.tracks.first().map(|t| t.label.clone()));
        let selection = JukeboxSelection::new(initial_label);
        let control = JukeboxControl::spawn(app_name, egui_ctx);
        eprintln!(
            "jukebox_core: catalog ready — {} songs / {} voices ({} on-disk)",
            library.song_index.len(),
            library.db_voice_count,
            library.tracks.len(),
        );
        Ok(Self {
            library,
            audio,
            selection,
            history,
            control,
            voice_picker,
        })
    }

    pub fn visible_tracks(&self) -> Vec<Track> {
        self.selection.visible(&self.library, &self.history)
    }

    pub fn rescan(&mut self) {
        self.library.rescan();
        self.history.refresh();
    }

    pub fn select_tile(&mut self, tile: Tile) {
        self.selection.tile = tile;
    }

    pub fn stop(&mut self) {
        self.audio.stop();
    }

    /// Resolve a label → MIDI render or direct WAV/MP3 path → cpal.
    /// MIDI cache misses spawn a background `render_midi` subprocess
    /// and return early; `tick` will pick up the result.
    pub fn play_label(&mut self, label: &str) {
        let track = match self.library.find_track(label) {
            Some(t) => t,
            None => {
                eprintln!("jukebox_core: play_label unknown label={label}");
                return;
            }
        };
        let song = self.library.song(label).cloned();
        let voice = (self.voice_picker)(self.selection.tile, song.as_ref(), &track).to_string();
        self.selection.selected_label = Some(track.label.clone());

        let playable: PathBuf = match track.format {
            Format::Mid => match lookup_in_cache(&track.path, &voice) {
                Ok(Some(p)) => p,
                Ok(None) => {
                    self.spawn_render(&track, &voice);
                    return;
                }
                Err(e) => {
                    eprintln!("jukebox_core: cache lookup {}: {e}", track.path.display());
                    return;
                }
            },
            Format::Wav | Format::Mp3 => track.path.clone(),
        };
        self.load_resolved(track.label.clone(), playable, &voice);
    }

    fn spawn_render(&mut self, track: &Track, voice: &str) {
        let path = track.path.clone();
        let voice_for_thread = voice.to_string();
        let label = track.label.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = match std::thread::Builder::new()
            .name("jukebox-core-render".into())
            .spawn(move || {
                let result = render_midi_blocking(&path, &voice_for_thread);
                let _ = tx.send(result);
            }) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("jukebox_core: spawn render thread: {e}");
                return;
            }
        };
        eprintln!(
            "jukebox_core: render queued ({}) — voice={}",
            track.path.display(),
            voice
        );
        self.audio.pending_render = Some(PendingRender {
            track_label: label,
            voice_id: voice.to_string(),
            started: Instant::now(),
            rx,
            _handle: handle,
        });
    }

    fn poll_pending_render(&mut self) {
        let pending = match self.audio.pending_render.as_ref() {
            Some(p) => p,
            None => return,
        };
        let result = match pending.rx.try_recv() {
            Ok(r) => r,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.audio.pending_render = None;
                return;
            }
        };
        let pending = self.audio.pending_render.take().expect("guarded by Some()");
        let elapsed = pending.started.elapsed().as_millis();
        match result {
            Ok(wav) => {
                eprintln!(
                    "jukebox_core: render done ({}) in {} ms",
                    wav.display(),
                    elapsed
                );
                self.load_resolved(pending.track_label, wav, &pending.voice_id);
            }
            Err(e) => {
                eprintln!("jukebox_core: render failed: {e}");
            }
        }
    }

    fn load_resolved(&mut self, label: String, playable: PathBuf, voice: &str) {
        let dec = match open_decoded(&playable) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("jukebox_core: open {}: {e}", playable.display());
                return;
            }
        };
        self.audio.load(label.clone(), dec);
        self.selection.selected_label = Some(label.clone());
        let v = if voice == "—" { None } else { Some(voice) };
        self.history.record_play(&label, v);
        self.history.refresh();
    }

    fn drain_cp_commands(&mut self) {
        for cmd in self.control.state.drain_commands() {
            match cmd {
                CpCommand::LoadTrack { label } => self.play_label(&label),
                CpCommand::Play => self.audio.resume(),
                CpCommand::Stop => self.audio.stop(),
                CpCommand::SelectTile { tile } => {
                    if let Some(t) = Tile::from_str(&tile) {
                        self.selection.tile = t;
                    }
                }
                CpCommand::Rescan => self.rescan(),
            }
        }
    }

    fn publish_snapshot(&mut self, visible: &[Track]) {
        self.control.frame_id = self.control.frame_id.saturating_add(1);
        let audio = self.audio.snapshot();
        let snap = CpSnapshot {
            frame_id: self.control.frame_id,
            track_count: self.library.tracks.len(),
            db_song_count: self.library.song_index.len(),
            db_voice_count: self.library.db_voice_count,
            selected_label: self.selection.selected_label.clone(),
            selected_tile: self.selection.tile.label().to_string(),
            any_playing: audio.is_playing,
            loaded_label: audio.label,
            cursor_frames: audio.cursor_frames,
            total_frames: audio.total_frames,
            sample_rate: self.audio.sample_rate_for_progress,
            visible_labels: visible.iter().map(|t| t.label.clone()).collect(),
            all_labels: self.library.tracks.iter().map(|t| t.label.clone()).collect(),
        };
        self.control.state.publish(snap);
    }

    /// Per-frame housekeeping. UI variants don't normally call this
    /// directly; `JukeboxRunner::update` calls it before the variant's
    /// `render`. Returns the visible-track list so the UI can iterate
    /// it without recomputing.
    pub fn tick(&mut self, _ctx: &egui::Context) -> Vec<Track> {
        self.drain_cp_commands();
        self.poll_pending_render();
        let visible = self.visible_tracks();
        self.publish_snapshot(&visible);
        visible
    }
}

// ---------------------------------------------------------------------------
// JukeboxApp + JukeboxRunner
// ---------------------------------------------------------------------------

/// UI hook implemented by each `jukebox_lite*` variant. Receives the
/// pre-computed visible-track list so the variant can skip the filter
/// pass on its own.
pub trait JukeboxApp {
    fn render(&mut self, ctx: &egui::Context, core: &mut JukeboxCore, visible: &[Track]);
}

/// Default `eframe::App` impl. Owns one `JukeboxCore` and one
/// `JukeboxApp`. Applies the two invariants in [module-level docs](self):
///
/// * `core.tick(ctx)` runs CP drain + render-queue poll + visible-track
///   recomputation before each paint;
/// * `ctx.request_repaint_after(500ms)` runs unconditionally at the
///   end so first-paint always completes and idle state still
///   refreshes 2× per second (pending-render arrival, CP commands
///   from a verifier, etc.).
pub struct JukeboxRunner<A: JukeboxApp> {
    pub core: JukeboxCore,
    pub app: A,
}

impl<A: JukeboxApp> JukeboxRunner<A> {
    pub fn new(core: JukeboxCore, app: A) -> Self {
        Self { core, app }
    }
}

impl<A: JukeboxApp> eframe::App for JukeboxRunner<A> {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let visible = self.core.tick(ctx);
        self.app.render(ctx, &mut self.core, &visible);
        // Backstop: unconditional repaint request so the first paint
        // completes (so eframe flips WS_VISIBLE), and idle frames still
        // re-tick at 2 Hz to pick up CP commands or finished renders.
        ctx.request_repaint_after(Duration::from_millis(500));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_from_extension() {
        assert_eq!(Format::from_ext("mid"), Some(Format::Mid));
        assert_eq!(Format::from_ext("MIDI"), Some(Format::Mid));
        assert_eq!(Format::from_ext("wav"), Some(Format::Wav));
        assert_eq!(Format::from_ext("mp3"), Some(Format::Mp3));
        assert_eq!(Format::from_ext("flac"), None);
    }

    #[test]
    fn tile_label_round_trip() {
        for &t in Tile::all() {
            assert_eq!(Tile::from_str(t.label()), Some(t));
        }
    }

    #[test]
    fn default_voice_uses_active_tile_first() {
        let track = Track {
            path: PathBuf::from("dummy.mid"),
            label: "dummy".into(),
            format: Format::Mid,
        };
        assert_eq!(default_pick_voice(Tile::Classical, None, &track), "piano-modal");
        assert_eq!(default_pick_voice(Tile::Game, None, &track), "square");
        assert_eq!(default_pick_voice(Tile::Folk, None, &track), "guitar-stk");
    }

    #[test]
    fn default_voice_falls_back_to_song_metadata() {
        let track = Track {
            path: PathBuf::from("x.mid"),
            label: "x".into(),
            format: Format::Mid,
        };
        let song = Song {
            id: "x".into(),
            title: "T".into(),
            composer: "C".into(),
            composer_key: "c".into(),
            era: Some("Baroque".into()),
            instrument: "guitar".into(),
            license: "PD".into(),
            source: None,
            source_url: None,
            suggested_voice: None,
            midi_path: PathBuf::from("x.mid"),
            context: None,
            tags: vec![],
        };
        assert_eq!(default_pick_voice(Tile::All, Some(&song), &track), "guitar-stk");
    }

    #[test]
    fn rendered_audio_uses_no_voice_sentinel() {
        let track = Track {
            path: PathBuf::from("foo.wav"),
            label: "foo".into(),
            format: Format::Wav,
        };
        assert_eq!(default_pick_voice(Tile::All, None, &track), "—");
    }

    #[test]
    fn classical_tile_includes_baroque_classical_romantic_modern() {
        let mk = |era: Option<&str>, ins: &str| Song {
            id: "x".into(),
            title: "T".into(),
            composer: "C".into(),
            composer_key: "c".into(),
            era: era.map(String::from),
            instrument: ins.into(),
            license: "PD".into(),
            source: None,
            source_url: None,
            suggested_voice: None,
            midi_path: PathBuf::from("x.mid"),
            context: None,
            tags: vec![],
        };
        let t = Track {
            path: PathBuf::from("x.mid"),
            label: "x".into(),
            format: Format::Mid,
        };
        let favs = HashSet::new();
        let recent = HashSet::new();
        for era in ["Baroque", "Classical", "Romantic", "Modern"] {
            let s = mk(Some(era), "piano");
            assert!(track_matches_tile(
                Tile::Classical,
                &t,
                Some(&s),
                &favs,
                &recent
            ));
        }
        let s = mk(Some("Traditional"), "guitar");
        assert!(!track_matches_tile(
            Tile::Classical,
            &t,
            Some(&s),
            &favs,
            &recent
        ));
        assert!(track_matches_tile(Tile::Folk, &t, Some(&s), &favs, &recent));
    }

    #[test]
    fn game_tile_catches_metadata_less_tracks() {
        let t = Track {
            path: PathBuf::from("v01_chip.mid"),
            label: "v01_chip".into(),
            format: Format::Mid,
        };
        let favs = HashSet::new();
        let recent = HashSet::new();
        assert!(track_matches_tile(Tile::Game, &t, None, &favs, &recent));
    }

    #[test]
    fn scan_dirs_dedups_mp3_over_same_stem_wav() {
        let dir = std::env::temp_dir().join("keysynth_jukebox_core_dedup_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.wav"), b"RIFF").unwrap();
        std::fs::write(dir.join("a.mp3"), b"ID3").unwrap();
        std::fs::write(dir.join("b.wav"), b"RIFF").unwrap();
        let tracks = scan_dirs(&[dir.as_path()]);
        assert_eq!(tracks.len(), 2);
        let a = tracks.iter().find(|t| t.label == "a").unwrap();
        assert_eq!(a.format, Format::Mp3);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
