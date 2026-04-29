//! jukebox_lite — YouTube-style minimal jukebox (team/jukebox-lite-rebuild).
//!
//! Why this exists: the main `jukebox` bin grew to 4600+ lines of
//! per-row mixer / marker / personal-note / multi-tag controls and a
//! flat catalog list — VLM critique reads "high willingness" but the
//! lived experience says "わかりにくい、作り直した方がいい". Root cause is panel
//! overload + per-row noise + no mood/genre nav.
//!
//! Lite layout, modeled on YouTube:
//!   - top header: logo / search / ⓘ stats popover
//!   - left sidebar: mood/genre tiles (All / Favorites / Recent /
//!     Classical / Game / Folk) — collapsible nav
//!   - center: song-card grid (composer + era badge + 1-click play)
//!   - bottom (when playing): Now Playing bar (title + composer +
//!     progress + stop)
//!   - right (when playing): "same composer / same era" recommendations
//!
//! Deliberately NOT in lite: mixer panel, research panel, marker tags,
//! personal-note editor, per-row voice picker, per-row checkbox / x2
//! play-count column. Voice is picked automatically from mood:
//!   Classical → piano-modal, Game → square, Folk → guitar-stk.
//!
//! Backend reuse: keysynth::library_db / play_log / preview_cache /
//! gui_cp / ui::setup_japanese_fonts. Audio path is a single decoded
//! slot (no multi-track mixer) wired to one cpal stream.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

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

use keysynth::gui_cp;
use keysynth::library_db::{LibraryDb, Song, SongFilter, SongSort, VoiceFilter};
use keysynth::play_log::{PlayEntry, PlayLogDb};

// ---------------------------------------------------------------------------
// Source files
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Wav,
    Mp3,
    Mid,
}

impl Format {
    fn from_ext(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "wav" => Some(Format::Wav),
            "mp3" => Some(Format::Mp3),
            "mid" | "midi" => Some(Format::Mid),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
struct Track {
    path: PathBuf,
    /// Stable id (file stem) — joins to `Song::id` and `play_log` keys.
    label: String,
    format: Format,
}

fn scan_dirs(dirs: &[&Path]) -> Vec<Track> {
    let mut out: Vec<Track> = Vec::new();
    for d in dirs {
        let read = match std::fs::read_dir(d) {
            Ok(r) => r,
            Err(_) => continue,
        };
        // Per-dir dedup: prefer MP3 over same-stem WAV (matches the
        // main jukebox's transition strategy).
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

struct Mp3State {
    /// Source path retained for diagnostic logs. Not read by the
    /// decode loop; jukebox_lite never reopens (no in-loop seek).
    #[allow(dead_code)]
    path: PathBuf,
    format: Box<dyn FormatReader>,
    decoder: Box<dyn Decoder>,
    track_id: u32,
    scratch: Vec<f32>,
    scratch_pos: usize,
    eof: bool,
}

enum DecoderState {
    Wav {
        reader: WavReader<BufReader<File>>,
        sample_format: WavSampleFormat,
        bits_per_sample: u16,
    },
    Mp3(Mp3State),
}

struct DecodedTrack {
    state: DecoderState,
    sample_rate: u32,
    channels: u16,
    total_samples: u64,
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

fn open_decoded(path: &Path) -> Result<DecodedTrack, String> {
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
            // Rarer formats — skip the packet rather than panic; the
            // jukebox-lite catalogue is overwhelmingly MIDI→WAV via
            // preview_cache, so this branch is best-effort.
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
// Audio mixer (single slot, mono-stereo aware)
// ---------------------------------------------------------------------------

struct AudioState {
    decoded: Option<DecodedTrack>,
    is_playing: bool,
    cursor_frames: u64,
    /// File label currently loaded (`Track::label`). Empty when idle.
    label: String,
    volume: f32,
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

struct Mixer {
    state: Arc<Mutex<AudioState>>,
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

fn build_mixer_stream(state: Arc<Mutex<AudioState>>) -> Result<Mixer, String> {
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
    let err_fn = |err| eprintln!("jukebox_lite audio stream error: {err}");
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
    Ok(Mixer {
        state,
        _stream: stream,
    })
}

// ---------------------------------------------------------------------------
// Preview cache (MIDI → WAV via render_midi subprocess)
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

fn cache_key(track_path: &Path, voice_id: &str) -> keysynth::preview_cache::CacheKey {
    keysynth::preview_cache::CacheKey {
        song_path: track_path.to_path_buf(),
        voice_id: voice_id.to_string(),
        voice_dll: None,
        render_params: keysynth::preview_cache::RenderParams::default(),
    }
}

fn open_preview_cache() -> Result<keysynth::preview_cache::Cache, String> {
    let cache_dir = PathBuf::from("bench-out/cache");
    keysynth::preview_cache::Cache::new(&cache_dir, 1_073_741_824)
        .map_err(|e| format!("Cache::new: {e}"))
}

fn lookup_in_cache(track_path: &Path, voice_id: &str) -> Result<Option<PathBuf>, String> {
    let cache = open_preview_cache()?;
    let key = cache_key(track_path, voice_id);
    cache.lookup(&key).map_err(|e| format!("lookup: {e}"))
}

fn render_midi_blocking(midi_path: &Path, voice_id: &str) -> Result<PathBuf, String> {
    let cache = open_preview_cache()?;
    let key = cache_key(midi_path, voice_id);
    let bin = render_midi_binary_path()
        .ok_or_else(|| "render_midi binary not found alongside jukebox_lite".to_string())?;
    keysynth::preview_cache::render_to_cache(&cache, &key, voice_id, &bin)
        .map_err(|e| format!("render_to_cache: {e}"))
}

// ---------------------------------------------------------------------------
// Sidebar tile semantics
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tile {
    All,
    Favorites,
    Recent,
    Classical,
    Game,
    Folk,
}

impl Tile {
    fn label(self) -> &'static str {
        match self {
            Tile::All => "All",
            Tile::Favorites => "Favorites",
            Tile::Recent => "Recent",
            Tile::Classical => "Classical",
            Tile::Game => "Game",
            Tile::Folk => "Folk",
        }
    }
    fn icon(self) -> &'static str {
        match self {
            Tile::All => "\u{1F3B5}",     // 🎵
            Tile::Favorites => "\u{2B50}", // ⭐
            Tile::Recent => "\u{1F551}",   // 🕑
            Tile::Classical => "\u{1F3B9}", // 🎹
            Tile::Game => "\u{1F3AE}",      // 🎮
            Tile::Folk => "\u{1FA95}",      // 🪕
        }
    }
    fn all() -> &'static [Tile] {
        &[
            Tile::All,
            Tile::Favorites,
            Tile::Recent,
            Tile::Classical,
            Tile::Game,
            Tile::Folk,
        ]
    }
}

/// Pick the default voice for a track based on its mood.
/// Mood is derived first from the active sidebar tile (when the user
/// clicked into a genre, that's their explicit context), then from
/// per-track Song metadata, then from a stem heuristic.
fn pick_voice(tile: Tile, song: Option<&Song>, track: &Track) -> &'static str {
    if matches!(track.format, Format::Wav | Format::Mp3) {
        // Already-rendered audio doesn't need a voice; cached WAVs play
        // verbatim. Returning a sentinel keeps the snapshot/CP wire
        // honest.
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

/// Decide whether `track` belongs in `tile`. Tracks without DB
/// metadata fall through `Game` (catch-all for filename-only entries
/// like `cc0_*` chiptune dumps).
fn track_matches_tile(
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
// Visual helpers
// ---------------------------------------------------------------------------

fn era_color(era: &str) -> egui::Color32 {
    match era {
        "Baroque" => egui::Color32::from_rgb(200, 160, 100),
        "Classical" => egui::Color32::from_rgb(140, 220, 160),
        "Romantic" => egui::Color32::from_rgb(220, 160, 230),
        "Modern" => egui::Color32::from_rgb(120, 200, 240),
        "Traditional" => egui::Color32::from_rgb(240, 200, 120),
        _ => egui::Color32::from_rgb(170, 170, 170),
    }
}

fn compact_composer(composer: &str) -> String {
    let core = match composer.split_once('(') {
        Some((before, _)) => before,
        None => composer,
    };
    core.trim().to_string()
}

// ---------------------------------------------------------------------------
// Library + history bootstrap
// ---------------------------------------------------------------------------

fn load_library_index() -> (HashMap<String, Song>, usize) {
    let db_path = PathBuf::from("bench-out/library.db");
    let manifest = PathBuf::from("bench-out/songs/manifest.json");
    let voices_live = PathBuf::from("voices_live");
    let mut db = match LibraryDb::open(&db_path) {
        Ok(db) => db,
        Err(e) => {
            eprintln!(
                "jukebox_lite: library_db open failed ({}): {e} — running without DB metadata",
                db_path.display()
            );
            return (HashMap::new(), 0);
        }
    };
    if let Err(e) = db.migrate() {
        eprintln!("jukebox_lite: library_db migrate failed: {e}");
        return (HashMap::new(), 0);
    }
    if manifest.is_file() {
        if let Err(e) = db.import_songs(&manifest) {
            eprintln!("jukebox_lite: library_db import_songs failed: {e}");
        }
    }
    if voices_live.is_dir() {
        if let Err(e) = db.import_voices(&voices_live) {
            eprintln!("jukebox_lite: library_db import_voices failed: {e}");
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

fn load_play_log() -> Option<PlayLogDb> {
    let db_path = PathBuf::from("bench-out/play_log.db");
    let mut db = match PlayLogDb::open(&db_path) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("jukebox_lite: play_log open failed: {e}");
            return None;
        }
    };
    if let Err(e) = db.migrate() {
        eprintln!("jukebox_lite: play_log migrate failed: {e}");
        return None;
    }
    Some(db)
}

const RECENT_WINDOW: usize = 24;

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
// CP server snapshot/command types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, Serialize)]
struct CpSnapshot {
    frame_id: u64,
    track_count: usize,
    db_song_count: usize,
    db_voice_count: usize,
    selected_label: Option<String>,
    selected_tile: String,
    any_playing: bool,
    /// `Track::label` of whatever the user has loaded into the slot.
    loaded_label: Option<String>,
    cursor_frames: u64,
    total_frames: u64,
    sample_rate: u32,
    /// Catalog rows visible after current tile + filter, in display order.
    visible_labels: Vec<String>,
    /// Every catalog row label, regardless of filter — verifiers use
    /// this to pick a known label without having to mirror filter
    /// semantics.
    all_labels: Vec<String>,
}

#[derive(Clone, Debug)]
enum CpCommand {
    LoadTrack { label: String },
    Play,
    Stop,
    SelectTile { tile: String },
    Rescan,
}

fn spawn_cp_server(
    state: gui_cp::State<CpSnapshot, CpCommand>,
) -> std::io::Result<gui_cp::Handle> {
    let endpoint = gui_cp::resolve_endpoint("jukebox_lite", None);
    let st_get = state.clone();
    let st_load = state.clone();
    let st_play = state.clone();
    let st_stop = state.clone();
    let st_tile = state.clone();
    let st_rescan = state.clone();
    let st_list = state.clone();
    gui_cp::Builder::new("jukebox_lite", &endpoint)
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
            Ok(json!({"queued": "load_track", "label": a.label}))
        })
        .register("play", move |_p| {
            st_play.push_command(CpCommand::Play);
            Ok(json!({"queued": "play"}))
        })
        .register("stop", move |_p| {
            st_stop.push_command(CpCommand::Stop);
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
            Ok(json!({"queued": "select_tile", "tile": a.tile}))
        })
        .register("rescan", move |_p| {
            st_rescan.push_command(CpCommand::Rescan);
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
        })
        .serve()
}

fn tile_from_str(s: &str) -> Option<Tile> {
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

// ---------------------------------------------------------------------------
// Pending background renders
// ---------------------------------------------------------------------------

struct PendingRender {
    track_label: String,
    voice_id: String,
    started: Instant,
    rx: std::sync::mpsc::Receiver<Result<PathBuf, String>>,
    _handle: std::thread::JoinHandle<()>,
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

struct JukeboxLite {
    tracks: Vec<Track>,
    song_index: HashMap<String, Song>,
    db_voice_count: usize,
    play_log: Option<PlayLogDb>,
    favorites: HashSet<String>,
    recent_ids: HashSet<String>,
    recent_entries: Vec<PlayEntry>,

    mixer: Mixer,
    sample_rate_for_progress: u32,
    pending_render: Option<PendingRender>,

    /// Currently active tile in the left sidebar.
    tile: Tile,
    /// Substring filter from the header search input.
    search: String,
    /// Track::label currently loaded (or last-loaded) — drives the
    /// selected-row glyph and recommendations panel.
    selected_label: Option<String>,
    /// Show the ⓘ stats popover next to the search box.
    show_stats: bool,

    refresh_dirs: Vec<PathBuf>,

    cp_state: gui_cp::State<CpSnapshot, CpCommand>,
    _cp_handle: Option<gui_cp::Handle>,
    cp_frame_id: u64,
}

impl JukeboxLite {
    fn new(dirs: Vec<PathBuf>) -> Result<Self, String> {
        let (song_index, db_voice_count) = load_library_index();
        let dir_refs: Vec<&Path> = dirs.iter().map(|p| p.as_path()).collect();
        let tracks = scan_dirs(&dir_refs);
        println!(
            "jukebox_lite: library_db catalog \u{2192} {} songs / {} voices ({} files on disk)",
            song_index.len(),
            db_voice_count,
            tracks.len(),
        );
        let play_log = load_play_log();
        let (favorites, recent_ids, recent_entries) = snapshot_history(play_log.as_ref());

        let audio = Arc::new(Mutex::new(AudioState::new()));
        let mixer = build_mixer_stream(audio).map_err(|e| format!("audio: {e}"))?;
        let sample_rate_for_progress = 44_100;

        let cp_state: gui_cp::State<CpSnapshot, CpCommand> = gui_cp::State::new();
        let cp_handle = match spawn_cp_server(cp_state.clone()) {
            Ok(h) => Some(h),
            Err(e) => {
                eprintln!("jukebox_lite: CP server failed: {e}");
                None
            }
        };

        let initial_label = recent_entries
            .first()
            .map(|e| e.song_id.clone())
            .or_else(|| favorites.iter().next().cloned())
            .or_else(|| tracks.first().map(|t| t.label.clone()));

        Ok(Self {
            tracks,
            song_index,
            db_voice_count,
            play_log,
            favorites,
            recent_ids,
            recent_entries,
            mixer,
            sample_rate_for_progress,
            pending_render: None,
            tile: Tile::All,
            search: String::new(),
            selected_label: initial_label,
            show_stats: false,
            refresh_dirs: dirs,
            cp_state,
            _cp_handle: cp_handle,
            cp_frame_id: 0,
        })
    }

    fn rescan(&mut self) {
        let dir_refs: Vec<&Path> = self.refresh_dirs.iter().map(|p| p.as_path()).collect();
        self.tracks = scan_dirs(&dir_refs);
        let (idx, vc) = load_library_index();
        self.song_index = idx;
        self.db_voice_count = vc;
        let (f, r, re) = snapshot_history(self.play_log.as_ref());
        self.favorites = f;
        self.recent_ids = r;
        self.recent_entries = re;
    }

    fn refresh_history(&mut self) {
        let (f, r, re) = snapshot_history(self.play_log.as_ref());
        self.favorites = f;
        self.recent_ids = r;
        self.recent_entries = re;
    }

    /// Compute the visible track set after applying current tile +
    /// search. Returns clones so the GUI can borrow `self` for
    /// playback during iteration.
    fn visible_tracks(&self) -> Vec<Track> {
        let needle = self.search.trim().to_ascii_lowercase();
        let mut out: Vec<Track> = self
            .tracks
            .iter()
            .filter(|t| {
                let song = self.song_index.get(&t.label);
                if !track_matches_tile(self.tile, t, song, &self.favorites, &self.recent_ids) {
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
        // For Recent, sort by most-recent-first using recent_entries
        // order; everything else stays in the catalog (composer) order
        // already produced by `scan_dirs` + the DB index.
        if self.tile == Tile::Recent {
            let order: HashMap<&str, usize> = self
                .recent_entries
                .iter()
                .enumerate()
                .map(|(i, e)| (e.song_id.as_str(), i))
                .collect();
            out.sort_by_key(|t| order.get(t.label.as_str()).copied().unwrap_or(usize::MAX));
        } else if !self.song_index.is_empty() {
            out.sort_by(|a, b| {
                let ka = self.song_index.get(&a.label).map(|s| s.composer_key.clone());
                let kb = self.song_index.get(&b.label).map(|s| s.composer_key.clone());
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

    fn play_track(&mut self, track: Track) {
        let song = self.song_index.get(&track.label).cloned();
        let voice = pick_voice(self.tile, song.as_ref(), &track).to_string();
        self.selected_label = Some(track.label.clone());

        let playable: PathBuf = match track.format {
            Format::Mid => match lookup_in_cache(&track.path, &voice) {
                Ok(Some(p)) => p,
                Ok(None) => {
                    self.spawn_render(&track, &voice);
                    return;
                }
                Err(e) => {
                    eprintln!("jukebox_lite: cache lookup {}: {e}", track.path.display());
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
        let handle = std::thread::Builder::new()
            .name("jukebox-lite-render".into())
            .spawn(move || {
                let result = render_midi_blocking(&path, &voice_for_thread);
                let _ = tx.send(result);
            })
            .expect("spawn render thread");
        eprintln!(
            "jukebox_lite: render queued ({}) — voice={}",
            track.path.display(),
            voice
        );
        self.pending_render = Some(PendingRender {
            track_label: label,
            voice_id: voice.to_string(),
            started: Instant::now(),
            rx,
            _handle: handle,
        });
    }

    fn poll_pending_render(&mut self) {
        let pending = match self.pending_render.as_ref() {
            Some(p) => p,
            None => return,
        };
        let result = match pending.rx.try_recv() {
            Ok(r) => r,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.pending_render = None;
                return;
            }
        };
        let pending = self.pending_render.take().expect("guarded");
        let elapsed = pending.started.elapsed().as_millis();
        match result {
            Ok(wav) => {
                eprintln!(
                    "jukebox_lite: render done ({}) in {} ms",
                    wav.display(),
                    elapsed
                );
                self.load_resolved(pending.track_label, wav, &pending.voice_id);
            }
            Err(e) => {
                eprintln!("jukebox_lite: render failed: {e}");
            }
        }
    }

    fn load_resolved(&mut self, label: String, playable: PathBuf, voice: &str) {
        let dec = match open_decoded(&playable) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("jukebox_lite: open {}: {e}", playable.display());
                return;
            }
        };
        let sr = dec.sample_rate;
        {
            let mut st = match self.mixer.state.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            st.decoded = Some(dec);
            st.is_playing = true;
            st.cursor_frames = 0;
            st.label = label.clone();
        }
        self.sample_rate_for_progress = sr;
        self.selected_label = Some(label.clone());

        // play_log: voice column carries the mood-derived voice id so
        // history reflects what was actually rendered/heard.
        if let Some(db) = self.play_log.as_mut() {
            let v = if voice == "—" { None } else { Some(voice) };
            if let Err(e) = db.record_play(&label, v, None) {
                eprintln!("jukebox_lite: record_play({label}): {e}");
            }
        }
        self.refresh_history();
    }

    fn stop(&mut self) {
        if let Ok(mut st) = self.mixer.state.lock() {
            st.is_playing = false;
        }
    }

    fn drain_cp_commands(&mut self) {
        for cmd in self.cp_state.drain_commands() {
            match cmd {
                CpCommand::LoadTrack { label } => {
                    if let Some(t) = self.tracks.iter().find(|t| t.label == label).cloned() {
                        self.play_track(t);
                    } else {
                        eprintln!("jukebox_lite: CP load_track unknown label={label}");
                    }
                }
                CpCommand::Play => {
                    if let Ok(mut st) = self.mixer.state.lock() {
                        if st.decoded.is_some() {
                            st.is_playing = true;
                        }
                    }
                }
                CpCommand::Stop => self.stop(),
                CpCommand::SelectTile { tile } => {
                    if let Some(t) = tile_from_str(&tile) {
                        self.tile = t;
                    }
                }
                CpCommand::Rescan => self.rescan(),
            }
        }
    }

    fn publish_cp_snapshot(&mut self, visible: &[Track]) {
        self.cp_frame_id = self.cp_frame_id.saturating_add(1);
        let (any_playing, loaded_label, cursor_frames, total_frames) = {
            let st = self.mixer.state.lock().ok();
            match st {
                Some(g) => (
                    g.is_playing,
                    if g.label.is_empty() {
                        None
                    } else {
                        Some(g.label.clone())
                    },
                    g.cursor_frames,
                    g.decoded
                        .as_ref()
                        .map(|d| d.total_samples / d.channels.max(1) as u64)
                        .unwrap_or(0),
                ),
                None => (false, None, 0, 0),
            }
        };
        let snap = CpSnapshot {
            frame_id: self.cp_frame_id,
            track_count: self.tracks.len(),
            db_song_count: self.song_index.len(),
            db_voice_count: self.db_voice_count,
            selected_label: self.selected_label.clone(),
            selected_tile: self.tile.label().to_string(),
            any_playing,
            loaded_label,
            cursor_frames,
            total_frames,
            sample_rate: self.sample_rate_for_progress,
            visible_labels: visible.iter().map(|t| t.label.clone()).collect(),
            all_labels: self.tracks.iter().map(|t| t.label.clone()).collect(),
        };
        self.cp_state.publish(snap);
    }
}

// ---------------------------------------------------------------------------
// UI rendering
// ---------------------------------------------------------------------------

const SIDEBAR_WIDTH: f32 = 200.0;
const CARD_MIN_W: f32 = 220.0;
const CARD_HEIGHT: f32 = 96.0;

impl eframe::App for JukeboxLite {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain any queued CP commands and pending background renders
        // first so the UI reflects them in the same paint.
        self.drain_cp_commands();
        self.poll_pending_render();

        let visible = self.visible_tracks();

        // Top header
        egui::TopBottomPanel::top("jukebox_lite_header")
            .exact_height(54.0)
            .show(ctx, |ui| {
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.add_space(8.0);
                    ui.heading("\u{1F3B5}  jukebox lite");
                    ui.add_space(20.0);
                    ui.label("\u{1F50D}");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.search)
                            .desired_width(280.0)
                            .hint_text("Search composer / title…"),
                    );
                    ui.add_space(8.0);
                    if ui.button("\u{2716} clear").clicked() {
                        self.search.clear();
                    }
                    // Push the stats button to the far right.
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(8.0);
                        let stats = format!(
                            "\u{2139} {} songs · {} voices",
                            self.song_index.len(),
                            self.db_voice_count
                        );
                        let resp = ui.button(stats);
                        if resp.clicked() {
                            self.show_stats = !self.show_stats;
                        }
                        let popup_id = ui.make_persistent_id("jukebox_lite_stats_popup");
                        if self.show_stats {
                            egui::popup::popup_below_widget(
                                ui,
                                popup_id,
                                &resp,
                                egui::PopupCloseBehavior::CloseOnClick,
                                |ui| {
                                    ui.set_min_width(280.0);
                                    ui.heading("Library");
                                    ui.label(format!(
                                        "On-disk catalog rows: {}",
                                        self.tracks.len()
                                    ));
                                    ui.label(format!("DB songs: {}", self.song_index.len()));
                                    ui.label(format!("DB voices: {}", self.db_voice_count));
                                    ui.label(format!(
                                        "Favorites: {}",
                                        self.favorites.len()
                                    ));
                                    ui.label(format!(
                                        "Recent plays (window): {}",
                                        self.recent_entries.len()
                                    ));
                                    ui.separator();
                                    if ui.button("Rescan files").clicked() {
                                        self.rescan();
                                    }
                                },
                            );
                            // Auto-close when a click happens outside.
                            ctx.memory_mut(|m| m.open_popup(popup_id));
                        }
                    });
                });
            });

        // Bottom Now Playing bar (always present so the layout doesn't
        // jump when audio starts; idle state shows a placeholder line).
        egui::TopBottomPanel::bottom("jukebox_lite_now_playing")
            .exact_height(72.0)
            .show(ctx, |ui| {
                ui.add_space(6.0);
                self.show_now_playing(ui);
            });

        // Right recommendations panel — visible only while audio is
        // actually rolling. Idle state stays clean (matches the
        // YouTube reference: rec rail appears once you start watching,
        // not while you're still browsing).
        let any_playing_now = self
            .mixer
            .state
            .lock()
            .map(|s| s.is_playing)
            .unwrap_or(false);
        let show_recommend = any_playing_now
            && self
                .selected_label
                .as_ref()
                .map(|l| self.song_index.contains_key(l))
                .unwrap_or(false);
        if show_recommend {
            egui::SidePanel::right("jukebox_lite_recommend")
                .resizable(false)
                .exact_width(220.0)
                .show(ctx, |ui| {
                    self.show_recommendations(ui);
                });
        }

        // Left sidebar (tiles).
        egui::SidePanel::left("jukebox_lite_sidebar")
            .resizable(false)
            .exact_width(SIDEBAR_WIDTH)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                ui.heading("Library");
                ui.add_space(6.0);
                for &tile in Tile::all() {
                    let count = self
                        .tracks
                        .iter()
                        .filter(|t| {
                            track_matches_tile(
                                tile,
                                t,
                                self.song_index.get(&t.label),
                                &self.favorites,
                                &self.recent_ids,
                            )
                        })
                        .count();
                    let selected = self.tile == tile;
                    let label = format!("{}  {}   ({})", tile.icon(), tile.label(), count);
                    let btn = egui::SelectableLabel::new(selected, label);
                    if ui.add_sized([SIDEBAR_WIDTH - 18.0, 32.0], btn).clicked() {
                        self.tile = tile;
                    }
                }
                ui.add_space(12.0);
                ui.separator();
                ui.label(
                    egui::RichText::new("Voice picks per tile:")
                        .small()
                        .color(egui::Color32::GRAY),
                );
                ui.label(egui::RichText::new("Classical → piano-modal").small());
                ui.label(egui::RichText::new("Game → square").small());
                ui.label(egui::RichText::new("Folk → guitar-stk").small());
            });

        // Center: card grid.
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.heading(self.tile.label());
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new(format!("{} results", visible.len()))
                        .color(egui::Color32::GRAY),
                );
                if let Some(p) = self.pending_render.as_ref() {
                    ui.add_space(20.0);
                    ui.spinner();
                    ui.label(
                        egui::RichText::new(format!(
                            "rendering {}…",
                            p.track_label
                        ))
                        .color(egui::Color32::LIGHT_BLUE),
                    );
                }
            });
            ui.separator();
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    self.draw_card_grid(ui, &visible);
                });
        });

        // Tail: publish CP snapshot for verifiers.
        self.publish_cp_snapshot(&visible);

        // Repaint while audio is rolling so the progress bar advances.
        let any_playing = self
            .mixer
            .state
            .lock()
            .map(|s| s.is_playing)
            .unwrap_or(false);
        if any_playing || self.pending_render.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(120));
        }
    }
}

impl JukeboxLite {
    fn draw_card_grid(&mut self, ui: &mut egui::Ui, visible: &[Track]) {
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
        let avail = ui.available_width();
        let cols = ((avail / CARD_MIN_W).floor() as usize).max(1);
        let card_w = (avail / cols as f32) - 12.0;
        let row_count = (visible.len() + cols - 1) / cols;
        let mut to_play: Option<Track> = None;
        for r in 0..row_count {
            ui.horizontal(|ui| {
                for c in 0..cols {
                    let i = r * cols + c;
                    if i >= visible.len() {
                        ui.add_space(card_w + 6.0);
                        continue;
                    }
                    let t = &visible[i];
                    let song = self.song_index.get(&t.label).cloned();
                    let is_selected = self
                        .selected_label
                        .as_ref()
                        .map(|l| l == &t.label)
                        .unwrap_or(false);
                    let is_fav = self.favorites.contains(&t.label);
                    if Self::draw_card(ui, card_w, t, song.as_ref(), is_selected, is_fav) {
                        to_play = Some(t.clone());
                    }
                    ui.add_space(6.0);
                }
            });
            ui.add_space(6.0);
        }
        if let Some(t) = to_play {
            self.play_track(t);
        }
    }

    /// Draws one card, returns true when the user clicked play.
    fn draw_card(
        ui: &mut egui::Ui,
        width: f32,
        track: &Track,
        song: Option<&Song>,
        selected: bool,
        fav: bool,
    ) -> bool {
        // Dark cards on the (light) panel — gives a YouTube-light
        // look where the user's eye lands on the song tiles, not on
        // the chrome. Selected card swaps to a brighter blue so a
        // verifier (and user) can see at a glance which row is loaded.
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
                // Top row: era badge + license + ★
                ui.horizontal(|ui| {
                    if let Some(s) = song {
                        if let Some(era) = s.era.as_deref() {
                            let badge = egui::RichText::new(format!(" {} ", era))
                                .background_color(era_color(era))
                                .color(egui::Color32::BLACK)
                                .small();
                            ui.label(badge);
                        }
                        ui.label(
                            egui::RichText::new(format!("· {}", s.instrument))
                                .small()
                                .color(egui::Color32::LIGHT_GRAY),
                        );
                    } else {
                        ui.label(
                            egui::RichText::new(format!("[{:?}]", track.format))
                                .small()
                                .color(egui::Color32::DARK_GRAY),
                        );
                    }
                    if fav {
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                ui.label(
                                    egui::RichText::new("\u{2605}")
                                        .color(egui::Color32::from_rgb(255, 200, 80)),
                                );
                            },
                        );
                    }
                });
                // Title (1 strong line)
                let title = song.map(|s| s.title.as_str()).unwrap_or(&track.label);
                ui.label(egui::RichText::new(title).strong().size(15.0));
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
                    let play = egui::Button::new(
                        egui::RichText::new("\u{25B6} play").size(14.0),
                    )
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

    fn show_now_playing(&mut self, ui: &mut egui::Ui) {
        let (label, is_playing, cursor, total) = {
            let st = self.mixer.state.lock().ok();
            match st {
                Some(g) => (
                    if g.label.is_empty() {
                        None
                    } else {
                        Some(g.label.clone())
                    },
                    g.is_playing,
                    g.cursor_frames,
                    g.decoded
                        .as_ref()
                        .map(|d| d.total_samples / d.channels.max(1) as u64)
                        .unwrap_or(0),
                ),
                None => (None, false, 0, 0),
            }
        };
        let song = label.as_ref().and_then(|l| self.song_index.get(l)).cloned();
        ui.horizontal(|ui| {
            ui.add_space(8.0);
            // Big play state icon
            let icon = if is_playing {
                "\u{25B6}"
            } else if label.is_some() {
                "\u{23F8}"
            } else {
                "\u{1F3B5}"
            };
            ui.label(egui::RichText::new(icon).size(22.0));
            ui.add_space(8.0);
            ui.vertical(|ui| {
                let title_text = match (label.as_ref(), song.as_ref()) {
                    (Some(_), Some(s)) => s.title.clone(),
                    (Some(l), None) => l.clone(),
                    (None, _) => "Nothing playing".to_string(),
                };
                ui.label(egui::RichText::new(title_text).strong().size(15.0));
                let sub = match song.as_ref() {
                    Some(s) => {
                        let mut line = compact_composer(&s.composer);
                        if let Some(era) = s.era.as_deref() {
                            line.push_str(" · ");
                            line.push_str(era);
                        }
                        line
                    }
                    None => label.clone().unwrap_or_else(|| {
                        "Pick a card on the right to start listening.".to_string()
                    }),
                };
                ui.label(
                    egui::RichText::new(sub)
                        .color(egui::Color32::LIGHT_GRAY)
                        .small(),
                );
            });
            // Progress bar fills the middle.
            ui.add_space(16.0);
            let progress = if total > 0 {
                (cursor as f32 / total as f32).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let bar = egui::ProgressBar::new(progress)
                .desired_width(ui.available_width() - 90.0)
                .text(if total > 0 {
                    let secs = cursor as u32 / self.sample_rate_for_progress.max(1);
                    let total_secs = total as u32 / self.sample_rate_for_progress.max(1);
                    format!(
                        "{:01}:{:02} / {:01}:{:02}",
                        secs / 60,
                        secs % 60,
                        total_secs / 60,
                        total_secs % 60
                    )
                } else {
                    "—:—".to_string()
                });
            ui.add(bar);
            // Stop on the right edge.
            ui.add_space(6.0);
            let stop = egui::Button::new(egui::RichText::new("\u{25A0} stop").size(14.0))
                .fill(egui::Color32::from_rgb(170, 80, 80));
            if ui.add(stop).clicked() {
                self.stop();
            }
            ui.add_space(8.0);
        });
    }

    fn show_recommendations(&mut self, ui: &mut egui::Ui) {
        ui.add_space(8.0);
        ui.heading("Up next");
        let label = match self.selected_label.clone() {
            Some(l) => l,
            None => return,
        };
        let song = match self.song_index.get(&label).cloned() {
            Some(s) => s,
            None => return,
        };
        ui.add_space(6.0);
        ui.label(
            egui::RichText::new(format!("Same composer · {}", compact_composer(&song.composer)))
                .small()
                .color(egui::Color32::LIGHT_GRAY),
        );
        let same_composer: Vec<Track> = self
            .tracks
            .iter()
            .filter(|t| {
                if t.label == label {
                    return false;
                }
                self.song_index
                    .get(&t.label)
                    .map(|s| s.composer_key == song.composer_key)
                    .unwrap_or(false)
            })
            .take(3)
            .cloned()
            .collect();
        if same_composer.is_empty() {
            ui.label(egui::RichText::new("(no other works in catalog)").small());
        } else {
            let mut to_play: Option<Track> = None;
            for t in &same_composer {
                let title = self
                    .song_index
                    .get(&t.label)
                    .map(|s| s.title.clone())
                    .unwrap_or_else(|| t.label.clone());
                if ui.button(format!("\u{25B6} {title}")).clicked() {
                    to_play = Some(t.clone());
                }
            }
            if let Some(t) = to_play {
                self.play_track(t);
            }
        }
        ui.add_space(10.0);
        if let Some(era) = song.era.as_deref() {
            ui.label(
                egui::RichText::new(format!("Same era · {era}"))
                    .small()
                    .color(egui::Color32::LIGHT_GRAY),
            );
            let same_era: Vec<Track> = self
                .tracks
                .iter()
                .filter(|t| {
                    if t.label == label {
                        return false;
                    }
                    self.song_index
                        .get(&t.label)
                        .and_then(|s| s.era.as_deref())
                        .map(|e| e == era)
                        .unwrap_or(false)
                })
                .filter(|t| {
                    self.song_index
                        .get(&t.label)
                        .map(|s| s.composer_key != song.composer_key)
                        .unwrap_or(true)
                })
                .take(3)
                .cloned()
                .collect();
            if same_era.is_empty() {
                ui.label(egui::RichText::new("(only this composer)").small());
            } else {
                let mut to_play: Option<Track> = None;
                for t in &same_era {
                    let title = self
                        .song_index
                        .get(&t.label)
                        .map(|s| s.title.clone())
                        .unwrap_or_else(|| t.label.clone());
                    if ui.button(format!("\u{25B6} {title}")).clicked() {
                        to_play = Some(t.clone());
                    }
                }
                if let Some(t) = to_play {
                    self.play_track(t);
                }
            }
        }
    }
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
    let app = JukeboxLite::new(dirs)?;
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 760.0])
            .with_title("keysynth jukebox lite"),
        ..Default::default()
    };
    eframe::run_native(
        "keysynth jukebox lite",
        options,
        Box::new(|cc| {
            keysynth::ui::setup_japanese_fonts(&cc.egui_ctx);
            // Lock to light theme. We deliberately want the
            // "dark-cards-on-light-chrome" YouTube-light feel: the
            // dark card surfaces give the user's eye a clear focal
            // target while the chrome (header / sidebar / nowplaying
            // bar) stays out of the way. egui's dark theme was tried
            // first but its panel bg (#303030) is too close to the
            // chosen card bg (#24242a) — the gui_audit primary-
            // contrast pair both ended up as near-identical dark
            // greys which trips the WCAG blocker.
            cc.egui_ctx.set_theme(egui::ThemePreference::Light);
            Ok(Box::new(app))
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
            assert_eq!(tile_from_str(t.label()), Some(t));
        }
    }

    #[test]
    fn pick_voice_uses_active_tile_first() {
        let track = Track {
            path: PathBuf::from("dummy.mid"),
            label: "dummy".into(),
            format: Format::Mid,
        };
        // Tile-driven mood overrides per-track metadata.
        assert_eq!(pick_voice(Tile::Classical, None, &track), "piano-modal");
        assert_eq!(pick_voice(Tile::Game, None, &track), "square");
        assert_eq!(pick_voice(Tile::Folk, None, &track), "guitar-stk");
    }

    #[test]
    fn pick_voice_falls_back_to_song_metadata() {
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
        // All-tile defers to instrument: guitar wins.
        assert_eq!(pick_voice(Tile::All, Some(&song), &track), "guitar-stk");
    }

    #[test]
    fn rendered_audio_uses_no_voice_sentinel() {
        let track = Track {
            path: PathBuf::from("foo.wav"),
            label: "foo".into(),
            format: Format::Wav,
        };
        assert_eq!(pick_voice(Tile::All, None, &track), "—");
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
        let dir = std::env::temp_dir().join("keysynth_jukebox_lite_dedup_test");
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
