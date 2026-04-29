//! jukebox_lite_g — Gemini version of the music browser.
//!
//! A sibling to `jukebox_lite` with a custom list-style UI.
//! Reuses the backend from `jukebox_lite.rs` but implements a denser,
//! table-focused interface inspired by foobar2000 and Spotify.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

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
use keysynth::library_db::{LibraryDb, Song, SongFilter, VoiceFilter};
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
// Decoder (reused from jukebox_lite)
// ---------------------------------------------------------------------------

struct Mp3State {
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
// Audio mixer (single slot)
// ---------------------------------------------------------------------------

struct AudioState {
    decoded: Option<DecodedTrack>,
    is_playing: bool,
    cursor_frames: u64,
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
    let err_fn = |err| eprintln!("jukebox_lite_g audio stream error: {err}");
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
// Preview cache
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
        .ok_or_else(|| "render_midi binary not found alongside jukebox_lite_g".to_string())?;
    keysynth::preview_cache::render_to_cache(&cache, &key, voice_id, &bin)
        .map_err(|e| format!("render_to_cache: {e}"))
}

// ---------------------------------------------------------------------------
// Sidebar / Filter logic
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
            Tile::All => "All Songs",
            Tile::Favorites => "Favorites",
            Tile::Recent => "Recently Played",
            Tile::Classical => "Classical",
            Tile::Game => "Game / Synth",
            Tile::Folk => "Folk / Guitar",
        }
    }
    fn icon(self) -> &'static str {
        match self {
            Tile::All => "\u{2630}",
            Tile::Favorites => "\u{2605}",
            Tile::Recent => "\u{21BB}",
            Tile::Classical => "\u{1F3B9}",
            Tile::Game => "\u{1F3AE}",
            Tile::Folk => "\u{1F3B8}",
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

fn pick_voice(tile: Tile, song: Option<&Song>, track: &Track) -> &'static str {
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
    }
    let stem = track.label.to_ascii_lowercase();
    if stem.starts_with("cc0_") || stem.contains("chiptune") {
        return "square";
    }
    "piano-modal"
}

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
                return track.label.to_ascii_lowercase().contains("cc0");
            }
            let s = song.unwrap();
            let ins = s.instrument.to_ascii_lowercase();
            ins.contains("synth") || ins.contains("game") || ins.contains("chip")
        }
    }
}

// ---------------------------------------------------------------------------
// CP types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, Serialize)]
struct CpSnapshot {
    frame_id: u64,
    any_playing: bool,
    loaded_label: Option<String>,
    cursor_frames: u64,
    total_frames: u64,
    visible_labels: Vec<String>,
    track_count: usize,
    db_song_count: usize,
    db_voice_count: usize,
}

#[derive(Clone, Debug, serde::Deserialize)]
enum CpCommand {
    LoadTrack { label: String },
    Play,
    Stop,
    SelectTile { tile: String },
    Rescan,
}

// ---------------------------------------------------------------------------
// App State
// ---------------------------------------------------------------------------

struct PendingRender {
    track_label: String,
    voice_id: String,
    rx: std::sync::mpsc::Receiver<Result<PathBuf, String>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SortColumn {
    Title,
    Composer,
    Era,
    Format,
}

struct JukeboxLiteG {
    tracks: Vec<Track>,
    song_index: HashMap<String, Song>,
    db_voice_count: usize,
    play_log: Option<PlayLogDb>,
    favorites: HashSet<String>,
    recent_ids: HashSet<String>,
    recent_entries: Vec<PlayEntry>,

    mixer: Mixer,
    sample_rate: u32,
    pending_render: Option<PendingRender>,

    tile: Tile,
    search: String,
    selected_label: Option<String>,

    sort_col: SortColumn,
    sort_asc: bool,

    cp_state: gui_cp::State<CpSnapshot, CpCommand>,
    _cp_handle: Option<gui_cp::Handle>,
    cp_frame_id: u64,
}

impl JukeboxLiteG {
    fn new() -> Result<Self, String> {
        let db_path = PathBuf::from("bench-out/library.db");
        let mut db = LibraryDb::open(&db_path).map_err(|e| e.to_string())?;
        let _ = db.migrate();
        let songs = db.query_songs(&SongFilter::default()).unwrap_or_default();
        let voice_count = db.query_voices(&VoiceFilter::default()).map(|v| v.len()).unwrap_or(0);
        let mut song_index = HashMap::new();
        for s in songs {
            song_index.insert(s.id.clone(), s);
        }

        let dirs = vec![
            PathBuf::from("bench-out/songs"),
            PathBuf::from("bench-out/CHIPTUNE"),
        ];
        let dir_refs: Vec<&Path> = dirs.iter().map(|p| p.as_path()).collect();
        let tracks = scan_dirs(&dir_refs);

        let play_log = PlayLogDb::open(&PathBuf::from("bench-out/play_log.db")).ok();
        let mut favs = HashSet::new();
        let mut recent_ids = HashSet::new();
        let mut recent_entries = Vec::new();
        if let Some(ref db) = play_log {
            if let Ok(f) = db.favorites() {
                favs = f.into_iter().collect();
            }
            if let Ok(r) = db.recent_plays(100) {
                recent_ids = r.iter().map(|e| e.song_id.clone()).collect();
                recent_entries = r;
            }
        }

        let audio = Arc::new(Mutex::new(AudioState::new()));
        let mixer = build_mixer_stream(audio).map_err(|e| format!("audio: {e}"))?;

        let cp_state: gui_cp::State<CpSnapshot, CpCommand> = gui_cp::State::new();
        let st_get = cp_state.clone();
        let st_load = cp_state.clone();
        let st_play = cp_state.clone();
        let st_stop = cp_state.clone();
        let st_tile = cp_state.clone();
        let st_rescan = cp_state.clone();
        
        let endpoint = gui_cp::resolve_endpoint("jukebox_lite_g", None);
        let cp_handle = gui_cp::Builder::new("jukebox_lite_g", &endpoint)
            .register("get_state", move |_p| match st_get.read_snapshot() {
                Some(snap) => Ok(json!(snap)),
                None => Ok(json!({"warming_up": true})),
            })
            .register("load_track", move |p| {
                #[derive(serde::Deserialize)]
                struct Args { label: String }
                let a: Args = serde_json::from_value(p).map_err(|e| cp_core::RpcError::invalid_params(e.to_string()))?;
                st_load.push_command(CpCommand::LoadTrack { label: a.label });
                Ok(json!({"status": "ok"}))
            })
            .register("play", move |_p| {
                st_play.push_command(CpCommand::Play);
                Ok(json!({"status": "ok"}))
            })
            .register("stop", move |_p| {
                st_stop.push_command(CpCommand::Stop);
                Ok(json!({"status": "ok"}))
            })
            .register("select_tile", move |p| {
                #[derive(serde::Deserialize)]
                struct Args { tile: String }
                let a: Args = serde_json::from_value(p).map_err(|e| cp_core::RpcError::invalid_params(e.to_string()))?;
                st_tile.push_command(CpCommand::SelectTile { tile: a.tile });
                Ok(json!({"status": "ok"}))
            })
            .register("rescan", move |_p| {
                st_rescan.push_command(CpCommand::Rescan);
                Ok(json!({"status": "ok"}))
            })
            .serve()
            .ok();

        Ok(Self {
            tracks,
            song_index,
            db_voice_count: voice_count,
            play_log,
            favorites: favs,
            recent_ids,
            recent_entries,
            mixer,
            sample_rate: 44100,
            pending_render: None,
            tile: Tile::All,
            search: String::new(),
            selected_label: None,
            sort_col: SortColumn::Composer,
            sort_asc: true,
            cp_state,
            _cp_handle: cp_handle,
            cp_frame_id: 0,
        })
    }

    fn rescan(&mut self) {
        let db_path = PathBuf::from("bench-out/library.db");
        if let Ok(mut db) = LibraryDb::open(&db_path) {
            let _ = db.migrate();
            let songs = db.query_songs(&SongFilter::default()).unwrap_or_default();
            self.db_voice_count = db.query_voices(&VoiceFilter::default()).map(|v| v.len()).unwrap_or(0);
            self.song_index.clear();
            for s in songs {
                self.song_index.insert(s.id.clone(), s);
            }
        }

        let dirs = vec![
            PathBuf::from("bench-out/songs"),
            PathBuf::from("bench-out/CHIPTUNE"),
        ];
        let dir_refs: Vec<&Path> = dirs.iter().map(|p| p.as_path()).collect();
        self.tracks = scan_dirs(&dir_refs);

        if let Some(ref db) = self.play_log {
            if let Ok(f) = db.favorites() {
                self.favorites = f.into_iter().collect();
            }
            if let Ok(r) = db.recent_plays(100) {
                self.recent_ids = r.iter().map(|e| e.song_id.clone()).collect();
                self.recent_entries = r;
            }
        }
    }

    fn play_track(&mut self, track: Track) {
        let song = self.song_index.get(&track.label);
        let voice = pick_voice(self.tile, song, &track).to_string();
        self.selected_label = Some(track.label.clone());

        match track.format {
            Format::Mid => {
                if let Ok(Some(cached)) = lookup_in_cache(&track.path, &voice) {
                    self.load_resolved(track.label.clone(), cached, &voice);
                } else {
                    let path = track.path.clone();
                    let v = voice.clone();
                    let (tx, rx) = std::sync::mpsc::channel();
                    std::thread::spawn(move || {
                        let res = render_midi_blocking(&path, &v);
                        let _ = tx.send(res);
                    });
                    self.pending_render = Some(PendingRender {
                        track_label: track.label.clone(),
                        voice_id: voice,
                        rx,
                    });
                }
            }
            _ => self.load_resolved(track.label.clone(), track.path.clone(), &voice),
        }
    }

    fn load_resolved(&mut self, label: String, path: PathBuf, voice: &str) {
        if let Ok(dec) = open_decoded(&path) {
            self.sample_rate = dec.sample_rate;
            if let Ok(mut st) = self.mixer.state.lock() {
                st.decoded = Some(dec);
                st.is_playing = true;
                st.cursor_frames = 0;
                st.label = label.clone();
            }
            if let Some(ref mut db) = self.play_log {
                let v = if voice == "—" { None } else { Some(voice) };
                let _ = db.record_play(&label, v, None);
            }
        }
    }

    fn visible_tracks(&self) -> Vec<Track> {
        let needle = self.search.trim().to_lowercase();
        let mut out: Vec<Track> = self.tracks.iter()
            .filter(|t| {
                let song = self.song_index.get(&t.label);
                if !track_matches_tile(self.tile, t, song, &self.favorites, &self.recent_ids) {
                    return false;
                }
                if needle.is_empty() { return true; }
                let title = song.map(|s| s.title.as_str()).unwrap_or("");
                let composer = song.map(|s| s.composer.as_str()).unwrap_or("");
                t.label.to_lowercase().contains(&needle) ||
                title.to_lowercase().contains(&needle) ||
                composer.to_lowercase().contains(&needle)
            })
            .cloned()
            .collect();

        out.sort_by(|a, b| {
            let sa = self.song_index.get(&a.label);
            let sb = self.song_index.get(&b.label);
            let ord = match self.sort_col {
                SortColumn::Title => {
                    let ta = sa.map(|s| s.title.as_str()).unwrap_or(&a.label);
                    let tb = sb.map(|s| s.title.as_str()).unwrap_or(&b.label);
                    ta.cmp(tb)
                }
                SortColumn::Composer => {
                    let ca = sa.map(|s| s.composer_key.as_str()).unwrap_or("");
                    let cb = sb.map(|s| s.composer_key.as_str()).unwrap_or("");
                    ca.cmp(cb).then(a.label.cmp(&b.label))
                }
                SortColumn::Era => {
                    let ea = sa.and_then(|s| s.era.as_deref()).unwrap_or("");
                    let eb = sb.and_then(|s| s.era.as_deref()).unwrap_or("");
                    ea.cmp(eb)
                }
                SortColumn::Format => format!("{:?}", a.format).cmp(&format!("{:?}", b.format)),
            };
            if self.sort_asc { ord } else { ord.reverse() }
        });
        out
    }

    fn drain_cp_commands(&mut self) {
        for cmd in self.cp_state.drain_commands() {
            match cmd {
                CpCommand::LoadTrack { label } => {
                    if let Some(t) = self.tracks.iter().find(|t| t.label == label).cloned() {
                        self.play_track(t);
                    }
                }
                CpCommand::Play => {
                    if let Ok(mut st) = self.mixer.state.lock() {
                        if st.decoded.is_some() { st.is_playing = true; }
                    }
                }
                CpCommand::Stop => {
                    if let Ok(mut st) = self.mixer.state.lock() { st.is_playing = false; }
                }
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
                    if g.label.is_empty() { None } else { Some(g.label.clone()) },
                    g.cursor_frames,
                    g.decoded.as_ref().map(|d| d.total_samples / d.channels.max(1) as u64).unwrap_or(0),
                ),
                None => (false, None, 0, 0),
            }
        };
        let snap = CpSnapshot {
            frame_id: self.cp_frame_id,
            any_playing,
            loaded_label,
            cursor_frames,
            total_frames,
            visible_labels: visible.iter().map(|t| t.label.clone()).collect(),
            track_count: self.tracks.len(),
            db_song_count: self.song_index.len(),
            db_voice_count: self.db_voice_count,
        };
        self.cp_state.publish(snap);
    }
}

fn tile_from_str(s: &str) -> Option<Tile> {
    match s.to_lowercase().as_str() {
        "all" => Some(Tile::All),
        "favorites" | "favs" => Some(Tile::Favorites),
        "recent" => Some(Tile::Recent),
        "classical" => Some(Tile::Classical),
        "game" => Some(Tile::Game),
        "folk" => Some(Tile::Folk),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// UI implementation
// ---------------------------------------------------------------------------

impl eframe::App for JukeboxLiteG {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_cp_commands();
        if let Some(pending) = self.pending_render.take() {
            match pending.rx.try_recv() {
                Ok(Ok(path)) => self.load_resolved(pending.track_label, path, &pending.voice_id),
                Ok(Err(_)) => {},
                Err(std::sync::mpsc::TryRecvError::Empty) => self.pending_render = Some(pending),
                Err(_) => {},
            }
        }

        let visible = self.visible_tracks();
        self.publish_cp_snapshot(&visible);

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
                    let selected = self.tile == tile;
                    let text = format!("{}  {}", tile.icon(), tile.label());
                    if ui.selectable_label(selected, text).clicked() {
                        self.tile = tile;
                    }
                    ui.add_space(4.0);
                }

                ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
                    ui.add_space(10.0);
                    let stats = format!("{} tracks / {} voices", self.tracks.len(), self.db_voice_count);
                    ui.label(egui::RichText::new(stats).small().color(egui::Color32::GRAY));
                    if ui.button("\u{21BB} Rescan").clicked() {
                        self.rescan();
                    }
                    ui.separator();
                });
            });

        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.add_space(8.0);
                ui.label(egui::RichText::new("\u{1F50D}").color(egui::Color32::GRAY));
                ui.add(egui::TextEdit::singleline(&mut self.search)
                    .hint_text("Search library...")
                    .desired_width(240.0));
                if !self.search.is_empty() {
                    if ui.button("\u{2715}").clicked() { self.search.clear(); }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(8.0);
                    ui.label(egui::RichText::new(format!("{} results", visible.len())).small());
                });
            });
            ui.add_space(8.0);
        });

        egui::TopBottomPanel::bottom("player")
            .exact_height(80.0)
            .show(ctx, |ui| {
                let st_lock = self.mixer.state.lock();
                let (playing, label, cursor, total) = if let Ok(ref st) = st_lock {
                    (st.is_playing, 
                     st.label.clone(), 
                     st.cursor_frames, 
                     st.decoded.as_ref().map(|d| d.total_samples / d.channels.max(1) as u64).unwrap_or(0))
                } else {
                    (false, String::new(), 0, 0)
                };

                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.add_space(20.0);
                    if playing {
                        if ui.button(egui::RichText::new("\u{23F8}").size(24.0)).clicked() {
                            if let Ok(mut st) = self.mixer.state.lock() { st.is_playing = false; }
                        }
                    } else {
                        let play_btn = ui.button(egui::RichText::new("\u{25B6}").size(24.0));
                        if play_btn.clicked() {
                            if let Ok(mut st) = self.mixer.state.lock() { st.is_playing = true; }
                        }
                    }
                    if ui.button(egui::RichText::new("\u{25A0}").size(24.0)).clicked() {
                        if let Ok(mut st) = self.mixer.state.lock() { st.is_playing = false; }
                    }

                    ui.add_space(20.0);
                    ui.vertical(|ui| {
                        let title = self.song_index.get(&label).map(|s| s.title.as_str()).unwrap_or(&label);
                        ui.label(egui::RichText::new(title).strong());
                        let composer = self.song_index.get(&label).map(|s| s.composer.as_str()).unwrap_or("—");
                        ui.label(egui::RichText::new(composer).small().color(egui::Color32::GRAY));
                    });

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(20.0);
                        let mut vol = if let Ok(st) = self.mixer.state.lock() { st.volume } else { 0.85 };
                        if ui.add(egui::Slider::new(&mut vol, 0.0..=1.5).show_value(false).text("\u{1F50A}")).changed() {
                            if let Ok(mut st) = self.mixer.state.lock() { st.volume = vol; }
                        }
                    });
                });

                ui.add_space(8.0);
                let progress = if total > 0 { cursor as f32 / total as f32 } else { 0.0 };
                ui.add(egui::ProgressBar::new(progress).desired_height(4.0));
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
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

                for (idx, track) in visible.iter().enumerate() {
                    let song = self.song_index.get(&track.label);
                    let is_selected = self.selected_label.as_ref() == Some(&track.label);
                    
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
                    
                    let response = frame.show(ui, |ui| {
                        let w = ui.available_width();
                        let cols = [w * 0.4, w * 0.3, w * 0.15, w * 0.1];
                        
                        ui.horizontal(|ui| {
                            let title = song.map(|s| s.title.as_str()).unwrap_or(&track.label);
                            let text_color = if is_selected { ui.visuals().selection.stroke.color } else { ui.visuals().text_color() };
                            ui.add_sized([cols[0], 20.0], egui::Label::new(egui::RichText::new(title).strong().color(text_color)));
                            
                            let composer = song.map(|s| s.composer.as_str()).unwrap_or("—");
                            ui.add_sized([cols[1], 20.0], egui::Label::new(composer));
                            
                            let era = song.and_then(|s| s.era.as_deref()).unwrap_or("");
                            ui.add_sized([cols[2], 20.0], egui::Label::new(era));
                            
                            ui.add_sized([cols[3], 20.0], egui::Label::new(format!("{:?}", track.format).to_uppercase()));
                        });
                    }).response;

                    let response = ui.interact(response.rect, ui.id().with(idx), egui::Sense::click());
                    if response.clicked() {
                        self.play_track(track.clone());
                    }
                    if response.hovered() && !is_selected {
                        ui.painter().rect_filled(response.rect, 0.0, ui.visuals().widgets.hovered.bg_fill.gamma_multiply(0.2));
                    }
                }
            });
        });

        if let Ok(st) = self.mixer.state.lock() {
            if st.is_playing || self.pending_render.is_some() {
                ctx.request_repaint_after(std::time::Duration::from_millis(100));
            }
        }
    }
}

impl JukeboxLiteG {
    fn header_cell(&mut self, ui: &mut egui::Ui, text: &str, width: f32, col: SortColumn) {
        let selected = self.sort_col == col;
        let mut text = text.to_string();
        if selected {
            text.push(' ');
            text.push(if self.sort_asc { '\u{25B4}' } else { '\u{25BE}' });
        }
        
        let button = egui::Button::new(egui::RichText::new(text).small().strong())
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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = JukeboxLiteG::new()?;
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 700.0])
            .with_title("JUKEBOX G"),
        ..Default::default()
    };
    eframe::run_native(
        "JUKEBOX G",
        options,
        Box::new(|cc| {
            keysynth::ui::setup_japanese_fonts(&cc.egui_ctx);
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| format!("eframe: {e}").into())
}
