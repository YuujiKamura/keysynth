//! jukebox_lite_c — dense list-view music browser for keysynth.
//!
//! This is a sibling to `jukebox_lite`, but deliberately not a card grid.
//! The interaction model is closer to Finder list view / foobar2000 /
//! Spotify's Songs table: filter rail, ledger-style track list, detail
//! inspector, and a persistent transport strip.
//!
//! Backend reuse stays unchanged: `library_db`, `play_log`,
//! `preview_cache`, `gui_cp`, `ui::setup_japanese_fonts`, and
//! `audio_audit`.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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

use keysynth::audio_audit::{self, DominantFrequency};
use keysynth::gui_cp;
use keysynth::library_db::{LibraryDb, Song, SongFilter, SongSort, VoiceFilter};
use keysynth::play_log::{PlayEntry, PlayLogDb};

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

    fn label(self) -> &'static str {
        match self {
            Format::Wav => "WAV",
            Format::Mp3 => "MP3",
            Format::Mid => "MID",
        }
    }
}

#[derive(Clone, Debug)]
struct Track {
    path: PathBuf,
    label: String,
    format: Format,
}

fn scan_dirs(dirs: &[&Path]) -> Vec<Track> {
    let mut out = Vec::new();
    for d in dirs {
        let read = match std::fs::read_dir(d) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let mut dir_tracks = Vec::new();
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
            volume: 0.82,
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
    let mut cursor_advanced = 0u64;
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
    let err_fn = |err| eprintln!("jukebox_lite_c audio stream error: {err}");
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
            let mut scratch = Vec::<f32>::new();
            device
                .build_output_stream(
                    &stream_cfg,
                    move |out: &mut [i16], _| {
                        if scratch.len() != out.len() {
                            scratch.resize(out.len(), 0.0);
                        }
                        fill_callback(&mut scratch, channels, &st);
                        for (dst, src) in out.iter_mut().zip(scratch.iter()) {
                            *dst = (src.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                        }
                    },
                    err_fn,
                    None,
                )
                .map_err(|e| format!("build i16: {e}"))?
        }
        SampleFormat::U16 => {
            let st = state.clone();
            let mut scratch = Vec::<f32>::new();
            device
                .build_output_stream(
                    &stream_cfg,
                    move |out: &mut [u16], _| {
                        if scratch.len() != out.len() {
                            scratch.resize(out.len(), 0.0);
                        }
                        fill_callback(&mut scratch, channels, &st);
                        for (dst, src) in out.iter_mut().zip(scratch.iter()) {
                            *dst = (((src.clamp(-1.0, 1.0) + 1.0) * 0.5) * u16::MAX as f32) as u16;
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
        .ok_or_else(|| "render_midi binary not found alongside jukebox_lite_c".to_string())?;
    keysynth::preview_cache::render_to_cache(&cache, &key, voice_id, &bin)
        .map_err(|e| format!("render_to_cache: {e}"))
}

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
        return "direct";
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

fn compact_composer(composer: &str) -> String {
    let core = match composer.split_once('(') {
        Some((before, _)) => before,
        None => composer,
    };
    core.trim().to_string()
}

fn humanize_ago(secs: i64) -> String {
    let s = secs.max(0);
    if s < 60 {
        format!("{s}s ago")
    } else if s < 3_600 {
        format!("{}m ago", s / 60)
    } else if s < 86_400 {
        format!("{}h ago", s / 3_600)
    } else {
        format!("{}d ago", s / 86_400)
    }
}

fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn era_badge_color(era: Option<&str>) -> egui::Color32 {
    match era {
        Some("Baroque") => egui::Color32::from_rgb(201, 172, 112),
        Some("Classical") => egui::Color32::from_rgb(165, 199, 155),
        Some("Romantic") => egui::Color32::from_rgb(211, 173, 190),
        Some("Modern") => egui::Color32::from_rgb(153, 187, 208),
        Some("Traditional") => egui::Color32::from_rgb(213, 189, 134),
        _ => egui::Color32::from_rgb(191, 191, 191),
    }
}

fn source_bucket(track: &Track, song: Option<&Song>) -> &'static str {
    match track.format {
        Format::Mid if song.is_some() => "CAT",
        Format::Mid => "MID",
        Format::Mp3 => "MP3",
        Format::Wav => "WAV",
    }
}

fn display_title(song: Option<&Song>, track: &Track) -> String {
    song.map(|s| s.title.clone())
        .unwrap_or_else(|| track.label.clone())
}

fn display_composer(song: Option<&Song>) -> String {
    song.map(|s| compact_composer(&s.composer))
        .unwrap_or_else(|| "-".to_string())
}

fn display_voice_label(voice: Option<&str>) -> &'static str {
    match voice {
        Some("direct") => "direct file",
        Some("piano-modal") => "piano-modal",
        Some("guitar-stk") => "guitar-stk",
        Some("square") => "square",
        Some(_) => "custom",
        None => "-",
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
                "jukebox_lite_c: library_db open failed ({}): {e} — running without DB metadata",
                db_path.display()
            );
            return (HashMap::new(), 0);
        }
    };
    if let Err(e) = db.migrate() {
        eprintln!("jukebox_lite_c: library_db migrate failed: {e}");
        return (HashMap::new(), 0);
    }
    if manifest.is_file() {
        if let Err(e) = db.import_songs(&manifest) {
            eprintln!("jukebox_lite_c: library_db import_songs failed: {e}");
        }
    }
    if voices_live.is_dir() {
        if let Err(e) = db.import_voices(&voices_live) {
            eprintln!("jukebox_lite_c: library_db import_voices failed: {e}");
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
            eprintln!("jukebox_lite_c: play_log open failed: {e}");
            return None;
        }
    };
    if let Err(e) = db.migrate() {
        eprintln!("jukebox_lite_c: play_log migrate failed: {e}");
        return None;
    }
    Some(db)
}

const RECENT_WINDOW: usize = 24;

fn snapshot_history(
    log: Option<&PlayLogDb>,
) -> (
    HashSet<String>,
    HashSet<String>,
    Vec<PlayEntry>,
    HashMap<String, i64>,
) {
    let mut favs = HashSet::new();
    let mut recent_ids = HashSet::new();
    let mut recent_entries = Vec::new();
    let mut play_counts = HashMap::new();
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
        if let Ok(p) = db.play_counts() {
            play_counts = p;
        }
    }
    (favs, recent_ids, recent_entries, play_counts)
}

#[derive(Clone, Debug, Default, Serialize)]
struct CpSnapshot {
    frame_id: u64,
    track_count: usize,
    db_song_count: usize,
    db_voice_count: usize,
    selected_label: Option<String>,
    selected_tile: String,
    any_playing: bool,
    loaded_label: Option<String>,
    cursor_frames: u64,
    total_frames: u64,
    sample_rate: u32,
    visible_labels: Vec<String>,
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

fn spawn_cp_server(state: gui_cp::State<CpSnapshot, CpCommand>) -> std::io::Result<gui_cp::Handle> {
    let endpoint = gui_cp::resolve_endpoint("jukebox_lite_c", None);
    let st_get = state.clone();
    let st_load = state.clone();
    let st_play = state.clone();
    let st_stop = state.clone();
    let st_tile = state.clone();
    let st_rescan = state.clone();
    let st_list = state.clone();
    gui_cp::Builder::new("jukebox_lite_c", &endpoint)
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

#[derive(Clone, Debug)]
struct AudioAuditSnapshot {
    track_label: String,
    playable_path: PathBuf,
    duration_sec: f32,
    peak_dbfs: f32,
    rms_dbfs: f32,
    crest_factor_db: f32,
    silence_count: usize,
    longest_silence_sec: f32,
    dominant: Vec<DominantFrequency>,
    tone_balance: [f32; 3],
    envelope: Vec<f32>,
}

fn decode_mono_for_audit(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let mut dec = open_decoded(path)?;
    let ch = dec.channels.max(1) as usize;
    let sr = dec.sample_rate;
    let reserve = (dec.total_samples / ch as u64).min(44_100 * 180) as usize;
    let mut mono = Vec::with_capacity(reserve);
    loop {
        let mut acc = 0.0f32;
        let mut got = 0usize;
        for _ in 0..ch {
            match read_one_sample(&mut dec) {
                Some(v) => {
                    acc += v;
                    got += 1;
                }
                None => break,
            }
        }
        if got == 0 {
            break;
        }
        mono.push(acc / got as f32);
        if got < ch {
            break;
        }
    }
    Ok((mono, sr))
}

fn build_audio_audit(track_label: &str, path: &Path) -> Result<AudioAuditSnapshot, String> {
    let (samples, sr) = decode_mono_for_audit(path)?;
    if samples.is_empty() || sr == 0 {
        return Err("decoded audio was empty".to_string());
    }
    let silences = audio_audit::silence_intervals(&samples, sr, -52.0, 180);
    let longest_silence_sec = silences
        .iter()
        .map(|s| s.duration_sec)
        .fold(0.0f32, f32::max);
    let spectral = audio_audit::spectral_envelope(&samples, sr, 9);
    let low: f32 = spectral.bands.iter().take(3).sum();
    let mid: f32 = spectral.bands.iter().skip(3).take(3).sum();
    let high: f32 = spectral.bands.iter().skip(6).take(3).sum();
    let max_band = low.max(mid).max(high).max(1e-6);
    let envelope_raw = audio_audit::rms_envelope(&samples, sr, 300);
    let max_env = envelope_raw
        .iter()
        .copied()
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let envelope = envelope_raw.into_iter().map(|v| v / max_env).collect();
    Ok(AudioAuditSnapshot {
        track_label: track_label.to_string(),
        playable_path: path.to_path_buf(),
        duration_sec: samples.len() as f32 / sr as f32,
        peak_dbfs: audio_audit::peak_dbfs(&samples),
        rms_dbfs: audio_audit::rms_dbfs(&samples),
        crest_factor_db: audio_audit::crest_factor_db(&samples),
        silence_count: silences.len(),
        longest_silence_sec,
        dominant: audio_audit::dominant_frequencies(&samples, sr, 3),
        tone_balance: [low / max_band, mid / max_band, high / max_band],
        envelope,
    })
}

struct PendingRender {
    track_label: String,
    voice_id: String,
    started: Instant,
    rx: std::sync::mpsc::Receiver<Result<PathBuf, String>>,
    _handle: std::thread::JoinHandle<()>,
}

struct PendingAudit {
    track_label: String,
    rx: std::sync::mpsc::Receiver<Result<AudioAuditSnapshot, String>>,
    _handle: std::thread::JoinHandle<()>,
}

#[derive(Clone, Default)]
struct PlaybackSnapshot {
    loaded_label: Option<String>,
    is_playing: bool,
    cursor_frames: u64,
    total_frames: u64,
    volume: f32,
}

struct JukeboxLiteC {
    tracks: Vec<Track>,
    song_index: HashMap<String, Song>,
    db_voice_count: usize,
    play_log: Option<PlayLogDb>,
    favorites: HashSet<String>,
    recent_ids: HashSet<String>,
    recent_entries: Vec<PlayEntry>,
    play_counts: HashMap<String, i64>,

    mixer: Mixer,
    sample_rate_for_progress: u32,
    pending_render: Option<PendingRender>,
    pending_audit: Option<PendingAudit>,
    audio_audit: Option<AudioAuditSnapshot>,
    audio_audit_error: Option<String>,
    loaded_voice: Option<String>,

    tile: Tile,
    search: String,
    selected_label: Option<String>,
    refresh_dirs: Vec<PathBuf>,

    cp_state: gui_cp::State<CpSnapshot, CpCommand>,
    _cp_handle: Option<gui_cp::Handle>,
    cp_frame_id: u64,
}

impl JukeboxLiteC {
    fn new(dirs: Vec<PathBuf>) -> Result<Self, String> {
        let (song_index, db_voice_count) = load_library_index();
        let dir_refs: Vec<&Path> = dirs.iter().map(|p| p.as_path()).collect();
        let tracks = scan_dirs(&dir_refs);
        println!(
            "jukebox_lite_c: library_db catalog -> {} songs / {} voices ({} files on disk)",
            song_index.len(),
            db_voice_count,
            tracks.len(),
        );
        let play_log = load_play_log();
        let (favorites, recent_ids, recent_entries, play_counts) =
            snapshot_history(play_log.as_ref());

        let audio = Arc::new(Mutex::new(AudioState::new()));
        let mixer = build_mixer_stream(audio).map_err(|e| format!("audio: {e}"))?;
        let sample_rate_for_progress = 44_100;

        let cp_state: gui_cp::State<CpSnapshot, CpCommand> = gui_cp::State::new();
        let cp_handle = match spawn_cp_server(cp_state.clone()) {
            Ok(h) => Some(h),
            Err(e) => {
                eprintln!("jukebox_lite_c: CP server failed: {e}");
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
            play_counts,
            mixer,
            sample_rate_for_progress,
            pending_render: None,
            pending_audit: None,
            audio_audit: None,
            audio_audit_error: None,
            loaded_voice: None,
            tile: Tile::All,
            search: String::new(),
            selected_label: initial_label,
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
        self.refresh_history();
    }

    fn refresh_history(&mut self) {
        let (f, r, re, counts) = snapshot_history(self.play_log.as_ref());
        self.favorites = f;
        self.recent_ids = r;
        self.recent_entries = re;
        self.play_counts = counts;
    }

    fn playback_snapshot(&self) -> PlaybackSnapshot {
        let st = match self.mixer.state.lock() {
            Ok(g) => g,
            Err(_) => return PlaybackSnapshot::default(),
        };
        PlaybackSnapshot {
            loaded_label: if st.label.is_empty() {
                None
            } else {
                Some(st.label.clone())
            },
            is_playing: st.is_playing,
            cursor_frames: st.cursor_frames,
            total_frames: st
                .decoded
                .as_ref()
                .map(|d| d.total_samples / d.channels.max(1) as u64)
                .unwrap_or(0),
            volume: st.volume,
        }
    }

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
                let ka = self
                    .song_index
                    .get(&a.label)
                    .map(|s| s.composer_key.clone());
                let kb = self
                    .song_index
                    .get(&b.label)
                    .map(|s| s.composer_key.clone());
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

    fn ensure_selection(&mut self, visible: &[Track]) {
        if self
            .selected_label
            .as_ref()
            .map(|label| self.tracks.iter().any(|t| &t.label == label))
            .unwrap_or(false)
        {
            return;
        }
        self.selected_label = visible.first().map(|t| t.label.clone());
    }

    fn selected_track(&self) -> Option<Track> {
        let label = self.selected_label.as_ref()?;
        self.tracks.iter().find(|t| &t.label == label).cloned()
    }

    fn toggle_favorite(&mut self, label: &str) {
        let Some(db) = self.play_log.as_mut() else {
            return;
        };
        if let Err(e) = db.toggle_favorite(label) {
            eprintln!("jukebox_lite_c: toggle_favorite({label}): {e}");
        }
        self.refresh_history();
    }

    fn resume_loaded(&mut self) {
        if let Ok(mut st) = self.mixer.state.lock() {
            if st.decoded.is_some() {
                st.is_playing = true;
            }
        }
    }

    fn set_volume(&mut self, volume: f32) {
        if let Ok(mut st) = self.mixer.state.lock() {
            st.volume = volume.clamp(0.0, 1.5);
        }
    }

    fn activate_track(&mut self, track: Track, playback: &PlaybackSnapshot) {
        if playback.loaded_label.as_deref() == Some(track.label.as_str()) {
            if playback.is_playing {
                self.stop();
            } else {
                self.resume_loaded();
            }
            self.selected_label = Some(track.label);
            return;
        }
        self.play_track(track);
    }

    fn play_track(&mut self, track: Track) {
        let song = self.song_index.get(&track.label).cloned();
        let voice = pick_voice(self.tile, song.as_ref(), &track).to_string();
        self.selected_label = Some(track.label.clone());

        let playable = match track.format {
            Format::Mid => match lookup_in_cache(&track.path, &voice) {
                Ok(Some(p)) => p,
                Ok(None) => {
                    self.spawn_render(&track, &voice);
                    return;
                }
                Err(e) => {
                    eprintln!("jukebox_lite_c: cache lookup {}: {e}", track.path.display());
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
            .name("jukebox-lite-c-render".into())
            .spawn(move || {
                let result = render_midi_blocking(&path, &voice_for_thread);
                let _ = tx.send(result);
            })
            .expect("spawn render thread");
        eprintln!(
            "jukebox_lite_c: render queued ({}) — voice={voice}",
            track.path.display(),
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
                    "jukebox_lite_c: render done ({}) in {} ms",
                    wav.display(),
                    elapsed,
                );
                self.load_resolved(pending.track_label, wav, &pending.voice_id);
            }
            Err(e) => {
                eprintln!("jukebox_lite_c: render failed: {e}");
            }
        }
    }

    fn spawn_audio_audit(&mut self, label: &str, playable: &Path) {
        let label = label.to_string();
        let label_for_thread = label.clone();
        let path = playable.to_path_buf();
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = std::thread::Builder::new()
            .name("jukebox-lite-c-audit".into())
            .spawn(move || {
                let result = build_audio_audit(&label_for_thread, &path);
                let _ = tx.send(result);
            })
            .expect("spawn audit thread");
        self.pending_audit = Some(PendingAudit {
            track_label: label,
            rx,
            _handle: handle,
        });
    }

    fn poll_pending_audit(&mut self) {
        let pending = match self.pending_audit.as_ref() {
            Some(p) => p,
            None => return,
        };
        let result = match pending.rx.try_recv() {
            Ok(r) => r,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.pending_audit = None;
                return;
            }
        };
        let pending = self.pending_audit.take().expect("guarded");
        match result {
            Ok(report) => {
                if report.track_label == pending.track_label {
                    self.audio_audit_error = None;
                    self.audio_audit = Some(report);
                }
            }
            Err(e) => {
                self.audio_audit = None;
                self.audio_audit_error = Some(e);
            }
        }
    }

    fn load_resolved(&mut self, label: String, playable: PathBuf, voice: &str) {
        let dec = match open_decoded(&playable) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("jukebox_lite_c: open {}: {e}", playable.display());
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
        self.loaded_voice = Some(voice.to_string());
        self.audio_audit = None;
        self.audio_audit_error = None;
        self.spawn_audio_audit(&label, &playable);

        if let Some(db) = self.play_log.as_mut() {
            let stored_voice = if voice == "direct" { None } else { Some(voice) };
            if let Err(e) = db.record_play(&label, stored_voice, None) {
                eprintln!("jukebox_lite_c: record_play({label}): {e}");
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
                        eprintln!("jukebox_lite_c: CP load_track unknown label={label}");
                    }
                }
                CpCommand::Play => self.resume_loaded(),
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

    fn publish_cp_snapshot(&mut self, visible: &[Track], playback: &PlaybackSnapshot) {
        self.cp_frame_id = self.cp_frame_id.saturating_add(1);
        let snap = CpSnapshot {
            frame_id: self.cp_frame_id,
            track_count: self.tracks.len(),
            db_song_count: self.song_index.len(),
            db_voice_count: self.db_voice_count,
            selected_label: self.selected_label.clone(),
            selected_tile: self.tile.label().to_string(),
            any_playing: playback.is_playing,
            loaded_label: playback.loaded_label.clone(),
            cursor_frames: playback.cursor_frames,
            total_frames: playback.total_frames,
            sample_rate: self.sample_rate_for_progress,
            visible_labels: visible.iter().map(|t| t.label.clone()).collect(),
            all_labels: self.tracks.iter().map(|t| t.label.clone()).collect(),
        };
        self.cp_state.publish(snap);
    }

    fn show_header(&mut self, ui: &mut egui::Ui, visible_count: usize) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.heading("keysynth / jukebox lite c");
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("list browser")
                    .small()
                    .color(egui::Color32::from_rgb(96, 100, 93)),
            );
            ui.add_space(16.0);
            stat_chip(
                ui,
                "View",
                format!("{} rows", visible_count),
                egui::Color32::from_rgb(225, 229, 218),
            );
            stat_chip(
                ui,
                "Catalog",
                format!("{} songs", self.song_index.len()),
                egui::Color32::from_rgb(225, 222, 214),
            );
            stat_chip(
                ui,
                "Voices",
                self.db_voice_count.to_string(),
                egui::Color32::from_rgb(220, 229, 228),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Rescan").clicked() {
                    self.rescan();
                }
                if ui.button("Clear").clicked() {
                    self.search.clear();
                }
            });
        });
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("Find")
                    .strong()
                    .color(egui::Color32::from_rgb(48, 53, 49)),
            );
            ui.add_sized(
                [300.0, 30.0],
                egui::TextEdit::singleline(&mut self.search)
                    .hint_text("title, composer, stem, or catalog metadata"),
            );
            ui.add_space(10.0);
            ui.label(
                egui::RichText::new(format!("{} view", self.tile.label()))
                    .color(egui::Color32::from_rgb(83, 98, 88)),
            );
            if let Some(pending) = self.pending_render.as_ref() {
                ui.add_space(14.0);
                ui.spinner();
                ui.label(
                    egui::RichText::new(format!("rendering {}", pending.track_label))
                        .color(egui::Color32::from_rgb(69, 111, 103)),
                );
            } else if let Some(pending) = self.pending_audit.as_ref() {
                ui.add_space(14.0);
                ui.spinner();
                ui.label(
                    egui::RichText::new(format!("auditing {}", pending.track_label))
                        .color(egui::Color32::from_rgb(92, 93, 131)),
                );
            }
        });
    }

    fn show_sidebar(&mut self, ui: &mut egui::Ui) {
        ui.add_space(8.0);
        ui.label(
            egui::RichText::new("Filters")
                .small()
                .color(egui::Color32::from_rgb(96, 100, 93)),
        );
        ui.add_space(4.0);
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
            let fill = if selected {
                egui::Color32::from_rgb(80, 115, 103)
            } else {
                egui::Color32::from_rgb(229, 226, 218)
            };
            let text = if selected {
                egui::Color32::from_rgb(247, 245, 240)
            } else {
                egui::Color32::from_rgb(44, 47, 45)
            };
            let button = egui::Button::new(
                egui::RichText::new(format!("{:<10} {:>4}", tile.label(), count)).color(text),
            )
            .fill(fill);
            if ui.add_sized([170.0, 30.0], button).clicked() {
                self.tile = tile;
            }
            ui.add_space(4.0);
        }

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(10.0);
        ui.label(
            egui::RichText::new("Routing")
                .small()
                .color(egui::Color32::from_rgb(96, 100, 93)),
        );
        ui.add_space(4.0);
        routing_row(ui, "Classical", "piano-modal");
        routing_row(ui, "Game", "square");
        routing_row(ui, "Folk", "guitar-stk");
        routing_row(ui, "Audio files", "direct");

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(10.0);
        ui.label(
            egui::RichText::new("Recent")
                .small()
                .color(egui::Color32::from_rgb(96, 100, 93)),
        );
        ui.add_space(4.0);
        let mut to_select = None;
        let mut shown = 0usize;
        for entry in &self.recent_entries {
            if shown >= 5 {
                break;
            }
            let Some(track) = self.tracks.iter().find(|t| t.label == entry.song_id) else {
                continue;
            };
            let song = self.song_index.get(&track.label);
            let title = display_title(song, track);
            let subtitle = display_composer(song);
            let button = egui::Button::new(
                egui::RichText::new(title).color(egui::Color32::from_rgb(44, 47, 45)),
            )
            .fill(egui::Color32::from_rgb(240, 236, 228));
            if ui.add_sized([170.0, 26.0], button).clicked() {
                to_select = Some(track.label.clone());
            }
            ui.label(
                egui::RichText::new(subtitle)
                    .small()
                    .color(egui::Color32::from_rgb(108, 111, 108)),
            );
            ui.add_space(4.0);
            shown += 1;
        }
        if let Some(label) = to_select {
            self.selected_label = Some(label);
        }
    }

    fn show_transport(&mut self, ui: &mut egui::Ui, playback: &PlaybackSnapshot) {
        egui::Frame::none()
            .fill(egui::Color32::from_rgb(57, 69, 63))
            .inner_margin(egui::Margin::symmetric(14.0, 12.0))
            .rounding(8.0)
            .show(ui, |ui| {
                let label = playback.loaded_label.clone();
                let song = label.as_ref().and_then(|l| self.song_index.get(l));
                let title = match (label.as_ref(), song) {
                    (Some(_), Some(s)) => s.title.clone(),
                    (Some(l), None) => l.clone(),
                    (None, _) => "No track loaded".to_string(),
                };
                let subtitle = match (label.as_ref(), song) {
                    (Some(_), Some(s)) => {
                        let mut out = compact_composer(&s.composer);
                        if let Some(era) = s.era.as_deref() {
                            out.push_str("  /  ");
                            out.push_str(era);
                        }
                        out
                    }
                    (Some(l), None) => l.clone(),
                    (None, _) => "Select a row and hit Play.".to_string(),
                };
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.label(
                            egui::RichText::new(title)
                                .size(17.0)
                                .strong()
                                .color(egui::Color32::from_rgb(246, 244, 240)),
                        );
                        ui.label(
                            egui::RichText::new(subtitle)
                                .small()
                                .color(egui::Color32::from_rgb(205, 213, 204)),
                        );
                    });
                    ui.add_space(14.0);
                    let mut vol = playback.volume;
                    let progress = if playback.total_frames > 0 {
                        (playback.cursor_frames as f32 / playback.total_frames as f32)
                            .clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    ui.add(egui::ProgressBar::new(progress).desired_width(340.0).text(
                        if playback.total_frames > 0 {
                            let secs = playback.cursor_frames as u32
                                / self.sample_rate_for_progress.max(1);
                            let total_secs =
                                playback.total_frames as u32 / self.sample_rate_for_progress.max(1);
                            format!(
                                "{:01}:{:02} / {:01}:{:02}",
                                secs / 60,
                                secs % 60,
                                total_secs / 60,
                                total_secs % 60
                            )
                        } else {
                            "--:--".to_string()
                        },
                    ));
                    ui.add_space(14.0);
                    if ui
                        .add_sized(
                            [62.0, 28.0],
                            egui::Button::new(if playback.is_playing { "Stop" } else { "Play" })
                                .fill(egui::Color32::from_rgb(208, 224, 216)),
                        )
                        .clicked()
                    {
                        if playback.is_playing {
                            self.stop();
                        } else {
                            self.resume_loaded();
                        }
                    }
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(display_voice_label(self.loaded_voice.as_deref()))
                            .small()
                            .color(egui::Color32::from_rgb(219, 223, 216)),
                    );
                    ui.add_space(8.0);
                    if ui
                        .add_sized(
                            [110.0, 24.0],
                            egui::Slider::new(&mut vol, 0.0..=1.25).show_value(false),
                        )
                        .changed()
                    {
                        self.set_volume(vol);
                    }
                    ui.label(
                        egui::RichText::new(format!("{:>3.0}%", vol * 100.0))
                            .small()
                            .color(egui::Color32::from_rgb(219, 223, 216)),
                    );
                });
            });
    }

    fn show_inspector(&mut self, ui: &mut egui::Ui, playback: &PlaybackSnapshot) {
        ui.add_space(8.0);
        let Some(track) = self.selected_track() else {
            ui.label("No selection.");
            return;
        };
        let song = self.song_index.get(&track.label).cloned();
        let title = display_title(song.as_ref(), &track);
        let composer = display_composer(song.as_ref());
        ui.label(
            egui::RichText::new(title)
                .heading()
                .color(egui::Color32::from_rgb(36, 40, 37)),
        );
        ui.label(
            egui::RichText::new(composer)
                .color(egui::Color32::from_rgb(93, 96, 91))
                .small(),
        );
        ui.add_space(10.0);

        let is_fav = self.favorites.contains(&track.label);
        ui.horizontal(|ui| {
            if ui
                .add_sized(
                    [76.0, 28.0],
                    egui::Button::new("Play").fill(egui::Color32::from_rgb(211, 224, 215)),
                )
                .clicked()
            {
                self.activate_track(track.clone(), playback);
            }
            if ui
                .add_sized(
                    [76.0, 28.0],
                    egui::Button::new(if is_fav { "Unsave" } else { "Save" })
                        .fill(egui::Color32::from_rgb(229, 224, 214)),
                )
                .clicked()
            {
                self.toggle_favorite(&track.label);
            }
        });

        ui.add_space(10.0);
        egui::Grid::new("jukebox_lite_c_meta")
            .num_columns(2)
            .spacing([12.0, 8.0])
            .show(ui, |ui| {
                inspector_key(ui, "Label");
                ui.label(egui::RichText::new(&track.label).monospace());
                ui.end_row();

                inspector_key(ui, "Format");
                ui.label(track.format.label());
                ui.end_row();

                inspector_key(ui, "Route");
                ui.label(display_voice_label(Some(pick_voice(
                    self.tile,
                    song.as_ref(),
                    &track,
                ))));
                ui.end_row();

                inspector_key(ui, "Era");
                ui.label(song.as_ref().and_then(|s| s.era.as_deref()).unwrap_or("-"));
                ui.end_row();

                inspector_key(ui, "Instrument");
                ui.label(song.as_ref().map(|s| s.instrument.as_str()).unwrap_or("-"));
                ui.end_row();

                inspector_key(ui, "License");
                ui.label(song.as_ref().map(|s| s.license.as_str()).unwrap_or("-"));
                ui.end_row();

                inspector_key(ui, "Plays");
                ui.label(
                    self.play_counts
                        .get(&track.label)
                        .copied()
                        .unwrap_or(0)
                        .to_string(),
                );
                ui.end_row();

                inspector_key(ui, "Last heard");
                let last_heard = self
                    .play_log
                    .as_ref()
                    .and_then(|db| db.last_played_at(&track.label).ok().flatten())
                    .map(|ts| humanize_ago(unix_now().saturating_sub(ts)))
                    .unwrap_or_else(|| "never".to_string());
                ui.label(last_heard);
                ui.end_row();

                inspector_key(ui, "Path");
                ui.label(
                    egui::RichText::new(track.path.display().to_string())
                        .small()
                        .monospace(),
                );
                ui.end_row();
            });

        if let Some(song) = song.as_ref() {
            if !song.tags.is_empty() {
                ui.add_space(10.0);
                ui.label(
                    egui::RichText::new("Tags")
                        .small()
                        .color(egui::Color32::from_rgb(96, 100, 93)),
                );
                ui.add_space(4.0);
                ui.horizontal_wrapped(|ui| {
                    for tag in &song.tags {
                        detail_chip(ui, tag, egui::Color32::from_rgb(232, 228, 219));
                    }
                });
            }

            if let Some(context) = song.context.as_deref() {
                ui.add_space(10.0);
                ui.label(
                    egui::RichText::new("Context")
                        .small()
                        .color(egui::Color32::from_rgb(96, 100, 93)),
                );
                ui.add_space(4.0);
                egui::Frame::none()
                    .fill(egui::Color32::from_rgb(240, 236, 228))
                    .inner_margin(egui::Margin::symmetric(10.0, 8.0))
                    .rounding(6.0)
                    .show(ui, |ui| {
                        ui.label(context);
                    });
            }

            if let Some(url) = song.source_url.as_deref() {
                ui.add_space(8.0);
                ui.hyperlink_to("Source link", url);
            }
        }

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(12.0);
        ui.label(
            egui::RichText::new("Audio audit")
                .small()
                .color(egui::Color32::from_rgb(96, 100, 93)),
        );
        ui.add_space(6.0);

        let audit_matches = self
            .audio_audit
            .as_ref()
            .map(|a| a.track_label == track.label)
            .unwrap_or(false);
        let audit_pending = self
            .pending_audit
            .as_ref()
            .map(|p| p.track_label == track.label)
            .unwrap_or(false);

        if audit_pending {
            ui.spinner();
            ui.label("Computing audio metrics...");
            return;
        }

        if !audit_matches {
            if self
                .pending_render
                .as_ref()
                .map(|p| p.track_label == track.label)
                .unwrap_or(false)
            {
                ui.label("This row is waiting on preview render before audio audit can run.");
            } else {
                ui.label("Play this row to populate the audio audit panel.");
            }
            return;
        }

        let Some(report) = self.audio_audit.as_ref() else {
            if let Some(err) = self.audio_audit_error.as_deref() {
                ui.label(err);
            }
            return;
        };

        ui.label(
            egui::RichText::new(report.playable_path.display().to_string())
                .small()
                .monospace()
                .color(egui::Color32::from_rgb(103, 107, 103)),
        );
        ui.add_space(8.0);
        draw_envelope(ui, &report.envelope);
        ui.add_space(8.0);
        egui::Grid::new("jukebox_lite_c_audit_metrics")
            .num_columns(2)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                inspector_key(ui, "Duration");
                ui.label(format!("{:.1}s", report.duration_sec));
                ui.end_row();

                inspector_key(ui, "Peak");
                ui.label(format!("{:.1} dBFS", report.peak_dbfs));
                ui.end_row();

                inspector_key(ui, "RMS");
                ui.label(format!("{:.1} dBFS", report.rms_dbfs));
                ui.end_row();

                inspector_key(ui, "Crest");
                ui.label(format!("{:.1} dB", report.crest_factor_db));
                ui.end_row();

                inspector_key(ui, "Silence");
                ui.label(format!(
                    "{} spans / {:.2}s longest",
                    report.silence_count, report.longest_silence_sec
                ));
                ui.end_row();
            });
        ui.add_space(8.0);
        meter_row(
            ui,
            "Body",
            report.tone_balance[0],
            egui::Color32::from_rgb(182, 196, 160),
        );
        meter_row(
            ui,
            "Presence",
            report.tone_balance[1],
            egui::Color32::from_rgb(123, 167, 151),
        );
        meter_row(
            ui,
            "Air",
            report.tone_balance[2],
            egui::Color32::from_rgb(144, 171, 201),
        );

        if !report.dominant.is_empty() {
            ui.add_space(10.0);
            ui.label(
                egui::RichText::new("Dominant frequencies")
                    .small()
                    .color(egui::Color32::from_rgb(96, 100, 93)),
            );
            for dom in &report.dominant {
                ui.label(format!(
                    "{:>6.1} Hz   {:>5.1} dB",
                    dom.freq_hz, dom.magnitude_db
                ));
            }
        }
    }

    fn show_table(&mut self, ui: &mut egui::Ui, visible: &[Track], playback: &PlaybackSnapshot) {
        if visible.is_empty() {
            ui.add_space(50.0);
            ui.vertical_centered(|ui| {
                ui.label(
                    egui::RichText::new("No rows in this view.")
                        .heading()
                        .color(egui::Color32::from_rgb(104, 108, 102)),
                );
                ui.label("Change the filter or clear the search box.");
            });
            return;
        }

        const ROW_H: f32 = 34.0;
        const STATUS_W: f32 = 54.0;
        const COMPOSER_W: f32 = 180.0;
        const ERA_W: f32 = 90.0;
        const INSTR_W: f32 = 92.0;
        const SOURCE_W: f32 = 56.0;
        const PLAYS_W: f32 = 42.0;
        const ACTION_W: f32 = 56.0;
        const FAV_W: f32 = 44.0;

        let title_w = (ui.available_width()
            - STATUS_W
            - COMPOSER_W
            - ERA_W
            - INSTR_W
            - SOURCE_W
            - PLAYS_W
            - ACTION_W
            - FAV_W
            - 48.0)
            .max(220.0);

        egui::Frame::none()
            .fill(egui::Color32::from_rgb(214, 221, 210))
            .rounding(6.0)
            .inner_margin(egui::Margin::symmetric(10.0, 8.0))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    header_cell(ui, STATUS_W, "State");
                    header_cell(ui, title_w, "Title");
                    header_cell(ui, COMPOSER_W, "Composer");
                    header_cell(ui, ERA_W, "Era");
                    header_cell(ui, INSTR_W, "Inst");
                    header_cell(ui, SOURCE_W, "Src");
                    header_cell(ui, PLAYS_W, "P");
                    header_cell(ui, FAV_W, "Save");
                    header_cell(ui, ACTION_W, "Go");
                });
            });

        ui.add_space(6.0);

        let mut select_label: Option<String> = None;
        let mut toggle_fav: Option<String> = None;
        let mut to_activate: Option<Track> = None;

        egui::ScrollArea::vertical()
            .auto_shrink([false; 2])
            .show_rows(ui, ROW_H + 4.0, visible.len(), |ui, range| {
                for row in range {
                    let track = &visible[row];
                    let song = self.song_index.get(&track.label);
                    let is_selected = self
                        .selected_label
                        .as_ref()
                        .map(|l| l == &track.label)
                        .unwrap_or(false);
                    let is_loaded = playback
                        .loaded_label
                        .as_ref()
                        .map(|l| l == &track.label)
                        .unwrap_or(false);
                    let is_pending = self
                        .pending_render
                        .as_ref()
                        .map(|p| p.track_label == track.label)
                        .unwrap_or(false);
                    let is_fav = self.favorites.contains(&track.label);
                    let plays = self.play_counts.get(&track.label).copied().unwrap_or(0);

                    let fill = if is_loaded && playback.is_playing {
                        egui::Color32::from_rgb(213, 229, 220)
                    } else if is_selected {
                        egui::Color32::from_rgb(225, 231, 223)
                    } else if row % 2 == 0 {
                        egui::Color32::from_rgb(248, 245, 239)
                    } else {
                        egui::Color32::from_rgb(241, 237, 229)
                    };

                    let (rect, row_resp) = ui.allocate_exact_size(
                        egui::vec2(ui.available_width(), ROW_H),
                        egui::Sense::click(),
                    );
                    ui.painter().rect_filled(rect, 6.0, fill);
                    let mut clicked_fav = false;
                    let mut clicked_play = false;
                    ui.allocate_new_ui(
                        egui::UiBuilder::new()
                            .max_rect(rect.shrink2(egui::vec2(10.0, 6.0)))
                            .layout(egui::Layout::left_to_right(egui::Align::Center)),
                        |ui| {
                            let state_text = if is_pending {
                                "Render"
                            } else if is_loaded && playback.is_playing {
                                "Live"
                            } else if is_loaded {
                                "Ready"
                            } else if self.recent_ids.contains(&track.label) {
                                "Recent"
                            } else {
                                "-"
                            };
                            cell_text(
                                ui,
                                STATUS_W,
                                egui::RichText::new(state_text)
                                    .small()
                                    .color(egui::Color32::from_rgb(79, 94, 86)),
                            );

                            let title = display_title(song, track);
                            let title_resp = ui.add_sized(
                                [title_w, 20.0],
                                egui::SelectableLabel::new(
                                    is_selected,
                                    egui::RichText::new(title)
                                        .strong()
                                        .color(egui::Color32::from_rgb(34, 38, 35)),
                                ),
                            );
                            if title_resp.clicked() {
                                select_label = Some(track.label.clone());
                            }
                            if title_resp.double_clicked() {
                                to_activate = Some(track.clone());
                            }

                            cell_text(
                                ui,
                                COMPOSER_W,
                                egui::RichText::new(display_composer(song))
                                    .color(egui::Color32::from_rgb(83, 87, 82)),
                            );
                            badge_cell(
                                ui,
                                ERA_W,
                                song.and_then(|s| s.era.as_deref()).unwrap_or("-"),
                                era_badge_color(song.and_then(|s| s.era.as_deref())),
                            );
                            cell_text(
                                ui,
                                INSTR_W,
                                egui::RichText::new(
                                    song.map(|s| s.instrument.as_str()).unwrap_or("-"),
                                )
                                .color(egui::Color32::from_rgb(83, 87, 82)),
                            );
                            cell_text(
                                ui,
                                SOURCE_W,
                                egui::RichText::new(source_bucket(track, song))
                                    .monospace()
                                    .color(egui::Color32::from_rgb(86, 90, 99)),
                            );
                            cell_text(
                                ui,
                                PLAYS_W,
                                egui::RichText::new(plays.to_string())
                                    .monospace()
                                    .color(egui::Color32::from_rgb(83, 87, 82)),
                            );

                            let fav_resp = ui.add_sized(
                                [FAV_W, 20.0],
                                egui::Button::new(if is_fav { "On" } else { "Save" })
                                    .fill(egui::Color32::from_rgb(232, 226, 215)),
                            );
                            if fav_resp.clicked() {
                                clicked_fav = true;
                                toggle_fav = Some(track.label.clone());
                            }

                            let play_label = if is_loaded && playback.is_playing {
                                "Stop"
                            } else {
                                "Play"
                            };
                            let play_resp = ui.add_sized(
                                [ACTION_W, 20.0],
                                egui::Button::new(play_label)
                                    .fill(egui::Color32::from_rgb(206, 221, 212)),
                            );
                            if play_resp.clicked() {
                                clicked_play = true;
                                to_activate = Some(track.clone());
                            }
                        },
                    );
                    if row_resp.clicked() && !clicked_fav && !clicked_play {
                        select_label = Some(track.label.clone());
                    }
                    if row_resp.double_clicked() && !clicked_fav && !clicked_play {
                        to_activate = Some(track.clone());
                    }
                    ui.add_space(4.0);
                }
            });

        if let Some(label) = select_label {
            self.selected_label = Some(label);
        }
        if let Some(label) = toggle_fav {
            self.toggle_favorite(&label);
        }
        if let Some(track) = to_activate {
            self.activate_track(track, playback);
        }
    }
}

impl eframe::App for JukeboxLiteC {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_cp_commands();
        self.poll_pending_render();
        self.poll_pending_audit();

        let visible = self.visible_tracks();
        self.ensure_selection(&visible);
        let playback = self.playback_snapshot();

        egui::TopBottomPanel::top("jukebox_lite_c_header")
            .exact_height(78.0)
            .show(ctx, |ui| self.show_header(ui, visible.len()));

        egui::TopBottomPanel::bottom("jukebox_lite_c_transport")
            .exact_height(92.0)
            .show(ctx, |ui| self.show_transport(ui, &playback));

        egui::SidePanel::left("jukebox_lite_c_sidebar")
            .resizable(false)
            .exact_width(190.0)
            .show(ctx, |ui| self.show_sidebar(ui));

        egui::SidePanel::right("jukebox_lite_c_inspector")
            .resizable(false)
            .exact_width(320.0)
            .show(ctx, |ui| self.show_inspector(ui, &playback));

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(8.0);
            self.show_table(ui, &visible, &playback);
        });

        self.publish_cp_snapshot(&visible, &playback);

        if playback.is_playing || self.pending_render.is_some() || self.pending_audit.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(120));
        }
    }
}

fn stat_chip(ui: &mut egui::Ui, label: &str, value: String, fill: egui::Color32) {
    egui::Frame::none()
        .fill(fill)
        .rounding(6.0)
        .inner_margin(egui::Margin::symmetric(8.0, 4.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(label)
                        .small()
                        .color(egui::Color32::from_rgb(96, 100, 93)),
                );
                ui.label(
                    egui::RichText::new(value)
                        .strong()
                        .color(egui::Color32::from_rgb(45, 48, 45)),
                );
            });
        });
}

fn routing_row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(label)
                .small()
                .color(egui::Color32::from_rgb(72, 76, 74)),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                egui::RichText::new(value)
                    .small()
                    .monospace()
                    .color(egui::Color32::from_rgb(108, 111, 108)),
            );
        });
    });
}

fn inspector_key(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text)
            .small()
            .color(egui::Color32::from_rgb(96, 100, 93)),
    );
}

fn detail_chip(ui: &mut egui::Ui, text: &str, fill: egui::Color32) {
    egui::Frame::none()
        .fill(fill)
        .rounding(6.0)
        .inner_margin(egui::Margin::symmetric(8.0, 4.0))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(text)
                    .small()
                    .color(egui::Color32::from_rgb(56, 58, 55)),
            );
        });
}

fn meter_row(ui: &mut egui::Ui, label: &str, value: f32, fill: egui::Color32) {
    ui.horizontal(|ui| {
        ui.add_sized(
            [62.0, 18.0],
            egui::Label::new(
                egui::RichText::new(label)
                    .small()
                    .color(egui::Color32::from_rgb(96, 100, 93)),
            ),
        );
        ui.add(
            egui::ProgressBar::new(value.clamp(0.0, 1.0))
                .desired_width(ui.available_width() - 10.0)
                .fill(fill),
        );
    });
}

fn draw_envelope(ui: &mut egui::Ui, values: &[f32]) {
    let desired = egui::vec2(ui.available_width(), 44.0);
    let (rect, _) = ui.allocate_exact_size(desired, egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 6.0, egui::Color32::from_rgb(240, 236, 228));
    if values.len() < 2 {
        return;
    }
    let mut points = Vec::with_capacity(values.len());
    for (i, value) in values.iter().enumerate() {
        let t = i as f32 / (values.len() - 1) as f32;
        let x = egui::lerp(rect.left()..=rect.right(), t);
        let y = egui::lerp(rect.bottom()..=rect.top(), value.clamp(0.0, 1.0));
        points.push(egui::pos2(x, y));
    }
    painter.add(egui::Shape::line(
        points,
        egui::Stroke::new(1.6, egui::Color32::from_rgb(74, 116, 104)),
    ));
}

fn header_cell(ui: &mut egui::Ui, width: f32, text: &str) {
    ui.add_sized(
        [width, 18.0],
        egui::Label::new(
            egui::RichText::new(text)
                .small()
                .strong()
                .color(egui::Color32::from_rgb(52, 58, 54)),
        ),
    );
}

fn cell_text(ui: &mut egui::Ui, width: f32, text: egui::RichText) {
    ui.add_sized([width, 18.0], egui::Label::new(text));
}

fn badge_cell(ui: &mut egui::Ui, width: f32, text: &str, fill: egui::Color32) {
    ui.allocate_ui_with_layout(
        egui::vec2(width, 20.0),
        egui::Layout::left_to_right(egui::Align::Center),
        |ui| {
            egui::Frame::none()
                .fill(fill)
                .rounding(6.0)
                .inner_margin(egui::Margin::symmetric(6.0, 2.0))
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new(text)
                            .small()
                            .color(egui::Color32::from_rgb(31, 34, 31)),
                    );
                });
        },
    );
}

fn apply_visual_style(ctx: &egui::Context) {
    ctx.set_theme(egui::ThemePreference::Light);

    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(10.0, 5.0);
    style.spacing.interact_size.y = 24.0;
    style.text_styles.insert(
        egui::TextStyle::Heading,
        egui::FontId::new(24.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Body,
        egui::FontId::new(14.5, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Button,
        egui::FontId::new(13.5, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Small,
        egui::FontId::new(11.5, egui::FontFamily::Proportional),
    );
    ctx.set_style(style);

    let mut visuals = egui::Visuals::light();
    visuals.override_text_color = Some(egui::Color32::from_rgb(37, 40, 37));
    visuals.panel_fill = egui::Color32::from_rgb(236, 232, 224);
    visuals.window_fill = egui::Color32::from_rgb(246, 243, 237);
    visuals.extreme_bg_color = egui::Color32::from_rgb(228, 224, 214);
    visuals.selection.bg_fill = egui::Color32::from_rgb(91, 126, 112);
    visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(246, 244, 240));
    visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(240, 236, 228);
    visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(239, 235, 226);
    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(231, 227, 217);
    visuals.widgets.active.bg_fill = egui::Color32::from_rgb(221, 229, 220);
    ctx.set_visuals(visuals);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dirs = vec![
        PathBuf::from("bench-out/songs"),
        PathBuf::from("bench-out/CHIPTUNE"),
        PathBuf::from("bench-out/iterH"),
    ];
    let app = JukeboxLiteC::new(dirs)?;
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1380.0, 840.0])
            .with_title("keysynth jukebox lite c"),
        ..Default::default()
    };
    eframe::run_native(
        "keysynth jukebox lite c",
        options,
        Box::new(|cc| {
            keysynth::ui::setup_japanese_fonts(&cc.egui_ctx);
            apply_visual_style(&cc.egui_ctx);
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| -> Box<dyn std::error::Error> { format!("eframe: {e}").into() })?;
    Ok(())
}

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
        assert_eq!(pick_voice(Tile::All, Some(&song), &track), "guitar-stk");
    }

    #[test]
    fn rendered_audio_uses_direct_route() {
        let track = Track {
            path: PathBuf::from("foo.wav"),
            label: "foo".into(),
            format: Format::Wav,
        };
        assert_eq!(pick_voice(Tile::All, None, &track), "direct");
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
        let dir = std::env::temp_dir().join("keysynth_jukebox_lite_c_dedup_test");
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

    #[test]
    fn humanize_ago_picks_natural_unit() {
        assert_eq!(humanize_ago(15), "15s ago");
        assert_eq!(humanize_ago(90), "1m ago");
        assert_eq!(humanize_ago(7_200), "2h ago");
        assert_eq!(humanize_ago(172_800), "2d ago");
    }
}
