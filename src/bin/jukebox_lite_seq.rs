//! jukebox_lite_seq — list-view browser + live BMS-style sequencer view.
//!
//! Two modes share one window:
//!
//! * `ViewMode::List` inherits the Codex `jukebox_lite_c` browser:
//!   filter rail, ledger track list, detail inspector, audio audit. WAV /
//!   MP3 stream straight; MIDI rows hand off to the existing `render_midi`
//!   preview cache so the static-asset path keeps working for the casual
//!   "play this file" flow.
//!
//! * `ViewMode::Sequencer` is the new live-paradigm window. It parses the
//!   selected MIDI in-process, draws a pitch × time note grid, and drives
//!   `voices_live/*` cdylibs **directly** through `LiveFactory::make_voice`
//!   — no `render_midi` subprocess, no preview cache. Voice swaps,
//!   mute/solo, and tempo changes take effect mid-playback because they
//!   hit the audio thread immediately through `ArcSwap` / `Mutex` reads
//!   at the next sample-block boundary.
//!
//! Backend reuse: `library_db`, `play_log`, `preview_cache`, `gui_cp`,
//! `ui::setup_japanese_fonts`, `audio_audit`, and `live_reload`.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use arc_swap::ArcSwap;
use cp_core::RpcError;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use eframe::egui;
use hound::{SampleFormat as WavSampleFormat, WavReader};
use midly::{MetaMessage, MidiMessage, Smf, Timing, TrackEventKind};
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
use keysynth::live_reload::{build_and_load, LiveFactory};
use keysynth::play_log::{PlayEntry, PlayLogDb};
use keysynth::synth::{midi_to_freq, VoiceImpl};

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
    let err_fn = |err| eprintln!("jukebox_lite_seq audio stream error: {err}");
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
        .ok_or_else(|| "render_midi binary not found alongside jukebox_lite_seq".to_string())?;
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
                "jukebox_lite_seq: library_db open failed ({}): {e} — running without DB metadata",
                db_path.display()
            );
            return (HashMap::new(), 0);
        }
    };
    if let Err(e) = db.migrate() {
        eprintln!("jukebox_lite_seq: library_db migrate failed: {e}");
        return (HashMap::new(), 0);
    }
    if manifest.is_file() {
        if let Err(e) = db.import_songs(&manifest) {
            eprintln!("jukebox_lite_seq: library_db import_songs failed: {e}");
        }
    }
    if voices_live.is_dir() {
        if let Err(e) = db.import_voices(&voices_live) {
            eprintln!("jukebox_lite_seq: library_db import_voices failed: {e}");
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
            eprintln!("jukebox_lite_seq: play_log open failed: {e}");
            return None;
        }
    };
    if let Err(e) = db.migrate() {
        eprintln!("jukebox_lite_seq: play_log migrate failed: {e}");
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
    let endpoint = gui_cp::resolve_endpoint("jukebox_lite_seq", None);
    let st_get = state.clone();
    let st_load = state.clone();
    let st_play = state.clone();
    let st_stop = state.clone();
    let st_tile = state.clone();
    let st_rescan = state.clone();
    let st_list = state.clone();
    gui_cp::Builder::new("jukebox_lite_seq", &endpoint)
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

// ---------------------------------------------------------------------------
// Sequencer view: live in-process playback through voices_live cdylibs.
//
// The List view above streams pre-rendered audio off disk. This block is
// the live counterpart: a parsed SMF turns into per-track NoteEvents, and
// the audio thread fires `LiveFactory::make_voice` directly on each
// note_on. Voice swap, tempo, and mute/solo all read shared state at
// every render-block boundary, so the user hears the change inside one
// audio buffer (~6 ms at 256 frames / 44.1 kHz).
// ---------------------------------------------------------------------------

const SEQ_VOICE_CAP: usize = 64;
const SEQ_RELEASE_TAIL_SEC: f64 = 1.5;

#[derive(Clone, Debug)]
struct SeqNote {
    track_idx: u8,
    midi_note: u8,
    velocity: u8,
    /// Native-tempo event time (seconds from song start). Tempo slider
    /// rescales how fast the playhead walks this axis at playback time;
    /// the events themselves stay in native tempo so the slider can
    /// move in either direction without round-trip drift.
    start_sec: f64,
    duration_sec: f64,
}

#[derive(Clone, Debug)]
struct SeqProgram {
    notes: Arc<Vec<SeqNote>>,
    track_count: usize,
    end_sec: f64,
    pitch_min: u8,
    pitch_max: u8,
    label: String,
}

impl SeqProgram {
    fn empty() -> Self {
        Self {
            notes: Arc::new(Vec::new()),
            track_count: 0,
            end_sec: 0.0,
            pitch_min: 60,
            pitch_max: 72,
            label: String::new(),
        }
    }
}

/// Parse a Standard MIDI File into a flat, sorted `SeqNote` list with
/// per-track tags preserved. Mirrors `render_midi::parse_smf` but keeps
/// the track index and emits f64 seconds without applying tempo_scale —
/// the live engine multiplies tempo at playback time.
fn parse_smf_for_seq(bytes: &[u8]) -> Result<SeqProgram, String> {
    let smf = Smf::parse(bytes).map_err(|e| format!("midly parse: {e}"))?;
    let ppq: u32 = match smf.header.timing {
        Timing::Metrical(t) => t.as_int() as u32,
        Timing::Timecode(_, _) => return Err("SMPTE timing not supported".into()),
    };

    #[derive(Clone, Copy)]
    enum Kind {
        On(u8, u8, u8, u8),
        Off(u8, u8, u8),
        Tempo(u32),
    }
    let mut events: Vec<(u64, usize, Kind)> = Vec::new();
    for (track_idx, track) in smf.tracks.iter().enumerate() {
        let mut tick: u64 = 0;
        for ev in track {
            tick += ev.delta.as_int() as u64;
            match ev.kind {
                TrackEventKind::Midi { channel, message } => match message {
                    MidiMessage::NoteOn { key, vel } => {
                        let v = vel.as_int();
                        if v > 0 {
                            events.push((
                                tick,
                                track_idx,
                                Kind::On(channel.as_int(), key.as_int(), v, track_idx as u8),
                            ));
                        } else {
                            events.push((
                                tick,
                                track_idx,
                                Kind::Off(channel.as_int(), key.as_int(), track_idx as u8),
                            ));
                        }
                    }
                    MidiMessage::NoteOff { key, .. } => {
                        events.push((
                            tick,
                            track_idx,
                            Kind::Off(channel.as_int(), key.as_int(), track_idx as u8),
                        ));
                    }
                    _ => {}
                },
                TrackEventKind::Meta(MetaMessage::Tempo(t)) => {
                    events.push((tick, track_idx, Kind::Tempo(t.as_int())));
                }
                _ => {}
            }
        }
    }
    events.sort_by_key(|(t, _, _)| *t);

    let mut us_per_q: u32 = 500_000;
    let mut last_tick: u64 = 0;
    let mut now_sec: f64 = 0.0;
    let mut active: HashMap<(u8, u8, u8), (f64, u8)> = HashMap::new();
    let mut out: Vec<SeqNote> = Vec::new();
    for (tick, _, kind) in &events {
        if *tick > last_tick {
            let delta_ticks = (*tick - last_tick) as f64;
            let sec_per_tick = (us_per_q as f64) / (ppq as f64 * 1_000_000.0);
            now_sec += delta_ticks * sec_per_tick;
            last_tick = *tick;
        }
        match *kind {
            Kind::Tempo(t) => us_per_q = t,
            Kind::On(ch, n, v, t_idx) => {
                active.insert((t_idx, ch, n), (now_sec, v));
            }
            Kind::Off(ch, n, t_idx) => {
                if let Some((start, vel)) = active.remove(&(t_idx, ch, n)) {
                    let dur = (now_sec - start).max(0.005);
                    out.push(SeqNote {
                        track_idx: t_idx,
                        midi_note: n,
                        velocity: vel,
                        start_sec: start,
                        duration_sec: dur,
                    });
                }
            }
        }
    }
    for ((t_idx, _ch, n), (start, vel)) in active {
        let dur = (now_sec - start).max(0.005);
        out.push(SeqNote {
            track_idx: t_idx,
            midi_note: n,
            velocity: vel,
            start_sec: start,
            duration_sec: dur,
        });
    }
    out.sort_by(|a, b| {
        a.start_sec
            .partial_cmp(&b.start_sec)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let track_count = out
        .iter()
        .map(|n| n.track_idx as usize + 1)
        .max()
        .unwrap_or(0);
    let end_sec = out
        .iter()
        .map(|n| n.start_sec + n.duration_sec)
        .fold(0.0, f64::max);
    let pitch_min = out.iter().map(|n| n.midi_note).min().unwrap_or(60);
    let pitch_max = out.iter().map(|n| n.midi_note).max().unwrap_or(72);

    Ok(SeqProgram {
        notes: Arc::new(out),
        track_count: track_count.max(1),
        end_sec,
        pitch_min,
        pitch_max,
        label: String::new(),
    })
}

struct ActiveSeqVoice {
    voice: Box<dyn VoiceImpl + Send>,
    /// Carried even though unread because future feature ideas (per-
    /// track gain, per-voice cleanup-by-track) need it; tagging here
    /// is virtually free vs. backfilling later from the note list.
    #[allow(dead_code)]
    track_idx: u8,
    #[allow(dead_code)]
    midi_note: u8,
    release_at_sec: f64,
    released: bool,
}

#[derive(Clone, Debug)]
struct SeqStatus {
    is_playing: bool,
    playhead_sec: f64,
    end_sec: f64,
    tempo_scale: f32,
    voice_count: usize,
    #[allow(dead_code)]
    track_count: usize,
}

struct SeqMutState {
    is_playing: bool,
    playhead_sec: f64,
    next_idx: usize,
    program: SeqProgram,
    voices: Vec<ActiveSeqVoice>,
    track_mute: Vec<bool>,
    track_solo: Vec<bool>,
    /// Per-track voice slug. `None` means "use master". Audio thread
    /// looks this up against the shared `VoicePool` on every note_on
    /// — present in pool ⇒ that factory makes the voice; absent ⇒
    /// fall through to master. This is the "each channel has its own
    /// engine" feature: switching a track's voice is just a pool
    /// lookup, no rebuild.
    track_voice_slugs: Vec<Option<String>>,
    tempo_scale: f32,
    master: f32,
}

impl SeqMutState {
    fn new() -> Self {
        Self {
            is_playing: false,
            playhead_sec: 0.0,
            next_idx: 0,
            program: SeqProgram::empty(),
            voices: Vec::with_capacity(SEQ_VOICE_CAP),
            track_mute: Vec::new(),
            track_solo: Vec::new(),
            track_voice_slugs: Vec::new(),
            tempo_scale: 1.0,
            master: 0.55,
        }
    }

    fn any_solo(&self) -> bool {
        self.track_solo.iter().any(|&s| s)
    }

    fn track_audible(&self, track_idx: u8) -> bool {
        let i = track_idx as usize;
        if self.track_mute.get(i).copied().unwrap_or(false) {
            return false;
        }
        if self.any_solo() {
            return self.track_solo.get(i).copied().unwrap_or(false);
        }
        true
    }

    fn ensure_track_vecs(&mut self, n: usize) {
        if self.track_mute.len() < n {
            self.track_mute.resize(n, false);
        }
        if self.track_solo.len() < n {
            self.track_solo.resize(n, false);
        }
        if self.track_voice_slugs.len() < n {
            self.track_voice_slugs.resize(n, None);
        }
    }
}

/// Memory-resident pool of every `LiveFactory` we've ever loaded. Once
/// a voice is in here it stays there, so re-selecting it is an
/// `ArcSwap::store` (or a HashMap lookup for per-track), never a
/// `cargo build`. Shared between the GUI thread (writes on load
/// completion) and the audio thread (reads on each note_on for
/// per-track lookup).
type VoicePool = Arc<Mutex<HashMap<String, Arc<LiveFactory>>>>;

#[derive(Clone, Debug, PartialEq, Eq)]
enum VoiceSlotState {
    /// Never requested. UI shows this as "—".
    Idle,
    /// Build / load is currently running on a worker thread.
    Loading,
    /// Factory is in the pool; ready to use instantly.
    Ready,
    /// Build failed; the message is surfaced in the dropdown tooltip.
    Failed(String),
}

// SeqVoiceLoadStatus replaced by per-slot VoiceSlotState above; the
// dropdown reads the slot map directly so every option shows ✓ /
// spinner / × independently rather than a single global status line.

struct SeqEngine {
    shared: Arc<Mutex<SeqMutState>>,
    /// Master voice — used for any track whose `track_voice_slugs[i]`
    /// is `None` (or whose slug is not yet in the pool). Hot-swappable
    /// via `ArcSwap` so re-selecting in the master dropdown is one
    /// pointer store.
    factory: Arc<ArcSwap<Option<Arc<LiveFactory>>>>,
    /// Memory-resident voice pool. Both the master swap above and
    /// per-track lookups in the render block resolve against this.
    pool: VoicePool,
    /// State of every voice slot the user has touched (or that the
    /// background warm-up has triggered). The dropdown reads this to
    /// show ✓ / spinner / × per option without taking the pool lock.
    slot_states: Arc<Mutex<HashMap<String, VoiceSlotState>>>,
    #[allow(dead_code)]
    sr: u32,
    voice_load_rx: Receiver<SeqVoiceLoadEvent>,
    voice_load_tx: Sender<SeqVoiceLoadEvent>,
    /// Currently-active master voice slug. Set the moment we kick off
    /// a load (so a duplicate click is a no-op) and cleared if that
    /// load fails.
    current_voice: Option<String>,
    _stream: cpal::Stream,
}

enum SeqVoiceLoadEvent {
    Loaded {
        name: String,
        factory: Arc<LiveFactory>,
        took_ms: u128,
        /// `true` when this load was triggered by the master-voice
        /// dropdown; `false` for warm-up / per-track loads that just
        /// want the factory in the pool without rotating the master.
        set_as_master: bool,
    },
    Failed {
        name: String,
        message: String,
    },
}

fn seq_render_block(
    out: &mut [f32],
    channels: u16,
    state: &Arc<Mutex<SeqMutState>>,
    factory: &Arc<ArcSwap<Option<Arc<LiveFactory>>>>,
    pool: &VoicePool,
    sr: u32,
) {
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
    let frames = out.len() / channels.max(1) as usize;
    if frames == 0 {
        return;
    }
    let dt_wall = frames as f64 / sr as f64;
    let dt_event = dt_wall * st.tempo_scale.max(0.05) as f64;

    let block_start = st.playhead_sec;
    let block_end = block_start + dt_event;
    let master_factory = factory.load_full();
    let master_factory: Option<Arc<LiveFactory>> =
        master_factory.as_ref().as_ref().cloned();

    // Snapshot pool once per render block. Audio thread takes the
    // mutex briefly; GUI rarely contends because writes only happen
    // when a load completes (rare event, < once per second worst case).
    let pool_snapshot: HashMap<String, Arc<LiveFactory>> = match pool.lock() {
        Ok(g) => g.clone(),
        Err(_) => HashMap::new(),
    };

    // Spawn note_ons whose start_sec falls inside this block. We grab
    // one Arc<Vec<SeqNote>> snapshot up front; mutating commands swap a
    // fresh Arc rather than editing the existing one.
    let notes_arc = st.program.notes.clone();
    {
        let mut idx = st.next_idx;
        while idx < notes_arc.len() {
            let n = &notes_arc[idx];
            if n.start_sec >= block_end {
                break;
            }
            if n.start_sec >= block_start && st.track_audible(n.track_idx) {
                if st.voices.len() >= SEQ_VOICE_CAP {
                    st.voices.remove(0);
                }
                // Resolve the factory for this track: per-track slug
                // first (if both set and present in the pool), then
                // master. Falling through silently when no factory is
                // available is intentional — the user just hasn't
                // built any voice yet, and the alternative is a
                // crash.
                let track_slug = st
                    .track_voice_slugs
                    .get(n.track_idx as usize)
                    .and_then(|s| s.as_ref());
                let factory_for_note: Option<&Arc<LiveFactory>> = track_slug
                    .and_then(|slug| pool_snapshot.get(slug.as_str()))
                    .or(master_factory.as_ref());
                if let Some(fac) = factory_for_note {
                    let freq = midi_to_freq(n.midi_note);
                    let v = fac.make_voice(sr as f32, freq, n.velocity);
                    st.voices.push(ActiveSeqVoice {
                        voice: v,
                        track_idx: n.track_idx,
                        midi_note: n.midi_note,
                        release_at_sec: n.start_sec + n.duration_sec,
                        released: false,
                    });
                }
            }
            idx += 1;
        }
        st.next_idx = idx;
    }

    // Trigger releases for any active voice whose release-time falls
    // inside this block. We compare against block_end (the playhead at
    // end of block) because firing a hair early at the block boundary
    // is inaudible and saves a per-sample inner branch.
    for v in st.voices.iter_mut() {
        if !v.released && v.release_at_sec <= block_end {
            v.voice.trigger_release();
            v.released = true;
        }
    }

    // Mix all active voices into the (mono) output. We allocate a tmp
    // mono buffer per call; the alternative is per-voice rendering into
    // `out` interleaved which means each voice has to know about
    // channel layout. With at most SEQ_VOICE_CAP=64 voices and ~256
    // frames, the alloc cost is negligible.
    let mut mono = vec![0.0_f32; frames];
    let mut tmp = vec![0.0_f32; frames];
    for v in st.voices.iter_mut() {
        tmp.fill(0.0);
        v.voice.render_add(&mut tmp);
        for (m, t) in mono.iter_mut().zip(tmp.iter()) {
            *m += *t;
        }
    }
    let master = st.master;
    for s in mono.iter_mut() {
        *s = (*s * master).tanh();
    }

    // Evict voices that have rendered themselves to silence (or held
    // notes whose release window is well past).
    st.voices.retain(|v| {
        let past_release =
            v.released && v.release_at_sec + SEQ_RELEASE_TAIL_SEC < block_end;
        !past_release && !v.voice.is_done()
    });

    // Fan out mono → output channel layout.
    for f in 0..frames {
        let s = mono[f];
        let base = f * channels as usize;
        match channels {
            1 => out[base] = s,
            2 => {
                out[base] = s;
                out[base + 1] = s;
            }
            n => {
                out[base] = s;
                out[base + 1] = s;
                for c in 2..n as usize {
                    out[base + c] = 0.0;
                }
            }
        }
    }

    st.playhead_sec = block_end;
    if st.playhead_sec >= st.program.end_sec + SEQ_RELEASE_TAIL_SEC {
        st.is_playing = false;
        st.next_idx = 0;
        st.playhead_sec = 0.0;
        st.voices.clear();
    }
}

impl SeqEngine {
    fn build() -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "no default output device for sequencer".to_string())?;
        let supported = device
            .default_output_config()
            .map_err(|e| format!("default_output_config: {e}"))?;
        let sample_format = supported.sample_format();
        let channels = supported.channels();
        let stream_cfg: StreamConfig = supported.into();
        let sr = stream_cfg.sample_rate.0;

        let shared = Arc::new(Mutex::new(SeqMutState::new()));
        let factory: Arc<ArcSwap<Option<Arc<LiveFactory>>>> =
            Arc::new(ArcSwap::from_pointee(None));
        let pool: VoicePool = Arc::new(Mutex::new(HashMap::new()));

        let err_fn = |e| eprintln!("jukebox_lite_seq sequencer audio error: {e}");
        let stream = match sample_format {
            SampleFormat::F32 => {
                let st = shared.clone();
                let fac = factory.clone();
                let pl = pool.clone();
                device
                    .build_output_stream(
                        &stream_cfg,
                        move |out: &mut [f32], _| {
                            seq_render_block(out, channels, &st, &fac, &pl, sr);
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| format!("build f32: {e}"))?
            }
            SampleFormat::I16 => {
                let st = shared.clone();
                let fac = factory.clone();
                let pl = pool.clone();
                let mut scratch = Vec::<f32>::new();
                device
                    .build_output_stream(
                        &stream_cfg,
                        move |out: &mut [i16], _| {
                            if scratch.len() != out.len() {
                                scratch.resize(out.len(), 0.0);
                            }
                            seq_render_block(&mut scratch, channels, &st, &fac, &pl, sr);
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
                let st = shared.clone();
                let fac = factory.clone();
                let pl = pool.clone();
                let mut scratch = Vec::<f32>::new();
                device
                    .build_output_stream(
                        &stream_cfg,
                        move |out: &mut [u16], _| {
                            if scratch.len() != out.len() {
                                scratch.resize(out.len(), 0.0);
                            }
                            seq_render_block(&mut scratch, channels, &st, &fac, &pl, sr);
                            for (dst, src) in out.iter_mut().zip(scratch.iter()) {
                                *dst =
                                    (((src.clamp(-1.0, 1.0) + 1.0) * 0.5) * u16::MAX as f32) as u16;
                            }
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| format!("build u16: {e}"))?
            }
            other => return Err(format!("unsupported sample format: {other:?}")),
        };
        stream.play().map_err(|e| format!("seq stream.play: {e}"))?;

        let (tx, rx) = std::sync::mpsc::channel::<SeqVoiceLoadEvent>();
        Ok(Self {
            shared,
            factory,
            pool,
            slot_states: Arc::new(Mutex::new(HashMap::new())),
            sr,
            voice_load_rx: rx,
            voice_load_tx: tx,
            current_voice: None,
            _stream: stream,
        })
    }

    /// User-visible status string for the currently-active master
    /// voice. Reads slot_states (cheap) so the dropdown renders every
    /// frame without contending the pool mutex.
    #[allow(dead_code)]
    fn status_label(&self) -> String {
        let slug = match self.current_voice.as_ref() {
            Some(s) => s.clone(),
            None => return "no master voice".into(),
        };
        let state = self
            .slot_states
            .lock()
            .ok()
            .and_then(|g| g.get(&slug).cloned());
        match state {
            Some(VoiceSlotState::Ready) => format!("✓ {slug}"),
            Some(VoiceSlotState::Loading) => format!("building {slug}..."),
            Some(VoiceSlotState::Failed(msg)) => format!("× {slug}: {msg}"),
            _ => format!("? {slug}"),
        }
    }

    fn slot_state(&self, slug: &str) -> VoiceSlotState {
        self.slot_states
            .lock()
            .ok()
            .and_then(|g| g.get(slug).cloned())
            .unwrap_or(VoiceSlotState::Idle)
    }

    fn pool_has(&self, slug: &str) -> bool {
        self.pool.lock().map(|g| g.contains_key(slug)).unwrap_or(false)
    }

    /// Fast path for "voice already in pool" (no rebuild). Returns
    /// `true` if the master swap could be satisfied from cache; the
    /// caller can skip the load thread entirely in that case.
    fn try_set_master_from_pool(&mut self, slug: &str) -> bool {
        let factory = match self.pool.lock() {
            Ok(g) => g.get(slug).cloned(),
            Err(_) => None,
        };
        if let Some(f) = factory {
            self.factory.store(Arc::new(Some(f)));
            self.current_voice = Some(slug.to_string());
            return true;
        }
        false
    }

    /// Request `slug` to be loaded into the pool. If it's already
    /// there, swap the master pointer and return immediately
    /// (no thread spawn). If it's currently building, do nothing.
    /// Otherwise spawn a background `build_and_load`.
    ///
    /// `set_as_master` controls whether successful load also rotates
    /// the master ArcSwap to this voice; warm-up calls pass `false`.
    fn ensure_voice_loaded(&mut self, slug: &str, set_as_master: bool) {
        if self.try_set_master_from_pool(slug) && set_as_master {
            // Cache hit + master rotation done in try_set_master_from_pool.
            return;
        }
        // Already loaded but caller didn't want to rotate master:
        // mark slot Ready and bail.
        if self.pool_has(slug) {
            if let Ok(mut g) = self.slot_states.lock() {
                g.insert(slug.into(), VoiceSlotState::Ready);
            }
            return;
        }
        // Already building? Don't double-spawn.
        if matches!(self.slot_state(slug), VoiceSlotState::Loading) {
            if set_as_master {
                self.current_voice = Some(slug.to_string());
            }
            return;
        }

        let crate_root = PathBuf::from("voices_live").join(slug);
        let name = slug.to_string();
        if let Ok(mut g) = self.slot_states.lock() {
            g.insert(name.clone(), VoiceSlotState::Loading);
        }
        if set_as_master {
            // Tentatively claim master so duplicate clicks don't
            // re-spawn. Actual ArcSwap update happens in poll once
            // the build lands.
            self.current_voice = Some(name.clone());
        }

        let tx = self.voice_load_tx.clone();
        let name_for_thread = name;
        let set_master_for_thread = set_as_master;
        std::thread::Builder::new()
            .name("jukebox-lite-seq-voice".into())
            .spawn(move || {
                let started = Instant::now();
                match build_and_load(&crate_root) {
                    Ok(f) => {
                        let _ = tx.send(SeqVoiceLoadEvent::Loaded {
                            name: name_for_thread,
                            factory: Arc::new(f),
                            took_ms: started.elapsed().as_millis(),
                            set_as_master: set_master_for_thread,
                        });
                    }
                    Err(e) => {
                        let _ = tx.send(SeqVoiceLoadEvent::Failed {
                            name: name_for_thread,
                            message: e,
                        });
                    }
                }
            })
            .expect("spawn seq voice loader");
    }

    fn poll_voice_loads(&mut self) {
        loop {
            let ev = match self.voice_load_rx.try_recv() {
                Ok(e) => e,
                Err(_) => return,
            };
            match ev {
                SeqVoiceLoadEvent::Loaded {
                    name,
                    factory,
                    took_ms,
                    set_as_master,
                } => {
                    if let Ok(mut g) = self.pool.lock() {
                        g.insert(name.clone(), factory.clone());
                    }
                    if let Ok(mut g) = self.slot_states.lock() {
                        g.insert(name.clone(), VoiceSlotState::Ready);
                    }
                    if set_as_master {
                        self.factory.store(Arc::new(Some(factory)));
                    }
                    eprintln!(
                        "jukebox_lite_seq: pool += '{name}' ({took_ms} ms){}",
                        if set_as_master { " [master]" } else { "" }
                    );
                }
                SeqVoiceLoadEvent::Failed { name, message } => {
                    eprintln!(
                        "jukebox_lite_seq: voice '{name}' failed to load: {message}"
                    );
                    if let Ok(mut g) = self.slot_states.lock() {
                        g.insert(name.clone(), VoiceSlotState::Failed(message));
                    }
                    if self.current_voice.as_deref() == Some(name.as_str()) {
                        self.current_voice = None;
                    }
                }
            }
        }
    }

    fn load_program(&self, program: SeqProgram) {
        if let Ok(mut st) = self.shared.lock() {
            st.program = program.clone();
            st.ensure_track_vecs(program.track_count);
            st.next_idx = 0;
            st.playhead_sec = 0.0;
            st.voices.clear();
            st.is_playing = false;
        }
    }

    fn play(&self) {
        if let Ok(mut st) = self.shared.lock() {
            if st.program.notes.is_empty() {
                return;
            }
            // If the playhead has already run past the end, rewind so a
            // second Play after auto-stop replays from the top instead
            // of silently doing nothing.
            if st.playhead_sec >= st.program.end_sec {
                st.playhead_sec = 0.0;
                st.next_idx = 0;
                st.voices.clear();
            }
            st.is_playing = true;
        }
    }

    fn pause(&self) {
        if let Ok(mut st) = self.shared.lock() {
            st.is_playing = false;
            // Trigger release on every still-held voice so paused notes
            // ring out instead of cutting hard. The voices keep
            // existing in `st.voices` so resume continues their decay
            // tails seamlessly.
            for v in st.voices.iter_mut() {
                if !v.released {
                    v.voice.trigger_release();
                    v.released = true;
                }
            }
        }
    }

    fn stop(&self) {
        if let Ok(mut st) = self.shared.lock() {
            st.is_playing = false;
            st.playhead_sec = 0.0;
            st.next_idx = 0;
            st.voices.clear();
        }
    }

    fn seek(&self, sec: f64) {
        if let Ok(mut st) = self.shared.lock() {
            let target = sec.clamp(0.0, st.program.end_sec);
            st.playhead_sec = target;
            // Re-find next event index so spawning resumes correctly.
            st.next_idx = st
                .program
                .notes
                .iter()
                .position(|n| n.start_sec >= target)
                .unwrap_or(st.program.notes.len());
            // Pre-existing held voices are stale relative to a seek;
            // cut them so the user hears the new position cleanly.
            st.voices.clear();
        }
    }

    fn set_tempo(&self, tempo_scale: f32) {
        if let Ok(mut st) = self.shared.lock() {
            st.tempo_scale = tempo_scale.clamp(0.05, 4.0);
        }
    }

    #[allow(dead_code)]
    fn set_master(&self, master: f32) {
        if let Ok(mut st) = self.shared.lock() {
            st.master = master.clamp(0.0, 1.5);
        }
    }

    fn toggle_mute(&self, track: usize) {
        if let Ok(mut st) = self.shared.lock() {
            st.ensure_track_vecs(track + 1);
            let v = !st.track_mute[track];
            st.track_mute[track] = v;
        }
    }

    fn toggle_solo(&self, track: usize) {
        if let Ok(mut st) = self.shared.lock() {
            st.ensure_track_vecs(track + 1);
            let v = !st.track_solo[track];
            st.track_solo[track] = v;
        }
    }

    fn snapshot_status(&self) -> SeqStatus {
        match self.shared.lock() {
            Ok(st) => SeqStatus {
                is_playing: st.is_playing,
                playhead_sec: st.playhead_sec,
                end_sec: st.program.end_sec,
                tempo_scale: st.tempo_scale,
                voice_count: st.voices.len(),
                track_count: st.program.track_count,
            },
            Err(_) => SeqStatus {
                is_playing: false,
                playhead_sec: 0.0,
                end_sec: 0.0,
                tempo_scale: 1.0,
                voice_count: 0,
                track_count: 0,
            },
        }
    }

    fn snapshot_program(&self) -> SeqProgram {
        match self.shared.lock() {
            Ok(st) => st.program.clone(),
            Err(_) => SeqProgram::empty(),
        }
    }

    fn snapshot_track_flags(&self) -> (Vec<bool>, Vec<bool>) {
        match self.shared.lock() {
            Ok(st) => (st.track_mute.clone(), st.track_solo.clone()),
            Err(_) => (Vec::new(), Vec::new()),
        }
    }

    #[allow(dead_code)]
    fn current_voice(&self) -> Option<&str> {
        self.current_voice.as_deref()
    }
}

/// Voices the dropdown surfaces. Slugs match `voices_live/<slug>/`.
const SEQ_VOICE_CHOICES: &[(&str, &str)] = &[
    ("guitar_stk", "Guitar (STK)"),
    ("guitar", "Guitar (KS)"),
    ("piano_modal", "Piano Modal"),
    ("piano_lite", "Piano Lite"),
    ("piano", "Piano"),
    ("piano_thick", "Piano Thick"),
    ("piano_5am", "Piano 5AM"),
];

struct JukeboxLiteSeq {
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

    // Sequencer view state. The engine owns its own cpal stream that
    // sits idle until the user enters the Sequencer tab and hits Play.
    seq: Option<SeqEngine>,
    seq_init_error: Option<String>,
    seq_loaded_label: Option<String>,
    seq_voice_slug: String,
    /// `--autoplay` flag: deferred until the first voice cdylib has
    /// finished building. We can't just call `seq.play()` at boot
    /// because the factory may still be `None`; the audio thread would
    /// silently swallow note_ons until the build lands.
    seq_autoplay_pending: bool,
}

impl JukeboxLiteSeq {
    fn new(dirs: Vec<PathBuf>) -> Result<Self, String> {
        let (song_index, db_voice_count) = load_library_index();
        let dir_refs: Vec<&Path> = dirs.iter().map(|p| p.as_path()).collect();
        let tracks = scan_dirs(&dir_refs);
        println!(
            "jukebox_lite_seq: library_db catalog -> {} songs / {} voices ({} files on disk)",
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
                eprintln!("jukebox_lite_seq: CP server failed: {e}");
                None
            }
        };

        let initial_label = recent_entries
            .first()
            .map(|e| e.song_id.clone())
            .or_else(|| favorites.iter().next().cloned())
            .or_else(|| tracks.first().map(|t| t.label.clone()));

        // Bring the sequencer engine up and start warming every voice
        // in `SEQ_VOICE_CHOICES`. Each `ensure_voice_loaded` call
        // either swaps an in-memory factory in (cache hit, free) or
        // kicks a background `cargo build` (cold). Master rotation
        // happens for the first slug only; the rest just populate the
        // pool so subsequent dropdown picks are instant. Cargo's
        // incremental cache means the second time the user runs the
        // app every voice is "✓" within a few hundred ms.
        let (mut seq, seq_init_error) = match SeqEngine::build() {
            Ok(e) => (Some(e), None),
            Err(e) => {
                eprintln!("jukebox_lite_seq: sequencer engine init failed: {e}");
                (None, Some(e))
            }
        };
        if let Some(s) = seq.as_mut() {
            for (idx, (slug, _)) in SEQ_VOICE_CHOICES.iter().enumerate() {
                s.ensure_voice_loaded(slug, idx == 0);
            }
        }

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
            seq,
            seq_init_error,
            seq_loaded_label: None,
            seq_voice_slug: "guitar_stk".to_string(),
            seq_autoplay_pending: false,
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
            eprintln!("jukebox_lite_seq: toggle_favorite({label}): {e}");
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
                    eprintln!("jukebox_lite_seq: cache lookup {}: {e}", track.path.display());
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
            "jukebox_lite_seq: render queued ({}) — voice={voice}",
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
                    "jukebox_lite_seq: render done ({}) in {} ms",
                    wav.display(),
                    elapsed,
                );
                self.load_resolved(pending.track_label, wav, &pending.voice_id);
            }
            Err(e) => {
                eprintln!("jukebox_lite_seq: render failed: {e}");
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
                eprintln!("jukebox_lite_seq: open {}: {e}", playable.display());
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
                eprintln!("jukebox_lite_seq: record_play({label}): {e}");
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
                        eprintln!("jukebox_lite_seq: CP load_track unknown label={label}");
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
            ui.heading("keysynth / jukebox lite seq");
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
        // Branch on the selected row's format:
        //   - .mid  → drive the live `SeqEngine` (Play/Stop/tempo,
        //             progress comes from playhead vs end_sec).
        //   - .wav/.mp3 → drive the existing file-stream `Mixer`
        //             (Play/Stop/volume, progress from PCM cursor).
        // Both branches paint the same dark frame; the user reads
        // playing-state from the `Stop`/`Play` button label.
        let selected_format = self
            .selected_track()
            .map(|t| t.format)
            .unwrap_or(Format::Wav);
        match selected_format {
            Format::Mid => self.show_seq_transport(ui),
            Format::Wav | Format::Mp3 => self.show_file_transport(ui, playback),
        }
    }

    fn show_seq_transport(&mut self, ui: &mut egui::Ui) {
        let status = match self.seq.as_ref() {
            Some(s) => s.snapshot_status(),
            None => return,
        };
        let title = self
            .seq_loaded_label
            .clone()
            .or_else(|| self.selected_label.clone())
            .unwrap_or_else(|| "No MIDI selected".to_string());
        let subtitle = self
            .selected_label
            .as_ref()
            .and_then(|l| self.song_index.get(l))
            .map(|s| {
                let mut out = compact_composer(&s.composer);
                if let Some(era) = s.era.as_deref() {
                    out.push_str("  /  ");
                    out.push_str(era);
                }
                out
            })
            .unwrap_or_else(|| "Live sequencer · in-process voices_live".to_string());

        let mut clicked_play = false;
        let mut clicked_stop = false;
        let mut new_tempo: Option<f32> = None;

        egui::Frame::none()
            .fill(egui::Color32::from_rgb(57, 69, 63))
            .inner_margin(egui::Margin::symmetric(14.0, 12.0))
            .rounding(8.0)
            .show(ui, |ui| {
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
                    let progress = if status.end_sec > 0.0 {
                        (status.playhead_sec as f32 / status.end_sec as f32).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    ui.add(
                        egui::ProgressBar::new(progress)
                            .desired_width(280.0)
                            .text(format!(
                                "{:.1}s / {:.1}s",
                                status.playhead_sec, status.end_sec
                            )),
                    );
                    ui.add_space(14.0);
                    if ui
                        .add_sized(
                            [62.0, 28.0],
                            egui::Button::new(if status.is_playing { "Pause" } else { "Play" })
                                .fill(egui::Color32::from_rgb(208, 224, 216)),
                        )
                        .clicked()
                    {
                        clicked_play = true;
                    }
                    ui.add_space(4.0);
                    if ui
                        .add_sized(
                            [54.0, 28.0],
                            egui::Button::new("Stop")
                                .fill(egui::Color32::from_rgb(225, 218, 208)),
                        )
                        .clicked()
                    {
                        clicked_stop = true;
                    }
                    ui.add_space(8.0);
                    let mut tempo = status.tempo_scale;
                    if ui
                        .add_sized(
                            [180.0, 24.0],
                            egui::Slider::new(&mut tempo, 0.25..=2.0).text("tempo"),
                        )
                        .changed()
                    {
                        new_tempo = Some(tempo);
                    }
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(format!("voices {}", status.voice_count))
                            .small()
                            .color(egui::Color32::from_rgb(205, 213, 204)),
                    );
                });
            });

        if let Some(t) = new_tempo {
            if let Some(seq) = self.seq.as_ref() {
                seq.set_tempo(t);
            }
        }
        if clicked_stop {
            if let Some(seq) = self.seq.as_ref() {
                seq.stop();
            }
        }
        if clicked_play {
            if status.is_playing {
                if let Some(seq) = self.seq.as_ref() {
                    seq.pause();
                }
            } else {
                self.seq_play();
            }
        }
    }

    fn show_file_transport(&mut self, ui: &mut egui::Ui, playback: &PlaybackSnapshot) {
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
                    ui.add(egui::ProgressBar::new(progress).desired_width(280.0).text(
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
        egui::Grid::new("jukebox_lite_seq_meta")
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
        egui::Grid::new("jukebox_lite_seq_audit_metrics")
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

    // ----------------------------------------------------------------
    // Sequencer-view methods. Live paradigm: voices_live cdylib in
    // process, no render_midi subprocess, no preview_cache.
    // ----------------------------------------------------------------

    fn seq_load_selected(&mut self) {
        let track = match self.selected_track() {
            Some(t) => t,
            None => return,
        };
        if track.format != Format::Mid {
            return;
        }
        let bytes = match std::fs::read(&track.path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "jukebox_lite_seq: read midi {}: {e}",
                    track.path.display()
                );
                return;
            }
        };
        let mut program = match parse_smf_for_seq(&bytes) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("jukebox_lite_seq: parse midi: {e}");
                return;
            }
        };
        program.label = track.label.clone();
        if let Some(seq) = self.seq.as_ref() {
            seq.load_program(program);
        }
        self.seq_loaded_label = Some(track.label.clone());
    }

    fn seq_play(&mut self) {
        if self.seq_loaded_label.is_none() {
            self.seq_load_selected();
        }
        if let Some(seq) = self.seq.as_ref() {
            seq.play();
        }
    }

    fn seq_change_master_voice(&mut self, voice_slug: &str) {
        self.seq_voice_slug = voice_slug.to_string();
        if let Some(seq) = self.seq.as_mut() {
            // Cache hit: this returns after a single ArcSwap::store.
            // Cache miss: queues a background build, master rotates
            // when the build lands.
            seq.ensure_voice_loaded(voice_slug, true);
        }
    }

    fn seq_change_track_voice(&mut self, track_idx: usize, voice_slug: Option<&str>) {
        let Some(seq) = self.seq.as_mut() else {
            return;
        };
        if let Some(slug) = voice_slug {
            // Same warm-up rule: if the slug isn't pooled yet, kick a
            // build that drops it into the pool without touching
            // master. Until then audio thread falls back to master
            // for this track, so playback never goes silent.
            seq.ensure_voice_loaded(slug, false);
        }
        if let Ok(mut st) = seq.shared.lock() {
            let slot = track_idx;
            if st.track_voice_slugs.len() <= slot {
                st.track_voice_slugs.resize(slot + 1, None);
            }
            st.track_voice_slugs[slot] = voice_slug.map(str::to_string);
        }
    }

    // Master-voice combo embedded directly in the per-track strip
    // (see `show_sequencer_panel`); no separate header function.

    fn show_sequencer_panel(&mut self, ui: &mut egui::Ui) {
        let Some(seq) = self.seq.as_ref() else {
            ui.label(
                egui::RichText::new(
                    self.seq_init_error
                        .clone()
                        .unwrap_or_else(|| "sequencer engine unavailable".into()),
                )
                .small()
                .color(egui::Color32::from_rgb(168, 84, 84)),
            );
            return;
        };
        let program = seq.snapshot_program();
        let status = seq.snapshot_status();
        let (mute, solo) = seq.snapshot_track_flags();

        if program.notes.is_empty() {
            // Silent placeholder. The grid lives where the program
            // would be drawn so the layout doesn't shift when a row
            // gets selected.
            let avail = ui.available_size();
            let (rect, _) =
                ui.allocate_exact_size(avail, egui::Sense::hover());
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 6.0, egui::Color32::from_rgb(28, 36, 32));
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "select a MIDI row above",
                egui::FontId::proportional(13.0),
                egui::Color32::from_rgba_unmultiplied(160, 180, 168, 120),
            );
            return;
        }

        // Snapshot per-track voice slugs + slot states once for the
        // whole strip. The track combo deferred-mutates via these
        // intent vectors so we never carry a borrow on `seq` across
        // the &mut self call.
        let track_slug_snapshot: Vec<Option<String>> = self
            .seq
            .as_ref()
            .and_then(|s| s.shared.lock().ok().map(|st| st.track_voice_slugs.clone()))
            .unwrap_or_default();
        let states: HashMap<String, VoiceSlotState> = self
            .seq
            .as_ref()
            .and_then(|s| s.slot_states.lock().ok().map(|g| g.clone()))
            .unwrap_or_default();

        let mut track_voice_intents: Vec<(usize, Option<&'static str>)> = Vec::new();

        // Track strip (master voice + Mute / Solo / Voice combo per
        // track). The leading "Master" cell is the default fallback
        // for any track set to "(master)" in its own combo.
        let mut master_chosen: Option<&'static str> = None;
        ui.horizontal_wrapped(|ui| {
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(232, 228, 219))
                .rounding(6.0)
                .inner_margin(egui::Margin::symmetric(6.0, 3.0))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("Master")
                                .small()
                                .strong()
                                .color(egui::Color32::from_rgb(48, 53, 49)),
                        );
                        let current_label = SEQ_VOICE_CHOICES
                            .iter()
                            .find(|(slug, _)| *slug == self.seq_voice_slug.as_str())
                            .map(|(_, label)| *label)
                            .unwrap_or("(unknown)");
                        egui::ComboBox::from_id_salt("seq_master_voice_strip")
                            .selected_text(format!(
                                "{} {current_label}",
                                state_glyph(states.get(self.seq_voice_slug.as_str()))
                            ))
                            .width(140.0)
                            .show_ui(ui, |ui| {
                                for (slug, label) in SEQ_VOICE_CHOICES {
                                    let glyph = state_glyph(states.get(*slug));
                                    let txt = format!("{glyph} {label}");
                                    if ui
                                        .selectable_label(
                                            self.seq_voice_slug == *slug,
                                            txt,
                                        )
                                        .clicked()
                                    {
                                        master_chosen = Some(*slug);
                                    }
                                }
                            });
                    });
                });
            for t in 0..program.track_count {
                let is_m = mute.get(t).copied().unwrap_or(false);
                let is_s = solo.get(t).copied().unwrap_or(false);
                let track_slug = track_slug_snapshot
                    .get(t)
                    .and_then(|s| s.as_deref());
                ui.add_space(4.0);
                egui::Frame::none()
                    .fill(egui::Color32::from_rgb(232, 228, 219))
                    .rounding(6.0)
                    .inner_margin(egui::Margin::symmetric(6.0, 3.0))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!("T{}", t + 1))
                                    .small()
                                    .strong()
                                    .color(seq_track_color(t)),
                            );
                            let mute_btn = egui::Button::new(
                                egui::RichText::new("M").color(if is_m {
                                    egui::Color32::WHITE
                                } else {
                                    egui::Color32::from_rgb(80, 80, 80)
                                }),
                            )
                            .fill(if is_m {
                                egui::Color32::from_rgb(170, 96, 96)
                            } else {
                                egui::Color32::from_rgb(238, 234, 226)
                            });
                            if ui.add_sized([22.0, 18.0], mute_btn).clicked() {
                                seq.toggle_mute(t);
                            }
                            let solo_btn = egui::Button::new(
                                egui::RichText::new("S").color(if is_s {
                                    egui::Color32::WHITE
                                } else {
                                    egui::Color32::from_rgb(80, 80, 80)
                                }),
                            )
                            .fill(if is_s {
                                egui::Color32::from_rgb(96, 134, 170)
                            } else {
                                egui::Color32::from_rgb(238, 234, 226)
                            });
                            if ui.add_sized([22.0, 18.0], solo_btn).clicked() {
                                seq.toggle_solo(t);
                            }

                            // Per-track voice combo. "(master)" routes
                            // through the shared master factory; any
                            // other slug pulls that track's notes
                            // through its own pooled factory.
                            let track_label = match track_slug {
                                None => "(master)".to_string(),
                                Some(slug) => {
                                    let name = SEQ_VOICE_CHOICES
                                        .iter()
                                        .find(|(s, _)| *s == slug)
                                        .map(|(_, n)| *n)
                                        .unwrap_or(slug);
                                    format!(
                                        "{} {name}",
                                        state_glyph(states.get(slug))
                                    )
                                }
                            };
                            egui::ComboBox::from_id_salt(("seq_track_voice", t))
                                .selected_text(track_label)
                                .width(120.0)
                                .show_ui(ui, |ui| {
                                    if ui
                                        .selectable_label(
                                            track_slug.is_none(),
                                            "(master)",
                                        )
                                        .clicked()
                                    {
                                        track_voice_intents.push((t, None));
                                    }
                                    for (slug, label) in SEQ_VOICE_CHOICES {
                                        let glyph = state_glyph(states.get(*slug));
                                        let txt = format!("{glyph} {label}");
                                        let selected = track_slug == Some(*slug);
                                        if ui.selectable_label(selected, txt).clicked() {
                                            track_voice_intents
                                                .push((t, Some(*slug)));
                                        }
                                    }
                                });
                        });
                    });
            }
        });
        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // Note grid: pitch (vertical) × time (horizontal). Click to seek.
        let avail = ui.available_size();
        let grid_h = (avail.y - 16.0).max(160.0);
        let grid_w = avail.x - 8.0;
        let (rect, resp) = ui.allocate_exact_size(
            egui::vec2(grid_w, grid_h),
            egui::Sense::click_and_drag(),
        );
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 6.0, egui::Color32::from_rgb(28, 36, 32));

        let pitch_min = program.pitch_min.saturating_sub(1);
        let pitch_max = program.pitch_max.saturating_add(1);
        let pitch_span = (pitch_max as f32 - pitch_min as f32).max(1.0);
        let total_sec = program.end_sec.max(1.0) as f32;

        // Octave ledger lines.
        for p in pitch_min..=pitch_max {
            if p % 12 == 0 {
                let t = (p as f32 - pitch_min as f32) / pitch_span;
                let y = egui::lerp(rect.bottom()..=rect.top(), t);
                painter.line_segment(
                    [
                        egui::pos2(rect.left(), y),
                        egui::pos2(rect.right(), y),
                    ],
                    egui::Stroke::new(0.5, egui::Color32::from_rgba_unmultiplied(180, 200, 180, 50)),
                );
            }
        }
        // Beat ledger lines (every 1.0 sec at native tempo — close
        // enough for visual orientation; precise meter would require
        // parsing time signatures).
        let beat_count = total_sec.ceil() as i32;
        for b in 0..=beat_count {
            let t = b as f32 / total_sec;
            let x = egui::lerp(rect.left()..=rect.right(), t);
            painter.line_segment(
                [
                    egui::pos2(x, rect.top()),
                    egui::pos2(x, rect.bottom()),
                ],
                egui::Stroke::new(
                    0.5,
                    egui::Color32::from_rgba_unmultiplied(120, 150, 130, 35),
                ),
            );
        }

        // Notes themselves.
        for n in program.notes.iter() {
            let x0 = egui::lerp(
                rect.left()..=rect.right(),
                (n.start_sec as f32 / total_sec).clamp(0.0, 1.0),
            );
            let x1 = egui::lerp(
                rect.left()..=rect.right(),
                ((n.start_sec + n.duration_sec) as f32 / total_sec).clamp(0.0, 1.0),
            );
            let y_top = egui::lerp(
                rect.bottom()..=rect.top(),
                (n.midi_note as f32 - pitch_min as f32 + 1.0) / pitch_span,
            );
            let y_bot = egui::lerp(
                rect.bottom()..=rect.top(),
                (n.midi_note as f32 - pitch_min as f32) / pitch_span,
            );
            let cell = egui::Rect::from_min_max(
                egui::pos2(x0, y_top),
                egui::pos2((x1 - 1.0).max(x0 + 1.0), y_bot),
            );
            let muted = mute.get(n.track_idx as usize).copied().unwrap_or(false);
            let any_solo = solo.iter().any(|&s| s);
            let solo_active = solo.get(n.track_idx as usize).copied().unwrap_or(false);
            let dimmed = muted || (any_solo && !solo_active);
            let mut col = seq_track_color(n.track_idx as usize);
            if dimmed {
                col = egui::Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 50);
            } else {
                let v = n.velocity as f32 / 127.0;
                let factor = 0.55 + 0.45 * v;
                col = egui::Color32::from_rgb(
                    (col.r() as f32 * factor) as u8,
                    (col.g() as f32 * factor) as u8,
                    (col.b() as f32 * factor) as u8,
                );
            }
            painter.rect_filled(cell, 1.5, col);
        }

        // Playhead.
        let head_t =
            (status.playhead_sec as f32 / total_sec).clamp(0.0, 1.0);
        let head_x = egui::lerp(rect.left()..=rect.right(), head_t);
        painter.line_segment(
            [
                egui::pos2(head_x, rect.top()),
                egui::pos2(head_x, rect.bottom()),
            ],
            egui::Stroke::new(1.6, egui::Color32::from_rgb(244, 226, 138)),
        );

        // Click / drag to seek. Intent-collection style so we don't
        // hold a `seq` borrow across the `&mut self` apply pass below.
        let mut seek_to: Option<f64> = None;
        if let Some(pos) = resp.interact_pointer_pos() {
            if rect.contains(pos) {
                let t =
                    ((pos.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0);
                seek_to = Some(t as f64 * total_sec as f64);
            }
        }
        // Release the `seq` borrow we've been holding for snapshots,
        // mute/solo button handlers, etc., then apply collected
        // intents that need `&mut self`. At this point we're below
        // every site that read from `seq`.
        let _ = seq;
        if let Some(slug) = master_chosen {
            self.seq_change_master_voice(slug);
        }
        for (track, slug) in track_voice_intents {
            self.seq_change_track_voice(track, slug);
        }
        if let Some(target) = seek_to {
            if let Some(seq) = self.seq.as_ref() {
                seq.seek(target);
            }
        }
    }

}

/// Single-character status glyph for a voice slot. Drives the combo
/// labels so the user sees ✓ for "instantly available", spinner for
/// "still building", × for "build broke". Avoids dropdown jitter.
fn state_glyph(s: Option<&VoiceSlotState>) -> &'static str {
    match s {
        Some(VoiceSlotState::Ready) => "✓",
        Some(VoiceSlotState::Loading) => "…",
        Some(VoiceSlotState::Failed(_)) => "×",
        Some(VoiceSlotState::Idle) | None => "·",
    }
}

fn seq_track_color(track_idx: usize) -> egui::Color32 {
    // Picks one of a small palette deterministically by index. The
    // palette favours legible-on-dark hues — saturation kept down so
    // the playhead's yellow stays the brightest thing on screen.
    const PALETTE: &[egui::Color32] = &[
        egui::Color32::from_rgb(159, 213, 173),
        egui::Color32::from_rgb(209, 184, 145),
        egui::Color32::from_rgb(157, 196, 217),
        egui::Color32::from_rgb(218, 173, 197),
        egui::Color32::from_rgb(193, 200, 145),
        egui::Color32::from_rgb(184, 162, 215),
        egui::Color32::from_rgb(214, 197, 132),
        egui::Color32::from_rgb(146, 207, 199),
    ];
    PALETTE[track_idx % PALETTE.len()]
}

impl eframe::App for JukeboxLiteSeq {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_cp_commands();
        self.poll_pending_render();
        self.poll_pending_audit();
        if let Some(seq) = self.seq.as_mut() {
            seq.poll_voice_loads();
        }
        if self.seq_autoplay_pending
            && self
                .seq
                .as_ref()
                .map(|s| {
                    matches!(
                        s.slot_state(self.seq_voice_slug.as_str()),
                        VoiceSlotState::Ready
                    )
                })
                .unwrap_or(false)
        {
            self.seq_play();
            self.seq_autoplay_pending = false;
        }

        // Selection-driven auto-load: whenever the row that's
        // selected in the table differs from what's loaded in the
        // sequencer engine, swap the program in. No tabs, no buttons
        // — clicking a `.mid` row IS the load.
        let want_mid = self
            .selected_label
            .as_deref()
            .filter(|label| {
                self.tracks
                    .iter()
                    .any(|t| t.label == *label && t.format == Format::Mid)
            })
            .map(str::to_string);
        let have_mid = self.seq_loaded_label.clone();
        let seq_idle = self
            .seq
            .as_ref()
            .map(|s| !s.snapshot_status().is_playing)
            .unwrap_or(true);
        if want_mid.is_some() && want_mid != have_mid && seq_idle {
            self.seq_load_selected();
        }

        let visible = self.visible_tracks();
        self.ensure_selection(&visible);
        let playback = self.playback_snapshot();

        // Single unified layout — list + sequencer share one window.
        // Header carries the search/filter row plus the master voice
        // combo, sidebar holds tile filters, right pane is the
        // inspector for the selected row, central panel is split
        // vertically: top half = list table (the browser), bottom
        // half = sequencer grid (live program for the selected row).
        // Bottom transport adapts to the selected row's format.
        egui::TopBottomPanel::top("jukebox_lite_seq_header")
            .exact_height(80.0)
            .show(ctx, |ui| self.show_header(ui, visible.len()));

        egui::TopBottomPanel::bottom("jukebox_lite_seq_transport")
            .exact_height(92.0)
            .show(ctx, |ui| self.show_transport(ui, &playback));

        egui::SidePanel::left("jukebox_lite_seq_sidebar")
            .resizable(false)
            .exact_width(190.0)
            .show(ctx, |ui| self.show_sidebar(ui));

        egui::SidePanel::right("jukebox_lite_seq_inspector")
            .resizable(false)
            .exact_width(320.0)
            .show(ctx, |ui| self.show_inspector(ui, &playback));

        egui::CentralPanel::default().show(ctx, |ui| {
            // Vertical split: top = list rows, bottom = sequencer
            // grid. We give the sequencer ~45% of the height because
            // the grid scales better than the table — the table just
            // wraps text on narrow rows; the grid loses pitch
            // resolution if it's too short.
            let total_h = ui.available_height();
            let seq_h = (total_h * 0.45).max(180.0).min(total_h - 120.0);
            let table_h = total_h - seq_h - 6.0;
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), table_h),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    ui.add_space(6.0);
                    self.show_table(ui, &visible, &playback);
                },
            );
            ui.add_space(4.0);
            ui.separator();
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), seq_h),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    ui.add_space(2.0);
                    self.show_sequencer_panel(ui);
                },
            );
        });

        self.publish_cp_snapshot(&visible, &playback);

        let seq_active = self
            .seq
            .as_ref()
            .map(|s| s.snapshot_status().is_playing)
            .unwrap_or(false);
        if playback.is_playing
            || seq_active
            || self.pending_render.is_some()
            || self.pending_audit.is_some()
        {
            ctx.request_repaint_after(std::time::Duration::from_millis(33));
        } else {
            // Repaint at idle just slow enough that the grid view
            // updates after a click but doesn't burn CPU at 30 fps.
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

struct CliArgs {
    initial_midi_label: Option<String>,
    autoplay: bool,
}

fn parse_cli() -> CliArgs {
    let mut initial_midi_label = None;
    let mut autoplay = false;
    let mut iter = std::env::args().skip(1);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--midi" | "--start-midi" => {
                initial_midi_label = iter.next();
            }
            "--autoplay" => autoplay = true,
            _ => {}
        }
    }
    CliArgs {
        initial_midi_label,
        autoplay,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_cli();
    let dirs = vec![
        PathBuf::from("bench-out/songs"),
        PathBuf::from("bench-out/CHIPTUNE"),
        PathBuf::from("bench-out/iterH"),
    ];
    let mut app = JukeboxLiteSeq::new(dirs)?;
    if let Some(label) = args.initial_midi_label.as_ref() {
        if app.tracks.iter().any(|t| &t.label == label) {
            app.selected_label = Some(label.clone());
            // Load the program now so the grid is non-empty on the
            // very first frame; the audio engine waits until the user
            // hits Play (or `--autoplay` flips it after voice load).
            app.seq_load_selected();
        }
    }
    if args.autoplay {
        app.seq_autoplay_pending = true;
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1380.0, 840.0])
            .with_title("keysynth jukebox lite seq"),
        ..Default::default()
    };
    eframe::run_native(
        "keysynth jukebox lite seq",
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
        let dir = std::env::temp_dir().join("keysynth_jukebox_lite_seq_dedup_test");
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
