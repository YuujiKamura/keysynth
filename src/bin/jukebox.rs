//! Minimal WAV jukebox: scan bench-out/{iterH,songs}/ for *.wav, list
//! as selectable rows, click to play through default audio output via
//! a single-stream cpal mixer. For quick A/B listening of multiple
//! engine renders without launching a media player per file.
//!
//! Stage 1 (issue #7) replaces rodio with a hand-rolled cpal mixer:
//! one cpal::Stream is opened on the chosen output device and N tracks
//! are summed inside the audio callback. Each track is decoded
//! incrementally with hound::WavReader and the playback position is
//! tracked as a frame cursor so loop / mute / solo / pan / volume are
//! sample-accurate (no per-track Sink, no start-skew).

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use keysynth::gui_cp;
use keysynth::library_db::{LibraryDb, Song, SongFilter, SongSort, VoiceFilter};
use keysynth::play_log::{PlayEntry, PlayLogDb};

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

/// Container format of a Track on disk. Drives which decoder we pick
/// (hound for WAV, symphonia for MP3). Kept separate from per-engine
/// suffixes so the catalogue can show / hide formats without re-parsing
/// filenames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Wav,
    Mp3,
    /// Standard MIDI File. The decoder cannot play these directly — a
    /// click on a Mid track triggers a lazy `preview_cache` render
    /// through the user's currently selected voice, which produces a
    /// WAV that the standard decoder path can play.
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
    /// Display name = filename without extension.
    label: String,
    /// Group label inferred from filename prefix (everything up to the
    /// last "_<engine>" suffix; "_sfz" / "_modal" / etc. drop into a
    /// separate column so the same piece across engines lines up).
    piece: String,
    engine: String,
    /// Provenance / source repo of the underlying material:
    /// "keysynth", "chiptune-demo", "listener-lab", "midi" etc.
    source: String,
    size_kb: u64,
    /// Container format. Selects WAV (hound) vs MP3 (symphonia) decoder.
    format: Format,
}

const ENGINE_SUFFIXES: &[&str] = &[
    // Order matters: longer / more specific suffixes are listed first so the
    // matcher catches them before shorter prefixes (e.g. "modal-r16" before
    // "modal"). 2026-04-26 presets:
    //   modal-r16     = round-16 (CDPAM-optimal but "muffled")
    //   modal-arch1   = Arch-1 (Hertzian hammer LPF + commuted residual)
    //   modal-physics = pure analytic physics (Stulov hammer + missing
    //                   modes + Valimaki T60, no LUT, no residual)
    "modal-physics",
    "modal-arch1",
    "modal-r16",
    "sfz",
    "modal",
    "square",
    "ks",
    "ks-rich",
    "sub",
    "fm",
    "piano",
    "piano-thick",
    "piano-lite",
    "piano-5am",
    "koto",
    // "pure" marks an NSF/libgme ground-truth render — it isn't a keysynth
    // engine, but tagging it like one lets the fold-by-piece grouper pair
    // nsf_parodius_03_vic_viper_pure with audio_parodius_03_vic_viper /
    // listener_parodius_03_vic_viper_square in the same row.
    "pure",
];

/// Strip the trailing "(YYYY-YYYY)" / "(b. YYYY)" / "(Spanish)" parens
/// from a `composer` so the inline song-list cell stays readable. The
/// dates already drive era classification; once the era badge is shown
/// they're noise on the row itself. Trims trailing whitespace so the
/// result is hover-tooltip-clean too.
fn compact_composer(composer: &str) -> String {
    let core = match composer.split_once('(') {
        Some((before, _)) => before,
        None => composer,
    };
    core.trim().to_string()
}

/// Background colour for an era badge. Picked to be distinguishable at
/// a glance against the egui dark theme — same idea as `source_label`'s
/// palette in `render_piece_row`. Unknown eras fall back to neutral
/// grey so adding a new era doesn't crash the UI.
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

/// Compact license label for the inline badge ("Public Domain" → "PD",
/// "CC-BY-SA 4.0" → "CC-BY-SA"). Keeps the row narrow enough that
/// composer + era + license still fit before the engine buttons wrap.
/// The full string remains available via the row tooltip.
fn license_short(license: &str) -> String {
    let l = license.trim();
    let lower = l.to_lowercase();
    if lower.contains("public domain") || lower == "pd" {
        "PD".to_string()
    } else if lower.starts_with("cc0") {
        "CC0".to_string()
    } else if lower.starts_with("cc-by-sa") || lower.starts_with("cc by-sa") {
        "CC-BY-SA".to_string()
    } else if lower.starts_with("cc-by") || lower.starts_with("cc by") {
        "CC-BY".to_string()
    } else if lower.is_empty() {
        "?".to_string()
    } else {
        l.to_string()
    }
}

/// Colour-code license category. Public-domain / CC0 are "free to use,
/// no strings"; CC-BY needs attribution; CC-BY-SA also locks downstream
/// licensing — surface that distinction so a user picking material for
/// downstream redistribution can spot share-alike at a glance.
fn license_color(short: &str) -> egui::Color32 {
    match short {
        "PD" | "CC0" => egui::Color32::from_rgb(140, 220, 160),
        "CC-BY" => egui::Color32::from_rgb(180, 200, 255),
        "CC-BY-SA" => egui::Color32::from_rgb(255, 200, 120),
        _ => egui::Color32::from_rgb(170, 170, 170),
    }
}

fn classify_source(path: &Path, piece: &str) -> &'static str {
    // Path-driven first: which directory is this file in?
    let path_str = path.to_string_lossy().to_lowercase();
    if path_str.contains("/chiptune/") || path_str.contains("\\chiptune\\") {
        // Filenames within bench-out/CHIPTUNE/ are derived from the
        // input JSON's stem. Sub-classify by that stem prefix.
        let p = piece.to_lowercase();
        if p.starts_with("rec_")
            || p.starts_with("manual_")
            || p.starts_with("arp_demo")
            || p.starts_with("arrange_")
            || p.starts_with("compose_")
            || p.starts_with("twincle")
            || p.starts_with("twinclestar")
        {
            return "listener-lab";
        }
        // v0..v99 = ai-chiptune-demo competition naming
        let bytes = p.as_bytes();
        if bytes.len() >= 2 && bytes[0] == b'v' && bytes[1].is_ascii_digit() {
            return "chiptune-demo";
        }
        return "chiptune";
    }
    if path_str.contains("midi_") || piece.starts_with("midi_") {
        return "midi";
    }
    if piece.starts_with("tofu_") {
        // Acceleration-of-Tofu game asset — SFX or BGM rendered from
        // its Pygame synth (drum_machine, famicom_drum, etc.).
        return "tofu";
    }
    if piece.starts_with("audio_") {
        // Direct mp3-to-wav conversion of a reference recording
        // (archive.org piano roll capture, etc.). Source material
        // not synthesised by us.
        return "archive-ref";
    }
    if piece.starts_with("nsf_") {
        // libgme-rendered ground truth from a Nintendo Sound Format file.
        // This IS what the original NES game sounds like — every other
        // *_parodius_* row should be measured against the matching
        // nsf_parodius_*_pure track.
        return "nsf-truth";
    }
    "keysynth"
}

/// Voice IDs the per-track voice picker exposes in the jukebox row's
/// dropdown. Each one maps 1:1 to a `render_midi --engine <id>` alias
/// so the cache subprocess can render with no extra flags. Order is
/// the dropdown display order — we lead with the two heuristic
/// defaults (`piano-modal`, `guitar-stk`) so the most common picks
/// are at the top, then walk through the alternative pianos and
/// finally the chiptune / synth tail.
const AVAILABLE_VOICES: &[&str] = &[
    "piano-modal",
    "guitar-stk",
    "piano",
    "piano-lite",
    "piano-5am",
    "koto",
    "square",
    "ks",
    "fm",
    "sub",
];

/// Pick a default voice for a MIDI track based on filename heuristic.
/// Piano-leaning composer prefixes (mozart, satie, albeniz, bach, ...)
/// route through `piano-modal`; everything else through the STK guitar
/// voice. Used as the fallback when the user has not yet picked a
/// voice for this track via the GUI dropdown (`play_log.track_voices`).
fn default_voice_for_midi(path: &Path) -> &'static str {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    const PIANO_PREFIXES: &[&str] = &[
        "mozart_",
        "satie_",
        "albeniz_",
        "chopin_",
        "beethoven_",
        "schubert_",
        "scarlatti_",
    ];
    if PIANO_PREFIXES.iter().any(|p| stem.starts_with(p)) {
        "piano-modal"
    } else {
        "guitar-stk"
    }
}

/// Where to find the `render_midi` binary that the cache subprocess
/// calls. Both jukebox and render_midi live in the same target
/// directory, so we anchor at `current_exe()` and replace the file
/// name. Returns `None` if the executable cannot be discovered.
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

/// Build the `CacheKey` for a MIDI file under an explicit voice id.
/// Pulled out of the resolve flow so the synchronous cache hit check
/// and the background render-and-store path share the same key
/// construction. The voice id is whatever the user has currently
/// picked for this track (or the heuristic default if they have not
/// touched the picker yet).
fn build_midi_cache_key(midi_path: &Path, voice_id: &str) -> keysynth::preview_cache::CacheKey {
    keysynth::preview_cache::CacheKey {
        song_path: midi_path.to_path_buf(),
        voice_id: voice_id.to_string(),
        // Voice .dll mtime tracking is wired in `preview_cache::CacheKey`
        // itself (and exercised by the unit tests); the GUI side just
        // doesn't track which .dll is currently loaded yet, so we go
        // through the "builtin" hash bucket. When the GUI grows a voice
        // selector with hot-reload status this should plug in the real
        // .dll path so rebuild → cache invalidate flows automatically.
        voice_dll: None,
        render_params: keysynth::preview_cache::RenderParams::default(),
    }
}

fn open_preview_cache() -> Result<keysynth::preview_cache::Cache, String> {
    let cache_dir = PathBuf::from("bench-out/cache");
    keysynth::preview_cache::Cache::new(&cache_dir, 1_073_741_824) // 1 GiB
        .map_err(|e| format!("Cache::new {}: {e}", cache_dir.display()))
}

/// Cache-only lookup, never spawns a render. Returns `Ok(Some(path))`
/// if there's already a rendered WAV for the (song, voice) pair.
/// Used by the GUI thread for the synchronous fast path before
/// deciding whether to spawn a background render.
fn lookup_midi_in_preview_cache(
    midi_path: &Path,
    voice_id: &str,
) -> Result<Option<PathBuf>, String> {
    let cache = open_preview_cache()?;
    let key = build_midi_cache_key(midi_path, voice_id);
    cache.lookup(&key).map_err(|e| format!("lookup: {e}"))
}

/// One-shot stat sweep that returns, per `Track::label`, the set of
/// voice ids whose preview WAV is already on disk. The central panel
/// uses this to draw the "✓" hit glyph next to the picker — it only
/// goes green when the user's *currently selected* voice for that
/// track is cached. Call sites refresh on startup, after `rescan`,
/// and after a successful render (foreground or prewarm).
///
/// The sweep stats only the (track, selected-voice) pair so cost stays
/// O(N midi tracks) rather than O(N × |AVAILABLE_VOICES|). When the
/// user changes their pick we re-stat just that one entry from the
/// click handler, not the whole catalogue.
fn snapshot_cached_midi_voices(
    tracks: &[Track],
    selected_voices: &HashMap<String, String>,
) -> HashMap<String, HashSet<String>> {
    let cache = match open_preview_cache() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("jukebox: cached-set snapshot skipped: {e}");
            return HashMap::new();
        }
    };
    let mut out: HashMap<String, HashSet<String>> = HashMap::new();
    for t in tracks {
        if t.format != Format::Mid {
            continue;
        }
        let voice = selected_voices
            .get(&t.label)
            .cloned()
            .unwrap_or_else(|| default_voice_for_midi(&t.path).to_string());
        let key = build_midi_cache_key(&t.path, &voice);
        if matches!(cache.lookup(&key), Ok(Some(_))) {
            out.entry(t.label.clone()).or_default().insert(voice);
        }
    }
    out
}

/// Cheap single-entry refresh used after a voice change or a finished
/// render. Stats one (path, voice) pair and updates the snapshot map
/// in place.
fn refresh_cached_voice_entry(
    cached: &mut HashMap<String, HashSet<String>>,
    label: &str,
    midi_path: &Path,
    voice_id: &str,
) {
    let cache = match open_preview_cache() {
        Ok(c) => c,
        Err(_) => return,
    };
    let key = build_midi_cache_key(midi_path, voice_id);
    let entry = cached.entry(label.to_string()).or_default();
    if matches!(cache.lookup(&key), Ok(Some(_))) {
        entry.insert(voice_id.to_string());
    } else {
        entry.remove(voice_id);
    }
    if entry.is_empty() {
        cached.remove(label);
    }
}

/// Synchronous render-and-store. Called from a background thread so
/// the GUI stays responsive while `render_midi` synthesises
/// (typically 1–10 s for a short piece). Returns the on-disk WAV
/// path of the freshly-cached render. Cache hits short-circuit
/// inside `render_to_cache` so a stale background thread that fires
/// after a peer thread already populated the entry returns
/// immediately.
fn render_midi_blocking(midi_path: &Path, voice_id: &str) -> Result<PathBuf, String> {
    let cache = open_preview_cache()?;
    let key = build_midi_cache_key(midi_path, voice_id);
    let bin = render_midi_binary_path()
        .ok_or_else(|| "render_midi binary not found alongside jukebox".to_string())?;
    keysynth::preview_cache::render_to_cache(&cache, &key, voice_id, &bin)
        .map_err(|e| format!("render_to_cache: {e}"))
}

fn parse_track(path: &Path) -> Option<Track> {
    let stem = path.file_stem()?.to_str()?.to_string();
    let ext = path.extension().and_then(|s| s.to_str())?;
    let format = Format::from_ext(ext)?;
    let size_kb = std::fs::metadata(path).ok()?.len() / 1024;
    // Try splitting "<piece>_<engine>" by recognised engine suffix.
    let (piece, engine) = ENGINE_SUFFIXES
        .iter()
        .find_map(|s| {
            let suf = format!("_{s}");
            stem.strip_suffix(&suf)
                .map(|p| (p.to_string(), s.to_string()))
        })
        .unwrap_or_else(|| (stem.clone(), "—".to_string()));
    let source = classify_source(path, &piece).to_string();
    Some(Track {
        path: path.to_path_buf(),
        label: stem.clone(),
        piece,
        engine,
        source,
        size_kb,
        format,
    })
}

/// 2026-04-26: bench-out/songs/ is being migrated WAV → MP3 (Gemini
/// audio modality requirement, ~1/7 disk footprint). During the
/// transition both formats coexist on disk; this scanner keeps MP3 and
/// drops the matching WAV so the catalogue isn't doubled. Same-stem
/// pairs are detected per-directory using `(stem)` as the dedup key.
fn scan_dirs(dirs: &[&Path]) -> Vec<Track> {
    let mut out: Vec<Track> = Vec::new();
    for d in dirs {
        let read = match std::fs::read_dir(d) {
            Ok(r) => r,
            Err(_) => continue,
        };
        // Two-pass per directory: collect everything, then drop WAVs
        // that have a same-stem MP3 sibling. Per-dir (not global) so a
        // /tmp.wav and bench/tmp.mp3 don't accidentally cancel.
        let mut dir_tracks: Vec<Track> = Vec::new();
        for ent in read.flatten() {
            let p = ent.path();
            let ext = match p.extension().and_then(|s| s.to_str()) {
                Some(e) => e.to_ascii_lowercase(),
                None => continue,
            };
            // wav / mp3: decode-and-play directly. mid: list now,
            // lazy-render on user click (issue #62 Phase 2).
            if ext != "wav" && ext != "mp3" && ext != "mid" && ext != "midi" {
                continue;
            }
            if let Some(t) = parse_track(&p) {
                dir_tracks.push(t);
            }
        }
        // Build set of stems that have an MP3 in this directory.
        let mp3_stems: std::collections::HashSet<String> = dir_tracks
            .iter()
            .filter(|t| t.format == Format::Mp3)
            .map(|t| t.label.clone())
            .collect();
        for t in dir_tracks {
            if t.format == Format::Wav && mp3_stems.contains(&t.label) {
                continue; // MP3 sibling wins
            }
            out.push(t);
        }
    }
    // Sort by piece then engine for grouped display.
    out.sort_by(|a, b| a.piece.cmp(&b.piece).then_with(|| a.engine.cmp(&b.engine)));
    out
}

// ---------- Mixer ----------------------------------------------------

/// MP3 decode state. symphonia decodes packet-by-packet into a planar
/// AudioBuffer; we eagerly flatten each packet into an interleaved f32
/// scratch and drain it sample-by-sample to match the existing
/// `read_one_sample` contract used by the WAV path.
struct Mp3State {
    /// Source path retained for re-open on loop / seek-to-zero (symphonia
    /// `seek` to packet 0 is awkward across formats; reopening is robust
    /// and only happens at end-of-file).
    path: PathBuf,
    format: Box<dyn FormatReader>,
    decoder: Box<dyn Decoder>,
    track_id: u32,
    /// Interleaved f32 samples already decoded but not yet drained.
    /// Filled from the AudioBuffer when empty; consumed one f32 at a
    /// time by `read_one_sample`.
    scratch: Vec<f32>,
    /// Read position into `scratch`.
    scratch_pos: usize,
    /// True once next_packet returned UnexpectedEof (or any terminal
    /// error). The caller treats this like WAV EOF.
    eof: bool,
}

/// Per-format decoded handle. WAV uses hound (incremental sample iter);
/// MP3 uses symphonia (packet decode → flatten → drain).
enum DecoderState {
    Wav {
        reader: WavReader<BufReader<File>>,
        sample_format: WavSampleFormat,
        bits_per_sample: u16,
    },
    Mp3(Mp3State),
}

/// Common per-track metadata + decoder state, regardless of format.
/// `total_samples` is the interleaved sample count (frames * channels)
/// — for MP3 this is best-effort from symphonia's track metadata; the
/// audio callback never *requires* it (EOF is detected by the decoder),
/// but the UI uses it for the position readout.
struct DecodedTrack {
    state: DecoderState,
    sample_rate: u32,
    channels: u16,
    total_samples: u64,
}

fn open_wav(path: &Path) -> Result<DecodedTrack, String> {
    let reader = WavReader::open(path).map_err(|e| format!("hound open: {e}"))?;
    let spec = reader.spec();
    let total_samples = reader.duration() as u64 * spec.channels as u64;
    Ok(DecodedTrack {
        sample_rate: spec.sample_rate,
        channels: spec.channels,
        total_samples,
        state: DecoderState::Wav {
            reader,
            sample_format: spec.sample_format,
            bits_per_sample: spec.bits_per_sample,
        },
    })
}

/// Open an MP3 file with symphonia. Initialises the format reader, picks
/// the first decodable track, and constructs the matching decoder. The
/// scratch buffer starts empty — first `read_one_sample` call triggers
/// `decode_next_packet`.
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
    let codec_params = track.codec_params.clone();
    let sample_rate = codec_params
        .sample_rate
        .ok_or_else(|| "mp3: missing sample rate".to_string())?;
    let channels_count = codec_params.channels.map(|c| c.count() as u16).unwrap_or(2);
    // n_frames is "frames" in the *codec* sense (PCM frame = one sample
    // per channel). Multiply by channels to get the interleaved sample
    // count expected by total_samples.
    let total_samples = codec_params
        .n_frames
        .map(|n| n * channels_count as u64)
        .unwrap_or(0);
    let decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| format!("mp3 codec: {e}"))?;
    Ok(DecodedTrack {
        sample_rate,
        channels: channels_count,
        total_samples,
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

/// Dispatch by extension. Used by `MixerTrack::load`.
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

/// Drain one packet from symphonia, flattening planar channels into the
/// interleaved scratch. Sets `eof` on terminal errors. Returns Ok(()) on
/// success or recoverable skip; the caller re-checks scratch length.
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
            Err(SymphoniaError::DecodeError(_)) => continue, // skip bad frame
            Err(SymphoniaError::IoError(_)) => {
                state.eof = true;
                return Ok(());
            }
            Err(e) => return Err(format!("mp3 decode: {e}")),
        };
        // Flatten planar -> interleaved f32. Match on the concrete
        // AudioBufferRef so each numeric format can be normalised before
        // the cast (i16/i32/u8 differ in midpoint and range).
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
            AudioBufferRef::U8(buf) => {
                for f in 0..frames {
                    for c in 0..ch {
                        let v = buf.chan(c)[f] as f32;
                        state.scratch.push((v - 128.0) / 128.0);
                    }
                }
            }
            AudioBufferRef::S8(buf) => {
                for f in 0..frames {
                    for c in 0..ch {
                        state.scratch.push(buf.chan(c)[f] as f32 / 128.0);
                    }
                }
            }
            AudioBufferRef::S24(buf) => {
                let inv = 1.0f32 / ((1i32 << 23) - 1) as f32;
                for f in 0..frames {
                    for c in 0..ch {
                        state.scratch.push(buf.chan(c)[f].inner() as f32 * inv);
                    }
                }
            }
            AudioBufferRef::U16(buf) => {
                for f in 0..frames {
                    for c in 0..ch {
                        let v = buf.chan(c)[f] as f32;
                        state.scratch.push((v - 32768.0) / 32768.0);
                    }
                }
            }
            AudioBufferRef::U24(buf) => {
                let mid = (1u32 << 23) as f32;
                for f in 0..frames {
                    for c in 0..ch {
                        let v = buf.chan(c)[f].inner() as f32;
                        state.scratch.push((v - mid) / mid);
                    }
                }
            }
            AudioBufferRef::U32(buf) => {
                let mid = (1u64 << 31) as f32;
                for f in 0..frames {
                    for c in 0..ch {
                        let v = buf.chan(c)[f] as f32;
                        state.scratch.push((v - mid) / mid);
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
        }
        if state.scratch.is_empty() {
            // Empty packet (rare); pull another.
            continue;
        }
        return Ok(());
    }
}

/// One slot in the multi-track mixer. Owns its own hound reader (with a
/// frame cursor so seek/loop is exact). The audio callback advances the
/// cursor; the UI thread reads it back through the shared Mutex.
struct MixerTrack {
    file_path: PathBuf,
    file_label: String,
    /// `None` when the slot is empty. When loaded, the reader is owned
    /// by this slot and consumed sample-by-sample in the audio
    /// callback.
    decoded: Option<DecodedTrack>,
    volume: f32,
    /// Pan -1.0 (full left) .. +1.0 (full right). 0 = centre.
    pan: f32,
    muted: bool,
    solo: bool,
    loop_enabled: bool,
    /// True while the cursor is < total_samples (or always while looping).
    /// Audio callback flips this to false at end-of-file when not looping.
    is_playing: bool,
    /// Frames consumed from the start of the file. Reset to 0 on a
    /// fresh play_all(). Frame == one stereo (or mono) sample point.
    cursor_frames: u64,
    /// Last-block peak / RMS per channel. Updated each audio callback
    /// from the stereo-summed output. Read by UI for VU meters.
    vu_peak_l: f32,
    vu_peak_r: f32,
    vu_rms_l: f32,
    vu_rms_r: f32,
}

impl MixerTrack {
    fn empty() -> Self {
        Self {
            file_path: PathBuf::new(),
            file_label: String::new(),
            decoded: None,
            volume: 1.0,
            pan: 0.0,
            muted: false,
            solo: false,
            loop_enabled: false,
            is_playing: false,
            cursor_frames: 0,
            vu_peak_l: 0.0,
            vu_peak_r: 0.0,
            vu_rms_l: 0.0,
            vu_rms_r: 0.0,
        }
    }

    /// Load a fresh decoder for `path`. Format is detected from the file
    /// extension (`.wav` → hound, `.mp3` → symphonia). Replaces any
    /// previous loaded reader; cursor is reset to 0 and `is_playing` is
    /// left false until the user hits play_all.
    fn load(&mut self, path: &Path, label: String) -> Result<(), String> {
        let dec = open_decoded(path)?;
        self.file_path = path.to_path_buf();
        self.file_label = label;
        self.decoded = Some(dec);
        self.cursor_frames = 0;
        self.is_playing = false;
        Ok(())
    }

    fn unload(&mut self) {
        self.decoded = None;
        self.file_path.clear();
        self.file_label.clear();
        self.cursor_frames = 0;
        self.is_playing = false;
        self.vu_peak_l = 0.0;
        self.vu_peak_r = 0.0;
        self.vu_rms_l = 0.0;
        self.vu_rms_r = 0.0;
    }

    /// Rewind decoder to sample 0. For WAV this is a cheap hound seek;
    /// for MP3 we re-open the file (symphonia's seek API is per-format
    /// flaky and end-of-file rewinds are rare enough that the extra
    /// fopen is not measurable). Returns Err if reader missing.
    fn seek_start(&mut self) -> Result<(), String> {
        let dec = self.decoded.as_mut().ok_or("no decoded track")?;
        match &mut dec.state {
            DecoderState::Wav { reader, .. } => {
                reader.seek(0).map_err(|e| format!("hound seek: {e}"))?;
            }
            DecoderState::Mp3(_) => {
                let path = self.file_path.clone();
                let fresh = open_mp3(&path)?;
                *dec = fresh;
            }
        }
        self.cursor_frames = 0;
        Ok(())
    }
}

/// Rewind the decoder back to the first sample. Used by the audio
/// callback when `loop_enabled` is true and the decoder reports EOF.
/// Returns Err on failure (caller should give up looping).
fn rewind_decoder(dec: &mut DecodedTrack) -> Result<(), String> {
    match &mut dec.state {
        DecoderState::Wav { reader, .. } => reader.seek(0).map_err(|e| format!("hound seek: {e}")),
        DecoderState::Mp3(state) => {
            let path = state.path.clone();
            let fresh = open_mp3(&path)?;
            *dec = fresh;
            Ok(())
        }
    }
}

/// Read one f32 sample from the active decoder honouring its native
/// sample format. Returns None at EOF.
fn read_one_sample(dec: &mut DecodedTrack) -> Option<f32> {
    match &mut dec.state {
        DecoderState::Wav {
            reader,
            sample_format,
            bits_per_sample,
        } => match *sample_format {
            WavSampleFormat::Int => {
                let s: i32 = reader.samples::<i32>().next()?.ok()?;
                // Normalise by max int magnitude for the bit depth.
                let max = ((1u32 << (*bits_per_sample as u32 - 1)) - 1) as f32;
                Some(s as f32 / max)
            }
            WavSampleFormat::Float => {
                let s: f32 = reader.samples::<f32>().next()?.ok()?;
                Some(s)
            }
        },
        DecoderState::Mp3(state) => {
            // Refill scratch when drained.
            if state.scratch_pos >= state.scratch.len() {
                if state.eof {
                    return None;
                }
                if let Err(e) = mp3_decode_next(state) {
                    eprintln!("jukebox mp3 decode: {e}");
                    state.eof = true;
                    return None;
                }
                if state.scratch_pos >= state.scratch.len() {
                    // mp3_decode_next set eof or yielded nothing.
                    return None;
                }
            }
            let s = state.scratch[state.scratch_pos];
            state.scratch_pos += 1;
            Some(s)
        }
    }
}

/// Pull one stereo (L, R) frame from a mixer track. Handles mono ->
/// stereo expansion, applies pan / volume / mute / solo. `solo_active`
/// is true when *any* track has solo on; non-solo tracks get virtual-
/// muted in that case.
///
/// Returns (left, right). On EOF: if loop_enabled, seeks to 0 and
/// continues; otherwise sets is_playing=false and returns silence.
fn pull_frame(track: &mut MixerTrack, solo_active: bool) -> (f32, f32) {
    if !track.is_playing {
        return (0.0, 0.0);
    }
    let dec = match track.decoded.as_mut() {
        Some(d) => d,
        None => return (0.0, 0.0),
    };
    // Read raw frame from file.
    let (raw_l, raw_r) = match dec.channels {
        1 => {
            let s = match read_one_sample(dec) {
                Some(v) => v,
                None => {
                    if track.loop_enabled {
                        let _ = rewind_decoder(dec);
                        track.cursor_frames = 0;
                        match read_one_sample(dec) {
                            Some(v) => v,
                            None => {
                                track.is_playing = false;
                                return (0.0, 0.0);
                            }
                        }
                    } else {
                        track.is_playing = false;
                        return (0.0, 0.0);
                    }
                }
            };
            (s, s)
        }
        2 => {
            let l = match read_one_sample(dec) {
                Some(v) => v,
                None => {
                    if track.loop_enabled {
                        let _ = rewind_decoder(dec);
                        track.cursor_frames = 0;
                        match read_one_sample(dec) {
                            Some(v) => v,
                            None => {
                                track.is_playing = false;
                                return (0.0, 0.0);
                            }
                        }
                    } else {
                        track.is_playing = false;
                        return (0.0, 0.0);
                    }
                }
            };
            let r = match read_one_sample(dec) {
                Some(v) => v,
                None => l, // truncated stereo file: duplicate L
            };
            (l, r)
        }
        n => {
            // Multichannel: take first two, drop the rest.
            let l = read_one_sample(dec).unwrap_or(0.0);
            let r = read_one_sample(dec).unwrap_or(l);
            for _ in 2..n {
                let _ = read_one_sample(dec);
            }
            (l, r)
        }
    };
    track.cursor_frames += 1;

    // Apply mute / solo.
    let muted_effective = track.muted || (solo_active && !track.solo);
    if muted_effective {
        return (0.0, 0.0);
    }

    // Equal-power pan: pan in [-1, 1] -> angle in [0, pi/2].
    // L gain = cos(angle), R gain = sin(angle). pan=0 -> L=R=sqrt(0.5).
    let pan = track.pan.clamp(-1.0, 1.0);
    let theta = (pan + 1.0) * 0.25 * std::f32::consts::PI;
    let pan_l = theta.cos();
    let pan_r = theta.sin();

    let vol = track.volume.clamp(0.0, 2.0);
    let out_l = raw_l * vol * pan_l;
    let out_r = raw_r * vol * pan_r;
    (out_l, out_r)
}

/// Shared state owned by both audio callback and UI thread. The Mutex
/// is held briefly inside the audio callback (long enough to fill one
/// device buffer, typically ~5 ms at 44.1 kHz / 256 frames). Spikes are
/// possible but acceptable for an MVP A/B player; lock-free ringbuffer
/// would be premature optimisation here.
struct MixerState {
    tracks: Vec<MixerTrack>,
    /// True when any track has had play() pressed; flipped to false
    /// when stop_all is hit.
    transport_running: bool,
}

impl MixerState {
    fn new(initial_slots: usize) -> Self {
        let mut tracks = Vec::new();
        for _ in 0..initial_slots {
            tracks.push(MixerTrack::empty());
        }
        Self {
            tracks,
            transport_running: false,
        }
    }

    fn solo_active(&self) -> bool {
        self.tracks.iter().any(|t| t.solo)
    }
}

/// Owns the live cpal::Stream. Dropping this drops the stream and
/// disconnects from the audio device. `state` is shared with the UI
/// thread for reads / writes to track parameters.
struct Mixer {
    state: Arc<Mutex<MixerState>>,
    _stream: cpal::Stream,
    sample_rate: u32,
    output_channels: u16,
    device_name: String,
}

fn build_mixer_stream(device_name: &str, state: Arc<Mutex<MixerState>>) -> Result<Mixer, String> {
    let host = cpal::default_host();
    let device = if device_name == "(default)" {
        host.default_output_device()
            .ok_or_else(|| "no default output device".to_string())?
    } else {
        let mut found = None;
        let iter = host
            .output_devices()
            .map_err(|e| format!("enumerate: {e}"))?;
        for d in iter {
            if d.name().ok().as_deref() == Some(device_name) {
                found = Some(d);
                break;
            }
        }
        found.ok_or_else(|| format!("device not found: {device_name}"))?
    };
    let resolved_name = device.name().unwrap_or_else(|_| device_name.to_string());
    let supported = device
        .default_output_config()
        .map_err(|e| format!("default_output_config: {e}"))?;
    let sample_format = supported.sample_format();
    let sample_rate = supported.sample_rate().0;
    let channels = supported.channels();
    let stream_cfg: StreamConfig = supported.into();

    let err_fn = |err| eprintln!("jukebox audio stream error: {err}");

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
                .map_err(|e| format!("build f32 stream: {e}"))?
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
                .map_err(|e| format!("build i16 stream: {e}"))?
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

    Ok(Mixer {
        state,
        _stream: stream,
        sample_rate,
        output_channels: channels,
        device_name: resolved_name,
    })
}

/// Audio callback body. Called on the cpal audio thread; locks the
/// shared state, pulls one frame per track, sums and writes into the
/// interleaved output buffer.
fn fill_callback(out: &mut [f32], channels: u16, state: &Arc<Mutex<MixerState>>) {
    // Zero first so any early-return leaves silence (not stale data).
    for s in out.iter_mut() {
        *s = 0.0;
    }
    let mut st = match state.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    if !st.transport_running {
        // Still update VU smoothing toward zero so meters fall.
        for t in st.tracks.iter_mut() {
            t.vu_peak_l *= 0.85;
            t.vu_peak_r *= 0.85;
            t.vu_rms_l *= 0.85;
            t.vu_rms_r *= 0.85;
        }
        return;
    }
    let solo = st.solo_active();
    let frames = out.len() / channels.max(1) as usize;

    // Per-track VU accumulators for this block.
    let n = st.tracks.len();
    let mut sum_sq_l: Vec<f32> = vec![0.0; n];
    let mut sum_sq_r: Vec<f32> = vec![0.0; n];
    let mut peak_l: Vec<f32> = vec![0.0; n];
    let mut peak_r: Vec<f32> = vec![0.0; n];
    let mut counted: Vec<u32> = vec![0; n];

    for f in 0..frames {
        let mut mix_l = 0.0f32;
        let mut mix_r = 0.0f32;
        for (i, t) in st.tracks.iter_mut().enumerate() {
            if t.decoded.is_none() || !t.is_playing {
                continue;
            }
            let (l, r) = pull_frame(t, solo);
            mix_l += l;
            mix_r += r;
            // Track-level peak / RMS (post-volume, post-pan, but without
            // the bus sum) so the VU reflects this slot's contribution.
            peak_l[i] = peak_l[i].max(l.abs());
            peak_r[i] = peak_r[i].max(r.abs());
            sum_sq_l[i] += l * l;
            sum_sq_r[i] += r * r;
            counted[i] += 1;
        }
        // Soft-clip the bus to [-1, 1] so a 4-track sum-to-1.5 doesn't
        // ruin the device. tanh is gentle and consistent with the live
        // synth's master path.
        let bus_l = mix_l.tanh();
        let bus_r = mix_r.tanh();
        let base = f * channels as usize;
        match channels {
            1 => {
                out[base] = (bus_l + bus_r) * 0.5;
            }
            2 => {
                out[base] = bus_l;
                out[base + 1] = bus_r;
            }
            n => {
                // Multichannel device: write L/R to channels 0/1, zero
                // the rest. Good enough for stereo content on a 5.1.
                out[base] = bus_l;
                out[base + 1] = bus_r;
                for c in 2..n as usize {
                    out[base + c] = 0.0;
                }
            }
        }
    }

    // Write back VU stats. Smooth with previous block (one-pole
    // ~50 ms decay) so the meter doesn't strobe.
    let alpha = 0.4f32; // weight of new block
    for (i, t) in st.tracks.iter_mut().enumerate() {
        let cnt = counted[i].max(1) as f32;
        let new_rms_l = (sum_sq_l[i] / cnt).sqrt();
        let new_rms_r = (sum_sq_r[i] / cnt).sqrt();
        t.vu_peak_l = t.vu_peak_l * (1.0 - alpha) + peak_l[i] * alpha;
        t.vu_peak_r = t.vu_peak_r * (1.0 - alpha) + peak_r[i] * alpha;
        t.vu_rms_l = t.vu_rms_l * (1.0 - alpha) + new_rms_l * alpha;
        t.vu_rms_r = t.vu_rms_r * (1.0 - alpha) + new_rms_r * alpha;
    }
}

// ---------- App -----------------------------------------------------

// ---------------------------------------------------------------------------
// Control-Protocol (CP) snapshot + command types — shared with the
// embedded `gui_cp` server so external tooling can drive verification
// without scripting the windowing layer. See `verify_jukebox.sh` and
// the user-path verification skill for the consumer side.
// ---------------------------------------------------------------------------

/// Per-mix-slot summary published over CP. Only the fields a
/// verification script actually needs to reason about playback state
/// — full mixer detail (VU peaks, pan, etc.) stays internal.
#[derive(Clone, Debug, Serialize)]
struct CpSlotSnap {
    slot: usize,
    /// `Track::label` of the file currently in this slot, or `None`.
    label: Option<String>,
    is_playing: bool,
    cursor_frames: u64,
    total_frames: u64,
    sample_rate: u32,
}

/// Wire-format snapshot of the jukebox state. Refreshed at the tail
/// of every `update()` frame; CP `get_state` reads serve a clone.
#[derive(Clone, Debug, Default, Serialize)]
struct CpJukeboxSnapshot {
    /// Increments every painted frame so a verification script can
    /// confirm the GUI is alive (and wait for command effects to land
    /// in the next frame).
    frame_id: u64,
    /// Cataloged tracks (jukebox-side, includes WAV/MP3/MIDI rows).
    track_count: usize,
    /// Voice rows materialized in `bench-out/library.db`. Mirrors the
    /// header line that proves DB discovery succeeded.
    db_voice_count: usize,
    /// Songs in `bench-out/library.db` keyed by stem. Together with
    /// `db_voice_count` this matches the stdout boot line
    /// `library_db catalog → N songs / M voices`.
    db_song_count: usize,
    /// Currently-selected catalog row label, or `None`.
    selected_label: Option<String>,
    /// Aggregate transport state: any slot is_playing.
    any_playing: bool,
    /// Per-slot detail (length matches `slot_count`).
    slots: Vec<CpSlotSnap>,
    /// Number of (label, voice) pairs that already have a preview WAV
    /// on disk. The verification script asserts this matches the
    /// disk-scan in `bench-out/cache/`.
    cache_hit_pairs: usize,
    /// Distinct labels that have at least one cached voice. Lighter to
    /// post-filter against on the consumer side than the full map.
    cached_labels: Vec<String>,
    /// Effective voice id per `Track::label` (empty when the user
    /// hasn't picked one yet — caller can still fall back to the
    /// jukebox's heuristic by querying `get_state` after a
    /// `set_voice`).
    selected_voices: HashMap<String, String>,
    /// True while the background prewarm worker is still draining its
    /// queue.
    prewarm_active: bool,
    /// Every catalog row's `Track::label`, in display order. Verifiers
    /// use this to pick a real label for `load_track` without having
    /// to glob `bench-out/songs/`. Kept on the snapshot so it stays
    /// in sync with `track_count` automatically.
    all_labels: Vec<String>,
}

/// Commands CP handlers can enqueue for the egui thread to apply on
/// the next frame. Stays small on purpose — anything more ambitious
/// belongs as a dedicated method that locks its own state.
#[derive(Clone, Debug)]
enum CpJukeboxCommand {
    /// Find a catalog row whose `Track::label` matches and start it
    /// in slot 0 (same code path as the per-row "▶" button).
    LoadTrack { label: String },
    /// Resume playback for whatever is already loaded.
    Play,
    /// Stop every slot. Equivalent to the header "■ stop" button.
    Stop,
    /// Persist a voice pick for `label`; if the row currently plays,
    /// re-trigger so the user hears the new voice.
    SetVoice { label: String, voice: String },
    /// Re-scan source dirs + library.db (matches the "rescan files"
    /// header button). Mostly here so a verification script can force
    /// a refresh after staging a new bench-out file.
    Rescan,
}

struct Jukebox {
    tracks: Vec<Track>,
    selected: Option<usize>,
    filter: String,
    /// Audio mixer (single cpal::Stream). Replaced on device change.
    mixer: Mixer,
    refresh_dirs: Vec<PathBuf>,
    /// Available output devices' display names. Re-polled when the user
    /// clicks "rescan dev" so newly-plugged devices appear.
    devices: Vec<String>,
    /// Currently bound device name (the one whose stream we hold).
    /// "(default)" means cpal default at startup.
    current_device: String,
    /// Map from mix slot -> tracks index (so the combo box state lives
    /// outside the audio-locked MixerState).
    slot_file_idx: Vec<Option<usize>>,
    /// Mirror of mixer.state's track count, set when the user hits
    /// "+ track" or "×".
    slot_count: usize,
    /// Pending background render. Set when the user clicks a MIDI
    /// track on cache miss; the GUI thread spawns a worker that calls
    /// `render_to_cache` and signals completion through this channel.
    /// `update()` polls the receiver every frame and finalises the
    /// playback hand-off once a path arrives. The `tracks` index is
    /// kept so the user can see which row is being rendered, and so
    /// finished renders for stale clicks (user already moved on) don't
    /// clobber the currently-playing track.
    pending_render: Option<PendingRender>,
    /// DB-backed song metadata index keyed by file stem
    /// (e.g. "bach_bwv999_prelude"). Lookup-only — the catalog is
    /// rebuilt at startup from `bench-out/songs/manifest.json` so the
    /// GUI never writes here. Populated to `None` on environments
    /// without a writable bench-out/library.db (e.g. read-only mount,
    /// CI image without the dir) so the rest of the app stays
    /// functional.
    song_index: HashMap<String, Song>,
    /// Total voice rows reported by `keysynth-db` at startup. Surfaces
    /// in the header so the user can see at-a-glance whether discovery
    /// found the expected modeled-voices set.
    db_voice_count: usize,
    /// Local play-history + favorites store
    /// (`bench-out/play_log.db`, gitignored). `None` when the file is
    /// unwritable so the rest of the jukebox keeps functioning — every
    /// read of this field treats `None` as "no history yet".
    play_log: Option<PlayLogDb>,
    /// In-memory mirror of the favorites table; the DB stays
    /// authoritative. Refreshed after every toggle so the central
    /// panel can render ★ glyphs without a per-row SELECT.
    favorites: HashSet<String>,
    /// Lifetime play counts keyed by `Track::label` (file stem).
    /// Refreshed after each `record_play` so the side panel and
    /// per-row badges stay in sync without polling.
    play_counts: HashMap<String, i64>,
    /// Most-recent-first slice of the play history, capped at
    /// `RECENT_PLAYS_LIMIT`. Used by the right side panel.
    recent_plays: Vec<PlayEntry>,
    /// When true the central catalogue is filtered down to just the
    /// rows whose `Track::label` is in `favorites`. Lets the user
    /// flick between "everything" and "my list" without retyping a
    /// substring filter.
    favorites_only: bool,
    /// Per-track-label set of voice ids whose preview WAV is currently
    /// in `bench-out/cache/`. Drives the "✓" hit glyph on each MIDI
    /// row so users can see at a glance whether their currently-picked
    /// voice will play instantly. Refreshed on startup, after
    /// `rescan`, after a render completes, and after the user picks a
    /// new voice from the per-row dropdown.
    cached_midi_labels: HashMap<String, HashSet<String>>,
    /// Per-track-label voice id the user has picked from the row's
    /// dropdown. Persisted in `play_log.track_voices` so the next
    /// session re-opens with the same picks; absent labels fall back
    /// to `default_voice_for_midi(path)`.
    selected_voices: HashMap<String, String>,
    /// Optional background prewarmer that walks the Top-N MIDI tracks
    /// at startup and renders each through `preview_cache::render_to_cache`
    /// so the user's first click on a popular song is a cache hit.
    /// `None` once the worker has finished or never started.
    prewarm: Option<PrewarmState>,
    /// DB-driven catalogue filters. Each is a value pulled from the
    /// matching `Song` column ("Baroque", "Public Domain", "guitar"…)
    /// and `None` means "show all". A track without DB metadata is
    /// hidden whenever any of these is set, since by definition we
    /// can't tell whether it matches.
    era_filter: Option<String>,
    license_filter: Option<String>,
    instrument_filter: Option<String>,
    selected_label: Option<String>,
    user_tags: HashMap<String, Vec<String>>,
    notes: HashMap<String, String>,
    note_draft: String,
    note_draft_song: Option<String>,
    custom_tag_input: String,
    cp_state: gui_cp::State<CpJukeboxSnapshot, CpJukeboxCommand>,
    _cp_handle: Option<gui_cp::Handle>,
    cp_frame_id: u64,
}

const RECENT_PLAYS_LIMIT: usize = 12;
const TOP_PLAYED_LIMIT: usize = 5;
/// Number of MIDI tracks the startup prewarmer pre-renders. Picked to
/// dominate the user's first listening session without paying the full
/// (#songs × #voices) wall time on launch — typical bench-out/songs/
/// has ~14 entries, so 6 covers most of "what was I listening to last
/// time" without delaying the GUI's first paint.
const PREWARM_TOP_N: usize = 6;

/// Background pre-render driver. Owns the worker thread and a one-shot
/// channel that streams completion records back to the GUI. The worker
/// renders sequentially: parallel renders would compete for the same
/// `target/release/render_midi` subprocess and disk bandwidth without a
/// real perceived-latency win for a 6-track warmup.
struct PrewarmState {
    rx: std::sync::mpsc::Receiver<PrewarmDone>,
    /// JoinHandle parked here so the thread isn't detached. Held until
    /// the channel disconnects (worker has finished its queue).
    _handle: std::thread::JoinHandle<()>,
}

/// One completion record emitted by the prewarm worker. `label` is the
/// `Track::label` (file stem) — we send only the label, not an index,
/// because the `tracks` Vec can be re-sorted by a `rescan` while the
/// worker is mid-queue.
struct PrewarmDone {
    label: String,
    /// Voice id the prewarmer rendered through. Carried alongside the
    /// label so the GUI can update `cached_midi_labels` for the right
    /// (label, voice) pair — picking a different voice from the GUI
    /// is racy with the prewarm thread, so the worker's report has to
    /// be self-describing.
    voice_id: String,
    /// `Some(elapsed_ms)` on a successful render, `None` if the cache
    /// already had the entry (cheap stat-only path), `Err(msg)` if
    /// render_midi failed.
    outcome: Result<Option<u128>, String>,
}

/// One in-flight MIDI → WAV render dispatched from the GUI thread.
struct PendingRender {
    track_idx: usize,
    label: String,
    /// Voice id this render is producing. Stored so the completion
    /// handler can update `cached_midi_labels` for the correct
    /// (label, voice) pair even when the user has since picked a
    /// different voice on the row.
    voice_id: String,
    started: std::time::Instant,
    rx: std::sync::mpsc::Receiver<Result<PathBuf, String>>,
    /// JoinHandle is parked here so the worker thread isn't detached
    /// (it still runs to completion if dropped, but holding the handle
    /// is the standard lifecycle pattern).
    _handle: std::thread::JoinHandle<()>,
}

/// Open `bench-out/library.db`, refresh the songs/voices import from
/// disk, and return a stem-keyed song index plus the total voice row
/// count. Failures are non-fatal: an unwritable DB just yields an
/// empty index so the jukebox keeps working without DB metadata. The
/// stderr log line documents the fallback for the operator.
fn load_library_db_index() -> (HashMap<String, Song>, usize) {
    let db_path = PathBuf::from("bench-out/library.db");
    let manifest = PathBuf::from("bench-out/songs/manifest.json");
    let voices_live = PathBuf::from("voices_live");
    let mut db = match LibraryDb::open(&db_path) {
        Ok(db) => db,
        Err(e) => {
            eprintln!(
                "jukebox: library_db open failed ({}): {e} — running without DB metadata",
                db_path.display()
            );
            return (HashMap::new(), 0);
        }
    };
    if let Err(e) = db.migrate() {
        eprintln!("jukebox: library_db migrate failed: {e}");
        return (HashMap::new(), 0);
    }
    if manifest.is_file() {
        if let Err(e) = db.import_songs(&manifest) {
            eprintln!("jukebox: library_db import_songs failed: {e}");
        }
    }
    if voices_live.is_dir() {
        if let Err(e) = db.import_voices(&voices_live) {
            eprintln!("jukebox: library_db import_voices failed: {e}");
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
    let mut index = HashMap::new();
    for s in songs {
        index.insert(s.id.clone(), s);
    }
    (index, voice_count)
}

/// Open the local play-log database. Failures degrade gracefully so a
/// read-only working tree (e.g. a CI image) still lets the user browse
/// the catalogue — they just lose history persistence for the session.
fn load_play_log() -> Option<PlayLogDb> {
    let db_path = PathBuf::from("bench-out/play_log.db");
    let mut db = match PlayLogDb::open(&db_path) {
        Ok(db) => db,
        Err(e) => {
            eprintln!(
                "jukebox: play_log open failed ({}): {e} — running without history",
                db_path.display()
            );
            return None;
        }
    };
    if let Err(e) = db.migrate() {
        eprintln!("jukebox: play_log migrate failed: {e} — running without history");
        return None;
    }
    Some(db)
}

/// Cheap "5m ago" / "2h ago" / "3d ago" formatter used by the recent-
/// plays panel. Keeps the side panel readable without dragging in a
/// `chrono` / `humantime` dependency for one-line text.
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

impl Jukebox {
    fn new(dirs: Vec<PathBuf>) -> Result<Self, String> {
        let (song_index, db_voice_count) = load_library_db_index();
        let dir_refs: Vec<&Path> = dirs.iter().map(|p| p.as_path()).collect();
        let mut tracks = scan_dirs(&dir_refs);
        // For bench-out/songs/* MIDI tracks the DB is the source of
        // truth for piece display: it carries composer / era / license
        // metadata that the on-disk filename can't. Replace the
        // filename-derived `piece` with the DB title for any MIDI row
        // whose stem matches a manifest entry.
        if !song_index.is_empty() {
            for t in tracks.iter_mut() {
                if t.format != Format::Mid {
                    continue;
                }
                if let Some(song) = song_index.get(&t.label) {
                    t.piece = song.id.clone();
                }
            }
            // Resort: songs whose stem is in the DB sort by composer
            // (DB-driven order); everything else keeps the filename
            // ordering after them. Matches `SongFilter { sort:
            // ByComposer }` from the brief while preserving the
            // iterH / CHIPTUNE rendering rows the DB doesn't know
            // about.
            tracks.sort_by(|a, b| {
                let ka = song_index.get(&a.label).map(|s| s.composer_key.clone());
                let kb = song_index.get(&b.label).map(|s| s.composer_key.clone());
                match (ka, kb) {
                    (Some(x), Some(y)) => x.cmp(&y).then_with(|| a.piece.cmp(&b.piece)),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => a.piece.cmp(&b.piece).then_with(|| a.engine.cmp(&b.engine)),
                }
            });
        }
        // Diagnostic line consumed by E2E gate 7 — proves the DB path
        // is reachable from the GUI binary even on an air-gapped /
        // headless launch. Kept on stdout (not stderr) so a
        // `keysynth jukebox | head -1` smoke captures it cheaply.
        println!(
            "jukebox: library_db catalog \u{2192} {} songs / {} voices",
            song_index.len(),
            db_voice_count,
        );
        let devices = list_output_devices();
        let initial_slots = 2usize;
        let state = Arc::new(Mutex::new(MixerState::new(initial_slots)));
        let mixer =
            build_mixer_stream("(default)", state).map_err(|e| format!("audio output: {e}"))?;
        let current_device = mixer.device_name.clone();
        let play_log = load_play_log();
        let (favorites, play_counts, recent_plays) = snapshot_play_stats(play_log.as_ref());
        let selected_voices = play_log
            .as_ref()
            .and_then(|db| db.track_voices().ok())
            .unwrap_or_default();
        let cached_midi_labels = snapshot_cached_midi_voices(&tracks, &selected_voices);
        let prewarm =
            spawn_startup_prewarm(&tracks, &play_counts, &cached_midi_labels, &selected_voices);
        let user_tags = play_log
            .as_ref()
            .and_then(|db| db.all_user_tags().ok())
            .unwrap_or_default();
        let notes = play_log
            .as_ref()
            .and_then(|db| db.all_notes().ok())
            .unwrap_or_default();
        let cp_state: gui_cp::State<CpJukeboxSnapshot, CpJukeboxCommand> = gui_cp::State::new();
        let cp_handle = match spawn_cp_server(cp_state.clone()) {
            Ok(h) => Some(h),
            Err(e) => {
                eprintln!("jukebox: CP server failed to start: {e} — verification disabled");
                None
            }
        };
        let mut state = Self {
            tracks,
            selected: None,
            filter: String::new(),
            mixer,
            refresh_dirs: dirs,
            devices,
            current_device,
            slot_file_idx: vec![None; initial_slots],
            slot_count: initial_slots,
            pending_render: None,
            song_index,
            db_voice_count,
            play_log,
            favorites,
            play_counts,
            recent_plays,
            favorites_only: false,
            cached_midi_labels,
            selected_voices,
            prewarm,
            era_filter: None,
            license_filter: None,
            instrument_filter: None,
            selected_label: None,
            user_tags,
            notes,
            note_draft: String::new(),
            note_draft_song: None,
            custom_tag_input: String::new(),
            cp_state,
            _cp_handle: cp_handle,
            cp_frame_id: 0,
        };
        // Open the detail pane on the most-recently-played track (or
        // first ★ favorite, or first track) so the user has a concrete
        // example of what marking + notes look like the moment the GUI
        // opens. Also confirms the panel renders during E2E verify.
        let initial = state
            .recent_plays
            .first()
            .map(|e| e.song_id.clone())
            .or_else(|| state.favorites.iter().next().cloned())
            .or_else(|| state.tracks.first().map(|t| t.label.clone()));
        if let Some(stem) = initial {
            state.set_selected_label(&stem);
        }
        Ok(state)
    }

    /// Built-in marker palette surfaced at the top of the detail pane
    /// so a brand-new user has obvious next steps. Custom strings can
    /// still be added via the "+ tag" input — order here is the
    /// preferred display sequence.
    const KNOWN_TAGS: &'static [(&'static str, &'static str, &'static str)] = &[
        ("study", "\u{1F393}", "Mark as material to learn — theory, structure, technique."),
        ("perform", "\u{1F3A4}", "Want to play this myself."),
        ("revisit", "\u{2753}", "Come back to this later — undecided."),
        ("compare", "\u{1F4DD}", "Reference for A/B against another arrangement."),
    ];

    /// Set the focused row and bind the note-edit buffer to it. Called
    /// on row click. Idempotent: re-focusing the same row does not
    /// overwrite an in-flight edit.
    fn set_selected_label(&mut self, label: &str) {
        if self.selected_label.as_deref() == Some(label) {
            return;
        }
        self.flush_note_draft();
        self.selected_label = Some(label.to_string());
        self.note_draft = self.notes.get(label).cloned().unwrap_or_default();
        self.note_draft_song = Some(label.to_string());
    }

    /// Persist the active note draft to SQLite if it changed. Called
    /// before switching rows and from the explicit "Save note" button
    /// so the user controls when the database write happens.
    fn flush_note_draft(&mut self) {
        let Some(song) = self.note_draft_song.clone() else {
            return;
        };
        let prev = self.notes.get(&song).cloned().unwrap_or_default();
        let current = self.note_draft.trim().to_string();
        if current == prev.trim() {
            return;
        }
        if let Some(db) = self.play_log.as_mut() {
            if let Err(e) = db.set_note(&song, &current) {
                eprintln!("jukebox: set_note({song}): {e}");
                return;
            }
        }
        if current.is_empty() {
            self.notes.remove(&song);
        } else {
            self.notes.insert(song, current);
        }
    }

    /// Toggle a single user_tag for `song_id` and refresh in-memory
    /// mirror so the row chips and detail pane update in the same
    /// frame.
    fn toggle_user_tag(&mut self, song_id: &str, tag: &str) {
        let tag = tag.trim();
        if tag.is_empty() {
            return;
        }
        let Some(db) = self.play_log.as_mut() else {
            return;
        };
        match db.toggle_user_tag(song_id, tag) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("jukebox: toggle_user_tag({song_id}, {tag}): {e}");
                return;
            }
        }
        // Re-read just this song's tags rather than re-snapshotting the
        // whole table — keeps the cost flat as the catalogue grows.
        match db.user_tags(song_id) {
            Ok(tags) => {
                if tags.is_empty() {
                    self.user_tags.remove(song_id);
                } else {
                    self.user_tags.insert(song_id.to_string(), tags);
                }
            }
            Err(e) => {
                eprintln!("jukebox: refresh user_tags({song_id}): {e}");
            }
        }
    }

    /// Drain any prewarm completions queued since the last frame and
    /// fold them into `cached_midi_labels` so the "✓" glyph appears
    /// on a row the moment its background render finishes.
    fn poll_prewarm(&mut self) {
        let Some(state) = self.prewarm.as_ref() else {
            return;
        };
        let mut disconnected = false;
        loop {
            match state.rx.try_recv() {
                Ok(done) => match done.outcome {
                    Ok(Some(ms)) => {
                        eprintln!(
                            "jukebox: prewarm rendered {} ({}) in {} ms",
                            done.label, done.voice_id, ms
                        );
                        self.cached_midi_labels
                            .entry(done.label)
                            .or_default()
                            .insert(done.voice_id);
                    }
                    Ok(None) => {
                        self.cached_midi_labels
                            .entry(done.label)
                            .or_default()
                            .insert(done.voice_id);
                    }
                    Err(e) => {
                        eprintln!(
                            "jukebox: prewarm failed for {} ({}): {e}",
                            done.label, done.voice_id
                        );
                    }
                },
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        if disconnected {
            self.prewarm = None;
        }
    }

    /// Resolve the effective voice id the user has picked for `label`.
    /// Falls back to the heuristic default when the user hasn't
    /// touched the picker for this track yet.
    fn voice_for(&self, label: &str, midi_path: &Path) -> String {
        self.selected_voices
            .get(label)
            .cloned()
            .unwrap_or_else(|| default_voice_for_midi(midi_path).to_string())
    }

    /// Persist the user's voice pick for `label` and re-stat the cache
    /// for the new (label, voice) pair so the "✓" glyph reflects
    /// disk truth in the same frame. Returns true when the pick
    /// actually changed (used by the UI to decide whether to fire
    /// `preview()` immediately so the user hears the new voice).
    fn set_track_voice(&mut self, label: &str, midi_path: &Path, voice_id: &str) -> bool {
        let prev = self.selected_voices.get(label).cloned();
        if prev.as_deref() == Some(voice_id) {
            return false;
        }
        self.selected_voices
            .insert(label.to_string(), voice_id.to_string());
        if let Some(db) = self.play_log.as_mut() {
            if let Err(e) = db.set_track_voice(label, voice_id) {
                eprintln!("jukebox: set_track_voice({label}, {voice_id}): {e}");
            }
        }
        refresh_cached_voice_entry(
            &mut self.cached_midi_labels,
            label,
            midi_path,
            voice_id,
        );
        true
    }

    /// Pull cached favorites / counts / recent-plays from the DB.
    /// Called after every mutation (record_play, toggle_favorite) so
    /// the side panel reflects the new state without holding a live
    /// borrow into `play_log` from the egui closures.
    fn refresh_play_stats(&mut self) {
        let (favs, counts, recent) = snapshot_play_stats(self.play_log.as_ref());
        self.favorites = favs;
        self.play_counts = counts;
        self.recent_plays = recent;
    }

    /// Apply a batch of `(song_id, fav)` favorite assignments and
    /// refresh caches once at the end. Used by the per-piece ★ button:
    /// a single click can flip several engine renders that share a
    /// piece root, and we want the side panel to settle in one frame
    /// instead of mid-loop.
    fn set_favorites_batch(&mut self, ops: &[(String, bool)]) {
        if ops.is_empty() {
            return;
        }
        let Some(db) = self.play_log.as_mut() else {
            return;
        };
        for (song_id, fav) in ops {
            if let Err(e) = db.set_favorite(song_id, *fav) {
                eprintln!("jukebox: set_favorite({song_id}, {fav}): {e}");
            }
        }
        self.refresh_play_stats();
    }

    /// Log a play row keyed by `Track::label` (the catalog stem).
    /// Called every time `load_resolved_track` actually starts audio.
    fn record_play_for(&mut self, idx: usize) {
        let Some(track) = self.tracks.get(idx) else {
            return;
        };
        let song_id = track.label.clone();
        let voice_id = track.engine.clone();
        let Some(db) = self.play_log.as_mut() else {
            return;
        };
        let voice = if voice_id.is_empty() || voice_id == "—" {
            None
        } else {
            Some(voice_id.as_str())
        };
        if let Err(e) = db.record_play(&song_id, voice, None) {
            eprintln!("jukebox: record_play({song_id}): {e}");
            return;
        }
        self.refresh_play_stats();
    }

    /// Find the first track matching `song_id` (which is `Track::label`)
    /// and start it. Used by the side-panel "▶" buttons.
    fn play_song_id(&mut self, song_id: &str) {
        let idx = self.tracks.iter().position(|t| t.label == song_id);
        if let Some(i) = idx {
            self.preview(i);
        }
    }
}

/// Pick the first `PREWARM_TOP_N` MIDI tracks worth pre-rendering.
/// Ordering: descending lifetime play count (so the user's actual
/// favourites win), with stem-alphabetical fallback so a fresh DB
/// still pre-warms a deterministic slate. Already-cached labels are
/// skipped so a second jukebox launch doesn't re-queue work it's
/// already done.
fn pick_prewarm_targets(
    tracks: &[Track],
    play_counts: &HashMap<String, i64>,
    cached: &HashMap<String, HashSet<String>>,
    selected_voices: &HashMap<String, String>,
) -> Vec<(PathBuf, String)> {
    let mut candidates: Vec<(&Track, i64, String)> = tracks
        .iter()
        .filter(|t| t.format == Format::Mid)
        .map(|t| {
            let voice = selected_voices
                .get(&t.label)
                .cloned()
                .unwrap_or_else(|| default_voice_for_midi(&t.path).to_string());
            let already = cached
                .get(&t.label)
                .map(|s| s.contains(&voice))
                .unwrap_or(false);
            (t, already, voice)
        })
        .filter(|(_, already, _)| !already)
        .map(|(t, _, voice)| {
            let count = play_counts.get(&t.label).copied().unwrap_or(0);
            (t, count, voice)
        })
        .collect();
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.label.cmp(&b.0.label)));
    candidates
        .into_iter()
        .take(PREWARM_TOP_N)
        .map(|(t, _, voice)| (t.path.clone(), voice))
        .collect()
}

/// Kick off the background prewarm worker. Returns `None` when the
/// prewarm queue is empty (everything already cached, or no MIDIs
/// found) so callers don't have to wait on a no-op thread.
fn spawn_startup_prewarm(
    tracks: &[Track],
    play_counts: &HashMap<String, i64>,
    cached: &HashMap<String, HashSet<String>>,
    selected_voices: &HashMap<String, String>,
) -> Option<PrewarmState> {
    let queue = pick_prewarm_targets(tracks, play_counts, cached, selected_voices);
    if queue.is_empty() {
        return None;
    }
    let render_bin = render_midi_binary_path()?;
    let (tx, rx) = std::sync::mpsc::channel::<PrewarmDone>();
    eprintln!(
        "jukebox: prewarm queue → {} MIDI track(s) (top by play count)",
        queue.len()
    );
    let handle = std::thread::Builder::new()
        .name("preview-prewarm".into())
        .spawn(move || {
            // Re-open the cache inside the worker so we don't leak any
            // GUI-thread handle into the background; the on-disk
            // representation is the only shared state.
            let cache = match open_preview_cache() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("jukebox: prewarm cache open failed: {e}");
                    return;
                }
            };
            for (path, voice_id) in queue {
                let label = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("?")
                    .to_string();
                let key = build_midi_cache_key(&path, &voice_id);
                // Cache may have been populated by a peer process
                // (another jukebox window, a manual ksprerender run)
                // since `pick_prewarm_targets` last looked. Cheap stat
                // here saves a redundant subprocess spawn.
                let outcome = match cache.lookup(&key) {
                    Ok(Some(_)) => Ok(None),
                    Ok(None) => {
                        let t0 = Instant::now();
                        match keysynth::preview_cache::render_to_cache(
                            &cache,
                            &key,
                            &voice_id,
                            &render_bin,
                        ) {
                            Ok(_) => Ok(Some(t0.elapsed().as_millis())),
                            Err(e) => Err(format!("render_to_cache: {e}")),
                        }
                    }
                    Err(e) => Err(format!("lookup: {e}")),
                };
                // Receiver-drop on the GUI side just discards the
                // notification — the on-disk WAV still exists, so a
                // later `snapshot_cached_midi_voices` will pick it up.
                if tx
                    .send(PrewarmDone {
                        label,
                        voice_id,
                        outcome,
                    })
                    .is_err()
                {
                    return;
                }
            }
        })
        .ok()?;
    Some(PrewarmState {
        rx,
        _handle: handle,
    })
}

/// Pulls the small caches the GUI reads every frame out of the DB in
/// one shot. Returning a tuple keeps `Jukebox::new` and
/// `refresh_play_stats` symmetric.
fn snapshot_play_stats(
    db: Option<&PlayLogDb>,
) -> (HashSet<String>, HashMap<String, i64>, Vec<PlayEntry>) {
    let Some(db) = db else {
        return (HashSet::new(), HashMap::new(), Vec::new());
    };
    let favorites: HashSet<String> = db.favorites().unwrap_or_default().into_iter().collect();
    let play_counts = db.play_counts().unwrap_or_default();
    let recent_plays = db.recent_plays(RECENT_PLAYS_LIMIT).unwrap_or_default();
    (favorites, play_counts, recent_plays)
}

impl Jukebox {

    /// Reset all loaded tracks' cursor to 0 and start the transport.
    /// Sample-accurate: every track that's loaded begins from frame 0
    /// in the same audio callback iteration.
    fn play_all(&mut self) {
        let mut st = match self.mixer.state.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        for t in st.tracks.iter_mut() {
            if t.decoded.is_none() {
                continue;
            }
            if let Err(e) = t.seek_start() {
                eprintln!("jukebox: seek_start: {e}");
                continue;
            }
            t.is_playing = true;
        }
        st.transport_running = true;
    }

    fn stop_all(&mut self) {
        let mut st = match self.mixer.state.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        for t in st.tracks.iter_mut() {
            t.is_playing = false;
        }
        st.transport_running = false;
    }

    /// Single-track preview (the per-row "▶ engine" buttons in the
    /// catalogue). Loads the file into slot 0, leaves other slots
    /// muted-as-they-are, and starts the transport. Reuses the mixer
    /// machinery so we don't have a parallel rodio path.
    fn preview(&mut self, idx: usize) {
        // Cancel any in-flight render that's not for the track the
        // user just clicked. Without this, switching from a slow
        // uncached MIDI back to a cached track would cause the late
        // render to clobber the freshly-loaded playback when it
        // eventually completes — see issue #66 PR review feedback
        // ("再生中に他の曲が再生切り替え出来なくなる"). The old worker
        // thread keeps running but its result is discarded, since
        // dropping the receiver here turns its `tx.send` into a no-op
        // SendError. The cache file still gets populated, so a future
        // click on the cancelled track lands as an instant cache hit.
        if let Some(pending) = self.pending_render.as_ref() {
            if pending.track_idx != idx {
                eprintln!(
                    "jukebox: cancelling pending render for track #{} — user switched to #{}",
                    pending.track_idx, idx
                );
                self.pending_render = None;
            }
        }
        let track = match self.tracks.get(idx) {
            Some(t) => t.clone(),
            None => return,
        };
        let label = format!("[{}] {}_{}", track.source, track.piece, track.engine);

        // For MIDI tracks we go through the lazy preview cache (issue
        // #62 Phase 2). Two paths:
        //   - cache hit  → load synchronously on the GUI thread; the
        //                  stat + open is fast (sub-millisecond).
        //   - cache miss → spawn a background render worker. The
        //                  worker runs `render_midi` as a subprocess
        //                  (1–10 s for a typical short piece) without
        //                  touching the audio callback or the UI
        //                  thread. `update()` polls the channel and
        //                  starts playback once the WAV is ready.
        let playable_path: PathBuf = if track.format == Format::Mid {
            let voice = self.voice_for(&track.label, &track.path);
            match lookup_midi_in_preview_cache(&track.path, &voice) {
                Ok(Some(p)) => p,
                Ok(None) => {
                    self.spawn_midi_render(idx, label, voice);
                    return;
                }
                Err(e) => {
                    eprintln!(
                        "jukebox: preview_cache lookup {} ({voice}): {e}",
                        track.path.display(),
                    );
                    return;
                }
            }
        } else {
            track.path.clone()
        };
        self.load_resolved_track(idx, label, playable_path);
    }

    /// Spawn a background thread that renders `tracks[idx]` through the
    /// preview-cache subprocess path. Stores the pending state on
    /// `self.pending_render`; `poll_pending_render` (called from
    /// `update()`) finalises the playback hand-off when the worker
    /// finishes.
    fn spawn_midi_render(&mut self, idx: usize, label: String, voice_id: String) {
        let path = match self.tracks.get(idx) {
            Some(t) => t.path.clone(),
            None => return,
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let path_for_thread = path.clone();
        let voice_for_thread = voice_id.clone();
        let handle = std::thread::Builder::new()
            .name("preview-render".into())
            .spawn(move || {
                let result = render_midi_blocking(&path_for_thread, &voice_for_thread);
                // Receiver drop on the GUI side is non-fatal: we just
                // discard the rendered path. The cache file itself
                // already lives at its hashed location, so a future
                // click will see the cache hit.
                let _ = tx.send(result);
            })
            .expect("spawn preview-render thread");
        eprintln!(
            "jukebox: render queued ({}) — voice={} (background thread)",
            path.display(),
            voice_id,
        );
        self.pending_render = Some(PendingRender {
            track_idx: idx,
            label,
            voice_id,
            started: std::time::Instant::now(),
            rx,
            _handle: handle,
        });
    }

    /// Drain the pending-render channel. Called once per egui frame.
    /// Non-blocking: if the worker hasn't finished, returns
    /// immediately so the UI can keep redrawing at 60 fps. When the
    /// worker delivers a result, we load the WAV into slot 0 and
    /// start playback.
    fn poll_pending_render(&mut self) {
        let pending = match self.pending_render.as_ref() {
            Some(p) => p,
            None => return,
        };
        let result = match pending.rx.try_recv() {
            Ok(r) => r,
            Err(std::sync::mpsc::TryRecvError::Empty) => return, // still rendering
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                // Worker died without sending; clear and bail.
                self.pending_render = None;
                return;
            }
        };
        let pending = self.pending_render.take().expect("guarded above");
        let elapsed_ms = pending.started.elapsed().as_millis();
        match result {
            Ok(wav_path) => {
                eprintln!(
                    "jukebox: render done ({}) in {} ms",
                    wav_path.display(),
                    elapsed_ms,
                );
                // Mark the (label, voice) pair as cached BEFORE
                // handing off to `load_resolved_track` so the row's
                // "✓" glyph appears in the same frame the audio
                // starts.
                if let Some(t) = self.tracks.get(pending.track_idx) {
                    self.cached_midi_labels
                        .entry(t.label.clone())
                        .or_default()
                        .insert(pending.voice_id.clone());
                }
                self.load_resolved_track(pending.track_idx, pending.label, wav_path);
            }
            Err(e) => {
                eprintln!("jukebox: render failed: {e}");
            }
        }
    }

    /// Inner half of `preview()` after the path has been resolved.
    /// Pulled out so both the synchronous (cache hit) and asynchronous
    /// (cache miss → background render) paths reach the same playback
    /// state without duplicating mixer-locking boilerplate.
    fn load_resolved_track(&mut self, idx: usize, label: String, playable_path: PathBuf) {
        // Ensure we have at least one slot.
        {
            let mut st = self.mixer.state.lock().unwrap();
            if st.tracks.is_empty() {
                st.tracks.push(MixerTrack::empty());
                self.slot_file_idx.push(None);
                self.slot_count = st.tracks.len();
            }
        }
        self.stop_all();
        {
            let mut st = self.mixer.state.lock().unwrap();
            for (i, t) in st.tracks.iter_mut().enumerate() {
                if i == 0 {
                    if let Err(e) = t.load(&playable_path, label.clone()) {
                        eprintln!("jukebox: load {}: {e}", playable_path.display());
                        return;
                    }
                    t.is_playing = true;
                } else {
                    t.is_playing = false;
                }
            }
            st.transport_running = true;
            self.slot_file_idx[0] = Some(idx);
        }
        self.selected = Some(idx);
        // Append a row to play_log.db. Done after the mixer state is
        // committed so a failed load (handled above with an early
        // return) doesn't pollute the history with a play that never
        // produced audio.
        self.record_play_for(idx);
    }

    fn rebind_device(&mut self, name: &str) {
        // Snapshot the current mixer state so the new stream picks up
        // the same tracks / volumes / cursor.
        let state = self.mixer.state.clone();
        match build_mixer_stream(name, state) {
            Ok(m) => {
                self.current_device = m.device_name.clone();
                self.mixer = m;
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
        // Re-import the DB so a fresh manifest.json edit (or a new
        // voices_live/<name>/) shows up without restarting the jukebox.
        let (index, voice_count) = load_library_db_index();
        self.song_index = index;
        self.db_voice_count = voice_count;
        // The play_log is independent of the catalog rebuild, but the
        // counts/recent caches may be stale if another process wrote
        // them (e.g. a parallel kssong run); refresh them here too.
        self.refresh_play_stats();
        // New tracks may have shipped with their own pre-rendered
        // cache entries (e.g. a parallel `ksprerender` populated
        // `bench-out/cache/`); rebuild the hit set so the row glyph
        // is honest.
        self.cached_midi_labels =
            snapshot_cached_midi_voices(&self.tracks, &self.selected_voices);
    }

    /// Append a new empty mix slot. UI callable.
    fn add_slot(&mut self) {
        let mut st = self.mixer.state.lock().unwrap();
        st.tracks.push(MixerTrack::empty());
        self.slot_file_idx.push(None);
        self.slot_count = st.tracks.len();
    }

    fn remove_slot(&mut self, idx: usize) {
        let mut st = self.mixer.state.lock().unwrap();
        if idx < st.tracks.len() {
            st.tracks.remove(idx);
            self.slot_count = st.tracks.len();
        }
        if idx < self.slot_file_idx.len() {
            self.slot_file_idx.remove(idx);
        }
    }

    /// Apply slot_file_idx changes: load files into mixer tracks where
    /// the UI's chosen file differs from the currently-loaded one.
    fn sync_slot_files(&mut self) {
        // Build the list of (slot_idx, path, label) outside the lock so
        // we don't hold the audio Mutex during file open.
        let mut to_load: Vec<(usize, PathBuf, String)> = Vec::new();
        let mut to_unload: Vec<usize> = Vec::new();
        {
            let st = self.mixer.state.lock().unwrap();
            for (i, want) in self.slot_file_idx.iter().enumerate() {
                let cur_path = st.tracks.get(i).and_then(|t| {
                    if t.decoded.is_some() {
                        Some(t.file_path.clone())
                    } else {
                        None
                    }
                });
                // For MIDI tracks the audio decoder cannot read the .mid
                // path directly — it needs the rendered WAV from the
                // preview cache. If the cache is empty, skip the slot
                // sync entirely; `preview()` is the path that triggers
                // a render, so re-trying every frame here would just
                // spam errors against an empty cache.
                let want_path = want.and_then(|idx| self.tracks.get(idx)).and_then(|t| {
                    let label = format!("[{}] {}_{}", t.source, t.piece, t.engine);
                    if t.format == Format::Mid {
                        let voice_id = self.voice_for(&label, &t.path);
                        match lookup_midi_in_preview_cache(&t.path, &voice_id) {
                            Ok(Some(p)) => Some((p, label)),
                            _ => None,
                        }
                    } else {
                        Some((t.path.clone(), label))
                    }
                });
                match (cur_path, want_path) {
                    (None, Some((p, l))) => to_load.push((i, p, l)),
                    (Some(cur), Some((p, l))) if cur != p => to_load.push((i, p, l)),
                    (Some(_), None) => to_unload.push(i),
                    _ => {}
                }
            }
        }
        if !to_load.is_empty() || !to_unload.is_empty() {
            let mut st = self.mixer.state.lock().unwrap();
            for i in &to_unload {
                if let Some(t) = st.tracks.get_mut(*i) {
                    t.unload();
                }
            }
            for (i, p, l) in to_load {
                if let Some(t) = st.tracks.get_mut(i) {
                    if let Err(e) = t.load(&p, l) {
                        eprintln!("jukebox: load {}: {e}", p.display());
                    }
                }
            }
        }
    }

    /// Left side panel: deep-dive metadata + marking + free-form note
    /// for the row currently in `selected_label`. No-op (panel hidden)
    /// when nothing is selected so the catalogue can run full-width
    /// for the casual "shuffle and listen" workflow.
    fn render_detail_panel(&mut self, ctx: &egui::Context) {
        let label = match self.selected_label.clone() {
            Some(l) => l,
            None => return,
        };
        // Pull every read off `self` up-front so the closure body can
        // mutate `self.note_draft` / call `self.toggle_user_tag` etc.
        let song_meta = self.song_index.get(&label).cloned();
        let track_meta = self
            .tracks
            .iter()
            .find(|t| t.label == label)
            .map(|t| (t.path.clone(), t.engine.clone(), t.source.clone()));
        let plays = self.play_counts.get(&label).copied().unwrap_or(0);
        let last_played = self
            .play_log
            .as_ref()
            .and_then(|db| db.last_played_at(&label).ok().flatten());
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let assigned_tags: Vec<String> = self
            .user_tags
            .get(&label)
            .cloned()
            .unwrap_or_default();
        let known_keys: Vec<&str> = Self::KNOWN_TAGS.iter().map(|(k, _, _)| *k).collect();
        let custom_assigned: Vec<String> = assigned_tags
            .iter()
            .filter(|t| !known_keys.contains(&t.as_str()))
            .cloned()
            .collect();
        let is_favorite = self.favorites.contains(&label);

        let mut tag_ops: Vec<String> = Vec::new();
        let mut clear_selection = false;
        let mut close_save = false;
        let mut play_now = false;
        let mut toggle_fav = false;
        let mut open_url: Option<String> = None;

        egui::SidePanel::left("detail-panel")
            .resizable(true)
            .default_width(320.0)
            .min_width(260.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("\u{1F50D} research");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .small_button("\u{2715}")
                            .on_hover_text("Close detail panel")
                            .clicked()
                        {
                            clear_selection = true;
                        }
                    });
                });
                ui.separator();

                // Title block — DB title + raw stem so the user can
                // grep-confirm the catalogue row.
                let title = song_meta
                    .as_ref()
                    .map(|s| s.title.clone())
                    .unwrap_or_else(|| label.clone());
                ui.label(
                    egui::RichText::new(title)
                        .strong()
                        .size(15.0),
                );
                ui.label(
                    egui::RichText::new(&label)
                        .monospace()
                        .small()
                        .color(egui::Color32::from_rgb(140, 140, 160)),
                );
                if let Some((_, engine, source)) = &track_meta {
                    ui.label(
                        egui::RichText::new(format!("source: {source}  ·  engine: {engine}"))
                            .small()
                            .color(egui::Color32::from_rgb(150, 170, 200)),
                    );
                }

                ui.add_space(6.0);

                // Metadata grid (composer / era / instrument / license).
                if let Some(song) = &song_meta {
                    egui::Grid::new("detail-meta")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            meta_row(ui, "composer", &song.composer);
                            if let Some(e) = &song.era {
                                ui.label(egui::RichText::new("era").small().weak());
                                ui.label(
                                    egui::RichText::new(e)
                                        .color(era_color(e))
                                        .strong()
                                        .small(),
                                );
                                ui.end_row();
                            }
                            meta_row(ui, "instrument", &song.instrument);
                            ui.label(egui::RichText::new("license").small().weak());
                            let lic_short = license_short(&song.license);
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(lic_short.clone())
                                        .color(license_color(&lic_short))
                                        .strong()
                                        .monospace()
                                        .small(),
                                );
                                ui.label(
                                    egui::RichText::new(&song.license)
                                        .small()
                                        .color(egui::Color32::from_rgb(180, 180, 180)),
                                );
                            });
                            ui.end_row();
                            if let Some(v) = &song.suggested_voice {
                                meta_row(ui, "suggested voice", v);
                            }
                            if let Some(src) = &song.source {
                                meta_row(ui, "source", src);
                            }
                        });

                    if let Some(url) = song.source_url.as_deref() {
                        ui.add_space(4.0);
                        if ui
                            .add(
                                egui::Hyperlink::from_label_and_url(
                                    egui::RichText::new(format!("\u{1F517} {url}"))
                                        .small()
                                        .color(egui::Color32::from_rgb(140, 200, 255)),
                                    url,
                                )
                                .open_in_new_tab(true),
                            )
                            .on_hover_text(
                                "Open the manifest source URL in your default browser.",
                            )
                            .clicked()
                        {
                            open_url = Some(url.to_string());
                        }
                    }
                    if !song.tags.is_empty() {
                        ui.add_space(4.0);
                        ui.label(egui::RichText::new("manifest tags").small().weak());
                        ui.horizontal_wrapped(|ui| {
                            for tag in &song.tags {
                                ui.label(
                                    egui::RichText::new(format!("#{tag}"))
                                        .small()
                                        .color(egui::Color32::from_rgb(200, 200, 230))
                                        .background_color(egui::Color32::from_rgb(40, 40, 60)),
                                );
                            }
                        });
                    }
                    if let Some(ctx_blurb) = song.context.as_deref() {
                        ui.add_space(6.0);
                        ui.label(egui::RichText::new("context").small().weak());
                        ui.label(
                            egui::RichText::new(ctx_blurb)
                                .small()
                                .color(egui::Color32::from_rgb(220, 220, 220)),
                        );
                    }
                } else {
                    ui.label(
                        egui::RichText::new(
                            "no library_db entry — non-MIDI / non-catalog source",
                        )
                        .small()
                        .italics()
                        .color(egui::Color32::from_rgb(150, 150, 150)),
                    );
                }

                ui.add_space(8.0);
                ui.separator();

                // Listening stats block.
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("\u{1F39A} plays").small().weak(),
                    );
                    ui.label(
                        egui::RichText::new(format!("{plays}"))
                            .strong()
                            .color(egui::Color32::from_rgb(160, 200, 255))
                            .monospace(),
                    );
                    if let Some(ts) = last_played {
                        ui.label(
                            egui::RichText::new(format!("last {}", humanize_ago(now - ts)))
                                .small()
                                .color(egui::Color32::from_rgb(180, 180, 200)),
                        );
                    } else {
                        ui.label(
                            egui::RichText::new("never played")
                                .small()
                                .italics()
                                .color(egui::Color32::from_rgb(140, 140, 140)),
                        );
                    }
                });

                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if ui
                        .button(egui::RichText::new("\u{25B6} preview").strong())
                        .on_hover_text("Preview this track in the active mixer slot.")
                        .clicked()
                    {
                        play_now = true;
                    }
                    let fav_glyph = if is_favorite { "\u{2605} unfav" } else { "\u{2606} fav" };
                    if ui
                        .button(fav_glyph)
                        .on_hover_text("Toggle ★ favorite (also shown in side panel).")
                        .clicked()
                    {
                        toggle_fav = true;
                    }
                });

                ui.add_space(8.0);
                ui.separator();

                // Marker palette: known toggles + custom-tag entry.
                ui.label(
                    egui::RichText::new("marker tags")
                        .strong()
                        .color(egui::Color32::from_rgb(220, 220, 220)),
                );
                ui.horizontal_wrapped(|ui| {
                    for (key, glyph, hover) in Self::KNOWN_TAGS {
                        let on = assigned_tags.iter().any(|t| t == key);
                        let chip = if on {
                            egui::RichText::new(format!("{glyph} {key}"))
                                .strong()
                                .color(egui::Color32::from_rgb(20, 20, 20))
                                .background_color(egui::Color32::from_rgb(255, 210, 80))
                        } else {
                            egui::RichText::new(format!("{glyph} {key}"))
                                .color(egui::Color32::from_rgb(220, 220, 220))
                                .background_color(egui::Color32::from_rgb(50, 50, 60))
                        };
                        if ui
                            .button(chip)
                            .on_hover_text(*hover)
                            .clicked()
                        {
                            tag_ops.push((*key).to_string());
                        }
                    }
                });
                if !custom_assigned.is_empty() {
                    ui.label(
                        egui::RichText::new("custom")
                            .small()
                            .weak(),
                    );
                    ui.horizontal_wrapped(|ui| {
                        for tag in &custom_assigned {
                            if ui
                                .button(
                                    egui::RichText::new(format!("\u{2716} #{tag}"))
                                        .small()
                                        .color(egui::Color32::from_rgb(255, 200, 200))
                                        .background_color(egui::Color32::from_rgb(60, 30, 30)),
                                )
                                .on_hover_text("Click to remove this custom tag.")
                                .clicked()
                            {
                                tag_ops.push(tag.clone());
                            }
                        }
                    });
                }
                ui.horizontal(|ui| {
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut self.custom_tag_input)
                            .desired_width(160.0)
                            .hint_text("custom tag…"),
                    );
                    let pressed_enter = resp.lost_focus()
                        && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    let clicked_add = ui.button("+ tag").clicked();
                    if pressed_enter || clicked_add {
                        let new_tag = self.custom_tag_input.trim().to_string();
                        if !new_tag.is_empty() {
                            tag_ops.push(new_tag);
                            self.custom_tag_input.clear();
                        }
                    }
                });

                ui.add_space(8.0);
                ui.separator();

                // Notes editor. Bound to a draft buffer so typing
                // doesn't fight egui's frame cycle.
                ui.label(
                    egui::RichText::new("personal note")
                        .strong()
                        .color(egui::Color32::from_rgb(220, 220, 220)),
                );
                let resp = ui.add(
                    egui::TextEdit::multiline(&mut self.note_draft)
                        .desired_rows(4)
                        .desired_width(f32::INFINITY)
                        .hint_text(
                            "private note — what to study, why this caught your ear, …",
                        ),
                );
                if resp.lost_focus() {
                    close_save = true;
                }
                ui.horizontal(|ui| {
                    if ui
                        .button("\u{1F4BE} save")
                        .on_hover_text("Persist the note to bench-out/play_log.db.")
                        .clicked()
                    {
                        close_save = true;
                    }
                    if ui
                        .button("clear")
                        .on_hover_text("Erase the note (deletes the row).")
                        .clicked()
                    {
                        self.note_draft.clear();
                        close_save = true;
                    }
                });
            });

        // Side-effect drain. egui closures hold &mut self.note_draft
        // already, so every state write happens after `show()` returns.
        if let Some(url) = open_url {
            ctx.open_url(egui::OpenUrl::new_tab(url));
        }
        for tag in tag_ops {
            self.toggle_user_tag(&label, &tag);
        }
        if toggle_fav {
            self.set_favorites_batch(&[(label.clone(), !is_favorite)]);
        }
        if play_now {
            self.play_song_id(&label);
        }
        if close_save {
            self.flush_note_draft();
        }
        if clear_selection {
            self.flush_note_draft();
            self.selected_label = None;
            self.note_draft.clear();
            self.note_draft_song = None;
        }
    }

    // ---------------------------------------------------------------
    // CP integration
    // ---------------------------------------------------------------

    /// Apply one queued CP command. Called from `update()` head, on
    /// the egui thread, so it can freely use the same `&mut self`
    /// methods the UI buttons use (preview / stop_all / set_track_voice
    /// / rescan).
    fn apply_cp_command(&mut self, cmd: CpJukeboxCommand) {
        match cmd {
            CpJukeboxCommand::LoadTrack { label } => {
                let idx = self.tracks.iter().position(|t| t.label == label);
                match idx {
                    Some(i) => self.preview(i),
                    None => eprintln!("cp: load_track: no catalog row matches label '{label}'"),
                }
            }
            CpJukeboxCommand::Play => {
                self.play_all();
            }
            CpJukeboxCommand::Stop => {
                self.stop_all();
            }
            CpJukeboxCommand::SetVoice { label, voice } => {
                let path = match self.tracks.iter().find(|t| t.label == label) {
                    Some(t) => t.path.clone(),
                    None => {
                        eprintln!("cp: set_voice: no catalog row matches label '{label}'");
                        return;
                    }
                };
                if self.set_track_voice(&label, &path, &voice) {
                    // Mirror the UI's behaviour: a voice change re-fires
                    // preview so the user (or verifier) hears the new
                    // voice on the row that's currently selected.
                    if let Some(idx) = self.tracks.iter().position(|t| t.label == label) {
                        self.preview(idx);
                    }
                }
            }
            CpJukeboxCommand::Rescan => {
                self.rescan();
            }
        }
    }

    /// Build the `CpJukeboxSnapshot` published over `get_state` for
    /// this frame. Computed after all command application + UI mutation
    /// so the result reflects what the user (or verification script)
    /// will see in the *next* frame's GUI paint.
    fn build_cp_snapshot(&self) -> CpJukeboxSnapshot {
        let (slot_snaps, any_playing) = {
            let st = self.mixer.state.lock().unwrap();
            let snaps: Vec<CpSlotSnap> = st
                .tracks
                .iter()
                .enumerate()
                .map(|(i, t)| CpSlotSnap {
                    slot: i,
                    label: if t.decoded.is_some() {
                        Some(t.file_label.clone())
                    } else {
                        None
                    },
                    is_playing: t.is_playing,
                    cursor_frames: t.cursor_frames,
                    total_frames: t
                        .decoded
                        .as_ref()
                        .map(|d| d.total_samples / d.channels.max(1) as u64)
                        .unwrap_or(0),
                    sample_rate: t.decoded.as_ref().map(|d| d.sample_rate).unwrap_or(0),
                })
                .collect();
            let any = snaps.iter().any(|s| s.is_playing);
            (snaps, any)
        };
        let cache_hit_pairs: usize = self
            .cached_midi_labels
            .values()
            .map(|s| s.len())
            .sum();
        let cached_labels: Vec<String> = self
            .cached_midi_labels
            .iter()
            .filter(|(_, v)| !v.is_empty())
            .map(|(k, _)| k.clone())
            .collect();
        let selected_label = self
            .selected
            .and_then(|i| self.tracks.get(i))
            .map(|t| t.label.clone());
        let all_labels: Vec<String> = self.tracks.iter().map(|t| t.label.clone()).collect();
        CpJukeboxSnapshot {
            frame_id: self.cp_frame_id,
            track_count: self.tracks.len(),
            db_voice_count: self.db_voice_count,
            db_song_count: self.song_index.len(),
            selected_label,
            any_playing,
            slots: slot_snaps,
            cache_hit_pairs,
            cached_labels,
            selected_voices: self.selected_voices.clone(),
            prewarm_active: self.prewarm.is_some(),
            all_labels,
        }
    }
}

/// Two-cell row helper for the detail-pane metadata grid: weak left
/// label, strong right value, end_row(). Keeps the gutter consistent
/// across optional fields without copy-pasting the grid plumbing.
fn meta_row(ui: &mut egui::Ui, key: &str, value: &str) {
    ui.label(egui::RichText::new(key).small().weak());
    ui.label(egui::RichText::new(value).small());
    ui.end_row();
}

/// Spawn the embedded jukebox CP server. Registers `get_state`,
/// `load_track`, `play`, `stop`, `set_voice`, and `rescan` against a
/// shared `gui_cp::State`. The egui thread drains the command queue
/// each frame; CP handlers do nothing more than queue or read.
fn spawn_cp_server(
    state: gui_cp::State<CpJukeboxSnapshot, CpJukeboxCommand>,
) -> std::io::Result<gui_cp::Handle> {
    let endpoint = gui_cp::resolve_endpoint("jukebox", None);

    let st_get = state.clone();
    let st_load = state.clone();
    let st_play = state.clone();
    let st_stop = state.clone();
    let st_voice = state.clone();
    let st_rescan = state.clone();
    let st_list = state.clone();

    gui_cp::Builder::new("jukebox", &endpoint)
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
            let label = a.label.clone();
            st_load.push_command(CpJukeboxCommand::LoadTrack { label });
            Ok(json!({"queued": "load_track", "label": a.label}))
        })
        .register("play", move |_p| {
            st_play.push_command(CpJukeboxCommand::Play);
            Ok(json!({"queued": "play"}))
        })
        .register("stop", move |_p| {
            st_stop.push_command(CpJukeboxCommand::Stop);
            Ok(json!({"queued": "stop"}))
        })
        .register("set_voice", move |p| {
            #[derive(serde::Deserialize)]
            struct Args {
                label: String,
                voice: String,
            }
            let a: Args = gui_cp::decode_params(p)?;
            let pair = json!({"label": a.label, "voice": a.voice});
            st_voice.push_command(CpJukeboxCommand::SetVoice {
                label: a.label,
                voice: a.voice,
            });
            Ok(json!({"queued": "set_voice", "args": pair}))
        })
        .register("rescan", move |_p| {
            st_rescan.push_command(CpJukeboxCommand::Rescan);
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
            // Read the most-recent snapshot so list_tracks reflects
            // the same set the GUI is rendering. Fall back to "warming
            // up" if no frame has published yet.
            let snap = match st_list.read_snapshot() {
                Some(s) => s,
                None => return Ok(json!({"warming_up": true, "tracks": []})),
            };
            let needle = a.contains.unwrap_or_default().to_ascii_lowercase();
            let limit = a.limit.unwrap_or(usize::MAX);
            let tracks: Vec<&str> = snap
                .all_labels
                .iter()
                .filter(|label| {
                    needle.is_empty() || label.to_ascii_lowercase().contains(&needle)
                })
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

/// Render a single VU meter bar. peak / rms are 0..1+ (we clamp).
fn vu_meter_bar(ui: &mut egui::Ui, label: &str, peak: f32, rms: f32) {
    let desired_w = 120.0f32;
    let h = 10.0f32;
    let (rect, _resp) = ui.allocate_exact_size(egui::vec2(desired_w, h), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    // Background
    painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(40, 40, 40));
    let p = peak.clamp(0.0, 1.2) / 1.2;
    let r = rms.clamp(0.0, 1.2) / 1.2;
    // RMS fill (green-ish, behind)
    let mut rms_rect = rect;
    rms_rect.set_width(rect.width() * r);
    painter.rect_filled(rms_rect, 2.0, egui::Color32::from_rgb(60, 160, 80));
    // Peak tick (yellow line at peak position)
    let peak_x = rect.left() + rect.width() * p;
    painter.line_segment(
        [
            egui::pos2(peak_x, rect.top()),
            egui::pos2(peak_x, rect.bottom()),
        ],
        egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 220, 80)),
    );
    // Clip indicator (>1.0 -> red overlay on right edge)
    if peak > 1.0 {
        let mut clip_rect = rect;
        clip_rect.set_left(rect.left() + rect.width() * (1.0 / 1.2));
        painter.rect_filled(
            clip_rect,
            2.0,
            egui::Color32::from_rgba_premultiplied(180, 40, 40, 120),
        );
    }
    ui.label(egui::RichText::new(label).monospace().small());
}

impl eframe::App for Jukebox {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(std::time::Duration::from_millis(50));

        // Drain the CP command queue first so external verification
        // commands (load_track / play / stop / set_voice) take effect
        // *this* frame — by the time we publish the snapshot at the
        // tail, the resulting state mutation is already visible.
        for cmd in self.cp_state.drain_commands() {
            self.apply_cp_command(cmd);
        }

        // Lazy-MIDI render: drain the background-render channel before
        // anything else so a finished render becomes the active track
        // for THIS frame (no extra one-frame latency between worker
        // completion and audible playback).
        self.poll_pending_render();
        // Background prewarm completions: each Ok flips the row's "✓"
        // glyph on. Cheap (try_recv loop) so it's safe to call every
        // frame.
        self.poll_prewarm();

        // Apply pending file-load changes from combo boxes.
        self.sync_slot_files();

        // Snapshot mixer state for the UI (volumes / pan / VU). We
        // copy out so we don't hold the audio Mutex during egui draw.
        let snapshot: Vec<MixerSnap> = {
            let st = self.mixer.state.lock().unwrap();
            st.tracks
                .iter()
                .map(|t| MixerSnap {
                    has_file: t.decoded.is_some(),
                    file_label: t.file_label.clone(),
                    volume: t.volume,
                    pan: t.pan,
                    muted: t.muted,
                    solo: t.solo,
                    loop_enabled: t.loop_enabled,
                    is_playing: t.is_playing,
                    cursor_frames: t.cursor_frames,
                    sr: t.decoded.as_ref().map(|d| d.sample_rate).unwrap_or(0),
                    total_frames: t
                        .decoded
                        .as_ref()
                        .map(|d| d.total_samples / d.channels.max(1) as u64)
                        .unwrap_or(0),
                    vu_peak_l: t.vu_peak_l,
                    vu_peak_r: t.vu_peak_r,
                    vu_rms_l: t.vu_rms_l,
                    vu_rms_r: t.vu_rms_r,
                })
                .collect()
        };
        let any_playing = snapshot.iter().any(|s| s.is_playing);

        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("keysynth jukebox");
                ui.separator();
                if ui
                    .button(if any_playing { "■ stop" } else { "—" })
                    .clicked()
                {
                    self.stop_all();
                }
                ui.separator();
                if ui.button("rescan files").clicked() {
                    self.rescan();
                }
                ui.separator();
                let db_label = format!(
                    "{} tracks · DB: {} songs / {} voices",
                    self.tracks.len(),
                    self.song_index.len(),
                    self.db_voice_count,
                );
                ui.label(db_label).on_hover_text(
                    "DB counts come from bench-out/library.db (issue #66 — \
                     materialized index over voices_live/* + bench-out/songs/manifest.json). \
                     Hover any song row for full composer / era / license metadata.",
                );
                ui.separator();
                ui.label("filter:");
                ui.text_edit_singleline(&mut self.filter);
                ui.separator();
                let total_midi = self
                    .tracks
                    .iter()
                    .filter(|t| t.format == Format::Mid)
                    .count();
                // "cached" counts tracks whose *currently selected*
                // voice has a WAV on disk — switching a row from
                // piano-modal to guitar-stk drops it from the green
                // count until the new voice is rendered.
                let cached = self
                    .tracks
                    .iter()
                    .filter(|t| t.format == Format::Mid)
                    .filter(|t| {
                        let voice = self.voice_for(&t.label, &t.path);
                        self.cached_midi_labels
                            .get(&t.label)
                            .map(|s| s.contains(&voice))
                            .unwrap_or(false)
                    })
                    .count();
                let cache_color = if total_midi == 0 || cached >= total_midi {
                    egui::Color32::from_rgb(120, 220, 140)
                } else {
                    egui::Color32::from_rgb(220, 200, 120)
                };
                let cache_text = if self.prewarm.is_some() {
                    format!("cache {cached}/{total_midi} \u{231B}")
                } else {
                    format!("cache {cached}/{total_midi} \u{2713}")
                };
                ui.label(
                    egui::RichText::new(cache_text)
                        .color(cache_color)
                        .monospace()
                        .small(),
                )
                .on_hover_text(
                    "MIDI preview cache coverage. Green ✓ = every MIDI in the catalogue \
                     has a rendered WAV in bench-out/cache/. Yellow ⌛ = background \
                     prewarmer still working. Run `ksprerender` to fill the cache for \
                     every (song, voice) pair offline.",
                );
                ui.separator();
                let fav_label = format!("\u{2605} only ({})", self.favorites.len());
                ui.checkbox(&mut self.favorites_only, fav_label).on_hover_text(
                    "Restrict the catalogue to starred songs. Use the \u{2605}/\u{2606} \
                     button on each row to toggle. Persists across sessions in \
                     bench-out/play_log.db (gitignored, machine-local).",
                );
            });
            // ---- DB-driven catalogue filters ----
            // Each combo is built once per frame from the song_index so a
            // freshly imported manifest immediately surfaces new eras /
            // licenses / instruments without needing a separate refresh
            // path. Empty index → no DB metadata available, skip the row.
            if !self.song_index.is_empty() {
                let mut eras: Vec<String> = self
                    .song_index
                    .values()
                    .filter_map(|s| s.era.clone())
                    .collect();
                eras.sort();
                eras.dedup();
                let mut licenses: Vec<String> = self
                    .song_index
                    .values()
                    .map(|s| s.license.clone())
                    .collect();
                licenses.sort();
                licenses.dedup();
                let mut instruments: Vec<String> = self
                    .song_index
                    .values()
                    .map(|s| s.instrument.clone())
                    .collect();
                instruments.sort();
                instruments.dedup();

                ui.horizontal(|ui| {
                    ui.label("browse:");
                    let era_cur = self
                        .era_filter
                        .clone()
                        .unwrap_or_else(|| "(any era)".to_string());
                    egui::ComboBox::from_id_salt("era-filter")
                        .selected_text(era_cur.clone())
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_label(self.era_filter.is_none(), "(any era)")
                                .clicked()
                            {
                                self.era_filter = None;
                            }
                            for e in &eras {
                                if ui
                                    .selectable_label(
                                        self.era_filter.as_deref() == Some(e.as_str()),
                                        e,
                                    )
                                    .clicked()
                                {
                                    self.era_filter = Some(e.clone());
                                }
                            }
                        })
                        .response
                        .on_hover_text(
                            "Era buckets are derived from each composer's death year \
                             (Baroque ≤1750, Classical ≤1820, Romantic ≤1910, Modern \
                             >1910, Traditional for unattributed folk pieces).",
                        );
                    let lic_cur = self
                        .license_filter
                        .clone()
                        .unwrap_or_else(|| "(any license)".to_string());
                    egui::ComboBox::from_id_salt("license-filter")
                        .selected_text(lic_cur)
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_label(self.license_filter.is_none(), "(any license)")
                                .clicked()
                            {
                                self.license_filter = None;
                            }
                            for l in &licenses {
                                if ui
                                    .selectable_label(
                                        self.license_filter.as_deref() == Some(l.as_str()),
                                        l,
                                    )
                                    .clicked()
                                {
                                    self.license_filter = Some(l.clone());
                                }
                            }
                        })
                        .response
                        .on_hover_text(
                            "Filter by source license. Use Public Domain / CC0 for \
                             unrestricted downstream reuse; CC-BY-SA pieces propagate \
                             share-alike to anything you publish that contains them.",
                        );
                    let ins_cur = self
                        .instrument_filter
                        .clone()
                        .unwrap_or_else(|| "(any instrument)".to_string());
                    egui::ComboBox::from_id_salt("instrument-filter")
                        .selected_text(ins_cur)
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_label(self.instrument_filter.is_none(), "(any instrument)")
                                .clicked()
                            {
                                self.instrument_filter = None;
                            }
                            for i in &instruments {
                                if ui
                                    .selectable_label(
                                        self.instrument_filter.as_deref() == Some(i.as_str()),
                                        i,
                                    )
                                    .clicked()
                                {
                                    self.instrument_filter = Some(i.clone());
                                }
                            }
                        });
                    let any_filter_set = self.era_filter.is_some()
                        || self.license_filter.is_some()
                        || self.instrument_filter.is_some();
                    if any_filter_set
                        && ui
                            .button("clear")
                            .on_hover_text("Reset era / license / instrument filters.")
                            .clicked()
                    {
                        self.era_filter = None;
                        self.license_filter = None;
                        self.instrument_filter = None;
                    }
                });
            }
            ui.horizontal(|ui| {
                ui.label("output:");
                let current = self.current_device.clone();
                let mut new_pick: Option<String> = None;
                egui::ComboBox::from_id_salt("audio-device")
                    .selected_text(&current)
                    .width(360.0)
                    .show_ui(ui, |ui| {
                        for dev in &self.devices {
                            if ui.selectable_label(*dev == current, dev).clicked() {
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
                ui.separator();
                ui.monospace(format!(
                    "sr={} ch={}",
                    self.mixer.sample_rate, self.mixer.output_channels
                ));
            });
        });

        // ----- Bottom panel: multi-track mixer ----------------------
        egui::TopBottomPanel::bottom("mixer")
            .resizable(true)
            .min_height(220.0)
            .default_height(280.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("mixer");
                    ui.separator();
                    if ui.button("▶ play all").clicked() {
                        self.play_all();
                    }
                    if ui.button("■ stop all").clicked() {
                        self.stop_all();
                    }
                    ui.separator();
                    if ui.button("+ track").clicked() {
                        self.add_slot();
                    }
                    ui.separator();
                    let n_active = snapshot.iter().filter(|s| s.has_file).count();
                    ui.label(format!(
                        "{n_active}/{} loaded · {}",
                        snapshot.len(),
                        if any_playing { "playing" } else { "stopped" }
                    ));
                });
                ui.separator();

                // Track-list snapshot for combo box (label + source).
                let track_choices: Vec<(usize, String)> = self
                    .tracks
                    .iter()
                    .enumerate()
                    .map(|(i, t)| (i, format!("[{}] {}_{}", t.source, t.piece, t.engine)))
                    .collect();
                egui::ScrollArea::vertical()
                    .id_salt("mixer-scroll")
                    .show(ui, |ui| {
                        // Pending mutations applied after the loop so we
                        // don't simultaneously hold a borrow into self
                        // and try to lock state inside the UI closure.
                        let mut to_remove: Option<usize> = None;
                        let mut volume_changes: Vec<(usize, f32)> = Vec::new();
                        let mut pan_changes: Vec<(usize, f32)> = Vec::new();
                        let mut mute_changes: Vec<(usize, bool)> = Vec::new();
                        let mut solo_changes: Vec<(usize, bool)> = Vec::new();
                        let mut loop_changes: Vec<(usize, bool)> = Vec::new();

                        for (slot_idx, snap) in snapshot.iter().enumerate() {
                            ui.horizontal(|ui| {
                                ui.monospace(format!("{:>2}", slot_idx + 1));
                                let current = self
                                    .slot_file_idx
                                    .get(slot_idx)
                                    .copied()
                                    .flatten()
                                    .and_then(|i| {
                                        track_choices
                                            .iter()
                                            .find(|(j, _)| *j == i)
                                            .map(|(_, l)| l.as_str())
                                    })
                                    .unwrap_or("(empty)");
                                let cur_idx_opt =
                                    self.slot_file_idx.get(slot_idx).copied().flatten();
                                egui::ComboBox::from_id_salt(format!("mix-{slot_idx}"))
                                    .selected_text(current)
                                    .width(360.0)
                                    .show_ui(ui, |ui| {
                                        if ui
                                            .selectable_label(cur_idx_opt.is_none(), "(empty)")
                                            .clicked()
                                        {
                                            if slot_idx < self.slot_file_idx.len() {
                                                self.slot_file_idx[slot_idx] = None;
                                            }
                                        }
                                        for (i, label) in &track_choices {
                                            if ui
                                                .selectable_label(cur_idx_opt == Some(*i), label)
                                                .clicked()
                                            {
                                                if slot_idx < self.slot_file_idx.len() {
                                                    self.slot_file_idx[slot_idx] = Some(*i);
                                                }
                                            }
                                        }
                                    });
                                let mut vol = snap.volume;
                                if ui
                                    .add(
                                        egui::Slider::new(&mut vol, 0.0..=2.0)
                                            .text("vol")
                                            .step_by(0.05),
                                    )
                                    .changed()
                                {
                                    volume_changes.push((slot_idx, vol));
                                }
                                let mut pan = snap.pan;
                                if ui
                                    .add(
                                        egui::Slider::new(&mut pan, -1.0..=1.0)
                                            .text("pan")
                                            .step_by(0.05),
                                    )
                                    .changed()
                                {
                                    pan_changes.push((slot_idx, pan));
                                }
                                let mut muted = snap.muted;
                                if ui.checkbox(&mut muted, "mute").changed() {
                                    mute_changes.push((slot_idx, muted));
                                }
                                let mut solo = snap.solo;
                                if ui.checkbox(&mut solo, "solo").changed() {
                                    solo_changes.push((slot_idx, solo));
                                }
                                let mut loop_on = snap.loop_enabled;
                                if ui.checkbox(&mut loop_on, "loop").changed() {
                                    loop_changes.push((slot_idx, loop_on));
                                }
                                if snap.is_playing {
                                    ui.colored_label(egui::Color32::from_rgb(255, 200, 80), "▶");
                                }
                                if ui.button("×").clicked() {
                                    to_remove = Some(slot_idx);
                                }
                            });
                            // Per-slot VU row.
                            ui.horizontal(|ui| {
                                ui.add_space(28.0);
                                vu_meter_bar(ui, "L", snap.vu_peak_l, snap.vu_rms_l);
                                vu_meter_bar(ui, "R", snap.vu_peak_r, snap.vu_rms_r);
                                if snap.has_file && snap.sr > 0 {
                                    let pos_s = snap.cursor_frames as f64 / snap.sr as f64;
                                    let total_s = snap.total_frames as f64 / snap.sr as f64;
                                    ui.monospace(format!("{:>5.1}/{:>5.1}s", pos_s, total_s));
                                }
                            });
                        }

                        // Commit mutations.
                        if !volume_changes.is_empty()
                            || !pan_changes.is_empty()
                            || !mute_changes.is_empty()
                            || !solo_changes.is_empty()
                            || !loop_changes.is_empty()
                        {
                            let mut st = self.mixer.state.lock().unwrap();
                            for (i, v) in volume_changes {
                                if let Some(t) = st.tracks.get_mut(i) {
                                    t.volume = v;
                                }
                            }
                            for (i, p) in pan_changes {
                                if let Some(t) = st.tracks.get_mut(i) {
                                    t.pan = p;
                                }
                            }
                            for (i, m) in mute_changes {
                                if let Some(t) = st.tracks.get_mut(i) {
                                    t.muted = m;
                                }
                            }
                            for (i, s) in solo_changes {
                                if let Some(t) = st.tracks.get_mut(i) {
                                    t.solo = s;
                                }
                            }
                            for (i, l) in loop_changes {
                                if let Some(t) = st.tracks.get_mut(i) {
                                    t.loop_enabled = l;
                                }
                            }
                        }
                        if let Some(idx) = to_remove {
                            self.remove_slot(idx);
                        }
                    });
            });

        // ----- Right panel: play history + favorites ----------------
        // Three stacked sections: starred songs, recently played
        // window, and lifetime leaderboard. Every entry is a button
        // that re-launches the matching track via `play_song_id`.
        // The panel is collapsible — drag the splitter to hide it
        // when the user wants the catalogue full-width.
        egui::SidePanel::right("history-panel")
            .resizable(true)
            .default_width(240.0)
            .min_width(180.0)
            .show(ctx, |ui| {
                let mut to_play_song: Option<String> = None;
                let mut to_clear_fav: Option<String> = None;

                ui.horizontal(|ui| {
                    ui.heading("\u{2605} favorites");
                    ui.label(format!("({})", self.favorites.len()));
                });
                ui.separator();
                if self.play_log.is_none() {
                    ui.colored_label(
                        egui::Color32::from_rgb(200, 130, 130),
                        "play_log unavailable",
                    )
                    .on_hover_text(
                        "bench-out/play_log.db could not be opened — favorites and \
                         history are disabled this session. See stderr for details.",
                    );
                } else if self.favorites.is_empty() {
                    ui.label(
                        egui::RichText::new("no favorites yet — click \u{2606} on any row")
                            .small()
                            .italics(),
                    );
                } else {
                    egui::ScrollArea::vertical()
                        .id_salt("favorites-scroll")
                        .max_height(180.0)
                        .show(ui, |ui| {
                            // DB returns most-recently-added first.
                            // Re-derive from the snapshot we already
                            // have so the panel doesn't hit SQLite per
                            // frame.
                            let mut favs: Vec<&String> = self.favorites.iter().collect();
                            favs.sort();
                            for song_id in favs {
                                ui.horizontal(|ui| {
                                    let display = self
                                        .song_index
                                        .get(song_id)
                                        .map(|s| s.title.clone())
                                        .unwrap_or_else(|| song_id.clone());
                                    let exists = self
                                        .tracks
                                        .iter()
                                        .any(|t| &t.label == song_id);
                                    let label = if exists {
                                        format!("\u{25B6} {display}")
                                    } else {
                                        format!("(missing) {display}")
                                    };
                                    if ui
                                        .add_enabled(
                                            exists,
                                            egui::Button::new(
                                                egui::RichText::new(label).small(),
                                            )
                                            .frame(false),
                                        )
                                        .on_hover_text(song_id.as_str())
                                        .clicked()
                                    {
                                        to_play_song = Some(song_id.clone());
                                    }
                                    if ui
                                        .small_button("\u{2605}")
                                        .on_hover_text("Remove from favorites")
                                        .clicked()
                                    {
                                        to_clear_fav = Some(song_id.clone());
                                    }
                                });
                            }
                        });
                }

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.heading("\u{1F551} recent");
                    ui.label(format!("({})", self.recent_plays.len()));
                });
                ui.separator();
                if self.recent_plays.is_empty() {
                    ui.label(
                        egui::RichText::new("nothing played yet")
                            .small()
                            .italics(),
                    );
                } else {
                    egui::ScrollArea::vertical()
                        .id_salt("recent-scroll")
                        .max_height(220.0)
                        .show(ui, |ui| {
                            let now = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_secs() as i64)
                                .unwrap_or(0);
                            for entry in &self.recent_plays {
                                ui.horizontal(|ui| {
                                    let display = self
                                        .song_index
                                        .get(&entry.song_id)
                                        .map(|s| s.title.clone())
                                        .unwrap_or_else(|| entry.song_id.clone());
                                    let voice = entry
                                        .voice_id
                                        .as_deref()
                                        .unwrap_or("?");
                                    let ago = humanize_ago(now - entry.played_at);
                                    let exists = self
                                        .tracks
                                        .iter()
                                        .any(|t| t.label == entry.song_id);
                                    let label = format!(
                                        "\u{25B6} {display}  ({voice}, {ago})"
                                    );
                                    if ui
                                        .add_enabled(
                                            exists,
                                            egui::Button::new(
                                                egui::RichText::new(label).small(),
                                            )
                                            .frame(false),
                                        )
                                        .on_hover_text(entry.song_id.as_str())
                                        .clicked()
                                    {
                                        to_play_song = Some(entry.song_id.clone());
                                    }
                                });
                            }
                        });
                }

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.heading("\u{1F525} top");
                });
                ui.separator();
                let top: Vec<(String, i64)> = {
                    let mut v: Vec<(String, i64)> = self
                        .play_counts
                        .iter()
                        .map(|(k, v)| (k.clone(), *v))
                        .collect();
                    v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                    v.truncate(TOP_PLAYED_LIMIT);
                    v
                };
                if top.is_empty() {
                    ui.label(egui::RichText::new("no plays yet").small().italics());
                } else {
                    for (song_id, n) in &top {
                        ui.horizontal(|ui| {
                            let display = self
                                .song_index
                                .get(song_id)
                                .map(|s| s.title.clone())
                                .unwrap_or_else(|| song_id.clone());
                            let exists = self.tracks.iter().any(|t| &t.label == song_id);
                            let label = format!("\u{25B6} {display}");
                            ui.add_sized(
                                [40.0, 18.0],
                                egui::Label::new(
                                    egui::RichText::new(format!("\u{00D7}{n}"))
                                        .color(egui::Color32::from_rgb(255, 180, 120))
                                        .small()
                                        .monospace(),
                                ),
                            );
                            if ui
                                .add_enabled(
                                    exists,
                                    egui::Button::new(egui::RichText::new(label).small())
                                        .frame(false),
                                )
                                .on_hover_text(song_id.as_str())
                                .clicked()
                            {
                                to_play_song = Some(song_id.clone());
                            }
                        });
                    }
                }

                if let Some(song_id) = to_play_song {
                    self.play_song_id(&song_id);
                }
                if let Some(song_id) = to_clear_fav {
                    self.set_favorites_batch(&[(song_id, false)]);
                }
            });

        // ----- Left panel: research / marking detail ---------------
        // Open whenever a row is selected. Carries everything the
        // catalogue row can't fit inline: full composer name, license
        // text, manifest context blurb, clickable source URL, the
        // user's marker chips for this piece, and a free-form note
        // editor backed by `play_log.notes`.
        self.render_detail_panel(ctx);

        egui::CentralPanel::default().show(ctx, |ui| {
            let needle = self.filter.to_lowercase();
            // Two-level grouping. The first key folds the iter-variant
            // sprawl (twinkle / twinkle_h2 / twinkle_p / twinkle_revert)
            // under one collapsible header; the second is the exact
            // piece name that lines up engine renders into one row.
            //
            // fold_key heuristic: known piece roots are detected first
            // (multi-word names like "maple_leaf_rag" / "bach_invention"),
            // otherwise fall back to the first underscore-delimited
            // token. The default-collapsed state hides 80 % of the
            // iter-spam from the catalogue while keeping every render
            // reachable.
            fn fold_key(piece: &str) -> String {
                const KNOWN_ROOTS: &[&str] = &[
                    "maple_leaf_rag",
                    "bach_invention",
                    "bach_prelude_c",
                    "bach_prelude12",
                    "twelfth_street_rag",
                    "st_louis_blues",
                    "king_porter_stomp",
                    "fur_elise",
                    "canon_d",
                    "blues_in_c",
                    "c_progression",
                    "minor_cadence",
                    "demo_four_on_the_floor",
                    "demo_breakbeat",
                    "midi_bach",
                    "midi_maple",
                    "midi_maple12",
                    "tap",
                    "bach_baseline",
                    "bach_prelude",
                ];
                let p = piece.to_lowercase();
                for root in KNOWN_ROOTS {
                    if p == *root || p.starts_with(&format!("{root}_")) {
                        return (*root).to_string();
                    }
                }
                // midi_<composer>_<rest> → root "midi_<composer>".
                // Otherwise the whole midi import dumps into one
                // mega-group and you can't browse Chopin vs Beethoven
                // separately. The composer token is whatever follows
                // "midi_" up to the next underscore.
                if let Some(rest) = p.strip_prefix("midi_") {
                    if let Some(composer) = rest.split('_').next() {
                        if !composer.is_empty() {
                            return format!("midi_{composer}");
                        }
                    }
                }
                // Fallback: first underscore-delimited token.
                p.split('_').next().unwrap_or(&p).to_string()
            }

            // root → piece → Vec<track index>
            let mut folded: std::collections::BTreeMap<
                String,
                std::collections::BTreeMap<String, Vec<usize>>,
            > = std::collections::BTreeMap::new();
            for (i, t) in self.tracks.iter().enumerate() {
                if !needle.is_empty() && !t.piece.to_lowercase().contains(&needle) {
                    continue;
                }
                if self.favorites_only && !self.favorites.contains(&t.label) {
                    continue;
                }
                // DB-driven filters: era / license / instrument. A track
                // without a song_index entry can't be classified, so any
                // active DB filter excludes it. Once all three are None
                // this branch is a noop and non-DB tracks (chiptune /
                // listener-lab / tofu / archive-ref) flow through as
                // before.
                if self.era_filter.is_some()
                    || self.license_filter.is_some()
                    || self.instrument_filter.is_some()
                {
                    let song = match self.song_index.get(&t.label) {
                        Some(s) => s,
                        None => continue,
                    };
                    if let Some(want) = &self.era_filter {
                        if song.era.as_deref() != Some(want.as_str()) {
                            continue;
                        }
                    }
                    if let Some(want) = &self.license_filter {
                        if &song.license != want {
                            continue;
                        }
                    }
                    if let Some(want) = &self.instrument_filter {
                        if &song.instrument != want {
                            continue;
                        }
                    }
                }
                let root = fold_key(&t.piece);
                folded
                    .entry(root)
                    .or_default()
                    .entry(t.piece.clone())
                    .or_default()
                    .push(i);
            }

            egui::ScrollArea::vertical().show(ui, |ui| {
                let mut to_play: Option<usize> = None;
                // Queued favorite-state updates from the per-piece ★
                // buttons. Drained outside the closure so the egui
                // borrow on `self` stays read-only inside it.
                let mut to_set_fav: Vec<(String, bool)> = Vec::new();
                // Queued voice-picker selections: (track_idx, new
                // voice_id). After the loop we persist each via
                // `set_track_voice` and then auto-fire `preview(idx)`
                // for the changed row so the user immediately hears
                // the new voice (cache hit = instant, miss = render
                // spinner).
                let mut to_set_voice: Vec<(usize, String)> = Vec::new();
                // Stem of the row the user clicked to focus the detail
                // panel. Drained after the closure body returns so we
                // can borrow `&mut self` for `set_selected_label` (the
                // closure only holds a shared ref).
                let mut to_select: Option<String> = None;
                let avail_w = ui.available_width();
                for (root, pieces) in &folded {
                    let total_renders: usize = pieces.values().map(|v| v.len()).sum();
                    let n_pieces = pieces.len();
                    // If the root has only one piece (and ≤ 4 renders),
                    // skip the collapsing header — render the row
                    // inline for cheap browse.
                    let inline = n_pieces <= 1 && total_renders <= 4;
                    let any_active_in_root = pieces
                        .values()
                        .flatten()
                        .any(|i| self.selected == Some(*i) && any_playing);

                    let render_piece_row =
                        |ui: &mut egui::Ui,
                         piece: &str,
                         indices: &[usize],
                         to_play: &mut Option<usize>,
                         to_set_fav: &mut Vec<(String, bool)>,
                         to_set_voice: &mut Vec<(usize, String)>,
                         to_select: &mut Option<String>| {
                            ui.horizontal_wrapped(|ui| {
                                let active_in_group = indices
                                    .iter()
                                    .any(|i| self.selected == Some(*i) && any_playing);
                                // Favorite glyph: solid \u{2605} if any
                                // render in this row is starred, hollow
                                // \u{2606} otherwise. Click flips every
                                // render's state to the opposite of
                                // current "any" so a single button is
                                // enough to favorite / unfavorite a
                                // whole piece (the underlying primitive
                                // stays per-render).
                                let any_fav = indices.iter().any(|i| {
                                    self.tracks
                                        .get(*i)
                                        .map(|t| self.favorites.contains(&t.label))
                                        .unwrap_or(false)
                                });
                                let star_glyph = if any_fav { "\u{2605}" } else { "\u{2606}" };
                                let star_color = if any_fav {
                                    egui::Color32::from_rgb(255, 210, 80)
                                } else {
                                    egui::Color32::from_rgb(140, 140, 140)
                                };
                                let star_btn = egui::Button::new(
                                    egui::RichText::new(star_glyph)
                                        .color(star_color)
                                        .strong(),
                                )
                                .frame(false);
                                if ui
                                    .add_sized([22.0, 22.0], star_btn)
                                    .on_hover_text(if any_fav {
                                        "Unfavorite this piece (all engine renders)"
                                    } else {
                                        "Favorite this piece — appears in the \u{2605} side panel \
                                         and the \u{2605}-only filter."
                                    })
                                    .clicked()
                                {
                                    let target = !any_fav;
                                    for i in indices {
                                        if let Some(t) = self.tracks.get(*i) {
                                            to_set_fav.push((t.label.clone(), target));
                                        }
                                    }
                                }
                                // Lifetime play count (sum across this
                                // piece's renders). Hidden when zero so
                                // unplayed rows stay visually quiet.
                                let total_plays: i64 = indices
                                    .iter()
                                    .filter_map(|i| self.tracks.get(*i))
                                    .map(|t| {
                                        self.play_counts
                                            .get(&t.label)
                                            .copied()
                                            .unwrap_or(0)
                                    })
                                    .sum();
                                if total_plays > 0 {
                                    ui.add_sized(
                                        [40.0, 22.0],
                                        egui::Label::new(
                                            egui::RichText::new(format!("\u{00D7}{total_plays}"))
                                                .color(egui::Color32::from_rgb(160, 200, 255))
                                                .small()
                                                .monospace(),
                                        ),
                                    )
                                    .on_hover_text(
                                        "Lifetime play count from bench-out/play_log.db. \
                                         Counts every preview launched from this row across all \
                                         engine variants.",
                                    );
                                } else {
                                    ui.add_space(40.0);
                                }
                                let source_label = indices
                                    .first()
                                    .and_then(|i| self.tracks.get(*i))
                                    .map(|t| t.source.as_str())
                                    .unwrap_or("?");
                                let source_color = match source_label {
                                    "keysynth" => egui::Color32::from_rgb(180, 200, 255),
                                    "chiptune-demo" => egui::Color32::from_rgb(120, 220, 140),
                                    "listener-lab" => egui::Color32::from_rgb(220, 180, 255),
                                    "midi" => egui::Color32::from_rgb(255, 200, 120),
                                    "tofu" => egui::Color32::from_rgb(255, 140, 140),
                                    "archive-ref" => egui::Color32::from_rgb(160, 240, 220),
                                    // Ground-truth NSF render: bright yellow-green = "this is the
                                    // measuring stick, line everything else up against it".
                                    "nsf-truth" => egui::Color32::from_rgb(220, 255, 100),
                                    _ => egui::Color32::from_rgb(180, 180, 180),
                                };
                                ui.add_sized(
                                    [110.0, 22.0],
                                    egui::Label::new(
                                        egui::RichText::new(source_label)
                                            .color(source_color)
                                            .monospace()
                                            .small(),
                                    ),
                                );
                                // Notation indicator: ♪ if at least one
                                // render in the row goes through a known
                                // synth engine (sfz / modal / square /
                                // ...) — that proves the source has a
                                // symbolic score (midi / chiptune JSON /
                                // hand-coded piece function), so it can
                                // be re-rendered through any engine.
                                // ▮ when every render's engine is "—":
                                // a static wav with no score backing
                                // (tofu game asset, archive.org rip).
                                let any_notated = indices.iter().any(|i| {
                                    self.tracks
                                        .get(*i)
                                        .map(|t| t.engine != "—")
                                        .unwrap_or(false)
                                });
                                let kind_glyph = if any_notated { "♪" } else { "▮" };
                                let kind_color = if any_notated {
                                    egui::Color32::from_rgb(255, 215, 100)
                                } else {
                                    egui::Color32::from_rgb(150, 150, 150)
                                };
                                let kind_hover = if any_notated {
                                    "score-backed (midi / chiptune JSON / piece fn) — \
                                     can re-render through any engine"
                                } else {
                                    "wav-only — no symbolic source, fixed audio"
                                };
                                ui.add_sized(
                                    [22.0, 22.0],
                                    egui::Label::new(
                                        egui::RichText::new(kind_glyph).color(kind_color).strong(),
                                    ),
                                )
                                .on_hover_text(kind_hover);

                                // Stable file-stem we route selection
                                // through. play_log + library_db both
                                // key off this, so picking it once here
                                // means the detail pane, favorites, and
                                // play history all line up. Uses the
                                // first index's label since every
                                // render in a piece shares the same
                                // stem.
                                let piece_label_stem: Option<String> = indices
                                    .first()
                                    .and_then(|i| self.tracks.get(*i))
                                    .map(|t| t.label.clone());
                                let is_selected = piece_label_stem
                                    .as_deref()
                                    .map(|s| self.selected_label.as_deref() == Some(s))
                                    .unwrap_or(false);
                                let piece_text = if active_in_group {
                                    egui::RichText::new(piece)
                                        .color(egui::Color32::from_rgb(255, 200, 80))
                                        .strong()
                                        .monospace()
                                } else if is_selected {
                                    egui::RichText::new(piece)
                                        .color(egui::Color32::from_rgb(150, 220, 255))
                                        .strong()
                                        .monospace()
                                } else {
                                    egui::RichText::new(piece).monospace()
                                };
                                let piece_w = (avail_w * 0.22).max(180.0);
                                let piece_resp = ui.add_sized(
                                    [piece_w, 22.0],
                                    egui::SelectableLabel::new(is_selected, piece_text),
                                );
                                if piece_resp
                                    .on_hover_text(
                                        "Click to open the research / marking detail panel \
                                         for this piece.",
                                    )
                                    .clicked()
                                {
                                    if let Some(stem) = &piece_label_stem {
                                        *to_select = Some(stem.clone());
                                    }
                                }
                                // Inline user-tag chips for this piece.
                                // Single-glyph ribbon so the row stays
                                // narrow; full taxonomy lives in the
                                // detail pane. Custom tags collapse
                                // into a generic "#" badge.
                                if let Some(stem) = &piece_label_stem {
                                    if let Some(tags) = self.user_tags.get(stem) {
                                        for tag in tags {
                                            let glyph = match tag.as_str() {
                                                "study" => "\u{1F393}",
                                                "perform" => "\u{1F3A4}",
                                                "revisit" => "\u{2753}",
                                                "compare" => "\u{1F4DD}",
                                                _ => "\u{1F3F7}",
                                            };
                                            ui.add_sized(
                                                [18.0, 22.0],
                                                egui::Label::new(
                                                    egui::RichText::new(glyph)
                                                        .small(),
                                                ),
                                            )
                                            .on_hover_text(format!("user tag: {tag}"));
                                        }
                                    }
                                }

                                // ---- DB metadata badges --------------
                                // Pull metadata off the first track that
                                // resolves in song_index (typically all
                                // renders in a piece share the same DB
                                // row since the file stem is the same).
                                // Tracks without a DB row (chiptune /
                                // listener-lab / tofu) get neutral
                                // placeholders so column widths stay
                                // stable across rows.
                                let song_meta: Option<&Song> = indices
                                    .iter()
                                    .filter_map(|i| self.tracks.get(*i))
                                    .find_map(|t| self.song_index.get(&t.label));
                                let composer_disp = song_meta
                                    .map(|s| compact_composer(&s.composer))
                                    .unwrap_or_else(|| "—".to_string());
                                let composer_color = if song_meta.is_some() {
                                    egui::Color32::from_rgb(220, 220, 220)
                                } else {
                                    egui::Color32::from_rgb(110, 110, 110)
                                };
                                ui.add_sized(
                                    [160.0, 22.0],
                                    egui::Label::new(
                                        egui::RichText::new(composer_disp)
                                            .color(composer_color)
                                            .small(),
                                    ),
                                )
                                .on_hover_text(
                                    song_meta
                                        .map(|s| s.composer.clone())
                                        .unwrap_or_else(|| {
                                            "no library_db entry — non-MIDI source"
                                                .to_string()
                                        }),
                                );
                                let era_disp = song_meta
                                    .and_then(|s| s.era.clone())
                                    .unwrap_or_else(|| "—".to_string());
                                ui.add_sized(
                                    [80.0, 22.0],
                                    egui::Label::new(
                                        egui::RichText::new(era_disp.clone())
                                            .color(era_color(&era_disp))
                                            .strong()
                                            .small(),
                                    ),
                                )
                                .on_hover_text(
                                    "Musicological era bucket (derived from composer \
                                     death year). Use the era filter at the top to \
                                     pull just one period out of the catalogue.",
                                );
                                let lic_disp = song_meta
                                    .map(|s| license_short(&s.license))
                                    .unwrap_or_else(|| "—".to_string());
                                ui.add_sized(
                                    [70.0, 22.0],
                                    egui::Label::new(
                                        egui::RichText::new(lic_disp.clone())
                                            .color(license_color(&lic_disp))
                                            .small()
                                            .monospace(),
                                    ),
                                )
                                .on_hover_text(
                                    song_meta
                                        .map(|s| s.license.clone())
                                        .unwrap_or_else(|| {
                                            "license unknown (no DB entry)".to_string()
                                        }),
                                );
                                let instr_disp = song_meta
                                    .map(|s| s.instrument.clone())
                                    .unwrap_or_else(|| "—".to_string());
                                ui.add_sized(
                                    [60.0, 22.0],
                                    egui::Label::new(
                                        egui::RichText::new(instr_disp)
                                            .color(egui::Color32::from_rgb(180, 180, 200))
                                            .small()
                                            .monospace(),
                                    ),
                                )
                                .on_hover_text(
                                    "Recommended-instrument tag from the manifest \
                                     (drives the suggested-voice pairing).",
                                );

                                ui.separator();
                                for &i in indices {
                                    let t = &self.tracks[i];
                                    let active = self.selected == Some(i) && any_playing;
                                    let is_rendering = self
                                        .pending_render
                                        .as_ref()
                                        .map(|p| p.track_idx == i)
                                        .unwrap_or(false);
                                    // MIDI tracks reach the audio
                                    // pipeline through `preview_cache`,
                                    // so a cache hit means "click =
                                    // instant playback". Surface that
                                    // contract on the row so the user
                                    // can distinguish a sub-millisecond
                                    // click from a multi-second render
                                    // BEFORE clicking. Non-MIDI rows
                                    // are always instant (decoded
                                    // straight from disk) so we skip
                                    // the glyph for them.
                                    let selected_voice = if t.format == Format::Mid {
                                        Some(self.voice_for(&t.label, &t.path))
                                    } else {
                                        None
                                    };
                                    let cache_hit = match &selected_voice {
                                        Some(v) => self
                                            .cached_midi_labels
                                            .get(&t.label)
                                            .map(|s| s.contains(v))
                                            .unwrap_or(false),
                                        None => false,
                                    };
                                    if cache_hit {
                                        ui.add_sized(
                                            [16.0, 22.0],
                                            egui::Label::new(
                                                egui::RichText::new("\u{2713}")
                                                    .color(egui::Color32::from_rgb(120, 220, 140))
                                                    .strong(),
                                            ),
                                        )
                                        .on_hover_text(
                                            "Preview already rendered to bench-out/cache/ \
                                             for the selected voice — click is instant.",
                                        );
                                    } else if t.format == Format::Mid {
                                        ui.add_sized(
                                            [16.0, 22.0],
                                            egui::Label::new(
                                                egui::RichText::new("\u{2218}")
                                                    .color(egui::Color32::from_rgb(140, 140, 140))
                                                    .small(),
                                            ),
                                        )
                                        .on_hover_text(
                                            "No cached preview for the selected voice — \
                                             first click triggers a background render (1–10 s).",
                                        );
                                    }
                                    // Per-row voice picker. Only shown
                                    // for MIDI tracks because non-MIDI
                                    // rows have a fixed engine baked
                                    // into the file name (sfz / modal
                                    // / etc.) — there's nothing to
                                    // re-render. Selecting a different
                                    // voice persists immediately to
                                    // play_log.track_voices and queues
                                    // an auto-preview so the user
                                    // hears the new pick without a
                                    // second click.
                                    if let Some(cur_voice) = &selected_voice {
                                        let mut picked: Option<String> = None;
                                        let cur_text = cur_voice.clone();
                                        let cached_voices_for_label = self
                                            .cached_midi_labels
                                            .get(&t.label)
                                            .cloned()
                                            .unwrap_or_default();
                                        egui::ComboBox::from_id_salt(format!("voice-{i}"))
                                            .selected_text(
                                                egui::RichText::new(&cur_text).monospace().small(),
                                            )
                                            .width(112.0)
                                            .show_ui(ui, |ui| {
                                                for v in AVAILABLE_VOICES {
                                                    let is_cur = *v == cur_text;
                                                    let is_cached =
                                                        cached_voices_for_label.contains(*v);
                                                    let glyph = if is_cached { "\u{2713} " } else { "  " };
                                                    let item = format!("{glyph}{v}");
                                                    if ui
                                                        .selectable_label(
                                                            is_cur,
                                                            egui::RichText::new(item)
                                                                .monospace()
                                                                .small(),
                                                        )
                                                        .on_hover_text(if is_cached {
                                                            "Already rendered for this track \
                                                             — switching is instant."
                                                        } else {
                                                            "Will trigger a background render \
                                                             (1–10 s) the first time."
                                                        })
                                                        .clicked()
                                                    {
                                                        picked = Some((*v).to_string());
                                                    }
                                                }
                                            });
                                        if let Some(v) = picked {
                                            if v != cur_text {
                                                to_set_voice.push((i, v));
                                            }
                                        }
                                    }
                                    let engine_text = match &selected_voice {
                                        Some(v) => v.as_str(),
                                        None => t.engine.as_str(),
                                    };
                                    let label = if active {
                                        format!("\u{25B6} {}", engine_text)
                                    } else if is_rendering {
                                        // Live elapsed-time counter so
                                        // the user knows the click
                                        // landed and roughly how long
                                        // the render has been in
                                        // flight. Updated by the
                                        // standard 50 ms repaint
                                        // schedule.
                                        let secs = self
                                            .pending_render
                                            .as_ref()
                                            .map(|p| p.started.elapsed().as_secs_f32())
                                            .unwrap_or(0.0);
                                        format!("\u{231B} {} ({:.1}s)", engine_text, secs)
                                    } else if t.format == Format::Mid {
                                        format!("\u{25B6} {}", engine_text)
                                    } else {
                                        t.engine.clone()
                                    };
                                    let mut hover = format!(
                                        "{} ({} KB)",
                                        t.path.display(),
                                        t.size_kb
                                    );
                                    if let Some(song) = self.song_index.get(&t.label) {
                                        // DB-backed metadata. Newline-
                                        // separated so egui's hover
                                        // tooltip wraps it properly,
                                        // and ordered the way a user
                                        // skims (title -> composer ->
                                        // era / instrument / license).
                                        hover.push_str("\n\n");
                                        hover.push_str(&song.title);
                                        hover.push('\n');
                                        hover.push_str(&song.composer);
                                        if let Some(era) = &song.era {
                                            hover.push_str(&format!("  ·  {era}"));
                                        }
                                        hover.push_str(&format!(
                                            "  ·  {}",
                                            song.instrument
                                        ));
                                        hover.push_str(&format!(
                                            "  ·  {}",
                                            song.license
                                        ));
                                        if let Some(v) = &song.suggested_voice {
                                            hover.push_str(&format!(
                                                "\nsuggested voice: {v}"
                                            ));
                                        }
                                        if let Some(ctx) = &song.context {
                                            hover.push_str("\n\n");
                                            hover.push_str(ctx);
                                        }
                                    }
                                    if ui
                                        .selectable_label(active, label)
                                        .on_hover_text(hover)
                                        .clicked()
                                    {
                                        *to_play = Some(i);
                                    }
                                }
                            });
                        };

                    if inline {
                        for (piece, indices) in pieces {
                            render_piece_row(
                                ui,
                                piece,
                                indices,
                                &mut to_play,
                                &mut to_set_fav,
                                &mut to_set_voice,
                                &mut to_select,
                            );
                            ui.add_space(2.0);
                        }
                    } else {
                        let header_text = if any_active_in_root {
                            egui::RichText::new(format!(
                                "{root}  ({n_pieces} pieces · {total_renders} renders)  ▶"
                            ))
                            .color(egui::Color32::from_rgb(255, 200, 80))
                            .strong()
                        } else {
                            egui::RichText::new(format!(
                                "{root}  ({n_pieces} pieces · {total_renders} renders)"
                            ))
                            .strong()
                        };
                        let header = egui::CollapsingHeader::new(header_text)
                            .default_open(any_active_in_root || total_renders <= 3)
                            .id_salt(format!("fold-{root}"));
                        header.show(ui, |ui| {
                            for (piece, indices) in pieces {
                                render_piece_row(
                                    ui,
                                    piece,
                                    indices,
                                    &mut to_play,
                                    &mut to_set_fav,
                                    &mut to_set_voice,
                                    &mut to_select,
                                );
                                ui.add_space(2.0);
                            }
                        });
                    }
                }
                // Voice picks have to settle BEFORE `to_play`: a
                // voice change auto-fires preview for the same row, so
                // we want the latest pick reflected when `preview()`
                // resolves the cache key. Multiple picks on the same
                // row in one frame are unlikely (egui delivers one
                // selectable click at a time) but we defensively keep
                // only the last per-row.
                let mut auto_play_after_voice: Option<usize> = None;
                if !to_set_voice.is_empty() {
                    let mut last_per_idx: HashMap<usize, String> = HashMap::new();
                    for (idx, voice) in to_set_voice {
                        last_per_idx.insert(idx, voice);
                    }
                    for (idx, voice) in last_per_idx {
                        let path = match self.tracks.get(idx) {
                            Some(t) => t.path.clone(),
                            None => continue,
                        };
                        let label = match self.tracks.get(idx) {
                            Some(t) => t.label.clone(),
                            None => continue,
                        };
                        if self.set_track_voice(&label, &path, &voice) {
                            // Auto-fire preview so the user immediately
                            // hears the new voice. Cache hits play
                            // straight away; misses kick off the
                            // background render with progress UI.
                            auto_play_after_voice = Some(idx);
                        }
                    }
                }
                if let Some(i) = auto_play_after_voice.or(to_play) {
                    self.preview(i);
                }
                if !to_set_fav.is_empty() {
                    self.set_favorites_batch(&to_set_fav);
                }
                if let Some(stem) = to_select {
                    self.set_selected_label(&stem);
                }
            });
        });

        // Publish the post-paint snapshot for any waiting CP reader.
        // Done last so a verifier polling `get_state` after a queued
        // command observes the resulting mutation in this same frame.
        self.cp_frame_id = self.cp_frame_id.wrapping_add(1);
        let snap = self.build_cp_snapshot();
        self.cp_state.publish(snap);
    }
}

/// Lightweight UI-side mirror of MixerTrack so the egui frame can read
/// state without holding the audio Mutex while drawing.
#[allow(dead_code)]
struct MixerSnap {
    has_file: bool,
    file_label: String,
    volume: f32,
    pan: f32,
    muted: bool,
    solo: bool,
    loop_enabled: bool,
    is_playing: bool,
    cursor_frames: u64,
    sr: u32,
    total_frames: u64,
    vu_peak_l: f32,
    vu_peak_r: f32,
    vu_rms_l: f32,
    vu_rms_r: f32,
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
            .with_inner_size([1280.0, 880.0])
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_from_extension() {
        assert_eq!(Format::from_ext("wav"), Some(Format::Wav));
        assert_eq!(Format::from_ext("WAV"), Some(Format::Wav));
        assert_eq!(Format::from_ext("mp3"), Some(Format::Mp3));
        assert_eq!(Format::from_ext("MP3"), Some(Format::Mp3));
        assert_eq!(Format::from_ext("flac"), None);
    }

    #[test]
    fn parse_track_assigns_format() {
        let dir = std::env::temp_dir().join("keysynth_jukebox_parse_test");
        let _ = std::fs::create_dir_all(&dir);
        let wav = dir.join("piece_a_sfz.wav");
        let mp3 = dir.join("piece_a_modal.mp3");
        std::fs::write(&wav, b"RIFF").unwrap();
        std::fs::write(&mp3, b"ID3").unwrap();
        let tw = parse_track(&wav).unwrap();
        let tm = parse_track(&mp3).unwrap();
        assert_eq!(tw.format, Format::Wav);
        assert_eq!(tw.engine, "sfz");
        assert_eq!(tw.piece, "piece_a");
        assert_eq!(tm.format, Format::Mp3);
        assert_eq!(tm.engine, "modal");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Verify the symphonia path actually decodes a real MP3 from
    /// bench-out/songs/. Uses `twinkle_sfz.mp3` if present (skipped
    /// otherwise so the test passes on a fresh checkout). Reads ~10
    /// frames worth of samples and checks they're finite + non-zero.
    #[test]
    fn open_mp3_real_file_decodes_samples() {
        let path = std::path::PathBuf::from("bench-out/songs/twinkle_sfz.mp3");
        if !path.exists() {
            eprintln!("skip: {} not present", path.display());
            return;
        }
        let mut dec = open_mp3(&path).expect("open_mp3");
        assert!(dec.sample_rate > 0);
        assert!(dec.channels > 0);
        let mut got_nonzero = false;
        // Drain enough samples to span at least one MP3 frame (1152
        // samples per channel) plus a margin.
        for _ in 0..(1152 * dec.channels as usize * 2) {
            match read_one_sample(&mut dec) {
                Some(s) => {
                    assert!(s.is_finite(), "sample {s} not finite");
                    if s.abs() > 0.0 {
                        got_nonzero = true;
                    }
                }
                None => break,
            }
        }
        assert!(got_nonzero, "MP3 decoded as silence — decoder broken");
    }

    #[test]
    fn compact_composer_strips_year_paren() {
        assert_eq!(compact_composer("J.S. Bach (1685-1750)"), "J.S. Bach");
        assert_eq!(
            compact_composer("Francisco Tárrega (1852-1909)"),
            "Francisco Tárrega"
        );
        assert_eq!(compact_composer("Traditional (American)"), "Traditional");
        assert_eq!(compact_composer("Anonymous"), "Anonymous");
    }

    #[test]
    fn license_short_buckets() {
        assert_eq!(license_short("Public Domain"), "PD");
        assert_eq!(license_short("public domain"), "PD");
        assert_eq!(license_short("CC0 1.0"), "CC0");
        assert_eq!(license_short("CC-BY 4.0"), "CC-BY");
        assert_eq!(license_short("CC-BY-SA 4.0"), "CC-BY-SA");
        assert_eq!(license_short(""), "?");
        // Anything unrecognised passes through verbatim so the user can
        // still see the raw string and decide.
        assert_eq!(license_short("Mutopia restricted"), "Mutopia restricted");
    }

    #[test]
    fn humanize_ago_picks_natural_unit() {
        assert_eq!(humanize_ago(-5), "0s ago"); // clock skew → clamp
        assert_eq!(humanize_ago(0), "0s ago");
        assert_eq!(humanize_ago(45), "45s ago");
        assert_eq!(humanize_ago(60), "1m ago");
        assert_eq!(humanize_ago(125), "2m ago");
        assert_eq!(humanize_ago(3_600), "1h ago");
        assert_eq!(humanize_ago(7_200 + 30), "2h ago");
        assert_eq!(humanize_ago(86_400), "1d ago");
        assert_eq!(humanize_ago(3 * 86_400 + 7_200), "3d ago");
    }

    #[test]
    fn scan_dirs_prefers_mp3_over_same_stem_wav() {
        let dir = std::env::temp_dir().join("keysynth_jukebox_dedup_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Same stem as both wav and mp3 — mp3 should win.
        let wav_dup = dir.join("twinkle_modal.wav");
        let mp3_dup = dir.join("twinkle_modal.mp3");
        // Wav-only stem should survive.
        let wav_solo = dir.join("twinkle_sfz.wav");
        // Mp3-only stem should survive.
        let mp3_solo = dir.join("twinkle_piano.mp3");
        std::fs::write(&wav_dup, b"RIFF").unwrap();
        std::fs::write(&mp3_dup, b"ID3").unwrap();
        std::fs::write(&wav_solo, b"RIFF").unwrap();
        std::fs::write(&mp3_solo, b"ID3").unwrap();
        let tracks = scan_dirs(&[dir.as_path()]);
        // Expect 3 tracks: dup as MP3, wav-solo as WAV, mp3-solo as MP3.
        assert_eq!(tracks.len(), 3, "tracks: {tracks:#?}");
        let dup = tracks
            .iter()
            .find(|t| t.label == "twinkle_modal")
            .expect("twinkle_modal present");
        assert_eq!(dup.format, Format::Mp3, "mp3 should beat wav for same stem");
        let solo_wav = tracks
            .iter()
            .find(|t| t.label == "twinkle_sfz")
            .expect("twinkle_sfz present");
        assert_eq!(solo_wav.format, Format::Wav);
        let solo_mp3 = tracks
            .iter()
            .find(|t| t.label == "twinkle_piano")
            .expect("twinkle_piano present");
        assert_eq!(solo_mp3.format, Format::Mp3);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
