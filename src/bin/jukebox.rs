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

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use eframe::egui;
use hound::{SampleFormat as WavSampleFormat, WavReader};
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
}

impl Format {
    fn from_ext(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "wav" => Some(Format::Wav),
            "mp3" => Some(Format::Mp3),
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
    "sfz", "modal", "square", "ks", "ks-rich", "sub", "fm", "piano", "piano-thick",
    "piano-lite", "piano-5am", "koto",
    // "pure" marks an NSF/libgme ground-truth render — it isn't a keysynth
    // engine, but tagging it like one lets the fold-by-piece grouper pair
    // nsf_parodius_03_vic_viper_pure with audio_parodius_03_vic_viper /
    // listener_parodius_03_vic_viper_square in the same row.
    "pure",
];

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
            stem.strip_suffix(&suf).map(|p| (p.to_string(), s.to_string()))
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
            if ext != "wav" && ext != "mp3" {
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
    out.sort_by(|a, b| {
        a.piece
            .cmp(&b.piece)
            .then_with(|| a.engine.cmp(&b.engine))
    });
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
    let channels_count = codec_params
        .channels
        .map(|c| c.count() as u16)
        .unwrap_or(2);
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
            Err(SymphoniaError::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
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
        DecoderState::Wav { reader, .. } => {
            reader.seek(0).map_err(|e| format!("hound seek: {e}"))
        }
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

fn build_mixer_stream(
    device_name: &str,
    state: Arc<Mutex<MixerState>>,
) -> Result<Mixer, String> {
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
        let dir_refs: Vec<&Path> = dirs.iter().map(|p| p.as_path()).collect();
        let tracks = scan_dirs(&dir_refs);
        let devices = list_output_devices();
        let initial_slots = 2usize;
        let state = Arc::new(Mutex::new(MixerState::new(initial_slots)));
        let mixer = build_mixer_stream("(default)", state)
            .map_err(|e| format!("audio output: {e}"))?;
        let current_device = mixer.device_name.clone();
        Ok(Self {
            tracks,
            selected: None,
            filter: String::new(),
            mixer,
            refresh_dirs: dirs,
            devices,
            current_device,
            slot_file_idx: vec![None; initial_slots],
            slot_count: initial_slots,
        })
    }

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
        let track = match self.tracks.get(idx) {
            Some(t) => t.clone(),
            None => return,
        };
        let label = format!("[{}] {}_{}", track.source, track.piece, track.engine);
        // Ensure we have at least one slot.
        {
            let mut st = self.mixer.state.lock().unwrap();
            if st.tracks.is_empty() {
                st.tracks.push(MixerTrack::empty());
                self.slot_file_idx.push(None);
                self.slot_count = st.tracks.len();
            }
        }
        // Stop everything first so we don't pile previews.
        self.stop_all();
        {
            let mut st = self.mixer.state.lock().unwrap();
            // Solo slot 0 implicitly by muting other slots' is_playing
            // for this preview. (We don't toggle their stored .solo /
            // .muted so the user's mixer setup survives.)
            for (i, t) in st.tracks.iter_mut().enumerate() {
                if i == 0 {
                    if let Err(e) = t.load(&track.path, label.clone()) {
                        eprintln!("jukebox: load {}: {e}", track.path.display());
                        return;
                    }
                    t.is_playing = true;
                } else {
                    // Don't auto-play other slots in preview mode.
                    t.is_playing = false;
                }
            }
            st.transport_running = true;
            self.slot_file_idx[0] = Some(idx);
        }
        self.selected = Some(idx);
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
                let cur_path = st
                    .tracks
                    .get(i)
                    .and_then(|t| {
                        if t.decoded.is_some() {
                            Some(t.file_path.clone())
                        } else {
                            None
                        }
                    });
                let want_path = want.and_then(|idx| self.tracks.get(idx)).map(|t| {
                    let label = format!("[{}] {}_{}", t.source, t.piece, t.engine);
                    (t.path.clone(), label)
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
}

/// Render a single VU meter bar. peak / rms are 0..1+ (we clamp).
fn vu_meter_bar(ui: &mut egui::Ui, label: &str, peak: f32, rms: f32) {
    let desired_w = 120.0f32;
    let h = 10.0f32;
    let (rect, _resp) =
        ui.allocate_exact_size(egui::vec2(desired_w, h), egui::Sense::hover());
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
        painter.rect_filled(clip_rect, 2.0, egui::Color32::from_rgba_premultiplied(180, 40, 40, 120));
    }
    ui.label(egui::RichText::new(label).monospace().small());
}

impl eframe::App for Jukebox {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(std::time::Duration::from_millis(50));

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
                if ui.button(if any_playing { "■ stop" } else { "—" }).clicked() {
                    self.stop_all();
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
                                let cur_idx_opt = self
                                    .slot_file_idx
                                    .get(slot_idx)
                                    .copied()
                                    .flatten();
                                egui::ComboBox::from_id_salt(format!("mix-{slot_idx}"))
                                    .selected_text(current)
                                    .width(360.0)
                                    .show_ui(ui, |ui| {
                                        if ui
                                            .selectable_label(
                                                cur_idx_opt.is_none(),
                                                "(empty)",
                                            )
                                            .clicked()
                                        {
                                            if slot_idx < self.slot_file_idx.len() {
                                                self.slot_file_idx[slot_idx] = None;
                                            }
                                        }
                                        for (i, label) in &track_choices {
                                            if ui
                                                .selectable_label(
                                                    cur_idx_opt == Some(*i),
                                                    label,
                                                )
                                                .clicked()
                                            {
                                                if slot_idx < self.slot_file_idx.len() {
                                                    self.slot_file_idx[slot_idx] =
                                                        Some(*i);
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
                                    ui.colored_label(
                                        egui::Color32::from_rgb(255, 200, 80),
                                        "▶",
                                    );
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
                                    let pos_s = snap.cursor_frames as f64
                                        / snap.sr as f64;
                                    let total_s = snap.total_frames as f64
                                        / snap.sr as f64;
                                    ui.monospace(format!(
                                        "{:>5.1}/{:>5.1}s",
                                        pos_s, total_s
                                    ));
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
                let avail_w = ui.available_width();
                for (root, pieces) in &folded {
                    let total_renders: usize =
                        pieces.values().map(|v| v.len()).sum();
                    let n_pieces = pieces.len();
                    // If the root has only one piece (and ≤ 4 renders),
                    // skip the collapsing header — render the row
                    // inline for cheap browse.
                    let inline = n_pieces <= 1 && total_renders <= 4;
                    let any_active_in_root = pieces.values().flatten().any(|i| {
                        self.selected == Some(*i) && any_playing
                    });

                    let render_piece_row =
                        |ui: &mut egui::Ui,
                         piece: &str,
                         indices: &[usize],
                         to_play: &mut Option<usize>| {
                            ui.horizontal_wrapped(|ui| {
                                let active_in_group = indices.iter().any(|i| {
                                    self.selected == Some(*i) && any_playing
                                });
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
                                        egui::RichText::new(kind_glyph)
                                            .color(kind_color)
                                            .strong(),
                                    ),
                                )
                                .on_hover_text(kind_hover);

                                let piece_text = if active_in_group {
                                    egui::RichText::new(piece)
                                        .color(egui::Color32::from_rgb(255, 200, 80))
                                        .strong()
                                        .monospace()
                                } else {
                                    egui::RichText::new(piece).monospace()
                                };
                                let piece_w = (avail_w * 0.30).max(200.0);
                                ui.add_sized(
                                    [piece_w, 22.0],
                                    egui::Label::new(piece_text),
                                );
                                ui.separator();
                                for &i in indices {
                                    let t = &self.tracks[i];
                                    let active = self.selected == Some(i) && any_playing;
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
                                        *to_play = Some(i);
                                    }
                                }
                            });
                        };

                    if inline {
                        for (piece, indices) in pieces {
                            render_piece_row(ui, piece, indices, &mut to_play);
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
                                render_piece_row(ui, piece, indices, &mut to_play);
                                ui.add_space(2.0);
                            }
                        });
                    }
                }
                if let Some(i) = to_play {
                    self.preview(i);
                }
            });
        });
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
