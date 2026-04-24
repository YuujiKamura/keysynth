//! Minimal SFZ sampler engine.
//!
//! Goal: load a real recorded piano library (Salamander Grand V3 or similar
//! SFZ manifest), trigger sample playback per MIDI note_on / note_off, and
//! render into the keysynth mono bus. The intent is "stop fighting the
//! 1990s SF2 bank, substitute real 48kHz/16-bit Yamaha C5 recordings".
//!
//! Scope deliberately narrow. We support only the subset of SFZ that
//! Salamander V3 actually uses:
//!
//!   <region>
//!   sample=        relative WAV path
//!   lokey / hikey  inclusive MIDI note range
//!   key            single-note shortcut (sets lokey==hikey==pitch_keycenter)
//!   lovel / hivel  inclusive velocity range (the 16-velocity stratification)
//!   pitch_keycenter  MIDI note the sample was recorded at
//!   volume         per-region gain in dB
//!   tune           per-region pitch offset in cents
//!   loop_mode      ignored (Salamander uses full-length one-shots)
//!   trigger        release/attack — release samples mix in on note_off
//!   ampeg_release  release envelope time in seconds (capped, fallback to 0.05)
//!
//! Skipped (would bloat scope for minimal gain on Salamander):
//!   - `<group>` opcode inheritance (we only honour per-<region>; Salamander's
//!     groups add at most a common path prefix and loop_mode=no_loop, both
//!     handled other ways).
//!   - `<global>`, filters, LFOs, effects, random, seq_position.
//!   - Stereo-sample-to-stereo-bus routing. keysynth is mono; we downmix.
//!
//! Pitch-shifting is linear interpolation on sample index — cheap, causes
//! some aliasing/imaging for shifts beyond ~3 semitones but Salamander V3
//! records every 3rd note so any given note is at most 1.5 semitones from a
//! recorded sample. Aliasing stays below audibility.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use hound::{SampleFormat, WavReader};

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

/// One `<region>` parsed out of the SFZ manifest.
#[derive(Debug, Clone)]
pub struct SfzRegion {
    pub sample_idx: usize, // index into SfzPlayer::samples
    pub lokey: u8,
    pub hikey: u8,
    pub lovel: u8,
    pub hivel: u8,
    pub pitch_keycenter: u8,
    /// Linear gain (already-converted from volume=... dB).
    pub gain: f32,
    /// Pitch ratio multiplier from `tune` (cents). 1.0 by default.
    pub tune_ratio: f32,
    /// Is this a release-triggered region (plays on note_off instead of note_on)?
    pub trigger_release: bool,
    /// Release envelope time in seconds. Default 0.05s if unspecified.
    pub ampeg_release: f32,
}

/// A decoded WAV sample held in memory. All channels interleaved as f32.
/// We keep stereo when present; `num_channels` is 1 or 2.
pub struct SfzSample {
    pub data: Vec<f32>,
    pub num_channels: u16,
    pub sample_rate: u32,
    pub num_frames: usize,
    /// Filesystem path (for debugging / logging only).
    pub path: PathBuf,
}

/// One voice actively rendering a sample.
struct SfzVoice {
    region_idx: usize,
    /// Fractional read position in source frames.
    pos: f64,
    /// Frames to advance per output frame (pitch ratio × source/output SR ratio).
    increment: f64,
    /// Per-voice gain (velocity × region.gain, pre-release).
    gain: f32,
    /// AR release envelope: starts at 1.0, decays to 0 across release_samples.
    release_samples_remaining: f32,
    release_samples_total: f32,
    released: bool,
    /// MIDI (channel, note) for note-off matching.
    key: (u8, u8),
    /// Velocity captured at trigger, needed if we later need to match a
    /// release-sample region (plays at the same velocity class as the hit).
    velocity: u8,
}

pub struct SfzPlayer {
    pub regions: Vec<SfzRegion>,
    pub samples: Vec<SfzSample>,
    /// Output sample rate. Used to scale `increment` against each sample's
    /// recorded sample rate so pitch math produces the right speed.
    output_sr: f32,
    voices: Vec<SfzVoice>,
    /// Path of the loaded .sfz (for display).
    pub manifest_path: PathBuf,
    /// Cap voices to bound CPU under chord-hammering. 32 is enough for a
    /// sustain-pedal-down arpeggio without perceptible stealing.
    max_voices: usize,
}

impl SfzPlayer {
    pub fn regions_len(&self) -> usize {
        self.regions.len()
    }
    pub fn samples_len(&self) -> usize {
        self.samples.len()
    }

    /// Load an SFZ manifest + all referenced WAV samples. WAVs are resolved
    /// relative to the manifest's parent directory (standard SFZ convention).
    pub fn load(manifest: &Path, output_sr: f32) -> Result<Self, String> {
        let text = std::fs::read_to_string(manifest)
            .map_err(|e| format!("opening SFZ {:?}: {e}", manifest))?;
        let base = manifest
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));

        let raw_regions = parse_sfz_regions(&text)?;
        if raw_regions.is_empty() {
            return Err(format!("SFZ {:?} contained no <region> blocks", manifest));
        }

        // Intern samples by normalised path so 30+ regions sharing a WAV only
        // decode it once. Salamander actually has one sample per region, so
        // this is mostly a safety net.
        let mut sample_index: HashMap<PathBuf, usize> = HashMap::new();
        let mut samples: Vec<SfzSample> = Vec::new();
        let mut regions: Vec<SfzRegion> = Vec::new();

        for raw in raw_regions {
            let Some(sample_rel) = raw.get("sample") else {
                // Region without a sample — Salamander doesn't do this, skip
                // defensively rather than error.
                continue;
            };
            // SFZ uses backslashes on Windows-authored packs. Normalise.
            let normalised = sample_rel.replace('\\', "/");
            let sample_path = base.join(&normalised);

            let sample_idx = if let Some(&idx) = sample_index.get(&sample_path) {
                idx
            } else {
                let sample = load_wav(&sample_path)
                    .map_err(|e| format!("loading sample {:?}: {e}", sample_path))?;
                let idx = samples.len();
                samples.push(sample);
                sample_index.insert(sample_path.clone(), idx);
                idx
            };

            // Resolve key ranges. `key=N` sets all three; then lokey/hikey
            // and pitch_keycenter can override. Salamander uses lokey/hikey
            // explicitly with pitch_keycenter per-region.
            let default_center: u8 = 60;
            let (mut lokey, mut hikey, mut keycenter) = (0u8, 127u8, default_center);
            if let Some(k) = raw.get("key").and_then(|v| parse_midi_note(v)) {
                lokey = k;
                hikey = k;
                keycenter = k;
            }
            if let Some(v) = raw.get("lokey").and_then(|v| parse_midi_note(v)) {
                lokey = v;
            }
            if let Some(v) = raw.get("hikey").and_then(|v| parse_midi_note(v)) {
                hikey = v;
            }
            if let Some(v) = raw.get("pitch_keycenter").and_then(|v| parse_midi_note(v)) {
                keycenter = v;
            }
            let lovel = raw
                .get("lovel")
                .and_then(|v| v.parse::<u8>().ok())
                .unwrap_or(0);
            let hivel = raw
                .get("hivel")
                .and_then(|v| v.parse::<u8>().ok())
                .unwrap_or(127);

            let volume_db = raw
                .get("volume")
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(0.0);
            let tune_cents = raw
                .get("tune")
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(0.0);

            let trigger_release = raw
                .get("trigger")
                .map(|v| v.eq_ignore_ascii_case("release"))
                .unwrap_or(false);
            let ampeg_release = raw
                .get("ampeg_release")
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(0.05)
                .clamp(0.005, 10.0);

            regions.push(SfzRegion {
                sample_idx,
                lokey,
                hikey,
                lovel,
                hivel,
                pitch_keycenter: keycenter,
                gain: db_to_gain(volume_db),
                tune_ratio: cents_to_ratio(tune_cents),
                trigger_release,
                ampeg_release,
            });
        }

        Ok(Self {
            regions,
            samples,
            output_sr,
            voices: Vec::with_capacity(64),
            manifest_path: manifest.to_path_buf(),
            max_voices: 32,
        })
    }

    /// Trigger a note. Picks the best-matching non-release region for
    /// (note, velocity). If no region is in range the closest by note/vel
    /// distance is used so the user never hits silence.
    pub fn note_on(&mut self, channel: u8, note: u8, velocity: u8) {
        let velocity = velocity.max(1);
        let Some(region_idx) = self.pick_region(note, velocity, false) else {
            return;
        };
        let region = &self.regions[region_idx];
        let sample = &self.samples[region.sample_idx];

        // Pitch ratio: (2^((note - keycenter)/12)) * tune_ratio, then scale
        // for source-sample-rate mismatch against the output rate.
        let semi = note as f32 - region.pitch_keycenter as f32;
        let pitch_ratio = 2.0_f32.powf(semi / 12.0) * region.tune_ratio;
        let sr_ratio = (sample.sample_rate as f32) / self.output_sr;
        let increment = (pitch_ratio * sr_ratio) as f64;

        // Velocity gain: linear 0..1 curve. SFZ's "proper" curve is
        // amp_velcurve, but keeping it simple passes Salamander fine
        // since it already stratifies velocity into 16 layers.
        let vel_gain = (velocity as f32) / 127.0;
        let gain = region.gain * vel_gain;

        let release_s = region.ampeg_release;
        let release_total = (release_s * self.output_sr).max(1.0);

        // Evict if at cap: prefer an already-released voice, else oldest.
        if self.voices.len() >= self.max_voices {
            let evict = self.voices.iter().position(|v| v.released).unwrap_or(0);
            self.voices.remove(evict);
        }

        self.voices.push(SfzVoice {
            region_idx,
            pos: 0.0,
            increment,
            gain,
            release_samples_remaining: release_total,
            release_samples_total: release_total,
            released: false,
            key: (channel, note),
            velocity,
        });
    }

    pub fn note_off(&mut self, channel: u8, note: u8) {
        // Start release on any matching attack voice.
        let mut release_vel: Option<u8> = None;
        for v in self.voices.iter_mut() {
            if v.key == (channel, note) && !v.released {
                v.released = true;
                release_vel = Some(v.velocity);
            }
        }
        // Fire a release-trigger region if one matches. Salamander V3 ships
        // dedicated release samples: trigger=release, lokey/hikey matching
        // the held note. Velocity of the released region is typically the
        // note-on velocity (that's what "release" libraries key off), so we
        // reuse the captured attack velocity.
        if let Some(vel) = release_vel {
            if let Some(idx) = self.pick_region(note, vel, true) {
                let region = &self.regions[idx];
                let sample = &self.samples[region.sample_idx];
                let semi = note as f32 - region.pitch_keycenter as f32;
                let pitch_ratio = 2.0_f32.powf(semi / 12.0) * region.tune_ratio;
                let sr_ratio = (sample.sample_rate as f32) / self.output_sr;
                let increment = (pitch_ratio * sr_ratio) as f64;
                // Release samples are short; they play once and die with
                // the sample end, so release envelope is effectively the
                // sample itself. Keep release_samples_remaining long enough
                // not to fade prematurely.
                let release_total = (sample.num_frames as f32).max(1.0);
                self.voices.push(SfzVoice {
                    region_idx: idx,
                    pos: 0.0,
                    increment,
                    gain: region.gain * (vel as f32 / 127.0) * 0.35,
                    release_samples_remaining: release_total,
                    release_samples_total: release_total,
                    // Already "released" so the AR envelope decays — but
                    // we set both timers identical so the multiplier stays
                    // near 1.0 for the duration of the short release hit.
                    released: true,
                    key: (channel, note),
                    velocity: vel,
                });
            }
        }
    }

    /// Render `frames` samples of mixed audio into left/right stereo buffers.
    /// Caller provides buffers; we ADD into them (so multiple engines can
    /// share the bus). Mono samples get copied to both channels.
    pub fn render(&mut self, left: &mut [f32], right: &mut [f32]) {
        debug_assert_eq!(left.len(), right.len());
        let frames = left.len();

        // Borrow-checker: we mutate voices AND index samples. Index samples
        // through a raw pointer-ish pattern by taking a snapshot reference
        // before the loop — samples Vec never mutates during render.
        let samples_ptr: *const SfzSample = self.samples.as_ptr();
        let samples_len = self.samples.len();
        let regions_ptr: *const SfzRegion = self.regions.as_ptr();
        let regions_len = self.regions.len();

        for voice in self.voices.iter_mut() {
            if voice.region_idx >= regions_len {
                continue;
            }
            // SAFETY: regions/samples Vecs are only mutated at load time;
            // during render they're effectively read-only, and we bounds-
            // check the indices above and below.
            let region = unsafe { &*regions_ptr.add(voice.region_idx) };
            if region.sample_idx >= samples_len {
                continue;
            }
            let sample = unsafe { &*samples_ptr.add(region.sample_idx) };
            voice_render_add(voice, sample, left, right, frames);
        }

        // Drop finished voices (position past end OR release fully decayed).
        self.voices.retain(|v| {
            let sample = &self.samples[self.regions[v.region_idx].sample_idx];
            let done_by_position = v.pos as usize >= sample.num_frames;
            let done_by_release = v.released && v.release_samples_remaining <= 0.0;
            !(done_by_position || done_by_release)
        });
    }

    /// Pick the region matching (note, velocity), optionally filtering to
    /// `trigger=release` regions. If no region is strictly in-range, falls
    /// back to the nearest (by note then velocity distance) of the same
    /// release/attack class so note_on never produces silence.
    fn pick_region(&self, note: u8, velocity: u8, want_release: bool) -> Option<usize> {
        let mut best_exact: Option<usize> = None;
        let mut best_fallback: Option<(usize, i32)> = None;
        for (i, r) in self.regions.iter().enumerate() {
            if r.trigger_release != want_release {
                continue;
            }
            let in_key = note >= r.lokey && note <= r.hikey;
            let in_vel = velocity >= r.lovel && velocity <= r.hivel;
            if in_key && in_vel {
                best_exact = Some(i);
                break;
            }
            // Distance metric: 4× key distance + velocity distance, so we
            // prefer the right pitch over the right velocity layer.
            let kdist = if note < r.lokey {
                (r.lokey - note) as i32
            } else if note > r.hikey {
                (note - r.hikey) as i32
            } else {
                0
            };
            let vdist = if velocity < r.lovel {
                (r.lovel - velocity) as i32
            } else if velocity > r.hivel {
                (velocity - r.hivel) as i32
            } else {
                0
            };
            let score = kdist * 4 + vdist;
            match best_fallback {
                Some((_, s)) if s <= score => {}
                _ => best_fallback = Some((i, score)),
            }
        }
        best_exact.or(best_fallback.map(|(i, _)| i))
    }
}

fn voice_render_add(
    voice: &mut SfzVoice,
    sample: &SfzSample,
    left: &mut [f32],
    right: &mut [f32],
    frames: usize,
) {
    let ch = sample.num_channels as usize;
    let n_frames = sample.num_frames;
    if n_frames < 2 || ch == 0 {
        return;
    }
    let data = sample.data.as_slice();

    for i in 0..frames {
        let p = voice.pos;
        let p_int = p as usize;
        if p_int + 1 >= n_frames {
            // Off the end. Let retain() remove it.
            voice.pos = n_frames as f64;
            break;
        }
        let frac = (p - p_int as f64) as f32;

        // Linear interpolation per channel.
        let (l, r) = if ch == 1 {
            let a = data[p_int];
            let b = data[p_int + 1];
            let s = a + (b - a) * frac;
            (s, s)
        } else {
            // Interleaved stereo.
            let base_a = p_int * ch;
            let base_b = (p_int + 1) * ch;
            let la = data[base_a];
            let ra = data[base_a + 1];
            let lb = data[base_b];
            let rb = data[base_b + 1];
            let l = la + (lb - la) * frac;
            let r = ra + (rb - ra) * frac;
            (l, r)
        };

        // Release envelope multiplier (AR shape: 1.0 during hold, linear
        // decay over `release_samples_total` once released).
        let env = if voice.released {
            let rem = voice.release_samples_remaining;
            let env = (rem / voice.release_samples_total).max(0.0);
            voice.release_samples_remaining = rem - 1.0;
            env
        } else {
            1.0
        };

        let g = voice.gain * env;
        left[i] += l * g;
        right[i] += r * g;
        voice.pos = p + voice.increment;
    }
}

// ---------------------------------------------------------------------------
// SFZ text parser
// ---------------------------------------------------------------------------
//
// Salamander-shaped SFZ is very regular:
//
//   // comment
//   <region>
//   sample=samples/A0v1.wav
//   lokey=21 hikey=23 pitch_keycenter=21
//   lovel=1 hivel=15
//   volume=-5
//
// Formally, SFZ allows opcodes on the SAME line space-separated. We support
// that. We also honour `#define`-free output of most exporters.

/// Parse the SFZ text into a list of per-region opcode maps.
///
/// Returns one HashMap per `<region>`; keys are opcode names, values are
/// raw string values (trimmed). Non-region headers (`<group>`, `<global>`,
/// `<control>`) are currently skipped, but any opcodes under an active
/// `<group>` are attached to regions that follow until the next `<group>`,
/// so common group-level defaults (e.g. `loop_mode=no_loop`) propagate.
pub fn parse_sfz_regions(text: &str) -> Result<Vec<HashMap<String, String>>, String> {
    let mut regions: Vec<HashMap<String, String>> = Vec::new();
    let mut group: HashMap<String, String> = HashMap::new();
    let mut current: Option<HashMap<String, String>> = None;
    let mut in_group = false;

    for raw_line in text.lines() {
        // Strip `//` comments but keep the pre-comment part.
        let line = if let Some(idx) = raw_line.find("//") {
            &raw_line[..idx]
        } else {
            raw_line
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Process headers + opcodes token-by-token so "lokey=21 hikey=23"
        // on one line splits properly.
        let mut rest = line;
        while !rest.is_empty() {
            rest = rest.trim_start();
            if rest.is_empty() {
                break;
            }
            if let Some(stripped) = rest.strip_prefix('<') {
                // Header: <region>, <group>, etc.
                if let Some(end) = stripped.find('>') {
                    let tag = &stripped[..end];
                    rest = &stripped[end + 1..];
                    match tag {
                        "region" => {
                            if let Some(r) = current.take() {
                                regions.push(r);
                            }
                            // Start region with a copy of the active group's
                            // opcodes so group-level defaults inherit.
                            current = Some(group.clone());
                            in_group = false;
                        }
                        "group" => {
                            if let Some(r) = current.take() {
                                regions.push(r);
                            }
                            group = HashMap::new();
                            in_group = true;
                        }
                        "global" | "master" | "control" => {
                            if let Some(r) = current.take() {
                                regions.push(r);
                            }
                            // We don't model these; opcodes following them
                            // fall into `group` as an approximation.
                            group = HashMap::new();
                            in_group = true;
                        }
                        _ => {
                            // Unknown header — ignore but keep parsing.
                            if let Some(r) = current.take() {
                                regions.push(r);
                            }
                            in_group = false;
                        }
                    }
                } else {
                    return Err(format!("unterminated header: {raw_line:?}"));
                }
                continue;
            }

            // Opcode `key=value`. Value runs until the next whitespace
            // EXCEPT for opcodes whose values can contain spaces or path
            // separators. `sample=` is the classic offender. Handle by
            // greedy-to-end-of-line for that specific key.
            if let Some(eq) = rest.find('=') {
                let key = rest[..eq].trim().to_string();
                let after = &rest[eq + 1..];
                let value = if key == "sample" {
                    // Greedy: the value is everything up to the next opcode.
                    // Scan forward for the next `WHITESPACE + word + =` pattern.
                    match find_next_opcode_boundary(after) {
                        Some(bound) => {
                            let v = after[..bound].trim().to_string();
                            rest = &after[bound..];
                            v
                        }
                        None => {
                            let v = after.trim().to_string();
                            rest = "";
                            v
                        }
                    }
                } else {
                    // Value is next whitespace-delimited token.
                    let token_end = after.find(char::is_whitespace).unwrap_or(after.len());
                    let v = after[..token_end].trim().to_string();
                    rest = &after[token_end..];
                    v
                };
                if in_group {
                    group.insert(key, value);
                } else if let Some(r) = current.as_mut() {
                    r.insert(key, value);
                }
                // Opcode outside any header is technically ignored.
            } else {
                // Unparseable junk; skip the rest of the line.
                break;
            }
        }
    }

    if let Some(r) = current.take() {
        regions.push(r);
    }
    Ok(regions)
}

/// Given the substring starting *after* a `sample=` value, find where the
/// value ends — i.e. the start of the next `name=...` opcode. SFZ lets the
/// sample path contain spaces, so we can't just split on whitespace.
///
/// Heuristic: scan forward for ` name=` where `name` is [a-zA-Z_][a-zA-Z0-9_]+.
fn find_next_opcode_boundary(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b' ' || bytes[i] == b'\t' {
            let start = i;
            // Skip whitespace
            while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
                i += 1;
            }
            // Look for identifier followed by '='
            let ident_start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            if i > ident_start && i < bytes.len() && bytes[i] == b'=' {
                return Some(start);
            }
            // Not an opcode; continue scanning.
            continue;
        }
        i += 1;
    }
    None
}

/// MIDI notes can appear either as `60` or as `c4` / `c#4` / `db4`. Accept
/// both so we don't crash on mixed-convention SFZ packs.
pub fn parse_midi_note(s: &str) -> Option<u8> {
    if let Ok(n) = s.parse::<u8>() {
        return Some(n);
    }
    let s = s.trim().to_ascii_lowercase();
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    // Match letter + optional # or b + signed octave.
    let letter = bytes[0];
    let semitone_base: i32 = match letter {
        b'c' => 0,
        b'd' => 2,
        b'e' => 4,
        b'f' => 5,
        b'g' => 7,
        b'a' => 9,
        b'b' => 11,
        _ => return None,
    };
    let (mut accidental, rest_idx) = (0i32, 1);
    let rest_idx = if bytes.len() > 1 && bytes[1] == b'#' {
        accidental = 1;
        2
    } else if bytes.len() > 1 && bytes[1] == b'b' && letter != b'b' {
        accidental = -1;
        2
    } else {
        rest_idx
    };
    if rest_idx >= bytes.len() {
        return None;
    }
    let octave_str = &s[rest_idx..];
    let octave: i32 = octave_str.parse().ok()?;
    // C4 = MIDI 60 convention (Yamaha / Apple). SFZ uses this.
    let midi = 12 * (octave + 1) + semitone_base + accidental;
    if (0..=127).contains(&midi) {
        Some(midi as u8)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// WAV loading
// ---------------------------------------------------------------------------

fn load_wav(path: &Path) -> Result<SfzSample, String> {
    let file = File::open(path).map_err(|e| format!("open {:?}: {e}", path))?;
    let reader = BufReader::new(file);
    let mut wav = WavReader::new(reader).map_err(|e| format!("WavReader {:?}: {e}", path))?;
    let spec = wav.spec();
    let num_channels = spec.channels;
    if num_channels == 0 || num_channels > 2 {
        return Err(format!(
            "unsupported channel count {} in {:?}",
            num_channels, path
        ));
    }
    let sample_rate = spec.sample_rate;
    let bits = spec.bits_per_sample;

    let data: Vec<f32> = match (spec.sample_format, bits) {
        (SampleFormat::Int, 16) => wav
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("reading 16-bit {:?}: {e}", path))?,
        (SampleFormat::Int, 24) => wav
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / 8_388_608.0)) // 2^23
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("reading 24-bit {:?}: {e}", path))?,
        (SampleFormat::Int, 32) => wav
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("reading 32-bit int {:?}: {e}", path))?,
        (SampleFormat::Float, 32) => wav
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("reading 32-bit float {:?}: {e}", path))?,
        (fmt, b) => {
            return Err(format!(
                "unsupported WAV format: {:?} {}-bit in {:?}",
                fmt, b, path
            ));
        }
    };

    let num_frames = data.len() / num_channels as usize;
    Ok(SfzSample {
        data,
        num_channels,
        sample_rate,
        num_frames,
        path: path.to_path_buf(),
    })
}

fn db_to_gain(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

fn cents_to_ratio(cents: f32) -> f32 {
    2.0_f32.powf(cents / 1200.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_region() {
        let sfz = "\
<region>
sample=samples/a4.wav
lokey=57 hikey=60 pitch_keycenter=60
lovel=1 hivel=63
volume=-3
";
        let regions = parse_sfz_regions(sfz).unwrap();
        assert_eq!(regions.len(), 1);
        let r = &regions[0];
        assert_eq!(r.get("sample").map(|s| s.as_str()), Some("samples/a4.wav"));
        assert_eq!(r.get("lokey").map(|s| s.as_str()), Some("57"));
        assert_eq!(r.get("hikey").map(|s| s.as_str()), Some("60"));
        assert_eq!(r.get("pitch_keycenter").map(|s| s.as_str()), Some("60"));
        assert_eq!(r.get("lovel").map(|s| s.as_str()), Some("1"));
        assert_eq!(r.get("hivel").map(|s| s.as_str()), Some("63"));
        assert_eq!(r.get("volume").map(|s| s.as_str()), Some("-3"));
    }

    #[test]
    fn parse_group_inheritance() {
        let sfz = "\
<group>
loop_mode=no_loop
ampeg_release=0.5
<region>
sample=samples/c4.wav
key=60
<region>
sample=samples/d4.wav
key=62
";
        let regions = parse_sfz_regions(sfz).unwrap();
        assert_eq!(regions.len(), 2);
        for r in &regions {
            assert_eq!(r.get("loop_mode").map(|s| s.as_str()), Some("no_loop"));
            assert_eq!(r.get("ampeg_release").map(|s| s.as_str()), Some("0.5"));
        }
    }

    #[test]
    fn parse_comments_and_whitespace() {
        let sfz = "\
// Top comment
<region> // inline comment
sample=a.wav lokey=1 hikey=2
";
        let regions = parse_sfz_regions(sfz).unwrap();
        assert_eq!(regions.len(), 1);
        let r = &regions[0];
        assert_eq!(r.get("sample").map(|s| s.as_str()), Some("a.wav"));
        assert_eq!(r.get("lokey").map(|s| s.as_str()), Some("1"));
        assert_eq!(r.get("hikey").map(|s| s.as_str()), Some("2"));
    }

    #[test]
    fn midi_note_letter_form() {
        assert_eq!(parse_midi_note("c4"), Some(60));
        assert_eq!(parse_midi_note("a4"), Some(69));
        assert_eq!(parse_midi_note("c#4"), Some(61));
        assert_eq!(parse_midi_note("db4"), Some(61));
        assert_eq!(parse_midi_note("60"), Some(60));
    }

    #[test]
    fn midi_note_invalid_returns_none() {
        assert_eq!(parse_midi_note(""), None);
        assert_eq!(parse_midi_note("z4"), None);
        assert_eq!(parse_midi_note("c"), None);
    }

    #[test]
    fn midi_note_out_of_range_returns_none() {
        // C-2 = -12 (negative MIDI), out of range.
        assert_eq!(parse_midi_note("c-2"), None);
        // G10 = 127+? — far above 127.
        assert_eq!(parse_midi_note("c10"), None);
    }

    #[test]
    fn midi_note_uppercase_accepted() {
        // parse_midi_note lowercases internally, so uppercase letters
        // should still match.
        assert_eq!(parse_midi_note("C4"), Some(60));
        assert_eq!(parse_midi_note("A4"), Some(69));
    }

    #[test]
    fn parse_empty_sfz_yields_empty_regions() {
        let regions = parse_sfz_regions("").unwrap();
        assert!(regions.is_empty());
    }

    #[test]
    fn parse_only_comments_yields_empty_regions() {
        let regions = parse_sfz_regions("// only a comment\n// another\n").unwrap();
        assert!(regions.is_empty());
    }

    #[test]
    fn parse_unterminated_header_returns_err() {
        let r = parse_sfz_regions("<region\nsample=a.wav\n");
        assert!(r.is_err(), "unterminated header should be Err");
    }

    #[test]
    fn parse_sample_with_spaces_in_path() {
        let sfz = "\
<region>
sample=samples/A 3.wav lokey=57 hikey=60
";
        let regions = parse_sfz_regions(sfz).unwrap();
        assert_eq!(regions.len(), 1);
        assert_eq!(
            regions[0].get("sample").map(|s| s.as_str()),
            Some("samples/A 3.wav")
        );
        assert_eq!(regions[0].get("lokey").map(|s| s.as_str()), Some("57"));
    }

    #[test]
    fn parse_multiple_regions_independent() {
        let sfz = "\
<region>
sample=a.wav key=60
<region>
sample=b.wav key=62
";
        let regions = parse_sfz_regions(sfz).unwrap();
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].get("key").map(|s| s.as_str()), Some("60"));
        assert_eq!(regions[1].get("key").map(|s| s.as_str()), Some("62"));
    }

    #[test]
    fn db_to_gain_zero_db_unity() {
        assert!((db_to_gain(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn db_to_gain_minus_six_db_half_amplitude() {
        // -6 dB ≈ 0.5012 amplitude.
        let g = db_to_gain(-6.0);
        assert!((g - 0.5012).abs() < 0.01, "expected ~0.5012, got {g}");
    }

    #[test]
    fn cents_to_ratio_zero_unity() {
        assert!((cents_to_ratio(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cents_to_ratio_one_octave() {
        // 1200 cents = 1 octave = ratio 2.
        let r = cents_to_ratio(1200.0);
        assert!((r - 2.0).abs() < 1e-3);
    }
}
