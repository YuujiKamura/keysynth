//! Modal piano voice — parallel bandpass resonators driven by a hammer
//! impulse. Pure projection model (issue #3 alt path): each mode is a
//! single (freq, T60, init_amp) triple; the voice is a parallel bank of
//! 2nd-order biquad bandpass IIRs, each tuned to one mode and decaying
//! at exactly its measured T60.
//!
//! No physical string/soundboard model — the modes ARE the projection.
//! Modal LUTs are produced offline from a reference recording via
//! `keysynth::extract::*` and fed into `ModalPianoVoice::new`. This
//! voice is a counterpoint to the physical-model `PianoVoice`
//! (`voices/piano.rs`): instead of approaching the SFZ Salamander
//! reference by tuning string/soundboard parameters, it consumes the
//! reference's measured spectrum directly.
//!
//! Per-mode resonator (textbook biquad bandpass with prescribed pole
//! radius for an exact T60):
//!
//! ```text
//!     omega_k = 2π · f_k / sr
//!     r_k = exp(-3 · ln(10) / (T60_k · sr))     // r^(T60·sr) = 1e-3 = -60 dB
//!     y[n] = b0 · x[n] - a1 · y[n-1] - a2 · y[n-2]
//!     a1 = -2 r_k cos(omega_k)
//!     a2 = r_k²
//!     b0 = 1 - r_k²                             // unity gain at resonance
//! ```
//!
//! Each mode is excited by the same scalar input `x[n]` (the hammer
//! impulse) and contributes `init_amp_k · y_k[n]` to the output. The
//! release envelope multiplies the summed output, identical to the
//! existing physical voices.

use std::path::Path;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::synth::{ReleaseEnvelope, VoiceImpl};

/// One modal resonance: (f, T60, gain). `init_amp` is the LINEAR
/// amplitude (not dB), so callers converting from `extract::Partial::init_db`
/// need to do `10f32.powf(init_db / 20.0)` first.
#[derive(Clone, Copy, Debug)]
pub struct Mode {
    pub freq_hz: f32,
    pub t60_sec: f32,
    pub init_amp: f32,
}

struct ModalResonator {
    /// Cosine of the resonant angular frequency. Coefficient `a1` is
    /// `-2·r·cos_omega` and is recomputed on damper engage to swap the
    /// pole radius without rebuilding the resonator from scratch.
    cos_omega: f32,
    a1: f32,
    a2: f32,
    b0: f32,
    y1: f32,
    y2: f32,
    init_amp: f32,
    sr: f32,
}

impl ModalResonator {
    fn new(mode: Mode, sr: f32) -> Self {
        let omega = 2.0 * std::f32::consts::PI * mode.freq_hz / sr;
        let cos_omega = omega.cos();
        let t60 = mode.t60_sec.max(1e-3);
        let (a1, a2, b0) = compute_coefs(cos_omega, t60, sr);
        Self {
            cos_omega,
            a1,
            a2,
            b0,
            y1: 0.0,
            y2: 0.0,
            init_amp: mode.init_amp,
            sr,
        }
    }

    /// Engage the damper: swap the pole radius to one that decays to
    /// -60 dB in `damper_t60_sec` (typically 0.05-0.20 s). This kills
    /// the mode much faster than its natural T60, mimicking a felt
    /// damper landing on a piano string. Cheap: just re-derives the
    /// three IIR coefficients; internal state (y1, y2) is preserved
    /// so there's no click.
    fn engage_damper(&mut self, damper_t60_sec: f32) {
        let t60 = damper_t60_sec.max(1e-3);
        let (a1, a2, b0) = compute_coefs(self.cos_omega, t60, self.sr);
        self.a1 = a1;
        self.a2 = a2;
        self.b0 = b0;
    }

    #[inline]
    fn step(&mut self, x: f32) -> f32 {
        let y0 = self.b0 * x - self.a1 * self.y1 - self.a2 * self.y2;
        self.y2 = self.y1;
        self.y1 = y0;
        y0 * self.init_amp
    }
}

/// Pole radius chosen so r^(T60·sr) = 1e-3 (i.e. -60 dB at exactly
/// T60 seconds). Returns (a1, a2, b0) for the biquad bandpass with
/// unity peak gain at resonance.
fn compute_coefs(cos_omega: f32, t60_sec: f32, sr: f32) -> (f32, f32, f32) {
    let r = (-3.0 * 10f32.ln() / (t60_sec * sr)).exp();
    let a1 = -2.0 * r * cos_omega;
    let a2 = r * r;
    // |H(e^{jω₀})| = 1 / (1 - r²) at resonance; pre-multiplying b0
    // by (1 - r²) normalises peak to 1 — at the moment of damper
    // engage this rescales the steady-state gain, which is the
    // physically correct behaviour (less coupling area = less
    // radiated amplitude).
    let b0 = 1.0 - r * r;
    (a1, a2, b0)
}

/// Modal piano voice. Holds a parallel bank of resonators (one per
/// mode) plus a hammer-impulse excitation buffer that's played out at
/// note-on. After the impulse runs out, the resonators ring out at
/// their per-mode T60.
///
/// On `trigger_release` the voice engages a virtual damper:
/// `damper_t60_sec` (default 0.12 s) is swapped into every resonator
/// in place of its natural T60, which mimics a wool damper landing on
/// a piano string and stopping it within ~100 ms. Without this the
/// per-mode resonators would happily ring through their full T60
/// (h1 ~ 18 s) regardless of note_off, producing the audible "電子
/// ピアノっぽい / 減衰しない" character of the early pilot.
pub struct ModalPianoVoice {
    resonators: Vec<ModalResonator>,
    /// Hammer-impulse waveform played into every resonator at note-on.
    /// Single-sample delta in the default constructor (broadband, every
    /// mode excited equally); callers can supply any shape.
    excitation: Vec<f32>,
    excitation_idx: usize,
    /// Damper T60 applied to every resonator on `trigger_release`. Real
    /// piano felt dampers stop the string in ~80-200 ms; 0.12 s is a
    /// reasonable middle.
    damper_t60_sec: f32,
    /// Becomes true after `trigger_release` so the damper engages on
    /// the next render pass. Idempotent.
    damper_pending: bool,
    release: ReleaseEnvelope,
}

impl ModalPianoVoice {
    /// Construct from a list of modes and an explicit excitation buffer.
    /// `velocity_amp` (typically `velocity / 127.0`) scales the
    /// excitation peak so per-velocity loudness behaves intuitively.
    /// Each mode is also expanded into 3 sub-modes detuned by ±0.7
    /// cents around the centre frequency, mimicking the natural
    /// detune across a piano's 3-string unison and producing audible
    /// inter-partial beating ("生っぽさ"). Pass `with_modes_no_detune`
    /// if you don't want this expansion.
    pub fn with_modes(sr: f32, modes: &[Mode], excitation: Vec<f32>, velocity_amp: f32) -> Self {
        const DETUNE_CENTS: f32 = 0.7;
        let cents_to_ratio = |c: f32| 2.0_f32.powf(c / 1200.0);
        let mut detuned: Vec<Mode> = Vec::with_capacity(modes.len() * 3);
        for m in modes {
            // Each sub-mode gets 1/3 of the original amplitude so total
            // energy stays constant. Centre is exactly on-pitch; flanks
            // are ±DETUNE_CENTS.
            let third = m.init_amp / 3.0_f32.sqrt();
            detuned.push(Mode {
                freq_hz: m.freq_hz * cents_to_ratio(-DETUNE_CENTS),
                t60_sec: m.t60_sec,
                init_amp: third,
            });
            detuned.push(Mode {
                freq_hz: m.freq_hz,
                t60_sec: m.t60_sec,
                init_amp: third,
            });
            detuned.push(Mode {
                freq_hz: m.freq_hz * cents_to_ratio(DETUNE_CENTS),
                t60_sec: m.t60_sec,
                init_amp: third,
            });
        }
        Self::with_modes_no_detune(sr, &detuned, excitation, velocity_amp)
    }

    /// Construct without the 3-sub-mode detune expansion — modes are
    /// used exactly as given. Useful for testing the resonator bank
    /// in isolation; the user-facing path is `with_modes`.
    pub fn with_modes_no_detune(
        sr: f32,
        modes: &[Mode],
        excitation: Vec<f32>,
        velocity_amp: f32,
    ) -> Self {
        let resonators = modes
            .iter()
            .copied()
            .map(|m| ModalResonator::new(m, sr))
            .collect();
        let scaled_excitation: Vec<f32> = excitation.iter().map(|&s| s * velocity_amp).collect();
        Self {
            resonators,
            excitation: scaled_excitation,
            excitation_idx: 0,
            // Damper T60 raised 0.12 → 0.8 s after live A/B against SFZ
            // ("音圧が足りない" feedback): 0.12 s killed modes within
            // ~0.4 s of note_off while SFZ samples sustain ~1.5-3 s.
            // 0.8 s effective T60 gives ~2.6 s audible tail (-20 dB at
            // T60/3, well above floor) — closer to the recorded
            // release envelope.
            damper_t60_sec: 0.8,
            damper_pending: false,
            release: ReleaseEnvelope::new(0.800, sr),
        }
    }

    /// Override the damper T60 (default 0.8 s). Useful for the una-corda
    /// (slower damper) or for the SF2 placeholder pattern where dampers
    /// are emulated by silence (set very small).
    pub fn set_damper_t60_sec(&mut self, t60_sec: f32) {
        self.damper_t60_sec = t60_sec;
    }

    /// Construct with a default broadband impulse (single-sample delta).
    /// A delta has a flat spectrum across the whole band, so each mode
    /// receives equal excitation energy regardless of its frequency.
    /// Longer / shaped impulses (3 ms half-sine etc.) sound more
    /// "hammer-like" but starve high partials because their spectrum
    /// rolls off — measured deficit was ~27 dB on h2 of the SFZ
    /// Salamander C4 reference, dominating the pilot's amplitude error.
    pub fn new_default_excitation(sr: f32, modes: &[Mode], velocity: u8) -> Self {
        let velocity_amp = (velocity.max(1) as f32) / 127.0;
        // Single-sample delta. The resonator chain shapes the audible
        // transient (each mode rings up over its first cycle), so we
        // don't need an explicit hammer envelope here.
        let excitation = vec![1.0_f32];
        Self::with_modes(sr, modes, excitation, velocity_amp)
    }

    /// Construct a voice for `midi_note` from a runtime-loaded `ModalLut`
    /// (per-note partial / T60 / init_db table; issue #3 A3).
    ///
    /// Pick strategy: nearest-neighbour by `|entry.midi_note - midi_note|`.
    /// The chosen entry's modes are then frequency-scaled by the ratio
    /// `target_f0 / nearest.f0_hz`, where `target_f0` is the equal-tempered
    /// pitch for `midi_note` (A4 = 440 Hz). T60 and init_db are kept as-is
    /// (we don't have a model for how T60 should scale across notes; the
    /// nearest entry's measured T60 is the best estimate available).
    ///
    /// T60 sentinel handling: any mode with `t60_sec <= 0` is treated as
    /// "extractor failed" and substituted with `12.0` s, the same default
    /// as `bin/render_modal --default-t60`.
    ///
    /// Empty LUT panics: callers must guarantee `lut.entries` is non-empty
    /// (the auto-load path enforces this — fallback C4 LUT is always at
    /// least 1 entry).
    pub fn from_lut(lut: &ModalLut, sr: f32, midi_note: u8, velocity: u8) -> Self {
        let entry = lut.nearest_entry(midi_note);
        let target_f0 = 440.0_f32 * 2.0_f32.powf((midi_note as f32 - 69.0) / 12.0);
        let ratio = if entry.f0_hz > 0.0 {
            target_f0 / entry.f0_hz
        } else {
            1.0
        };
        const FALLBACK_T60: f32 = 12.0;

        // Per-partial T60 floor. The `extract::t60` extractor systematically
        // under-reports T60 on some partials (h2 ≈ 2 s vs analyse's 11 s
        // because near-equal-amplitude detuned-string beating defeats the
        // R²/slope gate). For perceptual realism we floor each T60 to a
        // partial-index-aware minimum that mirrors the empirical decay
        // curve of a real grand piano (h1 longest, monotone descent).
        // Visualised diff vs SFZ ("音圧が足りない") showed the modal voice
        // was crashing to silence ~30 dB below the SFZ sustain floor;
        // the LUT-reported T60s were the dominant reason.
        fn t60_floor_for_partial(n: usize) -> f32 {
            // Matches analyse's reference values for SFZ Salamander C4
            // (h1=18, h2=11, h3=10, h4=7, h5=7, h6=8, h7=9, h8=9...);
            // beyond h8 we extrapolate downward on a gentle slope.
            match n {
                0 | 1 => 18.0,
                2 => 10.0,
                3 => 9.0,
                4 => 7.0,
                5 => 7.0,
                6 => 7.5,
                7 => 8.0,
                8 => 8.0,
                _ => (8.0_f32 - (n as f32 - 8.0) * 0.4).max(3.0),
            }
        }

        let modes: Vec<Mode> = entry
            .modes
            .iter()
            .enumerate()
            .map(|(i, m)| {
                // i+1 = partial index 1..N, matching the n=1 .. fundamental
                // numbering used in the extract crate.
                let n = i + 1;
                let raw = if m.t60_sec > 0.0 {
                    m.t60_sec
                } else {
                    FALLBACK_T60
                };
                let t60 = raw.max(t60_floor_for_partial(n));
                Mode {
                    freq_hz: m.freq_hz * ratio,
                    t60_sec: t60,
                    init_amp: m.init_amp,
                }
            })
            .collect();
        let velocity_amp = (velocity.max(1) as f32) / 127.0;
        let excitation = vec![1.0_f32];
        Self::with_modes(sr, &modes, excitation, velocity_amp)
    }
}

// ===========================================================================
// Modal LUT (per-note partial / T60 / init_db table)
// ===========================================================================

/// On-disk JSON representation of one mode within a per-note entry. Uses
/// `init_db` (decibels) on disk because the upstream extractor reports
/// dB; we convert to linear `init_amp` at parse time so the runtime
/// `Mode` shape stays uniform.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ModalLutModeJson {
    pub freq_hz: f32,
    pub t60_sec: f32,
    pub init_db: f32,
}

/// On-disk attack envelope summary (informational; not yet used to shape
/// the hammer impulse, but parsed so the schema round-trips losslessly).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct ModalLutAttackJson {
    #[serde(default)]
    pub time_to_peak_s: f32,
    #[serde(default)]
    pub peak_db: f32,
    #[serde(default)]
    pub post_peak_slope_db_s: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModalLutEntryJson {
    pub midi_note: u8,
    pub f0_hz: f32,
    pub modes: Vec<ModalLutModeJson>,
    #[serde(default)]
    pub attack: Option<ModalLutAttackJson>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModalLutJson {
    pub schema_version: u32,
    pub lut: Vec<ModalLutEntryJson>,
}

/// One per-note entry in the modal LUT. Holds the linear-amplitude
/// `Mode` triples already converted from the on-disk dB representation,
/// plus the source MIDI note and measured fundamental for nearest-neighbour
/// scaling in `ModalPianoVoice::from_lut`.
#[derive(Clone, Debug)]
pub struct ModalLutEntry {
    pub midi_note: u8,
    pub f0_hz: f32,
    pub modes: Vec<Mode>,
    pub time_to_peak_s: f32,
    pub peak_db: f32,
    pub post_peak_slope_db_s: f32,
}

/// Per-note modal partials / T60 / init_amp table. Built from a JSON file
/// (produced by `bin/build_modal_lut`) or from the hardcoded C4 fallback
/// when the JSON isn't available.
#[derive(Clone, Debug)]
pub struct ModalLut {
    pub entries: Vec<ModalLutEntry>,
}

/// Process-wide modal LUT, set once at startup by `main`. Read by
/// `synth::make_voice` when `Engine::PianoModal` is selected. Using a
/// `OnceLock` keeps the audio thread lock-free after startup; the LUT
/// itself is `Clone` but every voice needs only a borrowed view, so we
/// just keep one instance.
pub static MODAL_LUT: OnceLock<ModalLut> = OnceLock::new();

/// Errors produced by `ModalLut::from_json_path`.
#[derive(Debug)]
pub enum ModalLutError {
    Io(std::io::Error),
    Parse(serde_json::Error),
    SchemaVersion(u32),
    Empty,
}

impl std::fmt::Display for ModalLutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "modal LUT I/O error: {e}"),
            Self::Parse(e) => write!(f, "modal LUT parse error: {e}"),
            Self::SchemaVersion(v) => write!(f, "modal LUT unsupported schema_version: {v}"),
            Self::Empty => write!(f, "modal LUT is empty (no per-note entries)"),
        }
    }
}

impl std::error::Error for ModalLutError {}

impl From<std::io::Error> for ModalLutError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for ModalLutError {
    fn from(e: serde_json::Error) -> Self {
        Self::Parse(e)
    }
}

impl ModalLut {
    /// Load a modal LUT from a JSON file produced by `bin/build_modal_lut`.
    /// On-disk schema:
    /// ```json
    /// {
    ///   "schema_version": 1,
    ///   "lut": [
    ///     {
    ///       "midi_note": 60,
    ///       "f0_hz": 261.63,
    ///       "modes": [{ "freq_hz": 261.6, "t60_sec": 18.14, "init_db": -2.7 }, ...],
    ///       "attack": { "time_to_peak_s": 0.039, "peak_db": -7.62, "post_peak_slope_db_s": -10.37 }
    ///     }
    ///   ]
    /// }
    /// ```
    /// `init_db` is converted to linear `init_amp = 10^(init_db/20)` here
    /// so the runtime `Mode` shape stays unified with the rest of the
    /// modal pipeline (`Mode::init_amp` is always linear).
    pub fn from_json_path(path: &Path) -> Result<Self, ModalLutError> {
        let raw = std::fs::read(path)?;
        let json: ModalLutJson = serde_json::from_slice(&raw)?;
        if json.schema_version != 1 {
            return Err(ModalLutError::SchemaVersion(json.schema_version));
        }
        if json.lut.is_empty() {
            return Err(ModalLutError::Empty);
        }
        let entries: Vec<ModalLutEntry> = json
            .lut
            .into_iter()
            .map(|e| {
                let modes: Vec<Mode> = e
                    .modes
                    .iter()
                    .map(|m| Mode {
                        freq_hz: m.freq_hz,
                        t60_sec: m.t60_sec,
                        init_amp: 10f32.powf(m.init_db / 20.0),
                    })
                    .collect();
                let attack = e.attack.unwrap_or_default();
                ModalLutEntry {
                    midi_note: e.midi_note,
                    f0_hz: e.f0_hz,
                    modes,
                    time_to_peak_s: attack.time_to_peak_s,
                    peak_db: attack.peak_db,
                    post_peak_slope_db_s: attack.post_peak_slope_db_s,
                }
            })
            .collect();
        Ok(Self { entries })
    }

    /// Hardcoded C4 fallback LUT used when no JSON file is available.
    /// Mirrors `ANALYSE_LUT_C4` from `bin/render_modal.rs`: the SFZ
    /// Salamander Grand V3 C4 reference's first 8 partials as measured
    /// by the existing `analyse` binary. Sufficient to keep
    /// `Engine::PianoModal` audible at MIDI 60 in dev/test runs that
    /// haven't built the per-note LUT yet.
    pub fn fallback_c4() -> Self {
        const ANALYSE_LUT_C4: [(f32, f32, f32); 8] = [
            (261.6, 18.14, -2.7),
            (523.1, 11.21, 0.0),
            (785.6, 9.58, -18.5),
            (1048.8, 7.38, -18.4),
            (1311.7, 6.98, -14.3),
            (1577.3, 7.93, -26.1),
            (1843.7, 8.98, -17.4),
            (2111.5, 8.66, -32.9),
        ];
        let modes: Vec<Mode> = ANALYSE_LUT_C4
            .iter()
            .map(|&(f, t60, db)| Mode {
                freq_hz: f,
                t60_sec: t60,
                init_amp: 10f32.powf(db / 20.0),
            })
            .collect();
        Self {
            entries: vec![ModalLutEntry {
                midi_note: 60,
                f0_hz: 261.63,
                modes,
                time_to_peak_s: 0.039,
                peak_db: -7.62,
                post_peak_slope_db_s: -10.37,
            }],
        }
    }

    /// Auto-load helper used by `main` and `bin/bench`. Tries
    /// `--modal-lut PATH` (if `Some`); otherwise tries the conventional
    /// `bench-out/REF/sfz_salamander_multi/modal_lut.json`. If neither
    /// loads cleanly, falls back to the hardcoded C4 entry so
    /// `Engine::PianoModal` still produces sound at MIDI 60.
    ///
    /// Returns the source description as the second tuple element so the
    /// caller can log which path was actually used (real JSON vs C4
    /// fallback).
    pub fn auto_load(explicit: Option<&Path>) -> (Self, String) {
        let default_path = Path::new("bench-out/REF/sfz_salamander_multi/modal_lut.json");
        let candidate = explicit.unwrap_or(default_path);
        match Self::from_json_path(candidate) {
            Ok(lut) => {
                let n = lut.entries.len();
                (lut, format!("{} ({n} entries)", candidate.display()))
            }
            Err(e) => (
                Self::fallback_c4(),
                format!(
                    "hardcoded C4 fallback (loading {} failed: {e})",
                    candidate.display()
                ),
            ),
        }
    }

    /// Find the entry whose `midi_note` is closest to `midi_note`.
    /// Ties broken by lower midi_note (stable: the iterator returns the
    /// first minimum). Panics if `entries` is empty — the auto-load path
    /// guarantees at least the hardcoded C4 entry is present.
    pub fn nearest_entry(&self, midi_note: u8) -> &ModalLutEntry {
        self.entries
            .iter()
            .min_by_key(|e| (e.midi_note as i32 - midi_note as i32).unsigned_abs())
            .expect("ModalLut::nearest_entry called on empty LUT")
    }
}

impl VoiceImpl for ModalPianoVoice {
    fn render_add(&mut self, buf: &mut [f32]) {
        // Lazy damper engage: cheaper than wrapping every step in a
        // branch, and we can do it once at the top of each render pass.
        if self.damper_pending {
            for r in &mut self.resonators {
                r.engage_damper(self.damper_t60_sec);
            }
            self.damper_pending = false;
        }
        for sample in buf.iter_mut() {
            // Pull next excitation sample (or 0 once the impulse runs
            // out — the resonators carry the sound from there).
            let x = if self.excitation_idx < self.excitation.len() {
                let v = self.excitation[self.excitation_idx];
                self.excitation_idx += 1;
                v
            } else {
                0.0
            };
            let mut sum = 0.0_f32;
            for r in &mut self.resonators {
                sum += r.step(x);
            }
            let env = self.release.step();
            *sample += sum * env;
        }
    }
    fn trigger_release(&mut self) {
        // Default behaviour: fade the output via ReleaseEnvelope (~0.3 s
        // multiplicative) AND engage the damper on the next render pass
        // so the resonators themselves stop ringing rather than just
        // getting quieter.
        if let Some(env) = self.release_env_mut() {
            env.trigger();
        }
        self.damper_pending = true;
    }
    fn release_env(&self) -> Option<&ReleaseEnvelope> {
        Some(&self.release)
    }
    fn release_env_mut(&mut self) -> Option<&mut ReleaseEnvelope> {
        Some(&mut self.release)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 44100.0;

    fn render(voice: &mut ModalPianoVoice, n: usize) -> Vec<f32> {
        let mut buf = vec![0.0_f32; n];
        voice.render_add(&mut buf);
        buf
    }

    /// Single mode at 440 Hz, T60 = 1.0 s, hit by a default impulse.
    /// Output should peak in the first ~10 ms then decay; after 1 s
    /// the RMS should be < -55 dB of the peak.
    #[test]
    fn single_mode_decays_to_t60() {
        let modes = [Mode {
            freq_hz: 440.0,
            t60_sec: 1.0,
            init_amp: 1.0,
        }];
        let mut v = ModalPianoVoice::new_default_excitation(SR, &modes, 100);
        let buf = render(&mut v, (SR * 1.5) as usize);

        // Peak in first 50 ms. With a single-sample delta excitation the
        // resonator's transient response peaks at b0 = 1 - r² which for
        // T60 = 1 s, sr = 44100 is ~3e-4 — small but resolvable.
        let early_peak = buf[..((SR * 0.05) as usize)]
            .iter()
            .copied()
            .fold(0.0_f32, |a, b| a.max(b.abs()));
        assert!(
            early_peak > 1e-4,
            "early peak too small ({}); resonator not excited",
            early_peak
        );

        // RMS over [1.0, 1.5] s: should be ~ -60 dB of early peak (T60 =
        // 1.0 s by construction). Allow generous slack: -45 dB or
        // better.
        let tail_start = (SR * 1.0) as usize;
        let tail = &buf[tail_start..];
        let tail_rms = (tail.iter().map(|&s| s * s).sum::<f32>() / tail.len() as f32).sqrt();
        let ratio_db = 20.0 * (tail_rms / early_peak).max(1e-9).log10();
        assert!(
            ratio_db < -45.0,
            "tail RMS {ratio_db:.1} dB > -45 dB → not decaying as fast as T60"
        );
    }

    /// Multiple modes at harmonic ratios, single hammer impulse: each
    /// must contribute energy at its own frequency. We spot-check by
    /// looking at the initial 200 ms RMS — three modes with init_amp
    /// 1.0/0.5/0.25 should give a roughly 1.0 + 0.5 + 0.25 ≈ 1.75x
    /// dynamic range vs a single 1.0 mode (rough sanity, not exact).
    #[test]
    fn multi_mode_renders() {
        let modes = vec![
            Mode {
                freq_hz: 261.63,
                t60_sec: 5.0,
                init_amp: 1.0,
            },
            Mode {
                freq_hz: 523.25,
                t60_sec: 4.0,
                init_amp: 0.5,
            },
            Mode {
                freq_hz: 784.88,
                t60_sec: 3.0,
                init_amp: 0.25,
            },
        ];
        let mut v = ModalPianoVoice::new_default_excitation(SR, &modes, 100);
        let buf = render(&mut v, 8192);
        let peak = buf.iter().copied().fold(0.0_f32, |a, b| a.max(b.abs()));
        // Delta excitation peak ≈ velocity_amp · b0 ≈ 0.78 · 1e-4 ~ 1e-4.
        // Loose lower bound that catches "no signal" but doesn't pin exact
        // amplitude (which depends on resonator b0 normalisation).
        assert!(peak > 5e-4, "multi-mode peak too small: {peak}");
        // No NaN / Inf.
        for &s in &buf {
            assert!(s.is_finite(), "non-finite sample in multi-mode render");
        }
    }

    #[test]
    fn release_lifecycle() {
        let modes = [Mode {
            freq_hz: 261.63,
            t60_sec: 2.0,
            init_amp: 1.0,
        }];
        let mut v = ModalPianoVoice::new_default_excitation(SR, &modes, 100);
        assert!(!v.is_releasing());
        v.trigger_release();
        assert!(v.is_releasing());
        let _ = render(&mut v, (SR * 0.7) as usize);
        assert!(v.is_done());
    }

    #[test]
    fn empty_modes_silent_after_excitation() {
        let modes: Vec<Mode> = Vec::new();
        let mut v = ModalPianoVoice::new_default_excitation(SR, &modes, 100);
        let buf = render(&mut v, 4096);
        // No resonators → output is always zero (Σ over empty bank).
        for &s in &buf {
            assert_eq!(s, 0.0, "empty modal bank produced non-zero sample");
        }
    }

    // -- ModalLut / from_lut path -----------------------------------------

    fn one_entry_lut(midi_note: u8, f0_hz: f32, modes: Vec<Mode>) -> ModalLut {
        ModalLut {
            entries: vec![ModalLutEntry {
                midi_note,
                f0_hz,
                modes,
                time_to_peak_s: 0.0,
                peak_db: 0.0,
                post_peak_slope_db_s: 0.0,
            }],
        }
    }

    /// Exact-note hit: `from_lut` for the LUT's own `midi_note` should
    /// keep mode frequencies on-pitch (modulo the ±0.7 cents 3-string
    /// detune `with_modes` injects). Each LUT mode expands into 3
    /// resonators whose centre frequencies bracket the original.
    #[test]
    fn from_lut_exact_match_uses_lut_directly() {
        let lut = one_entry_lut(
            60,
            261.63,
            vec![
                Mode {
                    freq_hz: 261.6,
                    t60_sec: 1.0,
                    init_amp: 1.0,
                },
                Mode {
                    freq_hz: 523.1,
                    t60_sec: 0.8,
                    init_amp: 0.5,
                },
            ],
        );
        let v = ModalPianoVoice::from_lut(&lut, SR, 60, 100);
        // 2 modes × 3 detune sub-modes = 6 resonators, frequency-ordered
        // around their centres. Recover each centre frequency from the
        // resonator's `cos_omega` and compare against the LUT.
        assert_eq!(v.resonators.len(), 6);
        let recovered: Vec<f32> = v
            .resonators
            .iter()
            .map(|r| r.cos_omega.acos() * SR / (2.0 * std::f32::consts::PI))
            .collect();
        // Sub-modes for f1 = 261.6: centre and ±0.7 cents.
        // Centre band straddles 261.6.
        let f1_band = &recovered[0..3];
        let f1_min = f1_band.iter().copied().fold(f32::MAX, f32::min);
        let f1_max = f1_band.iter().copied().fold(f32::MIN, f32::max);
        assert!(
            f1_min < 261.6 && f1_max > 261.6,
            "f1 sub-modes don't bracket: {f1_band:?}"
        );
        // ±0.7 cents at 261.6 Hz is ±0.106 Hz; allow generous slack.
        assert!(
            (f1_min - 261.6).abs() < 0.5,
            "f1 lower flank far off: {f1_min}"
        );
        let f2_band = &recovered[3..6];
        let f2_min = f2_band.iter().copied().fold(f32::MAX, f32::min);
        let f2_max = f2_band.iter().copied().fold(f32::MIN, f32::max);
        assert!(
            f2_min < 523.1 && f2_max > 523.1,
            "f2 sub-modes don't bracket: {f2_band:?}"
        );
    }

    /// LUT only has note 60 (C4). Calling `from_lut` for note 72 (C5)
    /// should scale every mode frequency by 2.0 (one octave up).
    #[test]
    fn from_lut_nearest_octave_scales_correctly() {
        let lut = one_entry_lut(
            60,
            261.63,
            vec![
                Mode {
                    freq_hz: 261.63,
                    t60_sec: 1.0,
                    init_amp: 1.0,
                },
                Mode {
                    freq_hz: 523.25,
                    t60_sec: 0.8,
                    init_amp: 0.5,
                },
            ],
        );
        let v = ModalPianoVoice::from_lut(&lut, SR, 72, 100);
        assert_eq!(v.resonators.len(), 6);
        // Recover sub-mode centres and check they scale by 2.0 (with
        // ratio = 440·2^((72-69)/12) / 261.63 = 523.25 / 261.63 ≈ 2.0).
        let recovered: Vec<f32> = v
            .resonators
            .iter()
            .map(|r| r.cos_omega.acos() * SR / (2.0 * std::f32::consts::PI))
            .collect();
        // f1 centre band should sit around 523.25 Hz (261.63 × 2).
        let f1_centre = recovered[1];
        assert!(
            (f1_centre - 523.25).abs() < 1.0,
            "f1 octave scale wrong: got {f1_centre}, expected ~523.25"
        );
        // f2 centre band should sit around 1046.5 Hz (523.25 × 2).
        let f2_centre = recovered[4];
        assert!(
            (f2_centre - 1046.5).abs() < 2.0,
            "f2 octave scale wrong: got {f2_centre}, expected ~1046.5"
        );
    }

    /// Sentinel T60 = -1 (extractor failure marker) → resonator should
    /// be built with the 12 s fallback, not blow up with a tiny pole
    /// radius and produce silence / NaN.
    #[test]
    fn from_lut_handles_t60_sentinel() {
        let lut = one_entry_lut(
            60,
            261.63,
            vec![Mode {
                freq_hz: 261.63,
                t60_sec: -1.0,
                init_amp: 1.0,
            }],
        );
        let v = ModalPianoVoice::from_lut(&lut, SR, 60, 100);
        assert_eq!(v.resonators.len(), 3);
        // Pole radius for 12 s T60 at 44100 Hz: r ≈ 0.99987.
        // r² should be > 0.999.
        for r in &v.resonators {
            assert!(
                r.a2 > 0.999,
                "sentinel T60 not substituted with 12 s fallback: a2={}",
                r.a2
            );
        }
    }

    /// Auto-load with no real JSON path falls back to the hardcoded C4
    /// entry; the resulting LUT must be non-empty so `nearest_entry`
    /// can't panic.
    #[test]
    fn modal_lut_fallback_c4_has_one_entry() {
        let lut = ModalLut::fallback_c4();
        assert_eq!(lut.entries.len(), 1);
        assert_eq!(lut.entries[0].midi_note, 60);
        assert_eq!(lut.entries[0].modes.len(), 8);
    }

    #[test]
    fn high_t60_pole_inside_unit_circle() {
        // Sanity: very long T60 → r approaches 1 but never reaches it.
        let mode = Mode {
            freq_hz: 440.0,
            t60_sec: 100.0,
            init_amp: 1.0,
        };
        let res = ModalResonator::new(mode, SR);
        let r2 = res.a2;
        assert!(r2 > 0.99 && r2 < 1.0, "pole radius² out of range: {r2}");
    }
}
