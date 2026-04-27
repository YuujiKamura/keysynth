//! Piano-leaning DWS engine.
//!
//! Differences from `ks-rich`:
//!   - WIDER, ms-scale hammer contact (20-200 samples vs 3-19)
//!     -> contact pulse closer to real piano (hammer felt compresses over ms)
//!   - ASYMMETRIC hammer profile: linear rise + exponential decay
//!     -> mimics felt strike physics (sharp impact, slow release)
//!   - FREQUENCY-DEPENDENT decay: high notes die faster
//!     -> matches real piano where treble strings have less mass / faster damp
//!   - Stronger allpass dispersion -> more inharmonicity (piano stretch tuning)

use super::super::synth::{
    piano_hammer_excitation, piano_hammer_width, KsString, ReleaseEnvelope, VoiceImpl,
};
use super::hammer_stulov::{self, HammerParams};
use super::longitudinal::LongitudinalString;

// ---------------------------------------------------------------------------
// Piano preset container (issue #2 transition: unify Piano/PianoThick/PianoLite
// into one parameterised voice). Adding a new preset = one const-struct entry.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoundboardKind {
    /// Full 16-mode bank used by the original `Piano` preset.
    ConcertGrand,
    /// 12-mode lite bank — drops the high-Q upper-band modes that were
    /// ringing through the bridge feedback loop in the full preset.
    Lite,
}

#[derive(Clone, Copy, Debug)]
pub struct PianoPreset {
    /// Number of detuned KS strings ganged in unison.
    pub string_count: usize,
    /// Half-spread of the detune in cents. Strings are placed symmetrically
    /// around 1.0 (centre). For odd counts the centre string is at 0 cents.
    pub detune_cents_half_spread: f32,
    /// Which soundboard mode bank to use.
    pub soundboard: SoundboardKind,
    /// Per-string KS in-loop decay at low freq (~110 Hz).
    pub decay_base: f32,
    /// Decay slope at 2 kHz. The per-string decay multiplier is
    /// `decay_base - decay_slope * (freq/2000).clamp(0,1)`.
    pub decay_slope: f32,
    /// Bridge feedback coefficient injected into each string per sample.
    /// Already pre-divided by string_count for presets where total bridge
    /// energy is normalised; raw (un-divided) for presets that explicitly
    /// want stronger per-string coupling.
    pub coupling_per_string: f32,
    /// Dry string-sum gain in the final mix.
    pub dry_gain: f32,
    /// Wet soundboard-output gain in the final mix.
    pub wet_gain: f32,
    /// Longitudinal-mode gain in the final mix.
    pub long_gain: f32,
    /// Voice release time (seconds) handed to the shared ReleaseEnvelope.
    pub release_sec: f32,
    /// Use Stulov non-linear hammer model instead of linear-rise + exp-decay.
    pub use_stulov: bool,
    /// Parameters for the Stulov hammer model.
    pub hammer_params: HammerParams,
}

impl PianoPreset {
    /// Original `Piano` engine: 7 strings, full ConcertGrand 16-mode body,
    /// dry/wet 0.5/0.35. Bridge feedback normalised across the 7 strings.
    pub const PIANO: Self = Self {
        string_count: 7,
        detune_cents_half_spread: 3.0,
        soundboard: SoundboardKind::ConcertGrand,
        decay_base: 0.9985,
        decay_slope: 0.0035,
        coupling_per_string: 0.012 / 7.0,
        dry_gain: 0.5,
        wet_gain: 0.35,
        long_gain: 0.0,
        release_sec: 0.300,
        use_stulov: false,
        hammer_params: HammerParams {
            mass_kg: 8.7e-3,
            k_stiffness: 4.5e9,
            p_exponent: 2.5,
            eps_hysteresis: 1e-4,
        },
    };
    /// `PianoThick`: 7 strings + 12-mode lite body so the high-Q upper-band
    /// modes don't ring through the bridge loop. Decay base lifted to keep
    /// the fundamental sustaining (closer to the SFZ Salamander C4 ref) and
    /// dry-dominant 0.7/0.20 mix to suppress the metallic body sustain.
    pub const PIANO_THICK: Self = Self {
        string_count: 7,
        detune_cents_half_spread: 3.0,
        soundboard: SoundboardKind::Lite,
        decay_base: 0.9990,
        decay_slope: 0.0050,
        coupling_per_string: 0.015 / 7.0,
        dry_gain: 0.7,
        wet_gain: 0.20,
        long_gain: 0.0,
        release_sec: 0.300,
        use_stulov: false,
        hammer_params: HammerParams {
            mass_kg: 8.7e-3,
            k_stiffness: 4.5e9,
            p_exponent: 2.5,
            eps_hysteresis: 1e-4,
        },
    };
    /// `PianoLite`: 3 strings + 12-mode lite body. v4 tuning lands h1 T60
    /// within 8% of the SFZ Salamander C4 reference. Coupling is NOT
    /// pre-divided here — the original PianoLite ran 0.015/string × 3
    /// strings = 0.045 total bridge energy by design.
    pub const PIANO_LITE: Self = Self {
        string_count: 3,
        detune_cents_half_spread: 1.5,
        soundboard: SoundboardKind::Lite,
        decay_base: 0.9980,
        decay_slope: 0.0025,
        coupling_per_string: 0.015,
        dry_gain: 0.5,
        wet_gain: 0.35,
        long_gain: 0.0,
        release_sec: 0.300,
        use_stulov: false,
        hammer_params: HammerParams {
            mass_kg: 8.7e-3,
            k_stiffness: 4.5e9,
            p_exponent: 2.5,
            eps_hysteresis: 1e-4,
        },
    };
    /// `PianoLong`: T1.2 longitudinal coupling preset. 7 strings, full
    /// body, plus longitudinal phantom partials at sum-frequencies.
    pub const PIANO_LONG: Self = Self {
        string_count: 7,
        detune_cents_half_spread: 3.0,
        soundboard: SoundboardKind::ConcertGrand,
        decay_base: 0.9985,
        decay_slope: 0.0035,
        coupling_per_string: 0.012 / 7.0,
        dry_gain: 0.5,
        wet_gain: 0.35,
        long_gain: 0.15,
        release_sec: 0.300,
        use_stulov: false,
        hammer_params: HammerParams {
            mass_kg: 8.7e-3,
            k_stiffness: 4.5e9,
            p_exponent: 2.5,
            eps_hysteresis: 1e-4,
        },
    };
}

/// Build the symmetric per-string detune ratios for a preset's string count
/// and half-spread. Reproduces the explicit arrays the pre-unification
/// `PianoVoice` / `PianoVoiceThick` (7-string, ±3 c) and `PianoVoiceLite`
/// (3-string, ±1.5 c) used.
pub(crate) fn piano_detunes(count: usize, half_cents: f32) -> Vec<f32> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![1.0];
    }
    let cents_to_ratio = |c: f32| 2.0_f32.powf(c / 1200.0);
    let mut v = Vec::with_capacity(count);
    if count % 2 == 1 {
        let half = (count - 1) / 2;
        let step = half_cents / (half as f32);
        for i in 0..count {
            let offset_cents = (i as f32 - half as f32) * step;
            v.push(if offset_cents == 0.0 {
                1.0
            } else {
                cents_to_ratio(offset_cents)
            });
        }
    } else {
        let step = (2.0 * half_cents) / ((count - 1) as f32);
        for i in 0..count {
            let offset_cents = -half_cents + (i as f32) * step;
            v.push(cents_to_ratio(offset_cents));
        }
    }
    v
}

pub struct PianoVoice {
    /// Detuned KS strings (count comes from `PianoPreset::string_count`).
    /// `Vec` rather than a fixed array so a single struct serves all
    /// presets — this is the issue #2 transition fix.
    strings: Vec<KsString>,
    /// Optional longitudinal mode per transverse string.
    longitudinal: Vec<LongitudinalString>,
    release: ReleaseEnvelope,
    /// Modal soundboard resonator bank. Receives the summed string output
    /// every sample and feeds back into the strings via `coupling_per_string`
    /// for physical bridge coupling (Bank/Lehtonen DWS).
    soundboard: crate::soundboard::Soundboard,
    coupling_per_string: f32,
    dry_gain: f32,
    wet_gain: f32,
    long_gain: f32,
    /// 1.0 / string_count — applied to the string sum so the in-phase
    /// attack peak stays bounded regardless of preset.
    string_norm: f32,
}

impl PianoVoice {
    /// Construct a piano voice using `PianoPreset::PIANO` (the original
    /// 7-string ConcertGrand defaults). Kept as `new` for backward
    /// compatibility with existing tests.
    pub fn new(sr: f32, freq: f32, velocity: u8) -> Self {
        Self::with_preset(PianoPreset::PIANO, sr, freq, velocity)
    }

    /// Construct a piano voice using an arbitrary preset. This is the
    /// canonical entry point — `Engine::Piano`/`PianoThick`/`PianoLite`
    /// all dispatch here with different presets.
    pub fn with_preset(mut preset: PianoPreset, sr: f32, freq: f32, velocity: u8) -> Self {
        if std::env::var("KS_PIANO_LONG")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
        {
            preset = PianoPreset::PIANO_LONG;
        }
        let amp = (velocity.max(1) as f32) / 127.0;
        let detunes = piano_detunes(preset.string_count, preset.detune_cents_half_spread);
        // Lower stiffness — measured B at C4 ~3e-4 from SF2 reference; the
        // shallow fixed range puts B near the reference.
        let ap = (0.10 + (freq / 4000.0).min(0.05)).min(0.18);
        let decay_for = |f: f32| -> f32 {
            let high = (f / 2000.0).clamp(0.0, 1.0);
            preset.decay_base - preset.decay_slope * high
        };

        let mk = |freq_string: f32| -> KsString {
            let (n, _frac) = KsString::delay_length_compensated(sr, freq_string, ap);
            let force_stulov = std::env::var("KS_STULOV")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            let buf = if preset.use_stulov || force_stulov {
                let v_mps = 0.5 + (velocity as f32 / 127.0) * 4.5;
                hammer_stulov::stulov_pulse(v_mps, sr, n, &preset.hammer_params)
            } else {
                let hammer_w = piano_hammer_width(velocity);
                piano_hammer_excitation(n, hammer_w, amp)
            };
            KsString::with_buf(sr, freq_string, buf, decay_for(freq_string), ap)
                .with_attack_lpf(sr, 0.0, 0.97, 0.97)
        };
        let strings: Vec<KsString> = detunes.iter().map(|d| mk(freq * *d)).collect();

        // Longitudinal strings (same count and tuning as transverse)
        let mut longitudinal = Vec::with_capacity(strings.len());
        if preset.long_gain > 0.0 {
            for d in &detunes {
                let f_s = freq * d;
                // Steel piano string longitudinal speed c_long ≈ 14× c_trans
                longitudinal.push(LongitudinalString::new(sr, f_s, 14.0, decay_for(f_s)));
            }
        }

        let soundboard = match preset.soundboard {
            SoundboardKind::ConcertGrand => crate::soundboard::Soundboard::new_concert_grand(sr),
            SoundboardKind::Lite => crate::soundboard::Soundboard::new_concert_grand_lite(sr),
        };
        let string_norm = if preset.string_count == 0 {
            1.0
        } else {
            1.0 / (preset.string_count as f32)
        };
        Self {
            strings,
            longitudinal,
            release: ReleaseEnvelope::new(preset.release_sec, sr),
            soundboard,
            coupling_per_string: preset.coupling_per_string,
            dry_gain: preset.dry_gain,
            wet_gain: preset.wet_gain,
            long_gain: preset.long_gain,
            string_norm,
        }
    }
}

impl VoiceImpl for PianoVoice {
    fn render_add(&mut self, buf: &mut [f32]) {
        let dry_gain = self.dry_gain;
        let wet_gain = self.wet_gain;
        let long_gain = self.long_gain;
        let string_norm = self.string_norm;
        for sample in buf.iter_mut() {
            // 1. Inject the previous-sample soundboard output back into
            //    every string via the bridge. One-sample delay keeps the
            //    feedback loop strictly causal (no algebraic loop).
            let fb = self.soundboard.last_output() * self.coupling_per_string;
            if fb != 0.0 {
                for s in &mut self.strings {
                    s.inject_feedback(fb);
                }
            }
            // 2. Step the strings (consumes the feedback we just queued).
            let mut s_sum = 0.0_f32;
            let mut s_long_total = 0.0_f32;
            for i in 0..self.strings.len() {
                let s_trans = self.strings[i].step();
                s_sum += s_trans;
                if !self.longitudinal.is_empty() {
                    s_long_total += self.longitudinal[i].step(s_trans);
                }
            }
            let s_avg = s_sum * string_norm;
            // 3. Drive the soundboard with the averaged string output.
            let board_out = self.soundboard.process(s_avg);
            // 4. Mix dry strings + wet soundboard + longitudinal into the audio bus.
            let env = self.release.step();
            *sample += (s_avg * dry_gain + board_out * wet_gain + s_long_total * long_gain) * env;
        }
    }
    fn release_env(&self) -> Option<&ReleaseEnvelope> {
        Some(&self.release)
    }
    fn release_env_mut(&mut self) -> Option<&mut ReleaseEnvelope> {
        Some(&mut self.release)
    }
}
