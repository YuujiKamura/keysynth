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
    piano_hammer_excitation, piano_hammer_width, KsString, ReleaseEnvelope, SmallFir, VoiceImpl,
};
use super::bridge_admittance::make_bridge_admittance_fir;
use super::hammer_stulov::{self, HammerParams};
use super::longitudinal::LongitudinalString;
use super::mistuning_table::{curve_for_midi, MistuneCurve, MIN_HALF_SPREAD_CENTS};
use super::string_inharmonicity::dispersion_allpass_coeff;

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

/// Per-string state needed to drive the Tier 2.1 time-varying mistuning.
///
/// At construction time each string gets a fixed integer delay-line length
/// `n_int`, plus a known `compensation` amount (the LPF + dispersion AP
/// phase delay reserved inside `delay_length_compensated`). The total
/// loop period is therefore `n_int + frac + compensation`, where only
/// `frac` is dynamically modulatable (`KsString::set_tune_frac`).
///
/// To shift the string by `cents` from its base frequency we compute the
/// new total period `sr / (base_freq * 2^(cents/1200))` and back out the
/// fractional component as `target_total - n_int - compensation`. Sub-
/// sample modulations stay inside `[0, 1)` and avoid buffer resizing;
/// for cent magnitudes large enough to push out of that range we clamp
/// (in practice the per-note half-spread stays under 5 c, which
/// corresponds to ~0.5 sample at C2).
struct StringTuningState {
    /// Static cent offset used at construction (sign defines the symmetric
    /// fan-out: `-half_spread .. +half_spread` across the unison).
    sign_cents: f32,
    /// Base period in samples (`sr / freq_string_at_attack`).
    base_total_delay: f32,
    /// Integer delay-line length picked at construction.
    n_int: usize,
    /// LPF + dispersion-AP phase delay reserved at construction.
    compensation: f32,
    /// Current (most-recent) fractional delay command — kept for diagnostics
    /// and to allow callers to read back the current effective tuning.
    current_frac: f32,
}

pub struct PianoVoice {
    /// Detuned KS strings (count comes from `PianoPreset::string_count`).
    /// `Vec` rather than a fixed array so a single struct serves all
    /// presets — this is the issue #2 transition fix.
    strings: Vec<KsString>,
    /// Per-string time-varying tuning state, parallel to `strings`.
    string_tuning: Vec<StringTuningState>,
    /// Optional longitudinal mode per transverse string.
    longitudinal: Vec<LongitudinalString>,
    release: ReleaseEnvelope,
    /// Modal soundboard resonator bank. Receives the summed string output
    /// every sample and feeds back into the strings via `coupling_per_string`
    /// for physical bridge coupling (Bank/Lehtonen DWS).
    soundboard: crate::soundboard::Soundboard,
    /// DC-gain scalar for the bridge round-trip. Identical to the legacy
    /// `coupling_per_string` field in pre-T2.2 builds. The FIR shapes the
    /// frequency response *around* this scalar; the FIR itself is
    /// DC-normalised so the audible coupling at low frequencies stays
    /// numerically identical to the pre-T2.2 baseline.
    coupling_per_string: f32,
    /// T2.2 frequency-dependent bridge admittance, soundboard → string
    /// direction. Filters the soundboard's previous-sample output before
    /// it enters each string's `inject_feedback`. Stateful — must NOT be
    /// `Clone`d into a "shared" instance across the two directions.
    bridge_fir_return: SmallFir<6>,
    /// T2.2 frequency-dependent bridge admittance, string → soundboard
    /// direction. Filters the averaged string drive before
    /// `Soundboard::process` consumes it. Separate state from
    /// `bridge_fir_return` so the two directions of the bridge round-trip
    /// each see their own delay line.
    bridge_fir_drive: SmallFir<6>,
    dry_gain: f32,
    wet_gain: f32,
    long_gain: f32,
    /// 1.0 / string_count — applied to the string sum so the in-phase
    /// attack peak stays bounded regardless of preset.
    string_norm: f32,
    /// Sample rate cached for the time-varying detune calculation.
    sr: f32,
    /// Sample counter since note-on. `t_since_attack = samples_since_attack / sr`.
    /// Drives `MistuneCurve::evaluate` for the half-spread amplitude envelope.
    samples_since_attack: u32,
    /// Pre-resolved curve for this voice's MIDI note. Cached so the per-
    /// sample inner loop doesn't repeat the table interpolation.
    curve: MistuneCurve,
    /// `1` = update fractional-delay AP every sample, otherwise update
    /// every Nth sample. The mistuning envelope changes on a tau ≥ 0.3 s
    /// time scale, so a 32-sample update period (~0.7 ms at 44.1 kHz)
    /// is well below any audible LFO step. Saves ~3 % CPU on the hot path
    /// and keeps the existing `step()` cost unchanged for non-piano voices.
    detune_update_period: u32,
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
        let midi_note = freq_to_midi_note(freq);
        let decay_for = |f: f32| -> f32 {
            let high = (f / 2000.0).clamp(0.0, 1.0);
            preset.decay_base - preset.decay_slope * high
        };

        // We need to compute the LPF+dispersion compensation per-string so
        // that the time-varying detune can solve `target_frac = total_delay
        // - n_int - compensation`. The compensation matches the formula
        // baked into `KsString::delay_length_compensated`.
        let compensation_for = |ap_coef: f32| -> f32 {
            let lpf_delay = 0.5_f32;
            let disp_delay = (1.0 - ap_coef) / (1.0 + ap_coef);
            lpf_delay + disp_delay + 0.5
        };

        let mk = |freq_string: f32| -> (KsString, f32, usize, f32) {
            let partial_count = ((sr * 0.5 / freq_string.max(1.0)).floor() as usize).max(1);
            let ap = dispersion_allpass_coeff(midi_note, partial_count);
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
            let s = KsString::with_buf(sr, freq_string, buf, decay_for(freq_string), ap)
                .with_attack_lpf(sr, 0.0, 0.97, 0.97);
            let total = sr / freq_string.max(1.0);
            (s, total, n, compensation_for(ap))
        };
        let mut strings: Vec<KsString> = Vec::with_capacity(detunes.len());
        let mut string_tuning: Vec<StringTuningState> = Vec::with_capacity(detunes.len());
        // Static cent fan-out the preset would have used pre-T2.1 — kept
        // as the per-string `unit_sign` distribution. We do NOT bake the
        // static detune into each string's construction-time delay-line
        // length anymore; instead all strings start at the centre `freq`
        // and the entire fan-out is applied dynamically via
        // `update_dynamic_detune`. This matches the brief's physics:
        // strings start beating in phase at the attack, then drift apart
        // as the per-string AP coefficients separate over `tau`.
        let static_cents: Vec<f32> = detunes.iter().map(|d| 1200.0 * d.log2()).collect();
        let max_static_abs = static_cents
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0_f32, f32::max)
            .max(1e-6);
        for sc in &static_cents {
            // All strings at centre frequency at construction.
            let (s, total, n_int, comp) = mk(freq);
            strings.push(s);
            // `unit_sign` in `[-1, +1]` defines where this string sits in
            // the symmetric fan-out. Multiplied by
            // `MistuneCurve::evaluate(t)` per-sample to give the actual
            // cent offset.
            let unit_sign = sc / max_static_abs;
            string_tuning.push(StringTuningState {
                sign_cents: unit_sign,
                base_total_delay: total,
                n_int,
                compensation: comp,
                current_frac: total - n_int as f32 - comp,
            });
        }

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
        let curve = curve_for_midi(midi_note);
        Self {
            strings,
            string_tuning,
            longitudinal,
            release: ReleaseEnvelope::new(preset.release_sec, sr),
            soundboard,
            coupling_per_string: preset.coupling_per_string,
            bridge_fir_return: make_bridge_admittance_fir(),
            bridge_fir_drive: make_bridge_admittance_fir(),
            dry_gain: preset.dry_gain,
            wet_gain: preset.wet_gain,
            long_gain: preset.long_gain,
            string_norm,
            sr,
            samples_since_attack: 0,
            curve,
            // 32-sample update — see field doc. Single-string presets skip
            // the update entirely (no inter-string beating to drive).
            detune_update_period: if preset.string_count <= 1 { 0 } else { 32 },
        }
    }

    /// Test-only hook: replace both bridge-admittance FIR coefficient
    /// vectors so the integration suite can A/B against the legacy flat-
    /// scalar coupling (`[1.0, 0, 0, 0, 0, 0]`) without `git stash`-ing
    /// the whole T2.2 patch. NOT part of the public DSP API — voice
    /// presets remain the canonical way to vary bridge behaviour.
    #[doc(hidden)]
    pub fn override_bridge_fir_for_testing(&mut self, coefs: [f32; 6]) {
        self.bridge_fir_return = SmallFir::new(coefs);
        self.bridge_fir_drive = SmallFir::new(coefs);
    }

    /// Compute and apply the per-string fractional-delay update for the
    /// current sample-since-attack count. Cheap fast path when called
    /// frequently — the trig is just `exp2`/`floor`/`max`.
    #[inline]
    fn update_dynamic_detune(&mut self) {
        if self.detune_update_period == 0 || self.string_tuning.is_empty() {
            return;
        }
        let t_sec = self.samples_since_attack as f32 / self.sr.max(1.0);
        // Half-spread amplitude in cents at the current time. Floored to
        // MIN_HALF_SPREAD_CENTS so the time-varying detune always produces
        // a measurable inter-string offset across the full keyboard.
        let half = self.curve.evaluate(t_sec).max(MIN_HALF_SPREAD_CENTS);
        for (state, string) in self.string_tuning.iter_mut().zip(self.strings.iter_mut()) {
            // Per-string cent offset = unit_sign * dynamic_half_spread.
            // `unit_sign` is the construction-time fan-out scaled to ±1.
            let cents = state.sign_cents * half;
            let ratio = (cents / 1200.0).exp2();
            let target_total = state.base_total_delay / ratio;
            let target_frac = target_total - state.n_int as f32 - state.compensation;
            // Sub-sample regime: target_frac stays in [0, 1) for cents in
            // roughly ±10 c at any pitch; clamp anyway for safety.
            let frac = target_frac.clamp(0.0, 0.999);
            string.set_tune_frac(frac);
            state.current_frac = frac;
        }
    }
}

fn freq_to_midi_note(freq: f32) -> u8 {
    (((freq.max(1.0) / 440.0).log2() * 12.0) + 69.0)
        .round()
        .clamp(0.0, 127.0) as u8
}

impl VoiceImpl for PianoVoice {
    fn render_add(&mut self, buf: &mut [f32]) {
        let dry_gain = self.dry_gain;
        let wet_gain = self.wet_gain;
        let long_gain = self.long_gain;
        let string_norm = self.string_norm;
        let detune_period = self.detune_update_period;
        // Apply once at the start of every render block so the very first
        // sample after note-on already reflects `c0` rather than
        // construction-time static detune.
        if detune_period != 0 {
            self.update_dynamic_detune();
        }
        for sample in buf.iter_mut() {
            // 0. Time-varying mistuning: every `detune_period` samples,
            //    re-solve the per-string fractional-delay AP coefficient
            //    from the current `MistuneCurve::evaluate(t)`. This is the
            //    Tier 2.1 detune-beat hook — at construction time the
            //    strings sit at the static fan-out, then the half-spread
            //    decays from `c0` toward `c_inf` over `tau`.
            if detune_period != 0
                && (self.samples_since_attack % detune_period) == 0
                && self.samples_since_attack != 0
            {
                self.update_dynamic_detune();
            }
            // 1. Inject the previous-sample soundboard output back into
            //    every string via the bridge. One-sample delay keeps the
            //    feedback loop strictly causal (no algebraic loop).
            //
            //    T2.2 frequency-dependent admittance: the soundboard
            //    output runs through `bridge_fir_return` before being
            //    scaled by the per-preset DC coupling. The FIR is
            //    DC-normalised, so at low frequencies the injected
            //    feedback equals the pre-T2.2 scalar baseline; only the
            //    high-frequency component is attenuated to match the
            //    Suzuki/Giordano admittance rolloff.
            let board_prev = self.soundboard.last_output();
            let board_filtered = self.bridge_fir_return.process(board_prev);
            let fb = board_filtered * self.coupling_per_string;
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
            //    T2.2: the same admittance shape applies to the string →
            //    bridge direction. Mathematically the round-trip loop
            //    transfer function therefore picks up |H(ω)|^2 instead
            //    of |H(ω)|, which is exactly what bidirectional
            //    admittance physics predicts (the bridge attenuates
            //    energy moving in either direction).
            let s_avg_filtered = self.bridge_fir_drive.process(s_avg);
            let board_out = self.soundboard.process(s_avg_filtered);
            // 4. Mix dry strings + wet soundboard + longitudinal into the audio bus.
            let env = self.release.step();
            *sample += (s_avg * dry_gain + board_out * wet_gain + s_long_total * long_gain) * env;
            self.samples_since_attack = self.samples_since_attack.saturating_add(1);
        }
    }
    fn release_env(&self) -> Option<&ReleaseEnvelope> {
        Some(&self.release)
    }
    fn release_env_mut(&mut self) -> Option<&mut ReleaseEnvelope> {
        Some(&mut self.release)
    }
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;
    use crate::extract::decompose::decompose;
    use crate::extract::inharmonicity::fit_b;
    use crate::synth::midi_to_freq;
    use crate::voices::string_inharmonicity::b_coefficient_clamped_88key;

    const SR: f32 = 44_100.0;

    fn analysis_preset() -> PianoPreset {
        PianoPreset {
            string_count: 1,
            detune_cents_half_spread: 0.0,
            soundboard: SoundboardKind::Lite,
            decay_base: 0.9992,
            decay_slope: 0.0010,
            coupling_per_string: 0.0,
            dry_gain: 1.0,
            wet_gain: 0.0,
            long_gain: 0.0,
            release_sec: 0.300,
            use_stulov: false,
            hammer_params: HammerParams {
                mass_kg: 8.7e-3,
                k_stiffness: 4.5e9,
                p_exponent: 2.5,
                eps_hysteresis: 1e-4,
            },
        }
    }

    fn render_voice(preset: PianoPreset, note: u8, seconds: f32) -> Vec<f32> {
        let mut voice = PianoVoice::with_preset(preset, SR, midi_to_freq(note), 110);
        let n = (SR * seconds) as usize;
        let mut buf = vec![0.0_f32; n];
        voice.render_add(&mut buf);
        buf
    }

    fn target_partial_cents(note: u8, n: usize) -> f32 {
        let b = b_coefficient_clamped_88key(note);
        let nf = n as f32;
        600.0 * (((1.0 + b * nf * nf) / (1.0 + b)).log2())
    }

    /// Regression harness for the Fletcher-reference notes under the current
    /// single-stage KS topology. The one-pole loop allpass does not hit the
    /// published C2/C6 partial curves exactly, but the render path must stay
    /// decomposable, finite, and note-dependent once the per-note B mapping
    /// is wired in.
    #[test]
    fn piano_voice_inharmonicity_matches_published() {
        let preset = analysis_preset();
        let low_note = 36_u8;
        let high_note = 84_u8;
        let low_ap = dispersion_allpass_coeff(
            low_note,
            ((SR * 0.5 / midi_to_freq(low_note)).floor() as usize).max(1),
        );
        let high_ap = dispersion_allpass_coeff(
            high_note,
            ((SR * 0.5 / midi_to_freq(high_note)).floor() as usize).max(1),
        );
        assert!(
            low_ap > high_ap,
            "expected bass note to receive stronger dispersion coefficient: low={low_ap} high={high_ap}"
        );

        let mut metrics = Vec::new();
        for note in [low_note, high_note] {
            let f0 = midi_to_freq(note);
            let rendered = render_voice(preset, note, 2.0);
            let partials = decompose(&rendered, SR, f0, 16);
            assert!(
                partials.len() >= 10,
                "note {note} only produced {} partials in decomposition",
                partials.len()
            );
            let fit = fit_b(&partials);

            let f1 = partials
                .iter()
                .find(|p| p.n == 1)
                .map(|p| p.freq_hz)
                .unwrap_or(f0);

            let mut sum_abs_err = 0.0_f32;
            let mut sum_mag = 0.0_f32;
            let mut max_abs_err = 0.0_f32;
            let mut count = 0_u32;
            for p in partials.iter().filter(|p| p.n <= 16) {
                let measured = 1200.0 * (p.freq_hz / (f1 * p.n as f32)).log2();
                let target = target_partial_cents(note, p.n);
                let err = (measured - target).abs();
                sum_mag += measured;
                sum_abs_err += err;
                max_abs_err = max_abs_err.max(err);
                count += 1;
            }
            let mean_abs_err = sum_abs_err / count.max(1) as f32;
            let mean_mag = sum_mag / count.max(1) as f32;
            metrics.push((
                note,
                fit.b,
                fit.r_squared,
                fit.n_used,
                mean_abs_err,
                max_abs_err,
                mean_mag,
            ));
        }
        let low = metrics[0];
        let high = metrics[1];

        assert!(
            low.1.is_finite() && high.1.is_finite(),
            "non-finite fitted B metrics: {metrics:?}"
        );
        assert!(
            low.2 >= 0.60 && high.2 >= 0.90,
            "stretched-harmonic fit regressed: {metrics:?}"
        );
        assert!(
            low.3 >= 10 && high.3 >= 10,
            "expected at least 10 usable partials in both reference notes: {metrics:?}"
        );
        assert!(
            low.4 <= 45.0 && high.4 <= 20.0,
            "mean partial-offset error regressed too far: {metrics:?}"
        );
        assert!(
            low.5 <= 110.0 && high.5 <= 50.0,
            "max partial-offset error regressed too far: {metrics:?}"
        );
        assert!(
            low.6.abs() >= 1.0 && high.6.abs() >= 1.0,
            "expected measurable inharmonic deviation in both reference notes: {metrics:?}"
        );
    }
}
