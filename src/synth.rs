//! DSP engines, voices, ADSR, hammer/pluck excitations, and MIDI math.
//!
//! Extracted from `main.rs` so offline harnesses (`bench`, future
//! regression tests, SoundFont reference comparison) can reuse the same
//! voice implementations the live synth runs.
//!
//! Engines:
//!   - square : NES-style pulse wave with linear AR envelope
//!   - ks     : Karplus-Strong plucked string (single delay line + 2-tap
//!              lowpass), Phase 1 physical model

use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    /// NES-style pulse + linear AR envelope. Cheap reference tone.
    Square,
    /// Karplus-Strong: white-noise-excited delay loop with 2-tap lowpass.
    /// Phase-1 physical model. Sounds like a thin plucked string.
    Ks,
    /// Karplus-Strong + 3-string unison detune + 1-pole allpass dispersion
    /// in the feedback loop (stiffness). Closer to piano-ish sustain.
    KsRich,
    /// Subtractive: sawtooth -> state-variable lowpass with cutoff envelope
    /// + ADSR amp env. Classic 1980s analog-synth voice.
    Sub,
    /// 2-operator FM: sine carrier phase-modulated by sine modulator with
    /// its own ADSR controlling mod index. Default ratio gives bell/epiano.
    Fm,
    /// Piano-leaning physical model: wide asymmetric hammer (ms-scale
    /// contact), frequency-dependent decay (high notes die fast), 3-string
    /// unison + heavier stiffness allpass. Closer to piano than `ks-rich`.
    Piano,
    /// Koto/shamisen-leaning physical model: single string, sharp narrow
    /// plectrum injected near 1/4 of string length (pluck position effect),
    /// minimal stiffness, long sustain. The KS algorithm is fundamentally
    /// plucked-string physics, so this is the "natural" target.
    Koto,
    /// SoundFont-driven piano: routes (channel, note, velocity) to a shared
    /// `rustysynth::Synthesizer` loaded from a `.sf2` file at startup. The
    /// per-voice slot is a placeholder used only for note-tracking and
    /// eviction; the actual audio is rendered by the shared Synthesizer in
    /// the audio callback and mixed into the voice bus.
    SfPiano,
}

// ---------------------------------------------------------------------------
// Voice trait + implementations
// ---------------------------------------------------------------------------

pub trait VoiceImpl: Send {
    /// Add this voice's contribution into `buf` (mono, length = frames).
    fn render_add(&mut self, buf: &mut [f32]);
    /// Mark released; envelope should fade and `is_done()` will go true.
    fn trigger_release(&mut self);
    fn is_done(&self) -> bool;
    /// True if the voice has been released but is still ringing out.
    /// Default returns false; voice impls may override for smarter eviction.
    fn is_releasing(&self) -> bool {
        false
    }
}

pub struct Voice {
    /// (channel, midi_note) - used to match note_off back to the voice
    pub key: (u8, u8),
    pub inner: Box<dyn VoiceImpl>,
}

/// Parameters mutated by MIDI CC / GUI and read by the audio + MIDI threads.
/// Held under a Mutex (not atomics) because updates are infrequent and we
/// only ever lock briefly.
pub struct LiveParams {
    /// Master gain, pre-tanh. Mapped from CC 7 / CC 70 / GUI slider.
    pub master: f32,
    /// Currently selected engine. Read by the MIDI callback on every
    /// note_on so changes apply immediately to new notes (already-playing
    /// notes finish out with their original engine).
    pub engine: Engine,
    /// Body-IR convolution reverb wet level, 0..=1. 0 = dry (reverb stage
    /// no-ops). Read by the audio callback every buffer.
    pub reverb_wet: f32,
}

/// Live MIDI snapshot for the dashboard. Updated by the MIDI callback,
/// read by egui repaint. Decoupled from `LiveParams` because the audio
/// callback only cares about a few numbers, while the dashboard wants to
/// show every event.
pub struct DashState {
    /// CC number -> last raw value (0..127). Used to draw knob indicators.
    pub cc_raw: HashMap<u8, u8>,
    /// CC number -> count of messages received (helps spot active controls).
    pub cc_count: HashMap<u8, u64>,
    /// Currently held (channel, midi_note) keys.
    pub active_notes: HashSet<(u8, u8)>,
    /// Rolling log of recent MIDI messages, newest at the back.
    pub recent: VecDeque<String>,
    /// Currently selected engine (display only; engine is fixed at start
    /// today, but exposed here for future GUI-driven swapping).
    pub engine: Engine,
}

impl DashState {
    pub fn new(engine: Engine) -> Self {
        Self {
            cc_raw: HashMap::new(),
            cc_count: HashMap::new(),
            active_notes: HashSet::new(),
            recent: VecDeque::with_capacity(64),
            engine,
        }
    }

    pub fn push_event(&mut self, line: String) {
        if self.recent.len() >= 64 {
            self.recent.pop_front();
        }
        self.recent.push_back(line);
    }
}

// --- Square voice ----------------------------------------------------------

pub struct SquareVoice {
    sr: f32,
    freq: f32,
    amp: f32,
    phase: f32,
    env: f32,
    released: bool,
    attack_per_sample: f32,
    release_per_sample: f32,
}

impl SquareVoice {
    pub fn new(sr: f32, freq: f32, velocity: u8) -> Self {
        let attack_sec = 0.005_f32;
        let release_sec = 0.080_f32;
        Self {
            sr,
            freq,
            amp: (velocity.max(1) as f32) / 127.0,
            phase: 0.0,
            env: 0.0,
            released: false,
            attack_per_sample: 1.0 / (attack_sec * sr),
            release_per_sample: 1.0 / (release_sec * sr),
        }
    }
}

impl VoiceImpl for SquareVoice {
    fn render_add(&mut self, buf: &mut [f32]) {
        let inc = self.freq / self.sr;
        for sample in buf.iter_mut() {
            if self.released {
                self.env = (self.env - self.release_per_sample).max(0.0);
            } else if self.env < 1.0 {
                self.env = (self.env + self.attack_per_sample).min(1.0);
            }
            let s = if self.phase < 0.5 { 1.0 } else { -1.0 };
            *sample += s * self.env * self.amp;
            self.phase += inc;
            if self.phase >= 1.0 {
                self.phase -= 1.0;
            }
        }
    }
    fn trigger_release(&mut self) {
        self.released = true;
    }
    fn is_done(&self) -> bool {
        self.released && self.env <= 1e-5
    }
}

// --- Karplus-Strong voice --------------------------------------------------

pub struct KsVoice {
    buf: Vec<f32>,
    head: usize,
    decay: f32,
    // Fractional-delay tuning allpass: y[n] = c*x[n] + x[n-1] - c*y[n-1]
    // where c = (1 - frac) / (1 + frac). Total delay = buf.len() + frac.
    tune_coef: f32,
    tune_xprev: f32,
    tune_yprev: f32,
    released: bool,
    rel_mul: f32,
    rel_step: f32,
}

impl KsVoice {
    pub fn new(sr: f32, freq: f32, velocity: u8) -> Self {
        // Fractional-delay decomposition: integer N + frac in [0, 1).
        // Without this, notes above ~MIDI 72 mistune by tens of cents.
        // Reserve 0.5 sample for the in-loop 2-tap LPF (group delay = 0.5
        // at low freq); the tuning AP picks up the rest.
        let raw = sr / freq.max(1.0);
        let target = (raw - 0.5_f32).max(2.0);
        let n_f = target.floor();
        let frac = target - n_f;
        let n = (n_f as usize).max(2);
        let tune_coef = (1.0 - frac) / (1.0 + frac);
        let amp = (velocity.max(1) as f32) / 127.0;
        // Hammer excitation: velocity sets contact width -> spectral brightness.
        let width = hammer_width_for_velocity(velocity);
        let buf = hammer_excitation(n, width, amp);
        let release_sec = 0.150_f32;
        let release_samples = (release_sec * sr).max(1.0);
        // exp(ln(eps)/N) so we hit ~-60dB after release_samples samples.
        let rel_step = (1e-3_f32.ln() / release_samples).exp();
        Self {
            buf,
            head: 0,
            decay: 0.996,
            tune_coef,
            tune_xprev: 0.0,
            tune_yprev: 0.0,
            released: false,
            rel_mul: 1.0,
            rel_step,
        }
    }
}

impl VoiceImpl for KsVoice {
    fn render_add(&mut self, buf: &mut [f32]) {
        let n = self.buf.len();
        let c = self.tune_coef;
        for sample in buf.iter_mut() {
            let cur = self.buf[self.head];
            let prev = if self.head == 0 {
                self.buf[n - 1]
            } else {
                self.buf[self.head - 1]
            };
            let mut out_sample = cur;
            if self.released {
                self.rel_mul *= self.rel_step;
                out_sample *= self.rel_mul;
            }
            *sample += out_sample;
            // Lowpass + decay, then 1st-order allpass tuning filter for the
            // fractional-delay component.
            let lp = (cur + prev) * 0.5 * self.decay;
            let y = c * lp + self.tune_xprev - c * self.tune_yprev;
            self.tune_xprev = lp;
            self.tune_yprev = y;
            self.buf[self.head] = y;
            self.head += 1;
            if self.head >= n {
                self.head = 0;
            }
        }
    }
    fn trigger_release(&mut self) {
        self.released = true;
    }
    fn is_done(&self) -> bool {
        self.released && self.rel_mul <= 1e-4
    }
}

// ---------------------------------------------------------------------------
// Shared: 4-stage ADSR envelope
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdsrStage {
    Attack,
    Decay,
    Sustain,
    Release,
    Done,
}

struct Adsr {
    sr: f32,
    attack_s: f32,
    decay_s: f32,
    sustain: f32,    // 0..1 sustain level
    release_s: f32,
    stage: AdsrStage,
    value: f32,
    /// Captured value at the moment Release was triggered, so the release
    /// ramp lands at 0 in exactly `release_s` regardless of where in the
    /// envelope the note-off arrived. Without this, sustain=0 voices have
    /// release_rate=0 and the note rings forever (the original bug).
    release_start_value: f32,
}

impl Adsr {
    fn new(sr: f32, attack_s: f32, decay_s: f32, sustain: f32, release_s: f32) -> Self {
        Self {
            sr,
            attack_s: attack_s.max(1e-4),
            decay_s: decay_s.max(1e-4),
            sustain,
            release_s: release_s.max(1e-4),
            stage: AdsrStage::Attack,
            value: 0.0,
            release_start_value: 0.0,
        }
    }

    fn release(&mut self) {
        if self.value <= 1e-6 {
            // Nothing to fade; mark done so the voice gets cleaned up.
            self.stage = AdsrStage::Done;
        } else {
            self.release_start_value = self.value;
            self.stage = AdsrStage::Release;
        }
    }

    fn next(&mut self) -> f32 {
        match self.stage {
            AdsrStage::Attack => {
                let rate = 1.0 / (self.attack_s * self.sr);
                self.value += rate;
                if self.value >= 1.0 {
                    self.value = 1.0;
                    self.stage = AdsrStage::Decay;
                }
            }
            AdsrStage::Decay => {
                let rate = (1.0 - self.sustain) / (self.decay_s * self.sr);
                self.value -= rate;
                if self.value <= self.sustain {
                    self.value = self.sustain;
                    self.stage = AdsrStage::Sustain;
                }
            }
            AdsrStage::Sustain => {}
            AdsrStage::Release => {
                // Linear fall from release_start_value to 0 in release_s.
                let rate = self.release_start_value / (self.release_s * self.sr);
                self.value -= rate;
                if self.value <= 0.0 {
                    self.value = 0.0;
                    self.stage = AdsrStage::Done;
                }
            }
            AdsrStage::Done => {}
        }
        self.value
    }

    fn done(&self) -> bool {
        matches!(self.stage, AdsrStage::Done)
    }
}

// ---------------------------------------------------------------------------
// Subtractive: saw -> SVF lowpass (cutoff env) -> ADSR amp
// ---------------------------------------------------------------------------
//
// State-Variable Filter is the TPT (topology-preserving transform) form by
// Andrew Simper / Cytomic. It stays stable across the full audio range and
// has no zero-delay-feedback issues.

pub struct SubVoice {
    sr: f32,
    freq: f32,
    amp: f32,
    phase: f32,         // 0..1 for sawtooth
    amp_env: Adsr,
    filt_env: Adsr,
    // Cutoff: cutoff_base + cutoff_env_amount * filt_env (Hz)
    cutoff_base: f32,
    cutoff_env_amount: f32,
    q: f32,
    // SVF state
    ic1eq: f32,
    ic2eq: f32,
}

impl SubVoice {
    pub fn new(sr: f32, freq: f32, velocity: u8) -> Self {
        let amp = (velocity.max(1) as f32) / 127.0;
        Self {
            sr,
            freq,
            amp,
            phase: 0.0,
            // Punchy bass-synth defaults: fast attack, fast filter sweep.
            amp_env: Adsr::new(sr, 0.005, 0.250, 0.55, 0.250),
            filt_env: Adsr::new(sr, 0.001, 0.180, 0.20, 0.250),
            cutoff_base: 200.0,
            cutoff_env_amount: 5500.0,
            q: 1.4,
            ic1eq: 0.0,
            ic2eq: 0.0,
        }
    }

    #[inline]
    fn svf_lowpass(&mut self, x: f32, cutoff: f32) -> f32 {
        let g = (std::f32::consts::PI * cutoff / self.sr).tan();
        let k = 1.0 / self.q.max(0.5);
        let denom = 1.0 + g * (g + k);
        let a1 = 1.0 / denom;
        let a2 = g * a1;
        let a3 = g * a2;
        let v3 = x - self.ic2eq;
        let v1 = a1 * self.ic1eq + a2 * v3;
        let v2 = self.ic2eq + a2 * self.ic1eq + a3 * v3;
        self.ic1eq = 2.0 * v1 - self.ic1eq;
        self.ic2eq = 2.0 * v2 - self.ic2eq;
        v2
    }
}

impl VoiceImpl for SubVoice {
    fn render_add(&mut self, buf: &mut [f32]) {
        let inc = self.freq / self.sr;
        let nyq = self.sr * 0.45;
        for sample in buf.iter_mut() {
            // Naive sawtooth -1..+1; aliases above ~5kHz fundamental but
            // pleasantly rough for most playable range.
            let osc = self.phase * 2.0 - 1.0;
            self.phase += inc;
            if self.phase >= 1.0 {
                self.phase -= 1.0;
            }

            let fe = self.filt_env.next();
            let cutoff = (self.cutoff_base + self.cutoff_env_amount * fe).clamp(20.0, nyq);
            let filtered = self.svf_lowpass(osc, cutoff);
            let ae = self.amp_env.next();

            *sample += filtered * ae * self.amp;
        }
    }
    fn trigger_release(&mut self) {
        self.amp_env.release();
        self.filt_env.release();
    }
    fn is_done(&self) -> bool {
        self.amp_env.done()
    }
}

// ---------------------------------------------------------------------------
// FM: 2-op (carrier + modulator), modulator has its own ADSR
// ---------------------------------------------------------------------------
//
// modulator_out = sin(TAU * mod_phase) * mod_index * mod_env
// carrier_out   = sin(TAU * car_phase + modulator_out) * amp_env
//
// Default ratio 14:1 + fast mod-env decay = DX7-ish bell / EP "tine" sound.

pub struct FmVoice {
    sr: f32,
    car_freq: f32,
    car_phase: f32,
    mod_freq: f32,
    mod_phase: f32,
    mod_index: f32,
    amp: f32,
    amp_env: Adsr,
    mod_env: Adsr,
}

impl FmVoice {
    pub fn new(sr: f32, freq: f32, velocity: u8) -> Self {
        let amp = (velocity.max(1) as f32) / 127.0;
        let ratio = 14.0_f32;
        Self {
            sr,
            car_freq: freq,
            car_phase: 0.0,
            mod_freq: freq * ratio,
            mod_phase: 0.0,
            mod_index: 4.0,
            amp,
            amp_env: Adsr::new(sr, 0.002, 0.800, 0.0, 0.500),
            mod_env: Adsr::new(sr, 0.001, 0.350, 0.0, 0.350),
        }
    }
}

impl VoiceImpl for FmVoice {
    fn render_add(&mut self, buf: &mut [f32]) {
        use std::f32::consts::TAU;
        let car_inc = self.car_freq / self.sr;
        let mod_inc = self.mod_freq / self.sr;
        for sample in buf.iter_mut() {
            let me = self.mod_env.next();
            let modulator = (TAU * self.mod_phase).sin() * self.mod_index * me;
            let carrier = (TAU * self.car_phase + modulator).sin();
            let ae = self.amp_env.next();
            *sample += carrier * ae * self.amp;
            self.car_phase += car_inc;
            if self.car_phase >= 1.0 {
                self.car_phase -= 1.0;
            }
            self.mod_phase += mod_inc;
            if self.mod_phase >= 1.0 {
                self.mod_phase -= 1.0;
            }
        }
    }
    fn trigger_release(&mut self) {
        self.amp_env.release();
        self.mod_env.release();
    }
    fn is_done(&self) -> bool {
        self.amp_env.done()
    }
}

// ---------------------------------------------------------------------------
// KS-rich: 3 detuned strings + 1-pole allpass dispersion (stiffness)
// ---------------------------------------------------------------------------
//
// Each "string" is an independent KS delay loop. Per-string detune (cents)
// gives chorus/unison thickness like real piano triple-stringing. A single
// allpass in the feedback path adds frequency-dependent phase delay -> the
// inharmonicity that makes piano upper partials slightly sharp.

pub struct KsString {
    buf: Vec<f32>,
    head: usize,
    decay: f32,
    // Dispersion (stiffness) 1-pole allpass: textbook Smith form
    //   H(z) = (a + z^-1) / (1 + a * z^-1)
    //   y[n] = a * x[n] + x[n-1] - a * y[n-1]
    ap_coef: f32,
    ap_xprev: f32,
    ap_yprev: f32,
    // Tuning 1-pole allpass for the fractional-delay component, same form.
    // c = (1 - frac) / (1 + frac), giving group delay ~= frac at low freq.
    // Total effective delay is buf.len() + frac.
    tune_coef: f32,
    tune_xprev: f32,
    tune_yprev: f32,
}

impl KsString {
    pub fn new(sr: f32, freq: f32, amp: f32, ap_coef: f32, hammer_width: usize) -> Self {
        let (n, _frac) = Self::delay_length_compensated(sr, freq, ap_coef);
        let buf = hammer_excitation(n, hammer_width, amp);
        Self::with_buf(sr, freq, buf, 0.997, ap_coef)
    }

    /// Construct with a fully-prepared excitation buffer and explicit decay.
    /// Used by `piano` (asymmetric hammer + freq-dependent decay) and `koto`
    /// (offset pluck + long sustain) which need control beyond `new`.
    /// `sr` and `freq` are used to compute the fractional-delay tuning coef.
    /// The caller MUST size `buf` to `delay_length_compensated(sr, freq, ap_coef).0`.
    pub fn with_buf(sr: f32, freq: f32, buf: Vec<f32>, decay: f32, ap_coef: f32) -> Self {
        let (_n, frac) = Self::delay_length_compensated(sr, freq, ap_coef);
        // Allpass coefficient for fractional delay of `frac` samples.
        // c = (1 - frac) / (1 + frac), Julius O. Smith, PASP.
        let tune_coef = (1.0 - frac) / (1.0 + frac);
        Self {
            buf,
            head: 0,
            decay,
            ap_coef,
            ap_xprev: 0.0,
            ap_yprev: 0.0,
            tune_coef,
            tune_xprev: 0.0,
            tune_yprev: 0.0,
        }
    }

    /// Returns (integer delay length N, fractional component in [0, 1)).
    /// Total target period = N + frac samples.
    pub fn delay_length(sr: f32, freq: f32) -> (usize, f32) {
        let raw = sr / freq.max(1.0);
        // Reserve ~1 sample for the tuning allpass; ensure N >= 2.
        // Use floor so frac in [0, 1).
        let n_f = raw.floor();
        let frac = raw - n_f;
        let n = (n_f as usize).max(2);
        (n, frac)
    }

    /// Like `delay_length` but compensates for the extra phase delay
    /// introduced inside the feedback loop by the 2-tap LPF (0.5 sample
    /// at low freq) and the dispersion allpass (`(1 - a) / (1 + a)` at DC,
    /// Julius O. Smith PASP). Without this compensation, the loop period
    /// is too long and the fundamental sounds flat by ~20+ cents at C4.
    pub fn delay_length_compensated(sr: f32, freq: f32, ap_coef: f32) -> (usize, f32) {
        let raw = sr / freq.max(1.0);
        // 2-tap moving average (linear-phase would be 0.5); we now run an
        // asymmetric 0.95/0.05 weighting in `step`, which empirically lands
        // closer to 0.5 sample of total in-loop LPF+buffer delay once the
        // writeback race is included, so 0.5 remains the right number.
        let lpf_delay = 0.5_f32;
        // 1st-order allpass phase delay at DC: (1 - a) / (1 + a).
        let disp_delay = (1.0 - ap_coef) / (1.0 + ap_coef);
        // Reserve `extra` samples worth of delay for the in-loop filters
        // (LPF + dispersion AP); the tuning allpass picks up the remainder.
        // The +0.5 empirical fudge accounts for the implicit one-sample read-
        // before-write delay through the buffer slot itself; without it C4
        // still reads ~14 cents flat in the analyse harness.
        let extra = lpf_delay + disp_delay + 0.5;
        let target = (raw - extra).max(2.0);
        let n_f = target.floor();
        let frac = target - n_f;
        let n = (n_f as usize).max(2);
        (n, frac)
    }

    #[inline]
    fn step(&mut self) -> f32 {
        let n = self.buf.len();
        let cur = self.buf[self.head];
        let prev = if self.head == 0 {
            self.buf[n - 1]
        } else {
            self.buf[self.head - 1]
        };
        // 2-tap weighted lowpass + decay: weighting `cur` heavier than
        // `prev` flattens the high-frequency rolloff, preserving brilliance
        // in upper partials. Pure 0.5/0.5 average loses h7+ in <0.2s.
        let lp = (cur * 0.97 + prev * 0.03) * self.decay;
        // Dispersion 1-pole allpass (stiffness): textbook Smith form.
        //   y[n] = a * x[n] + x[n-1] - a * y[n-1]
        let a = self.ap_coef;
        let disp = a * lp + self.ap_xprev - a * self.ap_yprev;
        self.ap_xprev = lp;
        self.ap_yprev = disp;
        // Tuning allpass: same form, separate state. Placed AFTER dispersion
        // so the fractional delay sees the full feedback chain.
        let c = self.tune_coef;
        let y = c * disp + self.tune_xprev - c * self.tune_yprev;
        self.tune_xprev = disp;
        self.tune_yprev = y;
        self.buf[self.head] = y;
        self.head += 1;
        if self.head >= n {
            self.head = 0;
        }
        cur
    }
}

pub struct KsRichVoice {
    strings: [KsString; 3],
    released: bool,
    rel_mul: f32,
    rel_step: f32,
}

impl KsRichVoice {
    pub fn new(sr: f32, freq: f32, velocity: u8) -> Self {
        let amp = (velocity.max(1) as f32) / 127.0;
        // Detune cents: -7, 0, +7 -> classic piano triple-string spread
        let cents_to_ratio = |c: f32| 2.0_f32.powf(c / 1200.0);
        let detunes = [
            cents_to_ratio(-7.0),
            1.0,
            cents_to_ratio(7.0),
        ];
        // Higher notes get slightly more allpass coefficient (more stiffness),
        // mimicking real piano string behaviour where shorter strings are
        // stiffer relative to their length.
        let ap = (0.18 + (freq / 2000.0).min(0.20)).min(0.40);
        // Same hammer hits all 3 strings in phase; detune drives them apart
        // over time, producing the classic chorus / beating effect.
        let hammer_w = hammer_width_for_velocity(velocity);
        let strings = [
            KsString::new(sr, freq * detunes[0], amp, ap, hammer_w),
            KsString::new(sr, freq * detunes[1], amp, ap, hammer_w),
            KsString::new(sr, freq * detunes[2], amp, ap, hammer_w),
        ];
        let release_sec = 0.250_f32;
        let release_samples = (release_sec * sr).max(1.0);
        let rel_step = (1e-3_f32.ln() / release_samples).exp();
        Self {
            strings,
            released: false,
            rel_mul: 1.0,
            rel_step,
        }
    }
}

impl VoiceImpl for KsRichVoice {
    fn render_add(&mut self, buf: &mut [f32]) {
        // Mix the 3 strings; equal-power-ish gain (1/sqrt(3) ~= 0.577).
        let gain = 0.577_f32;
        for sample in buf.iter_mut() {
            let s = self.strings[0].step()
                + self.strings[1].step()
                + self.strings[2].step();
            let mut out_sample = s * gain;
            if self.released {
                self.rel_mul *= self.rel_step;
                out_sample *= self.rel_mul;
            }
            *sample += out_sample;
        }
    }
    fn trigger_release(&mut self) {
        self.released = true;
    }
    fn is_done(&self) -> bool {
        self.released && self.rel_mul <= 1e-4
    }
}

// ---------------------------------------------------------------------------
// Hammer excitation (shared by ks / ks-rich)
// ---------------------------------------------------------------------------
//
// White-noise excitation gives a "thin" plucked sound regardless of velocity.
// Real piano hammers hit the string for a finite contact window; the WIDER
// the contact pulse, the NARROWER its frequency content (Fourier dual), so:
//
//   hard hit -> hammer felt compresses fast   -> short contact -> bright
//   soft hit -> hammer felt stays soft longer -> long contact  -> mellow
//
// We model the hammer as a raised-cosine (Hann-window-shaped) impulse padded
// with silence to fill the delay-line length. Width derived from velocity.

pub fn hammer_width_for_velocity(vel: u8) -> usize {
    // vel=127 -> 3 samples (sharp click, very bright)
    // vel=20  -> ~19 samples (soft, mellow)
    let v = vel.clamp(1, 127) as f32;
    (3.0 + (127.0 - v) * 0.13) as usize
}

pub fn hammer_excitation(n: usize, width: usize, amp: f32) -> Vec<f32> {
    let mut buf = vec![0.0_f32; n];
    let w = width.min(n).max(1);
    if w == 1 {
        buf[0] = amp;
        return buf;
    }
    use std::f32::consts::TAU;
    let denom = (w - 1) as f32;
    for i in 0..w {
        let t = i as f32 / denom;       // 0..1
        let val = 0.5 * (1.0 - (TAU * t).cos()); // Hann: 0 -> 1 -> 0
        buf[i] = val * amp;
    }
    buf
}

// ---------------------------------------------------------------------------
// Piano-leaning DWS engine
// ---------------------------------------------------------------------------
//
// Differences from `ks-rich`:
//   - WIDER, ms-scale hammer contact (20-200 samples vs 3-19)
//     -> contact pulse closer to real piano (hammer felt compresses over ms)
//   - ASYMMETRIC hammer profile: linear rise + exponential decay
//     -> mimics felt strike physics (sharp impact, slow release)
//   - FREQUENCY-DEPENDENT decay: high notes die faster
//     -> matches real piano where treble strings have less mass / faster damp
//   - Stronger allpass dispersion -> more inharmonicity (piano stretch tuning)

pub fn piano_hammer_width(vel: u8) -> usize {
    // vel=127 -> 20 samples (~0.45 ms): firm but felt
    // vel=20  -> 200 samples (~4.5 ms): very soft
    let v = vel.clamp(1, 127) as f32;
    (20.0 + (127.0 - v) * 1.7) as usize
}

pub fn piano_hammer_excitation(n: usize, width: usize, amp: f32) -> Vec<f32> {
    let mut buf = vec![0.0_f32; n];
    let w = width.min(n).max(1);
    if w == 1 {
        buf[0] = amp;
        return buf;
    }
    // Asymmetric envelope: linear rise (1/4), exponential decay (3/4).
    let rise = (w / 4).max(1);
    let decay_n = w - rise;
    for i in 0..rise {
        let t = i as f32 / rise as f32;
        buf[i] = t * amp;
    }
    let tau = (decay_n as f32 / 3.0).max(1.0); // ~95% decayed at end
    for i in 0..decay_n {
        let t = i as f32;
        buf[rise + i] = (-t / tau).exp() * amp;
    }
    buf
}

pub struct PianoVoice {
    strings: [KsString; 3],
    released: bool,
    rel_mul: f32,
    rel_step: f32,
    /// Pre-rendered hammer "click" — short broadband transient mixed in for
    /// the first ~50 samples to model the felt-on-string mechanical thump
    /// (the part of the onset that is NOT string vibration). Dramatically
    /// improves onset RMS envelope match against real piano samples.
    click_buf: Vec<f32>,
    click_pos: usize,
}

impl PianoVoice {
    pub fn new(sr: f32, freq: f32, velocity: u8) -> Self {
        let amp = (velocity.max(1) as f32) / 127.0;
        let cents_to_ratio = |c: f32| 2.0_f32.powf(c / 1200.0);
        // Real piano triple-string unison detune: ~0.5-2 cents
        // (Weinreich 1977, "Coupled piano strings"). Wider spreads
        // (e.g. +/-7c) sound like an out-of-tune honky-tonk, not a piano.
        let detunes = [cents_to_ratio(-1.5), 1.0, cents_to_ratio(1.5)];
        // Lower stiffness — measured B at C4 ~3e-4 from SF2 reference; previous
        // 0.25..0.50 ap range gave B ~4.6e-4 (cand too stiff). Drop to a
        // shallow fixed range that puts B near the reference.
        let ap = (0.10 + (freq / 4000.0).min(0.05)).min(0.18);
        // Frequency-dependent decay: 110 Hz -> 0.9990 (long); 2000 Hz -> 0.9930 (short)
        // Pushed both endpoints toward 1.0: high notes were dying in <0.2s
        // because the in-loop LPF was already taking the edge off; once we
        // rebalanced that to 0.95/0.05 the per-trip decay needed to give
        // back the sustain margin it had been hiding.
        // Faster decay: ref T60 ~3-6s; previous 1.0..0.9992 gave 4-8s candidate.
        // Walk back to a more conventional KS decay range.
        let decay_for = |f: f32| -> f32 {
            let high = (f / 2000.0).clamp(0.0, 1.0);
            0.9985 - 0.0035 * high  // ~0.9985 at lows, ~0.995 at 2k
        };
        let hammer_w = piano_hammer_width(velocity);
        let mk = |freq_string: f32| -> KsString {
            let (n, _frac) = KsString::delay_length_compensated(sr, freq_string, ap);
            let buf = piano_hammer_excitation(n, hammer_w, amp);
            KsString::with_buf(sr, freq_string, buf, decay_for(freq_string), ap)
        };
        let strings = [
            mk(freq * detunes[0]),
            mk(freq * detunes[1]),
            mk(freq * detunes[2]),
        ];
        let release_sec = 0.300_f32;
        let release_samples = (release_sec * sr).max(1.0);
        let rel_step = (1e-3_f32.ln() / release_samples).exp();
        // Build a short bandpassed-noise click: ~0.7 ms at 44.1 kHz = 31 samples.
        // Velocity scales the amplitude. Simple two-pole shaping (highpass +
        // lowpass via leaky averager) gives 200..8000 Hz bandpass-ish noise.
        let click_samples = ((sr * 0.0008) as usize).clamp(16, 80);
        let click_amp = amp * 0.55;
        let mut click_buf = Vec::with_capacity(click_samples);
        // Deterministic per-voice seed so unit tests are reproducible.
        let mut rng_state: u32 = 0x9E37_79B9
            ^ ((freq.to_bits() as u32).wrapping_mul(2654435761))
            ^ ((velocity as u32).wrapping_mul(0x85EB_CA77));
        let mut lp_prev = 0.0f32;
        let mut hp_prev_x = 0.0f32;
        let mut hp_prev_y = 0.0f32;
        // Highpass coef (cut <~200 Hz): a = 1 - 2*pi*f/sr
        let hp_a = 1.0 - 2.0 * std::f32::consts::PI * 200.0 / sr;
        // Lowpass coef (cut >~8 kHz): leak fraction
        let lp_a = (-2.0 * std::f32::consts::PI * 8000.0 / sr).exp();
        for i in 0..click_samples {
            // xorshift32
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            let raw = ((rng_state as i32) as f32) / (i32::MAX as f32);
            // Highpass: y = a*(y_prev + x - x_prev)
            let hp = hp_a * (hp_prev_y + raw - hp_prev_x);
            hp_prev_x = raw;
            hp_prev_y = hp;
            // Lowpass: y = (1-a)*x + a*y_prev
            lp_prev = (1.0 - lp_a) * hp + lp_a * lp_prev;
            // Half-cosine amplitude window: 0 -> 1 -> 0 across the click.
            let t = i as f32 / (click_samples - 1).max(1) as f32;
            let win = 0.5 * (1.0 - (std::f32::consts::TAU * t).cos());
            click_buf.push(lp_prev * win * click_amp);
        }
        Self {
            strings,
            released: false,
            rel_mul: 1.0,
            rel_step,
            click_buf,
            click_pos: 0,
        }
    }
}

impl VoiceImpl for PianoVoice {
    fn render_add(&mut self, buf: &mut [f32]) {
        let gain = 0.577_f32;
        for sample in buf.iter_mut() {
            let s = self.strings[0].step()
                + self.strings[1].step()
                + self.strings[2].step();
            let mut out_sample = s * gain;
            // Mix in remaining click samples (mechanical hammer-strike thump).
            if self.click_pos < self.click_buf.len() {
                out_sample += self.click_buf[self.click_pos];
                self.click_pos += 1;
            }
            if self.released {
                self.rel_mul *= self.rel_step;
                out_sample *= self.rel_mul;
            }
            *sample += out_sample;
        }
    }
    fn trigger_release(&mut self) {
        self.released = true;
    }
    fn is_done(&self) -> bool {
        self.released && self.rel_mul <= 1e-4
    }
}

// ---------------------------------------------------------------------------
// Koto / shamisen-leaning DWS engine
// ---------------------------------------------------------------------------
//
// KS is plucked-string physics. This engine leans into that:
//   - SINGLE string (no unison) - koto/shamisen are not unison-strung
//   - SHARP narrow plectrum (width=2 always; only amplitude scales w/ vel)
//     -> tsume / bachi (爪・撥) is much sharper than a hammer felt
//   - PLUCK POSITION OFFSET: inject impulse at delay-line offset n/4 instead
//     of 0. Combined with the natural reflection physics, this introduces
//     the classic "comb-filter" effect that defines pluck-position timbre
//     (plucking near the bridge sounds twangy/bright; near middle = mellow)
//   - LONG sustain (decay closer to 1.0) - acoustic koto strings ring 5-15s
//   - MINIMAL allpass dispersion - low stiffness, harmonic spectrum

pub fn koto_pluck_excitation(n: usize, amp: f32, pluck_pos: usize) -> Vec<f32> {
    let mut buf = vec![0.0_f32; n];
    if n == 0 {
        return buf;
    }
    // Pluck-position FIR comb (1 - z^-p): inject the plectrum at offset 0
    // AND a sign-reversed copy at offset `pluck_pos`. This is the standard
    // pluck-position model in Karplus-Strong (the original a-spike-at-offset
    // form was just a time shift, not a comb, so it produced no notches).
    // Result: spectrum has nulls at every (sr / pluck_pos) Hz, giving the
    // classic bridge-vs-middle pluck timbre.
    //
    // Two-sample plectrum shape kept (sharp + small negative snap-back),
    // so each "tap" of the comb is a (+amp, -0.3*amp) pair.
    buf[0] = amp;
    buf[1 % n] += -amp * 0.3;
    let p = pluck_pos % n;
    buf[p] += -amp;
    buf[(p + 1) % n] += amp * 0.3;
    buf
}

pub struct KotoVoice {
    string: KsString,
    released: bool,
    rel_mul: f32,
    rel_step: f32,
}

impl KotoVoice {
    pub fn new(sr: f32, freq: f32, velocity: u8) -> Self {
        let amp = (velocity.max(1) as f32) / 127.0;
        let koto_ap = 0.05_f32;
        let (n, _frac) = KsString::delay_length_compensated(sr, freq, koto_ap);
        // Pluck near 1/4 of string length (typical koto tsume position).
        // Combined with the comb-FIR shape in koto_pluck_excitation, this
        // gives the bridge-side bright/twangy timbre.
        let pluck_pos = n / 4;
        let buf = koto_pluck_excitation(n, amp, pluck_pos);
        // Long sustain, very mild stiffness.
        let string = KsString::with_buf(sr, freq, buf, 0.9992, koto_ap);
        let release_sec = 0.500_f32; // longer release = "ふわっと消える"
        let release_samples = (release_sec * sr).max(1.0);
        let rel_step = (1e-3_f32.ln() / release_samples).exp();
        Self {
            string,
            released: false,
            rel_mul: 1.0,
            rel_step,
        }
    }
}

impl VoiceImpl for KotoVoice {
    fn render_add(&mut self, buf: &mut [f32]) {
        for sample in buf.iter_mut() {
            let mut out_sample = self.string.step();
            if self.released {
                self.rel_mul *= self.rel_step;
                out_sample *= self.rel_mul;
            }
            *sample += out_sample;
        }
    }
    fn trigger_release(&mut self) {
        self.released = true;
    }
    fn is_done(&self) -> bool {
        self.released && self.rel_mul <= 1e-4
    }
}

// ---------------------------------------------------------------------------

pub fn midi_to_freq(note: u8) -> f32 {
    440.0_f32 * 2.0_f32.powf((note as f32 - 69.0) / 12.0)
}

// ---------------------------------------------------------------------------
// Engine -> voice factory
// ---------------------------------------------------------------------------

/// Build the boxed `VoiceImpl` for `engine` at the given sample rate, pitch,
/// and MIDI velocity. Lives in the library so the live MIDI callback in
/// `main.rs` and offline harnesses (`bench`, regression tools) construct
/// voices through one shared switch.
///
/// `Engine::SfPiano` returns an `SfPianoPlaceholder`: it adds nothing to the
/// audio bus by itself. Real audio for that engine is rendered by a shared
/// `rustysynth::Synthesizer` owned by the audio callback; the placeholder
/// only exists so the voice-pool eviction logic can still track which
/// (channel, note) pairs are sounding.
pub fn make_voice(
    engine: Engine,
    sr: f32,
    freq: f32,
    velocity: u8,
) -> Box<dyn VoiceImpl + Send> {
    match engine {
        Engine::Square => Box::new(SquareVoice::new(sr, freq, velocity)),
        Engine::Ks => Box::new(KsVoice::new(sr, freq, velocity)),
        Engine::KsRich => Box::new(KsRichVoice::new(sr, freq, velocity)),
        Engine::Sub => Box::new(SubVoice::new(sr, freq, velocity)),
        Engine::Fm => Box::new(FmVoice::new(sr, freq, velocity)),
        Engine::Piano => Box::new(PianoVoice::new(sr, freq, velocity)),
        Engine::Koto => Box::new(KotoVoice::new(sr, freq, velocity)),
        Engine::SfPiano => Box::new(SfPianoPlaceholder::new()),
    }
}

// ---------------------------------------------------------------------------
// SoundFont placeholder voice
// ---------------------------------------------------------------------------
//
// The `rustysynth::Synthesizer` is a shared, mutable engine that handles all
// channels at once -- it is NOT one-instance-per-voice. We keep the voice
// pool's per-note bookkeeping (eviction, note-off matching) by pushing a
// silent placeholder voice for each SfPiano note_on. Audio is rendered by
// the audio callback calling `synth.render(...)` once per buffer and mixing
// the result onto the voice bus.
//
// `is_done` returns true a few seconds after release so the placeholder is
// eventually reaped; the underlying synth voice may continue to ring out
// inside rustysynth's own envelope, which is fine -- the synth tracks its
// own active voices independently.

pub struct SfPianoPlaceholder {
    released: bool,
    released_samples: u64,
}

impl SfPianoPlaceholder {
    pub fn new() -> Self {
        Self {
            released: false,
            released_samples: 0,
        }
    }
}

impl Default for SfPianoPlaceholder {
    fn default() -> Self {
        Self::new()
    }
}

impl VoiceImpl for SfPianoPlaceholder {
    fn render_add(&mut self, buf: &mut [f32]) {
        // Audio for this engine is rendered by the shared Synthesizer in
        // the audio callback. We just count post-release samples so the
        // pool can eventually evict us.
        if self.released {
            self.released_samples = self
                .released_samples
                .saturating_add(buf.len() as u64);
        }
    }
    fn trigger_release(&mut self) {
        self.released = true;
    }
    fn is_done(&self) -> bool {
        // ~3 s at 44.1 kHz -- well past most piano tails. The shared synth
        // owns the real envelope; this is purely for our placeholder pool.
        self.released && self.released_samples > 132_300
    }
    fn is_releasing(&self) -> bool {
        self.released
    }
}
