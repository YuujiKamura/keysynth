//! Real-time MIDI keyboard -> speaker synth.
//!
//! Single-binary Rust replacement for the Python prototype. midir delivers
//! MIDI events on its own callback thread; cpal's audio callback runs on a
//! dedicated audio thread; both share a Mutex<Vec<Voice>> for the active
//! voice pool. Voices are released (env decay) when note_off arrives and
//! removed from the pool once their envelope reaches silence.
//!
//! Engines:
//!   - square : NES-style pulse wave with linear AR envelope
//!   - ks     : Karplus-Strong plucked string (single delay line + 2-tap
//!              lowpass), Phase 1 physical model

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use midir::{Ignore, MidiInput};

mod ui;

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
}

#[derive(Clone, Debug)]
struct Args {
    engine: Engine,
    port: Option<String>,
    list: bool,
    master: f32,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            engine: Engine::Square,
            port: None,
            list: false,
            master: 0.3,
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let mut out = Args::default();
    let mut iter = std::env::args().skip(1);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--list" => out.list = true,
            "--engine" => {
                let v = iter.next().ok_or("--engine needs a value (square|ks)")?;
                out.engine = match v.as_str() {
                    "square" => Engine::Square,
                    "ks" => Engine::Ks,
                    "ks-rich" => Engine::KsRich,
                    "sub" => Engine::Sub,
                    "fm" => Engine::Fm,
                    "piano" => Engine::Piano,
                    "koto" => Engine::Koto,
                    other => return Err(format!(
                        "unknown engine: {other} (square|ks|ks-rich|sub|fm|piano|koto)"
                    )),
                };
            }
            "--port" => {
                out.port = Some(iter.next().ok_or("--port needs a value")?);
            }
            "--master" => {
                let v = iter.next().ok_or("--master needs a float")?;
                out.master = v.parse().map_err(|e| format!("bad --master: {e}"))?;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    Ok(out)
}

fn print_help() {
    println!(
        "keysynth - real-time MIDI keyboard -> speaker synth\n\n\
         USAGE:\n  \
            keysynth [--engine square|ks] [--port NAME] [--master FLOAT]\n  \
            keysynth --list\n\n\
         OPTIONS:\n  \
            --engine ENGINE   square|ks|ks-rich|sub|fm|piano|koto\n  \
                              (sub = analog subtractive, fm = 2-op bell,\n  \
                               piano = wide-hammer DWS, koto = pluck-pos DWS)\n  \
            --port NAME       MIDI input port (default: first available)\n  \
            --master FLOAT    Master gain pre-tanh (default: 0.3)\n  \
            --list            List MIDI input ports and exit\n  \
            -h, --help        Show this help"
    );
}

// ---------------------------------------------------------------------------
// Voice trait + implementations
// ---------------------------------------------------------------------------

trait VoiceImpl: Send {
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

struct Voice {
    /// (channel, midi_note) - used to match note_off back to the voice
    key: (u8, u8),
    inner: Box<dyn VoiceImpl>,
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

struct SquareVoice {
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
    fn new(sr: f32, freq: f32, velocity: u8) -> Self {
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

struct KsVoice {
    buf: Vec<f32>,
    head: usize,
    decay: f32,
    released: bool,
    rel_mul: f32,
    rel_step: f32,
}

impl KsVoice {
    fn new(sr: f32, freq: f32, velocity: u8) -> Self {
        let n = ((sr / freq.max(1.0)).round() as usize).max(2);
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
            released: false,
            rel_mul: 1.0,
            rel_step,
        }
    }
}

impl VoiceImpl for KsVoice {
    fn render_add(&mut self, buf: &mut [f32]) {
        let n = self.buf.len();
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
            self.buf[self.head] = (cur + prev) * 0.5 * self.decay;
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

struct SubVoice {
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
    fn new(sr: f32, freq: f32, velocity: u8) -> Self {
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

struct FmVoice {
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
    fn new(sr: f32, freq: f32, velocity: u8) -> Self {
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

struct KsString {
    buf: Vec<f32>,
    head: usize,
    decay: f32,
    // 1-pole allpass state: y[n] = -a*x[n] + x[n-1] + a*y[n-1]
    ap_coef: f32,
    ap_xprev: f32,
    ap_yprev: f32,
}

impl KsString {
    fn new(sr: f32, freq: f32, amp: f32, ap_coef: f32, hammer_width: usize) -> Self {
        let n = ((sr / freq.max(1.0)).round() as usize).max(2);
        let buf = hammer_excitation(n, hammer_width, amp);
        Self::with_buf(buf, 0.997, ap_coef)
    }

    /// Construct with a fully-prepared excitation buffer and explicit decay.
    /// Used by `piano` (asymmetric hammer + freq-dependent decay) and `koto`
    /// (offset pluck + long sustain) which need control beyond `new`.
    fn with_buf(buf: Vec<f32>, decay: f32, ap_coef: f32) -> Self {
        Self {
            buf,
            head: 0,
            decay,
            ap_coef,
            ap_xprev: 0.0,
            ap_yprev: 0.0,
        }
    }

    fn delay_length(sr: f32, freq: f32) -> usize {
        ((sr / freq.max(1.0)).round() as usize).max(2)
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
        // Standard 2-tap lowpass average + decay
        let lp = (cur + prev) * 0.5 * self.decay;
        // 1-pole allpass for dispersion (stiffness)
        let a = self.ap_coef;
        let y = -a * lp + self.ap_xprev + a * self.ap_yprev;
        self.ap_xprev = lp;
        self.ap_yprev = y;
        self.buf[self.head] = y;
        self.head += 1;
        if self.head >= n {
            self.head = 0;
        }
        cur
    }
}

struct KsRichVoice {
    strings: [KsString; 3],
    released: bool,
    rel_mul: f32,
    rel_step: f32,
}

impl KsRichVoice {
    fn new(sr: f32, freq: f32, velocity: u8) -> Self {
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

fn hammer_width_for_velocity(vel: u8) -> usize {
    // vel=127 -> 3 samples (sharp click, very bright)
    // vel=20  -> ~19 samples (soft, mellow)
    let v = vel.max(1).min(127) as f32;
    (3.0 + (127.0 - v) * 0.13) as usize
}

fn hammer_excitation(n: usize, width: usize, amp: f32) -> Vec<f32> {
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

fn piano_hammer_width(vel: u8) -> usize {
    // vel=127 -> 20 samples (~0.45 ms): firm but felt
    // vel=20  -> 200 samples (~4.5 ms): very soft
    let v = vel.max(1).min(127) as f32;
    (20.0 + (127.0 - v) * 1.7) as usize
}

fn piano_hammer_excitation(n: usize, width: usize, amp: f32) -> Vec<f32> {
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

struct PianoVoice {
    strings: [KsString; 3],
    released: bool,
    rel_mul: f32,
    rel_step: f32,
}

impl PianoVoice {
    fn new(sr: f32, freq: f32, velocity: u8) -> Self {
        let amp = (velocity.max(1) as f32) / 127.0;
        let cents_to_ratio = |c: f32| 2.0_f32.powf(c / 1200.0);
        let detunes = [cents_to_ratio(-7.0), 1.0, cents_to_ratio(7.0)];
        // Stronger stiffness than ks-rich + scaled with frequency.
        let ap = (0.25 + (freq / 1500.0).min(0.30)).min(0.50);
        // Frequency-dependent decay: 110 Hz -> 0.9990 (long); 2000 Hz -> 0.9930 (short)
        let decay_for = |f: f32| -> f32 {
            let high = (f / 2000.0).clamp(0.0, 1.0);
            0.999 - 0.006 * high
        };
        let hammer_w = piano_hammer_width(velocity);
        let mk = |freq_string: f32| -> KsString {
            let n = KsString::delay_length(sr, freq_string);
            let buf = piano_hammer_excitation(n, hammer_w, amp);
            KsString::with_buf(buf, decay_for(freq_string), ap)
        };
        let strings = [
            mk(freq * detunes[0]),
            mk(freq * detunes[1]),
            mk(freq * detunes[2]),
        ];
        let release_sec = 0.300_f32;
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

impl VoiceImpl for PianoVoice {
    fn render_add(&mut self, buf: &mut [f32]) {
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

fn koto_pluck_excitation(n: usize, amp: f32, offset: usize) -> Vec<f32> {
    let mut buf = vec![0.0_f32; n];
    if n == 0 {
        return buf;
    }
    // Two-sample plectrum: a sharp positive spike followed by a smaller
    // negative tail (string snaps back). This gives the characteristic
    // "click" of fingernail/tsume contact.
    buf[offset % n] = amp;
    buf[(offset + 1) % n] = -amp * 0.3;
    buf
}

struct KotoVoice {
    string: KsString,
    released: bool,
    rel_mul: f32,
    rel_step: f32,
}

impl KotoVoice {
    fn new(sr: f32, freq: f32, velocity: u8) -> Self {
        let amp = (velocity.max(1) as f32) / 127.0;
        let n = KsString::delay_length(sr, freq);
        // Pluck near 1/4 of string length (typical koto tsume position)
        let pluck_offset = n / 4;
        let buf = koto_pluck_excitation(n, amp, pluck_offset);
        // Long sustain, very mild stiffness.
        let string = KsString::with_buf(buf, 0.9992, 0.05);
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

// xorshift32, thread-local. No extra crate.
// Kept available for future engines (noise oscillator, sample-and-hold, etc.)
#[allow(dead_code)]
fn fastrand_f32() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u32> = Cell::new(0x9E37_79B9);
    }
    STATE.with(|s| {
        let mut x = s.get();
        if x == 0 {
            x = 0x1234_5678;
        }
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        s.set(x);
        (x as f32) / (u32::MAX as f32)
    })
}

// ---------------------------------------------------------------------------

fn midi_to_freq(note: u8) -> f32 {
    440.0_f32 * 2.0_f32.powf((note as f32 - 69.0) / 12.0)
}

// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            print_help();
            std::process::exit(2);
        }
    };

    let mut midi_in = MidiInput::new("keysynth")?;
    midi_in.ignore(Ignore::None);

    let ports = midi_in.ports();
    if args.list {
        if ports.is_empty() {
            println!("(no MIDI input ports)");
        } else {
            for p in &ports {
                println!("{}", midi_in.port_name(p)?);
            }
        }
        return Ok(());
    }

    if ports.is_empty() {
        return Err("no MIDI input ports found - keyboard plugged in?".into());
    }

    let chosen_port = if let Some(want) = &args.port {
        ports
            .iter()
            .find(|p| midi_in.port_name(p).map(|n| n == *want).unwrap_or(false))
            .ok_or_else(|| {
                let available: Vec<String> = ports
                    .iter()
                    .filter_map(|p| midi_in.port_name(p).ok())
                    .collect();
                format!("port {want:?} not found. Available: {available:?}")
            })?
            .clone()
    } else {
        ports[0].clone()
    };
    let port_name = midi_in.port_name(&chosen_port)?;

    // -- Audio output --
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or("no default output device")?;
    let out_name = device.name().unwrap_or_else(|_| "<unknown>".into());

    let supported = device.default_output_config()?;
    let sample_format = supported.sample_format();
    let sr_hz = supported.sample_rate().0;
    let channels = supported.channels();
    let stream_cfg: StreamConfig = supported.into();

    eprintln!(
        "keysynth: midi='{port_name}' audio='{out_name}' sr={sr_hz} ch={channels} \
         engine={:?} master={:.2}",
        args.engine, args.master
    );
    eprintln!("press keys on your MIDI keyboard - Ctrl-C to quit");

    let voices: Arc<Mutex<Vec<Voice>>> = Arc::new(Mutex::new(Vec::with_capacity(64)));
    let live: Arc<Mutex<LiveParams>> = Arc::new(Mutex::new(LiveParams {
        master: args.master,
        engine: args.engine,
    }));
    let dash: Arc<Mutex<DashState>> = Arc::new(Mutex::new(DashState::new(args.engine)));

    let voices_for_midi = voices.clone();
    let live_for_midi = live.clone();
    let dash_for_midi = dash.clone();
    let sr_for_voices = sr_hz as f32;
    let _conn = midi_in.connect(
        &chosen_port,
        "keysynth-in",
        move |_stamp, raw, _| {
            if raw.len() < 3 {
                return;
            }
            let status = raw[0];
            let msg_type = status & 0xF0;
            let channel = status & 0x0F;
            let note = raw[1];
            let velocity = raw[2];

            match msg_type {
                0x90 if velocity > 0 => {
                    {
                        let mut d = dash_for_midi.lock().unwrap();
                        d.active_notes.insert((channel, note));
                        d.push_event(format!("note_on  ch{channel} n{note} v{velocity}"));
                    }
                    let freq = midi_to_freq(note);
                    // Read currently-selected engine fresh each note so GUI
                    // changes apply immediately to subsequent keypresses.
                    let engine = live_for_midi.lock().unwrap().engine;
                    let inner: Box<dyn VoiceImpl> = match engine {
                        Engine::Square => Box::new(SquareVoice::new(sr_for_voices, freq, velocity)),
                        Engine::Ks => Box::new(KsVoice::new(sr_for_voices, freq, velocity)),
                        Engine::KsRich => Box::new(KsRichVoice::new(sr_for_voices, freq, velocity)),
                        Engine::Sub => Box::new(SubVoice::new(sr_for_voices, freq, velocity)),
                        Engine::Fm => Box::new(FmVoice::new(sr_for_voices, freq, velocity)),
                        Engine::Piano => Box::new(PianoVoice::new(sr_for_voices, freq, velocity)),
                        Engine::Koto => Box::new(KotoVoice::new(sr_for_voices, freq, velocity)),
                    };
                    let v = Voice {
                        key: (channel, note),
                        inner,
                    };
                    let mut pool = voices_for_midi.lock().unwrap();
                    if let Some(slot) = pool.iter_mut().find(|x| x.key == (channel, note)) {
                        *slot = v;
                    } else {
                        // Hard cap voice pool to bound CPU/memory growth under
                        // sustained MIDI input. When at cap, prefer evicting an
                        // already-released voice; fall back to the oldest entry.
                        const MAX_VOICES: usize = 32;
                        if pool.len() >= MAX_VOICES {
                            let evict_idx = pool
                                .iter()
                                .position(|x| x.inner.is_done() || x.inner.is_releasing())
                                .unwrap_or(0);
                            pool.remove(evict_idx);
                        }
                        pool.push(v);
                    }
                }
                0x80 | 0x90 => {
                    // Note off, or note_on with velocity 0 (running-status note off)
                    {
                        let mut d = dash_for_midi.lock().unwrap();
                        d.active_notes.remove(&(channel, note));
                        d.push_event(format!("note_off ch{channel} n{note}"));
                    }
                    let mut pool = voices_for_midi.lock().unwrap();
                    if let Some(slot) = pool.iter_mut().find(|x| x.key == (channel, note)) {
                        slot.inner.trigger_release();
                    }
                }
                0xB0 => {
                    // Control Change. raw[1] = CC number, raw[2] = value (0..127).
                    let cc_num = note;       // (raw[1])
                    let cc_val = velocity;   // (raw[2])
                    {
                        let mut d = dash_for_midi.lock().unwrap();
                        d.cc_raw.insert(cc_num, cc_val);
                        *d.cc_count.entry(cc_num).or_insert(0) += 1;
                        d.push_event(format!("CC{cc_num}={cc_val} ch{channel}"));
                    }
                    eprintln!("midi: CC{cc_num}={cc_val} (ch={channel})");

                    // MPK mini 3 K1-K8 are ROTARY ENCODERS in relative mode:
                    //   1..63   = +N step(s) clockwise
                    //   65..127 = -N step(s) counter-clockwise (encoded as 128-N)
                    // Standard MIDI Volume (CC 7) is absolute (a "real" pot).
                    let delta_ticks: i32 = match cc_val {
                        0 | 64 => 0,
                        1..=63 => cc_val as i32,
                        65..=127 => -((128 - cc_val as i32)),
                        _ => 0, // MIDI CC values are 0..127 in spec; defensive
                    };

                    match cc_num {
                        // CC 7 = absolute MIDI Volume (rare on MPK).
                        7 => {
                            let new_master = (cc_val as f32 / 127.0) * 2.0;
                            live_for_midi.lock().unwrap().master = new_master;
                            eprintln!("    -> master = {new_master:.3} (absolute)");
                        }
                        // CC 70 = MPK K1 (relative encoder by default).
                        // 1 tick = +/- 0.05 master gain, clamped 0..3.0.
                        70 => {
                            let mut p = live_for_midi.lock().unwrap();
                            p.master = (p.master + delta_ticks as f32 * 0.05).clamp(0.0, 3.0);
                            eprintln!("    -> master = {:.3} (delta={delta_ticks:+})", p.master);
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        },
        (),
    )?;

    let voices_for_audio = voices.clone();
    let live_for_audio = live.clone();
    let err_fn = |err| eprintln!("audio stream error: {err}");

    // Build an F32 output stream. Used both for the native F32 path and as a
    // fallback when cpal reports a non-{F32,I16,U16} format we don't handle.
    let build_f32_stream = |dev: &cpal::Device, cfg: &StreamConfig| {
        let voices_arc = voices_for_audio.clone();
        let live_arc = live_for_audio.clone();
        // Preallocated mono scratch reused across audio callbacks. Sized lazily
        // on the first callback once we know the buffer size; cpal callbacks
        // are FnMut so capturing `mut` state is fine.
        let mut mono_scratch: Vec<f32> = Vec::new();
        dev.build_output_stream(
            cfg,
            move |out: &mut [f32], _| {
                let master = live_arc.lock().unwrap().master;
                audio_callback(out, channels, &voices_arc, master, &mut mono_scratch);
            },
            err_fn,
            None,
        )
    };

    let stream = match sample_format {
        SampleFormat::F32 => build_f32_stream(&device, &stream_cfg)?,
        SampleFormat::I16 => {
            let voices_arc = voices_for_audio.clone();
            let live_arc = live_for_audio.clone();
            let mut mono_scratch: Vec<f32> = Vec::new();
            let mut interleaved_scratch: Vec<f32> = Vec::new();
            device.build_output_stream(
                &stream_cfg,
                move |out: &mut [i16], _| {
                    let master = live_arc.lock().unwrap().master;
                    if interleaved_scratch.len() != out.len() {
                        interleaved_scratch.resize(out.len(), 0.0);
                    }
                    audio_callback(
                        &mut interleaved_scratch,
                        channels,
                        &voices_arc,
                        master,
                        &mut mono_scratch,
                    );
                    for (dst, &src) in out.iter_mut().zip(interleaved_scratch.iter()) {
                        let clamped = src.max(-1.0).min(1.0);
                        *dst = (clamped * i16::MAX as f32) as i16;
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let voices_arc = voices_for_audio.clone();
            let live_arc = live_for_audio.clone();
            let mut mono_scratch: Vec<f32> = Vec::new();
            let mut interleaved_scratch: Vec<f32> = Vec::new();
            device.build_output_stream(
                &stream_cfg,
                move |out: &mut [u16], _| {
                    let master = live_arc.lock().unwrap().master;
                    if interleaved_scratch.len() != out.len() {
                        interleaved_scratch.resize(out.len(), 0.0);
                    }
                    audio_callback(
                        &mut interleaved_scratch,
                        channels,
                        &voices_arc,
                        master,
                        &mut mono_scratch,
                    );
                    for (dst, &src) in out.iter_mut().zip(interleaved_scratch.iter()) {
                        let clamped = src.max(-1.0).min(1.0);
                        let unsigned = ((clamped + 1.0) * 0.5 * u16::MAX as f32) as u16;
                        *dst = unsigned;
                    }
                },
                err_fn,
                None,
            )?
        }
        other => {
            // cpal's SampleFormat is #[non_exhaustive]; modern backends may
            // hand back I32/I8/U8/F64/etc. Try opening the device with an F32
            // stream regardless - many backends will accept it - and surface
            // the original error only if that also fails.
            eprintln!(
                "keysynth: sample format {other:?} not natively handled, \
                 attempting F32 fallback"
            );
            match build_f32_stream(&device, &stream_cfg) {
                Ok(s) => s,
                Err(e) => {
                    return Err(format!(
                        "unsupported sample format: {other:?} (F32 fallback also failed: {e})"
                    )
                    .into());
                }
            }
        }
    };
    stream.play()?;

    // Launch egui dashboard. The cpal Stream and midir InputConnection are
    // moved into the App struct so they live as long as the GUI window.
    ui::run_app(ui::AppContext {
        stream,
        midi_conn: _conn,
        live,
        dash,
        port_name,
        out_name,
        sr_hz,
    })?;

    eprintln!("keysynth: stopping");
    Ok(())
}

fn audio_callback(
    out: &mut [f32],
    channels: u16,
    voices: &Arc<Mutex<Vec<Voice>>>,
    master: f32,
    mono: &mut Vec<f32>,
) {
    let frames = out.len() / channels as usize;
    // Reuse the caller-owned scratch buffer to avoid per-callback heap
    // allocation on the audio thread (~every 11 ms at 1024 frames / 48 kHz).
    if mono.len() != frames {
        mono.resize(frames, 0.0);
    } else {
        mono.fill(0.0);
    }

    {
        let mut pool = voices.lock().unwrap();
        for v in pool.iter_mut() {
            v.inner.render_add(mono.as_mut_slice());
        }
        pool.retain(|v| !v.inner.is_done());
    }

    for sample in mono.iter_mut() {
        *sample = (*sample * master).tanh();
    }

    for (frame_idx, frame) in out.chunks_mut(channels as usize).enumerate() {
        let s = mono[frame_idx];
        for slot in frame.iter_mut() {
            *slot = s;
        }
    }
}
