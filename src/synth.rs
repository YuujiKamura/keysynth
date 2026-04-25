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
//!
//! Per-engine voice impls live under `crate::voices::*` (issue #2 split).
//! This module keeps the shared primitives (`ReleaseEnvelope`, `Adsr`,
//! `KsString`, hammer/pluck excitations) plus the `Engine` enum, the
//! `VoiceImpl` trait, and the `make_voice` factory; voice types are
//! re-exported below so external callers (`main.rs`, `bench`) can keep
//! importing them via `keysynth::synth::*`.

use std::collections::{HashMap, HashSet, VecDeque};

pub mod voices {
    //! Re-exported for callers that prefer `keysynth::synth::voices::...`.
    pub use crate::voices::*;
}

// Public re-exports of the voice impls so `keysynth::synth::SquareVoice`
// etc. keep resolving for `main.rs` / `bench` / external tests.
pub use crate::voices::fm::FmVoice;
pub use crate::voices::koto::{koto_pluck_excitation, KotoVoice};
pub use crate::voices::ks::KsVoice;
pub use crate::voices::ks_rich::KsRichVoice;
pub use crate::voices::piano::{PianoPreset, PianoVoice, SoundboardKind};
pub use crate::voices::placeholder::SfPianoPlaceholder;
pub use crate::voices::square::SquareVoice;
pub use crate::voices::sub::SubVoice;

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
    /// SFZ-driven sampler piano: routes (channel, note, velocity) to a
    /// shared `SfzPlayer` loaded from a `.sfz` manifest at startup. Same
    /// placeholder pattern as SfPiano — audio is rendered by the shared
    /// player in the audio callback. Intended for high-quality sample
    /// libraries (Salamander Grand V3 etc.) that the SF2 format can't
    /// represent at full fidelity.
    SfzPiano,
    /// Mid-tier piano: 7 strings + 12-mode "lite" soundboard, no
    /// shared sympathetic bank. Same per-voice DSP as `Piano`, but
    /// uses the lite soundboard so the high-Q upper-band modes don't
    /// ring through the bridge feedback loop. Decay/coupling/dry-wet
    /// tuned 2026-04-25 against the SFZ Salamander Grand Piano V3 C4
    /// reference T60 vector.
    PianoThick,
    /// Most ref-faithful variant: 3 strings + 12-mode lite soundboard,
    /// no sym bank, decay/coupling tuned ("v4") to land h1 T60 within
    /// 8% of the SFZ Salamander C4 ground truth (16.68 s vs ref 18.14 s).
    /// Mid-band partials h2-h5 also within 30%. Best per-partial T60
    /// match across all variants per the 2026-04-25 measurement loop.
    PianoLite,
}

impl Engine {
    /// True if this engine is one of the KS-string-based piano models
    /// (`Piano`, `PianoThick`, `PianoLite`). Sample-based engines (`SfPiano`,
    /// `SfzPiano`) and non-piano engines (`Square`, `Ks`, `Sub`, etc.) return
    /// false. Used by the audio callback to gate the shared sympathetic
    /// string bank — sample-based engines already include body resonance in
    /// the recording, so adding sympathetic on top double-resonates.
    pub fn is_piano_family(self) -> bool {
        matches!(self, Engine::Piano | Engine::PianoThick | Engine::PianoLite)
    }
}

// ---------------------------------------------------------------------------
// ReleaseEnvelope: shared multiplicative exponential release helper.
// ---------------------------------------------------------------------------
//
// History: every multiplicative-release voice (KS, KS-rich, Piano, Koto,
// PianoThick, SfPianoPlaceholder) re-rolled the same `released: bool +
// rel_mul: f32 + rel_step: f32` triplet plus identical `trigger_release` /
// `is_done` / `is_releasing` impls. ReleaseEnvelope centralises that so
// changes (e.g. done_threshold tuning) happen in one place and so it is
// impossible for a voice to forget to override `is_releasing` (the
// pre-refactor cause of "no sound after pressing many keys" — the voice
// pool's eviction policy couldn't find released voices because most
// VoiceImpls fell back to the trait default `is_releasing() == false`).

/// Multiplicative exponential release envelope shared by every voice.
/// Encapsulates the released-flag + rel_mul accumulator + rel_step
/// per-sample multiplier so individual voice impls don't re-roll it.
pub struct ReleaseEnvelope {
    released: bool,
    rel_mul: f32,
    rel_step: f32,
    /// Threshold under which `is_done()` returns true. Keeps the
    /// per-voice "voice can be evicted" check uniform across types.
    done_threshold: f32,
}

impl ReleaseEnvelope {
    /// `release_sec` is the target -60dB time once `trigger()` is called.
    /// Picks a per-sample multiplicative step that gets us there.
    pub fn new(release_sec: f32, sr: f32) -> Self {
        let release_samples = (release_sec * sr).max(1.0);
        // exp(ln(eps)/N) so we hit ~-60dB after release_samples samples.
        let rel_step = (1e-3_f32.ln() / release_samples).exp();
        Self {
            released: false,
            rel_mul: 1.0,
            rel_step,
            done_threshold: 1e-4,
        }
    }

    /// Construct with an explicit done-threshold (default is 1e-4).
    /// Used by SfPianoPlaceholder which historically used a fixed-time
    /// eviction window rather than amplitude-based.
    pub fn with_done_threshold(release_sec: f32, sr: f32, done_threshold: f32) -> Self {
        let mut env = Self::new(release_sec, sr);
        env.done_threshold = done_threshold;
        env
    }

    /// Mark released. Subsequent `step()` calls will multiplicatively
    /// decay rel_mul.
    pub fn trigger(&mut self) {
        self.released = true;
    }

    /// Per-sample tick. Call once per output sample. Returns the current
    /// envelope multiplier — voices apply this to their own output.
    /// While not released, returns 1.0; after release, decays toward 0.
    #[inline]
    pub fn step(&mut self) -> f32 {
        if self.released {
            self.rel_mul *= self.rel_step;
        }
        self.rel_mul
    }

    /// Current multiplier without advancing. Useful for tests.
    #[inline]
    pub fn current(&self) -> f32 {
        self.rel_mul
    }

    /// True once the release decay has reached `done_threshold`.
    pub fn is_done(&self) -> bool {
        self.released && self.rel_mul <= self.done_threshold
    }

    /// True if the voice has been released and is just ringing out.
    /// Used by main.rs voice eviction.
    pub fn is_releasing(&self) -> bool {
        self.released
    }
}

// ---------------------------------------------------------------------------
// Voice trait
// ---------------------------------------------------------------------------

pub trait VoiceImpl: Send {
    /// Add this voice's contribution into `buf` (mono, length = frames).
    fn render_add(&mut self, buf: &mut [f32]);

    /// Optional reference to the shared `ReleaseEnvelope`. Voices that
    /// use the multiplicative-release pattern return `Some` so the
    /// default `trigger_release`/`is_done`/`is_releasing` impls below
    /// just delegate. Voices with custom envelope shapes (linear AR,
    /// ADSR) return `None` and override the trait methods directly.
    fn release_env(&self) -> Option<&ReleaseEnvelope> {
        None
    }
    fn release_env_mut(&mut self) -> Option<&mut ReleaseEnvelope> {
        None
    }

    /// Mark released; envelope should fade and `is_done()` will go true.
    fn trigger_release(&mut self) {
        if let Some(env) = self.release_env_mut() {
            env.trigger();
        }
    }
    fn is_done(&self) -> bool {
        self.release_env().map(|e| e.is_done()).unwrap_or(false)
    }
    /// True if the voice has been released but is still ringing out.
    /// Critical for the voice-pool eviction policy: when at the cap the
    /// pool prefers to drop releasing voices over still-sustained ones.
    /// Pre-refactor most voices forgot to override this and returned
    /// `false`, which caused new notes to evict slot-0 (often a still-
    /// sounding note) instead of an actually-releasing voice.
    fn is_releasing(&self) -> bool {
        self.release_env()
            .map(|e| e.is_releasing())
            .unwrap_or(false)
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
    /// SoundFont GM program (0..=127). Used by the `sf-piano` engine to pick
    /// which patch the shared `rustysynth::Synthesizer` plays. The MIDI
    /// callback diffs this against the last applied value before each
    /// note_on and emits a Program Change (and Bank Select if needed)
    /// only on actual change, so per-note overhead stays at one comparison.
    pub sf_program: u8,
    /// SoundFont bank select (CC 0 MSB). 0 = melodic GM, 128 = drum kit
    /// for GeneralUser GS / many GM2/GS soundfonts. Other values are
    /// SF2-specific variation banks. Diffed alongside `sf_program`.
    pub sf_bank: u8,
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

// ---------------------------------------------------------------------------
// Shared: 4-stage ADSR envelope (used by SubVoice + FmVoice).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AdsrStage {
    Attack,
    Decay,
    Sustain,
    Release,
    Done,
}

pub(crate) struct Adsr {
    sr: f32,
    attack_s: f32,
    decay_s: f32,
    sustain: f32, // 0..1 sustain level
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
    pub(crate) fn new(sr: f32, attack_s: f32, decay_s: f32, sustain: f32, release_s: f32) -> Self {
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

    pub(crate) fn release(&mut self) {
        if self.value <= 1e-6 {
            // Nothing to fade; mark done so the voice gets cleaned up.
            self.stage = AdsrStage::Done;
        } else {
            self.release_start_value = self.value;
            self.stage = AdsrStage::Release;
        }
    }

    pub(crate) fn next(&mut self) -> f32 {
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

    pub(crate) fn done(&self) -> bool {
        matches!(self.stage, AdsrStage::Done)
    }

    pub(crate) fn is_releasing(&self) -> bool {
        matches!(self.stage, AdsrStage::Release | AdsrStage::Done)
    }
}

// ---------------------------------------------------------------------------
// KsString: shared single-delay-line primitive used by every plucked-string
// voice (Ks/KsRich/Piano/Koto) and by the SympatheticBank.
// ---------------------------------------------------------------------------

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
    // Optional second 1-pole allpass placed in series with the first to give
    // a 2-stage dispersion network (issue #3 P2). A 2nd-order (or two
    // cascaded 1st-order) allpass has two coefficients, so it can match the
    // measured stretched-harmonic curve of a real piano string across the
    // audible range — a 1-stage section can only produce a single
    // phase-delay-vs-frequency shape and that shape doesn't fit piano
    // stiffness (Fletcher / Smith PASP).
    //
    // Backward compatibility: `ap2_coef == 0.0` is the legacy 1-stage mode
    // and the second section is BYPASSED entirely in `step` (no extra
    // 1-sample delay, no state update). The legacy `with_buf` /
    // `KsString::new` constructors leave `ap2_coef = 0.0`, so every
    // pre-existing call site (PianoVoice, KotoVoice, KsRichVoice, the
    // SympatheticBank) keeps producing bit-identical output.
    ap2_coef: f32,
    ap2_xprev: f32,
    ap2_yprev: f32,
    // Tuning 1-pole allpass for the fractional-delay component, same form.
    // c = (1 - frac) / (1 + frac), giving group delay ~= frac at low freq.
    // Total effective delay is buf.len() + frac.
    tune_coef: f32,
    tune_xprev: f32,
    tune_yprev: f32,
    // Fix B: time-varying KS LPF strength. The 2-tap weighted average inside
    // `step` is implemented as `lp = cur*lpf_w0 + prev*lpf_w1` where
    // (lpf_w0, lpf_w1) interpolates from (attack, steady) over the first
    // `attack_total_samples` samples after note_on. Attack favours `cur`
    // heavily (preserves highs); steady is the gentler average that yields
    // natural piano-like rolloff.
    lpf_w0_attack: f32,
    lpf_w0_steady: f32,
    attack_samples_remaining: u32,
    attack_total_samples: u32,
    /// One-sample additive feedback register. Anything written here via
    /// `inject_feedback` is summed into the in-loop LPF output (`lp`)
    /// during the next `step`, then cleared. Used by `PianoVoice` to
    /// route soundboard output back into the string via the bridge.
    pending_feedback: f32,
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
        // Default attack envelope on LPF strength: 80 ms preserving highs.
        // The piano voice rebuilds with a different attack profile via
        // `with_attack_lpf`; other callers (ks-rich, koto) get the steady
        // value throughout (attack window length = 0).
        let attack_total_samples = 0;
        Self {
            buf,
            head: 0,
            decay,
            ap_coef,
            ap_xprev: 0.0,
            ap_yprev: 0.0,
            // 2-stage dispersion is OPT-IN via `with_two_stage_dispersion`.
            // `ap2_coef == 0.0` is the bypass sentinel: the second section
            // is skipped in `step` entirely (not run with a=0, which would
            // be a 1-sample delay and detune the loop).
            ap2_coef: 0.0,
            ap2_xprev: 0.0,
            ap2_yprev: 0.0,
            tune_coef,
            tune_xprev: 0.0,
            tune_yprev: 0.0,
            lpf_w0_attack: 0.97,
            lpf_w0_steady: 0.97,
            attack_samples_remaining: attack_total_samples,
            attack_total_samples,
            pending_feedback: 0.0,
        }
    }

    /// Engage a SECOND 1-pole allpass in series with the primary dispersion
    /// allpass (issue #3 P2). Calling with `a2 = 0.0` is a no-op: the second
    /// section stays bypassed in `step` and the string is bit-identical to
    /// the legacy 1-stage configuration. With `a2 != 0.0` the loop gains the
    /// extra degree of freedom needed to match measured piano-string
    /// stiffness curves (B coefficient ≈ 2.86e-4 per the SFZ Salamander C4
    /// reference) — voices that want this behaviour set both `a1` (via the
    /// constructor) and `a2` here.
    ///
    /// NOTE: this method does NOT recompute the fractional-delay
    /// compensation. Adding a second allpass shifts the loop's group delay
    /// at low frequency by `(1 - a2) / (1 + a2)` samples, so callers that
    /// care about fundamental-pitch accuracy will need a P3 follow-up to
    /// fold that into `delay_length_compensated`. For P2 we're only
    /// validating the topology — voices keep using the 1-stage call path.
    pub fn with_two_stage_dispersion(mut self, _a1: f32, a2: f32) -> Self {
        // `_a1` is accepted for API symmetry; the primary coefficient was
        // already set by the constructor. Asserting equality would force
        // callers to pass it twice, so we just consume the argument.
        self.ap2_coef = a2;
        self.ap2_xprev = 0.0;
        self.ap2_yprev = 0.0;
        self
    }

    /// Add `x` to the one-sample feedback register. Called BEFORE `step`
    /// each sample by `PianoVoice` so the soundboard's previous-sample
    /// output enters the next loop iteration. Additive across multiple
    /// callers per sample; cleared inside `step` after consumption.
    #[inline]
    pub fn inject_feedback(&mut self, x: f32) {
        self.pending_feedback += x;
    }

    /// Fix B: configure a per-voice attack envelope on the in-loop LPF
    /// strength. Over the first `attack_ms` ms after the voice starts,
    /// the `cur`-tap weight interpolates linearly from `w0_attack` (high
    /// cur weight = preserves brilliance) toward `w0_steady` (the natural
    /// settled rolloff). Should be called immediately after construction.
    pub fn with_attack_lpf(
        mut self,
        sr: f32,
        attack_ms: f32,
        w0_attack: f32,
        w0_steady: f32,
    ) -> Self {
        let total = (sr * attack_ms / 1000.0) as u32;
        self.lpf_w0_attack = w0_attack;
        self.lpf_w0_steady = w0_steady;
        self.attack_samples_remaining = total;
        self.attack_total_samples = total;
        self
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

    /// Advance the delay line one sample, return the sample that just left
    /// the line (i.e. the plain KS string output). Consumes any pending
    /// feedback injected via `inject_feedback` before this call.
    ///
    /// Exposed `pub` so out-of-crate callers (e.g. the shared
    /// `SympatheticBank`) can drive a bare KS string without going through
    /// a `PianoVoice`.
    #[inline]
    pub fn step(&mut self) -> f32 {
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
        //
        // Fix B: time-varying LPF — during the attack window, `w0` is
        // closer to 1.0 (minimal LPF, preserves highs). Fades linearly
        // to `w0_steady` over `attack_total_samples`.
        let w0 = if self.attack_samples_remaining > 0 && self.attack_total_samples > 0 {
            let t =
                1.0 - (self.attack_samples_remaining as f32) / (self.attack_total_samples as f32);
            self.attack_samples_remaining -= 1;
            self.lpf_w0_attack + (self.lpf_w0_steady - self.lpf_w0_attack) * t
        } else {
            self.lpf_w0_steady
        };
        let w1 = 1.0 - w0;
        let mut lp = (cur * w0 + prev * w1) * self.decay;
        // Bridge feedback: sum any externally-injected sample (e.g. from
        // the soundboard) into the post-LPF point of the loop, then
        // clear the register. Done BEFORE dispersion so the feedback
        // experiences the same downstream filter chain as the string
        // signal itself — this is where the bridge sits in the physical
        // model.
        if self.pending_feedback != 0.0 {
            lp += self.pending_feedback;
            self.pending_feedback = 0.0;
        }
        // Dispersion 1-pole allpass (stiffness): textbook Smith form.
        //   y[n] = a * x[n] + x[n-1] - a * y[n-1]
        let a = self.ap_coef;
        let disp = a * lp + self.ap_xprev - a * self.ap_yprev;
        self.ap_xprev = lp;
        self.ap_yprev = disp;
        // Optional 2nd dispersion allpass in series with the first (P2).
        // `ap2_coef == 0.0` is the bypass sentinel and we skip the section
        // entirely — running an a=0 allpass would inject a free 1-sample
        // delay and shift the loop period, breaking tuning. This branch
        // costs one comparison in the legacy 1-stage path (≤ 1 % CPU).
        let disp = if self.ap2_coef != 0.0 {
            let a2 = self.ap2_coef;
            let d2 = a2 * disp + self.ap2_xprev - a2 * self.ap2_yprev;
            self.ap2_xprev = disp;
            self.ap2_yprev = d2;
            d2
        } else {
            disp
        };
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
        let t = i as f32 / denom; // 0..1
        let val = 0.5 * (1.0 - (TAU * t).cos()); // Hann: 0 -> 1 -> 0
        buf[i] = val * amp;
    }
    buf
}

// ---------------------------------------------------------------------------
// Piano hammer: ms-scale asymmetric (Hann rise + exponential decay).
// ---------------------------------------------------------------------------

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
    // Asymmetric envelope: Hann-shaped rise (1/4), exponential decay (3/4).
    // Was linear rise which has a slope corner at i=0 (silence -> linear
    // ramp = discontinuous first derivative = audible broadband click).
    // Hann half-window (0.5*(1-cos(pi*t))) starts at zero VALUE AND zero
    // SLOPE, so the onset is C1-continuous and the click disappears.
    let rise = (w / 4).max(1);
    let decay_n = w - rise;
    for i in 0..rise {
        let t = i as f32 / rise as f32;
        let hann = 0.5 * (1.0 - (std::f32::consts::PI * t).cos());
        buf[i] = hann * amp;
    }
    let tau = (decay_n as f32 / 3.0).max(1.0); // ~95% decayed at end
    for i in 0..decay_n {
        let t = i as f32;
        buf[rise + i] = (-t / tau).exp() * amp;
    }
    buf
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
pub fn make_voice(engine: Engine, sr: f32, freq: f32, velocity: u8) -> Box<dyn VoiceImpl + Send> {
    match engine {
        Engine::Square => Box::new(SquareVoice::new(sr, freq, velocity)),
        Engine::Ks => Box::new(KsVoice::new(sr, freq, velocity)),
        Engine::KsRich => Box::new(KsRichVoice::new(sr, freq, velocity)),
        Engine::Sub => Box::new(SubVoice::new(sr, freq, velocity)),
        Engine::Fm => Box::new(FmVoice::new(sr, freq, velocity)),
        Engine::Piano => Box::new(PianoVoice::with_preset(
            PianoPreset::PIANO,
            sr,
            freq,
            velocity,
        )),
        Engine::Koto => Box::new(KotoVoice::new(sr, freq, velocity)),
        Engine::SfPiano => Box::new(SfPianoPlaceholder::new(sr)),
        // SfzPiano shares the placeholder voice strategy with SfPiano: the
        // pool entry only tracks (channel, note) for eviction; audio comes
        // from the shared SfzPlayer rendered in the audio callback.
        Engine::SfzPiano => Box::new(SfPianoPlaceholder::new(sr)),
        Engine::PianoThick => Box::new(PianoVoice::with_preset(
            PianoPreset::PIANO_THICK,
            sr,
            freq,
            velocity,
        )),
        Engine::PianoLite => Box::new(PianoVoice::with_preset(
            PianoPreset::PIANO_LITE,
            sr,
            freq,
            velocity,
        )),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voices::piano::piano_detunes;

    const SR: f32 = 44_100.0;

    fn nonzero_peak(buf: &[f32]) -> f32 {
        buf.iter().fold(0.0_f32, |a, &x| a.max(x.abs()))
    }

    fn render_n(voice: &mut dyn VoiceImpl, frames: usize) -> Vec<f32> {
        let mut buf = vec![0.0_f32; frames];
        voice.render_add(&mut buf);
        buf
    }

    // --- ReleaseEnvelope -------------------------------------------------

    #[test]
    fn release_env_starts_unity_pre_trigger() {
        let env = ReleaseEnvelope::new(0.1, SR);
        assert_eq!(env.current(), 1.0);
        assert!(!env.is_releasing());
        assert!(!env.is_done());
    }

    #[test]
    fn release_env_step_pre_trigger_stays_unity() {
        let mut env = ReleaseEnvelope::new(0.1, SR);
        for _ in 0..10_000 {
            assert_eq!(env.step(), 1.0);
        }
    }

    #[test]
    fn release_env_trigger_marks_releasing() {
        let mut env = ReleaseEnvelope::new(0.1, SR);
        env.trigger();
        assert!(env.is_releasing());
    }

    #[test]
    fn release_env_decays_after_trigger() {
        let mut env = ReleaseEnvelope::new(0.1, SR);
        env.trigger();
        let mut prev = 1.0_f32;
        for _ in 0..100 {
            let v = env.step();
            assert!(
                v < prev || v == 0.0,
                "envelope should be monotonically decreasing"
            );
            prev = v;
        }
    }

    #[test]
    fn release_env_is_done_after_full_release_window() {
        let release_sec = 0.05_f32;
        let mut env = ReleaseEnvelope::new(release_sec, SR);
        env.trigger();
        // The envelope hits ~-60dB after release_samples; done_threshold is
        // 1e-4 (-80dB). Run for 2× the release window to be safe.
        let n = (release_sec * SR * 2.0) as usize;
        for _ in 0..n {
            let _ = env.step();
        }
        assert!(
            env.is_done(),
            "envelope should be done after 2x release window"
        );
    }

    #[test]
    fn release_env_not_done_pre_trigger() {
        let mut env = ReleaseEnvelope::new(0.05, SR);
        for _ in 0..(SR as usize) {
            let _ = env.step();
        }
        assert!(!env.is_done());
    }

    #[test]
    fn release_env_with_done_threshold_overrides() {
        let mut env = ReleaseEnvelope::with_done_threshold(0.05, SR, 0.5);
        env.trigger();
        // With threshold 0.5 we should reach done quickly (well before -60dB).
        let mut steps = 0;
        for _ in 0..10_000 {
            let _ = env.step();
            steps += 1;
            if env.is_done() {
                break;
            }
        }
        assert!(env.is_done(), "high threshold should trigger done quickly");
        assert!(steps < 1000, "should be done in <1000 steps, took {steps}");
    }

    // --- midi_to_freq ----------------------------------------------------

    #[test]
    fn midi_to_freq_a4() {
        assert!((midi_to_freq(69) - 440.0).abs() < 1e-3);
    }

    #[test]
    fn midi_to_freq_c4() {
        // C4 = MIDI 60. 440 / 2^(9/12) = ~261.626 Hz.
        let f = midi_to_freq(60);
        assert!((f - 261.6256).abs() < 0.01, "expected ~261.63, got {f}");
    }

    #[test]
    fn midi_to_freq_octave_above_a4() {
        let f = midi_to_freq(81); // A5
        assert!((f - 880.0).abs() < 1e-3, "expected 880, got {f}");
    }

    #[test]
    fn midi_to_freq_octave_below_a4() {
        let f = midi_to_freq(57); // A3
        assert!((f - 220.0).abs() < 1e-3, "expected 220, got {f}");
    }

    // --- hammer excitations ----------------------------------------------

    #[test]
    fn hammer_width_decreasing_with_velocity() {
        let w_soft = hammer_width_for_velocity(20);
        let w_hard = hammer_width_for_velocity(127);
        assert!(
            w_hard < w_soft,
            "hard hits should have narrower contact ({w_hard} < {w_soft})"
        );
    }

    #[test]
    fn hammer_width_clamps_zero_velocity() {
        // vel=0 is clamped to 1 internally; should give the widest pulse.
        let w0 = hammer_width_for_velocity(0);
        let w1 = hammer_width_for_velocity(1);
        assert_eq!(w0, w1);
    }

    #[test]
    fn hammer_excitation_length_matches_n() {
        let buf = hammer_excitation(100, 5, 1.0);
        assert_eq!(buf.len(), 100);
    }

    #[test]
    fn hammer_excitation_starts_zero() {
        let buf = hammer_excitation(100, 10, 1.0);
        // Hann starts at 0
        assert!(
            buf[0].abs() < 1e-6,
            "Hann hammer should start at 0, got {}",
            buf[0]
        );
    }

    #[test]
    fn hammer_excitation_ends_zero_after_window() {
        let buf = hammer_excitation(100, 10, 1.0);
        // After the 10-sample window, all subsequent samples are zero.
        for v in buf.iter().skip(10) {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn hammer_excitation_peak_within_amp() {
        let amp = 0.5_f32;
        let buf = hammer_excitation(100, 20, amp);
        let p = nonzero_peak(&buf);
        assert!(p <= amp + 1e-6, "peak {p} exceeds amp {amp}");
        assert!(p > amp * 0.5, "peak {p} too small for amp {amp}");
    }

    #[test]
    fn hammer_excitation_unit_width_is_impulse() {
        let buf = hammer_excitation(10, 1, 0.7);
        assert_eq!(buf[0], 0.7);
        for v in buf.iter().skip(1) {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn piano_hammer_width_decreasing_with_velocity() {
        let w_soft = piano_hammer_width(20);
        let w_hard = piano_hammer_width(127);
        assert!(w_hard < w_soft);
        // Piano hammer is significantly wider than the generic one.
        assert!(w_hard >= 20);
        assert!(w_soft >= 100);
    }

    #[test]
    fn piano_hammer_excitation_length() {
        let buf = piano_hammer_excitation(500, 100, 1.0);
        assert_eq!(buf.len(), 500);
    }

    #[test]
    fn piano_hammer_excitation_starts_zero() {
        let buf = piano_hammer_excitation(500, 100, 1.0);
        assert!(buf[0].abs() < 1e-6);
    }

    #[test]
    fn piano_hammer_excitation_unit_width() {
        let buf = piano_hammer_excitation(10, 1, 0.5);
        assert_eq!(buf[0], 0.5);
    }

    #[test]
    fn piano_hammer_excitation_decay_after_rise() {
        // Find the peak; everything after should be decreasing.
        let buf = piano_hammer_excitation(200, 80, 1.0);
        let (peak_idx, _) =
            buf.iter()
                .enumerate()
                .fold((0usize, 0.0_f32), |(i_max, v_max), (i, &v)| {
                    if v.abs() > v_max {
                        (i, v.abs())
                    } else {
                        (i_max, v_max)
                    }
                });
        // After peak: monotonic decay (allow 1 sample tolerance).
        let mut prev = buf[peak_idx];
        for v in buf.iter().skip(peak_idx + 1).take(50) {
            assert!(*v <= prev + 1e-6, "decay should be monotonic past peak");
            prev = *v;
        }
    }

    #[test]
    fn koto_pluck_excitation_length() {
        let buf = koto_pluck_excitation(40, 1.0, 10);
        assert_eq!(buf.len(), 40);
    }

    #[test]
    fn koto_pluck_excitation_zero_n() {
        let buf = koto_pluck_excitation(0, 1.0, 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn koto_pluck_excitation_has_positive_and_negative() {
        let buf = koto_pluck_excitation(40, 1.0, 10);
        assert!(buf.iter().any(|&x| x > 0.0), "should have positive samples");
        assert!(buf.iter().any(|&x| x < 0.0), "should have negative samples");
    }

    #[test]
    fn koto_pluck_excitation_amplitude_at_zero() {
        let buf = koto_pluck_excitation(40, 0.8, 10);
        assert!((buf[0] - 0.8).abs() < 1e-6);
    }

    // --- KsString --------------------------------------------------------

    #[test]
    fn ks_string_delay_length_basic() {
        let (n, frac) = KsString::delay_length(SR, 440.0);
        // sr/freq = 100.227 → N=100, frac~0.227
        assert_eq!(n, 100);
        assert!(frac >= 0.0 && frac < 1.0);
    }

    #[test]
    fn ks_string_delay_length_minimum_two() {
        let (n, _) = KsString::delay_length(SR, 100_000.0);
        // Way above SR, would compute < 2.
        assert!(n >= 2);
    }

    #[test]
    fn ks_string_delay_length_compensated_smaller_than_raw() {
        let (n_raw, _) = KsString::delay_length(SR, 261.63);
        let (n_comp, _) = KsString::delay_length_compensated(SR, 261.63, 0.18);
        assert!(n_comp <= n_raw, "compensated length should be ≤ raw");
    }

    #[test]
    fn ks_string_step_produces_finite_output() {
        let mut s = KsString::new(SR, 440.0, 1.0, 0.18, 5);
        for _ in 0..1000 {
            let y = s.step();
            assert!(y.is_finite());
        }
    }

    #[test]
    fn ks_string_with_buf_produces_audio() {
        let (n, _) = KsString::delay_length_compensated(SR, 440.0, 0.18);
        let buf = hammer_excitation(n, 5, 1.0);
        let mut s = KsString::with_buf(SR, 440.0, buf, 0.997, 0.18);
        let mut peak = 0.0_f32;
        for _ in 0..1000 {
            peak = peak.max(s.step().abs());
        }
        assert!(peak > 0.0, "string should produce non-zero output");
    }

    #[test]
    fn ks_string_with_attack_lpf_chains() {
        let s = KsString::new(SR, 440.0, 1.0, 0.18, 5);
        let _ = s.with_attack_lpf(SR, 80.0, 0.99, 0.97);
        // Just verifying the builder pattern compiles & returns a usable struct.
    }

    #[test]
    fn ks_string_inject_feedback_changes_output() {
        // Two identical strings; feed feedback into one and confirm output diverges.
        let mut a = KsString::new(SR, 440.0, 0.5, 0.10, 3);
        let mut b = KsString::new(SR, 440.0, 0.5, 0.10, 3);
        let mut diff_seen = false;
        for _ in 0..200 {
            b.inject_feedback(0.1);
            let ya = a.step();
            let yb = b.step();
            if (ya - yb).abs() > 1e-6 {
                diff_seen = true;
            }
        }
        assert!(diff_seen, "feedback injection should affect output");
    }

    #[test]
    fn ks_string_decays_over_time() {
        let mut s = KsString::new(SR, 440.0, 1.0, 0.18, 5);
        let early_peak: f32 = (0..500).map(|_| s.step().abs()).fold(0.0_f32, f32::max);
        for _ in 0..50_000 {
            let _ = s.step();
        }
        let late_peak: f32 = (0..500).map(|_| s.step().abs()).fold(0.0_f32, f32::max);
        assert!(
            late_peak < early_peak,
            "string should decay: early {early_peak} late {late_peak}"
        );
    }

    // --- KsString 2-stage dispersion (issue #3 P2) -----------------------

    /// Build two clones of the same KS string config; one uses the legacy
    /// 1-stage call path, the other goes through `with_two_stage_dispersion`
    /// with `a2 = 0.0` (the bypass sentinel). Outputs MUST be bit-identical
    /// over a 4096-sample render — this is the regression contract that
    /// guarantees no existing voice changes its sound.
    #[test]
    fn ks_string_two_stage_bypass_is_bit_identical() {
        let ap1 = 0.18_f32;
        let (n, _) = KsString::delay_length_compensated(SR, 440.0, ap1);
        let buf_a = hammer_excitation(n, 5, 1.0);
        let buf_b = buf_a.clone();
        let mut a = KsString::with_buf(SR, 440.0, buf_a, 0.997, ap1);
        let mut b =
            KsString::with_buf(SR, 440.0, buf_b, 0.997, ap1).with_two_stage_dispersion(ap1, 0.0);
        for i in 0..4096 {
            let ya = a.step();
            let yb = b.step();
            assert_eq!(
                ya.to_bits(),
                yb.to_bits(),
                "sample {i} differs: 1-stage {ya} vs 2-stage-bypassed {yb}"
            );
        }
    }

    /// With a non-trivial second-stage coefficient, the 2-stage string MUST
    /// produce an audibly different output from the 1-stage one — at least
    /// 1 % L1 distance over 4096 samples. Catches "ap2 wired but never
    /// applied" regressions.
    #[test]
    fn ks_string_two_stage_active_diverges_from_one_stage() {
        let ap1 = 0.10_f32;
        let (n, _) = KsString::delay_length_compensated(SR, 440.0, ap1);
        let buf_a = hammer_excitation(n, 5, 1.0);
        let buf_b = buf_a.clone();
        let mut one = KsString::with_buf(SR, 440.0, buf_a, 0.997, ap1);
        let mut two =
            KsString::with_buf(SR, 440.0, buf_b, 0.997, ap1).with_two_stage_dispersion(ap1, 0.05);
        let mut sum_one = 0.0_f64;
        let mut sum_diff = 0.0_f64;
        for _ in 0..4096 {
            let y1 = one.step();
            let y2 = two.step();
            sum_one += y1.abs() as f64;
            sum_diff += (y1 - y2).abs() as f64;
        }
        assert!(sum_one > 0.0, "1-stage output is silent");
        let rel = sum_diff / sum_one;
        assert!(
            rel >= 0.01,
            "2-stage output must diverge from 1-stage by ≥1 % L1; got {rel:.4}"
        );
    }

    /// Stretched-harmonic check: an allpass with `0 < a < 1` has positive
    /// group delay → the loop period is longer for HF than for LF, so the
    /// measured partials should sit at or above `n · f1` (never below). We
    /// render an impulse-excited 1-second buffer and FFT-decompose it, then
    /// assert the per-partial frequency is ≥ `n · f1` for h1..=h8.
    #[test]
    fn ks_string_two_stage_partials_are_stretched_upward() {
        use crate::extract::decompose::decompose;
        let f0 = 220.0_f32;
        let ap1 = 0.10_f32;
        let ap2 = 0.05_f32;
        let (n, _) = KsString::delay_length_compensated(SR, f0, ap1);
        // Narrow hammer excitation → wide spectrum → all partials excited.
        let buf = hammer_excitation(n, 3, 1.0);
        let mut s = KsString::with_buf(SR, f0, buf, 0.999, ap1).with_two_stage_dispersion(ap1, ap2);
        let total_samples = SR as usize;
        let mut sig = Vec::with_capacity(total_samples);
        for _ in 0..total_samples {
            sig.push(s.step());
        }
        let partials = decompose(&sig, SR, f0, 8);
        assert!(
            partials.len() >= 4,
            "expected ≥4 partials for stretching check, got {}",
            partials.len()
        );
        let f1 = partials
            .iter()
            .find(|p| p.n == 1)
            .map(|p| p.freq_hz)
            .expect("h1 must be present");
        for p in &partials {
            // Allow 5 cents below `n*f1` for STFT bin quantisation noise on
            // h1 (the low partials are < 1 bin wide). Above that, partials
            // should monotonically stretch with stiffness.
            let expected = (p.n as f32) * f1;
            let cents = 1200.0 * (p.freq_hz / expected).log2();
            assert!(
                cents > -5.0,
                "h{}: measured {:.2} Hz, expected ≥ {:.2} Hz (cents={cents:.2})",
                p.n,
                p.freq_hz,
                expected
            );
        }
    }

    /// Stability: a 5-second render with both coefficients at the upper
    /// edge of the existing 1-stage range (0.18) must stay finite. Guards
    /// against the textbook way 2-stage allpass nets go unstable when both
    /// coefficients sit near the unit circle.
    #[test]
    fn ks_string_two_stage_stable_at_upper_range() {
        let ap1 = 0.18_f32;
        let ap2 = 0.18_f32;
        let (n, _) = KsString::delay_length_compensated(SR, 261.63, ap1);
        let buf = hammer_excitation(n, 5, 1.0);
        let mut s =
            KsString::with_buf(SR, 261.63, buf, 0.998, ap1).with_two_stage_dispersion(ap1, ap2);
        let n_samples = (SR * 5.0) as usize;
        for i in 0..n_samples {
            let y = s.step();
            assert!(
                y.is_finite(),
                "sample {i} not finite: {y} (ap1={ap1}, ap2={ap2})"
            );
            assert!(
                y.abs() < 10.0,
                "sample {i} too large: {y} (instability — ap1={ap1}, ap2={ap2})"
            );
        }
    }

    // --- SquareVoice -----------------------------------------------------

    #[test]
    fn square_voice_renders_audio() {
        let mut v = SquareVoice::new(SR, 440.0, 100);
        let buf = render_n(&mut v, 1024);
        assert!(nonzero_peak(&buf) > 0.0);
    }

    #[test]
    fn square_voice_release_then_done() {
        let mut v = SquareVoice::new(SR, 440.0, 100);
        let _ = render_n(&mut v, 1024);
        assert!(!v.is_done());
        assert!(!v.is_releasing());
        v.trigger_release();
        assert!(v.is_releasing());
        // After ~80 ms (release time) of rendering, env should reach 0.
        let _ = render_n(&mut v, ((0.080 * SR) as usize) + 100);
        assert!(v.is_done());
    }

    #[test]
    fn square_voice_attack_ramps_up() {
        let mut v = SquareVoice::new(SR, 440.0, 127);
        let early = render_n(&mut v, 4); // 4 samples — well below attack time
        let later = render_n(&mut v, 1024);
        assert!(nonzero_peak(&later) > nonzero_peak(&early));
    }

    // --- KsVoice ---------------------------------------------------------

    #[test]
    fn ks_voice_renders_audio() {
        let mut v = KsVoice::new(SR, 440.0, 100);
        let buf = render_n(&mut v, 1024);
        assert!(nonzero_peak(&buf) > 0.0);
    }

    #[test]
    fn ks_voice_releasing_before_trigger_is_false() {
        let v = KsVoice::new(SR, 440.0, 100);
        assert!(!v.is_releasing());
        assert!(!v.is_done());
    }

    #[test]
    fn ks_voice_trigger_release_marks_releasing() {
        let mut v = KsVoice::new(SR, 440.0, 100);
        v.trigger_release();
        assert!(v.is_releasing());
    }

    #[test]
    fn ks_voice_eventually_done() {
        let mut v = KsVoice::new(SR, 440.0, 100);
        v.trigger_release();
        // 0.150 s release window, render 0.3 s.
        let _ = render_n(&mut v, (SR * 0.3) as usize);
        assert!(v.is_done());
    }

    // --- KsRichVoice -----------------------------------------------------

    #[test]
    fn ks_rich_voice_renders_audio() {
        let mut v = KsRichVoice::new(SR, 440.0, 100);
        let buf = render_n(&mut v, 1024);
        assert!(nonzero_peak(&buf) > 0.0);
    }

    #[test]
    fn ks_rich_voice_release_lifecycle() {
        let mut v = KsRichVoice::new(SR, 440.0, 100);
        assert!(!v.is_releasing());
        v.trigger_release();
        assert!(v.is_releasing());
        let _ = render_n(&mut v, (SR * 0.6) as usize);
        assert!(v.is_done());
    }

    // --- SubVoice --------------------------------------------------------

    #[test]
    fn sub_voice_renders_audio() {
        let mut v = SubVoice::new(SR, 220.0, 100);
        // SubVoice has a slow filter sweep; render a few buffers to reach audible output.
        let _ = render_n(&mut v, 1024);
        let buf = render_n(&mut v, 4096);
        assert!(nonzero_peak(&buf) > 0.0);
    }

    #[test]
    fn sub_voice_release_completes() {
        let mut v = SubVoice::new(SR, 220.0, 100);
        let _ = render_n(&mut v, 1024);
        v.trigger_release();
        assert!(v.is_releasing());
        // Release time is 0.250 s; render generously.
        let _ = render_n(&mut v, (SR * 0.5) as usize);
        assert!(v.is_done());
    }

    // --- FmVoice ---------------------------------------------------------

    #[test]
    fn fm_voice_renders_audio() {
        let mut v = FmVoice::new(SR, 440.0, 100);
        let buf = render_n(&mut v, 1024);
        assert!(nonzero_peak(&buf) > 0.0);
    }

    #[test]
    fn fm_voice_release_completes() {
        let mut v = FmVoice::new(SR, 440.0, 100);
        let _ = render_n(&mut v, 512);
        v.trigger_release();
        assert!(v.is_releasing());
        let _ = render_n(&mut v, (SR * 1.0) as usize);
        assert!(v.is_done());
    }

    // --- PianoVoice ------------------------------------------------------

    #[test]
    fn piano_voice_renders_audio() {
        let mut v = PianoVoice::new(SR, 261.63, 100);
        let buf = render_n(&mut v, 4096);
        assert!(nonzero_peak(&buf) > 0.0);
    }

    #[test]
    fn piano_voice_release_lifecycle() {
        let mut v = PianoVoice::new(SR, 261.63, 100);
        assert!(!v.is_releasing());
        v.trigger_release();
        assert!(v.is_releasing());
        let _ = render_n(&mut v, (SR * 0.7) as usize);
        assert!(v.is_done());
    }

    // --- KotoVoice -------------------------------------------------------

    #[test]
    fn koto_voice_renders_audio() {
        let mut v = KotoVoice::new(SR, 261.63, 100);
        let buf = render_n(&mut v, 1024);
        assert!(nonzero_peak(&buf) > 0.0);
    }

    #[test]
    fn koto_voice_release_lifecycle() {
        let mut v = KotoVoice::new(SR, 261.63, 100);
        v.trigger_release();
        assert!(v.is_releasing());
        let _ = render_n(&mut v, (SR * 1.1) as usize);
        assert!(v.is_done());
    }

    // --- PianoVoice (PIANO_THICK preset) ---------------------------------

    #[test]
    fn piano_thick_preset_renders_audio() {
        let mut v = PianoVoice::with_preset(PianoPreset::PIANO_THICK, SR, 261.63, 100);
        let buf = render_n(&mut v, 4096);
        assert!(nonzero_peak(&buf) > 0.0);
    }

    #[test]
    fn piano_thick_preset_release_lifecycle() {
        let mut v = PianoVoice::with_preset(PianoPreset::PIANO_THICK, SR, 261.63, 100);
        v.trigger_release();
        assert!(v.is_releasing());
        let _ = render_n(&mut v, (SR * 0.7) as usize);
        assert!(v.is_done());
    }

    #[test]
    fn piano_lite_preset_renders_audio() {
        let mut v = PianoVoice::with_preset(PianoPreset::PIANO_LITE, SR, 261.63, 100);
        let buf = render_n(&mut v, 4096);
        assert!(nonzero_peak(&buf) > 0.0);
    }

    #[test]
    fn piano_lite_preset_release_lifecycle() {
        let mut v = PianoVoice::with_preset(PianoPreset::PIANO_LITE, SR, 261.63, 100);
        v.trigger_release();
        assert!(v.is_releasing());
        let _ = render_n(&mut v, (SR * 0.7) as usize);
        assert!(v.is_done());
    }

    #[test]
    fn piano_detunes_symmetric_odd_count() {
        let d = piano_detunes(7, 3.0);
        assert_eq!(d.len(), 7);
        // Centre is exactly 1.0
        assert!((d[3] - 1.0).abs() < 1e-9);
        // Symmetric: outermost ratio = 1 / opposite ratio (within FP).
        assert!((d[0] * d[6] - 1.0).abs() < 1e-6);
        assert!((d[1] * d[5] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn piano_detunes_three_string_lite() {
        let d = piano_detunes(3, 1.5);
        assert_eq!(d.len(), 3);
        assert!((d[1] - 1.0).abs() < 1e-9);
        assert!((d[0] * d[2] - 1.0).abs() < 1e-6);
    }

    // --- SfPianoPlaceholder ---------------------------------------------

    #[test]
    fn sf_placeholder_default_constructs() {
        let p = SfPianoPlaceholder::default();
        assert!(!p.is_releasing());
        assert!(!p.is_done());
    }

    #[test]
    fn sf_placeholder_release_lifecycle() {
        let mut p = SfPianoPlaceholder::new(SR);
        assert!(!p.is_releasing());
        p.trigger_release();
        assert!(p.is_releasing());
        // 3 s release window; render 4 s of buffers.
        let frames_per_buf = 1024;
        let total_frames = (SR * 4.0) as usize;
        let mut buf = vec![0.0_f32; frames_per_buf];
        let mut rendered = 0;
        while rendered < total_frames {
            for s in buf.iter_mut() {
                *s = 0.0;
            }
            p.render_add(&mut buf);
            rendered += frames_per_buf;
        }
        assert!(
            p.is_done(),
            "placeholder should be done after >3s post-release"
        );
    }

    #[test]
    fn sf_placeholder_render_add_does_not_modify_buffer() {
        // Placeholder is silent — it must not write into the audio bus
        // (audio is rendered by the shared synth elsewhere).
        let mut p = SfPianoPlaceholder::new(SR);
        let mut buf = vec![1.5_f32; 256];
        p.render_add(&mut buf);
        for v in &buf {
            assert_eq!(*v, 1.5, "placeholder must not modify the buffer");
        }
    }

    // --- make_voice factory ---------------------------------------------

    #[test]
    fn make_voice_square_renders() {
        let mut v = make_voice(Engine::Square, SR, 440.0, 100);
        let buf = render_n(v.as_mut(), 1024);
        assert!(nonzero_peak(&buf) > 0.0);
    }

    #[test]
    fn make_voice_ks_renders() {
        let mut v = make_voice(Engine::Ks, SR, 440.0, 100);
        let buf = render_n(v.as_mut(), 1024);
        assert!(nonzero_peak(&buf) > 0.0);
    }

    #[test]
    fn make_voice_ks_rich_renders() {
        let mut v = make_voice(Engine::KsRich, SR, 440.0, 100);
        let buf = render_n(v.as_mut(), 1024);
        assert!(nonzero_peak(&buf) > 0.0);
    }

    #[test]
    fn make_voice_sub_renders() {
        let mut v = make_voice(Engine::Sub, SR, 220.0, 100);
        let _ = render_n(v.as_mut(), 1024);
        let buf = render_n(v.as_mut(), 4096);
        assert!(nonzero_peak(&buf) > 0.0);
    }

    #[test]
    fn make_voice_fm_renders() {
        let mut v = make_voice(Engine::Fm, SR, 440.0, 100);
        let buf = render_n(v.as_mut(), 1024);
        assert!(nonzero_peak(&buf) > 0.0);
    }

    #[test]
    fn make_voice_piano_renders() {
        let mut v = make_voice(Engine::Piano, SR, 261.63, 100);
        let buf = render_n(v.as_mut(), 4096);
        assert!(nonzero_peak(&buf) > 0.0);
    }

    #[test]
    fn make_voice_koto_renders() {
        let mut v = make_voice(Engine::Koto, SR, 261.63, 100);
        let buf = render_n(v.as_mut(), 1024);
        assert!(nonzero_peak(&buf) > 0.0);
    }

    #[test]
    fn make_voice_sf_piano_silent() {
        let mut v = make_voice(Engine::SfPiano, SR, 261.63, 100);
        let buf = render_n(v.as_mut(), 1024);
        assert_eq!(nonzero_peak(&buf), 0.0);
    }

    #[test]
    fn make_voice_sfz_piano_silent() {
        let mut v = make_voice(Engine::SfzPiano, SR, 261.63, 100);
        let buf = render_n(v.as_mut(), 1024);
        assert_eq!(nonzero_peak(&buf), 0.0);
    }

    #[test]
    fn make_voice_piano_thick_renders() {
        let mut v = make_voice(Engine::PianoThick, SR, 261.63, 100);
        let buf = render_n(v.as_mut(), 4096);
        assert!(nonzero_peak(&buf) > 0.0);
    }

    // --- Eviction-policy regression --------------------------------------
    //
    // Pre-refactor bug: most voice impls forgot to override `is_releasing`,
    // so the voice-pool `find(|x| x.is_done() || x.is_releasing())` scan
    // returned None and the eviction fell through to `unwrap_or(0)`,
    // killing slot 0 — often a still-sounding note. After the refactor
    // every multiplicative-release voice (and the SfPianoPlaceholder)
    // returns is_releasing() == true once trigger_release is called.

    #[test]
    fn every_engine_reports_is_releasing_after_trigger() {
        let engines = [
            Engine::Square,
            Engine::Ks,
            Engine::KsRich,
            Engine::Sub,
            Engine::Fm,
            Engine::Piano,
            Engine::Koto,
            Engine::SfPiano,
            Engine::SfzPiano,
            Engine::PianoThick,
        ];
        for engine in engines.iter().copied() {
            let mut v = make_voice(engine, SR, 440.0, 100);
            // SubVoice needs at least one sample of render before its
            // amp_env leaves the Attack stage — without that, releasing
            // an immediately-released voice is treated as Done with
            // value <= 1e-6. Render a tiny block first.
            let _ = render_n(v.as_mut(), 64);
            assert!(
                !v.is_releasing(),
                "engine {:?} reports releasing pre-trigger",
                engine
            );
            v.trigger_release();
            assert!(
                v.is_releasing(),
                "engine {:?} must report is_releasing() == true after trigger_release",
                engine,
            );
        }
    }

    #[test]
    fn voice_pool_eviction_prefers_released_voices() {
        // Simulate the real eviction policy: at cap, the pool finds the
        // first voice that is_done() OR is_releasing(). Construct a pool
        // of 4 voices, release the second one, confirm the eviction picks
        // index 1 not index 0.
        let mut pool: Vec<Box<dyn VoiceImpl + Send>> = (0..4)
            .map(|_| make_voice(Engine::Piano, SR, 440.0, 100))
            .collect();
        pool[1].trigger_release();
        let evict_idx = pool
            .iter()
            .position(|x| x.is_done() || x.is_releasing())
            .unwrap_or(0);
        assert_eq!(
            evict_idx, 1,
            "should evict the released voice (slot 1), got {evict_idx}"
        );
    }

    #[test]
    fn voice_pool_all_unreleased_falls_back_to_zero() {
        let pool: Vec<Box<dyn VoiceImpl + Send>> = (0..4)
            .map(|_| make_voice(Engine::Piano, SR, 440.0, 100))
            .collect();
        let evict_idx = pool
            .iter()
            .position(|x| x.is_done() || x.is_releasing())
            .unwrap_or(0);
        assert_eq!(evict_idx, 0, "no released voice → fall back to slot 0");
    }
}
