//! Kira-backed audio engine for the web build.
//!
//! This module owns:
//!   - the `kira::AudioManager<DefaultBackend>` (page-lifetime)
//!   - a custom `Sound` impl wrapping a keysynth `Voice`
//!   - an optional bus `Effect` (sympathetic + reverb + MixMode tanh)
//!   - a `HashMap<u8, SoundHandle>` so `note_off(midi_note)` can find
//!     the right active sound to release
//!
//! Public API expected by `src/bin/web.rs`:
//!
//! ```ignore
//! pub struct KiraEngine { /* … */ }
//!
//! impl KiraEngine {
//!     pub fn new() -> Result<Self, String>;
//!     pub fn note_on(&mut self, engine: Engine, note: u8, velocity: u8);
//!     pub fn note_off(&mut self, note: u8);
//!     pub fn set_engine(&mut self, engine: Engine);
//!     pub fn set_modal_preset(&mut self, preset: ModalPreset);
//!     pub fn sample_rate(&self) -> u32;
//! }
//! ```
//!
//! The Codex dispatch fills in the body. This skeleton exists so the
//! Gemini side (web.rs glue) can compile against the public type
//! while Codex iterates on the implementation.

#![cfg(feature = "web")]

use std::{
    collections::HashMap,
    convert::Infallible,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use kira::{
    backend::DefaultBackend,
    effect::{Effect, EffectBuilder},
    info::Info,
    sound::{Sound, SoundData},
    track::{TrackBuilder, TrackHandle},
    AudioManager, AudioManagerSettings, Easing, Frame, Tween,
};

use crate::{
    reverb::{self, Reverb},
    sympathetic::SympatheticBank,
    synth::{make_voice, midi_to_freq, Engine, MixMode, ModalPreset, VoiceImpl},
};

const DEFAULT_MASTER: f32 = 1.0;
const DEFAULT_REVERB_WET: f32 = 0.3;
const NOTE_STOP_FADE: Duration = Duration::from_millis(25);
const SCRATCH_CAPACITY_FRAMES: usize = 8192;
const SYMPATHETIC_COUPLING: f32 = 0.0002;
const SYMPATHETIC_MIX: f32 = 0.3;

pub struct KiraEngine {
    #[allow(dead_code)]
    manager: AudioManager<DefaultBackend>,
    /// Sub-track that hosts the shared bus chain (sympathetic →
    /// reverb → MixMode). Every voice plays into THIS track so the
    /// chain runs once on the summed mix instead of being smeared
    /// across N independent voice instances. Without this, 3 keys
    /// pressed together = 3 reverb tails layered + 3 sympathetic
    /// banks ringing in isolation + 3 ParallelComp+tanh saturations
    /// summed — exactly the "ゴミ" the user reported.
    bus_track: TrackHandle,
    voices: HashMap<u8, VoiceSoundHandle>,
    current_engine: Engine,
    current_modal_preset: ModalPreset,
    sample_rate: u32,
}

impl KiraEngine {
    /// Initialise Kira's AudioManager and capture the negotiated sample rate.
    pub fn new() -> Result<Self, String> {
        let mut manager = AudioManager::<DefaultBackend>::new(AudioManagerSettings::default())
            .map_err(|err| format!("AudioManager::new failed: {err:?}"))?;

        let sample_rate_cell = Arc::new(AtomicU32::new(0));
        let mut bus_builder = TrackBuilder::new();
        bus_builder.add_effect(SampleRateProbeBuilder::new(sample_rate_cell.clone()));
        bus_builder.add_effect(BusEffectBuilder);
        let bus_track = manager
            .add_sub_track(bus_builder)
            .map_err(|err| format!("bus sub-track failed: {err:?}"))?;

        let sample_rate = sample_rate_cell.load(Ordering::SeqCst);
        if sample_rate == 0 {
            return Err("Kira initialised, but no sample rate was reported".to_string());
        }

        let preset = ModalPreset::Default;
        preset.apply();

        Ok(Self {
            manager,
            bus_track,
            voices: HashMap::new(),
            current_engine: Engine::Square,
            current_modal_preset: preset,
            sample_rate,
        })
    }

    pub fn note_on(&mut self, engine: Engine, note: u8, velocity: u8) {
        let engine = if self.current_engine != engine {
            self.current_engine = engine;
            engine
        } else {
            self.current_engine
        };

        if is_unsupported_sample_engine(engine) {
            eprintln!("KiraEngine note_on: {engine:?} is not yet supported");
            return;
        }

        if let Some(mut old_handle) = self.voices.remove(&note) {
            old_handle.stop(default_stop_tween());
        }

        let sound_data =
            VoiceSoundData::new(engine, self.sample_rate, midi_to_freq(note), velocity);
        match self.bus_track.play(sound_data) {
            Ok(handle) => {
                self.voices.insert(note, handle);
            }
            Err(err) => {
                eprintln!("KiraEngine note_on failed for note {note}: {err:?}");
            }
        }
    }

    pub fn note_off(&mut self, note: u8) {
        if let Some(mut handle) = self.voices.remove(&note) {
            handle.stop(default_stop_tween());
        }
    }

    pub fn set_engine(&mut self, engine: Engine) {
        self.current_engine = engine;
    }

    pub fn set_modal_preset(&mut self, preset: ModalPreset) {
        if self.current_modal_preset != preset {
            self.current_modal_preset = preset;
        }
        preset.apply();
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

fn default_stop_tween() -> Tween {
    Tween {
        duration: NOTE_STOP_FADE,
        ..Tween::default()
    }
}

fn is_unsupported_sample_engine(engine: Engine) -> bool {
    matches!(engine, Engine::SfPiano | Engine::SfzPiano)
}

struct VoiceSoundData {
    engine: Engine,
    sample_rate: u32,
    freq: f32,
    velocity: u8,
}

impl VoiceSoundData {
    fn new(engine: Engine, sample_rate: u32, freq: f32, velocity: u8) -> Self {
        Self {
            engine,
            sample_rate,
            freq,
            velocity,
        }
    }
}

impl SoundData for VoiceSoundData {
    type Error = Infallible;
    type Handle = VoiceSoundHandle;

    fn into_sound(self) -> Result<(Box<dyn Sound>, Self::Handle), Self::Error> {
        let shared = Arc::new(Mutex::new(VoiceControl::default()));
        let sound = VoiceSound::new(self, shared.clone());
        Ok((Box::new(sound), VoiceSoundHandle { shared }))
    }
}

#[derive(Default)]
struct VoiceControl {
    pending_stop: Option<Tween>,
}

struct VoiceSoundHandle {
    shared: Arc<Mutex<VoiceControl>>,
}

impl VoiceSoundHandle {
    fn stop(&mut self, tween: Tween) {
        if let Ok(mut control) = self.shared.lock() {
            control.pending_stop = Some(tween);
        }
    }
}

/// Per-voice Kira sound. RAW voice render only — no sympathetic /
/// reverb / MixMode. Those run once on the summed bus inside the
/// shared `BusEffect` attached to `KiraEngine::bus_track`. Doing
/// the bus chain per-voice (as the first-cut implementation did)
/// gave 3 voices = 3 reverb tails + 3 sympathetic banks running
/// in isolation + 3 saturation stages summed — the user heard
/// that as completely broken audio.
///
/// `engine` is kept for diagnostics / future per-voice tagging
/// but does not influence the bus chain anymore (the bus has no
/// way to know which engine each summed sample originated from;
/// the sympathetic gate is permissive — it always runs but only
/// drives audibly when piano-family voices feed in piano-shaped
/// energy).
struct VoiceSound {
    #[allow(dead_code)]
    engine: Engine,
    voice: Box<dyn VoiceImpl + Send>,
    mono_scratch: Vec<f32>,
    control: Arc<Mutex<VoiceControl>>,
    stop_state: StopState,
}

impl VoiceSound {
    fn new(data: VoiceSoundData, control: Arc<Mutex<VoiceControl>>) -> Self {
        Self {
            engine: data.engine,
            voice: make_voice(
                data.engine,
                data.sample_rate as f32,
                data.freq,
                data.velocity,
            ),
            mono_scratch: vec![0.0; SCRATCH_CAPACITY_FRAMES],
            control,
            stop_state: StopState::Running,
        }
    }

    fn begin_stop(&mut self, tween: Tween) {
        if matches!(self.stop_state, StopState::Finished | StopState::Fading(_)) {
            return;
        }
        self.voice.trigger_release();
        let duration = tween.duration.as_secs_f64();
        if duration <= 0.0 {
            self.stop_state = StopState::Finished;
        } else {
            self.stop_state = StopState::Fading(FadeOut {
                elapsed: 0.0,
                duration,
                easing: tween.easing,
            });
        }
    }
}

impl Sound for VoiceSound {
    fn on_start_processing(&mut self) {
        let pending_stop = self
            .control
            .lock()
            .ok()
            .and_then(|mut control| control.pending_stop.take());
        if let Some(tween) = pending_stop {
            self.begin_stop(tween);
        }
    }

    fn process(&mut self, out: &mut [Frame], dt: f64, _info: &Info) {
        if out.is_empty() {
            return;
        }
        if matches!(self.stop_state, StopState::Finished) || self.voice.is_done() {
            out.fill(Frame::ZERO);
            self.stop_state = StopState::Finished;
            return;
        }

        let frames = out.len();
        if frames > self.mono_scratch.len() {
            eprintln!(
                "KiraEngine voice buffer too large: requested {frames}, capacity {}",
                self.mono_scratch.len()
            );
            out.fill(Frame::ZERO);
            self.stop_state = StopState::Finished;
            return;
        }

        let mono = &mut self.mono_scratch[..frames];
        mono.fill(0.0);
        self.voice.render_add(mono);
        apply_stop_fade(mono, dt, &mut self.stop_state);

        for (frame, sample) in out.iter_mut().zip(mono.iter().copied()) {
            *frame = Frame::from_mono(sample);
        }

        if self.voice.is_done() {
            self.stop_state = StopState::Finished;
        }
    }

    fn finished(&self) -> bool {
        matches!(self.stop_state, StopState::Finished) || self.voice.is_done()
    }
}

enum StopState {
    Running,
    Fading(FadeOut),
    Finished,
}

struct FadeOut {
    elapsed: f64,
    duration: f64,
    easing: Easing,
}

fn apply_mix_mode(
    mono: &mut [f32],
    mono_compressed: &mut [f32],
    limiter_gain: &mut f32,
    mix_mode: MixMode,
    master: f32,
) {
    match mix_mode {
        MixMode::Plain => {
            for sample in mono.iter_mut() {
                *sample = (*sample * master).tanh();
            }
        }
        MixMode::Limiter => {
            const ATTACK: f32 = 0.5;
            const RELEASE: f32 = 0.0001;
            for sample in mono.iter_mut() {
                let abs_s = sample.abs();
                if abs_s > *limiter_gain {
                    *limiter_gain += (abs_s - *limiter_gain) * ATTACK;
                } else {
                    *limiter_gain += (abs_s - *limiter_gain) * RELEASE;
                }
                let gr = if *limiter_gain > 1.0 {
                    1.0 / *limiter_gain
                } else {
                    1.0
                };
                *sample = (*sample * gr * master).tanh();
            }
        }
        MixMode::ParallelComp => {
            const ALPHA: f32 = 0.7;
            const BETA: f32 = 0.6;
            const ATTACK: f32 = 0.5;
            const RELEASE: f32 = 0.0001;

            for (i, sample) in mono.iter().copied().enumerate() {
                let abs_s = sample.abs();
                if abs_s > *limiter_gain {
                    *limiter_gain += (abs_s - *limiter_gain) * ATTACK;
                } else {
                    *limiter_gain += (abs_s - *limiter_gain) * RELEASE;
                }
                let gr = if *limiter_gain > 1.0 {
                    1.0 / *limiter_gain
                } else {
                    1.0
                };
                mono_compressed[i] = sample * gr;
            }

            for (i, sample) in mono.iter_mut().enumerate() {
                let combined = (*sample * ALPHA + mono_compressed[i] * BETA) * master;
                *sample = combined.tanh();
            }
        }
    }
}

fn apply_stop_fade(samples: &mut [f32], dt: f64, stop_state: &mut StopState) {
    let StopState::Fading(fade) = stop_state else {
        return;
    };

    for sample in samples.iter_mut() {
        let progress = if fade.duration <= 0.0 {
            1.0
        } else {
            (fade.elapsed / fade.duration).clamp(0.0, 1.0)
        };
        let gain = (1.0 - apply_easing(fade.easing, progress)).clamp(0.0, 1.0) as f32;
        *sample *= gain;
        fade.elapsed += dt;
    }

    if fade.elapsed >= fade.duration {
        *stop_state = StopState::Finished;
    }
}

fn apply_easing(easing: Easing, mut x: f64) -> f64 {
    match easing {
        Easing::Linear => x,
        Easing::InPowi(power) => x.powi(power),
        Easing::OutPowi(power) => 1.0 - apply_easing(Easing::InPowi(power), 1.0 - x),
        Easing::InOutPowi(power) => {
            x *= 2.0;
            if x < 1.0 {
                0.5 * apply_easing(Easing::InPowi(power), x)
            } else {
                x = 2.0 - x;
                0.5 * (1.0 - apply_easing(Easing::InPowi(power), x)) + 0.5
            }
        }
        Easing::InPowf(power) => x.powf(power),
        Easing::OutPowf(power) => 1.0 - apply_easing(Easing::InPowf(power), 1.0 - x),
        Easing::InOutPowf(power) => {
            x *= 2.0;
            if x < 1.0 {
                0.5 * apply_easing(Easing::InPowf(power), x)
            } else {
                x = 2.0 - x;
                0.5 * (1.0 - apply_easing(Easing::InPowf(power), x)) + 0.5
            }
        }
    }
}

struct SampleRateProbeBuilder {
    sample_rate: Arc<AtomicU32>,
}

impl SampleRateProbeBuilder {
    fn new(sample_rate: Arc<AtomicU32>) -> Self {
        Self { sample_rate }
    }
}

impl EffectBuilder for SampleRateProbeBuilder {
    type Handle = ();

    fn build(self) -> (Box<dyn Effect>, Self::Handle) {
        (
            Box::new(SampleRateProbeEffect {
                sample_rate: self.sample_rate,
            }),
            (),
        )
    }
}

struct SampleRateProbeEffect {
    sample_rate: Arc<AtomicU32>,
}

impl Effect for SampleRateProbeEffect {
    fn init(&mut self, sample_rate: u32, _internal_buffer_size: usize) {
        self.sample_rate.store(sample_rate, Ordering::SeqCst);
    }

    fn on_change_sample_rate(&mut self, sample_rate: u32) {
        self.sample_rate.store(sample_rate, Ordering::SeqCst);
    }

    fn process(&mut self, _input: &mut [Frame], _dt: f64, _info: &Info) {}
}

/// Builder for the shared bus chain effect. Lazily initialised by
/// Kira with the actual sample rate.
struct BusEffectBuilder;

impl EffectBuilder for BusEffectBuilder {
    type Handle = ();

    fn build(self) -> (Box<dyn Effect>, Self::Handle) {
        (Box::new(BusEffect::uninit()), ())
    }
}

/// Shared bus chain — runs once on the summed mix of every voice
/// playing into the bus track. Mirrors the native
/// `src/main.rs::audio_callback` graph: incoming samples → drop
/// stereo down to mono → sympathetic bank (always driven; we
/// can't tell per-sample which voices are piano-family from a
/// summed bus, so we run permissively at the same coupling/mix
/// the native bus uses) → NaN/denormal flush → body IR reverb
/// (single shared instance, single tail) → MixMode (default
/// ParallelComp at master=1.0) → mirror back to stereo.
struct BusEffect {
    sample_rate: u32,
    sympathetic: SympatheticBank,
    reverb: Reverb,
    limiter_gain: f32,
    mono_scratch: Vec<f32>,
    mono_compressed: Vec<f32>,
}

impl BusEffect {
    fn uninit() -> Self {
        Self {
            sample_rate: 0,
            sympathetic: SympatheticBank::new_piano(48_000.0),
            reverb: Reverb::new(reverb::synthetic_body_ir(48_000)),
            limiter_gain: 1.0,
            mono_scratch: vec![0.0; SCRATCH_CAPACITY_FRAMES],
            mono_compressed: vec![0.0; SCRATCH_CAPACITY_FRAMES],
        }
    }

    fn rebuild(&mut self, sample_rate: u32) {
        self.sample_rate = sample_rate;
        self.sympathetic = SympatheticBank::new_piano(sample_rate as f32);
        self.reverb = Reverb::new(reverb::synthetic_body_ir(sample_rate));
        self.limiter_gain = 1.0;
    }
}

impl Effect for BusEffect {
    fn init(&mut self, sample_rate: u32, _internal_buffer_size: usize) {
        self.rebuild(sample_rate);
    }

    fn on_change_sample_rate(&mut self, sample_rate: u32) {
        self.rebuild(sample_rate);
    }

    fn process(&mut self, input: &mut [Frame], _dt: f64, _info: &Info) {
        let frames = input.len();
        if frames == 0 {
            return;
        }
        if frames > self.mono_scratch.len() {
            // Fail loudly instead of silently truncating. The user
            // explicitly asked for hard failure over silent
            // degradation; if Kira hands us a bigger buffer than
            // SCRATCH_CAPACITY_FRAMES, that's a sizing assumption
            // we got wrong and we want to know.
            eprintln!(
                "BusEffect: buffer too large (got {frames}, cap {}); zeroing",
                self.mono_scratch.len()
            );
            for f in input.iter_mut() {
                *f = Frame::ZERO;
            }
            return;
        }

        let mono = &mut self.mono_scratch[..frames];
        let mono_compressed = &mut self.mono_compressed[..frames];

        // Pull bus down to mono. VoiceSound mirrors mono → both
        // channels via Frame::from_mono so left == right; either
        // channel works. Average for safety in case future code
        // routes a stereo synth here.
        for (i, f) in input.iter().enumerate() {
            mono[i] = 0.5 * (f.left + f.right);
        }

        // Sympathetic bank — drive permissively (can't tell which
        // voices are piano-family from a summed bus). For non-
        // piano-only sessions the bank stays near silent because
        // square / FM / sub voices don't share modes with the
        // piano-tuned strings. Native gates per-voice but the
        // shared bus is the structural correct unit, so accept
        // the slightly different gating semantics.
        for sample in mono.iter_mut() {
            let drive = *sample;
            let sym_out = self.sympathetic.process(drive, SYMPATHETIC_COUPLING);
            *sample += sym_out * SYMPATHETIC_MIX;
        }

        // NaN/Inf + denormal flush.
        for sample in mono.iter_mut() {
            if !sample.is_finite() || sample.abs() < 1e-30 {
                *sample = 0.0;
            }
        }

        // Body IR reverb — single shared tail.
        self.reverb.process(mono, DEFAULT_REVERB_WET);

        // MixMode (ParallelComp default, master=1.0). Same
        // constants as `src/main.rs::audio_callback`.
        apply_mix_mode(
            mono,
            mono_compressed,
            &mut self.limiter_gain,
            MixMode::ParallelComp,
            DEFAULT_MASTER,
        );

        // Mirror back to stereo Frame.
        for (f, &m) in input.iter_mut().zip(mono.iter()) {
            *f = Frame::from_mono(m);
        }
    }
}
