//! Subtractive voice: saw -> SVF lowpass (cutoff env) -> ADSR amp.
//!
//! State-Variable Filter is the TPT (topology-preserving transform) form by
//! Andrew Simper / Cytomic. It stays stable across the full audio range and
//! has no zero-delay-feedback issues.

use super::super::synth::{Adsr, VoiceImpl};

pub struct SubVoice {
    sr: f32,
    freq: f32,
    amp: f32,
    phase: f32, // 0..1 for sawtooth
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
    // SubVoice uses dual ADSR envelopes (amp + filter) rather than the
    // shared ReleaseEnvelope, so it overrides the trait methods.
    fn trigger_release(&mut self) {
        self.amp_env.release();
        self.filt_env.release();
    }
    fn is_done(&self) -> bool {
        self.amp_env.done()
    }
    fn is_releasing(&self) -> bool {
        self.amp_env.is_releasing()
    }
}
