//! Karplus-Strong voice: white-noise-excited delay loop with 2-tap lowpass.

use super::super::synth::{
    hammer_excitation, hammer_width_for_velocity, ReleaseEnvelope, VoiceImpl,
};

pub struct KsVoice {
    buf: Vec<f32>,
    head: usize,
    decay: f32,
    // Fractional-delay tuning allpass: y[n] = c*x[n] + x[n-1] - c*y[n-1]
    // where c = (1 - frac) / (1 + frac). Total delay = buf.len() + frac.
    tune_coef: f32,
    tune_xprev: f32,
    tune_yprev: f32,
    release: ReleaseEnvelope,
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
        Self {
            buf,
            head: 0,
            decay: 0.996,
            tune_coef,
            tune_xprev: 0.0,
            tune_yprev: 0.0,
            release: ReleaseEnvelope::new(0.150, sr),
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
            let env = self.release.step();
            *sample += cur * env;
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
    fn release_env(&self) -> Option<&ReleaseEnvelope> {
        Some(&self.release)
    }
    fn release_env_mut(&mut self) -> Option<&mut ReleaseEnvelope> {
        Some(&mut self.release)
    }
}
