//! Square voice: NES-style pulse with linear AR envelope.

use super::super::synth::VoiceImpl;

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
    // SquareVoice uses a linear AR envelope rather than the shared
    // multiplicative ReleaseEnvelope, so it overrides the trait methods.
    fn trigger_release(&mut self) {
        self.released = true;
    }
    fn is_done(&self) -> bool {
        self.released && self.env <= 1e-5
    }
    fn is_releasing(&self) -> bool {
        self.released
    }
}
