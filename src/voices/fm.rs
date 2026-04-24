//! FM voice: 2-op (carrier + modulator), modulator has its own ADSR.
//!
//! modulator_out = sin(TAU * mod_phase) * mod_index * mod_env
//! carrier_out   = sin(TAU * car_phase + modulator_out) * amp_env
//!
//! Default ratio 14:1 + fast mod-env decay = DX7-ish bell / EP "tine" sound.

use super::super::synth::{Adsr, VoiceImpl};

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
    // FmVoice uses dual ADSR envelopes (amp + mod-index) rather than the
    // shared ReleaseEnvelope, so it overrides the trait methods.
    fn trigger_release(&mut self) {
        self.amp_env.release();
        self.mod_env.release();
    }
    fn is_done(&self) -> bool {
        self.amp_env.done()
    }
    fn is_releasing(&self) -> bool {
        self.amp_env.is_releasing()
    }
}
