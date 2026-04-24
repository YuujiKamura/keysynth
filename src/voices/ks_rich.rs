//! KS-rich voice: 3 detuned strings + 1-pole allpass dispersion (stiffness).
//!
//! Each "string" is an independent KS delay loop. Per-string detune (cents)
//! gives chorus/unison thickness like real piano triple-stringing. A single
//! allpass in the feedback path adds frequency-dependent phase delay -> the
//! inharmonicity that makes piano upper partials slightly sharp.

use super::super::synth::{hammer_width_for_velocity, KsString, ReleaseEnvelope, VoiceImpl};

pub struct KsRichVoice {
    strings: [KsString; 3],
    release: ReleaseEnvelope,
}

impl KsRichVoice {
    pub fn new(sr: f32, freq: f32, velocity: u8) -> Self {
        let amp = (velocity.max(1) as f32) / 127.0;
        // Detune cents: -7, 0, +7 -> classic piano triple-string spread
        let cents_to_ratio = |c: f32| 2.0_f32.powf(c / 1200.0);
        let detunes = [cents_to_ratio(-7.0), 1.0, cents_to_ratio(7.0)];
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
        Self {
            strings,
            release: ReleaseEnvelope::new(0.250, sr),
        }
    }
}

// (is_releasing override added on each piano-family voice impl below so the
// MIDI eviction in main.rs can identify "release-tail" voices for eviction
// before falling back to slot-0 (which silently kills currently-sounding
// notes when the user plays >= 32 simultaneous notes).)

impl VoiceImpl for KsRichVoice {
    fn render_add(&mut self, buf: &mut [f32]) {
        // Mix the 3 strings; equal-power-ish gain (1/sqrt(3) ~= 0.577).
        let gain = 0.577_f32;
        for sample in buf.iter_mut() {
            let s = self.strings[0].step() + self.strings[1].step() + self.strings[2].step();
            let env = self.release.step();
            *sample += s * gain * env;
        }
    }
    fn release_env(&self) -> Option<&ReleaseEnvelope> {
        Some(&self.release)
    }
    fn release_env_mut(&mut self) -> Option<&mut ReleaseEnvelope> {
        Some(&mut self.release)
    }
}
