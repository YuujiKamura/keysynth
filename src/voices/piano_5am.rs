//! PianoVoice as it stood at commit `8f0df23` (2026-04-25 05:38).
//!
//! Containerised on 2026-04-25 because that morning's tuning state was
//! perceptually the most balanced before the 7-string + soundboard +
//! sympathetic-bank work that followed. Captured here as a static
//! reproduction so the ear-driven path that produced it doesn't get
//! lost when later tuning loops walk further away.
//!
//! Differences vs current `voices::piano::PianoVoice`:
//!   - 3 KsStrings (not 7), simple equal-gain sum 0.333
//!   - No `Soundboard`, no bridge coupling — the body resonance the
//!     current piano gets from the modal soundboard came in at d81bf97
//!     (05:58), AFTER this snapshot.
//!   - No shared `SympatheticBank` interaction (bank itself was added
//!     d0ad787 at 06:14, so this preset predates it).
//!   - Pre-rendered `click_buf` (broadband thump + 4-ping high-band
//!     transient) preserved as written, including the "amp = 0" gating
//!     that was already in place at 8f0df23 — i.e. the click is
//!     audibly disabled by default but the structure is kept so the
//!     original code reads identically.
//!   - Legacy multiplicative release (`released: bool + rel_mul + rel_step`)
//!     instead of the shared `ReleaseEnvelope` (which only landed at
//!     8c2bdd9, 07:29).
//!
//! Bug note: at 8f0df23 PianoVoice did NOT override `is_releasing()`,
//! which caused the voice-pool eviction bug ("音が出なくなる on rapid
//! repeated keypresses" — the pool's `is_releasing` scan returned
//! `false` for every Piano voice, so eviction fell back to
//! `unwrap_or(0)` and killed slot 0, often a still-sounding note).
//! This snapshot adds the missing override so the sound balance is
//! preserved while the eviction policy actually works. THIS is the
//! one functional change vs the 8f0df23 source.

#![allow(dead_code)]

use crate::synth::{piano_hammer_excitation, piano_hammer_width, KsString, VoiceImpl};

pub struct Piano5AMVoice {
    strings: [KsString; 3],
    released: bool,
    rel_mul: f32,
    rel_step: f32,
    /// Pre-rendered hammer "click". Allocated and (optionally) populated
    /// at `new`. At commit 8f0df23 both component amplitudes were 0.0,
    /// so this buffer is silent in practice — kept for byte-faithfulness
    /// with the morning-of-2026-04-25 reproduction.
    click_buf: Vec<f32>,
    click_pos: usize,
}

impl Piano5AMVoice {
    pub fn new(sr: f32, freq: f32, velocity: u8) -> Self {
        let amp = (velocity.max(1) as f32) / 127.0;
        let cents_to_ratio = |c: f32| 2.0_f32.powf(c / 1200.0);
        // Real piano triple-string unison detune ~0.5-2 cents (Weinreich
        // 1977). Wider spreads sound honky-tonk, not piano.
        let detunes = [cents_to_ratio(-1.5), 1.0, cents_to_ratio(1.5)];
        // Lower stiffness — measured B at C4 ~3e-4 from SF2 reference.
        let ap = (0.10 + (freq / 4000.0).min(0.05)).min(0.18);
        // Frequency-dependent decay tuned 5:38 against SFZ Salamander.
        let decay_for = |f: f32| -> f32 {
            let high = (f / 2000.0).clamp(0.0, 1.0);
            0.9985 - 0.0035 * high
        };
        let hammer_w = piano_hammer_width(velocity);
        // Fix B disabled — see 8f0df23 commit history. attack_lpf
        // params are flat (0 ms, w0 = w1 = 0.97).
        let attack_ms = 0.0_f32;
        let w0_attack = 0.97_f32;
        let w0_steady = 0.97_f32;
        let mk = |freq_string: f32| -> KsString {
            let (n, _frac) = KsString::delay_length_compensated(sr, freq_string, ap);
            let buf = piano_hammer_excitation(n, hammer_w, amp);
            KsString::with_buf(sr, freq_string, buf, decay_for(freq_string), ap)
                .with_attack_lpf(sr, attack_ms, w0_attack, w0_steady)
        };
        let strings = [
            mk(freq * detunes[0]),
            mk(freq * detunes[1]),
            mk(freq * detunes[2]),
        ];
        let release_sec = 0.300_f32;
        let release_samples = (release_sec * sr).max(1.0);
        let rel_step = (1e-3_f32.ln() / release_samples).exp();

        // Pre-rendered hammer click — verbatim from 8f0df23 with
        // amplitudes already set to 0.0 (effectively silent).
        let vel_norm = (velocity.max(1) as f32) / 127.0;
        let thump_samples = ((sr * 0.0008) as usize).clamp(16, 80);
        let thump_amp = 0.0_f32;
        let bright_samples = ((sr * 0.080) as usize).clamp(256, 4096);
        let bright_amp = 0.0_f32;
        let _ping_freq_hz = 6500.0 + vel_norm * 1500.0;
        let bright_tau = sr * 0.018;
        let bright_attack_n = ((sr * 0.0015) as usize).max(1);

        let click_samples = thump_samples.max(bright_samples);
        let mut click_buf = vec![0.0_f32; click_samples];
        let mut rng_state: u32 = 0x9E37_79B9
            ^ ((freq.to_bits()).wrapping_mul(2654435761))
            ^ ((velocity as u32).wrapping_mul(0x85EB_CA77));
        // (1) Legacy thump: HP 200 Hz + LP 8 kHz, half-cosine windowed.
        {
            let mut lp_prev = 0.0f32;
            let mut hp_prev_x = 0.0f32;
            let mut hp_prev_y = 0.0f32;
            let hp_a = 1.0 - 2.0 * std::f32::consts::PI * 200.0 / sr;
            let lp_a = (-2.0 * std::f32::consts::PI * 8000.0 / sr).exp();
            for i in 0..thump_samples {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 17;
                rng_state ^= rng_state << 5;
                let raw = ((rng_state as i32) as f32) / (i32::MAX as f32);
                let hp = hp_a * (hp_prev_y + raw - hp_prev_x);
                hp_prev_x = raw;
                hp_prev_y = hp;
                lp_prev = (1.0 - lp_a) * hp + lp_a * lp_prev;
                let t = i as f32 / (thump_samples - 1).max(1) as f32;
                let win = 0.5 * (1.0 - (std::f32::consts::TAU * t).cos());
                click_buf[i] += lp_prev * win * thump_amp;
            }
        }
        // (2) High-band ping chord at 5500/6500/7500/8500 Hz.
        {
            let ping_freqs = [5500.0_f32, 6500.0, 7500.0, 8500.0];
            let n_pings = ping_freqs.len() as f32;
            let per_amp = bright_amp / n_pings.sqrt();
            let mut phases = [0.0f32; 4];
            let phase_incs: [f32; 4] = [
                std::f32::consts::TAU * ping_freqs[0] / sr,
                std::f32::consts::TAU * ping_freqs[1] / sr,
                std::f32::consts::TAU * ping_freqs[2] / sr,
                std::f32::consts::TAU * ping_freqs[3] / sr,
            ];
            for i in 0..bright_samples {
                let env = if i < bright_attack_n {
                    (i as f32) / (bright_attack_n as f32)
                } else {
                    let t = (i - bright_attack_n) as f32;
                    (-t / bright_tau).exp()
                };
                let mut sum = 0.0f32;
                for k in 0..4 {
                    sum += phases[k].sin();
                    phases[k] += phase_incs[k];
                    if phases[k] >= std::f32::consts::TAU {
                        phases[k] -= std::f32::consts::TAU;
                    }
                }
                click_buf[i] += sum * env * per_amp;
            }
            let _ = rng_state;
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

impl VoiceImpl for Piano5AMVoice {
    fn render_add(&mut self, buf: &mut [f32]) {
        // 3-string sum: in-phase at attack (~3x peak), decorrelated
        // after detune drift. 0.333 keeps in-phase peak ≤ 1.0.
        let gain = 0.333_f32;
        for sample in buf.iter_mut() {
            let s = self.strings[0].step() + self.strings[1].step() + self.strings[2].step();
            let mut out_sample = s * gain;
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
    /// One functional change vs the 8f0df23 original: provide
    /// `is_releasing` so the voice-pool eviction policy can find this
    /// voice when it's in its release tail. Without this override the
    /// trait default returns `false`, which is the root cause of the
    /// "音が出なくなる on rapid repeated keypresses" bug — the pool's
    /// `is_releasing` scan returned `false` for every Piano voice, so
    /// `unwrap_or(0)` killed slot 0 (often a still-sounding note).
    fn is_releasing(&self) -> bool {
        self.released
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 44100.0;

    fn render_n(v: &mut dyn VoiceImpl, n: usize) -> Vec<f32> {
        let mut buf = vec![0.0_f32; n];
        v.render_add(&mut buf);
        buf
    }

    #[test]
    fn renders_audio() {
        let mut v = Piano5AMVoice::new(SR, 261.63, 100);
        let buf = render_n(&mut v, 4096);
        let peak = buf.iter().copied().fold(0.0_f32, |a, b| a.max(b.abs()));
        assert!(peak > 0.01, "peak too small: {peak}");
        for &s in &buf {
            assert!(s.is_finite(), "non-finite sample");
        }
    }

    #[test]
    fn release_lifecycle() {
        let mut v = Piano5AMVoice::new(SR, 261.63, 100);
        assert!(!v.is_releasing());
        v.trigger_release();
        // Eviction-bug regression: once released, is_releasing() MUST
        // return true so the voice pool can pick this voice for
        // eviction instead of slot 0.
        assert!(v.is_releasing());
        let _ = render_n(&mut v, (SR * 0.7) as usize);
        assert!(v.is_done());
    }
}
