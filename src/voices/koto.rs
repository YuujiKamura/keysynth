//! Koto / shamisen-leaning DWS engine.
//!
//! KS is plucked-string physics. This engine leans into that:
//!   - SINGLE string (no unison) - koto/shamisen are not unison-strung
//!   - SHARP narrow plectrum (width=2 always; only amplitude scales w/ vel)
//!     -> tsume / bachi (爪・撥) is much sharper than a hammer felt
//!   - PLUCK POSITION OFFSET: inject impulse at delay-line offset n/4 instead
//!     of 0. Combined with the natural reflection physics, this introduces
//!     the classic "comb-filter" effect that defines pluck-position timbre
//!     (plucking near the bridge sounds twangy/bright; near middle = mellow)
//!   - LONG sustain (decay closer to 1.0) - acoustic koto strings ring 5-15s
//!   - MINIMAL allpass dispersion - low stiffness, harmonic spectrum

use super::super::synth::{KsString, ReleaseEnvelope, VoiceImpl};

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
    release: ReleaseEnvelope,
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
        Self {
            string,
            release: ReleaseEnvelope::new(0.500, sr), // longer release = "ふわっと消える"
        }
    }
}

impl VoiceImpl for KotoVoice {
    fn render_add(&mut self, buf: &mut [f32]) {
        for sample in buf.iter_mut() {
            let s = self.string.step();
            let env = self.release.step();
            *sample += s * env;
        }
    }
    fn release_env(&self) -> Option<&ReleaseEnvelope> {
        Some(&self.release)
    }
    fn release_env_mut(&mut self) -> Option<&mut ReleaseEnvelope> {
        Some(&mut self.release)
    }
}
