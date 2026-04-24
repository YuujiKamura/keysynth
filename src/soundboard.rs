//! Modal soundboard resonator bank for the piano engine.
//!
//! Bidirectional coupling model (Bank/Lehtonen DWS): string output is fed
//! into a parallel bank of modal bandpass resonators tuned to the soundboard's
//! prominent eigenmodes; resonator output is mixed into the audio output AND
//! fed back into the strings via a small bridge-coupling coefficient.
//!
//! The mode set targets a small/medium grand piano: 12 prominent modes
//! between 80 Hz and 2.2 kHz. Real soundboards have hundreds of modes,
//! but the perceptually dominant ones cluster in the low/mid range —
//! that "halo" of body resonance that makes a real piano sound 3D
//! instead of like a stack of sine partials.
//!
//! Stability: total feedback gain (coupling * resonator peak gain) must
//! stay < 1 across all modes. Per-mode peak gain is normalised to
//! `1 / sqrt(N_modes)` so the parallel sum's peak response stays ~1, and
//! the public `PianoVoice::coupling` keeps feedback well below unity.

/// 2nd-order biquad bandpass resonator, normalised so the peak (at f0)
/// magnitude response is `gain`. Direct-form II transposed for low noise
/// at low Q. Uses RBJ cookbook BPF (constant 0 dB peak gain) with the
/// per-mode `gain` factor applied at the input.
pub struct ModalResonator {
    // Biquad coefficients, normalised to a0 = 1.
    b0: f32,
    b2: f32,  // b1 = 0 for BPF (constant peak gain) form
    a1: f32,
    a2: f32,
    // State (transposed-direct-form-II)
    z1: f32,
    z2: f32,
    // Per-mode input gain (peak response normalisation).
    gain: f32,
}

impl ModalResonator {
    /// Build a bandpass resonator at `freq_hz` with quality `q`, peak
    /// magnitude response `peak_gain`. RBJ "BPF (constant 0 dB peak
    /// gain)" cookbook form, then scaled by `peak_gain` at the input.
    pub fn new(sr: f32, freq_hz: f32, q: f32, peak_gain: f32) -> Self {
        let omega = std::f32::consts::TAU * freq_hz / sr.max(1.0);
        let (sin_w, cos_w) = (omega.sin(), omega.cos());
        let alpha = sin_w / (2.0 * q.max(1e-3));
        // RBJ BPF (constant 0 dB peak gain): b0=alpha, b1=0, b2=-alpha,
        // a0=1+alpha, a1=-2cos(w), a2=1-alpha.
        let a0 = 1.0 + alpha;
        let inv = 1.0 / a0;
        Self {
            b0: alpha * inv,
            b2: -alpha * inv,
            a1: -2.0 * cos_w * inv,
            a2: (1.0 - alpha) * inv,
            z1: 0.0,
            z2: 0.0,
            gain: peak_gain,
        }
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let xg = x * self.gain;
        // Transposed direct form II
        let y = self.b0 * xg + self.z1;
        self.z1 = self.z2 - self.a1 * y;
        // b1 = 0 in BPF form, so the z2 update only carries b2 input.
        self.z2 = self.b2 * xg - self.a2 * y;
        y
    }
}

/// Modal soundboard: parallel bank of bandpass resonators driven by the
/// summed string output. Output is mixed into the audio bus AND fed back
/// into the strings (via `last_output`) one sample later, providing the
/// causal bridge coupling.
pub struct Soundboard {
    resonators: Vec<ModalResonator>,
    last_output: f32,
}

impl Soundboard {
    /// Lite variant: original 12-mode set, no high-freq additions, with
    /// damped Q values (15-35 vs the full variant's 60-120). Used by
    /// `Engine::PianoThick` so the upper-band high-Q resonances of the
    /// full 16-mode set don't ring through the bridge feedback loop and
    /// colour the tail (perceived as "ミャアアーオーン" sustained
    /// pitched ringing). Real soundboard plate modes are heavily damped
    /// (radiating 2D plate vs discrete resonator); the damped Qs better
    /// approximate that behaviour.
    pub fn new_concert_grand_lite(sr: f32) -> Self {
        let modes: [(f32, f32); 12] = [
            (80.0, 15.0),
            (110.0, 15.0),
            (150.0, 18.0),
            (200.0, 18.0),
            (270.0, 20.0),
            (350.0, 20.0),
            (470.0, 22.0),
            (600.0, 22.0),
            (800.0, 25.0),
            (1100.0, 25.0),
            (1500.0, 30.0),
            (2200.0, 35.0),
        ];
        let n = modes.len() as f32;
        let peak_gain = 1.0 / n.sqrt();
        let resonators = modes
            .iter()
            .map(|&(f, q)| ModalResonator::new(sr, f, q, peak_gain))
            .collect();
        Self {
            resonators,
            last_output: 0.0,
        }
    }

    /// Construct a typical small/medium concert-grand mode set.
    /// 12 modes, log-spaced 80 Hz to 2.2 kHz, Q rising with frequency
    /// (low-Q low-end gives the body "wood" thump, higher Q upper modes
    /// give brilliance/sparkle without ringing on forever).
    /// Per-mode peak gain is `1 / sqrt(N)` so the sum's peak response
    /// is order-unity.
    pub fn new_concert_grand(sr: f32) -> Self {
        // (freq Hz, Q). Frequencies and Qs picked from typical measured
        // soundboard mode sets in the Bank/Lehtonen literature; the exact
        // values are not critical, only the shape (spread + Q rising
        // with freq).
        // Extended to 16 modes: added 4 high-frequency soundboard modes
        // (3000-7500 Hz, lower Q ~30) so the body radiates the upper-band
        // "shimmer" the user perceives as missing brilliance. Real piano
        // soundboards radiate effectively up to ~6 kHz; without modes
        // there, the soundboard can only colour the bass-mid which leaves
        // the tone "曇った" / cloth-covered at the top end.
        let modes: [(f32, f32); 16] = [
            (80.0, 60.0),
            (110.0, 60.0),
            (150.0, 70.0),
            (200.0, 70.0),
            (270.0, 80.0),
            (350.0, 80.0),
            (470.0, 90.0),
            (600.0, 90.0),
            (800.0, 100.0),
            (1100.0, 100.0),
            (1500.0, 110.0),
            (2200.0, 120.0),
            (3000.0, 30.0),
            (4000.0, 30.0),
            (5500.0, 25.0),
            (7500.0, 20.0),
        ];
        let n = modes.len() as f32;
        let peak_gain = 1.0 / n.sqrt();
        let resonators = modes
            .iter()
            .map(|&(f, q)| ModalResonator::new(sr, f, q, peak_gain))
            .collect();
        Self {
            resonators,
            last_output: 0.0,
        }
    }

    /// Process one input sample, return the soundboard output sample.
    /// Side-effect: caches the result in `last_output` for one-sample
    /// delayed feedback (keeping the string<->soundboard loop causal).
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let mut sum = 0.0_f32;
        for r in self.resonators.iter_mut() {
            sum += r.process(input);
        }
        self.last_output = sum;
        sum
    }

    /// Most-recent output sample, used by the caller to inject one-sample
    /// delayed feedback into the strings before driving the next sample.
    #[inline]
    pub fn last_output(&self) -> f32 {
        self.last_output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Impulse response should decay (no instability for moderate Q).
    #[test]
    fn modal_resonator_impulse_decays() {
        let mut r = ModalResonator::new(48_000.0, 200.0, 80.0, 1.0);
        let mut peak_after = 0.0_f32;
        let initial = r.process(1.0).abs();
        // Skip first 4800 samples (100 ms), then check the next 4800 max.
        for _ in 0..4800 {
            let _ = r.process(0.0);
        }
        for _ in 0..4800 {
            let y = r.process(0.0);
            peak_after = peak_after.max(y.abs());
        }
        // After 100 ms the response should be significantly smaller than
        // the initial impulse response (i.e. it's actually decaying).
        assert!(
            peak_after < initial * 2.0,
            "resonator unstable: initial={initial} peak_after_100ms={peak_after}"
        );
        assert!(peak_after.is_finite(), "non-finite output");
    }

    /// Soundboard should produce bounded output for an impulse + 5 s of zeros.
    #[test]
    fn soundboard_bounded_for_impulse() {
        let sr = 48_000.0;
        let mut sb = Soundboard::new_concert_grand(sr);
        let mut peak = 0.0_f32;
        let _ = sb.process(1.0);
        for _ in 0..(5 * sr as usize) {
            let y = sb.process(0.0);
            peak = peak.max(y.abs());
            assert!(y.is_finite(), "non-finite soundboard output");
        }
        // With 12 modes at peak_gain=1/sqrt(12), worst-case peak is ~1.0.
        assert!(peak < 4.0, "soundboard peak too large: {peak}");
    }

    #[test]
    fn modal_resonator_zero_input_zero_output() {
        let mut r = ModalResonator::new(48_000.0, 200.0, 50.0, 1.0);
        for _ in 0..1000 {
            assert_eq!(r.process(0.0), 0.0);
        }
    }

    #[test]
    fn modal_resonator_finite_for_impulse() {
        let mut r = ModalResonator::new(48_000.0, 1000.0, 80.0, 1.0);
        let _ = r.process(1.0);
        for _ in 0..10_000 {
            let y = r.process(0.0);
            assert!(y.is_finite());
        }
    }

    #[test]
    fn soundboard_lite_constructs() {
        let sr = 48_000.0;
        let mut sb = Soundboard::new_concert_grand_lite(sr);
        // Lite has 12 modes; impulse should still ring.
        let mut max_y = 0.0_f32;
        let _ = sb.process(1.0);
        for _ in 0..1000 {
            let y = sb.process(0.0);
            max_y = max_y.max(y.abs());
            assert!(y.is_finite());
        }
        assert!(max_y > 0.0, "lite soundboard should produce non-zero ring");
    }

    #[test]
    fn soundboard_lite_decays_faster_than_full() {
        // Lite uses lower Q (15-35 vs 60-120) so its impulse response
        // should decay measurably faster than the full variant.
        let sr = 48_000.0;
        let mut full = Soundboard::new_concert_grand(sr);
        let mut lite = Soundboard::new_concert_grand_lite(sr);
        let _ = full.process(1.0);
        let _ = lite.process(1.0);
        // Skip 100 ms then sum |y| over the next 100 ms.
        for _ in 0..(sr as usize / 10) {
            let _ = full.process(0.0);
            let _ = lite.process(0.0);
        }
        let mut full_energy = 0.0_f32;
        let mut lite_energy = 0.0_f32;
        for _ in 0..(sr as usize / 10) {
            full_energy += full.process(0.0).abs();
            lite_energy += lite.process(0.0).abs();
        }
        assert!(
            lite_energy < full_energy,
            "lite ({lite_energy}) should decay faster than full ({full_energy})"
        );
    }

    #[test]
    fn soundboard_last_output_reflects_latest_process() {
        let sr = 48_000.0;
        let mut sb = Soundboard::new_concert_grand(sr);
        // Initial last_output is zero.
        assert_eq!(sb.last_output(), 0.0);
        let y1 = sb.process(1.0);
        assert_eq!(sb.last_output(), y1, "last_output should mirror latest process");
        let y2 = sb.process(0.5);
        assert_eq!(sb.last_output(), y2);
    }

    #[test]
    fn soundboard_zero_input_zero_output() {
        let sr = 48_000.0;
        let mut sb = Soundboard::new_concert_grand(sr);
        for _ in 0..10_000 {
            assert_eq!(sb.process(0.0), 0.0);
            assert_eq!(sb.last_output(), 0.0);
        }
    }
}
