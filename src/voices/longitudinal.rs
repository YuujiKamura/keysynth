use std::collections::VecDeque;

pub struct LongitudinalString {
    pub delay_long: VecDeque<f32>,
    pub decay: f32,
    pub sr: f32,
}

impl LongitudinalString {
    pub fn new(sr: f32, fundamental_hz: f32, c_long_ratio: f32, decay: f32) -> Self {
        // Steel piano string longitudinal speed c_long ≈ 14× c_trans
        // The longitudinal "fundamental" sits at ~14 × f_transverse.
        let delay_len = (sr / (c_long_ratio * fundamental_hz)).round() as usize;
        let mut delay_long = VecDeque::with_capacity(delay_len);
        for _ in 0..delay_len {
            delay_long.push_back(0.0);
        }
        Self {
            delay_long,
            decay,
            sr,
        }
    }

    pub fn step(&mut self, transverse_displacement: f32) -> f32 {
        // Geometric source term: force = K_long × transverse²
        // The transverse displacement squared drives the longitudinal mode
        // through string-length variation.
        const K_LONG: f32 = 0.01;
        let force = K_LONG * transverse_displacement * transverse_displacement;

        // Basic delay line step (same shape as KS but simple for longitudinal)
        let back = self.delay_long.pop_back().unwrap_or(0.0);
        let val = force + back * self.decay;

        if !val.is_finite() {
            panic!("K_long produced a non-finite longitudinal output: {}", val);
        }

        self.delay_long.push_front(val);
        val
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longitudinal_decays_to_zero() {
        let sr = 44100.0;
        let mut s = LongitudinalString::new(sr, 261.63, 14.0, 0.99);
        // Drive with a transient
        s.step(1.0);
        // Verify output decays below 1e-6 within 5 seconds.
        for _ in 0..5 * 44100 {
            s.step(0.0);
        }
        let out = s.step(0.0);
        assert!(
            out.abs() < 1e-6,
            "Output should decay below 1e-6, got {}",
            out
        );
    }

    #[test]
    fn longitudinal_amplitude_quadratic_in_input() {
        let sr = 44100.0;
        let mut s1 = LongitudinalString::new(sr, 261.63, 14.0, 0.99);
        let mut s2 = LongitudinalString::new(sr, 261.63, 14.0, 0.99);

        // Drive with small input
        let out1 = s1.step(0.5);
        // Drive with 2x input
        let out2 = s2.step(1.0);

        // Geometric source is squared: doubling input amplitude (0.5 -> 1.0)
        // should quadruple the output amplitude (0.25 -> 1.0).
        let ratio = out2 / out1;
        assert!(
            (ratio - 4.0).abs() < 0.1 * 4.0,
            "Expected ratio ~4.0, got {}",
            ratio
        );
    }

    #[test]
    fn longitudinal_finite_under_max_drive() {
        let sr = 44100.0;
        let mut s = LongitudinalString::new(sr, 261.63, 14.0, 0.99);
        // Drive at amplitude 1.0 for 1 second
        for _ in 0..44100 {
            let out = s.step(1.0);
            assert!(out.is_finite());
            assert!(out.abs() < 100.0, "Peak should be < 100.0, got {}", out);
        }
    }
}
