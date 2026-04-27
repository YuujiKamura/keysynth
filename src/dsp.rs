//! Per-track DSP chain primitives (Stage 4 of issue #7).
//!
//! Stage 1 of the per-track mixer (custom cpal mixer) is being built in a
//! separate worktree; this module is intentionally self-contained so it
//! can be dropped into the mixer's per-track signal path without touching
//! `jukebox.rs` or any voice code. The integration receiver is
//! [`process_track_chain`], which fixes the EQ -> compressor -> reverb
//! ordering and a single `send_level` knob — that's the only function the
//! mixer needs to call once Stage 1 lands.
//!
//! Building blocks:
//!
//! - [`Biquad`]              — RBJ-cookbook biquad in direct-form II
//!                             transposed (numerically well-behaved at
//!                             low coefficients, smooth coefficient
//!                             updates, single-sample state).
//! - [`Eq3Band`]              — three RBJ filters wired serially:
//!                             low shelf -> mid peaking -> high shelf.
//! - [`Compressor`]           — peak envelope follower with attack /
//!                             release coefficients; gain is `1/peak`
//!                             above threshold so the output peak sits
//!                             at the threshold (true peak limiter at
//!                             ratio = ∞, threshold-only knee).
//! - [`TrackReverb`]          — wraps [`crate::reverb::Reverb`] with a
//!                             per-track send level so multiple tracks
//!                             can share a reverb shape but feed it at
//!                             different amounts.
//!
//! All structures expose `process_inplace(&mut self, &mut [f32])`.
//! Allocation happens at construction time only; the audio thread is
//! pure arithmetic.

use crate::reverb::Reverb;

/// Direct-form II transposed biquad.
///
/// Transfer function:
///   H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)
///
/// State is the two-element delay line `(z1, z2)` of the transposed
/// form — two multiplies of the input, two of the past output, and two
/// state updates per sample. Coefficients are stored normalised by `a0`
/// (cookbook convention) so the recurrence is just:
///
/// ```text
///   y     = b0 * x + z1
///   z1    = b1 * x - a1 * y + z2
///   z2    = b2 * x - a2 * y
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    /// Identity (pass-through) biquad. Useful when a band needs to be
    /// disabled without branching in the audio loop.
    pub fn identity() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Construct from already-normalised cookbook coefficients (the
    /// caller has already divided by `a0`).
    pub fn from_coefs(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Reset internal delay line. Call when re-seeking, splicing audio,
    /// or otherwise breaking continuity.
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// RBJ low-shelf at `f0` Hz, slope `s` (1.0 = max slope without
    /// resonance bump), `gain_db` shelf gain.
    pub fn low_shelf(sample_rate: f32, f0: f32, gain_db: f32, s: f32) -> Self {
        let a = 10.0_f32.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f32::consts::PI * f0 / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let s = s.clamp(1e-3, 1.0);
        // alpha = sin(w0)/2 * sqrt((A + 1/A)*(1/S - 1) + 2)
        let alpha = sin_w0 / 2.0 * ((a + 1.0 / a) * (1.0 / s - 1.0) + 2.0).sqrt();
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

        let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
        let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
        let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
        let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha;

        Self::from_coefs(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// RBJ high-shelf at `f0` Hz.
    pub fn high_shelf(sample_rate: f32, f0: f32, gain_db: f32, s: f32) -> Self {
        let a = 10.0_f32.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f32::consts::PI * f0 / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let s = s.clamp(1e-3, 1.0);
        let alpha = sin_w0 / 2.0 * ((a + 1.0 / a) * (1.0 / s - 1.0) + 2.0).sqrt();
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
        let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha;

        Self::from_coefs(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// RBJ peaking-EQ at `f0` Hz with `q` resonance and `gain_db`
    /// boost / cut at the centre.
    pub fn peaking(sample_rate: f32, f0: f32, gain_db: f32, q: f32) -> Self {
        let a = 10.0_f32.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f32::consts::PI * f0 / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let q = q.max(1e-3);
        let alpha = sin_w0 / (2.0 * q);

        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha / a;

        Self::from_coefs(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// Process a single sample through the transposed direct-form II.
    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    /// In-place processing of a sample buffer.
    pub fn process_inplace(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            *s = self.process_sample(*s);
        }
    }
}

/// Three-band cascade EQ: low-shelf -> mid peaking -> high-shelf.
///
/// Wiring is deliberately fixed-topology so the mixer never has to
/// reason about active band counts in the audio thread; disabled bands
/// are just `Biquad::identity()` and still cheap.
#[derive(Debug, Clone, Copy)]
pub struct Eq3Band {
    pub low_shelf: Biquad,
    pub mid_peak: Biquad,
    pub high_shelf: Biquad,
}

impl Eq3Band {
    /// Default flat EQ: all three bands are identity, ready to be
    /// reconfigured per-track without ever introducing a click.
    pub fn flat() -> Self {
        Self {
            low_shelf: Biquad::identity(),
            mid_peak: Biquad::identity(),
            high_shelf: Biquad::identity(),
        }
    }

    /// Convenience builder using musical defaults (200 Hz / 1 kHz / 5 kHz).
    pub fn new(sample_rate: f32, low_gain_db: f32, mid_gain_db: f32, high_gain_db: f32) -> Self {
        Self {
            low_shelf: Biquad::low_shelf(sample_rate, 200.0, low_gain_db, 0.7),
            mid_peak: Biquad::peaking(sample_rate, 1_000.0, mid_gain_db, 1.0),
            high_shelf: Biquad::high_shelf(sample_rate, 5_000.0, high_gain_db, 0.7),
        }
    }

    /// Reset all three biquad delay lines.
    pub fn reset(&mut self) {
        self.low_shelf.reset();
        self.mid_peak.reset();
        self.high_shelf.reset();
    }

    /// In-place serial filtering low -> mid -> high.
    pub fn process_inplace(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            let mut y = self.low_shelf.process_sample(*s);
            y = self.mid_peak.process_sample(y);
            y = self.high_shelf.process_sample(y);
            *s = y;
        }
    }
}

/// Peak-following compressor with a fixed-ratio (= ∞) limiter style:
/// once the envelope rises above `threshold`, gain becomes
/// `threshold / envelope` so the output peak parks at the threshold.
/// The envelope itself uses standard one-pole attack / release smoothing
/// driven by `|x|`.
///
/// Coefficients are derived from time constants in seconds; the usual
/// `exp(-1 / (tau * sr))` form is used so longer time constants yield a
/// coefficient closer to 1.0 (slower follower).
#[derive(Debug, Clone, Copy)]
pub struct Compressor {
    /// Linear threshold in [0, 1] (peak amplitude).
    threshold: f32,
    /// One-pole coefficient for envelope rise (0..1, larger = faster).
    attack_coef: f32,
    /// One-pole coefficient for envelope fall.
    release_coef: f32,
    /// Make-up gain applied post-compression.
    makeup: f32,
    /// Internal envelope state.
    env: f32,
}

impl Compressor {
    /// Build a compressor.
    ///
    /// - `sample_rate` Hz
    /// - `threshold` linear (0..1)
    /// - `attack_secs` time to reach ~63 % of a step rise
    /// - `release_secs` time to reach ~63 % of a step fall
    /// - `makeup` linear post-gain (use `1.0` for unity)
    pub fn new(
        sample_rate: f32,
        threshold: f32,
        attack_secs: f32,
        release_secs: f32,
        makeup: f32,
    ) -> Self {
        let attack_coef = time_constant_coef(attack_secs, sample_rate);
        let release_coef = time_constant_coef(release_secs, sample_rate);
        Self {
            threshold: threshold.max(1e-6),
            attack_coef,
            release_coef,
            makeup: makeup.max(0.0),
            env: 0.0,
        }
    }

    /// Reset envelope state.
    pub fn reset(&mut self) {
        self.env = 0.0;
    }

    /// Linear threshold getter (used by tests).
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Process a single sample: update peak follower, compute static gain.
    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        let target = x.abs();
        // Asymmetric one-pole: rise fast (attack), fall slow (release).
        let coef = if target > self.env {
            self.attack_coef
        } else {
            self.release_coef
        };
        // env = coef * env + (1-coef) * target
        self.env = coef * self.env + (1.0 - coef) * target;

        let gain = if self.env > self.threshold {
            self.threshold / self.env
        } else {
            1.0
        };
        x * gain * self.makeup
    }

    /// In-place processing.
    pub fn process_inplace(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            *s = self.process_sample(*s);
        }
    }
}

/// One-pole smoothing coefficient for a `tau`-second time constant at
/// `sr` Hz. `tau <= 0` yields `0.0` (instant follow).
fn time_constant_coef(tau: f32, sr: f32) -> f32 {
    if tau <= 0.0 || sr <= 0.0 {
        0.0
    } else {
        (-1.0 / (tau * sr)).exp().clamp(0.0, 0.9999_999)
    }
}

/// Per-track wrapper around the existing convolution [`Reverb`].
///
/// `Reverb` already does in-place wet/dry mixing controlled by a `wet`
/// argument, which is exactly what we want for a "send level" knob:
/// `0.0` skips the convolution entirely (fast path inside `Reverb`),
/// `1.0` is fully wet, and intermediate values blend dry input with
/// the convolved signal. Wrapping it here lets the per-track DSP
/// chain hold its own state (separate `history` ring per track) so
/// crossfeed between tracks doesn't bleed through a shared buffer.
pub struct TrackReverb {
    inner: Reverb,
    send_level: f32,
}

impl TrackReverb {
    /// Build a per-track reverb from an IR (typically the synthetic
    /// body IR or a loaded WAV). Empty IR => no-op reverb.
    pub fn new(ir: Vec<f32>, send_level: f32) -> Self {
        Self {
            inner: Reverb::new(ir),
            send_level: send_level.clamp(0.0, 1.0),
        }
    }

    /// Update send level [0, 1].
    pub fn set_send_level(&mut self, send_level: f32) {
        self.send_level = send_level.clamp(0.0, 1.0);
    }

    /// Current send level [0, 1].
    pub fn send_level(&self) -> f32 {
        self.send_level
    }

    /// IR length actually in use.
    pub fn ir_len(&self) -> usize {
        self.inner.ir_len()
    }

    /// Apply the track's reverb in place. Send level is forwarded to
    /// `Reverb::process` as the wet ratio: at `send=0` this is a true
    /// pass-through; at `send=1` the dry signal is fully replaced by
    /// the convolved version. Stage 1's mixer therefore only needs to
    /// twiddle the send level — there is no parallel "send bus" to
    /// wire up at this stage.
    pub fn process_inplace(&mut self, buf: &mut [f32]) {
        self.inner.process(buf, self.send_level);
    }
}

/// Stage 1 integration receiver. Fixes the per-track DSP chain order:
///
/// ```text
///   buf -> EQ -> Compressor -> Reverb (send_level) -> buf
/// ```
///
/// `send_level` overrides the reverb's stored send level (by writing
/// it through `set_send_level` first) so the mixer can automate it
/// per-buffer without reaching into `TrackReverb` directly.
pub fn process_track_chain(
    eq: &mut Eq3Band,
    comp: &mut Compressor,
    rev: &mut TrackReverb,
    send_level: f32,
    buf: &mut [f32],
) {
    eq.process_inplace(buf);
    comp.process_inplace(buf);
    rev.set_send_level(send_level);
    rev.process_inplace(buf);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// RMS of a buffer.
    fn rms(buf: &[f32]) -> f32 {
        if buf.is_empty() {
            return 0.0;
        }
        let s: f32 = buf.iter().map(|x| x * x).sum();
        (s / buf.len() as f32).sqrt()
    }

    /// Render a sine, run a closure on the buffer, return its peak.
    fn render_sine(sr: f32, freq: f32, secs: f32) -> Vec<f32> {
        let n = (sr * secs) as usize;
        let mut buf = Vec::with_capacity(n);
        let dt = 1.0 / sr;
        for i in 0..n {
            buf.push((2.0 * PI * freq * i as f32 * dt).sin());
        }
        buf
    }

    #[test]
    fn biquad_identity_passthrough() {
        let mut bq = Biquad::identity();
        let mut buf = vec![0.1, -0.2, 0.3, -0.4, 0.5];
        let original = buf.clone();
        bq.process_inplace(&mut buf);
        for (a, b) in buf.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn biquad_zero_input_zero_output() {
        let mut bq = Biquad::low_shelf(44_100.0, 200.0, 6.0, 0.7);
        let mut buf = vec![0.0_f32; 1024];
        bq.process_inplace(&mut buf);
        for v in &buf {
            assert!(v.abs() < 1e-9);
        }
    }

    #[test]
    fn eq3_flat_is_passthrough() {
        let mut eq = Eq3Band::flat();
        let mut buf = render_sine(44_100.0, 1_000.0, 0.05);
        let original = buf.clone();
        eq.process_inplace(&mut buf);
        for (a, b) in buf.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "flat EQ should pass through, got {a} vs {b}"
            );
        }
    }

    /// Low-shelf boost should raise the RMS of a sub-shelf sine, and a
    /// low-shelf cut should lower it. We compare RMS before/after to a
    /// generous tolerance — this is a smoke test for "the cookbook
    /// coefficients shift energy in the right direction" rather than a
    /// numerical magnitude verification (which would require steady-state
    /// settling and a proper FFT bin extraction).
    #[test]
    fn low_shelf_boost_raises_lowband_rms() {
        let sr = 44_100.0;
        // Pick a sine well below the 200 Hz shelf so the full shelf gain
        // applies to it: 80 Hz.
        let buf_in = render_sine(sr, 80.0, 0.5);
        let mut buf = buf_in.clone();
        let mut eq = Eq3Band::new(sr, 12.0, 0.0, 0.0);
        eq.process_inplace(&mut buf);

        // Skip the first ~50 ms so the biquad has settled.
        let skip = (sr * 0.05) as usize;
        let in_rms = rms(&buf_in[skip..]);
        let out_rms = rms(&buf[skip..]);
        assert!(
            out_rms > in_rms * 1.5,
            "low-shelf +12 dB should boost 80 Hz sine RMS noticeably (in={in_rms}, out={out_rms})"
        );
    }

    #[test]
    fn low_shelf_cut_reduces_lowband_rms() {
        let sr = 44_100.0;
        let buf_in = render_sine(sr, 80.0, 0.5);
        let mut buf = buf_in.clone();
        let mut eq = Eq3Band::new(sr, -12.0, 0.0, 0.0);
        eq.process_inplace(&mut buf);
        let skip = (sr * 0.05) as usize;
        let in_rms = rms(&buf_in[skip..]);
        let out_rms = rms(&buf[skip..]);
        assert!(
            out_rms < in_rms * 0.6,
            "low-shelf -12 dB should attenuate 80 Hz sine (in={in_rms}, out={out_rms})"
        );
    }

    /// A high-band signal (5 kHz) should be largely unaffected by a low
    /// shelf at 200 Hz — this catches accidental coefficient inversions
    /// where a "low shelf" actually nukes the highs.
    #[test]
    fn low_shelf_does_not_affect_highs() {
        let sr = 44_100.0;
        let buf_in = render_sine(sr, 5_000.0, 0.2);
        let mut buf = buf_in.clone();
        let mut eq = Eq3Band::new(sr, 12.0, 0.0, 0.0);
        eq.process_inplace(&mut buf);
        let skip = (sr * 0.05) as usize;
        let in_rms = rms(&buf_in[skip..]);
        let out_rms = rms(&buf[skip..]);
        let ratio = out_rms / in_rms;
        assert!(
            (ratio - 1.0).abs() < 0.15,
            "low-shelf at 200 Hz should not move 5 kHz RMS far (ratio={ratio})"
        );
    }

    #[test]
    fn compressor_passes_quiet_signals_through() {
        // Threshold 0.5; signal peaks at 0.1 — should never engage gain
        // reduction. With unity makeup the output should equal the input
        // (modulo float rounding).
        let mut comp = Compressor::new(44_100.0, 0.5, 0.005, 0.05, 1.0);
        let mut buf = render_sine(44_100.0, 440.0, 0.05);
        for s in buf.iter_mut() {
            *s *= 0.1;
        }
        let original = buf.clone();
        comp.process_inplace(&mut buf);
        for (a, b) in buf.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-3);
        }
    }

    #[test]
    fn compressor_reduces_loud_peaks() {
        // Threshold 0.2; signal peaks at 1.0 — gain should pull post-attack
        // peaks down to roughly the threshold. We measure the latter half
        // of the buffer so attack transients don't dominate.
        let sr = 44_100.0_f32;
        let mut comp = Compressor::new(sr, 0.2, 0.001, 0.05, 1.0);
        let mut buf = render_sine(sr, 440.0, 0.5);
        comp.process_inplace(&mut buf);
        let tail = &buf[buf.len() / 2..];
        let peak = tail.iter().fold(0.0_f32, |a, &x| a.max(x.abs()));
        assert!(
            peak < 0.35,
            "compressor should keep tail peak near threshold, got {peak}"
        );
        // Sanity: ensure compressor didn't kill the signal.
        assert!(
            peak > 0.05,
            "compressor should not silence the signal, got {peak}"
        );
    }

    #[test]
    fn compressor_reset_clears_envelope() {
        let mut comp = Compressor::new(44_100.0, 0.2, 0.001, 0.05, 1.0);
        let mut buf = vec![1.0_f32; 1024];
        comp.process_inplace(&mut buf);
        comp.reset();
        // After reset, env is 0.0; first sample passes through unscaled
        // (env starts below threshold).
        let first = comp.process_sample(0.05);
        assert!((first - 0.05).abs() < 1e-6);
    }

    #[test]
    fn track_reverb_zero_send_is_passthrough() {
        let mut rev = TrackReverb::new(crate::reverb::synthetic_body_ir(44_100), 0.0);
        let mut buf = vec![0.3_f32, -0.5, 0.7, -0.2, 0.9];
        let original = buf.clone();
        rev.process_inplace(&mut buf);
        for (a, b) in buf.iter().zip(original.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn track_reverb_send_level_clamped() {
        let mut rev = TrackReverb::new(vec![1.0, 0.5], 5.0);
        assert_eq!(rev.send_level(), 1.0);
        rev.set_send_level(-1.0);
        assert_eq!(rev.send_level(), 0.0);
    }

    #[test]
    fn track_reverb_finite_for_random_input() {
        let mut rev = TrackReverb::new(crate::reverb::synthetic_body_ir(44_100), 0.5);
        let mut state: u32 = 0xC0FFEE;
        let mut buf = vec![0.0_f32; 4096];
        for s in buf.iter_mut() {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            *s = (state as i32 as f32) / (i32::MAX as f32);
        }
        rev.process_inplace(&mut buf);
        for v in &buf {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn process_track_chain_runs_clean_with_flat_settings() {
        let sr = 44_100.0_f32;
        let mut eq = Eq3Band::flat();
        // Generous threshold so the compressor never engages.
        let mut comp = Compressor::new(sr, 10.0, 0.005, 0.05, 1.0);
        let mut rev = TrackReverb::new(crate::reverb::synthetic_body_ir(sr as u32), 0.0);
        let mut buf = render_sine(sr, 440.0, 0.05);
        let original = buf.clone();
        process_track_chain(&mut eq, &mut comp, &mut rev, 0.0, &mut buf);
        // Flat EQ + dormant compressor + send=0 reverb should be a near
        // bit-perfect pass-through. Allow a tiny tolerance for biquad-
        // introduced denormal-clearing arithmetic.
        for (a, b) in buf.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "expected near-passthrough, got {a} vs {b}"
            );
        }
    }

    #[test]
    fn process_track_chain_active_chain_stays_finite() {
        let sr = 44_100.0_f32;
        let mut eq = Eq3Band::new(sr, 6.0, -3.0, 4.0);
        let mut comp = Compressor::new(sr, 0.3, 0.002, 0.05, 1.2);
        let mut rev = TrackReverb::new(crate::reverb::synthetic_body_ir(sr as u32), 0.4);
        let mut buf = render_sine(sr, 440.0, 0.5);
        // Bump the level so the compressor has work to do.
        for s in buf.iter_mut() {
            *s *= 1.5;
        }
        process_track_chain(&mut eq, &mut comp, &mut rev, 0.4, &mut buf);
        for v in &buf {
            assert!(v.is_finite());
            assert!(v.abs() < 5.0, "chain output blew up: {v}");
        }
    }
}
