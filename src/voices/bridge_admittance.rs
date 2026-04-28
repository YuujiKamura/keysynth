//! Frequency-dependent piano-bridge admittance, expressed as a 6-tap FIR.
//!
//! Tier 2.2 of the Pianoteq-beat campaign (issue #32). A real piano bridge
//! is not a flat scalar coupler: its mechanical input admittance `Y(ω)`
//! varies by an order of magnitude over the audible range. Pre-T2.2
//! `PianoVoice` shipped a single `coupling_per_string: f32` for the
//! soundboard ↔ string return path, which makes the bridge feel inert and
//! flat — every harmonic of every string couples by the same amount, so
//! the body never colours the radiation pattern with a frequency-dependent
//! resonance shape.
//!
//! # Reference admittance shape
//!
//! Suzuki (1986, JASA 80) and Giordano & Korty (1996, JASA 100) report:
//!
//!   * Below ~250 Hz: `|Y(ω)|` is roughly flat at the "low-frequency
//!     plateau" (the soundboard mass dominates the local impedance, the
//!     bridge moves like a piston).
//!   * 250 Hz – 1 kHz: a series of sharp `|Y|` peaks corresponding to
//!     soundboard plate modes, with antiresonance dips between them.
//!   * 1 kHz – 5 kHz: ~−6 dB/oct mass-controlled rolloff with smaller
//!     residual modal ripple.
//!   * Above 5 kHz: smooth `−6 dB/oct` mass-line.
//!
//! A 6-tap FIR cannot resolve the 250 Hz – 1 kHz modal peaks (those live
//! in the soundboard modal resonator we already have), but it CAN model
//! the broad low-pass envelope plus a controlled dip in the 4 – 8 kHz
//! region. That is what the modelled-piano literature refers to as the
//! *macroscopic* admittance — the modal structure is layered on top by
//! `crate::soundboard::Soundboard`.
//!
//! # FIR design
//!
//! Linear-phase Type-I, symmetric coefficients, normalised so the filter
//! has DC gain = 1.0. PianoVoice scales the filter output by the preset's
//! `coupling_per_string` so the *baseline* (DC) feedback amount matches
//! the pre-T2.2 audible behaviour exactly — only the high-frequency
//! component of the round-trip loop changes.
//!
//! Coefficient values were hand-fit (least-squares against a Suzuki
//! fig.7 abstraction with the modal peaks averaged out) so the magnitude
//! response satisfies:
//!
//!   * `|H(0)|` = 1.000  (DC gain unity by construction)
//!   * `|H(2 kHz)|` ≈ 0.93
//!   * `|H(5 kHz)|` ≈ 0.62 (≈ −4.2 dB)
//!   * `|H(10 kHz)|` ≈ 0.09 (≈ −20.5 dB)
//!   * `|H(Nyquist)|` = 0  (zero by Type-I linear-phase construction)
//!
//! The unit tests below pin those magnitudes and reject any silent edit
//! to the table.
//!
//! # Why not the soundboard mode bank itself?
//!
//! The soundboard module already runs a 12 – 32 mode resonator stack
//! (Bank/Lehtonen DWS layout). That resonator provides the modal
//! structure — sharp peaks and dips — but it does NOT change the broad
//! coupling strength as a function of frequency. The string ↔ bridge
//! feedback path, before it even hits the resonator, used to be a flat
//! scalar; this FIR puts the broad admittance shape onto that path so
//! the modes are excited with frequency-appropriate weight.

#![allow(dead_code)]

use crate::synth::SmallFir;

/// Baked piano-bridge admittance, 6-tap FIR. Symmetric (linear-phase
/// Type-I), DC-normalised. Caller scales the FIR output by the per-preset
/// `coupling_per_string` to recover the legacy DC coupling amplitude.
///
/// Sums to exactly 1.0 (validated by `bridge_admittance_dc_gain_unity`).
pub const BRIDGE_ADMITTANCE_FIR_COEFS: [f32; 6] = [0.06, 0.18, 0.26, 0.26, 0.18, 0.06];

/// Construct a fresh `SmallFir<6>` initialised with the bridge admittance
/// coefficient table and zero state. Cheap — the constructor is just a
/// pair of array copies.
pub fn make_bridge_admittance_fir() -> SmallFir<6> {
    SmallFir::new(BRIDGE_ADMITTANCE_FIR_COEFS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// `|H(e^jω)|` at angular frequency `omega_rad_per_sample` for the
    /// hard-coded admittance table. Used to pin the magnitude-response
    /// landmarks quoted in the module docs.
    fn admittance_magnitude(omega: f32) -> f32 {
        let mut re = 0.0_f32;
        let mut im = 0.0_f32;
        for (k, &c) in BRIDGE_ADMITTANCE_FIR_COEFS.iter().enumerate() {
            re += c * (omega * k as f32).cos();
            im -= c * (omega * k as f32).sin();
        }
        (re * re + im * im).sqrt()
    }

    fn omega_for_hz(hz: f32, sr: f32) -> f32 {
        2.0 * PI * hz / sr
    }

    #[test]
    fn bridge_admittance_dc_gain_unity() {
        let sum: f32 = BRIDGE_ADMITTANCE_FIR_COEFS.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "expected DC gain 1.0, got {sum}");
        let fir = make_bridge_admittance_fir();
        assert!((fir.dc_gain() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bridge_admittance_is_lowpass_shape() {
        // Sanity: passband at low audio freq, attenuation rising
        // monotonically through the audible band, zero at Nyquist.
        let sr = 44_100.0;
        let h_dc = admittance_magnitude(0.0);
        let h_500 = admittance_magnitude(omega_for_hz(500.0, sr));
        let h_2k = admittance_magnitude(omega_for_hz(2_000.0, sr));
        let h_5k = admittance_magnitude(omega_for_hz(5_000.0, sr));
        let h_10k = admittance_magnitude(omega_for_hz(10_000.0, sr));
        let h_nyq = admittance_magnitude(PI);

        assert!((h_dc - 1.0).abs() < 1e-5);
        assert!(h_500 > 0.97 && h_500 <= 1.0);
        assert!(h_2k > 0.85 && h_2k < 0.97);
        assert!(h_5k > 0.55 && h_5k < 0.75);
        assert!(h_10k > 0.05 && h_10k < 0.20);
        assert!(h_nyq < 1e-5, "Type-I linear-phase: zero at Nyquist");

        // Monotone-decreasing across the landmarks we pinned.
        assert!(h_dc >= h_500);
        assert!(h_500 >= h_2k);
        assert!(h_2k >= h_5k);
        assert!(h_5k >= h_10k);
        assert!(h_10k >= h_nyq);
    }

    #[test]
    fn bridge_admittance_high_freq_attenuation_db() {
        // Suzuki's reported admittance drops by ≥ 10 dB between 1 kHz
        // and 8 kHz. Verify the FIR matches that order of magnitude so a
        // future coefficient edit can't silently flatten the bridge.
        let sr = 44_100.0;
        let h_1k = admittance_magnitude(omega_for_hz(1_000.0, sr));
        let h_8k = admittance_magnitude(omega_for_hz(8_000.0, sr));
        let drop_db = 20.0 * (h_1k / h_8k.max(1e-9)).log10();
        assert!(
            (8.0..=20.0).contains(&drop_db),
            "expected 8–20 dB rolloff between 1 kHz and 8 kHz, got {drop_db:.2} dB"
        );
    }

    #[test]
    fn smallfir_step_matches_direct_convolution() {
        let mut fir = make_bridge_admittance_fir();
        let xs = [1.0_f32, 0.5, -0.3, 0.0, 0.7, 0.2, -0.1, 0.8, 0.0, 0.0];
        let ys: Vec<f32> = xs.iter().map(|&x| fir.process(x)).collect();
        // Reference: y[n] = sum_k c[k] * x[n-k] (zero-padded).
        for n in 0..xs.len() {
            let mut expected = 0.0_f32;
            for k in 0..6 {
                if n >= k {
                    expected += BRIDGE_ADMITTANCE_FIR_COEFS[k] * xs[n - k];
                }
            }
            assert!(
                (ys[n] - expected).abs() < 1e-5,
                "n={n}: ys={} expected={}",
                ys[n],
                expected
            );
        }
    }

    #[test]
    fn smallfir_state_resets_to_silence() {
        let mut fir = make_bridge_admittance_fir();
        for _ in 0..6 {
            let _ = fir.process(0.5);
        }
        fir.reset();
        // After reset, processing zero must return zero.
        for _ in 0..6 {
            let y = fir.process(0.0);
            assert!(y.abs() < 1e-9);
        }
    }

    #[test]
    fn smallfir_impulse_response_returns_coefs() {
        let mut fir = make_bridge_admittance_fir();
        let mut h = [0.0_f32; 6];
        h[0] = fir.process(1.0);
        for i in 1..6 {
            h[i] = fir.process(0.0);
        }
        for i in 0..6 {
            assert!(
                (h[i] - BRIDGE_ADMITTANCE_FIR_COEFS[i]).abs() < 1e-6,
                "h[{i}] = {} vs coef {}",
                h[i],
                BRIDGE_ADMITTANCE_FIR_COEFS[i]
            );
        }
    }
}
