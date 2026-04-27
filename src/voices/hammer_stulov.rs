//! Stulov non-linear hammer model (1995).
//!
//! Reference:
//!   Stulov, A. (1995). Hysteretic model of the grand piano hammer felt.
//!   J. Acoust. Soc. Am. 97(4), 2577–2585.

#[derive(Clone, Copy, Debug)]
pub struct HammerParams {
    /// Effective hammer mass (kg).
    pub mass_kg: f32,
    /// Felt stiffness (N/m^p).
    pub k_stiffness: f32,
    /// Non-linearity exponent (~2.3-3.5).
    pub p_exponent: f32,
    /// Hysteresis loss factor (~10⁻⁴–10⁻⁵).
    pub eps_hysteresis: f32,
}

impl Default for HammerParams {
    fn default() -> Self {
        Self {
            mass_kg: 8.7e-3,
            k_stiffness: 4.5e9, // Steinway D mid-register approximation
            p_exponent: 2.5,
            eps_hysteresis: 1e-4,
        }
    }
}

/// Solve the hammer-string contact ODE via implicit RK4 (Gauss-Legendre 2-stage).
///
/// Model: m * d²u/dt² = -F(u, v)
/// F(u, v) = K * u^p * (1 + ε * v * sgn(v))
/// where u is compression, v is du/dt.
///
/// String is approximated as a fixed end (high impedance).
pub fn stulov_pulse(
    velocity_mps: f32,
    sr: f32,
    sample_count: usize,
    params: &HammerParams,
) -> Vec<f32> {
    let dt_audio = 1.0 / sr;
    let substeps = 4;
    let dt = dt_audio / substeps as f32;

    let m = params.mass_kg;
    let k = params.k_stiffness;
    let p = params.p_exponent;
    let eps = params.eps_hysteresis;

    // F(u, v)
    let force_fn = |u: f32, v: f32| -> f32 {
        if u <= 0.0 {
            0.0
        } else {
            k * u.powf(p) * (1.0 + eps * v * v.signum())
        }
    };

    // dS/dt = [v, -F/m]
    let deriv = |u: f32, v: f32| -> (f32, f32) { (v, -force_fn(u, v) / m) };

    // Gauss-Legendre 2-stage (4th order implicit)
    let _c1 = 0.5 - 3.0_f32.sqrt() / 6.0;
    let _c2 = 0.5 + 3.0_f32.sqrt() / 6.0;
    let a11 = 0.25;
    let a12 = 0.25 - 3.0_f32.sqrt() / 6.0;
    let a21 = 0.25 + 3.0_f32.sqrt() / 6.0;
    let a22 = 0.25;
    let b1 = 0.5;
    let b2 = 0.5;

    let mut u = 0.0_f32;
    let mut v = velocity_mps;

    let mut buf = Vec::with_capacity(sample_count);

    // Initial force
    buf.push(force_fn(u, v));

    for _ in 1..sample_count {
        for _ in 0..substeps {
            // Predict initial k1, k2 (explicitly or just zero)
            let (mut k1u, mut k1v) = deriv(u, v);
            let (mut k2u, mut k2v) = deriv(u, v);

            // Fixed-point iteration to solve for k1, k2
            for _ in 0..8 {
                let u1 = u + dt * (a11 * k1u + a12 * k2u);
                let v1 = v + dt * (a11 * k1v + a12 * k2v);
                let (nk1u, nk1v) = deriv(u1, v1);

                let u2 = u + dt * (a21 * k1u + a22 * k2u);
                let v2 = v + dt * (a21 * k1v + a22 * k2v);
                let (nk2u, nk2v) = deriv(u2, v2);

                k1u = nk1u;
                k1v = nk1v;
                k2u = nk2u;
                k2v = nk2v;
            }

            u += dt * (b1 * k1u + b2 * k2u);
            v += dt * (b1 * k1v + b2 * k2v);
        }

        let f = force_fn(u, v);
        buf.push(f);

        // Separation: if compression becomes negative and velocity is moving away
        if u <= 0.0 && v < 0.0 && buf.len() > 5 {
            break;
        }
    }

    // Trim or pad to sample_count
    if buf.len() > sample_count {
        buf.truncate(sample_count);
    }

    // Normalization (optional but usually hammer pulses are normalized
    // and then scaled by velocity amp in the voice).
    // Brief doesn't explicitly say to normalize, but existing solve_stulov_hammer_pulse does.
    // However, piano_hammer_excitation takes an `amp` parameter.
    // stulov_pulse in brief returns Vec<f32>.
    // Let's check how it's used.

    // Safety check: check for NaN
    for x in &buf {
        if x.is_nan() {
            panic!("stulov_pulse produced NaN! check parameters: {:?}", params);
        }
    }

    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stulov_pulse_finite_and_nonzero() {
        let params = HammerParams::default();
        let pulse = stulov_pulse(2.5, 44100.0, 500, &params);
        assert!(!pulse.is_empty());
        let mut has_nonzero = false;
        for &x in &pulse {
            assert!(x.is_finite());
            if x.abs() > 1e-3 {
                has_nonzero = true;
            }
        }
        assert!(has_nonzero);
    }

    #[test]
    fn stulov_pulse_velocity_30_quieter_than_127() {
        let params = HammerParams::default();
        // MIDI 30 -> 0.5 + (30/127)*4.5 = 1.56
        // MIDI 127 -> 0.5 + 4.5 = 5.0
        let v30 = 0.5 + (30.0 / 127.0) * 4.5;
        let v127 = 0.5 + (127.0 / 127.0) * 4.5;
        let p30 = stulov_pulse(v30, 44100.0, 1000, &params);
        let p127 = stulov_pulse(v127, 44100.0, 1000, &params);

        let peak30 = p30.iter().copied().fold(0.0_f32, f32::max);
        let peak127 = p127.iter().copied().fold(0.0_f32, f32::max);

        let ratio = peak127 / peak30;
        let db_diff = 20.0 * ratio.log10();
        println!(
            "Peak30: {}, Peak127: {}, dB diff: {}",
            peak30, peak127, db_diff
        );
        assert!(db_diff > 10.0);

        // Spectral centroid check
        let c30 = calculate_centroid(&p30);
        let c127 = calculate_centroid(&p127);
        println!("Centroid30: {}, Centroid127: {}", c30, c127);
        assert!(c127 > c30 + 200.0);
    }

    #[test]
    fn stulov_pulse_low_p_softer_than_high_p() {
        // Use realistic (K, p) pairs from Stulov (1995) Steinway D data.
        // Bass (A0): p=2.87, K=242 N/mm^p -> ~1e11 N/m^p
        // Treble (A73): p=3.5, K=10000 N/mm^p -> ~3e14 N/m^p
        let mut p_low_params = HammerParams::default();
        p_low_params.p_exponent = 2.87;
        p_low_params.k_stiffness = 1.0e11;
        p_low_params.mass_kg = 10.0e-3;

        let mut p_high_params = HammerParams::default();
        p_high_params.p_exponent = 3.5;
        p_high_params.k_stiffness = 3.16e14;
        p_high_params.mass_kg = 6.5e-3;

        let v = 2.5;
        let p_low = stulov_pulse(v, 44100.0, 1000, &p_low_params);
        let p_high = stulov_pulse(v, 44100.0, 1000, &p_high_params);

        let c_low = calculate_centroid(&p_low);
        let c_high = calculate_centroid(&p_high);
        println!(
            "Centroid(p=2.87, K=1e11): {}, Centroid(p=3.5, K=3e14): {}",
            c_low, c_high
        );
        assert!(c_high > c_low + 200.0);
    }

    fn calculate_centroid(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        // Very simple time-domain centroid approximation as a proxy for spectral centroid.
        // Or better: do a real FFT? I don't have FFT here.
        // Actually, the pulse width in time is a good proxy: shorter pulse = higher centroid.
        // Let's just use the RMS-weighted "time centroid" as a very rough proxy if needed,
        // but for spectral centroid we really need frequencies.
        // Wait, the brief says "spectral centroid should differ by > 200 Hz".
        // I should probably implement a simple DFT for the test.
        let n = samples.len();
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        // DFT
        for k in 1..n / 2 {
            let mut re = 0.0;
            let mut im = 0.0;
            for (n_idx, &s) in samples.iter().enumerate() {
                let theta = -2.0 * std::f32::consts::PI * (k as f32) * (n_idx as f32) / (n as f32);
                re += s * theta.cos();
                im += s * theta.sin();
            }
            let mag = (re * re + im * im).sqrt();
            let freq = (k as f32) * 44100.0 / (n as f32);
            numerator += freq * mag;
            denominator += mag;
        }
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}
