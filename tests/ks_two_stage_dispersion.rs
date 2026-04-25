//! Integration test for the 2-stage allpass dispersion in `KsString`
//! (issue #3 P2).
//!
//! The unit tests in `src/synth.rs` cover the bypass-equivalence contract
//! (1-arg API stays bit-identical), the "actually does something"
//! divergence check, monotonic stretching of partials, and stability under
//! the upper-range coefficient pair.
//!
//! What this file adds: an end-to-end check that the new topology
//! introduces a *measurable extra degree of freedom* in the partial-
//! frequency domain. With the same primary coefficient `ap1`, engaging
//! the second stage with `ap2 > 0` must produce STRICTLY MORE
//! cents-from-harmonic deviation across h2..h12 than the 1-stage path.
//!
//! Metric choice: we use mean-absolute deviation of partial frequencies
//! from `n · f1` (in cents) rather than the linearised-LS Fletcher B fit
//! because the bare KS string under a 1st-order allpass produces a
//! phase-delay curve that is NOT quadratic-in-n, so the `B · n²` fit
//! returns near-zero noise and is too unstable for a CI gate. The cents-
//! deviation magnitude IS robust — it captures the same "more
//! inharmonicity" idea the brief asks for, just measured directly.
//!
//! This is NOT a "matches the SFZ Salamander C4 reference" assertion —
//! that's a P4 tuneloop concern. We only assert that the topology has
//! gained a real partial-shaping knob, which is the structural ceiling
//! P2 was commissioned to break.

use keysynth::extract::decompose::decompose;
use keysynth::synth::{hammer_excitation, KsString};

const SR: f32 = 44_100.0;

/// Render `total_samples` from a fresh KS string at `freq` with the given
/// dispersion config. `ap2 == 0.0` keeps the second stage bypassed (legacy
/// 1-stage path); otherwise the 2-stage cascade is engaged. Decay is held
/// at 0.999 so high partials live long enough for `decompose` to
/// time-average them above the SNR floor.
fn render_ks(freq: f32, ap1: f32, ap2: f32, total_samples: usize) -> Vec<f32> {
    let (n, _) = KsString::delay_length_compensated(SR, freq, ap1);
    // Narrow hammer (3 samples) → broad excitation spectrum so all partials
    // we care about (h1..h12) start with comparable energy.
    let buf = hammer_excitation(n, 3, 1.0);
    let mut s = if ap2 == 0.0 {
        KsString::with_buf(SR, freq, buf, 0.999, ap1)
    } else {
        KsString::with_buf(SR, freq, buf, 0.999, ap1).with_two_stage_dispersion(ap1, ap2)
    };
    let mut out = Vec::with_capacity(total_samples);
    for _ in 0..total_samples {
        out.push(s.step());
    }
    out
}

/// Decompose the signal and return mean-absolute deviation (in cents) of
/// h2..hN from `n · f1`. Larger value = more inharmonicity = more
/// stiffness in the loop. Returns `(mean_abs_cents, n_partials)`.
fn measure_inharmonicity_cents(signal: &[f32], f0: f32) -> (f32, usize) {
    let partials = decompose(signal, SR, f0, 12);
    let n = partials.len();
    let f1 = partials
        .iter()
        .find(|p| p.n == 1)
        .map(|p| p.freq_hz)
        .unwrap_or(f0);
    if f1 <= 0.0 || n < 4 {
        return (0.0, n);
    }
    let mut sum = 0.0_f64;
    let mut count = 0_u32;
    for p in partials.iter().filter(|p| p.n >= 2) {
        let cents = 1200.0 * (p.freq_hz / (p.n as f32 * f1)).log2();
        sum += cents.abs() as f64;
        count += 1;
    }
    if count == 0 {
        (0.0, n)
    } else {
        ((sum / count as f64) as f32, n)
    }
}

/// Contrast test: same primary coefficient `ap1`, second stage off vs on.
/// With `ap2 != 0` the 2-stage path should produce a measurably larger
/// mean-absolute partial deviation than the 1-stage one — that's the
/// new degree of freedom the P2 topology unlocked, observed directly in
/// the partial frequencies rather than through a fragile B fit.
#[test]
fn two_stage_dispersion_increases_partial_deviation() {
    // C4 — matches the SFZ Salamander reference note.
    let f0 = 261.63_f32;
    // 1.5 s, plenty for 12 partials.
    let total_samples = (SR * 1.5) as usize;
    // Larger coefficients than the piano voices use (0.18) so the
    // stiffness response is unambiguous against STFT bin-quantisation
    // noise on a 4096-pt FFT. P4 will tune `ap2` for piano voices to a
    // smaller, perceptually-tuned value; here we only need values big
    // enough for the topology check to be statistically clean.
    let ap1 = 0.30_f32;
    let ap2 = 0.30_f32;

    let sig_one = render_ks(f0, ap1, 0.0, total_samples);
    let sig_two = render_ks(f0, ap1, ap2, total_samples);

    let (cents_one, n_one) = measure_inharmonicity_cents(&sig_one, f0);
    let (cents_two, n_two) = measure_inharmonicity_cents(&sig_two, f0);

    assert!(
        n_one >= 4,
        "1-stage decompose found {n_one} partials, need ≥4 for the comparison"
    );
    assert!(
        n_two >= 4,
        "2-stage decompose found {n_two} partials, need ≥4 for the comparison"
    );

    // Require ≥0.3 cents of additional deviation. Empirical baseline at
    // ap1=ap2=0.30 is ~0.5 cents (1-stage) vs ~1.0 cents (2-stage), so
    // 0.3 cents leaves headroom against bin-quantisation jitter while
    // still failing if the second stage ever silently no-ops.
    let delta = cents_two - cents_one;
    assert!(
        delta >= 0.3,
        "2-stage path must add ≥0.3 cents mean partial deviation: \
         1-stage mean|cents|={cents_one:.3} (n={n_one}), \
         2-stage mean|cents|={cents_two:.3} (n={n_two}), \
         Δ={delta:.3}"
    );
}
