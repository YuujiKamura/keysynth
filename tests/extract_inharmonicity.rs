//! Integration tests for `keysynth::extract::inharmonicity::fit_b`.
//!
//! Round-trip coverage already lives in the in-module `#[cfg(test)] mod
//! tests` block. This file holds the **golden** check against the SFZ
//! Salamander C4 reference: load
//! `bench-out/REF/sfz_salamander_grand_v3_C4.wav`, decompose it via
//! `extract::decompose::decompose`, then assert that `fit_b` reports
//! the same B / R² that the existing `analyse` binary reports for that
//! file.
//!
//! See issue #3.

use keysynth::extract::decompose::{decompose, Partial};
use keysynth::extract::inharmonicity::fit_b;
use std::path::Path;

/// Read a 16-bit PCM mono/stereo WAV and return mono f32 samples + sample
/// rate. Stereo is downmixed by averaging.
fn read_wav_mono(path: &Path) -> Option<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path).ok()?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let sr = spec.sample_rate;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample as i32;
            let scale = (1i64 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / scale)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
    };
    if channels <= 1 {
        Some((samples, sr))
    } else {
        let mut mono = Vec::with_capacity(samples.len() / channels);
        for frame in samples.chunks_exact(channels) {
            let sum: f32 = frame.iter().sum();
            mono.push(sum / channels as f32);
        }
        Some((mono, sr))
    }
}

/// Golden test on the SFZ Salamander C4 reference.
///
/// `#[ignore]` because `extract::decompose::decompose` is being
/// implemented in parallel under issue #3 P1 and is still a `todo!()`
/// stub at the time this test was written. The reference WAV is already
/// committed at `bench-out/REF/sfz_salamander_grand_v3_C4.wav`
/// (commit `0fb3840`), so once decompose lands this test should pass
/// against real data.
///
/// Run with `cargo test --test extract_inharmonicity -- --ignored`
/// after decompose lands. See issue #3.
#[test]
#[ignore = "needs decompose() (issue #3 P1) — run with --ignored once that lands"]
fn golden_sfz_c4() {
    let wav_path = Path::new("bench-out")
        .join("REF")
        .join("sfz_salamander_grand_v3_C4.wav");
    let (sig, sr) = match read_wav_mono(&wav_path) {
        Some(x) => x,
        None => panic!("{} not found or unreadable.", wav_path.display()),
    };

    let f0 = 261.63_f32; // C4
    let max_partials = 16;
    let partials: Vec<Partial> = decompose(&sig, sr as f32, f0, max_partials);
    assert!(
        !partials.is_empty(),
        "decompose() returned no partials for REF_sfz_C4.wav"
    );

    let fit = fit_b(&partials);

    // Existing `analyse` binary reports B ≈ 2.859e-4, R² ≈ 0.999. We
    // allow some drift: B in [2.5e-4, 3.2e-4] (≈ ±12% around 2.86e-4)
    // and R² ≥ 0.95 to absorb decompose-impl differences.
    assert!(
        fit.b >= 2.5e-4 && fit.b <= 3.2e-4,
        "B out of golden band [2.5e-4, 3.2e-4]: got {:e}",
        fit.b
    );
    assert!(
        fit.r_squared >= 0.95,
        "R² below golden floor 0.95: got {}",
        fit.r_squared
    );
    assert_eq!(
        fit.n_used,
        partials.len(),
        "n_used should equal the partial count returned by decompose()"
    );
}
