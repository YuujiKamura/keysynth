//! Crate-level golden test for [`keysynth::extract::decompose`] against
//! the SoundFont reference recording. Pinned values come from the
//! existing `analyse` binary's harmonic-tracker output for
//! `bench-out/REF/sfz_salamander_grand_v3_C4.wav` (committed on main at
//! `0fb3840`). See issue #3.

use std::path::PathBuf;

use keysynth::extract::decompose::decompose;

/// Load `bench-out/REF/sfz_salamander_grand_v3_C4.wav` as mono f32 at
/// 44.1 kHz. Returns `None` if the file is missing — the test that uses
/// this is `#[ignore]`d when that happens. (Path corrected from the
/// pre-`0fb3840` brief that called the file `bench-out/REF_sfz_C4.wav`.)
fn load_ref_sfz_c4() -> Option<(Vec<f32>, f32)> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("bench-out");
    path.push("REF");
    path.push("sfz_salamander_grand_v3_C4.wav");
    if !path.exists() {
        return None;
    }
    let mut reader = hound::WavReader::open(&path).ok()?;
    let spec = reader.spec();
    let sr = spec.sample_rate as f32;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(Result::ok).collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / max)
                .collect()
        }
    };

    // Down-mix to mono if needed.
    let mono: Vec<f32> = if channels <= 1 {
        samples
    } else {
        samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    Some((mono, sr))
}

/// Pinned per-partial expectations for the SFZ Salamander C4 reference
/// at f0 = 261.63 Hz. Frequencies are in Hz, tolerance widens for higher
/// harmonics because the SFZ Salamander C4 has progressive piano-style
/// stretching.
struct PartialExpect {
    n: usize,
    freq_hz: f32,
    freq_tol_hz: f32,
}

const EXPECTED: &[PartialExpect] = &[
    PartialExpect {
        n: 1,
        freq_hz: 261.6,
        freq_tol_hz: 1.5,
    },
    PartialExpect {
        n: 2,
        freq_hz: 523.1,
        freq_tol_hz: 2.0,
    },
    PartialExpect {
        n: 3,
        freq_hz: 785.6,
        freq_tol_hz: 2.5,
    },
    PartialExpect {
        n: 4,
        freq_hz: 1048.8,
        freq_tol_hz: 3.0,
    },
    PartialExpect {
        n: 5,
        freq_hz: 1311.7,
        freq_tol_hz: 3.5,
    },
    PartialExpect {
        n: 6,
        freq_hz: 1577.3,
        freq_tol_hz: 4.0,
    },
    PartialExpect {
        n: 7,
        freq_hz: 1843.7,
        freq_tol_hz: 4.5,
    },
    PartialExpect {
        n: 8,
        freq_hz: 2111.5,
        freq_tol_hz: 5.0,
    },
];

// `#[ignore]`d on this worktree because the SFZ Salamander reference WAV
// landed on main at commit `0fb3840` AFTER this worktree forked, so the
// file isn't physically present here. Once the worktree is rebased onto
// `0fb3840` (or later) the file will be at
// `bench-out/REF/sfz_salamander_grand_v3_C4.wav` and this test can be
// run with `cargo test --test extract_decompose -- --ignored`. See
// issue #3 for the pinned per-partial values' provenance.
#[test]
#[ignore = "requires bench-out/REF/sfz_salamander_grand_v3_C4.wav (issue #3)"]
fn golden_ref_sfz_c4() {
    let Some((signal, sr)) = load_ref_sfz_c4() else {
        panic!(
            "golden_ref_sfz_c4: bench-out/REF/sfz_salamander_grand_v3_C4.wav \
             missing — rebase onto main (>= 0fb3840) to pull the committed \
             reference, or see issue #3 for regeneration steps."
        );
    };

    assert!(
        (sr - 44100.0).abs() < 1.0,
        "sfz_salamander_grand_v3_C4.wav must be 44.1 kHz, got {}",
        sr
    );

    let partials = decompose(&signal, 44100.0, 261.63, 8);

    assert_eq!(
        partials.len(),
        8,
        "expected exactly 8 partials, got {}",
        partials.len()
    );

    for (got, want) in partials.iter().zip(EXPECTED.iter()) {
        assert_eq!(got.n, want.n, "partial index mismatch");
        let err_hz = (got.freq_hz - want.freq_hz).abs();
        assert!(
            err_hz <= want.freq_tol_hz,
            "h{}: got {:.2} Hz, want {} ± {} Hz (err {:.3})",
            want.n,
            got.freq_hz,
            want.freq_hz,
            want.freq_tol_hz,
            err_hz
        );
    }

    // h1 should be the strongest (or near-strongest) partial in the SFZ
    // recording, so its init_db lands in the [-3.0, 0.0] band.
    let h1 = partials.iter().find(|p| p.n == 1).expect("h1 missing");
    assert!(
        h1.init_db <= 0.0 && h1.init_db >= -3.0,
        "h1 init_db expected in [-3.0, 0.0] dB, got {}",
        h1.init_db
    );
}
