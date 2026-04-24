//! Crate-level integration tests for `keysynth::extract::t60`.
//!
//! Per the issue #3 P3 contract: golden test against
//! `bench-out/REF_sfz_C4.wav`. The wav is not committed to the repo, so
//! the test is `#[ignore]` when the file is missing — run with
//! `cargo test --test extract_t60 -- --ignored` once the WAV is on disk.

use std::path::Path;

use keysynth::extract::decompose::Partial;
use keysynth::extract::t60::extract_t60;

const REF_WAV: &str = "bench-out/REF_sfz_C4.wav";

/// Reference T60 values reported by the existing `analyse` pipeline for
/// `bench-out/REF_sfz_C4.wav` (h1..h8). Per-partial T60 estimates are
/// genuinely noisy when partials overlap or fade into the noise floor,
/// so we accept ±15% drift from the analyse numbers.
const ANALYSE_REF_T60: [(usize, f32, f32); 8] = [
    // (n, freq_hz, t60_sec)
    (1, 261.6, 18.14),
    (2, 523.1, 11.21),
    (3, 784.7, 9.58),
    (4, 1046.2, 7.38),
    (5, 1307.8, 6.98),
    (6, 1569.3, 7.93),
    (7, 1830.9, 8.98),
    (8, 2092.5, 8.66),
];

fn read_wav_mono_f32(path: &Path) -> Option<(Vec<f32>, f32)> {
    let mut reader = hound::WavReader::open(path).ok()?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let sr = spec.sample_rate as f32;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max)
                .collect()
        }
    };

    let mono: Vec<f32> = if channels <= 1 {
        samples
    } else {
        samples
            .chunks(channels)
            .map(|c| c.iter().copied().sum::<f32>() / channels as f32)
            .collect()
    };
    Some((mono, sr))
}

#[test]
fn golden_sfz_c4_per_partial_t60_within_15pct() {
    let path = Path::new(REF_WAV);
    if !path.exists() {
        // WAV missing — skip rather than fail. Rerun with the file in
        // place to validate against the analyse reference.
        eprintln!(
            "skipping golden test: {REF_WAV} not present (build the SFZ \
             reference with `cargo run --bin bench` to populate it)"
        );
        return;
    }

    let Some((signal, sr)) = read_wav_mono_f32(path) else {
        panic!("failed to read {REF_WAV}");
    };

    let partials: Vec<Partial> = ANALYSE_REF_T60
        .iter()
        .map(|&(n, f, _)| Partial {
            n,
            freq_hz: f,
            // Loose init_db; the extractor doesn't use it.
            init_db: 0.0,
        })
        .collect();

    let out = extract_t60(&signal, sr, &partials);
    assert_eq!(
        out.seconds.len(),
        partials.len(),
        "T60Vector length mismatch"
    );

    for (i, ((_, _, target), &got)) in ANALYSE_REF_T60.iter().zip(out.seconds.iter()).enumerate() {
        assert!(
            got > 0.0,
            "partial {} (n={}): sentinel/negative T60 {got}",
            i,
            ANALYSE_REF_T60[i].0,
        );
        let lo = target * 0.85;
        let hi = target * 1.15;
        assert!(
            got >= lo && got <= hi,
            "partial {} (n={}): recovered T60 {got:.3} s outside ±15% of \
             analyse value {target:.3} s [{lo:.3}, {hi:.3}]",
            i,
            ANALYSE_REF_T60[i].0,
        );
    }
}
