//! Integration tests for `keysynth::extract::attack`.
//!
//! Establishes a golden test against the SFZ Salamander C4 reference
//! render. The pinned values were measured by running this binary once
//! against `bench-out/REF/sfz_salamander_grand_v3_C4.wav` and read off
//! the assertion failures; tolerances are loose so legitimate refactors
//! (different RMS window length, different slope-fit window, etc.) only
//! trip the test if they materially change the extracted attack shape.

use std::path::PathBuf;

use keysynth::extract::attack::extract_attack;

const REF_WAV: &str = "bench-out/REF/sfz_salamander_grand_v3_C4.wav";

fn ref_wav_path() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest).join(REF_WAV)
}

fn load_mono_f32(path: &std::path::Path) -> (Vec<f32>, f32) {
    let mut reader = hound::WavReader::open(path).expect("open ref wav");
    let spec = reader.spec();
    let sr = spec.sample_rate as f32;

    // Read interleaved samples and downmix to mono.
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| (s.expect("sample") as f32) / max)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.expect("sample"))
            .collect(),
    };

    if spec.channels <= 1 {
        return (samples, sr);
    }
    let ch = spec.channels as usize;
    let mut mono = Vec::with_capacity(samples.len() / ch);
    for frame in samples.chunks_exact(ch) {
        let s: f32 = frame.iter().copied().sum::<f32>() / ch as f32;
        mono.push(s);
    }
    (mono, sr)
}

#[test]
fn golden_sfz_c4_attack() {
    let path = ref_wav_path();
    if !path.exists() {
        eprintln!("ref wav missing at {:?}, skipping", path);
        return;
    }

    let (signal, sr) = load_mono_f32(&path);
    assert!(!signal.is_empty(), "ref wav loaded but empty");

    let env = extract_attack(&signal, sr, 100.0);

    // Pinned (measured 2026-04-25) values for the SFZ Salamander C4
    // reference render:
    //   time_to_peak_s       = 0.0390  (39 ms — Salamander C4 has a
    //                                   slow build-up before the peak)
    //   peak_db              = -7.62
    //   post_peak_slope_db_s = -10.37 dB/sec
    // Tolerances: ±5 ms on time-to-peak, ±3 dB on peak level, ±20%
    // (clamped to ±5 dB/sec) on slope. Loose enough that legitimate
    // refactors of the RMS window length or the slope-fit window do
    // not trip the test, tight enough to catch a real regression in
    // the extractor or the upstream render.

    let want_ttp = 0.0390_f32;
    assert!(
        (env.time_to_peak_s - want_ttp).abs() <= 0.005,
        "time_to_peak_s = {} not within ±5 ms of pinned {}",
        env.time_to_peak_s,
        want_ttp
    );

    let want_peak_db = -7.62_f32;
    assert!(
        (env.peak_db - want_peak_db).abs() <= 3.0,
        "peak_db = {} not within ±3 dB of pinned {}",
        env.peak_db,
        want_peak_db
    );

    assert!(
        env.post_peak_slope_db_s < 0.0,
        "post_peak_slope_db_s = {} should be NEGATIVE (always decaying)",
        env.post_peak_slope_db_s
    );

    let want_slope = -10.37_f32;
    let slope_tol = (want_slope.abs() * 0.20).max(5.0);
    assert!(
        (env.post_peak_slope_db_s - want_slope).abs() <= slope_tol,
        "post_peak_slope_db_s = {} not within ±{} of pinned {}",
        env.post_peak_slope_db_s,
        slope_tol,
        want_slope
    );

    // Envelope sanity: ~100 hops at 1 ms / 100 ms window.
    assert!(
        env.rms_envelope_db.len() >= 90 && env.rms_envelope_db.len() <= 110,
        "rms_envelope_db.len() = {}",
        env.rms_envelope_db.len()
    );
}
