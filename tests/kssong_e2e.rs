#![cfg(feature = "native")]

use std::path::PathBuf;
use std::process::Command;

use keysynth::song::{parse_chord, parse_progression, parse_roman, Key, PitchClass, Quality};

fn unique_wav_path(stem: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("kssong-{stem}-{nanos}.wav"))
}

fn read_wav_peak_and_frames(path: &std::path::Path) -> (f32, usize) {
    let mut reader = hound::WavReader::open(path).expect("wav should open");
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(Result::ok).collect(),
        hound::SampleFormat::Int => {
            let max = (1_i64 << (spec.bits_per_sample.saturating_sub(1))) as f32;
            reader
                .samples::<i32>()
                .filter_map(Result::ok)
                .map(|sample| sample as f32 / max)
                .collect()
        }
    };
    let frames = samples.len() / channels;
    let peak = samples
        .iter()
        .fold(0.0_f32, |acc, sample| acc.max(sample.abs()));
    (peak, frames)
}

#[test]
fn gate_1_parse_c_major_close_voicing() {
    let chord = parse_chord("C").expect("C should parse");
    assert_eq!(chord.root, PitchClass::C);
    assert_eq!(chord.quality, Quality::Major);
    assert_eq!(chord.voice(), vec![60, 64, 67, 72]);
}

#[test]
fn gate_2_parse_a_minor_close_voicing() {
    let chord = parse_chord("Am").expect("Am should parse");
    assert_eq!(chord.root, PitchClass::A);
    assert_eq!(chord.quality, Quality::Minor);
    assert_eq!(chord.voice(), vec![57, 60, 64, 69]);
}

#[test]
fn gate_3_parse_major_seventh_with_five_notes() {
    let chord = parse_chord("Cmaj7").expect("Cmaj7 should parse");
    assert_eq!(chord.quality, Quality::Major7);
    assert_eq!(chord.voice(), vec![60, 64, 67, 71, 72]);
}

#[test]
fn gate_4_parse_progression_four_chords() {
    let progression = parse_progression("C - G - Am - F").expect("progression should parse");
    assert_eq!(progression.len(), 4);
}

#[test]
fn gate_5_parse_roman_vi_in_c_major() {
    let chord = parse_roman("vi", Key::C).expect("roman vi should parse");
    assert_eq!(chord.root, PitchClass::A);
    assert_eq!(chord.quality, Quality::Minor);
}

#[test]
fn gate_6_and_7_cli_renders_nonempty_audible_wav_with_expected_length() {
    let out_path = unique_wav_path("cli");
    let status = Command::new(env!("CARGO_BIN_EXE_kssong"))
        .args([
            "play",
            "C - G",
            "--voice",
            "piano",
            "--bpm",
            "120",
            "--out",
            out_path.to_str().expect("temp path must be utf-8"),
        ])
        .status()
        .expect("kssong binary should launch");

    assert!(status.success(), "kssong CLI exited with {status}");
    assert!(out_path.exists(), "output wav should exist");

    let (peak, frames) = read_wav_peak_and_frames(&out_path);
    let expected_sec = 2.0_f32 * 4.0 * 60.0 / 120.0 + 2.5;
    let expected_frames = (expected_sec * 44_100.0) as f32;
    let frame_error = (frames as f32 - expected_frames).abs() / expected_frames.max(1.0);
    assert!(
        frame_error <= 0.10,
        "wav length should be within 10%: got {frames} frames, expected about {expected_frames}"
    );
    assert!(peak > 0.1, "wav should be audible, got peak {peak}");

    let _ = std::fs::remove_file(out_path);
}
