#![cfg(feature = "native")]

use std::path::{Path, PathBuf};
use std::process::Command;

use keysynth::song::{
    parse_chord, parse_progression, parse_roman, Chord, Key, PitchClass, Quality, Voicing,
};
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

const STANDARD_GUITAR_OPEN_STRINGS: [u8; 6] = [40, 45, 50, 55, 59, 64];

fn unique_wav_path(stem: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("kssong-{stem}-{nanos}.wav"))
}

fn read_wav_peak_and_frames(path: &Path) -> (f32, usize) {
    let (samples, _sr) = read_wav_mono(path);
    let peak = samples
        .iter()
        .fold(0.0_f32, |acc, sample| acc.max(sample.abs()));
    (peak, samples.len())
}

fn read_wav_mono(path: &Path) -> (Vec<f32>, u32) {
    let mut reader = hound::WavReader::open(path).expect("wav should open");
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
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

    if channels == 1 {
        (interleaved, spec.sample_rate)
    } else {
        let mut mono = Vec::with_capacity(interleaved.len() / channels);
        for frame in interleaved.chunks_exact(channels) {
            let sum: f32 = frame.iter().sum();
            mono.push(sum / channels as f32);
        }
        (mono, spec.sample_rate)
    }
}

fn band_peak_relative_db(
    samples: &[f32],
    sr_hz: u32,
    band_low_hz: f32,
    band_high_hz: f32,
) -> (f32, f32) {
    let n = samples.len().min(131_072);
    assert!(n >= 1024, "need at least 1024 samples for FFT gate");

    let fft_len = n.next_power_of_two();
    let mean = samples[..n].iter().sum::<f32>() / n as f32;
    let mut fft_buf = vec![Complex32::new(0.0, 0.0); fft_len];
    for (idx, sample) in samples[..n].iter().enumerate() {
        let window = if n > 1 {
            let phase = 2.0 * std::f32::consts::PI * idx as f32 / (n as f32 - 1.0);
            0.5 - 0.5 * phase.cos()
        } else {
            1.0
        };
        fft_buf[idx].re = (*sample - mean) * window;
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_len);
    fft.process(&mut fft_buf);

    let bin_hz = sr_hz as f32 / fft_len as f32;
    let mut global_max = 0.0_f32;
    let mut band_max = 0.0_f32;
    let mut band_peak_hz = 0.0_f32;
    for (idx, bin) in fft_buf.iter().take(fft_len / 2).enumerate().skip(1) {
        let hz = idx as f32 * bin_hz;
        let mag = bin.norm();
        if hz >= 20.0 && mag > global_max {
            global_max = mag;
        }
        if hz >= band_low_hz && hz <= band_high_hz && mag > band_max {
            band_max = mag;
            band_peak_hz = hz;
        }
    }

    let rel_db = 20.0 * (band_max / global_max.max(1e-12)).log10();
    (band_peak_hz, rel_db)
}

fn chord_tone_intervals(chord: &Chord) -> Vec<u8> {
    let mut out = Vec::new();
    for interval in chord.quality.intervals() {
        let normalized = interval % 12;
        if !out.contains(&normalized) {
            out.push(normalized);
        }
    }
    out
}

fn is_chord_tone(chord: &Chord, midi_note: u8) -> bool {
    let relative_pc = (12 + midi_note % 12 - chord.root.as_u8()) % 12;
    chord_tone_intervals(chord).contains(&relative_pc)
}

fn assert_strictly_ascending(notes: &[u8]) {
    assert!(
        notes.windows(2).all(|pair| pair[0] < pair[1]),
        "notes must be strictly ascending: {notes:?}"
    );
}

#[test]
fn gate_1_close_voicing_keeps_legacy_c_major() {
    let chord = parse_chord("C").expect("C should parse");
    assert_eq!(chord.root, PitchClass::C);
    assert_eq!(chord.quality, Quality::Major);
    assert_eq!(chord.voice(Voicing::Close), vec![60, 64, 67, 72]);
}

#[test]
fn gate_2_piano_voicing_adds_c2_bass_root() {
    let chord = parse_chord("C").expect("C should parse");
    assert_eq!(chord.voice(Voicing::Piano), vec![36, 60, 64, 67, 72]);
    assert_eq!(chord.voice(Voicing::Piano)[0], 36);
}

#[test]
fn gate_3_guitar_voicing_uses_six_string_spread_for_c_major() {
    let chord = parse_chord("C").expect("C should parse");
    assert_eq!(chord.voice(Voicing::Guitar), vec![40, 48, 52, 55, 60, 64]);
    assert_eq!(chord.voice(Voicing::Guitar).len(), 6);
    assert_eq!(chord.voice(Voicing::Guitar)[0], 40);
}

#[test]
fn gate_4_open_voicing_spreads_c_major_over_28_semitones() {
    let chord = parse_chord("C").expect("C should parse");
    let notes = chord.voice(Voicing::Open);
    assert_eq!(notes, vec![36, 43, 52, 55, 60, 64]);
    assert_eq!(notes[0], 36);
    assert_eq!(notes[notes.len() - 1], 64);
    assert_eq!(notes[notes.len() - 1] - notes[0], 28);
}

#[test]
fn gate_5_voicing_rules_hold_for_c_g_am_f() {
    for symbol in ["C", "G", "Am", "F"] {
        let chord = parse_chord(symbol).expect("progression chord should parse");

        let close = chord.voice(Voicing::Close);
        assert_strictly_ascending(&close);

        let piano = chord.voice(Voicing::Piano);
        assert_eq!(
            piano[0],
            36 + chord.root.as_u8(),
            "{symbol}: piano bass root"
        );
        assert_eq!(&piano[1..], &close, "{symbol}: piano upper close voicing");

        let guitar = chord.voice(Voicing::Guitar);
        assert_eq!(guitar.len(), 6, "{symbol}: guitar mode should emit 6 notes");
        for (note, open_string) in guitar.iter().zip(STANDARD_GUITAR_OPEN_STRINGS) {
            assert!(
                *note >= open_string,
                "{symbol}: guitar note {note} must not sit below open string {open_string}"
            );
            assert!(
                is_chord_tone(&chord, *note),
                "{symbol}: guitar note {note} must be a chord tone"
            );
        }
        assert_strictly_ascending(&guitar);

        let open = chord.voice(Voicing::Open);
        assert_eq!(open[0], 36 + chord.root.as_u8(), "{symbol}: open bass root");
        assert!(open.len() >= 6, "{symbol}: open voicing should be wide");
        assert_strictly_ascending(&open);
    }
}

#[test]
fn gate_6_existing_close_voicing_parse_gates_still_hold() {
    let chord = parse_chord("Am").expect("Am should parse");
    assert_eq!(chord.root, PitchClass::A);
    assert_eq!(chord.quality, Quality::Minor);
    assert_eq!(chord.voice(Voicing::Close), vec![57, 60, 64, 69]);

    let maj7 = parse_chord("Cmaj7").expect("Cmaj7 should parse");
    assert_eq!(maj7.quality, Quality::Major7);
    assert_eq!(maj7.voice(Voicing::Close), vec![60, 64, 67, 71, 72]);
}

#[test]
fn gate_7_progression_and_roman_parsing_still_hold() {
    let progression = parse_progression("C - G - Am - F").expect("progression should parse");
    assert_eq!(progression.len(), 4);

    let chord = parse_roman("vi", Key::C).expect("roman vi should parse");
    assert_eq!(chord.root, PitchClass::A);
    assert_eq!(chord.quality, Quality::Minor);
}

#[test]
fn gate_8_default_close_cli_is_bit_identical_to_explicit_close() {
    let default_out = unique_wav_path("default-close");
    let explicit_out = unique_wav_path("explicit-close");

    let default_status = Command::new(env!("CARGO_BIN_EXE_kssong"))
        .args([
            "play",
            "C - G",
            "--voice",
            "piano",
            "--bpm",
            "120",
            "--out",
            default_out.to_str().expect("temp path must be utf-8"),
        ])
        .status()
        .expect("kssong default close should launch");
    assert!(
        default_status.success(),
        "default close exited with {default_status}"
    );

    let explicit_status = Command::new(env!("CARGO_BIN_EXE_kssong"))
        .args([
            "play",
            "C - G",
            "--voicing",
            "close",
            "--voice",
            "piano",
            "--bpm",
            "120",
            "--out",
            explicit_out.to_str().expect("temp path must be utf-8"),
        ])
        .status()
        .expect("kssong explicit close should launch");
    assert!(
        explicit_status.success(),
        "explicit close exited with {explicit_status}"
    );

    let default_bytes = std::fs::read(&default_out).expect("default wav should exist");
    let explicit_bytes = std::fs::read(&explicit_out).expect("explicit wav should exist");
    assert_eq!(
        default_bytes, explicit_bytes,
        "default close must stay bit-identical"
    );

    let _ = std::fs::remove_file(default_out);
    let _ = std::fs::remove_file(explicit_out);
}

#[test]
fn gate_9_cli_default_close_render_keeps_length_and_audibility() {
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

#[test]
fn gate_10_cli_piano_voicing_has_c2_band_energy() {
    let out_path = unique_wav_path("piano-voicing");
    let status = Command::new(env!("CARGO_BIN_EXE_kssong"))
        .args([
            "play",
            "C",
            "--voicing",
            "piano",
            "--voice",
            "piano",
            "--bpm",
            "120",
            "--out",
            out_path.to_str().expect("temp path must be utf-8"),
        ])
        .status()
        .expect("kssong piano voicing should launch");

    assert!(
        status.success(),
        "kssong piano voicing exited with {status}"
    );
    let (samples, sr_hz) = read_wav_mono(&out_path);
    let (peak_hz, rel_db) = band_peak_relative_db(&samples, sr_hz, 60.0, 70.0);
    assert!(
        (60.0..=70.0).contains(&peak_hz),
        "expected a bass-band peak near C2, got {peak_hz:.2} Hz"
    );
    assert!(
        rel_db >= -40.0,
        "expected bass-band energy to sit within -40 dB of spectral max, got {rel_db:.2} dB"
    );

    let _ = std::fs::remove_file(out_path);
}
