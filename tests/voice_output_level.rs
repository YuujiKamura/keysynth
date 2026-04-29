//! Issue #13: VoiceImpl output-level contract.
//!
//! Each engine's `make_voice` is wrapped in a calibration coefficient
//! that brings a sustained C4 (MIDI 60, vel 100, 1.0 s render) to within
//! ±1.5 dB of `keysynth::calibration::TARGET_PEAK_DBFS`. Without this,
//! A/B comparison between voices measures loudness, not timbre.
//!
//! Placeholder engines (SfPiano / SfzPiano / Live) render silence from
//! `make_voice` because their audio comes from a shared synthesiser
//! owned by the audio callback; they are excluded from the contract.
//!
//! Diagnostic mode — to remeasure baseline peaks (the values baked into
//! `engine_calibration_gain`), run:
//!
//!   cargo test --test voice_output_level --release \
//!       voice_output_level_diagnostic -- --ignored --nocapture
//!
//! and copy the printed `raw_peak_dbfs` values into `src/calibration.rs`.

use keysynth::calibration::{
    db_from_linear, engine_calibration_gain, make_voice_raw, TARGET_PEAK_DBFS,
};
use keysynth::synth::{make_voice, Engine, ModalLut, VoiceImpl, MODAL_LUT};

const SR: f32 = 44_100.0;
const C4_FREQ: f32 = 261.625_56;
const VEL: u8 = 100;
const RENDER_SECS: f32 = 1.0;

fn render_secs(voice: &mut dyn VoiceImpl, secs: f32) -> Vec<f32> {
    let n = (SR * secs) as usize;
    let mut buf = vec![0.0_f32; n];
    voice.render_add(&mut buf);
    buf
}

fn peak(buf: &[f32]) -> f32 {
    buf.iter().fold(0.0_f32, |a, &x| a.max(x.abs()))
}

/// Engines whose `make_voice` returns a placeholder; excluded from the
/// contract because they don't render audio through the trait.
fn is_placeholder(e: Engine) -> bool {
    matches!(e, Engine::SfPiano | Engine::SfzPiano | Engine::Live)
}

const ENGINES: &[Engine] = &[
    Engine::Square,
    Engine::Ks,
    Engine::KsRich,
    Engine::Sub,
    Engine::Fm,
    Engine::Piano,
    Engine::PianoThick,
    Engine::PianoLite,
    Engine::Piano5AM,
    Engine::PianoModal,
    Engine::Koto,
];

fn ensure_modal_lut() {
    let _ = MODAL_LUT.set(ModalLut::fallback_c4());
}

#[test]
fn voice_output_level_aligned() {
    ensure_modal_lut();

    // ±1.5 dB is roughly 17% — tight enough that loudness differences
    // between voices are inaudible (the just-noticeable difference for
    // sustained tones is ~1 dB), loose enough that envelope shape and
    // partial structure don't get steamrollered into a single peak.
    const TOLERANCE_DB: f32 = 1.5;

    let mut report = String::from("\nvoice output-level contract:\n");
    report.push_str(&format!(
        "  target = {TARGET_PEAK_DBFS:.1} dBFS, tolerance = ±{TOLERANCE_DB:.1} dB\n",
    ));
    report.push_str("  engine            calibrated_peak_dbfs   delta_db\n");

    let mut violations: Vec<(Engine, f32)> = Vec::new();
    for &engine in ENGINES.iter().filter(|e| !is_placeholder(**e)) {
        let mut voice = make_voice(engine, SR, C4_FREQ, VEL);
        let buf = render_secs(voice.as_mut(), RENDER_SECS);
        let p = peak(&buf);
        let p_db = db_from_linear(p);
        let delta = p_db - TARGET_PEAK_DBFS;
        report.push_str(&format!(
            "  {:>14?}        {:>8.2}            {:+6.2}\n",
            engine, p_db, delta,
        ));
        if delta.abs() > TOLERANCE_DB {
            violations.push((engine, delta));
        }
    }
    println!("{report}");
    assert!(
        violations.is_empty(),
        "engines outside ±{TOLERANCE_DB} dB of {TARGET_PEAK_DBFS} dBFS target: {violations:?}",
    );
}

#[test]
fn calibration_gain_is_positive_and_finite() {
    for &engine in ENGINES {
        let g = engine_calibration_gain(engine);
        assert!(
            g.is_finite() && g > 0.0,
            "engine {engine:?} has non-positive/non-finite gain {g}",
        );
    }
}

/// Diagnostic: prints raw (uncalibrated) peaks so the calibration table
/// can be remeasured if engines change. Ignored by default — run with
/// `--ignored --nocapture` to see numbers.
#[test]
#[ignore]
fn voice_output_level_diagnostic() {
    ensure_modal_lut();

    println!();
    println!("raw voice peaks (C4, vel=100, 1.0 s, no calibration):");
    println!("  engine                raw_peak    raw_peak_dbfs  needed_gain   needed_gain_db");
    for &engine in ENGINES.iter().filter(|e| !is_placeholder(**e)) {
        let mut voice = make_voice_raw(engine, SR, C4_FREQ, VEL);
        let buf = render_secs(voice.as_mut(), RENDER_SECS);
        let p = peak(&buf);
        let p_db = db_from_linear(p);
        let target_lin = 10.0_f32.powf(TARGET_PEAK_DBFS / 20.0);
        let need_gain = target_lin / p.max(1e-9);
        let need_gain_db = db_from_linear(need_gain);
        println!(
            "  {:>14?}        {:>7.4}    {:>+8.2}      {:>7.4}      {:>+7.2}",
            engine, p, p_db, need_gain, need_gain_db,
        );
    }
}
