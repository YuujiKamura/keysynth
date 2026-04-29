//! Headless regression test for the web audio bus.
//!
//! Reproduces the exact DSP graph that `src/bin/web.rs::render_mono_chunk`
//! runs in the browser (voice mix → sympathetic bank → NaN/denormal flush
//! → body IR reverb → MixMode tanh) but as a normal `cargo test` so we
//! catch web-specific regressions (the master=3.0 + Plain tanh era's
//! hard-clipping audio quality crash) without booting a browser.
//!
//! Strategy: render the C-E-G chord through `Engine::PianoModal` at the
//! browser-typical 48 kHz with the same defaults `WebApp::default` uses
//! today, then assert peak / RMS / crest-factor thresholds that catch
//! the known bad regimes.
//!
//! Note: issue #13 (`crate::calibration`) wraps `make_voice` in a
//! per-engine output-level coefficient that brings every voice to
//! −12 dBFS peak at C4 vel 100. PianoModal's coefficient is ≈0.535,
//! which shifts the post-bus levels recorded here ≈5 dB lower than
//! the pre-calibration measurements that motivated the original
//! thresholds. The bands below were retightened against post-
//! calibration numbers measured 2026-04-29 so the sandwich still
//! catches the master=3.0+Plain regime.
//!
//! The bin/web.rs module itself is wasm32-only so we can't re-use its
//! `render_mono_chunk` symbol directly; the math is duplicated here from
//! the same source. Keep this in sync with `render_mono_chunk` (same
//! coupling constants, same NaN guard, same MixMode branches) — drift
//! between the two means web playback diverges from what the test
//! verifies.

use keysynth::reverb::{self, Reverb};
use keysynth::sympathetic::SympatheticBank;
use keysynth::synth::{make_voice, midi_to_freq, Engine, MixMode};

const SR: f32 = 48_000.0;
const SR_U: u32 = 48_000;

/// Mirror of `WebApp::default()` `LiveParams` fields that affect the
/// audio bus. If the web defaults change, these must follow.
const MASTER: f32 = 1.0;
const REVERB_WET: f32 = 0.3;
const MIX_MODE: MixMode = MixMode::ParallelComp;

/// Apply the live audio bus chain in-place on `mono`. Same as
/// `render_mono_chunk` minus the SF2 path (which is silent for any
/// engine other than `SfPiano` so it's a no-op for this test).
fn apply_bus(
    mono: &mut Vec<f32>,
    sympathetic: &mut SympatheticBank,
    reverb: &mut Reverb,
    limiter_gain: &mut f32,
    mono_compressed: &mut Vec<f32>,
    engine: Engine,
) {
    if engine.is_piano_family() {
        const COUPLING: f32 = 0.0002;
        const MIX: f32 = 0.3;
        for sample in mono.iter_mut() {
            let drive = *sample;
            let sym_out = sympathetic.process(drive, COUPLING);
            *sample += sym_out * MIX;
        }
    } else {
        for _ in 0..mono.len() {
            let _ = sympathetic.process(0.0, 0.0);
        }
    }
    for s in mono.iter_mut() {
        if !s.is_finite() {
            *s = 0.0;
        } else if s.abs() < 1e-30 {
            *s = 0.0;
        }
    }
    if REVERB_WET > 0.0 {
        reverb.process(mono.as_mut_slice(), REVERB_WET);
    }
    match MIX_MODE {
        MixMode::Plain => {
            for sample in mono.iter_mut() {
                *sample = (*sample * MASTER).tanh();
            }
        }
        MixMode::Limiter => {
            const ATTACK: f32 = 0.5;
            const RELEASE: f32 = 0.0001;
            for sample in mono.iter_mut() {
                let abs_s = sample.abs();
                if abs_s > *limiter_gain {
                    *limiter_gain += (abs_s - *limiter_gain) * ATTACK;
                } else {
                    *limiter_gain += (abs_s - *limiter_gain) * RELEASE;
                }
                let gr = if *limiter_gain > 1.0 {
                    1.0 / *limiter_gain
                } else {
                    1.0
                };
                *sample = (*sample * gr * MASTER).tanh();
            }
        }
        MixMode::ParallelComp => {
            const ALPHA: f32 = 0.7;
            const BETA: f32 = 0.6;
            const ATTACK: f32 = 0.5;
            const RELEASE: f32 = 0.0001;
            if mono_compressed.len() != mono.len() {
                mono_compressed.resize(mono.len(), 0.0);
            }
            for (i, sample) in mono.iter().enumerate() {
                let abs_s = sample.abs();
                if abs_s > *limiter_gain {
                    *limiter_gain += (abs_s - *limiter_gain) * ATTACK;
                } else {
                    *limiter_gain += (abs_s - *limiter_gain) * RELEASE;
                }
                let gr = if *limiter_gain > 1.0 {
                    1.0 / *limiter_gain
                } else {
                    1.0
                };
                mono_compressed[i] = sample * gr;
            }
            for (i, sample) in mono.iter_mut().enumerate() {
                let combined = (*sample * ALPHA + mono_compressed[i] * BETA) * MASTER;
                *sample = combined.tanh();
            }
        }
    }
}

fn render_chord_through_bus(notes: &[u8], duration_s: f32, engine: Engine) -> Vec<f32> {
    let frames = (duration_s * SR) as usize;
    let mut voices: Vec<Box<dyn keysynth::synth::VoiceImpl + Send>> = notes
        .iter()
        .map(|&n| make_voice(engine, SR, midi_to_freq(n), 100))
        .collect();
    let mut reverb = Reverb::new(reverb::synthetic_body_ir(SR_U));
    let mut sympathetic = SympatheticBank::new_piano(SR);
    let mut limiter_gain = 1.0f32;
    let mut mono_compressed: Vec<f32> = Vec::new();

    // Render in 1024-frame chunks, same shape the worklet uses.
    let chunk = 1024;
    let mut out = Vec::with_capacity(frames);
    let mut buf = vec![0.0f32; chunk];
    let mut produced = 0;
    while produced < frames {
        let take = chunk.min(frames - produced);
        buf.iter_mut().take(take).for_each(|s| *s = 0.0);
        let slice = &mut buf[..take];
        for v in voices.iter_mut() {
            v.render_add(slice);
        }
        let mut chunk_vec: Vec<f32> = slice.to_vec();
        apply_bus(
            &mut chunk_vec,
            &mut sympathetic,
            &mut reverb,
            &mut limiter_gain,
            &mut mono_compressed,
            engine,
        );
        out.extend_from_slice(&chunk_vec);
        produced += take;
    }
    out
}

fn db(x: f32) -> f32 {
    20.0 * x.abs().max(1e-12).log10()
}

#[test]
fn web_bus_pianomodal_default_chord_within_thresholds() {
    // C-E-G major chord, 1.5 s, PianoModal default voice — same as the
    // browser's first-click experience after `▶ Start` + selecting
    // "Modal (default)" + tapping the on-screen keys.
    let mono = render_chord_through_bus(&[60, 64, 67], 1.5, Engine::PianoModal);

    let peak = mono.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let rms =
        (mono.iter().map(|s| (*s as f64).powi(2)).sum::<f64>() / mono.len() as f64).sqrt() as f32;
    let crest = peak / rms.max(1e-12);

    let peak_db = db(peak);
    let rms_db = db(rms);

    eprintln!(
        "web_bus_pianomodal: peak={peak:.4} ({peak_db:.2} dB)  \
         rms={rms:.4} ({rms_db:.2} dB)  crest={crest:.2}"
    );

    // Threshold strategy: pass on what the *current* master=1.0 +
    // ParallelComp + reverb=0.3 + issue-#13 calibration chain
    // produces (peak ≈ 0.65, rms ≈ -16.96 dB, crest ≈ 4.59 measured
    // 2026-04-29) and trip on the OLD broken master=3.0 + Plain
    // regime (peak ≈ 0.946, rms ≈ -10.85 dB, crest ≈ 3.30 measured
    // by the sibling sanity test below).
    //
    // The sibling `web_bus_old_master_3_plain_actually_does_clip`
    // test asserts the OLD numbers do trip these thresholds, so the
    // pair forms a one-trip regression sandwich: the NEW defaults
    // sit in a band these thresholds reject the OLD defaults from.
    //
    // Margin chosen so small DSP retunings of `ModalParams::default`
    // (output_gain swing, residual_amp tweaks) don't false-alarm but
    // re-broken master values get caught.
    assert!(
        peak < 0.85,
        "peak {peak:.3} ≥ 0.85 — hard clipping. Likely master gain \
         regressed up or `MixMode` flipped to `Plain` on a hot bus. \
         (master=3.0 + Plain hits 0.946.)"
    );
    assert!(
        rms_db < -14.0,
        "rms {rms_db:.2} dB ≥ -14 — bus too loud. Likely a master / \
         output_gain regression. (master=3.0 + Plain hits -10.85 dB.)"
    );
    assert!(
        crest > 3.5,
        "crest factor {crest:.2} ≤ 3.5 — dynamics squashed nearly to \
         a square wave. Likely saturating into the tanh wall. \
         (master=3.0 + Plain hits 3.30.)"
    );
}

#[test]
fn web_bus_old_master_3_plain_actually_does_clip() {
    // Sanity: build the same chord with the OLD broken defaults
    // (master=3.0, Plain tanh) and confirm at least one of the
    // thresholds trips. If this test ever passes, the thresholds are
    // wrong (too loose) — we want them tight enough to catch the very
    // regression they were chosen against.
    let frames = (1.5 * SR) as usize;
    let mut voices: Vec<Box<dyn keysynth::synth::VoiceImpl + Send>> = [60, 64, 67]
        .iter()
        .map(|&n| make_voice(Engine::PianoModal, SR, midi_to_freq(n), 100))
        .collect();
    let mut reverb = Reverb::new(reverb::synthetic_body_ir(SR_U));
    let mut sympathetic = SympatheticBank::new_piano(SR);

    let chunk = 1024;
    let mut out = Vec::with_capacity(frames);
    let mut buf = vec![0.0f32; chunk];
    let mut produced = 0;
    while produced < frames {
        let take = chunk.min(frames - produced);
        buf.iter_mut().take(take).for_each(|s| *s = 0.0);
        let slice = &mut buf[..take];
        for v in voices.iter_mut() {
            v.render_add(slice);
        }
        // Apply just enough of the bus chain to land near the broken
        // regime: sym bank + reverb + master=3.0 Plain tanh.
        for sample in slice.iter_mut() {
            let drive = *sample;
            let sym_out = sympathetic.process(drive, 0.0002);
            *sample += sym_out * 0.3;
        }
        for s in slice.iter_mut() {
            if !s.is_finite() {
                *s = 0.0;
            }
        }
        reverb.process(slice, 0.3);
        for sample in slice.iter_mut() {
            *sample = (*sample * 3.0).tanh();
        }
        out.extend_from_slice(slice);
        produced += take;
    }

    let peak = out.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let rms =
        (out.iter().map(|s| (*s as f64).powi(2)).sum::<f64>() / out.len() as f64).sqrt() as f32;
    let crest = peak / rms.max(1e-12);
    let peak_db = db(peak);
    let rms_db = db(rms);

    eprintln!(
        "OLD broken bus (master=3.0+Plain): peak={peak:.4} ({peak_db:.2} dB)  \
         rms={rms:.4} ({rms_db:.2} dB)  crest={crest:.2}"
    );

    let trips_peak = peak >= 0.85;
    let trips_rms = rms_db >= -14.0;
    let trips_crest = crest <= 3.5;
    assert!(
        trips_peak || trips_rms || trips_crest,
        "Sanity check failed: the OLD master=3.0+Plain regime did NOT \
         trip any of peak≥0.85 / rms≥-14 dB / crest≤3.5. The thresholds \
         in `web_bus_pianomodal_default_chord_within_thresholds` are \
         too loose to catch the regression they exist for."
    );
}
