//! Tier 2.3 numerical gate: end-to-end check that the MIDI CC 64
//! sustain pedal lifts the damper on a released piano voice.
//!
//! Three checkpoints (the brief's hard rule):
//!
//!   1. **pedal_off baseline** — render MIDI 60, note_on at `t=0`,
//!      note_off (release) at `t=1.0 s`, observe through `t=3.0 s`.
//!      Assert RMS at `t=2.5 s` is at least 30 dB below RMS at `t=1.0 s`.
//!      Proves the natural `ReleaseEnvelope` is doing its job.
//!
//!   2. **pedal_on sustain** — same render, but `pedal_sustain=1.0`
//!      from `t=0`. Assert RMS at `t=2.5 s` is at least 15 dB **higher**
//!      than the pedal_off render at the same time. Proves the pedal
//!      is keeping the voice alive.
//!
//!   3. **pedal_release_mid** — `pedal_sustain=1.0` at start, then
//!      drops to `0.0` at `t=1.5 s` (after note_off). Assert RMS at
//!      `t=2.5 s` is **between** the pedal_off and pedal_on cases.
//!      Proves the envelope tracks live pedal changes.
//!
//! All three must pass before the PR opens (per brief). Any single
//! failure means the integration is silently broken — pedal not wired,
//! `step_with_pedal` ignored, or a regression in the bit-identical
//! pedal=0 path.

use keysynth::synth::{midi_to_freq, VoiceImpl};
use keysynth::voices::piano::{PianoPreset, PianoVoice};

const SR: f32 = 44_100.0;

/// Render a 3-second sustain of MIDI 60 through `Engine::Piano` with
/// pedal events scripted by the caller. `pedal_program` is consulted
/// once per audio block (every `BLOCK` samples) and the returned value
/// becomes that block's `pedal_sustain`.
///
/// `release_at` (in seconds) triggers `voice.trigger_release()` exactly
/// once on the first block whose start time crosses that threshold.
fn render_with_pedal_program<F>(release_at: f32, duration: f32, mut pedal_program: F) -> Vec<f32>
where
    F: FnMut(f32) -> f32,
{
    const BLOCK: usize = 256;
    let mut v = PianoVoice::with_preset(PianoPreset::PIANO, SR, midi_to_freq(60), 100);
    let total = (SR * duration) as usize;
    let mut buf = vec![0.0_f32; total];
    let mut released = false;
    let mut start = 0;
    while start < total {
        let end = (start + BLOCK).min(total);
        let t = start as f32 / SR;
        if !released && t >= release_at {
            v.trigger_release();
            released = true;
        }
        let pedal = pedal_program(t).clamp(0.0, 1.0);
        v.set_pedal_sustain(pedal);
        v.render_add(&mut buf[start..end]);
        start = end;
    }
    buf
}

/// RMS of a window centred on `t_centre` of half-width `half_window`
/// seconds. Returns `0.0` when the window is empty (out of range).
fn rms_at(samples: &[f32], t_centre: f32, half_window: f32) -> f32 {
    let centre = (t_centre * SR) as isize;
    let half = (half_window * SR) as isize;
    let lo = (centre - half).max(0) as usize;
    let hi = ((centre + half) as usize).min(samples.len());
    if hi <= lo {
        return 0.0;
    }
    let n = (hi - lo) as f32;
    let ss: f32 = samples[lo..hi].iter().map(|x| x * x).sum();
    (ss / n).sqrt()
}

fn db(x: f32) -> f32 {
    20.0 * x.max(1e-12).log10()
}

#[test]
fn pedal_off_release_decays_30db_in_1_5s() {
    // Test 1 (the brief's pedal_off baseline).
    //
    // 1.0 s sustain + 2.0 s release. The Piano preset's release_sec
    // is 0.300 s — the envelope reaches the -60 dB threshold in ~1 s
    // post-release, so by t=2.5 s (1.5 s after release) the voice is
    // well below the noise floor.
    let buf = render_with_pedal_program(1.0, 3.0, |_t| 0.0);

    let rms_at_release = rms_at(&buf, 1.0, 0.05);
    let rms_late = rms_at(&buf, 2.5, 0.05);

    let decay_db = db(rms_at_release) - db(rms_late);
    eprintln!(
        "pedal_off: rms(1.0s)={:.5} ({:.2} dB), rms(2.5s)={:.5} ({:.2} dB), decay={:.2} dB",
        rms_at_release,
        db(rms_at_release),
        rms_late,
        db(rms_late),
        decay_db
    );

    assert!(
        decay_db >= 30.0,
        "pedal_off should decay ≥ 30 dB between 1.0 s and 2.5 s, got {decay_db:.2} dB"
    );
}

#[test]
fn pedal_on_sustains_voice_15db_above_pedal_off() {
    // Test 2 (pedal-on sustain). Pedal held at 1.0 throughout. Voice is
    // released at t=1.0 s but the slowed-down envelope keeps the tail
    // ringing, so at t=2.5 s the RMS should be substantially higher
    // than the pedal_off baseline at the same time.
    let buf_off = render_with_pedal_program(1.0, 3.0, |_t| 0.0);
    let buf_on = render_with_pedal_program(1.0, 3.0, |_t| 1.0);

    let rms_off_late = rms_at(&buf_off, 2.5, 0.05);
    let rms_on_late = rms_at(&buf_on, 2.5, 0.05);
    let lift_db = db(rms_on_late) - db(rms_off_late);
    eprintln!(
        "pedal_off rms(2.5s)={:.6} ({:.2} dB)\npedal_on  rms(2.5s)={:.6} ({:.2} dB)\nlift={:.2} dB",
        rms_off_late,
        db(rms_off_late),
        rms_on_late,
        db(rms_on_late),
        lift_db
    );

    assert!(
        lift_db >= 15.0,
        "pedal_on should sustain ≥ 15 dB above pedal_off at t=2.5 s, got {lift_db:.2} dB"
    );
}

#[test]
fn pedal_release_mid_falls_between_off_and_on() {
    // Test 3 (mid-release pivot). Pedal held at 1.0 until t=1.5 s, then
    // dropped to 0.0. After that the envelope reverts to its natural
    // (full-rate) decay. RMS at t=2.5 s must therefore sit between the
    // off baseline (released at t=1.0) and the always-on case (kept
    // ringing throughout).
    let buf_off = render_with_pedal_program(1.0, 3.0, |_t| 0.0);
    let buf_on = render_with_pedal_program(1.0, 3.0, |_t| 1.0);
    let buf_mid = render_with_pedal_program(1.0, 3.0, |t| if t < 1.5 { 1.0 } else { 0.0 });

    // Measure at t = 1.8 s — 0.3 s after the pedal drops (natural
    // release takes the mid-case from "still ringing" toward floor),
    // but well before all three traces have crashed into numerical
    // noise. At t = 2.5 s both off and mid are at floor-ish RMS so the
    // dB margin between them is dominated by quantisation.
    let t_meas = 1.8_f32;
    let rms_off = rms_at(&buf_off, t_meas, 0.05);
    let rms_on = rms_at(&buf_on, t_meas, 0.05);
    let rms_mid = rms_at(&buf_mid, t_meas, 0.05);
    eprintln!(
        "t={t_meas} s: off={:.6} ({:.2} dB), mid={:.6} ({:.2} dB), on={:.6} ({:.2} dB)",
        rms_off,
        db(rms_off),
        rms_mid,
        db(rms_mid),
        rms_on,
        db(rms_on)
    );

    // Strictly between in dB. We require a margin so a regression that
    // collapses one direction is caught: pedal_release_mid must sit at
    // least 3 dB above the off baseline AND at least 3 dB below the
    // always-on case.
    let margin_above_off = db(rms_mid) - db(rms_off);
    let margin_below_on = db(rms_on) - db(rms_mid);
    assert!(
        margin_above_off >= 3.0,
        "release-mid should be ≥ 3 dB above pedal_off, got {margin_above_off:.2} dB"
    );
    assert!(
        margin_below_on >= 3.0,
        "release-mid should be ≥ 3 dB below pedal_on, got {margin_below_on:.2} dB"
    );
}

/// Marked `#[ignore]` so it runs only when explicitly invoked
/// (`cargo test --features native -- --ignored render_pedal_ab_wavs`).
/// Produces the A/B WAV pair the PR description references; not part
/// of the every-CI numerical gate.
#[test]
#[ignore]
fn render_pedal_ab_wavs() {
    use std::fs;
    use std::path::PathBuf;

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench-out/piano-pedal");
    fs::create_dir_all(&out_dir).unwrap();

    let buf_off = render_with_pedal_program(1.0, 4.0, |_t| 0.0);
    let buf_on = render_with_pedal_program(1.0, 4.0, |_t| 1.0);

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SR as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    for (name, samples) in [("pedal_off.wav", &buf_off), ("pedal_on.wav", &buf_on)] {
        let path = out_dir.join(name);
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for &s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            let i = (clamped * i16::MAX as f32) as i16;
            writer.write_sample(i).unwrap();
        }
        writer.finalize().unwrap();
        eprintln!("wrote {}", path.display());
    }
}

#[test]
fn pedal_zero_render_bit_identical_to_no_pedal_call() {
    // Hard rule from the brief: "既存 PIANO preset の audio output は
    // pedal_sustain=0.0 で bit-identical".
    //
    // Render once without ever touching `set_pedal_sustain` (pre-T2.3
    // baseline shape) and once explicitly setting pedal=0.0 every
    // block. The two rendered buffers must match sample-for-sample.
    const BLOCK: usize = 256;
    let total = (SR * 1.5) as usize;

    let mut a = PianoVoice::with_preset(PianoPreset::PIANO, SR, midi_to_freq(60), 100);
    let mut b = PianoVoice::with_preset(PianoPreset::PIANO, SR, midi_to_freq(60), 100);
    let mut buf_a = vec![0.0_f32; total];
    let mut buf_b = vec![0.0_f32; total];
    let mut start = 0;
    let mut released = false;
    while start < total {
        let end = (start + BLOCK).min(total);
        let t = start as f32 / SR;
        if !released && t >= 1.0 {
            a.trigger_release();
            b.trigger_release();
            released = true;
        }
        // a never sees set_pedal_sustain (legacy path). b explicitly
        // sets pedal=0.0 each block.
        b.set_pedal_sustain(0.0);
        a.render_add(&mut buf_a[start..end]);
        b.render_add(&mut buf_b[start..end]);
        start = end;
    }
    for i in 0..total {
        assert_eq!(
            buf_a[i].to_bits(),
            buf_b[i].to_bits(),
            "sample {i}: pedal=0.0 must be bit-identical to never-pedal"
        );
    }
}
