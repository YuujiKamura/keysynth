//! Render a chord (3+ simultaneous MIDI notes) through any engine and
//! write to a mono WAV. Used to build the SFZ Salamander chord-level
//! ground-truth set and to render modal/keysynth candidates against
//! that same chord configuration for A/B comparison.
//!
//! Why a separate bin: the existing `bench` binary renders one note
//! at a time. Single-note evaluation hides the polyphony-stacking
//! artifact that the listener flagged as "和音が不自然なハードル";
//! GT and candidate renders both need to be at chord level to make
//! the comparison meaningful.
//!
//! Usage:
//!     render_chord --engine ENGINE --notes "60,64,67" \
//!                  --duration 5 --hold 3 \
//!                  [--sfz PATH] [--velocity V] \
//!                  --out bench-out/path.wav
//!
//! For `sfz-piano` the `--sfz PATH` argument is required (or the
//! Salamander SFZ is auto-discovered at the path used by the live
//! synth). For keysynth engines (`square`/`ks`/`piano-modal`/etc.) the
//! voices are constructed via `make_voice` for each note and summed
//! into a shared mono buffer, then peak-normalised to -3 dBFS.

use std::env;
use std::path::PathBuf;

use hound::{SampleFormat, WavSpec, WavWriter};

use keysynth::sfz::SfzPlayer;
use keysynth::synth::{make_voice, midi_to_freq, modal_params, set_modal_params, Engine, ModalLut, MODAL_LUT};

const SR: u32 = 44100;

struct Args {
    engine: Engine,
    notes: Vec<u8>,
    velocity: u8,
    duration_sec: f32,
    hold_sec: f32,
    sfz_path: Option<PathBuf>,
    out_path: PathBuf,
    modal_lut_path: Option<PathBuf>,
    /// Stagger note onsets by N ms so simultaneous-key strikes don't
    /// land on bit-identical sample 0 across voices. Real piano players
    /// can't perfectly synchronise fingers; recorded chords always have
    /// 1-5 ms inter-onset jitter. 0 = perfectly aligned (the current
    /// modal voice's failure mode in chord context).
    onset_jitter_ms: f32,
}

fn parse_args() -> Result<Args, String> {
    let mut engine: Option<Engine> = None;
    let mut notes: Vec<u8> = Vec::new();
    let mut velocity: u8 = 100;
    let mut duration_sec: f32 = 5.0;
    let mut hold_sec: f32 = 3.0;
    let mut sfz_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut modal_lut_path: Option<PathBuf> = None;
    let mut onset_jitter_ms: f32 = 0.0;

    let mut iter = env::args().skip(1);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--engine" => {
                let v = iter.next().ok_or("--engine needs a value")?;
                engine = Some(match v.as_str() {
                    "square" => Engine::Square,
                    "ks" => Engine::Ks,
                    "ks-rich" => Engine::KsRich,
                    "sub" => Engine::Sub,
                    "fm" => Engine::Fm,
                    "piano" => Engine::Piano,
                    "koto" => Engine::Koto,
                    "sfz-piano" => Engine::SfzPiano,
                    "piano-thick" => Engine::PianoThick,
                    "piano-lite" => Engine::PianoLite,
                    "piano-5am" => Engine::Piano5AM,
                    "piano-modal" => Engine::PianoModal,
                    other => return Err(format!("unknown engine: {other}")),
                });
            }
            "--notes" => {
                let v = iter.next().ok_or("--notes needs a value (e.g. 60,64,67)")?;
                for tok in v.split(|c: char| c == ',' || c.is_whitespace()) {
                    let t = tok.trim();
                    if t.is_empty() {
                        continue;
                    }
                    notes.push(t.parse().map_err(|e| format!("bad note '{t}': {e}"))?);
                }
            }
            "--velocity" => {
                velocity = iter
                    .next()
                    .ok_or("--velocity needs an integer")?
                    .parse()
                    .map_err(|e| format!("bad --velocity: {e}"))?;
            }
            "--duration" => {
                duration_sec = iter
                    .next()
                    .ok_or("--duration needs a float")?
                    .parse()
                    .map_err(|e| format!("bad --duration: {e}"))?;
            }
            "--hold" => {
                hold_sec = iter
                    .next()
                    .ok_or("--hold needs a float")?
                    .parse()
                    .map_err(|e| format!("bad --hold: {e}"))?;
            }
            "--sfz" => {
                sfz_path = Some(PathBuf::from(iter.next().ok_or("--sfz needs a path")?));
            }
            "--out" => {
                out_path = Some(PathBuf::from(iter.next().ok_or("--out needs a path")?));
            }
            "--modal-lut" => {
                modal_lut_path = Some(PathBuf::from(
                    iter.next().ok_or("--modal-lut needs a path")?,
                ));
            }
            "--onset-jitter" => {
                onset_jitter_ms = iter
                    .next()
                    .ok_or("--onset-jitter needs a float (ms)")?
                    .parse()
                    .map_err(|e| format!("bad --onset-jitter: {e}"))?;
            }
            "--help" | "-h" => {
                eprintln!(
                    "render_chord — render a chord through any engine and write a mono WAV.\n\n\
                     options:\n  \
                     --engine ENGINE   square|ks|ks-rich|sub|fm|piano|piano-thick|piano-lite|piano-5am|\
                                       piano-modal|sfz-piano|koto\n  \
                     --notes \"N,N,N\"   comma- or space-separated MIDI notes\n  \
                     --velocity V      MIDI velocity 1..127 (default 100)\n  \
                     --duration SEC    total render length (default 5)\n  \
                     --hold SEC        time before note_off (default 3)\n  \
                     --sfz PATH        SFZ manifest (required for sfz-piano)\n  \
                     --modal-lut PATH  modal LUT JSON (auto-discovered if omitted)\n  \
                     --onset-jitter MS spread voice onsets across N ms (default 0)\n  \
                     --out PATH        output WAV path (required)"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }

    let engine = engine.ok_or("--engine is required")?;
    if notes.is_empty() {
        return Err("--notes is required and must contain at least one MIDI note".into());
    }
    let out_path = out_path.ok_or("--out is required")?;

    Ok(Args {
        engine,
        notes,
        velocity,
        duration_sec,
        hold_sec,
        sfz_path,
        out_path,
        modal_lut_path,
        onset_jitter_ms,
    })
}

fn write_wav_stereo(
    path: &std::path::Path,
    left: &[f32],
    right: &[f32],
) -> Result<(), String> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).map_err(|e| format!("create_dir_all {}: {e}", p.display()))?;
    }
    let spec = WavSpec {
        channels: 2,
        sample_rate: SR,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut w = WavWriter::create(path, spec).map_err(|e| format!("WavWriter::create: {e}"))?;
    let n = left.len().min(right.len());
    for i in 0..n {
        let l = (left[i].clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        let r = (right[i].clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        w.write_sample(l)
            .map_err(|e| format!("write_sample(L): {e}"))?;
        w.write_sample(r)
            .map_err(|e| format!("write_sample(R): {e}"))?;
    }
    w.finalize().map_err(|e| format!("finalize: {e}"))?;
    Ok(())
}

fn constant_power_pan(pan: f32) -> (f32, f32) {
    let p = pan.clamp(-1.0, 1.0);
    let theta = (p + 1.0) * std::f32::consts::FRAC_PI_4;
    (theta.cos(), theta.sin())
}

fn pan_for_note(midi_note: u8) -> f32 {
    const CENTRE: f32 = 60.0;
    const SPAN: f32 = 36.0;
    ((midi_note as f32 - CENTRE) / SPAN).clamp(-1.0, 1.0)
}

fn render_sfz(args: &Args) -> Result<(Vec<f32>, Vec<f32>), String> {
    let sfz_path = args
        .sfz_path
        .as_ref()
        .ok_or("--sfz PATH required for engine sfz-piano")?;
    let mut player = SfzPlayer::load(sfz_path, SR as f32)
        .map_err(|e| format!("SfzPlayer::load: {e}"))?;
    let total_samples = (args.duration_sec * SR as f32) as usize;
    let release_at = ((args.hold_sec * SR as f32) as usize).min(total_samples);
    let jitter_samples = (args.onset_jitter_ms * SR as f32 / 1000.0) as usize;

    let mut left = vec![0.0_f32; total_samples];
    let mut right = vec![0.0_f32; total_samples];

    // Stagger note_on across the jitter window so the SFZ player's
    // internal voices don't all fire at sample 0. We render in chunks
    // between successive note_ons.
    let n = args.notes.len();
    let mut cursor = 0usize;
    for (idx, &note) in args.notes.iter().enumerate() {
        let start = if jitter_samples == 0 || n <= 1 {
            0
        } else {
            (idx * jitter_samples) / n.max(1)
        };
        if start > cursor {
            let span = start.min(total_samples) - cursor;
            if span > 0 {
                player.render(
                    &mut left[cursor..cursor + span],
                    &mut right[cursor..cursor + span],
                );
                cursor += span;
            }
        }
        player.note_on(0, note, args.velocity);
    }
    if cursor < release_at {
        player.render(&mut left[cursor..release_at], &mut right[cursor..release_at]);
    }
    for &note in &args.notes {
        player.note_off(0, note);
    }
    if release_at < total_samples {
        player.render(
            &mut left[release_at..total_samples],
            &mut right[release_at..total_samples],
        );
    }
    Ok((left, right))
}

fn render_keysynth(args: &Args) -> Result<(Vec<f32>, Vec<f32>), String> {
    if args.engine == Engine::PianoModal {
        let (lut, source) = ModalLut::auto_load(args.modal_lut_path.as_deref());
        eprintln!("render_chord: modal LUT source = {source}");
        let _ = MODAL_LUT.set(lut);
        // Optional residual layer (Smith commuted synthesis). Loaded
        // when bench-out/RESIDUAL/ exists. Silently skipped otherwise
        // so legacy renders (and the `build_residual_ir.py` pipeline
        // itself, which renders modal-only) keep working.
        let res_dir = std::path::PathBuf::from("bench-out/RESIDUAL");
        if res_dir.is_dir() {
            if let Ok(rl) = keysynth::voices::piano_modal::ResidualLut::from_dir(&res_dir) {
                eprintln!(
                    "render_chord: residual LUT source = {} ({} entries)",
                    rl.source,
                    rl.entries.len(),
                );
                let _ = keysynth::voices::piano_modal::RESIDUAL_LUT.set(rl);
            }
        }
    }
    let total_samples = (args.duration_sec * SR as f32) as usize;
    let release_at = ((args.hold_sec * SR as f32) as usize).min(total_samples);
    let jitter_samples = (args.onset_jitter_ms * SR as f32 / 1000.0) as usize;
    let n = args.notes.len();

    let mut left = vec![0.0_f32; total_samples];
    let mut right = vec![0.0_f32; total_samples];
    for (idx, &note) in args.notes.iter().enumerate() {
        let freq = midi_to_freq(note);
        let mut voice = make_voice(args.engine, SR as f32, freq, args.velocity);
        let offset = if jitter_samples == 0 || n <= 1 {
            0
        } else {
            (idx * jitter_samples) / n.max(1)
        };
        let voice_release_at = if release_at > offset {
            release_at - offset
        } else {
            0
        };
        let voice_total = if total_samples > offset {
            total_samples - offset
        } else {
            0
        };
        let mut voice_buf = vec![0.0_f32; voice_total];
        if voice_release_at > 0 {
            voice.render_add(&mut voice_buf[..voice_release_at]);
        }
        voice.trigger_release();
        if voice_release_at < voice_total {
            voice.render_add(&mut voice_buf[voice_release_at..voice_total]);
        }
        let (lg, rg) = constant_power_pan(pan_for_note(note));
        for (i, s) in voice_buf.iter().enumerate() {
            let dst = offset + i;
            if dst < total_samples {
                left[dst] += *s * lg;
                right[dst] += *s * rg;
            }
        }
    }
    Ok((left, right))
}

fn peak_normalise_stereo(left: &mut [f32], right: &mut [f32], target_dbfs: f32) {
    let peak_l = left.iter().copied().fold(0.0_f32, |a, b| a.max(b.abs()));
    let peak_r = right.iter().copied().fold(0.0_f32, |a, b| a.max(b.abs()));
    let peak = peak_l.max(peak_r);
    if peak > 1e-9 {
        let target = 10f32.powf(target_dbfs / 20.0);
        let scale = target / peak;
        for s in left.iter_mut() {
            *s *= scale;
        }
        for s in right.iter_mut() {
            *s *= scale;
        }
    }
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("render_chord: {e}");
            std::process::exit(2);
        }
    };

    // Optional ModalParams override via env vars (one-shot tuning experiments).
    {
        let mut p = modal_params();
        if let Ok(v) = std::env::var("KS_DETUNE")    { if let Ok(x) = v.parse() { p.detune_cents = x; } }
        if let Ok(v) = std::env::var("KS_POL_H")     { if let Ok(x) = v.parse() { p.pol_h_weight = x; } }
        if let Ok(v) = std::env::var("KS_T60_CAP")   { if let Ok(x) = v.parse() { p.t60_cap_sec = x; } }
        if let Ok(v) = std::env::var("KS_STAGE_B")   { if let Ok(x) = v.parse() { p.stage_b_gain = x; } }
        if let Ok(v) = std::env::var("KS_OUT_GAIN")  { if let Ok(x) = v.parse() { p.output_gain = x; } }
        if let Ok(v) = std::env::var("KS_RESIDUAL")  { if let Ok(x) = v.parse() { p.residual_amp = x; } }
        eprintln!("render_chord: ModalParams = {:?}", p);
        set_modal_params(p);
    }

    let (mut left, mut right) = match args.engine {
        Engine::SfzPiano => render_sfz(&args),
        _ => render_keysynth(&args),
    }
    .unwrap_or_else(|e| {
        eprintln!("render_chord: {e}");
        std::process::exit(2);
    });

    let raw_peak_l = left.iter().copied().fold(0.0_f32, |a, b| a.max(b.abs()));
    let raw_peak_r = right.iter().copied().fold(0.0_f32, |a, b| a.max(b.abs()));
    peak_normalise_stereo(&mut left, &mut right, -3.0);

    eprintln!(
        "render_chord: engine={:?} notes={:?} duration={:.2}s hold={:.2}s",
        args.engine, args.notes, args.duration_sec, args.hold_sec
    );
    eprintln!(
        "render_chord: raw peak L={:.3} R={:.3} → normalised -3 dBFS",
        raw_peak_l, raw_peak_r
    );

    if let Err(e) = write_wav_stereo(&args.out_path, &left, &right) {
        eprintln!("render_chord: {e}");
        std::process::exit(2);
    }
    eprintln!("render_chord: wrote {} (stereo)", args.out_path.display());
}
