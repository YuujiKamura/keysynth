//! Render a chiptune-lab song JSON (ai-chiptune-demo or listener-lab
//! format) through any keysynth engine. Lead+harm+bass tracks become
//! pitched NoteEvents on the chosen voice; drum string overlays via
//! the rhythm-box module.
//!
//! Usage:
//!     render_chiptune --in <song.json> --engine ENGINE --out <OUT.wav>
//!                     [--sfz PATH] [--modal-lut PATH]

use std::env;
use std::path::PathBuf;

use hound::{SampleFormat, WavSpec, WavWriter};

use keysynth::chiptune_import::{parse_song_json, ImportedNote};
use keysynth::drums::DrumEvent;
use keysynth::sfz::SfzPlayer;
use keysynth::synth::{make_voice, midi_to_freq, Engine, ModalLut, MODAL_LUT};

const SR: u32 = 44100;
const RELEASE_TAIL_SEC: f32 = 2.0;

struct Args {
    input: PathBuf,
    engine: Engine,
    out_path: PathBuf,
    sfz_path: Option<PathBuf>,
    modal_lut_path: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut input: Option<PathBuf> = None;
    let mut engine: Option<Engine> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut sfz_path: Option<PathBuf> = None;
    let mut modal_lut_path: Option<PathBuf> = None;

    let mut iter = env::args().skip(1);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--in" => input = Some(PathBuf::from(iter.next().ok_or("--in needs a value")?)),
            "--engine" => {
                let v = iter.next().ok_or("--engine needs a value")?;
                engine = Some(match v.as_str() {
                    "square" => Engine::Square,
                    "ks" => Engine::Ks,
                    "ks-rich" => Engine::KsRich,
                    "sub" => Engine::Sub,
                    "fm" => Engine::Fm,
                    "piano" => Engine::Piano,
                    "piano-thick" => Engine::PianoThick,
                    "piano-lite" => Engine::PianoLite,
                    "piano-5am" => Engine::Piano5AM,
                    "piano-modal" => Engine::PianoModal,
                    "sfz-piano" => Engine::SfzPiano,
                    "koto" => Engine::Koto,
                    other => return Err(format!("unknown engine: {other}")),
                });
            }
            "--out" => out_path = Some(PathBuf::from(iter.next().ok_or("--out needs a value")?)),
            "--sfz" => sfz_path = Some(PathBuf::from(iter.next().ok_or("--sfz needs a value")?)),
            "--modal-lut" => {
                modal_lut_path = Some(PathBuf::from(
                    iter.next().ok_or("--modal-lut needs a value")?,
                ))
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    Ok(Args {
        input: input.ok_or("--in is required")?,
        engine: engine.ok_or("--engine is required")?,
        out_path: out_path.ok_or("--out is required")?,
        sfz_path,
        modal_lut_path,
    })
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

fn render_keysynth(
    engine: Engine,
    notes: &[ImportedNote],
    drums: &[DrumEvent],
    modal_lut_path: Option<&std::path::Path>,
) -> (Vec<f32>, Vec<f32>) {
    if engine == Engine::PianoModal {
        let (lut, source) = ModalLut::auto_load(modal_lut_path);
        eprintln!("render_chiptune: modal LUT source = {source}");
        let _ = MODAL_LUT.set(lut);
        let res_dir = std::path::PathBuf::from("bench-out/RESIDUAL");
        if res_dir.is_dir() {
            if let Ok(rl) = keysynth::voices::piano_modal::ResidualLut::from_dir(&res_dir) {
                eprintln!(
                    "render_chiptune: residual LUT source = {} ({} entries)",
                    rl.source,
                    rl.entries.len(),
                );
                let _ = keysynth::voices::piano_modal::RESIDUAL_LUT.set(rl);
            }
        }
    }

    let max_end = notes
        .iter()
        .map(|e| e.start_sec + e.duration_sec)
        .fold(0.0_f32, f32::max);
    let drum_end = drums
        .iter()
        .map(|e| e.start_sec + e.duration_sec())
        .fold(0.0_f32, f32::max);
    let total_sec = max_end.max(drum_end) + RELEASE_TAIL_SEC;
    let total_samples = (total_sec * SR as f32) as usize;
    let mut left = vec![0.0_f32; total_samples];
    let mut right = vec![0.0_f32; total_samples];

    const RELEASE_BUDGET_SEC: f32 = 1.5;
    let release_budget = (RELEASE_BUDGET_SEC * SR as f32) as usize;

    for ev in notes {
        let start_sample = (ev.start_sec * SR as f32) as usize;
        let release_sample = ((ev.start_sec + ev.duration_sec) * SR as f32) as usize;
        if start_sample >= total_samples {
            continue;
        }
        let release_at = release_sample.saturating_sub(start_sample);
        let voice_total = (release_at + release_budget).min(total_samples - start_sample);
        let mut voice_buf = vec![0.0_f32; voice_total];

        let freq = midi_to_freq(ev.midi_note);
        let mut voice = make_voice(engine, SR as f32, freq, ev.velocity);
        if release_at > 0 {
            voice.render_add(&mut voice_buf[..release_at.min(voice_total)]);
        }
        voice.trigger_release();
        if release_at < voice_total {
            voice.render_add(&mut voice_buf[release_at..voice_total]);
        }
        let (lg, rg) = constant_power_pan(pan_for_note(ev.midi_note));
        for (i, s) in voice_buf.iter().enumerate() {
            left[start_sample + i] += *s * lg;
            right[start_sample + i] += *s * rg;
        }
    }

    // Mix drums center-pan
    let mut scratch: Vec<f32> = Vec::new();
    for ev in drums {
        let start = (ev.start_sec * SR as f32) as usize;
        if start >= total_samples {
            continue;
        }
        let n = ((ev.duration_sec()) * SR as f32) as usize;
        let n = n.min(total_samples - start);
        if n == 0 {
            continue;
        }
        if scratch.len() < n {
            scratch.resize(n, 0.0);
        }
        for s in scratch[..n].iter_mut() {
            *s = 0.0;
        }
        ev.render(SR as f32, &mut scratch[..n], 0);
        for i in 0..n {
            left[start + i] += scratch[i];
            right[start + i] += scratch[i];
        }
    }

    (left, right)
}

fn render_sfz(
    sfz_path: &std::path::Path,
    notes: &[ImportedNote],
    drums: &[DrumEvent],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let mut player = SfzPlayer::load(sfz_path, SR as f32).map_err(|e| format!("sfz: {e}"))?;
    let max_end = notes
        .iter()
        .map(|e| e.start_sec + e.duration_sec)
        .fold(0.0_f32, f32::max);
    let drum_end = drums
        .iter()
        .map(|e| e.start_sec + e.duration_sec())
        .fold(0.0_f32, f32::max);
    let total_sec = max_end.max(drum_end) + RELEASE_TAIL_SEC;
    let total_samples = (total_sec * SR as f32) as usize;
    let mut left = vec![0.0_f32; total_samples];
    let mut right = vec![0.0_f32; total_samples];

    let mut event_idx = 0;
    let mut releases: Vec<(usize, u8)> = Vec::new();
    for sample_i in 0..total_samples {
        while event_idx < notes.len()
            && (notes[event_idx].start_sec * SR as f32) as usize <= sample_i
        {
            let ev = &notes[event_idx];
            player.note_on(0, ev.midi_note, ev.velocity);
            let release_at = ((ev.start_sec + ev.duration_sec) * SR as f32) as usize;
            releases.push((release_at, ev.midi_note));
            event_idx += 1;
        }
        releases.retain(|(at, note)| {
            if *at <= sample_i {
                player.note_off(0, *note);
                false
            } else {
                true
            }
        });
        let mut l = [0.0_f32; 1];
        let mut r = [0.0_f32; 1];
        player.render(&mut l, &mut r);
        left[sample_i] = l[0];
        right[sample_i] = r[0];
    }

    let mut scratch: Vec<f32> = Vec::new();
    for ev in drums {
        let start = (ev.start_sec * SR as f32) as usize;
        if start >= total_samples {
            continue;
        }
        let n = ((ev.duration_sec()) * SR as f32) as usize;
        let n = n.min(total_samples - start);
        if n == 0 {
            continue;
        }
        if scratch.len() < n {
            scratch.resize(n, 0.0);
        }
        for s in scratch[..n].iter_mut() {
            *s = 0.0;
        }
        ev.render(SR as f32, &mut scratch[..n], 0);
        for i in 0..n {
            left[start + i] += scratch[i];
            right[start + i] += scratch[i];
        }
    }

    Ok((left, right))
}

fn write_wav_stereo(path: &std::path::Path, left: &[f32], right: &[f32]) -> Result<(), String> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).map_err(|e| format!("create_dir_all: {e}"))?;
    }
    let peak = left
        .iter()
        .chain(right.iter())
        .fold(0.0_f32, |a, b| a.max(b.abs()));
    let target = 10f32.powf(-3.0 / 20.0);
    let gain = if peak > 0.0 { target / peak } else { 1.0 };
    eprintln!(
        "render_chiptune: raw peak={:.3} → normalised -3 dBFS (gain {:.3})",
        peak, gain,
    );
    let spec = WavSpec {
        channels: 2,
        sample_rate: SR,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut w = WavWriter::create(path, spec).map_err(|e| format!("WavWriter: {e}"))?;
    let n = left.len().min(right.len());
    for i in 0..n {
        let l = ((left[i] * gain).clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        let r = ((right[i] * gain).clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        w.write_sample(l).map_err(|e| format!("write L: {e}"))?;
        w.write_sample(r).map_err(|e| format!("write R: {e}"))?;
    }
    w.finalize().map_err(|e| format!("finalize: {e}"))?;
    Ok(())
}

fn main() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
        let csr = _mm_getcsr();
        _mm_setcsr(csr | 0x8040);
    }

    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("render_chiptune: {e}");
            std::process::exit(2);
        }
    };
    let json_text = match std::fs::read_to_string(&args.input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("render_chiptune: read {}: {e}", args.input.display());
            std::process::exit(2);
        }
    };
    let (notes, drums, song) = match parse_song_json(&json_text) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("render_chiptune: parse: {e}");
            std::process::exit(2);
        }
    };
    eprintln!(
        "render_chiptune: name={:?} bpm={} notes={} drums={}",
        song.name.as_deref().unwrap_or("?"),
        song.bpm,
        notes.len(),
        drums.len(),
    );

    let (left, right) = if args.engine == Engine::SfzPiano {
        let sfz = match args.sfz_path.as_ref() {
            Some(p) => p.clone(),
            None => {
                eprintln!("render_chiptune: --sfz required for sfz-piano");
                std::process::exit(2);
            }
        };
        match render_sfz(&sfz, &notes, &drums) {
            Ok(buf) => buf,
            Err(e) => {
                eprintln!("render_chiptune: sfz: {e}");
                std::process::exit(2);
            }
        }
    } else {
        render_keysynth(args.engine, &notes, &drums, args.modal_lut_path.as_deref())
    };

    if let Err(e) = write_wav_stereo(&args.out_path, &left, &right) {
        eprintln!("render_chiptune: write: {e}");
        std::process::exit(2);
    }
    eprintln!("render_chiptune: wrote {}", args.out_path.display());
}
