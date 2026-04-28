//! Render a Standard MIDI File (.mid) through any keysynth engine.
//!
//! Usage:
//!     render_midi --in <FILE.mid> --engine ENGINE --out <OUT.wav>
//!                 [--sfz PATH] [--modal-lut PATH] [--tempo-scale FLOAT]
//!
//! Lets us test the synth against real human-played MIDI performances
//! (piano-midi.de, Mutopia, Joplin piano rolls) instead of hand-coded
//! note tuples. Hand-coded data lacks microtiming; recorded MIDI
//! captures it directly.

use std::env;
use std::path::PathBuf;

use hound::{SampleFormat, WavSpec, WavWriter};
use midly::{MetaMessage, MidiMessage, Smf, Timing, TrackEventKind};

use keysynth::sfz::SfzPlayer;
use keysynth::synth::{make_voice, midi_to_freq, Engine, ModalLut, MODAL_LUT};

pub const SR: u32 = 44100;
pub const RELEASE_TAIL_SEC: f32 = 2.5;

#[derive(Clone, Copy, Debug)]
pub struct NoteEvent {
    pub start_sec: f32,
    pub midi_note: u8,
    pub duration_sec: f32,
    pub velocity: u8,
}

struct Args {
    input: PathBuf,
    engine: Engine,
    out_path: PathBuf,
    sfz_path: Option<PathBuf>,
    modal_lut_path: Option<PathBuf>,
    tempo_scale: f32,
}

fn parse_args() -> Result<Args, String> {
    let mut input: Option<PathBuf> = None;
    let mut engine: Option<Engine> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut sfz_path: Option<PathBuf> = None;
    let mut modal_lut_path: Option<PathBuf> = None;
    let mut tempo_scale: f32 = 1.0;

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
            "--tempo-scale" => {
                tempo_scale = iter
                    .next()
                    .ok_or("--tempo-scale needs a value")?
                    .parse()
                    .map_err(|e| format!("--tempo-scale parse: {e}"))?;
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
        tempo_scale,
    })
}

/// Parse a Standard MIDI File and convert to a flat list of NoteEvent
/// (note_on/note_off pairs collapsed into start_sec + duration_sec).
/// Handles format-0 (single track) and format-1 (multi-track merged by
/// absolute tick); tempo changes mid-piece are honoured.
pub fn parse_smf(bytes: &[u8], tempo_scale: f32) -> Result<Vec<NoteEvent>, String> {
    let smf = Smf::parse(bytes).map_err(|e| format!("midly parse: {e}"))?;
    let ppq: u32 = match smf.header.timing {
        Timing::Metrical(t) => t.as_int() as u32,
        Timing::Timecode(_, _) => return Err("SMPTE timing not supported".to_string()),
    };

    // Build a sorted, merged list of (abs_tick, kind) across all tracks.
    #[derive(Clone, Copy)]
    enum Kind {
        On(u8, u8, u8), // channel, note, velocity (velocity > 0)
        Off(u8, u8),    // channel, note
        Tempo(u32),     // microseconds per quarter note
    }
    let mut events: Vec<(u64, Kind)> = Vec::new();
    for track in smf.tracks.iter() {
        let mut tick: u64 = 0;
        for ev in track {
            tick += ev.delta.as_int() as u64;
            match ev.kind {
                TrackEventKind::Midi { channel, message } => match message {
                    MidiMessage::NoteOn { key, vel } => {
                        let v = vel.as_int();
                        if v > 0 {
                            events.push((tick, Kind::On(channel.as_int(), key.as_int(), v)));
                        } else {
                            events.push((tick, Kind::Off(channel.as_int(), key.as_int())));
                        }
                    }
                    MidiMessage::NoteOff { key, .. } => {
                        events.push((tick, Kind::Off(channel.as_int(), key.as_int())));
                    }
                    _ => {}
                },
                TrackEventKind::Meta(MetaMessage::Tempo(us_per_q)) => {
                    events.push((tick, Kind::Tempo(us_per_q.as_int())));
                }
                _ => {}
            }
        }
    }
    // Stable sort by absolute tick. Tempo events processed in the same
    // pass before note conversion at that tick.
    events.sort_by_key(|(t, _)| *t);

    // Walk events, maintain current tempo, convert ticks → seconds.
    // Pair note_on with the next note_off for the same (channel, note).
    let mut us_per_q: u32 = 500_000; // default 120 BPM
    let mut last_tick: u64 = 0;
    let mut now_sec: f64 = 0.0;
    // Active per-(channel, note) start_sec + velocity.
    use std::collections::HashMap;
    let mut active: HashMap<(u8, u8), (f64, u8)> = HashMap::new();
    let mut out: Vec<NoteEvent> = Vec::new();

    for (tick, kind) in &events {
        if *tick > last_tick {
            let delta_ticks = (*tick - last_tick) as f64;
            let sec_per_tick = (us_per_q as f64) / (ppq as f64 * 1_000_000.0);
            now_sec += delta_ticks * sec_per_tick;
            last_tick = *tick;
        }
        match *kind {
            Kind::Tempo(t) => {
                us_per_q = t;
            }
            Kind::On(ch, n, v) => {
                active.insert((ch, n), (now_sec, v));
            }
            Kind::Off(ch, n) => {
                if let Some((start, vel)) = active.remove(&(ch, n)) {
                    let dur = (now_sec - start).max(0.005) as f32;
                    out.push(NoteEvent {
                        start_sec: (start as f32) / tempo_scale,
                        midi_note: n,
                        duration_sec: dur / tempo_scale,
                        velocity: vel,
                    });
                }
            }
        }
    }
    // Any still-active notes at end-of-track: synthesise an off at end.
    for ((_ch, n), (start, vel)) in active {
        let dur = (now_sec - start).max(0.005) as f32;
        out.push(NoteEvent {
            start_sec: (start as f32) / tempo_scale,
            midi_note: n,
            duration_sec: dur / tempo_scale,
            velocity: vel,
        });
    }
    out.sort_by(|a, b| {
        a.start_sec
            .partial_cmp(&b.start_sec)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(out)
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

pub fn render_keysynth(
    engine: Engine,
    events: &[NoteEvent],
    modal_lut_path: Option<&std::path::Path>,
) -> (Vec<f32>, Vec<f32>) {
    if engine == Engine::PianoModal {
        let (lut, source) = ModalLut::auto_load(modal_lut_path);
        eprintln!("render_midi: modal LUT source = {source}");
        let _ = MODAL_LUT.set(lut);
        let res_dir = std::path::PathBuf::from("bench-out/RESIDUAL");
        if res_dir.is_dir() {
            if let Ok(rl) = keysynth::voices::piano_modal::ResidualLut::from_dir(&res_dir) {
                eprintln!(
                    "render_midi: residual LUT source = {} ({} entries)",
                    rl.source,
                    rl.entries.len(),
                );
                let _ = keysynth::voices::piano_modal::RESIDUAL_LUT.set(rl);
            }
        }
    }
    let max_end = events
        .iter()
        .map(|e| e.start_sec + e.duration_sec)
        .fold(0.0_f32, f32::max);
    let total_sec = max_end + RELEASE_TAIL_SEC;
    let total_samples = (total_sec * SR as f32) as usize;
    let mut left = vec![0.0_f32; total_samples];
    let mut right = vec![0.0_f32; total_samples];

    // Per-event tail budget: hold + this much release ringing.
    // Bounding voice_buf at hold+1.5s prevents the per-event allocation
    // from exploding on multi-thousand-event MIDI files (a 3-min piece
    // with 2000 notes would otherwise allocate ~GB of f32 buffers).
    const RELEASE_BUDGET_SEC: f32 = 1.5;
    let release_budget_samples = (RELEASE_BUDGET_SEC * SR as f32) as usize;

    for ev in events {
        let start_sample = (ev.start_sec * SR as f32) as usize;
        let release_sample = ((ev.start_sec + ev.duration_sec) * SR as f32) as usize;
        if start_sample >= total_samples {
            continue;
        }
        let release_at = release_sample.saturating_sub(start_sample);
        let voice_total = (release_at + release_budget_samples).min(total_samples - start_sample);
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
    (left, right)
}

pub fn render_sfz(
    sfz_path: &std::path::Path,
    events: &[NoteEvent],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let mut player = SfzPlayer::load(sfz_path, SR as f32).map_err(|e| format!("sfz load: {e}"))?;
    let max_end = events
        .iter()
        .map(|e| e.start_sec + e.duration_sec)
        .fold(0.0_f32, f32::max);
    let total_sec = max_end + RELEASE_TAIL_SEC;
    let total_samples = (total_sec * SR as f32) as usize;
    let mut left = vec![0.0_f32; total_samples];
    let mut right = vec![0.0_f32; total_samples];

    let mut event_idx = 0;
    let mut releases: Vec<(usize, u8)> = Vec::new();
    for sample_i in 0..total_samples {
        // Trigger any note_on whose start is <= sample_i.
        while event_idx < events.len()
            && (events[event_idx].start_sec * SR as f32) as usize <= sample_i
        {
            let ev = &events[event_idx];
            player.note_on(0, ev.midi_note, ev.velocity);
            let release_at = ((ev.start_sec + ev.duration_sec) * SR as f32) as usize;
            releases.push((release_at, ev.midi_note));
            event_idx += 1;
        }
        // Trigger any pending note_off.
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
    Ok((left, right))
}

pub fn write_wav_stereo(path: &std::path::Path, left: &[f32], right: &[f32]) -> Result<(), String> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).map_err(|e| format!("create_dir_all: {e}"))?;
    }
    // Peak-normalise to -3 dBFS.
    let peak = left
        .iter()
        .chain(right.iter())
        .fold(0.0_f32, |a, b| a.max(b.abs()));
    let target = 10f32.powf(-3.0 / 20.0);
    let gain = if peak > 0.0 { target / peak } else { 1.0 };
    eprintln!(
        "render_midi: raw peak={:.3} → normalised -3 dBFS (gain {:.3})",
        peak, gain,
    );

    let spec = WavSpec {
        channels: 2,
        sample_rate: SR,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut w = WavWriter::create(path, spec).map_err(|e| format!("WavWriter::create: {e}"))?;
    let n = left.len().min(right.len());
    for i in 0..n {
        let l = ((left[i] * gain).clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        let r = ((right[i] * gain).clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        w.write_sample(l)
            .map_err(|e| format!("write_sample(L): {e}"))?;
        w.write_sample(r)
            .map_err(|e| format!("write_sample(R): {e}"))?;
    }
    w.finalize().map_err(|e| format!("finalize: {e}"))?;
    Ok(())
}

fn main() {
    // FTZ/DAZ for offline render so high-Q biquads don't denormal-stall.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
        let csr = _mm_getcsr();
        _mm_setcsr(csr | 0x8040);
    }

    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("render_midi: {e}");
            std::process::exit(2);
        }
    };

    let bytes = match std::fs::read(&args.input) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("render_midi: read {}: {e}", args.input.display());
            std::process::exit(2);
        }
    };
    let events = match parse_smf(&bytes, args.tempo_scale) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("render_midi: SMF parse: {e}");
            std::process::exit(2);
        }
    };
    let last_end = events
        .iter()
        .map(|e| e.start_sec + e.duration_sec)
        .fold(0.0_f32, f32::max);
    eprintln!(
        "render_midi: input='{}' engine={:?} events={} duration={:.1}s tempo_scale={:.2}",
        args.input.display(),
        args.engine,
        events.len(),
        last_end,
        args.tempo_scale,
    );

    let (left, right) = if args.engine == Engine::SfzPiano {
        let sfz = match args.sfz_path.as_ref() {
            Some(p) => p.clone(),
            None => {
                eprintln!("render_midi: --sfz required for sfz-piano engine");
                std::process::exit(2);
            }
        };
        match render_sfz(&sfz, &events) {
            Ok(buf) => buf,
            Err(e) => {
                eprintln!("render_midi: sfz render: {e}");
                std::process::exit(2);
            }
        }
    } else {
        render_keysynth(args.engine, &events, args.modal_lut_path.as_deref())
    };

    if let Err(e) = write_wav_stereo(&args.out_path, &left, &right) {
        eprintln!("render_midi: write: {e}");
        std::process::exit(2);
    }
    eprintln!("render_midi: wrote {} (stereo)", args.out_path.display());
}
