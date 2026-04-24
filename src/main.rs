//! Real-time MIDI keyboard -> speaker synth.
//!
//! Single-binary Rust replacement for the Python prototype. midir delivers
//! MIDI events on its own callback thread; cpal's audio callback runs on a
//! dedicated audio thread; both share a Mutex<Vec<Voice>> for the active
//! voice pool. Voices are released (env decay) when note_off arrives and
//! removed from the pool once their envelope reaches silence.
//!
//! DSP / voice / engine code lives in the `keysynth` library
//! (`src/synth.rs`) so offline harnesses (`bench`, regression tools) reuse
//! the exact same voices the live synth runs.

use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use midir::{Ignore, MidiInput};

use keysynth::synth::{
    make_voice, midi_to_freq, DashState, Engine, LiveParams, Voice, VoiceImpl,
};
use keysynth::ui;

#[derive(Clone, Debug)]
struct Args {
    engine: Engine,
    port: Option<String>,
    list: bool,
    master: f32,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            engine: Engine::Square,
            port: None,
            list: false,
            master: 0.3,
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let mut out = Args::default();
    let mut iter = std::env::args().skip(1);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--list" => out.list = true,
            "--engine" => {
                let v = iter.next().ok_or("--engine needs a value (square|ks)")?;
                out.engine = match v.as_str() {
                    "square" => Engine::Square,
                    "ks" => Engine::Ks,
                    "ks-rich" => Engine::KsRich,
                    "sub" => Engine::Sub,
                    "fm" => Engine::Fm,
                    "piano" => Engine::Piano,
                    "koto" => Engine::Koto,
                    other => return Err(format!(
                        "unknown engine: {other} (square|ks|ks-rich|sub|fm|piano|koto)"
                    )),
                };
            }
            "--port" => {
                out.port = Some(iter.next().ok_or("--port needs a value")?);
            }
            "--master" => {
                let v = iter.next().ok_or("--master needs a float")?;
                out.master = v.parse().map_err(|e| format!("bad --master: {e}"))?;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    Ok(out)
}

fn print_help() {
    println!(
        "keysynth - real-time MIDI keyboard -> speaker synth\n\n\
         USAGE:\n  \
            keysynth [--engine square|ks|ks-rich|sub|fm|piano|koto] [--port NAME] [--master FLOAT]\n  \
            keysynth --list\n\n\
         OPTIONS:\n  \
            --engine ENGINE   square|ks|ks-rich|sub|fm|piano|koto\n  \
                              (sub = analog subtractive, fm = 2-op bell,\n  \
                               piano = wide-hammer DWS, koto = pluck-pos DWS)\n  \
            --port NAME       MIDI input port (default: first available)\n  \
            --master FLOAT    Master gain pre-tanh (default: 0.3)\n  \
            --list            List MIDI input ports and exit\n  \
            -h, --help        Show this help"
    );
}

// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            print_help();
            std::process::exit(2);
        }
    };

    let mut midi_in = MidiInput::new("keysynth")?;
    midi_in.ignore(Ignore::None);

    let ports = midi_in.ports();
    if args.list {
        if ports.is_empty() {
            println!("(no MIDI input ports)");
        } else {
            for p in &ports {
                println!("{}", midi_in.port_name(p)?);
            }
        }
        return Ok(());
    }

    if ports.is_empty() {
        return Err("no MIDI input ports found - keyboard plugged in?".into());
    }

    let chosen_port = if let Some(want) = &args.port {
        ports
            .iter()
            .find(|p| midi_in.port_name(p).map(|n| n == *want).unwrap_or(false))
            .ok_or_else(|| {
                let available: Vec<String> = ports
                    .iter()
                    .filter_map(|p| midi_in.port_name(p).ok())
                    .collect();
                format!("port {want:?} not found. Available: {available:?}")
            })?
            .clone()
    } else {
        ports[0].clone()
    };
    let port_name = midi_in.port_name(&chosen_port)?;

    // -- Audio output --
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or("no default output device")?;
    let out_name = device.name().unwrap_or_else(|_| "<unknown>".into());

    let supported = device.default_output_config()?;
    let sample_format = supported.sample_format();
    let sr_hz = supported.sample_rate().0;
    let channels = supported.channels();
    let stream_cfg: StreamConfig = supported.into();

    eprintln!(
        "keysynth: midi='{port_name}' audio='{out_name}' sr={sr_hz} ch={channels} \
         engine={:?} master={:.2}",
        args.engine, args.master
    );
    eprintln!("press keys on your MIDI keyboard - Ctrl-C to quit");

    let voices: Arc<Mutex<Vec<Voice>>> = Arc::new(Mutex::new(Vec::with_capacity(64)));
    let live: Arc<Mutex<LiveParams>> = Arc::new(Mutex::new(LiveParams {
        master: args.master,
        engine: args.engine,
    }));
    let dash: Arc<Mutex<DashState>> = Arc::new(Mutex::new(DashState::new(args.engine)));

    let voices_for_midi = voices.clone();
    let live_for_midi = live.clone();
    let dash_for_midi = dash.clone();
    let sr_for_voices = sr_hz as f32;
    let _conn = midi_in.connect(
        &chosen_port,
        "keysynth-in",
        move |_stamp, raw, _| {
            if raw.len() < 3 {
                return;
            }
            let status = raw[0];
            let msg_type = status & 0xF0;
            let channel = status & 0x0F;
            let note = raw[1];
            let velocity = raw[2];

            match msg_type {
                0x90 if velocity > 0 => {
                    {
                        let mut d = dash_for_midi.lock().unwrap();
                        d.active_notes.insert((channel, note));
                        d.push_event(format!("note_on  ch{channel} n{note} v{velocity}"));
                    }
                    let freq = midi_to_freq(note);
                    // Read currently-selected engine fresh each note so GUI
                    // changes apply immediately to subsequent keypresses.
                    let engine = live_for_midi.lock().unwrap().engine;
                    let inner: Box<dyn VoiceImpl> =
                        make_voice(engine, sr_for_voices, freq, velocity);
                    let v = Voice {
                        key: (channel, note),
                        inner,
                    };
                    let mut pool = voices_for_midi.lock().unwrap();
                    if let Some(slot) = pool.iter_mut().find(|x| x.key == (channel, note)) {
                        *slot = v;
                    } else {
                        // Hard cap voice pool to bound CPU/memory growth under
                        // sustained MIDI input. When at cap, prefer evicting an
                        // already-released voice; fall back to the oldest entry.
                        const MAX_VOICES: usize = 32;
                        if pool.len() >= MAX_VOICES {
                            let evict_idx = pool
                                .iter()
                                .position(|x| x.inner.is_done() || x.inner.is_releasing())
                                .unwrap_or(0);
                            pool.remove(evict_idx);
                        }
                        pool.push(v);
                    }
                }
                0x80 | 0x90 => {
                    // Note off, or note_on with velocity 0 (running-status note off)
                    {
                        let mut d = dash_for_midi.lock().unwrap();
                        d.active_notes.remove(&(channel, note));
                        d.push_event(format!("note_off ch{channel} n{note}"));
                    }
                    let mut pool = voices_for_midi.lock().unwrap();
                    if let Some(slot) = pool.iter_mut().find(|x| x.key == (channel, note)) {
                        slot.inner.trigger_release();
                    }
                }
                0xB0 => {
                    // Control Change. raw[1] = CC number, raw[2] = value (0..127).
                    let cc_num = note;       // (raw[1])
                    let cc_val = velocity;   // (raw[2])
                    {
                        let mut d = dash_for_midi.lock().unwrap();
                        d.cc_raw.insert(cc_num, cc_val);
                        *d.cc_count.entry(cc_num).or_insert(0) += 1;
                        d.push_event(format!("CC{cc_num}={cc_val} ch{channel}"));
                    }

                    // MPK mini 3 K1-K8 are ROTARY ENCODERS in relative mode:
                    //   1..63   = +N step(s) clockwise
                    //   65..127 = -N step(s) counter-clockwise (encoded as 128-N)
                    // Standard MIDI Volume (CC 7) is absolute (a "real" pot).
                    let delta_ticks: i32 = match cc_val {
                        0 | 64 => 0,
                        1..=63 => cc_val as i32,
                        65..=127 => -((128 - cc_val as i32)),
                        _ => 0, // MIDI CC values are 0..127 in spec; defensive
                    };

                    match cc_num {
                        // CC 7 = absolute MIDI Volume (rare on MPK).
                        // Top out at 3.0 to match the GUI slider + CC70 range.
                        7 => {
                            let new_master = (cc_val as f32 / 127.0) * 3.0;
                            live_for_midi.lock().unwrap().master = new_master;
                        }
                        // CC 70 = MPK K1 (relative encoder by default).
                        // 1 tick = +/- 0.05 master gain, clamped 0..3.0.
                        70 => {
                            let mut p = live_for_midi.lock().unwrap();
                            p.master = (p.master + delta_ticks as f32 * 0.05).clamp(0.0, 3.0);
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        },
        (),
    )?;

    let voices_for_audio = voices.clone();
    let live_for_audio = live.clone();
    let err_fn = |err| eprintln!("audio stream error: {err}");

    // Build an F32 output stream. Used both for the native F32 path and as a
    // fallback when cpal reports a non-{F32,I16,U16} format we don't handle.
    let build_f32_stream = |dev: &cpal::Device, cfg: &StreamConfig| {
        let voices_arc = voices_for_audio.clone();
        let live_arc = live_for_audio.clone();
        // Preallocated mono scratch reused across audio callbacks. Sized lazily
        // on the first callback once we know the buffer size; cpal callbacks
        // are FnMut so capturing `mut` state is fine.
        let mut mono_scratch: Vec<f32> = Vec::new();
        dev.build_output_stream(
            cfg,
            move |out: &mut [f32], _| {
                let master = live_arc.lock().unwrap().master;
                audio_callback(out, channels, &voices_arc, master, &mut mono_scratch);
            },
            err_fn,
            None,
        )
    };

    let stream = match sample_format {
        SampleFormat::F32 => build_f32_stream(&device, &stream_cfg)?,
        SampleFormat::I16 => {
            let voices_arc = voices_for_audio.clone();
            let live_arc = live_for_audio.clone();
            let mut mono_scratch: Vec<f32> = Vec::new();
            let mut interleaved_scratch: Vec<f32> = Vec::new();
            device.build_output_stream(
                &stream_cfg,
                move |out: &mut [i16], _| {
                    let master = live_arc.lock().unwrap().master;
                    if interleaved_scratch.len() != out.len() {
                        interleaved_scratch.resize(out.len(), 0.0);
                    }
                    audio_callback(
                        &mut interleaved_scratch,
                        channels,
                        &voices_arc,
                        master,
                        &mut mono_scratch,
                    );
                    for (dst, &src) in out.iter_mut().zip(interleaved_scratch.iter()) {
                        let clamped = src.clamp(-1.0, 1.0);
                        *dst = (clamped * i16::MAX as f32) as i16;
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let voices_arc = voices_for_audio.clone();
            let live_arc = live_for_audio.clone();
            let mut mono_scratch: Vec<f32> = Vec::new();
            let mut interleaved_scratch: Vec<f32> = Vec::new();
            device.build_output_stream(
                &stream_cfg,
                move |out: &mut [u16], _| {
                    let master = live_arc.lock().unwrap().master;
                    if interleaved_scratch.len() != out.len() {
                        interleaved_scratch.resize(out.len(), 0.0);
                    }
                    audio_callback(
                        &mut interleaved_scratch,
                        channels,
                        &voices_arc,
                        master,
                        &mut mono_scratch,
                    );
                    for (dst, &src) in out.iter_mut().zip(interleaved_scratch.iter()) {
                        let clamped = src.clamp(-1.0, 1.0);
                        let unsigned = ((clamped + 1.0) * 0.5 * u16::MAX as f32) as u16;
                        *dst = unsigned;
                    }
                },
                err_fn,
                None,
            )?
        }
        other => {
            // cpal's SampleFormat is #[non_exhaustive]; modern backends may
            // hand back I32/I8/U8/F64/etc. Try opening the device with an F32
            // stream regardless - many backends will accept it - and surface
            // the original error only if that also fails.
            eprintln!(
                "keysynth: sample format {other:?} not natively handled, \
                 attempting F32 fallback"
            );
            match build_f32_stream(&device, &stream_cfg) {
                Ok(s) => s,
                Err(e) => {
                    return Err(format!(
                        "unsupported sample format: {other:?} (F32 fallback also failed: {e})"
                    )
                    .into());
                }
            }
        }
    };
    stream.play()?;

    // Launch egui dashboard. The cpal Stream and midir InputConnection are
    // moved into the App struct so they live as long as the GUI window.
    ui::run_app(ui::AppContext {
        stream,
        midi_conn: _conn,
        live,
        dash,
        port_name,
        out_name,
        sr_hz,
    })?;

    eprintln!("keysynth: stopping");
    Ok(())
}

fn audio_callback(
    out: &mut [f32],
    channels: u16,
    voices: &Arc<Mutex<Vec<Voice>>>,
    master: f32,
    mono: &mut Vec<f32>,
) {
    let frames = out.len() / channels as usize;
    // Reuse the caller-owned scratch buffer to avoid per-callback heap
    // allocation on the audio thread (~every 11 ms at 1024 frames / 48 kHz).
    if mono.len() != frames {
        mono.resize(frames, 0.0);
    } else {
        mono.fill(0.0);
    }

    {
        let mut pool = voices.lock().unwrap();
        for v in pool.iter_mut() {
            v.inner.render_add(mono.as_mut_slice());
        }
        pool.retain(|v| !v.inner.is_done());
    }

    for sample in mono.iter_mut() {
        *sample = (*sample * master).tanh();
    }

    for (frame_idx, frame) in out.chunks_mut(channels as usize).enumerate() {
        let s = mono[frame_idx];
        for slot in frame.iter_mut() {
            *slot = s;
        }
    }
}
