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
//!
//! Two extras live alongside the original engines:
//!   - `sf-piano` : routes notes to a shared `rustysynth::Synthesizer`
//!     loaded from a `.sf2` file (`--sf2 PATH`). The voice pool only
//!     holds placeholder slots; audio comes out of the shared synth.
//!   - body IR convolution reverb : applied to the mono mix
//!     after voice rendering, before the tanh soft-clip. Synthetic IR by
//!     default; pass `--ir PATH` for a real WAV impulse response.

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use midir::{Ignore, MidiInput};
use rustysynth::{SoundFont, Synthesizer, SynthesizerSettings};

use keysynth::reverb::{self, Reverb};
use keysynth::sfz::SfzPlayer;
use keysynth::sympathetic::SympatheticBank;
use keysynth::synth::{
    make_voice, midi_to_freq, DashState, Engine, LiveParams, Voice, VoiceImpl,
};
use keysynth::ui;

/// Shared rustysynth instance for `sf-piano`. `None` until `--sf2` is loaded
/// (or always `None` for the other engines). Behind a Mutex because the
/// synth is mutable and accessed from both the MIDI callback (note on/off)
/// and the audio callback (render).
pub type SharedSynth = Arc<Mutex<Option<Synthesizer>>>;

/// Shared SFZ sampler instance for `sfz-piano`. `None` until `--sfz` is
/// loaded. Same Mutex pattern as `SharedSynth`: MIDI thread calls
/// note_on/off, audio thread calls render.
pub type SharedSfz = Arc<Mutex<Option<SfzPlayer>>>;

#[derive(Clone, Debug)]
struct Args {
    engine: Engine,
    port: Option<String>,
    list: bool,
    master: f32,
    sf2: Option<PathBuf>,
    sf2_program: u8,
    sf2_bank: u8,
    sfz: Option<PathBuf>,
    ir_path: Option<PathBuf>,
    reverb_wet: f32,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            engine: Engine::Square,
            port: None,
            list: false,
            master: 0.3,
            sf2: None,
            sf2_program: 0,
            sf2_bank: 0,
            sfz: None,
            ir_path: None,
            // Body-IR wet default lowered 0.3 -> 0.15 once the piano engine
            // gained a bidirectional modal soundboard (see synth::PianoVoice
            // / soundboard::Soundboard). The soundboard now provides the
            // body / "halo" character that the IR reverb was masking with;
            // keep the IR as room-character seasoning, not as the body.
            // Easy to dial back up via GUI / CC for sustain-pedal vibes.
            reverb_wet: 0.15,
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
                let v = iter.next().ok_or("--engine needs a value")?;
                out.engine = match v.as_str() {
                    "square" => Engine::Square,
                    "ks" => Engine::Ks,
                    "ks-rich" => Engine::KsRich,
                    "sub" => Engine::Sub,
                    "fm" => Engine::Fm,
                    "piano" => Engine::Piano,
                    "koto" => Engine::Koto,
                    "sf-piano" => Engine::SfPiano,
                    "sfz-piano" => Engine::SfzPiano,
                    other => return Err(format!(
                        "unknown engine: {other} (square|ks|ks-rich|sub|fm|piano|koto|sf-piano|sfz-piano)"
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
            "--sf2" => {
                out.sf2 = Some(PathBuf::from(
                    iter.next().ok_or("--sf2 needs a path")?,
                ));
            }
            "--sf2-program" => {
                out.sf2_program = iter.next().ok_or("--sf2-program needs an integer")?
                    .parse().map_err(|e| format!("bad --sf2-program: {e}"))?;
            }
            "--sf2-bank" => {
                out.sf2_bank = iter.next().ok_or("--sf2-bank needs an integer")?
                    .parse().map_err(|e| format!("bad --sf2-bank: {e}"))?;
            }
            "--sfz" => {
                out.sfz = Some(PathBuf::from(
                    iter.next().ok_or("--sfz needs a path")?,
                ));
            }
            "--ir" => {
                out.ir_path = Some(PathBuf::from(
                    iter.next().ok_or("--ir needs a path")?,
                ));
            }
            "--reverb" => {
                let v = iter.next().ok_or("--reverb needs a float 0..1")?;
                out.reverb_wet = v.parse::<f32>()
                    .map_err(|e| format!("bad --reverb: {e}"))?
                    .clamp(0.0, 1.0);
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
            keysynth [--engine ENGINE] [--port NAME] [--master FLOAT]\n  \
                     [--sf2 PATH [--sf2-program N] [--sf2-bank N]]\n  \
                     [--sfz PATH]\n  \
                     [--ir PATH] [--reverb 0..1]\n  \
            keysynth --list\n\n\
         OPTIONS:\n  \
            --engine ENGINE   square|ks|ks-rich|sub|fm|piano|koto|sf-piano|sfz-piano\n  \
                              (sub = analog subtractive, fm = 2-op bell,\n  \
                               piano = wide-hammer DWS, koto = pluck-pos DWS,\n  \
                               sf-piano = SoundFont real-piano via rustysynth,\n  \
                               sfz-piano = SFZ sample library, e.g. Salamander Grand V3)\n  \
            --port NAME       MIDI input port (default: first available)\n  \
            --master FLOAT    Master gain pre-tanh (default: 0.3)\n  \
            --sf2 PATH        SoundFont .sf2 (required for --engine sf-piano)\n  \
            --sf2-program N   GM program 0..127 (default: 0 = Acoustic Grand)\n  \
            --sf2-bank N      Bank 0..128 (default: 0)\n  \
            --sfz PATH        SFZ manifest .sfz (required for --engine sfz-piano)\n  \
            --ir PATH         WAV impulse response for body reverb\n  \
                              (default: built-in synthetic piano body IR)\n  \
            --reverb FLOAT    Reverb wet 0..1 (default: 0 = dry)\n  \
            --list            List MIDI input ports and exit\n  \
            -h, --help        Show this help"
    );
}

// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            print_help();
            std::process::exit(2);
        }
    };

    // Auto-discover SF2 / SFZ in CWD if not explicitly given via CLI.
    // Lets the user freely live-switch between sf-piano / sfz-piano in
    // the GUI without having to remember to pass both flags at startup.
    if args.sf2.is_none() {
        if let Ok(entries) = std::fs::read_dir(".") {
            for e in entries.flatten() {
                let p = e.path();
                if p.extension().and_then(|s| s.to_str()) == Some("sf2") {
                    eprintln!("keysynth: auto-discovered SF2 {}", p.display());
                    args.sf2 = Some(p);
                    break;
                }
            }
        }
    }
    if args.sfz.is_none() {
        for root in [PathBuf::from("."), PathBuf::from("Salamander")] {
            if let Ok(entries) = std::fs::read_dir(&root) {
                let mut found: Option<PathBuf> = None;
                for e in entries.flatten() {
                    let p = e.path();
                    if p.is_dir() {
                        // One level deep — Salamander's SFZ lives in a
                        // subdirectory next to its sample folder.
                        if let Ok(sub) = std::fs::read_dir(&p) {
                            for se in sub.flatten() {
                                let sp = se.path();
                                if sp.extension().and_then(|s| s.to_str())
                                    == Some("sfz")
                                {
                                    found = Some(sp);
                                    break;
                                }
                            }
                        }
                    } else if p.extension().and_then(|s| s.to_str()) == Some("sfz") {
                        found = Some(p);
                        break;
                    }
                    if found.is_some() {
                        break;
                    }
                }
                if let Some(p) = found {
                    eprintln!("keysynth: auto-discovered SFZ {}", p.display());
                    args.sfz = Some(p);
                    break;
                }
            }
        }
    }

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

    // -- SoundFont (sf-piano engine) --
    //
    // Loaded eagerly when --sf2 is given so any errors (missing file, bad
    // format) surface at startup rather than the first key press. Hard
    // error if the user picked sf-piano but didn't pass an sf2 path.
    let shared_synth: SharedSynth = Arc::new(Mutex::new(None));
    if let Some(sf2_path) = &args.sf2 {
        let mut file = BufReader::new(
            File::open(sf2_path)
                .map_err(|e| format!("opening SoundFont {:?}: {e}", sf2_path))?,
        );
        let sf = Arc::new(SoundFont::new(&mut file)?);
        let mut settings = SynthesizerSettings::new(sr_hz as i32);
        // rustysynth default maximum_polyphony = 64. Piano patches can use
        // 2-4 internal voices per note (stereo + velocity layers), so chords
        // hit the cap fast and notes get stolen mid-decay. Raise to 256 -
        // tests show it stays under 5% CPU even when fully saturated.
        settings.maximum_polyphony = 256;
        let mut synth = Synthesizer::new(&sf, &settings)?;
        if args.sf2_bank > 0 {
            synth.process_midi_message(0, 0xB0, 0, args.sf2_bank as i32);
        }
        synth.process_midi_message(0, 0xC0, args.sf2_program as i32, 0);
        *shared_synth.lock().unwrap() = Some(synth);
        eprintln!(
            "keysynth: loaded SoundFont '{}' (program={} bank={})",
            sf2_path.display(), args.sf2_program, args.sf2_bank
        );
    } else if args.engine == Engine::SfPiano {
        return Err(
            "engine 'sf-piano' requires --sf2 PATH (no SoundFont loaded)".into(),
        );
    }

    // -- SFZ sampler (sfz-piano engine) --
    //
    // Loaded eagerly when --sfz is given. Decodes every WAV referenced by the
    // manifest into memory (~400-1300 MB for Salamander Grand V3 depending on
    // bit depth / sample rate); the read is one-shot at startup so the audio
    // thread never blocks on disk I/O.
    let shared_sfz: SharedSfz = Arc::new(Mutex::new(None));
    if let Some(sfz_path) = &args.sfz {
        let started = std::time::Instant::now();
        let player = SfzPlayer::load(sfz_path, sr_hz as f32)
            .map_err(|e| format!("loading SFZ {:?}: {e}", sfz_path))?;
        eprintln!(
            "keysynth: loaded SFZ '{}' ({} regions, {} samples, {:.1}s)",
            sfz_path.display(),
            player.regions_len(),
            player.samples_len(),
            started.elapsed().as_secs_f32()
        );
        *shared_sfz.lock().unwrap() = Some(player);
    } else if args.engine == Engine::SfzPiano {
        return Err(
            "engine 'sfz-piano' requires --sfz PATH (no SFZ manifest loaded)".into(),
        );
    }

    // -- Body IR reverb --
    //
    // IR is built / loaded once at startup; the audio callback owns the
    // Reverb instance directly so there's no per-callback locking and no
    // heap allocation on the audio thread.
    let ir_samples: Vec<f32> = match &args.ir_path {
        Some(p) => {
            let ir = reverb::load_ir_wav(p)?;
            eprintln!(
                "keysynth: loaded IR '{}' ({} samples, ~{:.0} ms @ {} Hz)",
                p.display(), ir.len(), 1000.0 * ir.len() as f32 / sr_hz as f32, sr_hz
            );
            ir
        }
        None => reverb::synthetic_body_ir(sr_hz),
    };

    eprintln!(
        "keysynth: midi='{port_name}' audio='{out_name}' sr={sr_hz} ch={channels} \
         engine={:?} master={:.2} reverb={:.2}",
        args.engine, args.master, args.reverb_wet
    );
    eprintln!("press keys on your MIDI keyboard - Ctrl-C to quit");

    let voices: Arc<Mutex<Vec<Voice>>> = Arc::new(Mutex::new(Vec::with_capacity(64)));
    let live: Arc<Mutex<LiveParams>> = Arc::new(Mutex::new(LiveParams {
        master: args.master,
        engine: args.engine,
        reverb_wet: args.reverb_wet,
        sf_program: args.sf2_program,
        sf_bank: args.sf2_bank,
    }));
    let dash: Arc<Mutex<DashState>> = Arc::new(Mutex::new(DashState::new(args.engine)));

    let voices_for_midi = voices.clone();
    let live_for_midi = live.clone();
    let dash_for_midi = dash.clone();
    let synth_for_midi = shared_synth.clone();
    let sfz_for_midi = shared_sfz.clone();
    let sr_for_voices = sr_hz as f32;
    // Track the last (program, bank) we pushed into the shared synth so
    // we don't spam a Program Change per note. `u16::MAX` sentinel forces
    // the first note_on after startup to send the initial pair, even if
    // the GUI never moved the picker (covers the case where args.sf2_program
    // != 0 was already applied at SF2 load time -- still cheap).
    let mut last_applied_program: u16 = u16::MAX;
    let mut last_applied_bank: u16 = u16::MAX;
    let _conn = midi_in.connect(
        &chosen_port,
        "keysynth-in",
        move |_stamp, raw, _| {
            if raw.is_empty() {
                return;
            }
            let status = raw[0];
            let msg_type = status & 0xF0;
            let channel = status & 0x0F;

            // Program Change is a 2-byte message: status + program.
            // Handle BEFORE the >=3 guard below, otherwise it gets dropped.
            if msg_type == 0xC0 {
                if raw.len() < 2 {
                    return;
                }
                let program = raw[1].min(127);
                {
                    let mut lp = live_for_midi.lock().unwrap();
                    lp.sf_program = program;
                }
                {
                    let mut d = dash_for_midi.lock().unwrap();
                    d.push_event(format!("PC ch{channel} prog={program}"));
                }
                return;
            }

            if raw.len() < 3 {
                return;
            }
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
                    // Read currently-selected engine + GM patch fresh each
                    // note so GUI changes apply immediately to subsequent
                    // keypresses.
                    let (engine, want_program, want_bank) = {
                        let lp = live_for_midi.lock().unwrap();
                        (lp.engine, lp.sf_program, lp.sf_bank)
                    };

                    // Drive the shared SoundFont synth too if we're in
                    // sf-piano (or whenever a synth is loaded -- harmless
                    // for other engines because the placeholder is what
                    // routes audio for them, not the synth).
                    if engine == Engine::SfzPiano {
                        if let Some(player) = sfz_for_midi.lock().unwrap().as_mut() {
                            player.note_on(channel, note, velocity);
                        }
                    } else if engine == Engine::SfPiano {
                        if let Some(synth) = synth_for_midi.lock().unwrap().as_mut() {
                            // Push Bank Select + Program Change ONLY when
                            // the desired pair has actually changed since
                            // last note. Order: Bank Select (CC 0 MSB)
                            // first, then Program Change -- bank select
                            // takes effect on the next PC per GM/GS spec.
                            if (want_bank as u16) != last_applied_bank
                                || (want_program as u16) != last_applied_program
                            {
                                synth.process_midi_message(
                                    channel as i32,
                                    0xB0,
                                    0,
                                    want_bank as i32,
                                );
                                synth.process_midi_message(
                                    channel as i32,
                                    0xC0,
                                    want_program as i32,
                                    0,
                                );
                                last_applied_bank = want_bank as u16;
                                last_applied_program = want_program as u16;
                            }
                            synth.note_on(channel as i32, note as i32, velocity as i32);
                        }
                    }

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
                    // Forward note-off to the shared synth too so its envelope
                    // moves into release. We can't tell from here whether this
                    // particular note was an sf-piano voice without scanning
                    // the pool, but rustysynth ignores note_off for inactive
                    // notes, so the call is safe and cheap.
                    if let Some(synth) = synth_for_midi.lock().unwrap().as_mut() {
                        synth.note_off(channel as i32, note as i32);
                    }
                    // Forward to SFZ player too. Like rustysynth above, this
                    // is safe whether or not sfz-piano is the active engine —
                    // an inactive note_off is a no-op inside the player.
                    if let Some(player) = sfz_for_midi.lock().unwrap().as_mut() {
                        player.note_off(channel, note);
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
                        // Top out at 10.0 to match the GUI slider + CC70 range.
                        // tanh soft-clip handles saturation past ~3.
                        7 => {
                            let new_master = (cc_val as f32 / 127.0) * 10.0;
                            live_for_midi.lock().unwrap().master = new_master;
                        }
                        // CC 70 = MPK K1 (relative encoder by default).
                        // 1 tick = +/- 0.1 master gain, clamped 0..10.0.
                        70 => {
                            let mut p = live_for_midi.lock().unwrap();
                            p.master = (p.master + delta_ticks as f32 * 0.1).clamp(0.0, 10.0);
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
    let synth_for_audio = shared_synth.clone();
    let sfz_for_audio = shared_sfz.clone();
    let err_fn = |err| eprintln!("audio stream error: {err}");

    // Build an F32 output stream. Used both for the native F32 path and as a
    // fallback when cpal reports a non-{F32,I16,U16} format we don't handle.
    let build_f32_stream = |dev: &cpal::Device, cfg: &StreamConfig| {
        let voices_arc = voices_for_audio.clone();
        let live_arc = live_for_audio.clone();
        let synth_arc = synth_for_audio.clone();
        let sfz_arc = sfz_for_audio.clone();
        // Preallocated mono / sf-stereo scratch reused across audio callbacks.
        // Sized lazily on the first callback once we know the buffer size;
        // cpal callbacks are FnMut so capturing `mut` state is fine.
        let mut mono_scratch: Vec<f32> = Vec::new();
        let mut sf_left: Vec<f32> = Vec::new();
        let mut sf_right: Vec<f32> = Vec::new();
        let mut reverb = Reverb::new(ir_samples.clone());
        let mut limiter_gain: f32 = 1.0;
        // Shared sympathetic string bank — lives for the whole stream so any
        // note played excites the SAME 24 resonator strings and builds up a
        // cross-voice "halo of neighbors" the way a real piano's undamped
        // strings do.
        let mut sympathetic = SympatheticBank::new_piano(sr_hz as f32);
        dev.build_output_stream(
            cfg,
            move |out: &mut [f32], _| {
                let (master, wet) = {
                    let lp = live_arc.lock().unwrap();
                    (lp.master, lp.reverb_wet)
                };
                audio_callback(
                    out,
                    channels,
                    &voices_arc,
                    &synth_arc,
                    &sfz_arc,
                    master,
                    wet,
                    &mut mono_scratch,
                    &mut sf_left,
                    &mut sf_right,
                    &mut reverb,
                    &mut limiter_gain,
                    &mut sympathetic,
                );
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
            let synth_arc = synth_for_audio.clone();
            let sfz_arc = sfz_for_audio.clone();
            let mut mono_scratch: Vec<f32> = Vec::new();
            let mut sf_left: Vec<f32> = Vec::new();
            let mut sf_right: Vec<f32> = Vec::new();
            let mut interleaved_scratch: Vec<f32> = Vec::new();
            let mut reverb = Reverb::new(ir_samples.clone());
            let mut limiter_gain: f32 = 1.0;
            let mut sympathetic = SympatheticBank::new_piano(sr_hz as f32);
            device.build_output_stream(
                &stream_cfg,
                move |out: &mut [i16], _| {
                    let (master, wet) = {
                        let lp = live_arc.lock().unwrap();
                        (lp.master, lp.reverb_wet)
                    };
                    if interleaved_scratch.len() != out.len() {
                        interleaved_scratch.resize(out.len(), 0.0);
                    }
                    audio_callback(
                        &mut interleaved_scratch,
                        channels,
                        &voices_arc,
                        &synth_arc,
                        &sfz_arc,
                        master,
                        wet,
                        &mut mono_scratch,
                        &mut sf_left,
                        &mut sf_right,
                        &mut reverb,
                        &mut limiter_gain,
                        &mut sympathetic,
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
            let synth_arc = synth_for_audio.clone();
            let sfz_arc = sfz_for_audio.clone();
            let mut mono_scratch: Vec<f32> = Vec::new();
            let mut sf_left: Vec<f32> = Vec::new();
            let mut sf_right: Vec<f32> = Vec::new();
            let mut interleaved_scratch: Vec<f32> = Vec::new();
            let mut reverb = Reverb::new(ir_samples.clone());
            let mut limiter_gain: f32 = 1.0;
            let mut sympathetic = SympatheticBank::new_piano(sr_hz as f32);
            device.build_output_stream(
                &stream_cfg,
                move |out: &mut [u16], _| {
                    let (master, wet) = {
                        let lp = live_arc.lock().unwrap();
                        (lp.master, lp.reverb_wet)
                    };
                    if interleaved_scratch.len() != out.len() {
                        interleaved_scratch.resize(out.len(), 0.0);
                    }
                    audio_callback(
                        &mut interleaved_scratch,
                        channels,
                        &voices_arc,
                        &synth_arc,
                        &sfz_arc,
                        master,
                        wet,
                        &mut mono_scratch,
                        &mut sf_left,
                        &mut sf_right,
                        &mut reverb,
                        &mut limiter_gain,
                        &mut sympathetic,
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

#[allow(clippy::too_many_arguments)]
fn audio_callback(
    out: &mut [f32],
    channels: u16,
    voices: &Arc<Mutex<Vec<Voice>>>,
    synth: &SharedSynth,
    sfz: &SharedSfz,
    master: f32,
    reverb_wet: f32,
    mono: &mut Vec<f32>,
    sf_left: &mut Vec<f32>,
    sf_right: &mut Vec<f32>,
    reverb: &mut Reverb,
    // Persistent gain state for the limiter, kept across callbacks so we
    // can ramp smoothly from prev block's gain to this block's target,
    // avoiding the per-block step discontinuity that sounds like chopping.
    limiter_gain: &mut f32,
    // Shared sympathetic string bank: 24 KS strings ringing across the
    // piano range, driven by the mixed voice output as a soundboard proxy.
    // Lives for the stream lifetime — any note played pumps energy into
    // the bank, and bank strings whose partials overlap with the struck
    // note's partials keep ringing after the note is released (the
    // "halo of neighbor strings" the user perceived missing from our
    // piano vs Salamander SFZ samples).
    sympathetic: &mut SympatheticBank,
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

    // Mix the shared SoundFont synth into the mono bus. The synth always
    // renders stereo; we downmix (L+R)/2. If no synth is loaded (no --sf2)
    // this whole branch is a single None check.
    {
        let mut guard = synth.lock().unwrap();
        if let Some(s) = guard.as_mut() {
            if sf_left.len() != frames {
                sf_left.resize(frames, 0.0);
                sf_right.resize(frames, 0.0);
            } else {
                sf_left.fill(0.0);
                sf_right.fill(0.0);
            }
            s.render(sf_left.as_mut_slice(), sf_right.as_mut_slice());
            for i in 0..frames {
                mono[i] += (sf_left[i] + sf_right[i]) * 0.5;
            }
        }
    }

    // Mix the shared SFZ player into the mono bus. Same downmix story as
    // the SF2 path; reuses the same sf_left/sf_right scratch since the two
    // engines are mutually exclusive in any sane configuration (and even
    // if both were loaded, mixing both is harmless).
    {
        let mut guard = sfz.lock().unwrap();
        if let Some(player) = guard.as_mut() {
            if sf_left.len() != frames {
                sf_left.resize(frames, 0.0);
                sf_right.resize(frames, 0.0);
            } else {
                sf_left.fill(0.0);
                sf_right.fill(0.0);
            }
            player.render(sf_left.as_mut_slice(), sf_right.as_mut_slice());
            for i in 0..frames {
                mono[i] += (sf_left[i] + sf_right[i]) * 0.5;
            }
        }
    }

    // Sympathetic string bank (shared across voices): drive the bank with
    // the current mixed bus as a soundboard-output proxy, then add its
    // output back into the bus. Coupling 0.02 (bank takes ~2% of the bus
    // per sample); mix gain 0.3 (bank contributes ~30% of its output
    // amplitude to the final mix). No closed feedback loop — the bank
    // only reads from the bus, never writes into any voice's private
    // soundboard — so stability reduces to the per-string KS decay
    // (0.9994) being below unity.
    {
        const COUPLING: f32 = 0.02;
        const MIX: f32 = 0.3;
        for sample in mono.iter_mut() {
            let board_drive = *sample;
            let sym_out = sympathetic.process(board_drive, COUPLING);
            *sample += sym_out * MIX;
        }
    }

    // Body IR convolution before soft-clip so saturation acts on the
    // post-reverb signal -- otherwise reverb tail clips against tanh.
    reverb.process(mono.as_mut_slice(), reverb_wet);

    // Compressor reverted — didn't help perceptibly. Physical ceiling
    // (±1.0 digital full-scale × Shokz hardware max) dominates. Plain tanh.
    let _ = limiter_gain;
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
