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
    make_voice, midi_to_freq, DashState, Engine, LiveParams, MixMode, Voice, VoiceImpl,
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
    mix_mode: MixMode,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            engine: Engine::Square,
            port: None,
            list: false,
            // Master gain pre-tanh. 3.0 sits well into the soft-clip
            // saturation region for any non-trivial signal — single
            // voice peak (1.0) → tanh(3.0) ≈ 0.995, two voices already
            // bumping the ceiling. This trades early saturation for an
            // "always loud, always present" feel under live MIDI, vs
            // the older 0.3 which bench-favoured a clean linear region
            // but felt distant on real hardware. Override via --master.
            master: 3.0,
            sf2: None,
            sf2_program: 0,
            sf2_bank: 0,
            sfz: None,
            ir_path: None,
            // Body-IR wet default. 0.3 gives the piano engines a
            // perceptible room character without burying the soundboard
            // halo. Was 0.15 briefly when the piano engines first
            // grew their own modal soundboard, but live playing felt
            // dry; restore to 0.3.
            reverb_wet: 0.3,
            // Default to plain tanh; user can switch to Limiter or
            // ParallelComp via GUI dropdown / `--mix-mode <label>`.
            // Issue #4 phase 1.
            mix_mode: MixMode::Plain,
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
                    "piano-thick" => Engine::PianoThick,
                    "piano-lite" => Engine::PianoLite,
                    "piano-5am" => Engine::Piano5AM,
                    other => return Err(format!(
                        "unknown engine: {other} (square|ks|ks-rich|sub|fm|piano|koto|sf-piano|sfz-piano|piano-thick|piano-lite|piano-5am)"
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
                out.sf2 = Some(PathBuf::from(iter.next().ok_or("--sf2 needs a path")?));
            }
            "--sf2-program" => {
                out.sf2_program = iter
                    .next()
                    .ok_or("--sf2-program needs an integer")?
                    .parse()
                    .map_err(|e| format!("bad --sf2-program: {e}"))?;
            }
            "--sf2-bank" => {
                out.sf2_bank = iter
                    .next()
                    .ok_or("--sf2-bank needs an integer")?
                    .parse()
                    .map_err(|e| format!("bad --sf2-bank: {e}"))?;
            }
            "--sfz" => {
                out.sfz = Some(PathBuf::from(iter.next().ok_or("--sfz needs a path")?));
            }
            "--ir" => {
                out.ir_path = Some(PathBuf::from(iter.next().ok_or("--ir needs a path")?));
            }
            "--reverb" => {
                let v = iter.next().ok_or("--reverb needs a float 0..1")?;
                out.reverb_wet = v
                    .parse::<f32>()
                    .map_err(|e| format!("bad --reverb: {e}"))?
                    .clamp(0.0, 1.0);
            }
            "--mix-mode" => {
                let v = iter.next().ok_or("--mix-mode needs a label")?;
                out.mix_mode = MixMode::from_label(&v)
                    .ok_or_else(|| format!("bad --mix-mode: {v} (plain|limiter|parallel-comp)"))?;
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
                                if sp.extension().and_then(|s| s.to_str()) == Some("sfz") {
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
            File::open(sf2_path).map_err(|e| format!("opening SoundFont {:?}: {e}", sf2_path))?,
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
            sf2_path.display(),
            args.sf2_program,
            args.sf2_bank
        );
    } else if args.engine == Engine::SfPiano {
        return Err("engine 'sf-piano' requires --sf2 PATH (no SoundFont loaded)".into());
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
        return Err("engine 'sfz-piano' requires --sfz PATH (no SFZ manifest loaded)".into());
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
                p.display(),
                ir.len(),
                1000.0 * ir.len() as f32 / sr_hz as f32,
                sr_hz
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
        mix_mode: args.mix_mode,
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
                        //
                        // Pre-2026-04-25: most VoiceImpls forgot to override
                        // `is_releasing` (only SfPianoPlaceholder did). The
                        // released-voice scan therefore returned None on
                        // every Piano/Ks/etc. voice and `unwrap_or(0)`
                        // killed slot 0 — often a still-sustained note,
                        // perceived as "no sound after pressing many keys".
                        // ReleaseEnvelope-based VoiceImpl trait now provides
                        // is_releasing() by default, so the scan finds a
                        // real candidate as long as ANY voice in the pool
                        // has been released.
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
                    let cc_num = note; // (raw[1])
                    let cc_val = velocity; // (raw[2])
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
                        65..=127 => -(128 - cc_val as i32),
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
        let mut mono_compressed: Vec<f32> = Vec::new();
        // Shared sympathetic string bank — lives for the whole stream so any
        // note played excites the SAME 24 resonator strings and builds up a
        // cross-voice "halo of neighbors" the way a real piano's undamped
        // strings do.
        let mut sympathetic = SympatheticBank::new_piano(sr_hz as f32);
        dev.build_output_stream(
            cfg,
            move |out: &mut [f32], _| {
                let (master, wet, engine, mix_mode) = {
                    let lp = live_arc.lock().unwrap();
                    (lp.master, lp.reverb_wet, lp.engine, lp.mix_mode)
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
                    engine,
                    mix_mode,
                    &mut mono_compressed,
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
            let mut mono_compressed: Vec<f32> = Vec::new();
            let mut sympathetic = SympatheticBank::new_piano(sr_hz as f32);
            device.build_output_stream(
                &stream_cfg,
                move |out: &mut [i16], _| {
                    let (master, wet, engine, mix_mode) = {
                        let lp = live_arc.lock().unwrap();
                        (lp.master, lp.reverb_wet, lp.engine, lp.mix_mode)
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
                        engine,
                        mix_mode,
                        &mut mono_compressed,
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
            let mut mono_compressed: Vec<f32> = Vec::new();
            let mut sympathetic = SympatheticBank::new_piano(sr_hz as f32);
            device.build_output_stream(
                &stream_cfg,
                move |out: &mut [u16], _| {
                    let (master, wet, engine, mix_mode) = {
                        let lp = live_arc.lock().unwrap();
                        (lp.master, lp.reverb_wet, lp.engine, lp.mix_mode)
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
                        engine,
                        mix_mode,
                        &mut mono_compressed,
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
    // Currently selected engine. The sympathetic bank only runs when this
    // is in the piano family (`Engine::is_piano_family()`); for SF/SFZ
    // engines it would double-resonate the sample-recorded body, and for
    // non-piano engines (square/sub/fm/koto) the piano-tuned bank just
    // adds inappropriate piano halo. Per issue #2 audit acb2f0360b347a623.
    engine: Engine,
    // Final-stage bus mixing strategy (issue #4 polyphony headroom).
    // Selectable live so the user can A/B Plain / Limiter /
    // ParallelComp under chord playing.
    mix_mode: MixMode,
    // Persistent scratch buffer used by ParallelComp for the
    // "compressed" path (bus B). Caller-owned so we don't allocate on
    // the audio thread.
    mono_compressed: &mut Vec<f32>,
) {
    // Flush-To-Zero + Denormals-Are-Zero on the audio thread.
    //
    // High-Q biquad resonators (the soundboard mode bank in PianoVoice
    // and family) ring out exponentially. Once `z1`/`z2` cross
    // ~1.18e-38 they enter f32 denormal range, where x86 SSE arithmetic
    // is 100-1000x slower per op. With 32 voices × 12 modes × 2 state
    // vars × 48 kHz the cumulative per-callback cost spikes enough to
    // miss the audio deadline → cpal underruns → DAC outputs zeros →
    // the user hears "音が出なくなる on rapid striking" specifically
    // on piano family (Piano5AM has no soundboard, no denormals, no
    // bug). Setting MXCSR FZ (bit 15) + DAZ (bit 6) — i.e. mask
    // 0x8040 — makes the SSE unit treat denormals as zero on both
    // input and output, killing the slowdown without changing audible
    // behaviour (denormals are < -750 dB FS).
    //
    // Calling once per audio_callback on x86_64 is harmless (single
    // CSR write, sub-nanosecond). On non-x86 architectures the call
    // is compiled out; denormals on those platforms are a separate
    // story (ARM has FPCR.FZ etc.) but most consumer audio runs x86.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
        let csr = _mm_getcsr();
        if csr & 0x8040 != 0x8040 {
            _mm_setcsr(csr | 0x8040);
        }
    }

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
    // output back into the bus.
    //
    // Gated to the piano family (Engine::Piano / PianoThick / PianoLite).
    // For SF/SFZ engines the recorded sample already includes the body
    // resonance — running the bank on top double-resonates and produced
    // the audit-noted (#2 / acb2f0360b347a623) sustained metallic ring on
    // the SFZ Salamander reference during bench. For non-piano engines
    // (square/sub/fm/koto) the piano-tuned 24-string bank just adds
    // inappropriate piano halo. Bank state is still ticked when other
    // engines are active so any prior energy decays naturally — we just
    // skip the drive-and-mix.
    // Sympathetic-string bank (shared across voices) — drive with the
    // current bus level scaled by COUPLING, mix output back at MIX.
    //
    // Stability budget (2026-04-25 incident, "連打バグ"): the original
    // COUPLING = 0.02 / MIX = 0.3 pair was unstable under rapid piano-
    // family playing. Each bank string is a KS delay loop with decay
    // 0.9994 → steady-state amplitude per unit drive ≈ 1/(1−decay) ≈
    // 1667. With 24 strings summing through 1/√24 ≈ 0.204 normalisation
    // and MIX = 0.3, the per-bank-string bus contribution per unit
    // drive reached ≈ 102. Under 32-voice piano-family playing the bus
    // peak transients hit 5–30, drive 0.1–0.6, bank ring-up exploded
    // into f32 saturation territory; eventually `Inf − Inf = NaN`
    // appeared in the dispersion allpass, propagated through reverb to
    // the DAC, and the user heard it as "音が出なくなる". Verified via
    // bisect that this was structural in the bank, not in any later
    // commit — Piano5AM (no sym-bank gate) was symptom-free, all other
    // piano-family voices produced the silence on rapid striking.
    //
    // Empirically: COUPLING = 0.0002 (1/100 of the original) is stable
    // under sustained piano-family playing while still leaving the
    // sympathetic halo audible. Re-tightening this requires either a
    // per-string limiter inside the bank or a smaller MIX; tracked as
    // a follow-up to issue #2 (sym-bank stability).
    if engine.is_piano_family() {
        const COUPLING: f32 = 0.0002;
        const MIX: f32 = 0.3;
        for sample in mono.iter_mut() {
            let board_drive = *sample;
            let sym_out = sympathetic.process(board_drive, COUPLING);
            *sample += sym_out * MIX;
        }
    } else {
        for _ in 0..mono.len() {
            let _ = sympathetic.process(0.0, 0.0);
        }
    }

    // Belt-and-braces NaN/Inf guard. If ANY upstream stage (a high-Q
    // soundboard mode running away under cumulative drive, a future sym
    // bank regression, etc.) produces a non-finite sample, replace with
    // 0 before reverb gets it. Reverb is a convolution and would smear
    // a single NaN across its full IR length, turning a momentary glitch
    // into permanent silence-via-NaN-propagation.
    for s in mono.iter_mut() {
        if !s.is_finite() {
            *s = 0.0;
        }
    }

    // Body IR convolution before soft-clip so saturation acts on the
    // post-reverb signal -- otherwise reverb tail clips against tanh.
    reverb.process(mono.as_mut_slice(), reverb_wet);

    // Final-stage bus mixing — dispatched on `mix_mode` so the user can
    // A/B Plain / Limiter / ParallelComp live (issue #4). Each branch
    // is responsible for writing the final tanh-clipped sample into
    // `mono` in-place.
    match mix_mode {
        MixMode::Plain => {
            // Honest physical model: just tanh the bus. Saturates hard
            // at chord peaks; preserves exact dynamics in linear range.
            for sample in mono.iter_mut() {
                *sample = (*sample * master).tanh();
            }
        }
        MixMode::Limiter => {
            // Polyphony-aware peak limiter (one-pole envelope follower
            // → 1/peak_env attenuation when peak_env > 1.0). Catches
            // chord peaks but flattens dynamics — the listener noted
            // it as "圧縮は本来の自然音では絶対ない挙動". Kept here as a
            // baseline comparison for the parallel-comp variant.
            const ATTACK_COEF: f32 = 0.5;
            const RELEASE_COEF: f32 = 0.0001;
            for sample in mono.iter_mut() {
                let abs_s = sample.abs();
                if abs_s > *limiter_gain {
                    *limiter_gain += (abs_s - *limiter_gain) * ATTACK_COEF;
                } else {
                    *limiter_gain += (abs_s - *limiter_gain) * RELEASE_COEF;
                }
                let gain_reduction = if *limiter_gain > 1.0 {
                    1.0 / *limiter_gain
                } else {
                    1.0
                };
                *sample = (*sample * gain_reduction * master).tanh();
            }
        }
        MixMode::ParallelComp => {
            // Parallel compression / NY trick (issue #4 phase 1).
            //
            //   bus_clean      = mono                  (no dynamics)
            //   bus_compressed = mono * peak_limiter   (heavy 1/peak)
            //   final          = α · clean + β · compressed
            //
            // α + β > 1 so the compressed path actively LIFTS the
            // sustained portion of chords (where the limiter is
            // pulling gain) instead of just clamping them. Result: the
            // attack transient survives via the clean path (full
            // punch) AND the chord's sustain stays audible via the
            // compressed path (which gets a gain boost from β > 1
            // that the clean path can't because it would overshoot).
            //
            // Default α = 0.7, β = 0.6 → unity gain on a single voice
            // (peak_env < 1, compressed = clean, total = 1.3 entering
            // tanh — slight saturation for warmth) and α + β = 1.3
            // which lifts the limited sustain ~+2.3 dB over what a
            // pure limiter would give. Tunable; ear-validate later.
            const ALPHA: f32 = 0.7;
            const BETA: f32 = 0.6;
            const ATTACK_COEF: f32 = 0.5;
            const RELEASE_COEF: f32 = 0.0001;

            // Build the compressed copy in-place into `mono_compressed`.
            if mono_compressed.len() != mono.len() {
                mono_compressed.resize(mono.len(), 0.0);
            }
            for (i, sample) in mono.iter().enumerate() {
                let abs_s = sample.abs();
                if abs_s > *limiter_gain {
                    *limiter_gain += (abs_s - *limiter_gain) * ATTACK_COEF;
                } else {
                    *limiter_gain += (abs_s - *limiter_gain) * RELEASE_COEF;
                }
                let gr = if *limiter_gain > 1.0 {
                    1.0 / *limiter_gain
                } else {
                    1.0
                };
                mono_compressed[i] = sample * gr;
            }

            // Sum the two paths, scale by master, soft-clip.
            for (i, sample) in mono.iter_mut().enumerate() {
                let combined = (*sample * ALPHA + mono_compressed[i] * BETA) * master;
                *sample = combined.tanh();
            }
        }
    }

    for (frame_idx, frame) in out.chunks_mut(channels as usize).enumerate() {
        let s = mono[frame_idx];
        for slot in frame.iter_mut() {
            *slot = s;
        }
    }
}
