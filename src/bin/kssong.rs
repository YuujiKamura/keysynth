use std::path::PathBuf;

use midly::num::{u15, u24, u28, u4, u7};
use midly::{Format, Header, MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind};

use keysynth::library_db::{LibraryDb, SongFilter, SongSort};
use keysynth::song::{parse_progression_with_key, Chord, Key, Voicing};
use keysynth::synth::{midi_to_freq, Engine, VoiceImpl};
use keysynth::voices::guitar::GuitarVoice;

#[allow(dead_code, deprecated)]
#[path = "render_midi.rs"]
mod render_midi_impl;

const PPQ: u16 = 480;
const BEATS_PER_BAR: u32 = 4;
const DEFAULT_BPM: u32 = 120;
const DEFAULT_VELOCITY: u8 = 90;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VoiceChoice {
    Engine(Engine),
    Guitar,
}

impl VoiceChoice {
    fn parse(input: &str) -> Result<Self, String> {
        match input {
            "square" => Ok(Self::Engine(Engine::Square)),
            "ks" => Ok(Self::Engine(Engine::Ks)),
            "ks-rich" => Ok(Self::Engine(Engine::KsRich)),
            "sub" => Ok(Self::Engine(Engine::Sub)),
            "fm" => Ok(Self::Engine(Engine::Fm)),
            "piano" => Ok(Self::Engine(Engine::Piano)),
            "piano-thick" => Ok(Self::Engine(Engine::PianoThick)),
            "piano-lite" => Ok(Self::Engine(Engine::PianoLite)),
            "piano-5am" => Ok(Self::Engine(Engine::Piano5AM)),
            "piano-modal" => Ok(Self::Engine(Engine::PianoModal)),
            "koto" => Ok(Self::Engine(Engine::Koto)),
            "sf-piano" => Ok(Self::Engine(Engine::SfPiano)),
            "sfz-piano" => Ok(Self::Engine(Engine::SfzPiano)),
            "guitar" => Ok(Self::Guitar),
            // The library DB stores manifest-level suggested_voice
            // strings ("guitar-stk", "piano-stk", ...) that name a
            // voices_live/<id>/ cdylib slot rather than a static
            // Engine variant. kssong has no live-reload path of its
            // own, so we route those onto the closest static voice:
            // guitar-stk → GuitarVoice (the in-tree STK guitar port),
            // piano-stk  → Engine::Piano (in-tree KS+modal hybrid).
            // This means "kssong --recommended-voice guitar-stk" picks
            // a Tárrega piece and renders it through the offline
            // guitar voice, which is the closest stand-in available
            // without spinning up the live_reload subsystem.
            "guitar-stk" => Ok(Self::Guitar),
            "piano-stk" => Ok(Self::Engine(Engine::Piano)),
            other => Err(format!(
                "unknown --voice: {other} (piano|piano-modal|piano-thick|piano-lite|piano-5am|guitar|guitar-stk|square|ks|ks-rich|fm|sub|koto|sf-piano|sfz-piano)"
            )),
        }
    }
}

#[derive(Debug)]
struct PlayArgs {
    /// Either a literal chord progression ("C - G - Am - F") or `None`
    /// when the song is being chosen from the library DB via the
    /// composer/era/tag/recommended-voice flags.
    progression: Option<String>,
    voice: VoiceChoice,
    /// True when --voice was not passed by the user — lets the
    /// library-DB path opportunistically pick the song's
    /// suggested_voice (e.g. guitar-stk for Tárrega) instead of the
    /// generic Engine::Piano default.
    voice_was_default: bool,
    voicing: Voicing,
    bpm: u32,
    bars: Option<usize>,
    key: Option<Key>,
    out_path: PathBuf,
    sfz_path: Option<PathBuf>,
    modal_lut_path: Option<PathBuf>,
    /// Optional library-DB filter. When any field is `Some`, kssong
    /// switches from "render the supplied progression" to "find the
    /// first matching song in bench-out/library.db and render its
    /// MIDI". Mutually exclusive with the progression positional in
    /// the sense that one or the other must produce notes.
    db_filter: SongFilter,
}

fn db_filter_active(f: &SongFilter) -> bool {
    f.composer.is_some()
        || f.era.is_some()
        || f.instrument.is_some()
        || f.tag.is_some()
        || f.recommended_voice.is_some()
}

fn print_help() {
    eprintln!(
        "kssong - chord-progression + library-DB song renderer\n\n\
         usage:\n  \
         kssong play \"C - G - Am - F\" --voice piano --bpm 120 --out out.wav\n  \
         kssong play --composer bach --voice piano-modal --out bach.wav\n  \
         kssong play --recommended-voice guitar-stk --out tarrega.wav\n\n\
         play options (chord-progression mode):\n  \
         --voice NAME      piano|piano-modal|piano-thick|piano-lite|piano-5am|guitar|square|ks|ks-rich|fm|sub|koto|sf-piano|sfz-piano\n  \
         --voicing MODE    close|piano|guitar|open (default close)\n  \
         --bpm N           tempo in beats per minute (default 120)\n  \
         --bars N          total bars to render; repeats/truncates progression to fit\n  \
         --key KEY         base key for roman numerals (example: C, F#, Bb)\n  \
         --sfz PATH        required when --voice sfz-piano\n  \
         --modal-lut PATH  optional modal LUT override for piano-modal\n  \
         --out PATH        output WAV path (required)\n\n\
         play options (library-DB mode, issue #66):\n  \
         --composer NAME           pick a song matching composer key (\"bach\", \"tarrega\")\n  \
         --era ERA                 Baroque|Classical|Romantic|Modern|Traditional\n  \
         --tag TAG                 song tag from manifest.json\n  \
         --recommended-voice V     song whose `suggested_voice` is V (e.g. guitar-stk)\n  \
         When any of the four flags above is set, --voice / --voicing default to\n  \
         the song's suggested voice if --voice is not given.\n  \
         Bench-out/library.db is auto-rebuilt from manifest.json before the query.\n  \
         --voicing/--voice still apply on top to override the engine."
    );
}

fn parse_args() -> Result<PlayArgs, String> {
    let mut iter = std::env::args().skip(1);
    let Some(subcommand) = iter.next() else {
        print_help();
        return Err("missing subcommand".to_string());
    };

    if matches!(subcommand.as_str(), "--help" | "-h") {
        print_help();
        std::process::exit(0);
    }

    if subcommand != "play" {
        return Err(format!("unknown subcommand: {subcommand}"));
    }

    // The first positional after "play" is treated as the chord
    // progression unless it starts with "--", in which case the user
    // has gone straight into library-DB mode (no progression at all).
    let mut iter = iter.peekable();
    let progression: Option<String> = match iter.peek() {
        Some(arg) if arg.starts_with("--") => None,
        Some(_) => iter.next(),
        None => None,
    };
    let mut voice = VoiceChoice::Engine(Engine::Piano);
    let mut voice_was_default = true;
    let mut voicing = Voicing::Close;
    let mut bpm = DEFAULT_BPM;
    let mut bars = None;
    let mut key = None;
    let mut out_path: Option<PathBuf> = None;
    let mut sfz_path = None;
    let mut modal_lut_path = None;
    let mut db_filter = SongFilter {
        sort: SongSort::ByComposer,
        ..Default::default()
    };

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--voice" | "--engine" => {
                let value = iter.next().ok_or("--voice needs a value")?;
                voice = VoiceChoice::parse(&value)?;
                voice_was_default = false;
            }
            "--voicing" => {
                let value = iter.next().ok_or("--voicing needs a value")?;
                voicing = Voicing::parse(&value).map_err(|e| format!("bad --voicing: {e}"))?;
            }
            "--bpm" => {
                bpm = iter
                    .next()
                    .ok_or("--bpm needs a value")?
                    .parse()
                    .map_err(|e| format!("bad --bpm: {e}"))?;
                if bpm == 0 {
                    return Err("--bpm must be > 0".to_string());
                }
            }
            "--bars" => {
                let value: usize = iter
                    .next()
                    .ok_or("--bars needs a value")?
                    .parse()
                    .map_err(|e| format!("bad --bars: {e}"))?;
                if value == 0 {
                    return Err("--bars must be > 0".to_string());
                }
                bars = Some(value);
            }
            "--key" => {
                let value = iter.next().ok_or("--key needs a value")?;
                key = Some(Key::parse(&value).map_err(|e| format!("bad --key: {e}"))?);
            }
            "--out" => {
                out_path = Some(PathBuf::from(iter.next().ok_or("--out needs a value")?));
            }
            "--sfz" => {
                sfz_path = Some(PathBuf::from(iter.next().ok_or("--sfz needs a value")?));
            }
            "--modal-lut" => {
                modal_lut_path = Some(PathBuf::from(
                    iter.next().ok_or("--modal-lut needs a value")?,
                ));
            }
            "--composer" => {
                db_filter.composer = Some(iter.next().ok_or("--composer needs a value")?);
            }
            "--era" => {
                db_filter.era = Some(iter.next().ok_or("--era needs a value")?);
            }
            "--instrument" => {
                db_filter.instrument = Some(iter.next().ok_or("--instrument needs a value")?);
            }
            "--tag" => {
                db_filter.tag = Some(iter.next().ok_or("--tag needs a value")?);
            }
            "--recommended-voice" => {
                db_filter.recommended_voice =
                    Some(iter.next().ok_or("--recommended-voice needs a value")?);
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }

    if progression.is_none() && !db_filter_active(&db_filter) {
        return Err(
            "play requires either a chord progression string or one of \
             --composer / --era / --tag / --instrument / --recommended-voice"
                .to_string(),
        );
    }

    Ok(PlayArgs {
        progression,
        voice,
        voice_was_default,
        voicing,
        bpm,
        bars,
        key,
        out_path: out_path.ok_or("--out is required")?,
        sfz_path,
        modal_lut_path,
        db_filter,
    })
}

fn expand_progression(chords: &[Chord], total_bars: usize) -> Vec<Chord> {
    (0..total_bars)
        .map(|bar| chords[bar % chords.len()].clone())
        .collect()
}

fn build_smf_bytes(chords: &[Chord], bpm: u32, voicing: Voicing) -> Result<Vec<u8>, String> {
    #[derive(Clone, Copy)]
    enum MidiKind {
        NoteOff(u8),
        NoteOn(u8),
    }

    let bar_ticks = BEATS_PER_BAR * PPQ as u32;
    let mut timeline: Vec<(u32, MidiKind)> = Vec::new();
    for (bar_idx, chord) in chords.iter().enumerate() {
        let start_tick = bar_idx as u32 * bar_ticks;
        let end_tick = start_tick + bar_ticks;
        for note in chord.voice(voicing) {
            timeline.push((end_tick, MidiKind::NoteOff(note)));
            timeline.push((start_tick, MidiKind::NoteOn(note)));
        }
    }
    timeline.sort_by_key(|(tick, kind)| {
        let order = match kind {
            MidiKind::NoteOff(_) => 0_u8,
            MidiKind::NoteOn(_) => 1_u8,
        };
        (*tick, order)
    });

    let tempo_us_per_quarter = 60_000_000_u32 / bpm;
    let header = Header {
        format: Format::Parallel,
        timing: Timing::Metrical(u15::new(PPQ)),
    };

    let tempo_track = vec![
        TrackEvent {
            delta: u28::new(0),
            kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::new(tempo_us_per_quarter))),
        },
        TrackEvent {
            delta: u28::new(0),
            kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
        },
    ];

    let mut note_track: Vec<TrackEvent<'static>> = Vec::with_capacity(timeline.len() + 1);
    let mut last_tick = 0_u32;
    for (tick, kind) in timeline {
        let delta = tick.saturating_sub(last_tick);
        last_tick = tick;
        let message = match kind {
            MidiKind::NoteOn(note) => MidiMessage::NoteOn {
                key: u7::new(note),
                vel: u7::new(DEFAULT_VELOCITY),
            },
            MidiKind::NoteOff(note) => MidiMessage::NoteOff {
                key: u7::new(note),
                vel: u7::new(0),
            },
        };
        note_track.push(TrackEvent {
            delta: u28::new(delta),
            kind: TrackEventKind::Midi {
                channel: u4::new(0),
                message,
            },
        });
    }
    note_track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });

    let smf = Smf {
        header,
        tracks: vec![tempo_track, note_track],
    };

    let mut bytes = Vec::new();
    smf.write_std(&mut bytes)
        .map_err(|e| format!("midly write: {e}"))?;
    Ok(bytes)
}

fn render_guitar(events: &[render_midi_impl::NoteEvent]) -> (Vec<f32>, Vec<f32>) {
    let max_end = events
        .iter()
        .map(|event| event.start_sec + event.duration_sec)
        .fold(0.0_f32, f32::max);
    let total_sec = max_end + render_midi_impl::RELEASE_TAIL_SEC;
    let total_samples = (total_sec * render_midi_impl::SR as f32) as usize;
    let mut left = vec![0.0_f32; total_samples];
    let mut right = vec![0.0_f32; total_samples];

    const RELEASE_BUDGET_SEC: f32 = 1.5;
    let release_budget_samples = (RELEASE_BUDGET_SEC * render_midi_impl::SR as f32) as usize;

    for event in events {
        let start_sample = (event.start_sec * render_midi_impl::SR as f32) as usize;
        let release_sample =
            ((event.start_sec + event.duration_sec) * render_midi_impl::SR as f32) as usize;
        if start_sample >= total_samples {
            continue;
        }
        let release_at = release_sample.saturating_sub(start_sample);
        let voice_total = (release_at + release_budget_samples).min(total_samples - start_sample);
        let mut voice_buf = vec![0.0_f32; voice_total];

        let freq = midi_to_freq(event.midi_note);
        let mut voice: Box<dyn VoiceImpl + Send> = Box::new(GuitarVoice::new(
            render_midi_impl::SR as f32,
            freq,
            event.velocity,
        ));
        if release_at > 0 {
            voice.render_add(&mut voice_buf[..release_at.min(voice_total)]);
        }
        voice.trigger_release();
        if release_at < voice_total {
            voice.render_add(&mut voice_buf[release_at..voice_total]);
        }

        let (left_gain, right_gain) = constant_power_pan(pan_for_note(event.midi_note));
        for (idx, sample) in voice_buf.iter().enumerate() {
            left[start_sample + idx] += *sample * left_gain;
            right[start_sample + idx] += *sample * right_gain;
        }
    }

    (left, right)
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

fn render_to_wav(args: &PlayArgs, midi_bytes: &[u8]) -> Result<(), String> {
    let events = render_midi_impl::parse_smf(midi_bytes, 1.0)?;
    let (left, right) = match args.voice {
        VoiceChoice::Engine(Engine::SfzPiano) => {
            let sfz_path = args
                .sfz_path
                .as_deref()
                .ok_or("--sfz PATH is required when --voice sfz-piano")?;
            render_midi_impl::render_sfz(sfz_path, &events)?
        }
        VoiceChoice::Engine(Engine::SfPiano) => {
            return Err(
                "sf-piano is live-only in keysynth; use sfz-piano or a synthesized engine"
                    .to_string(),
            );
        }
        VoiceChoice::Engine(engine) => {
            render_midi_impl::render_keysynth(engine, &events, args.modal_lut_path.as_deref())
        }
        VoiceChoice::Guitar => render_guitar(&events),
    };
    render_midi_impl::write_wav_stereo(&args.out_path, &left, &right)
}

fn run_play(mut args: PlayArgs) -> Result<(), String> {
    if db_filter_active(&args.db_filter) {
        return run_play_from_library(&mut args);
    }
    let progression = args
        .progression
        .as_ref()
        .ok_or("play requires a chord progression or a library-DB filter")?;
    let chords = parse_progression_with_key(progression, args.key)
        .map_err(|e| format!("progression parse: {e}"))?;
    let total_bars = args.bars.unwrap_or(chords.len());
    let expanded = expand_progression(&chords, total_bars);
    let midi_bytes = build_smf_bytes(&expanded, args.bpm, args.voicing)?;
    render_to_wav(&args, &midi_bytes)?;
    eprintln!(
        "kssong: wrote {} (bars={} bpm={} voicing={})",
        args.out_path.display(),
        total_bars,
        args.bpm,
        args.voicing.as_str()
    );
    Ok(())
}

/// Library-DB rendering path (issue #66). Opens bench-out/library.db,
/// re-imports manifest.json so the catalog is up to date, runs the
/// supplied filter, and renders the first match through the existing
/// render_midi path. The MIDI file from the manifest is loaded
/// verbatim — kssong's chord-progression generator is bypassed.
///
/// Domain note: when the user only specifies a filter (no --voice),
/// the song's `suggested_voice` from the manifest wins. That's the
/// editorial pick from PR #63 — Tárrega + Bach guitar pieces lean on
/// guitar-stk; Mozart / Satie / Albéniz lean on piano-modal. Picking
/// up that hint by default is the whole point of having the metadata
/// in the DB, so kssong honours it unless the user overrides.
fn run_play_from_library(args: &mut PlayArgs) -> Result<(), String> {
    let db_path = PathBuf::from("bench-out/library.db");
    let manifest = PathBuf::from("bench-out/songs/manifest.json");
    let voices_live = PathBuf::from("voices_live");
    let mut db = LibraryDb::open(&db_path).map_err(|e| format!("library_db open: {e}"))?;
    db.migrate().map_err(|e| format!("library_db migrate: {e}"))?;
    if manifest.is_file() {
        db.import_songs(&manifest)
            .map_err(|e| format!("library_db import_songs: {e}"))?;
    } else {
        return Err(format!(
            "library_db: missing manifest at {} — run `keysynth_db rebuild` from a checkout",
            manifest.display()
        ));
    }
    if voices_live.is_dir() {
        let _ = db.import_voices(&voices_live);
    }

    let songs = db
        .query_songs(&args.db_filter)
        .map_err(|e| format!("library_db query: {e}"))?;
    let song = songs.into_iter().next().ok_or_else(|| {
        format!(
            "library_db: no song matched filter {{composer={:?}, era={:?}, instrument={:?}, \
             tag={:?}, recommended_voice={:?}}}",
            args.db_filter.composer,
            args.db_filter.era,
            args.db_filter.instrument,
            args.db_filter.tag,
            args.db_filter.recommended_voice,
        )
    })?;

    if args.voice_was_default {
        if let Some(suggested) = &song.suggested_voice {
            match VoiceChoice::parse(suggested) {
                Ok(v) => {
                    args.voice = v;
                    eprintln!("kssong: using song's suggested voice '{suggested}'");
                }
                Err(e) => {
                    eprintln!(
                        "kssong: ignoring suggested_voice {suggested:?} ({e}); \
                         keeping default piano. Pass --voice to override."
                    );
                }
            }
        }
    }

    let midi_bytes = std::fs::read(&song.midi_path).map_err(|e| {
        format!(
            "library_db: read MIDI {}: {e}",
            song.midi_path.display()
        )
    })?;
    render_to_wav_bytes(args, &midi_bytes)?;
    eprintln!(
        "kssong: wrote {} from library_db song={} ({}, {})",
        args.out_path.display(),
        song.id,
        song.composer,
        song.era.as_deref().unwrap_or("?"),
    );
    let voice_label = match args.voice {
        VoiceChoice::Engine(e) => format!("{e:?}"),
        VoiceChoice::Guitar => "Guitar".to_string(),
    };
    let _ = db.record_play(&song.id, &voice_label, None);
    Ok(())
}

/// Wrap `render_to_wav` for the library-DB path. `render_to_wav`
/// already takes `&PlayArgs + &[u8]`, so this is a thin alias to keep
/// both call sites symmetrical and to leave a single insertion point
/// if a future patch wants to record render duration on the play row.
fn render_to_wav_bytes(args: &PlayArgs, midi_bytes: &[u8]) -> Result<(), String> {
    render_to_wav(args, midi_bytes)
}

fn main() {
    #[cfg(target_arch = "x86_64")]
    #[allow(deprecated)]
    unsafe {
        use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
        let csr = _mm_getcsr();
        _mm_setcsr(csr | 0x8040);
    }

    let args = match parse_args() {
        Ok(args) => args,
        Err(err) => {
            eprintln!("kssong: {err}");
            std::process::exit(2);
        }
    };

    if let Err(err) = run_play(args) {
        eprintln!("kssong: {err}");
        std::process::exit(2);
    }
}
