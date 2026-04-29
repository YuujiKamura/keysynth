//! Phase 2 keysynth Steel bindings — issue #56.
//!
//! Registers the four functions an AI agent (or any human at the REPL)
//! needs to drive keysynth from Lisp without going through the GUI or
//! the `ksctl` JSON-RPC sidecar:
//!
//! ```scheme
//! (list-voices)                     ; -> list of voice labels
//! (load-voice 'piano)               ; -> selects a voice for render
//! (query-songs :composer "Bach")    ; -> list of "id<TAB>title<TAB>composer"
//! (render-wav '(60 64 67) "out.wav"); -> "wrote out.wav (N frames)"
//! ```
//!
//! Bindings are stateful: a shared `Arc<Mutex<KsRuntime>>` holds the
//! currently-selected voice across `eval` calls so a session reads
//! naturally — `(load-voice 'piano)` followed by `(render-wav ...)`
//! renders through `Engine::Piano` without re-passing the voice.
//!
//! Errors raised by the Rust closures surface as Steel runtime errors
//! (`ErrorKind::Generic`) via the [`KsResult`] wrapper; the REPL prints
//! them and stays alive, mirroring how a Scheme `(error ...)` would
//! behave at the prompt.
//!
//! Native-only — relies on `hound` for the WAV writer and `library_db`
//! (rusqlite) for the catalog query, both gated under the `native`
//! cargo feature alongside the Steel runtime itself.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use steel::rerrs::{ErrorKind, SteelErr};
use steel::rvals::{IntoSteelVal, Result as SteelResult, SteelVal};
use steel::steel_vm::engine::Engine as SteelEngine;
use steel::steel_vm::register_fn::RegisterFn;

use crate::library_db::{LibraryDb, SongFilter, SongSort};
use crate::synth::{make_voice, midi_to_freq, Engine, VoiceImpl};
use crate::voice_lib::discover_plugin_voices;

/// Shared runtime state held by the registered Steel callables.
///
/// Each closure clones an `Arc<Mutex<KsRuntime>>` so they all see —
/// and mutate — the same "current voice", DB path, and sample rate.
struct KsRuntime {
    /// Engine used by `(render-wav ...)`. Defaults to `Piano` so a
    /// freshly-launched REPL renders something sensible without the
    /// caller having to invoke `(load-voice ...)` first.
    current_engine: Engine,
    current_label: String,
    /// SQLite library catalog. Defaults to `bench-out/library.db`,
    /// matching the `keysynth_db` CLI. Overridable at process start
    /// via the `KEYSYNTH_DB` environment variable.
    db_path: PathBuf,
    /// Render sample rate. 48 kHz matches `cp.rs`'s offline render.
    sample_rate: u32,
    /// Per-note window length used by `(render-wav ...)`. 0.5 s gives
    /// each note ~half a second of sustain plus a 0.5 s release tail.
    note_seconds: f32,
}

impl KsRuntime {
    fn new() -> Self {
        let db_path = std::env::var_os("KEYSYNTH_DB")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("bench-out/library.db"));
        Self {
            current_engine: Engine::Piano,
            current_label: "Piano".to_string(),
            db_path,
            sample_rate: 48_000,
            note_seconds: 0.5,
        }
    }
}

/// `Result<T, String>` adapter that maps the `Err` arm onto a Steel
/// `Generic` error so registered Rust closures can surface validation
/// failures to user code without panicking.
pub(crate) struct KsResult<T>(Result<T, String>);

impl<T: IntoSteelVal> IntoSteelVal for KsResult<T> {
    fn into_steelval(self) -> SteelResult<SteelVal> {
        match self.0 {
            Ok(v) => v.into_steelval(),
            Err(msg) => Err(SteelErr::new(ErrorKind::Generic, msg)),
        }
    }
}

const PRELUDE: &str = r#"
;; (query-songs :composer "Bach") sugar around the Rust-side
;; positional `__query-songs`. Three filter slots are supported:
;; :composer, :era, :instrument. Pass `#f` to skip any slot.
(define-syntax query-songs
  (syntax-rules (:composer :era :instrument)
    ((_) (__query-songs #f #f #f))
    ((_ :composer c) (__query-songs c #f #f))
    ((_ :era e) (__query-songs #f e #f))
    ((_ :instrument i) (__query-songs #f #f i))
    ((_ :composer c :era e) (__query-songs c e #f))
    ((_ :composer c :instrument i) (__query-songs c #f i))
    ((_ :era e :instrument i) (__query-songs #f e i))
    ((_ :composer c :era e :instrument i) (__query-songs c e i))))
"#;

/// Register the Phase 2 keysynth API on `engine`. Idempotent — calling
/// it twice would simply reinstall the bindings against a fresh shared
/// runtime, which is a no-op for callers that only ever construct one
/// `scripting::Engine`.
pub(crate) fn register(engine: &mut SteelEngine) -> Result<(), String> {
    let runtime = Arc::new(Mutex::new(KsRuntime::new()));

    // (list-voices) -> list of voice labels
    engine.register_fn("list-voices", || -> Vec<String> { list_voices_impl() });

    // (load-voice 'piano) -> "loaded: Piano"
    {
        let rt = runtime.clone();
        engine.register_fn(
            "load-voice",
            move |name: String| -> KsResult<String> {
                KsResult(load_voice_impl(&rt, &name))
            },
        );
    }

    // (current-voice) -> "Piano" — diagnostic helper, not in the
    // four-function spec but useful for scripted assertions.
    {
        let rt = runtime.clone();
        engine.register_fn("current-voice", move || -> String {
            rt.lock()
                .map(|s| s.current_label.clone())
                .unwrap_or_else(|_| "<poisoned>".into())
        });
    }

    // (__query-songs composer era instrument) — positional Rust seam.
    // The `(query-songs :composer ...)` user-facing form is defined by
    // the syntax-rules macro in PRELUDE.
    {
        let rt = runtime.clone();
        engine.register_fn(
            "__query-songs",
            move |composer: Option<String>,
                  era: Option<String>,
                  instrument: Option<String>|
                  -> KsResult<Vec<String>> {
                KsResult(query_songs_impl(&rt, composer, era, instrument))
            },
        );
    }

    // (render-wav '(60 64 67) "out.wav") -> "wrote out.wav (N frames)"
    {
        let rt = runtime.clone();
        engine.register_fn(
            "render-wav",
            move |notes: Vec<i32>, out_path: String| -> KsResult<String> {
                KsResult(render_wav_impl(&rt, &notes, &out_path))
            },
        );
    }

    engine
        .run(PRELUDE.to_string())
        .map(|_| ())
        .map_err(|e| format!("scripting prelude failed: {e}"))
}

// ─── voice listing & resolution ─────────────────────────────────

fn list_voices_impl() -> Vec<String> {
    let mut names = static_voice_names();
    // Plugin discovery only contributes when the cwd has a
    // `voices_live/` tree (typical for repo work). Failures are
    // already logged inside `discover_plugin_voices`.
    let plugin_root = PathBuf::from("voices_live");
    if plugin_root.is_dir() {
        for slot in discover_plugin_voices(&plugin_root) {
            names.push(slot.label);
        }
    }
    names
}

fn static_voice_names() -> Vec<String> {
    vec![
        "Square".into(),
        "KS".into(),
        "KS Rich".into(),
        "Sub".into(),
        "FM".into(),
        "Piano".into(),
        "Koto".into(),
        "Piano Thick".into(),
        "Piano Lite".into(),
        "Piano 5AM".into(),
        "Piano Modal".into(),
    ]
}

fn load_voice_impl(rt: &Mutex<KsRuntime>, name: &str) -> Result<String, String> {
    let (engine, label) = resolve_voice(name).ok_or_else(|| {
        format!("unknown voice: {name}. Use (list-voices) to see available names.")
    })?;
    let mut state = rt.lock().map_err(|e| format!("runtime mutex: {e}"))?;
    state.current_engine = engine;
    state.current_label = label.clone();
    Ok(format!("loaded: {label}"))
}

/// Map a user-supplied voice name to a static `Engine`. Lookup is
/// case-insensitive and ignores spaces / hyphens / underscores so
/// `(load-voice 'piano-lite)`, `(load-voice "Piano Lite")`, and
/// `(load-voice "PIANOLITE")` all land on the same slot.
fn resolve_voice(name: &str) -> Option<(Engine, String)> {
    let key: String = name
        .trim()
        .chars()
        .filter(|c| !matches!(c, ' ' | '-' | '_'))
        .flat_map(|c| c.to_lowercase())
        .collect();
    let pair = match key.as_str() {
        "square" => (Engine::Square, "Square"),
        "ks" => (Engine::Ks, "KS"),
        "ksrich" => (Engine::KsRich, "KS Rich"),
        "sub" => (Engine::Sub, "Sub"),
        "fm" => (Engine::Fm, "FM"),
        "piano" => (Engine::Piano, "Piano"),
        "koto" => (Engine::Koto, "Koto"),
        "pianothick" => (Engine::PianoThick, "Piano Thick"),
        "pianolite" => (Engine::PianoLite, "Piano Lite"),
        "piano5am" => (Engine::Piano5AM, "Piano 5AM"),
        "pianomodal" => (Engine::PianoModal, "Piano Modal"),
        _ => return None,
    };
    Some((pair.0, pair.1.to_string()))
}

// ─── library DB ─────────────────────────────────────────────────

fn query_songs_impl(
    rt: &Mutex<KsRuntime>,
    composer: Option<String>,
    era: Option<String>,
    instrument: Option<String>,
) -> Result<Vec<String>, String> {
    let db_path = {
        let state = rt.lock().map_err(|e| format!("runtime mutex: {e}"))?;
        state.db_path.clone()
    };
    if !db_path.exists() {
        return Err(format!(
            "library db not found at {} — run `keysynth_db rebuild` or set KEYSYNTH_DB",
            db_path.display()
        ));
    }
    let db = LibraryDb::open(&db_path)
        .map_err(|e| format!("library db open ({}): {e}", db_path.display()))?;
    let filter = SongFilter {
        composer,
        era,
        instrument,
        sort: SongSort::ByComposer,
        ..SongFilter::default()
    };
    let songs = db
        .query_songs(&filter)
        .map_err(|e| format!("query: {e}"))?;
    Ok(songs
        .into_iter()
        .map(|s| format!("{}\t{}\t{}", s.id, s.title, s.composer))
        .collect())
}

// ─── render path ────────────────────────────────────────────────

fn render_wav_impl(
    rt: &Mutex<KsRuntime>,
    notes: &[i32],
    out_path: &str,
) -> Result<String, String> {
    if notes.is_empty() {
        return Err("notes list is empty".into());
    }
    let (engine, sr_hz, note_seconds) = {
        let state = rt.lock().map_err(|e| format!("runtime mutex: {e}"))?;
        (state.current_engine, state.sample_rate, state.note_seconds)
    };
    if matches!(engine, Engine::SfPiano | Engine::SfzPiano | Engine::Live) {
        // SF2/SFZ/Live render through shared resources owned by the
        // GUI / live reloader; the offline `make_voice` path returns a
        // silent placeholder for them. Refuse explicitly so the user
        // doesn't get a 0-byte WAV.
        return Err(format!(
            "engine {engine:?} requires the GUI / live reloader and is not \
             available offline"
        ));
    }

    let sr = sr_hz as f32;
    let per_note_frames = ((sr * note_seconds) as usize).max(1);
    let max_tail_frames = (sr * 0.5) as usize;
    let mut buf: Vec<f32> =
        Vec::with_capacity((per_note_frames + max_tail_frames) * notes.len());

    for &n in notes {
        if !(0..=127).contains(&n) {
            return Err(format!("note {n} outside 0..=127"));
        }
        let freq = midi_to_freq(n as u8);
        let mut voice: Box<dyn VoiceImpl + Send> = make_voice(engine, sr, freq, 100);
        let mut sustain = vec![0.0f32; per_note_frames];
        voice.render_add(&mut sustain);
        buf.extend_from_slice(&sustain);

        voice.trigger_release();
        let chunk = ((sr * 0.05) as usize).max(64);
        let mut tail_chunk = vec![0.0f32; chunk];
        let mut tail_rendered = 0usize;
        while tail_rendered < max_tail_frames && !voice.is_done() {
            for s in tail_chunk.iter_mut() {
                *s = 0.0;
            }
            voice.render_add(&mut tail_chunk);
            buf.extend_from_slice(&tail_chunk);
            tail_rendered += tail_chunk.len();
        }
    }

    // Peak-normalise when the bus exceeds full-scale (mirrors cp.rs).
    let peak = buf.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    if peak > 1.0 {
        let inv = 1.0 / peak;
        for s in buf.iter_mut() {
            *s *= inv;
        }
    }

    let path = Path::new(out_path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
        }
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr_hz,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer =
        hound::WavWriter::create(path, spec).map_err(|e| format!("WAV create: {e}"))?;
    for &s in &buf {
        let v = (s.clamp(-1.0, 1.0) * (i16::MAX as f32 - 1.0)).round() as i16;
        writer
            .write_sample(v)
            .map_err(|e| format!("WAV write_sample: {e}"))?;
    }
    writer
        .finalize()
        .map_err(|e| format!("WAV finalize: {e}"))?;
    Ok(format!("wrote {} ({} frames)", out_path, buf.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_voice_handles_aliases() {
        assert!(matches!(resolve_voice("piano"), Some((Engine::Piano, _))));
        assert!(matches!(
            resolve_voice("Piano Lite"),
            Some((Engine::PianoLite, _))
        ));
        assert!(matches!(
            resolve_voice("piano-lite"),
            Some((Engine::PianoLite, _))
        ));
        assert!(matches!(
            resolve_voice("PIANOLITE"),
            Some((Engine::PianoLite, _))
        ));
        assert!(matches!(resolve_voice("KS Rich"), Some((Engine::KsRich, _))));
        assert!(matches!(resolve_voice("ks_rich"), Some((Engine::KsRich, _))));
        assert!(resolve_voice("does-not-exist").is_none());
    }

    #[test]
    fn static_voice_names_cover_all_offline_engines() {
        // Sanity: every name returned by `list-voices` resolves back to
        // an engine. If a future PR adds a new label without wiring it
        // in `resolve_voice`, this test catches it.
        for n in static_voice_names() {
            assert!(
                resolve_voice(&n).is_some(),
                "label {n:?} returned by list-voices must round-trip through resolve_voice"
            );
        }
    }
}
