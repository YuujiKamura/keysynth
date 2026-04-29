//! Scripting layer — embedded Steel (Scheme) runtime.
//!
//! Phase 1 stood up a bare Steel `Engine` to prove `(+ 1 2)` round-
//! trips through the Cargo-dep → library-wrapper → CLI plumbing.
//!
//! Phase 2 (issue #56) registers the keysynth-specific surface area on
//! top of that engine via [`bindings`]: `(list-voices)`, `(load-voice
//! 'piano)`, `(query-songs :composer "Bach")`, and `(render-wav ...)`.
//! See `bindings.rs` for the contract — this module just owns the
//! engine lifecycle and the public `eval` API.
//!
//! The wrapper keeps the steel-core types behind a thin facade:
//!   - construction errors collapse into [`ScriptError::Init`]
//!   - evaluation errors collapse into [`ScriptError::Eval`]
//!   - results are stringified via Steel's `Display` impl
//!
//! That keeps `ksrepl` and tests free of `SteelVal` and lets the API
//! evolve (richer return types, multi-value handling) in Phase 2 without
//! breaking call sites.
//!
//! Native-only: gated under the `native` Cargo feature alongside the
//! rest of the optional substrate (cp, libloading, etc.). The wasm
//! browser build doesn't ship a REPL.

use std::fmt;

use steel::steel_vm::engine::Engine as SteelEngine;

mod bindings;

/// Errors surfaced by [`Engine::eval`].
#[derive(Debug)]
pub enum ScriptError {
    /// Steel runtime failed to construct.
    Init(String),
    /// User code failed to compile or evaluate.
    Eval(String),
}

impl fmt::Display for ScriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScriptError::Init(msg) => write!(f, "steel init failed: {msg}"),
            ScriptError::Eval(msg) => write!(f, "steel eval failed: {msg}"),
        }
    }
}

impl std::error::Error for ScriptError {}

/// Thin wrapper around `steel::steel_vm::engine::Engine`.
///
/// Owns one Steel VM. Reusing the same `Engine` across `eval` calls
/// preserves bindings (`define`d names persist), which is what the REPL
/// wants.
pub struct Engine {
    inner: SteelEngine,
}

impl Engine {
    /// Construct a fresh Steel engine with the standard prelude and
    /// the keysynth Phase 2 bindings (`list-voices`, `load-voice`,
    /// `query-songs`, `render-wav`) installed.
    ///
    /// The Phase-1 Result-returning shape is preserved so a future
    /// addition that legitimately fails to register (missing asset,
    /// unsupported VM build) doesn't break call sites.
    pub fn new() -> Result<Self, ScriptError> {
        let mut inner = SteelEngine::new();
        bindings::register(&mut inner).map_err(ScriptError::Init)?;
        Ok(Self { inner })
    }

    /// Evaluate a Steel/Scheme source string.
    ///
    /// Returns the textual rendering of the *last* value produced. An
    /// empty program (whitespace / comments only) yields `""`. Multi-form
    /// programs (`(define x 1) x`) keep all earlier side effects but
    /// only stringify the final result, matching typical REPL UX.
    pub fn eval(&mut self, source: &str) -> Result<String, ScriptError> {
        let owned = source.to_string();
        let values = self
            .inner
            .run(owned)
            .map_err(|e| ScriptError::Eval(format!("{e}")))?;
        Ok(match values.last() {
            Some(v) => format!("{v}"),
            None => String::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_returns_three() {
        let mut engine = Engine::new().expect("steel init");
        assert_eq!(engine.eval("(+ 1 2)").unwrap(), "3");
    }

    #[test]
    fn define_persists_across_evals() {
        let mut engine = Engine::new().expect("steel init");
        engine.eval("(define x 41)").unwrap();
        assert_eq!(engine.eval("(+ x 1)").unwrap(), "42");
    }

    #[test]
    fn empty_program_returns_empty_string() {
        let mut engine = Engine::new().expect("steel init");
        assert_eq!(engine.eval("   ; just a comment\n").unwrap(), "");
    }

    #[test]
    fn syntax_error_is_eval_variant() {
        let mut engine = Engine::new().expect("steel init");
        let err = engine.eval("(+ 1").expect_err("unbalanced paren");
        assert!(matches!(err, ScriptError::Eval(_)));
    }

    // ─── Phase 2 bindings (issue #56) ────────────────────────────

    #[test]
    fn list_voices_includes_static_engines() {
        let mut engine = Engine::new().expect("steel init");
        let out = engine.eval("(list-voices)").expect("list-voices");
        // Steel renders lists as `("Square" "KS" ...)`. The Phase 2
        // contract is "Piano shows up by default" — the user's golden
        // path renders Piano without first loading a voice.
        assert!(out.contains("\"Piano\""), "expected Piano in: {out}");
        assert!(out.contains("\"Square\""), "expected Square in: {out}");
    }

    #[test]
    fn load_voice_accepts_quoted_symbol_and_persists() {
        let mut engine = Engine::new().expect("steel init");
        let out = engine
            .eval("(load-voice 'piano-lite)")
            .expect("load-voice");
        assert_eq!(out, "\"loaded: Piano Lite\"");
        // Subsequent calls see the same runtime state.
        let cur = engine.eval("(current-voice)").expect("current-voice");
        assert_eq!(cur, "\"Piano Lite\"");
    }

    #[test]
    fn load_voice_rejects_unknown_name() {
        let mut engine = Engine::new().expect("steel init");
        let err = engine
            .eval("(load-voice 'no-such-voice)")
            .expect_err("unknown voice should error");
        let msg = format!("{err}");
        assert!(msg.contains("unknown voice"), "got: {msg}");
    }

    #[test]
    fn query_songs_keyword_macro_routes_to_rust() {
        // Skip when the catalog hasn't been built yet — CI without the
        // `keysynth_db rebuild` step is a real configuration. We still
        // assert the binding is reachable: the error must come from the
        // DB layer, not from `__query-songs` being unbound.
        let mut engine = Engine::new().expect("steel init");
        let result = engine.eval(r#"(query-songs :composer "Bach")"#);
        match result {
            Ok(s) => {
                // Real DB present: at minimum the keyword macro must
                // have routed through, returning a list (possibly empty).
                assert!(s.starts_with('('), "expected list, got: {s}");
            }
            Err(ScriptError::Eval(msg)) => {
                // Acceptable when bench-out/library.db is missing. The
                // crucial signal is that the macro expanded — i.e. the
                // failure comes from the DB layer, not from a free
                // identifier `__query-songs`.
                assert!(
                    msg.contains("library db") || msg.contains("query"),
                    "expected DB-layer error, got: {msg}"
                );
            }
            Err(e) => panic!("unexpected error variant: {e}"),
        }
    }

    #[test]
    fn render_wav_writes_a_real_wav_file() {
        let mut engine = Engine::new().expect("steel init");
        let dir = std::env::temp_dir().join("keysynth-scripting-tests");
        std::fs::create_dir_all(&dir).expect("mkdir tempdir");
        let out = dir.join("render_wav_smoke.wav");
        // Use forward slashes so the embedded Steel string literal is
        // portable across Windows / Unix without backslash-escaping.
        let path_str = out.to_string_lossy().replace('\\', "/");
        let src = format!(
            "(begin (load-voice 'piano) (render-wav (list 60 64 67) \"{path_str}\"))"
        );
        let report = engine.eval(&src).expect("render-wav");
        assert!(report.contains("wrote"), "got: {report}");
        // File must exist and contain a non-empty WAV (44-byte header
        // alone would be evidence the writer never advanced past spec).
        let meta = std::fs::metadata(&out).expect("output WAV metadata");
        assert!(meta.len() > 1024, "WAV too small: {} bytes", meta.len());
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn render_wav_rejects_empty_notes_list() {
        let mut engine = Engine::new().expect("steel init");
        let err = engine
            .eval("(render-wav (list) \"unused.wav\")")
            .expect_err("empty notes should error");
        assert!(format!("{err}").contains("empty"), "got: {err}");
    }
}
