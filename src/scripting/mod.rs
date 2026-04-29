//! Scripting layer — embedded Steel (Scheme) runtime.
//!
//! Issue #56 Phase 1: stand up a Steel `Engine` so the rest of the
//! roadmap (voice construction DSL, chord progressions, live edit) has a
//! Lisp host to grow on top of. This module deliberately exposes nothing
//! keysynth-specific yet — Phase 2 is where `(list-voices)` /
//! `(load-voice 'piano)` / `(render-wav ...)` get registered as Steel
//! callables. Phase 1 only proves `(+ 1 2)` round-trips.
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
    /// Construct a fresh Steel engine with the standard prelude loaded.
    pub fn new() -> Result<Self, ScriptError> {
        // `Engine::new()` itself is infallible in steel-core 0.8, but we
        // keep `Result` in the signature so Phase 2 (which will register
        // keysynth callables and may legitimately fail) doesn't break the
        // public API.
        Ok(Self {
            inner: SteelEngine::new(),
        })
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
}
