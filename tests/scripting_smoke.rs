//! Integration smoke tests for the Steel scripting layer (issue #56 Phase 1).
//!
//! These ride alongside the in-module unit tests in `src/scripting/mod.rs`
//! and exercise the `keysynth::scripting::Engine` wrapper as an external
//! consumer would — through the public crate surface, without poking at
//! steel-core types directly.
//!
//! Phase 1 asserts only that the host language plumbing works:
//! arithmetic, top-level `define`, lambdas, list ops, and that bindings
//! survive across multiple `eval` calls on the same engine. Phase 2 will
//! add tests for keysynth-specific Steel callables once they're
//! registered.

#![cfg(feature = "native")]

use keysynth::scripting::{Engine, ScriptError};

#[test]
fn helloworld_arithmetic() {
    let mut engine = Engine::new().expect("steel init");
    assert_eq!(engine.eval("(+ 1 2)").unwrap(), "3");
}

#[test]
fn lambda_closure_evaluates() {
    let mut engine = Engine::new().expect("steel init");
    let out = engine
        .eval("((lambda (x y) (+ (* x x) (* y y))) 3 4)")
        .unwrap();
    assert_eq!(out, "25");
}

#[test]
fn defined_function_persists() {
    let mut engine = Engine::new().expect("steel init");
    engine
        .eval("(define double (lambda (n) (* n 2)))")
        .unwrap();
    assert_eq!(engine.eval("(double 21)").unwrap(), "42");
    // Same engine, third call: the binding survives.
    assert_eq!(engine.eval("(double (double 5))").unwrap(), "20");
}

#[test]
fn list_ops_round_trip() {
    let mut engine = Engine::new().expect("steel init");
    // The wrapper returns Steel's textual `Display` form; we only assert
    // the numeric reduction so we're not coupled to list-print formatting.
    let out = engine
        .eval("(apply + (list 10 20 30 40))")
        .unwrap();
    assert_eq!(out, "100");
}

#[test]
fn syntax_error_surfaces_as_eval_error() {
    let mut engine = Engine::new().expect("steel init");
    let err = engine.eval("(((").expect_err("bad syntax must error");
    assert!(
        matches!(err, ScriptError::Eval(_)),
        "expected ScriptError::Eval, got {err:?}"
    );
}

#[test]
fn engine_recovers_after_error() {
    // After a failed eval the runtime must still be usable — that's the
    // contract a REPL needs to keep running across user typos.
    let mut engine = Engine::new().expect("steel init");
    let _ = engine.eval("(undefined-symbol)");
    assert_eq!(engine.eval("(+ 1 1)").unwrap(), "2");
}
