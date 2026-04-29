//! `ksrepl` — Steel (embedded Scheme) REPL for keysynth.
//!
//! Issue #56 Phase 2 entry point. Hosts a Steel runtime preloaded with
//! the keysynth API surface so AI agents (or any human at the prompt)
//! can drive voice selection, library queries, and offline render from
//! Lisp:
//!
//! ```text
//! $ ksrepl
//! ksrepl> (list-voices)
//! ("Square" "KS" "KS Rich" ... "Piano Modal")
//! ksrepl> (load-voice 'piano)
//! "loaded: Piano"
//! ksrepl> (query-songs :composer "Bach")
//! ("bach_bwv999_prelude\tJ.S. Bach — BWV 999 ..." ...)
//! ksrepl> (render-wav (list 60 64 67) "out.wav")
//! "wrote out.wav (136800 frames)"
//! ksrepl> :quit
//! ```
//!
//! See `keysynth::scripting::bindings` for the full contract. Phase 1
//! kept this file deliberately bare; Phase 2 layers the four-function
//! agent surface (`list-voices` / `load-voice` / `query-songs` /
//! `render-wav`) on top of the same `scripting::Engine`.
//!
//! Modes:
//!   * `ksrepl`                — interactive REPL on stdin/stdout
//!   * `ksrepl -e "<src>"`     — evaluate one expression and exit
//!   * `ksrepl --eval "<src>"` — same, long form
//!   * `ksrepl -- <file.scm>`  — load + run a script (minimal Phase 1)
//!
//! Bindings persist within a session: each `eval` reuses the same
//! `scripting::Engine` so `(define x 10)` followed by `x` returns `10`.

use std::io::{self, BufRead, Write};
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use keysynth::gui_cp;
use keysynth::scripting::{Engine, ScriptError};
use serde_json::json;

const PROMPT: &str = "ksrepl> ";
const HELP: &str = r#"ksrepl — keysynth Steel (Scheme) REPL (issue #56 Phase 2)

USAGE:
  ksrepl                       interactive REPL
  ksrepl -e "<src>"            evaluate <src> and exit
  ksrepl --eval "<src>"        same, long form
  ksrepl <file.scm>            load and run <file.scm>
  ksrepl -h | --help           print this help

REPL COMMANDS:
  :quit, :q, :exit             exit the REPL
  :help                        print this help

KEYSYNTH API (Phase 2):
  (list-voices)                       list available voice labels
  (load-voice 'piano)                 select voice for subsequent renders
  (current-voice)                     report the currently-selected voice
  (query-songs :composer "Bach")      search the library catalog
  (query-songs :era "baroque" :instrument "guitar")
  (render-wav '(60 64 67) "out.wav")  render notes to a 16-bit mono WAV

ENVIRONMENT:
  KEYSYNTH_DB                  override library.db path
                               (default: bench-out/library.db)"#;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ksrepl: {e}");
            ExitCode::from(1)
        }
    }
}

fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{HELP}");
        return Ok(());
    }

    let mut engine = Engine::new()?;

    // One-shot eval modes.
    if let Some(idx) = args.iter().position(|a| a == "-e" || a == "--eval") {
        let src = args
            .get(idx + 1)
            .ok_or("`-e/--eval` requires an argument")?;
        let out = engine.eval(src)?;
        if !out.is_empty() {
            println!("{out}");
        }
        return Ok(());
    }

    // Single positional argument → treat as a script path.
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
    if let Some(path) = positional.first() {
        let src = std::fs::read_to_string(path)?;
        let out = engine.eval(&src)?;
        if !out.is_empty() {
            println!("{out}");
        }
        return Ok(());
    }

    interactive(engine)
}

fn interactive(mut engine: Engine) -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut line = String::new();

    eprintln!(
        "ksrepl (Steel — issue #56 Phase 2). Type :help for the keysynth API, :quit to exit."
    );

    // Embed a CP server so verification scripts can confirm the REPL
    // is alive and observe how many expressions have been evaluated.
    // Steel's `Engine` isn't `Send + Sync`, so we don't expose `eval`
    // over CP — the REPL stdin loop stays the only mutator. The
    // server is dropped automatically when this function returns.
    let eval_count = Arc::new(AtomicU64::new(0));
    let _cp = match spawn_ksrepl_cp(eval_count.clone()) {
        Ok(h) => Some(h),
        Err(e) => {
            eprintln!("ksrepl: CP server failed to start: {e}");
            None
        }
    };

    loop {
        write!(stdout, "{PROMPT}")?;
        stdout.flush()?;

        line.clear();
        let n = stdin.lock().read_line(&mut line)?;
        if n == 0 {
            // EOF (Ctrl-D / piped input drained) — newline for clean prompt.
            writeln!(stdout)?;
            return Ok(());
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match trimmed {
            ":quit" | ":q" | ":exit" => return Ok(()),
            ":help" => {
                println!("{HELP}");
                continue;
            }
            _ => {}
        }

        match engine.eval(trimmed) {
            Ok(out) if out.is_empty() => {
                eval_count.fetch_add(1, Ordering::Relaxed);
            }
            Ok(out) => {
                eval_count.fetch_add(1, Ordering::Relaxed);
                println!("{out}");
            }
            // Print eval errors but stay in the REPL — that's the whole
            // point of an interactive session.
            Err(ScriptError::Eval(msg)) => eprintln!("error: {msg}"),
            Err(e) => return Err(e.into()),
        }
    }
}

/// Spawn the ksrepl CP server. Read-only: `get_state` returns the
/// REPL's running flag and the number of successful evaluations so a
/// verifier can confirm the process is alive without injecting input.
fn spawn_ksrepl_cp(eval_count: Arc<AtomicU64>) -> std::io::Result<gui_cp::Handle> {
    let endpoint = gui_cp::resolve_endpoint("ksrepl", None);
    gui_cp::Builder::new("ksrepl", &endpoint)
        .register("get_state", move |_p| {
            Ok(json!({
                "running": true,
                "eval_count": eval_count.load(Ordering::Relaxed),
                "phase": "interactive",
            }))
        })
        .serve()
}
