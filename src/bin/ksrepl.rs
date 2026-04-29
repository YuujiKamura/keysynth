//! `ksrepl` — Steel (embedded Scheme) REPL for keysynth.
//!
//! Issue #56 Phase 1 entry point. Hosts a bare Steel runtime so the
//! Lisp scripting layer roadmap has a place to land:
//!
//! ```text
//! $ ksrepl
//! ksrepl> (+ 1 2)
//! 3
//! ksrepl> (define greet (lambda (n) (* n 2)))
//! ksrepl> (greet 21)
//! 42
//! ksrepl> :quit
//! ```
//!
//! Phase 1 deliberately exposes nothing keysynth-specific — Phase 2
//! is where `(list-voices)` / `(load-voice 'piano)` / `(render-wav ...)`
//! get registered. This binary just proves the host-language plumbing
//! is in place end-to-end (Cargo dep → library wrapper → CLI).
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

use keysynth::scripting::{Engine, ScriptError};

const PROMPT: &str = "ksrepl> ";
const HELP: &str = r#"ksrepl — keysynth Steel (Scheme) REPL (issue #56 Phase 1)

USAGE:
  ksrepl                       interactive REPL
  ksrepl -e "<src>"            evaluate <src> and exit
  ksrepl --eval "<src>"        same, long form
  ksrepl <file.scm>            load and run <file.scm>
  ksrepl -h | --help           print this help

REPL COMMANDS:
  :quit, :q, :exit             exit the REPL
  :help                        print this help

NOTES:
  Phase 1 hosts only the Steel standard prelude. keysynth Rust API
  (voice construction, render, live params) is registered in Phase 2."#;

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

    eprintln!("ksrepl (Steel — issue #56 Phase 1). Type :help for usage, :quit to exit.");

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
            Ok(out) if out.is_empty() => {}
            Ok(out) => println!("{out}"),
            // Print eval errors but stay in the REPL — that's the whole
            // point of an interactive session.
            Err(ScriptError::Eval(msg)) => eprintln!("error: {msg}"),
            Err(e) => return Err(e.into()),
        }
    }
}
