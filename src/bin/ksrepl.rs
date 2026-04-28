use keysynth::scripting::Engine;
use std::io::{self, IsTerminal, Write};
use std::process;

fn main() {
    let mut engine = Engine::new();
    let mut buffer = String::new();
    let stdin = io::stdin();
    let mut has_error = false;

    loop {
        if stdin.is_terminal() {
            print!("> ");
            let _ = io::stdout().flush();
        }

        buffer.clear();
        match stdin.read_line(&mut buffer) {
            Ok(0) => break, // EOF
            Ok(_) => {
                let line = buffer.trim();
                if line == "(exit)" {
                    break;
                }
                if line.is_empty() {
                    continue;
                }
                match engine.eval(line) {
                    Ok(res) => {
                        if !res.is_empty() {
                            println!("{}", res);
                        }
                    }
                    Err(e) => {
                        eprintln!("\x1b[31mError: {}\x1b[0m", e);
                        has_error = true;
                    }
                }
            }
            Err(_) => break,
        }
    }

    if has_error {
        process::exit(1);
    }
}
