//! `ksctl` — Voice Control Protocol CLI for keysynth.
//!
//! First consumer of the workspace-local `cp-core` JSON-RPC 2.0
//! substrate. Connects to a running `keysynth --cp` over TCP localhost,
//! sends one JSON-RPC call per invocation, prints the result.
//!
//! A consolidated `cpctl` that can drive any cp-core server is planned
//! for a later PR. This binary intentionally stays keysynth-shaped
//! (the subcommands map 1:1 to keysynth's method names) so the CLI
//! ergonomics aren't blocked on the substrate-CLI design discussion.
//!
//! Connection target priority:
//!   1. `--endpoint <host:port>`
//!   2. `KEYSYNTH_CP` env
//!   3. `127.0.0.1:5577`
//!
//! Examples:
//!
//! ```text
//! ksctl status
//! ksctl list
//! ksctl build --slot A --src voices_live
//! ksctl set A
//! ksctl render --notes 60,64,67,72 --duration 6 --out arp.wav
//! ```

use std::process::ExitCode;

use cp_core::{client as rpc, RpcResponse};
use serde_json::{json, Value};

use keysynth::cp::resolve_endpoint;

fn print_help() {
    println!(
        "{}",
        r#"ksctl — keysynth Control Protocol client (cp-core JSON-RPC 2.0)

USAGE:
  ksctl [GLOBAL OPTIONS] <COMMAND> [ARGS...]

GLOBAL OPTIONS:
  --endpoint <host:port>   Override the CP endpoint (default: 127.0.0.1:5577)
  --json                   Print raw JSON-RPC response instead of friendly output
  -h, --help               Show this help

COMMANDS:
  status                                          Show server status
  list                                            List loaded slots + active
  set <slot>                                      Switch active slot
  load   --slot <name> --dll <path>               Load a pre-built dll into slot
  unload --slot <name>                            Drop a slot
  build  --slot <name> --src <crate-dir>          Build a sibling cargo crate, load into slot
  render --notes 60,64,67,72 [--duration 6.0]
         [--velocity 100] --out <path.wav>        Render arpeggio through active slot

EXAMPLES:
  ksctl status
  ksctl list
  ksctl build --slot A --src voices_live
  ksctl set A
  ksctl render --notes 60,64,67,72 --out arp.wav
"#
    );
}

#[derive(Debug, Default)]
struct Cli {
    endpoint: Option<String>,
    json: bool,
    command: Option<String>,
    rest: Vec<String>,
}

impl Cli {
    fn parse() -> Result<Self, String> {
        let mut out = Cli::default();
        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                "--endpoint" | "--socket" => {
                    out.endpoint = Some(iter.next().ok_or("--endpoint needs a host:port")?);
                }
                "--json" => out.json = true,
                _ if out.command.is_none() => out.command = Some(arg),
                _ => out.rest.push(arg),
            }
        }
        Ok(out)
    }
}

fn parse_kv(rest: &[String]) -> Result<std::collections::HashMap<String, String>, String> {
    let mut map = std::collections::HashMap::new();
    let mut i = 0;
    while i < rest.len() {
        let key = &rest[i];
        if !key.starts_with("--") {
            return Err(format!("unexpected positional argument '{key}'"));
        }
        let val = rest
            .get(i + 1)
            .ok_or_else(|| format!("{key} needs a value"))?;
        if val.starts_with("--") {
            return Err(format!("{key} needs a value (got another flag)"));
        }
        map.insert(key.trim_start_matches("--").to_string(), val.clone());
        i += 2;
    }
    Ok(map)
}

fn main() -> ExitCode {
    let cli = match Cli::parse() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ksctl: {e}");
            print_help();
            return ExitCode::from(2);
        }
    };

    let Some(cmd) = cli.command.clone() else {
        print_help();
        return ExitCode::from(2);
    };

    let endpoint = resolve_endpoint(cli.endpoint.as_deref());

    let (method, params): (&str, Value) = match cmd.as_str() {
        "status" => ("status", json!({})),
        "list" => ("list", json!({})),
        "set" => match cli.rest.first() {
            Some(slot) => ("set", json!({ "slot": slot })),
            None => {
                eprintln!("ksctl set: missing <slot>");
                return ExitCode::from(2);
            }
        },
        "load" => {
            let kv = match parse_kv(&cli.rest) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("ksctl load: {e}");
                    return ExitCode::from(2);
                }
            };
            let slot = require(&kv, "slot", "load");
            let dll = require(&kv, "dll", "load");
            ("load", json!({ "slot": slot, "dll": dll }))
        }
        "unload" => {
            let kv = match parse_kv(&cli.rest) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("ksctl unload: {e}");
                    return ExitCode::from(2);
                }
            };
            let slot = require(&kv, "slot", "unload");
            ("unload", json!({ "slot": slot }))
        }
        "build" => {
            let kv = match parse_kv(&cli.rest) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("ksctl build: {e}");
                    return ExitCode::from(2);
                }
            };
            let slot = require(&kv, "slot", "build");
            let src = require(&kv, "src", "build");
            ("build", json!({ "slot": slot, "src": src }))
        }
        "render" => {
            let kv = match parse_kv(&cli.rest) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("ksctl render: {e}");
                    return ExitCode::from(2);
                }
            };
            let notes_s = require(&kv, "notes", "render");
            let notes: Vec<u8> = match notes_s
                .split(',')
                .map(|s| s.trim().parse::<u8>())
                .collect::<Result<Vec<_>, _>>()
            {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("ksctl render: --notes parse: {e}");
                    return ExitCode::from(2);
                }
            };
            let out = require(&kv, "out", "render");
            let duration = kv
                .get("duration")
                .map(|s| s.parse::<f32>().unwrap_or(6.0))
                .unwrap_or(6.0);
            let velocity = kv
                .get("velocity")
                .map(|s| s.parse::<u8>().unwrap_or(100))
                .unwrap_or(100);
            (
                "render",
                json!({
                    "notes": notes,
                    "duration": duration,
                    "velocity": velocity,
                    "out": out,
                }),
            )
        }
        other => {
            eprintln!("ksctl: unknown command '{other}'");
            print_help();
            return ExitCode::from(2);
        }
    };

    let resp = match rpc::call(&endpoint, method, params) {
        Ok(r) => r,
        Err(e) => {
            eprintln!(
                "ksctl: connect to {endpoint} failed: {e}\n\
                 (is keysynth running with --cp?)"
            );
            return ExitCode::from(1);
        }
    };

    if cli.json {
        let raw = serde_json::to_string(&resp).unwrap_or_else(|_| "{}".into());
        println!("{raw}");
        return if resp.error.is_some() {
            ExitCode::from(1)
        } else {
            ExitCode::SUCCESS
        };
    }

    print_pretty(&cmd, &resp)
}

/// Pull a `--key VALUE` from a parsed kv map or print a friendly
/// "missing --key" message and exit 2. Centralised here because every
/// subcommand's required flags share the same failure path.
fn require(kv: &std::collections::HashMap<String, String>, key: &str, cmd: &str) -> String {
    match kv.get(key) {
        Some(v) => v.clone(),
        None => {
            eprintln!("ksctl {cmd}: missing --{key}");
            std::process::exit(2);
        }
    }
}

fn print_pretty(cmd: &str, resp: &RpcResponse) -> ExitCode {
    if let Some(err) = &resp.error {
        eprintln!("ksctl: server error (code {}): {}", err.code, err.message);
        return ExitCode::from(1);
    }
    let data = resp.result.as_ref().cloned().unwrap_or(Value::Null);
    match cmd {
        "list" => {
            let active = data.get("active").and_then(|v| v.as_str()).unwrap_or("");
            let empty = Vec::new();
            let slots = data
                .get("slots")
                .and_then(|v| v.as_array())
                .unwrap_or(&empty);
            if slots.is_empty() {
                println!("(no slots loaded; active = {active})");
                return ExitCode::SUCCESS;
            }
            println!("{:<10} {:<6} dll", "slot", "active");
            for s in slots {
                let name = s.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                let dll = s.get("dll").and_then(|v| v.as_str()).unwrap_or("");
                let marker = if name == active { "*" } else { " " };
                println!("{name:<10} {marker:<6} {dll}");
            }
        }
        "status" => {
            let active = data.get("active").and_then(|v| v.as_str()).unwrap_or("");
            let n_slots = data.get("slots").and_then(|v| v.as_u64()).unwrap_or(0);
            let sr = data.get("sr_hz").and_then(|v| v.as_u64()).unwrap_or(0);
            let st = data.get("status").and_then(|v| v.as_str()).unwrap_or("");
            println!("active = {active}");
            println!("slots  = {n_slots}");
            println!("sr_hz  = {sr}");
            println!("status = {st}");
        }
        "render" => {
            let out = data.get("out").and_then(|v| v.as_str()).unwrap_or("");
            let frames = data.get("frames").and_then(|v| v.as_u64()).unwrap_or(0);
            let dur = data
                .get("duration_s")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let fp = data
                .get("fingerprint")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let active = data.get("active").and_then(|v| v.as_str()).unwrap_or("");
            println!("rendered {frames} frames ({dur:.2} s) → {out}");
            println!("active slot:  {active}");
            println!("fingerprint:  {fp}");
        }
        "build" => {
            let slot = data.get("slot").and_then(|v| v.as_str()).unwrap_or("");
            let dll = data.get("dll").and_then(|v| v.as_str()).unwrap_or("");
            let ms = data.get("build_ms").and_then(|v| v.as_u64()).unwrap_or(0);
            println!("slot {slot}: built + loaded in {ms} ms");
            println!("dll: {dll}");
        }
        "set" | "load" | "unload" => {
            let slot = data.get("slot").and_then(|v| v.as_str()).unwrap_or("");
            println!("slot {slot}: ok");
        }
        _ => {
            let pretty = serde_json::to_string_pretty(&data).unwrap_or_default();
            println!("{pretty}");
        }
    }
    ExitCode::SUCCESS
}
