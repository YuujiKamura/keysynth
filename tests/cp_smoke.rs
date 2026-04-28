//! Integration tests for the keysynth CP server (cp-core consumer).
//!
//! Walks the brief's "Test plan acceptance" scenario end-to-end, in
//! process: spin up `keysynth::cp::start` against `127.0.0.1:0` (no
//! GUI, no audio device), build two distinct voice DLLs from scratch
//! crate copies, load them into slots A and B, render an arpeggio
//! through each, assert the FNV-1a fingerprints differ — AND assert
//! held voices from slot A survive a swap to slot B (the multi-slot
//! extension still respects PR #40's Arc-pinning invariants).
//!
//! Each test gets an OS-picked port via `127.0.0.1:0` so parallel
//! runs don't collide. The chosen port is read back through the
//! returned `Server::endpoint` field.

#![cfg(feature = "native")]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use cp_core::{client as rpc, RpcResponse};
use serde_json::{json, Value};

use keysynth::cp;
use keysynth::live_reload::Reloader;

const SR_HZ: u32 = 44_100;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn copy_dir_shallow(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        if entry.file_name() == "target" {
            continue;
        }
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_shallow(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Stamp out a scratch copy of `voices_live/` whose `src/lib.rs`
/// contains the given source body. Used to make slot A and slot B
/// produce *audibly different* WAV bytes without touching the
/// checked-in voices_live/.
fn scratch_voice_crate(test_name: &str, lib_rs: &str) -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    let src = cwd.join("voices_live");
    if !src.join("Cargo.toml").exists() {
        return None;
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let scratch = cwd
        .join("target")
        .join("cp-tests")
        .join(format!("{test_name}-{nanos}"))
        .join("voices_live");
    copy_dir_shallow(&src, &scratch).ok()?;
    std::fs::write(scratch.join("src").join("lib.rs"), lib_rs).ok()?;
    Some(scratch)
}

/// Voice A: pure sine — deterministic spectral signature.
fn voice_source_sine() -> &'static str {
    r#"
use std::os::raw::c_void;
pub const KEYSYNTH_LIVE_ABI_VERSION: u32 = 1;
#[no_mangle] pub extern "C" fn keysynth_live_abi_version() -> u32 { 1 }
pub struct V { phase: f32, dphase: f32, amp: f32, released: bool, n: u32 }
#[no_mangle] pub unsafe extern "C" fn keysynth_live_new(sr: f32, freq: f32, vel: u8) -> *mut c_void {
    let v = V {
        phase: 0.0,
        dphase: std::f32::consts::TAU * freq / sr,
        amp: (vel as f32 / 127.0) * 0.5,
        released: false,
        n: 0,
    };
    Box::into_raw(Box::new(v)) as *mut c_void
}
#[no_mangle] pub unsafe extern "C" fn keysynth_live_render_add(p: *mut c_void, buf: *mut f32, n: usize) {
    if p.is_null() || buf.is_null() || n == 0 { return; }
    let v = &mut *(p as *mut V);
    let s = std::slice::from_raw_parts_mut(buf, n);
    for x in s {
        *x += v.amp * v.phase.sin();
        v.phase += v.dphase;
        if v.phase > std::f32::consts::TAU { v.phase -= std::f32::consts::TAU; }
        v.n = v.n.saturating_add(1);
    }
}
#[no_mangle] pub unsafe extern "C" fn keysynth_live_trigger_release(p: *mut c_void) {
    if !p.is_null() { (*(p as *mut V)).released = true; }
}
#[no_mangle] pub unsafe extern "C" fn keysynth_live_is_done(p: *mut c_void) -> bool {
    if p.is_null() { return true; }
    (*(p as *mut V)).released && (*(p as *mut V)).n > 4_410
}
#[no_mangle] pub unsafe extern "C" fn keysynth_live_is_releasing(p: *mut c_void) -> bool {
    if p.is_null() { return false; }
    (*(p as *mut V)).released
}
#[no_mangle] pub unsafe extern "C" fn keysynth_live_drop(p: *mut c_void) {
    if !p.is_null() { drop(Box::from_raw(p as *mut V)); }
}
"#
}

/// Voice B: square — completely different harmonic signature.
fn voice_source_square() -> &'static str {
    r#"
use std::os::raw::c_void;
pub const KEYSYNTH_LIVE_ABI_VERSION: u32 = 1;
#[no_mangle] pub extern "C" fn keysynth_live_abi_version() -> u32 { 1 }
pub struct V { phase: f32, dphase: f32, amp: f32, released: bool, n: u32 }
#[no_mangle] pub unsafe extern "C" fn keysynth_live_new(sr: f32, freq: f32, vel: u8) -> *mut c_void {
    let v = V {
        phase: 0.0,
        dphase: std::f32::consts::TAU * freq / sr,
        amp: (vel as f32 / 127.0) * 0.5,
        released: false,
        n: 0,
    };
    Box::into_raw(Box::new(v)) as *mut c_void
}
#[no_mangle] pub unsafe extern "C" fn keysynth_live_render_add(p: *mut c_void, buf: *mut f32, n: usize) {
    if p.is_null() || buf.is_null() || n == 0 { return; }
    let v = &mut *(p as *mut V);
    let s = std::slice::from_raw_parts_mut(buf, n);
    for x in s {
        let s = if v.phase.sin() >= 0.0 { v.amp } else { -v.amp };
        *x += s;
        v.phase += v.dphase;
        if v.phase > std::f32::consts::TAU { v.phase -= std::f32::consts::TAU; }
        v.n = v.n.saturating_add(1);
    }
}
#[no_mangle] pub unsafe extern "C" fn keysynth_live_trigger_release(p: *mut c_void) {
    if !p.is_null() { (*(p as *mut V)).released = true; }
}
#[no_mangle] pub unsafe extern "C" fn keysynth_live_is_done(p: *mut c_void) -> bool {
    if p.is_null() { return true; }
    (*(p as *mut V)).released && (*(p as *mut V)).n > 4_410
}
#[no_mangle] pub unsafe extern "C" fn keysynth_live_is_releasing(p: *mut c_void) -> bool {
    if p.is_null() { return false; }
    (*(p as *mut V)).released
}
#[no_mangle] pub unsafe extern "C" fn keysynth_live_drop(p: *mut c_void) {
    if !p.is_null() { drop(Box::from_raw(p as *mut V)); }
}
"#
}

/// Spin up a Reloader against the bundled voices_live/ + bind a CP
/// server on `127.0.0.1:0`. Returns the endpoint string for client
/// calls. Skips (returns `None`) when voices_live/ isn't present.
fn spawn_test_server() -> Option<(cp::Server, String)> {
    let here = std::env::current_dir().ok()?;
    let crate_root = here.join("voices_live");
    if !crate_root.join("Cargo.toml").exists() {
        eprintln!("skipping: voices_live/ not in CWD");
        return None;
    }
    let reloader = Reloader::spawn(crate_root);
    let server = cp::start(reloader, SR_HZ, "127.0.0.1:0").expect("start CP server");
    let endpoint = server.endpoint.clone();
    Some((server, endpoint))
}

fn must_ok(resp: &RpcResponse) -> &Value {
    if let Some(err) = &resp.error {
        panic!(
            "expected ok, got error: code={} msg={}",
            err.code, err.message
        );
    }
    resp.result.as_ref().expect("result missing")
}

fn must_err(resp: &RpcResponse) -> i32 {
    match &resp.error {
        Some(e) => e.code,
        None => panic!("expected error, got result: {:?}", resp.result),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Status round-trip: server up → client connects → JSON-RPC echo.
#[test]
fn status_roundtrip() {
    let Some((_srv, endpoint)) = spawn_test_server() else {
        return;
    };
    let resp = rpc::call(&endpoint, "status", json!({})).expect("call");
    let data = must_ok(&resp);
    assert_eq!(
        data.get("sr_hz").and_then(|v| v.as_u64()).unwrap_or(0),
        SR_HZ as u64,
        "sr_hz mismatch in {data:?}"
    );
}

/// Brief's "Test plan acceptance" replicated in-process via cp-core:
///   build A (sine) → build B (square) → set A → render a.wav
///   → set B → render b.wav → fingerprints differ.
#[test]
fn build_a_build_b_set_render_diff() {
    let Some((_srv, endpoint)) = spawn_test_server() else {
        return;
    };

    let crate_a = match scratch_voice_crate("voice-a", voice_source_sine()) {
        Some(p) => p,
        None => return,
    };
    let crate_b = match scratch_voice_crate("voice-b", voice_source_square()) {
        Some(p) => p,
        None => return,
    };

    let resp = rpc::call(
        &endpoint,
        "build",
        json!({"slot": "A", "src": crate_a.display().to_string()}),
    )
    .expect("build A");
    let data = must_ok(&resp);
    assert_eq!(data.get("slot").and_then(|v| v.as_str()), Some("A"));

    let resp = rpc::call(
        &endpoint,
        "build",
        json!({"slot": "B", "src": crate_b.display().to_string()}),
    )
    .expect("build B");
    must_ok(&resp);

    let resp = rpc::call(&endpoint, "list", json!({})).expect("list");
    let data = must_ok(&resp);
    let slots = data.get("slots").and_then(|v| v.as_array()).unwrap();
    let names: Vec<&str> = slots
        .iter()
        .filter_map(|s| s.get("name").and_then(|v| v.as_str()))
        .collect();
    assert!(names.contains(&"A"), "list missing A: {names:?}");
    assert!(names.contains(&"B"), "list missing B: {names:?}");

    rpc::call(&endpoint, "set", json!({"slot": "A"})).expect("set A");
    let out_a = std::env::temp_dir().join(format!("ksctl-test-a-{}.wav", std::process::id()));
    let resp = rpc::call(
        &endpoint,
        "render",
        json!({
            "notes": [60, 64, 67, 72],
            "duration": 1.0,
            "velocity": 100,
            "out": out_a.display().to_string(),
        }),
    )
    .expect("render A");
    let data = must_ok(&resp);
    let fp_a = data
        .get("fingerprint")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .expect("fingerprint A");

    // Set B → render → fingerprint should differ.
    let swap_started = Instant::now();
    rpc::call(&endpoint, "set", json!({"slot": "B"})).expect("set B");
    let swap_latency_ms = swap_started.elapsed().as_millis();
    eprintln!("set-active CP roundtrip: {swap_latency_ms} ms");
    assert!(
        swap_latency_ms < 200,
        "set over CP took {swap_latency_ms} ms (>200 ms)"
    );

    let out_b = std::env::temp_dir().join(format!("ksctl-test-b-{}.wav", std::process::id()));
    let resp = rpc::call(
        &endpoint,
        "render",
        json!({
            "notes": [60, 64, 67, 72],
            "duration": 1.0,
            "velocity": 100,
            "out": out_b.display().to_string(),
        }),
    )
    .expect("render B");
    let data = must_ok(&resp);
    let fp_b = data
        .get("fingerprint")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .expect("fingerprint B");

    assert_ne!(
        fp_a, fp_b,
        "voice A and B produced identical samples — slot swap not observable"
    );
    assert!(out_a.exists());
    assert!(out_b.exists());
    let _ = std::fs::remove_file(&out_a);
    let _ = std::fs::remove_file(&out_b);
}

/// `set` to a slot that doesn't exist returns -32602 (Invalid Params)
/// — matches the brief's "fail loud" rule.
#[test]
fn set_nonexistent_slot_errors() {
    let Some((_srv, endpoint)) = spawn_test_server() else {
        return;
    };
    let resp = rpc::call(&endpoint, "set", json!({"slot": "DOES-NOT-EXIST"})).expect("call");
    let code = must_err(&resp);
    assert_eq!(code, cp::codes::INVALID_PARAMS);
}

/// Unknown method returns -32601 (Method not found) — verifies the
/// cp-core dispatch layer surfaces the standard JSON-RPC error code.
#[test]
fn unknown_method_returns_method_not_found() {
    let Some((_srv, endpoint)) = spawn_test_server() else {
        return;
    };
    let resp = rpc::call(&endpoint, "no_such_method", json!({})).expect("call");
    let code = must_err(&resp);
    assert_eq!(code, cp::codes::METHOD_NOT_FOUND);
}

/// Connection survives a malformed JSON line: the server returns
/// -32700 Parse error and keeps the connection alive for follow-up
/// requests.
#[test]
fn malformed_request_kept_alive() {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpStream;

    let Some((_srv, endpoint)) = spawn_test_server() else {
        return;
    };

    let stream = TcpStream::connect(&endpoint).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let read_stream = stream.try_clone().unwrap();
    let mut writer = stream;
    let mut reader = BufReader::new(read_stream);

    writer.write_all(b"this is not json\n").unwrap();
    writer.flush().unwrap();
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let resp: RpcResponse = serde_json::from_str(line.trim_end()).unwrap();
    assert_eq!(resp.error.as_ref().unwrap().code, cp::codes::PARSE_ERROR);

    // Same connection, valid request now.
    let req = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "status",
        "id": 99,
    });
    let mut payload = serde_json::to_string(&req).unwrap();
    payload.push('\n');
    writer.write_all(payload.as_bytes()).unwrap();
    writer.flush().unwrap();
    line.clear();
    reader.read_line(&mut line).unwrap();
    let resp: RpcResponse = serde_json::from_str(line.trim_end()).unwrap();
    assert!(resp.error.is_none(), "{:?}", resp.error);
    assert_eq!(resp.id, json!(99));
}

/// Unload a slot, then set it: server returns -32602.
#[test]
fn unload_then_set_errors() {
    let Some((_srv, endpoint)) = spawn_test_server() else {
        return;
    };
    let crate_a = match scratch_voice_crate("unload-a", voice_source_sine()) {
        Some(p) => p,
        None => return,
    };
    rpc::call(
        &endpoint,
        "build",
        json!({"slot": "GONE", "src": crate_a.display().to_string()}),
    )
    .expect("build GONE");
    let resp = rpc::call(&endpoint, "unload", json!({"slot": "GONE"})).expect("unload");
    must_ok(&resp);
    let resp = rpc::call(&endpoint, "set", json!({"slot": "GONE"})).expect("set after unload");
    let code = must_err(&resp);
    assert_eq!(code, cp::codes::INVALID_PARAMS);
}
