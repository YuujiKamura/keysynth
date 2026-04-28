//! keysynth's Voice Control Protocol — first consumer of `cp-core`.
//!
//! This file is now thin: cp-core owns the JSON-RPC 2.0 wire, the
//! TCP listener, and the per-connection worker. Keysynth's job is to
//! register method handlers (status / list / set / load / unload /
//! build / render) that close over an `Arc<Reloader>` plus the
//! audio device's sample rate.
//!
//! Default endpoint: `127.0.0.1:5577`. Override via `--cp-endpoint
//! <host:port>` or the `KEYSYNTH_CP` env var. Local-only; the bind
//! address is hard-coded to `127.0.0.1` so we never expose ourselves
//! to the network even if the user typoes `0.0.0.0` (we still bind
//! `127.0.0.1` regardless of the host they pass).
//!
//! See `cp-core/README.md` for the substrate philosophy and
//! `.dispatch/cp-design.md` for keysynth-specific design notes.

use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Instant;

use cp_core::{client as rpc_client, CpServer, RpcError, ServerHandle};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::live_reload::Reloader;
use crate::synth::{midi_to_freq, VoiceImpl};

/// Default endpoint when neither `--cp-endpoint` nor `KEYSYNTH_CP` is
/// set. Picked over a randomised port so a user's `ksctl status` works
/// without ceremony as long as only one keysynth is running.
pub const DEFAULT_ENDPOINT: &str = "127.0.0.1:5577";

/// Resolve the endpoint string from CLI / env / default. Tests pass
/// `"127.0.0.1:0"` to get an OS-picked port and read it back via
/// `Server::local_addr`.
pub fn resolve_endpoint(cli_override: Option<&str>) -> String {
    if let Some(s) = cli_override {
        return s.to_string();
    }
    if let Ok(env) = std::env::var("KEYSYNTH_CP") {
        if !env.is_empty() {
            return env;
        }
    }
    DEFAULT_ENDPOINT.to_string()
}

// ---------------------------------------------------------------------------
// Per-method param types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct LoadParams {
    slot: String,
    dll: String,
}
#[derive(Debug, Clone, Deserialize)]
struct UnloadParams {
    slot: String,
}
#[derive(Debug, Clone, Deserialize)]
struct SetParams {
    slot: String,
}
#[derive(Debug, Clone, Deserialize)]
struct BuildParams {
    slot: String,
    src: String,
}
#[derive(Debug, Clone, Deserialize)]
struct RenderParams {
    notes: Vec<u8>,
    duration: f32,
    #[serde(default = "default_velocity")]
    velocity: u8,
    out: String,
}
fn default_velocity() -> u8 {
    100
}

#[derive(Debug, Clone, Serialize)]
struct SlotInfoWire {
    name: String,
    dll: String,
    loaded_at_unix_ms: u64,
}

// ---------------------------------------------------------------------------
// Server façade — keeps consumers' main.rs trivial.
// ---------------------------------------------------------------------------

/// keysynth's CP server handle. Wraps a `cp_core::ServerHandle` plus
/// the connection-count atomic so the GUI can render it. Drop the
/// last `Server` to stop the listener (cp_core::ServerHandle::Drop
/// signals shutdown).
#[derive(Clone)]
pub struct Server {
    pub endpoint: String,
    pub sr_hz: u32,
    pub connections: Arc<AtomicUsize>,
    inner: Arc<ServerHandle>,
}

impl Server {
    pub fn connection_count(&self) -> usize {
        self.inner.connection_count()
    }
    pub fn is_ready(&self) -> bool {
        // cp-core binds before returning from `serve()`, so once we
        // hold a Server the listener is necessarily up.
        true
    }
}

/// Build the cp-core server with all keysynth method handlers
/// registered, bind it, and return the running handle.
pub fn start(reloader: Reloader, sr_hz: u32, endpoint: &str) -> std::io::Result<Server> {
    // We always bind to 127.0.0.1 regardless of the user's host
    // string, so a typo can't accidentally expose the pipe to the
    // LAN. The port and any explicit "[::1]:N" the user passed are
    // honoured.
    let bind_addr = if let Some(rest) = endpoint.rsplit_once(':') {
        let port = rest.1;
        format!("127.0.0.1:{port}")
    } else {
        // No port → assume default.
        format!("127.0.0.1:5577")
    };

    let r_status = reloader.clone();
    let r_list = reloader.clone();
    let r_set = reloader.clone();
    let r_load = reloader.clone();
    let r_unload = reloader.clone();
    let r_build = reloader.clone();
    let r_render = reloader.clone();

    let handle = CpServer::new(bind_addr)
        .register("status", move |_p| op_status(&r_status, sr_hz))
        .register("list", move |_p| op_list(&r_list))
        .register("set", move |p| op_set(&r_set, p))
        .register("load", move |p| op_load(&r_load, p))
        .register("unload", move |p| op_unload(&r_unload, p))
        .register("build", move |p| op_build(&r_build, p))
        .register("render", move |p| op_render(&r_render, sr_hz, p))
        .serve()?;

    let local_addr = handle.local_addr();
    let connections = handle.connections.clone();
    Ok(Server {
        endpoint: local_addr.to_string(),
        sr_hz,
        connections,
        inner: Arc::new(handle),
    })
}

// ---------------------------------------------------------------------------
// Op implementations — each returns Result<Value, RpcError>.
// ---------------------------------------------------------------------------

fn op_status(reloader: &Reloader, sr_hz: u32) -> Result<Value, RpcError> {
    let n_slots = reloader.list_slots().len();
    let active = reloader.active_slot_name();
    let status = reloader.status_snapshot();
    Ok(json!({
        "active": active,
        "slots": n_slots,
        "sr_hz": sr_hz,
        "status": status.short_label(),
    }))
}

fn op_list(reloader: &Reloader) -> Result<Value, RpcError> {
    let slots: Vec<SlotInfoWire> = reloader
        .list_slots()
        .into_iter()
        .map(|s| SlotInfoWire {
            name: s.name,
            dll: s.dll_path.display().to_string(),
            loaded_at_unix_ms: s.loaded_at_unix_ms as u64,
        })
        .collect();
    Ok(json!({
        "slots": slots,
        "active": reloader.active_slot_name(),
    }))
}

fn op_set(reloader: &Reloader, params: Value) -> Result<Value, RpcError> {
    let p: SetParams = parse_params(params)?;
    reloader
        .set_active(&p.slot)
        .map_err(|e| RpcError::new(cp_core::codes::INVALID_PARAMS, e))?;
    Ok(json!({ "slot": p.slot }))
}

fn op_load(reloader: &Reloader, params: Value) -> Result<Value, RpcError> {
    let p: LoadParams = parse_params(params)?;
    let path = Path::new(&p.dll);
    if !path.exists() {
        return Err(RpcError::invalid_params(format!(
            "dll not found: {}",
            path.display()
        )));
    }
    reloader
        .load_dll_into_slot(path, &p.slot)
        .map_err(|e| RpcError::internal(e))?;
    Ok(json!({ "slot": p.slot }))
}

fn op_unload(reloader: &Reloader, params: Value) -> Result<Value, RpcError> {
    let p: UnloadParams = parse_params(params)?;
    reloader
        .unset_slot(&p.slot)
        .map_err(|e| RpcError::invalid_params(e))?;
    Ok(json!({ "slot": p.slot }))
}

fn op_build(reloader: &Reloader, params: Value) -> Result<Value, RpcError> {
    let p: BuildParams = parse_params(params)?;
    let crate_root = Path::new(&p.src);
    if !crate_root.exists() {
        return Err(RpcError::invalid_params(format!(
            "src dir not found: {}",
            crate_root.display()
        )));
    }
    let started = Instant::now();
    let factory = reloader
        .build_into_slot(crate_root, &p.slot)
        .map_err(|e| RpcError::internal(e))?;
    Ok(json!({
        "slot": p.slot,
        "dll": factory.dll_path.display().to_string(),
        "build_ms": started.elapsed().as_millis() as u64,
    }))
}

fn op_render(reloader: &Reloader, sr_hz: u32, params: Value) -> Result<Value, RpcError> {
    let p: RenderParams = parse_params(params)?;
    if p.notes.is_empty() {
        return Err(RpcError::invalid_params("notes list is empty"));
    }
    if !(0.05..=120.0).contains(&p.duration) {
        return Err(RpcError::invalid_params(format!(
            "duration {} s outside [0.05, 120] s",
            p.duration
        )));
    }
    let sr = sr_hz as f32;
    let per_note_frames = ((p.duration * sr) / p.notes.len() as f32).round() as usize;
    if per_note_frames == 0 {
        return Err(RpcError::invalid_params(
            "per-note window evaluates to 0 frames",
        ));
    }
    let max_tail_frames = (sr * 0.5) as usize;

    let mut buf: Vec<f32> = Vec::with_capacity((per_note_frames + max_tail_frames) * p.notes.len());
    let mut total_frames = 0usize;

    for &note in &p.notes {
        let freq = midi_to_freq(note);
        let mut voice: Box<dyn VoiceImpl + Send> = match reloader.make_voice(sr, freq, p.velocity) {
            Some(v) => v,
            None => {
                return Err(RpcError::internal(format!(
                    "active slot '{}' has no factory loaded",
                    reloader.active_slot_name()
                )));
            }
        };
        let mut sustain = vec![0.0f32; per_note_frames];
        voice.render_add(&mut sustain);
        buf.extend_from_slice(&sustain);
        total_frames += per_note_frames;

        voice.trigger_release();
        let chunk = ((sr * 0.05) as usize).max(64);
        let mut tail_rendered = 0usize;
        let mut tail_chunk = vec![0.0f32; chunk];
        while tail_rendered < max_tail_frames && !voice.is_done() {
            for s in tail_chunk.iter_mut() {
                *s = 0.0;
            }
            voice.render_add(&mut tail_chunk);
            buf.extend_from_slice(&tail_chunk);
            total_frames += tail_chunk.len();
            tail_rendered += tail_chunk.len();
        }
    }

    let peak = buf.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    if peak > 1.0 {
        let inv = 1.0 / peak;
        for s in buf.iter_mut() {
            *s *= inv;
        }
    }

    let out_path = Path::new(&p.out);
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr_hz,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(out_path, spec)
        .map_err(|e| RpcError::internal(format!("WAV writer create: {e}")))?;
    // Hand-rolled SHA-256 fingerprint via the `sha2`-free path: we
    // already pull `serde_json` so a 64-bit FNV-1a is a deterministic,
    // dep-free choice. Tests just need "differs across slots", which
    // FNV satisfies — they don't need cryptographic strength.
    let mut hash = fnv1a64_init();
    for &s in &buf {
        let i = (s.clamp(-1.0, 1.0) * (i16::MAX as f32 - 1.0)).round() as i16;
        writer
            .write_sample(i)
            .map_err(|e| RpcError::internal(format!("WAV write_sample: {e}")))?;
        for byte in i.to_le_bytes() {
            hash = fnv1a64_step(hash, byte);
        }
    }
    writer
        .finalize()
        .map_err(|e| RpcError::internal(format!("WAV finalize: {e}")))?;

    Ok(json!({
        "out": out_path.display().to_string(),
        "frames": total_frames as u64,
        "duration_s": (total_frames as f32) / sr,
        "fingerprint": format!("fnv1a64:{:016x}", hash),
        "active": reloader.active_slot_name(),
    }))
}

fn parse_params<T: for<'de> Deserialize<'de>>(v: Value) -> Result<T, RpcError> {
    serde_json::from_value::<T>(v)
        .map_err(|e| RpcError::invalid_params(format!("param parse: {e}")))
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x100_0000_01b3;
fn fnv1a64_init() -> u64 {
    FNV_OFFSET
}
fn fnv1a64_step(state: u64, byte: u8) -> u64 {
    (state ^ byte as u64).wrapping_mul(FNV_PRIME)
}

// ---------------------------------------------------------------------------
// Re-exports + thin client helper for ksctl
// ---------------------------------------------------------------------------

pub use cp_core::{codes, RpcResponse};

/// One-shot client helper. Resolves the endpoint via the same rules
/// as the server (CLI override → KEYSYNTH_CP env → DEFAULT_ENDPOINT),
/// then runs a single JSON-RPC round-trip.
pub fn one_shot(endpoint: &str, method: &str, params: Value) -> std::io::Result<RpcResponse> {
    rpc_client::call(endpoint, method, params)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a64_is_deterministic_and_position_sensitive() {
        let mut a = fnv1a64_init();
        for &b in b"abc" {
            a = fnv1a64_step(a, b);
        }
        let mut b = fnv1a64_init();
        for &c in b"acb" {
            b = fnv1a64_step(b, c);
        }
        assert_ne!(a, b, "byte order must affect hash");

        let mut c = fnv1a64_init();
        for &c2 in b"abc" {
            c = fnv1a64_step(c, c2);
        }
        assert_eq!(a, c, "hash must be deterministic");
    }

    #[test]
    fn resolve_endpoint_priority() {
        std::env::remove_var("KEYSYNTH_CP");
        assert_eq!(resolve_endpoint(None), DEFAULT_ENDPOINT);
        std::env::set_var("KEYSYNTH_CP", "127.0.0.1:9999");
        assert_eq!(resolve_endpoint(None), "127.0.0.1:9999");
        assert_eq!(resolve_endpoint(Some("127.0.0.1:1234")), "127.0.0.1:1234");
        std::env::remove_var("KEYSYNTH_CP");
    }
}
