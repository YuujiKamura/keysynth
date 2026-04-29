//! Generic Control Protocol (CP) scaffolding for keysynth GUI binaries.
//!
//! Each GUI binary (jukebox, chord_pad, ksabtest, ksrepl) embeds a
//! `cp-core` JSON-RPC 2.0 server so external tooling can drive
//! verification (load-track, play, stop, set-voice, get-state) without
//! synthesising click input or reading the back-buffer with
//! desk_capture / VLM. Window-driven QA is fragile (font fallback,
//! obscured surfaces, IME state) — CP-driven QA is deterministic.
//!
//! Substrate principle: this module depends only on `cp_core` (which
//! itself is std + serde + serde_json). No additional RPC SDKs, no
//! async runtimes, no third-party schema generators. Every keysynth
//! GUI shares the same wire so verification scripts written against
//! one GUI port over to the next.
//!
//! ## Shared state model
//!
//! cp-core dispatches each request on its own worker thread, so GUI
//! state must reach the handler through an `Arc<Mutex<...>>`. We
//! standardise on two channels:
//!
//! - **Snapshot** (`Arc<Mutex<Option<S>>>`): the egui thread re-publishes
//!   a fresh snapshot at the tail of every `update()`; CP handlers
//!   answer `get_state` reads from this clone. The GUI thread is the
//!   sole writer.
//! - **Command queue** (`Arc<Mutex<Vec<C>>>`): CP handlers push typed
//!   commands; the GUI thread drains and applies them at the head of
//!   each frame. This keeps every actual mutation on the eframe thread
//!   so we don't deadlock against the audio mixer's own `Arc<Mutex>`.
//!
//! ## Endpoint discovery
//!
//! Default bind is `127.0.0.1:0` (OS-picked port). The resolved
//! `SocketAddr` is published two ways so verification scripts can find
//! it without scraping window titles:
//!
//! 1. stdout line: `gui-cp: <app> listening on <addr> (pid <pid>)`.
//! 2. file at `bench-out/cp/<app>-<pid>.endpoint` containing the addr.
//!
//! Override the bind via `--gui-cp-endpoint <host:port>`,
//! `KEYSYNTH_GUI_CP_<APP>` (per-binary), or `KEYSYNTH_GUI_CP` (shared
//! fallback). The host is always rewritten to `127.0.0.1` regardless
//! of what the user passes — local-only by construction.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use cp_core::{CpServer, RpcError, ServerHandle};
use serde::Serialize;
use serde_json::{json, Value};

/// Where per-process endpoint advertisements live. Relative to CWD on
/// purpose: a verification script invoked from the workspace root can
/// glob `bench-out/cp/<app>-*.endpoint` without configuration.
pub const ANNOUNCE_DIR: &str = "bench-out/cp";

/// Resolve the bind endpoint for `app` from CLI / env / default. Tests
/// pass `Some("127.0.0.1:0")` directly to dodge env contamination.
pub fn resolve_endpoint(app: &str, cli_override: Option<&str>) -> String {
    if let Some(s) = cli_override {
        return s.to_string();
    }
    let app_var = format!("KEYSYNTH_GUI_CP_{}", app.to_ascii_uppercase());
    if let Ok(v) = std::env::var(&app_var) {
        if !v.is_empty() {
            return v;
        }
    }
    if let Ok(v) = std::env::var("KEYSYNTH_GUI_CP") {
        if !v.is_empty() {
            return v;
        }
    }
    "127.0.0.1:0".to_string()
}

/// Force loopback regardless of the host the user passed. Mirrors the
/// approach in `keysynth::cp::start` — typoing `0.0.0.0` must not
/// expose the GUI to the LAN.
fn force_loopback(endpoint: &str) -> String {
    match endpoint.rsplit_once(':') {
        Some((_, port)) if !port.is_empty() => format!("127.0.0.1:{port}"),
        _ => "127.0.0.1:0".to_string(),
    }
}

/// Shared snapshot + command bus. `S` is whatever `Serialize` blob the
/// GUI publishes for `get_state`; `C` is the GUI's command enum.
///
/// Cloning a `State` clones the inner `Arc`s, so the egui thread and
/// the CP handler closures share the same buffers.
pub struct State<S, C> {
    snapshot: Arc<Mutex<Option<S>>>,
    commands: Arc<Mutex<Vec<C>>>,
    last_command_unix_ms: Arc<Mutex<u64>>,
}

impl<S, C> Clone for State<S, C> {
    fn clone(&self) -> Self {
        Self {
            snapshot: self.snapshot.clone(),
            commands: self.commands.clone(),
            last_command_unix_ms: self.last_command_unix_ms.clone(),
        }
    }
}

impl<S, C> Default for State<S, C> {
    fn default() -> Self {
        Self {
            snapshot: Arc::new(Mutex::new(None)),
            commands: Arc::new(Mutex::new(Vec::new())),
            last_command_unix_ms: Arc::new(Mutex::new(0)),
        }
    }
}

impl<S: Clone, C> State<S, C> {
    /// Construct an empty state. `Default::default()` works too, but
    /// this reads better at the call site.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the published snapshot. Call once per egui frame from
    /// the GUI thread.
    pub fn publish(&self, snap: S) {
        if let Ok(mut g) = self.snapshot.lock() {
            *g = Some(snap);
        }
    }

    /// Read the latest snapshot. Returns `None` until the GUI has
    /// published its first frame.
    pub fn read_snapshot(&self) -> Option<S> {
        self.snapshot.lock().ok().and_then(|g| g.clone())
    }

    /// Pull every queued command. Call from the egui thread at the
    /// start of `update()`; apply each one before painting.
    pub fn drain_commands(&self) -> Vec<C> {
        match self.commands.lock() {
            Ok(mut g) => std::mem::take(&mut *g),
            Err(_) => Vec::new(),
        }
    }

    /// Push a command from any thread (typically a CP handler).
    pub fn push_command(&self, c: C) {
        if let Ok(mut g) = self.commands.lock() {
            g.push(c);
        }
        if let Ok(mut g) = self.last_command_unix_ms.lock() {
            *g = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
        }
    }

    /// Wall-clock timestamp of the last `push_command` (0 before any).
    /// Useful for verification: a script can call `dispatch` then poll
    /// `get_state` until the frame counter advances past this point.
    pub fn last_command_unix_ms(&self) -> u64 {
        self.last_command_unix_ms.lock().map(|g| *g).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for an embedded GUI CP server. Wraps `cp_core::CpServer`
/// with consumer-friendly defaults: forced loopback bind, automatic
/// `ping` registration, and endpoint announcement (stdout + on-disk
/// file under `bench-out/cp/`).
pub struct Builder {
    app: String,
    requested_endpoint: String,
    inner: CpServer,
}

impl Builder {
    /// Start configuring a CP server for `app` (binary name, used in
    /// announcements and discovery). `endpoint` is the requested bind
    /// address; the host portion is always rewritten to 127.0.0.1.
    pub fn new(app: impl Into<String>, endpoint: &str) -> Self {
        let app = app.into();
        let bind = force_loopback(endpoint);
        Self {
            app,
            requested_endpoint: endpoint.to_string(),
            inner: CpServer::new(bind),
        }
    }

    /// Register a custom JSON-RPC method handler. Same semantics as
    /// `cp_core::CpServer::register` — the closure runs on a per-conn
    /// worker thread, so any captured state must be `Send + Sync`.
    pub fn register<H>(mut self, method: &str, handler: H) -> Self
    where
        H: Fn(Value) -> Result<Value, RpcError> + Send + Sync + 'static,
    {
        self.inner = self.inner.register(method, handler);
        self
    }

    /// Bind, spawn the listener, announce the endpoint, return the
    /// handle. Drops `self`.
    pub fn serve(self) -> std::io::Result<Handle> {
        let app_for_ping = self.app.clone();
        let pid = std::process::id();
        let started_unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let inner = self.inner.register("ping", move |_p| {
            Ok(json!({
                "ok": true,
                "app": app_for_ping,
                "pid": pid,
                "started_unix_ms": started_unix_ms,
                "protocol": "keysynth-gui-cp/1",
            }))
        });
        let inner = inner.serve()?;
        let local_addr = inner.local_addr();
        let announce_path = announce_endpoint(&self.app, pid, &local_addr).ok();
        eprintln!(
            "gui-cp: {} listening on {} (pid {pid}) [requested {}]",
            self.app, local_addr, self.requested_endpoint,
        );
        Ok(Handle {
            app: self.app,
            local_addr,
            announce_path,
            _inner: Arc::new(inner),
        })
    }
}

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

/// Running CP server handle. Drop to stop the listener; the announce
/// file is removed on drop so a discovery scan never sees a stale
/// endpoint pointing at a dead port.
pub struct Handle {
    pub app: String,
    pub local_addr: SocketAddr,
    pub announce_path: Option<PathBuf>,
    /// Kept alive so the cp-core thread doesn't shut down. `Arc` lets
    /// us hand out clones to internal helpers if needed without
    /// breaking the single-shutdown invariant.
    _inner: Arc<ServerHandle>,
}

impl Drop for Handle {
    fn drop(&mut self) {
        if let Some(p) = self.announce_path.as_ref() {
            let _ = std::fs::remove_file(p);
        }
        // `Arc<ServerHandle>` drops cp-core's listener once the last
        // clone is gone; since `_inner` is private we hold the only
        // strong ref here.
    }
}

// ---------------------------------------------------------------------------
// Endpoint announcement / discovery
// ---------------------------------------------------------------------------

fn announce_endpoint(app: &str, pid: u32, addr: &SocketAddr) -> std::io::Result<PathBuf> {
    let dir = Path::new(ANNOUNCE_DIR);
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{app}-{pid}.endpoint"));
    std::fs::write(&path, addr.to_string())?;
    Ok(path)
}

/// Walk `bench-out/cp/` for the most recently-modified
/// `<app>-*.endpoint` file and return the address it contains. Used by
/// verification scripts to find a running GUI without parsing stdout.
pub fn discover_latest(app: &str) -> Option<String> {
    let dir = Path::new(ANNOUNCE_DIR);
    let entries = std::fs::read_dir(dir).ok()?;
    let prefix = format!("{app}-");
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for e in entries.flatten() {
        let name = e.file_name();
        let n = name.to_string_lossy();
        if !n.starts_with(&prefix) || !n.ends_with(".endpoint") {
            continue;
        }
        let meta = match e.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        let modified = match meta.modified() {
            Ok(t) => t,
            Err(_) => continue,
        };
        let take = best
            .as_ref()
            .map(|(prev, _)| *prev < modified)
            .unwrap_or(true);
        if take {
            best = Some((modified, e.path()));
        }
    }
    let (_, path) = best?;
    let s = std::fs::read_to_string(&path).ok()?;
    Some(s.trim().to_string())
}

// ---------------------------------------------------------------------------
// Helpers for handler authors
// ---------------------------------------------------------------------------

/// Decode a handler param value into a typed struct, mapping serde
/// failures to a JSON-RPC `-32602 Invalid Params` error so the caller
/// sees a structured rejection instead of a generic 500.
pub fn decode_params<T: for<'de> serde::Deserialize<'de>>(v: Value) -> Result<T, RpcError> {
    serde_json::from_value::<T>(v).map_err(|e| RpcError::invalid_params(format!("param parse: {e}")))
}

/// Convenience: serialise an arbitrary `Serialize` value to a `Value`,
/// returning `RpcError::internal` if the conversion fails (it should
/// never fail for plain structs but we keep the path explicit).
pub fn encode_result<T: Serialize>(v: T) -> Result<Value, RpcError> {
    serde_json::to_value(v).map_err(|e| RpcError::internal(format!("encode: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cp_core::client as rpc;

    #[test]
    fn force_loopback_rewrites_host() {
        assert_eq!(force_loopback("0.0.0.0:5577"), "127.0.0.1:5577");
        assert_eq!(force_loopback("example.com:1234"), "127.0.0.1:1234");
        assert_eq!(force_loopback("127.0.0.1:0"), "127.0.0.1:0");
        assert_eq!(force_loopback("nonsense"), "127.0.0.1:0");
    }

    #[test]
    fn resolve_endpoint_priority() {
        std::env::remove_var("KEYSYNTH_GUI_CP");
        std::env::remove_var("KEYSYNTH_GUI_CP_TESTAPP");
        assert_eq!(resolve_endpoint("testapp", None), "127.0.0.1:0");
        std::env::set_var("KEYSYNTH_GUI_CP", "127.0.0.1:7777");
        assert_eq!(resolve_endpoint("testapp", None), "127.0.0.1:7777");
        std::env::set_var("KEYSYNTH_GUI_CP_TESTAPP", "127.0.0.1:8888");
        assert_eq!(resolve_endpoint("testapp", None), "127.0.0.1:8888");
        assert_eq!(
            resolve_endpoint("testapp", Some("127.0.0.1:9999")),
            "127.0.0.1:9999"
        );
        std::env::remove_var("KEYSYNTH_GUI_CP");
        std::env::remove_var("KEYSYNTH_GUI_CP_TESTAPP");
    }

    #[test]
    fn state_drain_is_fifo_and_clears() {
        let s: State<(), i32> = State::new();
        s.push_command(1);
        s.push_command(2);
        s.push_command(3);
        let drained = s.drain_commands();
        assert_eq!(drained, vec![1, 2, 3]);
        assert!(s.drain_commands().is_empty());
    }

    #[test]
    fn snapshot_publish_replaces() {
        let s: State<i32, ()> = State::new();
        assert_eq!(s.read_snapshot(), None);
        s.publish(7);
        s.publish(11);
        assert_eq!(s.read_snapshot(), Some(11));
    }

    #[test]
    fn ping_roundtrip_via_builder() {
        let handle = Builder::new("unit-test-app", "127.0.0.1:0")
            .serve()
            .expect("serve");
        let resp = rpc::call(handle.local_addr.to_string(), "ping", json!({})).expect("call");
        assert!(resp.error.is_none(), "{:?}", resp.error);
        let result = resp.result.unwrap();
        assert_eq!(result["ok"], json!(true));
        assert_eq!(result["app"], json!("unit-test-app"));
        assert_eq!(result["protocol"], json!("keysynth-gui-cp/1"));
    }

    #[test]
    fn announce_file_round_trips_for_discovery() {
        // Use a unique app name so other tests' announce files don't
        // confuse `discover_latest`.
        let app = format!("disco-test-{}", std::process::id());
        let handle = Builder::new(&app, "127.0.0.1:0").serve().expect("serve");
        let discovered = discover_latest(&app).expect("discovered");
        assert_eq!(discovered, handle.local_addr.to_string());
        // Drop the handle and re-discover: the announce file must be
        // removed so the discovery doesn't yield a stale port.
        let announce = handle.announce_path.clone();
        drop(handle);
        if let Some(p) = announce {
            assert!(!p.exists(), "announce file lingers after Drop");
        }
    }

    #[test]
    fn custom_handler_dispatches() {
        let handle = Builder::new("dispatch-test", "127.0.0.1:0")
            .register("double", |p| {
                let n = p
                    .get("n")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| RpcError::invalid_params("missing n"))?;
                Ok(json!({ "out": n * 2 }))
            })
            .serve()
            .expect("serve");
        let resp = rpc::call(handle.local_addr.to_string(), "double", json!({"n": 21}))
            .expect("call");
        assert!(resp.error.is_none(), "{:?}", resp.error);
        assert_eq!(resp.result.unwrap()["out"], json!(42));
    }
}
