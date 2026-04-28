//! Vendor-neutral JSON-RPC 2.0 substrate over TCP localhost.
//!
//! See `README.md` for the substrate philosophy. Dependency budget is
//! `std + serde + serde_json` — that's the entire allowlist.
//!
//! Server side:
//!
//! ```no_run
//! use cp_core::CpServer;
//! use serde_json::json;
//! let handle = CpServer::new("127.0.0.1:5577")
//!     .register("status", |_params| Ok(json!({"ok": true})))
//!     .serve()
//!     .unwrap();
//! println!("listening on {}", handle.local_addr());
//! ```
//!
//! Client side: see `cp_core::client::call`.

use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// JSON-RPC 2.0 wire types
// ---------------------------------------------------------------------------

/// Standard JSON-RPC 2.0 error codes. Application-defined codes go
/// outside the range `[-32768, -32000]`. Re-export of the spec table.
#[allow(dead_code)]
pub mod codes {
    pub const PARSE_ERROR: i32 = -32700;
    pub const INVALID_REQUEST: i32 = -32600;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INVALID_PARAMS: i32 = -32602;
    pub const INTERNAL_ERROR: i32 = -32603;
}

/// Request envelope. `id` is `Option<Value>` so we accept JSON numbers,
/// strings, and nulls (the spec allows all three for ids).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcRequest {
    pub jsonrpc: String,
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
    /// Missing field → notification (no response). Present field
    /// (including `null`) → request that wants a response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
}

impl RpcRequest {
    pub fn new<P: Serialize>(method: impl Into<String>, params: P, id: i64) -> Self {
        let params = serde_json::to_value(params).unwrap_or(Value::Null);
        RpcRequest {
            jsonrpc: "2.0".into(),
            method: method.into(),
            params: Some(params),
            id: Some(Value::from(id)),
        }
    }

    pub fn is_notification(&self) -> bool {
        self.id.is_none()
    }
}

/// Error body inside `RpcResponse::error`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl RpcError {
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        RpcError {
            code,
            message: message.into(),
            data: None,
        }
    }
    pub fn invalid_params(msg: impl Into<String>) -> Self {
        RpcError::new(codes::INVALID_PARAMS, msg)
    }
    pub fn internal(msg: impl Into<String>) -> Self {
        RpcError::new(codes::INTERNAL_ERROR, msg)
    }
}

impl std::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (code {})", self.message, self.code)
    }
}

impl std::error::Error for RpcError {}

/// Response envelope. Exactly one of `result` or `error` is present.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcResponse {
    pub jsonrpc: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
    /// Echoes the request id; `Value::Null` if the request was
    /// malformed before we could read its id.
    pub id: Value,
}

impl RpcResponse {
    pub fn success(result: Value, id: Value) -> Self {
        RpcResponse {
            jsonrpc: "2.0".into(),
            result: Some(result),
            error: None,
            id,
        }
    }
    pub fn failure(error: RpcError, id: Value) -> Self {
        RpcResponse {
            jsonrpc: "2.0".into(),
            result: None,
            error: Some(error),
            id,
        }
    }
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

/// Method handler signature: takes the `params` value, returns a
/// `Result<Value, RpcError>` for the spec's `result` / `error` slots.
/// `Send + Sync + 'static` because handlers are dispatched from a
/// per-connection worker thread.
pub type Handler = Arc<dyn Fn(Value) -> Result<Value, RpcError> + Send + Sync + 'static>;

/// Builder + runtime for a JSON-RPC server. Registered handlers map
/// method names to closures; `serve()` consumes the builder and
/// returns a `ServerHandle` to the running listener.
pub struct CpServer {
    addr: String,
    handlers: HashMap<String, Handler>,
}

impl CpServer {
    /// Create a builder targeting `addr`. Pass `127.0.0.1:0` for an
    /// OS-picked port and read it back via `ServerHandle::local_addr`
    /// after `serve()`. Pass `127.0.0.1:5577` (or whatever consumer
    /// default) for a fixed port.
    pub fn new<A: Into<String>>(addr: A) -> Self {
        Self {
            addr: addr.into(),
            handlers: HashMap::new(),
        }
    }

    /// Register a method handler. Panics if `method` is already
    /// registered — registering the same name twice is always a bug.
    pub fn register<H>(mut self, method: &str, handler: H) -> Self
    where
        H: Fn(Value) -> Result<Value, RpcError> + Send + Sync + 'static,
    {
        if self.handlers.contains_key(method) {
            panic!("cp-core: duplicate handler for method '{method}'");
        }
        self.handlers.insert(method.to_string(), Arc::new(handler));
        self
    }

    /// Bind the listener and spawn the accept loop. Returns once the
    /// socket is bound (so callers can immediately use
    /// `local_addr()`); the accept loop runs on its own thread.
    pub fn serve(self) -> io::Result<ServerHandle> {
        let socket: SocketAddr = first_socket_addr(&self.addr)?;
        let listener = TcpListener::bind(socket)?;
        let local_addr = listener.local_addr()?;
        let shutdown = Arc::new(AtomicBool::new(false));
        let connections = Arc::new(AtomicUsize::new(0));
        let handlers = Arc::new(self.handlers);

        // Set a short read timeout on the listener so the accept loop
        // can observe `shutdown` between accepts. Without this an
        // idle server would wait indefinitely on `accept()`.
        listener.set_nonblocking(true)?;

        let shutdown_for_thread = shutdown.clone();
        let connections_for_thread = connections.clone();
        let handlers_for_thread = handlers.clone();
        let join = thread::Builder::new()
            .name("cp-core-listener".into())
            .spawn(move || {
                run_accept_loop(
                    listener,
                    handlers_for_thread,
                    connections_for_thread,
                    shutdown_for_thread,
                );
            })?;

        Ok(ServerHandle {
            local_addr,
            connections,
            shutdown,
            join: Some(join),
        })
    }
}

/// Resolve the first `SocketAddr` from a host:port string. Errors if
/// the input is empty or DNS fails.
fn first_socket_addr(s: &str) -> io::Result<SocketAddr> {
    s.to_socket_addrs()?
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "no socket address resolved"))
}

/// Handle to a running cp-core server. Cheap pieces are clonable
/// `Arc`s; the join handle is single-owner so calling `shutdown()`
/// twice is fine but `join()` only works once.
pub struct ServerHandle {
    pub local_addr: SocketAddr,
    pub connections: Arc<AtomicUsize>,
    shutdown: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl ServerHandle {
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }
    pub fn connection_count(&self) -> usize {
        self.connections.load(Ordering::SeqCst)
    }
    /// Signal the listener to stop. The accept loop exits between
    /// accepts; in-flight connections drain naturally.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
    /// Block until the listener thread exits. Returns immediately
    /// after the first call (the JoinHandle is consumed).
    pub fn join(&mut self) {
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn run_accept_loop(
    listener: TcpListener,
    handlers: Arc<HashMap<String, Handler>>,
    connections: Arc<AtomicUsize>,
    shutdown: Arc<AtomicBool>,
) {
    loop {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }
        match listener.accept() {
            Ok((stream, _peer)) => {
                if shutdown.load(Ordering::SeqCst) {
                    break;
                }
                connections.fetch_add(1, Ordering::SeqCst);
                let connections_for_conn = connections.clone();
                let handlers_for_conn = handlers.clone();
                if let Err(e) =
                    thread::Builder::new()
                        .name("cp-core-conn".into())
                        .spawn(move || {
                            // Restore blocking mode for the per-conn
                            // socket; the non-blocking flag is on the
                            // listener for accept-poll semantics, but
                            // child sockets work better in blocking mode.
                            let _ = stream.set_nonblocking(false);
                            let _ = stream.set_read_timeout(Some(Duration::from_secs(60)));
                            if let Err(e) = handle_connection(stream, &handlers_for_conn) {
                                eprintln!("cp-core: connection error: {e}");
                            }
                            connections_for_conn.fetch_sub(1, Ordering::SeqCst);
                        })
                {
                    eprintln!("cp-core: spawn worker failed: {e}");
                    connections.fetch_sub(1, Ordering::SeqCst);
                }
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                // Idle: poll the shutdown flag periodically.
                thread::sleep(Duration::from_millis(20));
            }
            Err(e) => {
                eprintln!("cp-core: accept failed: {e}");
                thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

fn handle_connection(stream: TcpStream, handlers: &HashMap<String, Handler>) -> io::Result<()> {
    let stream_for_write = stream.try_clone()?;
    let mut reader = BufReader::new(stream);
    let mut writer = stream_for_write;
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => return Ok(()),
            Ok(_n) => {}
            Err(e) if e.kind() == io::ErrorKind::ConnectionReset => return Ok(()),
            Err(e) if e.kind() == io::ErrorKind::TimedOut => return Ok(()),
            Err(e) => return Err(e),
        }
        let trimmed = line.trim_end_matches(&['\r', '\n'][..]);
        if trimmed.is_empty() {
            continue;
        }
        let resp = dispatch_line(trimmed, handlers);
        // Notification (id-less request) → suppress response.
        let Some(resp) = resp else {
            continue;
        };
        let mut payload = serde_json::to_string(&resp).unwrap_or_else(|e| {
            format!(r#"{{"jsonrpc":"2.0","error":{{"code":-32603,"message":"{e}"}},"id":null}}"#)
        });
        payload.push('\n');
        writer.write_all(payload.as_bytes())?;
        writer.flush()?;
    }
}

/// Parse one line, dispatch, build a response. Returns `None` for
/// notifications (no response per JSON-RPC 2.0).
fn dispatch_line(line: &str, handlers: &HashMap<String, Handler>) -> Option<RpcResponse> {
    let req: RpcRequest = match serde_json::from_str(line) {
        Ok(r) => r,
        Err(e) => {
            return Some(RpcResponse::failure(
                RpcError::new(codes::PARSE_ERROR, format!("parse error: {e}")),
                Value::Null,
            ));
        }
    };
    if req.jsonrpc != "2.0" {
        let id = req.id.clone().unwrap_or(Value::Null);
        return Some(RpcResponse::failure(
            RpcError::new(
                codes::INVALID_REQUEST,
                format!("jsonrpc field must be \"2.0\", got {:?}", req.jsonrpc),
            ),
            id,
        ));
    }
    let id_for_response = req.id.clone();
    let is_notif = req.is_notification();

    let result = match handlers.get(&req.method) {
        Some(h) => h(req.params.unwrap_or(Value::Null)),
        None => Err(RpcError::new(
            codes::METHOD_NOT_FOUND,
            format!("method not found: {}", req.method),
        )),
    };

    if is_notif {
        return None;
    }

    let id = id_for_response.unwrap_or(Value::Null);
    let resp = match result {
        Ok(v) => RpcResponse::success(v, id),
        Err(e) => RpcResponse::failure(e, id),
    };
    Some(resp)
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// One-shot client API. Connect, send one request, read one response,
/// close. Suitable for ksctl-style CLI invocations and tests.
pub mod client {
    use super::*;

    /// Open a connection and run a single request/response round-trip.
    /// Uses an auto-incrementing id per call (the integer doesn't need
    /// to be unique across processes; the server only echoes it).
    pub fn call<A: ToSocketAddrs, P: Serialize>(
        addr: A,
        method: &str,
        params: P,
    ) -> io::Result<RpcResponse> {
        let stream = TcpStream::connect(addr)?;
        stream.set_read_timeout(Some(Duration::from_secs(60)))?;
        call_on(stream, method, params)
    }

    /// Like `call`, but reuses an existing TcpStream so a test can
    /// send several requests on one connection.
    pub fn call_on<P: Serialize>(
        stream: TcpStream,
        method: &str,
        params: P,
    ) -> io::Result<RpcResponse> {
        let req = RpcRequest::new(method, params, next_id());
        let mut payload = serde_json::to_string(&req)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        payload.push('\n');

        let read_stream = stream.try_clone()?;
        let mut writer = stream;
        writer.write_all(payload.as_bytes())?;
        writer.flush()?;

        let mut reader = BufReader::new(read_stream);
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let resp: RpcResponse = serde_json::from_str(line.trim_end_matches(&['\r', '\n'][..]))
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(resp)
    }

    /// Open a long-lived connection for callers that want to send
    /// multiple requests in series (e.g. test helpers asserting that
    /// malformed JSON doesn't tear the connection down).
    pub fn connect<A: ToSocketAddrs>(addr: A) -> io::Result<TcpStream> {
        let stream = TcpStream::connect(addr)?;
        stream.set_read_timeout(Some(Duration::from_secs(60)))?;
        Ok(stream)
    }

    fn next_id() -> i64 {
        use std::sync::atomic::AtomicI64;
        static N: AtomicI64 = AtomicI64::new(1);
        N.fetch_add(1, Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn roundtrip_request_response_serde() {
        let req = RpcRequest::new("status", json!({}), 7);
        let s = serde_json::to_string(&req).unwrap();
        assert!(s.contains("\"jsonrpc\":\"2.0\""));
        assert!(s.contains("\"method\":\"status\""));
        assert!(s.contains("\"id\":7"));
    }

    #[test]
    fn unknown_method_yields_method_not_found() {
        let mut handlers = HashMap::new();
        handlers.insert(
            "ping".to_string(),
            Arc::new(|_p: Value| Ok(json!("pong"))) as Handler,
        );
        let line = r#"{"jsonrpc":"2.0","method":"missing","id":1}"#;
        let resp = dispatch_line(line, &handlers).unwrap();
        let err = resp.error.as_ref().unwrap();
        assert_eq!(err.code, codes::METHOD_NOT_FOUND);
    }

    #[test]
    fn malformed_json_yields_parse_error() {
        let handlers: HashMap<String, Handler> = HashMap::new();
        let resp = dispatch_line("this is not json", &handlers).unwrap();
        let err = resp.error.as_ref().unwrap();
        assert_eq!(err.code, codes::PARSE_ERROR);
        assert_eq!(resp.id, Value::Null);
    }

    #[test]
    fn notification_suppresses_response() {
        let handlers: HashMap<String, Handler> = HashMap::new();
        let line = r#"{"jsonrpc":"2.0","method":"unused"}"#;
        assert!(
            dispatch_line(line, &handlers).is_none(),
            "id-less request must not generate a response"
        );
    }

    #[test]
    fn server_serves_and_echoes() {
        let handle = CpServer::new("127.0.0.1:0")
            .register("echo", |p| Ok(p))
            .serve()
            .expect("serve");
        let resp = client::call(handle.local_addr(), "echo", json!({"x": 42})).expect("call");
        assert!(resp.error.is_none(), "{:?}", resp.error);
        assert_eq!(resp.result.unwrap(), json!({"x": 42}));
    }

    #[test]
    fn invalid_jsonrpc_field_rejected() {
        let handlers: HashMap<String, Handler> = HashMap::new();
        let line = r#"{"jsonrpc":"1.0","method":"x","id":1}"#;
        let resp = dispatch_line(line, &handlers).unwrap();
        let err = resp.error.as_ref().unwrap();
        assert_eq!(err.code, codes::INVALID_REQUEST);
    }

    #[test]
    fn handler_can_return_invalid_params() {
        let mut handlers: HashMap<String, Handler> = HashMap::new();
        handlers.insert(
            "needs_x".to_string(),
            Arc::new(|p: Value| {
                p.get("x")
                    .and_then(|v| v.as_i64())
                    .map(|n| json!(n + 1))
                    .ok_or_else(|| RpcError::invalid_params("missing 'x' (i64)"))
            }) as Handler,
        );
        let line = r#"{"jsonrpc":"2.0","method":"needs_x","params":{},"id":3}"#;
        let resp = dispatch_line(line, &handlers).unwrap();
        let err = resp.error.as_ref().unwrap();
        assert_eq!(err.code, codes::INVALID_PARAMS);
    }
}
