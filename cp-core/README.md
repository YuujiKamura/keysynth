# cp-core

Vendor-neutral JSON-RPC 2.0 control-protocol substrate over TCP
localhost. Built for reuse across the studio's multi-process tools
(keysynth → ksctl, future ghostty-win / photo-ai-lisp consumers).

## Why hand-rolled

The studio runs many small Rust binaries that need a "let an external
process drive me" capability. We deliberately do not import
`jsonrpsee` / `rmcp` / `mcp-*` / `tonic` or any other third-party RPC
SDK — those crates ship someone else's protocol opinions, error
hierarchy, and version pinning into every consumer that touches them.
Once two crates depend on different RPC frameworks the whole graph
fragments.

`cp-core` is small enough (one file, ~400 lines) that owning it is
cheaper than picking sides.

Dependency budget: `std + serde + serde_json`. Anything beyond that
is rejected.

## Wire format: JSON-RPC 2.0

[Spec](https://www.jsonrpc.org/specification) is two pages.

Request:
```json
{"jsonrpc":"2.0","method":"<name>","params":<value-or-omitted>,"id":<int-or-string-or-null>}
```

Response (success):
```json
{"jsonrpc":"2.0","result":<value>,"id":<echoed>}
```

Response (error):
```json
{"jsonrpc":"2.0","error":{"code":<int>,"message":"...","data":<optional>},"id":<echoed-or-null>}
```

Notifications (no `id` field) are accepted but ignored — no response
is generated. We don't use them at this layer; consumers that need
push semantics open a second connection.

Standard error codes used:

| code    | name             | when                                  |
| ------- | ---------------- | ------------------------------------- |
| -32700  | Parse error      | Line wasn't valid JSON                |
| -32600  | Invalid Request  | Missing `jsonrpc` / `method` field    |
| -32601  | Method not found | No registered handler for the method  |
| -32602  | Invalid params   | Handler rejected `params` shape       |
| -32603  | Internal error   | Handler returned a generic Err string |

Application-defined errors use codes outside [-32768, -32000].

## Transport

TCP localhost (`127.0.0.1` / `::1`). Picked over Unix-socket / named
pipe because:

- `curl --json '{...}' http://127.0.0.1:5577` and `nc 127.0.0.1 5577`
  work out of the box for debug.
- One transport across all platforms — no Windows / Unix `cfg` forks
  in every consumer.
- Local-only: bind to `127.0.0.1`. We don't expose this to the
  network.

Wire framing is line-delimited (`\n`-terminated JSON). One request per
line, one response per line.

## API

```rust
use cp_core::{CpServer, RpcError};
use serde_json::json;

let handle = CpServer::new("127.0.0.1:5577")
    .register("status", |_params| Ok(json!({"ok": true})))
    .register("echo", |params| Ok(params))
    .serve()?;

println!("listening on {}", handle.local_addr());
// ... handle.shutdown() to stop.
```

The client side:

```rust
let resp = cp_core::client::call(handle.local_addr(), "status", json!(null))?;
```

## What lives WHERE

| Concern                           | Layer        |
| --------------------------------- | ------------ |
| JSON-RPC 2.0 framing + dispatch   | `cp-core`    |
| TCP accept loop, per-conn worker  | `cp-core`    |
| Method handlers (slot ops, etc.)  | consumer     |
| ksctl-style CLI clients           | consumer     |

## Future consumers

`ksctl` is consumer #1 (in keysynth). A consolidated CLI `cpctl`
which lets one client drive any cp-core server is planned for a later
PR.
