# Design: keysynth Control Protocol (CP)

Sibling of [PR #40 / `.dispatch/design.md`]. Where #40 introduced one
hot-reloadable voice slot driven by a filesystem watcher, this brief
generalises the system to N named slots driven by an external IPC
client (`ksctl`) so external processes can push, swap, and headlessly
render voices without restarting the keysynth GUI.

## Goal

Keep the GUI running. Let an external CLI (`ksctl`) do everything the
brief lists:

- list slots
- load a pre-built `.dll` into a named slot
- build a sibling cargo crate and load its output into a slot
- set the active slot (next note_on routes through it)
- render a note list to WAV through the active slot, headless
- query server status

…over a local-only line-delimited JSON pipe. The whole feature is
opt-in (`--cp` flag); with it off, behaviour is identical to today.

## Transport

| Platform | Path                                                     |
| -------- | -------------------------------------------------------- |
| Windows  | `\\.\pipe\keysynth-cp` (named pipe)                      |
| Unix     | `${TMPDIR:-/tmp}/keysynth-cp.sock` (AF_UNIX socket)      |

Implemented via the `interprocess` crate (`local_socket` API), which
gives a uniform `Stream` / `Listener` over both. Picked over hand-rolling
`CreateNamedPipeW` + `UnixListener` because the brief is a v1 with
local-only access — there's no value in writing more `unsafe` than
necessary, and `interprocess` is small (`~200kB compiled`, no transitive
async runtime).

Authentication: none. The brief defers it. The pipe is local-process
already (Windows: named-pipe ACLs default to user-only; Unix: socket
file permissions 0600). Documented as "trust your local user account".

## Wire format

Line-delimited JSON. One request per `\n`-terminated line, one response
per `\n`-terminated line. Connection-oriented: a client may send many
requests on one connection, but each is independent.

### Requests (op-tagged)

```json
{"op": "list"}
{"op": "status"}
{"op": "load",   "slot": "A", "dll": "C:/path/to/voice.dll"}
{"op": "unload", "slot": "A"}
{"op": "set",    "slot": "A"}
{"op": "build",  "src": "voices_live", "slot": "A"}
{"op": "render", "notes": [60,64,67,72], "duration": 6.0, "velocity": 100, "out": "arp.wav"}
```

### Responses

Always one of:

```json
{"ok": true,  "data": {...}}
{"ok": false, "error": "no such slot 'X'"}
```

Concrete shapes:

- `list.data` → `{"slots":[{"name":"A","dll":"...","loaded_at_unix_ms":...}], "active":"A"}`
- `status.data` → `{"active":"default","slots":3,"connections":1,"sr_hz":44100}`
- `load / unload / set / build` → `data: {"slot":"A"}` on success
- `render.data` → `{"out":"arp.wav","frames":264600,"duration_s":6.0,"sha256":"..."}` (sha256 lets tests assert determinism)

Unknown op → `{"ok":false,"error":"unknown op '...'"}`.

## Multi-slot Reloader

Generalising from PR #40's single-slot `Reloader`:

```rust
pub struct Reloader {
    // PR #40 (kept for back-compat with the watcher path):
    pub current: Arc<ArcSwap<Option<Arc<LiveFactory>>>>,  // legacy alias for the active slot's factory
    pub status:  Arc<ArcSwap<Status>>,                    // last build status

    // CP additions:
    slots:        Arc<Mutex<HashMap<String, Arc<LiveFactory>>>>,
    active_slot:  Arc<ArcSwap<String>>,  // name of the active slot
    rebuild_tx:   Sender<RebuildRequest>,
    pub crate_root: PathBuf,
}
```

The watcher path stores its built factory under slot name `"default"`
and (only at first build) sets `active_slot = "default"` so existing
behaviour is unchanged: a user with no CP usage sees exactly today's
single-watch loop.

`make_voice` in the trampoline registered with `synth.rs` reads
`active_slot` first, looks up that slot in `slots`, and constructs.
If the active slot is empty (e.g. user `ksctl unload`'d it), returns
`None` → `Engine::Live` falls back to silence (same as today when the
initial build hasn't finished).

`current` is updated as the active-slot mirror so the existing GUI
status panel keeps working without a special case for "no watcher".

### Slot operations

- `set_slot(name, factory)` — replace; old factory drops only when no
  voice references it (Arc pinning from PR #40 unchanged).
- `unset_slot(name)` — remove from map.
- `set_active(name)` — fails if slot doesn't exist; otherwise atomically
  swaps `active_slot`.
- `build_into_slot(crate_root, name)` — exposes PR #40's `build_and_load`
  for arbitrary crate roots, stores the result into the named slot.
- `list_slots()` → snapshot of slot metadata for the `list` op.

All callable from any thread; the audio thread never touches them
(audio path only reads `active_slot` and the slot map via fast paths).

## Render op

Non-realtime, runs entirely on the CP-server thread that received the
request. Constructs one voice per MIDI note via the active factory,
plays each note for `duration / N` seconds (arpeggio — sequential, not
chord, so timbre differences between voices are clearly audible per
attack), triggers release at the end of each note's slice, sums into
mono `f32`, and writes a 16-bit PCM WAV via `hound` (already in
`native` deps).

Sample rate: the server takes a `sr_hz` at startup matching the audio
device's SR (44100 by default). `render` produces WAVs the user can
diff bit-for-bit, which is what the test plan asserts.

A SHA-256 of the rendered samples goes in the response so the test
plan's `diff <(sha256sum a.wav) <(sha256sum b.wav)` becomes
`response.data.sha256` differing — no need for the test to actually
read the WAVs back.

## ksctl CLI

`src/bin/ksctl.rs`. Plain `clap`-free arg parsing (the existing crate
doesn't use clap; mirroring its style keeps the dep tree tight).
Subcommands: `list`, `load`, `unload`, `set`, `build`, `render`,
`status`. Each constructs the JSON request, opens the pipe, sends one
line, reads one line, prints terminal-friendly output (with `--json` to
get the raw response if scripting).

Default pipe / socket path matches the server's so the user usually
doesn't pass `--socket`; an env var `KEYSYNTH_CP` overrides for tests.

## Concurrency model

```
GUI thread (egui) ─── reads Reloader.status_snapshot() per frame
                  └── reads CP::Server.connection_count() per frame

audio thread     ─── reads active_slot + slots[active] (lock-free read)

CP listener thread ── accept() loop on the local socket
                  └── per-conn worker thread: blocking read line / write line
                                              dispatch via Reloader + render
```

The audio thread takes the `slots` mutex exactly once per `make_voice`
call (note_on rate, ~tens of Hz tops). That's ~µs of contention against
infrequent CP writes. ArcSwap on `active_slot` keeps the actual swap
wait-free.

Per-connection worker uses `serde_json::from_str` per line, dispatches,
serialises response. No async runtime — `interprocess`'s blocking API
plus one `thread::spawn` per connection is sufficient at the
"one human typing ksctl commands" scale the brief describes.

## Latency budget

Target: `ksctl set <slot>` → first sample of new voice on next note
under 50 ms.

Components:
- ksctl process spawn + connect: ~20 ms (Windows is the slow OS here)
- single line of JSON over local pipe: < 1 ms
- ArcSwap store: nanoseconds
- next note_on actually processed: bounded by the next MIDI message,
  not by us

We measure pipe-roundtrip `set` latency in the integration test
(connect → write → read → close) — that's the actual server-side
contribution, which is what a host loop / scripted A/B controls. CLI
spawn overhead is documented but not counted.

## Security / opt-in

- `--cp` flag default OFF.
- When ON, server logs the bound socket path on startup and on every
  client connect.
- No network bind. Local-pipe only.
- The build op runs `cargo build` against the requested crate path
  (resolved relative to server CWD). User must trust whoever has access
  to the pipe — same trust boundary as having shell access on the
  machine.

## What is NOT changed

- `Engine::Live` matching arm in `make_voice` — unchanged, still goes
  through the same registered trampoline.
- The watcher path — keeps its single "default" slot semantics.
- The voice browser entry "Live (hot edit)" — same UI as today.
- `LiveFactory::load` ABI version check, ABI version constant, the
  voices_live cdylib — untouched.

## Out of scope

- Authentication / network — local pipe only for v1.
- Cross-process voice MMAP — every slot is a `LiveFactory` loaded by
  the GUI process.
- Hot-reload watch on N slots — only the single watched crate root from
  PR #40 still triggers automatic rebuilds; CP-loaded slots are
  reload-on-demand via `ksctl build`.
- Wasm32 path — feature-gated identically to PR #40.

## Files

```
.dispatch/cp-design.md         — this doc
src/cp.rs                      — server module + JSON protocol
src/bin/ksctl.rs               — CLI client
src/live_reload.rs             — multi-slot extension (additive)
src/main.rs                    — --cp flag + Server::start wiring
src/ui.rs                      — CP status block in side panel
tests/cp_smoke.rs              — headless CP integration test
Cargo.toml                     — interprocess dep, ksctl bin entry
```
