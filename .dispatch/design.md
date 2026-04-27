# Design: keysynth voice hot-reload

## Goal

Sub-second iteration loop on voice DSP without restarting the GUI: edit a Rust
source file, save, hear the new voice on the next note. Held notes keep
playing with their original code; new notes route through the freshly-compiled
voice. Crashes and compile failures must surface in the GUI without taking
down the audio thread or losing held notes.

## Approach: Option B (libloading + cdylib)

Picked **B** over A (Steel) and C (IPC):

- **Tier 1 work is pure Rust DSP** (`hammer_stulov`, `string_inharmonicity`,
  `longitudinal`, `piano_modal`). The brief forbids modifying it. Steel would
  force a rewrite of the live-edit voice in a different language, abandoning
  the existing physical-model toolbox. libloading lets the live voice be
  written in the same Rust the rest of the codebase uses, with the same
  `VoiceImpl` trait, the same `KsString` / `Adsr` / `ReleaseEnvelope`
  primitives if needed.
- **Audio-thread perf**: Steel inside the audio callback is risky even at
  control rate; the brief flags it. libloading is a one-time symbol-lookup
  cost at construction; render_add stays a direct-call vtable hop, same as
  the in-tree voices.
- **Compile time on a tiny crate is good**. A single-voice cdylib with
  `opt-level = 1` rebuilds incrementally in 200–800 ms on a warm cache,
  comfortably under the 1 s target. (Measured below.)
- **C is overkill**. The brief calls it itself.

The trade-off accepted: Windows DLL ABI hazards. Section "ABI safety" handles
them.

## Architecture

```
                                 ┌──────────────────────────────────┐
                                 │ voices_live/  (sibling cargo crate)│
                                 │  src/lib.rs  ← USER EDITS THIS    │
                                 │  Cargo.toml  (cdylib, opt-level=1)│
                                 └──────────────────────────────────┘
                                              │ cargo build
                                              ▼
                                 voices_live/target/debug/keysynth_voices_live.dll
                                              │
                                              │ libloading::Library::new
                                              ▼
       ┌────────────────────────┐    ┌───────────────────────────────┐
       │ filesystem watcher     │───►│ LiveReloader                  │
       │ (notify on src/*.rs)   │    │  - latest: ArcSwap<Factory>   │
       │  → debounced rebuild   │    │  - status: Mutex<Status>      │
       │  → spawn cargo build   │    │  (held-Arc<Library> per voice)│
       └────────────────────────┘    └────────────────┬──────────────┘
                                                      │ read on note_on
                                                      ▼
                              ┌──────────────────────────────────────┐
                              │ MIDI thread make_voice(Engine::Live) │
                              │  → factory.construct(sr, freq, vel)  │
                              │  → wrap in LiveVoice { inner, _lib } │
                              └──────────────────────────────────────┘
                                                      │
                                                      ▼ pushed into pool
                              ┌──────────────────────────────────────┐
                              │ audio cb: voice.render_add (unchanged)│
                              └──────────────────────────────────────┘
```

### Why a sibling crate, not a workspace member

- Zero churn on the main crate's Cargo.toml feature matrix (the wasm32 build
  is fragile and the brief calls that out).
- Independent target dir → `cargo build` of the live crate can't touch the
  main crate's incremental cache.
- The wasm32 build never sees `voices_live/` because it's a separate cargo
  invocation, gated by `#[cfg(not(target_arch = "wasm32"))]` on the main
  side anyway.

## ABI safety

The danger: a `Box<dyn VoiceImpl>` constructed inside the live DLL holds a
vtable pointer into that DLL's text segment. Drop glue, `render_add`,
everything resolves through the vtable. Unloading the DLL while a Box is
still alive is UB.

**Solution**: every live-constructed voice is wrapped in a struct that pins
the library:

```rust
pub struct LiveVoice {
    inner: Option<Box<dyn VoiceImpl + Send>>,  // dropped FIRST
    _lib: Arc<Library>,                         // dropped AFTER inner
}
```

Rust drops struct fields top-to-bottom, so `inner`'s drop glue (which lives
in the lib) runs while `_lib` still holds the Library alive. Only then is
the Library Arc decremented; if it's the last reference, libloading unmaps
the DLL.

The reloader holds the *current* factory's `Arc<Library>`. Old factories
are replaced by writing a new `Arc<Library>` into an `ArcSwap`. The old
Arc's refcount only hits zero when the live-reloader no longer references
it AND every voice constructed from it has been dropped. This is exactly
what we want: held notes keep their old DSP code valid until they fall
silent.

The voice pool already evicts dead voices in `main.rs`. No new lifecycle
work needed.

## Atomic swap protocol

- `ArcSwap<Option<LiveFactory>>` shared between the watcher thread and
  the MIDI callback.
- Watcher: build → load → wrap in `LiveFactory { lib: Arc<Library>, ctor: ... }`
  → `arc_swap.store(Arc::new(Some(new_factory)))`. Old factory drops when
  no voice references it.
- MIDI thread (note_on): `arc_swap.load()` → call ctor → wrap → push into
  voice pool. The load is wait-free (arc-swap's primary feature).
- Audio thread: never touches the reloader. It only renders voices already
  in the pool; the Arc<Library> pinning is invisible to it.

`std::sync::RwLock` would also work, but the watcher's write-side may
block briefly if a note_on is mid-load. arc-swap removes that contention.

## C ABI

The cdylib exposes a small C-compatible facade so we don't depend on Rust's
unstable trait-object ABI across compilation units (different rustc, lto
settings, etc. could mismatch vtable layouts):

```rust
#[no_mangle] pub unsafe extern "C" fn keysynth_live_new(sr: f32, freq: f32, vel: u8) -> *mut c_void;
#[no_mangle] pub unsafe extern "C" fn keysynth_live_render_add(p: *mut c_void, buf: *mut f32, n: usize);
#[no_mangle] pub unsafe extern "C" fn keysynth_live_trigger_release(p: *mut c_void);
#[no_mangle] pub unsafe extern "C" fn keysynth_live_is_done(p: *mut c_void) -> bool;
#[no_mangle] pub unsafe extern "C" fn keysynth_live_is_releasing(p: *mut c_void) -> bool;
#[no_mangle] pub unsafe extern "C" fn keysynth_live_drop(p: *mut c_void);
#[no_mangle] pub extern "C" fn keysynth_live_abi_version() -> u32;
```

Host loads each symbol once, caches them in `LiveFactory`. ABI version is
verified on load — mismatch = reject and surface error in side panel.

## Crash isolation

Three boundaries wrapped in `catch_unwind`:

1. Library load (`libloading::Library::new` + symbol lookup + ABI check).
2. Voice construction (the FFI ctor).
3. Per-voice `render_add` (called from the audio callback).

A panic on (1) or (2) leaves the previous factory untouched and the
error in the status panel.

A panic in (3) marks the voice as done in `LiveVoice::is_done()` and
silences it for the rest of its life. The audio callback is the only
hard-crash-must-not-happen path; the wrapper makes it bounded.

cargo build failures aren't panics — they're stderr from a child process,
captured and surfaced verbatim in the side panel.

## Reload trigger

`notify` watches `voices_live/src/`. On any write (debounced 150 ms):

1. spawn `cargo build --manifest-path voices_live/Cargo.toml` as a child
   process. Capture stderr.
2. on success: `libloading::Library::new(<dll path>)`, re-check ABI version,
   atomic-swap.
3. on failure: status = Err(stderr tail).

A `Ctrl+R` keybind also forces a rebuild.

The brief mentioned `voices_live/*.scm`. Since we picked Rust, the watched
extension is `.rs`. Same effect.

## UI changes

A new "Live voice" entry in the existing voice browser side panel
(`src/ui.rs`). Selecting it sets `LiveParams.engine = Engine::Live`.

A new bottom strip on the voice-browser panel shows:

- current dll path
- last reload timestamp + duration
- last error (red, monospaced) if any

Silent failure is the brief's worst-case. Errors are red, large, and stay
until the next successful reload.

## What is NOT changed

- `Engine` gets one new variant: `Engine::Live`. Existing variants untouched.
- `make_voice` gets one new arm dispatching to the reloader's current
  factory.
- `voice_lib.rs` gets one new builtin slot pointing at `Engine::Live`.
- `audio.rs` / cpal callback / wasm32 path: zero change.
- Tier 1 voice files: zero change.

## Web build

`Engine::Live` is a native-only enum variant gated by
`#[cfg(feature = "live_reload")]`. The `live_reload` feature is auto-on
under `native` and auto-off under `web`. The wasm32 path never touches
libloading or notify. Verified by running
`cargo check --no-default-features --features web --target wasm32-unknown-unknown --bin keysynth-web`
in CI.

Actually simpler: the variant is gated by `#[cfg(not(target_arch = "wasm32"))]`
inline. No new feature flag needed.

## Held notes

Designer's call (per brief). Picked **keep playing with old voice**: simpler,
matches user expectation when iterating, no audible glitch on reload. A
fade-and-retrigger option can be added later as a checkbox.

The Arc<Library> pinning makes this safe automatically — a voice that was
constructed under lib v1 keeps lib v1's code mapped until it dies, even
after lib v2 has replaced it for new notes.

## Tests

Unit (in `live_reload.rs`):

- ABI version mismatch → error surfaced, factory unchanged.
- Atomic swap: two threads — one repeatedly stores new factories, one
  repeatedly loads — no deadlock, no torn read.
- Held-voice survives reload: construct voice from factory v1, swap in v2,
  drop v1 from reloader, voice still renders without UB.

Integration (in `tests/live_reload_smoke.rs`):

- Build the bundled `voices_live/` crate, load it, render a few buffers.
- Rewrite `voices_live/src/lib.rs` with broken Rust → cargo fails → status
  = Err, previous factory still works.
- Rewrite with valid different code → reload succeeds → next voice
  produces different samples.
- Delete dll → reload skipped, status surfaced, previous factory works.

The integration test is gated on `cfg(not(target_os = "macos"))` only if
codesign issues bite — we're targeting Windows/Linux primary; macOS is
untested. (If macOS works for free, drop the gate.)

## Reload latency budget

- notify debounce: 150 ms
- cargo incremental build (1-file edit, opt-level=1, sine voice): 300–700 ms
  on warm cache.
- `Library::new` + symbol resolution: < 5 ms.
- ArcSwap store: ~ns.

Worst-case end-to-end on a warm cache: ~ 850 ms. Within budget.

A cold first build (no incremental cache) takes 5–15 s. We log "first
build…" so the user knows the latency is one-time. Subsequent edits are
fast.

## Out of scope (deliberate)

- DSL-style voice scripting (Steel, etc.). Native Rust is fine for the
  iteration loop.
- Persisting the live source across runs (it's a checked-in file, git
  handles it).
- Multiple live voices at once. One live slot is the MVP.
- Hot-swapping shared state (Adsr presets etc.). Keep the live voice
  self-contained for now.
