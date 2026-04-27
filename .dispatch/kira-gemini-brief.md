# Brief A (Gemini): main-thread glue for Kira-backed web audio

## Context

The user is abandoning the hand-rolled AudioWorkletProcessor + chunk-
render path in `src/bin/web.rs`. New audio backend is Kira
(`kira = "0.12"`, already added to `[dependencies]` and to the `web`
feature in `Cargo.toml`).

Architecture is split between two parallel agents:

- **You (Gemini)** — main-thread glue layer in `src/bin/web.rs`.
- **Codex (other deck)** — `src/audio_kira.rs` implementation.

DO NOT TOUCH `src/audio_kira.rs`. Codex owns that file. Treat it as
a black box that exposes the public API documented at the top of
the file. If the API is missing something you need, comment in your
PR description; do not edit Codex's file.

## Working tree

- Worktree: `C:\Users\yuuji\AppData\Local\Temp\keysynth-web`
  (Unix: `/tmp/keysynth-web`)
- Branch: `claude/web-kira-experiment` (already checked out, based
  on `origin/main`)
- Local serve: `trunk serve web/index.html --port 8090
  --no-autoreload`. Use `localhost:8090` or `192.168.2.100:8090`
  (deckpilot occupies 127.0.0.1:8080).

## Goal

In `src/bin/web.rs`:

1. Delete the `WORKLET_PROCESSOR_JS` const and all the
   AudioWorkletNode handshake / chunk-render plumbing in
   `build_audio`.
2. Replace it with a `KiraEngine` instance held by `WebApp`.
   Initialise via `audio_kira::KiraEngine::new()` on the splash
   gate's `▶ Start` click. Stash in `WebApp` (Rc<RefCell<Option<…>>>
   or just `Option<…>` if you can avoid sharing).
3. Rewrite `WebApp::note_on` / `note_off` to delegate to
   `KiraEngine::note_on(engine, note, velocity)` /
   `note_off(note)`. The current `voices: Arc<Mutex<Vec<Voice>>>`
   pool on the main thread goes away — Kira owns the pool.
4. Voice slot click: call `KiraEngine::set_engine(slot.engine)`. If
   `slot.modal_preset.is_some()`, call
   `KiraEngine::set_modal_preset(preset)` too.
5. Master / reverb / mix UI changes don't need Kira sync for this
   first cut — leave `KiraEngine` at its default values. Note in
   the PR description what's deferred.

## What to keep

- The egui UI: voice browser side panel, MIDI log, on-screen
  keyboard, master/reverb sliders, mix-mode row, GM 128 patch
  picker. Only the audio backend swap; UI is fine.
- `WebApp` struct minus the audio-thread plumbing (drop the cpal /
  AudioContext / AudioWorkletNode fields, the `voices` pool, the
  `synth` SharedSynth, the SF2 fetch path).
- Splash gate. The `▶ Start` button is a user gesture required for
  AudioContext creation; same here for Kira's manager.
- PC keyboard handling (`pc_held`, `handle_pc_keyboard`,
  panic_release).
- Web MIDI handshake (`request_midi`, MIDI inbox).

## What to throw away

- `WORKLET_PROCESSOR_JS` constant.
- `WORKLET_PROCESSOR_JS`-specific blob URL / addModule plumbing in
  `build_audio`.
- `AudioWorkletNode::new_with_options` call.
- `port.onmessage` / postMessage chunk pump.
- `RenderState`, `render_mono_chunk`, `mono_compressed`,
  `limiter_gain` fields if they live in `web.rs` — Codex owns the
  bus chain now.
- The `synth: SharedSynth` (rustysynth) path. SF2 / `Engine::SfPiano`
  is OUT OF SCOPE for this experiment; leave a stub that logs
  "SfPiano not yet supported on Kira backend" and does nothing on
  note_on. Future work.
- The `recording: Rc<RefCell<Option<Vec<f32>>>>` capture button.
  Kira renders inline; capturing requires a different approach.
  Drop the button, drop the field, drop the encode/download
  helpers. Note in PR description.
- The async SF2 fetch (`fetch_and_load_sf2`). Drop it.

## Cargo.toml

- `web` feature already lists `dep:kira` (added in the scaffold
  commit). Keep it.
- DROP `dep:rustysynth` from the `web` feature (no SfPiano path
  for now). Keep `dep:rustysynth` available under `native` only.
- DROP `Response`, `HtmlAnchorElement` from web-sys features (the
  fetch + download paths are gone). Verify nothing else needs them.
- Keep `wasm-bindgen`, `wasm-bindgen-futures`, `web-sys` (Window,
  Document, Navigator, MIDI features), `js-sys`,
  `console_error_panic_hook`, `cpal/wasm-bindgen` (kira's cpal
  needs it).

## Verification

After your edits, all of these must pass before opening the PR:

1. `cargo check --bin keysynth` (native) — clean. The native
   binary must keep compiling.
2. `cargo check --no-default-features --features web --target
   wasm32-unknown-unknown --bin keysynth-web` — clean.
3. `cargo fmt --check` — clean.
4. `cargo build --release --no-default-features --features web
   --target wasm32-unknown-unknown --bin keysynth-web` — clean.
   Note the wasm size.
5. Trunk: `trunk serve web/index.html --port 8090
   --no-autoreload`. Wait for ✅ success.
6. Wait for Codex to finish `src/audio_kira.rs`. Once their commit
   lands on the same branch, pull, recompile.
7. Browser at http://localhost:8090/: ▶ Start, click "Square"
   voice, press `Z`. Should produce a clean square tone.
8. Test sustained: hold `Z` 5+ seconds while scrolling the egui
   panel — should NOT cut out (this is the whole point of the
   Kira swap).

## Coordination with Codex

- **Both branches push to `claude/web-kira-experiment`**. Pull
  before pushing.
- File-level no overlap: Gemini owns `src/bin/web.rs` +
  `Cargo.toml`. Codex owns `src/audio_kira.rs`.
- Public API contract (the doc-comment block at the top of
  `src/audio_kira.rs`) is the integration boundary. If the actual
  signatures differ from the doc, prefer the actual signatures —
  Codex is the authoritative source for the audio_kira API.
- Don't block on Codex finishing. You can stub-call the API and
  rely on the placeholder `audio_kira` module compiling. Once
  Codex lands their impl, your code links against the real thing.

## When stuck

- Kira docs: https://docs.rs/kira/0.12
- If Kira's `AudioManager::new()` fails on wasm32 with a missing
  AudioContext / cpal-wasm-bindgen error, that's a real finding —
  report it explicitly in the PR description rather than hiding.

## Deliverable

Commits on `claude/web-kira-experiment` with:
- `Cargo.toml` cleaned of dead deps (rustysynth in web, Response,
  HtmlAnchorElement)
- `src/bin/web.rs` migrated to `KiraEngine`
- `cargo check / fmt` clean
- A clear PR-ready commit message describing what moved where

When both you and Codex are done, open one PR
`feat(web): Kira-based audio backend (experiment)` to `main` with
both halves of the work in the description. CI must be green.
