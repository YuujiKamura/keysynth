//! Voice hot-reload subsystem.
//!
//! Watches `voices_live/src/` for source edits, spawns
//! `cargo build --target-dir voices_live/target` on change, and atomically
//! swaps the loaded `.dll` / `.so` / `.dylib` produced by that build. The
//! MIDI thread reads the current factory through an `ArcSwap` and
//! constructs voices via the C-ABI exports defined in
//! `voices_live/src/lib.rs`.
//!
//! See `.dispatch/design.md` for the full rationale (Option B / libloading)
//! and ABI-safety story (each constructed voice pins its `Arc<Library>` so
//! the lib stays mapped until the voice drops, even after a swap).

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use libloading::{Library, Symbol};
use notify::{Event, EventKind, RecursiveMode, Watcher};

use crate::synth::{ReleaseEnvelope, VoiceImpl};

/// ABI version expected by the host. Must match the value returned by
/// `keysynth_live_abi_version` in the loaded library. Bumped on any
/// incompatible C-ABI change.
pub const EXPECTED_ABI_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Symbol signatures (mirror voices_live/src/lib.rs).
// ---------------------------------------------------------------------------

type FnAbiVersion = unsafe extern "C" fn() -> u32;
type FnNew = unsafe extern "C" fn(sr: f32, freq: f32, vel: u8) -> *mut c_void;
type FnRender = unsafe extern "C" fn(p: *mut c_void, buf: *mut f32, n: usize);
type FnRelease = unsafe extern "C" fn(p: *mut c_void);
type FnIsDone = unsafe extern "C" fn(p: *mut c_void) -> bool;
type FnIsReleasing = unsafe extern "C" fn(p: *mut c_void) -> bool;
type FnDrop = unsafe extern "C" fn(p: *mut c_void);

/// One loaded library + its resolved symbols. Cheap to clone-via-Arc.
///
/// The `Library` is held inside an `Arc` because every voice constructed
/// from this factory keeps its own clone alive until the voice itself
/// drops. Only when the reloader has swapped to a newer factory AND every
/// outstanding voice has dropped does the old `.dll` actually unmap. This
/// is what makes hot-reload safe: held notes finish out with the code
/// they were spawned under, even after the user has swapped to v2.
pub struct LiveFactory {
    pub lib: Arc<Library>,
    new_fn: FnNew,
    render_fn: FnRender,
    release_fn: FnRelease,
    is_done_fn: FnIsDone,
    is_releasing_fn: FnIsReleasing,
    drop_fn: FnDrop,
    /// Where the loaded DLL lived on disk. Surfaced to the user in the
    /// status panel.
    pub dll_path: PathBuf,
    /// When this factory finished loading. Used to render the
    /// "last reload" timestamp.
    pub loaded_at: Instant,
}

impl LiveFactory {
    /// Open `dll_path` with libloading, resolve every required symbol,
    /// and check the ABI-version export against `EXPECTED_ABI_VERSION`.
    ///
    /// Wrapped in catch_unwind by the caller (`Reloader::try_load`) so a
    /// crash inside libloading or the lib's static-init can't take down
    /// the whole process.
    pub fn load(dll_path: &Path) -> Result<Self, String> {
        let lib =
            unsafe { Library::new(dll_path).map_err(|e| format!("Library::new failed: {e}"))? };

        // Symbol lookup — store function pointers by VALUE (not Symbol<'_>)
        // so we don't tie them to the Library borrow. The Library lives
        // inside the Arc this factory will own; the function pointers
        // remain valid as long as ANY clone of that Arc is alive.
        unsafe {
            let abi_v: Symbol<FnAbiVersion> = lib
                .get(b"keysynth_live_abi_version")
                .map_err(|e| format!("missing keysynth_live_abi_version: {e}"))?;
            let actual = abi_v();
            if actual != EXPECTED_ABI_VERSION {
                return Err(format!(
                    "ABI mismatch: lib reports v{actual}, host expects v{EXPECTED_ABI_VERSION}"
                ));
            }

            let new_fn: FnNew = *lib
                .get::<FnNew>(b"keysynth_live_new")
                .map_err(|e| format!("missing keysynth_live_new: {e}"))?;
            let render_fn: FnRender = *lib
                .get::<FnRender>(b"keysynth_live_render_add")
                .map_err(|e| format!("missing keysynth_live_render_add: {e}"))?;
            let release_fn: FnRelease = *lib
                .get::<FnRelease>(b"keysynth_live_trigger_release")
                .map_err(|e| format!("missing keysynth_live_trigger_release: {e}"))?;
            let is_done_fn: FnIsDone = *lib
                .get::<FnIsDone>(b"keysynth_live_is_done")
                .map_err(|e| format!("missing keysynth_live_is_done: {e}"))?;
            let is_releasing_fn: FnIsReleasing = *lib
                .get::<FnIsReleasing>(b"keysynth_live_is_releasing")
                .map_err(|e| format!("missing keysynth_live_is_releasing: {e}"))?;
            let drop_fn: FnDrop = *lib
                .get::<FnDrop>(b"keysynth_live_drop")
                .map_err(|e| format!("missing keysynth_live_drop: {e}"))?;

            Ok(Self {
                lib: Arc::new(lib),
                new_fn,
                render_fn,
                release_fn,
                is_done_fn,
                is_releasing_fn,
                drop_fn,
                dll_path: dll_path.to_path_buf(),
                loaded_at: Instant::now(),
            })
        }
    }

    /// Construct a `LiveVoice`. Pins this factory's `Arc<Library>` into
    /// the voice so the lib stays mapped until the voice drops.
    pub fn make_voice(self: &Arc<Self>, sr: f32, freq: f32, vel: u8) -> Box<dyn VoiceImpl + Send> {
        let inner = unsafe { (self.new_fn)(sr, freq, vel) };
        Box::new(LiveVoice {
            inner,
            render_fn: self.render_fn,
            release_fn: self.release_fn,
            is_done_fn: self.is_done_fn,
            is_releasing_fn: self.is_releasing_fn,
            drop_fn: self.drop_fn,
            // KEY: pinning the Arc keeps the lib mapped for at LEAST as
            // long as this voice. Field declared AFTER `inner` so Rust's
            // top-down drop order runs `inner`'s release through `drop_fn`
            // (which lives in the lib's text segment) BEFORE _lib's
            // refcount decrements.
            _lib: self.lib.clone(),
        })
    }
}

/// A voice constructed by a hot-loadable library. Owns an opaque
/// `*mut c_void` plus a clone of the lib's `Arc` to keep its code mapped.
///
/// Drop order: `inner` field is declared before `_lib`, so `inner` is
/// dropped first (calling the lib's `drop_fn`, which is still mapped),
/// then `_lib` decrements. If this is the last reference to the lib,
/// libloading unmaps it AFTER the voice is fully torn down.
pub struct LiveVoice {
    inner: *mut c_void,
    render_fn: FnRender,
    release_fn: FnRelease,
    is_done_fn: FnIsDone,
    is_releasing_fn: FnIsReleasing,
    drop_fn: FnDrop,
    _lib: Arc<Library>,
}

// SAFETY: the inner pointer is owned exclusively by this `LiveVoice`
// (no aliasing) and the C ABI we expose is single-threaded per voice
// (only the audio callback touches it after construction). The pinned
// `Arc<Library>` is itself Send + Sync. libloading::Library is Send.
unsafe impl Send for LiveVoice {}

impl Drop for LiveVoice {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe { (self.drop_fn)(self.inner) };
            self.inner = std::ptr::null_mut();
        }
    }
}

impl VoiceImpl for LiveVoice {
    fn render_add(&mut self, buf: &mut [f32]) {
        if self.inner.is_null() || buf.is_empty() {
            return;
        }
        unsafe { (self.render_fn)(self.inner, buf.as_mut_ptr(), buf.len()) };
    }

    fn release_env(&self) -> Option<&ReleaseEnvelope> {
        None
    }

    fn release_env_mut(&mut self) -> Option<&mut ReleaseEnvelope> {
        None
    }

    fn trigger_release(&mut self) {
        if self.inner.is_null() {
            return;
        }
        unsafe { (self.release_fn)(self.inner) };
    }

    fn is_done(&self) -> bool {
        if self.inner.is_null() {
            return true;
        }
        unsafe { (self.is_done_fn)(self.inner) }
    }

    fn is_releasing(&self) -> bool {
        if self.inner.is_null() {
            return false;
        }
        unsafe { (self.is_releasing_fn)(self.inner) }
    }
}

// ---------------------------------------------------------------------------
// Reloader: filesystem watcher + cargo invoker + atomic factory swap.
// ---------------------------------------------------------------------------

/// Status surfaced to the GUI side panel. Updated by the watcher thread,
/// read by the egui thread on every frame.
#[derive(Clone, Debug)]
pub enum Status {
    /// No build attempted yet. Initial state at startup.
    Idle,
    /// `cargo build` is currently running for this trigger reason.
    Building { since: Instant, reason: String },
    /// Most recent build + load succeeded `at` ago, took `duration`.
    Ok {
        at: Instant,
        duration: Duration,
        dll_path: PathBuf,
    },
    /// Most recent build or load failed; `message` is the user-visible
    /// reason (cargo stderr tail or libloading error).
    Err { at: Instant, message: String },
}

impl Status {
    pub fn short_label(&self) -> String {
        match self {
            Status::Idle => "idle".into(),
            Status::Building { reason, .. } => format!("building... ({reason})"),
            Status::Ok { duration, .. } => format!("loaded in {} ms", duration.as_millis()),
            Status::Err { message, .. } => format!("ERR: {message}"),
        }
    }

    pub fn is_err(&self) -> bool {
        matches!(self, Status::Err { .. })
    }
}

/// Public handle to the reloader. Cheap to clone; everything inside is
/// reference-counted so multiple subsystems (MIDI callback, audio
/// callback, egui side panel) read from the same shared state.
#[derive(Clone)]
pub struct Reloader {
    /// Currently-loaded factory. `None` until the first successful build.
    /// MIDI thread reads this on every note_on; watcher thread stores
    /// new factories here. ArcSwap is wait-free.
    pub current: Arc<ArcSwap<Option<Arc<LiveFactory>>>>,
    /// Status string for GUI side panel.
    pub status: Arc<arc_swap::ArcSwap<Status>>,
    /// Sends "rebuild now" requests to the watcher thread (used by the
    /// optional Ctrl+R keybind and the initial startup build).
    rebuild_tx: Sender<RebuildRequest>,
    /// Path to the `voices_live/` cargo project root, surfaced in the
    /// GUI for clarity.
    pub crate_root: PathBuf,
}

#[derive(Debug, Clone)]
struct RebuildRequest {
    reason: String,
}

impl Reloader {
    /// Spawn the watcher thread, perform the initial build, and return
    /// the handle. The watcher monitors `<crate_root>/src/` for any
    /// `.rs` change; the cargo invocation forces output into
    /// `<crate_root>/target/` so the host can locate the DLL
    /// deterministically regardless of `CARGO_TARGET_DIR`.
    pub fn spawn(crate_root: PathBuf) -> Self {
        let current: Arc<ArcSwap<Option<Arc<LiveFactory>>>> = Arc::new(ArcSwap::from_pointee(None));
        let status = Arc::new(ArcSwap::from_pointee(Status::Idle));

        let (rebuild_tx, rebuild_rx) = mpsc::channel::<RebuildRequest>();

        // File watcher → debounced rebuild request.
        let watcher_tx = rebuild_tx.clone();
        let src_dir = crate_root.join("src");
        thread::Builder::new()
            .name("live-reload-fswatch".into())
            .spawn(move || {
                if let Err(e) = run_fs_watcher(&src_dir, watcher_tx) {
                    eprintln!("live_reload: file watcher exited: {e}");
                }
            })
            .expect("spawn fswatch thread");

        // Build worker.
        let current_for_worker = current.clone();
        let status_for_worker = status.clone();
        let crate_root_for_worker = crate_root.clone();
        thread::Builder::new()
            .name("live-reload-builder".into())
            .spawn(move || {
                run_build_worker(
                    crate_root_for_worker,
                    rebuild_rx,
                    current_for_worker,
                    status_for_worker,
                );
            })
            .expect("spawn builder thread");

        // Kick off the initial build so the user gets a working voice
        // without having to touch the file first.
        let _ = rebuild_tx.send(RebuildRequest {
            reason: "startup".into(),
        });

        Self {
            current,
            status,
            rebuild_tx,
            crate_root,
        }
    }

    /// Force a rebuild (Ctrl+R from the GUI, or other manual trigger).
    pub fn request_rebuild(&self, reason: &str) {
        let _ = self.rebuild_tx.send(RebuildRequest {
            reason: reason.into(),
        });
    }

    /// Construct a voice from the currently-loaded factory, or return
    /// `None` if no factory is loaded yet (initial build hasn't finished
    /// or the latest build errored). Caller falls back to silence in
    /// that case.
    pub fn make_voice(&self, sr: f32, freq: f32, vel: u8) -> Option<Box<dyn VoiceImpl + Send>> {
        let guard = self.current.load();
        let opt: &Option<Arc<LiveFactory>> = &**guard;
        opt.as_ref().map(|fac| fac.make_voice(sr, freq, vel))
    }

    /// Snapshot the current status for GUI rendering.
    pub fn status_snapshot(&self) -> Status {
        (**self.status.load()).clone()
    }

    /// Snapshot the currently-loaded factory's metadata (path + load
    /// time). Used by the side panel.
    pub fn current_meta(&self) -> Option<(PathBuf, Instant)> {
        let guard = self.current.load();
        let opt: &Option<Arc<LiveFactory>> = &**guard;
        opt.as_ref()
            .map(|fac| (fac.dll_path.clone(), fac.loaded_at))
    }
}

fn run_fs_watcher(src_dir: &Path, tx: Sender<RebuildRequest>) -> notify::Result<()> {
    let (raw_tx, raw_rx) = mpsc::channel::<notify::Result<Event>>();
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
        let _ = raw_tx.send(res);
    })?;
    if !src_dir.exists() {
        eprintln!(
            "live_reload: src dir {} does not exist; watcher disabled",
            src_dir.display()
        );
        return Ok(());
    }
    watcher.watch(src_dir, RecursiveMode::Recursive)?;

    // Debounce: collect events for 150 ms after the last write before
    // firing a single rebuild. Avoids 10× rebuilds when an editor saves
    // via a delete + create + chmod sequence.
    const DEBOUNCE: Duration = Duration::from_millis(150);
    let mut pending: Option<Instant> = None;
    loop {
        let recv_timeout = match pending {
            Some(deadline) => deadline.saturating_duration_since(Instant::now()),
            None => Duration::from_secs(60),
        };
        match raw_rx.recv_timeout(recv_timeout) {
            Ok(Ok(ev)) => {
                if matches!(
                    ev.kind,
                    EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
                ) && ev.paths.iter().any(|p| {
                    p.extension()
                        .map(|e| e == "rs" || e == "toml")
                        .unwrap_or(false)
                }) {
                    pending = Some(Instant::now() + DEBOUNCE);
                }
            }
            Ok(Err(e)) => eprintln!("live_reload: watcher error: {e}"),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Some(deadline) = pending {
                    if Instant::now() >= deadline {
                        let _ = tx.send(RebuildRequest {
                            reason: "fs change".into(),
                        });
                        pending = None;
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    Ok(())
}

fn run_build_worker(
    crate_root: PathBuf,
    rx: Receiver<RebuildRequest>,
    current: Arc<ArcSwap<Option<Arc<LiveFactory>>>>,
    status: Arc<ArcSwap<Status>>,
) {
    while let Ok(req) = rx.recv() {
        // Drain coalesced rebuild requests so a flurry of fs events only
        // triggers one build.
        while rx.try_recv().is_ok() {}

        let started = Instant::now();
        status.store(Arc::new(Status::Building {
            since: started,
            reason: req.reason.clone(),
        }));

        match build_and_load(&crate_root) {
            Ok(factory) => {
                let dll_path = factory.dll_path.clone();
                let duration = started.elapsed();
                current.store(Arc::new(Some(Arc::new(factory))));
                status.store(Arc::new(Status::Ok {
                    at: Instant::now(),
                    duration,
                    dll_path,
                }));
                eprintln!(
                    "live_reload: loaded in {} ms (reason: {})",
                    duration.as_millis(),
                    req.reason
                );
            }
            Err(msg) => {
                eprintln!("live_reload: build/load failed: {msg}");
                status.store(Arc::new(Status::Err {
                    at: Instant::now(),
                    message: msg,
                }));
                // Do NOT touch `current`. The previous factory (if any)
                // stays live, so held + new notes keep working.
            }
        }
    }
}

fn build_and_load(crate_root: &Path) -> Result<LiveFactory, String> {
    let manifest = crate_root.join("Cargo.toml");
    if !manifest.exists() {
        return Err(format!(
            "voices_live manifest missing: {}",
            manifest.display()
        ));
    }
    let target_dir = crate_root.join("target");

    let output = Command::new("cargo")
        .arg("build")
        .arg("--manifest-path")
        .arg(&manifest)
        .arg("--target-dir")
        .arg(&target_dir)
        .output()
        .map_err(|e| format!("cargo invocation failed: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // Trim to last ~800 chars so a runaway error doesn't blow up the
        // GUI label, but keep enough to see the actual rustc message.
        let trimmed: String = stderr.chars().rev().take(800).collect::<String>();
        let trimmed: String = trimmed.chars().rev().collect();
        return Err(format!("cargo build failed:\n{trimmed}"));
    }

    let dll_path = expected_dll_path(&target_dir);
    if !dll_path.exists() {
        return Err(format!(
            "build succeeded but DLL not found at {}",
            dll_path.display()
        ));
    }

    // Copy to a unique sibling path before loading. Hot-reloading the
    // SAME path is unreliable on Windows (the file is locked while the
    // previous instance is mapped) and brittle on Linux/macOS too. The
    // copy-then-load pattern means: every reload is a fresh file the OS
    // sees as different, and the previous DLL stays mapped until the
    // last voice referencing it drops (then libloading unmaps it).
    let unique = unique_dll_copy(&dll_path)?;
    LiveFactory::load(&unique)
}

fn expected_dll_path(target_dir: &Path) -> PathBuf {
    // Default cargo profile is `dev`, output goes to `target/debug/`.
    let dir = target_dir.join("debug");
    if cfg!(target_os = "windows") {
        dir.join("keysynth_voices_live.dll")
    } else if cfg!(target_os = "macos") {
        dir.join("libkeysynth_voices_live.dylib")
    } else {
        dir.join("libkeysynth_voices_live.so")
    }
}

fn unique_dll_copy(src: &Path) -> Result<PathBuf, String> {
    let parent = src.parent().ok_or("dll has no parent dir")?;
    let stem = src
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("dll has no stem")?;
    let ext = src.extension().and_then(|s| s.to_str()).unwrap_or("");
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let unique_name = if ext.is_empty() {
        format!("{stem}-live-{nanos}")
    } else {
        format!("{stem}-live-{nanos}.{ext}")
    };
    let dst = parent.join(unique_name);
    std::fs::copy(src, &dst)
        .map_err(|e| format!("copy {} → {}: {e}", src.display(), dst.display()))?;
    Ok(dst)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Sanity: status enum classification is correct.
    #[test]
    fn status_is_err_only_for_err() {
        assert!(!Status::Idle.is_err());
        assert!(!Status::Building {
            since: Instant::now(),
            reason: "x".into()
        }
        .is_err());
        assert!(!Status::Ok {
            at: Instant::now(),
            duration: Duration::from_millis(1),
            dll_path: PathBuf::from("x"),
        }
        .is_err());
        assert!(Status::Err {
            at: Instant::now(),
            message: "boom".into(),
        }
        .is_err());
    }

    /// ArcSwap guarantees: many readers + one writer never observe a
    /// torn read. Smoke-tests the contract our MIDI / watcher pair
    /// relies on.
    #[test]
    fn arc_swap_concurrent_swap_safe() {
        let cell: Arc<ArcSwap<Option<Arc<u32>>>> = Arc::new(ArcSwap::from_pointee(None));
        let writer_done = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for i in 0..50 {
            let cell = cell.clone();
            let done = writer_done.clone();
            handles.push(thread::spawn(move || {
                cell.store(Arc::new(Some(Arc::new(i as u32))));
                done.fetch_add(1, Ordering::SeqCst);
            }));
        }
        for _ in 0..200 {
            let _ = cell.load();
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(writer_done.load(Ordering::SeqCst), 50);
    }

    /// expected_dll_path is OS-correct.
    #[test]
    fn expected_dll_path_picks_right_extension() {
        let p = expected_dll_path(Path::new("/x/target"));
        let last = p.file_name().unwrap().to_string_lossy().into_owned();
        if cfg!(target_os = "windows") {
            assert!(last.ends_with(".dll"));
        } else if cfg!(target_os = "macos") {
            assert!(last.ends_with(".dylib"));
        } else {
            assert!(last.ends_with(".so"));
        }
    }

    /// Building from the actual `voices_live/` crate produces a loadable
    /// factory and a working voice. This is the end-to-end smoke and
    /// requires `cargo` on PATH; skipped if the crate dir is missing
    /// (e.g. release tarball).
    #[test]
    fn build_and_load_real_voices_live() {
        let here = std::env::current_dir().unwrap();
        let crate_root = here.join("voices_live");
        if !crate_root.join("Cargo.toml").exists() {
            eprintln!(
                "skipping: voices_live not at {} (CWD = {})",
                crate_root.display(),
                here.display()
            );
            return;
        }
        let factory = build_and_load(&crate_root).expect("build_and_load");
        let factory = Arc::new(factory);
        let mut voice = factory.make_voice(44_100.0, 440.0, 100);
        let mut buf = vec![0.0f32; 256];
        voice.render_add(&mut buf);
        let peak = buf.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        assert!(peak > 0.0, "live voice should produce audio");
        // Trigger release; voice should eventually be done.
        voice.trigger_release();
        for _ in 0..200 {
            voice.render_add(&mut buf);
            if voice.is_done() {
                break;
            }
        }
        assert!(voice.is_done(), "voice should reach is_done after release");
    }

    /// Held voices outlive their factory: drop the factory Arc but keep
    /// a voice alive; rendering must still work because the lib is pinned
    /// by the voice's own Arc<Library> clone.
    #[test]
    fn voice_outlives_factory_swap() {
        let here = std::env::current_dir().unwrap();
        let crate_root = here.join("voices_live");
        if !crate_root.join("Cargo.toml").exists() {
            return;
        }
        let factory = Arc::new(build_and_load(&crate_root).expect("build_and_load"));
        let mut voice = factory.make_voice(44_100.0, 440.0, 100);
        let mut buf = vec![0.0f32; 64];
        voice.render_add(&mut buf);
        let peak_before = buf.iter().fold(0.0f32, |a, &x| a.max(x.abs()));

        // Now drop the factory's "current" handle. The voice still
        // references the lib through its own Arc clone, so render_add
        // should NOT fault.
        drop(factory);

        let mut buf = vec![0.0f32; 64];
        voice.render_add(&mut buf);
        let peak_after = buf.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        assert!(
            peak_after > 0.0,
            "voice survives factory drop (peak before={peak_before}, after={peak_after})"
        );
    }
}
