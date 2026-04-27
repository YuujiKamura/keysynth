//! Integration tests for the voice hot-reload subsystem.
//!
//! These exercise the user-facing reload loop end-to-end:
//!   - syntax error in source → cargo build fails → previous factory
//!     keeps working, error surfaces in `Status::Err`.
//!   - good edit → rebuild → next voice produces *different* samples.
//!   - DLL deletion mid-flight → graceful handling (status surfaces,
//!     previous factory still serves note_on).
//!
//! All tests use a *throw-away copy* of the bundled `voices_live/` crate
//! under a tempdir — we don't mutate the checked-in source file because
//! test parallelism would clobber it. Each test gets its own sibling
//! crate at `target/tmp/<test-name>/voices_live`.
//!
//! Tests are skipped (with `eprintln`) if `voices_live/` isn't present
//! next to the integration binary's CWD — covers running from a release
//! tarball that didn't ship the crate.

#![cfg(feature = "native")]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use keysynth::live_reload::{LiveFactory, Reloader, Status};
use keysynth::synth::VoiceImpl;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Copy the bundled `voices_live/` crate into a unique scratch dir so a
/// test can mutate `src/lib.rs` without touching the checked-in source
/// or racing other parallel tests. Returns the scratch crate root or
/// `None` if the source crate isn't present.
fn scratch_voices_live_for(test_name: &str) -> Option<PathBuf> {
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
        .join("live-reload-tests")
        .join(format!("{test_name}-{nanos}"))
        .join("voices_live");
    copy_dir_shallow(&src, &scratch).ok()?;
    Some(scratch)
}

/// Copy `src` → `dst` recursively, creating dirs as needed. Skips
/// `target/` so we don't haul a built artifact tree along (each scratch
/// crate gets its own fresh build dir).
fn copy_dir_shallow(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        if name == "target" {
            continue;
        }
        let from = entry.path();
        let to = dst.join(&name);
        if entry.file_type()?.is_dir() {
            copy_dir_shallow(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Block until `pred(reloader.status_snapshot())` is true or `timeout`
/// elapses. Polls every 50 ms.
fn wait_for_status<F>(reloader: &Reloader, timeout: Duration, pred: F) -> Status
where
    F: Fn(&Status) -> bool,
{
    let deadline = Instant::now() + timeout;
    loop {
        let s = reloader.status_snapshot();
        if pred(&s) {
            return s;
        }
        if Instant::now() >= deadline {
            return s;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Block until a NEW factory finishes loading (its `dll_path` differs
/// from `prev_dll`) or `timeout` elapses. Used by tests that swap source
/// and need to confirm the next make_voice is from the new factory, not
/// the still-cached previous one.
fn wait_for_new_factory(reloader: &Reloader, prev_dll: Option<&Path>, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Some((dll, _)) = reloader.current_meta() {
            let differs = match prev_dll {
                Some(p) => dll.as_path() != p,
                None => true,
            };
            if differs && matches!(reloader.status_snapshot(), Status::Ok { .. }) {
                return true;
            }
        }
        if matches!(reloader.status_snapshot(), Status::Err { .. }) {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

fn render_peak(voice: &mut Box<dyn VoiceImpl + Send>, frames: usize) -> f32 {
    let mut buf = vec![0.0f32; frames];
    voice.render_add(&mut buf);
    buf.iter().fold(0.0f32, |a, &x| a.max(x.abs()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// End-to-end: spawn the reloader against a scratch crate copy, wait
/// for the initial build, build a voice, render audio, confirm peak > 0.
/// Records the elapsed time as the reload-latency baseline.
#[test]
fn end_to_end_initial_build_renders_audio() {
    let Some(crate_root) = scratch_voices_live_for("initial-build") else {
        eprintln!("skipping: voices_live/ not in CWD");
        return;
    };
    let started = Instant::now();
    let reloader = Reloader::spawn(crate_root.clone());
    let status = wait_for_status(&reloader, Duration::from_secs(60), |s| {
        matches!(s, Status::Ok { .. } | Status::Err { .. })
    });
    let elapsed = started.elapsed();
    match &status {
        Status::Ok { duration, .. } => {
            eprintln!(
                "initial-build OK: build {} ms, total {} ms",
                duration.as_millis(),
                elapsed.as_millis()
            );
        }
        other => panic!("expected Ok status, got {other:?}"),
    }
    let mut voice = reloader
        .make_voice(44_100.0, 440.0, 100)
        .expect("factory loaded");
    let peak = render_peak(&mut voice, 256);
    assert!(peak > 0.0, "live voice silent? peak = {peak}");
}

/// Syntax error in source → cargo fails → status flips to Err but the
/// previously-loaded factory keeps producing audio. The brief calls
/// silent failure the worst-case UX outcome; this test asserts the
/// failure is loud (status surfaces) AND the previous voice survives.
#[test]
fn syntax_error_keeps_previous_factory_working() {
    let Some(crate_root) = scratch_voices_live_for("syntax-error") else {
        return;
    };
    let reloader = Reloader::spawn(crate_root.clone());
    let _ = wait_for_status(&reloader, Duration::from_secs(60), |s| {
        matches!(s, Status::Ok { .. } | Status::Err { .. })
    });
    assert!(
        matches!(reloader.status_snapshot(), Status::Ok { .. }),
        "initial build must succeed, got {:?}",
        reloader.status_snapshot()
    );

    // Take a voice from the GOOD factory; we'll keep rendering through it
    // after the bad reload to confirm the lib stays mapped.
    let mut survivor = reloader
        .make_voice(44_100.0, 440.0, 100)
        .expect("good factory");
    assert!(render_peak(&mut survivor, 256) > 0.0);

    // Inject a syntax error and force a rebuild. Don't depend on the
    // file watcher firing — call request_rebuild() directly so this
    // test isn't timing-coupled to inotify / ReadDirectoryChangesW.
    let lib_rs = crate_root.join("src").join("lib.rs");
    std::fs::write(&lib_rs, "this is not rust { fn fn fn ;;;").unwrap();
    reloader.request_rebuild("test:syntax-error");

    let bad_status = wait_for_status(&reloader, Duration::from_secs(60), |s| {
        matches!(s, Status::Err { .. })
    });
    match &bad_status {
        Status::Err { message, .. } => {
            assert!(
                !message.is_empty(),
                "Status::Err message must surface cargo stderr"
            );
            eprintln!("expected error surfaced: {} chars", message.len());
        }
        other => panic!("expected Err, got {other:?}"),
    }

    // Survivor still works.
    assert!(
        render_peak(&mut survivor, 256) > 0.0,
        "previously-loaded voice silenced after bad reload"
    );

    // Restoring a valid file recovers — same trick (direct rebuild call).
    let original = std::fs::read_to_string(
        std::env::current_dir()
            .unwrap()
            .join("voices_live/src/lib.rs"),
    )
    .unwrap();
    std::fs::write(&lib_rs, &original).unwrap();
    reloader.request_rebuild("test:recover");
    let recovered = wait_for_status(&reloader, Duration::from_secs(60), |s| {
        matches!(s, Status::Ok { .. })
    });
    assert!(
        matches!(recovered, Status::Ok { .. }),
        "did not recover, last status = {recovered:?}"
    );
}

/// Replacing the source with a different sample shape produces audibly
/// different audio. Asserts the swap actually exchanges code, not just
/// pretends to.
#[test]
fn reload_swaps_in_new_dsp_code() {
    let Some(crate_root) = scratch_voices_live_for("swap-dsp") else {
        return;
    };
    let reloader = Reloader::spawn(crate_root.clone());
    let _ = wait_for_status(&reloader, Duration::from_secs(60), |s| {
        matches!(s, Status::Ok { .. })
    });
    let prev_dll = reloader.current_meta().map(|(p, _)| p);
    let mut v1 = reloader
        .make_voice(44_100.0, 440.0, 100)
        .expect("factory v1 loaded");
    let mut buf1 = vec![0.0f32; 1024];
    v1.render_add(&mut buf1);
    let signature_v1: f32 = buf1.iter().take(64).copied().sum::<f32>().abs();

    // Replace voice DSP with a constant-DC output (very obviously
    // different from the default sine voice). Use the same C-ABI surface
    // so symbol resolution still succeeds.
    let new_lib = r#"
use std::os::raw::c_void;
pub const KEYSYNTH_LIVE_ABI_VERSION: u32 = 1;
#[no_mangle] pub extern "C" fn keysynth_live_abi_version() -> u32 { 1 }
pub struct V { released: bool, n: u32 }
#[no_mangle] pub unsafe extern "C" fn keysynth_live_new(_sr: f32, _f: f32, _v: u8) -> *mut c_void {
    Box::into_raw(Box::new(V { released: false, n: 0 })) as *mut c_void
}
#[no_mangle] pub unsafe extern "C" fn keysynth_live_render_add(p: *mut c_void, buf: *mut f32, n: usize) {
    if p.is_null() || buf.is_null() || n == 0 { return; }
    let v = &mut *(p as *mut V);
    let s = std::slice::from_raw_parts_mut(buf, n);
    for x in s { *x += 0.5; v.n = v.n.saturating_add(1); }
}
#[no_mangle] pub unsafe extern "C" fn keysynth_live_trigger_release(p: *mut c_void) {
    if !p.is_null() { (*(p as *mut V)).released = true; }
}
#[no_mangle] pub unsafe extern "C" fn keysynth_live_is_done(p: *mut c_void) -> bool {
    if p.is_null() { return true; }
    (*(p as *mut V)).released && (*(p as *mut V)).n > 100_000
}
#[no_mangle] pub unsafe extern "C" fn keysynth_live_is_releasing(p: *mut c_void) -> bool {
    if p.is_null() { return false; }
    (*(p as *mut V)).released
}
#[no_mangle] pub unsafe extern "C" fn keysynth_live_drop(p: *mut c_void) {
    if !p.is_null() { drop(Box::from_raw(p as *mut V)); }
}
"#;
    let lib_rs = crate_root.join("src").join("lib.rs");
    std::fs::write(&lib_rs, new_lib).unwrap();
    reloader.request_rebuild("test:swap");
    // Wait for a *different* dll path to land — the wait_for_status
    // above would return immediately because the previous Ok status is
    // still cached, racing make_voice() against the swap.
    let swapped = wait_for_new_factory(&reloader, prev_dll.as_deref(), Duration::from_secs(60));
    assert!(
        swapped,
        "new factory never loaded; status = {:?}",
        reloader.status_snapshot()
    );

    // A NEW note_on uses the swapped factory.
    let mut v2 = reloader
        .make_voice(44_100.0, 440.0, 100)
        .expect("factory v2");
    let mut buf2 = vec![0.0f32; 1024];
    v2.render_add(&mut buf2);
    // The old (sine) voice's first 64 samples sum to a small near-zero
    // value (sine is symmetric). The new (DC=0.5) voice sums to 32.0.
    // Compare against a generous threshold instead of exact values to
    // tolerate platform float differences.
    let signature_v2: f32 = buf2.iter().take(64).copied().sum::<f32>().abs();
    assert!(
        (signature_v2 - signature_v1).abs() > 5.0,
        "v1 sig = {signature_v1}, v2 sig = {signature_v2} — DSP swap not observable"
    );

    // The held v1 voice survives — its own Arc<Library> kept the OLD
    // lib mapped even though `current` now points at v2's lib.
    let mut buf_post = vec![0.0f32; 256];
    v1.render_add(&mut buf_post);
    let peak_post = buf_post.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    assert!(peak_post > 0.0, "v1 voice silenced after swap");
}

/// Simulate a user blowing away the produced DLL between rebuilds:
/// the next request_rebuild() must surface an error (or recover by
/// rebuilding) without panicking and without dropping the previous
/// factory.
#[test]
fn missing_dll_after_load_handled_gracefully() {
    let Some(crate_root) = scratch_voices_live_for("missing-dll") else {
        return;
    };
    let reloader = Reloader::spawn(crate_root.clone());
    let _ = wait_for_status(&reloader, Duration::from_secs(60), |s| {
        matches!(s, Status::Ok { .. })
    });
    assert!(matches!(reloader.status_snapshot(), Status::Ok { .. }));
    let mut v = reloader.make_voice(44_100.0, 440.0, 100).expect("factory");
    assert!(render_peak(&mut v, 256) > 0.0);

    // Wipe the build dir and force rebuild. cargo will rebuild from
    // source, so we expect Ok again — the test is really just confirming
    // we don't panic mid-process.
    let target = crate_root.join("target");
    if target.exists() {
        let _ = std::fs::remove_dir_all(&target);
    }
    reloader.request_rebuild("test:missing-dll");
    let final_status = wait_for_status(&reloader, Duration::from_secs(60), |s| {
        matches!(s, Status::Ok { .. } | Status::Err { .. })
    });
    eprintln!("after target wipe: {final_status:?}");
    // Survivor still renders (Arc<Library> kept the old DLL pinned).
    assert!(render_peak(&mut v, 256) > 0.0);
}

/// Direct LiveFactory::load against a non-existent path returns Err
/// without touching disk-locked state. Cheap negative-path coverage that
/// doesn't depend on cargo.
#[test]
fn load_nonexistent_dll_errors() {
    let res = LiveFactory::load(Path::new("does-not-exist-anywhere.dll"));
    assert!(res.is_err());
}

/// Held a factory's Arc, drop it after a swap, render → must still work
/// because the voice pins its own clone. (Mirrors the unit test in
/// `live_reload::tests::voice_outlives_factory_swap` but goes through
/// the public Reloader API.)
#[test]
fn arc_pinning_survives_reloader_swap() {
    let Some(crate_root) = scratch_voices_live_for("arc-pinning") else {
        return;
    };
    let reloader = Reloader::spawn(crate_root.clone());
    let _ = wait_for_status(&reloader, Duration::from_secs(60), |s| {
        matches!(s, Status::Ok { .. })
    });
    let prev_dll = reloader.current_meta().map(|(p, _)| p);
    let mut held = reloader
        .make_voice(44_100.0, 440.0, 100)
        .expect("factory v1");
    // Trigger a fresh build so `current` points at a NEW factory; the
    // old one's Arc<Library> should drop to refcount-1 (pinned by held).
    let lib_rs = crate_root.join("src").join("lib.rs");
    let original = std::fs::read_to_string(&lib_rs).unwrap();
    std::fs::write(&lib_rs, format!("// touch\n{original}")).unwrap();
    reloader.request_rebuild("test:second-build");
    let swapped = wait_for_new_factory(&reloader, prev_dll.as_deref(), Duration::from_secs(60));
    assert!(swapped, "second build never produced new factory");
    // Drop the reloader entirely → the OLD lib's Arc clone we have via
    // `held` is now the last reference. Rendering must still work.
    drop(reloader);
    let mut buf = vec![0.0f32; 256];
    held.render_add(&mut buf);
    let peak = buf.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    assert!(peak > 0.0, "held voice silenced after reloader drop");
    // Now drop held → OLD lib unmaps cleanly (no UB).
    drop(held);
}

/// Reload latency budget: assert the SECOND build (warm cache, no source
/// change beyond a touch) finishes within 5 seconds. The brief targets
/// < 1 s; we use a much looser bound here so CI on cold machines doesn't
/// flake. Locally on a warm cache it's 200–800 ms (logged).
#[test]
fn reload_latency_within_budget() {
    let Some(crate_root) = scratch_voices_live_for("latency") else {
        return;
    };
    let reloader = Reloader::spawn(crate_root.clone());
    let _ = wait_for_status(&reloader, Duration::from_secs(120), |s| {
        matches!(s, Status::Ok { .. })
    });
    let lib_rs = crate_root.join("src").join("lib.rs");
    let original = std::fs::read_to_string(&lib_rs).unwrap();
    let prev_dll = reloader.current_meta().map(|(p, _)| p);
    let started = Instant::now();
    // Trivial whitespace edit → cargo incremental should be near-no-op.
    std::fs::write(&lib_rs, format!("{original}\n// edit\n")).unwrap();
    reloader.request_rebuild("test:latency");
    let swapped = wait_for_new_factory(&reloader, prev_dll.as_deref(), Duration::from_secs(30));
    let elapsed = started.elapsed();
    eprintln!(
        "warm-cache reload: {} ms (status: {:?})",
        elapsed.as_millis(),
        reloader.status_snapshot()
    );
    assert!(
        swapped,
        "warm reload failed: {:?}",
        reloader.status_snapshot()
    );
    // Generous budget: the brief asks for < 1 s on a warm cache, but
    // CI containers can be slow. 5 s catches genuine regressions
    // (something making each reload re-link from scratch, etc.).
    assert!(
        elapsed < Duration::from_secs(5),
        "warm-cache reload too slow: {} ms",
        elapsed.as_millis()
    );
}
