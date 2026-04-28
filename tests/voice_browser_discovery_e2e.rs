//! Discovery contract for the GUI voice browser. Asserts the
//! plugin-substrate promise: a new voice plugin appears in the side
//! panel by landing a `voices_live/<name>/Cargo.toml` with a
//! `[package.metadata.keysynth-voice]` block — and only that. No edits
//! to `voice_lib.rs`, `ui.rs`, the `Engine` enum, or top-level Cargo
//! deps are required.
//!
//! Three gates:
//!
//!   1. **Repo plugins are discovered** — scanning the real
//!      `voices_live/` tree returns at least the seven Cargo.toml
//!      files we ship with metadata blocks today (5 piano variants
//!      + 2 guitar variants). Each has a non-empty display name,
//!      lands in the spec-mandated category (`Piano` / `Guitar`),
//!      and carries a `live:<slot>` engine slot reference.
//!
//!   2. **Editorial tiers survive the round trip** — `Piano` and
//!      `Piano Modal` come back as `Recommend::Best`, `Guitar (STK)`
//!      as `Best`, `Guitar (KS)` as `Experimental`. This is the
//!      single editorial signal the spec calls out and the test
//!      guards against an accidental flip in either direction.
//!
//!   3. **Discovery is symmetric under directory churn** — drop a
//!      throwaway `voices_live/<rand>/Cargo.toml` with the metadata
//!      block, re-scan, and the entry shows up; remove the directory,
//!      re-scan, and it's gone. Proves discovery is genuinely driven
//!      by the filesystem rather than a hidden hard-coded list.
//!
//! Audio source files are intentionally untouched by this PR — the
//! test never instantiates a voice, only inspects catalog metadata.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use keysynth::voice_lib::{discover_plugin_voices, Category, Recommend, VoiceSlot};

fn voices_live_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is the keysynth crate root at test time, so
    // joining `voices_live/` lands on the real plugin tree the
    // checked-in metadata blocks live in.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("voices_live")
}

fn find<'a>(slots: &'a [VoiceSlot], label: &str) -> &'a VoiceSlot {
    slots
        .iter()
        .find(|s| s.label == label)
        .unwrap_or_else(|| panic!("expected discovered slot '{label}', got {slots:#?}"))
}

#[test]
fn repo_plugin_set_is_discovered() {
    let root = voices_live_root();
    assert!(
        root.is_dir(),
        "voices_live/ not found at {} — test must run from the keysynth checkout",
        root.display(),
    );
    let slots = discover_plugin_voices(&root);

    // The seven plugins shipping with metadata as of this PR.
    let required = [
        "Piano",
        "Piano Modal",
        "Piano Thick",
        "Piano Lite",
        "Piano 5AM",
        "Guitar (STK)",
        "Guitar (KS)",
    ];
    for label in required {
        let slot = find(&slots, label);
        assert!(
            !slot.description.is_empty(),
            "discovered slot '{label}' has empty description",
        );
        assert!(
            slot.engine_slot_ref().starts_with("live:"),
            "discovered slot '{label}' must carry a live:<name> engine ref, got {}",
            slot.engine_slot_ref(),
        );
    }

    // Category routing must follow the metadata `category` field.
    for label in [
        "Piano",
        "Piano Modal",
        "Piano Thick",
        "Piano Lite",
        "Piano 5AM",
    ] {
        assert_eq!(
            find(&slots, label).category,
            Category::Piano,
            "{label} did not land in Category::Piano",
        );
    }
    for label in ["Guitar (STK)", "Guitar (KS)"] {
        assert_eq!(
            find(&slots, label).category,
            Category::Guitar,
            "{label} did not land in Category::Guitar",
        );
    }
}

#[test]
fn editorial_tiers_survive_round_trip() {
    let slots = discover_plugin_voices(&voices_live_root());

    assert_eq!(find(&slots, "Piano").recommend, Recommend::Best);
    assert_eq!(find(&slots, "Piano Modal").recommend, Recommend::Best);
    assert_eq!(find(&slots, "Guitar (STK)").recommend, Recommend::Best);
    assert_eq!(
        find(&slots, "Guitar (KS)").recommend,
        Recommend::Experimental,
    );

    // STK port must land on `live:guitar_stk`; the legacy crate must
    // override its dir-derived slot to `live:guitar_ks` so the two
    // coexist unambiguously.
    assert_eq!(
        find(&slots, "Guitar (STK)").engine_slot_ref(),
        "live:guitar_stk"
    );
    assert_eq!(
        find(&slots, "Guitar (KS)").engine_slot_ref(),
        "live:guitar_ks"
    );
}

#[test]
fn discovery_tracks_directory_churn() {
    // Use a unique sibling directory so multiple test runs don't
    // clobber each other and so a panic in the middle leaves the
    // checked-in tree untouched. We deliberately do NOT use the real
    // `voices_live/` root for this gate — adding an entry there
    // would corrupt the previous test's expectations under cargo's
    // parallel test runner.
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let scratch = std::env::temp_dir().join(format!("ks_voice_disco_{nanos}_{n}"));
    fs::create_dir_all(&scratch).expect("scratch root create");

    // Sentinel plugin that lives only for this test.
    let plugin = scratch.join("test_dummy");
    fs::create_dir_all(&plugin).expect("plugin dir create");
    fs::write(
        plugin.join("Cargo.toml"),
        r#"[package]
name = "ks_disco_test_dummy"
version = "0.0.0"
edition = "2021"
publish = false

[package.metadata.keysynth-voice]
display_name = "Disco Test Dummy"
category = "Synth"
recommend = "Experimental"
description = "fixture for voice_browser_discovery_e2e"
"#,
    )
    .expect("write Cargo.toml");

    // Also drop a sibling dir without the metadata block to prove the
    // scanner doesn't sweep up arbitrary Cargo crates.
    let opaque = scratch.join("opaque_crate");
    fs::create_dir_all(&opaque).expect("opaque dir create");
    fs::write(
        opaque.join("Cargo.toml"),
        r#"[package]
name = "ks_disco_test_opaque"
version = "0.0.0"
edition = "2021"
publish = false
"#,
    )
    .expect("write opaque Cargo.toml");

    let with_dummy = discover_plugin_voices(&scratch);
    assert!(
        with_dummy.iter().any(|s| s.label == "Disco Test Dummy"),
        "discovery missed the new plugin in {}: {with_dummy:#?}",
        scratch.display(),
    );
    let dummy = with_dummy
        .iter()
        .find(|s| s.label == "Disco Test Dummy")
        .unwrap();
    assert_eq!(dummy.category, Category::Synth);
    assert_eq!(dummy.recommend, Recommend::Experimental);
    assert_eq!(dummy.engine_slot_ref(), "live:test_dummy");
    assert!(
        with_dummy.iter().all(|s| s.label != "ks_disco_test_opaque"),
        "discovery picked up a Cargo crate without the keysynth-voice metadata",
    );

    // Removing the plugin dir must also remove the entry — discovery
    // is purely a filesystem read, not a sticky cache.
    fs::remove_dir_all(&plugin).expect("plugin dir remove");
    let without_dummy = discover_plugin_voices(&scratch);
    assert!(
        without_dummy.iter().all(|s| s.label != "Disco Test Dummy"),
        "discovery still surfacing 'Disco Test Dummy' after directory removal: {without_dummy:#?}",
    );

    // Best-effort cleanup; ignored on Windows if a slow indexer holds
    // a handle.
    let _ = fs::remove_dir_all(&scratch);
}
