//! Voice browser <-> CP slot pool synchronisation regression.
//!
//! Asserts that `VoiceLibrary::refresh_cp_slots` produces the
//! browser state the CP-GUI bridge promises:
//!
//!   1. ksctl build piano  → browser gets a "piano" entry under
//!      `Category::CpSlot`, no other entries change.
//!   2. ksctl unload piano → "piano" entry disappears.
//!   3. The `default` watcher slot is filtered out (the browser
//!      already has a "Live (hot edit)" Custom-category entry for
//!      that one).
//!   4. Calling refresh repeatedly with the same input is
//!      idempotent — slot count stays stable, active index doesn't
//!      drift.
//!   5. If the active slot is unloaded, the active index falls
//!      back gracefully without panicking.
//!
//! GUI rendering itself isn't testable here (eframe needs a
//! windowing context), but every state change the browser would
//! display flows through `VoiceLibrary::slots`, which IS testable.
//! That's the boundary the brief calls out: "library.slots の中身
//! を assert で確認すれば足りる".

#![cfg(feature = "native")]

use keysynth::synth::Engine;
use keysynth::voice_lib::{Category, VoiceLibrary, VoiceSlot};

fn names_in_category(lib: &VoiceLibrary, cat: Category) -> Vec<String> {
    lib.slots
        .iter()
        .filter(|s| s.category == cat)
        .map(|s| s.label.clone())
        .collect()
}

fn fresh_library() -> VoiceLibrary {
    // Builtins only; no persisted user entries (HOME-relative load
    // would pull in dev-machine state we don't want in a regression).
    VoiceLibrary {
        slots: VoiceLibrary::load(None, None, Engine::Square).slots,
        active: 0,
    }
}

#[test]
fn cp_slot_appears_when_reloader_reports_one() {
    let mut lib = fresh_library();
    let before = lib.slots.len();
    assert!(
        names_in_category(&lib, Category::CpSlot).is_empty(),
        "fresh library should have zero CP slots"
    );

    let names = vec!["piano".to_string()];
    let (added, removed) = lib.refresh_cp_slots(&names);
    assert_eq!((added, removed), (1, 0));
    let cp = names_in_category(&lib, Category::CpSlot);
    assert_eq!(cp, vec!["piano".to_string()]);
    // Builtins must be preserved.
    assert_eq!(lib.slots.len(), before + 1);
}

#[test]
fn cp_slot_disappears_when_unloaded() {
    let mut lib = fresh_library();
    let _ = lib.refresh_cp_slots(&["piano".to_string()]);
    assert_eq!(names_in_category(&lib, Category::CpSlot).len(), 1);

    let (added, removed) = lib.refresh_cp_slots(&[]);
    assert_eq!((added, removed), (0, 1));
    assert!(names_in_category(&lib, Category::CpSlot).is_empty());
}

#[test]
fn default_watcher_slot_is_filtered() {
    // The watcher's own DEFAULT_SLOT shows up in Reloader::list_slots()
    // alongside CP-loaded slots, but the browser already has a "Live
    // (hot edit)" Custom entry for it. refresh_cp_slots must not
    // duplicate it.
    let mut lib = fresh_library();
    let names = vec!["default".to_string(), "piano".to_string()];
    let (added, _) = lib.refresh_cp_slots(&names);
    assert_eq!(added, 1, "only piano should be added; default is filtered");
    let cp = names_in_category(&lib, Category::CpSlot);
    assert_eq!(cp, vec!["piano".to_string()]);
}

#[test]
fn refresh_is_idempotent() {
    let mut lib = fresh_library();
    let names = vec!["piano".to_string(), "piano_modal".to_string()];
    lib.refresh_cp_slots(&names);
    let snapshot: Vec<String> = lib.slots.iter().map(|s| s.label.clone()).collect();
    let active = lib.active;

    // Repeating with the same input must not change anything.
    let (added, removed) = lib.refresh_cp_slots(&names);
    assert_eq!((added, removed), (0, 0));
    let after: Vec<String> = lib.slots.iter().map(|s| s.label.clone()).collect();
    assert_eq!(snapshot, after);
    assert_eq!(active, lib.active);
}

#[test]
fn unloading_active_cp_slot_clamps_active_index() {
    let mut lib = fresh_library();
    lib.refresh_cp_slots(&["piano".to_string(), "piano_modal".to_string()]);
    let piano_idx = lib
        .slots
        .iter()
        .position(|s| s.category == Category::CpSlot && s.label == "piano")
        .unwrap();
    lib.active = piano_idx;

    // Unload piano. Active was pointing at it; expect a graceful
    // fallback without panic.
    lib.refresh_cp_slots(&["piano_modal".to_string()]);
    assert!(lib.active < lib.slots.len(), "active index out of bounds");
    let names: Vec<String> = lib.slots.iter().map(|s| s.label.clone()).collect();
    assert!(
        !names.iter().any(|n| n == "piano"),
        "stale piano entry survived unload: {names:?}"
    );
}

#[test]
fn cp_slot_constructor_targets_engine_live() {
    let s = VoiceSlot::cp_slot("piano_thick");
    assert_eq!(s.category, Category::CpSlot);
    assert_eq!(s.engine, Engine::Live);
    assert_eq!(s.label, "piano_thick");
    assert!(
        s.builtin,
        "CP slots must be marked builtin (no rename / remove via GUI)"
    );
}
