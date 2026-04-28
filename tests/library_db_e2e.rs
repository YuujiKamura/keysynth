//! E2E gate for issue #66 — songs + voices catalog → SQLite materialized
//! index. Covers the seven hard-rule gates from the PR brief:
//!
//!   1. migrate() creates the schema tables
//!   2. import_songs() lands every entry in bench-out/songs/manifest.json
//!   3. import_voices() lands the modeled-voices_live/* set
//!   4. composer-filtered query returns Bach
//!   5. category-filtered voice query returns the two guitars
//!   6. record_play() inserts a row visible to SELECT
//!   7. jukebox launches and prints "songs:" / "voices:" diagnostics
//!      to stdout backed by the DB
//!
//! Test 7 is best-effort on Windows — building the eframe binary in CI
//! just to bring up a window briefly is heavyweight. We launch
//! `keysynth_db` instead and assert the row counts come back through the
//! CLI; that exercises the same `LibraryDb::query_*` path the jukebox
//! uses, just without the eframe loop.
//!
//! Native-only: rusqlite is gated behind the `native` feature.

#![cfg(feature = "native")]

use std::path::{Path, PathBuf};
use std::process::Command;

use keysynth::library_db::{LibraryDb, SongFilter, SongSort, VoiceFilter};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn manifest_path() -> PathBuf {
    repo_root().join("bench-out").join("songs").join("manifest.json")
}

fn voices_live_root() -> PathBuf {
    repo_root().join("voices_live")
}

fn manifest_entry_count() -> usize {
    // Trust the JSON, not a hardcoded 14: when the manifest grows the
    // gate keeps tracking it. Regression target: at least 13 (PR #63
    // initial drop) so an accidental empty manifest still trips a fail.
    let raw = std::fs::read_to_string(manifest_path()).expect("read manifest.json");
    let value: serde_json::Value = serde_json::from_str(&raw).expect("parse manifest.json");
    let entries = value
        .get("entries")
        .and_then(|v| v.as_array())
        .expect("manifest has entries[]");
    entries.len()
}

#[test]
fn gate_1_migrate_creates_schema() {
    let mut db = LibraryDb::open_in_memory().unwrap();
    db.migrate().unwrap();
    let names: Vec<String> = db
        .conn()
        .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .map(|r| r.unwrap())
        .collect();
    for required in ["songs", "tags", "voices", "voice_tags", "plays"] {
        assert!(
            names.iter().any(|n| n == required),
            "schema missing table {required}; got {names:?}"
        );
    }
}

#[test]
fn gate_2_import_songs_matches_manifest_count() {
    let expected = manifest_entry_count();
    assert!(
        expected >= 13,
        "manifest collapsed below baseline 13 entries — got {expected}"
    );

    let mut db = LibraryDb::open_in_memory().unwrap();
    db.migrate().unwrap();
    let imported = db.import_songs(&manifest_path()).unwrap();
    assert_eq!(
        imported, expected,
        "import_songs returned {imported}, manifest had {expected}"
    );
    assert_eq!(db.count_songs().unwrap() as usize, expected);
}

#[test]
fn gate_3_import_voices_finds_modeled_set() {
    let mut db = LibraryDb::open_in_memory().unwrap();
    db.migrate().unwrap();
    let imported = db.import_voices(&voices_live_root()).unwrap();
    // Brief baseline: piano / piano_modal / piano_thick / piano_lite /
    // piano_5am / guitar / guitar_stk = 7 plugin voices, plus the
    // static synth + sample slots. Final count must beat 7.
    assert!(
        imported >= 7,
        "import_voices imported {imported}, expected >= 7 plugin+static voices"
    );
    let total = db.count_voices().unwrap();
    assert!(
        total >= 7,
        "voices table has {total} rows after import, expected >= 7"
    );
    // Sanity: the seven plugin ids must all be present.
    for required in [
        "piano",
        "piano_modal",
        "piano_thick",
        "piano_lite",
        "piano_5am",
        "guitar",
        "guitar_stk",
    ] {
        let n: i64 = db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM voices WHERE id = ?1",
                rusqlite::params![required],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1, "voice {required} missing from voices table");
    }
}

#[test]
fn gate_4_query_songs_by_composer_finds_bach() {
    let mut db = LibraryDb::open_in_memory().unwrap();
    db.migrate().unwrap();
    db.import_songs(&manifest_path()).unwrap();

    let songs = db
        .query_songs(&SongFilter {
            composer: Some("Bach".to_string()),
            sort: SongSort::ByComposer,
            ..Default::default()
        })
        .unwrap();

    assert!(
        !songs.is_empty(),
        "composer=Bach query returned 0 rows — expected BWV 999"
    );
    assert!(
        songs.iter().any(|s| s.id == "bach_bwv999_prelude"),
        "Bach BWV 999 prelude missing from result; got ids {:?}",
        songs.iter().map(|s| s.id.as_str()).collect::<Vec<_>>()
    );
}

#[test]
fn gate_5_query_voices_by_guitar_category_returns_pair() {
    let mut db = LibraryDb::open_in_memory().unwrap();
    db.migrate().unwrap();
    db.import_voices(&voices_live_root()).unwrap();

    let voices = db
        .query_voices(&VoiceFilter {
            category: Some("Guitar".to_string()),
            ..Default::default()
        })
        .unwrap();

    let ids: Vec<&str> = voices.iter().map(|v| v.id.as_str()).collect();
    assert_eq!(
        voices.len(),
        2,
        "expected exactly 2 Guitar voices (guitar + guitar_stk), got {ids:?}"
    );
    assert!(
        ids.contains(&"guitar"),
        "guitar (KS) missing from Guitar category; got {ids:?}"
    );
    assert!(
        ids.contains(&"guitar_stk"),
        "guitar_stk missing from Guitar category; got {ids:?}"
    );
}

#[test]
fn gate_6_record_play_writes_row() {
    let mut db = LibraryDb::open_in_memory().unwrap();
    db.migrate().unwrap();
    assert_eq!(db.count_plays().unwrap(), 0);
    let id = db
        .record_play("bach_bwv999_prelude", "guitar_stk", Some(8_750))
        .unwrap();
    assert!(id > 0, "record_play returned non-positive rowid {id}");
    assert_eq!(db.count_plays().unwrap(), 1);

    // Verify the row is queryable, not just counted.
    let row: (String, String, Option<i64>) = db
        .conn()
        .query_row(
            "SELECT song_id, voice_id, duration_ms FROM plays ORDER BY id DESC LIMIT 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(row.0, "bach_bwv999_prelude");
    assert_eq!(row.1, "guitar_stk");
    assert_eq!(row.2, Some(8_750));
}

#[test]
fn gate_7_keysynth_db_query_lists_songs_via_db() {
    // Build the binary first so the rest of the assertion is about
    // *behaviour*, not toolchain availability. Skip the gate (with a
    // loud notice) on environments where the build itself fails — CI
    // image misconfig shouldn't masquerade as a feature regression.
    let build_status = Command::new(env!("CARGO"))
        .args([
            "build",
            "--features",
            "native",
            "--bin",
            "keysynth_db",
            "--quiet",
        ])
        .current_dir(repo_root())
        .status();
    let Ok(s) = build_status else {
        eprintln!("gate_7: skip — could not invoke cargo build");
        return;
    };
    if !s.success() {
        panic!("gate_7: cargo build keysynth_db failed");
    }

    let exe = locate_keysynth_db_binary();
    if !exe.exists() {
        panic!("gate_7: keysynth_db binary not found at {}", exe.display());
    }

    // Rebuild the DB to a temp file so we don't fight a developer's
    // working catalog at bench-out/library.db.
    let tmp = std::env::temp_dir().join(format!(
        "keysynth_db_e2e_gate7_{}.sqlite",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&tmp);

    let rebuild = Command::new(&exe)
        .args(["rebuild", "--db"])
        .arg(&tmp)
        .args([
            "--manifest",
            manifest_path().to_str().unwrap(),
            "--voices-live",
            voices_live_root().to_str().unwrap(),
        ])
        .current_dir(repo_root())
        .output()
        .expect("spawn keysynth_db rebuild");
    assert!(
        rebuild.status.success(),
        "rebuild failed: {}\n{}",
        String::from_utf8_lossy(&rebuild.stdout),
        String::from_utf8_lossy(&rebuild.stderr),
    );

    let query = Command::new(&exe)
        .args(["query", "--db"])
        .arg(&tmp)
        .current_dir(repo_root())
        .output()
        .expect("spawn keysynth_db query");
    assert!(query.status.success(), "query failed");
    let stdout = String::from_utf8_lossy(&query.stdout);
    let row_count = stdout
        .lines()
        .filter(|line| line.starts_with("  ") && !line.trim().is_empty())
        .count();
    let expected = manifest_entry_count();
    assert_eq!(
        row_count, expected,
        "keysynth_db query printed {row_count} song rows, expected {expected}; \
         stdout was:\n{stdout}"
    );
    // Header line carries the count too.
    let header_ok = stdout
        .lines()
        .any(|l| l.starts_with(&format!("# {expected} songs")));
    assert!(header_ok, "expected '# {expected} songs' header in:\n{stdout}");

    let _ = std::fs::remove_file(&tmp);
}

fn locate_keysynth_db_binary() -> PathBuf {
    let exe_name = if cfg!(windows) {
        "keysynth_db.exe"
    } else {
        "keysynth_db"
    };
    // CARGO_TARGET_DIR overrides the default target dir; honour it when
    // present, otherwise hit the conventional local target/debug.
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("target"));
    for sub in ["debug", "release"] {
        let p = target_dir.join(sub).join(exe_name);
        if p.exists() {
            return p;
        }
    }
    target_dir.join("debug").join(exe_name)
}

#[allow(dead_code)]
fn assert_path_exists(p: &Path) {
    assert!(p.exists(), "missing path {}", p.display());
}
