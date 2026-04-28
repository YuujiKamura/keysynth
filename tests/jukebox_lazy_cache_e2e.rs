//! End-to-end gate for `preview_cache` (issue #62 Phase 2).
//!
//! Six gates from the dispatch brief:
//!
//! 1. **catalog scan**: a directory containing `*.mid` files yields
//!    one cache key per file (no scan-time render).
//! 2. **lazy nature**: list lookup with no prior store creates zero
//!    cache entries on disk.
//! 3. **on-demand render**: explicit `Cache::store` for one (song,
//!    voice) writes exactly one WAV under `cache_dir/<hash>.wav`.
//! 4. **cache hit speed**: a second `lookup` for the same key
//!    completes in < 50 ms wall-clock.
//! 5. **invalidation**: touching the voice .dll file (mtime bump)
//!    changes the cache key hash for the same (song, voice) pair.
//! 6. **eviction**: when total entry size exceeds the cap,
//!    `evict_lru` drops the oldest until ≤ cap.
//!
//! These cover the Issue #62 spec at the unit level. The jukebox GUI
//! integration (cache lookup before playback, background render
//! thread) is exercised by manual smoke tests recorded in the PR
//! description; tests beyond this file would require driving an egui
//! app headlessly, which is out of scope for Phase 2.

#![cfg(feature = "native")]

use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

use keysynth::preview_cache::{Cache, CacheKey, RenderParams};

/// Disposable test directory under the OS temp tree. Per-test
/// uniqueness via process id + nanosecond timestamp; the OS cleans
/// up at reboot so tests do not bother with explicit teardown.
fn tempdir(label: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "keysynth-jukebox-lazy-{}-{}-{}",
        label,
        std::process::id(),
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    ));
    fs::create_dir_all(&p).unwrap();
    p
}

fn touch(path: &PathBuf, bytes: &[u8]) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, bytes).unwrap();
}

/// Three valid-looking SMF files representing a "catalog" the GUI
/// would scan. Content is just enough to convince the file system
/// they exist and have content; the cache module never parses the
/// bytes.
fn write_minimal_smf(path: &PathBuf, payload: &[u8]) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"MThd");
    bytes.extend_from_slice(&6u32.to_be_bytes());
    bytes.extend_from_slice(&1u16.to_be_bytes()); // format
    bytes.extend_from_slice(&1u16.to_be_bytes()); // tracks
    bytes.extend_from_slice(&480u16.to_be_bytes()); // ppq
    bytes.extend_from_slice(b"MTrk");
    bytes.extend_from_slice(&((payload.len() + 3) as u32).to_be_bytes());
    bytes.extend_from_slice(payload);
    bytes.extend_from_slice(&[0x00, 0xff, 0x2f, 0x00][1..]);
    touch(path, &bytes);
}

fn make_key(song: PathBuf, voice_id: &str, voice_dll: Option<PathBuf>) -> CacheKey {
    CacheKey {
        song_path: song,
        voice_id: voice_id.to_string(),
        voice_dll,
        render_params: RenderParams::default(),
    }
}

#[test]
fn gate_1_catalog_scan_yields_one_key_per_song() {
    let dir = tempdir("catalog-scan");
    let songs_dir = dir.join("songs");
    let songs = ["song_a.mid", "song_b.mid", "song_c.mid"];
    for s in &songs {
        write_minimal_smf(&songs_dir.join(s), s.as_bytes());
    }

    // The GUI catalog scan emits one CacheKey per (song, voice) pair.
    // We model that here: gather the song paths, build cache keys
    // through the public API, and verify each yields a distinct hash.
    let voice_id = "guitar-stk";
    let mut hashes = Vec::new();
    for s in &songs {
        let k = make_key(songs_dir.join(s), voice_id, None);
        hashes.push(k.hash().expect("hash"));
    }
    assert_eq!(hashes.len(), 3);
    let unique: std::collections::HashSet<_> = hashes.iter().collect();
    assert_eq!(unique.len(), 3, "each song must hash distinctly");
}

#[test]
fn gate_2_lazy_nature_scan_creates_no_cache_entries() {
    let dir = tempdir("lazy");
    let cache_dir = dir.join("cache");
    let cache = Cache::new(&cache_dir, 1024 * 1024).unwrap();

    // Scan-style: build many keys, lookup each, store nothing. After
    // this loop the cache directory should be empty (only the dir
    // itself exists; no `.wav` files).
    for i in 0..10 {
        let song = dir.join(format!("song_{i}.mid"));
        write_minimal_smf(&song, format!("song-{i}").as_bytes());
        let k = make_key(song, "guitar-stk", None);
        let hit = cache.lookup(&k).unwrap();
        assert!(hit.is_none(), "cold cache should miss");
    }

    let mut wav_count = 0;
    for ent in fs::read_dir(&cache_dir).unwrap() {
        let p = ent.unwrap().path();
        if p.extension().and_then(|s| s.to_str()) == Some("wav") {
            wav_count += 1;
        }
    }
    assert_eq!(wav_count, 0, "list-style scan must not render");
    assert_eq!(cache.current_size_bytes().unwrap(), 0);
}

#[test]
fn gate_3_explicit_store_creates_one_cache_file() {
    let dir = tempdir("on-demand");
    let cache = Cache::new(dir.join("cache"), 1024 * 1024).unwrap();
    let song = dir.join("piece.mid");
    write_minimal_smf(&song, b"piece-bytes");
    let key = make_key(song, "guitar-stk", None);

    let wav = b"RIFF\x24\x00\x00\x00WAVEfmt synthetic-render";
    let stored = cache.store(&key, wav).unwrap();
    assert!(stored.is_file());
    assert_eq!(fs::read(&stored).unwrap(), wav);

    // Exactly one .wav file should now live in the cache dir.
    let count = fs::read_dir(cache.dir())
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("wav"))
        .count();
    assert_eq!(count, 1, "on-demand store should produce exactly one wav");
}

#[test]
fn gate_4_cache_hit_under_50_ms() {
    let dir = tempdir("hit");
    let cache = Cache::new(dir.join("cache"), 1024 * 1024).unwrap();
    let song = dir.join("piece.mid");
    write_minimal_smf(&song, b"piece-bytes");
    let key = make_key(song, "guitar-stk", None);

    // Prime cache.
    cache.store(&key, &vec![0u8; 4096]).unwrap();

    // Measure 10 sequential lookups; even on a slow CI runner the
    // average should be well under 5 ms. We assert ≤ 50 ms per
    // lookup (5x slack) so the gate is non-flaky on shared CI.
    let mut max_ms: u128 = 0;
    for _ in 0..10 {
        let t0 = Instant::now();
        let hit = cache.lookup(&key).unwrap();
        let elapsed_ms = t0.elapsed().as_millis();
        assert!(hit.is_some(), "primed cache must hit");
        if elapsed_ms > max_ms {
            max_ms = elapsed_ms;
        }
    }
    assert!(max_ms < 50, "cache hit took {max_ms} ms, expected < 50 ms");
}

#[test]
fn gate_5_voice_dll_mtime_change_invalidates_key() {
    let dir = tempdir("invalidate");
    let song = dir.join("piece.mid");
    let dll = dir.join("voice.dll");
    write_minimal_smf(&song, b"piece-bytes");
    touch(&dll, b"dll-v1");

    let key1 = make_key(song.clone(), "guitar-stk", Some(dll.clone()));
    let h1 = key1.hash().unwrap();

    // Bump the dll's mtime without changing content. A 1.1 second
    // sleep covers the 1-second mtime granularity that some
    // filesystems (FAT-derived, network mounts) round to.
    std::thread::sleep(Duration::from_millis(1100));
    fs::OpenOptions::new()
        .write(true)
        .open(&dll)
        .unwrap()
        .set_modified(SystemTime::now())
        .unwrap();

    let key2 = make_key(song, "guitar-stk", Some(dll));
    let h2 = key2.hash().unwrap();
    assert_ne!(h1, h2, "voice .dll mtime change must invalidate cache key");
}

#[test]
fn gate_6_lru_eviction_drops_oldest_until_under_cap() {
    let dir = tempdir("evict");
    // 3 KiB cap, three 2 KiB entries → after writes total = 6 KiB
    // → expect two evictions (oldest two).
    let cache = Cache::new(dir.join("cache"), 3 * 1024).unwrap();
    let song = dir.join("piece.mid");
    write_minimal_smf(&song, b"piece-bytes");

    let mut keys = Vec::new();
    for i in 0..3 {
        let k = make_key(song.clone(), &format!("voice-{i}"), None);
        cache.store(&k, &vec![0u8; 2048]).unwrap();
        keys.push(k);
        // Distinct mtimes per entry so LRU sort is well-defined.
        std::thread::sleep(Duration::from_millis(1100));
    }
    assert!(cache.current_size_bytes().unwrap() >= 6 * 1024);

    let evicted = cache.evict_lru().unwrap();
    assert!(evicted >= 2, "expected ≥ 2 evictions, got {evicted}");
    assert!(
        cache.current_size_bytes().unwrap() <= 3 * 1024,
        "size after eviction must be at-or-under cap"
    );
    // Most-recently-stored key (index 2) should survive.
    assert!(
        cache.lookup(&keys[2]).unwrap().is_some(),
        "newest entry must survive LRU eviction"
    );
}

#[test]
fn smoke_render_to_cache_uses_cache_hit_path() {
    // This is a non-spec smoke that proves the `render_to_cache`
    // wrapper short-circuits on a primed cache without spawning a
    // subprocess. We use an obviously-bogus binary path; if the
    // wrapper actually tried to invoke it the test would fail with
    // an OS error. The pre-stored entry should be returned instead.
    let dir = tempdir("render-shortcircuit");
    let cache = Cache::new(dir.join("cache"), 1024 * 1024).unwrap();
    let song = dir.join("piece.mid");
    write_minimal_smf(&song, b"piece-bytes");
    let key = make_key(song.clone(), "guitar-stk", None);

    let wav = b"RIFF\x24\x00\x00\x00WAVE pre-warm";
    cache.store(&key, wav).unwrap();

    let bogus_bin = PathBuf::from("/this/path/does/not/exist/render_midi");
    let resolved = keysynth::preview_cache::render_to_cache(&cache, &key, "guitar-stk", &bogus_bin)
        .expect("primed cache should bypass subprocess");
    assert_eq!(fs::read(&resolved).unwrap(), wav);
}
