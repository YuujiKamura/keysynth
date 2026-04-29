//! Lazy + cached audio preview for the GUI song selector (issue #62
//! Phase 2).
//!
//! ## Why this module exists
//!
//! The naive design — render every (song, voice) pair when the song
//! list refreshes — does N*M renders even though the user listens to
//! at most one preview at a time. That pattern explodes at scale:
//! catalog grows, voice plugin rebuilds invalidate everything,
//! list-update latency drowns the GUI.
//!
//! `preview_cache` rebuilds that flow around the user's listen
//! bandwidth: list refresh emits zero renders; the user's click is
//! the only event that triggers a render; renders go through a
//! content-addressed disk cache so re-clicking the same (song, voice)
//! hits the WAV instantly.
//!
//! ## Cache key
//!
//! `sha256(song_path, song_mtime, voice_id, voice_dll, voice_dll_mtime,
//! render_params)`. Each input is encoded as a length-prefixed UTF-8 /
//! u128 / u32 byte sequence so the same logical key is byte-identical
//! across runs even when filenames are short prefixes of each other.
//!
//! Mtime is included so any rebuild of `voices_live/<name>/target/
//! release/*.dll` (or any edit to the source `.mid`) automatically
//! invalidates the cached preview without explicit cache-clear logic.
//!
//! ## What this module does NOT do
//!
//! - It does not run the renderer itself: callers pass the rendered
//!   WAV bytes via `Cache::store`. A separate helper
//!   (`render_to_cache`) wires up subprocess invocation of the
//!   `render_midi` binary; tests can supply synthetic WAV bytes
//!   directly to exercise the cache mechanics in isolation.
//! - It does not deduplicate concurrent renders. Two threads asking
//!   for the same key at the same time both render, and the second
//!   `store` overwrites with the same content. (Phase 3, issue #62.)
//! - It does not pre-warm. Hover-prefetch is explicitly out of scope
//!   for Phase 2.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use sha2::{Digest, Sha256};

/// Render-side parameters that distinguish two otherwise-identical
/// (song, voice) lookups. Two renders with different `master_gain` are
/// considered different cache entries; mixing knobs that re-mix the
/// final output should NOT live in the cache key (those can be
/// applied at playback time).
#[derive(Clone, Debug, PartialEq)]
pub struct RenderParams {
    pub sample_rate: u32,
    /// Master gain applied to the rendered output, 0.0..=2.0. Only the
    /// fixed render-time gain — runtime mixer volume on the playback
    /// side does not enter the cache key.
    pub master_gain: f32,
}

impl Default for RenderParams {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            master_gain: 1.0,
        }
    }
}

/// Identity of one cache entry. Built from the song file, the voice's
/// loaded `.dll` (or built-in voice id when there is no .dll), and the
/// render-side parameters.
#[derive(Clone, Debug)]
pub struct CacheKey {
    pub song_path: PathBuf,
    pub voice_id: String,
    /// Path to the voice's compiled `.dll` / `.so` / `.dylib`. None for
    /// built-in `Engine::*` voices that ship inside the keysynth crate.
    pub voice_dll: Option<PathBuf>,
    pub render_params: RenderParams,
}

impl CacheKey {
    /// Hex-encoded SHA-256 of the canonicalised key inputs. The output
    /// is stable across runs as long as every input file's content +
    /// mtime are unchanged; touching the voice .dll bumps the hash and
    /// causes a cache miss next lookup.
    pub fn hash(&self) -> io::Result<String> {
        let mut h = Sha256::new();
        absorb_bytes(&mut h, b"keysynth-preview-cache-v1");

        // Song: canonical absolute path + mtime in nanoseconds.
        let song_abs = canonicalize_or_lossy(&self.song_path);
        absorb_bytes(&mut h, b"song:");
        absorb_str(&mut h, song_abs.to_string_lossy().as_ref());
        absorb_mtime_ns(&mut h, &self.song_path)?;

        // Voice id (string label, e.g. "guitar-stk", "piano-modal").
        absorb_bytes(&mut h, b"voice_id:");
        absorb_str(&mut h, &self.voice_id);

        // Voice dll: either a (path, mtime) pair, or the literal token
        // "builtin" so a built-in voice with the same id doesn't collide
        // with a hot-loaded plugin that happened to share a name.
        match &self.voice_dll {
            Some(p) => {
                let p_abs = canonicalize_or_lossy(p);
                absorb_bytes(&mut h, b"voice_dll:");
                absorb_str(&mut h, p_abs.to_string_lossy().as_ref());
                absorb_mtime_ns(&mut h, p)?;
            }
            None => {
                absorb_bytes(&mut h, b"voice_dll:builtin");
            }
        }

        // Render params: pack into a fixed-byte representation so
        // float-formatting variations don't shift the hash.
        absorb_bytes(&mut h, b"sr:");
        absorb_bytes(&mut h, &self.render_params.sample_rate.to_le_bytes());
        absorb_bytes(&mut h, b"gain:");
        absorb_bytes(&mut h, &self.render_params.master_gain.to_le_bytes());

        Ok(hex_lower(&h.finalize()))
    }
}

fn absorb_bytes(h: &mut Sha256, bytes: &[u8]) {
    h.update((bytes.len() as u64).to_le_bytes());
    h.update(bytes);
}

fn absorb_str(h: &mut Sha256, s: &str) {
    absorb_bytes(h, s.as_bytes());
}

fn absorb_mtime_ns(h: &mut Sha256, path: &Path) -> io::Result<()> {
    let meta = fs::metadata(path)?;
    let mtime = meta.modified()?;
    let nanos = mtime
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        // Pre-1970 modification time => use a sentinel rather than fail
        // (some build systems set mtime=0 deliberately).
        .unwrap_or(0);
    absorb_bytes(h, &nanos.to_le_bytes());
    Ok(())
}

fn canonicalize_or_lossy(p: &Path) -> PathBuf {
    fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

/// On-disk LRU-evicting cache for rendered preview WAVs.
///
/// Files live at `<dir>/<hash>.wav`. `lookup` is a stat-only operation
/// (no read, no parse — just `is_file`). `store` writes atomically
/// via a temp file rename so partial writes can never be observed by
/// a concurrent `lookup`.
pub struct Cache {
    dir: PathBuf,
    max_bytes: u64,
}

impl Cache {
    /// Open or create the cache directory at `dir`. `max_bytes` is the
    /// soft cap for `evict_lru` — total on-disk size after eviction
    /// will be ≤ `max_bytes`. Pass a generous default (e.g. 1 GiB)
    /// for production; tests can pass small values to force eviction.
    pub fn new(dir: impl Into<PathBuf>, max_bytes: u64) -> io::Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)?;
        Ok(Self { dir, max_bytes })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Compute the on-disk path that `key` would resolve to. Does NOT
    /// check whether the file exists.
    pub fn path_for(&self, key: &CacheKey) -> io::Result<PathBuf> {
        let h = key.hash()?;
        Ok(self.dir.join(format!("{h}.wav")))
    }

    /// Return `Some(path)` if a cached render exists for `key`.
    /// Touches the file's atime via `set_modified` so the LRU policy
    /// in `evict_lru` orders by last-use rather than first-render.
    pub fn lookup(&self, key: &CacheKey) -> io::Result<Option<PathBuf>> {
        let path = self.path_for(key)?;
        match fs::metadata(&path) {
            Ok(m) if m.is_file() && m.len() > 0 => {
                // Bump mtime so subsequent `evict_lru` treats this as
                // recently used. Best-effort — some filesystems reject
                // explicit mtime updates, in which case we still
                // return the path.
                let _ = filetime_now(&path);
                Ok(Some(path))
            }
            _ => Ok(None),
        }
    }

    /// Atomically store `wav_bytes` for `key`. Returns the final on-disk
    /// path. If a previous entry exists for the same key it is
    /// overwritten in place.
    pub fn store(&self, key: &CacheKey, wav_bytes: &[u8]) -> io::Result<PathBuf> {
        let path = self.path_for(key)?;
        let tmp = path.with_extension("wav.partial");
        // Write to a temp file in the same directory so the rename is
        // a same-filesystem atomic op on every platform we support.
        fs::write(&tmp, wav_bytes)?;
        // On Windows, std::fs::rename() fails if the destination
        // exists — explicitly remove first.
        let _ = fs::remove_file(&path);
        fs::rename(&tmp, &path)?;
        Ok(path)
    }

    /// Total on-disk size of all cache entries, in bytes. Cheap-ish:
    /// one stat per file, no reads.
    pub fn current_size_bytes(&self) -> io::Result<u64> {
        let mut total = 0_u64;
        for ent in fs::read_dir(&self.dir)? {
            let ent = ent?;
            if ent.file_type()?.is_file() {
                total = total.saturating_add(ent.metadata()?.len());
            }
        }
        Ok(total)
    }

    /// Drop oldest-mtime entries until total size ≤ `max_bytes`.
    /// Returns the number of files evicted.
    pub fn evict_lru(&self) -> io::Result<usize> {
        let mut entries: Vec<(PathBuf, u64, SystemTime)> = Vec::new();
        for ent in fs::read_dir(&self.dir)? {
            let ent = ent?;
            if !ent.file_type()?.is_file() {
                continue;
            }
            let meta = ent.metadata()?;
            let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            entries.push((ent.path(), meta.len(), mtime));
        }
        // Sort oldest-first.
        entries.sort_by_key(|(_, _, t)| *t);
        let mut total: u64 = entries.iter().map(|(_, sz, _)| *sz).sum();
        let mut evicted = 0_usize;
        for (path, sz, _) in &entries {
            if total <= self.max_bytes {
                break;
            }
            // Best-effort delete; if a file disappears (concurrent
            // store) just keep going.
            if fs::remove_file(path).is_ok() {
                total = total.saturating_sub(*sz);
                evicted += 1;
            }
        }
        Ok(evicted)
    }
}

/// Update a file's modified-time to "now". Used by `lookup` to keep
/// LRU ordering current; failure is swallowed by the caller because
/// some filesystems (e.g. read-only mounts, FAT32 with limited
/// mtime granularity) reject the call.
fn filetime_now(path: &Path) -> io::Result<()> {
    let now = SystemTime::now();
    set_file_mtime(path, now)
}

#[cfg(unix)]
fn set_file_mtime(path: &Path, t: SystemTime) -> io::Result<()> {
    // Use std::fs::File::set_modified once available; for now go via
    // libc::utimensat-equivalent through std (Rust 1.75+).
    let f = fs::OpenOptions::new().write(true).open(path)?;
    f.set_modified(t)
}

#[cfg(windows)]
fn set_file_mtime(path: &Path, t: SystemTime) -> io::Result<()> {
    let f = fs::OpenOptions::new().write(true).open(path)?;
    f.set_modified(t)
}

// ---------------------------------------------------------------------------
// Render driver
// ---------------------------------------------------------------------------

/// Look up `key` in `cache`; on miss, run `render_midi` as a subprocess
/// and store the produced WAV bytes under `key`. Returns the final
/// on-disk WAV path.
///
/// `engine_arg` is the value passed straight to `render_midi --engine
/// <engine_arg>` (e.g. `"guitar-stk"`, `"piano-modal"`). The function
/// uses the binary path from `render_midi_bin` (typically
/// `target/release/render_midi.exe` or `target/release/render_midi`).
pub fn render_to_cache(
    cache: &Cache,
    key: &CacheKey,
    engine_arg: &str,
    render_midi_bin: &Path,
) -> io::Result<PathBuf> {
    if let Some(p) = cache.lookup(key)? {
        return Ok(p);
    }
    // Render to a temp file then read its bytes into the cache via
    // `store`. We deliberately don't have render_midi write directly
    // to the cache directory because that would expose half-written
    // files to a concurrent `lookup` — `store` does the atomic rename.
    let tmp = std::env::temp_dir().join(format!(
        "keysynth-preview-{}.wav",
        key.hash().unwrap_or_else(|_| "unknown".to_string())
    ));
    let _ = fs::remove_file(&tmp);

    let status = std::process::Command::new(render_midi_bin)
        .arg("--in")
        .arg(&key.song_path)
        .arg("--engine")
        .arg(engine_arg)
        .arg("--out")
        .arg(&tmp)
        .status()?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "render_midi exited with status {status}"
        )));
    }
    let wav_bytes = fs::read(&tmp)?;
    let _ = fs::remove_file(&tmp);
    cache.store(key, &wav_bytes)
}

/// Look up `key` in `cache`; on miss, run `render_nsf` as a subprocess
/// and store the produced WAV bytes under `key`. Returns the final
/// on-disk WAV path.
///
/// `track` is the 0-indexed NSF track number. `render_nsf_bin` is
/// typically `target/release/render_nsf.exe`.
pub fn render_nsf_to_cache(
    cache: &Cache,
    key: &CacheKey,
    track: u32,
    render_nsf_bin: &Path,
) -> io::Result<PathBuf> {
    if let Some(p) = cache.lookup(key)? {
        return Ok(p);
    }
    let tmp = std::env::temp_dir().join(format!(
        "keysynth-preview-nsf-{}.wav",
        key.hash().unwrap_or_else(|_| "unknown".to_string())
    ));
    let _ = fs::remove_file(&tmp);

    let status = std::process::Command::new(render_nsf_bin)
        .arg("--in")
        .arg(&key.song_path)
        .arg("--track")
        .arg(track.to_string())
        .arg("--out")
        .arg(&tmp)
        .status()?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "render_nsf exited with status {status}"
        )));
    }
    let wav_bytes = fs::read(&tmp)?;
    let _ = fs::remove_file(&tmp);
    cache.store(key, &wav_bytes)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Construct a `CacheKey` with no voice .dll set — exercises the
    /// "builtin" hash path and lets tests that don't actually load a
    /// dynamic voice still drive every cache code path.
    fn key_for(song: PathBuf, voice_id: &str) -> CacheKey {
        CacheKey {
            song_path: song,
            voice_id: voice_id.to_string(),
            voice_dll: None,
            render_params: RenderParams::default(),
        }
    }

    /// Write a tiny non-empty file at `path`. Creates parent dirs.
    fn touch(path: &Path, bytes: &[u8]) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn hash_is_stable_across_calls_with_same_inputs() {
        let dir = tempdir();
        let song = dir.join("song.mid");
        touch(&song, b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x01\xe0song1");
        let k = key_for(song, "voice-A");
        let h1 = k.hash().unwrap();
        let h2 = k.hash().unwrap();
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
    }

    #[test]
    fn hash_changes_when_song_content_changes() {
        let dir = tempdir();
        let song = dir.join("song.mid");
        touch(&song, b"contents-A");
        let k = key_for(song.clone(), "voice-A");
        let h1 = k.hash().unwrap();

        // Sleep just enough to guarantee an mtime tick on filesystems
        // with second-granularity timestamps (FAT32 has 2-second
        // resolution; NTFS / ext4 are sub-millisecond).
        std::thread::sleep(std::time::Duration::from_millis(1100));
        touch(&song, b"contents-B");
        let h2 = k.hash().unwrap();
        assert_ne!(h1, h2, "song content change must invalidate hash");
    }

    #[test]
    fn hash_changes_when_voice_dll_mtime_changes() {
        let dir = tempdir();
        let song = dir.join("song.mid");
        let dll = dir.join("voice.dll");
        touch(&song, b"song-x");
        touch(&dll, b"dll-A");

        let k = CacheKey {
            song_path: song,
            voice_id: "voice-A".into(),
            voice_dll: Some(dll.clone()),
            render_params: RenderParams::default(),
        };
        let h1 = k.hash().unwrap();

        std::thread::sleep(std::time::Duration::from_millis(1100));
        // Same content; just touch the dll. Use OpenOptions+set_modified
        // so the test doesn't depend on an external `touch` binary.
        let f = fs::OpenOptions::new().write(true).open(&dll).unwrap();
        f.set_modified(SystemTime::now()).unwrap();

        let h2 = k.hash().unwrap();
        assert_ne!(h1, h2, "voice .dll mtime change must invalidate hash");
    }

    #[test]
    fn store_then_lookup_returns_path() {
        let dir = tempdir();
        let cache = Cache::new(dir.join("cache"), 1024 * 1024).unwrap();
        let song = dir.join("song.mid");
        touch(&song, b"midi-bytes");
        let k = key_for(song, "voice-A");

        assert!(cache.lookup(&k).unwrap().is_none(), "miss before store");

        let wav = b"RIFF\x24\x00\x00\x00WAVEfmt fake-wav-bytes";
        let stored = cache.store(&k, wav).unwrap();
        assert!(stored.is_file());
        assert_eq!(fs::read(&stored).unwrap(), wav);

        let hit = cache.lookup(&k).unwrap();
        assert!(hit.is_some());
        assert_eq!(hit.unwrap(), stored);
    }

    #[test]
    fn evict_lru_drops_oldest_until_under_limit() {
        let dir = tempdir();
        // 3 KiB cap. Each entry is 2 KiB → after 3 entries we're at
        // 6 KiB > 3 KiB cap → evict_lru should drop the 2 oldest.
        let cache = Cache::new(dir.join("cache"), 3 * 1024).unwrap();
        let song = dir.join("song.mid");
        touch(&song, b"midi-bytes");

        let mut keys = Vec::new();
        for i in 0..3 {
            let k = key_for(song.clone(), &format!("voice-{i}"));
            // 2 KiB body so cap math is exact.
            let body = vec![i as u8; 2 * 1024];
            cache.store(&k, &body).unwrap();
            keys.push(k);
            // Ensure each store has a distinct mtime.
            std::thread::sleep(std::time::Duration::from_millis(1100));
        }
        assert!(cache.current_size_bytes().unwrap() >= 6 * 1024);

        let evicted = cache.evict_lru().unwrap();
        assert!(evicted >= 2, "expected ≥2 evictions, got {evicted}");
        assert!(
            cache.current_size_bytes().unwrap() <= 3 * 1024,
            "size after eviction should be under cap"
        );

        // The most-recently-stored key (index 2) survives.
        assert!(cache.lookup(&keys[2]).unwrap().is_some());
    }

    /// Ephemeral test directory under the system tempdir, deleted by
    /// the OS at reboot. Tests do not bother cleaning up because the
    /// per-test prefix collides only across runs of the same test
    /// process.
    fn tempdir() -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "keysynth-preview-cache-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&base).unwrap();
        base
    }
}
