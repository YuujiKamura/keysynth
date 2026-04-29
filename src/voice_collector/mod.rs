//! Stage D: Voice Collector — public-domain SFZ catalog acquisition pipeline.
//!
//! ## Why a separate module
//!
//! Stage B's song collector (`bench-out/songs/manifest.json`, see
//! `src/library_db::import_songs`) acquires Standard MIDI Files. This is
//! the analogous pipeline for sample-based instruments: a curated list
//! of CC0 / CC-BY SFZ libraries is pinned in `samples/manifest.json`,
//! re-fetched on demand via `voice_collector` CLI, and imported into the
//! same SQLite catalog as `voices` rows with id prefix `sample:` so they
//! never collide with `voices_live/<dir>` ids or built-in `static:` ids.
//!
//! ## What gets vendored
//!
//! Only the `.sfz` text manifests (a few KB each) are downloaded and
//! committed under `samples/sfz/`. The referenced WAV samples are
//! large (often hundreds of MB per library) and stay upstream — the
//! `samples_root` URL on each manifest entry is the operator-friendly
//! pointer for whoever wants to enable real playback later. Importing
//! the SFZ alone is enough to surface the voice in `keysynth_db query
//! --voices` and to expose its instrument range / region count to
//! discovery tooling.
//!
//! ## License hygiene
//!
//! The manifest's `rejected_categories` list pins the license rules:
//! CC0 only (CC-BY would force per-voice attribution UI we don't have
//! yet, CC-BY-SA is copyleft, NC variants are non-commercial). Every
//! entry's `license` field is checked against this allowlist on import
//! so an accidentally-pasted GPL/proprietary entry trips an error
//! before it lands in the DB.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

/// Allowed license strings. Any entry carrying a different license
/// trips `VoiceCollectorError::DisallowedLicense` on import — picking
/// the rejection at the catalog boundary is cheaper than hoping the
/// downstream consumer notices.
pub const ALLOWED_LICENSES: &[&str] = &["CC0-1.0", "CC0", "Public Domain"];

#[derive(Debug)]
pub enum VoiceCollectorError {
    Io(std::io::Error),
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    DisallowedLicense {
        id: String,
        license: String,
    },
    Fetch {
        url: String,
        status: i32,
        stderr: String,
    },
    /// `curl` (or whichever fetch tool) is not on PATH.
    FetchToolMissing(String),
    /// SFZ file referenced by manifest entry doesn't exist after fetch.
    Missing {
        id: String,
        path: PathBuf,
    },
}

impl std::fmt::Display for VoiceCollectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::Json { path, source } => {
                write!(f, "manifest parse ({}): {source}", path.display())
            }
            Self::DisallowedLicense { id, license } => write!(
                f,
                "entry {id}: license '{license}' not in ALLOWED_LICENSES (must be CC0)"
            ),
            Self::Fetch { url, status, stderr } => {
                write!(f, "fetch {url} failed (curl exit {status}): {stderr}")
            }
            Self::FetchToolMissing(tool) => {
                write!(f, "fetch tool '{tool}' not found on PATH")
            }
            Self::Missing { id, path } => {
                write!(f, "entry {id}: file {} missing on disk", path.display())
            }
        }
    }
}

impl std::error::Error for VoiceCollectorError {}

impl From<std::io::Error> for VoiceCollectorError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, VoiceCollectorError>;

/// One row of `samples/manifest.json`. Field names mirror the JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleEntry {
    pub id: String,
    pub file: String,
    pub display_name: String,
    pub instrument: String,
    pub category: String,
    pub recommend: String,
    pub license: String,
    pub source: String,
    pub source_url: String,
    pub fetch_url: String,
    #[serde(default)]
    pub samples_root: Option<String>,
    pub description: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleManifest {
    pub schema_version: u32,
    pub directory: String,
    #[serde(default)]
    pub description: Option<String>,
    pub acquisition_date: String,
    #[serde(default)]
    pub fetch_method: Option<String>,
    pub entries: Vec<SampleEntry>,
}

impl SampleManifest {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        serde_json::from_str(&raw).map_err(|source| VoiceCollectorError::Json {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Resolve the on-disk SFZ path for an entry, relative to the
    /// manifest's `directory`. The manifest stores `directory` relative
    /// to repo root by default; if the caller passes a different
    /// `repo_root` we resolve under that instead so tests can stage in
    /// a tempdir.
    pub fn entry_path(&self, repo_root: &Path, entry: &SampleEntry) -> PathBuf {
        repo_root.join(&self.directory).join(&entry.file)
    }

    /// License gate. Returns the first violating entry id, if any.
    pub fn audit_licenses(&self) -> Result<()> {
        for e in &self.entries {
            if !ALLOWED_LICENSES.iter().any(|ok| ok.eq_ignore_ascii_case(&e.license)) {
                return Err(VoiceCollectorError::DisallowedLicense {
                    id: e.id.clone(),
                    license: e.license.clone(),
                });
            }
        }
        Ok(())
    }
}

/// Fetch a single entry via curl. Skips the network call if the file
/// already exists and `force` is false. We shell out to curl rather
/// than pulling in reqwest/ureq because keysynth has zero HTTP
/// dependencies today and the catalog is fetched at most once per
/// manifest revision — adding a TLS stack to ship a 5-shot sync job
/// is the wrong tradeoff. curl is a hard dependency on every dev box
/// the project supports (already used by tools/fetch-guitar-refs.sh).
pub fn fetch_entry(
    manifest: &SampleManifest,
    entry: &SampleEntry,
    repo_root: &Path,
    force: bool,
) -> Result<FetchOutcome> {
    let dest = manifest.entry_path(repo_root, entry);
    if dest.exists() && !force {
        return Ok(FetchOutcome::Skipped {
            reason: "file already on disk (pass --force to refetch)".to_string(),
            path: dest,
        });
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let output = Command::new("curl")
        .args([
            "-fsSL",
            "--retry",
            "2",
            "--connect-timeout",
            "20",
            "--max-time",
            "120",
            "-o",
        ])
        .arg(&dest)
        .arg(&entry.fetch_url)
        .output()
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => {
                VoiceCollectorError::FetchToolMissing("curl".to_string())
            }
            _ => VoiceCollectorError::Io(e),
        })?;

    if !output.status.success() {
        // Best-effort cleanup so a partial write doesn't masquerade as
        // a successful prior fetch on the next run.
        let _ = std::fs::remove_file(&dest);
        return Err(VoiceCollectorError::Fetch {
            url: entry.fetch_url.clone(),
            status: output.status.code().unwrap_or(-1),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        });
    }

    let bytes = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
    Ok(FetchOutcome::Fetched { path: dest, bytes })
}

/// Fetch every entry in the manifest in declaration order. Stops on
/// the first hard error so a flaky network failure doesn't pollute the
/// catalog with half-written files.
pub fn fetch_all(
    manifest: &SampleManifest,
    repo_root: &Path,
    force: bool,
) -> Result<Vec<(String, FetchOutcome)>> {
    manifest.audit_licenses()?;
    let mut out = Vec::with_capacity(manifest.entries.len());
    for e in &manifest.entries {
        let outcome = fetch_entry(manifest, e, repo_root, force)?;
        out.push((e.id.clone(), outcome));
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub enum FetchOutcome {
    Fetched { path: PathBuf, bytes: u64 },
    Skipped { reason: String, path: PathBuf },
}

impl FetchOutcome {
    pub fn path(&self) -> &Path {
        match self {
            Self::Fetched { path, .. } => path,
            Self::Skipped { path, .. } => path,
        }
    }
    pub fn was_fetched(&self) -> bool {
        matches!(self, Self::Fetched { .. })
    }
}

/// Light SFZ inspection — parse just enough to count `<region>` blocks
/// and extract the lowest/highest declared `lokey`/`hikey`. Used by
/// `voice_collector list` for at-a-glance "is this a piano-range library
/// or a sax-range library" output and by the importer to populate a
/// `region_count` tag. This is *not* the playback parser — that lives
/// in `src/sfz.rs` and accepts a much broader subset.
#[derive(Debug, Default, Clone)]
pub struct SfzSummary {
    pub region_count: usize,
    pub lokey: Option<u8>,
    pub hikey: Option<u8>,
    pub sample_refs: usize,
    /// Count of `#include "..."` directives. Karoryfer multi-articulation
    /// libraries split per-articulation regions into companion SFZs and
    /// reference them from a thin master file — when this is nonzero the
    /// real region inventory lives in the includes.
    pub include_count: usize,
}

pub fn summarize_sfz(path: &Path) -> Result<SfzSummary> {
    let raw = std::fs::read_to_string(path)?;
    let mut s = SfzSummary::default();
    for line in raw.lines() {
        let line = line.trim();
        // Region opener can appear mid-line in SFZ; count the substring
        // rather than checking starts_with so `<region>...` on the same
        // line as opcodes still tallies.
        s.region_count += line.matches("<region>").count();
        if line.starts_with("#include") {
            s.include_count += 1;
        }
        if let Some(rest) = line.strip_prefix("sample=") {
            // Some SFZs put multiple opcodes on one line — we only
            // care about counting the leading sample=, that's enough
            // for "does this library reference any WAV at all" sanity.
            let _ = rest; // explicit no-op; trimming kept simple
            s.sample_refs += 1;
        } else if line.contains("sample=") && !line.starts_with('/') {
            s.sample_refs += 1;
        }
        for kv in line.split_whitespace() {
            if let Some(v) = kv.strip_prefix("lokey=") {
                if let Some(n) = parse_keynum(v) {
                    s.lokey = Some(s.lokey.map(|cur| cur.min(n)).unwrap_or(n));
                }
            } else if let Some(v) = kv.strip_prefix("hikey=") {
                if let Some(n) = parse_keynum(v) {
                    s.hikey = Some(s.hikey.map(|cur| cur.max(n)).unwrap_or(n));
                }
            } else if let Some(v) = kv.strip_prefix("key=") {
                if let Some(n) = parse_keynum(v) {
                    s.lokey = Some(s.lokey.map(|cur| cur.min(n)).unwrap_or(n));
                    s.hikey = Some(s.hikey.map(|cur| cur.max(n)).unwrap_or(n));
                }
            }
        }
    }
    Ok(s)
}

/// SFZ accepts both numeric MIDI keys (`60`) and note names (`c4`,
/// `f#3`). Salamander uses numbers, Karoryfer mixes both. We accept
/// either; unknown forms return None and the summary just skips them.
fn parse_keynum(s: &str) -> Option<u8> {
    if let Ok(n) = s.parse::<i32>() {
        if (0..=127).contains(&n) {
            return Some(n as u8);
        }
        return None;
    }
    // Note-name form: [A-Ga-g][#bs]?-?\d
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let pitch = match bytes[0].to_ascii_lowercase() {
        b'c' => 0i32,
        b'd' => 2,
        b'e' => 4,
        b'f' => 5,
        b'g' => 7,
        b'a' => 9,
        b'b' => 11,
        _ => return None,
    };
    let mut i = 1;
    let mut accidental = 0i32;
    if i < bytes.len() {
        match bytes[i] {
            b'#' | b's' => {
                accidental = 1;
                i += 1;
            }
            b'b' => {
                // Ambiguous: 'b' could be the accidental flat OR the
                // note name 'b' (already handled above). Here we're
                // past index 0 so it's the flat sign.
                accidental = -1;
                i += 1;
            }
            _ => {}
        }
    }
    let octave_str = std::str::from_utf8(&bytes[i..]).ok()?;
    let octave: i32 = octave_str.parse().ok()?;
    // SFZ "c4" = MIDI 60 (Yamaha convention used by SFZ tooling).
    let midi = (octave + 1) * 12 + pitch + accidental;
    if (0..=127).contains(&midi) {
        Some(midi as u8)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn keynum_numeric_and_note_name() {
        assert_eq!(parse_keynum("60"), Some(60));
        assert_eq!(parse_keynum("c4"), Some(60));
        assert_eq!(parse_keynum("C4"), Some(60));
        assert_eq!(parse_keynum("a4"), Some(69));
        assert_eq!(parse_keynum("c#4"), Some(61));
        assert_eq!(parse_keynum("db4"), Some(61));
        // out of range
        assert_eq!(parse_keynum("128"), None);
        assert_eq!(parse_keynum("garbage"), None);
    }

    #[test]
    fn summarize_minimal_sfz() {
        let dir = tempdir_or_skip();
        let p = dir.join("test.sfz");
        let mut f = std::fs::File::create(&p).unwrap();
        writeln!(
            f,
            "<group>\nlokey=21 hikey=108 pitch_keycenter=60\n<region>\nsample=foo.wav\n<region>\nsample=bar.wav lokey=40 hikey=80"
        )
        .unwrap();
        let s = summarize_sfz(&p).unwrap();
        assert_eq!(s.region_count, 2);
        assert_eq!(s.lokey, Some(21));
        assert_eq!(s.hikey, Some(108));
        // Only the line-leading `sample=` form counts (parser is
        // intentionally light); the second region's sample= follows a
        // newline so both are counted.
        assert!(s.sample_refs >= 2);
    }

    #[test]
    fn license_audit_rejects_non_cc0() {
        let m = SampleManifest {
            schema_version: 1,
            directory: "samples/sfz".to_string(),
            description: None,
            acquisition_date: "2026-04-29".to_string(),
            fetch_method: None,
            entries: vec![SampleEntry {
                id: "bad".to_string(),
                file: "bad.sfz".to_string(),
                display_name: "Bad".to_string(),
                instrument: "piano".to_string(),
                category: "Piano".to_string(),
                recommend: "Stable".to_string(),
                license: "CC-BY-NC-3.0".to_string(),
                source: "x".to_string(),
                source_url: "x".to_string(),
                fetch_url: "x".to_string(),
                samples_root: None,
                description: "x".to_string(),
                tags: vec![],
            }],
        };
        let err = m.audit_licenses().unwrap_err();
        assert!(matches!(
            err,
            VoiceCollectorError::DisallowedLicense { .. }
        ));
    }

    fn tempdir_or_skip() -> PathBuf {
        // Use the OS temp dir so we don't depend on the tempfile crate.
        // This is test-only and we never delete it (CI-grade is fine).
        let p = std::env::temp_dir().join(format!(
            "keysynth_voice_collector_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
