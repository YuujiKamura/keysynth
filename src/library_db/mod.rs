//! Materialized library catalog for songs + voices (issue #66).
//!
//! ## Why a DB?
//!
//! Two source-of-truth manifests already exist:
//!
//! * `bench-out/songs/manifest.json` — public-domain MIDI catalog (PR #63)
//! * `voices_live/<name>/Cargo.toml` `[package.metadata.keysynth-voice]`
//!   block — voice plugin metadata (PR #58)
//!
//! Both stay authoritative. This module is a *materialized index* on top:
//! the JSON / TOML stay editable by hand, `LibraryDb::rebuild()` re-imports
//! them, and queries land on a SQLite database (`bench-out/library.db`)
//! where filtering and joins are cheap.
//!
//! ## Why SQLite (and not a JSON in-memory cache)?
//!
//! kssong / jukebox / a future "find me a Bach guitar étude" query path
//! all want the same filter primitives: by composer, by era, by tag, by
//! recommended voice category. A growing manifest will eventually have
//! piano + guitar + organ + lute repertoire and ad-hoc Vec scans get
//! awkward. SQLite gives us indices, joins, and one shared on-disk
//! schema for free, with `rusqlite[bundled]` so deployment stays
//! single-binary.
//!
//! Steel-side query access (issue #56 Phase 2/3) is out of scope here —
//! that's a future bridge through `cp-core`. This module exposes a
//! native-Rust API only.
//!
//! ## Determinism
//!
//! Every `import_*` uses `INSERT OR REPLACE` keyed on a stable id (song
//! file stem, voice crate dir name) so re-running rebuild against an
//! unchanged on-disk source is a no-op. Tags are wiped + re-inserted per
//! parent so renaming/removing a tag in the source is reflected.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OptionalExtension};
use serde::Deserialize;

/// Embedded schema. Lives next to the catalog at `bench-out/library.db.sql`
/// so the file is human-readable + grep-friendly, but we bake it into
/// the binary via `include_str!` so `migrate()` works without the source
/// tree present (e.g. when keysynth-db ships as a release binary).
const SCHEMA_SQL: &str = include_str!("../../bench-out/library.db.sql");

#[derive(Debug)]
pub struct LibraryDb {
    conn: Connection,
}

#[derive(Debug)]
pub enum LibraryDbError {
    Sqlite(rusqlite::Error),
    Io(std::io::Error),
    Manifest {
        path: PathBuf,
        source: serde_json::Error,
    },
    Toml {
        path: PathBuf,
        source: toml::de::Error,
    },
}

impl std::fmt::Display for LibraryDbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sqlite(e) => write!(f, "sqlite: {e}"),
            Self::Io(e) => write!(f, "io: {e}"),
            Self::Manifest { path, source } => {
                write!(f, "manifest parse ({}): {source}", path.display())
            }
            Self::Toml { path, source } => write!(f, "toml ({}): {source}", path.display()),
        }
    }
}

impl std::error::Error for LibraryDbError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Sqlite(e) => Some(e),
            Self::Io(e) => Some(e),
            Self::Manifest { source, .. } => Some(source),
            Self::Toml { source, .. } => Some(source),
        }
    }
}

impl From<rusqlite::Error> for LibraryDbError {
    fn from(e: rusqlite::Error) -> Self {
        Self::Sqlite(e)
    }
}

impl From<std::io::Error> for LibraryDbError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, LibraryDbError>;

// ─── Public domain types ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Song {
    pub id: String,
    pub title: String,
    pub composer: String,
    pub composer_key: String,
    pub era: Option<String>,
    pub instrument: String,
    pub license: String,
    pub source: Option<String>,
    pub source_url: Option<String>,
    pub suggested_voice: Option<String>,
    pub midi_path: PathBuf,
    pub context: Option<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Voice {
    pub id: String,
    pub display_name: String,
    pub category: String,
    pub recommend: String,
    pub description: String,
    pub slot_name: Option<String>,
    pub decay_model: Option<String>,
    pub cdylib_path: Option<PathBuf>,
    pub cdylib_mtime: Option<i64>,
    pub tags: Vec<String>,
}

#[derive(Debug, Default, Clone)]
pub struct SongFilter {
    /// Substring match against `composer_key` (lowercased "tarrega",
    /// "bach", ...). `None` returns all rows.
    pub composer: Option<String>,
    pub era: Option<String>,
    pub instrument: Option<String>,
    pub tag: Option<String>,
    /// Suggested voice id (e.g. "guitar-stk"). Joins the songs row.
    pub recommended_voice: Option<String>,
    pub sort: SongSort,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum SongSort {
    /// Composer key, then title — group by composer for browser display.
    #[default]
    ByComposer,
    /// Title alphabetically.
    ByTitle,
    /// Insertion order (effectively: as listed in manifest.json).
    ByImportOrder,
}

#[derive(Debug, Default, Clone)]
pub struct VoiceFilter {
    pub category: Option<String>,
    pub recommend: Option<String>,
}

// ─── Lifecycle ───────────────────────────────────────────────────────

impl LibraryDb {
    /// Open or create the SQLite database at `path`. The file is
    /// created on first open; call [`migrate`](Self::migrate) before
    /// any insert/query to ensure schema is present.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        // foreign_keys is per-connection in SQLite; the schema PRAGMA
        // line in library.db.sql sets it during migrate(), but we also
        // enable it eagerly so the very first session honours
        // ON DELETE CASCADE.
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        Ok(Self { conn })
    }

    /// In-memory variant used by tests so they don't churn disk or
    /// race against a developer's working catalog.
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        Ok(Self { conn })
    }

    /// Apply the schema (idempotent). Safe to call repeatedly.
    pub fn migrate(&mut self) -> Result<()> {
        self.conn.execute_batch(SCHEMA_SQL)?;
        Ok(())
    }

    /// Re-import everything: voices from `voices_live_root` (defaults to
    /// `voices_live/`) and songs from `manifest_path`. Equivalent to
    /// `import_voices` + `import_songs` in a single transaction.
    pub fn rebuild(&mut self, manifest_path: &Path, voices_live_root: &Path) -> Result<(usize, usize)> {
        self.migrate()?;
        let songs = self.import_songs(manifest_path)?;
        let voices = self.import_voices(voices_live_root)?;
        Ok((songs, voices))
    }

    /// Direct accessor for callers that need to run their own query —
    /// kept small on purpose; prefer the typed `query_*` helpers.
    pub fn conn(&self) -> &Connection {
        &self.conn
    }
}

// ─── Song import ─────────────────────────────────────────────────────

/// Mirror of one entry in `bench-out/songs/manifest.json`. Field names
/// match the JSON exactly. Optional fields stay `Option` so partial
/// manifests (older / new schema versions) don't fail the import.
#[derive(Debug, Deserialize)]
struct ManifestEntry {
    file: String,
    title: String,
    composer: String,
    instrument: String,
    license: String,
    source: Option<String>,
    source_url: Option<String>,
    suggested_voice: Option<String>,
    context: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ManifestRoot {
    #[serde(default)]
    directory: Option<String>,
    entries: Vec<ManifestEntry>,
}

impl LibraryDb {
    /// Import all entries from a `manifest.json` file. Returns the
    /// count of inserted/replaced rows. Wipes existing tag rows for
    /// each touched song before reinserting so removed tags don't
    /// linger.
    pub fn import_songs(&mut self, manifest_path: &Path) -> Result<usize> {
        let raw = std::fs::read_to_string(manifest_path)?;
        let root: ManifestRoot =
            serde_json::from_str(&raw).map_err(|source| LibraryDbError::Manifest {
                path: manifest_path.to_path_buf(),
                source,
            })?;
        // The manifest stores files relative to its `directory` field.
        // Fall back to the manifest file's own parent if not set.
        let song_dir: PathBuf = root
            .directory
            .map(PathBuf::from)
            .or_else(|| manifest_path.parent().map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("."));

        let now = unix_now();
        let tx = self.conn.transaction()?;
        let mut count = 0usize;
        for entry in root.entries {
            let id = stem_of(&entry.file);
            let composer_key = composer_lookup_key(&entry.composer);
            let era = derive_era(&entry.composer);
            let midi_path = song_dir.join(&entry.file);

            tx.execute(
                "INSERT OR REPLACE INTO songs (
                    id, title, composer, composer_key, era, instrument, license,
                    source, source_url, suggested_voice, midi_path, context, imported_at
                ) VALUES (
                    ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13
                )",
                params![
                    id,
                    entry.title,
                    entry.composer,
                    composer_key,
                    era,
                    entry.instrument,
                    entry.license,
                    entry.source,
                    entry.source_url,
                    entry.suggested_voice,
                    midi_path.to_string_lossy(),
                    entry.context,
                    now,
                ],
            )?;
            tx.execute("DELETE FROM tags WHERE song_id = ?1", params![id])?;
            for tag in &entry.tags {
                tx.execute(
                    "INSERT OR IGNORE INTO tags (song_id, tag) VALUES (?1, ?2)",
                    params![id, tag],
                )?;
            }
            count += 1;
        }
        tx.commit()?;
        Ok(count)
    }
}

// ─── Voice import ────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct VoiceManifestRoot {
    package: Option<VoiceManifestPackage>,
}

#[derive(Debug, Deserialize)]
struct VoiceManifestPackage {
    metadata: Option<VoiceManifestMetadata>,
}

#[derive(Debug, Deserialize)]
struct VoiceManifestMetadata {
    #[serde(rename = "keysynth-voice")]
    keysynth_voice: Option<VoiceManifestEntry>,
}

#[derive(Debug, Deserialize)]
struct VoiceManifestEntry {
    display_name: String,
    category: String,
    recommend: String,
    description: String,
    #[serde(default)]
    slot_name: Option<String>,
    /// Free-form decay-model identifier — present on physical-model
    /// voices (e.g. "stulov-felt", "valimaki-t60"). Treated as opaque
    /// metadata at the DB layer.
    #[serde(default)]
    decay_model: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
}

impl LibraryDb {
    /// Scan `voices_live_root` for `<dir>/Cargo.toml` files carrying a
    /// `[package.metadata.keysynth-voice]` block, and import each as a
    /// voice row. Also imports the static built-in catalog (synth
    /// engines + sample slots) under `static:<engine>` ids so every
    /// selectable voice in the GUI is queryable from one table.
    pub fn import_voices(&mut self, voices_live_root: &Path) -> Result<usize> {
        let now = unix_now();
        let tx = self.conn.transaction()?;
        let mut count = 0usize;

        // Plugin discovery: each voices_live/<id>/ directory.
        let read_dir = match std::fs::read_dir(voices_live_root) {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!(
                    "library_db: voices_live root unreadable ({}): {e}",
                    voices_live_root.display()
                );
                None
            }
        };
        if let Some(read_dir) = read_dir {
            for ent in read_dir.flatten() {
                let path = ent.path();
                if !path.is_dir() {
                    continue;
                }
                let manifest = path.join("Cargo.toml");
                if !manifest.is_file() {
                    continue;
                }
                let id = match path.file_name().and_then(|s| s.to_str()) {
                    Some(n) => n.to_string(),
                    None => continue,
                };
                let raw = std::fs::read_to_string(&manifest)?;
                let root: VoiceManifestRoot =
                    toml::from_str(&raw).map_err(|source| LibraryDbError::Toml {
                        path: manifest.clone(),
                        source,
                    })?;
                let Some(meta) = root
                    .package
                    .and_then(|p| p.metadata)
                    .and_then(|m| m.keysynth_voice)
                else {
                    continue;
                };

                let slot_name = meta
                    .slot_name
                    .clone()
                    .or_else(|| Some(format!("live:{id}")));
                let (cdylib_path, cdylib_mtime) = locate_cdylib(&path, &id);

                tx.execute(
                    "INSERT OR REPLACE INTO voices (
                        id, display_name, category, recommend, description,
                        slot_name, decay_model, cdylib_path, cdylib_mtime, imported_at
                    ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
                    params![
                        id,
                        meta.display_name,
                        meta.category,
                        meta.recommend,
                        meta.description,
                        slot_name,
                        meta.decay_model,
                        cdylib_path.as_ref().map(|p| p.to_string_lossy().to_string()),
                        cdylib_mtime,
                        now,
                    ],
                )?;
                tx.execute("DELETE FROM voice_tags WHERE voice_id = ?1", params![id])?;
                for tag in &meta.tags {
                    tx.execute(
                        "INSERT OR IGNORE INTO voice_tags (voice_id, tag) VALUES (?1, ?2)",
                        params![id, tag],
                    )?;
                }
                count += 1;
            }
        }

        // Static catalog (synth engines + sample slots). These live in
        // src/voice_lib::builtins() and have no Cargo.toml to discover
        // from, so we hand-roll a small table here. Kept short on
        // purpose — when one of these moves into voices_live/, drop the
        // row here and the discovery branch above picks it up.
        for row in static_voice_rows() {
            tx.execute(
                "INSERT OR REPLACE INTO voices (
                    id, display_name, category, recommend, description,
                    slot_name, decay_model, cdylib_path, cdylib_mtime, imported_at
                ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
                params![
                    row.id,
                    row.display_name,
                    row.category,
                    row.recommend,
                    row.description,
                    row.slot_name,
                    Option::<String>::None,
                    Option::<String>::None,
                    Option::<i64>::None,
                    now,
                ],
            )?;
            count += 1;
        }

        tx.commit()?;
        Ok(count)
    }
}

struct StaticVoiceRow {
    id: &'static str,
    display_name: &'static str,
    category: &'static str,
    recommend: &'static str,
    description: &'static str,
    slot_name: &'static str,
}

fn static_voice_rows() -> &'static [StaticVoiceRow] {
    // Mirrors `src/voice_lib::builtins()` — the synth engines and
    // sample-based piano slots that aren't backed by a voices_live/
    // crate but still need to show up in `query_voices`.
    &[
        StaticVoiceRow {
            id: "static:square",
            display_name: "Square",
            category: "Synth",
            recommend: "Stable",
            description: "NES-style pulse",
            slot_name: "Square",
        },
        StaticVoiceRow {
            id: "static:ks",
            display_name: "KS",
            category: "Synth",
            recommend: "Stable",
            description: "Karplus-Strong basic",
            slot_name: "Ks",
        },
        StaticVoiceRow {
            id: "static:ks_rich",
            display_name: "KS Rich",
            category: "Synth",
            recommend: "Stable",
            description: "KS with allpass dispersion",
            slot_name: "KsRich",
        },
        StaticVoiceRow {
            id: "static:sub",
            display_name: "Sub",
            category: "Synth",
            recommend: "Stable",
            description: "analog subtractive",
            slot_name: "Sub",
        },
        StaticVoiceRow {
            id: "static:fm",
            display_name: "FM",
            category: "Synth",
            recommend: "Stable",
            description: "2-op DX7-ish bell",
            slot_name: "Fm",
        },
        StaticVoiceRow {
            id: "static:koto",
            display_name: "Koto",
            category: "Synth",
            recommend: "Stable",
            description: "pluck-position digital waveguide",
            slot_name: "Koto",
        },
        StaticVoiceRow {
            id: "static:sfz_piano",
            display_name: "Salamander SFZ (Grand)",
            category: "Samples",
            recommend: "Best",
            description: "high-quality Salamander Grand reference",
            slot_name: "SfzPiano",
        },
        StaticVoiceRow {
            id: "static:sf_piano",
            display_name: "GeneralUser SF2 (GM)",
            category: "Samples",
            recommend: "Stable",
            description: "GM-set SoundFont, multi-instrument",
            slot_name: "SfPiano",
        },
    ]
}

// ─── Queries ─────────────────────────────────────────────────────────

impl LibraryDb {
    pub fn query_songs(&self, filter: &SongFilter) -> Result<Vec<Song>> {
        let mut sql = String::from(
            "SELECT id, title, composer, composer_key, era, instrument, license,
                    source, source_url, suggested_voice, midi_path, context
             FROM songs",
        );
        let mut joins = String::new();
        let mut wheres: Vec<String> = Vec::new();
        let mut binds: Vec<String> = Vec::new();

        if let Some(c) = &filter.composer {
            wheres.push(format!("composer_key LIKE ?{}", binds.len() + 1));
            binds.push(format!("%{}%", c.to_lowercase()));
        }
        if let Some(e) = &filter.era {
            wheres.push(format!("era = ?{}", binds.len() + 1));
            binds.push(e.clone());
        }
        if let Some(i) = &filter.instrument {
            wheres.push(format!("instrument = ?{}", binds.len() + 1));
            binds.push(i.clone());
        }
        if let Some(v) = &filter.recommended_voice {
            wheres.push(format!("suggested_voice = ?{}", binds.len() + 1));
            binds.push(v.clone());
        }
        if let Some(t) = &filter.tag {
            joins.push_str(" JOIN tags ON tags.song_id = songs.id");
            wheres.push(format!("tags.tag = ?{}", binds.len() + 1));
            binds.push(t.clone());
        }

        if !joins.is_empty() {
            sql.push_str(&joins);
        }
        if !wheres.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&wheres.join(" AND "));
        }
        sql.push_str(match filter.sort {
            SongSort::ByComposer => " ORDER BY composer_key ASC, title ASC",
            SongSort::ByTitle => " ORDER BY title ASC",
            SongSort::ByImportOrder => " ORDER BY imported_at ASC, id ASC",
        });

        let mut stmt = self.conn.prepare(&sql)?;
        let bind_refs: Vec<&dyn rusqlite::ToSql> =
            binds.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
        let rows = stmt.query_map(bind_refs.as_slice(), |row| {
            Ok(Song {
                id: row.get(0)?,
                title: row.get(1)?,
                composer: row.get(2)?,
                composer_key: row.get(3)?,
                era: row.get(4)?,
                instrument: row.get(5)?,
                license: row.get(6)?,
                source: row.get(7)?,
                source_url: row.get(8)?,
                suggested_voice: row.get(9)?,
                midi_path: PathBuf::from(row.get::<_, String>(10)?),
                context: row.get(11)?,
                tags: Vec::new(),
            })
        })?;
        let mut out: Vec<Song> = Vec::new();
        for r in rows {
            out.push(r?);
        }
        for s in out.iter_mut() {
            s.tags = self.song_tags(&s.id)?;
        }
        Ok(out)
    }

    fn song_tags(&self, song_id: &str) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT tag FROM tags WHERE song_id = ?1 ORDER BY tag ASC")?;
        let rows = stmt.query_map(params![song_id], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn query_voices(&self, filter: &VoiceFilter) -> Result<Vec<Voice>> {
        let mut sql = String::from(
            "SELECT id, display_name, category, recommend, description,
                    slot_name, decay_model, cdylib_path, cdylib_mtime
             FROM voices",
        );
        let mut wheres: Vec<String> = Vec::new();
        let mut binds: Vec<String> = Vec::new();
        if let Some(c) = &filter.category {
            wheres.push(format!("category = ?{}", binds.len() + 1));
            binds.push(c.clone());
        }
        if let Some(r) = &filter.recommend {
            wheres.push(format!("recommend = ?{}", binds.len() + 1));
            binds.push(r.clone());
        }
        if !wheres.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&wheres.join(" AND "));
        }
        // Recommend rank: Best < Stable < Experimental matches the
        // editorial convention in src/voice_lib::Recommend.
        sql.push_str(
            " ORDER BY
                CASE recommend
                  WHEN 'Best' THEN 0
                  WHEN 'Stable' THEN 1
                  WHEN 'Experimental' THEN 2
                  ELSE 3 END ASC,
                category ASC, display_name ASC",
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let bind_refs: Vec<&dyn rusqlite::ToSql> =
            binds.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
        let rows = stmt.query_map(bind_refs.as_slice(), |row| {
            Ok(Voice {
                id: row.get(0)?,
                display_name: row.get(1)?,
                category: row.get(2)?,
                recommend: row.get(3)?,
                description: row.get(4)?,
                slot_name: row.get(5)?,
                decay_model: row.get(6)?,
                cdylib_path: row.get::<_, Option<String>>(7)?.map(PathBuf::from),
                cdylib_mtime: row.get(8)?,
                tags: Vec::new(),
            })
        })?;
        let mut out: Vec<Voice> = Vec::new();
        for r in rows {
            out.push(r?);
        }
        for v in out.iter_mut() {
            v.tags = self.voice_tags(&v.id)?;
        }
        Ok(out)
    }

    fn voice_tags(&self, voice_id: &str) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT tag FROM voice_tags WHERE voice_id = ?1 ORDER BY tag ASC")?;
        let rows = stmt.query_map(params![voice_id], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Look up a single song by stable id (file stem).
    pub fn get_song(&self, id: &str) -> Result<Option<Song>> {
        let opt = self
            .conn
            .query_row(
                "SELECT id, title, composer, composer_key, era, instrument, license,
                        source, source_url, suggested_voice, midi_path, context
                 FROM songs WHERE id = ?1",
                params![id],
                |row| {
                    Ok(Song {
                        id: row.get(0)?,
                        title: row.get(1)?,
                        composer: row.get(2)?,
                        composer_key: row.get(3)?,
                        era: row.get(4)?,
                        instrument: row.get(5)?,
                        license: row.get(6)?,
                        source: row.get(7)?,
                        source_url: row.get(8)?,
                        suggested_voice: row.get(9)?,
                        midi_path: PathBuf::from(row.get::<_, String>(10)?),
                        context: row.get(11)?,
                        tags: Vec::new(),
                    })
                },
            )
            .optional()?;
        let mut song = match opt {
            Some(s) => s,
            None => return Ok(None),
        };
        song.tags = self.song_tags(&song.id)?;
        Ok(Some(song))
    }

    /// Append a play history row. `song_id` and `voice_id` are
    /// permissive (free-form strings) so test harnesses can record
    /// synthetic ids without the foreign-key dance.
    pub fn record_play(
        &mut self,
        song_id: &str,
        voice_id: &str,
        duration_ms: Option<i64>,
    ) -> Result<i64> {
        let now = unix_now();
        self.conn.execute(
            "INSERT INTO plays (song_id, voice_id, played_at, duration_ms)
             VALUES (?1, ?2, ?3, ?4)",
            params![song_id, voice_id, now, duration_ms],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Total plays — diagnostic helper for tests / `keysynth-db query`.
    pub fn count_plays(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM plays", [], |row| row.get(0))?)
    }

    /// Total songs — diagnostic helper.
    pub fn count_songs(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM songs", [], |row| row.get(0))?)
    }

    /// Total voices — diagnostic helper.
    pub fn count_voices(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM voices", [], |row| row.get(0))?)
    }
}

// ─── helpers ─────────────────────────────────────────────────────────

fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Filename stem with extension stripped — `bach_bwv999_prelude.mid` →
/// `bach_bwv999_prelude`. Used as a stable song id (the JSON manifest
/// has no id field of its own; the file path is the natural key).
fn stem_of(file: &str) -> String {
    Path::new(file)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(file)
        .to_string()
}

/// Normalize a composer string into a lowercase last-name lookup key.
/// "J.S. Bach (1685-1750)" → "bach", "Tárrega" → "tarrega",
/// "Traditional (American)" → "traditional". Used so a CLI flag like
/// `--composer bach` matches without requiring exact UTF-8 punctuation.
fn composer_lookup_key(composer: &str) -> String {
    // Strip trailing parenthetical (years / nationality / etc.).
    let head = composer.split('(').next().unwrap_or(composer).trim();
    // Walk tokens and pick the last alphabetic-looking one. ASCII-fold
    // common diacritics so "Tárrega" → "tarrega".
    let mut last = head;
    for tok in head.split(|c: char| c.is_whitespace() || c == '.' || c == ',') {
        if tok.chars().any(|c| c.is_alphabetic()) {
            last = tok;
        }
    }
    fold_ascii_lowercase(last)
}

fn fold_ascii_lowercase(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'á' | 'à' | 'â' | 'ä' | 'ã' | 'å' => 'a',
            'é' | 'è' | 'ê' | 'ë' => 'e',
            'í' | 'ì' | 'î' | 'ï' => 'i',
            'ó' | 'ò' | 'ô' | 'ö' | 'õ' => 'o',
            'ú' | 'ù' | 'û' | 'ü' => 'u',
            'ñ' => 'n',
            'ç' => 'c',
            other => other.to_ascii_lowercase(),
        })
        .collect()
}

/// Best-effort era classification from a composer year-range. Reads
/// the trailing "(YYYY-YYYY)" parenthetical out of the composer string;
/// returns one of Baroque / Classical / Romantic / Modern / Traditional
/// based on the death year (or birth year as fallback). "Traditional"
/// applies whenever the parens contain "Traditional" instead of years.
///
/// This is editorial — eras blur at the edges. A composer who died in
/// 1820 spans Classical → early Romantic. The bucket boundaries used
/// here are the standard musicological consensus, picked once so
/// `--era Romantic` always lines up the same set of pieces:
///   Baroque   ≤ 1750  (Bach's death year — the canonical close-out)
///   Classical 1751-1820  (Beethoven mid-career)
///   Romantic  1821-1910  (covers late-Romantic guitar virtuosi —
///                         Tárrega 1909, Albéniz 1909, Sarasate 1908)
///   Modern    > 1910
fn derive_era(composer: &str) -> Option<String> {
    if composer.to_lowercase().contains("traditional") {
        return Some("Traditional".to_string());
    }
    let inside = composer.split_once('(')?.1.split_once(')')?.0;
    let nums: Vec<i32> = inside
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse().ok())
        .collect();
    let key_year = nums.last().or(nums.first()).copied()?;
    let era = match key_year {
        y if y <= 1750 => "Baroque",
        y if y <= 1820 => "Classical",
        y if y <= 1910 => "Romantic",
        _ => "Modern",
    };
    Some(era.to_string())
}

/// Locate the most recent built cdylib for `voices_live/<id>/`. Returns
/// the path + mtime so the catalog can show "stale" indicators when the
/// source has been edited since the last build. Multi-platform: we look
/// for `.so`, `.dll`, `.dylib`, prefer the newest.
fn locate_cdylib(crate_root: &Path, id: &str) -> (Option<PathBuf>, Option<i64>) {
    let target_dir = crate_root.join("target");
    let candidates = [
        target_dir.join("release").join(format!("libkeysynth_voice_{id}.so")),
        target_dir.join("release").join(format!("keysynth_voice_{id}.dll")),
        target_dir.join("release").join(format!("libkeysynth_voice_{id}.dylib")),
        target_dir.join("debug").join(format!("libkeysynth_voice_{id}.so")),
        target_dir.join("debug").join(format!("keysynth_voice_{id}.dll")),
        target_dir.join("debug").join(format!("libkeysynth_voice_{id}.dylib")),
    ];
    let mut best: Option<(PathBuf, i64)> = None;
    for p in candidates {
        if let Ok(meta) = std::fs::metadata(&p) {
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            best = match best.take() {
                Some((bp, bm)) if bm >= mtime => Some((bp, bm)),
                _ => Some((p, mtime)),
            };
        }
    }
    match best {
        Some((p, m)) => (Some(p), Some(m)),
        None => (None, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composer_key_strips_dates_and_diacritics() {
        assert_eq!(composer_lookup_key("J.S. Bach (1685-1750)"), "bach");
        assert_eq!(
            composer_lookup_key("Francisco Tárrega (1852-1909)"),
            "tarrega"
        );
        assert_eq!(
            composer_lookup_key("Traditional (American)"),
            "traditional"
        );
        assert_eq!(composer_lookup_key("Erik Satie (1866-1925)"), "satie");
    }

    #[test]
    fn era_buckets_by_death_year() {
        assert_eq!(derive_era("J.S. Bach (1685-1750)").as_deref(), Some("Baroque"));
        assert_eq!(
            derive_era("Wolfgang Amadeus Mozart (1756-1791)").as_deref(),
            Some("Classical")
        );
        // Tárrega 1909 — late-Romantic guitar virtuoso, must stay
        // under Romantic. The 1910 cutoff (vs the more conventional
        // 1900) is what keeps Tárrega + Albéniz in the same bucket
        // their actual idiom belongs to.
        assert_eq!(
            derive_era("Francisco Tárrega (1852-1909)").as_deref(),
            Some("Romantic")
        );
        // Satie 1925 — past the 1910 boundary, lands in Modern.
        assert_eq!(
            derive_era("Erik Satie (1866-1925)").as_deref(),
            Some("Modern")
        );
        assert_eq!(
            derive_era("Traditional (American)").as_deref(),
            Some("Traditional")
        );
    }

    #[test]
    fn migrate_creates_expected_tables() {
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
                "missing table {required} in {names:?}"
            );
        }
    }

    #[test]
    fn record_play_increments_count() {
        let mut db = LibraryDb::open_in_memory().unwrap();
        db.migrate().unwrap();
        assert_eq!(db.count_plays().unwrap(), 0);
        db.record_play("bach_bwv999_prelude", "guitar_stk", Some(12_500))
            .unwrap();
        db.record_play("satie_danse", "piano_modal", None).unwrap();
        assert_eq!(db.count_plays().unwrap(), 2);
    }
}
