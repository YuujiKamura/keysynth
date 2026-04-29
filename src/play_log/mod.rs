//! Local play-history + favorites store (issue #66 follow-up).
//!
//! Lives in a sibling SQLite file (`bench-out/play_log.db`, gitignored)
//! so machine-local listening doesn't churn the committed `library.db`
//! catalog. Two tables:
//!
//!   * `plays` — append-only log written every time the jukebox starts
//!     a track. `song_id` matches the stable file-stem id used by
//!     `library_db::Song::id` so joins against the catalog work without
//!     a foreign key (the catalog is read-only from this code path).
//!   * `favorites` — one row per starred song. `song_id` is the same
//!     stem, `added_at` is when the user first toggled it on.
//!
//! No foreign key into `songs(id)` because the catalog DB is a separate
//! file. Cross-DB FKs aren't a thing in SQLite, and tying the play log
//! to a specific catalog version would defeat the "machine-local user
//! state" goal — a user who deletes their library.db and rebuilds it
//! should keep their listening history.

use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OptionalExtension};

const SCHEMA_SQL: &str = r#"
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS plays (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    song_id     TEXT NOT NULL,
    voice_id    TEXT,
    played_at   INTEGER NOT NULL,
    duration_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_plays_song_id   ON plays(song_id);
CREATE INDEX IF NOT EXISTS idx_plays_played_at ON plays(played_at);

CREATE TABLE IF NOT EXISTS favorites (
    song_id  TEXT PRIMARY KEY,
    added_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS track_voices (
    song_id    TEXT PRIMARY KEY,
    voice_id   TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);

-- Multi-axis user marking (study / perform / revisit / compare / custom).
-- Composite key (song_id, tag) so the same song can carry several
-- intents at once. The UI seeds a small known palette but ad-hoc
-- strings are allowed so the user can grow their own taxonomy.
CREATE TABLE IF NOT EXISTS user_tags (
    song_id  TEXT NOT NULL,
    tag      TEXT NOT NULL,
    added_at INTEGER NOT NULL,
    PRIMARY KEY (song_id, tag)
);
CREATE INDEX IF NOT EXISTS idx_user_tags_tag ON user_tags(tag);

-- Per-song free-form note. Single row per song (upsert on edit). Empty
-- string deletes the row so notes(song_id) only ever holds songs the
-- user has actually written something about.
CREATE TABLE IF NOT EXISTS notes (
    song_id    TEXT PRIMARY KEY,
    note       TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);
"#;

#[derive(Debug)]
pub struct PlayLogDb {
    conn: Connection,
}

#[derive(Debug)]
pub enum PlayLogError {
    Sqlite(rusqlite::Error),
    Io(std::io::Error),
}

impl std::fmt::Display for PlayLogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sqlite(e) => write!(f, "sqlite: {e}"),
            Self::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for PlayLogError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Sqlite(e) => Some(e),
            Self::Io(e) => Some(e),
        }
    }
}

impl From<rusqlite::Error> for PlayLogError {
    fn from(e: rusqlite::Error) -> Self {
        Self::Sqlite(e)
    }
}

impl From<std::io::Error> for PlayLogError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, PlayLogError>;

#[derive(Debug, Clone)]
pub struct PlayEntry {
    pub id: i64,
    pub song_id: String,
    pub voice_id: Option<String>,
    pub played_at: i64,
    pub duration_ms: Option<i64>,
}

impl PlayLogDb {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        Ok(Self { conn })
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        Ok(Self { conn })
    }

    pub fn migrate(&mut self) -> Result<()> {
        self.conn.execute_batch(SCHEMA_SQL)?;
        Ok(())
    }

    /// Append a play row. `song_id` should be the catalog-stable stem
    /// (e.g. "bach_bwv999_prelude"); `voice_id` is the engine label
    /// the jukebox actually rendered through. Both are free-form so
    /// non-catalog tracks (iter-spam, chiptune fixtures) can still be
    /// counted.
    pub fn record_play(
        &mut self,
        song_id: &str,
        voice_id: Option<&str>,
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

    pub fn count_plays(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM plays", [], |row| row.get(0))?)
    }

    /// Most-recent-first window of play history. `limit` caps the
    /// returned vec size; pass a small number (10–50) for the GUI side
    /// panel and a larger one for an export.
    pub fn recent_plays(&self, limit: usize) -> Result<Vec<PlayEntry>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, song_id, voice_id, played_at, duration_ms
             FROM plays
             ORDER BY played_at DESC, id DESC
             LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(PlayEntry {
                id: row.get(0)?,
                song_id: row.get(1)?,
                voice_id: row.get(2)?,
                played_at: row.get(3)?,
                duration_ms: row.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Aggregated play-count per song. Used to render "🔥 top played"
    /// and to badge engine rows with their lifetime count.
    pub fn play_counts(&self) -> Result<HashMap<String, i64>> {
        let mut stmt = self
            .conn
            .prepare("SELECT song_id, COUNT(*) FROM plays GROUP BY song_id")?;
        let rows =
            stmt.query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)))?;
        let mut out = HashMap::new();
        for r in rows {
            let (k, v) = r?;
            out.insert(k, v);
        }
        Ok(out)
    }

    /// Leaderboard projection. Empty `limit` returns the full table.
    pub fn top_played(&self, limit: usize) -> Result<Vec<(String, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT song_id, COUNT(*) AS n
             FROM plays
             GROUP BY song_id
             ORDER BY n DESC, song_id ASC
             LIMIT ?1",
        )?;
        let rows =
            stmt.query_map(params![limit as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn is_favorite(&self, song_id: &str) -> Result<bool> {
        let opt: Option<i64> = self
            .conn
            .query_row(
                "SELECT 1 FROM favorites WHERE song_id = ?1",
                params![song_id],
                |row| row.get(0),
            )
            .optional()?;
        Ok(opt.is_some())
    }

    pub fn set_favorite(&mut self, song_id: &str, fav: bool) -> Result<()> {
        if fav {
            let now = unix_now();
            self.conn.execute(
                "INSERT OR IGNORE INTO favorites (song_id, added_at) VALUES (?1, ?2)",
                params![song_id, now],
            )?;
        } else {
            self.conn
                .execute("DELETE FROM favorites WHERE song_id = ?1", params![song_id])?;
        }
        Ok(())
    }

    /// Toggle and return the new state. One round-trip from the GUI
    /// thread for the common "click the star" interaction.
    pub fn toggle_favorite(&mut self, song_id: &str) -> Result<bool> {
        let new_state = !self.is_favorite(song_id)?;
        self.set_favorite(song_id, new_state)?;
        Ok(new_state)
    }

    /// All starred song ids, most-recently-added first.
    pub fn favorites(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT song_id FROM favorites ORDER BY added_at DESC, song_id ASC")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn count_favorites(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM favorites", [], |row| row.get(0))?)
    }

    /// Lookup the user's chosen voice for `song_id`. `None` means the
    /// user has never picked a voice for this track and the caller
    /// should fall back to the catalog suggested voice / heuristic.
    pub fn get_track_voice(&self, song_id: &str) -> Result<Option<String>> {
        let v: Option<String> = self
            .conn
            .query_row(
                "SELECT voice_id FROM track_voices WHERE song_id = ?1",
                params![song_id],
                |row| row.get(0),
            )
            .optional()?;
        Ok(v)
    }

    /// Persist the user's voice pick for `song_id`. Upsert so toggling
    /// back and forth reuses the row instead of bloating history.
    pub fn set_track_voice(&mut self, song_id: &str, voice_id: &str) -> Result<()> {
        let now = unix_now();
        self.conn.execute(
            "INSERT INTO track_voices (song_id, voice_id, updated_at) VALUES (?1, ?2, ?3)
             ON CONFLICT(song_id) DO UPDATE SET voice_id = excluded.voice_id,
                                                updated_at = excluded.updated_at",
            params![song_id, voice_id, now],
        )?;
        Ok(())
    }

    /// Snapshot of every persisted (song_id -> voice_id) pick, used by
    /// the GUI at startup so the catalog reflects the user's last
    /// session without a per-row SELECT.
    pub fn track_voices(&self) -> Result<HashMap<String, String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT song_id, voice_id FROM track_voices")?;
        let rows = stmt
            .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))?;
        let mut out = HashMap::new();
        for r in rows {
            let (k, v) = r?;
            out.insert(k, v);
        }
        Ok(out)
    }

    /// Insert or remove `(song_id, tag)`. `tag` is trimmed; empty
    /// strings are rejected so the UI can hand off raw text-input
    /// without pre-validating. Idempotent in both directions.
    pub fn set_user_tag(&mut self, song_id: &str, tag: &str, on: bool) -> Result<()> {
        let tag = tag.trim();
        if tag.is_empty() {
            return Ok(());
        }
        if on {
            let now = unix_now();
            self.conn.execute(
                "INSERT OR IGNORE INTO user_tags (song_id, tag, added_at) VALUES (?1, ?2, ?3)",
                params![song_id, tag, now],
            )?;
        } else {
            self.conn.execute(
                "DELETE FROM user_tags WHERE song_id = ?1 AND tag = ?2",
                params![song_id, tag],
            )?;
        }
        Ok(())
    }

    /// Toggle a single tag and return the new state. Mirrors
    /// `toggle_favorite` for the per-piece tag chips.
    pub fn toggle_user_tag(&mut self, song_id: &str, tag: &str) -> Result<bool> {
        let tag_t = tag.trim();
        if tag_t.is_empty() {
            return Ok(false);
        }
        let exists: Option<i64> = self
            .conn
            .query_row(
                "SELECT 1 FROM user_tags WHERE song_id = ?1 AND tag = ?2",
                params![song_id, tag_t],
                |row| row.get(0),
            )
            .optional()?;
        let new_state = exists.is_none();
        self.set_user_tag(song_id, tag_t, new_state)?;
        Ok(new_state)
    }

    /// Tags assigned to a single song, alphabetised for stable display.
    pub fn user_tags(&self, song_id: &str) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT tag FROM user_tags WHERE song_id = ?1 ORDER BY tag ASC",
        )?;
        let rows = stmt.query_map(params![song_id], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Snapshot of every (song_id -> tags) assignment so the GUI can
    /// render row-level chips without a per-row SELECT. Tags inside
    /// each Vec come back alphabetised.
    pub fn all_user_tags(&self) -> Result<HashMap<String, Vec<String>>> {
        let mut stmt = self
            .conn
            .prepare("SELECT song_id, tag FROM user_tags ORDER BY song_id, tag")?;
        let rows = stmt
            .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))?;
        let mut out: HashMap<String, Vec<String>> = HashMap::new();
        for r in rows {
            let (sid, tag) = r?;
            out.entry(sid).or_default().push(tag);
        }
        Ok(out)
    }

    /// Distinct tag vocabulary across the whole table, ordered by use
    /// frequency desc then alphabetically. Drives the "your tags" UI
    /// suggestion list.
    pub fn distinct_user_tags(&self) -> Result<Vec<(String, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT tag, COUNT(*) AS n FROM user_tags
             GROUP BY tag ORDER BY n DESC, tag ASC",
        )?;
        let rows = stmt
            .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Read the user's free-form note for a song. `None` = no row yet.
    pub fn get_note(&self, song_id: &str) -> Result<Option<String>> {
        let v: Option<String> = self
            .conn
            .query_row(
                "SELECT note FROM notes WHERE song_id = ?1",
                params![song_id],
                |row| row.get(0),
            )
            .optional()?;
        Ok(v)
    }

    /// Persist a note. Empty/whitespace-only payload deletes the row
    /// so `get_note` returns `None` again — keeps the table clean and
    /// matches the user mental model of "I cleared it".
    pub fn set_note(&mut self, song_id: &str, note: &str) -> Result<()> {
        let trimmed = note.trim();
        if trimmed.is_empty() {
            self.conn
                .execute("DELETE FROM notes WHERE song_id = ?1", params![song_id])?;
        } else {
            let now = unix_now();
            self.conn.execute(
                "INSERT INTO notes (song_id, note, updated_at) VALUES (?1, ?2, ?3)
                 ON CONFLICT(song_id) DO UPDATE SET note = excluded.note,
                                                    updated_at = excluded.updated_at",
                params![song_id, trimmed, now],
            )?;
        }
        Ok(())
    }

    /// Snapshot every (song_id -> note) row for in-memory mirror.
    pub fn all_notes(&self) -> Result<HashMap<String, String>> {
        let mut stmt = self.conn.prepare("SELECT song_id, note FROM notes")?;
        let rows = stmt
            .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))?;
        let mut out = HashMap::new();
        for r in rows {
            let (k, v) = r?;
            out.insert(k, v);
        }
        Ok(out)
    }

    /// Latest play timestamp per song, used to surface "last played
    /// 3d ago" in the detail pane without scanning `recent_plays`.
    pub fn last_played_at(&self, song_id: &str) -> Result<Option<i64>> {
        // Two-step: COUNT first so we don't fight rusqlite's NULL→i64
        // type coercion when the song has no rows. (`MAX(...)` over an
        // empty filtered set returns SQL NULL, which `row.get::<_,i64>`
        // reports as InvalidColumnType.)
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM plays WHERE song_id = ?1",
            params![song_id],
            |row| row.get(0),
        )?;
        if n == 0 {
            return Ok(None);
        }
        let v: i64 = self.conn.query_row(
            "SELECT MAX(played_at) FROM plays WHERE song_id = ?1",
            params![song_id],
            |row| row.get(0),
        )?;
        Ok(Some(v))
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }
}

fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> PlayLogDb {
        let mut db = PlayLogDb::open_in_memory().unwrap();
        db.migrate().unwrap();
        db
    }

    #[test]
    fn migrate_is_idempotent() {
        let mut db = fresh();
        // Second migrate must be a no-op against an existing schema.
        db.migrate().unwrap();
        let names: Vec<String> = db
            .conn()
            .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        for required in ["favorites", "notes", "plays", "track_voices", "user_tags"] {
            assert!(
                names.iter().any(|n| n == required),
                "missing table {required} in {names:?}"
            );
        }
    }

    #[test]
    fn record_play_and_recent_plays_round_trip() {
        let mut db = fresh();
        assert_eq!(db.count_plays().unwrap(), 0);
        db.record_play("bach_bwv999_prelude", Some("guitar_stk"), Some(12_500))
            .unwrap();
        db.record_play("satie_danse", Some("piano_modal"), None)
            .unwrap();
        db.record_play("bach_bwv999_prelude", Some("piano_modal"), Some(11_900))
            .unwrap();
        assert_eq!(db.count_plays().unwrap(), 3);
        let recent = db.recent_plays(10).unwrap();
        assert_eq!(recent.len(), 3);
        // Most recent first — our insert order means id=3 ("bach piano_modal") leads.
        assert_eq!(recent[0].id, 3);
        assert_eq!(recent[0].voice_id.as_deref(), Some("piano_modal"));
    }

    #[test]
    fn play_counts_and_top_played() {
        let mut db = fresh();
        for _ in 0..5 {
            db.record_play("fur_elise", Some("piano"), None).unwrap();
        }
        for _ in 0..2 {
            db.record_play("twinkle", Some("ks"), None).unwrap();
        }
        db.record_play("rare_tune", Some("ks"), None).unwrap();
        let counts = db.play_counts().unwrap();
        assert_eq!(counts.get("fur_elise"), Some(&5));
        assert_eq!(counts.get("twinkle"), Some(&2));
        assert_eq!(counts.get("rare_tune"), Some(&1));
        let top = db.top_played(2).unwrap();
        assert_eq!(top.len(), 2);
        assert_eq!(top[0], ("fur_elise".to_string(), 5));
        assert_eq!(top[1], ("twinkle".to_string(), 2));
    }

    #[test]
    fn favorite_toggle_flips_membership() {
        let mut db = fresh();
        assert!(!db.is_favorite("bach_bwv999_prelude").unwrap());
        let after_on = db.toggle_favorite("bach_bwv999_prelude").unwrap();
        assert!(after_on);
        assert!(db.is_favorite("bach_bwv999_prelude").unwrap());
        // Idempotent: explicit set to true on an already-on row stays on
        // (no INSERT-OR-IGNORE blowup) and doesn't bump added_at.
        db.set_favorite("bach_bwv999_prelude", true).unwrap();
        let after_off = db.toggle_favorite("bach_bwv999_prelude").unwrap();
        assert!(!after_off);
        assert!(!db.is_favorite("bach_bwv999_prelude").unwrap());
        assert_eq!(db.count_favorites().unwrap(), 0);
    }

    #[test]
    fn track_voice_round_trip() {
        let mut db = fresh();
        assert!(db.get_track_voice("bach_bwv999").unwrap().is_none());
        db.set_track_voice("bach_bwv999", "guitar-stk").unwrap();
        assert_eq!(
            db.get_track_voice("bach_bwv999").unwrap().as_deref(),
            Some("guitar-stk")
        );
        // Upsert: second set rewrites the same row.
        db.set_track_voice("bach_bwv999", "piano-modal").unwrap();
        assert_eq!(
            db.get_track_voice("bach_bwv999").unwrap().as_deref(),
            Some("piano-modal")
        );
        db.set_track_voice("scott_joplin_maple", "guitar-stk").unwrap();
        let all = db.track_voices().unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all.get("bach_bwv999"), Some(&"piano-modal".to_string()));
        assert_eq!(
            all.get("scott_joplin_maple"),
            Some(&"guitar-stk".to_string())
        );
    }

    #[test]
    fn user_tags_toggle_and_snapshot() {
        let mut db = fresh();
        assert!(db.user_tags("bach_bwv999").unwrap().is_empty());
        // Toggle on.
        let after_on = db.toggle_user_tag("bach_bwv999", "study").unwrap();
        assert!(after_on);
        assert_eq!(
            db.user_tags("bach_bwv999").unwrap(),
            vec!["study".to_string()]
        );
        // Stacks: a second tag joins the same song, alphabetised.
        db.set_user_tag("bach_bwv999", "perform", true).unwrap();
        assert_eq!(
            db.user_tags("bach_bwv999").unwrap(),
            vec!["perform".to_string(), "study".to_string()]
        );
        // Idempotent INSERT OR IGNORE: re-setting the same tag is a no-op.
        db.set_user_tag("bach_bwv999", "study", true).unwrap();
        assert_eq!(db.user_tags("bach_bwv999").unwrap().len(), 2);
        // Empty / whitespace tag is silently ignored (UI passes raw input).
        db.set_user_tag("bach_bwv999", "   ", true).unwrap();
        assert_eq!(db.user_tags("bach_bwv999").unwrap().len(), 2);
        // Toggle off.
        let after_off = db.toggle_user_tag("bach_bwv999", "study").unwrap();
        assert!(!after_off);
        assert_eq!(
            db.user_tags("bach_bwv999").unwrap(),
            vec!["perform".to_string()]
        );
        // Cross-song snapshot.
        db.set_user_tag("satie_danse", "revisit", true).unwrap();
        let all = db.all_user_tags().unwrap();
        assert_eq!(all.get("bach_bwv999"), Some(&vec!["perform".to_string()]));
        assert_eq!(all.get("satie_danse"), Some(&vec!["revisit".to_string()]));
        // Distinct vocabulary, frequency-sorted.
        db.set_user_tag("twinkle", "perform", true).unwrap();
        let vocab = db.distinct_user_tags().unwrap();
        // "perform" appears twice (bach + twinkle) so it leads.
        assert_eq!(vocab[0], ("perform".to_string(), 2));
    }

    #[test]
    fn notes_upsert_and_clear() {
        let mut db = fresh();
        assert!(db.get_note("bach_bwv999").unwrap().is_none());
        db.set_note("bach_bwv999", "  good warm-up etude\n").unwrap();
        // Persisted with whitespace trimmed.
        assert_eq!(
            db.get_note("bach_bwv999").unwrap().as_deref(),
            Some("good warm-up etude")
        );
        // Upsert: second write replaces the first.
        db.set_note("bach_bwv999", "actually a prelude").unwrap();
        assert_eq!(
            db.get_note("bach_bwv999").unwrap().as_deref(),
            Some("actually a prelude")
        );
        // Empty payload deletes the row.
        db.set_note("bach_bwv999", "   ").unwrap();
        assert!(db.get_note("bach_bwv999").unwrap().is_none());
        // Snapshot covers multiple songs.
        db.set_note("a", "note A").unwrap();
        db.set_note("b", "note B").unwrap();
        let all = db.all_notes().unwrap();
        assert_eq!(all.get("a"), Some(&"note A".to_string()));
        assert_eq!(all.get("b"), Some(&"note B".to_string()));
    }

    #[test]
    fn last_played_at_tracks_most_recent_row() {
        let db = fresh();
        // Hand-roll timestamps for determinism.
        db.conn
            .execute(
                "INSERT INTO plays (song_id, voice_id, played_at, duration_ms)
                 VALUES (?1, ?2, ?3, NULL)",
                params!["fur_elise", "piano", 1_000i64],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO plays (song_id, voice_id, played_at, duration_ms)
                 VALUES (?1, ?2, ?3, NULL)",
                params!["fur_elise", "piano", 5_000i64],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO plays (song_id, voice_id, played_at, duration_ms)
                 VALUES (?1, ?2, ?3, NULL)",
                params!["other", "ks", 3_000i64],
            )
            .unwrap();
        assert_eq!(db.last_played_at("fur_elise").unwrap(), Some(5_000));
        assert_eq!(db.last_played_at("other").unwrap(), Some(3_000));
        assert_eq!(db.last_played_at("never_played").unwrap(), None);
    }

    #[test]
    fn favorites_listed_most_recent_first() {
        let db = fresh();
        // Hand-roll the timestamps so the test is deterministic instead
        // of racing wall-clock seconds.
        db.conn
            .execute(
                "INSERT INTO favorites (song_id, added_at) VALUES (?1, ?2)",
                params!["older", 1_000i64],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO favorites (song_id, added_at) VALUES (?1, ?2)",
                params!["newer", 2_000i64],
            )
            .unwrap();
        let favs = db.favorites().unwrap();
        assert_eq!(favs, vec!["newer".to_string(), "older".to_string()]);
    }
}
