//! Stage F: blind A/B harness data layer (issue #67 Stage F).
//!
//! Owns the SQLite trial log + cumulative stats query for the
//! `ksabtest` GUI binary. Lives next to `play_log` because they share
//! the "machine-local listening state" niche — neither belongs in the
//! committed `library.db` catalog.
//!
//! Schema rationale:
//!   * `trials` is append-only — every blind A/B comparison ends in
//!     exactly one row, even if the user reloaded the same render.
//!   * `voice_a` / `voice_b` are stored canonically (alphabetical) so
//!     "piano-modal vs piano-thick" and "piano-thick vs piano-modal"
//!     bucket into the same pair for stats.
//!   * `voice_in_slot_a` records which canonical voice was assigned to
//!     the physical Slot A this trial. With the canonical pair this is
//!     enough to derive which voice the user picked from `verdict`.
//!   * `verdict` is the slot-relative answer (a/b/tie). Combined with
//!     `voice_in_slot_a` it yields the voice-relative answer.

use std::path::Path;

use rusqlite::{params, Connection};

const SCHEMA_SQL: &str = r#"
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS trials (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    voice_a          TEXT NOT NULL,
    voice_b          TEXT NOT NULL,
    voice_in_slot_a  TEXT NOT NULL,
    midi_path        TEXT NOT NULL,
    verdict          TEXT NOT NULL CHECK (verdict IN ('slot_a', 'slot_b', 'tie')),
    decided_at       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trials_pair ON trials(voice_a, voice_b);
CREATE INDEX IF NOT EXISTS idx_trials_decided_at ON trials(decided_at);
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    SlotA,
    SlotB,
    Tie,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::SlotA => "slot_a",
            Verdict::SlotB => "slot_b",
            Verdict::Tie => "tie",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "slot_a" => Some(Verdict::SlotA),
            "slot_b" => Some(Verdict::SlotB),
            "tie" => Some(Verdict::Tie),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Trial {
    pub id: i64,
    /// Canonical voice id (alphabetically smaller of the pair).
    pub voice_a: String,
    /// Canonical voice id (alphabetically larger of the pair).
    pub voice_b: String,
    /// Which canonical voice was placed in physical Slot A this trial.
    /// Always equals either `voice_a` or `voice_b`.
    pub voice_in_slot_a: String,
    pub midi_path: String,
    pub verdict: Verdict,
    pub decided_at: i64,
}

#[derive(Debug)]
pub enum AbError {
    Sqlite(rusqlite::Error),
    Io(std::io::Error),
}

impl std::fmt::Display for AbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sqlite(e) => write!(f, "sqlite: {e}"),
            Self::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for AbError {}

impl From<rusqlite::Error> for AbError {
    fn from(e: rusqlite::Error) -> Self {
        Self::Sqlite(e)
    }
}

impl From<std::io::Error> for AbError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, AbError>;

/// Canonicalise a voice pair to (smaller, larger) so two orderings of
/// the same pair bucket together in stats.
pub fn canonical_pair(x: &str, y: &str) -> (String, String) {
    if x <= y {
        (x.to_string(), y.to_string())
    } else {
        (y.to_string(), x.to_string())
    }
}

pub struct AbTestDb {
    conn: Connection,
}

impl AbTestDb {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        Ok(Self { conn })
    }

    pub fn open_in_memory() -> Result<Self> {
        Ok(Self {
            conn: Connection::open_in_memory()?,
        })
    }

    pub fn migrate(&mut self) -> Result<()> {
        self.conn.execute_batch(SCHEMA_SQL)?;
        Ok(())
    }

    /// Append a trial. `voice_a`/`voice_b` are canonicalised on the way
    /// in — callers can pass them in user-input order. `voice_in_slot_a`
    /// must be one of the two canonical names; this is the only piece
    /// the caller has to keep straight, because it encodes the random
    /// blind assignment that happened just before the user clicked.
    pub fn record_trial(
        &mut self,
        voice_a_in: &str,
        voice_b_in: &str,
        voice_in_slot_a: &str,
        midi_path: &str,
        verdict: Verdict,
    ) -> Result<i64> {
        let (voice_a, voice_b) = canonical_pair(voice_a_in, voice_b_in);
        if voice_in_slot_a != voice_a && voice_in_slot_a != voice_b {
            return Err(AbError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "voice_in_slot_a {voice_in_slot_a} is not in canonical pair ({voice_a}, {voice_b})"
                ),
            )));
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        self.conn.execute(
            "INSERT INTO trials (voice_a, voice_b, voice_in_slot_a, midi_path, verdict, decided_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![voice_a, voice_b, voice_in_slot_a, midi_path, verdict.as_str(), now],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// All trials for the canonical pair `(voice_a, voice_b)`. Order is
    /// most-recent first, capped at `limit` rows.
    pub fn trials_for_pair(
        &self,
        voice_a_in: &str,
        voice_b_in: &str,
        limit: usize,
    ) -> Result<Vec<Trial>> {
        let (voice_a, voice_b) = canonical_pair(voice_a_in, voice_b_in);
        let mut stmt = self.conn.prepare(
            "SELECT id, voice_a, voice_b, voice_in_slot_a, midi_path, verdict, decided_at
             FROM trials
             WHERE voice_a = ?1 AND voice_b = ?2
             ORDER BY decided_at DESC
             LIMIT ?3",
        )?;
        let rows = stmt.query_map(params![voice_a, voice_b, limit as i64], |row| {
            let v: String = row.get(5)?;
            Ok(Trial {
                id: row.get(0)?,
                voice_a: row.get(1)?,
                voice_b: row.get(2)?,
                voice_in_slot_a: row.get(3)?,
                midi_path: row.get(4)?,
                verdict: Verdict::parse(&v).unwrap_or(Verdict::Tie),
                decided_at: row.get(6)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// All trials, most-recent first, capped at `limit` rows.
    pub fn all_trials(&self, limit: usize) -> Result<Vec<Trial>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, voice_a, voice_b, voice_in_slot_a, midi_path, verdict, decided_at
             FROM trials
             ORDER BY decided_at DESC
             LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            let v: String = row.get(5)?;
            Ok(Trial {
                id: row.get(0)?,
                voice_a: row.get(1)?,
                voice_b: row.get(2)?,
                voice_in_slot_a: row.get(3)?,
                midi_path: row.get(4)?,
                verdict: Verdict::parse(&v).unwrap_or(Verdict::Tie),
                decided_at: row.get(6)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }
}

/// Voice-relative tally for one canonical pair: how many trials picked
/// each voice as "more real", how many were declared indistinguishable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PairStats {
    /// Trials where the user picked the canonical voice_a (regardless
    /// of whether it was physically in slot A or B that trial).
    pub a_wins: u32,
    pub b_wins: u32,
    pub ties: u32,
}

impl PairStats {
    pub fn total(&self) -> u32 {
        self.a_wins + self.b_wins + self.ties
    }

    /// Fraction of decisive trials (ties excluded) that picked
    /// canonical voice_a. Returns 0.5 when there are no decisive
    /// trials yet, so the UI can render a neutral 50% bar.
    pub fn a_share_decisive(&self) -> f32 {
        let decided = self.a_wins + self.b_wins;
        if decided == 0 {
            0.5
        } else {
            self.a_wins as f32 / decided as f32
        }
    }
}

/// Aggregate trials for a canonical pair into voice-relative wins.
/// Pure function over an already-loaded vector so the GUI thread can
/// re-aggregate cheaply each frame instead of re-querying SQLite.
pub fn aggregate(trials: &[Trial], voice_a_canonical: &str) -> PairStats {
    let mut s = PairStats::default();
    for t in trials {
        match t.verdict {
            Verdict::Tie => s.ties += 1,
            Verdict::SlotA => {
                // Slot A user pick → voice that was in slot A wins.
                if t.voice_in_slot_a == voice_a_canonical {
                    s.a_wins += 1;
                } else {
                    s.b_wins += 1;
                }
            }
            Verdict::SlotB => {
                // Slot B user pick → the *other* canonical voice wins.
                if t.voice_in_slot_a == voice_a_canonical {
                    s.b_wins += 1;
                } else {
                    s.a_wins += 1;
                }
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> AbTestDb {
        let mut db = AbTestDb::open_in_memory().expect("in-memory open");
        db.migrate().expect("migrate");
        db
    }

    #[test]
    fn canonicalises_pair_alphabetically() {
        assert_eq!(
            canonical_pair("piano-thick", "piano-modal"),
            ("piano-modal".into(), "piano-thick".into())
        );
        assert_eq!(
            canonical_pair("piano-modal", "piano-thick"),
            ("piano-modal".into(), "piano-thick".into())
        );
    }

    #[test]
    fn round_trip_trial() {
        let mut db = fresh();
        let id = db
            .record_trial(
                "piano-thick",
                "piano-modal",
                "piano-modal",
                "songs/bach.mid",
                Verdict::SlotA,
            )
            .expect("record");
        assert!(id > 0);
        let rows = db
            .trials_for_pair("piano-modal", "piano-thick", 10)
            .expect("query");
        assert_eq!(rows.len(), 1);
        let t = &rows[0];
        assert_eq!(t.voice_a, "piano-modal");
        assert_eq!(t.voice_b, "piano-thick");
        assert_eq!(t.voice_in_slot_a, "piano-modal");
        assert_eq!(t.verdict, Verdict::SlotA);
    }

    #[test]
    fn rejects_bogus_voice_in_slot_a() {
        let mut db = fresh();
        let res = db.record_trial(
            "piano-modal",
            "piano-thick",
            "totally-different-voice",
            "songs/bach.mid",
            Verdict::Tie,
        );
        assert!(res.is_err(), "should reject voice not in canonical pair");
    }

    #[test]
    fn aggregate_translates_slot_to_voice() {
        // Trial 1: piano-modal in slot A, user picked slot A
        //   → canonical voice_a (piano-modal) wins.
        // Trial 2: piano-thick in slot A, user picked slot A
        //   → piano-thick wins → canonical voice_b wins.
        // Trial 3: piano-modal in slot A, user picked slot B
        //   → piano-thick wins → canonical voice_b wins.
        // Trial 4: tie.
        let trials = vec![
            Trial {
                id: 1,
                voice_a: "piano-modal".into(),
                voice_b: "piano-thick".into(),
                voice_in_slot_a: "piano-modal".into(),
                midi_path: "x.mid".into(),
                verdict: Verdict::SlotA,
                decided_at: 1,
            },
            Trial {
                id: 2,
                voice_a: "piano-modal".into(),
                voice_b: "piano-thick".into(),
                voice_in_slot_a: "piano-thick".into(),
                midi_path: "x.mid".into(),
                verdict: Verdict::SlotA,
                decided_at: 2,
            },
            Trial {
                id: 3,
                voice_a: "piano-modal".into(),
                voice_b: "piano-thick".into(),
                voice_in_slot_a: "piano-modal".into(),
                midi_path: "x.mid".into(),
                verdict: Verdict::SlotB,
                decided_at: 3,
            },
            Trial {
                id: 4,
                voice_a: "piano-modal".into(),
                voice_b: "piano-thick".into(),
                voice_in_slot_a: "piano-modal".into(),
                midi_path: "x.mid".into(),
                verdict: Verdict::Tie,
                decided_at: 4,
            },
        ];
        let s = aggregate(&trials, "piano-modal");
        assert_eq!(s.a_wins, 1);
        assert_eq!(s.b_wins, 2);
        assert_eq!(s.ties, 1);
        assert_eq!(s.total(), 4);
        assert!((s.a_share_decisive() - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn aggregate_empty_yields_neutral_share() {
        let s = aggregate(&[], "piano-modal");
        assert_eq!(s.total(), 0);
        assert!((s.a_share_decisive() - 0.5).abs() < 1e-6);
    }
}
