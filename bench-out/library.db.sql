-- keysynth library catalog schema (Issue #66).
--
-- Materialized index over two source-of-truth manifests:
--
--   * voices      ← voices_live/<name>/Cargo.toml `[package.metadata.keysynth-voice]`
--   * songs       ← bench-out/songs/manifest.json
--
-- Neither manifest is touched at query time. `keysynth-db rebuild`
-- re-imports both so the DB is always reproducible from `git checkout +
-- voices_live/`. plays / play_log entries are local user state and are
-- written to a sibling `play_log.db` (gitignored), not into this catalog.
--
-- Idempotent: every CREATE is `IF NOT EXISTS` so `migrate()` can be
-- run repeatedly. INSERT OR REPLACE on the import side keeps the
-- catalog in sync with whatever's currently on disk.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- ---- songs --------------------------------------------------------
-- Public-domain MIDI catalog. id = filename stem (e.g.
-- "bach_bwv999_prelude") so cross-references with bench-out/songs/<id>.mid
-- are stable across importers.
CREATE TABLE IF NOT EXISTS songs (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    composer        TEXT NOT NULL,
    composer_key    TEXT NOT NULL,    -- normalized lookup key ("bach", "tarrega")
    era             TEXT,             -- Baroque / Classical / Romantic / Modern / Traditional
    instrument      TEXT NOT NULL,    -- guitar | piano | ...
    license         TEXT NOT NULL,
    source          TEXT,             -- "Mutopia Project"
    source_url      TEXT,
    suggested_voice TEXT,             -- e.g. "guitar-stk", "piano-modal"
    midi_path       TEXT NOT NULL,    -- "bench-out/songs/bach_bwv999_prelude.mid"
    context         TEXT,             -- editorial note from manifest.json
    imported_at     INTEGER NOT NULL  -- unix epoch seconds
);

CREATE INDEX IF NOT EXISTS idx_songs_composer    ON songs(composer_key);
CREATE INDEX IF NOT EXISTS idx_songs_era         ON songs(era);
CREATE INDEX IF NOT EXISTS idx_songs_instrument  ON songs(instrument);

-- ---- tags (song-side) --------------------------------------------
-- Free-form labels (etude / tremolo / dance / ragtime / ...) attached
-- per song. Many-to-many; deletion of the parent song cascades.
CREATE TABLE IF NOT EXISTS tags (
    song_id TEXT NOT NULL,
    tag     TEXT NOT NULL,
    PRIMARY KEY (song_id, tag),
    FOREIGN KEY (song_id) REFERENCES songs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);

-- ---- voices ------------------------------------------------------
-- voices_live/<id>/ plugin discovery snapshot. id = directory name
-- (e.g. "guitar_stk", "piano_modal"). Built-in static-catalog voices
-- (synth Square/KS/FM, sample SFZ/SF2) are imported with id prefix
-- "static:" so every selectable slot is enumerable from one query.
CREATE TABLE IF NOT EXISTS voices (
    id            TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL,
    category      TEXT NOT NULL,    -- Piano | Guitar | Synth | Samples | Custom
    recommend     TEXT NOT NULL,    -- Best | Stable | Experimental
    description   TEXT NOT NULL,
    slot_name     TEXT,             -- engine slot ref (live:guitar_stk, Square, ...)
    decay_model   TEXT,             -- optional [package.metadata.keysynth-voice.decay_model]
    cdylib_path   TEXT,             -- last-built voices_live/<id>/target/.../libkeysynth_voice_<id>.{so,dll,dylib}
    cdylib_mtime  INTEGER,          -- mtime in unix seconds, NULL if not built
    imported_at   INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_voices_category   ON voices(category);
CREATE INDEX IF NOT EXISTS idx_voices_recommend  ON voices(recommend);

-- ---- voice_tags --------------------------------------------------
-- Free-form labels for voices (physical-model / sampled / lightweight / ...).
CREATE TABLE IF NOT EXISTS voice_tags (
    voice_id TEXT NOT NULL,
    tag      TEXT NOT NULL,
    PRIMARY KEY (voice_id, tag),
    FOREIGN KEY (voice_id) REFERENCES voices(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_voice_tags_tag ON voice_tags(tag);

-- ---- plays -------------------------------------------------------
-- Local play history. Lives in this file for the e2e gate, but the
-- production write path uses a sibling play_log.db that's gitignored
-- so machine-local listening doesn't bloat the committed catalog.
CREATE TABLE IF NOT EXISTS plays (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    song_id     TEXT,
    voice_id    TEXT,
    played_at   INTEGER NOT NULL,    -- unix epoch seconds
    duration_ms INTEGER
);

CREATE INDEX IF NOT EXISTS idx_plays_song_id  ON plays(song_id);
CREATE INDEX IF NOT EXISTS idx_plays_voice_id ON plays(voice_id);
