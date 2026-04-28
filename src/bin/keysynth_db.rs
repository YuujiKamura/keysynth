//! `keysynth_db` — maintenance CLI for the materialized library catalog
//! (issue #66).
//!
//! Subcommands
//! -----------
//!
//!   keysynth_db migrate
//!       Apply schema to bench-out/library.db (idempotent).
//!
//!   keysynth_db rebuild
//!       Re-import voices_live/* + bench-out/songs/manifest.json.
//!
//!   keysynth_db query [--composer NAME] [--era ERA] [--instrument I]
//!                     [--tag T] [--recommended-voice V] [--voices]
//!       Print rows matching the filter. Default: songs grouped by
//!       composer. `--voices` switches to voice rows.
//!
//! Defaults to `bench-out/library.db` so you can run it from the repo
//! root without arguments. `--db PATH` overrides.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use keysynth::library_db::{LibraryDb, SongFilter, SongSort, VoiceFilter};

const DEFAULT_DB: &str = "bench-out/library.db";
const DEFAULT_MANIFEST: &str = "bench-out/songs/manifest.json";
const DEFAULT_VOICES_LIVE: &str = "voices_live";

fn print_help() {
    eprintln!(
        "keysynth_db — library catalog maintenance CLI (issue #66)\n\n\
         usage:\n  \
         keysynth_db migrate                   # apply schema\n  \
         keysynth_db rebuild                   # re-import songs + voices\n  \
         keysynth_db query [filters]           # list songs\n  \
         keysynth_db query --voices [filters]  # list voices\n\n\
         common:\n  \
         --db PATH                  override bench-out/library.db\n  \
         --manifest PATH            override songs manifest path\n  \
         --voices-live ROOT         override voices_live/ root\n\n\
         query filters:\n  \
         --composer NAME            substring match on composer key (\"bach\", \"tarrega\")\n  \
         --era ERA                  Baroque|Classical|Romantic|Modern|Traditional\n  \
         --instrument INST          guitar | piano | ...\n  \
         --tag T                    exact tag match\n  \
         --recommended-voice V      song's `suggested_voice` matches V\n  \
         --category CAT             (--voices only) Piano|Guitar|Synth|Samples|Custom\n  \
         --json                     emit JSON instead of text rows"
    );
}

#[derive(Default)]
struct CliArgs {
    cmd: Option<String>,
    db: Option<PathBuf>,
    manifest: Option<PathBuf>,
    voices_live: Option<PathBuf>,
    composer: Option<String>,
    era: Option<String>,
    instrument: Option<String>,
    tag: Option<String>,
    recommended_voice: Option<String>,
    category: Option<String>,
    show_voices: bool,
    json: bool,
}

fn parse_args() -> Result<CliArgs, String> {
    let mut iter = std::env::args().skip(1);
    let mut a = CliArgs::default();
    let Some(cmd) = iter.next() else {
        return Err("missing subcommand".to_string());
    };
    if matches!(cmd.as_str(), "--help" | "-h" | "help") {
        print_help();
        std::process::exit(0);
    }
    a.cmd = Some(cmd);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--db" => a.db = Some(PathBuf::from(iter.next().ok_or("--db needs a value")?)),
            "--manifest" => {
                a.manifest = Some(PathBuf::from(
                    iter.next().ok_or("--manifest needs a value")?,
                ))
            }
            "--voices-live" => {
                a.voices_live = Some(PathBuf::from(
                    iter.next().ok_or("--voices-live needs a value")?,
                ))
            }
            "--composer" => a.composer = Some(iter.next().ok_or("--composer needs a value")?),
            "--era" => a.era = Some(iter.next().ok_or("--era needs a value")?),
            "--instrument" => {
                a.instrument = Some(iter.next().ok_or("--instrument needs a value")?)
            }
            "--tag" => a.tag = Some(iter.next().ok_or("--tag needs a value")?),
            "--recommended-voice" => {
                a.recommended_voice = Some(iter.next().ok_or("--recommended-voice needs a value")?)
            }
            "--category" => a.category = Some(iter.next().ok_or("--category needs a value")?),
            "--voices" => a.show_voices = true,
            "--json" => a.json = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    Ok(a)
}

fn run(args: CliArgs) -> Result<(), String> {
    let db_path = args.db.clone().unwrap_or_else(|| PathBuf::from(DEFAULT_DB));
    let mut db = LibraryDb::open(&db_path).map_err(|e| format!("open {}: {e}", db_path.display()))?;

    match args.cmd.as_deref().unwrap_or("") {
        "migrate" => {
            db.migrate().map_err(|e| format!("migrate: {e}"))?;
            eprintln!("keysynth_db: schema applied → {}", db_path.display());
            Ok(())
        }
        "rebuild" => {
            let manifest = args
                .manifest
                .clone()
                .unwrap_or_else(|| PathBuf::from(DEFAULT_MANIFEST));
            let vl = args
                .voices_live
                .clone()
                .unwrap_or_else(|| PathBuf::from(DEFAULT_VOICES_LIVE));
            let (songs, voices) = db
                .rebuild(&manifest, &vl)
                .map_err(|e| format!("rebuild: {e}"))?;
            eprintln!(
                "keysynth_db: rebuilt {} ({} songs from {}, {} voices from {})",
                db_path.display(),
                songs,
                manifest.display(),
                voices,
                vl.display(),
            );
            Ok(())
        }
        "query" => {
            // For convenience let `query` work against an un-migrated
            // file: applying the schema is idempotent and lets a user
            // run `keysynth_db query` without first remembering to
            // `migrate`.
            db.migrate().map_err(|e| format!("migrate: {e}"))?;
            if args.show_voices {
                run_query_voices(&db, &args)
            } else {
                run_query_songs(&db, &args)
            }
        }
        other => Err(format!("unknown subcommand: {other}")),
    }
}

fn run_query_songs(db: &LibraryDb, args: &CliArgs) -> Result<(), String> {
    let filter = SongFilter {
        composer: args.composer.clone(),
        era: args.era.clone(),
        instrument: args.instrument.clone(),
        tag: args.tag.clone(),
        recommended_voice: args.recommended_voice.clone(),
        sort: SongSort::ByComposer,
    };
    let songs = db
        .query_songs(&filter)
        .map_err(|e| format!("query: {e}"))?;
    if args.json {
        // Hand-rolled JSON to avoid pulling serde derive on the public
        // Song struct (kept slim on purpose). Field set mirrors what
        // jukebox / kssong consumers actually use.
        println!("[");
        for (i, s) in songs.iter().enumerate() {
            print!(
                "  {{\"id\":\"{}\", \"title\":\"{}\", \"composer\":\"{}\", \"era\":{}, \
                  \"instrument\":\"{}\", \"suggested_voice\":{}, \"midi_path\":\"{}\"}}",
                json_escape(&s.id),
                json_escape(&s.title),
                json_escape(&s.composer),
                opt_string(&s.era),
                json_escape(&s.instrument),
                opt_string(&s.suggested_voice),
                json_escape(&s.midi_path.to_string_lossy()),
            );
            if i + 1 < songs.len() {
                print!(",");
            }
            println!();
        }
        println!("]");
    } else {
        println!("# {} songs", songs.len());
        for s in &songs {
            let era = s.era.as_deref().unwrap_or("?");
            let voice = s.suggested_voice.as_deref().unwrap_or("?");
            println!(
                "  {:<28}  {:<10}  {:<7}  {:<11}  {}",
                s.id,
                era,
                s.instrument,
                voice,
                s.title.lines().next().unwrap_or(""),
            );
        }
    }
    Ok(())
}

fn run_query_voices(db: &LibraryDb, args: &CliArgs) -> Result<(), String> {
    let filter = VoiceFilter {
        category: args.category.clone(),
        recommend: None,
    };
    let voices = db
        .query_voices(&filter)
        .map_err(|e| format!("query voices: {e}"))?;
    if args.json {
        println!("[");
        for (i, v) in voices.iter().enumerate() {
            print!(
                "  {{\"id\":\"{}\", \"display_name\":\"{}\", \"category\":\"{}\", \
                  \"recommend\":\"{}\", \"description\":\"{}\"}}",
                json_escape(&v.id),
                json_escape(&v.display_name),
                json_escape(&v.category),
                json_escape(&v.recommend),
                json_escape(&v.description),
            );
            if i + 1 < voices.len() {
                print!(",");
            }
            println!();
        }
        println!("]");
    } else {
        println!("# {} voices", voices.len());
        for v in &voices {
            println!(
                "  {:<22}  {:<8}  {:<13}  {}",
                v.id, v.category, v.recommend, v.description
            );
        }
    }
    Ok(())
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn opt_string(s: &Option<String>) -> String {
    match s {
        Some(v) => format!("\"{}\"", json_escape(v)),
        None => "null".to_string(),
    }
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("keysynth_db: {e}");
            print_help();
            return ExitCode::from(2);
        }
    };
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("keysynth_db: {e}");
            ExitCode::from(1)
        }
    }
}

#[allow(dead_code)]
fn ensure_default_paths_exist() -> bool {
    Path::new(DEFAULT_MANIFEST).exists() && Path::new(DEFAULT_VOICES_LIVE).exists()
}
