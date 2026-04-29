//! `voice_collector` — Stage D: public-domain SFZ catalog acquisition CLI.
//!
//! Mirrors the role of Stage B's song collector: pin a curated list of
//! CC0 / CC-BY SFZ libraries in `samples/manifest.json`, fetch each
//! `.sfz` text manifest into `samples/sfz/`, and import the catalog
//! into `bench-out/library.db` as `voices` rows with id prefix
//! `sample:`. WAV samples referenced by the SFZ stay upstream — see
//! the `samples_root` URL on each manifest entry.
//!
//! Subcommands
//! -----------
//!   voice_collector list                            # describe manifest entries
//!   voice_collector fetch [--id ID] [--force]       # download .sfz files
//!   voice_collector import [--db PATH]              # write rows to library.db
//!   voice_collector audit                           # license + on-disk presence check
//!
//! Defaults
//! --------
//!   --manifest samples/manifest.json
//!   --db       bench-out/library.db
//!   --root     . (repo root; resolves manifest's `directory` field)

use std::path::PathBuf;
use std::process::ExitCode;

use keysynth::library_db::LibraryDb;
use keysynth::voice_collector::{
    fetch_all, fetch_entry, summarize_sfz, FetchOutcome, SampleManifest,
};

const DEFAULT_MANIFEST: &str = "samples/manifest.json";
const DEFAULT_DB: &str = "bench-out/library.db";

fn print_help() {
    eprintln!(
        "voice_collector — Stage D SFZ catalog pipeline\n\n\
         usage:\n  \
         voice_collector list                          describe manifest entries\n  \
         voice_collector fetch [--id ID] [--force]     download .sfz files via curl\n  \
         voice_collector import                        upsert sample rows into library.db\n  \
         voice_collector audit                         license + presence check (no writes)\n\n\
         common:\n  \
         --manifest PATH        override samples/manifest.json\n  \
         --db PATH              override bench-out/library.db\n  \
         --root PATH            override repo root (resolves manifest's `directory`)\n  \
         --json                 emit JSON instead of text rows (list / audit)"
    );
}

#[derive(Default)]
struct Args {
    cmd: Option<String>,
    manifest: Option<PathBuf>,
    db: Option<PathBuf>,
    root: Option<PathBuf>,
    id: Option<String>,
    force: bool,
    json: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut iter = std::env::args().skip(1);
    let mut a = Args::default();
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
            "--manifest" => {
                a.manifest = Some(PathBuf::from(
                    iter.next().ok_or("--manifest needs a value")?,
                ))
            }
            "--db" => a.db = Some(PathBuf::from(iter.next().ok_or("--db needs a value")?)),
            "--root" => a.root = Some(PathBuf::from(iter.next().ok_or("--root needs a value")?)),
            "--id" => a.id = Some(iter.next().ok_or("--id needs a value")?),
            "--force" => a.force = true,
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

fn run(args: Args) -> Result<(), String> {
    let manifest_path = args
        .manifest
        .clone()
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MANIFEST));
    let root = args.root.clone().unwrap_or_else(|| PathBuf::from("."));

    let manifest = SampleManifest::load(&manifest_path)
        .map_err(|e| format!("load {}: {e}", manifest_path.display()))?;

    match args.cmd.as_deref().unwrap_or("") {
        "list" => cmd_list(&manifest, &root, &args),
        "fetch" => cmd_fetch(&manifest, &root, &args),
        "import" => cmd_import(&manifest, &args),
        "audit" => cmd_audit(&manifest, &root, &args),
        other => Err(format!("unknown subcommand: {other}")),
    }
}

fn cmd_list(manifest: &SampleManifest, root: &std::path::Path, args: &Args) -> Result<(), String> {
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&manifest.entries)
                .map_err(|e| format!("serialize: {e}"))?
        );
        return Ok(());
    }
    println!("# {} sample entries (manifest schema v{})", manifest.entries.len(), manifest.schema_version);
    println!(
        "  {:<32}  {:<10}  {:<8}  {:<13}  status",
        "id", "instrument", "license", "category"
    );
    for e in &manifest.entries {
        let p = manifest.entry_path(root, e);
        let status = if p.exists() {
            match summarize_sfz(&p) {
                Ok(s) => {
                    let inc = if s.include_count > 0 {
                        format!(", {} #includes", s.include_count)
                    } else {
                        String::new()
                    };
                    format!(
                        "ok  ({} regions, key {}-{}{})",
                        s.region_count,
                        s.lokey.map(|n| n.to_string()).unwrap_or_else(|| "?".into()),
                        s.hikey.map(|n| n.to_string()).unwrap_or_else(|| "?".into()),
                        inc,
                    )
                },
                Err(_) => "ok  (unparseable)".to_string(),
            }
        } else {
            "MISSING (run `voice_collector fetch`)".to_string()
        };
        println!(
            "  {:<32}  {:<10}  {:<8}  {:<13}  {}",
            e.id, e.instrument, e.license, e.category, status
        );
    }
    Ok(())
}

fn cmd_fetch(manifest: &SampleManifest, root: &std::path::Path, args: &Args) -> Result<(), String> {
    if let Some(id) = &args.id {
        let entry = manifest
            .entries
            .iter()
            .find(|e| &e.id == id)
            .ok_or_else(|| format!("no entry with id={id} in manifest"))?;
        let outcome = fetch_entry(manifest, entry, root, args.force)
            .map_err(|e| format!("fetch {id}: {e}"))?;
        report_fetch(&entry.id, &outcome);
        return Ok(());
    }
    let outcomes = fetch_all(manifest, root, args.force).map_err(|e| format!("fetch_all: {e}"))?;
    let mut fetched = 0usize;
    for (id, o) in &outcomes {
        if o.was_fetched() {
            fetched += 1;
        }
        report_fetch(id, o);
    }
    eprintln!(
        "voice_collector: {} fetched, {} skipped, {} total",
        fetched,
        outcomes.len() - fetched,
        outcomes.len()
    );
    Ok(())
}

fn cmd_audit(manifest: &SampleManifest, root: &std::path::Path, args: &Args) -> Result<(), String> {
    manifest
        .audit_licenses()
        .map_err(|e| format!("license audit failed: {e}"))?;
    let mut missing = Vec::new();
    let mut total_regions = 0usize;
    for e in &manifest.entries {
        let p = manifest.entry_path(root, e);
        if !p.exists() {
            missing.push(e.id.clone());
            continue;
        }
        if let Ok(s) = summarize_sfz(&p) {
            total_regions += s.region_count;
        }
    }
    if args.json {
        println!(
            "{{\"entries\":{},\"missing\":{},\"total_regions\":{}}}",
            manifest.entries.len(),
            missing.len(),
            total_regions
        );
    } else {
        eprintln!(
            "voice_collector audit: {} entries, {} on-disk OK, {} missing, {} total regions",
            manifest.entries.len(),
            manifest.entries.len() - missing.len(),
            missing.len(),
            total_regions,
        );
        for id in &missing {
            eprintln!("  missing: {id}");
        }
    }
    if !missing.is_empty() {
        return Err(format!(
            "{} sfz file(s) missing — run `voice_collector fetch`",
            missing.len()
        ));
    }
    Ok(())
}

fn cmd_import(manifest: &SampleManifest, args: &Args) -> Result<(), String> {
    let db_path = args.db.clone().unwrap_or_else(|| PathBuf::from(DEFAULT_DB));
    let manifest_path = args
        .manifest
        .clone()
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MANIFEST));
    // License audit is the import gate too — no point inserting rows
    // we'd then have to delete on the next CI run when a CC-BY-NC
    // entry slipped through.
    manifest
        .audit_licenses()
        .map_err(|e| format!("license audit failed: {e}"))?;
    let mut db = LibraryDb::open(&db_path).map_err(|e| format!("open {}: {e}", db_path.display()))?;
    db.migrate().map_err(|e| format!("migrate: {e}"))?;
    let n = db
        .import_samples(&manifest_path)
        .map_err(|e| format!("import_samples: {e}"))?;
    eprintln!(
        "voice_collector: imported {n} sample voices into {} (id prefix `sample:`)",
        db_path.display()
    );
    Ok(())
}

fn report_fetch(id: &str, o: &FetchOutcome) {
    match o {
        FetchOutcome::Fetched { path, bytes } => {
            eprintln!("  fetched  {id:<32}  {bytes:>7} B  {}", path.display());
        }
        FetchOutcome::Skipped { reason, path } => {
            eprintln!("  skipped  {id:<32}  ({reason})  {}", path.display());
        }
    }
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("voice_collector: {e}");
            print_help();
            return ExitCode::from(2);
        }
    };
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("voice_collector: {e}");
            ExitCode::from(1)
        }
    }
}
