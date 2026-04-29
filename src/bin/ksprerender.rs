//! Pre-render every (song, voice) MIDI preview into the on-disk
//! `preview_cache` so the jukebox UI never spawns a foreground render
//! on click. Designed to run as a one-shot CLI before launching the
//! jukebox (or as a periodic task after dropping new MIDIs into
//! `bench-out/songs/`).
//!
//! Usage:
//!   ksprerender [--songs DIR]      (default: bench-out/songs)
//!               [--voices V,V,...] (default: guitar-stk,piano-modal)
//!               [--cache-dir DIR]  (default: bench-out/cache)
//!               [--render-bin P]   (default: <ksprerender's dir>/render_midi[.exe])
//!               [--max-bytes N]    (default: 1 GiB)
//!               [--limit N]        (process only the first N MIDI files;
//!                                    sort is stem-alphabetical)
//!               [--dry-run]        (print plan, run no renders)
//!
//! The renderer is invoked via subprocess (same path as
//! `preview_cache::render_to_cache`); the cache key matches the
//! jukebox's `build_midi_cache_key` so jukebox lookups land as hits
//! without a second render.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use keysynth::preview_cache::{Cache, CacheKey, RenderParams};

const DEFAULT_SONGS_DIR: &str = "bench-out/songs";
const DEFAULT_CACHE_DIR: &str = "bench-out/cache";
const DEFAULT_VOICES: &[&str] = &["guitar-stk", "piano-modal"];
const DEFAULT_MAX_BYTES: u64 = 1_073_741_824; // 1 GiB

struct Args {
    songs_dir: PathBuf,
    voices: Vec<String>,
    cache_dir: PathBuf,
    render_bin: Option<PathBuf>,
    max_bytes: u64,
    limit: Option<usize>,
    dry_run: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut songs_dir = PathBuf::from(DEFAULT_SONGS_DIR);
    let mut voices: Vec<String> = DEFAULT_VOICES.iter().map(|s| (*s).to_string()).collect();
    let mut cache_dir = PathBuf::from(DEFAULT_CACHE_DIR);
    let mut render_bin: Option<PathBuf> = None;
    let mut max_bytes: u64 = DEFAULT_MAX_BYTES;
    let mut limit: Option<usize> = None;
    let mut dry_run = false;

    let mut iter = std::env::args().skip(1);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--songs" => {
                songs_dir = PathBuf::from(iter.next().ok_or("--songs needs a value")?);
            }
            "--voices" => {
                let v = iter.next().ok_or("--voices needs a value")?;
                voices = v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
                if voices.is_empty() {
                    return Err("--voices: empty list after parse".into());
                }
            }
            "--cache-dir" => {
                cache_dir = PathBuf::from(iter.next().ok_or("--cache-dir needs a value")?);
            }
            "--render-bin" => {
                render_bin = Some(PathBuf::from(
                    iter.next().ok_or("--render-bin needs a value")?,
                ));
            }
            "--max-bytes" => {
                max_bytes = iter
                    .next()
                    .ok_or("--max-bytes needs a value")?
                    .parse()
                    .map_err(|e| format!("--max-bytes parse: {e}"))?;
            }
            "--limit" => {
                limit = Some(
                    iter.next()
                        .ok_or("--limit needs a value")?
                        .parse()
                        .map_err(|e| format!("--limit parse: {e}"))?,
                );
            }
            "--dry-run" => dry_run = true,
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }

    Ok(Args {
        songs_dir,
        voices,
        cache_dir,
        render_bin,
        max_bytes,
        limit,
        dry_run,
    })
}

fn print_usage() {
    eprintln!(
        "ksprerender — pre-warm the jukebox preview cache.\n\n\
         Usage:\n\
           ksprerender [--songs DIR] [--voices V,V,...] [--cache-dir DIR]\n\
                       [--render-bin PATH] [--max-bytes N] [--limit N] [--dry-run]\n\n\
         Defaults: --songs {DEFAULT_SONGS_DIR}  --voices {voices}  --cache-dir {DEFAULT_CACHE_DIR}\n",
        voices = DEFAULT_VOICES.join(","),
    );
}

/// Locate `render_midi[.exe]` next to this binary. Mirrors the
/// jukebox's discovery path so both code paths agree on which renderer
/// produced a given cache entry.
fn discover_render_bin() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let name = if cfg!(windows) {
        "render_midi.exe"
    } else {
        "render_midi"
    };
    let cand = dir.join(name);
    if cand.is_file() {
        Some(cand)
    } else {
        None
    }
}

fn list_midi_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
    for ent in entries {
        let ent = ent.map_err(|e| format!("read_dir entry {}: {e}", dir.display()))?;
        let path = ent.path();
        if !path.is_file() {
            continue;
        }
        match path.extension().and_then(|s| s.to_str()) {
            Some(e) if e.eq_ignore_ascii_case("mid") || e.eq_ignore_ascii_case("midi") => {
                out.push(path);
            }
            _ => {}
        }
    }
    out.sort();
    Ok(out)
}

fn key_for(song: &Path, voice: &str) -> CacheKey {
    CacheKey {
        song_path: song.to_path_buf(),
        voice_id: voice.to_string(),
        voice_dll: None,
        render_params: RenderParams::default(),
    }
}

fn run(args: Args) -> Result<(), String> {
    let render_bin = match args.render_bin {
        Some(p) => p,
        None => discover_render_bin().ok_or_else(|| {
            "render_midi binary not found beside ksprerender. Pass --render-bin or run from \
             the same target dir."
                .to_string()
        })?,
    };
    if !render_bin.is_file() {
        return Err(format!(
            "render_midi binary not found at {}",
            render_bin.display()
        ));
    }

    let cache = Cache::new(&args.cache_dir, args.max_bytes)
        .map_err(|e| format!("Cache::new {}: {e}", args.cache_dir.display()))?;

    let mut songs = list_midi_files(&args.songs_dir)?;
    if let Some(n) = args.limit {
        songs.truncate(n);
    }

    let total = songs.len() * args.voices.len();
    println!(
        "ksprerender: songs={} voices={} → {total} (song,voice) pairs  cache={}  render={}",
        songs.len(),
        args.voices.len(),
        cache.dir().display(),
        render_bin.display(),
    );
    if total == 0 {
        println!("ksprerender: nothing to do.");
        return Ok(());
    }

    let mut hits = 0_usize;
    let mut misses = 0_usize;
    let mut errors = 0_usize;
    let mut total_render_ms: u128 = 0;
    let started = Instant::now();
    let mut idx = 0_usize;
    for song in &songs {
        for voice in &args.voices {
            idx += 1;
            let key = key_for(song, voice);
            let stem = song.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
            // Cache-only check first so the dry-run mode can still
            // distinguish "would render" from "already cached".
            let already = cache
                .lookup(&key)
                .map_err(|e| format!("lookup {}: {e}", song.display()))?;
            if already.is_some() {
                hits += 1;
                println!("[{idx:>3}/{total}] HIT  {stem}  voice={voice}");
                continue;
            }
            if args.dry_run {
                misses += 1;
                println!("[{idx:>3}/{total}] PLAN {stem}  voice={voice}");
                continue;
            }
            let t0 = Instant::now();
            match keysynth::preview_cache::render_to_cache(&cache, &key, voice, &render_bin) {
                Ok(_) => {
                    let ms = t0.elapsed().as_millis();
                    total_render_ms += ms;
                    misses += 1;
                    println!("[{idx:>3}/{total}] MISS {stem}  voice={voice}  rendered in {ms} ms");
                }
                Err(e) => {
                    errors += 1;
                    eprintln!("[{idx:>3}/{total}] FAIL {stem}  voice={voice}: {e}");
                }
            }
        }
    }

    if !args.dry_run {
        match cache.evict_lru() {
            Ok(n) if n > 0 => println!("ksprerender: evicted {n} entries to honour cap"),
            Ok(_) => {}
            Err(e) => eprintln!("ksprerender: evict_lru: {e}"),
        }
    }

    let wall_ms = started.elapsed().as_millis();
    println!(
        "ksprerender: done. hits={hits} misses={misses} errors={errors} wall={wall_ms}ms render_total={total_render_ms}ms",
    );
    if errors > 0 {
        return Err(format!("{errors} render(s) failed"));
    }
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("ksprerender: {e}");
            print_usage();
            return ExitCode::from(2);
        }
    };
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ksprerender: {e}");
            ExitCode::from(1)
        }
    }
}
