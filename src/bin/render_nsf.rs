//! Render an NSF (Nintendo Sound Format) track to a WAV file via libgme.
//!
//! Why this exists:
//!   Listener-AI (Gemini multimodal) transcribes audio into a Song schema,
//!   but we have no way to tell if the transcription is *right* — there's
//!   no ground truth. NSF is the original NES game's sound code dumped to
//!   a self-contained format; libgme runs that code on a 2A03 emulator and
//!   produces bit-exact original audio. So:
//!     listener_<song>_square.wav  = "what the AI thinks it sounds like"
//!     nsf_<song>_pure.wav         = "what it actually sounds like"
//!   Comparing the two turns Listener-AI from a hopeful black box into a
//!   measurable system.
//!
//! Why we bypass the crate's safe wrapper:
//!   game-music-emu 0.3.0's `EmuHandle::new` does
//!       Arc::new(transmute(emu))
//!   where `emu: *const MusicEmu`. On Windows MSVC this lands UB — the Arc
//!   tries to deref a non-Arc pointer for refcount tracking and segfaults
//!   in from_file/new_emu before any audio comes out. So we pull the crate
//!   in purely as a build-time provider of statically-linked libgme symbols
//!   (`gme_*` C ABI), declare them ourselves, and manage the raw pointer
//!   without Arc gymnastics.
//!
//! Usage:
//!     render_nsf --in <FILE.nsf> --track <N> --duration <SEC> --out <OUT.wav>
//!         [--samplerate 44100]

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;

// Touch the game-music-emu crate so cargo links the libgme static lib it
// builds. Without at least one symbol reference the linker would drop the
// archive as unused and our raw `gme_*` extern decls below wouldn't resolve.
#[allow(unused_imports)]
use game_music_emu::EmuType as _ForceLink;

const STEREO: u16 = 2;

// Opaque libgme types; we only ever pass these as raw pointers.
#[repr(C)]
struct MusicEmu {
    _opaque: [u8; 0],
}
#[repr(C)]
struct GmeType {
    _opaque: [u8; 0],
}

// gme_type_t is a pointer to a struct whose first two fields are
//   const char* system;
//   int track_count;
// We only deref `system` for diagnostic logging.
#[repr(C)]
struct GmeTypeStruct {
    system: *const c_char,
    track_count: c_int,
    // ... more fields we don't touch
}

unsafe extern "C" {
    fn gme_open_data(
        data: *const u8,
        size: c_int,
        out: *mut *const MusicEmu,
        sample_rate: c_int,
    ) -> *const c_char;
    fn gme_new_emu(gme_type: *const GmeTypeStruct, sample_rate: c_int) -> *const MusicEmu;
    fn gme_load_data(emu: *const MusicEmu, data: *const u8, size: usize) -> *const c_char;
    fn gme_start_track(emu: *const MusicEmu, index: c_int) -> *const c_char;
    fn gme_play(emu: *const MusicEmu, count: c_int, out: *mut i16) -> *const c_char;
    fn gme_track_count(emu: *const MusicEmu) -> c_int;
    fn gme_track_ended(emu: *const MusicEmu) -> bool;
    fn gme_delete(emu: *const MusicEmu);
    fn gme_type_list() -> *const *const GmeTypeStruct;
    // Direct pointer to the NSF type registration, exported by Nsf_Emu.cpp
    // when USE_GME_NSF is defined. Lets us bypass the identify_extension
    // path that fails under MSVC.
    static gme_nsf_type: *const GmeTypeStruct;
}

fn dump_type_list() {
    eprint!("render_nsf: compiled-in types:");
    let mut p = unsafe { gme_type_list() };
    let mut n = 0;
    while !p.is_null() && unsafe { !(*p).is_null() } {
        let entry = unsafe { *p };
        let sys = unsafe { CStr::from_ptr((*entry).system) }.to_string_lossy();
        eprint!(" [{sys}]");
        p = unsafe { p.add(1) };
        n += 1;
        if n > 32 {
            eprint!(" ...(truncated)");
            break;
        }
    }
    eprintln!(" (count={n})");
}

unsafe fn check(err: *const c_char, ctx: &str) -> Result<(), String> {
    if err.is_null() {
        return Ok(());
    }
    let msg = unsafe { CStr::from_ptr(err) }
        .to_string_lossy()
        .into_owned();
    if msg.is_empty() {
        return Ok(());
    }
    Err(format!("{ctx}: {msg}"))
}

struct NsfEmu {
    raw: *const MusicEmu,
}

impl NsfEmu {
    fn from_data(data: &[u8], sample_rate: u32) -> Result<Self, String> {
        if data.len() < 4 {
            return Err("file too small to be an NSF".into());
        }
        // Try gme_open_data first (auto-detect). On Windows MSVC this can
        // hit a strcmp-mismatch in gme_identify_extension despite the type
        // being compiled in. Fall back to forcing the NSF type directly via
        // gme_nsf_type, which is exported by Nsf_Emu.cpp when USE_GME_NSF
        // is defined.
        let mut raw: *const MusicEmu = std::ptr::null();
        let err = unsafe {
            gme_open_data(
                data.as_ptr(),
                data.len() as c_int,
                &mut raw as *mut *const MusicEmu,
                sample_rate as c_int,
            )
        };
        if let Ok(()) = unsafe { check(err, "gme_open_data") } {
            if !raw.is_null() {
                return Ok(Self { raw });
            }
        }
        eprintln!(
            "render_nsf: gme_open_data path failed; forcing NSF type via gme_nsf_type"
        );
        let nsf_type = unsafe { gme_nsf_type };
        if nsf_type.is_null() {
            return Err("gme_nsf_type symbol is null (NSF feature not built?)".into());
        }
        let raw = unsafe { gme_new_emu(nsf_type, sample_rate as c_int) };
        if raw.is_null() {
            return Err("gme_new_emu(NSF) returned null".into());
        }
        let load_err = unsafe { gme_load_data(raw, data.as_ptr(), data.len()) };
        if let Err(e) = unsafe { check(load_err, "gme_load_data") } {
            unsafe { gme_delete(raw) };
            return Err(e);
        }
        Ok(Self { raw })
    }

    fn track_count(&self) -> i32 {
        unsafe { gme_track_count(self.raw) }
    }

    fn start_track(&self, index: i32) -> Result<(), String> {
        unsafe { check(gme_start_track(self.raw, index as c_int), "gme_start_track") }
    }

    fn play(&self, samples: &mut [i16]) -> Result<(), String> {
        unsafe {
            check(
                gme_play(self.raw, samples.len() as c_int, samples.as_mut_ptr()),
                "gme_play",
            )
        }
    }

    fn track_ended(&self) -> bool {
        unsafe { gme_track_ended(self.raw) }
    }
}

impl Drop for NsfEmu {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { gme_delete(self.raw) };
        }
    }
}

struct Args {
    input: PathBuf,
    output: PathBuf,
    track: u32,
    duration_sec: f32,
    samplerate: u32,
}

fn parse_args() -> Result<Args, String> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut track: u32 = 0;
    let mut duration_sec: f32 = 60.0;
    let mut samplerate: u32 = 44_100;

    let mut iter = std::env::args().skip(1);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--in" => input = Some(PathBuf::from(iter.next().ok_or("--in needs a value")?)),
            "--out" => output = Some(PathBuf::from(iter.next().ok_or("--out needs a value")?)),
            "--track" => {
                track = iter
                    .next()
                    .ok_or("--track needs a value")?
                    .parse()
                    .map_err(|e| format!("--track: {e}"))?
            }
            "--duration" => {
                duration_sec = iter
                    .next()
                    .ok_or("--duration needs a value")?
                    .parse()
                    .map_err(|e| format!("--duration: {e}"))?
            }
            "--samplerate" => {
                samplerate = iter
                    .next()
                    .ok_or("--samplerate needs a value")?
                    .parse()
                    .map_err(|e| format!("--samplerate: {e}"))?
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }

    Ok(Args {
        input: input.ok_or("--in is required")?,
        output: output.ok_or("--out is required")?,
        track,
        duration_sec,
        samplerate,
    })
}

fn main() {
    if let Err(e) = run() {
        eprintln!("render_nsf: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    eprintln!(
        "render_nsf: input={:?} track={} duration={:.1}s sr={}Hz",
        args.input, args.track, args.duration_sec, args.samplerate
    );

    dump_type_list();
    let data = std::fs::read(&args.input).map_err(|e| format!("read input: {e}"))?;
    eprintln!(
        "render_nsf: file={} bytes, header={:02x}{:02x}{:02x}{:02x}",
        data.len(), data[0], data[1], data[2], data[3]
    );
    let emu = NsfEmu::from_data(&data, args.samplerate)?;

    let total_tracks = emu.track_count();
    eprintln!("render_nsf: track_count={total_tracks}");
    if (args.track as i32) >= total_tracks {
        return Err(format!(
            "--track {} out of range (file has {} tracks, 0-indexed)",
            args.track,
            total_tracks
        ));
    }

    emu.start_track(args.track as i32)?;

    let total_frames = (args.duration_sec * args.samplerate as f32) as usize;
    let total_samples = total_frames * STEREO as usize;

    // Pull in chunks. libgme is fine with sub-second blocks; it gives the
    // emulator a chance to update channel state without giant single calls
    // (the safer-wrapper crate also chunks internally).
    const CHUNK_FRAMES: usize = 4096;
    let chunk_samples = CHUNK_FRAMES * STEREO as usize;
    let mut buf: Vec<i16> = vec![0; total_samples];
    let mut filled = 0;
    while filled < total_samples {
        let take = chunk_samples.min(total_samples - filled);
        emu.play(&mut buf[filled..filled + take])?;
        filled += take;
        if emu.track_ended() {
            buf.truncate(filled);
            break;
        }
    }

    // 16-bit PCM stereo wav: keeps libgme samples bit-exact, file stays small.
    let spec = hound::WavSpec {
        channels: STEREO,
        sample_rate: args.samplerate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(&args.output, spec)
        .map_err(|e| format!("wav create: {e}"))?;
    for s in &buf {
        writer
            .write_sample(*s)
            .map_err(|e| format!("wav write: {e}"))?;
    }
    writer.finalize().map_err(|e| format!("wav finalize: {e}"))?;

    let peak = buf.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
    eprintln!(
        "render_nsf: wrote {:?} ({} frames, peak={})",
        args.output,
        buf.len() / STEREO as usize,
        peak,
    );
    Ok(())
}
