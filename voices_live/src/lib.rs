//! Hot-reloadable voice for keysynth.
//!
//! Edit this file, save, and the running keysynth GUI rebuilds + reloads
//! the produced .dll/.so/.dylib while audio keeps streaming. Held notes
//! finish with the OLD code (their library is pinned via Arc until the
//! voice drops); the next note_on routes through the NEW code.
//!
//! The host expects the C-ABI exports at the bottom of this file. The
//! `Voice` struct above is yours to mutate freely — anything that builds
//! to a stable `extern "C"` shape will work. Don't change the C-ABI
//! function signatures or the `KEYSYNTH_LIVE_ABI_VERSION` constant
//! without bumping the host.

use std::os::raw::c_void;

/// Bumped on any incompatible C-ABI change. Host rejects libs whose
/// version doesn't match its expectation. Live edits to the DSP body
/// don't need to touch this.
pub const KEYSYNTH_LIVE_ABI_VERSION: u32 = 1;

/// The voice DSP. Edit freely — this is the whole live-edit surface.
///
/// Default starter: a sine with exponential decay. Replace with FM,
/// karplus-strong, modal sum, whatever you want. Required behaviours:
///   - `render_add` MUST be additive (`buf[i] += sample`), never
///     overwrite. The host sums many voices into one bus.
///   - `is_done` should eventually return true so the host can evict
///     the voice from its pool. Otherwise pool fills up with ghosts.
///
/// Sample rate, fundamental frequency (Hz), and MIDI velocity (0..=127)
/// are passed to `new`. Save them on the voice; the host won't.
pub struct Voice {
    sr: f32,
    /// Phase accumulator in radians.
    phase: f32,
    /// Per-sample phase increment.
    dphase: f32,
    /// Initial amplitude from velocity.
    amp: f32,
    /// Multiplicative decay per sample (exp envelope).
    decay: f32,
    /// Set true on note_off; speeds up the decay.
    released: bool,
    /// Running envelope multiplier (starts at 1.0).
    env: f32,
}

impl Voice {
    pub fn new(sr: f32, freq: f32, velocity: u8) -> Self {
        let amp = (velocity as f32 / 127.0).clamp(0.0, 1.0) * 0.6;
        // ~3-second natural decay; release multiplies it by 5 so note-off
        // dies in ~0.6 s.
        let t60 = 3.0;
        let decay = (-(6.908) / (t60 * sr)).exp(); // 6.908 = ln(1000)
        Self {
            sr,
            phase: 0.0,
            dphase: std::f32::consts::TAU * freq / sr,
            amp,
            decay,
            released: false,
            env: 1.0,
        }
    }

    pub fn render_add(&mut self, buf: &mut [f32]) {
        let release_mul: f32 = if self.released { 5.0 } else { 1.0 };
        let _ = self.sr; // silence unused warning if user simplifies later
        for s in buf.iter_mut() {
            let sample = self.amp * self.env * self.phase.sin();
            *s += sample;
            self.phase += self.dphase;
            if self.phase > std::f32::consts::TAU {
                self.phase -= std::f32::consts::TAU;
            }
            self.env *= self.decay.powf(release_mul);
        }
    }

    pub fn trigger_release(&mut self) {
        self.released = true;
    }

    pub fn is_done(&self) -> bool {
        self.env < 1.0e-4
    }

    pub fn is_releasing(&self) -> bool {
        self.released
    }
}

// ---------------------------------------------------------------------------
// C ABI surface — DON'T edit signatures without bumping ABI version + host.
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn keysynth_live_abi_version() -> u32 {
    KEYSYNTH_LIVE_ABI_VERSION
}

/// Construct a Voice. Returned pointer must be freed via `keysynth_live_drop`.
#[no_mangle]
pub unsafe extern "C" fn keysynth_live_new(sr: f32, freq: f32, vel: u8) -> *mut c_void {
    let v = Box::new(Voice::new(sr, freq, vel));
    Box::into_raw(v) as *mut c_void
}

/// Add this voice's contribution into `buf[..n]` (mono).
#[no_mangle]
pub unsafe extern "C" fn keysynth_live_render_add(p: *mut c_void, buf: *mut f32, n: usize) {
    if p.is_null() || buf.is_null() || n == 0 {
        return;
    }
    let v = &mut *(p as *mut Voice);
    let slice = std::slice::from_raw_parts_mut(buf, n);
    v.render_add(slice);
}

#[no_mangle]
pub unsafe extern "C" fn keysynth_live_trigger_release(p: *mut c_void) {
    if p.is_null() {
        return;
    }
    (*(p as *mut Voice)).trigger_release();
}

#[no_mangle]
pub unsafe extern "C" fn keysynth_live_is_done(p: *mut c_void) -> bool {
    if p.is_null() {
        return true;
    }
    (*(p as *mut Voice)).is_done()
}

#[no_mangle]
pub unsafe extern "C" fn keysynth_live_is_releasing(p: *mut c_void) -> bool {
    if p.is_null() {
        return false;
    }
    (*(p as *mut Voice)).is_releasing()
}

#[no_mangle]
pub unsafe extern "C" fn keysynth_live_drop(p: *mut c_void) {
    if p.is_null() {
        return;
    }
    drop(Box::from_raw(p as *mut Voice));
}
