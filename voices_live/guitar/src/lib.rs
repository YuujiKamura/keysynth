//! `voices_live` plugin: steel-string acoustic guitar voice as a
//! hot-loadable cdylib (issue #44 — first non-piano voice family).
//!
//! Unlike the piano plugins under `voices_live/piano*` — which select
//! an `Engine` variant and route construction through
//! `keysynth::synth::make_voice` — this plugin builds a
//! `keysynth::voices::guitar::GuitarVoice` directly and hands the
//! boxed trait object to `keysynth::live_abi::live_new_boxed`. This
//! is the deliberate choice for non-piano families called out in the
//! brief: a new instrument family must NOT add an `Engine` variant.
//!
//! Audio output is constructed by the same `GuitarVoice::new`
//! invocation that `tests/guitar_e2e.rs` exercises in-process, so the
//! plugin path and the offline test path produce byte-identical
//! samples.

use std::os::raw::c_void;

use keysynth::live_abi as ka;
use keysynth::voices::guitar::GuitarVoice;

#[no_mangle]
pub extern "C" fn keysynth_live_abi_version() -> u32 {
    ka::KEYSYNTH_LIVE_ABI_VERSION
}

/// # Safety
/// Called by the host loader (`src/live_reload.rs`) after the ABI
/// version check. Returned pointer must be freed via
/// `keysynth_live_drop`.
#[no_mangle]
pub unsafe extern "C" fn keysynth_live_new(sr: f32, freq: f32, vel: u8) -> *mut c_void {
    let voice: Box<dyn keysynth::synth::VoiceImpl + Send> =
        Box::new(GuitarVoice::new(sr, freq, vel));
    unsafe { ka::live_new_boxed(voice) }
}

/// # Safety
/// `p` must come from `keysynth_live_new`; `buf` must point to `n`
/// writable `f32`s. Output is additive — never overwrites the buffer.
#[no_mangle]
pub unsafe extern "C" fn keysynth_live_render_add(p: *mut c_void, buf: *mut f32, n: usize) {
    unsafe { ka::live_render_add(p, buf, n) }
}

/// # Safety
/// `p` must come from `keysynth_live_new`.
#[no_mangle]
pub unsafe extern "C" fn keysynth_live_trigger_release(p: *mut c_void) {
    unsafe { ka::live_trigger_release(p) }
}

/// # Safety
/// `p` must come from `keysynth_live_new`.
#[no_mangle]
pub unsafe extern "C" fn keysynth_live_is_done(p: *mut c_void) -> bool {
    unsafe { ka::live_is_done(p) }
}

/// # Safety
/// `p` must come from `keysynth_live_new`.
#[no_mangle]
pub unsafe extern "C" fn keysynth_live_is_releasing(p: *mut c_void) -> bool {
    unsafe { ka::live_is_releasing(p) }
}

/// # Safety
/// `p` must come from `keysynth_live_new`. Must not be called twice.
#[no_mangle]
pub unsafe extern "C" fn keysynth_live_drop(p: *mut c_void) {
    unsafe { ka::live_drop(p) }
}
