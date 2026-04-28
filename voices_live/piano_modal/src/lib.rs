//! `voices_live` plugin: the `Engine::PianoModal` voice as a
//! hot-loadable cdylib.
//!
//! Constructs voices via `keysynth::synth::make_voice(Engine::PianoModal,
//! ...)`. The implementation (32-partial LUT projection, per-mode
//! biquad bandpass bank, hammer-impulse excitation) lives in
//! `keysynth::voices::piano_modal` and is not duplicated here.
//!
//! Per-process LUT bootstrap: each plugin cdylib statically links
//! its own copy of the keysynth crate, so the host's `MODAL_LUT`
//! `OnceLock` and the plugin's `MODAL_LUT` are different statics.
//! `keysynth_live_new` calls `ka::ensure_modal_lut_loaded()` once
//! to populate the plugin's LUT via the same `ModalLut::auto_load`
//! path the host's main.rs uses, so factory equivalence with the
//! enum-dispatch render path is preserved without the host having
//! to reach across the cdylib boundary.

use std::os::raw::c_void;

use keysynth::live_abi as ka;
use keysynth::synth::Engine;

const ENGINE: Engine = Engine::PianoModal;

#[no_mangle]
pub extern "C" fn keysynth_live_abi_version() -> u32 {
    ka::KEYSYNTH_LIVE_ABI_VERSION
}

/// # Safety
/// Called by the host loader after the ABI version check. Returned
/// pointer must be freed via `keysynth_live_drop`.
#[no_mangle]
pub unsafe extern "C" fn keysynth_live_new(sr: f32, freq: f32, vel: u8) -> *mut c_void {
    ka::ensure_modal_lut_loaded();
    unsafe { ka::live_new(ENGINE, sr, freq, vel) }
}

/// # Safety
/// `p` must come from `keysynth_live_new`; `buf` must point to `n`
/// writable `f32`s. Output is additive.
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
