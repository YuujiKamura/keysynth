//! `voices_live` plugin: the `Engine::PianoThick` voice as a
//! hot-loadable cdylib. Forwards to `keysynth::synth::make_voice`
//! so audio output matches the in-process enum-dispatch render
//! byte-for-byte.

use std::os::raw::c_void;

use keysynth::live_abi as ka;
use keysynth::synth::Engine;

const ENGINE: Engine = Engine::PianoThick;

#[no_mangle]
pub extern "C" fn keysynth_live_abi_version() -> u32 {
    ka::KEYSYNTH_LIVE_ABI_VERSION
}

/// # Safety
/// Host-loader call site only.
#[no_mangle]
pub unsafe extern "C" fn keysynth_live_new(sr: f32, freq: f32, vel: u8) -> *mut c_void {
    unsafe { ka::live_new(ENGINE, sr, freq, vel) }
}

/// # Safety
/// `p` from `keysynth_live_new`; `buf` is `n` writable f32s.
#[no_mangle]
pub unsafe extern "C" fn keysynth_live_render_add(p: *mut c_void, buf: *mut f32, n: usize) {
    unsafe { ka::live_render_add(p, buf, n) }
}

/// # Safety
/// `p` from `keysynth_live_new`.
#[no_mangle]
pub unsafe extern "C" fn keysynth_live_trigger_release(p: *mut c_void) {
    unsafe { ka::live_trigger_release(p) }
}

/// # Safety
/// `p` from `keysynth_live_new`.
#[no_mangle]
pub unsafe extern "C" fn keysynth_live_is_done(p: *mut c_void) -> bool {
    unsafe { ka::live_is_done(p) }
}

/// # Safety
/// `p` from `keysynth_live_new`.
#[no_mangle]
pub unsafe extern "C" fn keysynth_live_is_releasing(p: *mut c_void) -> bool {
    unsafe { ka::live_is_releasing(p) }
}

/// # Safety
/// `p` from `keysynth_live_new`. Must not be called twice.
#[no_mangle]
pub unsafe extern "C" fn keysynth_live_drop(p: *mut c_void) {
    unsafe { ka::live_drop(p) }
}
