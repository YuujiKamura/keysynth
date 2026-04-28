//! `voices_live` plugin: the `Engine::Piano` voice as a hot-loadable
//! cdylib.
//!
//! Constructs voices via `keysynth::synth::make_voice(Engine::Piano,
//! ...)` — the same factory the GUI's enum-dispatch path uses — so
//! audio output is byte-identical to the in-process `Engine::Piano`
//! render. Implementation logic (Stulov hammer, Fletcher
//! inharmonicity, KS+modal hybrid) lives in keysynth's voice
//! modules and is not duplicated here.
//!
//! Tier 1 voices (this one + piano_modal / piano_thick / piano_lite /
//! piano_5am) used to be reachable only through the in-process
//! `Engine` enum. After this PR they're swappable from outside via
//! `ksctl build --slot piano --src voices_live/piano` without
//! restarting the GUI.

use std::os::raw::c_void;

use keysynth::live_abi as ka;
use keysynth::synth::Engine;

const ENGINE: Engine = Engine::Piano;

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
    unsafe { ka::live_new(ENGINE, sr, freq, vel) }
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
