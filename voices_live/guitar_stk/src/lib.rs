//! `voices_live` plugin: Synthesis Toolkit (STK) Guitar/Twang port —
//! second steel-string acoustic guitar voice (issue #52, sibling of
//! `voices_live/guitar/`'s PR #51 voice). Provides an A/B reference
//! against the from-scratch implementation so we have a permanent
//! regression oracle.
//!
//! Adapted from The Synthesis Toolkit (STK) by Perry R. Cook and
//! Gary P. Scavone (1995–2023). Original license: MIT-equivalent
//! permissive (see `reference/stk-1.x/LICENSE`).
//!
//! Like the sibling `voices_live/guitar/` plugin — and unlike the
//! piano plugins under `voices_live/piano*` which select an `Engine`
//! variant — this plugin builds a
//! `keysynth::voices::guitar_stk::GuitarStkVoice` directly and hands
//! the boxed trait object to `keysynth::live_abi::live_new_boxed`.
//! Non-piano families do NOT add an `Engine` variant.

use std::os::raw::c_void;

use keysynth::live_abi as ka;
use keysynth::voices::guitar_stk::GuitarStkVoice;

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
        Box::new(GuitarStkVoice::new(sr, freq, vel));
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
