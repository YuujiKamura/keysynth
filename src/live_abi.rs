//! C ABI helper functions used by the `voices_live/<name>/` cdylib
//! plugins to expose Tier 1 voices (and any other `Engine` variant)
//! through the same `keysynth_live_*` symbols that PR #40 introduced.
//!
//! Each plugin's `src/lib.rs` is a ~30-line shim: it picks an
//! `Engine` variant and forwards every C-ABI export through the
//! helpers below. The actual voice construction goes through
//! `crate::synth::make_voice`, which is the same factory the GUI's
//! enum-dispatch path uses. That guarantees the byte / mr_stft
//! equivalence the user demanded — enum dispatch and plugin
//! dispatch both build voices via `make_voice`, so they cannot
//! diverge unless `make_voice` itself is non-deterministic.
//!
//! Lives in the keysynth library (not in voices_live) so plugin
//! crates only need a single `keysynth = { path = "../..",
//! default-features = false }` dep — no extra shim crates and no
//! per-plugin code duplication.
//!
//! ABI version is mirrored from `voices_live/src/lib.rs` so a
//! plugin built against this helper interoperates with the
//! existing host loader without bumping `EXPECTED_ABI_VERSION` in
//! `live_reload.rs`.

use std::os::raw::c_void;
use std::sync::OnceLock;

use crate::synth::{make_voice, Engine, VoiceImpl};
use crate::voices::piano_modal::{ModalLut, MODAL_LUT};

/// Bumped on any incompatible C-ABI change. Must equal
/// `live_reload::EXPECTED_ABI_VERSION` and the value already
/// returned by `voices_live/src/lib.rs::keysynth_live_abi_version`.
pub const KEYSYNTH_LIVE_ABI_VERSION: u32 = 1;

/// Opaque box around a `Box<dyn VoiceImpl + Send>`. The plugin only
/// hands the host a `*mut c_void` pointing at this struct; the host
/// loader (`src/live_reload.rs`) never inspects the contents.
struct PluginVoice {
    inner: Box<dyn VoiceImpl + Send>,
}

/// Lazy initialiser for the modal-piano LUT inside a plugin process.
///
/// The host's main.rs sets `MODAL_LUT` before loading any plugin, but
/// each plugin cdylib statically links its OWN copy of the keysynth
/// crate (and therefore its own `MODAL_LUT` static). Calling this from
/// the piano_modal plugin's `keysynth_live_new` ensures its private
/// LUT static gets the same auto-discovered JSON as the host's, so
/// `make_voice(Engine::PianoModal, ...)` produces byte-identical
/// audio across the enum-dispatch and plugin-dispatch paths.
///
/// Idempotent — `OnceLock::set` returns `Err` if already set, which
/// we ignore. Other plugins (Engine::Piano, PianoThick, PianoLite,
/// Piano5AM) don't touch the LUT and don't need to call this.
pub fn ensure_modal_lut_loaded() {
    static GUARD: OnceLock<()> = OnceLock::new();
    let _ = GUARD.get_or_init(|| {
        if MODAL_LUT.get().is_some() {
            return;
        }
        let (lut, _source) = ModalLut::auto_load(None);
        let _ = MODAL_LUT.set(lut);
    });
}

/// Construct a voice for `engine`. Returned pointer must be freed via
/// `live_drop`. Callers (the plugin's `keysynth_live_new`) wrap this
/// directly with `#[no_mangle] extern "C"`.
///
/// # Safety
/// Must only be called from the plugin's exported `keysynth_live_new`
/// symbol, which is itself called by the host loader after a successful
/// ABI-version check.
pub unsafe fn live_new(engine: Engine, sr: f32, freq: f32, vel: u8) -> *mut c_void {
    let v = Box::new(PluginVoice {
        inner: make_voice(engine, sr, freq, vel),
    });
    Box::into_raw(v) as *mut c_void
}

/// Add the voice's contribution into `buf[..n]` (mono, additive — never
/// overwrites the buffer, matching the host-side contract).
///
/// # Safety
/// `p` must come from a previous `live_new` call against the same
/// plugin and not have been passed to `live_drop`. `buf` must point
/// to `n` writable `f32`s.
pub unsafe fn live_render_add(p: *mut c_void, buf: *mut f32, n: usize) {
    if p.is_null() || buf.is_null() || n == 0 {
        return;
    }
    let v = &mut *(p as *mut PluginVoice);
    let slice = std::slice::from_raw_parts_mut(buf, n);
    v.inner.render_add(slice);
}

/// Mark the voice as released (note_off). Eventually the voice's
/// envelope will fall below the threshold `live_is_done` checks against
/// and the host will evict it.
///
/// # Safety
/// See `live_render_add`.
pub unsafe fn live_trigger_release(p: *mut c_void) {
    if p.is_null() {
        return;
    }
    (*(p as *mut PluginVoice)).inner.trigger_release();
}

/// Has the voice's release-envelope decayed to inaudible? Once true,
/// the host pool reclaims the slot and calls `live_drop`.
///
/// # Safety
/// See `live_render_add`.
pub unsafe fn live_is_done(p: *mut c_void) -> bool {
    if p.is_null() {
        return true;
    }
    (*(p as *mut PluginVoice)).inner.is_done()
}

/// Is the voice currently in its release tail? Used by the pool's
/// "evict the longest-released" eviction policy. Distinct from
/// `live_is_done` — a released voice can be `is_releasing == true`
/// for many seconds before `is_done` flips.
///
/// # Safety
/// See `live_render_add`.
pub unsafe fn live_is_releasing(p: *mut c_void) -> bool {
    if p.is_null() {
        return false;
    }
    (*(p as *mut PluginVoice)).inner.is_releasing()
}

/// Drop the voice and free its allocation.
///
/// # Safety
/// `p` must come from a previous `live_new` call. Must not be called
/// twice for the same pointer.
pub unsafe fn live_drop(p: *mut c_void) {
    if p.is_null() {
        return;
    }
    drop(Box::from_raw(p as *mut PluginVoice));
}
