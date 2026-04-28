//! Voice plugin / enum-dispatch equivalence test.
//!
//! For each Tier 1 piano engine (`Piano`, `PianoModal`, `PianoThick`,
//! `PianoLite`, `Piano5AM`) this test renders a fixed input through:
//!
//!   - the in-process enum-dispatch path: `keysynth::synth::make_voice(Engine::X, ...)`
//!   - the cdylib plugin path: `voices_live/<name>/` built + loaded via
//!     `keysynth::live_reload::build_and_load(...)` and then driven through
//!     the resulting `LiveFactory::make_voice(...)`.
//!
//! Pass condition (per the user's hard rule):
//!   byte-identical sha256 OR `mr_stft_l1 < 0.001`.
//!
//! The OR exists because each plugin cdylib statically links its own copy
//! of `keysynth`, so LLVM may legitimately reorder fp ops at the cdylib
//! boundary even when both sides ultimately call the same DSP function.
//! Spectral-distance equivalence under that constraint is the strongest
//! claim we can make from outside the cdylib.

#![cfg(feature = "native")]

use std::path::{Path, PathBuf};

use keysynth::analysis::mr_stft_l1;
use keysynth::live_reload::build_and_load;
use keysynth::synth::{make_voice, Engine, VoiceImpl};
use keysynth::voices::piano_modal::{ModalLut, MODAL_LUT};

const SR: f32 = 44_100.0;
const C4_HZ: f32 = 261.6256; // MIDI 60
const VEL: u8 = 100;
const N_FRAMES: usize = (SR as usize) * 2; // 2 seconds
const RELEASE_AT: usize = SR as usize; // release at 1.0 s
const BLOCK: usize = 256;

/// Render a single voice for `N_FRAMES` samples, triggering release at
/// `RELEASE_AT`. The voice is driven block-by-block to match how the
/// real audio callback exercises it. Buffers are zeroed before each
/// block so `render_add` doesn't accumulate across blocks (which would
/// alias state from the previous block into the next).
fn render_voice(mut voice: Box<dyn VoiceImpl + Send>) -> Vec<f32> {
    let mut out = vec![0.0f32; N_FRAMES];
    let mut released = false;
    let mut i = 0;
    while i < N_FRAMES {
        if !released && i >= RELEASE_AT {
            voice.trigger_release();
            released = true;
        }
        let end = (i + BLOCK).min(N_FRAMES);
        // The trait's render_add is additive into the caller's buffer,
        // so passing a slice over `out` writes straight into the right
        // place — no per-block scratch allocation.
        voice.render_add(&mut out[i..end]);
        i = end;
    }
    out
}

/// Plugin crate root resolution. Tests run with cwd = workspace root.
fn plugin_root(name: &str) -> PathBuf {
    std::env::current_dir()
        .expect("cwd")
        .join("voices_live")
        .join(name)
}

/// True if every plugin's Cargo.toml is on disk. Release tarballs may
/// ship without `voices_live/`; in that case we skip the whole test.
fn all_plugin_manifests_present() -> bool {
    for name in [
        "piano",
        "piano_modal",
        "piano_thick",
        "piano_lite",
        "piano_5am",
    ] {
        if !plugin_root(name).join("Cargo.toml").exists() {
            return false;
        }
    }
    true
}

/// Initialise the host-side MODAL_LUT exactly like `main.rs` does at
/// startup. The plugin cdylib carries its own static copy of MODAL_LUT
/// so it must self-bootstrap; if it doesn't, the plugin path will use
/// `ModalLut::fallback_c4()` and diverge from the host — which is the
/// designed failure mode and a correct test fail.
fn ensure_host_modal_lut() {
    if MODAL_LUT.get().is_some() {
        return;
    }
    let (lut, _msg) = ModalLut::auto_load(None);
    let _ = MODAL_LUT.set(lut);
}

/// Render via the enum-dispatch path.
fn render_via_enum(engine: Engine) -> Vec<f32> {
    if matches!(engine, Engine::PianoModal) {
        ensure_host_modal_lut();
    }
    let voice = make_voice(engine, SR, C4_HZ, VEL);
    render_voice(voice)
}

/// Render via the plugin path. Builds the cdylib at `crate_root`, loads
/// it with `LiveFactory::load`, constructs one voice, drives it.
fn render_via_plugin(crate_root: &Path) -> Result<Vec<f32>, String> {
    let factory = build_and_load(crate_root)?;
    let factory = std::sync::Arc::new(factory);
    let voice = factory.make_voice(SR, C4_HZ, VEL);
    Ok(render_voice(voice))
}

/// Minimal sha256 (FIPS 180-4). Embedded here because the keysynth crate
/// has no sha2 dep and the equivalence test only needs a 64-char hex
/// digest for human-readable reporting — the actual byte-equality check
/// uses `==` on the underlying buffer.
fn sha256_hex(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    // Pre-processing: append 0x80, pad with zeros, append 64-bit big-endian length.
    let mut msg = Vec::with_capacity(bytes.len() + 72);
    msg.extend_from_slice(bytes);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, c) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let mj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(mj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = String::with_capacity(64);
    for word in h.iter() {
        for byte in word.to_be_bytes().iter() {
            out.push_str(&format!("{:02x}", byte));
        }
    }
    out
}

fn buf_bytes(buf: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(buf.len() * 4);
    for s in buf {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    bytes
}

fn peak(buf: &[f32]) -> f32 {
    buf.iter().fold(0.0f32, |a, &x| a.max(x.abs()))
}

fn max_delta(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut m = 0.0f32;
    for i in 0..n {
        let d = (a[i] - b[i]).abs();
        if d > m {
            m = d;
        }
    }
    m
}

/// Run the equivalence check for one engine. Returns Ok(()) on PASS,
/// Err(diagnosis-string) on FAIL.
fn check_engine(engine: Engine, plugin_dir: &str) -> Result<(), String> {
    let label = format!("{:?}", engine);
    let crate_root = plugin_root(plugin_dir);

    // 1. Render via enum-dispatch.
    let host = render_via_enum(engine);
    let host_peak = peak(&host);
    if host_peak == 0.0 {
        return Err(format!(
            "[{label}] host (enum) render produced silence — engine itself is broken or LUT missing"
        ));
    }

    // 2. Render via plugin.
    let plug = render_via_plugin(&crate_root)
        .map_err(|e| format!("[{label}] plugin build_and_load failed: {e}"))?;
    let plug_peak = peak(&plug);
    if plug_peak == 0.0 {
        return Err(format!("[{label}] plugin render produced silence"));
    }

    // 3. Equivalence check: byte-identical OR mr_stft_l1 < 0.001.
    let host_bytes = buf_bytes(&host);
    let plug_bytes = buf_bytes(&plug);
    let host_sha = sha256_hex(&host_bytes);
    let plug_sha = sha256_hex(&plug_bytes);

    if host_bytes == plug_bytes {
        eprintln!(
            "[{label}] byte-equal sha256={host_sha} peak_host={host_peak:.4} peak_plug={plug_peak:.4}"
        );
        return Ok(());
    }

    // Not byte-equal — fall back to spectral metric.
    let mr = mr_stft_l1(&host, &plug, SR as u32);
    let md = max_delta(&host, &plug);
    eprintln!(
        "[{label}] sha256(host)={host_sha} sha256(plug)={plug_sha} mr_stft={mr:.6} max_delta={md:.6} peak_host={host_peak:.4} peak_plug={plug_peak:.4}"
    );

    if mr < 0.001 {
        Ok(())
    } else {
        Err(format!(
            "[{label}] diverged: mr_stft={mr:.6} max_delta={md:.6}; suggests plugin bypasses make_voice or uses a different DSP path"
        ))
    }
}

/// One test, all five engines. We do them all in a single `#[test]` so
/// the eprintln output is grouped and easy to paste into a PR.
#[test]
fn voice_factory_equivalence_all_tier1() {
    if !all_plugin_manifests_present() {
        eprintln!("skipping voice_factory_equivalence: voices_live/<name>/ not present (release tarball case)");
        return;
    }

    eprintln!("=== voice_factory_equivalence: enum-dispatch vs cdylib plugin ===");
    eprintln!(
        "SR={SR} hz={C4_HZ} vel={VEL} n_frames={N_FRAMES} release_at={RELEASE_AT} block={BLOCK}"
    );

    let cases: &[(Engine, &str)] = &[
        (Engine::Piano, "piano"),
        (Engine::PianoModal, "piano_modal"),
        (Engine::PianoThick, "piano_thick"),
        (Engine::PianoLite, "piano_lite"),
        (Engine::Piano5AM, "piano_5am"),
    ];

    let mut failures: Vec<String> = Vec::new();
    for (engine, dir) in cases {
        match check_engine(*engine, dir) {
            Ok(()) => {}
            Err(msg) => {
                eprintln!("FAIL: {msg}");
                failures.push(msg);
            }
        }
    }

    if failures.is_empty() {
        eprintln!("=== VERIFY PASS: all 5 Tier 1 plugins equivalent to enum-dispatch ===");
    } else {
        eprintln!("=== VERIFY FAIL: {} engine(s) diverged ===", failures.len());
        for f in &failures {
            eprintln!("  - {f}");
        }
        panic!(
            "voice_factory_equivalence: {} divergence(s)",
            failures.len()
        );
    }
}
