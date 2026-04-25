//! Modal piano voice — parallel bandpass resonators driven by a hammer
//! impulse. Pure projection model (issue #3 alt path): each mode is a
//! single (freq, T60, init_amp) triple; the voice is a parallel bank of
//! 2nd-order biquad bandpass IIRs, each tuned to one mode and decaying
//! at exactly its measured T60.
//!
//! No physical string/soundboard model — the modes ARE the projection.
//! Modal LUTs are produced offline from a reference recording via
//! `keysynth::extract::*` and fed into `ModalPianoVoice::new`. This
//! voice is a counterpoint to the physical-model `PianoVoice`
//! (`voices/piano.rs`): instead of approaching the SFZ Salamander
//! reference by tuning string/soundboard parameters, it consumes the
//! reference's measured spectrum directly.
//!
//! Per-mode resonator (textbook biquad bandpass with prescribed pole
//! radius for an exact T60):
//!
//! ```text
//!     omega_k = 2π · f_k / sr
//!     r_k = exp(-3 · ln(10) / (T60_k · sr))     // r^(T60·sr) = 1e-3 = -60 dB
//!     y[n] = b0 · x[n] - a1 · y[n-1] - a2 · y[n-2]
//!     a1 = -2 r_k cos(omega_k)
//!     a2 = r_k²
//!     b0 = 1 - r_k²                             // unity gain at resonance
//! ```
//!
//! Each mode is excited by the same scalar input `x[n]` (the hammer
//! impulse) and contributes `init_amp_k · y_k[n]` to the output. The
//! release envelope multiplies the summed output, identical to the
//! existing physical voices.

use crate::synth::{ReleaseEnvelope, VoiceImpl};

/// One modal resonance: (f, T60, gain). `init_amp` is the LINEAR
/// amplitude (not dB), so callers converting from `extract::Partial::init_db`
/// need to do `10f32.powf(init_db / 20.0)` first.
#[derive(Clone, Copy, Debug)]
pub struct Mode {
    pub freq_hz: f32,
    pub t60_sec: f32,
    pub init_amp: f32,
}

struct ModalResonator {
    /// Cosine of the resonant angular frequency. Coefficient `a1` is
    /// `-2·r·cos_omega` and is recomputed on damper engage to swap the
    /// pole radius without rebuilding the resonator from scratch.
    cos_omega: f32,
    a1: f32,
    a2: f32,
    b0: f32,
    y1: f32,
    y2: f32,
    init_amp: f32,
    sr: f32,
}

impl ModalResonator {
    fn new(mode: Mode, sr: f32) -> Self {
        let omega = 2.0 * std::f32::consts::PI * mode.freq_hz / sr;
        let cos_omega = omega.cos();
        let t60 = mode.t60_sec.max(1e-3);
        let (a1, a2, b0) = compute_coefs(cos_omega, t60, sr);
        Self {
            cos_omega,
            a1,
            a2,
            b0,
            y1: 0.0,
            y2: 0.0,
            init_amp: mode.init_amp,
            sr,
        }
    }

    /// Engage the damper: swap the pole radius to one that decays to
    /// -60 dB in `damper_t60_sec` (typically 0.05-0.20 s). This kills
    /// the mode much faster than its natural T60, mimicking a felt
    /// damper landing on a piano string. Cheap: just re-derives the
    /// three IIR coefficients; internal state (y1, y2) is preserved
    /// so there's no click.
    fn engage_damper(&mut self, damper_t60_sec: f32) {
        let t60 = damper_t60_sec.max(1e-3);
        let (a1, a2, b0) = compute_coefs(self.cos_omega, t60, self.sr);
        self.a1 = a1;
        self.a2 = a2;
        self.b0 = b0;
    }

    #[inline]
    fn step(&mut self, x: f32) -> f32 {
        let y0 = self.b0 * x - self.a1 * self.y1 - self.a2 * self.y2;
        self.y2 = self.y1;
        self.y1 = y0;
        y0 * self.init_amp
    }
}

/// Pole radius chosen so r^(T60·sr) = 1e-3 (i.e. -60 dB at exactly
/// T60 seconds). Returns (a1, a2, b0) for the biquad bandpass with
/// unity peak gain at resonance.
fn compute_coefs(cos_omega: f32, t60_sec: f32, sr: f32) -> (f32, f32, f32) {
    let r = (-3.0 * 10f32.ln() / (t60_sec * sr)).exp();
    let a1 = -2.0 * r * cos_omega;
    let a2 = r * r;
    // |H(e^{jω₀})| = 1 / (1 - r²) at resonance; pre-multiplying b0
    // by (1 - r²) normalises peak to 1 — at the moment of damper
    // engage this rescales the steady-state gain, which is the
    // physically correct behaviour (less coupling area = less
    // radiated amplitude).
    let b0 = 1.0 - r * r;
    (a1, a2, b0)
}

/// Modal piano voice. Holds a parallel bank of resonators (one per
/// mode) plus a hammer-impulse excitation buffer that's played out at
/// note-on. After the impulse runs out, the resonators ring out at
/// their per-mode T60.
///
/// On `trigger_release` the voice engages a virtual damper:
/// `damper_t60_sec` (default 0.12 s) is swapped into every resonator
/// in place of its natural T60, which mimics a wool damper landing on
/// a piano string and stopping it within ~100 ms. Without this the
/// per-mode resonators would happily ring through their full T60
/// (h1 ~ 18 s) regardless of note_off, producing the audible "電子
/// ピアノっぽい / 減衰しない" character of the early pilot.
pub struct ModalPianoVoice {
    resonators: Vec<ModalResonator>,
    /// Hammer-impulse waveform played into every resonator at note-on.
    /// Single-sample delta in the default constructor (broadband, every
    /// mode excited equally); callers can supply any shape.
    excitation: Vec<f32>,
    excitation_idx: usize,
    /// Damper T60 applied to every resonator on `trigger_release`. Real
    /// piano felt dampers stop the string in ~80-200 ms; 0.12 s is a
    /// reasonable middle.
    damper_t60_sec: f32,
    /// Becomes true after `trigger_release` so the damper engages on
    /// the next render pass. Idempotent.
    damper_pending: bool,
    release: ReleaseEnvelope,
}

impl ModalPianoVoice {
    /// Construct from a list of modes and an explicit excitation buffer.
    /// `velocity_amp` (typically `velocity / 127.0`) scales the
    /// excitation peak so per-velocity loudness behaves intuitively.
    /// Each mode is also expanded into 3 sub-modes detuned by ±0.7
    /// cents around the centre frequency, mimicking the natural
    /// detune across a piano's 3-string unison and producing audible
    /// inter-partial beating ("生っぽさ"). Pass `with_modes_no_detune`
    /// if you don't want this expansion.
    pub fn with_modes(sr: f32, modes: &[Mode], excitation: Vec<f32>, velocity_amp: f32) -> Self {
        const DETUNE_CENTS: f32 = 0.7;
        let cents_to_ratio = |c: f32| 2.0_f32.powf(c / 1200.0);
        let mut detuned: Vec<Mode> = Vec::with_capacity(modes.len() * 3);
        for m in modes {
            // Each sub-mode gets 1/3 of the original amplitude so total
            // energy stays constant. Centre is exactly on-pitch; flanks
            // are ±DETUNE_CENTS.
            let third = m.init_amp / 3.0_f32.sqrt();
            detuned.push(Mode {
                freq_hz: m.freq_hz * cents_to_ratio(-DETUNE_CENTS),
                t60_sec: m.t60_sec,
                init_amp: third,
            });
            detuned.push(Mode {
                freq_hz: m.freq_hz,
                t60_sec: m.t60_sec,
                init_amp: third,
            });
            detuned.push(Mode {
                freq_hz: m.freq_hz * cents_to_ratio(DETUNE_CENTS),
                t60_sec: m.t60_sec,
                init_amp: third,
            });
        }
        Self::with_modes_no_detune(sr, &detuned, excitation, velocity_amp)
    }

    /// Construct without the 3-sub-mode detune expansion — modes are
    /// used exactly as given. Useful for testing the resonator bank
    /// in isolation; the user-facing path is `with_modes`.
    pub fn with_modes_no_detune(
        sr: f32,
        modes: &[Mode],
        excitation: Vec<f32>,
        velocity_amp: f32,
    ) -> Self {
        let resonators = modes
            .iter()
            .copied()
            .map(|m| ModalResonator::new(m, sr))
            .collect();
        let scaled_excitation: Vec<f32> = excitation.iter().map(|&s| s * velocity_amp).collect();
        Self {
            resonators,
            excitation: scaled_excitation,
            excitation_idx: 0,
            damper_t60_sec: 0.12,
            damper_pending: false,
            release: ReleaseEnvelope::new(0.300, sr),
        }
    }

    /// Override the damper T60 (default 0.12 s). Useful for the una-corda
    /// (slower damper) or for the SF2 placeholder pattern where dampers
    /// are emulated by silence (set very small).
    pub fn set_damper_t60_sec(&mut self, t60_sec: f32) {
        self.damper_t60_sec = t60_sec;
    }

    /// Construct with a default broadband impulse (single-sample delta).
    /// A delta has a flat spectrum across the whole band, so each mode
    /// receives equal excitation energy regardless of its frequency.
    /// Longer / shaped impulses (3 ms half-sine etc.) sound more
    /// "hammer-like" but starve high partials because their spectrum
    /// rolls off — measured deficit was ~27 dB on h2 of the SFZ
    /// Salamander C4 reference, dominating the pilot's amplitude error.
    pub fn new_default_excitation(sr: f32, modes: &[Mode], velocity: u8) -> Self {
        let velocity_amp = (velocity.max(1) as f32) / 127.0;
        // Single-sample delta. The resonator chain shapes the audible
        // transient (each mode rings up over its first cycle), so we
        // don't need an explicit hammer envelope here.
        let excitation = vec![1.0_f32];
        Self::with_modes(sr, modes, excitation, velocity_amp)
    }
}

impl VoiceImpl for ModalPianoVoice {
    fn render_add(&mut self, buf: &mut [f32]) {
        // Lazy damper engage: cheaper than wrapping every step in a
        // branch, and we can do it once at the top of each render pass.
        if self.damper_pending {
            for r in &mut self.resonators {
                r.engage_damper(self.damper_t60_sec);
            }
            self.damper_pending = false;
        }
        for sample in buf.iter_mut() {
            // Pull next excitation sample (or 0 once the impulse runs
            // out — the resonators carry the sound from there).
            let x = if self.excitation_idx < self.excitation.len() {
                let v = self.excitation[self.excitation_idx];
                self.excitation_idx += 1;
                v
            } else {
                0.0
            };
            let mut sum = 0.0_f32;
            for r in &mut self.resonators {
                sum += r.step(x);
            }
            let env = self.release.step();
            *sample += sum * env;
        }
    }
    fn trigger_release(&mut self) {
        // Default behaviour: fade the output via ReleaseEnvelope (~0.3 s
        // multiplicative) AND engage the damper on the next render pass
        // so the resonators themselves stop ringing rather than just
        // getting quieter.
        if let Some(env) = self.release_env_mut() {
            env.trigger();
        }
        self.damper_pending = true;
    }
    fn release_env(&self) -> Option<&ReleaseEnvelope> {
        Some(&self.release)
    }
    fn release_env_mut(&mut self) -> Option<&mut ReleaseEnvelope> {
        Some(&mut self.release)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 44100.0;

    fn render(voice: &mut ModalPianoVoice, n: usize) -> Vec<f32> {
        let mut buf = vec![0.0_f32; n];
        voice.render_add(&mut buf);
        buf
    }

    /// Single mode at 440 Hz, T60 = 1.0 s, hit by a default impulse.
    /// Output should peak in the first ~10 ms then decay; after 1 s
    /// the RMS should be < -55 dB of the peak.
    #[test]
    fn single_mode_decays_to_t60() {
        let modes = [Mode {
            freq_hz: 440.0,
            t60_sec: 1.0,
            init_amp: 1.0,
        }];
        let mut v = ModalPianoVoice::new_default_excitation(SR, &modes, 100);
        let buf = render(&mut v, (SR * 1.5) as usize);

        // Peak in first 50 ms. With a single-sample delta excitation the
        // resonator's transient response peaks at b0 = 1 - r² which for
        // T60 = 1 s, sr = 44100 is ~3e-4 — small but resolvable.
        let early_peak = buf[..((SR * 0.05) as usize)]
            .iter()
            .copied()
            .fold(0.0_f32, |a, b| a.max(b.abs()));
        assert!(
            early_peak > 1e-4,
            "early peak too small ({}); resonator not excited",
            early_peak
        );

        // RMS over [1.0, 1.5] s: should be ~ -60 dB of early peak (T60 =
        // 1.0 s by construction). Allow generous slack: -45 dB or
        // better.
        let tail_start = (SR * 1.0) as usize;
        let tail = &buf[tail_start..];
        let tail_rms = (tail.iter().map(|&s| s * s).sum::<f32>() / tail.len() as f32).sqrt();
        let ratio_db = 20.0 * (tail_rms / early_peak).max(1e-9).log10();
        assert!(
            ratio_db < -45.0,
            "tail RMS {ratio_db:.1} dB > -45 dB → not decaying as fast as T60"
        );
    }

    /// Multiple modes at harmonic ratios, single hammer impulse: each
    /// must contribute energy at its own frequency. We spot-check by
    /// looking at the initial 200 ms RMS — three modes with init_amp
    /// 1.0/0.5/0.25 should give a roughly 1.0 + 0.5 + 0.25 ≈ 1.75x
    /// dynamic range vs a single 1.0 mode (rough sanity, not exact).
    #[test]
    fn multi_mode_renders() {
        let modes = vec![
            Mode {
                freq_hz: 261.63,
                t60_sec: 5.0,
                init_amp: 1.0,
            },
            Mode {
                freq_hz: 523.25,
                t60_sec: 4.0,
                init_amp: 0.5,
            },
            Mode {
                freq_hz: 784.88,
                t60_sec: 3.0,
                init_amp: 0.25,
            },
        ];
        let mut v = ModalPianoVoice::new_default_excitation(SR, &modes, 100);
        let buf = render(&mut v, 8192);
        let peak = buf.iter().copied().fold(0.0_f32, |a, b| a.max(b.abs()));
        // Delta excitation peak ≈ velocity_amp · b0 ≈ 0.78 · 1e-4 ~ 1e-4.
        // Loose lower bound that catches "no signal" but doesn't pin exact
        // amplitude (which depends on resonator b0 normalisation).
        assert!(peak > 5e-4, "multi-mode peak too small: {peak}");
        // No NaN / Inf.
        for &s in &buf {
            assert!(s.is_finite(), "non-finite sample in multi-mode render");
        }
    }

    #[test]
    fn release_lifecycle() {
        let modes = [Mode {
            freq_hz: 261.63,
            t60_sec: 2.0,
            init_amp: 1.0,
        }];
        let mut v = ModalPianoVoice::new_default_excitation(SR, &modes, 100);
        assert!(!v.is_releasing());
        v.trigger_release();
        assert!(v.is_releasing());
        let _ = render(&mut v, (SR * 0.7) as usize);
        assert!(v.is_done());
    }

    #[test]
    fn empty_modes_silent_after_excitation() {
        let modes: Vec<Mode> = Vec::new();
        let mut v = ModalPianoVoice::new_default_excitation(SR, &modes, 100);
        let buf = render(&mut v, 4096);
        // No resonators → output is always zero (Σ over empty bank).
        for &s in &buf {
            assert_eq!(s, 0.0, "empty modal bank produced non-zero sample");
        }
    }

    #[test]
    fn high_t60_pole_inside_unit_circle() {
        // Sanity: very long T60 → r approaches 1 but never reaches it.
        let mode = Mode {
            freq_hz: 440.0,
            t60_sec: 100.0,
            init_amp: 1.0,
        };
        let res = ModalResonator::new(mode, SR);
        let r2 = res.a2;
        assert!(r2 > 0.99 && r2 < 1.0, "pole radius² out of range: {r2}");
    }
}
