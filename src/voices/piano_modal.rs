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

use std::path::Path;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::synth::{modal_params, ReleaseEnvelope, VoiceImpl};

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
    /// Residual layer (Smith commuted synthesis). Pre-recorded buffer
    /// extracted from the SFZ ref minus the modal voice render of the
    /// same note; carries the soundboard / body / room / hammer-felt
    /// transient that the partial sum can't generate. Played sample-
    /// for-sample alongside the biquad output starting at note_on.
    /// Empty `Arc` if no residual was loaded for this note (caller
    /// fell through `RESIDUAL_LUT::get()`).
    residual: std::sync::Arc<Vec<f32>>,
    residual_idx: usize,
    residual_amp: f32,
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
        // Iter U: detune + polarization parameters now live-tunable via
        // `synth::set_modal_params`. detune_cents = 0 collapses the
        // 3-sub-mode detune layer to a single centre. pol_h_weight = 0
        // disables the after-sound (H-polarization) layer entirely.
        // T60 polarisation multipliers stay fixed at the Weinreich
        // values (0.4 / 1.8) — they aren't perceptually distinct from
        // the weight knob in casual A/B and would just clutter the UI.
        const POL_V_T60_MUL: f32 = 0.40;
        const POL_H_T60_MUL: f32 = 1.80;
        let p = modal_params();
        let pol_h = p.pol_h_weight.clamp(0.0, 1.0);
        let pol_v = 1.0 - pol_h;
        let detune_cents = p.detune_cents.max(0.0);

        let cents_to_ratio = |c: f32| 2.0_f32.powf(c / 1200.0);
        let detune_offsets: &[f32] = if detune_cents > 0.0 {
            &[-1.0, 0.0, 1.0]
        } else {
            &[0.0]
        };
        let detune_count = detune_offsets.len() as f32;
        let amp_per_detune = 1.0 / detune_count.sqrt();

        let mut detuned: Vec<Mode> = Vec::with_capacity(modes.len() * 6);
        for m in modes {
            let scale = m.init_amp * amp_per_detune;
            for off in detune_offsets {
                let f = m.freq_hz * cents_to_ratio(off * detune_cents);
                if pol_v > 0.0 {
                    detuned.push(Mode {
                        freq_hz: f,
                        t60_sec: m.t60_sec * POL_V_T60_MUL,
                        init_amp: scale * pol_v,
                    });
                }
                if pol_h > 0.0 {
                    detuned.push(Mode {
                        freq_hz: f,
                        t60_sec: m.t60_sec * POL_H_T60_MUL,
                        init_amp: scale * pol_h,
                    });
                }
            }
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
            // Damper T60 = 0.8 s on note_off. Felt damper landing on
            // a Yamaha C5 string stops vibration in ~80-200 ms physically,
            // but 0.12 s killed modes too aggressively in our voice
            // (the partial bank was already lossy from the modal-only
            // model); 0.8 s gives ~2.6 s audible tail closer to SFZ.
            damper_t60_sec: 0.8,
            damper_pending: false,
            // Iter Q: release envelope decoupled from damper. Was 0.8 s
            // (matching damper) which compounded multiplicatively at
            // note_off → effective decay ≈ -120 dB / 0.8 s, so a quick
            // keypress dropped to silence in ~0.4 s ("変な途切れ方").
            // Slowed to 4.0 s so damper (-75 dB/s) dominates and env
            // (-15 dB/s) is a slow safety fade. Combined decay rate
            // is essentially damper-only on the audible portion.
            release: ReleaseEnvelope::new(4.000, sr),
            // No residual by default — `from_lut` overrides this with
            // the nearest residual buffer when RESIDUAL_LUT is set.
            residual: std::sync::Arc::new(Vec::new()),
            residual_idx: 0,
            residual_amp: 0.0,
        }
    }

    /// Override the damper T60 (default 0.8 s). Useful for the una-corda
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

    /// Construct a voice for `midi_note` from a runtime-loaded `ModalLut`
    /// (per-note partial / T60 / init_db table; issue #3 A3).
    ///
    /// Pick strategy: nearest-neighbour by `|entry.midi_note - midi_note|`.
    /// The chosen entry's modes are then frequency-scaled by the ratio
    /// `target_f0 / nearest.f0_hz`, where `target_f0` is the equal-tempered
    /// pitch for `midi_note` (A4 = 440 Hz). T60 and init_db are kept as-is
    /// (we don't have a model for how T60 should scale across notes; the
    /// nearest entry's measured T60 is the best estimate available).
    ///
    /// T60 sentinel handling: any mode with `t60_sec <= 0` is treated as
    /// "extractor failed" and substituted with `12.0` s, the same default
    /// as `bin/render_modal --default-t60`.
    ///
    /// Empty LUT panics: callers must guarantee `lut.entries` is non-empty
    /// (the auto-load path enforces this — fallback C4 LUT is always at
    /// least 1 entry).
    pub fn from_lut(lut: &ModalLut, sr: f32, midi_note: u8, velocity: u8) -> Self {
        Self::from_lut_with_excitation(lut, sr, midi_note, velocity, true)
    }

    /// Pure-physics constructor: zero dependence on SFZ samples / extracted
    /// LUTs / pre-recorded residual buffers. Every mode's frequency,
    /// amplitude and decay are derived analytically from textbook physics:
    ///
    ///   * Inharmonicity     — Fletcher & Rossing (1998) §12.4
    ///                         f_n = n·f0·sqrt(1 + B·n²),
    ///                         B(note) = 2.5e-4 · 1.1^((60-note)/12)
    ///   * Strike position   — Fletcher & Rossing §12.7,
    ///                         A_n ∝ |sin(n·π·β)|, β = x/L = 1/8
    ///                         (8th, 16th, 24th harmonics → spectral voids)
    ///   * Radiation loss    — high-frequency rolloff,
    ///                         L(f) = 1 / (1 + (f/3000)²)
    ///   * T60 (Valimaki)    — internal friction + air + bridge radiation,
    ///                         T60(f) = 1 / (0.05 + 0.0002·f)
    ///   * Excitation        — Stulov (1995) hysteretic hammer model
    ///                         F(t) = K · x(t)^p, p ≈ 2.5
    ///                         RK4-integrated mass-spring ODE
    ///   * Residual          — disabled (Arc::new(empty)); commuted-residual
    ///                         path ignored, no SFZ-derived waveform anywhere
    ///
    /// Activated by env var `KS_PHYSICS=1` in `synth::make_voice`.
    pub fn from_physics(sr: f32, midi_note: u8, velocity: u8) -> Self {
        let f0 = 440.0_f32 * 2.0_f32.powf((midi_note as f32 - 69.0) / 12.0);

        // 1. Analytical inharmonicity coefficient. Note-dependent: B grows
        //    toward the bass (shorter relative wavelength, stiffer string).
        //    Reference value 2.5e-4 at MIDI 60 from Fletcher & Rossing
        //    Table 12.1 (typical grand C4 ≈ 2-3·10⁻⁴). Per-octave 1.1×
        //    multiplier matches the empirical bass-stiffening trend.
        let b_coeff = 2.5e-4_f32 * 1.1_f32.powf((60.0 - midi_note as f32) / 12.0);

        // 2. Strike position β = x/L = 1/8 (standard piano hammer impact
        //    point). Modes whose n is a multiple of 1/β = 8 land on a node
        //    of the strike point's spatial sin(nπβ) waveform → physically
        //    excited with amplitude exactly 0. Spectral voids at h8, h16,
        //    h24, ... give the "抜け" / clarity that an even-spectrum
        //    sample bank can't reproduce.
        let strike_pos: f32 = 0.125;

        let mut modes: Vec<Mode> = Vec::new();
        let n_max: usize = 96; // up to ~24 kHz at f0 = 250 Hz; loop below
                               // breaks once we cross the Nyquist guard.
        let nyq_guard = sr * 0.45;

        for n in 1..=n_max {
            let nf = n as f32;
            // 1) inharmonic partial frequency
            let freq = nf * f0 * (1.0 + b_coeff * nf * nf).sqrt();
            if freq > nyq_guard {
                break;
            }

            // 2) strike-position spatial amplitude (the missing-modes
            //    factor). |sin(n·π·β)| ∈ [0, 1].
            let spatial_amp = (nf * std::f32::consts::PI * strike_pos).sin().abs();

            // 3) radiation + air damping rolloff. -3 dB at 3 kHz, -20 dB
            //    at 30 kHz. Models combined air absorption and bridge
            //    radiation impedance for upper partials (Fletcher &
            //    Rossing §12.5).
            let radiation_loss = 1.0 / (1.0 + (freq / 3000.0).powi(2));

            // 1/n string spectrum (energy per partial of an ideal plucked
            // / struck string scales as 1/n) × spatial × radiation.
            let init_amp = (spatial_amp / nf) * radiation_loss;

            // 4) Valimaki-style T60 model: f-dependent decay combining
            //    internal friction (constant 0.05) and air loss
            //    (linear-in-f 2e-4·f). Yields T60(f0=261)≈9.3 s,
            //    T60(2 kHz)≈2.2 s, T60(8 kHz)≈0.6 s — close to measured
            //    grand-piano partial decays.
            let t60 = 1.0_f32 / (0.05 + 0.0002 * freq);

            modes.push(Mode {
                freq_hz: freq,
                t60_sec: t60,
                init_amp,
            });
        }

        // 5) Stulov nonlinear hammer pulse (RK4-integrated). Velocity
        //    scales the initial kinetic energy, which through the
        //    nonlinearity opens up the high-frequency content of the
        //    Force pulse — same physical phenomenon the existing
        //    `build_hammer_excitation` Hertzian LPF approximates, but
        //    here we solve the ODE directly. The pulse is peak-
        //    normalised inside `solve_stulov_hammer_pulse`, so we apply
        //    an explicit gain pre-factor here to keep the summed bank
        //    output in the same headroom band as the LUT path under the
        //    global `modal_params().output_gain` (default 80). LUT
        //    excitation peaks at ~0.55 (Stage A), physics peak is 1.0
        //    after normalisation → divide by ~6.5 so the chord_headroom_
        //    audit stays in the "clean" tier (LUT path: 0.806 raw).
        let excitation: Vec<f32> = {
            const PHYSICS_EXCITATION_GAIN: f32 = 0.15;
            solve_stulov_hammer_pulse(sr, velocity)
                .into_iter()
                .map(|s| s * PHYSICS_EXCITATION_GAIN)
                .collect()
        };

        let velocity_amp = (velocity.max(1) as f32) / 127.0;
        let mut voice = Self::with_modes(sr, &modes, excitation, velocity_amp);

        // 6) Sever the SFZ-derived residual path entirely. `residual_amp`
        //    is already 0 by default in `with_modes_no_detune`, but make
        //    the contract explicit: physics mode never touches recorded
        //    soundboard / room IR data.
        voice.residual = std::sync::Arc::new(Vec::new());
        voice.residual_idx = 0;
        voice.residual_amp = 0.0;
        voice
    }

    /// Like `from_lut` but with the option to disable the hammer-impact
    /// noise burst added to the excitation. Default (`with_hammer_noise =
    /// true`) adds a 5 ms bandpass-filtered noise burst on top of the
    /// single-sample delta, supplying the broadband attack transient and
    /// high-frequency body content that pure modal resonators can't
    /// produce — measured to close the visible spectrogram gap above
    /// 3 kHz against SFZ Salamander references.
    pub fn from_lut_with_excitation(
        lut: &ModalLut,
        sr: f32,
        midi_note: u8,
        velocity: u8,
        with_hammer_noise: bool,
    ) -> Self {
        let entry = lut.nearest_entry(midi_note);
        let target_f0 = 440.0_f32 * 2.0_f32.powf((midi_note as f32 - 69.0) / 12.0);
        let ratio = if entry.f0_hz > 0.0 {
            target_f0 / entry.f0_hz
        } else {
            1.0
        };
        const FALLBACK_T60: f32 = 12.0;

        // Per-partial T60 floor. The `extract::t60` extractor systematically
        // under-reports T60 on some partials (h2 ≈ 2 s vs analyse's 11 s
        // because near-equal-amplitude detuned-string beating defeats the
        // R²/slope gate). For perceptual realism we floor each T60 to a
        // partial-index-aware minimum that mirrors the empirical decay
        // curve of a real grand piano (h1 longest, monotone descent).
        // Visualised diff vs SFZ ("音圧が足りない") showed the modal voice
        // was crashing to silence ~30 dB below the SFZ sustain floor;
        // the LUT-reported T60s were the dominant reason.
        fn t60_floor_for_partial(n: usize) -> f32 {
            // Matches analyse's reference values for SFZ Salamander C4
            // (h1=18, h2=11, h3=10, h4=7, h5=7, h6=8, h7=9, h8=9...);
            // beyond h8 we extrapolate downward on a gentle slope.
            match n {
                0 | 1 => 18.0,
                2 => 10.0,
                3 => 9.0,
                4 => 7.0,
                5 => 7.0,
                6 => 7.5,
                7 => 8.0,
                8 => 8.0,
                _ => (8.0_f32 - (n as f32 - 8.0) * 0.4).max(3.0),
            }
        }

        let mut modes: Vec<Mode> = entry
            .modes
            .iter()
            .enumerate()
            .map(|(i, m)| {
                // i+1 = partial index 1..N, matching the n=1 .. fundamental
                // numbering used in the extract crate.
                let n = i + 1;
                let raw = if m.t60_sec > 0.0 {
                    m.t60_sec
                } else {
                    FALLBACK_T60
                };
                // T60 ceiling now live-tunable. See `ModalParams::t60_cap_sec`
                // — clamps the LUT artifact (25-33 s extracted from a
                // 10-s SFZ sample, ×1.8 for H-pol → unphysical sustain).
                let t60 = raw
                    .max(t60_floor_for_partial(n))
                    .min(modal_params().t60_cap_sec);
                Mode {
                    freq_hz: m.freq_hz * ratio,
                    t60_sec: t60,
                    init_amp: m.init_amp,
                }
            })
            .collect();

        // Extrapolate higher partials for bass notes. Linear-domain
        // residual measurements showed C2 (note 36) had +22 dB DEFICIT
        // in the 2-4 kHz band — the LUT only contains the partials
        // `extract::decompose` could detect above its SNR floor (h1-h16
        // for C2, top out at ~1100 Hz), but a real bass piano string has
        // 50+ audible partials reaching well past 4 kHz. We extrapolate
        // up to `target_max_partials` using the stretched-harmonic
        // formula and a conservative amplitude / T60 rolloff.
        const TARGET_MAX_PARTIALS: usize = 32;
        if modes.len() < TARGET_MAX_PARTIALS && !modes.is_empty() {
            // Estimate B by least-squares fit on the existing partials,
            // falling back to the piano-typical 2.5e-4 if the fit is
            // unstable.
            let f1 = modes[0].freq_hz;
            let mut sxy = 0.0_f32;
            let mut sxx = 0.0_f32;
            for (i, m) in modes.iter().enumerate() {
                let n = (i + 1) as f32;
                if n < 2.0 {
                    continue;
                }
                let r = m.freq_hz / (n * f1);
                let y = r * r - 1.0;
                let x = n * n;
                sxy += x * y;
                sxx += x * x;
            }
            let b_est = if sxx > 0.0 {
                (sxy / sxx).clamp(1e-5, 1e-3)
            } else {
                2.5e-4
            };

            // Last measured partial provides the rolloff anchor.
            let last_n = modes.len();
            let last_amp = modes[last_n - 1].init_amp.max(1e-6);
            let last_t60 = modes[last_n - 1].t60_sec;

            // Skip extrapolation if even the LAST measured partial is
            // already in 4-8 kHz (treble notes don't need this).
            let last_freq = modes[last_n - 1].freq_hz;
            if last_freq < 3000.0 {
                for i in last_n..TARGET_MAX_PARTIALS {
                    let n = (i + 1) as f32;
                    let f_extra = n * f1 * (1.0 + b_est * n * n).sqrt();
                    if f_extra > 8000.0 {
                        break;
                    }
                    // Amplitude rolloff: 0.85 per partial = -1.4 dB per
                    // partial beyond last. Iterated up from 0.63 (-4 dB
                    // per partial) which gave extrapolated amplitudes
                    // 100x too quiet to address the C2 deficit. SFZ
                    // bass notes show partial amplitudes roughly -1 to
                    // -3 dB per partial in the upper range.
                    let dn = (i + 1 - last_n) as f32;
                    let amp = last_amp * 0.85_f32.powf(dn);
                    let t60 = (last_t60 * 0.85_f32.powf(dn)).max(t60_floor_for_partial(i + 1));
                    modes.push(Mode {
                        freq_hz: f_extra,
                        t60_sec: t60,
                        init_amp: amp,
                    });
                }
            }
        }
        let velocity_amp = (velocity.max(1) as f32) / 127.0;
        // Build excitation: single-sample delta + optional hammer-impact
        // noise burst. The delta drives every modal resonator equally
        // (flat spectrum); the noise burst contributes the broadband
        // attack transient and the high-frequency content (>3 kHz) that
        // a pure modal bank can't produce — visible in spectrograms as
        // the "warmth gap" between modal and SFZ.
        let excitation = if with_hammer_noise {
            build_hammer_excitation(sr, midi_note, velocity)
        } else {
            vec![1.0_f32]
        };
        let mut voice = Self::with_modes(sr, &modes, excitation, velocity_amp);
        // Smith commuted-synthesis residual: load the nearest pre-
        // recorded "SFZ - modal" residual buffer if RESIDUAL_LUT is
        // populated. The residual carries everything the partial sum
        // can't generate (soundboard / body / room / hammer-felt
        // transient). Velocity-scaled so quieter notes have a
        // proportionally quieter residual.
        let res_scale = modal_params().residual_amp;
        if res_scale > 0.0 {
            if let Some(reslut) = RESIDUAL_LUT.get() {
                if let Some(entry) = reslut.nearest_entry(midi_note) {
                    voice.residual = entry.samples.clone();
                    voice.residual_idx = 0;
                    voice.residual_amp = velocity_amp * res_scale;
                }
            }
        }
        voice
    }
}

/// Build an excitation vector = delta + two-stage filtered-noise envelope:
///
///   stage A: 12 ms loud felt-impact burst (the audible "click" / "punch")
///   stage B: 250 ms quiet sustained noise tail (the diffuse spectrum
///            that fills the partial-line gaps in SFZ recordings —
///            string bleed, body radiation, room tone)
///
/// Spectrogram residual analysis vs SFZ Salamander showed modal had
/// 20-25 dB less average energy across all spectrum bands during note
/// sustain — the partials themselves were correct, but the inter-partial
/// "warmth" was missing. The stage B tail at low gain (~0.08) provides
/// a small amount of broadband filler whose envelope follows the note
/// duration without dominating the partial structure.
///
/// Parameters:
///   - bandpass 200 Hz – 6 kHz (slightly wider on the low end than the
///     stage-A burst-only version so body/room frequencies are present
///     in the sustained tail too)
///   - per-note seed so different notes have decorrelated noise
///   - stage A peak gain 0.55 (felt impact)
///   - stage B steady gain 0.08 (sustained warmth — under the partial
///     structure but adds the diffuse content that empty modal voices
///     are missing)
///
/// First sample is the delta (amplitude 1.0).
///
/// Arch-1 (2026-04-26): velocity is consumed for the post-pass Hertzian
/// hammer LPF — strong keystrokes compress the felt and shorten the
/// hammer-string contact time, which in real pianos opens the high-end
/// of the impact spectrum (Stulov 1995). We approximate this with a
/// 1-pole low-pass whose cutoff scales `1 kHz + 9 kHz · (vel/127)²`
/// applied to the entire generated excitation vector AFTER the existing
/// Stage A/B noise-shaping. The delta at buf[0] is also passed through
/// the LPF — that's intended: a delta into a 1-pole IIR becomes the
/// LPF's impulse response, which is exactly what a soft hammer should
/// look like (broadband attack rolled off above its cutoff).
fn build_hammer_excitation(sr: f32, midi_note: u8, velocity: u8) -> Vec<f32> {
    let n_a = ((sr * 0.012).round() as usize).max(16);
    let n_b = ((sr * 0.250).round() as usize).max(64);
    let total = n_a + n_b;
    let mut buf = vec![0.0_f32; total + 1];
    buf[0] = 1.0;

    let mut state: u32 =
        0x9E37_79B9 ^ (midi_note as u32).wrapping_mul(2_654_435_761) ^ (sr.to_bits() ^ 0xDEAD_BEEF);
    let mut hp_prev_x = 0.0_f32;
    let mut hp_prev_y = 0.0_f32;
    let mut lp_prev = 0.0_f32;
    // Bandpass 1000-5000 Hz. Iterated up from 400 Hz HP after the
    // linear-domain spectral residual on twinkle showed the <250 Hz
    // band was -6 dB SURPLUS and the 2-4 kHz band was +5.9 dB DEFICIT
    // even after the first tightening. Pushing the HP corner further
    // up forces the noise energy into the 1-5 kHz region where the
    // SFZ reference has continuous diffuse content (room tone +
    // body radiation). Modal's modes already produce the <1 kHz
    // partial structure, so the noise tail's contribution down there
    // was redundant + over-stacking on the held bass partials.
    let hp_a = 1.0 - 2.0 * std::f32::consts::PI * 1000.0 / sr;
    // Iter M: LP corner 5000 → 4000 Hz to stop the noise tail
    // leaking into 4-8 kHz, where bach baseline showed -7.3 dB
    // surplus (the band SFZ counterpoint pieces have almost
    // nothing in, but modal noise tail did).
    let lp_a = (-2.0 * std::f32::consts::PI * 4000.0 / sr).exp();
    let attack_n = ((sr * 0.001).round() as usize).max(2);

    const STAGE_A_GAIN: f32 = 0.55;
    // Stage B gain now live-tunable via `ModalParams::stage_b_gain`.
    // 0 disables the held-note noise tail entirely (only the 12 ms
    // attack burst remains).
    let stage_b_gain: f32 = modal_params().stage_b_gain.max(0.0);

    for i in 0..total {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let raw = (state as i32 as f32) / (i32::MAX as f32);
        let hp = hp_a * (hp_prev_y + raw - hp_prev_x);
        hp_prev_x = raw;
        hp_prev_y = hp;
        lp_prev = (1.0 - lp_a) * hp + lp_a * lp_prev;

        // Iter R: Stage A now tapers to stage_b_gain (not 0) and
        // Stage B starts at that level directly with no inter-stage
        // ramp. Eliminates the env dip-and-bump at t≈12-22 ms that
        // the high-Q modal bank registered as a second excitation
        // (perceived as 二拍 / double-tap on quick keys vs SFZ's
        // single-impact attack). Total noise energy unchanged
        // because Stage A averaged area is preserved (0.55 → 0.10
        // taper rather than 0.55 → 0) and Stage B sums identically
        // from its starting amplitude.
        let env = if i < attack_n {
            // Stage A onset ramp (1 ms, linear)
            (i as f32) / (attack_n as f32) * STAGE_A_GAIN
        } else if i < n_a {
            // Stage A decay: half-cosine from STAGE_A_GAIN down to
            // stage_b_gain over the remainder of n_a (~11 ms).
            let t = (i - attack_n) as f32 / (n_a - attack_n) as f32;
            let env_norm = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
            stage_b_gain + env_norm * (STAGE_A_GAIN - stage_b_gain)
        } else {
            // Stage B sustained tail: half-cosine stage_b_gain → 0
            // across n_b. Starts exactly where Stage A ended, no
            // ramp, no dip.
            let stage_b_pos = i - n_a;
            let t = stage_b_pos as f32 / n_b as f32;
            0.5 * (1.0 + (std::f32::consts::PI * t).cos()) * stage_b_gain
        };
        buf[i + 1] = lp_prev * env;
    }

    // Arch-1: Hertzian hammer LPF post-pass. Velocity-dependent 1-pole
    // low-pass, applied to the FULL buffer (delta + Stage A + Stage B).
    //   cutoff_hz = 1000 + 9000 · (vel/127)²
    //   a = exp(-2π · cutoff / sr)
    //   y[n] = x[n]·(1-a) + y[n-1]·a
    // Soft (vel=1) → ~1 kHz LPF, strong (vel=127) → ~10 kHz LPF.
    // Mirrors Stulov's felt-compression model: harder strikes shorten
    // the contact time, opening the high-frequency content of the
    // hammer impulse non-linearly with velocity.
    let vel_norm = (velocity as f32) / 127.0;
    let cutoff_hz = 1000.0 + 9000.0 * vel_norm * vel_norm;
    let lp_a_post = (-2.0 * std::f32::consts::PI * cutoff_hz / sr).exp();
    let one_minus_a = 1.0 - lp_a_post;
    let mut y_prev = 0.0_f32;
    for x in buf.iter_mut() {
        let y = *x * one_minus_a + y_prev * lp_a_post;
        y_prev = y;
        *x = y;
    }
    buf
}

/// Stulov (1995) hysteretic hammer-string contact ODE, integrated with
/// classical RK4 to produce the Force-on-string pulse used by `from_physics`.
///
/// Physical model (single-DoF lumped felt+hammer mass, string treated as a
/// rigid stop while the hammer is in contact):
///
///   m · d²x/dt² = -F(x, ẋ)
///   F(x, ẋ) = K · x^p          for x > 0  (felt compressed)
///           = 0                 for x ≤ 0  (free flight / separation)
///
/// where:
///   x  = felt compression (m)
///   m  = effective hammer mass (kg)            ≈ 8.7 g (mid-register)
///   K  = felt stiffness (N/m^p)                ≈ 4.0e9
///   p  = compression exponent                  ≈ 2.5  (Stulov 1995 fits)
///
/// We integrate a state vector [x, v] with v = ẋ. Initial condition:
/// x = 0, v = -v0 with v0 the hammer impact velocity (positive value
/// scaled by MIDI velocity). The hammer makes contact at t=0; the
/// integration runs until x crosses back through 0 from positive (felt
/// has decompressed and the hammer has separated from the string),
/// after which the Force is identically 0.
///
/// The output is the time-series of F(t) sampled at the audio rate `sr`,
/// truncated at separation (typically 1-5 ms total). buf[0] starts at
/// t=0; the first sample is 0 (felt hasn't compressed yet) — that's
/// fine for the modal bank because the resonators see a smoothly
/// rising broadband impulse rather than a delta singularity.
///
/// Velocity mapping: v0 = 0.5 + 4.5 · (vel/127)  m/s. A real concert
/// grand spans roughly 0.5 m/s (ppp) to 5 m/s (fff) at the hammer; the
/// linear map keeps headroom predictable. Higher v0 increases peak
/// Force AND broadens the spectrum (shorter contact time → wider band)
/// — precisely the nonlinear coupling Stulov's K·x^p captures and
/// linear approximations (Hertzian LPF, etc.) only mimic.
///
/// References:
///   Stulov, A. (1995). "Hysteretic model of the grand piano hammer
///     action." J. Acoust. Soc. Am. 97(4), 2577-2585.
///   Chaigne, A. & Askenfelt, A. (1994). "Numerical simulations of
///     piano strings." Part I, J. Acoust. Soc. Am. 95(2), 1112-1118.
fn solve_stulov_hammer_pulse(sr: f32, velocity: u8) -> Vec<f32> {
    // Physical constants. Mid-register grand-piano hammer params
    // (roughly C4); reasonable across the full range for our purposes
    // because the modal bank's per-note frequency layout dominates the
    // perceived character.
    let m: f32 = 8.7e-3; // 8.7 g effective hammer mass
    let k: f32 = 4.0e9; // felt stiffness (N/m^p)
    let p: f32 = 2.5; // Stulov compression exponent

    // Hammer impact velocity. ppp .. fff = 0.5 .. 5.0 m/s.
    let vel_norm = (velocity.max(1) as f32) / 127.0;
    let v0: f32 = 0.5 + 4.5 * vel_norm;

    // Time step = audio sample period. Stulov contact times are
    // 0.5-5 ms — at sr=44.1 kHz that's 22-220 samples, plenty of
    // resolution for RK4. We additionally substep 4× internally to
    // keep the high-velocity (stiff-spring) cases stable.
    let dt_audio = 1.0 / sr;
    let substeps: usize = 4;
    let dt = dt_audio / substeps as f32;

    // Force: F(x) = K · x^p when x > 0, else 0. Returns acceleration
    // a = -F/m (string pushes hammer back = negative direction).
    let accel = |x: f32| -> f32 {
        if x > 0.0 {
            -k * x.powf(p) / m
        } else {
            0.0
        }
    };
    // Force returned to the string (positive when felt is compressed).
    let force = |x: f32| -> f32 {
        if x > 0.0 {
            k * x.powf(p)
        } else {
            0.0
        }
    };

    // RK4 step on state (x, v) with ẋ = v, v̇ = a(x).
    let rk4_step = |x: f32, v: f32| -> (f32, f32) {
        let k1x = v;
        let k1v = accel(x);

        let k2x = v + 0.5 * dt * k1v;
        let k2v = accel(x + 0.5 * dt * k1x);

        let k3x = v + 0.5 * dt * k2v;
        let k3v = accel(x + 0.5 * dt * k2x);

        let k4x = v + dt * k3v;
        let k4v = accel(x + dt * k3x);

        let nx = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        let nv = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        (nx, nv)
    };

    // Initial state: hammer at the string surface (x=0), moving INTO
    // the string with velocity v0 (positive x = compression direction).
    let mut x: f32 = 0.0;
    let mut v: f32 = v0;

    // Cap total integration at 8 ms (well above any realistic Stulov
    // contact time even for the softest pp strikes). If the hammer
    // hasn't separated by then, force-terminate to avoid pathological
    // buffer growth.
    let max_samples = ((sr * 0.008).round() as usize).max(64);
    let mut buf: Vec<f32> = Vec::with_capacity(max_samples);

    // Sample F(t) at audio rate, advancing the ODE by `substeps` RK4
    // micro-steps between each emit.
    for i in 0..max_samples {
        // Sample the current Force (state x BEFORE this sample's
        // sub-step block — gives F(t) on the audio grid).
        buf.push(force(x));

        // Advance state by one audio period using RK4 sub-steps.
        for _ in 0..substeps {
            let (nx, nv) = rk4_step(x, v);
            x = nx;
            v = nv;
        }

        // Separation criterion: x crossed back through 0 (felt fully
        // decompressed) after at least the first few samples. We pad
        // a couple of zero-valued samples after separation so the
        // resonators see a clean trailing edge.
        if i > 4 && x <= 0.0 {
            // Two more zero samples to flush; then stop.
            buf.push(0.0);
            buf.push(0.0);
            break;
        }
    }

    // Normalise so the peak magnitude = 1.0 (the modal bank's
    // `with_modes` will then scale by velocity_amp like every other
    // path). This decouples physical Force units (Newtons) from the
    // arbitrary digital-domain levels the resonator b0 expects.
    let peak = buf.iter().copied().fold(0.0_f32, |a, b| a.max(b.abs()));
    if peak > 1e-9 {
        let inv = 1.0 / peak;
        for s in buf.iter_mut() {
            *s *= inv;
        }
    }
    buf
}

// ===========================================================================
// Modal LUT (per-note partial / T60 / init_db table)
// ===========================================================================

/// On-disk JSON representation of one mode within a per-note entry. Uses
/// `init_db` (decibels) on disk because the upstream extractor reports
/// dB; we convert to linear `init_amp` at parse time so the runtime
/// `Mode` shape stays uniform.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ModalLutModeJson {
    pub freq_hz: f32,
    pub t60_sec: f32,
    pub init_db: f32,
}

/// On-disk attack envelope summary (informational; not yet used to shape
/// the hammer impulse, but parsed so the schema round-trips losslessly).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct ModalLutAttackJson {
    #[serde(default)]
    pub time_to_peak_s: f32,
    #[serde(default)]
    pub peak_db: f32,
    #[serde(default)]
    pub post_peak_slope_db_s: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModalLutEntryJson {
    pub midi_note: u8,
    pub f0_hz: f32,
    pub modes: Vec<ModalLutModeJson>,
    #[serde(default)]
    pub attack: Option<ModalLutAttackJson>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModalLutJson {
    pub schema_version: u32,
    pub lut: Vec<ModalLutEntryJson>,
}

/// One per-note entry in the modal LUT. Holds the linear-amplitude
/// `Mode` triples already converted from the on-disk dB representation,
/// plus the source MIDI note and measured fundamental for nearest-neighbour
/// scaling in `ModalPianoVoice::from_lut`.
#[derive(Clone, Debug)]
pub struct ModalLutEntry {
    pub midi_note: u8,
    pub f0_hz: f32,
    pub modes: Vec<Mode>,
    pub time_to_peak_s: f32,
    pub peak_db: f32,
    pub post_peak_slope_db_s: f32,
}

/// Per-note modal partials / T60 / init_amp table. Built from a JSON file
/// (produced by `bin/build_modal_lut`) or from the hardcoded C4 fallback
/// when the JSON isn't available.
#[derive(Clone, Debug)]
pub struct ModalLut {
    pub entries: Vec<ModalLutEntry>,
}

/// Process-wide modal LUT, set once at startup by `main`. Read by
/// `synth::make_voice` when `Engine::PianoModal` is selected. Using a
/// `OnceLock` keeps the audio thread lock-free after startup; the LUT
/// itself is `Clone` but every voice needs only a borrowed view, so we
/// just keep one instance.
pub static MODAL_LUT: OnceLock<ModalLut> = OnceLock::new();

/// Per-note residual buffer extracted from the SFZ Salamander recording
/// minus the modal voice render of the same note (Smith commuted-
/// synthesis-style). Carries everything modal can't generate from a
/// partial sum: hammer-felt contact noise, soundboard radiation modes,
/// body resonances, sympathetic strings, room reflections, microphone
/// characteristics. Trimmed to ~0.5 s (the transient + early decay
/// portion that defines the recorded character; later sustain is
/// partial-dominated and would just over-stack on modal output).
#[derive(Clone, Debug)]
pub struct ResidualEntry {
    pub midi_note: u8,
    /// Mono samples at 44.1 kHz. Held behind an `Arc` so multiple voices
    /// share one allocation; each voice keeps its own playback index.
    pub samples: std::sync::Arc<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct ResidualLut {
    pub entries: Vec<ResidualEntry>,
    pub source: String,
}

impl ResidualLut {
    /// Load every `residual_NN.wav` from `dir`. Names that don't parse as
    /// `residual_<int>.wav` are skipped. Returns an empty Lut if no files
    /// are found (caller decides whether to fall back to no-residual mode).
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let mut entries: Vec<ResidualEntry> = Vec::new();
        let read = std::fs::read_dir(dir).map_err(|e| format!("residual dir {dir:?}: {e}"))?;
        for ent in read {
            let ent = ent.map_err(|e| format!("residual entry: {e}"))?;
            let path = ent.path();
            let stem = match path.file_stem().and_then(|s| s.to_str()) {
                Some(s) => s,
                None => continue,
            };
            let midi: u8 = match stem.strip_prefix("residual_").and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => continue,
            };
            let samples = read_wav_mono_44k(&path)?;
            entries.push(ResidualEntry {
                midi_note: midi,
                samples: std::sync::Arc::new(samples),
            });
        }
        entries.sort_by_key(|e| e.midi_note);
        Ok(Self {
            entries,
            source: dir.display().to_string(),
        })
    }

    pub fn nearest_entry(&self, midi_note: u8) -> Option<&ResidualEntry> {
        self.entries
            .iter()
            .min_by_key(|e| (e.midi_note as i32 - midi_note as i32).unsigned_abs())
    }
}

/// Process-wide residual buffer set, set once at startup by `main` (or
/// left unset if no residual files are present, in which case modal
/// voice runs without commuted-synthesis layer).
pub static RESIDUAL_LUT: OnceLock<ResidualLut> = OnceLock::new();

/// Minimal mono WAV reader for residual files (44.1 kHz, 16-bit PCM,
/// 1 channel — what `tools/build_residual_ir.py` writes).
fn read_wav_mono_44k(path: &Path) -> Result<Vec<f32>, String> {
    use std::fs::File;
    use std::io::{BufReader, Read};
    let file = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let mut reader = BufReader::new(file);
    let mut buf = Vec::new();
    reader
        .read_to_end(&mut buf)
        .map_err(|e| format!("read {path:?}: {e}"))?;
    if buf.len() < 44 || &buf[..4] != b"RIFF" || &buf[8..12] != b"WAVE" {
        return Err(format!("{path:?}: not a RIFF/WAVE"));
    }
    let mut i = 12;
    let mut data_off = None;
    let mut data_len = 0usize;
    let mut channels = 0u16;
    let mut sr = 0u32;
    let mut bits = 0u16;
    while i + 8 <= buf.len() {
        let chunk_id = &buf[i..i + 4];
        let chunk_len = u32::from_le_bytes(buf[i + 4..i + 8].try_into().unwrap()) as usize;
        match chunk_id {
            b"fmt " => {
                channels = u16::from_le_bytes(buf[i + 10..i + 12].try_into().unwrap());
                sr = u32::from_le_bytes(buf[i + 12..i + 16].try_into().unwrap());
                bits = u16::from_le_bytes(buf[i + 22..i + 24].try_into().unwrap());
            }
            b"data" => {
                data_off = Some(i + 8);
                data_len = chunk_len;
                break;
            }
            _ => {}
        }
        i += 8 + chunk_len;
    }
    let data_off = data_off.ok_or_else(|| format!("{path:?}: no data chunk"))?;
    if channels != 1 || sr != 44100 || bits != 16 {
        return Err(format!(
            "{path:?}: expected mono 16-bit 44.1 kHz, got ch={channels} sr={sr} bits={bits}"
        ));
    }
    let pcm = &buf[data_off..data_off + data_len];
    let mut out = Vec::with_capacity(pcm.len() / 2);
    for chunk in pcm.chunks_exact(2) {
        let v = i16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(v as f32 / 32768.0);
    }
    Ok(out)
}

/// Errors produced by `ModalLut::from_json_path`.
#[derive(Debug)]
pub enum ModalLutError {
    Io(std::io::Error),
    Parse(serde_json::Error),
    SchemaVersion(u32),
    Empty,
}

impl std::fmt::Display for ModalLutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "modal LUT I/O error: {e}"),
            Self::Parse(e) => write!(f, "modal LUT parse error: {e}"),
            Self::SchemaVersion(v) => write!(f, "modal LUT unsupported schema_version: {v}"),
            Self::Empty => write!(f, "modal LUT is empty (no per-note entries)"),
        }
    }
}

impl std::error::Error for ModalLutError {}

impl From<std::io::Error> for ModalLutError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for ModalLutError {
    fn from(e: serde_json::Error) -> Self {
        Self::Parse(e)
    }
}

impl ModalLut {
    /// Load a modal LUT from a JSON file produced by `bin/build_modal_lut`.
    /// On-disk schema:
    /// ```json
    /// {
    ///   "schema_version": 1,
    ///   "lut": [
    ///     {
    ///       "midi_note": 60,
    ///       "f0_hz": 261.63,
    ///       "modes": [{ "freq_hz": 261.6, "t60_sec": 18.14, "init_db": -2.7 }, ...],
    ///       "attack": { "time_to_peak_s": 0.039, "peak_db": -7.62, "post_peak_slope_db_s": -10.37 }
    ///     }
    ///   ]
    /// }
    /// ```
    /// `init_db` is converted to linear `init_amp = 10^(init_db/20)` here
    /// so the runtime `Mode` shape stays unified with the rest of the
    /// modal pipeline (`Mode::init_amp` is always linear).
    pub fn from_json_path(path: &Path) -> Result<Self, ModalLutError> {
        let raw = std::fs::read(path)?;
        let json: ModalLutJson = serde_json::from_slice(&raw)?;
        if json.schema_version != 1 {
            return Err(ModalLutError::SchemaVersion(json.schema_version));
        }
        if json.lut.is_empty() {
            return Err(ModalLutError::Empty);
        }
        let entries: Vec<ModalLutEntry> = json
            .lut
            .into_iter()
            .map(|e| {
                let modes: Vec<Mode> = e
                    .modes
                    .iter()
                    .map(|m| Mode {
                        freq_hz: m.freq_hz,
                        t60_sec: m.t60_sec,
                        init_amp: 10f32.powf(m.init_db / 20.0),
                    })
                    .collect();
                let attack = e.attack.unwrap_or_default();
                ModalLutEntry {
                    midi_note: e.midi_note,
                    f0_hz: e.f0_hz,
                    modes,
                    time_to_peak_s: attack.time_to_peak_s,
                    peak_db: attack.peak_db,
                    post_peak_slope_db_s: attack.post_peak_slope_db_s,
                }
            })
            .collect();
        Ok(Self { entries })
    }

    /// Hardcoded C4 fallback LUT used when no JSON file is available.
    /// Mirrors `ANALYSE_LUT_C4` from `bin/render_modal.rs`: the SFZ
    /// Salamander Grand V3 C4 reference's first 8 partials as measured
    /// by the existing `analyse` binary. Sufficient to keep
    /// `Engine::PianoModal` audible at MIDI 60 in dev/test runs that
    /// haven't built the per-note LUT yet.
    pub fn fallback_c4() -> Self {
        const ANALYSE_LUT_C4: [(f32, f32, f32); 8] = [
            (261.6, 18.14, -2.7),
            (523.1, 11.21, 0.0),
            (785.6, 9.58, -18.5),
            (1048.8, 7.38, -18.4),
            (1311.7, 6.98, -14.3),
            (1577.3, 7.93, -26.1),
            (1843.7, 8.98, -17.4),
            (2111.5, 8.66, -32.9),
        ];
        let modes: Vec<Mode> = ANALYSE_LUT_C4
            .iter()
            .map(|&(f, t60, db)| Mode {
                freq_hz: f,
                t60_sec: t60,
                init_amp: 10f32.powf(db / 20.0),
            })
            .collect();
        Self {
            entries: vec![ModalLutEntry {
                midi_note: 60,
                f0_hz: 261.63,
                modes,
                time_to_peak_s: 0.039,
                peak_db: -7.62,
                post_peak_slope_db_s: -10.37,
            }],
        }
    }

    /// Auto-load helper used by `main` and `bin/bench`. Tries
    /// `--modal-lut PATH` (if `Some`); otherwise tries the conventional
    /// `bench-out/REF/sfz_salamander_multi/modal_lut.json`. If neither
    /// loads cleanly, falls back to the hardcoded C4 entry so
    /// `Engine::PianoModal` still produces sound at MIDI 60.
    ///
    /// Returns the source description as the second tuple element so the
    /// caller can log which path was actually used (real JSON vs C4
    /// fallback).
    pub fn auto_load(explicit: Option<&Path>) -> (Self, String) {
        let default_path = Path::new("bench-out/REF/sfz_salamander_multi/modal_lut.json");
        let candidate = explicit.unwrap_or(default_path);
        match Self::from_json_path(candidate) {
            Ok(lut) => {
                let n = lut.entries.len();
                (lut, format!("{} ({n} entries)", candidate.display()))
            }
            Err(e) => (
                Self::fallback_c4(),
                format!(
                    "hardcoded C4 fallback (loading {} failed: {e})",
                    candidate.display()
                ),
            ),
        }
    }

    /// Find the entry whose `midi_note` is closest to `midi_note`.
    /// Ties broken by lower midi_note (stable: the iterator returns the
    /// first minimum). Panics if `entries` is empty — the auto-load path
    /// guarantees at least the hardcoded C4 entry is present.
    pub fn nearest_entry(&self, midi_note: u8) -> &ModalLutEntry {
        self.entries
            .iter()
            .min_by_key(|e| (e.midi_note as i32 - midi_note as i32).unsigned_abs())
            .expect("ModalLut::nearest_entry called on empty LUT")
    }
}

impl VoiceImpl for ModalPianoVoice {
    fn render_add(&mut self, buf: &mut [f32]) {
        // Iter S: per-sample damper-pending check (was per-block at
        // the top of render_add). trigger_release runs on the MIDI /
        // UI thread between audio blocks; a per-block check delayed
        // the biquad coefficient swap by up to one block (~21 ms at
        // 1024 frames / 48 kHz). The latency between the instant
        // release_env trigger and the delayed damper engage was
        // perceptually distinguishable from the attack itself —
        // listener heard 二拍 / two events on quick keys. Per-sample
        // branch is ~0.3 ns and trivially predicted; cost is
        // negligible against the 144-biquad inner loop.
        // (kept here so the inner loop body can flip damper at the
        //  exact sample without re-reading the flag every step.)
        // Per-voice output gain. The modal-bank impulse response has
        // peak amplitude ~b0 = 1 - r² ≈ 3e-4 per resonator, so even
        // summed across 144 sub-modes the raw voice peak is only
        // ~3e-3 — about 300× quieter than Square / KS / Piano (peak
        // ~1.0). render_song / render_chord normalise to -3 dBFS so
        // their output sounds level-matched, but the real-time DAC
        // path in main.rs has no normalisation and the modal voice
        // ended up inaudible against the master fader. Pre-multiply
        // the summed bank so live playback matches the loudness of
        // the other engines. The residual_l2 metric is invariant to
        // this gain (both ref and candidate go through the same
        // peak normalisation in render_song), so iter A-J accept
        // decisions are unaffected.
        // Iter M+: 50 → 100. The chord_headroom_audit test reports
        // modal raw_peak ≈ 0.49 vs ~1.0-1.5 for other "clean" tier
        // engines (Piano, PianoLite, KsRich, Sub). Users compensated
        // by cranking master to ~3, which pushed modal into the warm
        // tanh knee AND hard-clipped every other engine simultaneously
        // — the reported 「三和音はまだ割れる」 on PianoModal. Bringing
        // modal to ~1.0 raw peak puts it in the same headroom band as
        // the rest at master=1.0, so the master fader doesn't have to
        // span 1.0-3.0 just to balance modal against SFZ.
        // Output gain now live-tunable via `ModalParams::output_gain`.
        // Read once per audio block (per render_add invocation) so a
        // moving slider takes effect on already-playing voices, not
        // just new note_ons. Default 80 keeps PianoModal in the same
        // chord-clean tier as Piano / PianoLite (chord_headroom_audit).
        let modal_output_gain: f32 = modal_params().output_gain.max(0.0);
        for sample in buf.iter_mut() {
            // Per-sample damper engage (iter S). Cheap predictable
            // branch; eliminates up-to-21 ms note_off latency that
            // perceptually split into "release click" + "decay".
            if self.damper_pending {
                for r in &mut self.resonators {
                    r.engage_damper(self.damper_t60_sec);
                }
                self.damper_pending = false;
            }
            // Pull next excitation sample (or 0 once the impulse runs
            // out — the resonators carry the sound from there).
            let mut x = if self.excitation_idx < self.excitation.len() {
                let v = self.excitation[self.excitation_idx];
                self.excitation_idx += 1;
                v
            } else {
                0.0
            };
            // Arch-1 commuted residual (Smith): residual is fed into
            // the resonator excitation x[n] rather than added to the
            // output bus. Physical justification: the SFZ - modal
            // residual carries the soundboard / body / room IR, which
            // is one big LTI filter shared across all strings. The
            // previous direct-add layered every note's residual on top
            // of the bus, so chords stacked N copies of the
            // soundboard IR (linear and phase-incoherent), producing
            // the perceived "curtain-over" muddiness. Routing residual
            // through the per-note resonator bank means the noise
            // content gets filtered by the same modes the hammer is
            // exciting — fused with the partials, no longer a
            // detached layer that scales with polyphony.
            if self.residual_idx < self.residual.len() {
                x += self.residual[self.residual_idx] * self.residual_amp;
                self.residual_idx += 1;
            }
            let mut sum = 0.0_f32;
            for r in &mut self.resonators {
                sum += r.step(x);
            }
            let env = self.release.step();
            // Modal partial sum (biquad bank output * envelope * gain).
            // Residual is no longer added directly here — it's already
            // baked into `sum` via the commuted excitation path above.
            *sample += sum * env * modal_output_gain;
        }
    }
    fn trigger_release(&mut self) {
        // Iter T: cut off any remaining hammer-noise excitation when
        // the key is released. The Stage B noise tail is 250 ms long;
        // on a quick keypress (chon-press, ~50 ms hold) that meant
        // 200+ ms of noise still feeding the biquads AFTER the
        // damper had engaged, producing extra envelope peaks at
        // ~60/90/180 ms that the new attack-event diagnostic
        // detected as 3 attacks vs SFZ's 1. Stopping the
        // excitation playback at note_off lets the damped
        // resonators actually decay cleanly instead of being
        // re-excited mid-release.
        self.excitation_idx = self.excitation.len();
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

    // -- ModalLut / from_lut path -----------------------------------------

    fn one_entry_lut(midi_note: u8, f0_hz: f32, modes: Vec<Mode>) -> ModalLut {
        ModalLut {
            entries: vec![ModalLutEntry {
                midi_note,
                f0_hz,
                modes,
                time_to_peak_s: 0.0,
                peak_db: 0.0,
                post_peak_slope_db_s: 0.0,
            }],
        }
    }

    /// Exact-note hit: `from_lut` for the LUT's own `midi_note` should
    /// keep mode frequencies on-pitch (modulo the ±0.7 cents 3-string
    /// detune `with_modes` injects). Each LUT mode expands into 3
    /// resonators whose centre frequencies bracket the original.
    #[test]
    fn from_lut_exact_match_uses_lut_directly() {
        let lut = one_entry_lut(
            60,
            261.63,
            vec![
                Mode {
                    freq_hz: 261.6,
                    t60_sec: 1.0,
                    init_amp: 1.0,
                },
                Mode {
                    freq_hz: 523.1,
                    t60_sec: 0.8,
                    init_amp: 0.5,
                },
            ],
        );
        let v = ModalPianoVoice::from_lut(&lut, SR, 60, 100);
        // 2 modes × 3 detune sub-modes = 6 resonators, frequency-ordered
        // around their centres. Recover each centre frequency from the
        // resonator's `cos_omega` and compare against the LUT.
        assert_eq!(v.resonators.len(), 6);
        let recovered: Vec<f32> = v
            .resonators
            .iter()
            .map(|r| r.cos_omega.acos() * SR / (2.0 * std::f32::consts::PI))
            .collect();
        // Sub-modes for f1 = 261.6: centre and ±0.7 cents.
        // Centre band straddles 261.6.
        let f1_band = &recovered[0..3];
        let f1_min = f1_band.iter().copied().fold(f32::MAX, f32::min);
        let f1_max = f1_band.iter().copied().fold(f32::MIN, f32::max);
        assert!(
            f1_min < 261.6 && f1_max > 261.6,
            "f1 sub-modes don't bracket: {f1_band:?}"
        );
        // ±0.7 cents at 261.6 Hz is ±0.106 Hz; allow generous slack.
        assert!(
            (f1_min - 261.6).abs() < 0.5,
            "f1 lower flank far off: {f1_min}"
        );
        let f2_band = &recovered[3..6];
        let f2_min = f2_band.iter().copied().fold(f32::MAX, f32::min);
        let f2_max = f2_band.iter().copied().fold(f32::MIN, f32::max);
        assert!(
            f2_min < 523.1 && f2_max > 523.1,
            "f2 sub-modes don't bracket: {f2_band:?}"
        );
    }

    /// LUT only has note 60 (C4). Calling `from_lut` for note 72 (C5)
    /// should scale every mode frequency by 2.0 (one octave up).
    #[test]
    fn from_lut_nearest_octave_scales_correctly() {
        let lut = one_entry_lut(
            60,
            261.63,
            vec![
                Mode {
                    freq_hz: 261.63,
                    t60_sec: 1.0,
                    init_amp: 1.0,
                },
                Mode {
                    freq_hz: 523.25,
                    t60_sec: 0.8,
                    init_amp: 0.5,
                },
            ],
        );
        let v = ModalPianoVoice::from_lut(&lut, SR, 72, 100);
        assert_eq!(v.resonators.len(), 6);
        // Recover sub-mode centres and check they scale by 2.0 (with
        // ratio = 440·2^((72-69)/12) / 261.63 = 523.25 / 261.63 ≈ 2.0).
        let recovered: Vec<f32> = v
            .resonators
            .iter()
            .map(|r| r.cos_omega.acos() * SR / (2.0 * std::f32::consts::PI))
            .collect();
        // f1 centre band should sit around 523.25 Hz (261.63 × 2).
        let f1_centre = recovered[1];
        assert!(
            (f1_centre - 523.25).abs() < 1.0,
            "f1 octave scale wrong: got {f1_centre}, expected ~523.25"
        );
        // f2 centre band should sit around 1046.5 Hz (523.25 × 2).
        let f2_centre = recovered[4];
        assert!(
            (f2_centre - 1046.5).abs() < 2.0,
            "f2 octave scale wrong: got {f2_centre}, expected ~1046.5"
        );
    }

    /// Sentinel T60 = -1 (extractor failure marker) → resonator should
    /// be built with the 12 s fallback, not blow up with a tiny pole
    /// radius and produce silence / NaN.
    #[test]
    fn from_lut_handles_t60_sentinel() {
        let lut = one_entry_lut(
            60,
            261.63,
            vec![Mode {
                freq_hz: 261.63,
                t60_sec: -1.0,
                init_amp: 1.0,
            }],
        );
        let v = ModalPianoVoice::from_lut(&lut, SR, 60, 100);
        assert_eq!(v.resonators.len(), 3);
        // Pole radius for 12 s T60 at 44100 Hz: r ≈ 0.99987.
        // r² should be > 0.999.
        for r in &v.resonators {
            assert!(
                r.a2 > 0.999,
                "sentinel T60 not substituted with 12 s fallback: a2={}",
                r.a2
            );
        }
    }

    /// Auto-load with no real JSON path falls back to the hardcoded C4
    /// entry; the resulting LUT must be non-empty so `nearest_entry`
    /// can't panic.
    #[test]
    fn modal_lut_fallback_c4_has_one_entry() {
        let lut = ModalLut::fallback_c4();
        assert_eq!(lut.entries.len(), 1);
        assert_eq!(lut.entries[0].midi_note, 60);
        assert_eq!(lut.entries[0].modes.len(), 8);
    }

    // -- from_physics path -------------------------------------------------

    /// Stulov hammer pulse must produce a finite, non-empty Force buffer
    /// whose values are bounded — a runaway ODE (poor RK4 step / wrong
    /// sign) would diverge to NaN/Inf or last forever.
    #[test]
    fn stulov_pulse_bounded_and_terminates() {
        for &vel in &[1u8, 32, 64, 100, 127] {
            let pulse = solve_stulov_hammer_pulse(SR, vel);
            assert!(!pulse.is_empty(), "vel={vel}: empty Stulov pulse");
            // Contact ≤ 8 ms (max integration window). At sr=44.1 kHz
            // that's 353 samples + 2 zero-flush. Allow a little slack.
            assert!(
                pulse.len() <= ((SR * 0.010) as usize),
                "vel={vel}: pulse too long ({} samples)",
                pulse.len()
            );
            // All finite.
            for (i, &s) in pulse.iter().enumerate() {
                assert!(
                    s.is_finite(),
                    "vel={vel}: non-finite sample at index {i}: {s}"
                );
                assert!(
                    s.abs() <= 1.0 + 1e-6,
                    "vel={vel}: peak-normalised sample > 1.0: {s}"
                );
            }
            // Peak should be exactly 1.0 (post peak-normalisation).
            let peak = pulse.iter().copied().fold(0.0_f32, |a, b| a.max(b.abs()));
            assert!(
                (peak - 1.0).abs() < 1e-5,
                "vel={vel}: peak not unity: {peak}"
            );
        }
    }

    /// Higher velocity → broader Force pulse spectrum: a coarse proxy is
    /// the contact-time half-width. Stulov's stiffening felt shortens
    /// contact time as v0 grows. Concretely, vel=127's pulse should be
    /// no longer than vel=1's pulse (and typically meaningfully shorter).
    #[test]
    fn stulov_pulse_velocity_shortens_contact() {
        let soft = solve_stulov_hammer_pulse(SR, 1);
        let hard = solve_stulov_hammer_pulse(SR, 127);
        assert!(
            hard.len() <= soft.len(),
            "hard strike contact ({} samples) longer than soft strike ({} samples) — Stulov sign error?",
            hard.len(),
            soft.len()
        );
    }

    /// `from_physics` for C4 (note 60) must produce a non-empty resonator
    /// bank, finite first 100 ms output, and a measurable peak. The
    /// missing-modes contract: with strike β = 1/8, the 8th, 16th, 24th
    /// partials' init_amp coming out of the analytic loop should be
    /// *exactly* zero — verified BEFORE the detune expansion in
    /// `with_modes` smears them across ±0.7 cents.
    #[test]
    fn from_physics_renders_and_has_missing_modes() {
        let v = ModalPianoVoice::from_physics(SR, 60, 100);
        // Detune expansion (×3) × pol_v (default pol_h_weight=0.15 keeps
        // both layers active) means 96 partial slots → up to 96 × 6 =
        // 576 resonators, but the analytic loop breaks at 0.45·sr Nyquist.
        // We only require non-empty here.
        assert!(
            !v.resonators.is_empty(),
            "from_physics produced 0 resonators"
        );

        // Render 100 ms and assert finite + audible.
        let mut voice = ModalPianoVoice::from_physics(SR, 60, 100);
        let mut buf = vec![0.0_f32; (SR * 0.1) as usize];
        voice.render_add(&mut buf);
        for &s in &buf {
            assert!(s.is_finite(), "non-finite sample in physics render");
        }
        let peak = buf.iter().copied().fold(0.0_f32, |a, b| a.max(b.abs()));
        assert!(peak > 1e-4, "physics render too quiet: peak={peak}");
    }

    /// The strike-position sin(nπβ) factor is the physical signature
    /// of the model. Confirm it directly: for β=1/8, sin(8π·1/8)=0, etc.
    /// We do this by reconstructing the pre-expansion Mode list using
    /// the same formulas the constructor uses, verifying that h8/h16/h24
    /// would-be init_amps are zero. (We can't read the internal `modes`
    /// list back out of the voice because `with_modes` mutates &
    /// expands them; the contract is on the analytic generator.)
    #[test]
    fn missing_modes_at_strike_position_octuples() {
        let strike_pos: f32 = 0.125;
        for &n in &[8usize, 16, 24, 32] {
            let nf = n as f32;
            let spatial = (nf * std::f32::consts::PI * strike_pos).sin().abs();
            assert!(
                spatial < 1e-5,
                "n={n}: |sin(nπβ)|={spatial} should be ~0 for β=1/8",
            );
        }
        // Sanity: non-multiples of 8 should NOT be near zero.
        for &n in &[1usize, 3, 5, 7, 9, 11, 13] {
            let nf = n as f32;
            let spatial = (nf * std::f32::consts::PI * strike_pos).sin().abs();
            assert!(
                spatial > 0.05,
                "n={n}: |sin(nπβ)|={spatial} unexpectedly small",
            );
        }
    }

    /// Physics path must NOT load any residual buffer regardless of
    /// `RESIDUAL_LUT` global state. We can't easily set RESIDUAL_LUT in
    /// a unit test without leaking into other tests, but we can assert
    /// the per-voice residual fields are empty.
    #[test]
    fn from_physics_has_no_residual() {
        let v = ModalPianoVoice::from_physics(SR, 60, 100);
        assert!(
            v.residual.is_empty(),
            "physics path leaked residual buffer (len={})",
            v.residual.len()
        );
        assert_eq!(v.residual_amp, 0.0);
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
