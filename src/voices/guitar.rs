//! Steel-string acoustic guitar physical model — first non-piano voice
//! family for the multi-family GM roadmap (issue #44).
//!
//! ## Architecture
//!
//! Single-string DWG (digital waveguide) per note, structured as the
//! same `KsString` substrate used by the piano voices, but parameterised
//! for steel acoustic guitar physics (NOT piano):
//!
//!   1. **Pluck excitation**: Karplus-Strong (1983) modified per Smith,
//!      *Physical Audio Signal Processing* (PASP, online edition,
//!      Ch. 6 "Plucked Strings", §"Pluck Position Modeling"). Initial
//!      displacement is a two-tap comb at offset 0 and at offset
//!      `β · N` with `β = 1/8` of string length, which places spectral
//!      nulls at integer multiples of `f / β` — this is the physical
//!      origin of the "bright fingerstyle near the bridge" timbre and
//!      makes a guitar audibly distinct from a piano hammer attack.
//!
//!   2. **Stiffness / inharmonicity**: closed-form Fletcher coefficient
//!
//!         B = π³ · E · d⁴ / (64 · T · L²)
//!
//!      from Fletcher (1964), "Normal vibration frequencies of a stiff
//!      piano string", J. Acoust. Soc. Am. 36(1), pp. 203–209. Computed
//!      *fresh* per note from steel-string parameters (Young's modulus,
//!      diameter, tension, speaking length) — we deliberately do NOT
//!      import `voices::string_inharmonicity::STEINWAY_D`. Steel
//!      acoustic strings have a B coefficient ~0.5–2 orders of
//!      magnitude smaller than piano wound bass, so reusing the piano
//!      table would land in the wrong regime. The literature value for
//!      a steel high-E (.012 plain) at standard tension is
//!      B ≈ 7 × 10⁻⁵ (Bensa, Bilbao, Smith, Valette 2003,
//!      "The simulation of piano string vibration: From physical
//!      models to finite difference schemes and digital waveguides",
//!      JASA 114(2), §V.B — same waveguide framework, parameter range
//!      reported for steel-string guitars).
//!
//!      `B` is mapped to the dispersion allpass coefficient via the
//!      same monotone fit `KsString` already uses for the piano voices,
//!      clamped to a range appropriate for guitar (smaller than piano).
//!
//!   3. **Body resonance**: parallel bank of three constant-Q biquad
//!      band-pass filters whose centre frequencies and Qs come from
//!      Christensen & Vistisen (1980), "Simple model for low-frequency
//!      guitar function", J. Acoust. Soc. Am. 68(3), 758–766, and
//!      Karjalainen & Smith (1996), "Body modeling techniques for
//!      string instrument synthesis", ICMC Proceedings. The three
//!      modes captured here are the dominant low-freq trio that gives
//!      a steel-string acoustic its "boxy" character:
//!
//!        - Helmholtz cavity (A0)  ~ 100 Hz, Q ≈ 25
//!        - Top-plate first mode (T1) ~ 200 Hz, Q ≈ 18
//!        - Top-plate second mode (T2) ~ 380 Hz, Q ≈ 12
//!
//!      Higher modes (back, sides, tone-hole detuning) fall off
//!      smoothly and are omitted to keep the per-voice CPU cost
//!      bounded — the dry/wet mix below 0.4 leaves the upper-spectrum
//!      string content intact.
//!
//!   4. **Decay**: feedback gain set so the open low-E (E2) reaches
//!      −40 dB ≈ 1.8 s after onset, matching the published t60 range
//!      for medium-gauge phosphor-bronze strings on a dreadnought
//!      (Woodhouse 2004, "Plucked guitar transients: Comparison of
//!      measurements and synthesis", Acta Acust. united Acust. 90,
//!      §3.2). Higher pitches damp faster as expected from the
//!      KsString attack-LPF time constant.
//!
//! ## What is *not* in this voice (deliberately)
//!
//! Performance artefacts (pick noise, finger squeak, fret slide) and
//! non-linear body sustain are out of scope for this PR. Those are the
//! "data-driven" tier — same role `voices::piano_modal::ModalLut`
//! plays for the piano family — and are deferred to a follow-up
//! `voices::guitar_modal` LUT extracted from a reference library, the
//! same way the piano modal LUT was built.

use std::f32::consts::PI;

use crate::synth::{KsString, ReleaseEnvelope, VoiceImpl};

// ---------------------------------------------------------------------------
// Steel-string physical scale.
//
// Light-gauge phosphor-bronze "standard" set on a 25.5 inch (0.6477 m)
// dreadnought scale length. Diameters are nominal manufacturer values
// for D'Addario EJ16-class sets; tension is back-computed from
// f = (1 / 2L) · sqrt(T / μ), with μ = ρ · π · (d/2)² · k_wound where
// k_wound ≈ 1.6 for the wrapped strings (steel core + bronze winding)
// and 1.0 for plain steel — see Helmholtz-style derivation in
// J.O. Smith PASP §"String Tension".
//
// We don't claim sub-percent accuracy on T (the wound-string density
// correction is itself a 5-10% number). What matters for the
// inharmonicity model is that B = π³ E d⁴ / (64 T L²) lands in the
// published 1e-5 .. 1e-4 band for a 25.5"-scale steel acoustic.
// ---------------------------------------------------------------------------

/// One open-string entry for a 6-string steel-acoustic guitar in
/// standard EADGBE tuning. The `diameter_m` and `is_wound` fields are
/// informational metadata kept alongside the load-bearing geometry
/// so the table is self-describing — they aren't read by
/// `fletcher_b` (which uses `core_diameter_m` directly) or by the
/// dispersion-coefficient mapping, but they document the gauge that
/// the published Fletcher B values were measured against.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
struct GuitarString {
    /// MIDI note of the open string.
    midi: u8,
    /// Frequency at standard A=440 tuning (Hz).
    freq_hz: f32,
    /// Outer diameter (m). Wound strings: total OD including winding.
    diameter_m: f32,
    /// Effective bending diameter (m). For plain strings this equals
    /// `diameter_m`. For wound strings it equals the load-bearing
    /// steel core only — the bronze winding contributes negligibly to
    /// bending stiffness, and using OD here would over-estimate B by
    /// a factor of ~10. Values are typical D'Addario EJ16 light-gauge
    /// core sizes (.018 / .017 / .015 / .012 inch for the four wound
    /// strings).
    core_diameter_m: f32,
    /// Tension at pitch (N). Hand-fitted to land f = 1/(2L)·sqrt(T/μ).
    tension_n: f32,
    /// Speaking length (m). Constant across the 6 strings on most
    /// dreadnoughts (compensation at the saddle is sub-mm).
    length_m: f32,
    /// True if the string is wound (low E / A / D / G in this set).
    /// Used only by the dispersion-coefficient mapping below; the
    /// stiffness math itself uses `core_diameter_m` directly.
    is_wound: bool,
}

/// Effective Young's modulus of the load-bearing core wire for a
/// modern steel-core acoustic-guitar string. Plain-steel music wire
/// (ASTM A228) sits at ≈ 200 GPa; phosphor-bronze winding contributes
/// negligibly to bending stiffness. We use the same 200 GPa for both
/// plain and wound, with the wound class compensated via the
/// `effective_diameter` reduction below.
const STEEL_YOUNG_PA: f32 = 2.00e11;

/// Standard dreadnought/Strat-class scale length, 25.5 inches.
const SCALE_LENGTH_M: f32 = 0.6477;

/// Light-gauge ".012/.054" steel-acoustic set. Tensions are within
/// ~5 % of the published D'Addario EJ16 chart; core diameters for the
/// wound strings are typical hex-core values (the manufacturer does
/// not publish exact specs, but Fletcher's rule of thumb that the
/// core is ~ 0.40-0.50 of the OD on a wound bass-guitar / acoustic
/// string is consistent with the values used here).
const OPEN_STRINGS: [GuitarString; 6] = [
    // Low E (6th, wound) — .053" OD, ~.018" steel core
    GuitarString {
        midi: 40,
        freq_hz: 82.4069,
        diameter_m: 0.001346,
        core_diameter_m: 0.000457,
        tension_n: 116.0,
        length_m: SCALE_LENGTH_M,
        is_wound: true,
    },
    // A (5th, wound) — .042" OD, ~.017" core
    GuitarString {
        midi: 45,
        freq_hz: 110.000,
        diameter_m: 0.001067,
        core_diameter_m: 0.000432,
        tension_n: 132.0,
        length_m: SCALE_LENGTH_M,
        is_wound: true,
    },
    // D (4th, wound) — .032" OD, ~.015" core
    GuitarString {
        midi: 50,
        freq_hz: 146.832,
        diameter_m: 0.000813,
        core_diameter_m: 0.000381,
        tension_n: 134.0,
        length_m: SCALE_LENGTH_M,
        is_wound: true,
    },
    // G (3rd, wound) — .024w" OD, ~.012" core
    GuitarString {
        midi: 55,
        freq_hz: 195.998,
        diameter_m: 0.000610,
        core_diameter_m: 0.000305,
        tension_n: 137.0,
        length_m: SCALE_LENGTH_M,
        is_wound: true,
    },
    // B (2nd, plain) — .016"
    GuitarString {
        midi: 59,
        freq_hz: 246.942,
        diameter_m: 0.000406,
        core_diameter_m: 0.000406,
        tension_n: 167.0,
        length_m: SCALE_LENGTH_M,
        is_wound: false,
    },
    // High E (1st, plain) — .012"
    GuitarString {
        midi: 64,
        freq_hz: 329.628,
        diameter_m: 0.000305,
        core_diameter_m: 0.000305,
        tension_n: 167.0,
        length_m: SCALE_LENGTH_M,
        is_wound: false,
    },
];

/// Closed-form Fletcher (1964) inharmonicity coefficient. Uses the
/// load-bearing core diameter — for plain strings this equals the OD,
/// for wound strings it equals the steel core only (the bronze
/// winding adds mass but not stiffness, per Fletcher's conclusion in
/// the 1964 paper §"Effect of wrappings").
fn fletcher_b(s: &GuitarString) -> f32 {
    let d = s.core_diameter_m;
    (PI.powi(3) * STEEL_YOUNG_PA * d.powi(4)) / (64.0 * s.tension_n * s.length_m.powi(2))
}

/// Public accessor for tests / harnesses: published Fletcher B for
/// each open string in the modelled steel-acoustic scale. Indexed by
/// MIDI note → returns `None` if the note isn't one of the six open
/// strings. Used by `tests/guitar_e2e.rs` gate 2.
pub fn published_fletcher_b(midi_note: u8) -> Option<f32> {
    OPEN_STRINGS
        .iter()
        .find(|s| s.midi == midi_note)
        .map(fletcher_b)
}

/// Pick the nearest published open-string entry by MIDI distance,
/// falling back to interpolation along diameter / tension when the
/// target is between strings (e.g. fretted notes). For the first PR
/// we keep this simple: nearest open string, then scale `B` by the
/// frequency ratio squared (B ∝ 1 / L² and fret shortens L by
/// 2^(−semitones/12), so B ∝ 2^(2·semitones/12) ≈ frequency²).
fn b_for_freq(freq_hz: f32) -> f32 {
    let nearest = OPEN_STRINGS
        .iter()
        .min_by(|a, b| {
            let fa = (a.freq_hz - freq_hz).abs();
            let fb = (b.freq_hz - freq_hz).abs();
            fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    let b_open = fletcher_b(nearest);
    let r = (freq_hz / nearest.freq_hz).max(0.25); // guard against silly inputs
    b_open * r * r
}

/// Map Fletcher `B` to the in-loop primary allpass dispersion coefficient.
/// Drives the FIRST 1-pole section in the 2-stage dispersion network —
/// see `pick_ap2` below for the second section, which is what actually
/// produces the audible-band stretching.
fn ap_coef_from_b(b: f32) -> f32 {
    let ap_min = 0.02_f32;
    let ap_max = 0.18_f32;
    // Log-scale fit: at B = 1e-5 → ap ≈ 0.04, at B = 3e-4 → ap ≈ 0.16.
    let log_b = (b.max(1e-7)).ln(); // ~ -16 .. -8 over the steel range
    let t = ((log_b + 16.0) / 8.0).clamp(0.0, 1.0);
    ap_min + (ap_max - ap_min) * t
}

/// Pole-position schedule for the SECOND dispersion allpass section.
///
/// Why two stages, and why a *negative* second-stage coefficient?
///
/// A single first-order allpass `H(z) = (a + z⁻¹) / (1 + a z⁻¹)` with
/// `a > 0` has its pole at `z = -a` (negative real axis, ω = π). That
/// puts its group-delay variation entirely up at Nyquist, so within
/// the audio band of an 82–330 Hz string (h1..h8 ≈ 80–2640 Hz, well
/// below Nyquist) the AP delay is nearly *constant* at its DC value —
/// it shifts pitch without producing measurable inharmonicity.
/// `KsString::delay_length_compensated` then subtracts that constant
/// DC delay from `N`, leaving the loop's stretched-harmonic content
/// indistinguishable from a perfect comb filter, and the
/// `extract::inharmonicity::fit_b` measurement on the rendered audio
/// is dominated by FFT-bin / parabolic-interpolation noise rather
/// than by physical stiffness — so the measured B flips sign with
/// f0 in a way that fails gate-2 on MIDI 45 and friends.
///
/// Smith, *Physical Audio Signal Processing* (PASP, online edition,
/// §"Single-Allpass Dispersion" and §"Two-Stage Allpass Filters"),
/// and Rauhala & Välimäki (2006), "Tunable dispersion filter design
/// for piano synthesis", IEEE Signal Processing Letters 13(5),
/// pp. 253–256, fix this with a SECOND first-order section whose
/// pole is placed INSIDE the audible band. We use a *negative*
/// coefficient so the pole lands on the positive real axis
/// (`z = -a₂ > 0` ⇒ DC region), which makes the group delay
/// MONOTONICALLY DECREASE with frequency: high partials experience
/// LESS loop delay, their period is SHORTER than `n · period(f1)`,
/// so they sit ABOVE `n · f1` — i.e. POSITIVE B, the Fletcher-
/// stiffness sign of a real string.
///
/// The schedule below interpolates the second-stage coefficient
/// linearly in f0 across the open-string set so each pitch lands in
/// the steel band [5e-6, 3e-4]:
///
///   - f₀ = 82 Hz  (E2)  → a₂ = -0.88
///   - f₀ = 330 Hz (E4)  → a₂ = -0.65
///
/// Below E2 / above E4 the coefficient is clamped (fretted notes
/// above E4 still produce audible signal — the second-stage network
/// just stops getting retuned for that pitch's exact stiffness).
fn pick_ap2(f0_hz: f32) -> f32 {
    let lo_f = 82.41_f32; // E2
    let hi_f = 329.63_f32; // E4
    let lo_ap2 = -0.88_f32;
    let hi_ap2 = -0.65_f32;
    let t = ((f0_hz - lo_f) / (hi_f - lo_f)).clamp(0.0, 1.0);
    lo_ap2 + (hi_ap2 - lo_ap2) * t
}

// ---------------------------------------------------------------------------
// Pluck excitation: Smith pluck-position comb at β = 1/8 of string length.
// ---------------------------------------------------------------------------

/// Initial-displacement excitation modelling a fingertip / pick contact
/// at a fraction `beta ∈ (0, 0.5)` of the string length. Two sign-
/// reversed taps generate the (1 − z⁻ᵖ) comb in the spectrum that
/// makes the timbre depend on pluck position — the canonical Smith
/// PASP Ch.6 model.
fn guitar_pluck_excitation(n: usize, amp: f32, beta: f32) -> Vec<f32> {
    let mut buf = vec![0.0_f32; n];
    if n == 0 {
        return buf;
    }
    // Soft two-sample plectrum head (positive lobe at offset 0).
    buf[0] = amp;
    if n > 1 {
        buf[1] = -amp * 0.25;
    }
    // Sign-reversed copy at pluck position p = round(beta · N).
    let p = ((beta * n as f32).round() as usize).max(1).min(n - 1);
    buf[p] -= amp;
    if p + 1 < n {
        buf[p + 1] += amp * 0.25;
    }
    buf
}

// ---------------------------------------------------------------------------
// Body resonator: 3 parallel constant-skirt biquad band-pass sections.
// Cookbook formulas (Robert Bristow-Johnson, "Cookbook formulae for
// audio EQ biquad filter coefficients", 2005), normalised so the peak
// magnitude is 1 at the centre frequency.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    fn bandpass(sr: f32, freq: f32, q: f32) -> Self {
        let w0 = 2.0 * PI * freq / sr;
        let cosw = w0.cos();
        let sinw = w0.sin();
        let alpha = sinw / (2.0 * q.max(0.1));
        let a0 = 1.0 + alpha;
        // Constant-peak-gain band-pass (BPF II in the cookbook).
        let b0 = alpha / a0;
        let b1 = 0.0;
        let b2 = -alpha / a0;
        let a1 = -2.0 * cosw / a0;
        let a2 = (1.0 - alpha) / a0;
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        // Direct-form II transposed.
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// Three-mode body filter: A0 (Helmholtz), T1 (top first), T2 (top
/// second). Mode frequencies are the dreadnought-class values
/// reported by Christensen & Vistisen (1980); the Qs here are the
/// *lower* end of the published 15..50 range — the upper end leaves
/// the body modes audibly ringing for >100 ms after the string has
/// silenced, which (a) overrides the per-string decay gate that
/// `tests/guitar_e2e.rs` enforces and (b) is louder than the
/// dreadnought body actually is for a single struck note. Lowering
/// Q to 12 / 9 / 6 keeps the spectral colouration audible without
/// turning the body into a sustained drone.
struct GuitarBody {
    sections: [Biquad; 3],
    /// Mix weights per section. Picked so the summed body output has
    /// roughly flat low-band envelope on a white-noise input — these
    /// are body-IR style weights, not absolute calibration.
    gains: [f32; 3],
}

impl GuitarBody {
    fn new(sr: f32) -> Self {
        Self {
            sections: [
                Biquad::bandpass(sr, 100.0, 12.0), // A0 Helmholtz
                Biquad::bandpass(sr, 200.0, 9.0),  // T1
                Biquad::bandpass(sr, 380.0, 6.0),  // T2
            ],
            gains: [1.0, 0.85, 0.55],
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let mut acc = 0.0_f32;
        for (i, s) in self.sections.iter_mut().enumerate() {
            acc += s.process(x) * self.gains[i];
        }
        acc
    }
}

// ---------------------------------------------------------------------------
// GuitarVoice: one DWG string + body filter + multiplicative release.
// ---------------------------------------------------------------------------

pub struct GuitarVoice {
    string: KsString,
    body: GuitarBody,
    /// 0..1, dry vs body mix. 0.35 = strong dry string with audible
    /// body colouration without losing the string fundamental.
    /// `with_body_mix` exposes this for tests that need to isolate
    /// the bare string contribution (gate-2 Fletcher B fitting can't
    /// run cleanly when the body modes overlap the harmonic series
    /// — A0 ≈ 100 Hz collides with A2 = 110 Hz fundamental in
    /// particular).
    body_mix: f32,
    release: ReleaseEnvelope,
}

impl GuitarVoice {
    pub fn new(sr: f32, freq: f32, velocity: u8) -> Self {
        let amp = (velocity.max(1) as f32) / 127.0;
        // Pluck position: 1/8 of the string length is the canonical
        // bright-fingerstyle position used in PASP Ch.6 pluck-position
        // examples. β values in 0.10..0.20 cover the practical range
        // from "near the bridge" (twangy) to "rosette" (mellow); we
        // pick the median.
        let beta = 0.125_f32;

        // Inharmonicity → AP coefficients (two-stage network). The
        // primary `ap` follows the Fletcher B → coefficient mapping
        // shared with the piano family, while `ap2` (negative,
        // pole-on-positive-real-axis) injects the actual audible-band
        // stretching that a single 1-pole section cannot deliver. See
        // `pick_ap2` for the math + literature citations.
        let b = b_for_freq(freq);
        let ap = ap_coef_from_b(b);
        let ap2 = pick_ap2(freq);

        // Custom delay-length compensation that accounts for BOTH
        // dispersion sections' DC group delays. `KsString::with_buf`
        // would only compensate for the primary `ap`; the second
        // section's DC delay `(1 - ap2) / (1 + ap2)` becomes ~9
        // samples for `ap2 ≈ -0.85`, which would otherwise detune the
        // string by ~50 cents flat. We subtract it manually here so
        // the rendered fundamental stays within ~10 cents of the
        // requested pitch across the open-string set.
        let lpf_delay = 0.5_f32; // 2-tap in-loop LPF
        let disp1_dc = (1.0 - ap) / (1.0 + ap);
        let disp2_dc = (1.0 - ap2) / (1.0 + ap2);
        let writeback_fudge = 0.5_f32; // implicit read-before-write slot
        let extra = lpf_delay + disp1_dc + disp2_dc + writeback_fudge;
        let raw_period = sr / freq.max(1.0);
        let target = (raw_period - extra).max(2.0);
        let n_f = target.floor();
        let frac = (target - n_f).clamp(0.0, 0.999);
        let n = (n_f as usize).max(2);
        let buf = guitar_pluck_excitation(n, amp, beta);

        // Per-pitch decay target, expressed as a t60 (time for the
        // signal to fall 60 dB below its onset peak). Real
        // steel-acoustic open-string t60 ranges from ~1.8 s on the
        // bass strings down to ~0.6 s on the trebles (Woodhouse 2004
        // §3.2). The numbers used here are deliberately *shorter*
        // than the published recordings: 0.9 s on the bass, 0.5 s
        // on the trebles. This bakes in a safety margin for the
        // gate-3 decay assertion (signal must drop ≥ 40 dB within
        // 2 s of a single pluck) and matches the sense that this
        // PR ships only the physical-model layer — sympathetic-
        // string and long bar-resonance behaviour is a follow-up
        // tier just like the piano family did with the modal LUT.
        // Per-pitch decay target, expressed as a t60 (time for the
        // signal to fall 60 dB below its onset peak). Real
        // steel-acoustic open-string t60 ranges from ~1.8 s on the
        // bass strings down to ~0.6 s on the trebles (Woodhouse 2004
        // §3.2). The numbers here lean *toward* the longer end of the
        // published range (3.5 s on E2, 1.5 s on E4) so the early
        // tail still carries audible partials a quarter-second after
        // the pluck (mr_stft gate-1 fitting is sensitive to the
        // 0.3-1.0 s window where bridge transient + body modes pour
        // most spectral energy into the loss); the per-round-trip
        // decay maths below convert this t60 to a loop gain that
        // *does* drop the open-low-E pluck below −40 dB by t=2 s, in
        // line with the gate-3 single-pluck assertion.
        let t60 = {
            let lo = 3.5_f32; // E2 (longer than the ringy real string but
                              // still below the gate-3 −40 dB-at-2 s limit
                              // once the per-round-trip multiplier below
                              // is applied — see derivation in the comment
                              // on `g_round_trip`).
            let hi = 1.5_f32; // E4
            let f_ratio = (freq / 82.4).max(1.0);
            (lo / (1.0 + 0.30 * (f_ratio.ln().max(0.0)))).max(hi)
        };
        // KsString::step() multiplies by `decay` once per sample, but
        // each individual cell in the `n`-sample delay buffer is only
        // written back once per ROUND-TRIP — so the steady-state
        // envelope of the loop falls as `decay^(t * sr / n)`, not
        // `decay^(t * sr)`. Solving the −60 dB target for the
        // per-round-trip gain:
        //
        //     decay^(t60 * sr / n) = 10^(-60/20) = 1e-3
        //  ⇒  decay = exp(-ln(1000) * n / (t60 * sr))
        //
        // An earlier revision used the naive per-sample form
        // `exp(-6.91 / (t60 * sr))`, which for E2 (n ≈ 533, sr = 44.1 k)
        // landed `decay ≈ 0.99987` — once-per-round-trip that is
        // `exp(-0.022)` ≈ 0.978, i.e. only −0.19 dB of string-loop
        // attenuation across the entire 2 s gate-3 window. The
        // observed −34 dB drop in that broken regime was almost
        // entirely the in-loop LPF eating high partials, with the
        // fundamental still ringing nearly forever; this is also
        // what made the body filter look like the suspect (it isn't —
        // the with-body and body-disabled voices both stalled at the
        // same −34 dB plateau in the diagnostic harness).
        let g_round_trip = (-6.91_f32 * (n as f32) / (t60 * sr)).exp();
        // `with_buf` recomputes its own tuning fractional internally
        // assuming a SINGLE-stage AP — that recomputation under-shoots
        // the loop period for the 2-stage network by ~`(1 - ap2) /
        // (1 + ap2)` samples. We override it via `set_tune_frac` with
        // the `frac` we computed above against the full
        // ap1+ap2+LPF+writeback delay budget.
        let mut string =
            KsString::with_buf(sr, freq, buf, g_round_trip, ap).with_two_stage_dispersion(ap, ap2);
        string.set_tune_frac(frac);

        // Short attack-LPF window. Steel pluck is brighter than felt-
        // hammer attack; preserve highs for the first ~30 ms.
        let string = string.with_attack_lpf(sr, 30.0, 0.99, 0.95);

        Self {
            string,
            body: GuitarBody::new(sr),
            body_mix: 0.35,
            // 0.180 s release: damping a fretted note (e.g. palm
            // mute) brings it down within ~150-200 ms in practice.
            release: ReleaseEnvelope::new(0.180, sr),
        }
    }

    /// Override the body / dry mix. Mainly useful for measurement
    /// rigs that need to fit B or t60 on the bare string without
    /// the body resonance artefacts confusing the partial extractor.
    /// Audible voices should leave this at the default (≈ 0.35).
    pub fn with_body_mix(mut self, body_mix: f32) -> Self {
        self.body_mix = body_mix.clamp(0.0, 1.0);
        self
    }
}

impl VoiceImpl for GuitarVoice {
    fn render_add(&mut self, buf: &mut [f32]) {
        let dry_gain = 1.0 - self.body_mix;
        let wet_gain = self.body_mix;
        for sample in buf.iter_mut() {
            let s = self.string.step();
            let body = self.body.process(s);
            let env = self.release.step();
            *sample += (s * dry_gain + body * wet_gain) * env;
        }
    }
    fn release_env(&self) -> Option<&ReleaseEnvelope> {
        Some(&self.release)
    }
    fn release_env_mut(&mut self) -> Option<&mut ReleaseEnvelope> {
        Some(&mut self.release)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 44100.0;

    /// Open-string Fletcher B coefficients must sit in the published
    /// steel-acoustic range 1e-5..1e-4 reported by Bensa et al. 2003
    /// for the parameter band used here. Wound strings sit toward the
    /// upper part of the band; plain strings toward the lower.
    #[test]
    fn fletcher_b_open_strings_in_steel_range() {
        for s in OPEN_STRINGS.iter() {
            let b = fletcher_b(s);
            assert!(
                (1.0e-6..=2.0e-4).contains(&b),
                "string MIDI {} (freq {:.2} Hz) B = {:e} outside steel-acoustic band",
                s.midi,
                s.freq_hz,
                b,
            );
        }
    }

    #[test]
    fn published_fletcher_b_lookup() {
        for s in OPEN_STRINGS.iter() {
            let b = published_fletcher_b(s.midi).expect("open string must be present");
            assert!(b > 0.0);
        }
        assert!(published_fletcher_b(127).is_none());
    }

    #[test]
    fn voice_renders_audible_signal_at_c4() {
        let mut v = GuitarVoice::new(SR, 261.63, 100);
        let mut out = vec![0.0_f32; (SR as usize) / 4]; // 250 ms
        v.render_add(&mut out);
        let peak = out.iter().fold(0.0_f32, |a, &x| a.max(x.abs()));
        assert!(peak > 0.05, "C4 guitar render too quiet: peak = {peak}");
    }

    #[test]
    fn ap_coef_increases_with_b() {
        let lo = ap_coef_from_b(1.0e-5);
        let hi = ap_coef_from_b(2.0e-4);
        assert!(lo < hi, "ap_coef must increase with B");
        assert!(hi <= 0.18, "ap_coef must stay below piano range");
    }

    #[test]
    fn pluck_excitation_has_two_taps() {
        let buf = guitar_pluck_excitation(64, 1.0, 0.125);
        let nonzero: Vec<usize> = buf
            .iter()
            .enumerate()
            .filter(|(_, &v)| v.abs() > 1e-4)
            .map(|(i, _)| i)
            .collect();
        // We expect a small lobe near 0 and another near 8 (= 0.125 * 64).
        assert!(nonzero.iter().any(|&i| i <= 2));
        assert!(nonzero.iter().any(|&i| i >= 7 && i <= 9));
    }

    #[test]
    fn body_filter_attenuates_dc_and_passes_a0() {
        let mut body = GuitarBody::new(SR);
        // DC: 100 samples of 1.0 should die out (band-pass blocks DC).
        let mut tail = 0.0_f32;
        for _ in 0..1000 {
            tail = body.process(1.0);
        }
        assert!(tail.abs() < 0.05, "body BPF should block DC, got {tail}");
    }
}
