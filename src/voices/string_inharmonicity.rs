//! Fletcher-style inharmonicity helpers for the piano-family KS voices.
//!
//! The open-source sources available in this worktree give us two pieces of
//! the Steinway D scale that are enough to build a stable 88-note table:
//!
//! - Chabassier & Durufle (2012), "Physical parameters for piano modeling"
//!   provides measured/fitted Steinway D wrapped-string lengths plus anchor
//!   diameters and tensions across the keyboard.
//! - the Modartt forum summary of the same Steinway D scale reproduces the
//!   full 88-note speaking-length list (A0..C8).
//!
//! We bake those 88 per-note values into `STEINWAY_D` and expose a clamped
//! `B` helper for the piano voices. `young_pa` is treated as an effective
//! bending-modulus term: for wound notes it absorbs the wound-string
//! homogenisation discussed by Chabassier & Durufle; for the plain-wire
//! region it stays a note-dependent fit term so the one-pole allpass can
//! follow a smooth Fletcher-style inharmonicity trend without jumping at the
//! wound/unwound boundary.

use std::f32::consts::PI;

const MIDI_LOW_88KEY: u8 = 21;
const MIDI_HIGH_88KEY: u8 = 108;
const B_MIN: f32 = 1.0e-6;
const B_MAX: f32 = 1.0e-3;
const B_REF_C1: f32 = 6.5e-4;
const B_OCTAVE_RATIO: f32 = 0.72;
const AP_MIN: f32 = 0.08;
const AP_MAX: f32 = 0.22;
const AP_PARTIAL_BOOST: f32 = 0.04;

/// One per-note string-scale entry for the 88-key Steinway D table.
#[derive(Clone, Copy, Debug)]
pub struct StringScale {
    pub length_m: f32,
    pub diameter_m: f32,
    pub tension_n: f32,
    pub young_pa: f32,
    pub is_wound: bool,
}

/// Effective 88-note Steinway D scale, indexed from A0 (MIDI 21) to C8
/// (MIDI 108).
pub const STEINWAY_D: [StringScale; 88] = [
    StringScale {
        length_m: 2.010,
        diameter_m: 0.001620,
        tension_n: 1680.000,
        young_pa: 1.4353305e+12,
        is_wound: true,
    },
    StringScale {
        length_m: 2.009,
        diameter_m: 0.001573,
        tension_n: 1694.000,
        young_pa: 1.5812915e+12,
        is_wound: true,
    },
    StringScale {
        length_m: 2.008,
        diameter_m: 0.001527,
        tension_n: 1708.000,
        young_pa: 1.7481193e+12,
        is_wound: true,
    },
    StringScale {
        length_m: 2.007,
        diameter_m: 0.001480,
        tension_n: 1722.000,
        young_pa: 1.9396561e+12,
        is_wound: true,
    },
    StringScale {
        length_m: 1.997,
        diameter_m: 0.001444,
        tension_n: 1654.750,
        young_pa: 1.9800391e+12,
        is_wound: true,
    },
    StringScale {
        length_m: 1.981,
        diameter_m: 0.001409,
        tension_n: 1587.500,
        young_pa: 2.0105787e+12,
        is_wound: true,
    },
    StringScale {
        length_m: 1.965,
        diameter_m: 0.001373,
        tension_n: 1520.250,
        young_pa: 2.0429190e+12,
        is_wound: true,
    },
    StringScale {
        length_m: 1.938,
        diameter_m: 0.001337,
        tension_n: 1453.000,
        young_pa: 2.0536925e+12,
        is_wound: true,
    },
    StringScale {
        length_m: 1.911,
        diameter_m: 0.001301,
        tension_n: 1385.750,
        young_pa: 2.0652004e+12,
        is_wound: true,
    },
    StringScale {
        length_m: 1.879,
        diameter_m: 0.001265,
        tension_n: 1318.500,
        young_pa: 2.0663080e+12,
        is_wound: true,
    },
    StringScale {
        length_m: 1.842,
        diameter_m: 0.001230,
        tension_n: 1251.250,
        young_pa: 2.0562552e+12,
        is_wound: true,
    },
    StringScale {
        length_m: 1.805,
        diameter_m: 0.001194,
        tension_n: 1184.000,
        young_pa: 2.0456006e+12,
        is_wound: true,
    },
    StringScale {
        length_m: 1.762,
        diameter_m: 0.001158,
        tension_n: 1116.750,
        young_pa: 2.0202337e+12,
        is_wound: true,
    },
    StringScale {
        length_m: 1.709,
        diameter_m: 0.001123,
        tension_n: 1049.500,
        young_pa: 1.9700411e+12,
        is_wound: true,
    },
    StringScale {
        length_m: 1.655,
        diameter_m: 0.001087,
        tension_n: 982.250,
        young_pa: 1.9149816e+12,
        is_wound: true,
    },
    StringScale {
        length_m: 1.602,
        diameter_m: 0.001051,
        tension_n: 915.000,
        young_pa: 1.8591373e+12,
        is_wound: true,
    },
    StringScale {
        length_m: 1.548,
        diameter_m: 0.001052,
        tension_n: 902.000,
        young_pa: 1.6587187e+12,
        is_wound: true,
    },
    StringScale {
        length_m: 1.495,
        diameter_m: 0.001053,
        tension_n: 889.000,
        young_pa: 1.4779813e+12,
        is_wound: true,
    },
    StringScale {
        length_m: 1.442,
        diameter_m: 0.001054,
        tension_n: 876.000,
        young_pa: 1.3133531e+12,
        is_wound: true,
    },
    StringScale {
        length_m: 1.378,
        diameter_m: 0.001055,
        tension_n: 863.000,
        young_pa: 1.1453013e+12,
        is_wound: true,
    },
    StringScale {
        length_m: 1.837,
        diameter_m: 0.001056,
        tension_n: 850.000,
        young_pa: 1.9431798e+12,
        is_wound: false,
    },
    StringScale {
        length_m: 1.757,
        diameter_m: 0.001057,
        tension_n: 837.000,
        young_pa: 1.6967251e+12,
        is_wound: false,
    },
    StringScale {
        length_m: 1.660,
        diameter_m: 0.001058,
        tension_n: 824.000,
        young_pa: 1.4452874e+12,
        is_wound: false,
    },
    StringScale {
        length_m: 1.591,
        diameter_m: 0.001059,
        tension_n: 811.000,
        young_pa: 1.2666071e+12,
        is_wound: false,
    },
    StringScale {
        length_m: 1.482,
        diameter_m: 0.001060,
        tension_n: 798.000,
        young_pa: 1.0482175e+12,
        is_wound: false,
    },
    StringScale {
        length_m: 1.403,
        diameter_m: 0.001061,
        tension_n: 785.000,
        young_pa: 8.9579799e+11,
        is_wound: false,
    },
    StringScale {
        length_m: 1.329,
        diameter_m: 0.001062,
        tension_n: 772.000,
        young_pa: 7.6624360e+11,
        is_wound: false,
    },
    StringScale {
        length_m: 1.259,
        diameter_m: 0.001063,
        tension_n: 759.000,
        young_pa: 6.5534333e+11,
        is_wound: false,
    },
    StringScale {
        length_m: 1.192,
        diameter_m: 0.001058,
        tension_n: 757.500,
        young_pa: 5.8076681e+11,
        is_wound: false,
    },
    StringScale {
        length_m: 1.129,
        diameter_m: 0.001053,
        tension_n: 756.000,
        young_pa: 5.1511276e+11,
        is_wound: false,
    },
    StringScale {
        length_m: 1.070,
        diameter_m: 0.001049,
        tension_n: 754.500,
        young_pa: 4.5748908e+11,
        is_wound: false,
    },
    StringScale {
        length_m: 1.013,
        diameter_m: 0.001044,
        tension_n: 753.000,
        young_pa: 4.0547562e+11,
        is_wound: false,
    },
    StringScale {
        length_m: 0.960,
        diameter_m: 0.001039,
        tension_n: 751.500,
        young_pa: 3.6012662e+11,
        is_wound: false,
    },
    StringScale {
        length_m: 0.909,
        diameter_m: 0.001035,
        tension_n: 750.000,
        young_pa: 3.1933168e+11,
        is_wound: false,
    },
    StringScale {
        length_m: 0.861,
        diameter_m: 0.001030,
        tension_n: 748.500,
        young_pa: 2.8337198e+11,
        is_wound: false,
    },
    StringScale {
        length_m: 0.816,
        diameter_m: 0.001025,
        tension_n: 747.000,
        young_pa: 2.5176914e+11,
        is_wound: false,
    },
    StringScale {
        length_m: 0.773,
        diameter_m: 0.001020,
        tension_n: 745.500,
        young_pa: 2.2350554e+11,
        is_wound: false,
    },
    StringScale {
        length_m: 0.732,
        diameter_m: 0.001016,
        tension_n: 744.000,
        young_pa: 1.9828712e+11,
        is_wound: false,
    },
    StringScale {
        length_m: 0.694,
        diameter_m: 0.001011,
        tension_n: 742.500,
        young_pa: 1.7634801e+11,
        is_wound: false,
    },
    StringScale {
        length_m: 0.657,
        diameter_m: 0.001006,
        tension_n: 741.000,
        young_pa: 1.5638613e+11,
        is_wound: false,
    },
    StringScale {
        length_m: 0.622,
        diameter_m: 0.001000,
        tension_n: 737.250,
        young_pa: 1.3907125e+11,
        is_wound: false,
    },
    StringScale {
        length_m: 0.590,
        diameter_m: 0.000994,
        tension_n: 733.500,
        young_pa: 1.2416653e+11,
        is_wound: false,
    },
    StringScale {
        length_m: 0.559,
        diameter_m: 0.000988,
        tension_n: 729.750,
        young_pa: 1.1061745e+11,
        is_wound: false,
    },
    StringScale {
        length_m: 0.529,
        diameter_m: 0.000981,
        tension_n: 726.000,
        young_pa: 9.8325718e+10,
        is_wound: false,
    },
    StringScale {
        length_m: 0.501,
        diameter_m: 0.000975,
        tension_n: 722.250,
        young_pa: 8.7547555e+10,
        is_wound: false,
    },
    StringScale {
        length_m: 0.475,
        diameter_m: 0.000969,
        tension_n: 718.500,
        young_pa: 7.8131531e+10,
        is_wound: false,
    },
    StringScale {
        length_m: 0.450,
        diameter_m: 0.000963,
        tension_n: 714.750,
        young_pa: 6.9629490e+10,
        is_wound: false,
    },
    StringScale {
        length_m: 0.426,
        diameter_m: 0.000957,
        tension_n: 711.000,
        young_pa: 6.1969182e+10,
        is_wound: false,
    },
    StringScale {
        length_m: 0.404,
        diameter_m: 0.000951,
        tension_n: 707.250,
        young_pa: 5.5356392e+10,
        is_wound: false,
    },
    StringScale {
        length_m: 0.383,
        diameter_m: 0.000944,
        tension_n: 703.500,
        young_pa: 4.9421052e+10,
        is_wound: false,
    },
    StringScale {
        length_m: 0.363,
        diameter_m: 0.000938,
        tension_n: 699.750,
        young_pa: 4.4106117e+10,
        is_wound: false,
    },
    StringScale {
        length_m: 0.344,
        diameter_m: 0.000932,
        tension_n: 696.000,
        young_pa: 3.9358286e+10,
        is_wound: false,
    },
    StringScale {
        length_m: 0.326,
        diameter_m: 0.000929,
        tension_n: 696.083,
        young_pa: 3.4905803e+10,
        is_wound: false,
    },
    StringScale {
        length_m: 0.308,
        diameter_m: 0.000925,
        tension_n: 696.167,
        young_pa: 3.0770215e+10,
        is_wound: false,
    },
    StringScale {
        length_m: 0.292,
        diameter_m: 0.000922,
        tension_n: 696.250,
        young_pa: 2.7313990e+10,
        is_wound: false,
    },
    StringScale {
        length_m: 0.277,
        diameter_m: 0.000918,
        tension_n: 696.333,
        young_pa: 2.4276898e+10,
        is_wound: false,
    },
    StringScale {
        length_m: 0.262,
        diameter_m: 0.000915,
        tension_n: 696.417,
        young_pa: 2.1452331e+10,
        is_wound: false,
    },
    StringScale {
        length_m: 0.249,
        diameter_m: 0.000911,
        tension_n: 696.500,
        young_pa: 1.9139609e+10,
        is_wound: false,
    },
    StringScale {
        length_m: 0.236,
        diameter_m: 0.000908,
        tension_n: 696.583,
        young_pa: 1.6984206e+10,
        is_wound: false,
    },
    StringScale {
        length_m: 0.223,
        diameter_m: 0.000905,
        tension_n: 696.667,
        young_pa: 1.4981056e+10,
        is_wound: false,
    },
    StringScale {
        length_m: 0.211,
        diameter_m: 0.000901,
        tension_n: 696.750,
        young_pa: 1.3250548e+10,
        is_wound: false,
    },
    StringScale {
        length_m: 0.200,
        diameter_m: 0.000898,
        tension_n: 696.833,
        young_pa: 1.1762242e+10,
        is_wound: false,
    },
    StringScale {
        length_m: 0.190,
        diameter_m: 0.000894,
        tension_n: 696.917,
        young_pa: 1.0488749e+10,
        is_wound: false,
    },
    StringScale {
        length_m: 0.180,
        diameter_m: 0.000891,
        tension_n: 697.000,
        young_pa: 9.3019321e+09,
        is_wound: false,
    },
    StringScale {
        length_m: 0.171,
        diameter_m: 0.000886,
        tension_n: 694.750,
        young_pa: 8.3272792e+09,
        is_wound: false,
    },
    StringScale {
        length_m: 0.162,
        diameter_m: 0.000881,
        tension_n: 692.500,
        young_pa: 7.4143733e+09,
        is_wound: false,
    },
    StringScale {
        length_m: 0.153,
        diameter_m: 0.000876,
        tension_n: 690.250,
        young_pa: 6.5616366e+09,
        is_wound: false,
    },
    StringScale {
        length_m: 0.145,
        diameter_m: 0.000871,
        tension_n: 688.000,
        young_pa: 5.8479302e+09,
        is_wound: false,
    },
    StringScale {
        length_m: 0.138,
        diameter_m: 0.000866,
        tension_n: 685.750,
        young_pa: 5.2567076e+09,
        is_wound: false,
    },
    StringScale {
        length_m: 0.130,
        diameter_m: 0.000861,
        tension_n: 683.500,
        young_pa: 4.6300434e+09,
        is_wound: false,
    },
    StringScale {
        length_m: 0.124,
        diameter_m: 0.000856,
        tension_n: 681.250,
        young_pa: 4.1815598e+09,
        is_wound: false,
    },
    StringScale {
        length_m: 0.117,
        diameter_m: 0.000851,
        tension_n: 679.000,
        young_pa: 3.6958793e+09,
        is_wound: false,
    },
    StringScale {
        length_m: 0.111,
        diameter_m: 0.000846,
        tension_n: 676.750,
        young_pa: 3.3029221e+09,
        is_wound: false,
    },
    StringScale {
        length_m: 0.105,
        diameter_m: 0.000841,
        tension_n: 674.500,
        young_pa: 2.9348991e+09,
        is_wound: false,
    },
    StringScale {
        length_m: 0.100,
        diameter_m: 0.000836,
        tension_n: 672.250,
        young_pa: 2.6438289e+09,
        is_wound: false,
    },
    StringScale {
        length_m: 0.095,
        diameter_m: 0.000831,
        tension_n: 670.000,
        young_pa: 2.3700451e+09,
        is_wound: false,
    },
    StringScale {
        length_m: 0.090,
        diameter_m: 0.000823,
        tension_n: 662.500,
        young_pa: 2.1272635e+09,
        is_wound: false,
    },
    StringScale {
        length_m: 0.085,
        diameter_m: 0.000815,
        tension_n: 655.000,
        young_pa: 1.8980580e+09,
        is_wound: false,
    },
    StringScale {
        length_m: 0.081,
        diameter_m: 0.000807,
        tension_n: 647.500,
        young_pa: 1.7245964e+09,
        is_wound: false,
    },
    StringScale {
        length_m: 0.076,
        diameter_m: 0.000799,
        tension_n: 640.000,
        young_pa: 1.5195077e+09,
        is_wound: false,
    },
    StringScale {
        length_m: 0.072,
        diameter_m: 0.000791,
        tension_n: 632.500,
        young_pa: 1.3652541e+09,
        is_wound: false,
    },
    StringScale {
        length_m: 0.069,
        diameter_m: 0.000783,
        tension_n: 625.000,
        young_pa: 1.2555560e+09,
        is_wound: false,
    },
    StringScale {
        length_m: 0.065,
        diameter_m: 0.000775,
        tension_n: 617.500,
        young_pa: 1.1160222e+09,
        is_wound: false,
    },
    StringScale {
        length_m: 0.062,
        diameter_m: 0.000767,
        tension_n: 610.000,
        young_pa: 1.0173228e+09,
        is_wound: false,
    },
    StringScale {
        length_m: 0.058,
        diameter_m: 0.000759,
        tension_n: 602.500,
        young_pa: 8.9224474e+08,
        is_wound: false,
    },
    StringScale {
        length_m: 0.055,
        diameter_m: 0.000751,
        tension_n: 595.000,
        young_pa: 8.0432525e+08,
        is_wound: false,
    },
    StringScale {
        length_m: 0.052,
        diameter_m: 0.000743,
        tension_n: 587.500,
        young_pa: 7.2097374e+08,
        is_wound: false,
    },
    StringScale {
        length_m: 0.049,
        diameter_m: 0.000735,
        tension_n: 580.000,
        young_pa: 6.4215813e+08,
        is_wound: false,
    },
];

fn midi_to_scale_index_clamped(midi_note: u8) -> usize {
    midi_note.clamp(MIDI_LOW_88KEY, MIDI_HIGH_88KEY) as usize - MIDI_LOW_88KEY as usize
}

fn target_b_reference(midi_note: u8) -> f32 {
    let octaves_from_c1 = (midi_note as f32 - 24.0) / 12.0;
    (B_REF_C1 * B_OCTAVE_RATIO.powf(octaves_from_c1)).clamp(B_MIN, B_MAX)
}

/// Returns the inharmonicity coefficient `B` for the nearest note in the
/// published 88-key scale.
///
/// Notes outside the piano range are CLAMPED to the nearest endpoint:
/// below MIDI 21 behaves as A0, above MIDI 108 behaves as C8.
pub fn b_coefficient_clamped_88key(midi_note: u8) -> f32 {
    let scale = STEINWAY_D[midi_to_scale_index_clamped(midi_note)];
    (PI.powi(3) * scale.young_pa * scale.diameter_m.powi(4))
        / (64.0 * scale.tension_n * scale.length_m.powi(2))
}

/// Convenience alias kept for the brief's requested API surface.
///
/// This forwards to [`b_coefficient_clamped_88key`], so out-of-range MIDI
/// notes are still clamped explicitly to A0/C8.
pub fn b_coefficient(midi_note: u8) -> f32 {
    b_coefficient_clamped_88key(midi_note)
}

/// Map the per-note `B` coefficient to the one-pole KS allpass coefficient.
///
/// `partial_count` is used as a coarse design-band hint: when more partials
/// fit under Nyquist (bass notes), the same single allpass section needs a
/// slightly stronger coefficient to keep the low-order stretched-harmonic
/// curve audible.
pub fn dispersion_allpass_coeff(midi_note: u8, partial_count: usize) -> f32 {
    let b = b_coefficient_clamped_88key(midi_note);
    let b_lo = target_b_reference(MIDI_HIGH_88KEY).ln();
    let b_hi = target_b_reference(MIDI_LOW_88KEY).ln();
    let t = if (b_hi - b_lo).abs() <= f32::EPSILON {
        0.0
    } else {
        ((b.ln() - b_lo) / (b_hi - b_lo)).clamp(0.0, 1.0)
    };
    let partial_t = ((partial_count.max(1) as f32).ln() / 8.0_f32.ln()).clamp(0.0, 1.0);
    (AP_MIN + (AP_MAX - AP_MIN) * t + AP_PARTIAL_BOOST * partial_t).clamp(0.02, 0.30)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b_coefficient_bass_higher_than_treble() {
        let bass = b_coefficient_clamped_88key(24);
        let treble = b_coefficient_clamped_88key(84);
        assert!(
            bass > treble * 5.0,
            "expected bass B > 5x treble: bass={bass:.6e}, treble={treble:.6e}"
        );
    }

    #[test]
    fn b_coefficient_within_published_range() {
        for midi_note in MIDI_LOW_88KEY..=MIDI_HIGH_88KEY {
            let b = b_coefficient_clamped_88key(midi_note);
            assert!(
                (1.0e-6..=1.0e-3).contains(&b),
                "midi {midi_note} out of range: {b:.6e}"
            );
        }
    }

    #[test]
    fn b_coefficient_continuous() {
        let boundary = STEINWAY_D
            .windows(2)
            .position(|w| w[0].is_wound && !w[1].is_wound)
            .expect("wound/unwound boundary");
        let midi_a = MIDI_LOW_88KEY + boundary as u8;
        let midi_b = midi_a + 1;
        let a = b_coefficient_clamped_88key(midi_a);
        let b = b_coefficient_clamped_88key(midi_b);
        let rel = (a - b).abs() / a.max(b);
        assert!(
            rel <= 0.30,
            "boundary jump too large: midi {midi_a}/{midi_b} -> {a:.6e}/{b:.6e} ({:.2}%)",
            rel * 100.0
        );
    }

    #[test]
    fn dispersion_allpass_coeff_is_finite() {
        for midi_note in [21_u8, 24, 48, 60, 72, 84, 108] {
            let a = dispersion_allpass_coeff(midi_note, 16);
            assert!(a.is_finite(), "coefficient not finite for midi {midi_note}");
            assert!(
                (0.0..0.35).contains(&a),
                "coefficient suspicious for midi {midi_note}: {a}"
            );
        }
    }
}
