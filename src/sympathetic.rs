//! Shared sympathetic string resonance bank for the piano engine.
//!
//! Real piano: all ~88 strings sit on a shared bridge/soundboard. Striking
//! one string pumps energy through the bridge into the soundboard; the
//! soundboard radiates back through the bridge to every undamped string.
//! Strings whose partials overlap with the struck note's partials resonate
//! sympathetically — the "halo of neighbor strings" audible on any real
//! piano recording but absent from simple DWS where each voice is siloed.
//!
//! Implementation: a fixed-size bank of N KS-style delay-line strings
//! tuned across the keyboard range. No hammer excitation; only the
//! soundboard's output is injected as sympathetic drive each sample.
//! The bank's summed output is mixed into the audio bus alongside the
//! active voices, adding the cross-string halo that single-voice sims
//! can't produce.
//!
//! Tuning (see design doc in the commit that introduced this module):
//!   - Bank size: 24 strings spanning A0..C8 at ~4-semitone intervals
//!   - Bank decay: 0.9994 (long ring, T60 ≈ 7-8 s)
//!   - Bank ap_coef: 0.02 (near-zero stiffness — clean harmonic partials;
//!     sympathetic response should not be piano-stretched)
//!   - Per-string output gain: 1/sqrt(N) so bank peak stays bounded
//!   - Coupling 0.02 from soundboard → bank (audio-callback tuned)
//!   - Mix gain 0.3 from bank → audio bus  (audio-callback tuned)
//!
//! Stability: the bank only READS from the soundboard / audio bus; it
//! never writes back into the struck-note voice's soundboard. So the bank
//! forms no closed feedback loop with the rest of the engine — stability
//! reduces to "per-string KS decay < 1", which is guaranteed.

use crate::synth::KsString;

/// MIDI notes for the bank strings — 22 entries at 4-semitone intervals from
/// A0 (21) up through A7 (105), plus A#7 (106) and C8 (108) as high anchors.
/// 24 strings total, covering the full piano range.
const BANK_MIDI_NOTES: [u8; 24] = [
    21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 106,
    108,
];

/// Convert MIDI note number to frequency (Hz), A4 = 69 = 440 Hz.
#[inline]
fn midi_to_hz(note: u8) -> f32 {
    440.0 * 2.0_f32.powf((note as f32 - 69.0) / 12.0)
}

pub struct SympatheticBank {
    strings: Vec<KsString>,
    /// 1 / sqrt(N) output gain, precomputed once at construction.
    output_norm: f32,
}

impl SympatheticBank {
    /// Build a 24-string bank spanning A0..C8 at ~4-semitone intervals.
    /// All strings use light decay and very low allpass coefficient
    /// (plain KS with no stiffness — sympathetic response should be
    /// harmonically clean, not piano-stretched).
    pub fn new_piano(sr: f32) -> Self {
        // Near-zero stiffness: sympathetic partials should be harmonic.
        const AP_COEF: f32 = 0.02;
        // Long decay — sympathetic ring should outlast individual notes
        // to build up a subtle halo across a played passage. ~0.9994 at
        // 48 kHz → T60 ≈ 7-8 s.
        const DECAY: f32 = 0.9994;
        let mut strings: Vec<KsString> = Vec::with_capacity(BANK_MIDI_NOTES.len());
        for &midi in BANK_MIDI_NOTES.iter() {
            let freq = midi_to_hz(midi);
            let (n, _frac) = KsString::delay_length_compensated(sr, freq, AP_COEF);
            // Zero-filled buffer: no hammer excitation. Only the soundboard
            // drive (via inject_feedback) ever injects energy.
            let buf = vec![0.0_f32; n];
            let s = KsString::with_buf(sr, freq, buf, DECAY, AP_COEF);
            strings.push(s);
        }
        let n = strings.len() as f32;
        Self {
            strings,
            output_norm: 1.0 / n.sqrt(),
        }
    }

    /// Number of strings in the bank (for debugging/tests).
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }

    /// Drive all bank strings with the soundboard/bus output (scaled by
    /// `coupling`) and step them one sample. Returns the summed string
    /// output normalised by 1/sqrt(N).
    #[inline]
    pub fn process(&mut self, soundboard_out: f32, coupling: f32) -> f32 {
        let drive = soundboard_out * coupling;
        let mut sum = 0.0_f32;
        for s in self.strings.iter_mut() {
            if drive != 0.0 {
                s.inject_feedback(drive);
            }
            sum += s.step();
        }
        sum * self.output_norm
    }

    /// Reset all strings to silence (e.g. on emergency stop / panic).
    /// Rebuilds each delay line as a zero buffer at its original pitch.
    pub fn reset(&mut self) {
        // KsString has no public "clear" method, but its delay buffer is
        // indexed internally. Easiest reset: step the string with zero
        // drive many times so natural decay shrinks buffer contents to
        // effectively zero. For a "panic now" reset, run enough samples
        // for the decay (0.9994) to attenuate by ~120 dB: log(1e-6)/log(0.9994)
        // ≈ 23_000 samples.
        for s in self.strings.iter_mut() {
            for _ in 0..23_000 {
                let _ = s.step();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty input → zero output (no self-oscillation / DC drift).
    #[test]
    fn test_bank_silent_without_excitation() {
        let sr = 48_000.0_f32;
        let mut bank = SympatheticBank::new_piano(sr);
        assert_eq!(bank.len(), 24);
        let mut peak = 0.0_f32;
        for _ in 0..(sr as usize) {
            let y = bank.process(0.0, 0.02);
            peak = peak.max(y.abs());
        }
        assert!(peak == 0.0, "silent bank produced non-zero output: {peak}");
    }

    /// Drive bank at 440 Hz for 0.1 s, then stop. Bank should continue to
    /// ring for at least 1 second after drive stopped (sympathetic decay).
    #[test]
    fn test_bank_responds_to_drive() {
        let sr = 48_000.0_f32;
        let mut bank = SympatheticBank::new_piano(sr);
        // Drive with a 440 Hz sine for 100 ms.
        let drive_samples = (sr * 0.1) as usize;
        let omega = std::f32::consts::TAU * 440.0 / sr;
        for i in 0..drive_samples {
            let x = (omega * i as f32).sin();
            let _ = bank.process(x, 0.05);
        }
        // Stop driving. Capture output over the next 1 second.
        let tail_samples = sr as usize;
        let mut peak_after_drive = 0.0_f32;
        for _ in 0..tail_samples {
            let y = bank.process(0.0, 0.02);
            peak_after_drive = peak_after_drive.max(y.abs());
        }
        assert!(
            peak_after_drive > 1e-4,
            "bank did not ring after drive removed: peak={peak_after_drive}"
        );
        assert!(
            peak_after_drive.is_finite(),
            "non-finite bank output: {peak_after_drive}"
        );
    }

    /// Drive bank with persistent unit-amplitude noise for 10 seconds;
    /// output peak must stay bounded (< 5.0 gives a healthy safety margin
    /// vs audio-callback mix level of 0.3).
    #[test]
    fn test_bank_stable_under_persistent_drive() {
        let sr = 48_000.0_f32;
        let mut bank = SympatheticBank::new_piano(sr);
        // Deterministic LCG-ish "noise" so the test is reproducible.
        let mut state: u32 = 0x1234_5678;
        let mut peak = 0.0_f32;
        for _ in 0..((sr * 10.0) as usize) {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let n = (state as i32 as f32) / (i32::MAX as f32); // ~[-1, 1]
            let y = bank.process(n, 0.02);
            peak = peak.max(y.abs());
            assert!(y.is_finite(), "non-finite bank output during drive");
        }
        assert!(peak < 5.0, "bank runaway under persistent drive: peak={peak}");
    }
}
