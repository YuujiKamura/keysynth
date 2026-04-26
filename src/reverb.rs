//! Convolution reverb / "body IR" stage.
//!
//! Karplus-Strong and friends model the *string* but ignore the resonant
//! body (soundboard, cabinet, hammer-thump cavity). Real acoustic
//! instruments owe a huge fraction of their identity to that cavity. This
//! module convolves the dry voice mix with a short impulse response (IR)
//! to graft on the missing "wooden box" character.
//!
//! Two IR sources are supported:
//!   - `synthetic_body_ir` — Schroeder-style synthetic piano body IR
//!     (5 early reflections + ~100 ms exponential noise tail).
//!   - `load_ir_wav`       — load a real WAV impulse response (mono or
//!     downmixed stereo, 16/24/32-bit Int or Float).
//!
//! Convolution is naive direct (O(N*M) per buffer). IRs are capped at
//! ~150 ms (~6600 samples @ 44.1 kHz) so the per-callback cost stays
//! bounded and we don't need an FFT path.

#[cfg(feature = "native")]
use std::path::Path;

/// Hard cap on IR length in samples. ~150 ms at 44.1 kHz; the tail of a
/// piano body IR is well under this and direct convolution stays cheap.
pub const MAX_IR_SAMPLES: usize = 6600;

/// In-place wet/dry convolution reverb.
///
/// The IR is stored as a flat `Vec<f32>` (taps[0] = direct hit, taps[k] =
/// k-sample-delayed echo). `history` is a circular buffer of recent dry
/// input samples sized to IR length so we can read N taps back without
/// allocating each callback.
pub struct Reverb {
    ir: Vec<f32>,
    history: Vec<f32>,
    head: usize,
}

impl Reverb {
    /// Build a reverb from the given IR. IR is truncated at `MAX_IR_SAMPLES`.
    /// An empty IR yields a no-op reverb (`process` becomes pass-through).
    pub fn new(mut ir: Vec<f32>) -> Self {
        if ir.len() > MAX_IR_SAMPLES {
            ir.truncate(MAX_IR_SAMPLES);
        }
        let n = ir.len().max(1);
        Self {
            ir,
            history: vec![0.0_f32; n],
            head: 0,
        }
    }

    /// Number of IR taps actually in use (post-truncation).
    pub fn ir_len(&self) -> usize {
        self.ir.len()
    }

    /// In-place wet/dry mix. `wet` in 0..=1: 0 = dry only (no work), 1 =
    /// wet only. Anything else linearly mixes: `out = (1-wet)*dry + wet*conv`.
    /// Pre-allocated; no heap traffic in the audio thread.
    pub fn process(&mut self, samples: &mut [f32], wet: f32) {
        if wet <= 0.0 || self.ir.is_empty() {
            return;
        }
        let wet = wet.clamp(0.0, 1.0);
        let dry = 1.0 - wet;
        let n = self.history.len();
        let m = self.ir.len();
        for s in samples.iter_mut() {
            // Push current input into the circular history.
            self.history[self.head] = *s;
            // Convolve: y[t] = sum_k ir[k] * x[t - k].
            // x[t-k] lives at history[(head + n - k) mod n] given the
            // post-write head. We just wrote x[t] at `head`, so k=0 reads
            // from `head` itself; k=1 from one slot back; etc.
            let mut acc = 0.0_f32;
            let mut idx = self.head;
            for k in 0..m {
                acc += self.ir[k] * self.history[idx];
                idx = if idx == 0 { n - 1 } else { idx - 1 };
            }
            *s = dry * *s + wet * acc;
            self.head += 1;
            if self.head >= n {
                self.head = 0;
            }
        }
    }
}

/// Schroeder-style synthetic piano body IR.
///
/// Components:
///   - t=0: direct hit (1.0)
///   - 5 early reflections at 1/3/7/13/22 ms with alternating signs
///     [+0.45, -0.30, +0.22, -0.15, +0.10] -- mimics asymmetric cabinet
///     reflections off a wooden box.
///   - ~100 ms exponentially-decaying white-noise tail (`exp(-25*t)`)
///     scaled to roughly the same energy band as the early reflections.
///
/// Output is normalised so peak magnitude is approximately 1.0.
pub fn synthetic_body_ir(sr: u32) -> Vec<f32> {
    let sr_f = sr as f32;
    let tail_sec = 0.100_f32;
    let len = ((sr_f * tail_sec) as usize).min(MAX_IR_SAMPLES).max(1);
    let mut ir = vec![0.0_f32; len];

    // Direct hit
    ir[0] = 1.0;

    // Early reflections at fixed millisecond offsets with alternating signs.
    let early: &[(f32, f32)] = &[
        (0.001, 0.45),
        (0.003, -0.30),
        (0.007, 0.22),
        (0.013, -0.15),
        (0.022, 0.10),
    ];
    for &(t_sec, amp) in early {
        let idx = (t_sec * sr_f) as usize;
        if idx < len {
            ir[idx] += amp;
        }
    }

    // Exponentially-decaying noise tail. LCG keeps this dependency-free
    // and reproducible (same SR -> same IR).
    let mut state: u32 = 0x12345678_u32 ^ sr;
    let tail_amp = 0.25_f32;
    for i in 0..len {
        // xorshift32
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let n = (state as f32 / u32::MAX as f32) * 2.0 - 1.0; // [-1, 1)
        let t = i as f32 / sr_f;
        let env = (-25.0_f32 * t).exp();
        ir[i] += n * env * tail_amp;
    }

    // Normalise to peak ~= 1.0 so wet/dry math stays sane.
    let peak = ir.iter().fold(0.0_f32, |a, &x| a.max(x.abs()));
    if peak > 1e-6 {
        let scale = 1.0 / peak;
        for s in ir.iter_mut() {
            *s *= scale;
        }
    }

    ir
}

/// Load an impulse response from a WAV file. Accepts mono or stereo,
/// 16/24/32-bit Int or 32-bit Float. Stereo is downmixed to mono (L+R)/2.
/// Result is normalised to peak ~1.0 and truncated to `MAX_IR_SAMPLES`.
#[cfg(feature = "native")]
pub fn load_ir_wav(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader =
        hound::WavReader::open(path).map_err(|e| format!("opening IR WAV {:?}: {e}", path))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;

    // Pull samples as f32 regardless of underlying format. hound's typed
    // iterators dispatch on (sample_format, bits_per_sample); we cover the
    // common combinations and bail with a useful error on anything else.
    let raw: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("reading f32 samples: {e}"))?,
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|r| r.map(|s| s as f32 / i16::MAX as f32))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("reading i16 samples: {e}"))?,
        (hound::SampleFormat::Int, 24) => reader
            .samples::<i32>()
            .map(|r| r.map(|s| s as f32 / 8_388_608.0))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("reading i24 samples: {e}"))?,
        (hound::SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .map(|r| r.map(|s| s as f32 / i32::MAX as f32))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("reading i32 samples: {e}"))?,
        (fmt, bits) => {
            return Err(format!("unsupported WAV format: {fmt:?} {bits}-bit").into());
        }
    };

    // Downmix to mono.
    let mut mono: Vec<f32> = if channels <= 1 {
        raw
    } else {
        let frames = raw.len() / channels;
        let mut out = Vec::with_capacity(frames);
        for f in 0..frames {
            let mut acc = 0.0_f32;
            for c in 0..channels {
                acc += raw[f * channels + c];
            }
            out.push(acc / channels as f32);
        }
        out
    };

    if mono.len() > MAX_IR_SAMPLES {
        mono.truncate(MAX_IR_SAMPLES);
    }

    let peak = mono.iter().fold(0.0_f32, |a, &x| a.max(x.abs()));
    if peak > 1e-6 {
        let scale = 1.0 / peak;
        for s in mono.iter_mut() {
            *s *= scale;
        }
    }

    Ok(mono)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_ir_no_op() {
        let mut r = Reverb::new(vec![]);
        assert_eq!(r.ir_len(), 0);
        let mut buf = vec![1.0_f32; 16];
        r.process(&mut buf, 0.5);
        for v in &buf {
            assert_eq!(*v, 1.0, "empty IR must not modify the buffer");
        }
    }

    #[test]
    fn wet_zero_passthrough() {
        let ir = vec![0.5_f32, 0.25, 0.125];
        let mut r = Reverb::new(ir);
        let mut buf = vec![0.7_f32; 8];
        r.process(&mut buf, 0.0);
        for v in &buf {
            assert_eq!(*v, 0.7, "wet=0 must be a passthrough");
        }
    }

    #[test]
    fn impulse_ir_is_passthrough_at_wet_one() {
        // IR = [1.0]: convolution = identity. wet=1.0 should reproduce input.
        let ir = vec![1.0_f32];
        let mut r = Reverb::new(ir);
        let mut buf = vec![0.3_f32, 0.7, -0.5, 1.0];
        let original = buf.clone();
        r.process(&mut buf, 1.0);
        for (a, b) in buf.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6, "expected passthrough, got {a} vs {b}");
        }
    }

    #[test]
    fn ir_truncated_to_max_samples() {
        let big = vec![0.1_f32; MAX_IR_SAMPLES + 100];
        let r = Reverb::new(big);
        assert_eq!(r.ir_len(), MAX_IR_SAMPLES);
    }

    #[test]
    fn ir_len_reports_actual_taps() {
        let r = Reverb::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(r.ir_len(), 3);
    }

    #[test]
    fn synthetic_body_ir_nonempty() {
        let ir = synthetic_body_ir(44_100);
        assert!(!ir.is_empty());
        // Length is 100 ms = 4410 samples (capped to MAX_IR_SAMPLES).
        assert!(ir.len() <= MAX_IR_SAMPLES);
    }

    #[test]
    fn synthetic_body_ir_normalised_to_unit_peak() {
        let ir = synthetic_body_ir(44_100);
        let peak = ir.iter().fold(0.0_f32, |a, &x| a.max(x.abs()));
        assert!(peak > 0.0);
        assert!((peak - 1.0).abs() < 1e-3, "peak should be ~1.0, got {peak}");
    }

    #[test]
    fn synthetic_body_ir_starts_with_direct_hit() {
        let ir = synthetic_body_ir(44_100);
        // Direct hit at t=0 dominates (was 1.0 pre-normalisation, plus
        // a small noise sample). It should be one of the largest values.
        assert!(ir[0].abs() > 0.5);
    }

    #[test]
    fn synthetic_body_ir_reproducible_for_same_sr() {
        let a = synthetic_body_ir(44_100);
        let b = synthetic_body_ir(44_100);
        assert_eq!(a, b);
    }

    #[test]
    fn reverb_zero_input_zero_output() {
        let ir = vec![0.5_f32, 0.25];
        let mut r = Reverb::new(ir);
        let mut buf = vec![0.0_f32; 64];
        r.process(&mut buf, 0.7);
        for v in &buf {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn reverb_finite_for_random_input() {
        let ir = synthetic_body_ir(44_100);
        let mut r = Reverb::new(ir);
        let mut state: u32 = 0xC0FFEE;
        let mut buf = vec![0.0_f32; 4096];
        for s in buf.iter_mut() {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            *s = (state as i32 as f32) / (i32::MAX as f32);
        }
        r.process(&mut buf, 0.5);
        for v in &buf {
            assert!(v.is_finite());
        }
    }
}
