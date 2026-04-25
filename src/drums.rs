//! Rhythm box / drum synth ported from `ai-chiptune-demo/render.py`.
//!
//! Three voices (NES drum-channel-style), all noise-based or single-
//! oscillator with quick envelopes. Each `synth_*` function writes a
//! short additive hit into the start of a mono buffer; the buffer is
//! the caller's responsibility to size (typical hit lasts 50-200 ms).
//!
//! Source pattern syntax (from chiptune lab): one character per
//! sixteenth-step. `K` = kick, `S` = snare, `H` = hi-hat, `.` = rest.

use std::f32::consts::PI;

/// Kick: sine-wave frequency sweep 150 Hz → 40 Hz (exp falloff with
/// rate 50/s) + amplitude envelope exp(-t*15). Phase-accumulated so
/// the swept frequency tracks correctly at any sample rate.
pub fn synth_kick(buf: &mut [f32], sr: f32, gain: f32) {
    let mut phase = 0.0_f32;
    for (i, sample) in buf.iter_mut().enumerate() {
        let t = i as f32 / sr;
        let f = 150.0 * (-t * 50.0).exp() + 40.0;
        phase += 2.0 * PI * f / sr;
        let env = (-t * 15.0).exp();
        *sample += phase.sin() * env * gain * 1.2;
    }
}

/// Snare: bandpass-filtered white noise (1-3 kHz) + 220 Hz square
/// buzz under a fast amp envelope. The filter is implemented as a
/// single-iteration 2-pole bandpass biquad (RBJ cookbook). Pseudo-
/// random 32-bit xorshift seeds the noise so successive calls don't
/// produce correlated noise patterns.
pub fn synth_snare(buf: &mut [f32], sr: f32, gain: f32) {
    let n = buf.len() as f32;
    // Bandpass biquad coefficients for fc = 1.7 kHz, Q = 1.0.
    let fc = 1700.0_f32;
    let q = 1.0_f32;
    let w0 = 2.0 * PI * fc / sr;
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();
    let b0 = alpha;
    let b2 = -alpha;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;
    let (b0, b2, a1, a2) = (b0 / a0, b2 / a0, a1 / a0, a2 / a0);

    let mut state: u32 = 0xCAFE_BABE_u32 ^ (n as u32).wrapping_mul(2_654_435_761);
    let mut x1 = 0.0_f32;
    let mut x2 = 0.0_f32;
    let mut y1 = 0.0_f32;
    let mut y2 = 0.0_f32;
    let mut sq_phase = 0.0_f32;

    for (i, sample) in buf.iter_mut().enumerate() {
        let t = i as f32 / sr;
        // White noise (xorshift32 normalised to ±1).
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let raw = (state as i32 as f32) / (i32::MAX as f32);
        // Bandpass filter the noise.
        let y = b0 * raw + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1;
        x1 = raw;
        y2 = y1;
        y1 = y;
        // 220 Hz square buzz (decays fast with rate 40).
        sq_phase += 2.0 * PI * 220.0 / sr;
        if sq_phase > 2.0 * PI {
            sq_phase -= 2.0 * PI;
        }
        let tone = if sq_phase < PI { 1.0 } else { -1.0 };
        let tone_env = (-t * 40.0).exp();
        let env = (-t * 12.0).exp();
        *sample += (y * 0.8 + tone * 0.4 * tone_env) * env * gain;
    }
}

/// Hi-hat: high-pass-filtered white noise (7 kHz cutoff) under a
/// very fast amp envelope. Single-pole HP is sufficient for the
/// short hit; full RBJ filter would be inaudible benefit.
pub fn synth_hihat(buf: &mut [f32], sr: f32, gain: f32) {
    let mut state: u32 = 0xDEAD_BEEF;
    // Single-pole HP at 7 kHz: y[n] = a * (y[n-1] + x[n] - x[n-1])
    let a = 1.0 - 2.0 * PI * 7000.0 / sr;
    let mut x_prev = 0.0_f32;
    let mut y_prev = 0.0_f32;
    for (i, sample) in buf.iter_mut().enumerate() {
        let t = i as f32 / sr;
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let raw = (state as i32 as f32) / (i32::MAX as f32);
        let y = a * (y_prev + raw - x_prev);
        x_prev = raw;
        y_prev = y;
        let env = (-t * 40.0).exp();
        *sample += y * env * gain * 0.3;
    }
}

/// One drum hit at a given start time + kind. `vel` 0..=127 mapped
/// linearly to gain 0..1.
#[derive(Clone, Copy, Debug)]
pub struct DrumEvent {
    pub start_sec: f32,
    pub kind: char,
    pub velocity: u8,
}

impl DrumEvent {
    pub fn duration_sec(&self) -> f32 {
        match self.kind {
            'K' => 0.180,
            'S' => 0.150,
            'H' => 0.060,
            _ => 0.0,
        }
    }

    pub fn render(&self, sr: f32, dst: &mut [f32], dst_offset: usize) {
        let n = (self.duration_sec() * sr) as usize;
        if n == 0 {
            return;
        }
        let end = (dst_offset + n).min(dst.len());
        if end <= dst_offset {
            return;
        }
        let segment = &mut dst[dst_offset..end];
        let gain = (self.velocity.max(1) as f32) / 127.0;
        match self.kind {
            'K' => synth_kick(segment, sr, gain),
            'S' => synth_snare(segment, sr, gain),
            'H' => synth_hihat(segment, sr, gain),
            _ => {}
        }
    }
}

/// Parse a chiptune-style drum string into DrumEvents. Each char
/// occupies one `step_sec`-long slot starting at `start_sec`.
/// Characters: 'K' (kick), 'S' (snare), 'H' (hi-hat), '.' (rest),
/// any other char also treated as rest.
pub fn parse_drum_pattern(
    pattern: &str,
    start_sec: f32,
    step_sec: f32,
    velocity: u8,
) -> Vec<DrumEvent> {
    let mut out = Vec::new();
    for (i, c) in pattern.chars().enumerate() {
        if matches!(c, 'K' | 'S' | 'H') {
            out.push(DrumEvent {
                start_sec: start_sec + (i as f32) * step_sec,
                kind: c,
                velocity,
            });
        }
    }
    out
}
