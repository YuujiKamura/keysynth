//! SoundFont / SFZ placeholder voice.
//!
//! The `rustysynth::Synthesizer` is a shared, mutable engine that handles all
//! channels at once -- it is NOT one-instance-per-voice. We keep the voice
//! pool's per-note bookkeeping (eviction, note-off matching) by pushing a
//! silent placeholder voice for each SfPiano note_on. Audio is rendered by
//! the audio callback calling `synth.render(...)` once per buffer and mixing
//! the result onto the voice bus.
//!
//! `is_done` returns true a few seconds after release so the placeholder is
//! eventually reaped; the underlying synth voice may continue to ring out
//! inside rustysynth's own envelope, which is fine -- the synth tracks its
//! own active voices independently.

use super::super::synth::{ReleaseEnvelope, VoiceImpl};

pub struct SfPianoPlaceholder {
    /// Adopted ReleaseEnvelope so this placeholder participates in the
    /// shared eviction discipline (is_done / is_releasing). The release
    /// time is 3 s, matching the previous hard-coded 132_300-sample
    /// post-release window at 44.1 kHz; with done_threshold lowered to
    /// 1e-6 the envelope stays well below 1e-3 across the window so
    /// is_done() fires at roughly the same wall-clock instant.
    release: ReleaseEnvelope,
}

impl SfPianoPlaceholder {
    pub fn new(sr: f32) -> Self {
        // 3 s release × done_threshold 1e-3 reproduces the historical
        // ~3 s eviction window (which was sample-counted as 132_300 @
        // 44.1 kHz). The ReleaseEnvelope step is sized so rel_mul hits
        // 1e-3 exactly at `release_sec` post-trigger, matching the
        // original sample-counted semantics. Centralising via
        // ReleaseEnvelope means the placeholder now reports
        // `is_releasing()` like every other voice, fixing the
        // eviction-policy bug where the pool couldn't see SfPiano
        // placeholders as candidates for eviction.
        Self {
            release: ReleaseEnvelope::with_done_threshold(3.0, sr, 1e-3),
        }
    }
}

impl Default for SfPianoPlaceholder {
    fn default() -> Self {
        // 44.1 kHz default keeps the historical 3-s eviction window.
        Self::new(44_100.0)
    }
}

impl VoiceImpl for SfPianoPlaceholder {
    fn render_add(&mut self, buf: &mut [f32]) {
        // Audio for this engine is rendered by the shared Synthesizer in
        // the audio callback. We just step the release envelope so the
        // pool can eventually evict us via is_done().
        for _ in 0..buf.len() {
            let _ = self.release.step();
        }
    }
    fn release_env(&self) -> Option<&ReleaseEnvelope> {
        Some(&self.release)
    }
    fn release_env_mut(&mut self) -> Option<&mut ReleaseEnvelope> {
        Some(&mut self.release)
    }
}
