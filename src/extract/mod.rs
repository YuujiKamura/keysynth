//! Reference-waveform decomposition primitives (issue #3).
//!
//! Each submodule extracts one well-defined feature from a recorded note
//! (e.g. SFZ Salamander C4) so it can be (a) compared to the same feature
//! extracted from a keysynth-rendered candidate, (b) used as a loss
//! component in the `tuneloop` AI search, (c) frozen as a regression
//! anchor in `tests/extract_*.rs`.
//!
//! Every submodule MUST satisfy two test contracts:
//!   1. **Round-trip test** — synthesise a signal with known feature
//!      values, run the extractor, assert recovered values are within a
//!      stated tolerance. Locks in the extractor itself.
//!   2. **Golden test** — load `bench-out/REF_sfz_C4.wav`, run the
//!      extractor, assert pinned values that match the existing
//!      `analyse` output for that file. Locks in regression detection
//!      against future analysis-pipeline drift.
//!
//! Tests live alongside the modules in `#[cfg(test)] mod tests` blocks
//! AND under `tests/extract_<feature>.rs` for crate-level integration.

pub mod attack;
pub mod decompose;
pub mod inharmonicity;
pub mod t60;
