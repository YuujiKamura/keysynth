"""Extract per-note soundboard / body / room residual from SFZ Salamander
samples vs current modal voice renders (Smith commuted-synthesis-style).

For each LUT note we have:
  ref/note_NN.wav   = SFZ Salamander recording (10 s, stereo, normalised)
  cand/note_NN.wav  = current modal voice render of the same note (10 s)

The residual is what the SFZ recording carries that the modal partial
sum can't generate: hammer-felt contact noise, soundboard radiation
modes, body resonances, sympathetic strings, room reflections,
microphone characteristics. By extracting it once and adding it
back to the modal voice at note_on, we transplant the recorded
acoustic envelope onto the modal partial structure.

Pipeline per note:
  1. Load both wavs as float mono (downmix).
  2. Cross-correlate the first 100 ms to find the time offset and
     align modal to SFZ.
  3. RMS-match the first 200 ms (= attack region) so subtraction
     removes the partial portion proportionally.
  4. residual = sfz - scaled_modal (sample-by-sample).
  5. Trim to 0.5 s (transient + early decay portion is what carries
     the recorded character; later portion is partial-dominated and
     would just re-add modal partials).
  6. Apply a 5 ms half-cosine fade-out at the trim boundary so it
     doesn't click.
  7. Write as 16-bit mono WAV at 44.1 kHz.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from scipy.signal import correlate

LUT_NOTES = [36, 48, 60, 72, 84]
SR = 44100
TRIM_SEC = 0.05  # iter V': transient only (~50 ms). Polyphonic pieces
# (twinkle/bach) regressed +35-56 % residual_l2_aw at 0.5 s because
# every new note re-fired the full body+room tail and stacked on
# already-decaying voices. 50 ms keeps the hammer transient + initial
# body impulse (the "missing punch") and drops the sustained portion
# that would otherwise stack across polyphony.
FADE_SEC = 0.005
ALIGN_WINDOW_SEC = 0.1
RMS_MATCH_SEC = 0.2

REF_DIR = Path("bench-out/REF/sfz_salamander_multi")
CAND_DIR = Path("bench-out/MODAL_REF")
OUT_DIR = Path("bench-out/RESIDUAL")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with open(path, "rb") as f:
        d = f.read()
    if d[:4] != b"RIFF" or d[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE")
    fmt_idx = d.find(b"fmt ")
    fmt_size = struct.unpack("<I", d[fmt_idx + 4 : fmt_idx + 8])[0]
    audio_format, n_channels, sr, _byte_rate, _block_align, bits_per_sample = struct.unpack(
        "<HHIIHH", d[fmt_idx + 8 : fmt_idx + 8 + 16]
    )
    if audio_format != 1 or bits_per_sample != 16:
        raise ValueError(f"{path}: only PCM 16-bit supported")
    data_idx = d.find(b"data", fmt_idx + 8 + fmt_size)
    n_bytes = struct.unpack("<I", d[data_idx + 4 : data_idx + 8])[0]
    raw = np.frombuffer(d[data_idx + 8 : data_idx + 8 + n_bytes], dtype="<i2")
    if n_channels > 1:
        raw = raw.reshape(-1, n_channels).mean(axis=1)
    return raw.astype(np.float32) / 32768.0, sr


def write_wav_mono(path: Path, samples: np.ndarray, sr: int) -> None:
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2").tobytes()
    n_bytes = len(pcm)
    header = b"RIFF" + struct.pack("<I", 36 + n_bytes) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
    header += b"data" + struct.pack("<I", n_bytes)
    path.write_bytes(header + pcm)


def time_align(ref: np.ndarray, cand: np.ndarray, window_n: int) -> np.ndarray:
    """Shift cand so its first window_n samples best correlate with ref's."""
    a = ref[:window_n]
    b = cand[:window_n]
    corr = correlate(a, b, mode="full")
    lag = int(np.argmax(corr) - (len(b) - 1))
    if lag > 0:
        cand = np.concatenate([np.zeros(lag, dtype=cand.dtype), cand[:-lag]])
    elif lag < 0:
        cand = cand[-lag:]
        cand = np.concatenate([cand, np.zeros(-lag, dtype=cand.dtype)])
    return cand[: len(ref)]


def build_residual(midi_note: int) -> None:
    ref_path = REF_DIR / f"note_{midi_note}.wav"
    cand_path = CAND_DIR / f"note_{midi_note}.wav"
    out_path = OUT_DIR / f"residual_{midi_note}.wav"

    ref, sr_ref = read_wav_mono(ref_path)
    cand, sr_cand = read_wav_mono(cand_path)
    if sr_ref != SR or sr_cand != SR:
        raise ValueError(f"sr mismatch for midi {midi_note}: ref={sr_ref} cand={sr_cand}")

    n = min(len(ref), len(cand))
    ref = ref[:n]
    cand = cand[:n]

    # Time-align cand to ref using cross-correlation over the first 100 ms.
    align_n = int(ALIGN_WINDOW_SEC * SR)
    cand = time_align(ref, cand, align_n)

    # RMS-match cand to ref over the first 200 ms (attack region).
    match_n = int(RMS_MATCH_SEC * SR)
    ref_rms = float(np.sqrt(np.mean(ref[:match_n] ** 2)))
    cand_rms = float(np.sqrt(np.mean(cand[:match_n] ** 2)))
    scale = ref_rms / max(cand_rms, 1e-12)
    cand_scaled = cand * scale

    # Residual = ref - scaled cand.
    residual = ref - cand_scaled

    # Trim to TRIM_SEC and apply fade-out at the boundary.
    trim_n = int(TRIM_SEC * SR)
    residual = residual[:trim_n]
    fade_n = int(FADE_SEC * SR)
    if len(residual) > fade_n:
        fade = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_n)))
        residual[-fade_n:] *= fade

    write_wav_mono(out_path, residual, SR)
    res_peak = float(np.max(np.abs(residual)))
    res_rms = float(np.sqrt(np.mean(residual ** 2)))
    res_db = 20.0 * np.log10(max(res_rms, 1e-12))
    print(
        f"  midi {midi_note:>2}: align_lag-handled, scale={scale:.3f}, "
        f"residual peak={res_peak:.3f} rms={res_db:+.1f} dB → {out_path.name}"
    )


def main() -> None:
    print(f"build_residual_ir: {len(LUT_NOTES)} notes, trim={TRIM_SEC}s")
    for n in LUT_NOTES:
        build_residual(n)
    print(f"wrote residuals to {OUT_DIR}/")


if __name__ == "__main__":
    main()
