"""Side-by-side waveform comparison between two WAV files.

Generates a PNG with three rows for each WAV:
  - Time-domain waveform
  - RMS envelope (50 ms windows, dB FS)
  - Spectrogram (log magnitude, 0-8 kHz)

Used to visualise the difference between SFZ Salamander reference
renders and keysynth piano-modal candidate renders for issue #3.

Usage:
    python tools/compare_wavs.py REF.wav CAND.wav OUT.png [--label-ref X] [--label-cand Y]
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load a mono 16-bit WAV. Returns (samples in [-1, 1], sample_rate)."""
    with open(path, "rb") as f:
        d = f.read()
    if d[:4] != b"RIFF" or d[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    # fmt chunk
    fmt_idx = d.find(b"fmt ")
    fmt_size = struct.unpack("<I", d[fmt_idx + 4 : fmt_idx + 8])[0]
    audio_format, n_channels, sr, _byte_rate, _block_align, bits_per_sample = struct.unpack(
        "<HHIIHH", d[fmt_idx + 8 : fmt_idx + 8 + 16]
    )
    if audio_format != 1 or bits_per_sample != 16:
        raise ValueError(f"{path}: only PCM 16-bit supported (got fmt={audio_format} bits={bits_per_sample})")
    # data chunk
    data_idx = d.find(b"data", fmt_idx + 8 + fmt_size)
    n_bytes = struct.unpack("<I", d[data_idx + 4 : data_idx + 8])[0]
    raw = np.frombuffer(d[data_idx + 8 : data_idx + 8 + n_bytes], dtype="<i2")
    if n_channels > 1:
        raw = raw.reshape(-1, n_channels).mean(axis=1)
    samples = raw.astype(np.float32) / 32768.0
    return samples, sr


def rms_envelope_db(samples: np.ndarray, sr: int, window_ms: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """Sliding RMS envelope, returned in dB FS. Returns (time_axis, rms_db)."""
    win = int(sr * window_ms / 1000.0)
    if win < 1:
        win = 1
    n_chunks = len(samples) // win
    if n_chunks == 0:
        return np.array([]), np.array([])
    chunked = samples[: n_chunks * win].reshape(n_chunks, win)
    rms = np.sqrt(np.maximum(np.mean(chunked**2, axis=1), 1e-12))
    db = 20.0 * np.log10(rms)
    t = np.arange(n_chunks) * win / sr
    return t, db


def stft_db(samples: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the magnitude STFT in dB. Returns (time_axis, freq_axis, mag_db)."""
    if len(samples) < n_fft:
        return np.array([]), np.array([]), np.zeros((0, 0))
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / (n_fft - 1)))
    n_frames = 1 + (len(samples) - n_fft) // hop
    out = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for k in range(n_frames):
        start = k * hop
        frame = samples[start : start + n_fft] * window
        spec = np.fft.rfft(frame)
        out[:, k] = np.abs(spec)
    out_db = 20.0 * np.log10(np.maximum(out, 1e-9))
    t = np.arange(n_frames) * hop / sr
    f = np.arange(n_fft // 2 + 1) * sr / n_fft
    return t, f, out_db


def plot_compare(ref_path: Path, cand_path: Path, out_path: Path, label_ref: str, label_cand: str) -> None:
    ref, sr_r = read_wav_mono(ref_path)
    cand, sr_c = read_wav_mono(cand_path)
    if sr_r != sr_c:
        raise ValueError(f"sample-rate mismatch: ref={sr_r}, cand={sr_c}")
    sr = sr_r

    # Trim to shortest length so direct alignment works.
    n = min(len(ref), len(cand))
    ref, cand = ref[:n], cand[:n]

    # Compute analyses
    t_ref_rms, db_ref_rms = rms_envelope_db(ref, sr)
    t_cand_rms, db_cand_rms = rms_envelope_db(cand, sr)
    t_ref_st, f_ref_st, mag_ref_st = stft_db(ref, sr)
    t_cand_st, f_cand_st, mag_cand_st = stft_db(cand, sr)

    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex="col")
    fig.suptitle(f"WAV comparison: {label_ref} (left) vs {label_cand} (right)", fontsize=12)

    t_axis = np.arange(n) / sr

    # Row 1: waveform
    for ax, sig, label in [(axes[0, 0], ref, label_ref), (axes[0, 1], cand, label_cand)]:
        ax.plot(t_axis, sig, linewidth=0.4)
        ax.set_title(f"{label} — waveform")
        ax.set_ylabel("amplitude")
        ax.set_ylim(-1.0, 1.0)
        ax.grid(alpha=0.3)

    # Row 2: RMS envelope
    for ax, t, db, label in [
        (axes[1, 0], t_ref_rms, db_ref_rms, label_ref),
        (axes[1, 1], t_cand_rms, db_cand_rms, label_cand),
    ]:
        ax.plot(t, db, linewidth=1.0)
        ax.set_title(f"{label} — RMS envelope (50 ms windows)")
        ax.set_ylabel("dB FS")
        ax.set_ylim(-80, 0)
        ax.grid(alpha=0.3)

    # Row 3: spectrogram (clamp 0-8 kHz, dB-scaled)
    f_lim = 8000.0
    vmin, vmax = -80, 0
    for ax, t_st, f_st, mag, label in [
        (axes[2, 0], t_ref_st, f_ref_st, mag_ref_st, label_ref),
        (axes[2, 1], t_cand_st, f_cand_st, mag_cand_st, label_cand),
    ]:
        if mag.size == 0:
            ax.set_title(f"{label} — (signal too short for STFT)")
            continue
        # Normalise so 0 dB = peak across both files (consistent scale).
        peak = max(np.max(mag_ref_st), np.max(mag_cand_st))
        mag_norm = mag - peak  # dB relative to global peak
        f_mask = f_st <= f_lim
        ax.pcolormesh(
            t_st,
            f_st[f_mask],
            mag_norm[f_mask, :],
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            cmap="magma",
        )
        ax.set_title(f"{label} — spectrogram (dB rel. peak)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("freq (Hz)")
        ax.set_ylim(0, f_lim)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Side-by-side WAV comparison plot.")
    ap.add_argument("ref", type=Path, help="reference WAV path")
    ap.add_argument("cand", type=Path, help="candidate WAV path")
    ap.add_argument("out", type=Path, help="output PNG path")
    ap.add_argument("--label-ref", default="reference")
    ap.add_argument("--label-cand", default="candidate")
    args = ap.parse_args()

    plot_compare(args.ref, args.cand, args.out, args.label_ref, args.label_cand)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
