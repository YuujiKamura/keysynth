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
    """Load a WAV (mono or stereo) and downmix to mono in [-1, 1]."""
    with open(path, "rb") as f:
        d = f.read()
    if d[:4] != b"RIFF" or d[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    fmt_idx = d.find(b"fmt ")
    fmt_size = struct.unpack("<I", d[fmt_idx + 4 : fmt_idx + 8])[0]
    audio_format, n_channels, sr, _byte_rate, _block_align, bits_per_sample = struct.unpack(
        "<HHIIHH", d[fmt_idx + 8 : fmt_idx + 8 + 16]
    )
    if audio_format != 1 or bits_per_sample != 16:
        raise ValueError(f"{path}: only PCM 16-bit supported (got fmt={audio_format} bits={bits_per_sample})")
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

    fig, axes = plt.subplots(4, 2, figsize=(16, 13))
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
    peak_global = max(np.max(mag_ref_st), np.max(mag_cand_st))
    for ax, t_st, f_st, mag, label in [
        (axes[2, 0], t_ref_st, f_ref_st, mag_ref_st, label_ref),
        (axes[2, 1], t_cand_st, f_cand_st, mag_cand_st, label_cand),
    ]:
        if mag.size == 0:
            ax.set_title(f"{label} — (signal too short for STFT)")
            continue
        mag_norm = mag - peak_global
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
        ax.set_ylabel("freq (Hz)")
        ax.set_ylim(0, f_lim)

    # Row 4: residual = (ref - cand) in dB. Red = ref louder than cand
    # (modal lacks this content). Blue = cand louder than ref (modal
    # over-shoots here). Centred at 0, clip ±25 dB.
    if mag_ref_st.size > 0 and mag_cand_st.size > 0:
        # Align dimensions (truncate to min in each axis)
        n_t = min(mag_ref_st.shape[1], mag_cand_st.shape[1])
        n_f = min(mag_ref_st.shape[0], mag_cand_st.shape[0])
        diff = mag_ref_st[:n_f, :n_t] - mag_cand_st[:n_f, :n_t]
        f_axis = f_ref_st[:n_f]
        t_axis_st = t_ref_st[:n_t]
        f_mask = f_axis <= f_lim
        residual_ax = axes[3, 0]
        im = residual_ax.pcolormesh(
            t_axis_st,
            f_axis[f_mask],
            diff[f_mask, :],
            vmin=-25,
            vmax=25,
            shading="auto",
            cmap="RdBu_r",
        )
        residual_ax.set_title(
            "spectral residual (ref - cand, dB) — red = cand DEFICIT, blue = cand SURPLUS"
        )
        residual_ax.set_xlabel("time (s)")
        residual_ax.set_ylabel("freq (Hz)")
        residual_ax.set_ylim(0, f_lim)
        plt.colorbar(im, ax=residual_ax, fraction=0.04, pad=0.02)

        # Per-frequency-band average residual: positive → cand deficit
        band_ax = axes[3, 1]
        avg = np.mean(diff[f_mask, :], axis=1)
        band_ax.plot(f_axis[f_mask], avg, linewidth=1.0)
        band_ax.axhline(0, color="black", linewidth=0.5)
        band_ax.fill_between(f_axis[f_mask], 0, avg, where=(avg > 0), alpha=0.3, color="red")
        band_ax.fill_between(f_axis[f_mask], 0, avg, where=(avg < 0), alpha=0.3, color="blue")
        band_ax.set_title("avg residual per frequency band (red = deficit)")
        band_ax.set_xlabel("freq (Hz)")
        band_ax.set_ylabel("ref - cand (dB)")
        band_ax.set_xlim(0, f_lim)
        band_ax.grid(alpha=0.3)
    else:
        for ax in axes[3]:
            ax.set_title("(no residual — signal too short)")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)

    # ----- Numeric residual summary (no listener loop, supervised metric)
    if mag_ref_st.size > 0 and mag_cand_st.size > 0:
        # Recompute STFT in LINEAR-magnitude domain, not dB. dB residual
        # is inflated by silent bins (log of near-zero blows up); linear
        # mag residual measures actual energy gap.
        n_fft = 2048
        hop = 512

        def stft_linear(samples: np.ndarray) -> np.ndarray:
            window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / (n_fft - 1)))
            n_frames = 1 + (len(samples) - n_fft) // hop
            out = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
            for k in range(n_frames):
                start = k * hop
                frame = samples[start : start + n_fft] * window
                out[:, k] = np.abs(np.fft.rfft(frame))
            return out

        ref_lin = stft_linear(ref)
        cand_lin = stft_linear(cand)
        n_t = min(ref_lin.shape[1], cand_lin.shape[1])
        n_f = min(ref_lin.shape[0], cand_lin.shape[0])
        ref_lin = ref_lin[:n_f, :n_t]
        cand_lin = cand_lin[:n_f, :n_t]
        f_axis = np.arange(n_f) * 44100.0 / n_fft
        f_mask = f_axis <= 8000.0

        # L2 residual in linear magnitude. Treats "missing energy" and
        # "extra energy" symmetrically; doesn't blow up on silent bins.
        diff = ref_lin - cand_lin
        residual_l2 = float(np.sqrt(np.mean(diff[f_mask, :] ** 2)))

        # A-weighted residual_l2: same diff² but weighted per-frequency
        # by the IEC 61672 A-curve (linear power weight = 10^(A_dB/10)).
        # Ear-relevant: emphasises 500-5000 Hz where human sensitivity
        # peaks, de-emphasises <100 Hz and >10 kHz. Diverges from the
        # unweighted metric when modal+filter trades a low-band fix
        # for mid-band buildup that the listener still hears.
        f_safe = np.maximum(f_axis, 1.0)
        f2 = f_safe ** 2
        ra = (
            (12194.0 ** 2) * (f_safe ** 4)
            / (
                (f2 + 20.6 ** 2)
                * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
                * (f2 + 12194.0 ** 2)
            )
        )
        a_db = 20.0 * np.log10(np.maximum(ra, 1e-12)) + 2.00
        a_w = 10.0 ** (a_db / 10.0)  # linear power weight
        a_w_masked = np.where(f_mask, a_w, 0.0)[:, None]
        weighted_sq = (diff ** 2) * a_w_masked
        denom = float(np.sum(a_w_masked) * diff.shape[1])
        residual_l2_aw = float(np.sqrt(np.sum(weighted_sq) / max(denom, 1e-12)))

        # Per-band energy ratio: ref total / cand total in linear sum.
        # Ratio > 1 means cand is QUIETER than ref (deficit), <1 means
        # cand is LOUDER (surplus).
        bands = [
            (0, 250, "low (<250)"),
            (250, 500, "low-mid (250-500)"),
            (500, 1000, "mid (500-1k)"),
            (1000, 2000, "mid-hi (1-2k)"),
            (2000, 4000, "hi (2-4k)"),
            (4000, 8000, "very-hi (4-8k)"),
        ]
        # Inter-partial floor: how much energy lives BETWEEN the partial
        # peaks. Real recordings (SFZ Salamander) carry soundboard
        # resonances + room reverb + sympathetic strings → continuous
        # spectrum *between* partials. A pure modal bank produces ~0
        # energy between peaks (the "dead inter-partial silence" Risset
        # & Wessel called the missing piece of spectral fusion). Reported
        # as the ratio of floor energy ref/cand: large positive dB means
        # cand has a deader floor than ref ("lo-fi" signature).
        ref_avg = ref_lin.mean(axis=1)
        n_bins = ref_avg.shape[0]
        # Detect partial peaks in the averaged ref spectrum: local max
        # over ±10 bins, > 5 % of global max.
        peak_window = 10
        peak_thresh = 0.05 * float(ref_avg.max())
        is_peak = np.zeros(n_bins, dtype=bool)
        for i in range(peak_window, n_bins - peak_window):
            local = ref_avg[i - peak_window : i + peak_window + 1]
            if ref_avg[i] == local.max() and ref_avg[i] > peak_thresh:
                is_peak[i] = True
        # "Near peak" mask = peak ± 3 bins. Floor mask = the rest.
        near_radius = 3
        near_peak = np.zeros(n_bins, dtype=bool)
        for p in np.where(is_peak)[0]:
            near_peak[max(0, p - near_radius) : min(n_bins, p + near_radius + 1)] = True
        floor_mask = (~near_peak) & f_mask
        ref_floor_e = float(np.sum(ref_lin[floor_mask, :] ** 2))
        cand_floor_e = float(np.sum(cand_lin[floor_mask, :] ** 2))
        if cand_floor_e > 1e-12 and ref_floor_e > 1e-12:
            floor_ratio_db = 10.0 * np.log10(ref_floor_e / cand_floor_e)
        else:
            floor_ratio_db = float("inf") if ref_floor_e > 1e-12 else float("-inf")

        # Centroid trajectory MSE: spectral centroid SC(t) per frame
        # (Σ f·|X(f,t)| / Σ |X(f,t)|), MSE between ref and cand
        # trajectories. Captures dynamic brightness behaviour (Saitis &
        # Weinzierl 2019) which static energy comparison averages away.
        f_for_sc = f_axis[f_mask]
        ref_masked = ref_lin[f_mask, :]
        cand_masked = cand_lin[f_mask, :]
        ref_sc = (ref_masked * f_for_sc[:, None]).sum(axis=0) / np.maximum(
            ref_masked.sum(axis=0), 1e-12
        )
        cand_sc = (cand_masked * f_for_sc[:, None]).sum(axis=0) / np.maximum(
            cand_masked.sum(axis=0), 1e-12
        )
        centroid_mse_hz = float(np.sqrt(np.mean((ref_sc - cand_sc) ** 2)))

        print(f"residual_l2={residual_l2:.6f}")
        print(f"residual_l2_aw={residual_l2_aw:.6f}")
        print(
            f"inter_partial_floor: ref_e={ref_floor_e:.2e} cand_e={cand_floor_e:.2e} "
            f"ref/cand={floor_ratio_db:+.1f} dB"
        )
        print(f"centroid_mse_hz={centroid_mse_hz:.1f}")
        for lo, hi, name in bands:
            band_mask = (f_axis >= lo) & (f_axis < hi)
            if band_mask.any():
                ref_e = float(np.sum(ref_lin[band_mask, :] ** 2))
                cand_e = float(np.sum(cand_lin[band_mask, :] ** 2))
                if cand_e > 1e-12:
                    ratio_db = 10.0 * np.log10(ref_e / cand_e)
                else:
                    ratio_db = float("inf")
                print(
                    f"  band {name}: ref={ref_e:.2e} cand={cand_e:.2e} "
                    f"ref/cand={ratio_db:+.1f} dB"
                )


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
