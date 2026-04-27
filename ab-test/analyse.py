"""Waveform analysis for keysynth A/B reference WAV.

Reads ab-test/<name>.wav, dumps:
  - peak (linear + dBFS) per channel
  - RMS (linear + dBFS) per channel
  - DC offset
  - spectral centroid + roll-off
  - first 16 harmonic peaks (around the rendered chord's fundamentals)
  - PNG: waveform, spectrogram, magnitude spectrum
"""

import sys
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def db(x: float) -> float:
    return 20.0 * np.log10(max(abs(x), 1e-12))


def analyse(path: Path) -> None:
    sr, data = wavfile.read(path)
    if data.dtype.kind == "i":
        # i16 / i32 → normalise to ±1.0 float
        max_v = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_v
    elif data.dtype != np.float32:
        data = data.astype(np.float32)

    if data.ndim == 1:
        channels = [("mono", data)]
    else:
        channels = [(f"ch{i}", data[:, i]) for i in range(data.shape[1])]

    print(f"\n=== {path.name} ===")
    print(f"sample_rate: {sr} Hz")
    print(f"duration:    {len(data) / sr:.3f} s")
    print(f"channels:    {len(channels)}")
    print()

    print(f"{'name':<6} {'peak':>9} {'peak_dB':>9} {'rms':>9} {'rms_dB':>9} {'dc':>10}")
    for name, ch in channels:
        peak = float(np.max(np.abs(ch)))
        rms = float(np.sqrt(np.mean(ch.astype(np.float64) ** 2)))
        dc = float(np.mean(ch))
        print(
            f"{name:<6} {peak:>9.4f} {db(peak):>9.2f} "
            f"{rms:>9.4f} {db(rms):>9.2f} {dc:>10.2e}"
        )

    # Mix to mono for spectrum.
    mono = np.mean(np.stack([ch for _, ch in channels]), axis=0)

    # Spectral centroid + roll-off (85th percentile of cumulative power).
    nperseg = 4096
    f, t, Sxx = signal.spectrogram(
        mono, fs=sr, nperseg=nperseg, noverlap=nperseg // 2, scaling="spectrum"
    )
    power = Sxx.mean(axis=1)  # average across time → per-bin power
    bin_centres = f
    centroid = float(np.sum(bin_centres * power) / max(np.sum(power), 1e-12))
    cumpower = np.cumsum(power)
    rolloff_idx = int(np.searchsorted(cumpower, 0.85 * cumpower[-1]))
    rolloff = float(bin_centres[min(rolloff_idx, len(bin_centres) - 1)])
    print(f"\nspectral_centroid: {centroid:>8.1f} Hz")
    print(f"rolloff_85%:       {rolloff:>8.1f} Hz")

    # Top-K spectral peaks (post-attack window: 0.1–1.5 s into the file).
    start = int(0.1 * sr)
    end = int(min(1.5 * sr, len(mono)))
    seg = mono[start:end]
    if len(seg) >= nperseg:
        f2, P = signal.welch(seg, fs=sr, nperseg=nperseg)
        peaks_idx, _ = signal.find_peaks(P, distance=20, prominence=P.max() * 0.005)
        # Sort by power, keep top 16.
        peaks_idx = peaks_idx[np.argsort(-P[peaks_idx])][:16]
        peaks = sorted([(float(f2[i]), float(P[i])) for i in peaks_idx], key=lambda x: x[0])
        print(f"\ntop {len(peaks)} spectral peaks (0.1–1.5 s window):")
        print(f"  {'freq_Hz':>10} {'power_dB':>10}")
        max_p = max((p for _, p in peaks), default=1e-12)
        for fz, pw in peaks:
            print(f"  {fz:>10.1f} {db(pw / max_p):>10.2f}")

    # Plot.
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # Waveform
    t_axis = np.arange(len(mono)) / sr
    axes[0].plot(t_axis, mono, lw=0.6)
    axes[0].set_title(f"waveform — {path.name}")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("amplitude")
    axes[0].set_ylim(-1.0, 1.0)
    axes[0].grid(alpha=0.3)

    # Spectrogram (log-magnitude, dB)
    Sxx_db = 10.0 * np.log10(np.maximum(Sxx, 1e-10))
    im = axes[1].pcolormesh(
        t, f, Sxx_db, shading="auto", cmap="magma", vmin=Sxx_db.max() - 80
    )
    axes[1].set_ylim(0, 6000)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("freq (Hz)")
    axes[1].set_title("spectrogram (0–6 kHz, dB)")
    plt.colorbar(im, ax=axes[1], label="dB")

    # Magnitude spectrum (post-attack)
    if len(seg) >= nperseg:
        axes[2].semilogy(f2, P, lw=0.7)
        axes[2].set_xlim(20, 8000)
        axes[2].set_xscale("log")
        axes[2].set_xlabel("freq (Hz, log)")
        axes[2].set_ylabel("power (Welch PSD)")
        axes[2].set_title("magnitude spectrum, post-attack (0.1–1.5 s)")
        axes[2].grid(alpha=0.3, which="both")

    fig.tight_layout()
    out_png = path.with_suffix(".png")
    fig.savefig(out_png, dpi=110)
    print(f"\nplot: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        analyse(Path(arg))
