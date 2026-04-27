"""Apply native vs OLD-web vs NEW-web bus chain to the raw voice sum,
compare waveforms + spectra side-by-side.

Inputs:
  ab-test/native_pianomodal_default.wav  (raw voice sum, normalised)

Bus chains (mirrors src/main.rs::audio_callback / src/bin/web.rs::render_mono_chunk):

  - old_web    : master * tanh, Plain mode, master = 3.0   (PR #18..#28 era)
  - new_web    : master * ParallelComp + reverb, master=1.0 (PR #29 = native parity)
  - native_ref : same as new_web (identical defaults)

Output:
  ab-test/bus_compare.png  — 3-row plot, waveform left, spectrum right
  ab-test/old_web.wav      — re-rendered with broken old defaults
  ab-test/new_web.wav      — re-rendered with PR #29 defaults
"""

import sys
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-12))


def parallel_comp(mono: np.ndarray, master: float) -> np.ndarray:
    """Mirror src/main.rs::audio_callback ParallelComp branch.

    α=0.7, β=0.6 wet/dry blend; envelope follower attack=0.5 release=0.0001;
    final tanh saturation post-blend × master."""
    ALPHA = 0.7
    BETA = 0.6
    ATTACK = 0.5
    RELEASE = 0.0001
    out = np.empty_like(mono)
    compressed = np.empty_like(mono)
    gain = 1.0
    for i, s in enumerate(mono):
        a = abs(s)
        if a > gain:
            gain += (a - gain) * ATTACK
        else:
            gain += (a - gain) * RELEASE
        gr = 1.0 / gain if gain > 1.0 else 1.0
        compressed[i] = s * gr
    for i, s in enumerate(mono):
        combined = (s * ALPHA + compressed[i] * BETA) * master
        out[i] = np.tanh(combined)
    return out


def plain_tanh(mono: np.ndarray, master: float) -> np.ndarray:
    return np.tanh(mono * master)


def main() -> None:
    src = Path("native_pianomodal_default.wav")
    sr, data = wavfile.read(src)
    if data.dtype.kind == "i":
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    elif data.dtype != np.float32:
        data = data.astype(np.float32)

    if data.ndim == 2:
        # render_chord already normalises to -3 dBFS — sum to mono pre-bus.
        mono_in = data.mean(axis=1)
    else:
        mono_in = data
    mono_in = mono_in.astype(np.float64)

    # The render_chord output was peak-normalised to -3 dBFS so the input
    # to the bus is roughly what the live audio_callback would see post
    # voice mix (modulo reverb + sympathetic, which we don't simulate
    # here — we're isolating the master+MixMode stage that PR #29 fixed).

    old_web = plain_tanh(mono_in, master=3.0)
    new_web = parallel_comp(mono_in, master=1.0)

    out_dir = Path(".")
    wavfile.write(out_dir / "old_web.wav", sr, (old_web * 32767).astype(np.int16))
    wavfile.write(out_dir / "new_web.wav", sr, (new_web * 32767).astype(np.int16))

    rows = [
        ("input (raw voice sum, peak-norm)", mono_in),
        ("old_web (master=3.0, Plain tanh)", old_web),
        ("new_web (master=1.0, ParallelComp+tanh)", new_web),
    ]

    print(f"sr: {sr} Hz, duration: {len(mono_in)/sr:.2f}s")
    print(f"\n{'name':<42} {'peak':>9} {'peak_dB':>9} {'rms':>9} {'rms_dB':>9} {'crest':>7}")
    for name, x in rows:
        peak = float(np.max(np.abs(x)))
        rms = float(np.sqrt(np.mean(x ** 2)))
        crest = peak / max(rms, 1e-12)
        print(
            f"{name:<42} {peak:>9.4f} {db(peak):>9.2f} "
            f"{rms:>9.4f} {db(rms):>9.2f} {crest:>7.2f}"
        )

    fig, axes = plt.subplots(len(rows), 2, figsize=(14, 8))
    t = np.arange(len(mono_in)) / sr
    for r, (name, x) in enumerate(rows):
        axes[r, 0].plot(t, x, lw=0.5)
        axes[r, 0].set_xlim(0, min(2.0, t[-1]))
        axes[r, 0].set_ylim(-1.0, 1.0)
        axes[r, 0].set_title(f"waveform — {name}")
        axes[r, 0].grid(alpha=0.3)

        # Welch PSD on the post-attack window
        s = int(0.1 * sr)
        e = int(min(1.5 * sr, len(x)))
        f, P = signal.welch(x[s:e], fs=sr, nperseg=4096)
        axes[r, 1].semilogy(f, P, lw=0.7)
        axes[r, 1].set_xlim(20, 8000)
        axes[r, 1].set_xscale("log")
        axes[r, 1].set_title(f"PSD (post-attack) — {name}")
        axes[r, 1].grid(alpha=0.3, which="both")
        # mark fundamentals of C4-E4-G4
        for freq, lab in [(261.63, "C4"), (329.63, "E4"), (392.0, "G4")]:
            axes[r, 1].axvline(freq, color="r", lw=0.4, alpha=0.5)
            axes[r, 1].text(freq, P.max() * 0.7, lab, color="r", fontsize=7)

    fig.tight_layout()
    out = Path("bus_compare.png")
    fig.savefig(out, dpi=110)
    print(f"\nplot: {out.resolve()}")
    print(f"audio: {(out.parent / 'old_web.wav').resolve()}")
    print(f"audio: {(out.parent / 'new_web.wav').resolve()}")


if __name__ == "__main__":
    main()
