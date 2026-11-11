"""NOIZE CLEAN - optional DSP post-processing to reduce stem bleed and artifacts.

Methods (classical DSP, no promise of perfect removal):
  * reduce_bleed: Wiener-style soft spectral mask of a stem against the sum of the
    other stems, smoothed in time/frequency to limit 'musical noise'.
  * soft_gate: slow envelope gate that attenuates very quiet passages where
    residual bleed is most audible.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal import istft, stft

NPERSEG = 2048


def reduce_bleed(target: np.ndarray, others: np.ndarray, sr: int, strength: float = 0.5) -> np.ndarray:
    """target/others: (C, n). strength 0..1. Output has identical shape."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0.0 or target.shape[1] < NPERSEG:
        return target.copy()
    n = target.shape[1]
    out = np.empty_like(target)
    alpha = 0.5 + 2.5 * strength
    for c in range(target.shape[0]):
        _, _, T = stft(target[c], fs=sr, nperseg=NPERSEG)
        _, _, O = stft(others[c], fs=sr, nperseg=NPERSEG)
        pt, po = np.abs(T) ** 2, np.abs(O) ** 2
        mask = pt / (pt + alpha * po + 1e-12)
        mask = uniform_filter(mask, size=(3, 3), mode="nearest")
        # never fully zero a bin; stronger settings allow deeper attenuation
        gain = np.clip(mask, 1.0 - 0.95 * strength, 1.0)
        _, y = istft(T * gain, fs=sr, nperseg=NPERSEG)
        y = y[:n]
        if y.shape[0] < n:
            y = np.pad(y, (0, n - y.shape[0]))
        out[c] = y
    return out.astype(np.float32)


def soft_gate(x: np.ndarray, sr: int, threshold_db: float = -60.0, strength: float = 0.5) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0.0:
        return x.copy()
    win = max(1, int(0.02 * sr))
    env = uniform_filter(np.abs(x).max(axis=0), size=win, mode="nearest")
    thr = 10 ** (threshold_db / 20)
    g = np.clip(env / thr, 0.0, 1.0)
    g = uniform_filter(g, size=max(1, int(0.05 * sr)), mode="nearest")
    gain = (1.0 - strength) + strength * g
    return (x * gain).astype(np.float32)


def clean_stems(stems: dict[str, np.ndarray], sr: int, strength: float = 0.5) -> dict[str, np.ndarray]:
    total = np.sum(list(stems.values()), axis=0)
    out = {}
    for name, s in stems.items():
        others = total - s
        y = reduce_bleed(s, others, sr, strength)
        out[name] = soft_gate(y, sr, strength=strength * 0.5)
    return out
