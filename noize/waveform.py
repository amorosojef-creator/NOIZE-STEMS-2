"""Waveform peak computation for display."""
from __future__ import annotations
import numpy as np


def compute_peaks(audio: np.ndarray, bins: int) -> np.ndarray:
    """Return (bins, 2) array of (min, max) over the mono mix of `audio` (C, n)."""
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    n = mono.shape[0]
    if bins <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if n == 0:
        return np.zeros((bins, 2), dtype=np.float32)
    bins = min(bins, n)
    idx = np.linspace(0, n, bins, endpoint=False).astype(np.int64)
    out = np.empty((bins, 2), dtype=np.float32)
    out[:, 0] = np.minimum.reduceat(mono, idx)
    out[:, 1] = np.maximum.reduceat(mono, idx)
    return out
