"""Stem mixing: gain, pan, mute, solo, and clip protection."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class StemState:
    volume: float = 1.0     # linear 0..2
    pan: float = 0.0        # -1 (L) .. +1 (R)
    mute: bool = False
    solo: bool = False


@dataclass
class MixState:
    stems: dict[str, StemState] = field(default_factory=dict)

    @classmethod
    def for_names(cls, names) -> "MixState":
        return cls({n: StemState() for n in names})

    def audible(self, name: str) -> bool:
        any_solo = any(s.solo for s in self.stems.values())
        s = self.stems[name]
        return (s.solo if any_solo else not s.mute)


def pan_gains(pan: float) -> tuple[float, float]:
    """Constant-power pan with 0 dB at centre."""
    a = (np.clip(pan, -1.0, 1.0) + 1.0) * np.pi / 4.0
    return float(np.cos(a) * np.sqrt(2)), float(np.sin(a) * np.sqrt(2))


def render_mix(stems: dict[str, np.ndarray], state: MixState,
               start: int = 0, end: int | None = None) -> np.ndarray:
    """Return (2, n) mix of the audible stems for the sample range."""
    first = next(iter(stems.values()))
    end = first.shape[1] if end is None else end
    out = np.zeros((2, max(0, end - start)), dtype=np.float32)
    for name, data in stems.items():
        if not state.audible(name):
            continue
        st = state.stems[name]
        gl, gr = pan_gains(st.pan)
        seg = data[:, start:end]
        out[0, :seg.shape[1]] += seg[0] * st.volume * gl
        out[1, :seg.shape[1]] += seg[1] * st.volume * gr
    return out


def prevent_clipping(x: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    """Scale down (never up) if peak exceeds ceiling; preserves alignment."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return x * (ceiling / peak) if peak > ceiling else x
