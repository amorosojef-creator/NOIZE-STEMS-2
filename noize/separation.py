"""Demucs inference with chunking, progress, cancellation and error recovery.

`process_chunked` is pure NumPy so alignment can be unit-tested with a fake
model. `DemucsSeparator` runs real inference (needs torch + demucs + weights).
"""
from __future__ import annotations
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .audio_io import resample, to_stereo
from .models import get_model_info
from .settings import QUALITY

ProgressCB = Callable[[float, str], None]


class Cancelled(Exception):
    pass


class SeparationError(Exception):
    pass


@dataclass
class SeparationResult:
    stems: dict[str, np.ndarray]     # name -> (2, n) float32
    sample_rate: int
    model_id: str
    device: str


def resolve_device(preference: str, cuda_available: bool) -> str:
    """auto -> cuda if present else cpu; explicit cuda falls back to cpu."""
    if preference in ("auto", "cuda") and cuda_available:
        return "cuda"
    return "cpu"


def detect_gpu() -> dict:
    """Return {'cuda': bool, 'name': str|None, 'vram_mb': int|None}; never raises."""
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return {"cuda": True, "name": p.name, "vram_mb": int(p.total_memory // 2**20)}
    except Exception:
        pass
    return {"cuda": False, "name": None, "vram_mb": None}


def choose_chunk_seconds(device: str, vram_mb: int | None = None, ram_mb: int | None = None) -> float:
    """Memory-aware chunk length (seconds)."""
    if device == "cuda":
        v = vram_mb or 4000
        return 20.0 if v < 4000 else 40.0 if v < 8000 else 60.0
    r = ram_mb or available_ram_mb() or 8000
    return 15.0 if r < 6000 else 30.0 if r < 12000 else 45.0


def available_ram_mb() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().available // 2**20)
    except Exception:
        return None


def chunk_ranges(n: int, chunk: int, overlap: int) -> list[tuple[int, int]]:
    if n <= chunk:
        return [(0, n)]
    step = max(1, chunk - overlap)
    out, s = [], 0
    while True:
        e = min(s + chunk, n)
        out.append((s, e))
        if e >= n:
            break
        s += step
    return out


def process_chunked(audio: np.ndarray, run_chunk: Callable[[np.ndarray], np.ndarray],
                    chunk: int, overlap: int,
                    progress: ProgressCB | None = None,
                    cancel: threading.Event | None = None) -> np.ndarray:
    """Apply run_chunk (C,n)->(S,C,n) over overlapping chunks with linear crossfade."""
    n = audio.shape[1]
    ranges = chunk_ranges(n, chunk, overlap)
    out = None
    weight = np.zeros(n, dtype=np.float32)
    for i, (s, e) in enumerate(ranges):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        y = run_chunk(audio[:, s:e])
        if out is None:
            out = np.zeros((y.shape[0], y.shape[1], n), dtype=np.float32)
        length = e - s
        w = np.ones(length, dtype=np.float32)
        ov = min(overlap, length // 2)
        if ov > 0:
            ramp = (np.arange(ov, dtype=np.float32) + 1) / (ov + 1)
            if i > 0:
                w[:ov] = ramp
            if i < len(ranges) - 1:
                w[-ov:] = np.minimum(w[-ov:], ramp[::-1])
        out[:, :, s:e] += y[:, :, :length] * w
        weight[s:e] += w
        if progress:
            progress((i + 1) / len(ranges), f"Separating {i + 1}/{len(ranges)}")
    weight[weight == 0] = 1.0
    return out / weight


def vocals_instrumental(stems: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if "vocals" not in stems:
        raise KeyError("Model has no 'vocals' stem")
    others = [v for k, v in stems.items() if k != "vocals"]
    return {"vocals": stems["vocals"], "instrumental": np.sum(others, axis=0)}


class DemucsSeparator:
    def __init__(self, model_id: str, models_dir: Path, device_pref: str = "auto",
                 quality: str = "balanced"):
        self.info = get_model_info(model_id)
        self.models_dir = Path(models_dir)
        self.device_pref = device_pref
        self.quality = quality if quality in QUALITY else "balanced"

    def _load(self, device: str):
        try:
            import torch  # noqa: F401
            from demucs.pretrained import REMOTE_ROOT
            from demucs.repo import AnyModelRepo, BagOnlyRepo, LocalRepo
        except ImportError as e:
            raise SeparationError(f"AI runtime missing ({e}). Reinstall NOIZE.") from e
        missing = [f for f in self.info.files if not (self.models_dir / f).exists()]
        if missing:
            raise SeparationError(f"Model '{self.info.id}' is not installed. Open MODELS and install it.")
        local = LocalRepo(self.models_dir)
        repo = AnyModelRepo(local, BagOnlyRepo(REMOTE_ROOT, local))
        model = repo.get_model(self.info.id)
        model.eval()
        model.to(device)
        return model

    def separate(self, audio: np.ndarray, sr: int, progress: ProgressCB | None = None,
                 cancel: threading.Event | None = None) -> SeparationResult:
        gpu = detect_gpu()
        device = resolve_device(self.device_pref, gpu["cuda"])
        try:
            return self._run(audio, sr, device, gpu, progress, cancel)
        except Cancelled:
            raise
        except RuntimeError as e:
            if device == "cuda" and "out of memory" in str(e).lower():
                if progress:
                    progress(0.0, "GPU out of memory - retrying on CPU")
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return self._run(audio, sr, "cpu", gpu, progress, cancel)
            raise SeparationError(f"Separation failed: {e}") from e

    def _run(self, audio, sr, device, gpu, progress, cancel) -> SeparationResult:
        import torch
        from demucs.apply import apply_model
        if progress:
            progress(0.0, "Loading model")
        model = self._load(device)
        msr = int(getattr(model, "samplerate", 44100))
        x = resample(to_stereo(audio.astype(np.float32)), sr, msr)
        ref = x.mean(0)
        mean, std = float(ref.mean()), float(ref.std()) or 1.0
        xn = (x - mean) / std
        q = QUALITY[self.quality]
        chunk_s = choose_chunk_seconds(device, gpu.get("vram_mb"))
        chunk, overlap = int(chunk_s * msr), int(2.0 * msr)

        def run_chunk(c: np.ndarray) -> np.ndarray:
            t = torch.from_numpy(np.ascontiguousarray(c))[None]
            with torch.no_grad():
                y = apply_model(model, t, shifts=q["shifts"], split=True, overlap=q["overlap"],
                                device=device, progress=False)[0]
            return y.cpu().numpy()

        stems = process_chunked(xn, run_chunk, chunk, overlap, progress, cancel)
        stems = stems * std + mean
        names = list(model.sources)
        return SeparationResult({n: np.ascontiguousarray(stems[i]) for i, n in enumerate(names)},
                                msr, self.info.id, device)
