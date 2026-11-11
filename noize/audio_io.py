"""Audio validation, decoding (libsndfile / FFmpeg) and resampling.

Arrays are float32 with shape (channels, samples).
"""
from __future__ import annotations
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .paths import resource_path

AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aif", ".aiff", ".aac", ".wma"}
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv"}
SUPPORTED_EXT = AUDIO_EXT | VIDEO_EXT
_SF_NATIVE = {".wav", ".flac", ".ogg", ".aif", ".aiff", ".mp3"}


class AudioError(Exception):
    """User-presentable audio problem."""


def find_ffmpeg() -> str | None:
    bundled = resource_path("ffmpeg/ffmpeg.exe" if sys.platform == "win32" else "ffmpeg/ffmpeg")
    if bundled.exists():
        return str(bundled)
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:  # optional pip dependency that ships an ffmpeg binary
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def validate_input(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise AudioError(f"File not found: {p}")
    if not p.is_file():
        raise AudioError(f"Not a file: {p}")
    if p.suffix.lower() not in SUPPORTED_EXT:
        raise AudioError(f"Unsupported file type '{p.suffix}'. Supported: "
                         + ", ".join(sorted(SUPPORTED_EXT)))
    if p.stat().st_size == 0:
        raise AudioError(f"File is empty: {p.name}")
    return p


def _decode_ffmpeg(p: Path) -> tuple[np.ndarray, int]:
    ff = find_ffmpeg()
    if not ff:
        raise AudioError(f"Decoding {p.suffix} needs FFmpeg, which was not found. "
                         "Install FFmpeg or place ffmpeg.exe in the NOIZE 'ffmpeg' folder.")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "decoded.wav"
        cmd = [ff, "-nostdin", "-v", "error", "-y", "-i", str(p), "-vn",
               "-acodec", "pcm_f32le", str(out)]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        r = subprocess.run(cmd, capture_output=True, text=True, creationflags=flags)
        if r.returncode != 0 or not out.exists():
            msg = (r.stderr or "").strip().splitlines()
            raise AudioError(f"FFmpeg could not decode {p.name}: "
                             + (msg[-1] if msg else "unknown error"))
        data, sr = sf.read(str(out), dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.T), int(sr)


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    p = validate_input(path)
    ext = p.suffix.lower()
    if ext in _SF_NATIVE:
        try:
            data, sr = sf.read(str(p), dtype="float32", always_2d=True)
            if data.shape[0] == 0:
                raise AudioError(f"No audio data in {p.name}")
            return np.ascontiguousarray(data.T), int(sr)
        except AudioError:
            raise
        except Exception:
            pass  # fall through to FFmpeg
    audio, sr = _decode_ffmpeg(p)
    if audio.shape[1] == 0:
        raise AudioError(f"No audio data in {p.name}")
    return audio, sr


def probe_duration(path: str | Path) -> float | None:
    try:
        return float(sf.info(str(path)).duration)
    except Exception:
        return None


def to_stereo(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 2:
        return x
    if x.shape[0] == 1:
        return np.repeat(x, 2, axis=0)
    return np.ascontiguousarray(x[:2])


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    from scipy.signal import resample_poly
    g = math.gcd(int(sr_from), int(sr_to))
    y = resample_poly(x, int(sr_to) // g, int(sr_from) // g, axis=-1)
    return np.ascontiguousarray(y.astype(np.float32))
