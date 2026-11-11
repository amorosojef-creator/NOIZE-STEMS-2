"""Export engine: WAV / FLAC / MP3, 16/24-bit, resampling, safe output paths."""
from __future__ import annotations
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio_io import AudioError, find_ffmpeg, resample
from .mixing import MixState, prevent_clipping, render_mix

SUBTYPE = {16: "PCM_16", 24: "PCM_24"}
FORMATS = ("wav", "flac", "mp3")


def safe_name(name: str) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return s or "untitled"


def output_dir_for(track: str | Path, base: str | Path, organize: bool = True) -> Path:
    stem = safe_name(Path(track).stem)
    return Path(base) / stem if organize else Path(base)


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    i = 2
    while True:
        cand = path.with_name(f"{path.stem} ({i}){path.suffix}")
        if not cand.exists():
            return cand
        i += 1


def mp3_available() -> bool:
    return find_ffmpeg() is not None


def write_audio(path: Path, audio: np.ndarray, sr: int, fmt: str = "wav",
                bit_depth: int = 24, target_sr: int | None = None) -> Path:
    """Write (C, n) float audio. Clipping is prevented; length is preserved."""
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise AudioError(f"Unsupported export format: {fmt}")
    if bit_depth not in SUBTYPE:
        raise AudioError("Bit depth must be 16 or 24")
    if target_sr and target_sr != sr:
        audio = resample(audio, sr, target_sr)
        sr = target_sr
    audio = prevent_clipping(audio)
    path = unique_path(path.with_suffix("." + fmt))
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(audio.T)
    if fmt in ("wav", "flac"):
        sf.write(str(path), data, sr, subtype=SUBTYPE[bit_depth], format=fmt.upper())
        return path
    ff = find_ffmpeg()
    if not ff:
        raise AudioError("MP3 export needs FFmpeg (not found). Choose WAV or FLAC.")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "t.wav"
        sf.write(str(tmp), data, sr, subtype="PCM_24")
        r = subprocess.run([ff, "-nostdin", "-v", "error", "-y", "-i", str(tmp),
                            "-codec:a", "libmp3lame", "-b:a", "320k", str(path)],
                           capture_output=True, text=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        path.unlink(missing_ok=True)
        raise AudioError("MP3 encoding failed (libmp3lame unavailable?). Choose WAV or FLAC.")
    return path


def export_stems(stems: dict[str, np.ndarray], sr: int, out_dir: Path, track: str,
                 selected: list[str] | None = None, fmt: str = "wav", bit_depth: int = 24,
                 target_sr: int | None = None, extra: dict[str, np.ndarray] | None = None) -> list[Path]:
    pool = dict(stems)
    if extra:
        pool.update(extra)
    names = selected if selected else list(stems)
    unknown = [n for n in names if n not in pool]
    if unknown:
        raise AudioError(f"Unknown stems: {', '.join(unknown)}")
    base = safe_name(Path(track).stem)
    return [write_audio(Path(out_dir) / f"{base} - {n}", pool[n], sr, fmt, bit_depth, target_sr)
            for n in names]


def export_mix(stems: dict[str, np.ndarray], state: MixState, sr: int, out_dir: Path,
               track: str, fmt: str = "wav", bit_depth: int = 24,
               target_sr: int | None = None) -> Path:
    mix = render_mix(stems, state)
    base = safe_name(Path(track).stem)
    return write_audio(Path(out_dir) / f"{base} - custom mix", mix, sr, fmt, bit_depth, target_sr)
