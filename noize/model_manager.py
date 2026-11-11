"""Consent-based model downloads with SHA-256 prefix verification.

Demucs release files are named '<sig>-<sha256 prefix>.th'; the prefix is the
trusted checksum and is verified after download.
"""
from __future__ import annotations
import hashlib
import threading
import urllib.request
from pathlib import Path
from typing import Callable

from .models import MODELS, get_model_info


class ConsentRequired(Exception):
    pass


class DownloadCancelled(Exception):
    pass


class ChecksumError(Exception):
    pass


def verify_file(path: Path) -> bool:
    """Verify file against checksum embedded in its name."""
    try:
        want = path.stem.split("-", 1)[1]
    except IndexError:
        return False
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(1 << 20), b""):
            h.update(buf)
    return h.hexdigest().startswith(want)


class ModelManager:
    def __init__(self, directory: Path, opener: Callable = urllib.request.urlopen):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._open = opener

    def missing_files(self, model_id: str) -> list[str]:
        info = get_model_info(model_id)
        return [f for f in info.files if not (self.dir / f).exists()]

    def is_installed(self, model_id: str) -> bool:
        return not self.missing_files(model_id)

    def installed_ids(self) -> set[str]:
        return {m for m in MODELS if self.is_installed(m)}

    def verify(self, model_id: str) -> bool:
        return all((self.dir / f).exists() and verify_file(self.dir / f)
                   for f in get_model_info(model_id).files)

    def remove(self, model_id: str) -> None:
        for f in get_model_info(model_id).files:
            (self.dir / f).unlink(missing_ok=True)

    def install(self, model_id: str, consent: bool,
                progress: Callable[[float, str], None] | None = None,
                cancel: threading.Event | None = None) -> None:
        if not consent:
            raise ConsentRequired("Download requires user consent.")
        info = get_model_info(model_id)
        todo = self.missing_files(model_id)
        for i, fname in enumerate(todo):
            self._download(info.urls[fname], self.dir / fname,
                           lambda frac, i=i: progress and progress((i + frac) / len(todo), fname),
                           cancel)
        if progress:
            progress(1.0, "done")

    def _download(self, url: str, dest: Path, cb, cancel) -> None:
        part = dest.with_suffix(".part")
        try:
            with self._open(url, timeout=30) as resp, open(part, "wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                got = 0
                while True:
                    if cancel is not None and cancel.is_set():
                        raise DownloadCancelled()
                    buf = resp.read(1 << 18)
                    if not buf:
                        break
                    out.write(buf)
                    got += len(buf)
                    if total:
                        cb(got / total)
            if not _verify_part(part, dest.name):
                raise ChecksumError(f"Checksum mismatch for {dest.name}; file discarded.")
            part.replace(dest)
        except BaseException:
            part.unlink(missing_ok=True)
            raise


def _verify_part(part: Path, final_name: str) -> bool:
    want = Path(final_name).stem.split("-", 1)[1]
    h = hashlib.sha256()
    with open(part, "rb") as f:
        for buf in iter(lambda: f.read(1 << 20), b""):
            h.update(buf)
    return h.hexdigest().startswith(want)
