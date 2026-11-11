"""Filesystem locations (works from source and from a PyInstaller build)."""
from __future__ import annotations
import os
import sys
from pathlib import Path


def resource_path(relative: str) -> Path:
    """Locate a bundled resource (PyInstaller-aware)."""
    base = getattr(sys, "_MEIPASS", None)
    root = Path(base) if base else Path(__file__).resolve().parent
    candidate = root / relative
    if candidate.exists():
        return candidate
    return Path(__file__).resolve().parent / relative


def asset(name: str) -> Path:
    return resource_path(f"assets/{name}")


def app_data_dir() -> Path:
    override = os.environ.get("NOIZE_HOME")
    if override:
        p = Path(override)
    elif os.name == "nt":
        p = Path(os.environ.get("APPDATA", Path.home())) / "NOIZE"
    else:
        p = Path.home() / ".noize"
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = app_data_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    p = app_data_dir() / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p
