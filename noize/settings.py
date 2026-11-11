"""JSON-backed settings with defaults and corruption recovery."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any

from .paths import app_data_dir

DEFAULTS: dict[str, Any] = {
    "model": "htdemucs",
    "device": "auto",            # auto | cpu | cuda
    "quality": "balanced",       # fast | balanced | best
    "export_format": "wav",      # wav | flac | mp3
    "export_sample_rate": 44100,
    "export_bit_depth": 24,      # 16 | 24
    "output_dir": "",
    "organize_by_track": True,
    "clean_enabled": False,
    "clean_strength": 0.5,
    "models_consent": False,
    "last_input_dir": "",
}
QUALITY = {  # shifts (random-shift averaging), overlap
    "fast": {"shifts": 0, "overlap": 0.10},
    "balanced": {"shifts": 1, "overlap": 0.25},
    "best": {"shifts": 2, "overlap": 0.25},
}


class Settings:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else app_data_dir() / "settings.json"
        self.data: dict[str, Any] = dict(DEFAULTS)
        self.load()

    def load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if k in DEFAULTS and type(v) is type(DEFAULTS[k]):
                        self.data[k] = v
        except (OSError, ValueError):
            pass  # missing or corrupt -> defaults

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def get(self, key: str) -> Any:
        return self.data.get(key, DEFAULTS.get(key))

    def set(self, key: str, value: Any, save: bool = True) -> None:
        if key not in DEFAULTS:
            raise KeyError(key)
        self.data[key] = value
        if save:
            self.save()
