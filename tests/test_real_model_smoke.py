"""REAL-MODEL smoke test. Skipped unless torch, demucs and the htdemucs weights exist.

Run:  set NOIZE_REAL_MODEL=1 && pytest -m realmodel -s
This is the only test that exercises real neural inference. All other tests use
mathematical fakes and do NOT prove separation quality.
"""
import os

import numpy as np
import pytest

from noize.models import MODELS
from noize.paths import models_dir
from noize.separation import DemucsSeparator
from conftest import tone

pytestmark = pytest.mark.realmodel


def _ready():
    if not os.environ.get("NOIZE_REAL_MODEL"):
        return "set NOIZE_REAL_MODEL=1 to run"
    try:
        import demucs, torch  # noqa: F401,E401
    except ImportError:
        return "torch/demucs not installed"
    d = models_dir() if not os.environ.get("NOIZE_MODELS_DIR") else __import__("pathlib").Path(os.environ["NOIZE_MODELS_DIR"])
    if not all((d / f).exists() for f in MODELS["htdemucs"].files):
        return "htdemucs weights not installed"
    return None


def test_real_htdemucs_four_stems():
    why = _ready()
    if why:
        pytest.skip(why)
    import pathlib
    d = pathlib.Path(os.environ.get("NOIZE_MODELS_DIR") or models_dir())
    sr = 44100
    # Synthetic, self-generated (legally unencumbered) 6 s signal: bass + chords + noise bursts.
    mix = tone(sr, 6, 60, amp=0.3) + tone(sr, 6, 440, amp=0.2) + 0.05 * np.random.default_rng(1).standard_normal((2, 6 * sr)).astype(np.float32)
    res = DemucsSeparator("htdemucs", d, "cpu", "fast").separate(mix, sr)
    assert set(res.stems) == {"drums", "bass", "other", "vocals"}
    for s in res.stems.values():
        assert s.shape == (2, mix.shape[1]) and np.isfinite(s).all()
    recon = sum(res.stems.values())
    err = np.mean((recon - mix) ** 2) / np.mean(mix ** 2)
    assert err < 0.05          # stems should sum back to roughly the mixture
