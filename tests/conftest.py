import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("NOIZE_NO_AUDIO", "1")   # no audio server in CI/sandbox


def tone(sr=44100, secs=1.0, freq=440.0, channels=2, amp=0.5):
    t = np.arange(int(sr * secs)) / sr
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.tile(x, (channels, 1))


@pytest.fixture
def wav_file(tmp_path):
    p = tmp_path / "song.wav"
    sf.write(str(p), tone().T, 44100, subtype="PCM_16")
    return p


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NOIZE_HOME", str(tmp_path / "home"))
