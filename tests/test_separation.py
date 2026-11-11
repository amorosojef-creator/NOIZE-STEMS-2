import threading

import numpy as np
import pytest

from noize import separation as sep
from conftest import tone


def test_cpu_fallback_logic():
    assert sep.resolve_device("auto", True) == "cuda"
    assert sep.resolve_device("auto", False) == "cpu"
    assert sep.resolve_device("cuda", False) == "cpu"      # explicit CUDA without GPU -> CPU
    assert sep.resolve_device("cpu", True) == "cpu"


def test_chunk_ranges_cover_everything():
    r = sep.chunk_ranges(1000, 300, 50)
    assert r[0][0] == 0 and r[-1][1] == 1000
    for (a, b), (c, d) in zip(r, r[1:]):
        assert c < b            # overlap
    assert sep.chunk_ranges(100, 300, 50) == [(0, 100)]


def test_chunk_size_is_memory_aware():
    assert sep.choose_chunk_seconds("cpu", ram_mb=2000) < sep.choose_chunk_seconds("cpu", ram_mb=32000)
    assert sep.choose_chunk_seconds("cuda", vram_mb=2000) < sep.choose_chunk_seconds("cuda", vram_mb=12000)


def fake_model(c):
    """Pretend 2-stem model: stem0 = 0.25*x, stem1 = 0.75*x (sum == input)."""
    return np.stack([c * 0.25, c * 0.75])


def test_stitching_keeps_alignment_and_sums_to_input():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 10_000)).astype(np.float32)
    out = sep.process_chunked(x, fake_model, chunk=3000, overlap=500)
    assert out.shape == (2, 2, 10_000)
    np.testing.assert_allclose(out[0], x * 0.25, atol=1e-5)
    np.testing.assert_allclose(out.sum(0), x, atol=1e-5)


def test_stitching_short_tail():
    x = tone(secs=0.5)
    out = sep.process_chunked(x, fake_model, chunk=9000, overlap=2000)
    np.testing.assert_allclose(out.sum(0), x, atol=1e-5)


def test_progress_and_cancel():
    x = np.ones((2, 5000), np.float32)
    calls = []
    sep.process_chunked(x, fake_model, 1000, 100, progress=lambda f, m: calls.append(f))
    assert calls[-1] == pytest.approx(1.0) and calls == sorted(calls)
    ev = threading.Event()
    ev.set()
    with pytest.raises(sep.Cancelled):
        sep.process_chunked(x, fake_model, 1000, 100, cancel=ev)


def test_vocals_instrumental():
    st = {"drums": np.ones((2, 4), np.float32), "bass": np.ones((2, 4), np.float32),
          "vocals": np.full((2, 4), 3, np.float32)}
    r = sep.vocals_instrumental(st)
    assert set(r) == {"vocals", "instrumental"} and np.all(r["instrumental"] == 2)
    with pytest.raises(KeyError):
        sep.vocals_instrumental({"drums": st["drums"]})


def test_missing_model_gives_clear_error(tmp_path):
    s = sep.DemucsSeparator("htdemucs", tmp_path)
    pytest.importorskip("torch")
    pytest.importorskip("demucs")
    with pytest.raises(sep.SeparationError, match="not installed"):
        s.separate(tone(), 44100)
