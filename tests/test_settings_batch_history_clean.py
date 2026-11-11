import threading

import numpy as np
import pytest

from noize.audio_io import AudioError
from noize.batch import CANCELLED, DONE, FAILED, QUEUED, BatchQueue
from noize.clean import clean_stems, reduce_bleed
from noize.history import History
from noize.settings import DEFAULTS, Settings
from conftest import tone


def test_settings_persistence(tmp_path):
    p = tmp_path / "s.json"
    s = Settings(p)
    assert s.get("model") == DEFAULTS["model"]
    s.set("model", "htdemucs_ft")
    s.set("export_bit_depth", 16)
    assert Settings(p).get("model") == "htdemucs_ft"
    assert Settings(p).get("export_bit_depth") == 16


def test_settings_corrupt_and_wrong_types(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{broken")
    assert Settings(p).get("model") == "htdemucs"
    p.write_text('{"model": 5, "bogus": 1, "device": "cpu"}')
    s = Settings(p)
    assert s.get("model") == "htdemucs" and s.get("device") == "cpu"
    with pytest.raises(KeyError):
        s.set("bogus", 1)


def test_queue_operations(wav_file, tmp_path):
    q = BatchQueue()
    q.add(wav_file)
    with pytest.raises(AudioError, match="Already"):
        q.add(wav_file)
    bad = tmp_path / "x.txt"
    bad.write_text("x")
    assert len(q.add_many([bad, tmp_path / "gone.wav"])) == 2
    other = tmp_path / "o.wav"
    other.write_bytes(wav_file.read_bytes())
    q.add(other)
    q.move(1, 0)
    assert q.items[0].path == other
    q.remove(0)
    assert len(q.items) == 1


def test_queue_isolates_failures(wav_file, tmp_path):
    q = BatchQueue()
    other = tmp_path / "o.wav"
    other.write_bytes(wav_file.read_bytes())
    q.add_many([wav_file, other])

    def proc(path):
        if path == wav_file:
            raise RuntimeError("boom")

    q.run_all(proc)
    assert [i.status for i in q.items] == [FAILED, DONE]
    assert "boom" in q.items[0].message
    q.clear_finished()
    assert q.items == [] or all(i.status == QUEUED for i in q.items)


def test_queue_cancel(wav_file):
    q = BatchQueue()
    q.add(wav_file)
    ev = threading.Event()
    ev.set()
    q.run_all(lambda p: None, cancel=ev)
    assert q.items[0].status == CANCELLED


def test_cannot_remove_running(wav_file):
    q = BatchQueue()
    q.add(wav_file).status = "running"
    with pytest.raises(ValueError):
        q.remove(0)


def test_history(tmp_path):
    h = History(tmp_path / "h.db")
    i = h.add("/a.wav", "htdemucs", "cpu", "balanced", "/out", "done", 12.5, "ok")
    h.add("/b.wav", "htdemucs_ft", "cuda", "best", "/out2", "failed", 1.0, "err")
    rows = h.list()
    assert [r["input_path"] for r in rows] == ["/b.wav", "/a.wav"]
    h.delete(i)
    assert len(h.list()) == 1
    h.clear()
    assert h.list() == []
    h.close()


def test_clean_preserves_shape_and_reduces_bleed():
    sr = 44100
    target = tone(sr, 2.0, 440, amp=0.5)
    leak = tone(sr, 2.0, 3000, amp=0.05)
    others = tone(sr, 2.0, 3000, amp=0.5)
    dirty = target + leak
    out = reduce_bleed(dirty, others, sr, strength=0.8)
    assert out.shape == dirty.shape and out.dtype == np.float32

    def band(x, f):
        spec = np.abs(np.fft.rfft(x[0]))
        k = int(f * x.shape[1] / sr)
        return spec[k - 3:k + 4].max()
    assert band(out, 3000) < 0.6 * band(dirty, 3000)
    assert band(out, 440) > 0.8 * band(dirty, 440)


def test_clean_zero_strength_is_identity_and_all_stems():
    x = tone(secs=0.5)
    np.testing.assert_allclose(reduce_bleed(x, x * 0.1, 44100, 0.0), x)
    out = clean_stems({"a": x, "b": x * 0.2}, 44100, 0.5)
    assert set(out) == {"a", "b"} and out["a"].shape == x.shape
