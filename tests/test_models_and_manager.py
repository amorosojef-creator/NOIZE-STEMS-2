import hashlib
import io
import threading

import pytest

from noize import model_manager as mm
from noize.models import MODELS, ModelInfo, get_model_info, select_model


def test_registry_contents():
    assert {"htdemucs", "htdemucs_ft", "htdemucs_6s"} <= set(MODELS)
    assert len(MODELS["htdemucs_6s"].stems) == 6
    assert MODELS["htdemucs"].stems[-1] == "vocals"


def test_unknown_model():
    with pytest.raises(KeyError):
        get_model_info("nope")


def test_select_model_fallbacks():
    assert select_model("htdemucs_ft", {"htdemucs_ft", "htdemucs"}) == "htdemucs_ft"
    assert select_model("htdemucs_ft", {"htdemucs"}) == "htdemucs"
    with pytest.raises(LookupError):
        select_model("htdemucs", set())


class FakeResp(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


@pytest.fixture
def fake_model(monkeypatch):
    payload = b"weights" * 5000
    digest = hashlib.sha256(payload).hexdigest()[:8]
    fname = f"deadbeef-{digest}.th"
    info = ModelInfo("fake", "Fake", ("a", "b"), "d", (fname,), 1)
    monkeypatch.setattr(mm, "get_model_info", lambda mid: info)
    return payload, fname


def test_consent_required(tmp_path, fake_model):
    m = mm.ModelManager(tmp_path)
    with pytest.raises(mm.ConsentRequired):
        m.install("fake", consent=False)
    assert not m.is_installed("fake")


def test_install_verifies_and_reports_progress(tmp_path, fake_model):
    payload, fname = fake_model
    m = mm.ModelManager(tmp_path, opener=lambda url, timeout=0: FakeResp(payload))
    seen = []
    m.install("fake", consent=True, progress=lambda f, s: seen.append(f))
    assert m.is_installed("fake") and m.verify("fake")
    assert seen[-1] == 1.0


def test_bad_checksum_is_discarded(tmp_path, fake_model):
    _, fname = fake_model
    m = mm.ModelManager(tmp_path, opener=lambda url, timeout=0: FakeResp(b"tampered"))
    with pytest.raises(mm.ChecksumError):
        m.install("fake", consent=True)
    assert not (tmp_path / fname).exists() and not list(tmp_path.glob("*.part"))


def test_cancel_download(tmp_path, fake_model):
    payload, _ = fake_model
    ev = threading.Event()
    ev.set()
    m = mm.ModelManager(tmp_path, opener=lambda url, timeout=0: FakeResp(payload))
    with pytest.raises(mm.DownloadCancelled):
        m.install("fake", consent=True, cancel=ev)
    assert not list(tmp_path.glob("*.part"))
