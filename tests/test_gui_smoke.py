"""Offscreen GUI tests (no sound device or real model needed)."""
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from noize.history import History
from noize.model_manager import ModelManager
from noize.separation import SeparationResult
from noize.settings import Settings
from noize.ui import theme
from noize.ui.main_window import MainWindow
from conftest import tone

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(theme.STYLE)
    return a


@pytest.fixture
def win(app, tmp_path):
    w = MainWindow(Settings(tmp_path / "s.json"), History(tmp_path / "h.db"), ModelManager(tmp_path / "m"))
    yield w
    w.engine.stop()


def test_window_builds_with_all_pages(win):
    assert win.pages.count() == 5
    assert [b.text() for b in win.nav_group.buttons()] == ["EXTRACT", "BATCH", "HISTORY", "MODELS", "SETTINGS"]
    assert win.model_table.rowCount() == 3
    assert not win.export_btn.isEnabled()       # nothing to export yet


def test_style_respects_design_rules():
    import re
    assert "gradient" not in theme.STYLE
    for r in re.findall(r"border-radius:\s*(\d+)px", theme.STYLE):
        assert int(r) <= 3


def test_load_result_builds_lanes_and_selection(win, tmp_path):
    stems = {n: tone(freq=f, secs=1.0) for n, f in (("drums", 100), ("bass", 60), ("other", 300), ("vocals", 800))}
    win._load_result(SeparationResult(stems, 44100, "htdemucs", "cpu"))
    assert len(win.lanes) == 4
    assert set(win.stem_checks) == {"drums", "bass", "other", "vocals", "instrumental"}
    assert win.export_btn.isEnabled()
    win.lanes[0].m.setChecked(True)
    assert win.engine.mix.stems["drums"].mute
    win.lanes[1].s.setChecked(True)
    assert win.engine.mix.stems["bass"].solo
    # synchronized: every lane shares the single timeline object
    assert all(l.wave.tl is win.timeline for l in win.lanes)
    win.timeline.zoom(0.5)
    assert win.timeline.span < 1.0


def test_render_bytes_alignment_and_loop(win):
    st = {"a": tone(secs=1.0), "b": tone(secs=1.0, freq=200)}
    win.engine.set_stems(st, 44100)
    b = win.engine._render_bytes(4 * 1000)
    assert len(b) == 4000 and win.engine.pos == 1000
    win.engine.set_loop(0.0, 0.01)           # 441 samples
    win.engine.pos = 0
    out = win.engine._render_bytes(4 * 1000)
    assert len(out) == 4000 and win.engine.pos <= 441    # wrapped inside loop
    win.engine.mix.stems["a"].mute = True
    win.engine.mix.stems["b"].mute = True
    win.engine.set_loop(None, None)
    z = np.frombuffer(win.engine._render_bytes(400), dtype="<i2")
    assert not z.any()


def test_screenshot(win, tmp_path):
    stems = {n: tone(freq=f, secs=3.0) * np.hanning(132300) for n, f in (("drums", 100), ("bass", 60), ("other", 300), ("vocals", 800))}
    win._load_result(SeparationResult({k: v.astype(np.float32) for k, v in stems.items()}, 44100, "htdemucs", "cpu"))
    win.show()
    win.grab().save(str(tmp_path / "shot.png"))
    assert (tmp_path / "shot.png").stat().st_size > 5000
