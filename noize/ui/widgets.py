"""Reusable NOIZE widgets: drop area, shared timeline, waveform lane, stem lane."""
from __future__ import annotations
import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QCheckBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSlider,
                               QVBoxLayout, QWidget)

from ..waveform import compute_peaks
from . import theme


def section_label(text: str) -> QLabel:
    lab = QLabel(text.upper())
    lab.setObjectName("h")
    return lab


def fmt_time(s: float) -> str:
    s = max(0.0, s)
    return f"{int(s // 60)}:{s % 60:05.2f}"


class DropArea(QFrame):
    files_dropped = Signal(list)
    clicked = Signal()

    def __init__(self, text="DROP AUDIO OR VIDEO FILES HERE\nclick to browse"):
        super().__init__()
        self.setObjectName("card")
        self.setAcceptDrops(True)
        self.setMinimumHeight(120)
        lab = QLabel(text)
        lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lab.setObjectName("dim")
        lay = QVBoxLayout(self)
        lay.addWidget(lab)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.files_dropped.emit(paths)

    def mousePressEvent(self, e):
        self.clicked.emit()


class Timeline(QObject):
    """Shared view state so all lanes zoom/scroll/play in sync."""
    changed = Signal()
    seek_requested = Signal(float)
    loop_changed = Signal(object, object)

    def __init__(self):
        super().__init__()
        self.duration = 0.0
        self.start = 0.0
        self.span = 1.0
        self.position = 0.0
        self.loop: tuple[float, float] | None = None

    def set_duration(self, d: float):
        self.duration, self.start, self.span, self.position, self.loop = d, 0.0, max(d, 0.001), 0.0, None
        self.changed.emit()

    def zoom(self, factor: float, anchor: float | None = None):
        if self.duration <= 0:
            return
        anchor = self.start + self.span / 2 if anchor is None else anchor
        new = float(np.clip(self.span * factor, 0.25, self.duration))
        rel = (anchor - self.start) / self.span if self.span else 0.5
        self.start = float(np.clip(anchor - rel * new, 0, max(0, self.duration - new)))
        self.span = new
        self.changed.emit()

    def fit(self):
        self.start, self.span = 0.0, max(self.duration, 0.001)
        self.changed.emit()

    def scroll(self, delta_s: float):
        self.start = float(np.clip(self.start + delta_s, 0, max(0, self.duration - self.span)))
        self.changed.emit()

    def set_position(self, p: float):
        self.position = p
        self.changed.emit()

    def set_loop(self, a, b):
        self.loop = (a, b) if a is not None and b is not None and b > a else None
        self.loop_changed.emit(*(self.loop or (None, None)))
        self.changed.emit()


class WaveformLane(QWidget):
    """Draws one waveform against the shared Timeline. Click = seek, Shift+drag = loop."""

    def __init__(self, timeline: Timeline, sr: int = 44100, height: int = 64):
        super().__init__()
        self.tl, self.sr = timeline, sr
        self.audio: np.ndarray | None = None
        self.norm = 1.0
        self.color = QColor(theme.ACCENT)
        self.setMinimumHeight(height)
        self._drag_from: float | None = None
        timeline.changed.connect(self.update)

    def set_audio(self, audio: np.ndarray | None, sr: int):
        self.audio, self.sr = audio, sr
        self.norm = float(np.max(np.abs(audio))) if audio is not None and audio.size else 1.0
        self.norm = self.norm or 1.0
        self.update()

    def _t(self, x: float) -> float:
        return self.tl.start + (x / max(1, self.width())) * self.tl.span

    def paintEvent(self, _):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(theme.BG))
        p.setPen(QPen(QColor(theme.LINE), 1))
        p.drawLine(0, h // 2, w, h // 2)
        tl = self.tl
        if self.audio is not None and tl.duration > 0:
            s0 = int(tl.start * self.sr)
            s1 = min(self.audio.shape[1], int((tl.start + tl.span) * self.sr))
            if s1 > s0:
                peaks = compute_peaks(self.audio[:, s0:s1], w) / self.norm
                p.setPen(QPen(self.color, 1))
                mid, amp = h / 2, h / 2 - 2
                step = w / max(1, len(peaks))
                for i, (lo, hi) in enumerate(peaks):
                    x = i * step
                    p.drawLine(QPointF(x, mid - hi * amp), QPointF(x, mid - lo * amp - (0 if hi - lo > 0.01 else -1)))
        if tl.loop:
            x0 = (tl.loop[0] - tl.start) / tl.span * w
            x1 = (tl.loop[1] - tl.start) / tl.span * w
            c = QColor(theme.ACCENT)
            c.setAlpha(36)
            p.fillRect(QRectF(x0, 0, x1 - x0, h), c)
        if tl.duration > 0:
            x = (tl.position - tl.start) / tl.span * w
            if 0 <= x <= w:
                p.setPen(QPen(QColor(theme.TEXT), 1))
                p.drawLine(int(x), 0, int(x), h)
        p.setPen(QPen(QColor(theme.LINE), 1))
        p.drawRect(0, 0, w - 1, h - 1)

    def mousePressEvent(self, e):
        if self.tl.duration <= 0:
            return
        t = self._t(e.position().x())
        if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self._drag_from = t
        else:
            self.tl.seek_requested.emit(t)

    def mouseMoveEvent(self, e):
        if self._drag_from is not None:
            t = self._t(e.position().x())
            self.tl.set_loop(min(self._drag_from, t), max(self._drag_from, t))

    def mouseReleaseEvent(self, e):
        self._drag_from = None

    def mouseDoubleClickEvent(self, e):
        self.tl.set_loop(None, None)

    def wheelEvent(self, e):
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.tl.zoom(0.8 if e.angleDelta().y() > 0 else 1.25, self._t(e.position().x()))
        else:
            self.tl.scroll(-e.angleDelta().y() / 120 * self.tl.span * 0.1)


class StemLane(QWidget):
    """Name, mute/solo, volume, pan and waveform for one stem."""
    mute_toggled = Signal(str, bool)
    solo_toggled = Signal(str, bool)
    volume_changed = Signal(str, float)
    pan_changed = Signal(str, float)

    def __init__(self, name: str, timeline: Timeline, audio: np.ndarray, sr: int):
        super().__init__()
        self.name = name
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        ctl = QWidget()
        ctl.setFixedWidth(176)
        cl = QVBoxLayout(ctl)
        cl.setContentsMargins(0, 2, 0, 2)
        cl.setSpacing(3)
        top = QHBoxLayout()
        lab = QLabel(name.upper())
        lab.setObjectName("h")
        self.m, self.s = QPushButton("M"), QPushButton("S")
        for b in (self.m, self.s):
            b.setObjectName("tog")
            b.setCheckable(True)
            b.setFixedHeight(20)
        self.m.setToolTip("Mute")
        self.s.setToolTip("Solo")
        top.addWidget(lab, 1)
        top.addWidget(self.m)
        top.addWidget(self.s)
        self.vol = QSlider(Qt.Orientation.Horizontal)
        self.vol.setRange(0, 150)
        self.vol.setValue(100)
        self.vol.setToolTip("Volume (double-click resets)")
        self.pan = QSlider(Qt.Orientation.Horizontal)
        self.pan.setRange(-100, 100)
        self.pan.setValue(0)
        self.pan.setToolTip("Pan")
        cl.addLayout(top)
        cl.addWidget(self.vol)
        cl.addWidget(self.pan)
        self.wave = WaveformLane(timeline, sr)
        self.wave.set_audio(audio, sr)
        lay.addWidget(ctl)
        lay.addWidget(self.wave, 1)
        self.m.toggled.connect(lambda v: self.mute_toggled.emit(name, v))
        self.s.toggled.connect(lambda v: self.solo_toggled.emit(name, v))
        self.vol.valueChanged.connect(lambda v: self.volume_changed.emit(name, v / 100))
        self.pan.valueChanged.connect(lambda v: self.pan_changed.emit(name, v / 100))
