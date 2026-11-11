"""Synchronized multi-stem playback.

All stems are mixed sample-accurately in ONE buffer that feeds a single audio
sink, so stems cannot drift apart. Mute/solo/volume/pan changes take effect on
the next buffer (~20 ms).
"""
from __future__ import annotations
import os

import numpy as np
from PySide6.QtCore import QIODevice, QObject, QTimer, Signal
from PySide6.QtMultimedia import QAudio, QAudioFormat, QAudioSink, QMediaDevices

from .mixing import MixState, prevent_clipping, render_mix


class _MixDevice(QIODevice):
    def __init__(self, engine: "PlaybackEngine"):
        super().__init__()
        self.engine = engine

    def readData(self, maxlen: int):
        return self.engine._render_bytes(maxlen)

    def writeData(self, data):
        return 0

    def bytesAvailable(self):
        return (1 << 20) + super().bytesAvailable()

    def isSequential(self):
        return True


class PlaybackEngine(QObject):
    position_changed = Signal(float)     # seconds
    state_changed = Signal(str)          # playing | paused | stopped

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stems: dict[str, np.ndarray] = {}
        self.sr = 44100
        self.mix = MixState()
        self.pos = 0                      # sample index (render cursor)
        self.loop: tuple[int, int] | None = None
        self.sink: QAudioSink | None = None
        self.device: _MixDevice | None = None
        self.state = "stopped"
        self.error = ""
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(lambda: self.position_changed.emit(self.position_seconds()))

    # ---- data ----
    def set_stems(self, stems: dict[str, np.ndarray], sr: int):
        self.stop()
        self.stems, self.sr = stems, sr
        self.mix = MixState.for_names(stems)
        self.pos = 0
        self._build_sink()

    def _build_sink(self):
        fmt = QAudioFormat()
        fmt.setSampleRate(self.sr)
        fmt.setChannelCount(2)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        self.sink, self.device = None, None
        if os.environ.get("NOIZE_NO_AUDIO"):      # headless CI / sandboxes without an audio server
            self.error = "Audio output disabled (NOIZE_NO_AUDIO)"
            return
        dev = QMediaDevices.defaultAudioOutput()
        if dev.isNull() or not dev.isFormatSupported(fmt):
            self.error = ("No audio output device found" if dev.isNull()
                          else f"Audio device does not support {self.sr} Hz stereo")
            return
        self.error = ""
        self.sink = QAudioSink(dev, fmt, self)
        self.sink.setBufferSize(int(self.sr * 4 * 0.12))   # ~120 ms predictable latency
        self.device = _MixDevice(self)
        self.device.open(QIODevice.OpenModeFlag.ReadOnly)

    @property
    def total(self) -> int:
        return next(iter(self.stems.values())).shape[1] if self.stems else 0

    def duration(self) -> float:
        return self.total / self.sr if self.sr else 0.0

    def position_seconds(self) -> float:
        """Playhead = render cursor minus audio still buffered in the sink."""
        if not self.sink or self.state != "playing":
            return self.pos / self.sr
        buffered = (self.sink.bufferSize() - self.sink.bytesFree()) // 4
        return max(0.0, (self.pos - buffered) / self.sr)

    # ---- rendering (called from the audio thread by Qt) ----
    def _render_bytes(self, maxlen: int) -> bytes:
        frames = max(0, maxlen // 4)
        if not self.stems or frames == 0:
            return b""
        out = np.zeros((2, 0), np.float32)
        remaining = frames
        chunks = []
        while remaining > 0:
            looping = bool(self.loop) and self.loop[1] > self.loop[0]
            if looping and self.pos >= self.loop[1]:
                self.pos = self.loop[0]                 # wrap to loop start
            end_limit = min(self.loop[1], self.total) if looping else self.total
            n = min(remaining, end_limit - self.pos)
            if n <= 0:
                break
            chunks.append(render_mix(self.stems, self.mix, self.pos, self.pos + n))
            self.pos += n
            remaining -= n
        if not chunks:
            QTimer.singleShot(0, self._ended)
            return b""
        out = prevent_clipping(np.concatenate(chunks, axis=1), 1.0)
        pcm = (np.clip(out, -1, 1).T * 32767).astype("<i2")
        return pcm.tobytes()

    def _ended(self):
        if self.state == "playing":
            self.stop()

    # ---- transport ----
    def play(self):
        if not self.stems or not self.sink:
            if self.stems and self.error:
                self.state_changed.emit("error:" + self.error)
            return
        if self.pos >= self.total:
            self.pos = 0
        if self.state == "paused":
            self.sink.resume()
        else:
            self.sink.start(self.device)
        self.state = "playing"
        self._timer.start()
        self.state_changed.emit(self.state)

    def pause(self):
        if self.sink and self.state == "playing":
            self.position_changed.emit(self.position_seconds())
            self.sink.suspend()
            self.state = "paused"
            self._timer.stop()
            self.state_changed.emit(self.state)

    def stop(self):
        if self.sink:
            self.sink.stop()
        self.pos = 0
        self.state = "stopped"
        self._timer.stop()
        self.position_changed.emit(0.0)
        self.state_changed.emit(self.state)

    def seek(self, seconds: float):
        was = self.state == "playing"
        if self.sink and was:
            self.sink.stop()
        self.pos = int(np.clip(seconds * self.sr, 0, max(0, self.total - 1)))
        self.position_changed.emit(self.pos / self.sr)
        if was:
            self.state = "stopped"
            self.play()

    def set_loop(self, start_s: float | None, end_s: float | None):
        if start_s is None or end_s is None or end_s <= start_s:
            self.loop = None
        else:
            self.loop = (int(start_s * self.sr), int(end_s * self.sr))
