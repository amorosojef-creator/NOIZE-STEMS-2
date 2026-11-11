"""Background workers. All heavy work runs off the Qt UI thread."""
from __future__ import annotations
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from .audio_io import load_audio
from .batch import BatchQueue
from .clean import clean_stems
from .model_manager import ModelManager
from .separation import Cancelled, DemucsSeparator


class _Worker(QThread):
    progress = Signal(float, str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self):
        super().__init__()
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()


class SeparationWorker(_Worker):
    finished_ok = Signal(object, float)     # SeparationResult, seconds

    def __init__(self, path: Path, model_id: str, models_dir: Path, device: str, quality: str,
                 clean_strength: float = 0.0):
        super().__init__()
        self.path, self.model_id, self.models_dir = path, model_id, models_dir
        self.device, self.quality, self.clean_strength = device, quality, clean_strength

    def run(self):
        t0 = time.time()
        try:
            self.progress.emit(0.0, "Decoding audio")
            audio, sr = load_audio(self.path)
            sep = DemucsSeparator(self.model_id, self.models_dir, self.device, self.quality)
            scale = 0.9 if self.clean_strength > 0 else 1.0
            res = sep.separate(audio, sr, lambda f, m: self.progress.emit(f * scale, m), self.cancel_event)
            if self.clean_strength > 0:
                self.progress.emit(0.92, "NOIZE CLEAN")
                res.stems = clean_stems(res.stems, res.sample_rate, self.clean_strength)
            self.progress.emit(1.0, "Done")
            self.finished_ok.emit(res, time.time() - t0)
        except Cancelled:
            self.cancelled.emit()
        except Exception as e:  # noqa: BLE001 - surfaced to the user
            self.failed.emit(str(e) or e.__class__.__name__)


class DownloadWorker(_Worker):
    finished_ok = Signal(str)

    def __init__(self, manager: ModelManager, model_id: str, consent: bool):
        super().__init__()
        self.manager, self.model_id, self.consent = manager, model_id, consent

    def run(self):
        from .model_manager import DownloadCancelled
        try:
            self.manager.install(self.model_id, self.consent,
                                 lambda f, m: self.progress.emit(f, m), self.cancel_event)
            self.finished_ok.emit(self.model_id)
        except DownloadCancelled:
            self.cancelled.emit()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"Download failed: {e}")


class BatchWorker(_Worker):
    item_updated = Signal(int, str, str)    # index, status, message
    item_done = Signal(str, str, float)     # path, status, seconds

    def __init__(self, queue: BatchQueue, process_one):
        super().__init__()
        self.queue, self.process_one = queue, process_one

    def run(self):
        def on_update(item):
            self.item_updated.emit(self.queue.items.index(item), item.status, item.message)
        self.queue.run_all(lambda p: self.process_one(p, self.cancel_event, self.progress.emit),
                           self.cancel_event, on_update)


class FuncWorker(_Worker):
    """Run an arbitrary callable(progress_cb, cancel_event) off the UI thread."""
    done = Signal(object)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(lambda f, m="": self.progress.emit(f, m), self.cancel_event))
        except Cancelled:
            self.cancelled.emit()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e) or e.__class__.__name__)
