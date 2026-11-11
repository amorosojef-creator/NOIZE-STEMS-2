"""Sequential batch queue with per-item status and error isolation."""
from __future__ import annotations
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .audio_io import AudioError, validate_input

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"


@dataclass
class QueueItem:
    path: Path
    status: str = QUEUED
    message: str = ""


class BatchQueue:
    def __init__(self):
        self.items: list[QueueItem] = []

    def add(self, path) -> QueueItem:
        p = validate_input(path)          # raises AudioError
        if any(i.path == p for i in self.items):
            raise AudioError(f"Already in queue: {p.name}")
        item = QueueItem(p)
        self.items.append(item)
        return item

    def add_many(self, paths) -> list[str]:
        """Add what is valid; return error strings for the rest."""
        errors = []
        for p in paths:
            try:
                self.add(p)
            except AudioError as e:
                errors.append(str(e))
        return errors

    def remove(self, index: int) -> None:
        if self.items[index].status == RUNNING:
            raise ValueError("Cannot remove an item that is running")
        del self.items[index]

    def move(self, index: int, new_index: int) -> None:
        it = self.items.pop(index)
        self.items.insert(max(0, min(new_index, len(self.items))), it)

    def clear_finished(self) -> None:
        self.items = [i for i in self.items if i.status in (QUEUED, RUNNING)]

    def pending(self) -> list[QueueItem]:
        return [i for i in self.items if i.status == QUEUED]

    def run_all(self, process: Callable[[Path], None], cancel: threading.Event | None = None,
                on_update: Callable[[QueueItem], None] | None = None) -> None:
        """Run pending items one by one. A failure never stops the rest."""
        for item in self.pending():
            if cancel is not None and cancel.is_set():
                item.status = CANCELLED
                if on_update:
                    on_update(item)
                continue
            item.status = RUNNING
            if on_update:
                on_update(item)
            try:
                process(item.path)
                item.status = DONE
            except Exception as e:  # noqa: BLE001 - isolate per item
                if e.__class__.__name__ == "Cancelled":
                    item.status = CANCELLED
                else:
                    item.status, item.message = FAILED, str(e)
            if on_update:
                on_update(item)
