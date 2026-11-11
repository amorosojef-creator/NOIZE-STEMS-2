"""Local processing history (SQLite)."""
from __future__ import annotations
import sqlite3
import threading
import time
from pathlib import Path

from .paths import app_data_dir


class History:
    def __init__(self, db_path: Path | None = None):
        self.path = Path(db_path) if db_path else app_data_dir() / "history.db"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""CREATE TABLE IF NOT EXISTS runs(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, input_path TEXT, model TEXT,
            device TEXT, quality TEXT, output_dir TEXT, status TEXT, seconds REAL, message TEXT)""")
        self._conn.commit()

    def add(self, input_path, model, device, quality, output_dir, status, seconds=0.0, message="") -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO runs(ts,input_path,model,device,quality,output_dir,status,seconds,message)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                (time.time(), str(input_path), model, device, quality, str(output_dir), status, seconds, message))
            self._conn.commit()
            return int(cur.lastrowid)

    def list(self, limit: int = 500) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def delete(self, run_id: int) -> None:
        self._conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM runs")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
