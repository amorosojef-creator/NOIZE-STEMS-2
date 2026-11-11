"""NOIZE main window: EXTRACT / BATCH / HISTORY / MODELS / SETTINGS."""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QFileDialog,
                               QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
                               QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
                               QSlider, QStackedWidget, QTableWidget, QTableWidgetItem, QTextEdit,
                               QVBoxLayout, QWidget)

from .. import __version__
from ..audio_io import AUDIO_EXT, SUPPORTED_EXT, AudioError, find_ffmpeg, load_audio, validate_input
from ..batch import BatchQueue
from ..clean import clean_stems
from ..export import export_mix, export_stems, mp3_available, output_dir_for
from ..history import History
from ..model_manager import ModelManager
from ..models import MODELS
from ..paths import asset, models_dir
from ..separation import DemucsSeparator, detect_gpu, vocals_instrumental
from ..settings import Settings
from ..playback import PlaybackEngine
from ..workers import BatchWorker, DownloadWorker, FuncWorker, SeparationWorker
from . import theme
from .widgets import DropArea, StemLane, Timeline, WaveformLane, fmt_time, section_label

FILE_FILTER = "Audio/Video (" + " ".join("*" + e for e in sorted(SUPPORTED_EXT)) + ");;All files (*)"


def default_output_dir() -> Path:
    return Path.home() / "Music" / "NOIZE"


def open_folder(path) -> None:
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def panel(layout_cls=QVBoxLayout, margins=(12, 12, 12, 12), spacing=8) -> tuple[QWidget, object]:
    w = QWidget()
    w.setObjectName("panel")
    lay = layout_cls(w)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    return w, lay


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings | None = None, history: History | None = None,
                 manager: ModelManager | None = None):
        super().__init__()
        self.settings = settings or Settings()
        self.history = history or History()
        self.manager = manager or ModelManager(models_dir())
        self.queue = BatchQueue()
        self.timeline = Timeline()
        self.engine = PlaybackEngine(self)
        self.result = None          # active SeparationResult (possibly cleaned)
        self.raw_result = None      # original, un-cleaned result
        self.result_b = None        # Compare: second model
        self.showing_b = False
        self.current_path: Path | None = None
        self.preview = None         # (path, audio, sr)
        self._workers: set = set()
        self._sep_worker = None
        self.lanes: list[StemLane] = []
        self.gpu = {"cuda": False, "name": None, "vram_mb": None}

        self.setWindowTitle("NOIZE")
        self.setWindowIcon(QIcon(str(asset("NOIZE_app_icon.ico"))))
        self.setMinimumSize(1240, 740)
        self.resize(1440, 860)

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._build_topbar())
        self.pages = QStackedWidget()
        v.addWidget(self.pages, 1)
        self.pages.addWidget(self._build_extract())
        self.pages.addWidget(self._build_batch())
        self.pages.addWidget(self._build_history())
        self.pages.addWidget(self._build_models())
        self.pages.addWidget(self._build_settings())

        self.timeline.seek_requested.connect(self.engine.seek)
        self.timeline.loop_changed.connect(self._loop_changed)
        self.engine.position_changed.connect(self._on_position)
        self.engine.state_changed.connect(self._on_state)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self._toggle_play)

        self._refresh_models()
        self._probe_system()

    # ------------------------------------------------------------ helpers
    def _run_func(self, fn, on_done=None, on_fail=None, on_progress=None):
        w = FuncWorker(fn)
        self._workers.add(w)
        if on_done:
            w.done.connect(on_done)
        w.failed.connect(on_fail or self._error)
        if on_progress:
            w.progress.connect(on_progress)
        w.finished.connect(lambda: self._workers.discard(w))
        w.start()
        return w

    def _error(self, msg: str):
        self.status.setText(msg)
        QMessageBox.warning(self, "NOIZE", msg)

    def _info(self, msg: str):
        self.status.setText(msg)

    # ------------------------------------------------------------ top bar
    def _build_topbar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("topbar")
        h = QHBoxLayout(bar)
        h.setContentsMargins(16, 0, 16, 0)
        h.setSpacing(0)
        emb = QLabel()
        pm = QPixmap(str(asset("NOIZE_emblem_transparent.png")))
        emb.setPixmap(pm.scaled(34, 34, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        brand = QLabel("NOIZE")
        brand.setObjectName("brand")
        h.addWidget(emb)
        h.addSpacing(8)
        h.addWidget(brand)
        h.addSpacing(32)
        self.nav_group = QButtonGroup(self)
        for i, name in enumerate(["EXTRACT", "BATCH", "HISTORY", "MODELS", "SETTINGS"]):
            b = QPushButton(name)
            b.setObjectName("nav")
            b.setCheckable(True)
            b.setChecked(i == 0)
            self.nav_group.addButton(b, i)
            h.addWidget(b)
        self.nav_group.idClicked.connect(self._go_page)
        h.addStretch(1)
        ver = QLabel(f"v{__version__}")
        ver.setObjectName("dim")
        h.addWidget(ver)
        return bar

    def _go_page(self, i: int):
        self.pages.setCurrentIndex(i)
        if i == 2:
            self._refresh_history()
        if i == 3:
            self._refresh_models()
        if i == 1:
            self._refresh_queue()

    # ------------------------------------------------------------ EXTRACT page
    def _build_extract(self) -> QWidget:
        page = QWidget()
        h = QHBoxLayout(page)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        h.addWidget(self._build_left())
        h.addWidget(self._build_center(), 1)
        h.addWidget(self._build_right())
        return page

    def _build_left(self) -> QWidget:
        w, lay = panel()
        w.setFixedWidth(280)
        lay.addWidget(section_label("Queue"))
        self.qlist = QListWidget()
        self.qlist.setAcceptDrops(True)
        self.qlist.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.qlist.currentRowChanged.connect(self._queue_selected)
        lay.addWidget(self.qlist, 1)
        row = QHBoxLayout()
        add, rem = QPushButton("+ ADD FILES"), QPushButton("REMOVE")
        add.clicked.connect(self._add_files_dialog)
        rem.clicked.connect(self._remove_selected)
        row.addWidget(add)
        row.addWidget(rem)
        lay.addLayout(row)
        lay.addSpacing(6)
        lay.addWidget(section_label("Model"))
        self.model_combo = QComboBox()
        lay.addWidget(self.model_combo)
        lay.addWidget(section_label("Processor"))
        self.device_combo = QComboBox()
        for label, val in (("Auto (GPU if available)", "auto"), ("CPU", "cpu"), ("CUDA GPU", "cuda")):
            self.device_combo.addItem(label, val)
        self.device_combo.setCurrentIndex(max(0, self.device_combo.findData(self.settings.get("device"))))
        self.device_combo.currentIndexChanged.connect(lambda: self.settings.set("device", self.device_combo.currentData()))
        lay.addWidget(self.device_combo)
        self.gpu_label = QLabel("Detecting GPU...")
        self.gpu_label.setObjectName("dim")
        self.gpu_label.setWordWrap(True)
        lay.addWidget(self.gpu_label)
        lay.addWidget(section_label("Quality"))
        self.quality_combo = QComboBox()
        for label, val in (("Fast", "fast"), ("Balanced", "balanced"), ("Best (slower)", "best")):
            self.quality_combo.addItem(label, val)
        self.quality_combo.setCurrentIndex(max(0, self.quality_combo.findData(self.settings.get("quality"))))
        self.quality_combo.currentIndexChanged.connect(lambda: self.settings.set("quality", self.quality_combo.currentData()))
        lay.addWidget(self.quality_combo)
        self.extract_btn = QPushButton("EXTRACT STEMS")
        self.extract_btn.setObjectName("primary")
        self.extract_btn.setMinimumHeight(34)
        self.extract_btn.clicked.connect(self._extract_clicked)
        lay.addWidget(self.extract_btn)
        self.status = QLabel("Ready")
        self.status.setObjectName("dim")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        return w

    def _build_center(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(8)
        self.drop = DropArea()
        self.drop.files_dropped.connect(self._add_paths)
        self.drop.clicked.connect(self._add_files_dialog)
        v.addWidget(self.drop)
        info = QHBoxLayout()
        self.file_label = QLabel("No file loaded")
        self.file_label.setObjectName("title")
        self.dur_label = QLabel("")
        self.dur_label.setObjectName("dim")
        info.addWidget(self.file_label, 1)
        info.addWidget(self.dur_label)
        v.addLayout(info)
        v.addWidget(section_label("Overview"))
        self.overview = WaveformLane(self.timeline, height=52)
        v.addWidget(self.overview)
        prow = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("dim")
        self.cancel_btn = QPushButton("CANCEL")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        prow.addWidget(self.progress, 1)
        prow.addWidget(self.progress_label)
        prow.addWidget(self.cancel_btn)
        v.addLayout(prow)
        v.addWidget(section_label("Stems"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.lanes_host = QWidget()
        self.lanes_layout = QVBoxLayout(self.lanes_host)
        self.lanes_layout.setContentsMargins(0, 0, 0, 0)
        self.lanes_layout.setSpacing(4)
        self.lanes_layout.addStretch(1)
        scroll.setWidget(self.lanes_host)
        v.addWidget(scroll, 1)
        v.addLayout(self._build_transport())
        return w

    def _build_transport(self) -> QHBoxLayout:
        t = QHBoxLayout()
        self.play_btn, self.stop_btn = QPushButton("PLAY"), QPushButton("STOP")
        self.loop_btn = QPushButton("LOOP")
        self.loop_btn.setCheckable(True)
        self.loop_btn.setObjectName("tog")
        zo, zi, zf = QPushButton("-"), QPushButton("+"), QPushButton("FIT")
        for b in (zo, zi):
            b.setObjectName("sq")
            b.setFixedWidth(30)
        self.play_btn.clicked.connect(self._toggle_play)
        self.stop_btn.clicked.connect(self.engine.stop)
        self.loop_btn.toggled.connect(lambda _: self._apply_loop())
        zo.clicked.connect(lambda: self.timeline.zoom(1.5))
        zi.clicked.connect(lambda: self.timeline.zoom(1 / 1.5))
        zf.clicked.connect(self.timeline.fit)
        self.scrub = QSlider(Qt.Orientation.Horizontal)
        self.scrub.setRange(0, 1000)
        self.scrub.sliderMoved.connect(lambda v: self.engine.seek(v / 1000 * self.engine.duration()))
        self.time_label = QLabel("0:00.00 / 0:00.00")
        self.time_label.setObjectName("dim")
        for wdg in (self.play_btn, self.stop_btn, self.loop_btn):
            t.addWidget(wdg)
        t.addWidget(self.scrub, 1)
        t.addWidget(self.time_label)
        for wdg in (zo, zi, zf):
            t.addWidget(wdg)
        self.play_btn.setToolTip("Space")
        self.loop_btn.setToolTip("Shift+drag on a waveform sets the loop region; double-click clears it. Ctrl+wheel zooms.")
        return t

    def _build_right(self) -> QWidget:
        w, lay = panel()
        w.setFixedWidth(290)
        lay.addWidget(section_label("Stem selection"))
        self.stem_boxes_host = QWidget()
        self.stem_boxes = QVBoxLayout(self.stem_boxes_host)
        self.stem_boxes.setContentsMargins(0, 0, 0, 0)
        self.stem_boxes.setSpacing(2)
        lay.addWidget(self.stem_boxes_host)
        lay.addWidget(section_label("Export"))
        self.fmt_combo = QComboBox()
        for f in ("wav", "flac", "mp3"):
            self.fmt_combo.addItem(f.upper(), f)
        self.fmt_combo.setCurrentIndex(max(0, self.fmt_combo.findData(self.settings.get("export_format"))))
        self.sr_combo = QComboBox()
        for sr in (44100, 48000, 88200, 96000):
            self.sr_combo.addItem(f"{sr / 1000:g} kHz", sr)
        self.sr_combo.setCurrentIndex(max(0, self.sr_combo.findData(self.settings.get("export_sample_rate"))))
        self.depth_combo = QComboBox()
        for b in (16, 24):
            self.depth_combo.addItem(f"{b}-bit PCM", b)
        self.depth_combo.setCurrentIndex(max(0, self.depth_combo.findData(self.settings.get("export_bit_depth"))))
        for lab, cb, key in (("Format", self.fmt_combo, "export_format"), ("Sample rate", self.sr_combo, "export_sample_rate"),
                             ("Bit depth", self.depth_combo, "export_bit_depth")):
            lay.addWidget(QLabel(lab))
            lay.addWidget(cb)
            cb.currentIndexChanged.connect(lambda _=0, c=cb, k=key: self.settings.set(k, c.currentData()))
        self.mp3_note = QLabel("")
        self.mp3_note.setObjectName("dim")
        lay.addWidget(self.mp3_note)
        lay.addWidget(QLabel("Output folder"))
        orow = QHBoxLayout()
        self.out_edit = QLineEdit(self.settings.get("output_dir") or str(default_output_dir()))
        self.out_edit.editingFinished.connect(lambda: self.settings.set("output_dir", self.out_edit.text()))
        br = QPushButton("...")
        br.setObjectName("sq")
        br.setFixedWidth(30)
        br.clicked.connect(self._browse_out)
        orow.addWidget(self.out_edit, 1)
        orow.addWidget(br)
        lay.addLayout(orow)
        self.export_btn = QPushButton("EXPORT STEMS")
        self.export_btn.setObjectName("primary")
        self.export_btn.setMinimumHeight(32)
        self.export_btn.clicked.connect(self._export_stems)
        self.mix_btn = QPushButton("EXPORT CUSTOM MIX")
        self.mix_btn.clicked.connect(self._export_mix)
        lay.addWidget(self.export_btn)
        lay.addWidget(self.mix_btn)
        lay.addSpacing(4)
        lay.addWidget(section_label("Clean"))
        self.clean_slider = QSlider(Qt.Orientation.Horizontal)
        self.clean_slider.setRange(0, 100)
        self.clean_slider.setValue(int(self.settings.get("clean_strength") * 100))
        crow = QHBoxLayout()
        self.clean_apply = QPushButton("APPLY CLEAN")
        self.clean_restore = QPushButton("RESTORE")
        self.clean_apply.clicked.connect(self._apply_clean)
        self.clean_restore.clicked.connect(self._restore_original)
        crow.addWidget(self.clean_apply)
        crow.addWidget(self.clean_restore)
        lay.addWidget(self.clean_slider)
        lay.addLayout(crow)
        note = QLabel("Reduces bleed/artifacts with spectral masking. Not perfect removal.")
        note.setObjectName("dim")
        note.setWordWrap(True)
        lay.addWidget(note)
        lay.addSpacing(4)
        lay.addWidget(section_label("Compare"))
        self.cmp_combo = QComboBox()
        self.cmp_btn = QPushButton("PROCESS WITH THIS MODEL")
        self.ab_btn = QPushButton("A / B: A")
        self.ab_btn.setEnabled(False)
        self.cmp_btn.clicked.connect(self._run_compare)
        self.ab_btn.clicked.connect(self._toggle_ab)
        lay.addWidget(self.cmp_combo)
        lay.addWidget(self.cmp_btn)
        lay.addWidget(self.ab_btn)
        lay.addStretch(1)
        remix = QLabel("REMIX: use M / S / volume / pan on each stem lane, then EXPORT CUSTOM MIX.")
        remix.setObjectName("dim")
        remix.setWordWrap(True)
        lay.addWidget(remix)
        self._set_result_controls(False)
        return w

    # ------------------------------------------------------------ BATCH page
    def _build_batch(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(20, 16, 20, 16)
        v.addWidget(section_label("Batch queue - sequential processing, stems exported automatically"))
        self.batch_table = QTableWidget(0, 3)
        self.batch_table.setHorizontalHeaderLabels(["FILE", "STATUS", "MESSAGE"])
        self.batch_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.batch_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.batch_table.verticalHeader().hide()
        self.batch_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.batch_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        v.addWidget(self.batch_table, 1)
        row = QHBoxLayout()
        self.b_add, self.b_rem = QPushButton("+ ADD FILES"), QPushButton("REMOVE")
        self.b_up, self.b_down = QPushButton("UP"), QPushButton("DOWN")
        self.b_clear = QPushButton("CLEAR FINISHED")
        self.b_start = QPushButton("START BATCH")
        self.b_start.setObjectName("primary")
        self.b_cancel = QPushButton("CANCEL")
        self.b_cancel.setEnabled(False)
        for b in (self.b_add, self.b_rem, self.b_up, self.b_down, self.b_clear):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(self.b_cancel)
        row.addWidget(self.b_start)
        v.addLayout(row)
        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 1000)
        v.addWidget(self.batch_progress)
        self.b_add.clicked.connect(self._add_files_dialog)
        self.b_rem.clicked.connect(self._batch_remove)
        self.b_up.clicked.connect(lambda: self._batch_move(-1))
        self.b_down.clicked.connect(lambda: self._batch_move(1))
        self.b_clear.clicked.connect(lambda: (self.queue.clear_finished(), self._refresh_queue()))
        self.b_start.clicked.connect(self._batch_start)
        self.b_cancel.clicked.connect(self._cancel_batch)
        self._batch_worker = None
        return w

    # ------------------------------------------------------------ HISTORY page
    def _build_history(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(20, 16, 20, 16)
        v.addWidget(section_label("Processing history (stored locally)"))
        self.hist_table = QTableWidget(0, 7)
        self.hist_table.setHorizontalHeaderLabels(["WHEN", "FILE", "MODEL", "DEVICE", "STATUS", "SECONDS", "OUTPUT"])
        self.hist_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.hist_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.hist_table.verticalHeader().hide()
        self.hist_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.hist_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        v.addWidget(self.hist_table, 1)
        row = QHBoxLayout()
        o, d, c = QPushButton("OPEN OUTPUT FOLDER"), QPushButton("DELETE ENTRY"), QPushButton("CLEAR ALL")
        o.clicked.connect(self._hist_open)
        d.clicked.connect(self._hist_delete)
        c.clicked.connect(lambda: (self.history.clear(), self._refresh_history()))
        for b in (o, d, c):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        self._hist_rows: list[dict] = []
        return w

    # ------------------------------------------------------------ MODELS page
    def _build_models(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(20, 16, 20, 16)
        v.addWidget(section_label("Separation models - weights are downloaded only with your permission"))
        self.model_table = QTableWidget(0, 4)
        self.model_table.setHorizontalHeaderLabels(["MODEL", "STEMS", "SIZE", "STATUS"])
        self.model_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.model_table.verticalHeader().hide()
        self.model_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.model_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.model_table.currentCellChanged.connect(lambda *_: self._model_selected())
        v.addWidget(self.model_table)
        row = QHBoxLayout()
        self.m_install, self.m_verify, self.m_remove = QPushButton("INSTALL"), QPushButton("VERIFY"), QPushButton("REMOVE")
        self.m_install.setObjectName("primary")
        self.m_cancel = QPushButton("CANCEL")
        self.m_cancel.setEnabled(False)
        for b in (self.m_install, self.m_verify, self.m_remove, self.m_cancel):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        self.m_progress = QProgressBar()
        self.m_progress.setRange(0, 1000)
        v.addWidget(self.m_progress)
        self.m_info = QTextEdit()
        self.m_info.setReadOnly(True)
        self.m_info.setStyleSheet(f"QTextEdit {{ background: {theme.BG}; border: 1px solid {theme.LINE}; }}")
        v.addWidget(self.m_info, 1)
        self.m_install.clicked.connect(lambda: self._install_model(self._selected_model_id()))
        self.m_verify.clicked.connect(self._verify_model)
        self.m_remove.clicked.connect(self._remove_model)
        self.m_cancel.clicked.connect(lambda: self._dl_worker and self._dl_worker.cancel())
        self._dl_worker = None
        return w

    # ------------------------------------------------------------ SETTINGS page
    def _build_settings(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(10)
        head = QHBoxLayout()
        emb = QLabel()
        emb.setPixmap(QPixmap(str(asset("NOIZE_emblem_transparent.png"))).scaled(
            72, 72, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        t = QLabel("NOIZE")
        t.setObjectName("brand")
        sub = QLabel(f"AI Stem Extractor  -  v{__version__}  -  free, offline-capable")
        sub.setObjectName("dim")
        col = QVBoxLayout()
        col.addWidget(t)
        col.addWidget(sub)
        head.addWidget(emb)
        head.addLayout(col)
        head.addStretch(1)
        v.addLayout(head)
        v.addWidget(section_label("Output"))
        self.organize_cb = QCheckBox("Put each track's stems in its own subfolder")
        self.organize_cb.setChecked(self.settings.get("organize_by_track"))
        self.organize_cb.toggled.connect(lambda x: self.settings.set("organize_by_track", x))
        v.addWidget(self.organize_cb)
        self.autoclean_cb = QCheckBox("Apply NOIZE CLEAN automatically during extraction and batch")
        self.autoclean_cb.setChecked(self.settings.get("clean_enabled"))
        self.autoclean_cb.toggled.connect(lambda x: self.settings.set("clean_enabled", x))
        v.addWidget(self.autoclean_cb)
        reset = QPushButton("RESET SETTINGS TO DEFAULT")
        reset.setFixedWidth(240)
        reset.clicked.connect(self._reset_settings)
        v.addWidget(reset)
        v.addWidget(section_label("System"))
        self.sys_label = QLabel("")
        self.sys_label.setObjectName("dim")
        self.sys_label.setWordWrap(True)
        v.addWidget(self.sys_label)
        v.addWidget(section_label("About and licenses"))
        lic = QTextEdit()
        lic.setReadOnly(True)
        lic.setStyleSheet(f"QTextEdit {{ background: {theme.BG}; border: 1px solid {theme.LINE}; }}")
        lic.setPlainText(
            "NOIZE source code: MIT License.\n\n"
            "Third-party libraries: PySide6 / Qt (LGPLv3), PyTorch (BSD-3), Demucs (MIT), NumPy (BSD), "
            "SciPy (BSD), SoundFile / libsndfile (BSD / LGPL), PyInstaller (GPL with bootloader exception), "
            "psutil (BSD), FFmpeg (LGPL/GPL depending on build).\n\n"
            "Model weights are NOT bundled. They are downloaded on request from Meta's Demucs release "
            "and are governed by the Demucs repository license (MIT) - see docs/LICENSES.md.\n\n"
            "NOIZE does not promise perfect separation; results depend on the track and model.")
        v.addWidget(lic, 1)
        return w

    # ------------------------------------------------------------ system / models
    def _probe_system(self):
        def probe(_p, _c):
            return detect_gpu()
        self._run_func(probe, self._gpu_ready, on_fail=lambda m: self.gpu_label.setText("GPU detection unavailable"))

    def _gpu_ready(self, info):
        self.gpu = info
        txt = (f"CUDA: {info['name']} ({info['vram_mb']} MB)" if info["cuda"] else "No CUDA GPU detected - CPU mode")
        self.gpu_label.setText(txt)
        ff = find_ffmpeg()
        self.sys_label.setText(f"{txt}\nFFmpeg: {ff or 'not found (needed for M4A/video input and MP3 export)'}")
        self.mp3_note.setText("" if ff else "MP3 needs FFmpeg (not found)")

    def _refresh_models(self):
        installed = self.manager.installed_ids()
        cur = self.settings.get("model")
        self.model_combo.blockSignals(True)
        self.cmp_combo.clear()
        self.model_combo.clear()
        for mid, m in MODELS.items():
            tag = "" if mid in installed else "  (not installed)"
            self.model_combo.addItem(m.name + tag, mid)
            self.cmp_combo.addItem(m.name + tag, mid)
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(cur)))
        self.model_combo.blockSignals(False)
        if not getattr(self, "_model_combo_wired", False):
            self._model_combo_wired = True
            self.model_combo.currentIndexChanged.connect(
                lambda: self.settings.set("model", self.model_combo.currentData()))
        self.model_table.setRowCount(len(MODELS))
        for r, (mid, m) in enumerate(MODELS.items()):
            vals = [m.name, ", ".join(m.stems), f"~{m.approx_mb} MB", "Installed" if mid in installed else "Not installed"]
            for c, val in enumerate(vals):
                it = QTableWidgetItem(val)
                it.setData(Qt.ItemDataRole.UserRole, mid)
                self.model_table.setItem(r, c, it)
        if self.model_table.rowCount() and self.model_table.currentRow() < 0:
            self.model_table.selectRow(0)
        self._model_selected()

    def _selected_model_id(self) -> str:
        r = max(0, self.model_table.currentRow())
        return self.model_table.item(r, 0).data(Qt.ItemDataRole.UserRole)

    def _model_selected(self):
        if self.model_table.rowCount() == 0 or self.model_table.item(0, 0) is None:
            return
        m = MODELS[self._selected_model_id()]
        self.m_info.setPlainText(f"{m.name}\n\n{m.description}\n\nStems: {', '.join(m.stems)}\n"
                                 f"Files: {', '.join(m.files)}\nSource: dl.fbaipublicfiles.com (Meta)\n"
                                 f"Integrity: SHA-256 prefix in each file name is verified after download.\n\n"
                                 f"License: {m.license}")

    def _install_model(self, mid: str, then=None) -> bool:
        m = MODELS[mid]
        if self.manager.is_installed(mid):
            if then:
                then()
            return True
        ans = QMessageBox.question(
            self, "Download model",
            f"NOIZE needs to download '{m.name}' (about {m.approx_mb} MB) from Meta's Demucs release "
            f"(dl.fbaipublicfiles.com). Nothing else is uploaded.\n\nDownload now?")
        if ans != QMessageBox.StandardButton.Yes:
            return False
        w = DownloadWorker(self.manager, mid, True)
        self._dl_worker = w
        self._workers.add(w)
        self.m_cancel.setEnabled(True)
        self.m_install.setEnabled(False)
        w.progress.connect(lambda f, s: (self.m_progress.setValue(int(f * 1000)), self._info(f"Downloading {m.name}: {f * 100:.0f}%")))

        def done(_):
            self.m_cancel.setEnabled(False)
            self.m_install.setEnabled(True)
            self.m_progress.setValue(1000)
            self._refresh_models()
            self._info(f"{m.name} installed")
            if then:
                then()

        def fail(msg):
            self.m_cancel.setEnabled(False)
            self.m_install.setEnabled(True)
            self._error(msg)

        def cancelled():
            self.m_cancel.setEnabled(False)
            self.m_install.setEnabled(True)
            self._info("Download cancelled")

        w.finished_ok.connect(done)
        w.failed.connect(fail)
        w.cancelled.connect(cancelled)
        w.finished.connect(lambda: self._workers.discard(w))
        w.start()
        return True

    def _verify_model(self):
        mid = self._selected_model_id()
        if not self.manager.is_installed(mid):
            return self._info("Model is not installed")
        self._run_func(lambda p, c: self.manager.verify(mid),
                       lambda ok: self._info("Checksum OK" if ok else "Checksum FAILED - remove and reinstall"))

    def _remove_model(self):
        mid = self._selected_model_id()
        if QMessageBox.question(self, "Remove model", f"Delete downloaded weights for {MODELS[mid].name}?") == QMessageBox.StandardButton.Yes:
            self.manager.remove(mid)
            self._refresh_models()

    def _reset_settings(self):
        from ..settings import DEFAULTS
        for k, v in DEFAULTS.items():
            self.settings.set(k, v, save=False)
        self.settings.save()
        self._info("Settings reset - restart NOIZE to refresh all controls")

    # ------------------------------------------------------------ queue
    def _add_files_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Add audio or video", self.settings.get("last_input_dir"), FILE_FILTER)
        if files:
            self.settings.set("last_input_dir", str(Path(files[0]).parent))
            self._add_paths(files)

    def _add_paths(self, paths):
        errors = self.queue.add_many(paths)
        self._refresh_queue()
        if self.qlist.count() and self.qlist.currentRow() < 0:
            self.qlist.setCurrentRow(self.qlist.count() - 1)
        if errors:
            QMessageBox.warning(self, "Some files were skipped", "\n".join(errors[:8]))
        else:
            self._info(f"{len(self.queue.items)} file(s) in queue")

    def _remove_selected(self):
        r = self.qlist.currentRow()
        if r >= 0:
            try:
                self.queue.remove(r)
            except ValueError as e:
                return self._error(str(e))
            self._refresh_queue()

    def _refresh_queue(self):
        cur = self.qlist.currentRow()
        self.qlist.blockSignals(True)
        self.qlist.clear()
        for it in self.queue.items:
            self.qlist.addItem(f"{it.path.name}   [{it.status}]")
        self.qlist.setCurrentRow(min(cur, self.qlist.count() - 1) if cur >= 0 else -1)
        self.qlist.blockSignals(False)
        self.batch_table.setRowCount(len(self.queue.items))
        for r, it in enumerate(self.queue.items):
            for c, val in enumerate([it.path.name, it.status.upper(), it.message]):
                self.batch_table.setItem(r, c, QTableWidgetItem(val))

    def _queue_selected(self, row: int):
        if row < 0 or row >= len(self.queue.items):
            return
        path = self.queue.items[row].path
        self.current_path = path
        self.file_label.setText(path.name)
        self.dur_label.setText("decoding...")

        def decode(_p, _c):
            a, sr = load_audio(path)
            return path, a, sr

        def ready(res):
            p, a, sr = res
            if p != self.current_path:
                return
            self.preview = res
            self.dur_label.setText(f"{fmt_time(a.shape[1] / sr)}  -  {sr / 1000:g} kHz  -  {a.shape[0]} ch")
            if self.result is None:
                self.timeline.set_duration(a.shape[1] / sr)
                self.overview.set_audio(a, sr)

        self._run_func(decode, ready, on_fail=lambda m: (self.dur_label.setText("cannot decode"), self._info(m)))

    # ------------------------------------------------------------ extraction
    def _extract_clicked(self):
        if self._sep_worker is not None:
            return
        if self.current_path is None:
            return self._add_files_dialog()
        self._start_extraction(self.model_combo.currentData(), compare=False)

    def _start_extraction(self, model_id: str, compare: bool):
        if self.current_path is None:
            return self._info("Add a file first")
        if not self.manager.is_installed(model_id):
            self._install_model(model_id, then=lambda: self._start_extraction(model_id, compare))
            return
        clean = 0.0 if compare else (self.clean_slider.value() / 100 if self.settings.get("clean_enabled") else 0.0)
        w = SeparationWorker(self.current_path, model_id, self.manager.dir, self.settings.get("device"),
                             self.settings.get("quality"), clean)
        self._sep_worker = w
        self._workers.add(w)
        self.extract_btn.setEnabled(False)
        self.cmp_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.engine.stop()
        w.progress.connect(lambda f, m: (self.progress.setValue(int(f * 1000)), self.progress_label.setText(m)))
        w.finished_ok.connect(lambda res, secs: self._extraction_done(res, secs, compare))
        w.failed.connect(self._extraction_failed)
        w.cancelled.connect(lambda: self._extraction_finished_ui("Cancelled"))
        w.finished.connect(lambda: self._workers.discard(w))
        self.history_pending = (self.current_path, model_id, time.time())
        w.start()

    def _extraction_finished_ui(self, msg: str):
        self._sep_worker = None
        self.extract_btn.setEnabled(True)
        self.cmp_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_label.setText(msg)
        self._info(msg)

    def _extraction_failed(self, msg: str):
        path, mid, _ = self.history_pending
        self.history.add(path, mid, self.settings.get("device"), self.settings.get("quality"), "", "failed", 0.0, msg)
        self._extraction_finished_ui("Failed")
        self._error(msg)

    def _extraction_done(self, res, secs: float, compare: bool):
        path = self.history_pending[0]
        self.history.add(path, res.model_id, res.device, self.settings.get("quality"),
                         self._out_dir_for(path), "done", secs, f"{len(res.stems)} stems")
        if compare:
            self.result_b = res
            self.ab_btn.setEnabled(True)
            self.showing_b = True
            self.ab_btn.setText("A / B: B")
            self._load_result(res, keep_position=True)
        else:
            self.raw_result = self.result = res
            self.result_b, self.showing_b = None, False
            self.ab_btn.setEnabled(False)
            self.ab_btn.setText("A / B: A")
            self._load_result(res)
        self._extraction_finished_ui(f"Done in {secs:.0f}s on {res.device.upper()}")

    def _cancel(self):
        if self._sep_worker:
            self._sep_worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.progress_label.setText("Cancelling...")

    # ------------------------------------------------------------ result / lanes
    def _clear_lanes(self):
        for ln in self.lanes:
            ln.setParent(None)
            ln.deleteLater()
        self.lanes = []
        while self.stem_boxes.count():
            it = self.stem_boxes.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

    def _load_result(self, res, keep_position: bool = False):
        pos = self.engine.position_seconds() if keep_position else 0.0
        was_playing = self.engine.state == "playing"
        self._clear_lanes()
        if not keep_position:
            self.timeline.set_duration(next(iter(res.stems.values())).shape[1] / res.sample_rate)
        self.engine.set_stems(res.stems, res.sample_rate)
        mix = np.sum(list(res.stems.values()), axis=0)
        self.overview.set_audio(mix, res.sample_rate)
        for name, audio in res.stems.items():
            lane = StemLane(name, self.timeline, audio, res.sample_rate)
            lane.mute_toggled.connect(lambda n, v: setattr(self.engine.mix.stems[n], "mute", v))
            lane.solo_toggled.connect(lambda n, v: setattr(self.engine.mix.stems[n], "solo", v))
            lane.volume_changed.connect(lambda n, v: setattr(self.engine.mix.stems[n], "volume", v))
            lane.pan_changed.connect(lambda n, v: setattr(self.engine.mix.stems[n], "pan", v))
            self.lanes.append(lane)
            self.lanes_layout.insertWidget(self.lanes_layout.count() - 1, lane)
        self.stem_checks: dict[str, QCheckBox] = {}
        names = list(res.stems) + (["instrumental"] if "vocals" in res.stems else [])
        for n in names:
            cb = QCheckBox(n.capitalize() + ("  (all non-vocal stems)" if n == "instrumental" else ""))
            cb.setChecked(n != "instrumental")
            self.stem_boxes.addWidget(cb)
            self.stem_checks[n] = cb
        self._set_result_controls(True)
        if keep_position and pos > 0:
            self.engine.seek(pos)
            if was_playing:
                self.engine.play()

    def _set_result_controls(self, on: bool):
        for w in (self.export_btn, self.mix_btn, self.clean_apply, self.clean_restore, self.play_btn,
                  self.stop_btn, self.loop_btn, self.scrub):
            w.setEnabled(on)
        self.cmp_btn.setEnabled(on)

    # ------------------------------------------------------------ transport
    def _toggle_play(self):
        if not self.result:
            return
        (self.engine.pause if self.engine.state == "playing" else self.engine.play)()

    def _on_state(self, s: str):
        if s.startswith("error:"):
            return self._error(s[6:] + ". Playback is unavailable; export still works.")
        self.play_btn.setText("PAUSE" if s == "playing" else "PLAY")

    def _on_position(self, sec: float):
        self.timeline.set_position(sec)
        dur = self.engine.duration()
        if dur > 0 and not self.scrub.isSliderDown():
            self.scrub.blockSignals(True)
            self.scrub.setValue(int(sec / dur * 1000))
            self.scrub.blockSignals(False)
        self.time_label.setText(f"{fmt_time(sec)} / {fmt_time(dur)}")

    def _loop_changed(self, a, b):
        self._apply_loop()

    def _apply_loop(self):
        if self.loop_btn.isChecked() and self.timeline.loop:
            self.engine.set_loop(*self.timeline.loop)
        else:
            self.engine.set_loop(None, None)

    # ------------------------------------------------------------ clean / compare
    def _apply_clean(self):
        if not self.raw_result:
            return
        strength = self.clean_slider.value() / 100
        self.settings.set("clean_strength", strength)
        base = self.result_b if self.showing_b and self.result_b else self.raw_result
        self.clean_apply.setEnabled(False)

        def work(p, c):
            p(0.1, "NOIZE CLEAN")
            return clean_stems(base.stems, base.sample_rate, strength)

        def done(stems):
            from ..separation import SeparationResult
            res = SeparationResult(stems, base.sample_rate, base.model_id, base.device)
            self.result = res
            self.clean_apply.setEnabled(True)
            self._load_result(res, keep_position=True)
            self._info(f"CLEAN applied (strength {strength:.0%})")

        self._run_func(work, done, on_fail=lambda m: (self.clean_apply.setEnabled(True), self._error(m)))

    def _restore_original(self):
        base = self.result_b if self.showing_b and self.result_b else self.raw_result
        if base:
            self.result = base
            self._load_result(base, keep_position=True)
            self._info("Original stems restored")

    def _run_compare(self):
        if not self.raw_result:
            return self._info("Run a normal extraction first (model A)")
        mid = self.cmp_combo.currentData()
        if mid == self.raw_result.model_id:
            return self._info("Pick a different model for comparison")
        self._start_extraction(mid, compare=True)

    def _toggle_ab(self):
        if not self.result_b:
            return
        self.showing_b = not self.showing_b
        res = self.result_b if self.showing_b else self.raw_result
        self.result = res
        self.ab_btn.setText(f"A / B: {'B' if self.showing_b else 'A'}")
        self._load_result(res, keep_position=True)
        self._info(f"Now playing {res.model_id}")

    # ------------------------------------------------------------ export
    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.out_edit.text())
        if d:
            self.out_edit.setText(d)
            self.settings.set("output_dir", d)

    def _out_dir_for(self, track) -> Path:
        base = self.out_edit.text().strip() or str(default_output_dir())
        return output_dir_for(track, base, self.settings.get("organize_by_track"))

    def _export_args(self):
        if self.fmt_combo.currentData() == "mp3" and not mp3_available():
            raise AudioError("MP3 export needs FFmpeg, which was not found. Choose WAV or FLAC.")
        return self.fmt_combo.currentData(), self.depth_combo.currentData(), self.sr_combo.currentData()

    def _export_stems(self):
        if not self.result or not self.current_path:
            return
        try:
            fmt, depth, sr = self._export_args()
        except AudioError as e:
            return self._error(str(e))
        sel = [n for n, cb in self.stem_checks.items() if cb.isChecked()]
        if not sel:
            return self._info("Select at least one stem")
        res, track, out = self.result, self.current_path, self._out_dir_for(self.current_path)
        extra = vocals_instrumental(res.stems) if "vocals" in res.stems else None
        self.export_btn.setEnabled(False)

        def work(p, c):
            return export_stems(res.stems, res.sample_rate, out, track.name, sel, fmt, depth, sr,
                                {"instrumental": extra["instrumental"]} if extra else None)

        def done(files):
            self.export_btn.setEnabled(True)
            self._info(f"Exported {len(files)} file(s) to {out}")
            open_folder(out)

        self._run_func(work, done, on_fail=lambda m: (self.export_btn.setEnabled(True), self._error(m)))

    def _export_mix(self):
        if not self.result or not self.current_path:
            return
        try:
            fmt, depth, sr = self._export_args()
        except AudioError as e:
            return self._error(str(e))
        res, track, out = self.result, self.current_path, self._out_dir_for(self.current_path)
        import copy
        mix_state = copy.deepcopy(self.engine.mix)
        self.mix_btn.setEnabled(False)
        self._run_func(lambda p, c: export_mix(res.stems, mix_state, res.sample_rate, out, track.name, fmt, depth, sr),
                       lambda f: (self.mix_btn.setEnabled(True), self._info(f"Mix exported: {f.name}"), open_folder(out)),
                       on_fail=lambda m: (self.mix_btn.setEnabled(True), self._error(m)))

    # ------------------------------------------------------------ batch
    def _batch_selected(self) -> int:
        return max(-1, self.batch_table.currentRow())

    def _batch_remove(self):
        r = self._batch_selected()
        if r >= 0:
            try:
                self.queue.remove(r)
            except ValueError as e:
                return self._error(str(e))
            self._refresh_queue()

    def _batch_move(self, d: int):
        r = self._batch_selected()
        if 0 <= r < len(self.queue.items) and 0 <= r + d < len(self.queue.items):
            self.queue.move(r, r + d)
            self._refresh_queue()
            self.batch_table.selectRow(r + d)

    def _batch_start(self):
        if self._batch_worker or not self.queue.pending():
            return self._info("Nothing queued")
        mid = self.settings.get("model")
        if not self.manager.is_installed(mid):
            return self._install_model(mid, then=self._batch_start) and None
        try:
            fmt, depth, sr = self._export_args()
        except AudioError as e:
            return self._error(str(e))
        device, quality = self.settings.get("device"), self.settings.get("quality")
        strength = self.clean_slider.value() / 100 if self.settings.get("clean_enabled") else 0.0
        base_out = self.out_edit.text().strip() or str(default_output_dir())
        organize = self.settings.get("organize_by_track")
        n_total = len(self.queue.pending())
        done_count = [0]

        def process(path: Path, cancel, progress):
            t0 = time.time()
            out = output_dir_for(path, base_out, organize)
            try:
                audio, srate = load_audio(path)
                res = DemucsSeparator(mid, self.manager.dir, device, quality).separate(
                    audio, srate, lambda f, m: progress((done_count[0] + f) / n_total, m), cancel)
                stems = clean_stems(res.stems, res.sample_rate, strength) if strength > 0 else res.stems
                extra = vocals_instrumental(stems) if "vocals" in stems else None
                export_stems(stems, res.sample_rate, out, path.name, None, fmt, depth, sr,
                             {"instrumental": extra["instrumental"]} if extra else None)
                self.history.add(path, mid, res.device, quality, out, "done", time.time() - t0, "batch")
                done_count[0] += 1
            except Exception as e:  # noqa: BLE001
                if e.__class__.__name__ != "Cancelled":
                    self.history.add(path, mid, device, quality, out, "failed", time.time() - t0, str(e))
                raise

        w = BatchWorker(self.queue, process)
        self._batch_worker = w
        self._workers.add(w)
        self.b_start.setEnabled(False)
        self.b_cancel.setEnabled(True)
        w.progress.connect(lambda f, m: (self.batch_progress.setValue(int(f * 1000)), self._info(m)))
        w.item_updated.connect(lambda *_: self._refresh_queue())
        w.finished.connect(self._batch_finished)
        w.start()

    def _batch_finished(self):
        w, self._batch_worker = self._batch_worker, None
        self._workers.discard(w)
        self.b_start.setEnabled(True)
        self.b_cancel.setEnabled(False)
        self.batch_progress.setValue(1000)
        self._refresh_queue()
        ok = sum(i.status == "done" for i in self.queue.items)
        bad = sum(i.status == "failed" for i in self.queue.items)
        self._info(f"Batch finished: {ok} done, {bad} failed")

    def _cancel_batch(self):
        if self._batch_worker:
            self._batch_worker.cancel()

    # ------------------------------------------------------------ history
    def _refresh_history(self):
        self._hist_rows = self.history.list()
        self.hist_table.setRowCount(len(self._hist_rows))
        for r, h in enumerate(self._hist_rows):
            vals = [time.strftime("%Y-%m-%d %H:%M", time.localtime(h["ts"])), Path(h["input_path"]).name,
                    h["model"], h["device"], h["status"].upper(), f"{h['seconds']:.0f}", h["output_dir"]]
            for c, val in enumerate(vals):
                self.hist_table.setItem(r, c, QTableWidgetItem(val))

    def _hist_open(self):
        r = self.hist_table.currentRow()
        if r >= 0 and self._hist_rows[r]["output_dir"] and Path(self._hist_rows[r]["output_dir"]).exists():
            open_folder(self._hist_rows[r]["output_dir"])
        else:
            self._info("No output folder for this entry")

    def _hist_delete(self):
        r = self.hist_table.currentRow()
        if r >= 0:
            self.history.delete(self._hist_rows[r]["id"])
            self._refresh_history()

    def closeEvent(self, e):
        self.engine.stop()
        for w in list(self._workers):
            if hasattr(w, "cancel"):
                w.cancel()
            w.wait(2000)
        self.settings.save()
        super().closeEvent(e)
