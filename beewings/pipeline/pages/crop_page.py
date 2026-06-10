"""Stage 2: auto-crop, then review/edit boxes on each scan."""
from __future__ import annotations

from pathlib import Path

import cv2
from PyQt6.QtWidgets import (QHBoxLayout, QListWidget, QMessageBox, QProgressBar,
                             QPushButton, QVBoxLayout, QWidget)

from ..box_canvas import BoxEditorCanvas
from ..project import CropProject
from ..workers import CropWorker


class CropPage(QWidget):
    def __init__(self, ctx: dict, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._worker = None
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        self.scan_list = QListWidget()
        self.scan_list.currentRowChanged.connect(self._show_scan)
        left.addWidget(self.scan_list)
        self.auto_btn = QPushButton("Авто-нарезка всех")
        self.auto_btn.clicked.connect(self._auto)
        left.addWidget(self.auto_btn)
        self.recrop_btn = QPushButton("Пересоздать кропы")
        self.recrop_btn.clicked.connect(self._recrop)
        left.addWidget(self.recrop_btn)
        self.cancel_btn = QPushButton("Отмена")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        left.addWidget(self.cancel_btn)
        self.progress = QProgressBar()
        left.addWidget(self.progress)
        root.addLayout(left, 0)
        self.canvas = BoxEditorCanvas()
        self.canvas.boxesChanged.connect(self._sync_boxes)
        root.addWidget(self.canvas, 1)

    def enter(self) -> None:
        proj: CropProject = self.ctx["project"]
        self.scan_list.clear()
        self.scan_list.addItems([Path(s.path).name for s in proj.scans])
        if proj.scans:
            self.scan_list.setCurrentRow(0)

    def _show_scan(self, row: int) -> None:
        proj: CropProject = self.ctx["project"]
        if not (0 <= row < len(proj.scans)):
            return
        self._row = row
        entry = proj.scans[row]
        img = cv2.imread(entry.path)
        if img is not None:
            self.canvas.set_scan(img, entry.wing_boxes, entry.label_box)

    def _sync_boxes(self) -> None:
        proj: CropProject = self.ctx["project"]
        row = getattr(self, "_row", None)
        if row is None:
            return
        proj.scans[row].wing_boxes = self.canvas.wing_boxes()
        proj.scans[row].label_box = self.canvas.label_box()
        proj.save()

    def _auto(self) -> None:
        self._run(do_autodetect=True, do_recrop=False, refresh=True)

    def _recrop(self) -> None:
        self._run(do_autodetect=False, do_recrop=True, refresh=False)

    def _run(self, do_autodetect: bool, do_recrop: bool, refresh: bool) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        proj: CropProject = self.ctx["project"]
        self.progress.setRange(0, len(proj.scans))
        self._set_running(True)
        self._worker = CropWorker(proj, do_autodetect, do_recrop)
        self._worker.progress.connect(lambda i, t, n: self.progress.setValue(i))
        self._worker.failed.connect(
            lambda path, err: QMessageBox.warning(self, "Ошибка нарезки", f"{path}\n{err}"))
        self._worker.finished_ok.connect(lambda: self._on_done(refresh))
        self._worker.start()

    def _on_done(self, refresh: bool) -> None:
        self._set_running(False)
        if refresh:
            self._show_scan(getattr(self, "_row", 0))

    def _set_running(self, running: bool) -> None:
        for b in (self.auto_btn, self.recrop_btn):
            b.setEnabled(not running)
        self.cancel_btn.setEnabled(running)

    def _cancel(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()

    def commit(self) -> bool:
        proj: CropProject = self.ctx["project"]
        proj.stage = "landmarks"
        proj.save()
        return True
