"""Stage 2: auto-crop, then review/edit boxes on each scan."""
from __future__ import annotations

from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QMessageBox,
                             QProgressBar, QPushButton, QVBoxLayout, QWidget)


class _ObjectList(QListWidget):
    """List that emits deleteRequested on Del/Backspace."""

    deleteRequested = pyqtSignal()

    def keyPressEvent(self, ev) -> None:
        if ev.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.deleteRequested.emit()
            return
        super().keyPressEvent(ev)

from ..box_canvas import BoxEditorCanvas
from ..project import CropProject
from ..workers import CropWorker


class CropPage(QWidget):
    def __init__(self, ctx: dict, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._worker = None
        self._syncing = False
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        left.addWidget(QLabel("Сплиты:"))
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

        mid = QVBoxLayout()
        mid.addWidget(QLabel("Найденные крылья:"))
        self.objects = _ObjectList()
        self.objects.currentRowChanged.connect(self._on_object_selected)
        self.objects.deleteRequested.connect(self._delete_object)
        mid.addWidget(self.objects)
        self.del_obj_btn = QPushButton("Удалить крыло")
        self.del_obj_btn.clicked.connect(self._delete_object)
        mid.addWidget(self.del_obj_btn)
        self.label_status = QLabel("Этикетка: —")
        mid.addWidget(self.label_status)
        self.add_label_btn = QPushButton("Добавить этикетку")
        self.add_label_btn.clicked.connect(self._add_label)
        mid.addWidget(self.add_label_btn)
        self.del_label_btn = QPushButton("Удалить этикетку")
        self.del_label_btn.clicked.connect(self._remove_label)
        mid.addWidget(self.del_label_btn)
        root.addLayout(mid, 0)

        self.canvas = BoxEditorCanvas()
        self.canvas.boxesChanged.connect(self._on_boxes_changed)
        self.canvas.selectionChanged.connect(self._on_canvas_selection)
        self.canvas.labelSelected.connect(self._on_label_selected)
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
        entry = proj.scans[row]
        img = cv2.imread(entry.path)
        if img is None:
            return  # unreadable: keep previous scan/_row intact, don't corrupt
        self._row = row
        self.canvas.set_scan(img, entry.wing_boxes, entry.label_box)
        self._refresh_objects()

    def _refresh_objects(self) -> None:
        self._syncing = True
        self.objects.clear()
        self.objects.addItems([f"Крыло {i + 1}" for i in range(len(self.canvas.wing_boxes()))])
        sel = self.canvas.selected()
        if 0 <= sel < self.objects.count():
            self.objects.setCurrentRow(sel)
        self._syncing = False
        has_label = self.canvas.label_box() is not None
        self.label_status.setText("Этикетка: есть" if has_label else "Этикетка: не найдена")
        self.add_label_btn.setVisible(not has_label)
        self.del_label_btn.setVisible(has_label)

    def _on_object_selected(self, row: int) -> None:
        if self._syncing:
            return
        self.canvas.select_box(row)

    def _on_canvas_selection(self, idx: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        self.objects.setCurrentRow(idx)
        self._syncing = False

    def _on_label_selected(self, on: bool) -> None:
        if on:
            self._syncing = True
            self.objects.clearSelection()
            self.objects.setCurrentRow(-1)
            self._syncing = False
            self.label_status.setText("Этикетка: выбрана")
        else:
            self._refresh_objects()

    def _delete_object(self) -> None:
        idx = self.objects.currentRow()
        if idx >= 0:
            self.canvas.delete_box(idx)

    def _add_label(self) -> None:
        self.canvas.begin_label_draw()
        self.label_status.setText("Этикетка: нарисуйте рамку на скане…")

    def _remove_label(self) -> None:
        self.canvas.clear_label()

    def _on_boxes_changed(self) -> None:
        self._sync_boxes()
        self._refresh_objects()

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
