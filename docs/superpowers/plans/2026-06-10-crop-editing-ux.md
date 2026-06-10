# Crop-Editing UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]`.

**Goal:** Make the Нарезка tab convenient: side resize handles + keyboard, selection, an objects list synced to the canvas, and an "add label" action when none is detected.

**Architecture:** Extend `geometry.py` (side handles), `box_canvas.py` (persistent selection + visible handles + label-draw mode + keyboard), and `crop_page.py` (objects panel + label controls + two-way sync). Pure geometry stays unit-tested; Qt widgets get headless smokes.

**Tech Stack:** PyQt6, numpy, opencv, pytest.

---

## Task 1: Side resize handles (`geometry.py`)

**Files:** Modify `beewings/pipeline/geometry.py`; Test `tests/test_pipeline_geometry.py` (append).

- [ ] **Step 1: Append failing tests** to `tests/test_pipeline_geometry.py`:
```python
def test_hit_test_side_handles():
    boxes = [(10, 10, 100, 80)]
    assert hit_test(boxes, 60, 10, 8) == (0, "n")    # top edge midpoint
    assert hit_test(boxes, 60, 90, 8) == (0, "s")    # bottom edge midpoint
    assert hit_test(boxes, 10, 50, 8) == (0, "w")    # left edge midpoint
    assert hit_test(boxes, 110, 50, 8) == (0, "e")   # right edge midpoint


def test_corner_beats_side():
    boxes = [(10, 10, 100, 80)]
    assert hit_test(boxes, 10, 10, 8) == (0, "nw")   # corner wins over edge


def test_resize_sides():
    assert resize_box((10, 10, 100, 80), "n", 0, 5) == (10, 15, 100, 75)
    assert resize_box((10, 10, 100, 80), "s", 0, 5) == (10, 10, 100, 85)
    assert resize_box((10, 10, 100, 80), "w", 5, 0) == (15, 10, 95, 80)
    assert resize_box((10, 10, 100, 80), "e", 5, 0) == (10, 10, 105, 80)
```
- [ ] **Step 2:** Run → FAIL: `.venv/bin/python -m pytest tests/test_pipeline_geometry.py -q`
- [ ] **Step 3: Implement** — replace the `Handle` alias line and the `hit_test` and `resize_box` functions in `beewings/pipeline/geometry.py`:

Change the alias comment line:
```python
Handle = Optional[str]  # None | 'move' | 'nw' | 'ne' | 'sw' | 'se'
```
to:
```python
Handle = Optional[str]  # None|'move'|'nw'|'ne'|'sw'|'se'|'n'|'s'|'e'|'w'
```
Add a `_sides` helper after `_corners`:
```python
def _sides(box: Box):
    x, y, w, h = box
    return {"n": (x + w / 2, y), "s": (x + w / 2, y + h),
            "w": (x, y + h / 2), "e": (x + w, y + h / 2)}
```
Replace the body of `hit_test` (keep the signature/docstring) so corners are checked first, then sides, then body:
```python
    for idx in range(len(boxes) - 1, -1, -1):
        box = normalize_box(boxes[idx])
        for name, (cx, cy) in _corners(box).items():
            if abs(px - cx) <= handle_px and abs(py - cy) <= handle_px:
                return idx, name
        for name, (cx, cy) in _sides(box).items():
            if abs(px - cx) <= handle_px and abs(py - cy) <= handle_px:
                return idx, name
        x, y, w, h = box
        if x <= px <= x + w and y <= py <= y + h:
            return idx, "move"
    return None, None
```
In `resize_box`, before the final `return box`, add the four side cases:
```python
    if handle == "n":
        return normalize_box((x, y + dy, w, h - dy))
    if handle == "s":
        return normalize_box((x, y, w, h + dy))
    if handle == "w":
        return normalize_box((x + dx, y, w - dx, h))
    if handle == "e":
        return normalize_box((x, y, w + dx, h))
```
- [ ] **Step 4:** Run → PASS: `.venv/bin/python -m pytest tests/test_pipeline_geometry.py -q`
- [ ] **Step 5:** Commit:
```bash
git add beewings/pipeline/geometry.py tests/test_pipeline_geometry.py
git -c user.name="BeeWings" -c user.email="nurkal836@gmail.com" commit -m "feat(geometry): side resize handles n/s/e/w"
```

---

## Task 2: Selection, handles, label-draw, keyboard (`box_canvas.py`)

**Files:** Rewrite `beewings/pipeline/box_canvas.py`; Test `tests/test_pipeline_qt_smoke.py` (append).

- [ ] **Step 1: Append failing tests** to `tests/test_pipeline_qt_smoke.py`:
```python
def test_box_canvas_selection_and_label(qapp):
    import numpy as np
    from beewings.pipeline.box_canvas import BoxEditorCanvas
    c = BoxEditorCanvas()
    img = np.full((200, 400, 3), 255, np.uint8)
    c.set_scan(img, wing_boxes=[(10, 10, 50, 40), (100, 10, 50, 40)], label_box=None)
    seen = []
    c.selectionChanged.connect(seen.append)
    c.select_box(1)
    assert c.selected() == 1
    assert seen[-1] == 1
    # delete selected via public API clears selection
    c.delete_box(1)
    assert c.selected() == -1
    # label draw mode + programmatic clear
    assert c.label_box() is None
    c.begin_label_draw()
    assert c._label_mode is True
    c.set_label((0, 0, 20, 200))
    assert c.label_box() == (0, 0, 20, 200)
    assert c._label_mode is False
    c.clear_label()
    assert c.label_box() is None
```
- [ ] **Step 2:** Run → FAIL: `.venv/bin/python -m pytest tests/test_pipeline_qt_smoke.py::test_box_canvas_selection_and_label -q`
- [ ] **Step 3: Replace the whole file** `beewings/pipeline/box_canvas.py` with:
```python
"""Zoomable canvas for drawing and editing rectangular boxes over a scan."""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget

from .geometry import Box, hit_test, normalize_box, resize_box

_CURSORS = {
    "nw": Qt.CursorShape.SizeFDiagCursor, "se": Qt.CursorShape.SizeFDiagCursor,
    "ne": Qt.CursorShape.SizeBDiagCursor, "sw": Qt.CursorShape.SizeBDiagCursor,
    "n": Qt.CursorShape.SizeVerCursor, "s": Qt.CursorShape.SizeVerCursor,
    "e": Qt.CursorShape.SizeHorCursor, "w": Qt.CursorShape.SizeHorCursor,
    "move": Qt.CursorShape.SizeAllCursor,
}


class BoxEditorCanvas(QWidget):
    """Scan with editable wing boxes (green) + one label box (red).

    LMB drag on empty space draws a new wing box; drag a handle resizes; drag a
    body moves; click selects; RMB or Del deletes the selected/hit box; arrow
    keys nudge the selection (Shift = ×10). `begin_label_draw()` makes the next
    drag create the red label box. Ctrl+wheel zooms.
    """

    boxesChanged = pyqtSignal()
    selectionChanged = pyqtSignal(int)  # wing index, or -1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._pix: Optional[QPixmap] = None
        self._wing: List[Box] = []
        self._label: Optional[Box] = None
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self._selected = -1            # persistent selection (highlight + list)
        self._drag = None              # (index, handle) during an active drag
        self._drag_last: Optional[QPoint] = None
        self._new_origin = None
        self._preview = None
        self._label_mode = False
        self._drawing_label = False

    # ---- public API -----------------------------------------------------
    def set_scan(self, img_bgr: np.ndarray, wing_boxes: List[Box],
                 label_box: Optional[Box]) -> None:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._pix = QPixmap.fromImage(qimg)
        self._wing = [normalize_box(b) for b in wing_boxes]
        self._label = normalize_box(label_box) if label_box else None
        self._set_selected(-1)
        self._fit()
        self.update()

    def wing_boxes(self) -> List[Box]:
        return list(self._wing)

    def label_box(self) -> Optional[Box]:
        return self._label

    def selected(self) -> int:
        return self._selected

    def select_box(self, idx: int) -> None:
        self._set_selected(idx if 0 <= idx < len(self._wing) else -1)
        self.update()

    def delete_box(self, idx: int) -> None:
        if 0 <= idx < len(self._wing):
            self._wing.pop(idx)
            self._set_selected(-1)
            self.boxesChanged.emit()
            self.update()

    def make_label(self, idx: int) -> None:
        if 0 <= idx < len(self._wing):
            self._label = self._wing.pop(idx)
            self._set_selected(-1)
            self.boxesChanged.emit()
            self.update()

    def begin_label_draw(self) -> None:
        self._label_mode = True

    def set_label(self, box: Box) -> None:
        self._label = normalize_box(box)
        self._label_mode = False
        self.boxesChanged.emit()
        self.update()

    def clear_label(self) -> None:
        self._label = None
        self.boxesChanged.emit()
        self.update()

    # ---- internals ------------------------------------------------------
    def _set_selected(self, idx: int) -> None:
        if idx != self._selected:
            self._selected = idx
            self.selectionChanged.emit(idx)

    def _fit(self) -> None:
        if not self._pix:
            return
        pw, ph = self._pix.width(), self._pix.height()
        if pw == 0 or ph == 0:
            return
        self._scale = min(self.width() / pw, self.height() / ph) or 1.0
        self._offset = QPointF(0, 0)

    def _to_img(self, pos) -> QPoint:
        x = (pos.x() - self._offset.x()) / self._scale
        y = (pos.y() - self._offset.y()) / self._scale
        return QPoint(int(x), int(y))

    def _hpix(self) -> int:
        return int(8 / self._scale) + 1

    # ---- painting -------------------------------------------------------
    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.GlobalColor.darkGray)
        if not self._pix:
            return
        p.translate(self._offset)
        p.scale(self._scale, self._scale)
        p.drawPixmap(0, 0, self._pix)

        pen = QPen(); pen.setCosmetic(True); pen.setWidth(2)
        for i, (x, y, w, h) in enumerate(self._wing):
            if i == self._selected:
                pen.setColor(QColor(255, 196, 0)); pen.setWidth(3)
            else:
                pen.setColor(Qt.GlobalColor.green); pen.setWidth(2)
            p.setPen(pen)
            p.drawRect(x, y, w, h)
        if 0 <= self._selected < len(self._wing):
            self._draw_handles(p, self._wing[self._selected])

        if self._preview is not None:
            pen.setColor(Qt.GlobalColor.green); pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.DashLine); p.setPen(pen)
            px, py, pw, ph = self._preview
            p.drawRect(px, py, pw, ph)
            pen.setStyle(Qt.PenStyle.SolidLine)

        if self._label:
            pen.setColor(Qt.GlobalColor.red); pen.setWidth(2); p.setPen(pen)
            x, y, w, h = self._label
            p.drawRect(x, y, w, h)

    def _draw_handles(self, p: QPainter, box: Box) -> None:
        x, y, w, h = box
        s = max(3, int(7 / self._scale))
        pts = [(x, y), (x + w, y), (x, y + h), (x + w, y + h),
               (x + w // 2, y), (x + w // 2, y + h),
               (x, y + h // 2), (x + w, y + h // 2)]
        p.setPen(QPen(Qt.GlobalColor.black, 0))
        p.setBrush(QColor(255, 255, 255))
        for cx, cy in pts:
            p.drawRect(int(cx - s / 2), int(cy - s / 2), s, s)
        p.setBrush(Qt.BrushStyle.NoBrush)

    def resizeEvent(self, _ev) -> None:
        self._fit()

    # ---- mouse ----------------------------------------------------------
    def mousePressEvent(self, ev) -> None:
        ip = self._to_img(ev.position())
        if ev.button() == Qt.MouseButton.RightButton:
            idx, _ = hit_test(self._wing, ip.x(), ip.y(), self._hpix())
            if idx is not None:
                self.delete_box(idx)
            return
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        if self._label_mode:
            self._new_origin = ip
            self._drawing_label = True
            return
        idx, handle = hit_test(self._wing, ip.x(), ip.y(), self._hpix())
        if idx is not None:
            self._drag = (idx, handle)
            self._drag_last = ip
            self.select_box(idx)
        else:
            self._new_origin = ip
            self.select_box(-1)

    def mouseMoveEvent(self, ev) -> None:
        ip = self._to_img(ev.position())
        if self._drag is not None and self._drag_last is not None:
            idx, handle = self._drag
            dx, dy = ip.x() - self._drag_last.x(), ip.y() - self._drag_last.y()
            self._wing[idx] = resize_box(self._wing[idx], handle, dx, dy)
            self._drag_last = ip
            self.update()
        elif self._new_origin is not None:
            x0, y0 = self._new_origin.x(), self._new_origin.y()
            self._preview = normalize_box((x0, y0, ip.x() - x0, ip.y() - y0))
            self.update()
        else:
            _, handle = hit_test(self._wing, ip.x(), ip.y(), self._hpix())
            self.setCursor(_CURSORS.get(handle, Qt.CursorShape.ArrowCursor))

    def mouseReleaseEvent(self, ev) -> None:
        if self._new_origin is not None:
            x0, y0 = self._new_origin.x(), self._new_origin.y()
            ip = self._to_img(ev.position())
            box = normalize_box((x0, y0, ip.x() - x0, ip.y() - y0))
            big = box[2] > 5 and box[3] > 5
            if self._drawing_label:
                if big:
                    self._label = box
                    self.boxesChanged.emit()
                self._drawing_label = False
                self._label_mode = False
            elif big:
                self._wing.append(box)
                self.select_box(len(self._wing) - 1)
                self.boxesChanged.emit()
            self._new_origin = None
            self._preview = None
            self.update()
        elif self._drag is not None:
            idx, _ = self._drag
            self._wing[idx] = normalize_box(self._wing[idx])
            self._drag = None
            self._drag_last = None
            self.boxesChanged.emit()

    def wheelEvent(self, ev) -> None:
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.2 if ev.angleDelta().y() > 0 else 1 / 1.2
            self._scale *= factor
            self.update()

    def keyPressEvent(self, ev) -> None:
        if not (0 <= self._selected < len(self._wing)):
            super().keyPressEvent(ev)
            return
        key = ev.key()
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_box(self._selected)
            return
        step = 10 if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
        dx = dy = 0
        if key == Qt.Key.Key_Left: dx = -step
        elif key == Qt.Key.Key_Right: dx = step
        elif key == Qt.Key.Key_Up: dy = -step
        elif key == Qt.Key.Key_Down: dy = step
        else:
            super().keyPressEvent(ev); return
        x, y, w, h = self._wing[self._selected]
        self._wing[self._selected] = (x + dx, y + dy, w, h)
        self.boxesChanged.emit()
        self.update()
```
- [ ] **Step 4:** Run → PASS (selection test + the existing `test_box_canvas_set_and_edit`):
`.venv/bin/python -m pytest tests/test_pipeline_qt_smoke.py -q`
- [ ] **Step 5:** Commit:
```bash
git add beewings/pipeline/box_canvas.py tests/test_pipeline_qt_smoke.py
git -c user.name="BeeWings" -c user.email="nurkal836@gmail.com" commit -m "feat(box_canvas): selection, visible handles, label-draw, keyboard edit"
```

---

## Task 3: Objects panel + label controls (`crop_page.py`)

**Files:** Rewrite `beewings/pipeline/pages/crop_page.py`; Test `tests/test_app_qt_smoke.py` (append).

- [ ] **Step 1: Append failing test** to `tests/test_app_qt_smoke.py`:
```python
def test_crop_page_objects_panel(qapp, tmp_path):
    import cv2, numpy as np
    from beewings.pipeline.project import CropProject
    from beewings.pipeline.pages.crop_page import CropPage
    folder = tmp_path / "proj"; folder.mkdir()
    cv2.imwrite(str(folder / "a.jpg"), np.full((200, 300, 3), 255, np.uint8))
    proj = CropProject.open_folder(folder)
    proj.scans[0].wing_boxes = [(10, 10, 40, 30), (80, 10, 40, 30), (150, 10, 40, 30)]
    proj.scans[0].label_box = None
    page = CropPage({"project": proj})
    page.enter()
    assert page.objects.count() == 3                 # one row per wing
    assert page.add_label_btn.isVisible() or page.add_label_btn.isEnabled()
    page.objects.setCurrentRow(1)
    assert page.canvas.selected() == 1               # list -> canvas sync
    # deleting via the button removes a wing and refreshes the list
    page._delete_object()
    assert page.objects.count() == 2
```
- [ ] **Step 2:** Run → FAIL: `.venv/bin/python -m pytest tests/test_app_qt_smoke.py::test_crop_page_objects_panel -q`
- [ ] **Step 3: Replace the whole file** `beewings/pipeline/pages/crop_page.py` with:
```python
"""Stage 2: auto-crop, then review/edit boxes on each scan."""
from __future__ import annotations

from pathlib import Path

import cv2
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QMessageBox,
                             QProgressBar, QPushButton, QVBoxLayout, QWidget)

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

        # --- column 1: splits + run controls ---
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

        # --- column 2: found objects + label controls ---
        mid = QVBoxLayout()
        mid.addWidget(QLabel("Найденные крылья:"))
        self.objects = QListWidget()
        self.objects.currentRowChanged.connect(self._on_object_selected)
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

        # --- column 3: canvas ---
        self.canvas = BoxEditorCanvas()
        self.canvas.boxesChanged.connect(self._on_boxes_changed)
        self.canvas.selectionChanged.connect(self._on_canvas_selection)
        root.addWidget(self.canvas, 1)

    # ---- splits ----
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
        self._refresh_objects()

    # ---- objects panel ----
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

    # ---- run controls ----
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
```
- [ ] **Step 4:** Run → PASS:
`.venv/bin/python -m pytest tests/test_app_qt_smoke.py -q`
- [ ] **Step 5: Full suite + headless app check**
`.venv/bin/python -m pytest tests/ -q` → all pass
```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -c "
from PyQt6.QtWidgets import QApplication; app=QApplication([])
from beewings.app.app import AppController; AppController(); print('ok')"
```
- [ ] **Step 6:** Commit:
```bash
git add beewings/pipeline/pages/crop_page.py tests/test_app_qt_smoke.py
git -c user.name="BeeWings" -c user.email="nurkal836@gmail.com" commit -m "feat(crop): objects panel + label add/remove + canvas sync"
```

---

## Self-Review

**Spec coverage:** objects panel (§2 → Task 3); resize handles+keyboard+selection (§3 → Tasks 1,2); add/remove label (§4 → Tasks 2,3). ✓
**Placeholder scan:** full code in every step; no TBD. ✓
**Type/interface consistency:** geometry handles `n/s/e/w` (T1) used by box_canvas resize/hit_test (T2); `selectionChanged(int)`, `select_box`, `selected`, `begin_label_draw`, `set_label`, `clear_label` (T2) consumed by crop_page (T3); `_on_boxes_changed` replaces the old `boxesChanged→_sync_boxes` wiring and also refreshes objects. `_syncing` guard prevents list↔canvas selection recursion. ✓
**Regression note:** existing `test_box_canvas_set_and_edit` still valid — `delete_box`, `wing_boxes`, `label_box` unchanged in signature; `set_scan` now also resets selection (harmless).
