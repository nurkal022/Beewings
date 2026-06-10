# Crop/Points Navigation Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Steps use `- [ ]`.

**Goal:** (1) Del/Backspace deletes a wing from the objects list; (2) the label box is selectable/resizable/movable/deletable like wings; (3) the Точки tab uses a tree «scan → wings» that drives the embedded annotator.

**Architecture:** Small additions to `image_list.py` and `annotator_widget.py` (select by path, hide internal browser, show specific image); label editing in `box_canvas.py`; an `_ObjectList` with a delete signal in `crop_page.py`; a `QTreeWidget`-based `LandmarkTab` in `project_window.py`.

**Tech Stack:** PyQt6, numpy, opencv, pytest.

---

## Task 1: Annotator navigation API

**Files:** Modify `beewings/annotator/image_list.py`, `beewings/annotator/annotator_widget.py`; Test `tests/test_app_qt_smoke.py` (append).

- [ ] **Step 1: Append failing tests** to `tests/test_app_qt_smoke.py`:
```python
def test_image_list_select_path(qapp, tmp_path):
    import cv2, numpy as np
    from beewings.annotator.image_list import ImageList
    from beewings.core.profiles import get_profile
    for n in ("w0.jpg", "w1.jpg"):
        cv2.imwrite(str(tmp_path / n), np.full((20, 20, 3), 255, np.uint8))
    il = ImageList()
    il.load_folder(tmp_path, get_profile("Алпатов 12 точек"))
    assert il.select_path(tmp_path / "w1.jpg") is True
    assert il.current_path() == tmp_path / "w1.jpg"
    assert il.select_path(tmp_path / "nope.jpg") is False


def test_annotator_browser_and_show_image(qapp, tmp_path):
    import cv2, numpy as np
    from beewings.annotator.annotator_widget import AnnotatorWidget
    d = tmp_path / "crops"; d.mkdir()
    for i in range(2):
        cv2.imwrite(str(d / f"w_{i}.jpg"), np.full((30, 60, 3), 255, np.uint8))
    w = AnnotatorWidget()
    w.set_browser_visible(False)
    assert w.image_list.isVisible() is False
    w.show_image(d, d / "w_1.jpg")
    assert w.image_list.current_path() == d / "w_1.jpg"
```
- [ ] **Step 2:** Run → FAIL: `.venv/bin/python -m pytest tests/test_app_qt_smoke.py -k "select_path or show_image" -q`
- [ ] **Step 3a: Add `select_path` to `ImageList`** (`beewings/annotator/image_list.py`) — insert after `current_path`:
```python
    def select_path(self, path) -> bool:
        target = str(path)
        for i, p in enumerate(self._paths):
            if str(p) == target:
                self.list.setCurrentRow(i)
                return True
        return False
```
- [ ] **Step 3b: Add methods to `AnnotatorWidget`** (`beewings/annotator/annotator_widget.py`) — add these methods to the class (place near `load_folder`):
```python
    def set_browser_visible(self, visible: bool) -> None:
        self.image_list.setVisible(visible)

    def show_image(self, folder, path) -> None:
        from pathlib import Path as _P
        folder, path = _P(folder), _P(path)
        if self._project_dir != folder:
            self.load_folder(folder)
        self.image_list.select_path(path)
```
- [ ] **Step 4:** Run → PASS: `.venv/bin/python -m pytest tests/test_app_qt_smoke.py -k "select_path or show_image" -q`
- [ ] **Step 5:** Commit:
```bash
git add beewings/annotator/image_list.py beewings/annotator/annotator_widget.py tests/test_app_qt_smoke.py
git -c user.name="BeeWings" -c user.email="nurkal836@gmail.com" commit -m "feat(annotator): select_path, set_browser_visible, show_image"
```

---

## Task 2: Label box editing (`box_canvas.py`)

**Files:** Rewrite `beewings/pipeline/box_canvas.py`; Test `tests/test_pipeline_qt_smoke.py` (append).

- [ ] **Step 1: Append failing test** to `tests/test_pipeline_qt_smoke.py`:
```python
def test_box_canvas_label_editing(qapp):
    import numpy as np
    from beewings.pipeline.box_canvas import BoxEditorCanvas
    c = BoxEditorCanvas()
    c.set_scan(np.full((200, 400, 3), 255, np.uint8),
               wing_boxes=[(10, 10, 50, 40)], label_box=(0, 0, 30, 200))
    flags = []
    c.labelSelected.connect(flags.append)
    c.select_label()
    assert c.label_selected() is True
    assert flags[-1] is True
    assert c.selected() == -1            # wing selection cleared
    # selecting a wing clears label selection
    c.select_box(0)
    assert c.label_selected() is False
    # clear_label deselects label
    c.select_label()
    c.clear_label()
    assert c.label_box() is None
    assert c.label_selected() is False
```
- [ ] **Step 2:** Run → FAIL: `.venv/bin/python -m pytest tests/test_pipeline_qt_smoke.py::test_box_canvas_label_editing -q`
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
    """Scan with editable wing boxes (green) + one editable label box (red)."""

    boxesChanged = pyqtSignal()
    selectionChanged = pyqtSignal(int)   # wing index, or -1
    labelSelected = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._pix: Optional[QPixmap] = None
        self._wing: List[Box] = []
        self._label: Optional[Box] = None
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self._selected = -1
        self._label_sel = False
        self._drag = None             # (kind, idx, handle); kind 'wing'|'label'
        self._drag_last: Optional[QPoint] = None
        self._drag_start = None
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
        self._set_label_sel(False)
        self._fit()
        self.update()

    def wing_boxes(self) -> List[Box]:
        return list(self._wing)

    def label_box(self) -> Optional[Box]:
        return self._label

    def selected(self) -> int:
        return self._selected

    def label_selected(self) -> bool:
        return self._label_sel

    def select_box(self, idx: int) -> None:
        self._set_label_sel(False)
        self._set_selected(idx if 0 <= idx < len(self._wing) else -1)
        self.update()

    def select_label(self) -> None:
        if self._label is not None:
            self._set_selected(-1)
            self._set_label_sel(True)
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
        self._set_label_sel(False)
        self.boxesChanged.emit()
        self.update()

    # ---- selection helpers ---------------------------------------------
    def _set_selected(self, idx: int) -> None:
        if idx != self._selected:
            self._selected = idx
            self.selectionChanged.emit(idx)

    def _set_label_sel(self, on: bool) -> None:
        if on != self._label_sel:
            self._label_sel = on
            self.labelSelected.emit(on)

    # ---- view -----------------------------------------------------------
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
            pen.setColor(QColor(255, 140, 0) if self._label_sel else Qt.GlobalColor.red)
            pen.setWidth(3 if self._label_sel else 2); p.setPen(pen)
            x, y, w, h = self._label
            p.drawRect(x, y, w, h)
            if self._label_sel:
                self._draw_handles(p, self._label)

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
            self.select_box(idx)
            self._drag = ("wing", idx, handle)
            self._drag_last = ip
            self._drag_start = self._wing[idx]
            return
        if self._label is not None:
            lidx, lhandle = hit_test([self._label], ip.x(), ip.y(), self._hpix())
            if lidx is not None:
                self.select_label()
                self._drag = ("label", 0, lhandle)
                self._drag_last = ip
                self._drag_start = self._label
                return
        self._new_origin = ip
        self.select_box(-1)

    def mouseMoveEvent(self, ev) -> None:
        ip = self._to_img(ev.position())
        if self._drag is not None and self._drag_last is not None:
            kind, idx, handle = self._drag
            dx, dy = ip.x() - self._drag_last.x(), ip.y() - self._drag_last.y()
            if kind == "wing":
                self._wing[idx] = resize_box(self._wing[idx], handle, dx, dy)
            else:
                self._label = resize_box(self._label, handle, dx, dy)
            self._drag_last = ip
            self.update()
        elif self._new_origin is not None:
            x0, y0 = self._new_origin.x(), self._new_origin.y()
            self._preview = normalize_box((x0, y0, ip.x() - x0, ip.y() - y0))
            self.update()
        else:
            _, handle = hit_test(self._wing, ip.x(), ip.y(), self._hpix())
            if handle is None and self._label is not None:
                _, handle = hit_test([self._label], ip.x(), ip.y(), self._hpix())
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
                self._drawing_label = False
                self._label_mode = False
                self.boxesChanged.emit()
            elif big:
                self._wing.append(box)
                self.select_box(len(self._wing) - 1)
                self.boxesChanged.emit()
            self._new_origin = None
            self._preview = None
            self.update()
        elif self._drag is not None:
            kind, idx, _ = self._drag
            if kind == "wing":
                self._wing[idx] = normalize_box(self._wing[idx])
                changed = self._wing[idx] != self._drag_start
            else:
                self._label = normalize_box(self._label)
                changed = self._label != self._drag_start
            self._drag = None
            self._drag_last = None
            self._drag_start = None
            if changed:
                self.boxesChanged.emit()

    def wheelEvent(self, ev) -> None:
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.2 if ev.angleDelta().y() > 0 else 1 / 1.2
            self._scale *= factor
            self.update()

    def keyPressEvent(self, ev) -> None:
        key = ev.key()
        step = 10 if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
        delete_keys = (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
        arrows = {Qt.Key.Key_Left: (-step, 0), Qt.Key.Key_Right: (step, 0),
                  Qt.Key.Key_Up: (0, -step), Qt.Key.Key_Down: (0, step)}
        if self._label_sel and self._label is not None:
            if key in delete_keys:
                self.clear_label(); return
            if key in arrows:
                dx, dy = arrows[key]
                x, y, w, h = self._label
                self._label = (x + dx, y + dy, w, h)
                self.boxesChanged.emit(); self.update(); return
        elif 0 <= self._selected < len(self._wing):
            if key in delete_keys:
                self.delete_box(self._selected); return
            if key in arrows:
                dx, dy = arrows[key]
                x, y, w, h = self._wing[self._selected]
                self._wing[self._selected] = (x + dx, y + dy, w, h)
                self.boxesChanged.emit(); self.update(); return
        super().keyPressEvent(ev)
```
- [ ] **Step 4:** Run → PASS (existing box_canvas tests + new label test):
`.venv/bin/python -m pytest tests/test_pipeline_qt_smoke.py -q`
- [ ] **Step 5:** Commit:
```bash
git add beewings/pipeline/box_canvas.py tests/test_pipeline_qt_smoke.py
git -c user.name="BeeWings" -c user.email="nurkal836@gmail.com" commit -m "feat(box_canvas): selectable, resizable, deletable label box"
```

---

## Task 3: Objects-list delete key + label-selected hint (`crop_page.py`)

**Files:** Modify `beewings/pipeline/pages/crop_page.py`; Test `tests/test_app_qt_smoke.py` (append).

- [ ] **Step 1: Append failing test** to `tests/test_app_qt_smoke.py`:
```python
def test_crop_page_list_delete_signal(qapp, tmp_path):
    import cv2, numpy as np
    from beewings.pipeline.project import CropProject
    from beewings.pipeline.pages.crop_page import CropPage
    folder = tmp_path / "proj"; folder.mkdir()
    cv2.imwrite(str(folder / "a.jpg"), np.full((200, 300, 3), 255, np.uint8))
    proj = CropProject.open_folder(folder)
    proj.scans[0].wing_boxes = [(10, 10, 40, 30), (80, 10, 40, 30)]
    page = CropPage({"project": proj})
    page.enter()
    page.objects.setCurrentRow(0)
    page.objects.deleteRequested.emit()    # simulate Del keypress on the list
    assert page.objects.count() == 1
```
- [ ] **Step 2:** Run → FAIL: `.venv/bin/python -m pytest tests/test_app_qt_smoke.py::test_crop_page_list_delete_signal -q`
- [ ] **Step 3: Edit `crop_page.py`:**
  (a) Replace the imports line:
```python
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QMessageBox,
                             QProgressBar, QPushButton, QVBoxLayout, QWidget)
```
  with:
```python
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
```
  (b) In `__init__`, change `self.objects = QListWidget()` to:
```python
        self.objects = _ObjectList()
```
  and right after the existing `self.objects.currentRowChanged.connect(self._on_object_selected)` line add:
```python
        self.objects.deleteRequested.connect(self._delete_object)
```
  (c) After the canvas signal connections in `__init__` (after `self.canvas.selectionChanged.connect(self._on_canvas_selection)`), add:
```python
        self.canvas.labelSelected.connect(self._on_label_selected)
```
  (d) Add a handler method (place near `_on_canvas_selection`):
```python
    def _on_label_selected(self, on: bool) -> None:
        if on:
            self._syncing = True
            self.objects.clearSelection()
            self.objects.setCurrentRow(-1)
            self._syncing = False
            self.label_status.setText("Этикетка: выбрана")
        else:
            self._refresh_objects()
```
- [ ] **Step 4:** Run → PASS: `.venv/bin/python -m pytest tests/test_app_qt_smoke.py -q`
- [ ] **Step 5:** Commit:
```bash
git add beewings/pipeline/pages/crop_page.py tests/test_app_qt_smoke.py
git -c user.name="BeeWings" -c user.email="nurkal836@gmail.com" commit -m "feat(crop): Del/Backspace deletes from objects list; label-selected hint"
```

---

## Task 4: Tree navigation in the Точки tab (`project_window.py`)

**Files:** Modify `beewings/app/project_window.py`; Test `tests/test_app_qt_smoke.py` (append).

- [ ] **Step 1: Append failing test** to `tests/test_app_qt_smoke.py`:
```python
def test_landmark_tab_tree(qapp, tmp_path):
    import cv2, numpy as np
    from beewings.pipeline.project import CropProject, auto_detect, recrop
    from beewings.app.project_window import LandmarkTab
    folder = tmp_path / "proj"; folder.mkdir()
    # one real scan -> auto-detect + recrop so crops exist
    from tests.conftest import synthetic_scan  # not callable directly; build inline
    img = np.full((300, 600, 3), 255, np.uint8)
    cv2.ellipse(img, (200, 150), (120, 50), 0, 0, 360, (120, 120, 120), -1)
    cv2.ellipse(img, (430, 150), (120, 50), 0, 0, 360, (120, 120, 120), -1)
    cv2.imwrite(str(folder / "a.jpg"), img)
    proj = CropProject.open_folder(folder)
    auto_detect(proj.scans[0], proj.settings)
    recrop(proj, proj.scans[0])
    tab = LandmarkTab({"project": proj})
    tab.enter()
    assert tab.tree.topLevelItemCount() == 1            # one split
    parent = tab.tree.topLevelItem(0)
    tab.tree.expandItem(parent)                          # lazy-populate wings
    assert parent.childCount() >= 1                      # wings listed
    # annotator's internal browser hidden in the tab
    assert tab.annot.image_list.isVisible() is False
```
NOTE: remove the unused `from tests.conftest import synthetic_scan` line when implementing — it is illustrative; the inline image is what the test uses. Keep the rest.
- [ ] **Step 2:** Run → FAIL: `.venv/bin/python -m pytest tests/test_app_qt_smoke.py::test_landmark_tab_tree -q`
- [ ] **Step 3: Replace the `LandmarkTab` class** in `beewings/app/project_window.py`. Update the imports block to include the tree widgets and helpers — change:
```python
from PyQt6.QtWidgets import (QHBoxLayout, QListWidget, QMainWindow, QMessageBox,
                             QProgressBar, QPushButton, QStatusBar, QTabWidget,
                             QVBoxLayout, QWidget)
```
to:
```python
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QHBoxLayout, QMainWindow, QMessageBox, QProgressBar,
                             QPushButton, QStatusBar, QTabWidget, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core.profiles import get_profile
from ..core.schema import annotation_path, load_annotation
```
Then replace the entire `class LandmarkTab(QWidget): ...` definition with:
```python
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
_ROLE = Qt.ItemDataRole.UserRole


def _crop_files(cdir):
    skip = ("_label", "_debug")
    return sorted(p for p in cdir.glob("*")
                  if p.suffix.lower() in _IMAGE_EXTS and not any(t in p.stem for t in skip))


class LandmarkTab(QWidget):
    """Tree (scan -> wings) driving the embedded annotator; ML over all crops."""

    def __init__(self, ctx: dict, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._worker = None
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemExpanded.connect(self._populate)
        self.tree.itemClicked.connect(self._on_item_clicked)
        left.addWidget(self.tree)
        self.run_btn = QPushButton("\U0001f9e0 Расставить точки (ML)")
        self.run_btn.clicked.connect(self._run)
        left.addWidget(self.run_btn)
        self.progress = QProgressBar()
        left.addWidget(self.progress)
        root.addLayout(left, 0)
        self.annot = AnnotatorWidget()
        self.annot.set_browser_visible(False)
        root.addWidget(self.annot, 1)

    def enter(self) -> None:
        proj: CropProject = self.ctx["project"]
        self.tree.clear()
        for i, scan in enumerate(proj.scans):
            item = QTreeWidgetItem([Path(scan.path).name])
            item.setData(0, _ROLE, ("scan", i))
            QTreeWidgetItem(item, ["…"])  # placeholder so it is expandable
            self.tree.addTopLevelItem(item)

    def _populate(self, parent: "QTreeWidgetItem") -> None:
        kind, idx = parent.data(0, _ROLE)
        if kind != "scan":
            return
        if parent.childCount() == 1 and parent.child(0).data(0, _ROLE) is None:
            parent.takeChildren()  # drop placeholder
        else:
            return  # already populated
        proj: CropProject = self.ctx["project"]
        cdir = proj.crops_dir(proj.scans[idx])
        crops = _crop_files(cdir) if cdir.exists() else []
        if not crops:
            child = QTreeWidgetItem(parent, ["(не нарезано)"])
            child.setDisabled(True)
            return
        prof = get_profile(proj.settings.profile)
        for cp in crops:
            child = QTreeWidgetItem(parent, [self._wing_label(cp, cdir, prof)])
            child.setData(0, _ROLE, ("wing", idx, str(cp)))

    def _wing_label(self, cp, cdir, prof) -> str:
        ann = load_annotation(annotation_path(cp, cdir))
        if ann is None:
            return f"○  {cp.name}"
        done, total = ann.progress(prof.ids)
        mark = "✓" if done == total and total else "●"
        return f"{mark}  {cp.name}"

    def _on_item_clicked(self, item: "QTreeWidgetItem", _col: int) -> None:
        data = item.data(0, _ROLE)
        if not data:
            return
        if data[0] == "scan":
            item.setExpanded(True)
            if item.childCount() and item.child(0).data(0, _ROLE):
                self._open_wing(item.child(0))
        elif data[0] == "wing":
            self._open_wing(item)

    def _open_wing(self, item: "QTreeWidgetItem") -> None:
        data = item.data(0, _ROLE)
        if not data or data[0] != "wing":
            return
        _, idx, crop = data
        proj: CropProject = self.ctx["project"]
        cdir = proj.crops_dir(proj.scans[idx])
        self.annot.show_image(cdir, Path(crop))

    def _run(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        proj: CropProject = self.ctx["project"]
        self.progress.setRange(0, 0)
        self.run_btn.setEnabled(False)
        self._worker = LandmarkWorker(proj)
        self._worker.progress.connect(lambda i, t, name: self.progress.setFormat(name))
        self._worker.failed.connect(
            lambda path, err: QMessageBox.warning(self, "Ошибка разметки", f"{path}\n{err}"))
        self._worker.finished_ok.connect(self._done)
        self._worker.start()

    def _done(self) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.run_btn.setEnabled(True)
        self.enter()  # rebuild tree (refresh progress marks)
```
Keep the `from pathlib import Path` and `from ..annotator.annotator_widget import AnnotatorWidget`, `from ..pipeline.workers import LandmarkWorker`, `from ..pipeline.project import CropProject` imports already at the top of the file. Remove the now-unused `QListWidget` import (replaced by tree).
- [ ] **Step 4:** Run → PASS: `.venv/bin/python -m pytest tests/test_app_qt_smoke.py::test_landmark_tab_tree -q`
- [ ] **Step 5: Full suite + headless app check**
`.venv/bin/python -m pytest tests/ -q` → all pass
```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -c "
from PyQt6.QtWidgets import QApplication; app=QApplication([])
from beewings.app.app import AppController; AppController(); print('ok')"
```
- [ ] **Step 6:** Commit:
```bash
git add beewings/app/project_window.py tests/test_app_qt_smoke.py
git -c user.name="BeeWings" -c user.email="nurkal836@gmail.com" commit -m "feat(app): Точки tab tree (scan -> wings) driving embedded annotator"
```

---

## Self-Review

**Spec coverage:** Del/Backspace from list (§2 → Task 3); label editing select/resize/move/delete (§3 → Task 2); tree scan→wings + hidden internal browser + show_image (§4,§5 → Tasks 1,4). ✓
**Placeholder scan:** full code in steps; the Task-4 test note explicitly removes the illustrative import. No TBD. ✓
**Type/interface consistency:** `_drag=(kind,idx,handle)` used consistently in mouseMove/Release; `labelSelected(bool)`, `select_label`, `label_selected` (T2) consumed by crop_page (T3); `set_browser_visible`, `show_image` (T1) consumed by LandmarkTab (T4); `ImageList.select_path` (T1) used by `show_image`. Tree item roles: `("scan", i)` parents, `("wing", idx, crop)` children, placeholder has role None. ✓
**Regression:** existing box_canvas tests (`set_and_edit`, `selection_and_label`) still valid — wing APIs unchanged; `select_box` now also clears label selection (harmless). LandmarkTab previously used `split_list`; ProjectWindow references `self.landmark_tab.annot` (still present) — `_show_split` removed, but ProjectWindow only calls `enter()` on tab change (verify ProjectWindow doesn't call `landmark_tab._show_split`; it does not).
