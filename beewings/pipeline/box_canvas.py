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
    keys nudge the selection (Shift = x10). begin_label_draw() makes the next
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
        self._selected = -1
        self._drag = None
        self._drag_last: Optional[QPoint] = None
        self._new_origin = None
        self._preview = None
        self._label_mode = False
        self._drawing_label = False

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
