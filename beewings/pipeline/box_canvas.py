"""Zoomable canvas for drawing and editing rectangular boxes over a scan."""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget

from .geometry import Box, hit_test, normalize_box, resize_box

_GREEN = (0, 180, 0)
_RED = (230, 30, 60)


class BoxEditorCanvas(QWidget):
    """Displays a scan with editable wing boxes (green) and one label box (red).

    Mouse: drag on empty space draws a new wing box; drag a corner resizes; drag
    a body moves; right-click deletes. Ctrl+wheel zooms; middle/space drag pans.
    Emits boxesChanged whenever boxes are modified.
    """

    boxesChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._pix: Optional[QPixmap] = None
        self._wing: List[Box] = []
        self._label: Optional[Box] = None
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self._sel: Optional[int] = None
        self._handle = None
        self._drag_last: Optional[QPoint] = None
        self._new_origin = None
        self._preview = None

    def set_scan(self, img_bgr: np.ndarray, wing_boxes: List[Box],
                 label_box: Optional[Box]) -> None:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._pix = QPixmap.fromImage(qimg)
        self._wing = [normalize_box(b) for b in wing_boxes]
        self._label = normalize_box(label_box) if label_box else None
        self._fit()
        self.update()

    def wing_boxes(self) -> List[Box]:
        return list(self._wing)

    def label_box(self) -> Optional[Box]:
        return self._label

    def delete_box(self, idx: int) -> None:
        if 0 <= idx < len(self._wing):
            self._wing.pop(idx)
            self.boxesChanged.emit()
            self.update()

    def make_label(self, idx: int) -> None:
        if 0 <= idx < len(self._wing):
            self._label = self._wing.pop(idx)
            self.boxesChanged.emit()
            self.update()

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

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.GlobalColor.darkGray)
        if not self._pix:
            return
        p.translate(self._offset)
        p.scale(self._scale, self._scale)
        p.drawPixmap(0, 0, self._pix)
        pen = QPen(); pen.setCosmetic(True); pen.setWidth(2)
        pen.setColor(Qt.GlobalColor.green)
        p.setPen(pen)
        for (x, y, w, h) in self._wing:
            p.drawRect(x, y, w, h)
        if self._preview is not None:
            pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            px, py, pw, ph = self._preview
            p.drawRect(px, py, pw, ph)
            pen.setStyle(Qt.PenStyle.SolidLine)
            p.setPen(pen)
        if self._label:
            pen.setColor(Qt.GlobalColor.red); p.setPen(pen)
            x, y, w, h = self._label
            p.drawRect(x, y, w, h)

    def resizeEvent(self, _ev) -> None:
        self._fit()

    def mousePressEvent(self, ev) -> None:
        ip = self._to_img(ev.position())
        if ev.button() == Qt.MouseButton.RightButton:
            idx, _ = hit_test(self._wing, ip.x(), ip.y(), int(8 / self._scale) + 1)
            if idx is not None:
                self.delete_box(idx)
            return
        if ev.button() == Qt.MouseButton.LeftButton:
            idx, handle = hit_test(self._wing, ip.x(), ip.y(), int(8 / self._scale) + 1)
            if idx is not None:
                self._sel, self._handle, self._drag_last = idx, handle, ip
            else:
                self._new_origin = ip

    def mouseMoveEvent(self, ev) -> None:
        ip = self._to_img(ev.position())
        if self._sel is not None and self._drag_last is not None:
            dx, dy = ip.x() - self._drag_last.x(), ip.y() - self._drag_last.y()
            self._wing[self._sel] = resize_box(self._wing[self._sel], self._handle, dx, dy)
            self._drag_last = ip
            self.update()
        elif self._new_origin is not None:
            x0, y0 = self._new_origin.x(), self._new_origin.y()
            self._preview = normalize_box((x0, y0, ip.x() - x0, ip.y() - y0))
            self.update()

    def mouseReleaseEvent(self, ev) -> None:
        if self._new_origin is not None:
            x0, y0 = self._new_origin.x(), self._new_origin.y()
            ip = self._to_img(ev.position())
            box = normalize_box((x0, y0, ip.x() - x0, ip.y() - y0))
            if box[2] > 5 and box[3] > 5:
                self._wing.append(box)
                self.boxesChanged.emit()
            self._new_origin = None
            self._preview = None
            self.update()
        elif self._sel is not None:
            self._wing[self._sel] = normalize_box(self._wing[self._sel])
            self._sel = self._handle = self._drag_last = None
            self.boxesChanged.emit()

    def wheelEvent(self, ev) -> None:
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.2 if ev.angleDelta().y() > 0 else 1 / 1.2
            self._scale *= factor
            self.update()
