"""Floating magnifier widget that shows the image area near the cursor.

Pixel-perfect, no interpolation, with a crosshair at the exact target point.
Anchored to a screen corner of the canvas (configurable), out of the way.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout

from .canvas import ndarray_to_qimage


class Magnifier(QFrame):
    SIZE = 220        # widget size in pixels
    SAMPLE = 55       # area of source image to sample (smaller -> stronger zoom)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)
        self.setStyleSheet("QFrame { background: #111; border: 1px solid #444; }")
        self.setFixedSize(self.SIZE, self.SIZE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)

    def update_view(self, img: Optional[np.ndarray], x: float, y: float) -> None:
        if img is None:
            return
        h, w = img.shape[:2]
        half = self.SAMPLE // 2
        cx, cy = int(round(x)), int(round(y))
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(w, cx + half)
        y1 = min(h, cy + half)
        if x1 - x0 < 4 or y1 - y0 < 4:
            self._label.clear()
            return
        crop = img[y0:y1, x0:x1].copy()
        qimg = ndarray_to_qimage(crop)
        pm = QPixmap.fromImage(qimg).scaled(
            self.SIZE - 6, self.SIZE - 6,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,  # nearest-neighbor for pixel-accuracy
        )
        # Draw crosshair at where (x,y) lands in the scaled crop.
        painter = QPainter(pm)
        local_x = (cx - x0) / max(1, (x1 - x0)) * pm.width()
        local_y = (cy - y0) / max(1, (y1 - y0)) * pm.height()
        pen = QPen(QColor(0, 255, 120, 220))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(int(local_x), 0, int(local_x), pm.height())
        painter.drawLine(0, int(local_y), pm.width(), int(local_y))
        painter.end()
        self._label.setPixmap(pm)
