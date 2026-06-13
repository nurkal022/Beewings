from __future__ import annotations

import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_scan():
    """Return (img_bgr, meta) where meta has known label box and wing centers.

    Layout: white 255 canvas. Left 'label' = dense dark scribble strokes.
    Then a vertical empty gap. Then a 5x4 grid of dark ellipses (wings).
    """
    h, w = 600, 1200
    img = np.full((h, w, 3), 255, np.uint8)

    # Label block on the left: compact handwriting-like marks inside x in
    # [20, 180]. Each mark is far smaller than a wing, so the geometric filter
    # rejects them the way it rejects real handwritten characters; collectively
    # they still form a detectable left-anchored label block.
    label_box = (20, 40, 160, 300)  # x, y, w, h
    for ry in range(70, 341, 60):              # dense rows of writing
        cv2.line(img, (28, ry), (170, ry), (30, 30, 30), 9)

    # Wing grid: 5 columns x 4 rows of ellipses, starting well right of the gap.
    centers = []
    x0, y0, dx, dy = 360, 90, 160, 130
    for r in range(4):
        for c in range(5):
            cx, cy = x0 + c * dx, y0 + r * dy
            # A wing is a faint translucent membrane crossed by thin dark veins,
            # not a solid blob — so the ink-fraction filter keeps it.
            cv2.ellipse(img, (cx, cy), (55, 28), 0, 0, 360, (185, 185, 185), -1)
            cv2.line(img, (cx - 50, cy), (cx + 50, cy), (30, 30, 30), 2)
            cv2.line(img, (cx, cy - 24), (cx, cy + 24), (30, 30, 30), 2)
            centers.append((cx, cy))

    meta = {
        "label_box": label_box,
        "centers": centers,       # row-major order
        "n_wings": len(centers),
        "shape": (h, w),
    }
    return img, meta


import os


@pytest.fixture(scope="session")
def qapp():
    """A headless QApplication for widget construction tests."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    return app


@pytest.fixture
def scan_file(synthetic_scan, tmp_path):
    """Write the synthetic scan to a temp .jpg and return its Path."""
    img, meta = synthetic_scan
    path = tmp_path / "scan_0001.jpg"
    cv2.imwrite(str(path), img)
    return path, meta
