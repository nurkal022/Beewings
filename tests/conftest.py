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

    # Label block on the left: several dark strokes inside x in [20, 180].
    label_box = (20, 40, 160, 300)  # x, y, w, h
    rng_lines = [
        ((30, 80), (170, 90)),
        ((30, 140), (150, 150)),
        ((40, 200), (175, 215)),
        ((35, 300), (160, 320)),
    ]
    for (x1, y1), (x2, y2) in rng_lines:
        cv2.line(img, (x1, y1), (x2, y2), (30, 30, 30), 6)

    # Wing grid: 5 columns x 4 rows of ellipses, starting well right of the gap.
    centers = []
    x0, y0, dx, dy = 360, 90, 160, 130
    for r in range(4):
        for c in range(5):
            cx, cy = x0 + c * dx, y0 + r * dy
            cv2.ellipse(img, (cx, cy), (55, 28), 0, 0, 360, (120, 120, 120), -1)
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
