from __future__ import annotations

import cv2
import numpy as np


def _crops(tmp_path):
    d = tmp_path / "crops"
    d.mkdir()
    for i in range(2):
        cv2.imwrite(str(d / f"w_{i}.jpg"), np.full((40, 80, 3), 255, np.uint8))
    return d


def test_annotator_widget_loads_folder(qapp, tmp_path):
    from beewings.annotator.annotator_widget import AnnotatorWidget
    w = AnnotatorWidget()
    w.load_folder(_crops(tmp_path))
    assert w is not None


def test_mainwindow_wrapper_still_works(qapp, tmp_path):
    from beewings.annotator.main_window import MainWindow
    w = MainWindow()
    w.load_folder(_crops(tmp_path))
    assert w is not None
