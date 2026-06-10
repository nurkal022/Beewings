from __future__ import annotations

import cv2
import numpy as np


def _crops(tmp_path):
    d = tmp_path / "crops"
    d.mkdir()
    cv2.imwrite(str(d / "w0.jpg"), np.full((40, 80, 3), 255, np.uint8))
    return d


def test_annotator_slots_noop_after_delete(qapp, tmp_path):
    """If the embedded annotator's C++ object is torn down (Qt window/tab
    teardown) while a Python reference and a connected signal survive, its
    slots must no-op instead of touching deleted child widgets (which PyQt6
    turns into a SIGABRT)."""
    from PyQt6 import sip
    from beewings.annotator.annotator_widget import AnnotatorWidget
    w = AnnotatorWidget()
    w.load_folder(_crops(tmp_path))
    sip.delete(w)                      # simulate Qt deleting the widget tree
    # None of these must raise.
    w._run_ml_detect()
    w._run_auto_detect()
    w._on_image_selected(tmp_path / "crops" / "w0.jpg")
    w._save_current()
    w._after_change(advance=False)


def test_side_panel_refresh_noop_after_delete(qapp):
    from PyQt6 import sip
    from beewings.annotator.side_panel import SidePanel
    from beewings.core.profiles import get_profile
    from beewings.core.schema import WingAnnotation
    sp = SidePanel()
    sp._profile = get_profile("Алпатов 12 точек")
    ann = WingAnnotation(image="w.jpg", image_size=(10, 10),
                         profile=sp._profile.name, landmarks=[])
    sip.delete(sp)
    sp.refresh(ann)                    # must not raise
    sp.set_profile(sp._profile, ann, 1)  # must not raise
