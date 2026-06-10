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


def test_home_lists_recents(qapp):
    from beewings.app.home import HomeScreen
    recents = [{"path": "/a/Клат", "name": "Клат", "opened_at": "t",
                "n_splits": 3, "n_cropped": 2, "n_landmarked": 1}]
    opened = []
    h = HomeScreen(recents, on_open=lambda p: opened.append(p))
    assert h.cards_count() == 1
    h._emit_open("/a/Клат")
    assert opened == ["/a/Клат"]


def test_project_window_has_three_tabs(qapp, tmp_path):
    import cv2, numpy as np
    from beewings.pipeline.project import CropProject
    from beewings.app.project_window import ProjectWindow
    folder = tmp_path / "proj"
    folder.mkdir()
    cv2.imwrite(str(folder / "a.jpg"), np.full((20, 20, 3), 255, np.uint8))
    proj = CropProject.open_folder(folder)
    w = ProjectWindow(proj, on_home=lambda: None)
    assert w.tabs.count() == 3
    assert w.tabs.tabText(0) == "Нарезка"
    assert w.tabs.tabText(1) == "Точки"
    assert w.tabs.tabText(2) == "Экспорт"


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
    assert page.objects.count() == 3
    page.objects.setCurrentRow(1)
    assert page.canvas.selected() == 1
    page._delete_object()
    assert page.objects.count() == 2
