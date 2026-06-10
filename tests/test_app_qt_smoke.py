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


def test_crop_page_unreadable_scan_does_not_corrupt(qapp, tmp_path):
    import cv2, numpy as np
    from beewings.pipeline.project import CropProject, ScanEntry
    from beewings.pipeline.pages.crop_page import CropPage
    folder = tmp_path / "proj"; folder.mkdir()
    cv2.imwrite(str(folder / "a.jpg"), np.full((100, 100, 3), 255, np.uint8))
    proj = CropProject.open_folder(folder)
    proj.scans[0].wing_boxes = [(5, 5, 20, 20)]
    # add a second scan entry pointing at a missing file
    proj.scans.append(ScanEntry(path=str(folder / "missing.jpg")))
    page = CropPage({"project": proj})
    page.enter()                      # loads scan 0
    assert page._row == 0
    page._show_scan(1)                # unreadable -> must NOT switch _row
    assert page._row == 0
    # editing still targets scan 0, scan 1 stays empty
    page.canvas.select_box(0)
    page._delete_object()
    assert proj.scans[0].wing_boxes == []
    assert proj.scans[1].wing_boxes == []


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


def test_landmark_tab_tree(qapp, tmp_path):
    import cv2, numpy as np
    from beewings.pipeline.project import CropProject, auto_detect, recrop
    from beewings.app.project_window import LandmarkTab
    folder = tmp_path / "proj"; folder.mkdir()
    # one real scan -> auto-detect + recrop so crops exist
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
