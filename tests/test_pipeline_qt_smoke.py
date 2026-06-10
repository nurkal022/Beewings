from __future__ import annotations

import numpy as np


def test_box_canvas_set_and_edit(qapp):
    from beewings.pipeline.box_canvas import BoxEditorCanvas
    c = BoxEditorCanvas()
    img = np.full((200, 400, 3), 255, np.uint8)
    c.set_scan(img, wing_boxes=[(10, 10, 50, 40)], label_box=(0, 0, 20, 200))
    assert c.wing_boxes() == [(10, 10, 50, 40)]
    assert c.label_box() == (0, 0, 20, 200)
    c.delete_box(0)
    assert c.wing_boxes() == []


def test_crop_worker_runs(qapp, scan_file, tmp_path):
    from beewings.pipeline.project import CropProject
    from beewings.pipeline.workers import CropWorker
    scan_path, meta = scan_file
    proj = CropProject.create(tmp_path / "proj", scans_root=str(scan_path.parent),
                              scan_paths=[scan_path])
    seen = []
    w = CropWorker(proj, do_autodetect=True, do_recrop=True)
    w.progress.connect(lambda i, t, name: seen.append((i, t)))
    w.run()  # run synchronously in-test (QThread.run is a normal method)
    assert proj.scans[0].cropped is True
    assert len(proj.scans[0].wing_boxes) == meta["n_wings"]
    assert seen[-1][0] == seen[-1][1] == 1


def test_pages_construct(qapp, tmp_path):
    from beewings.pipeline.project import CropProject
    from beewings.pipeline.pages.select_page import SelectPage
    from beewings.pipeline.pages.crop_page import CropPage
    from beewings.pipeline.pages.landmark_page import LandmarkPage
    from beewings.pipeline.pages.export_page import ExportPage
    proj = CropProject.create(tmp_path / "proj", scans_root="/s", scan_paths=[])
    ctx = {"project": proj}
    for Page in (SelectPage, CropPage, LandmarkPage, ExportPage):
        w = Page(ctx)
        assert w is not None
