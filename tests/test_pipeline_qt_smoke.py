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


def test_box_canvas_selection_and_label(qapp):
    import numpy as np
    from beewings.pipeline.box_canvas import BoxEditorCanvas
    c = BoxEditorCanvas()
    img = np.full((200, 400, 3), 255, np.uint8)
    c.set_scan(img, wing_boxes=[(10, 10, 50, 40), (100, 10, 50, 40)], label_box=None)
    seen = []
    c.selectionChanged.connect(seen.append)
    c.select_box(1)
    assert c.selected() == 1
    assert seen[-1] == 1
    c.delete_box(1)
    assert c.selected() == -1
    assert c.label_box() is None
    c.begin_label_draw()
    assert c._label_mode is True
    c.set_label((0, 0, 20, 200))
    assert c.label_box() == (0, 0, 20, 200)
    assert c._label_mode is False
    c.clear_label()
    assert c.label_box() is None


def test_box_canvas_label_editing(qapp):
    import numpy as np
    from beewings.pipeline.box_canvas import BoxEditorCanvas
    c = BoxEditorCanvas()
    c.set_scan(np.full((200, 400, 3), 255, np.uint8),
               wing_boxes=[(10, 10, 50, 40)], label_box=(0, 0, 30, 200))
    flags = []
    c.labelSelected.connect(flags.append)
    c.select_label()
    assert c.label_selected() is True
    assert flags[-1] is True
    assert c.selected() == -1            # wing selection cleared
    # selecting a wing clears label selection
    c.select_box(0)
    assert c.label_selected() is False
    # clear_label deselects label
    c.select_label()
    c.clear_label()
    assert c.label_box() is None
    assert c.label_selected() is False
