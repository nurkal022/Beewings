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
