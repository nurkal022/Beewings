from __future__ import annotations

from beewings.pipeline.geometry import normalize_box, hit_test, resize_box


def test_normalize_negative_size():
    assert normalize_box((50, 50, -20, -10)) == (30, 40, 20, 10)


def test_hit_test_corner():
    boxes = [(10, 10, 100, 80)]
    idx, handle = hit_test(boxes, 109, 89, handle_px=8)
    assert idx == 0 and handle == "se"


def test_hit_test_body():
    boxes = [(10, 10, 100, 80)]
    idx, handle = hit_test(boxes, 60, 50, handle_px=8)
    assert idx == 0 and handle == "move"


def test_hit_test_miss():
    boxes = [(10, 10, 100, 80)]
    idx, handle = hit_test(boxes, 500, 500, handle_px=8)
    assert idx is None and handle is None


def test_resize_se_corner():
    assert resize_box((10, 10, 100, 80), "se", 10, 20) == (10, 10, 110, 100)


def test_resize_nw_corner():
    assert resize_box((10, 10, 100, 80), "nw", 5, 5) == (15, 15, 95, 75)


def test_resize_move():
    assert resize_box((10, 10, 100, 80), "move", 7, -3) == (17, 7, 100, 80)
