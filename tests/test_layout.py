from __future__ import annotations

from beewings.segment.layout import reading_order


def test_reading_order_row_major():
    # Two rows of three, deliberately shuffled. Centers via boxes (x,y,w,h).
    boxes = [
        (300, 210, 40, 20),  # row1 col3
        (100, 10, 40, 20),   # row0 col1
        (200, 205, 40, 20),  # row1 col2
        (300, 12, 40, 20),   # row0 col3
        (100, 200, 40, 20),  # row1 col1
        (200, 8, 40, 20),    # row0 col2
    ]
    order = reading_order(boxes)
    # Expected sorted: row0 (y~10) left->right, then row1 (y~205) left->right.
    ordered_centers = [(boxes[i][0]) for i in order]
    assert ordered_centers == [100, 200, 300, 100, 200, 300]


def test_reading_order_single_wing():
    assert reading_order([(50, 50, 10, 10)]) == [0]


def test_reading_order_empty():
    assert reading_order([]) == []
