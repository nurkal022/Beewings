"""Order wing boxes in human reading order (row-major: top->bottom, left->right).

Rows are found by clustering box centers on the Y axis: a new row starts when a
center is more than half the median box height below the current row's top.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

Box = Tuple[int, int, int, int]


def reading_order(boxes: List[Box]) -> List[int]:
    """Return indices of `boxes` sorted top-to-bottom, then left-to-right."""
    if not boxes:
        return []
    arr = np.array(boxes, dtype=float)
    cy = arr[:, 1] + arr[:, 3] / 2.0
    cx = arr[:, 0] + arr[:, 2] / 2.0
    median_h = float(np.median(arr[:, 3]))
    row_tol = max(1.0, median_h * 0.5)

    order_by_y = np.argsort(cy)
    rows: List[List[int]] = []
    cur: List[int] = [int(order_by_y[0])]
    cur_y = cy[order_by_y[0]]
    for idx in order_by_y[1:]:
        if cy[idx] - cur_y > row_tol:
            rows.append(cur)
            cur = []
            cur_y = cy[idx]
        cur.append(int(idx))
    rows.append(cur)

    result: List[int] = []
    for row in rows:
        result.extend(sorted(row, key=lambda i: cx[i]))
    return result
