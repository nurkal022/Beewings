"""Render a debug overlay: numbered wing boxes + the label box."""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]


def render_overlay(
    img: np.ndarray,
    boxes: List[Box],
    order: List[int],
    label_box: Optional[Box] = None,
) -> np.ndarray:
    """Return a copy of `img` with green numbered wing boxes and a red label box."""
    out = img.copy()
    thick = max(1, min(img.shape[:2]) // 400)
    scale = max(0.5, min(img.shape[:2]) / 800)
    for n, idx in enumerate(order):
        x, y, w, h = boxes[idx]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 180, 0), thick)
        cv2.putText(out, str(n), (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), thick, cv2.LINE_AA)
    if label_box is not None:
        x, y, w, h = label_box
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), thick)
    return out
