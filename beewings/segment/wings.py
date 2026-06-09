"""Wing foreground mask and per-wing bounding boxes."""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]


def wing_mask(
    img: np.ndarray,
    bg: float,
    delta: int = 25,
    close_frac: float = 0.012,
    exclude: Optional[Box] = None,
) -> np.ndarray:
    """Binary (0/255) mask of wings: content darker than background, with veins
    and membrane merged by morphological closing.

    `exclude` (the label box) is painted out before masking so it cannot be
    picked up as a wing.
    """
    work = img.copy()
    if exclude is not None:
        x, y, w, h = exclude
        work[y : y + h, x : x + w] = 255  # blank the label to background

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    mask = (gray < (bg - delta)).astype(np.uint8) * 255

    k = max(3, int(min(img.shape[:2]) * close_frac))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask


def find_wings(mask: np.ndarray, min_area_frac: float = 0.0003) -> List[Box]:
    """Return bounding boxes (x, y, w, h) of wing blobs, filtered by min area.

    Boxes are returned in arbitrary order; use layout.reading_order to sort.
    """
    total = mask.shape[0] * mask.shape[1]
    min_area = min_area_frac * total
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Box] = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        boxes.append(tuple(int(v) for v in cv2.boundingRect(c)))  # type: ignore
    return boxes
