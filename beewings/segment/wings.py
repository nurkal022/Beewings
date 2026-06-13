"""Wing foreground mask and per-wing bounding boxes."""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]


def wing_mask(
    img: np.ndarray,
    bg: float,
    delta: int = 45,
    close_frac: float = 0.001,
    exclude: Optional[Box] = None,
) -> np.ndarray:
    """Binary (0/255) mask of wings: content darker than background, with veins
    and membrane merged by morphological closing.

    `exclude` (the label box) is painted out before masking so it cannot be
    picked up as a wing.

    `delta` is deliberately high: it captures the dark wing body and veins while
    rejecting the faint membrane/shadow bridges between densely packed wings, so
    neighbouring wings stay separate. `close_frac` keeps the closing kernel small
    (a few pixels) so it merges veins within one wing without merging neighbours.
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


def drop_ink_blobs(
    img: np.ndarray,
    bg: float,
    boxes: List[Box],
    max_ink_frac: float = 0.33,
    fg_delta: int = 25,
    ink_delta: int = 95,
) -> List[Box]:
    """Drop boxes that are solid handwritten ink rather than a wing.

    A wing is a translucent membrane crossed by a few thin dark veins: of its
    foreground pixels, only a small fraction are ink-dark. A handwritten
    character is opaque ink: nearly all of its foreground is ink-dark. So the
    ratio ``(very dark) / (any foreground)`` cleanly separates the two —
    measured across real slides, wings stay below ~0.23 and characters above
    ~0.42, regardless of size, position, tilt, or paper colour.

    A box is kept when that ratio is at most ``max_ink_frac``. ``fg_delta``
    marks any content below the bright background; ``ink_delta`` marks opaque
    ink-dark pixels.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kept: List[Box] = []
    for box in boxes:
        x, y, w, h = box
        sub = gray[y : y + h, x : x + w]
        fg = int((sub < (bg - fg_delta)).sum())
        if fg == 0:
            continue
        ink = int((sub < (bg - ink_delta)).sum())
        if ink / fg <= max_ink_frac:
            kept.append(box)
    return kept


def filter_wings(
    boxes: List[Box],
    min_count: int = 6,
    area_lo: float = 0.35,
    area_hi: float = 3.0,
    aspect_lo: float = 0.45,
    aspect_hi: float = 3.0,
) -> List[Box]:
    """Reject boxes whose geometry is unlike the dominant wing shape.

    Wings on a slide are many objects of consistent, elongated shape; stray
    detections (handwritten-label characters that survived label exclusion,
    specks, or merged/label-leftover blobs) deviate strongly. We anchor on the
    *median* box and drop outliers, so the filter is robust to slide tilt and
    does not depend on the label being cut out perfectly.

    A box is kept when its area is within ``[area_lo, area_hi] * median_area``
    and its elongation (long/short side) is within
    ``[aspect_lo, aspect_hi] * median``. The upper aspect bound rejects thin
    handwritten strokes, which are far more elongated than any wing. With fewer
    than ``min_count`` boxes the median is unreliable, so the input is returned
    untouched.
    """
    if len(boxes) < min_count:
        return boxes

    arr = np.array(boxes, dtype=float)
    areas = arr[:, 2] * arr[:, 3]
    long_side = np.maximum(arr[:, 2], arr[:, 3])
    short_side = np.maximum(np.minimum(arr[:, 2], arr[:, 3]), 1.0)
    aspect = long_side / short_side

    med_area = float(np.median(areas))
    med_aspect = float(np.median(aspect))

    kept: List[Box] = []
    for box, a, asp in zip(boxes, areas, aspect):
        if a < area_lo * med_area or a > area_hi * med_area:
            continue
        if asp < aspect_lo * med_aspect or asp > aspect_hi * med_aspect:
            continue
        kept.append(box)
    return kept
