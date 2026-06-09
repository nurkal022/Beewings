"""Background estimation and handwritten-label detection for scan slides.

The label is always on the left of the slide, separated from the wing grid by
an empty vertical gap. We detect dark foreground content, then find the gap and
treat everything left of it as the label.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]


def estimate_background(img: np.ndarray) -> float:
    """Estimate the bright slide background as the median of the four corners."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    k = max(8, min(h, w) // 20)
    corners = np.concatenate([
        gray[:k, :k].ravel(),
        gray[:k, -k:].ravel(),
        gray[-k:, :k].ravel(),
        gray[-k:, -k:].ravel(),
    ])
    return float(np.median(corners))


def _foreground(gray: np.ndarray, bg: float, delta: int = 35) -> np.ndarray:
    """Binary mask of content darker than the background by `delta`."""
    mask = (gray < (bg - delta)).astype(np.uint8) * 255
    return mask


def detect_label_region(
    img: np.ndarray,
    bg: float,
    search_frac: float = 0.45,
    gap_frac: float = 0.04,
) -> Optional[Box]:
    """Return (x, y, w, h) of the left label block, or None if absent.

    Looks for dark content in the left `search_frac` of the image, then finds
    the first sustained empty vertical gap (width >= gap_frac * W) that ends the
    label block. The bounding box of content left of that gap is the label.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    fg = _foreground(gray, bg)

    col = fg.sum(axis=0)  # per-column foreground amount
    search_w = int(w * search_frac)
    if col[:search_w].max() == 0:
        return None  # no content on the left at all

    gap_w = max(5, int(w * gap_frac))
    empty = col <= (0.01 * col.max())

    # Find the rightmost column of the *first* label cluster: scan from the
    # leftmost content column until a run of `gap_w` empty columns appears.
    first = int(np.argmax(col > 0))
    # A real label is always anchored to the far-left edge; if the first
    # foreground pixel is beyond 15 % of the width it belongs to the wing grid.
    if first > int(w * 0.15):
        return None
    end = first
    run = 0
    for x in range(first, search_w):
        if empty[x]:
            run += 1
            if run >= gap_w:
                end = x - gap_w
                break
        else:
            run = 0
            end = x
    else:
        return None  # content never closed with a gap -> not a label block

    band = fg[:, : end + 1]
    ys, xs = np.where(band > 0)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
