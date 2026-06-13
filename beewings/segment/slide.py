"""Background estimation and handwritten-label detection for scan slides.

The label is always on the left of the slide, separated from the wing grid by
an empty vertical gap. We detect dark foreground content, then find the gap and
treat everything left of it as the label.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

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
    gap_frac: float = 0.006,
) -> Optional[Box]:
    """Return (x, y, w, h) of the left label block, or None if absent.

    The handwritten label sits on the far left, separated from the wing grid by
    an empty vertical band. We scan columns of the left `search_frac`, skip the
    faint outer margin to find where substantial content starts, then look for
    the first empty band at least `gap_frac * W` wide that separates the label
    from the wings. The bounding box of all content left of that band is the
    label. `gap_frac` is small because on high-resolution scans the separating
    band is only a few percent of the width, while character gaps are smaller.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    fg = _foreground(gray, bg)

    col = fg.sum(axis=0)  # per-column foreground amount
    search_w = int(w * search_frac)
    if col[:search_w].max() == 0:
        return None  # no content on the left at all

    gap_w = max(5, int(w * gap_frac))
    # A column is "empty" when it holds only negligible foreground. The threshold
    # is relative to the busiest column so faint speckle counts as empty.
    empty = col <= (0.01 * col.max())

    # Where does substantial label content begin? Skip the faint outer margin.
    substantial = np.where(~empty[:search_w])[0]
    if substantial.size == 0:
        return None
    first = int(substantial[0])
    # A real label is always anchored to the far-left edge; if substantial
    # content starts beyond 15 % of the width it belongs to the wing grid.
    if first > int(w * 0.15):
        return None

    # Scan rightward from the label start for the first separating empty band.
    run = 0
    gap_start = None
    for x in range(first, search_w):
        if empty[x]:
            run += 1
            if run >= gap_w:
                gap_start = x - run + 1
                break
        else:
            run = 0
    if gap_start is None:
        return None  # content never closed with a gap -> not a label block

    band = fg[:, :gap_start]
    ys, xs = np.where(band > 0)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def label_box_from_wings(
    img: np.ndarray,
    bg: float,
    wing_boxes: List[Box],
    gap_frac: float = 0.01,
    min_w_frac: float = 0.03,
) -> Optional[Box]:
    """Derive the label box as foreground content left of the leftmost wing.

    The wing grid is found first (label text removed by the geometric wing
    filter), so the wings themselves bound where the label can be. We take all
    dark content strictly left of the leftmost wing, minus a small gap. Because
    the cut is *left of every wing*, this can never eat a real wing — the
    deliberately safe bias, since a lost wing costs a manual re-add while a
    stray label box costs only a manual delete.

    Returns None when there is no room for a label (wings reach the edge), when
    the left content is too slight to be a label (`min_w_frac` of the slide
    width), or when content is not anchored near the far-left edge.
    """
    if not wing_boxes:
        return None
    h, w = img.shape[:2]
    leftmost = min(b[0] for b in wing_boxes)
    gap = max(5, int(w * gap_frac))
    cut = leftmost - gap
    if cut <= 0:
        return None  # wings start at the very edge -> no room for a label

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fg = _foreground(gray, bg)
    band = fg[:, :cut]
    ys, xs = np.where(band > 0)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    # A real label is anchored near the far-left edge and is a substantial
    # block; reject mid-slide speckle and slivers.
    if x0 > int(w * 0.15):
        return None
    if (x1 - x0 + 1) < int(w * min_w_frac):
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
