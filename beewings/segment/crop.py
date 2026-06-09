"""Crop a wing out of the full-resolution scan, with optional canonical rotate.

Default output is a plain axis-aligned rectangle (the chosen behaviour). With
`rotate=True` the wing is rotated so its long axis is horizontal and the narrow
(base/articulation) end is on the left — matching how the ML model was trained.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]


def crop_box(img: np.ndarray, box: Box, margin: float = 0.10) -> np.ndarray:
    """Plain rectangular crop with a fractional margin, clipped to image bounds."""
    x, y, w, h = box
    mx = int(round(w * margin))
    my = int(round(h * margin))
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(img.shape[1], x + w + mx)
    y1 = min(img.shape[0], y + h + my)
    return img[y0:y1, x0:x1].copy()


def crop_wing(
    img: np.ndarray,
    box: Box,
    margin: float = 0.10,
    rotate: bool = False,
    bg: float = 255.0,
) -> np.ndarray:
    """Crop one wing. If `rotate`, canonicalize to horizontal, base on the left."""
    patch = crop_box(img, box, margin)
    if not rotate:
        return patch

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    fg = (gray < (bg - 25)).astype(np.uint8)
    ys, xs = np.where(fg > 0)
    if xs.size < 10:
        return patch  # nothing to orient

    # Principal axis via PCA on foreground pixel coordinates.
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    angle = np.degrees(np.arctan2(major[1], major[0]))

    h, w = patch.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += nw / 2.0 - w / 2.0
    M[1, 2] += nh / 2.0 - h / 2.0
    rotated = cv2.warpAffine(patch, M, (nw, nh), borderValue=(int(bg),) * 3)

    # Ensure base (narrow end) is on the left: compare foreground height of the
    # left vs right 20% columns; the narrower side is the base.
    rg = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    rfg = (rg < (bg - 25)).astype(np.uint8)
    col_h = rfg.sum(axis=0)
    band = max(1, rfg.shape[1] // 5)
    left_h = col_h[:band].mean()
    right_h = col_h[-band:].mean()
    if left_h > right_h:  # base currently on the right -> flip 180
        rotated = cv2.rotate(rotated, cv2.ROTATE_180)
    return rotated
