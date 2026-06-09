"""Debug overlays for inspecting each stage of the pipeline.

Each `*_overlay` returns a BGR uint8 image suitable for cv2.imwrite.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

from .preprocess import WingMask
from .skeleton import VeinSkeleton


def mask_overlay(bgr: np.ndarray, wm: WingMask) -> np.ndarray:
    out = bgr.copy()
    blue = np.zeros_like(out)
    blue[..., 0] = wm.mask  # blue channel highlight
    out = cv2.addWeighted(out, 0.7, blue, 0.3, 0)
    cv2.rectangle(out, (wm.bbox[0], wm.bbox[1]),
                  (wm.bbox[0] + wm.bbox[2], wm.bbox[1] + wm.bbox[3]),
                  (0, 255, 255), 2)
    cv2.circle(out, (int(wm.centroid[0]), int(wm.centroid[1])), 6, (0, 0, 255), -1)
    # Major-axis line for visual sanity.
    L = wm.scale / 2
    dx = L * np.cos(wm.orientation)
    dy = L * np.sin(wm.orientation)
    p1 = (int(wm.centroid[0] - dx), int(wm.centroid[1] - dy))
    p2 = (int(wm.centroid[0] + dx), int(wm.centroid[1] + dy))
    cv2.line(out, p1, p2, (0, 255, 255), 2)
    return out


def skeleton_overlay(bgr: np.ndarray, sk: VeinSkeleton) -> np.ndarray:
    out = bgr.copy()
    # Highlight skeleton in green.
    ys, xs = np.where(sk.skeleton)
    out[ys, xs] = (60, 255, 60)
    # Dilate by 1 px so it's visible on a downscaled preview.
    return out


def junctions_overlay(bgr: np.ndarray, sk: VeinSkeleton,
                      draw_endpoints: bool = True) -> np.ndarray:
    out = skeleton_overlay(bgr, sk)
    if draw_endpoints:
        for x, y in sk.endpoints:
            cv2.circle(out, (int(round(x)), int(round(y))), 3, (255, 200, 0), -1)
    for x, y in sk.junctions:
        cv2.circle(out, (int(round(x)), int(round(y))), 6, (0, 0, 255), 2)
    return out


def landmarks_overlay(
    bgr: np.ndarray,
    predicted: Iterable[Tuple[int, Tuple[float, float]]],
    ground_truth: Optional[Iterable[Tuple[int, Tuple[float, float]]]] = None,
) -> np.ndarray:
    """Draw predicted landmarks (red) and, if given, ground truth (green) with id labels."""
    out = bgr.copy()
    if ground_truth is not None:
        for lid, (x, y) in ground_truth:
            cv2.circle(out, (int(round(x)), int(round(y))), 5, (0, 200, 0), 2)
    for lid, (x, y) in predicted:
        cv2.circle(out, (int(round(x)), int(round(y))), 5, (0, 0, 220), -1)
        cv2.putText(out, str(lid), (int(round(x)) + 6, int(round(y)) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1, cv2.LINE_AA)
    return out
