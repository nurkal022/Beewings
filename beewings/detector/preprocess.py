"""Wing isolation: separate the translucent wing from the bright background.

Input is a BGR scan with a wing on light background (paper or glass).
Output is a binary mask (uint8: 0/255) covering the wing region.

Strategy:
    1. To grayscale + light Gaussian blur.
    2. Otsu threshold INV (wing is darker than the white paper, so we invert).
    3. Morphological close to fill internal holes left by translucent membrane.
    4. Largest connected component — discards dust and the small bee leg
       sometimes visible at the edge of these scans.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class WingMask:
    mask: np.ndarray         # uint8 binary, 0 or 255
    gray: np.ndarray         # the grayscale image used (uint8)
    bbox: tuple              # (x, y, w, h) bounding box of the wing
    centroid: tuple          # (cx, cy) mass centroid
    orientation: float       # radians, angle of major axis vs image X axis
    scale: float             # major-axis length in pixels (rough span estimate)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    # index 0 is background; pick the largest of the rest
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    return (labels == biggest).astype(np.uint8) * 255


def _moments_pose(mask: np.ndarray) -> tuple:
    """Centroid, principal-axis angle (radians), major-axis length.

    The axis direction is canonicalized so its X component is positive — i.e.
    the angle is always in (-pi/2, pi/2]. Without this, two wings of the same
    physical orientation can come out 180° apart due to eigenvector sign,
    which destroys downstream pose averaging.
    """
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        h, w = mask.shape
        return (w / 2, h / 2), 0.0, max(w, h)
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, -1]
    if major[0] < 0:                  # force +X direction
        major = -major
    angle = float(np.arctan2(major[1], major[0]))
    span = 4.0 * float(np.sqrt(eigvals[-1]))
    return (float(centroid[0]), float(centroid[1])), angle, span


def isolate_wing(bgr: np.ndarray) -> WingMask:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu finds the dark vein scaffold reliably, but misses the translucent
    # lower membrane (which is only slightly darker than the background).
    _, dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                            iterations=2)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                            iterations=1)
    dark = _largest_component(dark)

    # The wing as a whole is convex enough that the convex hull of the
    # vein-scaffold mask covers the translucent area too. This gives us a
    # full-wing mask whose centroid + PCA describe the whole wing — without
    # which the projected mean shape sits on the dark top half only.
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        mask = dark
    else:
        biggest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(biggest)
        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)
        # Slight erosion to stay just inside the wing border (avoids the bright
        # halo around the wing edge contaminating the ridge filter later).
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                         iterations=1)

    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        bbox = (0, 0, w, h)
    else:
        bbox = (int(xs.min()), int(ys.min()),
                int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    centroid, theta, major = _moments_pose(mask)
    return WingMask(mask=mask, gray=gray, bbox=bbox, centroid=centroid,
                    orientation=theta, scale=major)
