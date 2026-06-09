"""Vein extraction → skeletonization → junction detection.

The wing's vein network is darker than the membrane. We isolate veins inside
the wing mask using local contrast (CLAHE + adaptive threshold + black-hat),
skeletonize them down to 1-pixel-wide lines, then locate pixels where 3 or
more skeleton neighbors meet (junctions). Endpoint pixels (1 neighbor) are
returned separately — some peripheral landmarks correspond to vein endpoints
where a vein meets the wing edge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from scipy.signal import convolve2d
from skimage.filters import meijering
from skimage.morphology import remove_small_objects, skeletonize


@dataclass
class VeinSkeleton:
    veins_binary: np.ndarray   # uint8 0/255: binarized vein pixels (pre-thinning)
    skeleton: np.ndarray       # bool: 1-px wide skeleton inside the wing mask
    junctions: List[Tuple[int, int]]   # (x, y) pixel coords with 3+ skeleton neighbors
    endpoints: List[Tuple[int, int]]   # (x, y) pixel coords with exactly 1 neighbor


# 3×3 kernel that, when convolved with a binary skeleton, gives at each pixel
# the count of its 8 neighbors that are also "on". Used to classify each
# skeleton pixel as endpoint / interior / junction.
_NEIGHBOR_KERNEL = np.array([[1, 1, 1],
                             [1, 0, 1],
                             [1, 1, 1]], dtype=np.uint8)


def extract_veins(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a binary (uint8 0/255) image of vein pixels inside the wing.

    The wing membrane has a granular dotted texture that gets misread as
    "veins" by isotropic dark-region detectors (black-hat, adaptive threshold).
    Meijering's neuriteness filter is a ridge detector — it responds strongly
    to elongated tubular structures (veins) and weakly to blob-shaped texture
    (membrane dots), giving a much cleaner binary.
    """
    # Slightly blur to take the edge off the granular dot texture before the
    # ridge filter. Keeps thin veins intact while smoothing 1-2px dots.
    g = cv2.bilateralFilter(gray, d=5, sigmaColor=25, sigmaSpace=5)
    # Erode the wing mask a bit to avoid the bright wing border ringing.
    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                      iterations=1)
    # Meijering on the *inverted* grayscale because veins are dark.
    inv = (255 - g)
    # `sigmas` set to the half-thickness of typical veins (in px) — small.
    ridge = meijering(inv.astype(np.float32) / 255.0,
                      sigmas=(1.0, 1.5, 2.0, 2.5),
                      black_ridges=False)
    # ridge is in [0, 1]. Normalize to robust range.
    rmax = np.percentile(ridge[inner > 0], 99.5) if inner.any() else 1.0
    ridge = np.clip(ridge / max(rmax, 1e-6), 0, 1)
    # Threshold: keep top ~ridge response inside the wing.
    binv = (ridge > 0.18).astype(np.uint8) * 255
    binv = cv2.bitwise_and(binv, inner)
    # Close 1-px gaps along the ridge; keep structure thin.
    binv = cv2.morphologyEx(binv, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                            iterations=1)
    return binv


def _classify_skeleton(skel: np.ndarray) -> tuple:
    """Return (junctions, endpoints) lists of (x, y) coords."""
    sk_u8 = skel.astype(np.uint8)
    neigh = convolve2d(sk_u8, _NEIGHBOR_KERNEL, mode="same", boundary="fill", fillvalue=0)
    neigh = neigh * sk_u8  # only count at skeleton pixels
    junc_yx = np.argwhere(neigh >= 3)
    end_yx = np.argwhere(neigh == 1)
    return [(int(x), int(y)) for y, x in junc_yx], [(int(x), int(y)) for y, x in end_yx]


def _cluster_close(points: List[Tuple[int, int]], radius: int = 4) -> List[Tuple[float, float]]:
    """Merge pixels that lie within `radius` of each other into one centroid.

    A junction in a thick vein can spread over a few skeleton pixels. We don't
    want to count them as separate candidates.
    """
    if not points:
        return []
    pts = np.array(points, dtype=np.float32)
    used = np.zeros(len(pts), dtype=bool)
    out: List[Tuple[float, float]] = []
    r2 = radius * radius
    for i in range(len(pts)):
        if used[i]:
            continue
        d2 = ((pts - pts[i]) ** 2).sum(axis=1)
        group_mask = d2 <= r2
        group = pts[group_mask]
        used |= group_mask
        cx, cy = group.mean(axis=0)
        out.append((float(cx), float(cy)))
    return out


def skeletonize_and_locate(gray: np.ndarray, mask: np.ndarray,
                           min_object_size: int = 250,
                           junction_merge_radius: int = 5) -> VeinSkeleton:
    veins = extract_veins(gray, mask)
    veins_bool = veins > 0
    # Drop tiny noise components before thinning. Vein segments are long;
    # the membrane texture passes Meijering only as short specks, so a higher
    # min_size kills them without eating real veins.
    veins_bool = remove_small_objects(veins_bool, min_size=min_object_size, connectivity=2)
    skel = skeletonize(veins_bool)
    junctions, endpoints = _classify_skeleton(skel)
    junctions = _cluster_close(junctions, radius=junction_merge_radius)
    endpoints = _cluster_close(endpoints, radius=junction_merge_radius)
    # Cast back to int tuples for downstream code; sub-pixel comes later.
    junctions = [(float(x), float(y)) for x, y in junctions]
    endpoints = [(float(x), float(y)) for x, y in endpoints]
    return VeinSkeleton(
        veins_binary=(veins_bool.astype(np.uint8) * 255),
        skeleton=skel,
        junctions=junctions,
        endpoints=endpoints,
    )
