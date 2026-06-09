"""Mean shape model built from ground-truth annotations.

For each training wing we compute TWO poses:
    - MASK pose: centroid + principal angle + major-axis length of the wing mask.
                 This is the only pose we have for a *new* wing.
    - LANDMARK pose: centroid + principal angle + cloud span of the 19 landmarks.

We normalize each specimen's landmarks into their OWN landmark-pose frame
(so the mean shape is built on a clean, consistent scale that doesn't depend
on how much wing-tip we segment). We also learn the systematic relation
between mask pose and landmark pose, which lets us project the mean shape
onto a brand-new wing using only its mask pose.

Learned relation (mean across training wings):
    offset_local: where the landmark centroid sits in the mask-local frame
    rel_angle:    landmark_angle - mask_angle
    rel_scale:    landmark_span / mask_scale
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .preprocess import WingMask, isolate_wing


@dataclass
class ShapeModel:
    landmark_ids: List[int]
    mean_local: np.ndarray            # (N, 2) landmarks in landmark-local frame
    std_local: np.ndarray             # (N, 2) per-landmark std (anisotropic uncertainty)
    # Mask→landmark systematic relation:
    offset_local: np.ndarray          # (2,) where landmark centroid sits in mask-local frame
    rel_angle: float                  # landmark_angle - mask_angle (radians)
    rel_scale: float                  # landmark_span / mask_scale
    n_specimens: int

    def project(self, pose: WingMask) -> np.ndarray:
        """Project the mean shape onto image coords using only the mask pose."""
        # 1. Estimate landmark frame from mask frame.
        lm_scale = pose.scale * self.rel_scale
        lm_angle = pose.orientation + self.rel_angle
        # offset_local is in mask-local frame (mask scale, mask angle).
        # Convert to image coords by rotating with mask angle, scaling by mask scale.
        c_m, s_m = np.cos(pose.orientation), np.sin(pose.orientation)
        Rm = np.array([[c_m, -s_m], [s_m, c_m]], dtype=np.float32)
        offset_image = (Rm @ (self.offset_local * pose.scale))
        lm_cx = pose.centroid[0] + float(offset_image[0])
        lm_cy = pose.centroid[1] + float(offset_image[1])

        # 2. Project mean_local (in landmark-local frame) by landmark frame.
        c_l, s_l = np.cos(lm_angle), np.sin(lm_angle)
        Rl = np.array([[c_l, -s_l], [s_l, c_l]], dtype=np.float32)
        scaled = self.mean_local * lm_scale
        rotated = scaled @ Rl.T
        return rotated + np.array([lm_cx, lm_cy], dtype=np.float32)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "landmark_ids": self.landmark_ids,
            "mean_local": self.mean_local.tolist(),
            "std_local": self.std_local.tolist(),
            "offset_local": self.offset_local.tolist(),
            "rel_angle": self.rel_angle,
            "rel_scale": self.rel_scale,
            "n_specimens": self.n_specimens,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ShapeModel":
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            landmark_ids=list(d["landmark_ids"]),
            mean_local=np.array(d["mean_local"], dtype=np.float32),
            std_local=np.array(d["std_local"], dtype=np.float32),
            offset_local=np.array(d["offset_local"], dtype=np.float32),
            rel_angle=float(d["rel_angle"]),
            rel_scale=float(d["rel_scale"]),
            n_specimens=int(d["n_specimens"]),
        )


def _pose_of_points(pts: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Compute (centroid, principal_angle, span). Direction forced to +X side."""
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, -1]
    if major[0] < 0:                  # canonicalize: principal axis points roughly right
        major = -major
    angle = float(np.arctan2(major[1], major[0]))
    proj = centered @ major
    span = 4.0 * float(np.std(proj))
    return centroid.astype(np.float32), angle, span


def _to_local(pts: np.ndarray, centroid: np.ndarray, angle: float, span: float) -> np.ndarray:
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return ((pts - centroid) @ R.T) / max(span, 1e-6)


def build_from_csv(csv_path: Path, image_root: Path,
                   profile: str = "19-point") -> ShapeModel:
    # Load CSV grouped by image.
    by_image: Dict[str, List[Tuple[int, float, float]]] = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["profile"] != profile or int(row["skipped"]):
                continue
            by_image.setdefault(row["image"], []).append(
                (int(row["point_id"]), float(row["x"]), float(row["y"]))
            )

    landmark_ids = sorted({pid for items in by_image.values() for pid, *_ in items})
    id_to_idx = {lid: i for i, lid in enumerate(landmark_ids)}

    # Accumulators
    local_shapes: List[np.ndarray] = []                   # list of (N,2) normalized shapes
    valid_mask_array: List[np.ndarray] = []               # per-specimen presence flags
    rel_angles: List[float] = []
    rel_scales: List[float] = []
    offsets_local: List[np.ndarray] = []                  # in mask-local frame

    for img_name, lms in by_image.items():
        img_path = image_root / img_name
        if not img_path.exists():
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        wm = isolate_wing(bgr)

        idmap = {pid: (x, y) for pid, x, y in lms}
        ids = sorted(idmap.keys())
        pts = np.array([idmap[i] for i in ids], dtype=np.float32)

        lm_centroid, lm_angle, lm_span = _pose_of_points(pts)
        mask_angle_c = wm.orientation                 # already canonicalized in moments_pose
        lm_angle_c = lm_angle                         # already canonicalized in _pose_of_points

        # Normalize landmarks into their own frame.
        local_pts = _to_local(pts, lm_centroid, lm_angle_c, lm_span)

        # Some specimens may miss landmarks (skipped) — pad to full length.
        local_full = np.full((len(landmark_ids), 2), np.nan, dtype=np.float32)
        valid_full = np.zeros(len(landmark_ids), dtype=bool)
        for i, lid in enumerate(ids):
            local_full[id_to_idx[lid]] = local_pts[i]
            valid_full[id_to_idx[lid]] = True
        local_shapes.append(local_full)
        valid_mask_array.append(valid_full)

        # Mask-to-landmark relation
        rel_angles.append(lm_angle_c - mask_angle_c)
        rel_scales.append(lm_span / max(wm.scale, 1e-6))
        # Offset (landmark centroid - mask centroid) in mask-local coords.
        diff = lm_centroid - np.array(wm.centroid, dtype=np.float32)
        c, s = np.cos(-mask_angle_c), np.sin(-mask_angle_c)
        Rinv = np.array([[c, -s], [s, c]], dtype=np.float32)
        offsets_local.append((Rinv @ diff) / max(wm.scale, 1e-6))

    if not local_shapes:
        raise RuntimeError("No specimens loaded — check CSV and image_root paths.")

    stack = np.stack(local_shapes, axis=0)               # (S, N, 2)
    valid = np.stack(valid_mask_array, axis=0)           # (S, N)
    # Compute per-landmark mean and std, ignoring NaN where missing.
    counts = valid.sum(axis=0)                            # (N,)
    mean_local = np.zeros((len(landmark_ids), 2), dtype=np.float32)
    std_local = np.zeros((len(landmark_ids), 2), dtype=np.float32)
    for i in range(len(landmark_ids)):
        col = stack[valid[:, i], i, :]
        if len(col):
            mean_local[i] = col.mean(axis=0)
            std_local[i] = col.std(axis=0) if len(col) > 1 else np.array([0.05, 0.05], dtype=np.float32)

    offset_local = np.mean(np.stack(offsets_local, axis=0), axis=0)
    rel_angle = float(np.mean(rel_angles))
    rel_scale = float(np.mean(rel_scales))

    return ShapeModel(
        landmark_ids=landmark_ids,
        mean_local=mean_local,
        std_local=std_local,
        offset_local=offset_local.astype(np.float32),
        rel_angle=rel_angle,
        rel_scale=rel_scale,
        n_specimens=len(local_shapes),
    )
