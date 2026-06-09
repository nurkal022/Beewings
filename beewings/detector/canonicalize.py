"""Find a consistent landmark ordering across specimens.

Problem: the annotator placed points by color group (1-8 red, 9-12 magenta,
13-19 cyan) but the order *within* each group may differ between wings.
So id "L9" on wing A and "L9" on wing B can refer to different biological
landmarks. Mean-shape modeling assumes consistent ids, so this breaks
training.

Fix (iterative consensus):
    1. Build a tentative mean shape from current labels.
    2. For each specimen, for each color group, find the permutation of its
       points that minimizes total distance to the current mean (Hungarian on
       distance matrix, restricted to the group's ids).
    3. Re-apply ids per the chosen permutation.
    4. Recompute mean shape.
    5. Repeat until no specimen changes anymore (typically 3-6 iterations).

We never invent points — we only relabel existing ones, preserving x/y values
and group membership.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .preprocess import isolate_wing
from .shape_model import _pose_of_points, _to_local


# Hard-coded grouping that matches the dataset's color scheme.
GROUPS: Dict[str, List[int]] = {
    "red":     list(range(1, 9)),       # 1..8
    "magenta": list(range(9, 13)),      # 9..12
    "cyan":    list(range(13, 20)),     # 13..19
}


def _load_csv(csv_path: Path, profile: str) -> Dict[str, Dict[int, Tuple[float, float]]]:
    out: Dict[str, Dict[int, Tuple[float, float]]] = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["profile"] != profile or int(row["skipped"]):
                continue
            out.setdefault(row["image"], {})[int(row["point_id"])] = (
                float(row["x"]), float(row["y"])
            )
    return out


def _normalize_to_local(pts_by_id: Dict[int, Tuple[float, float]],
                        image_root: Path, image_name: str) -> Dict[int, np.ndarray]:
    """Map each landmark from image coords into landmark-local frame."""
    bgr = cv2.imread(str(image_root / image_name))
    if bgr is None:
        return {}
    wm = isolate_wing(bgr)
    coords = np.array([pts_by_id[i] for i in sorted(pts_by_id)], dtype=np.float32)
    ids = sorted(pts_by_id)
    # Use the *landmark cloud* pose, not the mask pose — mask centroid drifts
    # if segmentation is imperfect on a given wing, but the landmark cloud
    # pose is intrinsic to the points themselves and gives a clean frame.
    lm_c, lm_a, lm_s = _pose_of_points(coords)
    local = _to_local(coords, lm_c, lm_a, lm_s)
    return {ids[i]: local[i] for i in range(len(ids))}


def _compute_mean_shape(locals_per_image: Dict[str, Dict[int, np.ndarray]],
                        all_ids: List[int]) -> Dict[int, np.ndarray]:
    sums = {lid: np.zeros(2, dtype=np.float64) for lid in all_ids}
    counts = {lid: 0 for lid in all_ids}
    for img, d in locals_per_image.items():
        for lid, xy in d.items():
            sums[lid] += xy
            counts[lid] += 1
    return {lid: (sums[lid] / counts[lid]) if counts[lid] else np.zeros(2)
            for lid in all_ids}


def _best_permutation(local_points: Dict[int, np.ndarray],
                      mean_local: Dict[int, np.ndarray],
                      group_ids: List[int]) -> Dict[int, int]:
    """For one specimen, one group: find old_id -> new_id mapping that
    minimizes the sum of squared distances to the mean shape."""
    available_ids = [i for i in group_ids if i in local_points]
    if not available_ids:
        return {}
    target_ids = [i for i in group_ids if i in mean_local]
    if not target_ids:
        return {i: i for i in available_ids}

    # Cost[i, j] = distance from specimen's point currently labeled `available_ids[i]`
    #             to the mean position of landmark `target_ids[j]`.
    pts = np.stack([local_points[i] for i in available_ids], axis=0)        # (Na, 2)
    targets = np.stack([mean_local[j] for j in target_ids], axis=0)         # (Nt, 2)
    diff = pts[:, None, :] - targets[None, :, :]                            # (Na, Nt, 2)
    cost = np.linalg.norm(diff, axis=2)                                     # (Na, Nt)
    rows, cols = linear_sum_assignment(cost)
    mapping: Dict[int, int] = {}
    for r, c in zip(rows, cols):
        old_id = available_ids[r]
        new_id = target_ids[c]
        mapping[old_id] = new_id
    return mapping


def canonicalize(csv_path: Path, image_root: Path, out_csv: Path,
                 profile: str = "19-point", max_iter: int = 12,
                 verbose: bool = True) -> Tuple[int, int]:
    """Run iterative consensus on the CSV and write a new one.

    Returns: (n_iterations, n_changes_in_last_iter).
    """
    truth = _load_csv(csv_path, profile)
    all_ids = sorted({i for d in truth.values() for i in d})

    # Pre-compute the landmark-local coordinates for every specimen (these don't
    # depend on labelling — only on point positions in image space).
    base_coords: Dict[str, Dict[int, Tuple[float, float]]] = {
        img: dict(d) for img, d in truth.items()
    }

    # Working labels per specimen — we will permute within groups.
    labels: Dict[str, Dict[int, int]] = {
        img: {i: i for i in d} for img, d in truth.items()      # id -> id
    }

    def apply_labels_to_local(img: str) -> Dict[int, np.ndarray]:
        # Re-normalize this specimen's coords each iteration using
        # *current* labels: pts list is the same, only ids attached differ.
        coords = base_coords[img]
        ids_sorted = sorted(coords)
        pts = np.array([coords[i] for i in ids_sorted], dtype=np.float32)
        lm_c, lm_a, lm_s = _pose_of_points(pts)
        local = _to_local(pts, lm_c, lm_a, lm_s)
        # Apply current labels: original_id -> new_id
        cur = labels[img]
        return {cur[ids_sorted[i]]: local[i] for i in range(len(ids_sorted))}

    last_changes = -1
    iters = 0
    for it in range(max_iter):
        iters = it + 1
        locals_per_image = {img: apply_labels_to_local(img) for img in truth}
        mean_local = _compute_mean_shape(locals_per_image, all_ids)

        changes = 0
        for img in truth:
            d_local = locals_per_image[img]
            for group_name, gids in GROUPS.items():
                # Filter to group ids present in this specimen.
                present = [lid for lid in gids if lid in d_local]
                if len(present) < 2:
                    continue
                # Build sub-mean for the group only.
                local_subset = {lid: d_local[lid] for lid in present}
                mapping = _best_permutation(local_subset, mean_local, gids)
                # Compose the new labels with the existing label dictionary.
                # mapping is in *current* id space; we need to update base->current.
                old_to_cur = labels[img]
                cur_to_new = mapping
                # Build new old_to_new
                new_map = {}
                for orig_id, cur_id in old_to_cur.items():
                    if cur_id in cur_to_new:
                        new_id = cur_to_new[cur_id]
                        new_map[orig_id] = new_id
                        if new_id != cur_id:
                            changes += 1
                    else:
                        new_map[orig_id] = cur_id
                labels[img] = new_map

        if verbose:
            print(f"  iter {iters}: {changes} label changes")
        if changes == 0:
            last_changes = 0
            break
        last_changes = changes

    # Write out the new CSV: same rows, but point_id replaced by the consensus label.
    with csv_path.open(encoding="utf-8") as fin, out_csv.open("w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            if row["profile"] != profile or int(row["skipped"]):
                writer.writerow(row)
                continue
            img = row["image"]
            old_id = int(row["point_id"])
            new_id = labels.get(img, {}).get(old_id, old_id)
            row["point_id"] = str(new_id)
            writer.writerow(row)

    return iters, last_changes


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Canonicalize landmark ids within color groups using "
                    "iterative consensus across specimens.")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--profile", default="19-point")
    args = p.parse_args(argv)
    iters, changes = canonicalize(args.csv, args.images, args.out, profile=args.profile)
    print(f"\nDone after {iters} iterations (last iter changes = {changes}).")
    print(f"Canonical CSV written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
