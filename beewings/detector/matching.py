"""Match junction/endpoint candidates to expected landmark positions.

Given:
    - candidates: list of (x, y) points detected on the wing
    - expected:   (N, 2) array of expected landmark positions in image coords
                  (output of ShapeModel.project)
    - radius:     per-landmark allowed distance from expected to candidate

Output:
    Dict[landmark_id, (x, y)] for landmarks that received a match.
    Landmarks without an in-range candidate are simply omitted (the GUI will
    leave them blank for the annotator to place manually).

Algorithm:
    Build a cost matrix [N candidates x M expected]; cost = distance, with
    +infinity replaced by 10 * radius for distances exceeding the per-landmark
    gate. Solve as a rectangular assignment problem.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_landmarks(
    candidates: Sequence[Tuple[float, float]],
    expected: np.ndarray,                   # (M, 2)
    landmark_ids: Sequence[int],            # length M
    gate_radius: float = 60.0,              # px; max distance for a valid match
) -> Dict[int, Tuple[float, float]]:
    if not candidates or expected.size == 0:
        return {}
    cand = np.asarray(candidates, dtype=np.float32)        # (N, 2)
    exp = np.asarray(expected, dtype=np.float32)           # (M, 2)
    n, m = len(cand), len(exp)
    # Distance matrix
    diff = cand[:, None, :] - exp[None, :, :]              # (N, M, 2)
    dist = np.linalg.norm(diff, axis=2)                    # (N, M)
    cost = dist.copy()
    # Soft penalty for out-of-gate matches so Hungarian still works but those
    # assignments will be filtered out afterwards.
    cost[cost > gate_radius] = gate_radius * 10.0

    # linear_sum_assignment handles rectangular matrices natively.
    rows, cols = linear_sum_assignment(cost)
    out: Dict[int, Tuple[float, float]] = {}
    for r, c in zip(rows, cols):
        if dist[r, c] <= gate_radius:
            out[int(landmark_ids[c])] = (float(cand[r, 0]), float(cand[r, 1]))
    return out
