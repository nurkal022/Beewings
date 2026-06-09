"""Live morphometric metrics computed from current landmark positions.

Routes to Alpatov-specific indices (CI, DsA, RI, etc.) when the active
profile follows the Alpatov methodology, and to generic distance metrics
for Tofilski (geometric morphometrics produces shape variables, not
linear indices).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .schema import WingAnnotation


def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def compute_metrics(ann: WingAnnotation) -> Dict[str, Optional[float]]:
    """Return a dict of metric_name -> value (or None if not computable).

    Routes by profile name.
    """
    pts: Dict[int, Tuple[float, float]] = {
        lm.id: (lm.x, lm.y)
        for lm in ann.landmarks
        if not lm.skipped and lm.x >= 0 and lm.y >= 0
    }

    metrics: Dict[str, Optional[float]] = {}

    # Alpatov-specific: classical indices
    if ann.profile in ("Алпатов 12 точек", "Алпатов 8 точек", "12-point", "8-point"):
        try:
            from .indices import compute_all_alpatov
            for ir in compute_all_alpatov(pts):
                metrics[ir.name] = ir.value
        except ImportError:
            pass

    def seg(a: int, b: int) -> Optional[float]:
        if a in pts and b in pts:
            return _dist(pts[a], pts[b])
        return None

    # Universal sanity distances (work for both schemes)
    metrics["размах L1 ↔ L2"] = seg(1, 2)
    metrics["размах L1 ↔ макс ID"] = seg(1, max(pts) if pts else 1)

    return metrics
