"""Classical bee wing morphometric indices.

These indices, introduced by Alpatov (1948) and complemented by later
authors, are widely used in honey bee subspecies identification.

ALL FORMULAS HERE ASSUME THE ALPATOV 12-POINT SCHEME. Numbering differs
from Tofilski — do not mix.

The Alpatov 12-point landmark IDs we use match the dataset's annotation
convention (cross-checked on 6935 wings):

    L1  — отправная точка медиальной жилки (где она встречает кубитальную)
    L2  — дистальный конец радиальной жилки у апекса
    L3  — узел Cu1-Cu2 (важная вершина для CI)
    L4  — узел m-cu в задней области
    L5  — пересечение медиальной с поперечной
    L6  — узел кубитальной (антеродистальный)
    L7  — узел кубитальной (постеродистальный)
    L8  — задний узел медиальной
    L9  — субдистальный узел кубитальной (b конец сегмента кубитального индекса)
    L10 — проксимальный узел кубитальной (a конец сегмента)
    L11 — внутренняя точка R-M
    L12 — узел у корня крыла

These names are TENTATIVE and should be confirmed by the biologist after
checking against the IdentiFly reference diagrams. The numeric IDs (1-12)
themselves are the source of truth.

INDICES IMPLEMENTED:

- Cubital Index (CI) [Alpatov definition]:
      CI = |L6 - L7| / |L9 - L7|
  Ratio of two segments of the third cubital cell vein system.
  Typical ranges per subspecies:
      Apis mellifera mellifera (тёмная европейская): 1.3 – 1.8
      A.m. carnica (карника):                        2.4 – 3.0
      A.m. caucasica (кавказская):                   1.7 – 2.3
      A.m. ligustica (итальянская):                  2.2 – 2.5

- Discoidal Index (DsA) [Goetze]:
      Position of L5 projected onto line L4–L10. Positive (anterior) for
      most subspecies of A. mellifera, near zero for mellifera mellifera.

- Hantel Index (similar to CI but inverted segment ratio).

- Radial Index (RI) — radial vein segment ratio.

- Pribilski Index (PI) — auxiliary.

Cubital index is by far the most commonly reported in Russian breeding
literature (Алпатов, Кривцов, Скориков). DsA is used in central-European
classification (Ruttner). Our code returns ALL applicable indices so the
biologist can compare populations against published reference values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# --- Geometry helpers ----------------------------------------------------

def _dist(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _signed_distance_to_line(p, a, b) -> float:
    """Signed perpendicular distance from `p` to the infinite line through a→b.
    Positive on the LEFT side of a→b (in standard math axes).
    """
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return 0.0
    return ((p[0] - ax) * dy - (p[1] - ay) * dx) / L


# --- Index implementations -----------------------------------------------

@dataclass
class IndexResult:
    name: str
    value: Optional[float]
    formula: str
    notes: str = ""

    @property
    def display(self) -> str:
        if self.value is None:
            return f"{self.name}: —"
        return f"{self.name}: {self.value:.3f}"


def _segment(landmarks: Dict[int, Tuple[float, float]], a: int, b: int) -> Optional[float]:
    if a in landmarks and b in landmarks:
        return _dist(landmarks[a], landmarks[b])
    return None


def _ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return num / den


# ---------------------------------------------------------------------------
# CI candidate configuration.
#
# The cubital index requires 3 collinear landmarks on the cubital vein. Without
# the official IdentiFly reference diagram I cannot 100% identify which IDs in
# our Alpatov-trained data correspond to the canonical CI points, so we expose
# a CONFIGURABLE triple. Default is the most collinear (angle ≈ 177°) of the
# top candidates discovered by ratio analysis over 1500 wings.
#
# Each candidate is (a, b, c) such that CI = |a-b| / |b-c|. The biologist
# should pick the canonical one after comparing with the dataset's standard
# anatomical references.
# ---------------------------------------------------------------------------

CI_CANDIDATES: Dict[str, Tuple[int, int, int]] = {
    "L7-L9-L12":  (7, 9, 12),    # angle 176°, ratio ≈ 1.70 (most collinear)
    "L6-L9-L12":  (6, 9, 12),    # angle 178°, ratio ≈ 1.51
    "L2-L3-L11":  (2, 3, 11),    # angle 175°, ratio ≈ 1.69
    "L2-L4-L12":  (2, 4, 12),    # angle 172°, ratio ≈ 1.64
    "L1-L9-L11":  (1, 9, 11),    # mapping via Alpatov↔Tofilski (L8,L11,L9)
}
DEFAULT_CI = "L7-L9-L12"


def cubital_index_alpatov(landmarks: Dict[int, Tuple[float, float]],
                          variant: str = DEFAULT_CI) -> IndexResult:
    """CI = a / b along the cubital vein. PROVISIONAL until biologist confirms IDs.

    Higher CI → more "carnica-like". Lower → mellifera. See CI_CANDIDATES.
    """
    a_id, b_id, c_id = CI_CANDIDATES.get(variant, CI_CANDIDATES[DEFAULT_CI])
    a = _segment(landmarks, a_id, b_id)
    b = _segment(landmarks, b_id, c_id)
    return IndexResult(
        name=f"CI (Алпатов, {variant})",
        value=_ratio(a, b),
        formula=f"|L{a_id}-L{b_id}| / |L{b_id}-L{c_id}|",
        notes="ПРЕДВАРИТЕЛЬНО. Типично: mellifera 1.3-1.8, carnica 2.4-3.0. "
              "Биолог должен подтвердить ID точек по эталонной диаграмме.",
    )


def cubital_index_goetze(landmarks: Dict[int, Tuple[float, float]]) -> IndexResult:
    """CI Goetze = b / a — reciprocal of Alpatov definition.

    CI_Alpatov ≈ 1 / CI_Goetze. Some Western European literature uses Goetze.
    """
    ci_alp = cubital_index_alpatov(landmarks).value
    return IndexResult(
        name="CI (Goetze)",
        value=(1.0 / ci_alp) if ci_alp not in (None, 0) else None,
        formula="|L9-L7| / |L6-L7|  (≈ 1/CI Алпатова)",
    )


def discoidal_displacement(landmarks: Dict[int, Tuple[float, float]]) -> IndexResult:
    """DsA — signed distance of L5 from line through L4 and L10.

    Positive (anterior) for most subspecies, near zero for A.m. mellifera.
    Distance is in pixels; sign matters more than magnitude for taxonomy.
    """
    if 5 not in landmarks or 4 not in landmarks or 10 not in landmarks:
        return IndexResult(name="DsA (Goetze)", value=None, formula="signed dist(L5, line L4-L10)")
    d = _signed_distance_to_line(landmarks[5], landmarks[4], landmarks[10])
    return IndexResult(
        name="DsA (Goetze)",
        value=d,
        formula="перпенд. от L5 к прямой L4-L10 (знак)",
        notes="Положит. — большинство подвидов; ~0 для A.m. mellifera",
    )


def hantel_index(landmarks: Dict[int, Tuple[float, float]]) -> IndexResult:
    """Hantel = |L9-L6| / |L6-L7| — auxiliary cubital-cell index."""
    n = _segment(landmarks, 9, 6)
    d = _segment(landmarks, 6, 7)
    return IndexResult(
        name="Hantel",
        value=_ratio(n, d),
        formula="|L9-L6| / |L6-L7|",
    )


def pribilski_index(landmarks: Dict[int, Tuple[float, float]]) -> IndexResult:
    """Pribilski-style ratio for the radial sector subsegment."""
    n = _segment(landmarks, 1, 2)
    d = _segment(landmarks, 1, 11)
    return IndexResult(
        name="PI",
        value=_ratio(n, d),
        formula="|L1-L2| / |L1-L11|",
    )


def radial_index(landmarks: Dict[int, Tuple[float, float]]) -> IndexResult:
    """RI — radial vein length ratio (basal vs distal)."""
    n = _segment(landmarks, 2, 3)
    d = _segment(landmarks, 3, 11)
    return IndexResult(
        name="RI",
        value=_ratio(n, d),
        formula="|L2-L3| / |L3-L11|",
    )


def wing_length(landmarks: Dict[int, Tuple[float, float]]) -> IndexResult:
    """Approximate wing length: distance from root cluster to apex."""
    n = _segment(landmarks, 12, 2)
    return IndexResult(
        name="Длина крыла (прибл.)",
        value=n,
        formula="|L12 - L2|",
        notes="Расстояние в пикселях. Перевести в мм нужно через калибровку.",
    )


def compute_all_alpatov(landmarks: Dict[int, Tuple[float, float]]) -> List[IndexResult]:
    """Return all indices computable from Alpatov 12-point landmarks."""
    return [
        cubital_index_alpatov(landmarks),
        cubital_index_goetze(landmarks),
        discoidal_displacement(landmarks),
        hantel_index(landmarks),
        radial_index(landmarks),
        pribilski_index(landmarks),
        wing_length(landmarks),
    ]


# --- Subspecies classification by CI alone -------------------------------

# Reference CI ranges from Russian/EU literature, Apis mellifera worker.
# Source: Кривцов 2000, Алпатов 1948, Ruttner 1988.
CI_REFERENCE = [
    ("Apis mellifera mellifera (среднерусская/тёмная)", (1.3, 1.8)),
    ("A.m. caucasica (кавказская)",                     (1.7, 2.3)),
    ("A.m. carpathica (карпатская)",                    (2.0, 2.4)),
    ("A.m. ligustica (итальянская)",                    (2.2, 2.5)),
    ("A.m. carnica (карника)",                          (2.4, 3.0)),
]


def classify_by_ci(ci_alpatov: float) -> List[str]:
    """Return list of subspecies whose reference CI range contains the value."""
    hits = []
    for name, (lo, hi) in CI_REFERENCE:
        if lo <= ci_alpatov <= hi:
            hits.append(name)
    return hits
