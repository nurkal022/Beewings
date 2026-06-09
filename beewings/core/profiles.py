"""Landmark profiles for bee wings.

Two INDEPENDENT methodologies are supported:

    Alpatov (8 / 12 points) — classical morphometry, Soviet/Russian school.
        Used to compute linear indices: cubital index (CI), discoidal (DsA),
        radial (RI), etc. Each point has a fixed biological role tied to
        which vein segment it terminates. Numbering is its OWN scheme.

    Tofilski (19 points) — modern geometric morphometrics, used by IdentiFly
        and DeepWings. Points are placed where blue circles tangent to vein
        outlines at >=3 points (geometrically stable). Numbering is its OWN
        scheme, different from Alpatov.

These two are NOT interchangeable. Even when two points sit at the same
pixel on the same wing, the biological INTERPRETATION differs and downstream
classification formulas expect specific point IDs per scheme. Mixing schemes
silently corrupts species identification.

In this project each profile has its OWN ML model checkpoint. The annotator
picks a profile up-front and stays in it for the whole session.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class LandmarkSpec:
    id: int
    label: str          # short human label, e.g. "L1"
    color: str          # hex color used on canvas
    description: str    # tooltip / side panel hint


@dataclass(frozen=True)
class Methodology:
    """A top-level scientific methodology — independent storage, model, ID space."""
    id: str                          # slug for filenames: "alpatov" | "tofilski"
    display_name: str                # human label
    short_label: str                 # 1-2 chars for badges, e.g. "А" / "Т"
    accent_color: str                # hex color for visual branding
    checkpoint_name: str             # default model file
    reference: str                   # citation / source
    description: str                 # multi-line tooltip


METHODOLOGIES: Dict[str, Methodology] = {
    "alpatov": Methodology(
        id="alpatov",
        display_name="Алпатов",
        short_label="А",
        accent_color="#c0392b",
        checkpoint_name="alpatov12.pt",
        reference="Алпатов В. В., 1948",
        description=(
            "Классическая (линейная) морфометрия. Точки выбраны для расчёта "
            "линейных индексов: кубитальный индекс (CI), дискоидальное смещение "
            "(DsA), радиальный (RI). Базовая схема — 8 точек, расширенная — 12. "
            "Используется в российской пчеловодческой литературе."
        ),
    ),
    "tofilski": Methodology(
        id="tofilski",
        display_name="Тофильский",
        short_label="Т",
        accent_color="#1f8a8a",
        checkpoint_name="tofilski19.pt",
        reference="Tofilski A., Apidologie 2017",
        description=(
            "Геометрическая морфометрия. 19 точек на пересечениях жилок, "
            "выбраны для устойчивости (вписанные окружности касаются жилок "
            "в ≥3 точках). Используется в ПО IdentiFly, DrawWing, DeepWings. "
            "Совместима с современным научным анализом форм Procrustes."
        ),
    ),
}


@dataclass(frozen=True)
class Profile:
    name: str
    landmarks: List[LandmarkSpec]
    methodology_id: str = "tofilski"  # which Methodology this profile belongs to
    checkpoint_name: str = ""         # may override methodology default
    description: str = ""

    @property
    def methodology(self) -> Methodology:
        return METHODOLOGIES[self.methodology_id]

    @property
    def ids(self) -> List[int]:
        return [lm.id for lm in self.landmarks]

    def get(self, lm_id: int) -> LandmarkSpec:
        for lm in self.landmarks:
            if lm.id == lm_id:
                return lm
        raise KeyError(lm_id)


_RED = "#e63946"
_MAGENTA = "#d041c8"
_CYAN = "#1bc7c0"


# Alpatov 8/12 - landmarks for classical morphometry.
# IDs and visual colors are local to this scheme; do NOT compare with Tofilski IDs.
_ALPATOV_12: List[LandmarkSpec] = [
    LandmarkSpec(1,  "A1",  _RED, "Алпатов L1"),
    LandmarkSpec(2,  "A2",  _RED, "Алпатов L2"),
    LandmarkSpec(3,  "A3",  _RED, "Алпатов L3"),
    LandmarkSpec(4,  "A4",  _RED, "Алпатов L4"),
    LandmarkSpec(5,  "A5",  _RED, "Алпатов L5"),
    LandmarkSpec(6,  "A6",  _RED, "Алпатов L6"),
    LandmarkSpec(7,  "A7",  _RED, "Алпатов L7"),
    LandmarkSpec(8,  "A8",  _RED, "Алпатов L8"),
    LandmarkSpec(9,  "A9",  _MAGENTA, "Алпатов L9 (расширение до 12)"),
    LandmarkSpec(10, "A10", _MAGENTA, "Алпатов L10"),
    LandmarkSpec(11, "A11", _MAGENTA, "Алпатов L11"),
    LandmarkSpec(12, "A12", _MAGENTA, "Алпатов L12"),
]

# Tofilski 19 — landmarks for geometric morphometrics (IdentiFly / DeepWings).
# IDs are independent of Alpatov scheme.
_TOFILSKI_19: List[LandmarkSpec] = [
    LandmarkSpec(1,  "T1",  _RED,     "Тофильский L1 (apex)"),
    LandmarkSpec(2,  "T2",  _RED,     "Тофильский L2"),
    LandmarkSpec(3,  "T3",  _RED,     "Тофильский L3"),
    LandmarkSpec(4,  "T4",  _RED,     "Тофильский L4"),
    LandmarkSpec(5,  "T5",  _RED,     "Тофильский L5"),
    LandmarkSpec(6,  "T6",  _RED,     "Тофильский L6"),
    LandmarkSpec(7,  "T7",  _RED,     "Тофильский L7 (anchor, выделен)"),
    LandmarkSpec(8,  "T8",  _RED,     "Тофильский L8"),
    LandmarkSpec(9,  "T9",  _MAGENTA, "Тофильский L9"),
    LandmarkSpec(10, "T10", _MAGENTA, "Тофильский L10"),
    LandmarkSpec(11, "T11", _MAGENTA, "Тофильский L11"),
    LandmarkSpec(12, "T12", _MAGENTA, "Тофильский L12"),
    LandmarkSpec(13, "T13", _CYAN,    "Тофильский L13"),
    LandmarkSpec(14, "T14", _CYAN,    "Тофильский L14 (anchor, выделен)"),
    LandmarkSpec(15, "T15", _CYAN,    "Тофильский L15"),
    LandmarkSpec(16, "T16", _CYAN,    "Тофильский L16"),
    LandmarkSpec(17, "T17", _CYAN,    "Тофильский L17"),
    LandmarkSpec(18, "T18", _CYAN,    "Тофильский L18"),
    LandmarkSpec(19, "T19", _CYAN,    "Тофильский L19 (apex выступ)"),
]


PROFILES: Dict[str, Profile] = {
    "Алпатов 8 точек": Profile(
        name="Алпатов 8 точек",
        landmarks=_ALPATOV_12[:8],
        methodology_id="alpatov",
        checkpoint_name="alpatov12.pt",   # 8 — подмножество 12, та же модель
        description="Базовая схема, 8 точек. Достаточно для CI.",
    ),
    "Алпатов 12 точек": Profile(
        name="Алпатов 12 точек",
        landmarks=_ALPATOV_12,
        methodology_id="alpatov",
        checkpoint_name="alpatov12.pt",
        description="Полная схема, 12 точек. Все индексы Алпатова: CI/DsA/RI/PI.",
    ),
    "Тофильский 19 точек": Profile(
        name="Тофильский 19 точек",
        landmarks=_TOFILSKI_19,
        methodology_id="tofilski",
        checkpoint_name="tofilski19.pt",
        description="Геометрическая морфометрия для Procrustes-анализа.",
    ),
}

# Default profile per methodology
DEFAULT_PROFILE_PER_METHODOLOGY: Dict[str, str] = {
    "alpatov": "Алпатов 12 точек",
    "tofilski": "Тофильский 19 точек",
}

DEFAULT_METHODOLOGY = "tofilski"
DEFAULT_PROFILE = DEFAULT_PROFILE_PER_METHODOLOGY[DEFAULT_METHODOLOGY]


def profiles_for_methodology(methodology_id: str) -> List[Profile]:
    """Return all profiles belonging to a methodology, in order."""
    return [p for p in PROFILES.values() if p.methodology_id == methodology_id]


def get_methodology(mid: str) -> Methodology:
    return METHODOLOGIES[mid]


def get_profile(name: str) -> Profile:
    # Backwards compat: old short names still resolve.
    legacy = {"8-point": "Алпатов 8 точек",
              "12-point": "Алпатов 12 точек",
              "19-point": "Тофильский 19 точек"}
    return PROFILES[legacy.get(name, name)]
