"""Pydantic schema for wing annotations.

One JSON file per image lives at <image_dir>/annotations/<image_stem>.json.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class Landmark(BaseModel):
    id: int
    x: float
    y: float
    uncertain: bool = False
    skipped: bool = False  # True means: annotator deliberately left this point blank


class WingAnnotation(BaseModel):
    image: str                          # filename relative to project root
    image_size: Tuple[int, int]         # (width, height) in pixels
    profile: str                        # profile name (e.g. "19-point")
    landmarks: List[Landmark] = Field(default_factory=list)
    annotator: str = ""
    notes: str = ""
    flipped: bool = False               # True if image was horizontally mirrored when annotated
    created: datetime = Field(default_factory=datetime.now)
    modified: datetime = Field(default_factory=datetime.now)
    version: int = 1

    def get_landmark(self, lm_id: int) -> Optional[Landmark]:
        for lm in self.landmarks:
            if lm.id == lm_id:
                return lm
        return None

    def set_landmark(self, lm_id: int, x: float, y: float) -> None:
        existing = self.get_landmark(lm_id)
        if existing is not None:
            existing.x = x
            existing.y = y
            existing.skipped = False
        else:
            self.landmarks.append(Landmark(id=lm_id, x=x, y=y))
        self.landmarks.sort(key=lambda lm: lm.id)
        self.modified = datetime.now()
        self.version += 1

    def remove_landmark(self, lm_id: int) -> None:
        self.landmarks = [lm for lm in self.landmarks if lm.id != lm_id]
        self.modified = datetime.now()
        self.version += 1

    def toggle_uncertain(self, lm_id: int) -> None:
        lm = self.get_landmark(lm_id)
        if lm is not None:
            lm.uncertain = not lm.uncertain
            self.modified = datetime.now()
            self.version += 1

    def mark_skipped(self, lm_id: int, skipped: bool = True) -> None:
        lm = self.get_landmark(lm_id)
        if lm is None and skipped:
            self.landmarks.append(Landmark(id=lm_id, x=-1, y=-1, skipped=True))
            self.landmarks.sort(key=lambda l: l.id)
        elif lm is not None:
            lm.skipped = skipped
        self.modified = datetime.now()
        self.version += 1

    def progress(self, expected_ids: List[int]) -> Tuple[int, int]:
        """Return (done, total). A skipped point counts as done."""
        done_ids = {lm.id for lm in self.landmarks if lm.skipped or (lm.x >= 0 and lm.y >= 0)}
        return sum(1 for i in expected_ids if i in done_ids), len(expected_ids)


def annotation_path(image_path: Path, project_dir: Path,
                    methodology_id: Optional[str] = None) -> Path:
    """Path of the annotation JSON for an image.

    With a methodology_id, files live under per-methodology subfolders so
    Alpatov and Tofilski annotations of the SAME wing can coexist without
    overwriting each other:

        <project_dir>/annotations/alpatov/<stem>.json
        <project_dir>/annotations/tofilski/<stem>.json

    When called without methodology_id (legacy path), files go to:
        <project_dir>/annotations/<stem>.json
    Used only by migration tooling and old tests.
    """
    if methodology_id:
        ann_dir = project_dir / "annotations" / methodology_id
    else:
        ann_dir = project_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    return ann_dir / (image_path.stem + ".json")


def migrate_legacy_annotations(project_dir: Path) -> Dict[str, int]:
    """Move legacy single-folder annotations into per-methodology subfolders.

    Inspects each <project_dir>/annotations/*.json. Uses its `profile` field
    to decide which methodology it belongs to, then moves it to the right
    subfolder. Returns a small report dict.
    """
    from .profiles import PROFILES, get_profile

    ann_root = project_dir / "annotations"
    if not ann_root.exists():
        return {"moved": 0, "skipped": 0}

    moved = 0
    skipped = 0
    for f in ann_root.iterdir():
        if not f.is_file() or f.suffix != ".json":
            continue
        try:
            ann = WingAnnotation.model_validate_json(f.read_text(encoding="utf-8"))
            profile = get_profile(ann.profile)
            mid = profile.methodology_id
        except Exception:
            skipped += 1
            continue
        dst = ann_root / mid / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            skipped += 1
            continue
        f.rename(dst)
        moved += 1
    return {"moved": moved, "skipped": skipped}


def load_annotation(path: Path) -> Optional[WingAnnotation]:
    if not path.exists():
        return None
    try:
        return WingAnnotation.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        # Corrupt / partially-written / hand-edited JSON: treat as "no annotation"
        # rather than crashing the list rendering or the annotator.
        return None


def save_annotation(ann: WingAnnotation, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ann.model_dump_json(indent=2), encoding="utf-8")
