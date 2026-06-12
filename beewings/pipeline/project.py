"""Pipeline project: a working directory + project.json manifest."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

Box = Tuple[int, int, int, int]


@dataclass
class Settings:
    margin: float = 0.10
    delta: int = 45
    label_frac: Optional[float] = None
    rotate: bool = True
    profile: str = "Алпатов 12 точек"
    checkpoint: str = "checkpoints/alpatov12.pt"


@dataclass
class ScanEntry:
    path: str
    label_box: Optional[Box] = None
    wing_boxes: List[Box] = field(default_factory=list)
    cropped: bool = False


@dataclass
class CropProject:
    root: Path
    scans_root: str
    settings: Settings = field(default_factory=Settings)
    stage: str = "select"
    scans: List[ScanEntry] = field(default_factory=list)

    @classmethod
    def create(cls, root: Path, scans_root: str, scan_paths: List[Path]) -> "CropProject":
        root = Path(root)
        scans = [ScanEntry(path=str(p)) for p in scan_paths]
        return cls(root=root, scans_root=str(scans_root), scans=scans)

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "scans_root": self.scans_root,
            "settings": asdict(self.settings),
            "stage": self.stage,
            "scans": [
                {
                    "path": s.path,
                    "label_box": list(s.label_box) if s.label_box else None,
                    "wing_boxes": [list(b) for b in s.wing_boxes],
                    "cropped": s.cropped,
                }
                for s in self.scans
            ],
        }
        tmp = self.root / "project.json.tmp"
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.root / "project.json")

    @classmethod
    def load(cls, root: Path) -> "CropProject":
        root = Path(root)
        data = json.loads((root / "project.json").read_text(encoding="utf-8"))

        def _box(v):
            return tuple(int(i) for i in v) if v else None

        scans = [
            ScanEntry(
                path=s["path"],
                label_box=_box(s.get("label_box")),
                wing_boxes=[tuple(int(i) for i in b) for b in s.get("wing_boxes", [])],
                cropped=bool(s.get("cropped", False)),
            )
            for s in data.get("scans", [])
        ]
        proj = cls(
            root=root,
            scans_root=data["scans_root"],
            settings=Settings(**data.get("settings", {})),
            stage=data.get("stage", "select"),
            scans=scans,
        )
        proj._heal_paths()
        return proj

    def _heal_paths(self) -> None:
        """Re-anchor scan image paths to this project folder.

        project.json stores absolute paths, so a project created on one machine
        breaks when copied to another (e.g. macOS -> Windows). When a stored
        path no longer exists, fall back to the same filename inside the project
        folder (or scans_root), which is where open_folder() found it.
        """
        search_dirs = [self.root, Path(self.scans_root)]
        for s in self.scans:
            if os.path.exists(s.path):
                continue
            name = Path(s.path).name
            for d in search_dirs:
                candidate = d / name
                if candidate.exists():
                    s.path = str(candidate)
                    break

    def crops_dir(self, entry: ScanEntry) -> Path:
        return self.root / "crops" / Path(entry.path).stem

    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    _SKIP = ("_crop_", "_label", "_debug")

    @classmethod
    def _scan_files(cls, folder: Path) -> List[Path]:
        folder = Path(folder)
        return sorted(
            p for p in folder.glob("*")
            if p.suffix.lower() in cls.IMAGE_EXTS and not any(t in p.stem for t in cls._SKIP)
        )

    @classmethod
    def open_folder(cls, folder: Path) -> "CropProject":
        """Load <folder>/project.json if present, else create from top-level images."""
        folder = Path(folder)
        if (folder / "project.json").exists():
            proj = cls.load(folder)
            proj.sync_new_scans()  # pick up images added since the project was made
            return proj
        proj = cls.create(folder, scans_root=str(folder),
                          scan_paths=cls._scan_files(folder))
        proj.save()
        return proj

    def sync_new_scans(self) -> int:
        """Append image files present in the project folder but not yet tracked.

        Lets users drop new scans into the folder and have them appear without
        rebuilding the project. Existing entries (and their boxes/annotations)
        are never touched or reordered. Returns the number of scans added.
        """
        have = {Path(s.path).name for s in self.scans}
        dirs, seen = [], set()
        for d in (self.root, Path(self.scans_root)):
            d = Path(d)
            if d not in seen and d.exists():
                dirs.append(d)
                seen.add(d)
        added = 0
        for d in dirs:
            for p in self._scan_files(d):
                if p.name in have:
                    continue
                self.scans.append(ScanEntry(path=str(p)))
                have.add(p.name)
                added += 1
        if added:
            self.save()
        return added

    def progress(self) -> dict:
        """Summary counts for the Home cards. A scan counts as landmarked only
        when at least one of its crops has an annotation with >=1 landmark."""
        from ..core.schema import load_annotation
        n_landmarked = 0
        for entry in self.scans:
            cdir = self.crops_dir(entry)
            ann_dir = cdir / "annotations"
            if not (entry.cropped and ann_dir.exists()):
                continue
            for jf in ann_dir.rglob("*.json"):
                a = load_annotation(jf)
                if a is not None and len(a.landmarks) > 0:
                    n_landmarked += 1
                    break
        return {
            "n_splits": len(self.scans),
            "n_cropped": sum(1 for e in self.scans if e.cropped),
            "n_landmarked": n_landmarked,
        }


from ..core.cvio import imread, imwrite
from ..segment.crop import crop_box, crop_wing
from ..segment.debug import render_overlay
from ..segment.layout import reading_order
from ..segment.slide import detect_label_region, estimate_background
from ..segment.wings import find_wings, wing_mask


def auto_detect(entry: ScanEntry, settings: Settings) -> None:
    """Fill entry.label_box and entry.wing_boxes by running segmentation."""
    img = imread(entry.path)
    if img is None:
        raise ValueError(f"cannot read image: {entry.path}")
    bg = estimate_background(img)
    if settings.label_frac is not None:
        h, w = img.shape[:2]
        label = (0, 0, int(round(settings.label_frac * w)), h)
    else:
        label = detect_label_region(img, bg)
    mask = wing_mask(img, bg, delta=settings.delta, exclude=label)
    boxes = find_wings(mask)
    order = reading_order(boxes)
    entry.label_box = label
    entry.wing_boxes = [boxes[i] for i in order]


def recrop(project: "CropProject", entry: ScanEntry) -> int:
    """Write per-wing crops (+ label + debug) for one scan from its boxes.

    Returns the number of wing crops written. Crops follow the manifest boxes
    exactly (manual edits respected), not a fresh auto-detection.
    """
    img = imread(entry.path)
    if img is None:
        raise ValueError(f"cannot read image: {entry.path}")
    bg = estimate_background(img)
    stem = Path(entry.path).stem
    out_dir = project.crops_dir(entry)
    out_dir.mkdir(parents=True, exist_ok=True)

    for n, box in enumerate(entry.wing_boxes):
        patch = crop_wing(img, box, margin=project.settings.margin,
                          rotate=project.settings.rotate, bg=bg)
        imwrite(out_dir / f"{stem}_crop_{n}.jpg", patch)
    if entry.label_box is not None:
        imwrite(out_dir / f"{stem}_label.jpg",
                crop_box(img, entry.label_box, 0.02))
    order = list(range(len(entry.wing_boxes)))
    overlay = render_overlay(img, entry.wing_boxes, order, label_box=entry.label_box)
    imwrite(out_dir / f"{stem}_debug.jpg", overlay)

    entry.cropped = True
    return len(entry.wing_boxes)
