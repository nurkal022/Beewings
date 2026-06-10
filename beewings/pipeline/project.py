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
        return cls(
            root=root,
            scans_root=data["scans_root"],
            settings=Settings(**data.get("settings", {})),
            stage=data.get("stage", "select"),
            scans=scans,
        )

    def crops_dir(self, entry: ScanEntry) -> Path:
        return self.root / "crops" / Path(entry.path).stem
