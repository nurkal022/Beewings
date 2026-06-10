"""Structured protocol export: data folders, summary CSV, TPS, report JSON."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set

from ..core.indices import compute_all_alpatov
from ..core.io_tps import export_tps
from ..core.profiles import get_profile
from ..core.schema import WingAnnotation, annotation_path, load_annotation
from .project import CropProject, ScanEntry

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _crop_paths(project: CropProject, entry: ScanEntry) -> List[Path]:
    cdir = project.crops_dir(entry)
    if not cdir.exists():
        return []
    skip = ("_label", "_debug")
    return sorted(
        p for p in cdir.glob("*")
        if p.suffix.lower() in IMAGE_EXTS and not any(t in p.stem for t in skip)
    )


def _collect(project: CropProject) -> List[tuple]:
    """Return list of (scan_stem, crop_path, WingAnnotation) for all cropped scans."""
    prof = get_profile(project.settings.profile)
    out = []
    for entry in project.scans:
        cdir = project.crops_dir(entry)
        for cp in _crop_paths(project, entry):
            ann = load_annotation(annotation_path(cp, cdir, prof.methodology_id))
            if ann is not None:
                out.append((Path(entry.path).stem, cp, ann))
    return out


def _summary_rows(items, n_points: int) -> List[dict]:
    rows = []
    for scan_stem, cp, ann in items:
        coords = {lm.id: (lm.x, lm.y) for lm in ann.landmarks}
        row = {"scan": scan_stem, "wing": cp.name}
        for i in range(1, n_points + 1):
            x, y = coords.get(i, ("", ""))
            row[f"x{i}"] = x
            row[f"y{i}"] = y
        for res in compute_all_alpatov(coords):  # returns List[IndexResult]
            row[res.name] = res.value if res.value is not None else ""
        rows.append(row)
    return rows


def export_protocol(project: CropProject, out_dir: Path,
                    include: Set[str]) -> Dict:
    """Write the selected protocol artifacts. Returns summary stats."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prof = get_profile(project.settings.profile)
    n_points = len(prof.ids)
    items = _collect(project)
    annotations: List[WingAnnotation] = [a for _, _, a in items]

    if "folders" in include:
        data_dir = out_dir / "data"
        for entry in project.scans:
            cdir = project.crops_dir(entry)
            if cdir.exists():
                shutil.copytree(cdir, data_dir / cdir.name, dirs_exist_ok=True)

    if "summary" in include:
        rows = _summary_rows(items, n_points)
        fields = (["scan", "wing"]
                  + [f"{ax}{i}" for i in range(1, n_points + 1) for ax in ("x", "y")])
        extra = [k for r in rows for k in r if k not in fields]
        seen = list(dict.fromkeys(extra))
        fields = fields + seen
        with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    if "tps" in include:
        export_tps(annotations, out_dir / "all_wings.tps")

    if "report" in include:
        report = {
            "profile": prof.name,
            "n_scans": sum(1 for e in project.scans if e.cropped),
            "n_wings": len(items),
        }
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"n_wings": len(items), "n_scans": sum(1 for e in project.scans if e.cropped)}
