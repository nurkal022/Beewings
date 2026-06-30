"""Per-scan export: one folder per scan (scan + crops + TPS + Excel + JSON).

Each cropped, annotated scan is exported into ``<scans folder>/<scan name>/``
containing the scan image, a ``crops/`` folder, and three artifacts scoped to
that scan only: a TPS file (geometric morphometrics), an Excel workbook
(landmark coordinates + classical indices), and a JSON file (the same data plus
provenance metadata). Scans with no annotated wings are skipped.
"""
from __future__ import annotations

import datetime as _dt
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import openpyxl

from ..core.indices import compute_all_alpatov
from ..core.io_dw import export_dw_png
from ..core.io_tps import export_tps
from ..core.profiles import (DEFAULT_PROFILE_PER_METHODOLOGY, METHODOLOGIES,
                             Profile, get_profile)
from ..core.schema import WingAnnotation, annotation_path, load_annotation
from .project import CropProject, ScanEntry

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Classical indices (CI/DsA/RI) are defined only for the Alpatov methodology.
_INDEX_METHODOLOGY = "alpatov"

# IdentiFly/DrawWing .dw.png export is a 19-point format → only the Tofilski set.
_DW_METHODOLOGY = "tofilski"

# Selectable export artifacts (UI checkboxes map onto these keys).
ALL_FORMATS: tuple = ("tps", "xlsx", "json", "dw")


def _export_profiles() -> List[Profile]:
    """The representative profile of each methodology, in display order."""
    return [get_profile(DEFAULT_PROFILE_PER_METHODOLOGY[mid]) for mid in METHODOLOGIES]


def _crop_paths(project: CropProject, entry: ScanEntry) -> List[Path]:
    cdir = project.crops_dir(entry)
    if not cdir.exists():
        return []
    skip = ("_label", "_debug")
    return sorted(
        p for p in cdir.glob("*")
        if p.suffix.lower() in IMAGE_EXTS and not any(t in p.stem for t in skip)
    )


def _items_for(project: CropProject, entry: ScanEntry, prof: Profile) -> List[tuple]:
    """Return [(crop_path, WingAnnotation)] annotated under `prof`'s methodology."""
    cdir = project.crops_dir(entry)
    out = []
    for cp in _crop_paths(project, entry):
        ann = load_annotation(annotation_path(cp, cdir, prof.methodology_id))
        if ann is not None:
            out.append((cp, ann))
    return out


def _wing_row(cp_name: str, ann: WingAnnotation, n_points: int,
              with_indices: bool) -> dict:
    coords = {lm.id: (lm.x, lm.y) for lm in ann.landmarks}
    row = {"wing": cp_name}
    for i in range(1, n_points + 1):
        x, y = coords.get(i, ("", ""))
        row[f"x{i}"] = x
        row[f"y{i}"] = y
    if with_indices:
        for res in compute_all_alpatov(coords):
            row[res.name] = res.value if res.value is not None else ""
    return row


def _write_xlsx(rows: List[dict], n_points: int, path: Path) -> None:
    fields = ["wing"] + [f"{ax}{i}" for i in range(1, n_points + 1) for ax in ("x", "y")]
    extra = list(dict.fromkeys(k for r in rows for k in r if k not in fields))
    fields = fields + extra
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "wings"
    ws.append(fields)
    for r in rows:
        ws.append([r.get(k, "") for k in fields])
    wb.save(path)


def _methodology_json(entry: ScanEntry, items: List[tuple], prof: Profile,
                      with_indices: bool) -> dict:
    index_acc: Dict[str, list] = {}
    uncertain = 0
    wings = []
    for cp, ann in items:
        coords = {lm.id: (lm.x, lm.y) for lm in ann.landmarks}
        uncertain += sum(1 for lm in ann.landmarks if lm.uncertain)
        indices = {}
        if with_indices:
            for res in compute_all_alpatov(coords):
                if res.value is not None:
                    index_acc.setdefault(res.name, []).append(res.value)
                    indices[res.name] = res.value
        wings.append({
            "wing": cp.name,
            "landmarks": [
                {"id": lm.id, "x": lm.x, "y": lm.y, "uncertain": bool(lm.uncertain)}
                for lm in ann.landmarks
            ],
            "indices": indices,
        })
    return {
        "scan": Path(entry.path).stem,
        "source_image": Path(entry.path).name,
        "profile": prof.name,
        "methodology": prof.methodology_id,
        "exported_at": _dt.date.today().isoformat(),
        "n_wings": len(items),
        "uncertain_points": uncertain,
        "index_means": {k: round(sum(v) / len(v), 4) for k, v in index_acc.items() if v},
        "wings": wings,
    }


def export_per_scan(project: CropProject,
                    formats: Optional[Iterable[str]] = None) -> Dict:
    """Export one folder per scan into the scans folder. Returns summary stats.

    For each scan with annotations, writes ``<root>/<stem>/`` holding the scan
    image, a ``crops/`` folder, and — *per methodology that has annotations* —
    suffixed artifacts ``<stem>_<methodology>.{tps,xlsx,json}`` plus, for the
    Tofilski methodology, IdentiFly ``dw/<wing>.dw.png`` files. Keeping the two
    methodologies in separate files makes Alpatov and Tofilski data unambiguous.

    ``formats`` selects which artifacts to write (subset of ``ALL_FORMATS``);
    ``None`` writes them all. The scan image and ``crops/`` are always copied.
    """
    fmts: Set[str] = set(formats) if formats is not None else set(ALL_FORMATS)
    root = Path(project.root)
    profiles = _export_profiles()
    n_scans = 0
    n_wings = 0

    for entry in project.scans:
        per_method = [(p, _items_for(project, entry, p)) for p in profiles]
        per_method = [(p, items) for p, items in per_method if items]
        if not per_method:
            continue  # nothing annotated in any methodology -> no folder
        n_scans += 1
        stem = Path(entry.path).stem
        sdir = root / stem
        crops_out = sdir / "crops"
        crops_out.mkdir(parents=True, exist_ok=True)

        src = Path(entry.path)
        if src.exists():
            shutil.copy2(src, sdir / src.name)
        # Copy every crop once (union across methodologies).
        for cp in _crop_paths(project, entry):
            shutil.copy2(cp, crops_out / cp.name)

        for prof, items in per_method:
            mid = prof.methodology_id
            with_idx = mid == _INDEX_METHODOLOGY
            n_points = len(prof.ids)
            n_wings += len(items)
            if "tps" in fmts:
                export_tps([a for _, a in items], sdir / f"{stem}_{mid}.tps")
            if "dw" in fmts and mid == _DW_METHODOLOGY:
                # One IdentiFly/DrawWing .dw.png per fully-annotated wing.
                dw_dir = sdir / "dw"
                for cp, ann in items:
                    export_dw_png(cp, ann, dw_dir / f"{cp.stem}.dw.png")
            if "xlsx" in fmts:
                rows = [_wing_row(cp.name, ann, n_points, with_idx) for cp, ann in items]
                _write_xlsx(rows, n_points, sdir / f"{stem}_{mid}.xlsx")
            if "json" in fmts:
                (sdir / f"{stem}_{mid}.json").write_text(
                    json.dumps(_methodology_json(entry, items, prof, with_idx),
                               ensure_ascii=False, indent=2),
                    encoding="utf-8")

    return {"n_wings": n_wings, "n_scans": n_scans}
