"""POST /export — build TPS / XLSX / JSON artifacts from supplied landmarks.

Reuses the existing exporters (``export_tps``) and the XLSX/JSON helpers from
``beewings.pipeline.export`` so the on-disk format matches the desktop app.

output=download → returns the file (or a zip for format=all).
output=files    → writes into out_dir (within BEEWINGS_DATA_ROOT) and returns paths.
"""
from __future__ import annotations

import datetime as _dt
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from ...core.indices import compute_all_alpatov
from ...core.io_tps import export_tps
from ...core.profiles import (DEFAULT_PROFILE_PER_METHODOLOGY, METHODOLOGIES,
                              get_profile)
from ...core.schema import Landmark, WingAnnotation
from ...pipeline.export import _wing_row, _write_xlsx
from ..io import resolve_path
from ..schemas import ExportFilesResponse, ExportRequest

router = APIRouter()

_FORMATS = {"tps", "xlsx", "json", "all"}


def _build_annotations(body: ExportRequest, profile_name: str) -> List[Tuple[str, WingAnnotation]]:
    out: List[Tuple[str, WingAnnotation]] = []
    for w in body.wings:
        ann = WingAnnotation(
            image=w.wing,
            image_size=tuple(body.image_size),
            profile=profile_name,
            annotator="api",
            landmarks=[Landmark(id=p.id, x=p.x, y=p.y, uncertain=p.uncertain)
                       for p in w.landmarks],
        )
        out.append((w.wing, ann))
    return out


def _json_payload(items: List[Tuple[str, WingAnnotation]], profile_name: str,
                  methodology: str, with_indices: bool) -> dict:
    index_acc: Dict[str, list] = {}
    uncertain = 0
    wings = []
    for name, ann in items:
        coords = {lm.id: (lm.x, lm.y) for lm in ann.landmarks}
        uncertain += sum(1 for lm in ann.landmarks if lm.uncertain)
        indices = {}
        if with_indices:
            for res in compute_all_alpatov(coords):
                if res.value is not None:
                    index_acc.setdefault(res.name, []).append(res.value)
                    indices[res.name] = res.value
        wings.append({
            "wing": name,
            "landmarks": [{"id": lm.id, "x": lm.x, "y": lm.y,
                           "uncertain": bool(lm.uncertain)} for lm in ann.landmarks],
            "indices": indices,
        })
    return {
        "profile": profile_name,
        "methodology": methodology,
        "exported_at": _dt.date.today().isoformat(),
        "n_wings": len(items),
        "uncertain_points": uncertain,
        "index_means": {k: round(sum(v) / len(v), 4) for k, v in index_acc.items() if v},
        "wings": wings,
    }


def _write_artifacts(items, profile_name, methodology, n_points, with_idx,
                     out_dir: Path, stem: str, fmt: str) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    anns = [a for _, a in items]
    if fmt in ("tps", "all"):
        p = out_dir / f"{stem}_{methodology}.tps"
        export_tps(anns, p)
        written["tps"] = str(p)
    if fmt in ("xlsx", "all"):
        p = out_dir / f"{stem}_{methodology}.xlsx"
        rows = [_wing_row(name, ann, n_points, with_idx) for name, ann in items]
        _write_xlsx(rows, n_points, p)
        written["xlsx"] = str(p)
    if fmt in ("json", "all"):
        p = out_dir / f"{stem}_{methodology}.json"
        p.write_text(json.dumps(_json_payload(items, profile_name, methodology, with_idx),
                                ensure_ascii=False, indent=2), encoding="utf-8")
        written["json"] = str(p)
    return written


@router.post("/export")
def export(body: ExportRequest, format: str = "all"):
    if format not in _FORMATS:
        raise HTTPException(status_code=400,
                            detail=f"format must be one of {sorted(_FORMATS)}")
    if body.methodology not in METHODOLOGIES:
        raise HTTPException(status_code=400,
                            detail=f"unknown methodology '{body.methodology}'")
    if not body.wings:
        raise HTTPException(status_code=400, detail="no wings supplied")

    profile_name = body.profile or DEFAULT_PROFILE_PER_METHODOLOGY[body.methodology]
    n_points = len(get_profile(profile_name).ids)
    with_idx = body.methodology == "alpatov"
    items = _build_annotations(body, profile_name)
    stem = "export"

    if body.output == "files":
        if not body.out_dir:
            raise HTTPException(status_code=400, detail="out_dir is required for output=files")
        written = _write_artifacts(items, profile_name, body.methodology, n_points,
                                   with_idx, resolve_path(body.out_dir), stem, format)
        return ExportFilesResponse(output="files", files=written)

    # output == "download": stage into a temp dir, return file (zip for 'all').
    tmp = Path(tempfile.mkdtemp(prefix="beewings_export_"))
    written = _write_artifacts(items, profile_name, body.methodology, n_points,
                               with_idx, tmp, stem, format)
    cleanup = BackgroundTask(shutil.rmtree, tmp, ignore_errors=True)
    if format == "all":
        zpath = tmp / f"{stem}_{body.methodology}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in written.values():
                zf.write(p, Path(p).name)
        return FileResponse(zpath, filename=zpath.name, media_type="application/zip",
                            background=cleanup)
    only = next(iter(written.values()))
    return FileResponse(only, filename=Path(only).name, background=cleanup)
