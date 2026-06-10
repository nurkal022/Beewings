from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from beewings.core.profiles import get_profile
from beewings.core.schema import Landmark, WingAnnotation, annotation_path, save_annotation
from beewings.pipeline.project import CropProject


def _make_project_with_crops(tmp_path):
    root = tmp_path / "proj"
    scan = tmp_path / "8.jpg"
    cv2.imwrite(str(scan), np.full((50, 50, 3), 255, np.uint8))
    proj = CropProject.create(root, scans_root=str(tmp_path), scan_paths=[scan])
    proj.settings.profile = "Алпатов 12 точек"
    entry = proj.scans[0]
    entry.cropped = True
    cdir = proj.crops_dir(entry)
    cdir.mkdir(parents=True)
    prof = get_profile("Алпатов 12 точек")
    for i in range(2):
        cp = cdir / f"8_crop_{i}.jpg"
        cv2.imwrite(str(cp), np.full((40, 80, 3), 255, np.uint8))
        lms = [Landmark(id=j, x=float(j * 5 + 1), y=float(j * 3 + 1)) for j in range(1, 13)]
        ann = WingAnnotation(image=cp.name, image_size=(80, 40),
                             profile=prof.name, landmarks=lms)
        save_annotation(ann, annotation_path(cp, cdir, prof.methodology_id))
    proj.save()
    return proj


def test_export_all(tmp_path):
    from beewings.pipeline.export import export_protocol
    proj = _make_project_with_crops(tmp_path)
    out = proj.root / "export"
    stats = export_protocol(proj, out,
                            include={"folders", "summary", "tps", "report"})
    assert (out / "summary.csv").exists()
    assert (out / "all_wings.tps").exists()
    assert (out / "report.json").exists()
    assert (out / "data" / "8").exists()

    rows = list(csv.DictReader(open(out / "summary.csv", encoding="utf-8")))
    assert len(rows) == 2
    assert "scan" in rows[0] and "wing" in rows[0]
    assert any(k.startswith("x1") for k in rows[0])

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["n_wings"] == 2
    assert report["n_scans"] == 1
    assert "index_means" in report
    assert "uncertain_points" in report


def test_export_subset_only_summary(tmp_path):
    from beewings.pipeline.export import export_protocol
    proj = _make_project_with_crops(tmp_path)
    out = proj.root / "export2"
    export_protocol(proj, out, include={"summary"})
    assert (out / "summary.csv").exists()
    assert not (out / "all_wings.tps").exists()
    assert not (out / "data").exists()
