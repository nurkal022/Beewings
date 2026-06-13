from __future__ import annotations

from pathlib import Path

import cv2

from beewings.pipeline.project import CropProject, auto_detect, recrop
from beewings.pipeline.export import export_per_scan
from tests._annotate import annotate_first_crop


def test_pipeline_crop_then_export(scan_file, tmp_path):
    """Scans -> auto-detect -> recrop -> annotate -> per-scan export, headless."""
    scan_path, meta = scan_file
    proj = CropProject.create(tmp_path / "proj", scans_root=str(scan_path.parent),
                              scan_paths=[scan_path])
    auto_detect(proj.scans[0], proj.settings)
    n = recrop(proj, proj.scans[0])
    proj.save()
    assert n == meta["n_wings"]

    annotate_first_crop(proj, proj.scans[0])
    stats = export_per_scan(proj)
    assert stats["n_scans"] == 1
    sdir = proj.root / scan_path.stem
    assert (sdir / f"{scan_path.stem}_alpatov.json").exists()
    assert (sdir / "crops").is_dir()
