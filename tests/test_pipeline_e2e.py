from __future__ import annotations

from pathlib import Path

import cv2

from beewings.pipeline.project import CropProject, auto_detect, recrop
from beewings.pipeline.export import export_protocol


def test_pipeline_crop_then_export(scan_file, tmp_path):
    """Scans -> auto-detect -> recrop -> export (no ML), fully headless."""
    scan_path, meta = scan_file
    proj = CropProject.create(tmp_path / "proj", scans_root=str(scan_path.parent),
                              scan_paths=[scan_path])
    auto_detect(proj.scans[0], proj.settings)
    n = recrop(proj, proj.scans[0])
    proj.save()
    assert n == meta["n_wings"]

    out = proj.root / "export"
    stats = export_protocol(proj, out, include={"folders", "report"})
    assert stats["n_scans"] == 1
    assert (out / "report.json").exists()
    assert (out / "data" / scan_path.stem).exists()
