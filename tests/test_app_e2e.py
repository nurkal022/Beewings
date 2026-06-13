from __future__ import annotations

from pathlib import Path

from beewings.pipeline.project import CropProject, auto_detect, recrop
from beewings.pipeline.export import export_per_scan
from tests._annotate import annotate_first_crop


def test_open_folder_crop_export(scan_file, tmp_path):
    scan_path, meta = scan_file
    folder = tmp_path / "project"
    folder.mkdir()
    dest = folder / scan_path.name
    dest.write_bytes(scan_path.read_bytes())

    proj = CropProject.open_folder(folder)
    assert len(proj.scans) == 1
    auto_detect(proj.scans[0], proj.settings)
    n = recrop(proj, proj.scans[0]); proj.save()
    assert n == meta["n_wings"]
    assert proj.progress()["n_cropped"] == 1

    annotate_first_crop(proj, proj.scans[0])
    stats = export_per_scan(proj)
    assert stats["n_scans"] == 1
    assert (proj.root / scan_path.stem / f"{scan_path.stem}_alpatov.json").exists()
