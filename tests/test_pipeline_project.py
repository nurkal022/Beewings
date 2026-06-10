from __future__ import annotations

from pathlib import Path

from beewings.pipeline.project import CropProject, ScanEntry, Settings


def test_create_and_roundtrip(tmp_path):
    root = tmp_path / "proj"
    proj = CropProject.create(root, scans_root="/scans",
                              scan_paths=[Path("/scans/a.jpg"), Path("/scans/b.jpg")])
    proj.save()
    assert (root / "project.json").exists()

    loaded = CropProject.load(root)
    assert loaded.scans_root == "/scans"
    assert [s.path for s in loaded.scans] == ["/scans/a.jpg", "/scans/b.jpg"]
    assert loaded.settings.margin == 0.10
    assert loaded.settings.delta == 45
    assert loaded.stage == "select"


def test_settings_roundtrip_and_edits(tmp_path):
    root = tmp_path / "proj"
    proj = CropProject.create(root, scans_root="/s", scan_paths=[Path("/s/a.jpg")])
    proj.settings.profile = "Тофильский 19 точек"
    proj.scans[0].wing_boxes = [(1, 2, 3, 4), (5, 6, 7, 8)]
    proj.scans[0].label_box = (0, 0, 10, 10)
    proj.save()
    loaded = CropProject.load(root)
    assert loaded.settings.profile == "Тофильский 19 точек"
    assert loaded.scans[0].wing_boxes == [(1, 2, 3, 4), (5, 6, 7, 8)]
    assert loaded.scans[0].label_box == (0, 0, 10, 10)


def test_crops_dir(tmp_path):
    root = tmp_path / "proj"
    proj = CropProject.create(root, scans_root="/s", scan_paths=[Path("/s/04401.jpg")])
    assert proj.crops_dir(proj.scans[0]) == root / "crops" / "04401"
