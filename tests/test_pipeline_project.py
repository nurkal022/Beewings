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


def test_auto_detect_fills_boxes(scan_file, tmp_path):
    from beewings.pipeline.project import CropProject, auto_detect
    scan_path, meta = scan_file
    proj = CropProject.create(tmp_path / "proj", scans_root=str(scan_path.parent),
                              scan_paths=[scan_path])
    auto_detect(proj.scans[0], proj.settings)
    assert len(proj.scans[0].wing_boxes) == meta["n_wings"]
    assert proj.scans[0].label_box is not None


def test_recrop_writes_crops(scan_file, tmp_path):
    from beewings.pipeline.project import CropProject, auto_detect, recrop
    scan_path, meta = scan_file
    proj = CropProject.create(tmp_path / "proj", scans_root=str(scan_path.parent),
                              scan_paths=[scan_path])
    auto_detect(proj.scans[0], proj.settings)
    n = recrop(proj, proj.scans[0])
    assert n == meta["n_wings"]
    crop_dir = proj.crops_dir(proj.scans[0])
    crops = list(crop_dir.glob(f"{scan_path.stem}_crop_*.jpg"))
    assert len(crops) == meta["n_wings"]
    assert (crop_dir / f"{scan_path.stem}_label.jpg").exists()
    assert proj.scans[0].cropped is True


def test_recrop_respects_manual_boxes(scan_file, tmp_path):
    from beewings.pipeline.project import CropProject, recrop
    scan_path, _ = scan_file
    proj = CropProject.create(tmp_path / "proj", scans_root=str(scan_path.parent),
                              scan_paths=[scan_path])
    proj.scans[0].wing_boxes = [(360, 90, 110, 56), (520, 90, 110, 56)]
    proj.settings.rotate = False
    n = recrop(proj, proj.scans[0])
    assert n == 2


def test_open_folder_creates_from_images(tmp_path):
    import cv2, numpy as np
    from beewings.pipeline.project import CropProject
    folder = tmp_path / "Клат"
    folder.mkdir()
    for n in ("a.jpg", "b.jpg"):
        cv2.imwrite(str(folder / n), np.full((10, 10, 3), 255, np.uint8))
    (folder / "notes.txt").write_text("x")
    proj = CropProject.open_folder(folder)
    assert sorted(Path(s.path).name for s in proj.scans) == ["a.jpg", "b.jpg"]
    assert (folder / "project.json").exists()


def test_open_folder_loads_existing(tmp_path):
    import cv2, numpy as np
    from beewings.pipeline.project import CropProject
    folder = tmp_path / "Клат"
    folder.mkdir()
    cv2.imwrite(str(folder / "a.jpg"), np.full((10, 10, 3), 255, np.uint8))
    p1 = CropProject.open_folder(folder)
    p1.settings.delta = 99
    p1.save()
    p2 = CropProject.open_folder(folder)
    assert p2.settings.delta == 99


def test_progress_counts(tmp_path):
    import cv2, numpy as np
    from beewings.pipeline.project import CropProject
    folder = tmp_path / "p"
    folder.mkdir()
    for n in ("a.jpg", "b.jpg"):
        cv2.imwrite(str(folder / n), np.full((10, 10, 3), 255, np.uint8))
    proj = CropProject.open_folder(folder)
    proj.scans[0].cropped = True
    prog = proj.progress()
    assert prog == {"n_splits": 2, "n_cropped": 1, "n_landmarked": 0}


def test_progress_landmarked_requires_real_landmarks(tmp_path):
    import cv2, numpy as np
    from beewings.pipeline.project import CropProject
    from beewings.core.profiles import get_profile
    from beewings.core.schema import (Landmark, WingAnnotation,
                                       annotation_path, save_annotation)
    folder = tmp_path / "p"
    folder.mkdir()
    cv2.imwrite(str(folder / "a.jpg"), np.full((10, 10, 3), 255, np.uint8))
    proj = CropProject.open_folder(folder)
    entry = proj.scans[0]
    entry.cropped = True
    cdir = proj.crops_dir(entry)
    cdir.mkdir(parents=True, exist_ok=True)
    crop = cdir / "a_crop_0.jpg"
    cv2.imwrite(str(crop), np.full((10, 10, 3), 255, np.uint8))
    prof = get_profile("Алпатов 12 точек")
    # empty annotation -> NOT landmarked
    empty = WingAnnotation(image=crop.name, image_size=(10, 10),
                           profile=prof.name, landmarks=[])
    save_annotation(empty, annotation_path(crop, cdir, prof.methodology_id))
    assert proj.progress()["n_landmarked"] == 0
    # with a real landmark -> landmarked
    filled = WingAnnotation(image=crop.name, image_size=(10, 10), profile=prof.name,
                            landmarks=[Landmark(id=1, x=1.0, y=1.0)])
    save_annotation(filled, annotation_path(crop, cdir, prof.methodology_id))
    assert proj.progress()["n_landmarked"] == 1
