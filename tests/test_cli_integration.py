from __future__ import annotations

import cv2

from beewings.segment.cli import process_scan


def test_process_scan_writes_crops_label_and_debug(synthetic_scan, tmp_path):
    img, meta = synthetic_scan
    scan_path = tmp_path / "0042.jpg"
    cv2.imwrite(str(scan_path), img)
    out_dir = tmp_path / "out"

    n = process_scan(
        scan_path, out_dir,
        margin=0.10, min_area_frac=0.0005,
        rotate=False, debug=True, dry_run=False,
    )

    assert n == meta["n_wings"]
    crops = sorted(out_dir.glob("0042_crop_*.jpg"))
    assert len(crops) == meta["n_wings"]
    assert (out_dir / "0042_label.jpg").exists()
    assert (out_dir / "0042_debug.jpg").exists()
    # Each crop is a real, non-empty image.
    for c in crops:
        im = cv2.imread(str(c))
        assert im is not None and im.size > 0


def test_process_scan_dry_run_writes_only_debug(synthetic_scan, tmp_path):
    img, _ = synthetic_scan
    scan_path = tmp_path / "0042.jpg"
    cv2.imwrite(str(scan_path), img)
    out_dir = tmp_path / "out"

    process_scan(scan_path, out_dir, margin=0.10, min_area_frac=0.0005,
                 rotate=False, debug=True, dry_run=True)

    assert list(out_dir.glob("0042_crop_*.jpg")) == []
    assert (out_dir / "0042_debug.jpg").exists()


def test_process_scan_dry_run_without_debug_creates_nothing(synthetic_scan, tmp_path):
    img, _ = synthetic_scan
    scan_path = tmp_path / "0042.jpg"
    cv2.imwrite(str(scan_path), img)
    out_dir = tmp_path / "out"

    process_scan(scan_path, out_dir, margin=0.10, min_area_frac=0.0005,
                 rotate=False, debug=False, dry_run=True)

    assert not out_dir.exists()  # truly read-only: no dir, no files


def test_label_frac_masks_left_fraction(synthetic_scan, tmp_path):
    from beewings.segment.cli import _resolve_label
    from beewings.segment.slide import estimate_background
    img, meta = synthetic_scan
    bg = estimate_background(img)
    box = _resolve_label(img, bg, label_box=None, label_frac=0.25)
    h, w = meta["shape"]
    assert box == (0, 0, int(round(0.25 * w)), h)


def test_label_box_overrides_autodetect_and_clamps(synthetic_scan, tmp_path):
    from beewings.segment.cli import _resolve_label
    from beewings.segment.slide import estimate_background
    img, meta = synthetic_scan
    h, w = meta["shape"]
    bg = estimate_background(img)
    # Box partly outside the image must be clamped to bounds.
    box = _resolve_label(img, bg, label_box=(w - 50, 10, 999, 999), label_frac=None)
    assert box == (w - 50, 10, 50, h - 10)


def test_parse_label_box_valid_and_invalid():
    from beewings.segment.cli import _parse_label_box
    import argparse
    import pytest as _pytest
    assert _parse_label_box("10, 20, 30, 40") == (10, 20, 30, 40)
    with _pytest.raises(argparse.ArgumentTypeError):
        _parse_label_box("1,2,3")
    with _pytest.raises(argparse.ArgumentTypeError):
        _parse_label_box("1,2,3,0")
