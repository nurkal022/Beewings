from __future__ import annotations

from beewings.segment.slide import estimate_background, detect_label_region
from beewings.segment.wings import wing_mask, find_wings


def test_find_wings_counts_grid(synthetic_scan):
    img, meta = synthetic_scan
    bg = estimate_background(img)
    label = detect_label_region(img, bg)
    mask = wing_mask(img, bg, exclude=label)
    boxes = find_wings(mask, min_area_frac=0.0005)
    assert len(boxes) == meta["n_wings"]


def test_find_wings_excludes_label(synthetic_scan):
    img, meta = synthetic_scan
    bg = estimate_background(img)
    label = detect_label_region(img, bg)
    mask = wing_mask(img, bg, exclude=label)
    boxes = find_wings(mask, min_area_frac=0.0005)
    lx, ly, lw, lh = meta["label_box"]
    for x, y, w, h in boxes:
        assert x >= lx + lw - 5  # every wing is right of the label block


def test_min_area_filters_specks(synthetic_scan):
    import cv2
    img, meta = synthetic_scan
    cv2.circle(img, (900, 550), 3, (50, 50, 50), -1)  # tiny speck
    bg = estimate_background(img)
    label = detect_label_region(img, bg)
    mask = wing_mask(img, bg, exclude=label)
    boxes = find_wings(mask, min_area_frac=0.0005)
    assert len(boxes) == meta["n_wings"]  # speck rejected
