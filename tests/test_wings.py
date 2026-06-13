from __future__ import annotations

from beewings.segment.slide import estimate_background, detect_label_region
from beewings.segment.wings import (
    wing_mask, find_wings, filter_wings, drop_ink_blobs,
)
import cv2
import numpy as np


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


def _wing_box(cx, cy, hw=55, hh=28):
    return (cx - hw, cy - hh, 2 * hw, 2 * hh)


def test_filter_wings_drops_small_letters():
    # 20 wing-sized boxes (consistent, elongated) + 4 small square "letters".
    wings = [_wing_box(360 + c * 160, 90 + r * 130)
             for r in range(4) for c in range(5)]
    letters = [(30, 80 + i * 60, 35, 40) for i in range(4)]  # small, ~square
    kept = filter_wings(wings + letters)
    assert sorted(kept) == sorted(wings)  # letters removed, wings intact


def test_filter_wings_drops_merged_blob():
    wings = [_wing_box(360 + c * 160, 90 + r * 130)
             for r in range(4) for c in range(5)]
    merged = (200, 200, 600, 300)  # huge merged/label leftover
    kept = filter_wings(wings + [merged])
    assert merged not in kept
    assert len(kept) == len(wings)


def test_drop_ink_blobs_removes_solid_ink_keeps_translucent():
    # bg=255. A solid-ink letter (uniformly very dark) vs a wing (faint
    # membrane with a couple of thin dark veins).
    img = np.full((200, 400, 3), 255, np.uint8)
    # letter: solid dark block
    img[40:140, 40:120] = 20
    # wing: faint membrane (slightly darker than bg) + two thin dark veins
    img[40:140, 240:360] = 215          # translucent membrane
    img[88:92, 240:360] = 20            # vein
    img[40:140, 295:299] = 20           # vein
    letter = (40, 40, 80, 100)
    wing = (240, 40, 120, 100)
    kept = drop_ink_blobs(img, 255.0, [letter, wing])
    assert wing in kept
    assert letter not in kept


def test_filter_wings_noop_when_too_few():
    # With too few boxes the median is unreliable; keep everything untouched.
    boxes = [_wing_box(360, 90), (30, 80, 35, 40)]
    assert filter_wings(boxes) == boxes
