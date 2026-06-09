from __future__ import annotations

from beewings.segment.slide import estimate_background, detect_label_region


def test_estimate_background_is_bright(synthetic_scan):
    img, _ = synthetic_scan
    bg = estimate_background(img)
    assert bg > 230  # white slide


def test_detect_label_region_left_block(synthetic_scan):
    img, meta = synthetic_scan
    bg = estimate_background(img)
    box = detect_label_region(img, bg)
    assert box is not None
    x, y, w, h = box
    lx, ly, lw, lh = meta["label_box"]
    # Detected box must cover the label strokes and stay on the left side.
    assert x <= lx + 20
    assert x + w <= meta["shape"][1] // 2     # never crosses into the grid
    assert x + w >= lx + lw - 20              # reaches the rightmost stroke


def test_detect_label_region_none_when_absent(synthetic_scan):
    img, _ = synthetic_scan
    bg = estimate_background(img)
    # Paint the whole left third white -> no label content.
    img[:, : img.shape[1] // 3] = 255
    assert detect_label_region(img, bg) is None
