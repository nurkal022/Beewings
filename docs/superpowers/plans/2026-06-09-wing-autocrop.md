# Wing Auto-Crop (`beewings-crop`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `beewings-crop` CLI that splits one big scan (a glass slide with 20–50 bee wings + a handwritten label) into individual per-wing images ready for `beewings-ml-batch`.

**Architecture:** Classical OpenCV pipeline with a grid layout prior. The scan is read once; detection runs on a downscaled working copy and all coordinates are scaled back to crop from the full-resolution original. New self-contained subpackage `beewings/segment/` with one responsibility per module: label/background (`slide.py`), wing mask + bounding boxes (`wings.py`), reading-order numbering (`layout.py`), cropping + optional canonical rotation (`crop.py`), debug overlay (`debug.py`), and CLI orchestration (`cli.py`).

**Tech Stack:** Python 3.10+, numpy, opencv-python (cv2), pytest (dev). All already in the project except pytest.

---

## File Structure

| File | Responsibility |
|---|---|
| `beewings/segment/__init__.py` | Package marker + public re-exports |
| `beewings/segment/slide.py` | `estimate_background`, `detect_label_region` |
| `beewings/segment/wings.py` | `wing_mask`, `find_wings` |
| `beewings/segment/layout.py` | `reading_order` |
| `beewings/segment/crop.py` | `crop_box`, `crop_wing` (+ optional rotation) |
| `beewings/segment/debug.py` | `render_overlay` |
| `beewings/segment/cli.py` | `main()` argparse entry point, per-scan orchestration |
| `tests/conftest.py` | `synthetic_scan` fixture builder |
| `tests/test_slide.py` | background + label detection tests |
| `tests/test_wings.py` | mask + bbox detection tests |
| `tests/test_layout.py` | reading-order tests |
| `tests/test_crop.py` | crop + rotation tests |
| `tests/test_cli_integration.py` | end-to-end on a synthetic scan |
| `pyproject.toml` | add `pytest` dev dep, register `beewings-crop` entry point |

### Shared types / conventions (used across tasks)

- A **box** is a tuple `(x, y, w, h)` of ints in pixel coordinates of the image it was found in.
- Images are BGR `np.ndarray` (cv2 convention), dtype `uint8`.
- "Background" is a single scalar grayscale value `float` (slides are bright/uniform).
- Foreground masks are `uint8` arrays with values `0` / `255`.

---

## Task 0: Project scaffolding

**Files:**
- Modify: `pyproject.toml`
- Create: `beewings/segment/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Add pytest dev dependency and the new entry point**

In `pyproject.toml`, add an optional dev group and register the script. After the existing `[project]` `dependencies` list, add:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0"]
```

In `[project.scripts]`, add this line after `beewings-ml-report = ...`:

```toml
beewings-crop = "beewings.segment.cli:main"
```

- [ ] **Step 2: Install pytest into the venv**

Run: `.venv/bin/pip install pytest`
Expected: `Successfully installed pytest-...`

- [ ] **Step 3: Create the package marker**

Create `beewings/segment/__init__.py`:

```python
"""Automatic cropping of individual wings from a full scan slide."""
from __future__ import annotations
```

- [ ] **Step 4: Create the synthetic-scan fixture**

Create `tests/conftest.py`. This builds a deterministic fake scan: white background, a dark "handwriting" block on the left, and a regular grid of dark filled ellipses ("wings"). No file IO — returns the array plus ground-truth metadata.

```python
from __future__ import annotations

import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_scan():
    """Return (img_bgr, meta) where meta has known label box and wing centers.

    Layout: white 255 canvas. Left 'label' = dense dark scribble strokes.
    Then a vertical empty gap. Then a 5x4 grid of dark ellipses (wings).
    """
    h, w = 600, 1200
    img = np.full((h, w, 3), 255, np.uint8)

    # Label block on the left: several dark strokes inside x in [20, 180].
    label_box = (20, 40, 160, 300)  # x, y, w, h
    rng_lines = [
        ((30, 80), (170, 90)),
        ((30, 140), (150, 150)),
        ((40, 200), (175, 215)),
        ((35, 300), (160, 320)),
    ]
    for (x1, y1), (x2, y2) in rng_lines:
        cv2.line(img, (x1, y1), (x2, y2), (30, 30, 30), 6)

    # Wing grid: 5 columns x 4 rows of ellipses, starting well right of the gap.
    centers = []
    x0, y0, dx, dy = 360, 90, 160, 130
    for r in range(4):
        for c in range(5):
            cx, cy = x0 + c * dx, y0 + r * dy
            cv2.ellipse(img, (cx, cy), (55, 28), 0, 0, 360, (120, 120, 120), -1)
            centers.append((cx, cy))

    meta = {
        "label_box": label_box,
        "centers": centers,       # row-major order
        "n_wings": len(centers),
        "shape": (h, w),
    }
    return img, meta
```

- [ ] **Step 5: Verify the fixture imports and pytest runs**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `no tests ran` (collection succeeds, zero tests yet — exit code 5 is fine).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml beewings/segment/__init__.py tests/conftest.py
git commit -m "chore: scaffold beewings-crop package and test fixture"
```

---

## Task 1: Background + label detection (`slide.py`)

**Files:**
- Create: `beewings/segment/slide.py`
- Test: `tests/test_slide.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_slide.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_slide.py -q`
Expected: FAIL — `ModuleNotFoundError` / `cannot import name 'estimate_background'`.

- [ ] **Step 3: Implement `slide.py`**

Create `beewings/segment/slide.py`:

```python
"""Background estimation and handwritten-label detection for scan slides.

The label is always on the left of the slide, separated from the wing grid by
an empty vertical gap. We detect dark foreground content, then find the gap and
treat everything left of it as the label.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]


def estimate_background(img: np.ndarray) -> float:
    """Estimate the bright slide background as the median of the four corners."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    k = max(8, min(h, w) // 20)
    corners = np.concatenate([
        gray[:k, :k].ravel(),
        gray[:k, -k:].ravel(),
        gray[-k:, :k].ravel(),
        gray[-k:, -k:].ravel(),
    ])
    return float(np.median(corners))


def _foreground(gray: np.ndarray, bg: float, delta: int = 35) -> np.ndarray:
    """Binary mask of content darker than the background by `delta`."""
    mask = (gray < (bg - delta)).astype(np.uint8) * 255
    return mask


def detect_label_region(
    img: np.ndarray,
    bg: float,
    search_frac: float = 0.45,
    gap_frac: float = 0.04,
) -> Optional[Box]:
    """Return (x, y, w, h) of the left label block, or None if absent.

    Looks for dark content in the left `search_frac` of the image, then finds
    the first sustained empty vertical gap (width >= gap_frac * W) that ends the
    label block. The bounding box of content left of that gap is the label.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    fg = _foreground(gray, bg)

    col = fg.sum(axis=0)  # per-column foreground amount
    search_w = int(w * search_frac)
    if col[:search_w].max() == 0:
        return None  # no content on the left at all

    gap_w = max(5, int(w * gap_frac))
    empty = col <= (0.01 * col.max())

    # Find the rightmost column of the *first* label cluster: scan from the
    # leftmost content column until a run of `gap_w` empty columns appears.
    first = int(np.argmax(col > 0))
    end = first
    run = 0
    for x in range(first, search_w):
        if empty[x]:
            run += 1
            if run >= gap_w:
                end = x - gap_w
                break
        else:
            run = 0
            end = x
    else:
        return None  # content never closed with a gap -> not a label block

    band = fg[:, : end + 1]
    ys, xs = np.where(band > 0)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_slide.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add beewings/segment/slide.py tests/test_slide.py
git commit -m "feat(segment): background estimation and label detection"
```

---

## Task 2: Wing mask + bounding boxes (`wings.py`)

**Files:**
- Create: `beewings/segment/wings.py`
- Test: `tests/test_wings.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_wings.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_wings.py -q`
Expected: FAIL — `cannot import name 'wing_mask'`.

- [ ] **Step 3: Implement `wings.py`**

Create `beewings/segment/wings.py`:

```python
"""Wing foreground mask and per-wing bounding boxes."""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]


def wing_mask(
    img: np.ndarray,
    bg: float,
    delta: int = 25,
    close_frac: float = 0.012,
    exclude: Optional[Box] = None,
) -> np.ndarray:
    """Binary (0/255) mask of wings: content darker than background, with veins
    and membrane merged by morphological closing.

    `exclude` (the label box) is painted out before masking so it cannot be
    picked up as a wing.
    """
    work = img.copy()
    if exclude is not None:
        x, y, w, h = exclude
        work[y : y + h, x : x + w] = 255  # blank the label to background

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    mask = (gray < (bg - delta)).astype(np.uint8) * 255

    k = max(3, int(min(img.shape[:2]) * close_frac))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask


def find_wings(mask: np.ndarray, min_area_frac: float = 0.0003) -> List[Box]:
    """Return bounding boxes (x, y, w, h) of wing blobs, filtered by min area.

    Boxes are returned in arbitrary order; use layout.reading_order to sort.
    """
    total = mask.shape[0] * mask.shape[1]
    min_area = min_area_frac * total
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Box] = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        boxes.append(tuple(int(v) for v in cv2.boundingRect(c)))  # type: ignore
    return boxes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_wings.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add beewings/segment/wings.py tests/test_wings.py
git commit -m "feat(segment): wing mask and bounding-box extraction"
```

---

## Task 3: Reading-order numbering (`layout.py`)

**Files:**
- Create: `beewings/segment/layout.py`
- Test: `tests/test_layout.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_layout.py`:

```python
from __future__ import annotations

from beewings.segment.layout import reading_order


def test_reading_order_row_major():
    # Two rows of three, deliberately shuffled. Centers via boxes (x,y,w,h).
    boxes = [
        (300, 210, 40, 20),  # row1 col3
        (100, 10, 40, 20),   # row0 col1
        (200, 205, 40, 20),  # row1 col2
        (300, 12, 40, 20),   # row0 col3
        (100, 200, 40, 20),  # row1 col1
        (200, 8, 40, 20),    # row0 col2
    ]
    order = reading_order(boxes)
    # Expected sorted: row0 (y~10) left->right, then row1 (y~205) left->right.
    ordered_centers = [(boxes[i][0]) for i in order]
    assert ordered_centers == [100, 200, 300, 100, 200, 300]


def test_reading_order_single_wing():
    assert reading_order([(50, 50, 10, 10)]) == [0]


def test_reading_order_empty():
    assert reading_order([]) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_layout.py -q`
Expected: FAIL — `cannot import name 'reading_order'`.

- [ ] **Step 3: Implement `layout.py`**

Create `beewings/segment/layout.py`:

```python
"""Order wing boxes in human reading order (row-major: top->bottom, left->right).

Rows are found by clustering box centers on the Y axis: a new row starts when a
center is more than half the median box height below the current row's top.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

Box = Tuple[int, int, int, int]


def reading_order(boxes: List[Box]) -> List[int]:
    """Return indices of `boxes` sorted top-to-bottom, then left-to-right."""
    if not boxes:
        return []
    arr = np.array(boxes, dtype=float)
    cy = arr[:, 1] + arr[:, 3] / 2.0
    cx = arr[:, 0] + arr[:, 2] / 2.0
    median_h = float(np.median(arr[:, 3]))
    row_tol = max(1.0, median_h * 0.5)

    order_by_y = np.argsort(cy)
    rows: List[List[int]] = []
    cur: List[int] = [int(order_by_y[0])]
    cur_y = cy[order_by_y[0]]
    for idx in order_by_y[1:]:
        if cy[idx] - cur_y > row_tol:
            rows.append(cur)
            cur = []
            cur_y = cy[idx]
        cur.append(int(idx))
    rows.append(cur)

    result: List[int] = []
    for row in rows:
        result.extend(sorted(row, key=lambda i: cx[i]))
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_layout.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add beewings/segment/layout.py tests/test_layout.py
git commit -m "feat(segment): row-major reading-order numbering"
```

---

## Task 4: Cropping + optional canonical rotation (`crop.py`)

**Files:**
- Create: `beewings/segment/crop.py`
- Test: `tests/test_crop.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_crop.py`:

```python
from __future__ import annotations

import numpy as np

from beewings.segment.crop import crop_box, crop_wing


def test_crop_box_adds_margin_and_clips():
    img = np.zeros((100, 200, 3), np.uint8)
    out = crop_box(img, (50, 40, 20, 20), margin=0.5)
    # 0.5 margin on a 20px box -> +10px each side -> 40x40 region.
    assert out.shape[0] == 40 and out.shape[1] == 40


def test_crop_box_clips_at_image_edge():
    img = np.zeros((100, 200, 3), np.uint8)
    out = crop_box(img, (0, 0, 20, 20), margin=0.5)
    # Cannot extend past the top-left corner; region is clipped, not padded.
    assert out.shape[0] == 30 and out.shape[1] == 30


def test_crop_wing_rotate_makes_landscape():
    # A tall vertical bar should come out wider-than-tall after canonical rotate.
    img = np.full((200, 200, 3), 255, np.uint8)
    img[40:160, 90:110] = 30  # vertical dark bar
    box = (90, 40, 20, 120)
    out = crop_wing(img, box, margin=0.1, rotate=True, bg=255.0)
    assert out.shape[1] >= out.shape[0]  # width >= height
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_crop.py -q`
Expected: FAIL — `cannot import name 'crop_box'`.

- [ ] **Step 3: Implement `crop.py`**

Create `beewings/segment/crop.py`:

```python
"""Crop a wing out of the full-resolution scan, with optional canonical rotate.

Default output is a plain axis-aligned rectangle (the chosen behaviour). With
`rotate=True` the wing is rotated so its long axis is horizontal and the narrow
(base/articulation) end is on the left — matching how the ML model was trained.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]


def crop_box(img: np.ndarray, box: Box, margin: float = 0.10) -> np.ndarray:
    """Plain rectangular crop with a fractional margin, clipped to image bounds."""
    x, y, w, h = box
    mx = int(round(w * margin))
    my = int(round(h * margin))
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(img.shape[1], x + w + mx)
    y1 = min(img.shape[0], y + h + my)
    return img[y0:y1, x0:x1].copy()


def crop_wing(
    img: np.ndarray,
    box: Box,
    margin: float = 0.10,
    rotate: bool = False,
    bg: float = 255.0,
) -> np.ndarray:
    """Crop one wing. If `rotate`, canonicalize to horizontal, base on the left."""
    patch = crop_box(img, box, margin)
    if not rotate:
        return patch

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    fg = (gray < (bg - 25)).astype(np.uint8)
    ys, xs = np.where(fg > 0)
    if xs.size < 10:
        return patch  # nothing to orient

    # Principal axis via PCA on foreground pixel coordinates.
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    angle = np.degrees(np.arctan2(major[1], major[0]))

    h, w = patch.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += nw / 2.0 - w / 2.0
    M[1, 2] += nh / 2.0 - h / 2.0
    rotated = cv2.warpAffine(patch, M, (nw, nh), borderValue=(int(bg),) * 3)

    # Ensure base (narrow end) is on the left: compare foreground height of the
    # left vs right 20% columns; the narrower side is the base.
    rg = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    rfg = (rg < (bg - 25)).astype(np.uint8)
    col_h = rfg.sum(axis=0)
    band = max(1, rfg.shape[1] // 5)
    left_h = col_h[:band].mean()
    right_h = col_h[-band:].mean()
    if left_h > right_h:  # base currently on the right -> flip 180
        rotated = cv2.rotate(rotated, cv2.ROTATE_180)
    return rotated
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_crop.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add beewings/segment/crop.py tests/test_crop.py
git commit -m "feat(segment): rectangular crop + optional canonical rotation"
```

---

## Task 5: Debug overlay (`debug.py`)

**Files:**
- Create: `beewings/segment/debug.py`
- Test: `tests/test_crop.py` (append) — overlay is visual; assert shape/non-mutation only

- [ ] **Step 1: Write the failing test**

Append to `tests/test_crop.py`:

```python
from beewings.segment.debug import render_overlay


def test_render_overlay_returns_same_shape_without_mutating():
    img = np.full((120, 240, 3), 255, np.uint8)
    boxes = [(20, 20, 40, 30), (120, 60, 40, 30)]
    order = [0, 1]
    before = img.copy()
    out = render_overlay(img, boxes, order, label_box=(0, 0, 15, 100))
    assert out.shape == img.shape
    assert np.array_equal(img, before)  # input not mutated
    assert not np.array_equal(out, before)  # something was drawn
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_crop.py::test_render_overlay_returns_same_shape_without_mutating -q`
Expected: FAIL — `cannot import name 'render_overlay'`.

- [ ] **Step 3: Implement `debug.py`**

Create `beewings/segment/debug.py`:

```python
"""Render a debug overlay: numbered wing boxes + the label box."""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]


def render_overlay(
    img: np.ndarray,
    boxes: List[Box],
    order: List[int],
    label_box: Optional[Box] = None,
) -> np.ndarray:
    """Return a copy of `img` with green numbered wing boxes and a red label box."""
    out = img.copy()
    thick = max(1, min(img.shape[:2]) // 400)
    scale = max(0.5, min(img.shape[:2]) / 800)
    for n, idx in enumerate(order):
        x, y, w, h = boxes[idx]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 180, 0), thick)
        cv2.putText(out, str(n), (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), thick, cv2.LINE_AA)
    if label_box is not None:
        x, y, w, h = label_box
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), thick)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_crop.py -q`
Expected: PASS (all crop + overlay tests pass).

- [ ] **Step 5: Commit**

```bash
git add beewings/segment/debug.py tests/test_crop.py
git commit -m "feat(segment): debug overlay rendering"
```

---

## Task 6: CLI orchestration (`cli.py`)

**Files:**
- Create: `beewings/segment/cli.py`
- Test: `tests/test_cli_integration.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_cli_integration.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cli_integration.py -q`
Expected: FAIL — `cannot import name 'process_scan'`.

- [ ] **Step 3: Implement `cli.py`**

Create `beewings/segment/cli.py`:

```python
"""beewings-crop: split a scan slide into individual wing images.

Usage:
    beewings-crop --input scan.jpg --out crops/ [--debug]
    beewings-crop --input scan_dir/ --out crops/ --recursive --debug
    beewings-crop --input scan.jpg --inplace
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import cv2

from .crop import crop_box, crop_wing
from .debug import render_overlay
from .layout import reading_order
from .slide import detect_label_region, estimate_background
from .wings import find_wings, wing_mask

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def process_scan(
    scan_path: Path,
    out_dir: Path,
    margin: float = 0.10,
    min_area_frac: float = 0.0003,
    rotate: bool = False,
    debug: bool = False,
    dry_run: bool = False,
) -> int:
    """Crop one scan. Returns the number of wings found. Writes outputs to
    `out_dir` named `<stem>_crop_N.jpg`, `<stem>_label.jpg`, `<stem>_debug.jpg`.
    """
    img = cv2.imread(str(scan_path))
    if img is None:
        raise ValueError(f"cannot read image: {scan_path}")

    stem = scan_path.stem
    bg = estimate_background(img)
    label = detect_label_region(img, bg)
    mask = wing_mask(img, bg, exclude=label)
    boxes = find_wings(mask, min_area_frac=min_area_frac)
    order = reading_order(boxes)

    out_dir.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        for n, idx in enumerate(order):
            patch = crop_wing(img, boxes[idx], margin=margin, rotate=rotate, bg=bg)
            cv2.imwrite(str(out_dir / f"{stem}_crop_{n}.jpg"), patch)
        if label is not None:
            cv2.imwrite(str(out_dir / f"{stem}_label.jpg"), crop_box(img, label, 0.02))

    if debug:
        overlay = render_overlay(img, boxes, order, label_box=label)
        cv2.imwrite(str(out_dir / f"{stem}_debug.jpg"), overlay)

    return len(order)


def _iter_scans(root: Path, recursive: bool) -> List[Path]:
    if root.is_file():
        return [root]
    globber = root.rglob if recursive else root.glob
    return sorted(p for p in globber("*") if p.suffix.lower() in IMAGE_EXTS)


def _out_dir_for(scan: Path, input_root: Path, out: Optional[Path], inplace: bool) -> Path:
    if inplace:
        return scan.parent
    assert out is not None
    if input_root.is_file():
        return out
    # Mirror the sub-folder structure under `out`.
    rel = scan.parent.relative_to(input_root)
    return out / rel


def main() -> int:
    p = argparse.ArgumentParser(description="Split a scan slide into per-wing crops.")
    p.add_argument("--input", type=Path, required=True, help="scan file or folder")
    p.add_argument("--out", type=Path, help="output folder (mirrors input tree)")
    p.add_argument("--inplace", action="store_true", help="write crops next to the scan")
    p.add_argument("--recursive", action="store_true", help="recurse into sub-folders")
    p.add_argument("--margin", type=float, default=0.10)
    p.add_argument("--min-area-frac", type=float, default=0.0003)
    p.add_argument("--rotate-canonical", action="store_true",
                   help="rotate wings to horizontal, base on the left (for ML)")
    p.add_argument("--debug", action="store_true", help="save <stem>_debug.jpg overlay")
    p.add_argument("--dry-run", action="store_true", help="only write debug overlay")
    args = p.parse_args()

    if not args.inplace and args.out is None:
        p.error("either --out or --inplace is required")

    # Skip files we generate ourselves so re-runs are idempotent.
    skip = ("_crop_", "_label", "_debug")
    scans = [s for s in _iter_scans(args.input, args.recursive)
             if not any(t in s.stem for t in skip)]
    if not scans:
        print(f"no scans found in {args.input}", file=sys.stderr)
        return 1

    total = 0
    for scan in scans:
        out_dir = _out_dir_for(scan, args.input, args.out, args.inplace)
        try:
            n = process_scan(
                scan, out_dir,
                margin=args.margin, min_area_frac=args.min_area_frac,
                rotate=args.rotate_canonical, debug=args.debug, dry_run=args.dry_run,
            )
            print(f"{scan}: {n} wings")
            total += n
        except Exception as exc:  # keep going on the rest of the batch
            print(f"{scan}: ERROR {exc}", file=sys.stderr)

    print(f"done: {total} wings from {len(scans)} scan(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cli_integration.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS (all tests across all files).

- [ ] **Step 6: Reinstall to register the entry point and smoke-test**

Run: `.venv/bin/pip install -e . >/dev/null && .venv/bin/beewings-crop --help`
Expected: argparse help text listing `--input`, `--out`, `--rotate-canonical`, etc.

- [ ] **Step 7: Commit**

```bash
git add beewings/segment/cli.py tests/test_cli_integration.py
git commit -m "feat(segment): beewings-crop CLI orchestration"
```

---

## Task 7: Validate on a real scan + document

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run on three real scans (different folders) with debug**

Run:
```bash
.venv/bin/beewings-crop --input "скан/Кх Владик/Кх Владик/04401/04401.jpg" \
    --out /tmp/crop_check/vladik --debug
.venv/bin/beewings-crop --input "скан/Клат/8.jpg" --out /tmp/crop_check/klat --debug
.venv/bin/beewings-crop --input "скан/Ип Мирных/02490.jpg" --out /tmp/crop_check/mirnyh --debug
```
Expected: each prints a wing count close to ground truth (Владик ≈ 28, Клат ≈ 50, Мирных ≈ a full grid).

- [ ] **Step 2: Visually inspect the debug overlays**

Open `/tmp/crop_check/*/*_debug.jpg`. Confirm: every wing has one green box, numbers run row-major, the label has a red box and is NOT among the wing crops. If a parameter is off (over/under-segmentation), note which `--min-area-frac` / `--margin` / `wing_mask` `delta` value fixes it and adjust the default in code, then re-run this task's Step 1.

- [ ] **Step 3: Spot-check the canonical-rotation path**

Run:
```bash
.venv/bin/beewings-crop --input "скан/Клат/8.jpg" --out /tmp/crop_check/klat_rot \
    --rotate-canonical --debug
```
Open a few `8_crop_*.jpg`: wings should be horizontal with the narrow base on the left.

- [ ] **Step 4: Document the new command in README**

In `README.md`, in the CLI table (the `| Команда | Назначение |` table), add this row after the `beewings-ml-batch` row:

```markdown
| `beewings-crop` | Авто-нарезка скана на отдельные крылья (контурная сегментация) |
```

And add a usage block after the "Пакетный пример" section:

````markdown
### Авто-нарезка скана

```bash
# нарезать один скан на отдельные крылья (+ debug-оверлей для проверки)
beewings-crop --input "скан/Клат/8.jpg" --out crops/Клат --debug

# вся папка рекурсивно, с поворотом крыльев в горизонталь для ML
beewings-crop --input "скан/" --out crops/ --recursive --rotate-canonical
```

Выход: `<скан>_crop_0.jpg … _crop_N.jpg` (отдельные крылья), `<скан>_label.jpg`
(этикетка), `<скан>_debug.jpg` (оверлей с рамками). Кропы подаются напрямую в
`beewings-ml-batch`.
````

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document beewings-crop auto-crop command"
```

---

## Self-Review

**Spec coverage:**
- §3 hybrid CV approach → Tasks 1–3 (mask, contours, grid order). ✓
- §4 module layout → Tasks 1–6 create exactly the specified files. ✓
- §5 pipeline steps 1–7 → load/bg (Task 1), label mask (Task 1+2 `exclude`), wing mask (Task 2), contours+area filter (Task 2), order (Task 3), crop+rotate (Task 4), debug (Task 5), orchestration (Task 6). ✓
- §6 label saved separately + masked out + `--ocr-label` deferred → `process_scan` writes `_label.jpg`, `wing_mask(exclude=...)`; OCR explicitly out of scope. ✓ (flag deferred per §11)
- §7 CLI flags → all present in Task 6 `main()` (`--input/--out/--inplace/--recursive/--margin/--min-area-frac/--rotate-canonical/--debug/--dry-run`). ✓
- §8 mirror output structure → `_out_dir_for`. ✓
- §9 edge cases: pale tip → morphology+margin (Task 2/4); varying background → `estimate_background` corners (Task 1); count mismatch → printed per-scan count + keeps going on error (Task 6); broken scan → `ValueError` caught in batch loop (Task 6). ✓ (watershed for touching wings noted in spec as the mechanism; current data is non-touching, so deferred — flagged here.)
- §10 testing → unit tests Tasks 1–5, integration Task 6, real-scan visual Task 7. ✓
- §11 out-of-scope (OCR, trained detector, .tps) → not implemented. ✓

**Known deferral:** Watershed splitting of touching wings (§5/§9) is NOT implemented because all sample scans have well-separated wings. If Task 7 Step 2 reveals merged boxes, add a follow-up task: split blobs whose bbox area >> median using `cv2.watershed` seeded by grid-cell centers from `layout`.

**Placeholder scan:** No TBD/TODO; every code step has complete code. ✓

**Type consistency:** `Box = (x,y,w,h)` used identically across `slide`/`wings`/`layout`/`crop`/`debug`/`cli`. `estimate_background -> float` consumed as `bg` everywhere. `wing_mask(img, bg, exclude=)` / `find_wings(mask, min_area_frac=)` / `reading_order(boxes)->List[int]` / `crop_wing(img, box, margin, rotate, bg)` signatures match their call sites in `cli.process_scan`. ✓
