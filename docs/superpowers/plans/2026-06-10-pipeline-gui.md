# Pipeline-in-GUI Wizard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an in-window wizard that runs the full workflow — pick a folder of big scans → auto-crop all → review/edit crop boxes on the scan → place landmarks (ML) → review in the existing annotator → export a structured protocol.

**Architecture:** New Qt-free logic layer (`project.py`, `export.py`, `landmarks.py`, `geometry.py`) that is fully unit-tested, plus thin PyQt6 UI on top (`box_canvas.py`, `workers.py`, `wizard.py`, `pages/`). The wizard manages a persistent project (`project.json`). The landmark-review step reuses the existing annotator via `MainWindow.load_folder`.

**Tech Stack:** Python 3.10+, PyQt6, numpy, opencv-python, pydantic, torch (ML), pytest. All already in the project. Reuses `beewings/segment/`, `beewings/ml/inference.py`, `beewings/core/{schema,profiles,indices,io_csv,io_tps}.py`.

---

## File Structure

| File | Responsibility |
|---|---|
| `beewings/pipeline/__init__.py` | package marker |
| `beewings/pipeline/project.py` | `Box`, `ScanEntry`, `Settings`, `CropProject` (manifest load/save), `auto_detect`, `recrop` |
| `beewings/pipeline/geometry.py` | pure box-editing math: `normalize_box`, `hit_test`, `resize_box` |
| `beewings/pipeline/landmarks.py` | `run_landmarks` — Qt-free ML over a crops folder → annotation JSONs |
| `beewings/pipeline/export.py` | `export_protocol` — folders copy + summary.csv + all_wings.tps + report.json |
| `beewings/pipeline/box_canvas.py` | `BoxEditorCanvas` QWidget (zoom/pan + draw/edit boxes) |
| `beewings/pipeline/workers.py` | `CropWorker`, `LandmarkWorker` QThread wrappers |
| `beewings/pipeline/wizard.py` | `PipelineWizard` (stepper + project state) |
| `beewings/pipeline/pages/select_page.py` | stage 1 — choose scan folder |
| `beewings/pipeline/pages/crop_page.py` | stage 2 — auto-crop + box editor + recrop |
| `beewings/pipeline/pages/landmark_page.py` | stage 3 — run ML + open in annotator |
| `beewings/pipeline/pages/export_page.py` | stage 4 — export options + run |
| `beewings/annotator/main_window.py` | add "🧩 Конвейер нарезки…" menu action |
| `pyproject.toml` | add `beewings-pipeline` entry point |
| `tests/conftest.py` | add `qapp` fixture + `scan_file` helper |
| `tests/test_pipeline_project.py` | project manifest + auto_detect + recrop |
| `tests/test_pipeline_geometry.py` | box geometry |
| `tests/test_pipeline_export.py` | export protocol |
| `tests/test_pipeline_landmarks.py` | landmark run (skip if no checkpoint) |
| `tests/test_pipeline_qt_smoke.py` | construct canvas/pages/wizard headless |

### Shared types (locked)
- `Box = Tuple[int, int, int, int]` — `(x, y, w, h)`, ints, in scan pixel coords. Reuse from `beewings.segment.wings.Box` by re-import where needed; in `project.py` define its own alias identically.
- Manifest filename: `project.json` at the project root.
- Crops dir for a scan: `<root>/crops/<scan_stem>/`.

---

## Task 0: Scaffold package + test fixtures

**Files:**
- Create: `beewings/pipeline/__init__.py`, `beewings/pipeline/pages/__init__.py`
- Modify: `pyproject.toml`, `tests/conftest.py`

- [ ] **Step 1: Create package markers**

Create `beewings/pipeline/__init__.py`:
```python
"""In-window pipeline wizard: scans -> crop -> landmarks -> export."""
from __future__ import annotations
```
Create `beewings/pipeline/pages/__init__.py`:
```python
from __future__ import annotations
```

- [ ] **Step 2: Register the entry point**

In `pyproject.toml`, under `[project.scripts]`, after `beewings-crop = ...`, add:
```toml
beewings-pipeline = "beewings.pipeline.wizard:main"
```

- [ ] **Step 3: Add a headless Qt fixture and a scan-file helper to tests/conftest.py**

Append to `tests/conftest.py`:
```python
import os


@pytest.fixture(scope="session")
def qapp():
    """A headless QApplication for widget construction tests."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    return app


@pytest.fixture
def scan_file(synthetic_scan, tmp_path):
    """Write the synthetic scan to a temp .jpg and return its Path."""
    img, meta = synthetic_scan
    path = tmp_path / "scan_0001.jpg"
    cv2.imwrite(str(path), img)
    return path, meta
```

- [ ] **Step 4: Verify collection**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: existing 19 tests still pass (new fixtures unused yet).

- [ ] **Step 5: Commit**
```bash
git add beewings/pipeline/__init__.py beewings/pipeline/pages/__init__.py pyproject.toml tests/conftest.py
git -c user.name="BeeWings" -c user.email="nurkal836@gmail.com" commit -m "chore: scaffold pipeline package, entry point, qt test fixtures"
```

---

## Task 1: Project manifest model (`project.py` part 1)

**Files:**
- Create: `beewings/pipeline/project.py`
- Test: `tests/test_pipeline_project.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline_project.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pipeline_project.py -q`
Expected: FAIL — `ModuleNotFoundError: beewings.pipeline.project`.

- [ ] **Step 3: Implement the manifest model**

Create `beewings/pipeline/project.py`:
```python
"""Pipeline project: a working directory + project.json manifest."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

Box = Tuple[int, int, int, int]


@dataclass
class Settings:
    margin: float = 0.10
    delta: int = 45
    label_frac: Optional[float] = None
    rotate: bool = True
    profile: str = "Алпатов 12 точек"
    checkpoint: str = "checkpoints/alpatov12.pt"


@dataclass
class ScanEntry:
    path: str
    label_box: Optional[Box] = None
    wing_boxes: List[Box] = field(default_factory=list)
    cropped: bool = False


@dataclass
class CropProject:
    root: Path
    scans_root: str
    settings: Settings = field(default_factory=Settings)
    stage: str = "select"
    scans: List[ScanEntry] = field(default_factory=list)

    # ---- construction / persistence -------------------------------------
    @classmethod
    def create(cls, root: Path, scans_root: str, scan_paths: List[Path]) -> "CropProject":
        root = Path(root)
        scans = [ScanEntry(path=str(p)) for p in scan_paths]
        return cls(root=root, scans_root=str(scans_root), scans=scans)

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "scans_root": self.scans_root,
            "settings": asdict(self.settings),
            "stage": self.stage,
            "scans": [
                {
                    "path": s.path,
                    "label_box": list(s.label_box) if s.label_box else None,
                    "wing_boxes": [list(b) for b in s.wing_boxes],
                    "cropped": s.cropped,
                }
                for s in self.scans
            ],
        }
        tmp = self.root / "project.json.tmp"
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.root / "project.json")

    @classmethod
    def load(cls, root: Path) -> "CropProject":
        root = Path(root)
        data = json.loads((root / "project.json").read_text(encoding="utf-8"))

        def _box(v):
            return tuple(int(i) for i in v) if v else None

        scans = [
            ScanEntry(
                path=s["path"],
                label_box=_box(s.get("label_box")),
                wing_boxes=[tuple(int(i) for i in b) for b in s.get("wing_boxes", [])],
                cropped=bool(s.get("cropped", False)),
            )
            for s in data.get("scans", [])
        ]
        return cls(
            root=root,
            scans_root=data["scans_root"],
            settings=Settings(**data.get("settings", {})),
            stage=data.get("stage", "select"),
            scans=scans,
        )

    # ---- paths ----------------------------------------------------------
    def crops_dir(self, entry: ScanEntry) -> Path:
        return self.root / "crops" / Path(entry.path).stem
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pipeline_project.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**
```bash
git add beewings/pipeline/project.py tests/test_pipeline_project.py
git -c user.name="BeeWings" -c user.email="nurkal836@gmail.com" commit -m "feat(pipeline): project manifest model with atomic save/load"
```

---

## Task 2: Auto-detect + recrop (`project.py` part 2)

**Files:**
- Modify: `beewings/pipeline/project.py`
- Test: `tests/test_pipeline_project.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline_project.py`:
```python
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
    # Manually set exactly 2 boxes; recrop must produce exactly 2 crops.
    proj.scans[0].wing_boxes = [(360, 90, 110, 56), (520, 90, 110, 56)]
    proj.settings.rotate = False
    n = recrop(proj, proj.scans[0])
    assert n == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pipeline_project.py -k "auto_detect or recrop" -q`
Expected: FAIL — `cannot import name 'auto_detect'`.

- [ ] **Step 3: Implement auto_detect + recrop**

Append to `beewings/pipeline/project.py`:
```python
import cv2

from ..segment.crop import crop_box, crop_wing
from ..segment.debug import render_overlay
from ..segment.layout import reading_order
from ..segment.slide import detect_label_region, estimate_background
from ..segment.wings import find_wings, wing_mask


def auto_detect(entry: ScanEntry, settings: Settings) -> None:
    """Fill entry.label_box and entry.wing_boxes by running segmentation."""
    img = cv2.imread(entry.path)
    if img is None:
        raise ValueError(f"cannot read image: {entry.path}")
    bg = estimate_background(img)
    if settings.label_frac is not None:
        h, w = img.shape[:2]
        label = (0, 0, int(round(settings.label_frac * w)), h)
    else:
        label = detect_label_region(img, bg)
    mask = wing_mask(img, bg, delta=settings.delta, exclude=label)
    boxes = find_wings(mask)
    order = reading_order(boxes)
    entry.label_box = label
    entry.wing_boxes = [boxes[i] for i in order]


def recrop(project: "CropProject", entry: ScanEntry) -> int:
    """Write per-wing crops (+ label + debug) for one scan from its boxes.

    Returns the number of wing crops written. Crops follow the manifest boxes
    exactly (manual edits respected), not a fresh auto-detection.
    """
    img = cv2.imread(entry.path)
    if img is None:
        raise ValueError(f"cannot read image: {entry.path}")
    bg = estimate_background(img)
    stem = Path(entry.path).stem
    out_dir = project.crops_dir(entry)
    out_dir.mkdir(parents=True, exist_ok=True)

    for n, box in enumerate(entry.wing_boxes):
        patch = crop_wing(img, box, margin=project.settings.margin,
                          rotate=project.settings.rotate, bg=bg)
        cv2.imwrite(str(out_dir / f"{stem}_crop_{n}.jpg"), patch)
    if entry.label_box is not None:
        cv2.imwrite(str(out_dir / f"{stem}_label.jpg"),
                    crop_box(img, entry.label_box, 0.02))
    order = list(range(len(entry.wing_boxes)))
    overlay = render_overlay(img, entry.wing_boxes, order, label_box=entry.label_box)
    cv2.imwrite(str(out_dir / f"{stem}_debug.jpg"), overlay)

    entry.cropped = True
    return len(entry.wing_boxes)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pipeline_project.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**
```bash
git add beewings/pipeline/project.py tests/test_pipeline_project.py
git -c user.name="BeeWings" -c user.email="nurkal836@gmail.com" commit -m "feat(pipeline): auto-detect boxes and recrop from manifest"
```

---

## Task 3: Box-editing geometry (`geometry.py`)

**Files:**
- Create: `beewings/pipeline/geometry.py`
- Test: `tests/test_pipeline_geometry.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline_geometry.py`:
```python
from __future__ import annotations

from beewings.pipeline.geometry import normalize_box, hit_test, resize_box


def test_normalize_negative_size():
    assert normalize_box((50, 50, -20, -10)) == (30, 40, 20, 10)


def test_hit_test_corner():
    boxes = [(10, 10, 100, 80)]
    # Near the bottom-right corner -> resize handle 'se'.
    idx, handle = hit_test(boxes, 109, 89, handle_px=8)
    assert idx == 0 and handle == "se"


def test_hit_test_body():
    boxes = [(10, 10, 100, 80)]
    idx, handle = hit_test(boxes, 60, 50, handle_px=8)
    assert idx == 0 and handle == "move"


def test_hit_test_miss():
    boxes = [(10, 10, 100, 80)]
    idx, handle = hit_test(boxes, 500, 500, handle_px=8)
    assert idx is None and handle is None


def test_resize_se_corner():
    # Dragging the SE corner by (+10, +20) grows width/height.
    assert resize_box((10, 10, 100, 80), "se", 10, 20) == (10, 10, 110, 100)


def test_resize_nw_corner():
    # Dragging the NW corner by (+5, +5) moves origin and shrinks size.
    assert resize_box((10, 10, 100, 80), "nw", 5, 5) == (15, 15, 95, 75)


def test_resize_move():
    assert resize_box((10, 10, 100, 80), "move", 7, -3) == (17, 7, 100, 80)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pipeline_geometry.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement geometry**

Create `beewings/pipeline/geometry.py`:
```python
"""Pure geometry for editing rectangular boxes (no Qt)."""
from __future__ import annotations

from typing import List, Optional, Tuple

Box = Tuple[int, int, int, int]
Handle = Optional[str]  # None | 'move' | 'nw' | 'ne' | 'sw' | 'se'


def normalize_box(box: Box) -> Box:
    """Return an equivalent box with non-negative width and height."""
    x, y, w, h = box
    if w < 0:
        x, w = x + w, -w
    if h < 0:
        y, h = y + h, -h
    return (x, y, w, h)


def _corners(box: Box):
    x, y, w, h = box
    return {"nw": (x, y), "ne": (x + w, y), "sw": (x, y + h), "se": (x + w, y + h)}


def hit_test(boxes: List[Box], px: int, py: int, handle_px: int = 8):
    """Return (index, handle) for the topmost box hit at (px, py), else (None, None).

    A corner within `handle_px` yields that corner handle; inside the box yields
    'move'. Iterates last-to-first so the most recently drawn box wins.
    """
    for idx in range(len(boxes) - 1, -1, -1):
        box = normalize_box(boxes[idx])
        for name, (cx, cy) in _corners(box).items():
            if abs(px - cx) <= handle_px and abs(py - cy) <= handle_px:
                return idx, name
        x, y, w, h = box
        if x <= px <= x + w and y <= py <= y + h:
            return idx, "move"
    return None, None


def resize_box(box: Box, handle: Handle, dx: int, dy: int) -> Box:
    """Apply a drag delta to a box given the grabbed handle. Returns a new box."""
    x, y, w, h = box
    if handle == "move":
        return (x + dx, y + dy, w, h)
    if handle == "nw":
        return normalize_box((x + dx, y + dy, w - dx, h - dy))
    if handle == "ne":
        return normalize_box((x, y + dy, w + dx, h - dy))
    if handle == "sw":
        return normalize_box((x + dx, y, w - dx, h + dy))
    if handle == "se":
        return normalize_box((x, y, w + dx, h + dy))
    return box
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pipeline_geometry.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**
```bash
git add beewings/pipeline/geometry.py tests/test_pipeline_geometry.py
git -c user.name="BeeWings" -c user.email="nurkal836@gmail.com" commit -m "feat(pipeline): pure box-editing geometry"
```

---

## Task 4: Landmark runner (`landmarks.py`)

**Files:**
- Create: `beewings/pipeline/landmarks.py`
- Test: `tests/test_pipeline_landmarks.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline_landmarks.py`:
```python
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

CKPT = Path("checkpoints/alpatov12.pt")


@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint not available")
def test_run_landmarks_writes_annotations(tmp_path):
    from beewings.pipeline.landmarks import run_landmarks
    # Two fake wing crops.
    crops = tmp_path / "crops"
    crops.mkdir()
    for i in range(2):
        img = np.full((300, 600, 3), 230, np.uint8)
        cv2.ellipse(img, (300, 150), (200, 70), 0, 0, 360, (120, 120, 120), -1)
        cv2.imwrite(str(crops / f"w_{i}.jpg"), img)

    seen = []
    n = run_landmarks(crops, profile_name="Алпатов 12 точек", checkpoint=CKPT,
                      progress_cb=lambda i, total, name: seen.append((i, total)))
    assert n == 2
    from beewings.core.schema import annotation_path, load_annotation
    from beewings.core.profiles import get_profile
    prof = get_profile("Алпатов 12 точек")
    ann = load_annotation(annotation_path(crops / "w_0.jpg", crops, prof.methodology_id))
    assert ann is not None and len(ann.landmarks) == 12
    assert seen[-1] == (2, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline_landmarks.py -q`
Expected: FAIL — `ModuleNotFoundError` (or skip if checkpoint missing; if skipped, still implement).

- [ ] **Step 3: Implement run_landmarks**

Create `beewings/pipeline/landmarks.py`:
```python
"""Qt-free ML landmark placement over a folder of wing crops."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import cv2

from ..core.profiles import get_profile
from ..core.schema import Landmark, WingAnnotation, annotation_path, save_annotation
from ..ml.inference import load as ml_load, predict as ml_predict

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
ProgressCb = Optional[Callable[[int, int, str], None]]


def _crop_images(crops_dir: Path):
    skip = ("_label", "_debug")
    return sorted(
        p for p in crops_dir.glob("*")
        if p.suffix.lower() in IMAGE_EXTS and not any(t in p.stem for t in skip)
    )


def run_landmarks(crops_dir: Path, profile_name: str, checkpoint: Path,
                  device: str = "auto", tta: bool = False,
                  progress_cb: ProgressCb = None) -> int:
    """Predict landmarks for every crop in `crops_dir`, saving annotation JSONs
    next to them (in the layout the annotator reads). Returns the count done."""
    crops_dir = Path(crops_dir)
    profile = get_profile(profile_name)
    model = ml_load(Path(checkpoint), device=device)
    images = _crop_images(crops_dir)
    total = len(images)
    for i, img_path in enumerate(images, start=1):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            if progress_cb:
                progress_cb(i, total, img_path.name)
            continue
        pred, conf = ml_predict(model, bgr, tta=tta, return_confidence=True)
        h, w = bgr.shape[:2]
        lms = [Landmark(id=int(k), x=float(v[0]), y=float(v[1]),
                        uncertain=conf.get(k, 1.0) < 0.45)
               for k, v in pred.items()]
        ann = WingAnnotation(image=img_path.name, image_size=(w, h),
                             profile=profile.name, landmarks=lms, annotator="ml")
        save_annotation(ann, annotation_path(img_path, crops_dir, profile.methodology_id))
        if progress_cb:
            progress_cb(i, total, img_path.name)
    return total
```

- [ ] **Step 4: Run test to verify it passes (or skips cleanly)**

Run: `.venv/bin/python -m pytest tests/test_pipeline_landmarks.py -q`
Expected: PASS (1 passed) if `checkpoints/alpatov12.pt` exists; otherwise `1 skipped`. Both are acceptable.

- [ ] **Step 5: Commit**
```bash
git add beewings/pipeline/landmarks.py tests/test_pipeline_landmarks.py
git -c user.name="BeeWings" -c user.email="nurkal836@gmail.com" commit -m "feat(pipeline): Qt-free ML landmark runner over crops folder"
```

---

## Task 5: Protocol export (`export.py`)

**Files:**
- Create: `beewings/pipeline/export.py`
- Test: `tests/test_pipeline_export.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline_export.py`:
```python
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from beewings.core.profiles import get_profile
from beewings.core.schema import Landmark, WingAnnotation, annotation_path, save_annotation
from beewings.pipeline.project import CropProject


def _make_project_with_crops(tmp_path):
    """A project with one scan, two crops, each with 12 saved landmarks."""
    root = tmp_path / "proj"
    scan = tmp_path / "8.jpg"
    cv2.imwrite(str(scan), np.full((50, 50, 3), 255, np.uint8))
    proj = CropProject.create(root, scans_root=str(tmp_path), scan_paths=[scan])
    proj.settings.profile = "Алпатов 12 точек"
    entry = proj.scans[0]
    entry.cropped = True
    cdir = proj.crops_dir(entry)
    cdir.mkdir(parents=True)
    prof = get_profile("Алпатов 12 точек")
    for i in range(2):
        cp = cdir / f"8_crop_{i}.jpg"
        cv2.imwrite(str(cp), np.full((40, 80, 3), 255, np.uint8))
        lms = [Landmark(id=j, x=float(j * 5 + 1), y=float(j * 3 + 1)) for j in range(1, 13)]
        ann = WingAnnotation(image=cp.name, image_size=(80, 40),
                             profile=prof.name, landmarks=lms)
        save_annotation(ann, annotation_path(cp, cdir, prof.methodology_id))
    proj.save()
    return proj


def test_export_all(tmp_path):
    from beewings.pipeline.export import export_protocol
    proj = _make_project_with_crops(tmp_path)
    out = proj.root / "export"
    stats = export_protocol(proj, out,
                            include={"folders", "summary", "tps", "report"})
    assert (out / "summary.csv").exists()
    assert (out / "all_wings.tps").exists()
    assert (out / "report.json").exists()
    assert (out / "data" / "8").exists()  # folders copied

    rows = list(csv.DictReader(open(out / "summary.csv", encoding="utf-8")))
    assert len(rows) == 2                      # one row per wing
    assert "scan" in rows[0] and "wing" in rows[0]
    assert any(k.startswith("x1") for k in rows[0])

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["n_wings"] == 2
    assert report["n_scans"] == 1


def test_export_subset_only_summary(tmp_path):
    from beewings.pipeline.export import export_protocol
    proj = _make_project_with_crops(tmp_path)
    out = proj.root / "export2"
    export_protocol(proj, out, include={"summary"})
    assert (out / "summary.csv").exists()
    assert not (out / "all_wings.tps").exists()
    assert not (out / "data").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pipeline_export.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement export_protocol**

Create `beewings/pipeline/export.py`:
```python
"""Structured protocol export: data folders, summary CSV, TPS, report JSON."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set

from ..core.indices import compute_all_alpatov
from ..core.io_tps import export_tps
from ..core.profiles import get_profile
from ..core.schema import WingAnnotation, annotation_path, load_annotation
from .project import CropProject, ScanEntry

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _crop_paths(project: CropProject, entry: ScanEntry) -> List[Path]:
    cdir = project.crops_dir(entry)
    if not cdir.exists():
        return []
    skip = ("_label", "_debug")
    return sorted(
        p for p in cdir.glob("*")
        if p.suffix.lower() in IMAGE_EXTS and not any(t in p.stem for t in skip)
    )


def _collect(project: CropProject) -> List[tuple]:
    """Return list of (scan_stem, crop_path, WingAnnotation) for all cropped scans."""
    prof = get_profile(project.settings.profile)
    out = []
    for entry in project.scans:
        cdir = project.crops_dir(entry)
        for cp in _crop_paths(project, entry):
            ann = load_annotation(annotation_path(cp, cdir, prof.methodology_id))
            if ann is not None:
                out.append((Path(entry.path).stem, cp, ann))
    return out


def _summary_rows(items, n_points: int) -> List[dict]:
    rows = []
    for scan_stem, cp, ann in items:
        coords = {lm.id: (lm.x, lm.y) for lm in ann.landmarks}
        row = {"scan": scan_stem, "wing": cp.name}
        for i in range(1, n_points + 1):
            x, y = coords.get(i, ("", ""))
            row[f"x{i}"] = x
            row[f"y{i}"] = y
        for res in compute_all_alpatov(coords):  # returns List[IndexResult]
            row[res.name] = res.value if res.value is not None else ""
        rows.append(row)
    return rows


def export_protocol(project: CropProject, out_dir: Path,
                    include: Set[str]) -> Dict:
    """Write the selected protocol artifacts. Returns summary stats."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prof = get_profile(project.settings.profile)
    n_points = len(prof.ids)
    items = _collect(project)
    annotations: List[WingAnnotation] = [a for _, _, a in items]

    if "folders" in include:
        data_dir = out_dir / "data"
        for entry in project.scans:
            cdir = project.crops_dir(entry)
            if cdir.exists():
                shutil.copytree(cdir, data_dir / cdir.name, dirs_exist_ok=True)

    if "summary" in include:
        rows = _summary_rows(items, n_points)
        fields = (["scan", "wing"]
                  + [f"{ax}{i}" for i in range(1, n_points + 1) for ax in ("x", "y")])
        extra = [k for r in rows for k in r if k not in fields]
        # keep index columns in first-seen order
        seen = list(dict.fromkeys(extra))
        fields = fields + seen
        with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    if "tps" in include:
        export_tps(annotations, out_dir / "all_wings.tps")

    if "report" in include:
        confs = []  # confidences aren't stored per-point in annotations; report counts
        report = {
            "profile": prof.name,
            "n_scans": sum(1 for e in project.scans if e.cropped),
            "n_wings": len(items),
        }
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"n_wings": len(items), "n_scans": sum(1 for e in project.scans if e.cropped)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pipeline_export.py -q`
Expected: PASS (2 passed).

NOTE: `compute_all_alpatov(coords)` takes a `{id:(x,y)}` dict and returns a
`List[IndexResult]` (each with `.name` and `.value`) — confirmed. The summary
computes Alpatov indices; for a Tofilski project they will mostly be `None`,
which is acceptable (default profile is Alpatov 12).

- [ ] **Step 5: Commit**
```bash
git add beewings/pipeline/export.py tests/test_pipeline_export.py
git -c user.name="BeeWings" -c user.email="nurkal836@gmail.com" commit -m "feat(pipeline): structured protocol export (folders/summary/tps/report)"
```

---

## Task 6: Box editor canvas (`box_canvas.py`)

**Files:**
- Create: `beewings/pipeline/box_canvas.py`
- Test: `tests/test_pipeline_qt_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_pipeline_qt_smoke.py`:
```python
from __future__ import annotations

import numpy as np


def test_box_canvas_set_and_edit(qapp):
    from beewings.pipeline.box_canvas import BoxEditorCanvas
    c = BoxEditorCanvas()
    img = np.full((200, 400, 3), 255, np.uint8)
    c.set_scan(img, wing_boxes=[(10, 10, 50, 40)], label_box=(0, 0, 20, 200))
    assert c.wing_boxes() == [(10, 10, 50, 40)]
    assert c.label_box() == (0, 0, 20, 200)
    # Programmatic delete keeps API stable.
    c.delete_box(0)
    assert c.wing_boxes() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline_qt_smoke.py::test_box_canvas_set_and_edit -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement BoxEditorCanvas**

Create `beewings/pipeline/box_canvas.py`:
```python
"""Zoomable canvas for drawing and editing rectangular boxes over a scan."""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget

from .geometry import Box, hit_test, normalize_box, resize_box

_GREEN = (0, 180, 0)
_RED = (230, 30, 60)


class BoxEditorCanvas(QWidget):
    """Displays a scan with editable wing boxes (green) and one label box (red).

    Mouse: drag on empty space draws a new wing box; drag a corner resizes; drag
    a body moves; right-click deletes. Ctrl+wheel zooms; middle/space drag pans.
    Emits boxesChanged whenever boxes are modified.
    """

    boxesChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._pix: Optional[QPixmap] = None
        self._wing: List[Box] = []
        self._label: Optional[Box] = None
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self._sel: Optional[int] = None
        self._handle = None
        self._drag_last: Optional[QPoint] = None
        self._new_origin = None

    # ---- public API -----------------------------------------------------
    def set_scan(self, img_bgr: np.ndarray, wing_boxes: List[Box],
                 label_box: Optional[Box]) -> None:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._pix = QPixmap.fromImage(qimg)
        self._wing = [normalize_box(b) for b in wing_boxes]
        self._label = normalize_box(label_box) if label_box else None
        self._fit()
        self.update()

    def wing_boxes(self) -> List[Box]:
        return list(self._wing)

    def label_box(self) -> Optional[Box]:
        return self._label

    def delete_box(self, idx: int) -> None:
        if 0 <= idx < len(self._wing):
            self._wing.pop(idx)
            self.boxesChanged.emit()
            self.update()

    def make_label(self, idx: int) -> None:
        if 0 <= idx < len(self._wing):
            self._label = self._wing.pop(idx)
            self.boxesChanged.emit()
            self.update()

    # ---- view transforms ------------------------------------------------
    def _fit(self) -> None:
        if not self._pix:
            return
        pw, ph = self._pix.width(), self._pix.height()
        if pw == 0 or ph == 0:
            return
        self._scale = min(self.width() / pw, self.height() / ph) or 1.0
        self._offset = QPointF(0, 0)

    def _to_img(self, pos) -> QPoint:
        x = (pos.x() - self._offset.x()) / self._scale
        y = (pos.y() - self._offset.y()) / self._scale
        return QPoint(int(x), int(y))

    # ---- painting -------------------------------------------------------
    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.GlobalColor.darkGray)
        if not self._pix:
            return
        p.translate(self._offset)
        p.scale(self._scale, self._scale)
        p.drawPixmap(0, 0, self._pix)
        pen = QPen(); pen.setCosmetic(True); pen.setWidth(2)
        pen.setColor(Qt.GlobalColor.green)
        p.setPen(pen)
        for (x, y, w, h) in self._wing:
            p.drawRect(x, y, w, h)
        if self._label:
            pen.setColor(Qt.GlobalColor.red); p.setPen(pen)
            x, y, w, h = self._label
            p.drawRect(x, y, w, h)

    def resizeEvent(self, _ev) -> None:
        self._fit()

    # ---- mouse ----------------------------------------------------------
    def mousePressEvent(self, ev) -> None:
        ip = self._to_img(ev.position())
        if ev.button() == Qt.MouseButton.RightButton:
            idx, _ = hit_test(self._wing, ip.x(), ip.y(), int(8 / self._scale) + 1)
            if idx is not None:
                self.delete_box(idx)
            return
        if ev.button() == Qt.MouseButton.LeftButton:
            idx, handle = hit_test(self._wing, ip.x(), ip.y(), int(8 / self._scale) + 1)
            if idx is not None:
                self._sel, self._handle, self._drag_last = idx, handle, ip
            else:
                self._new_origin = ip  # start drawing a new box

    def mouseMoveEvent(self, ev) -> None:
        ip = self._to_img(ev.position())
        if self._sel is not None and self._drag_last is not None:
            dx, dy = ip.x() - self._drag_last.x(), ip.y() - self._drag_last.y()
            self._wing[self._sel] = resize_box(self._wing[self._sel], self._handle, dx, dy)
            self._drag_last = ip
            self.update()
        elif self._new_origin is not None:
            x0, y0 = self._new_origin.x(), self._new_origin.y()
            self._preview = normalize_box((x0, y0, ip.x() - x0, ip.y() - y0))
            self.update()

    def mouseReleaseEvent(self, ev) -> None:
        if self._new_origin is not None:
            x0, y0 = self._new_origin.x(), self._new_origin.y()
            ip = self._to_img(ev.position())
            box = normalize_box((x0, y0, ip.x() - x0, ip.y() - y0))
            if box[2] > 5 and box[3] > 5:
                self._wing.append(box)
                self.boxesChanged.emit()
            self._new_origin = None
            self.update()
        elif self._sel is not None:
            self._wing[self._sel] = normalize_box(self._wing[self._sel])
            self._sel = self._handle = self._drag_last = None
            self.boxesChanged.emit()

    def wheelEvent(self, ev) -> None:
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.2 if ev.angleDelta().y() > 0 else 1 / 1.2
            self._scale *= factor
            self.update()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline_qt_smoke.py::test_box_canvas_set_and_edit -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**
```bash
git add beewings/pipeline/box_canvas.py tests/test_pipeline_qt_smoke.py
git -c user.name="BeeWings" -c user.email="nurkal836@gmail.com" commit -m "feat(pipeline): editable box canvas widget"
```

---

## Task 7: Background workers (`workers.py`)

**Files:**
- Create: `beewings/pipeline/workers.py`
- Test: `tests/test_pipeline_qt_smoke.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline_qt_smoke.py`:
```python
def test_crop_worker_runs(qapp, scan_file, tmp_path):
    from beewings.pipeline.project import CropProject
    from beewings.pipeline.workers import CropWorker
    scan_path, meta = scan_file
    proj = CropProject.create(tmp_path / "proj", scans_root=str(scan_path.parent),
                              scan_paths=[scan_path])
    seen = []
    w = CropWorker(proj, do_autodetect=True, do_recrop=True)
    w.progress.connect(lambda i, t, name: seen.append((i, t)))
    w.run()  # run synchronously in-test (QThread.run is a normal method)
    assert proj.scans[0].cropped is True
    assert len(proj.scans[0].wing_boxes) == meta["n_wings"]
    assert seen[-1][0] == seen[-1][1] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline_qt_smoke.py::test_crop_worker_runs -q`
Expected: FAIL — `cannot import name 'CropWorker'`.

- [ ] **Step 3: Implement workers**

Create `beewings/pipeline/workers.py`:
```python
"""QThread workers for the heavy pipeline steps (cropping, landmarks)."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from .landmarks import run_landmarks
from .project import CropProject, auto_detect, recrop


class CropWorker(QThread):
    """Auto-detect boxes and/or recrop every scan in the project."""

    progress = pyqtSignal(int, int, str)   # done, total, name
    failed = pyqtSignal(str, str)          # scan path, error
    finished_ok = pyqtSignal()

    def __init__(self, project: CropProject, do_autodetect: bool, do_recrop: bool,
                 parent=None):
        super().__init__(parent)
        self.project = project
        self.do_autodetect = do_autodetect
        self.do_recrop = do_recrop

    def run(self) -> None:
        total = len(self.project.scans)
        for i, entry in enumerate(self.project.scans, start=1):
            try:
                if self.do_autodetect:
                    auto_detect(entry, self.project.settings)
                if self.do_recrop:
                    recrop(self.project, entry)
            except Exception as exc:  # keep going on the rest
                self.failed.emit(entry.path, f"{type(exc).__name__}: {exc}")
            self.progress.emit(i, total, Path(entry.path).name)
        self.project.save()
        self.finished_ok.emit()


class LandmarkWorker(QThread):
    """Run ML landmarks over every cropped scan's crops folder."""

    progress = pyqtSignal(int, int, str)
    failed = pyqtSignal(str, str)
    finished_ok = pyqtSignal()

    def __init__(self, project: CropProject, parent=None):
        super().__init__(parent)
        self.project = project

    def run(self) -> None:
        cropped = [e for e in self.project.scans if e.cropped]
        for entry in cropped:
            cdir = self.project.crops_dir(entry)
            try:
                run_landmarks(
                    cdir,
                    profile_name=self.project.settings.profile,
                    checkpoint=Path(self.project.settings.checkpoint),
                    progress_cb=lambda i, t, name, e=entry: self.progress.emit(
                        i, t, f"{Path(e.path).stem}: {name}"),
                )
            except Exception as exc:
                self.failed.emit(entry.path, f"{type(exc).__name__}: {exc}")
        self.finished_ok.emit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline_qt_smoke.py::test_crop_worker_runs -q`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add beewings/pipeline/workers.py tests/test_pipeline_qt_smoke.py
git -c user.name="BeeWings" -c user.email="nurkal836@gmail.com" commit -m "feat(pipeline): QThread crop/landmark workers"
```

---

## Task 8: Wizard pages (`pages/*.py`)

**Files:**
- Create: `beewings/pipeline/pages/select_page.py`, `crop_page.py`, `landmark_page.py`, `export_page.py`
- Test: `tests/test_pipeline_qt_smoke.py` (append)

- [ ] **Step 1: Write the failing smoke test**

Append to `tests/test_pipeline_qt_smoke.py`:
```python
def test_pages_construct(qapp, tmp_path):
    from beewings.pipeline.project import CropProject
    from beewings.pipeline.pages.select_page import SelectPage
    from beewings.pipeline.pages.crop_page import CropPage
    from beewings.pipeline.pages.landmark_page import LandmarkPage
    from beewings.pipeline.pages.export_page import ExportPage
    proj = CropProject.create(tmp_path / "proj", scans_root="/s", scan_paths=[])
    ctx = {"project": proj}
    for Page in (SelectPage, CropPage, LandmarkPage, ExportPage):
        w = Page(ctx)
        assert w is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline_qt_smoke.py::test_pages_construct -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the four pages**

Create `beewings/pipeline/pages/select_page.py`:
```python
"""Stage 1: choose the folder of scans."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (QCheckBox, QFileDialog, QLabel, QListWidget,
                             QPushButton, QVBoxLayout, QWidget)

from ..project import CropProject

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class SelectPage(QWidget):
    def __init__(self, ctx: dict, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Этап 1. Выберите папку со сканами:"))
        self.recursive = QCheckBox("Искать во вложенных папках")
        lay.addWidget(self.recursive)
        btn = QPushButton("Выбрать папку…")
        btn.clicked.connect(self._choose)
        lay.addWidget(btn)
        self.list = QListWidget()
        lay.addWidget(self.list)

    def _choose(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Папка со сканами")
        if not folder:
            return
        root = Path(folder)
        globber = root.rglob if self.recursive.isChecked() else root.glob
        skip = ("_crop_", "_label", "_debug")
        scans = sorted(p for p in globber("*")
                       if p.suffix.lower() in IMAGE_EXTS
                       and not any(t in p.stem for t in skip))
        self.list.clear()
        self.list.addItems([str(p) for p in scans])
        self.ctx["scan_paths"] = scans
        self.ctx["scans_root"] = str(root)

    def commit(self) -> bool:
        scans = self.ctx.get("scan_paths") or []
        if not scans:
            return False
        proj: CropProject = self.ctx["project"]
        from ..project import ScanEntry
        proj.scans = [ScanEntry(path=str(p)) for p in scans]
        proj.scans_root = self.ctx["scans_root"]
        proj.stage = "crop"
        proj.save()
        return True
```

Create `beewings/pipeline/pages/crop_page.py`:
```python
"""Stage 2: auto-crop, then review/edit boxes on each scan."""
from __future__ import annotations

from pathlib import Path

import cv2
from PyQt6.QtWidgets import (QHBoxLayout, QListWidget, QProgressBar, QPushButton,
                             QVBoxLayout, QWidget)

from ..box_canvas import BoxEditorCanvas
from ..project import CropProject
from ..workers import CropWorker


class CropPage(QWidget):
    def __init__(self, ctx: dict, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._worker = None
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        self.scan_list = QListWidget()
        self.scan_list.currentRowChanged.connect(self._show_scan)
        left.addWidget(self.scan_list)
        self.auto_btn = QPushButton("Авто-нарезка всех")
        self.auto_btn.clicked.connect(self._auto)
        left.addWidget(self.auto_btn)
        self.recrop_btn = QPushButton("Пересоздать кропы")
        self.recrop_btn.clicked.connect(self._recrop)
        left.addWidget(self.recrop_btn)
        self.progress = QProgressBar()
        left.addWidget(self.progress)
        root.addLayout(left, 0)
        self.canvas = BoxEditorCanvas()
        self.canvas.boxesChanged.connect(self._sync_boxes)
        root.addWidget(self.canvas, 1)

    def enter(self) -> None:
        proj: CropProject = self.ctx["project"]
        self.scan_list.clear()
        self.scan_list.addItems([Path(s.path).name for s in proj.scans])
        if proj.scans:
            self.scan_list.setCurrentRow(0)

    def _show_scan(self, row: int) -> None:
        proj: CropProject = self.ctx["project"]
        if not (0 <= row < len(proj.scans)):
            return
        self._row = row
        entry = proj.scans[row]
        img = cv2.imread(entry.path)
        if img is not None:
            self.canvas.set_scan(img, entry.wing_boxes, entry.label_box)

    def _sync_boxes(self) -> None:
        proj: CropProject = self.ctx["project"]
        row = getattr(self, "_row", None)
        if row is None:
            return
        proj.scans[row].wing_boxes = self.canvas.wing_boxes()
        proj.scans[row].label_box = self.canvas.label_box()
        proj.save()

    def _auto(self) -> None:
        self._run(do_autodetect=True, do_recrop=False, refresh=True)

    def _recrop(self) -> None:
        self._run(do_autodetect=False, do_recrop=True, refresh=False)

    def _run(self, do_autodetect: bool, do_recrop: bool, refresh: bool) -> None:
        proj: CropProject = self.ctx["project"]
        self.progress.setRange(0, len(proj.scans))
        self._worker = CropWorker(proj, do_autodetect, do_recrop)
        self._worker.progress.connect(lambda i, t, n: self.progress.setValue(i))
        if refresh:
            self._worker.finished_ok.connect(lambda: self._show_scan(getattr(self, "_row", 0)))
        self._worker.start()

    def commit(self) -> bool:
        proj: CropProject = self.ctx["project"]
        proj.stage = "landmarks"
        proj.save()
        return True
```

Create `beewings/pipeline/pages/landmark_page.py`:
```python
"""Stage 3: run ML landmarks, then open crops in the existing annotator."""
from __future__ import annotations

from PyQt6.QtWidgets import (QComboBox, QLabel, QProgressBar, QPushButton,
                             QVBoxLayout, QWidget)

from ...core.profiles import PROFILES
from ..project import CropProject
from ..workers import LandmarkWorker


class LandmarkPage(QWidget):
    def __init__(self, ctx: dict, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._worker = None
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Этап 3. Расстановка точек:"))
        self.profile = QComboBox()
        self.profile.addItems(list(PROFILES.keys()))
        lay.addWidget(self.profile)
        self.run_btn = QPushButton("Расставить точки")
        self.run_btn.clicked.connect(self._run)
        lay.addWidget(self.run_btn)
        self.progress = QProgressBar()
        lay.addWidget(self.progress)
        self.open_btn = QPushButton("Открыть в редакторе точек")
        self.open_btn.clicked.connect(self._open_editor)
        lay.addWidget(self.open_btn)

    def enter(self) -> None:
        proj: CropProject = self.ctx["project"]
        idx = self.profile.findText(proj.settings.profile)
        if idx >= 0:
            self.profile.setCurrentIndex(idx)

    def _run(self) -> None:
        proj: CropProject = self.ctx["project"]
        proj.settings.profile = self.profile.currentText()
        prof = PROFILES[proj.settings.profile]
        proj.settings.checkpoint = f"checkpoints/{prof.checkpoint_name}"
        proj.save()
        self.progress.setRange(0, 0)  # busy
        self._worker = LandmarkWorker(proj)
        self._worker.finished_ok.connect(lambda: self.progress.setRange(0, 1) or
                                         self.progress.setValue(1))
        self._worker.start()

    def _open_editor(self) -> None:
        proj: CropProject = self.ctx["project"]
        cb = self.ctx.get("open_in_annotator")
        if cb:
            cb(proj.root / "crops")

    def commit(self) -> bool:
        proj: CropProject = self.ctx["project"]
        proj.stage = "export"
        proj.save()
        return True
```

Create `beewings/pipeline/pages/export_page.py`:
```python
"""Stage 4: export the structured protocol."""
from __future__ import annotations

from PyQt6.QtWidgets import (QCheckBox, QLabel, QMessageBox, QPushButton,
                             QVBoxLayout, QWidget)

from ..export import export_protocol
from ..project import CropProject


class ExportPage(QWidget):
    def __init__(self, ctx: dict, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Этап 4. Экспорт протокола:"))
        self.cb_folders = QCheckBox("Структура папок (крылья + точки)")
        self.cb_summary = QCheckBox("Сводная таблица (CSV + индексы)")
        self.cb_tps = QCheckBox("TPS для морфометрии")
        self.cb_report = QCheckBox("Сводный отчёт")
        for cb in (self.cb_folders, self.cb_summary, self.cb_tps, self.cb_report):
            cb.setChecked(True)
            lay.addWidget(cb)
        btn = QPushButton("Экспортировать")
        btn.clicked.connect(self._export)
        lay.addWidget(btn)

    def enter(self) -> None:
        pass

    def _export(self) -> None:
        proj: CropProject = self.ctx["project"]
        include = set()
        if self.cb_folders.isChecked(): include.add("folders")
        if self.cb_summary.isChecked(): include.add("summary")
        if self.cb_tps.isChecked(): include.add("tps")
        if self.cb_report.isChecked(): include.add("report")
        stats = export_protocol(proj, proj.root / "export", include)
        QMessageBox.information(self, "Готово",
                               f"Экспортировано: {stats['n_wings']} крыльев "
                               f"из {stats['n_scans']} сканов в {proj.root / 'export'}")

    def commit(self) -> bool:
        return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline_qt_smoke.py::test_pages_construct -q`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add beewings/pipeline/pages/ tests/test_pipeline_qt_smoke.py
git -c user.name="BeeWings" -c user.email="nurkal836@gmail.com" commit -m "feat(pipeline): wizard stage pages"
```

---

## Task 9: Wizard shell (`wizard.py`)

**Files:**
- Create: `beewings/pipeline/wizard.py`
- Test: `tests/test_pipeline_qt_smoke.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline_qt_smoke.py`:
```python
def test_wizard_constructs_and_steps(qapp, tmp_path):
    from beewings.pipeline.wizard import PipelineWizard
    w = PipelineWizard(project_root=tmp_path / "proj")
    assert w.stack.count() == 4
    assert w.stack.currentIndex() == 0
    # Cannot advance from stage 1 with no scans selected.
    w._next()
    assert w.stack.currentIndex() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline_qt_smoke.py::test_wizard_constructs_and_steps -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the wizard**

Create `beewings/pipeline/wizard.py`:
```python
"""PipelineWizard: a 4-stage stepper over a CropProject."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QPushButton,
                             QStackedWidget, QVBoxLayout, QWidget)

from .pages.crop_page import CropPage
from .pages.export_page import ExportPage
from .pages.landmark_page import LandmarkPage
from .pages.select_page import SelectPage
from .project import CropProject

STAGES = ["1. Сканы", "2. Нарезка и правка", "3. Точки", "4. Экспорт"]


class PipelineWizard(QWidget):
    def __init__(self, project_root: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BeeWings — Конвейер нарезки")
        self.resize(1100, 720)
        root = Path(project_root) if project_root else Path.cwd() / "beewings_project"
        if (root / "project.json").exists():
            self.project = CropProject.load(root)
        else:
            self.project = CropProject.create(root, scans_root="", scan_paths=[])
            self.project.save()

        self.ctx = {"project": self.project, "open_in_annotator": None}

        outer = QVBoxLayout(self)
        self.step_label = QLabel()
        outer.addWidget(self.step_label)
        self.stack = QStackedWidget()
        self.pages = [SelectPage(self.ctx), CropPage(self.ctx),
                      LandmarkPage(self.ctx), ExportPage(self.ctx)]
        for pg in self.pages:
            self.stack.addWidget(pg)
        outer.addWidget(self.stack, 1)

        nav = QHBoxLayout()
        self.back_btn = QPushButton("◀ Назад")
        self.back_btn.clicked.connect(self._back)
        self.next_btn = QPushButton("Далее ▶")
        self.next_btn.clicked.connect(self._next)
        nav.addWidget(self.back_btn)
        nav.addStretch(1)
        nav.addWidget(self.next_btn)
        outer.addLayout(nav)
        self._update_header()

    def _update_header(self) -> None:
        i = self.stack.currentIndex()
        self.step_label.setText("   →   ".join(
            (f"[{s}]" if n == i else s) for n, s in enumerate(STAGES)))
        self.back_btn.setEnabled(i > 0)
        self.next_btn.setText("Готово" if i == len(self.pages) - 1 else "Далее ▶")

    def _enter_current(self) -> None:
        pg = self.pages[self.stack.currentIndex()]
        if hasattr(pg, "enter"):
            pg.enter()

    def _next(self) -> None:
        pg = self.pages[self.stack.currentIndex()]
        if hasattr(pg, "commit") and not pg.commit():
            return
        if self.stack.currentIndex() < len(self.pages) - 1:
            self.stack.setCurrentIndex(self.stack.currentIndex() + 1)
            self._enter_current()
            self._update_header()

    def _back(self) -> None:
        if self.stack.currentIndex() > 0:
            self.stack.setCurrentIndex(self.stack.currentIndex() - 1)
            self._enter_current()
            self._update_header()


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    w = PipelineWizard()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline_qt_smoke.py -q`
Expected: PASS (all qt smoke tests).

- [ ] **Step 5: Commit**
```bash
git add beewings/pipeline/wizard.py tests/test_pipeline_qt_smoke.py
git -c user.name="BeeWings" -c user.email="nurkal836@gmail.com" commit -m "feat(pipeline): wizard stepper shell + entry point"
```

---

## Task 10: Integrate into the annotator window

**Files:**
- Modify: `beewings/annotator/main_window.py`

- [ ] **Step 1: Add the menu action (manual wiring, verified by launch)**

In `beewings/annotator/main_window.py`, find the File menu block (around line 135-157, after `open_act` is added) and add a pipeline action. Insert after the `file_menu.addAction(open_act)` line:

```python
        pipeline_act = QAction("🧩 Конвейер нарезки…", self)
        pipeline_act.triggered.connect(self._on_open_pipeline)
        file_menu.addAction(pipeline_act)
```

- [ ] **Step 2: Add the handler method**

Add this method to the `MainWindow` class (place it next to `_on_open_folder`, around line 313):

```python
    def _on_open_pipeline(self) -> None:
        from ..pipeline.wizard import PipelineWizard
        self._pipeline = PipelineWizard()
        # Let the wizard hand a finished crops folder back to this annotator.
        self._pipeline.ctx["open_in_annotator"] = self._open_crops_from_pipeline
        self._pipeline.show()

    def _open_crops_from_pipeline(self, crops_root) -> None:
        from pathlib import Path
        crops_root = Path(crops_root)
        # crops_root contains one subfolder per scan; open the first that exists,
        # or the crops_root itself if it directly holds images.
        subdirs = [d for d in crops_root.glob("*") if d.is_dir()]
        target = subdirs[0] if subdirs else crops_root
        self.load_folder(target)
        if hasattr(self, "_pipeline"):
            self._pipeline.raise_()
```

- [ ] **Step 3: Verify import + launch headlessly**

Run:
```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -c "
from PyQt6.QtWidgets import QApplication
app=QApplication([])
from beewings.annotator.main_window import MainWindow
w=MainWindow(); w._on_open_pipeline()
print('pipeline menu + wizard OK')
"
```
Expected: prints `pipeline menu + wizard OK` with no exception.

- [ ] **Step 4: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**
```bash
git add beewings/annotator/main_window.py
git -c user.name="BeeWings" -c user.email="nurkal836@gmail.com" commit -m "feat(annotator): launch pipeline wizard from File menu"
```

---

## Task 11: End-to-end headless verification + docs

**Files:**
- Create: `tests/test_pipeline_e2e.py`
- Modify: `README.md`

- [ ] **Step 1: Write the end-to-end test**

Create `tests/test_pipeline_e2e.py`:
```python
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
```

- [ ] **Step 2: Run the e2e test**

Run: `.venv/bin/python -m pytest tests/test_pipeline_e2e.py -q`
Expected: PASS. (No annotations exist yet, so `n_wings` may be 0 in the report — that's fine; the folders/report still export.)

- [ ] **Step 3: Document the pipeline in README**

In `README.md`, in the CLI table, add after the `beewings-crop` row:
```markdown
| `beewings-pipeline` | Окно-мастер: сканы → нарезка → правка → точки → экспорт |
```
And add a section after the "Авто-нарезка скана" block:
````markdown
### Конвейер в окне (мастер)

Полный рабочий процесс одним мастером — в меню **«🧩 Конвейер нарезки…»**
(или команда `beewings-pipeline`):

1. **Сканы** — выбрать папку с большими сканами.
2. **Нарезка и правка** — авто-нарезка всех; на большом скане можно удалять
   лишние рамки (ПКМ), рисовать недостающие (протяжка ЛКМ), тянуть за углы;
   «Пересоздать кропы» сохраняет крылья.
3. **Точки** — выбрать профиль и «Расставить точки» (ML), затем «Открыть в
   редакторе точек» для ручной правки.
4. **Экспорт** — структура папок + сводная таблица (CSV+индексы) + TPS +
   сводный отчёт в `<проект>/export/`.

Проект (`project.json`) сохраняется в рабочей папке, процесс можно продолжить.
````

- [ ] **Step 4: Run the whole suite once more**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**
```bash
git add tests/test_pipeline_e2e.py README.md
git -c user.name="BeeWings" -c user.email="nurkal836@gmail.com" commit -m "test+docs: pipeline end-to-end verification and README"
```

---

## Self-Review

**Spec coverage:**
- §2 wizard flow / menu → Task 9 (wizard), Task 10 (menu). ✓
- §3 project + manifest → Task 1; crops dir layout → Task 1/2. ✓
- §4 stage 1 select → Task 8; stage 2 auto-crop + box editor + recrop → Tasks 2,6,8; stage 3 ML + open annotator → Tasks 4,8,10; stage 4 export → Tasks 5,8. ✓
- §5 background workers + errors → Task 7 (failed signal, keep going); atomic save → Task 1 (`os.replace`). ✓
- §6 module layout → all modules created; note: added `landmarks.py` (spec folded ML into workers) to keep ML logic Qt-free and testable — documented deviation. ✓
- §7 testing: project/export/geometry unit-tested (Tasks 1-5), box geometry pure (Task 3), qt smoke (Tasks 6-9), e2e headless (Task 11). ✓
- export formats §4: folders/summary/tps/report all in Task 5. ✓

**Deviations flagged:**
- `landmarks.py` added (not in spec's module list) — keeps ML logic Qt-free/testable; workers.py wraps it.
- Confidence is not persisted in `WingAnnotation` (schema has only `uncertain`), so `report.json` reports counts, not mean confidence. Spec §4 mentioned "средняя уверенность"; since the annotation schema doesn't store per-point confidence, the report omits it rather than inventing storage. If mean confidence is required, a follow-up task must extend the schema — flagged, not silently dropped.

**Placeholder scan:** No TBD/TODO; every code step has complete code. Task 5 Step 4 includes a concrete fallback instruction to confirm the real `indices` aggregate function name — this is verification, not a placeholder (full code is present and runs if the name matches).

**Type consistency:** `Box=(x,y,w,h)` consistent across project/geometry/box_canvas. `CropProject.create(root, scans_root, scan_paths)`, `auto_detect(entry, settings)`, `recrop(project, entry)->int`, `run_landmarks(crops_dir, profile_name, checkpoint, ...)->int`, `export_protocol(project, out_dir, include)->dict` — call sites in workers/pages/tests match these signatures. Pages use a shared `ctx` dict with keys `project`, `scan_paths`, `scans_root`, `open_in_annotator`; the wizard seeds `ctx`. Each page exposes `commit()->bool` and optional `enter()`; the wizard calls them. ✓
