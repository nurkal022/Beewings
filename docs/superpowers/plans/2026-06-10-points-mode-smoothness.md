# Points-Mode Smoothness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Steps use `- [ ]`.

**Goal:** Stop the UI freeze on ML detect (run it on a background thread) and stop the side-panel "deformation" on methodology switch (wrap the panel in a scroll area).

**Architecture:** New `MLDetectWorker(QThread)` does model load + predict off the main thread; `AnnotatorWidget._run_ml_detect` becomes start→`_on_ml_done`/`_on_ml_failed`. `SidePanel` wraps its content in a `QScrollArea`.

**Tech Stack:** PyQt6, torch, numpy, opencv, pytest.

---

## Task 1: Background ML worker (`ml_worker.py`)

**Files:** Create `beewings/annotator/ml_worker.py`, `tests/test_ml_worker.py`.

- [ ] **Step 1: Write failing test** — `tests/test_ml_worker.py`:
```python
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

CKPT = Path("checkpoints/alpatov12.pt")


@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint not available")
def test_ml_worker_predicts(qapp):
    from beewings.annotator.ml_worker import MLDetectWorker
    bgr = np.full((300, 600, 3), 230, np.uint8)
    cv2.ellipse(bgr, (300, 150), (200, 70), 0, 0, 360, (120, 120, 120), -1)
    w = MLDetectWorker(CKPT, bgr, tta=False, expected_n=12, allow_alpatov8=False)
    out = {}
    w.done.connect(lambda pred, conf, model: out.update(pred=pred, conf=conf, model=model))
    w.failed.connect(lambda msg: out.update(err=msg))
    w.run()  # run synchronously in-test
    assert "pred" in out and len(out["pred"]) == 12
    assert out["model"] is not None


@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint not available")
def test_ml_worker_schema_mismatch(qapp):
    from beewings.annotator.ml_worker import MLDetectWorker
    bgr = np.full((100, 100, 3), 230, np.uint8)
    # alpatov12 checkpoint but we claim to expect 19 -> mismatch
    w = MLDetectWorker(CKPT, bgr, tta=False, expected_n=19, allow_alpatov8=False)
    out = {}
    w.failed.connect(lambda msg: out.update(err=msg))
    w.done.connect(lambda *a: out.update(done=True))
    w.run()
    assert "err" in out and "done" not in out
```
- [ ] **Step 2:** Run → FAIL: `.venv/bin/python -m pytest tests/test_ml_worker.py -q`
- [ ] **Step 3: Implement** `beewings/annotator/ml_worker.py`:
```python
"""Background thread for ML landmark detection (keeps the UI responsive)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class MLDetectWorker(QThread):
    """Load the checkpoint (if needed), validate its schema, and predict — all
    off the GUI thread. Emits done(pred, conf, model) or failed(message)."""

    done = pyqtSignal(object, object, object)   # pred dict, conf dict, model
    failed = pyqtSignal(str)

    def __init__(self, checkpoint: Path, bgr: np.ndarray, tta: bool,
                 expected_n: int, allow_alpatov8: bool, model=None, parent=None):
        super().__init__(parent)
        self.checkpoint = Path(checkpoint)
        self.bgr = bgr
        self.tta = tta
        self.expected_n = expected_n
        self.allow_alpatov8 = allow_alpatov8
        self.model = model

    def run(self) -> None:
        try:
            model = self.model
            if model is None:
                # Schema check before committing to a full load.
                import torch
                peek = torch.load(self.checkpoint, map_location="cpu",
                                  weights_only=False)
                ckpt_n = peek["config"]["n_points"]
                if ckpt_n != self.expected_n and not (
                    self.allow_alpatov8 and ckpt_n == 12 and self.expected_n == 8
                ):
                    self.failed.emit(
                        f"Несовпадение схемы: checkpoint {self.checkpoint.name} "
                        f"обучен на {ckpt_n} точек, а профиль ожидает "
                        f"{self.expected_n}. Это разные методики.")
                    return
                from ..ml.inference import load as load_ml
                model = load_ml(self.checkpoint, device="auto")
            from ..ml.inference import predict as ml_predict
            pred, conf = ml_predict(model, self.bgr, tta=self.tta,
                                    return_confidence=True)
            self.done.emit(pred, conf, model)
        except Exception as ex:  # surface to the UI thread
            self.failed.emit(f"{type(ex).__name__}: {ex}")
```
- [ ] **Step 4:** Run → PASS (or skip if no checkpoint): `.venv/bin/python -m pytest tests/test_ml_worker.py -q`
- [ ] **Step 5:** Commit:
```bash
git add beewings/annotator/ml_worker.py tests/test_ml_worker.py
git -c user.name="BeeWings" -c user.email="nurkal836@gmail.com" commit -m "feat(annotator): background ML detect worker"
```

---

## Task 2: Async `_run_ml_detect` in AnnotatorWidget

**Files:** Modify `beewings/annotator/annotator_widget.py`.

Context: the current `_run_ml_detect` (around lines 733–848) does, on the main thread: validation → checkpoint resolution → schema peek (torch.load) → model load → image read → predict → apply landmarks → refresh/save/status. Move the heavy part (peek + load + predict) to `MLDetectWorker`; keep validation + checkpoint resolution + image read on the main thread; apply results in a `done` callback.

- [ ] **Step 1: Replace the whole `_run_ml_detect` method** with the start-only version below. Find the method `def _run_ml_detect(self) -> None:` and replace its ENTIRE body (everything until the next method `def _run_auto_detect`) with:
```python
    def _run_ml_detect(self) -> None:
        """Run the trained UNet detector on the current image (off the GUI thread)."""
        if sip.isdeleted(self):
            return
        if self._current_ann is None or self._current_image_path is None or self._project_dir is None:
            return
        if getattr(self, "_ml_worker", None) is not None and self._ml_worker.isRunning():
            return

        ckpt_filename = self._profile.checkpoint_name or "tofilski19.pt"
        env_ckpt = os.environ.get("BEEWINGS_ML_CHECKPOINT", "").strip()
        ckpt_candidates: list[Path] = []
        if env_ckpt:
            ckpt_candidates.append(Path(env_ckpt))
        ckpt_candidates.extend([
            self._project_dir / "checkpoints" / ckpt_filename,
            Path("checkpoints") / ckpt_filename,
            Path("checkpoints/best.pt") if ckpt_filename == "tofilski19.pt" else Path(""),
            Path("runs/unet19_v1/best.pt") if ckpt_filename == "tofilski19.pt" else Path(""),
            Path("runs/unet12_v1/best.pt") if ckpt_filename == "alpatov12.pt" else Path(""),
        ])
        ckpt = next((p for p in ckpt_candidates if str(p) and p.is_file()), None)
        if ckpt is None:
            QMessageBox.warning(
                self, "Модель не найдена",
                f"Для профиля «{self._profile.name}» нужен checkpoint {ckpt_filename}.\n\n"
                f"Положи файл в:\n  • checkpoints/{ckpt_filename}\n"
                f"  • <папка_проекта>/checkpoints/{ckpt_filename}\n"
                f"  • или BEEWINGS_ML_CHECKPOINT env var")
            return
        try:
            from .ml_worker import MLDetectWorker
        except ImportError as ex:
            QMessageBox.critical(self, "ML не установлен", f"{ex}")
            return

        # Read the image bytes on the GUI thread (cheap), heavy work goes to the worker.
        with self._current_image_path.open("rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        expected_n = len(self._profile.ids)
        allow_alpatov8 = (self._profile.methodology_id == "alpatov" and expected_n == 8)
        cached = self._ml_model if self._ml_ckpt_path == str(ckpt) else None

        self.side.ml_btn.setEnabled(False)
        self.status.emit("Определяю точки (в фоне)…")
        self._ml_ckpt_pending = str(ckpt)
        self._ml_worker = MLDetectWorker(
            ckpt, bgr, tta=True, expected_n=expected_n,
            allow_alpatov8=allow_alpatov8, model=cached)
        self._ml_worker.done.connect(self._on_ml_done)
        self._ml_worker.failed.connect(self._on_ml_failed)
        self._ml_worker.start()

    def _on_ml_failed(self, msg: str) -> None:
        if sip.isdeleted(self):
            return
        self.side.ml_btn.setEnabled(True)
        title = "Несовпадение схемы" if msg.startswith("Несовпадение") else "Ошибка ML-детектора"
        QMessageBox.critical(self, title, msg)

    def _on_ml_done(self, pred, conf, model) -> None:
        if sip.isdeleted(self):
            return
        self._ml_model = model
        self._ml_ckpt_path = getattr(self, "_ml_ckpt_pending", None)
        self.side.ml_btn.setEnabled(True)
        if self._current_ann is None:
            return
        self._last_ml_confidences = conf
        active_ids = set(self._profile.ids)
        self._undo.push(self._current_ann.landmarks)
        applied = 0
        low_conf_ids: list = []
        for lid, (x, y) in pred.items():
            if lid not in active_ids:
                continue
            self._current_ann.set_landmark(lid, x, y)
            if conf.get(lid, 1.0) < 0.45:
                lm = self._current_ann.get_landmark(lid)
                if lm is not None:
                    lm.uncertain = True
                low_conf_ids.append(lid)
            applied += 1
        self.canvas.refresh_landmarks()
        self.canvas.set_current_id(self._current_lm_id)
        self.side.refresh(self._current_ann)
        self._save_current()
        active_conf = [conf[lid] for lid in active_ids if lid in conf]
        mean_conf = sum(active_conf) / max(1, len(active_conf))
        msg = f"ML: {applied}/{len(active_ids)} точек, средняя уверенность {mean_conf:.2f}"
        if low_conf_ids:
            msg += f" — проверь точки {low_conf_ids}"
        self.status.emit(msg)
```
- [ ] **Step 2: Initialise the new attributes.** In `AnnotatorWidget.__init__`, near the existing `self._ml_model = None` / `self._ml_ckpt_path = None` lines, add:
```python
        self._ml_worker = None
        self._ml_ckpt_pending = None
```
- [ ] **Step 3: Verify no syntax/import breakage + full suite**
Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass.
Run headless launch:
```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -c "
from PyQt6.QtWidgets import QApplication; app=QApplication([])
from beewings.annotator.annotator_widget import AnnotatorWidget; AnnotatorWidget(); print('ok')"
```
Expected: `ok`.
- [ ] **Step 4: Behavioural check (async path, real checkpoint)** — run:
```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python - <<'PY'
import cv2, numpy as np
from pathlib import Path
from PyQt6.QtWidgets import QApplication
app = QApplication([])
from beewings.annotator.annotator_widget import AnnotatorWidget
d = Path("/tmp/mlw"); d.mkdir(exist_ok=True)
img = np.full((300,600,3),230,np.uint8); cv2.ellipse(img,(300,150),(200,70),0,0,360,(120,120,120),-1)
cv2.imwrite(str(d/"w.jpg"), img)
w = AnnotatorWidget(); w.load_folder(d)
w._run_ml_detect()                     # returns immediately (worker started)
assert w._ml_worker is not None
w._ml_worker.wait(60000)               # block test until worker finishes
app.processEvents()                    # deliver done signal
print("ml_btn re-enabled:", w.side.ml_btn.isEnabled())
print("OK async")
PY
```
Expected: prints `OK async` and `ml_btn re-enabled: True`. (If no checkpoint, the worker emits failed and the button is also re-enabled — also acceptable.)
- [ ] **Step 5: Commit**
```bash
git add beewings/annotator/annotator_widget.py
git -c user.name="BeeWings" -c user.email="nurkal836@gmail.com" commit -m "feat(annotator): async ML detect (no UI freeze) via background worker"
```

---

## Task 3: Scrollable side panel (`side_panel.py`)

**Files:** Modify `beewings/annotator/side_panel.py`; Test `tests/test_app_qt_smoke.py` (append).

- [ ] **Step 1: Append failing test** to `tests/test_app_qt_smoke.py`:
```python
def test_side_panel_is_scrollable(qapp):
    from PyQt6.QtWidgets import QScrollArea
    from beewings.annotator.side_panel import SidePanel
    sp = SidePanel()
    assert sp.findChild(QScrollArea) is not None
    # methodology switch must not raise and list stays usable
    sp.set_methodology("alpatov", "Алпатов 12 точек")
    sp.set_methodology("tofilski", "Тофильский 19 точек")
    assert sp.list is not None
```
- [ ] **Step 2:** Run → FAIL: `.venv/bin/python -m pytest tests/test_app_qt_smoke.py::test_side_panel_is_scrollable -q`
- [ ] **Step 3: Wrap the panel content in a QScrollArea.**
  (a) Add `QScrollArea` to the `from PyQt6.QtWidgets import (...)` block in `side_panel.py`.
  (b) In `SidePanel.__init__`, replace the three opening lines:
```python
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
```
  with:
```python
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self._scroll)
        content = QWidget()
        self._scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
```
  (The rest of `__init__` keeps using `layout.addWidget(...)`, now adding to the scroll's inner content widget.)
  (c) Give the methodology info label a stable height so switching methodology text doesn't jump the layout. Find:
```python
        self.methodology_info = QLabel("")
        self.methodology_info.setWordWrap(True)
```
  and add right after:
```python
        self.methodology_info.setMinimumHeight(46)
```
- [ ] **Step 4:** Run → PASS: `.venv/bin/python -m pytest tests/test_app_qt_smoke.py::test_side_panel_is_scrollable -q`
- [ ] **Step 5: Full suite + headless app**
Run: `.venv/bin/python -m pytest tests/ -q` → all pass.
```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -c "
from PyQt6.QtWidgets import QApplication; app=QApplication([])
from beewings.app.app import AppController; AppController(); print('ok')"
```
- [ ] **Step 6: Commit**
```bash
git add beewings/annotator/side_panel.py tests/test_app_qt_smoke.py
git -c user.name="BeeWings" -c user.email="nurkal836@gmail.com" commit -m "feat(side_panel): scrollable panel + stable info height (no deform on methodology switch)"
```

---

## Self-Review

**Spec coverage:** async ML worker (§2 → Task 1,2); scroll panel + stable info height (§3 → Task 3); methodology switching preserved (Task 3 test exercises it); crash guards untouched. ✓
**Placeholder scan:** full code in steps. ✓
**Type/interface consistency:** `MLDetectWorker(checkpoint, bgr, tta, expected_n, allow_alpatov8, model=None)` with `done(pred, conf, model)`/`failed(str)` (Task 1) is exactly how `_run_ml_detect` constructs/handles it (Task 2). New attrs `_ml_worker`, `_ml_ckpt_pending` initialised in `__init__` (Task 2 Step 2) and used in the slots. `SidePanel.ml_btn` exists (toggled enabled in the async flow). ✓
**Risk:** torch model object created in the worker thread and reused on subsequent calls in another worker thread — inference-only; the existing `LandmarkWorker` already runs torch in a QThread, so this pattern is already used in the codebase. The `done`/`failed` callbacks are guarded by `sip.isdeleted(self)`.
