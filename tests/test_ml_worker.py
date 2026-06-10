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
