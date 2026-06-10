from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

CKPT = Path("checkpoints/alpatov12.pt")


@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint not available")
def test_run_landmarks_writes_annotations(tmp_path):
    from beewings.pipeline.landmarks import run_landmarks
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
