"""High-level detector that returns predicted landmarks for one wing image."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .matching import match_landmarks
from .preprocess import WingMask, isolate_wing
from .shape_model import ShapeModel
from .skeleton import VeinSkeleton, skeletonize_and_locate


@dataclass
class Detection:
    image_path: Path
    wing_mask: WingMask
    skeleton: VeinSkeleton
    expected_positions: np.ndarray           # (M, 2) projected mean shape
    landmark_ids: List[int]
    predicted: Dict[int, Tuple[float, float]]  # landmark_id -> (x, y)


class Detector:
    def __init__(self, model: ShapeModel, gate_radius: float = 70.0):
        self.model = model
        self.gate_radius = gate_radius

    def detect(self, bgr: np.ndarray, image_path: Optional[Path] = None) -> Detection:
        wm = isolate_wing(bgr)
        sk = skeletonize_and_locate(wm.gray, wm.mask)
        expected = self.model.project(wm)
        # Pool junctions + endpoints; both can correspond to landmarks (e.g. peripheral
        # landmarks at vein-edge intersections are detected as endpoints).
        candidates: List[Tuple[float, float]] = list(sk.junctions) + list(sk.endpoints)
        predicted = match_landmarks(
            candidates=candidates,
            expected=expected,
            landmark_ids=self.model.landmark_ids,
            gate_radius=self.gate_radius,
        )
        return Detection(
            image_path=image_path or Path(""),
            wing_mask=wm,
            skeleton=sk,
            expected_positions=expected,
            landmark_ids=list(self.model.landmark_ids),
            predicted=predicted,
        )

    def detect_file(self, image_path: Path) -> Detection:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(image_path)
        return self.detect(bgr, image_path)
