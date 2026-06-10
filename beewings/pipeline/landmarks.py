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
