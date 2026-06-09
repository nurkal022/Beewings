"""COCO keypoints export for downstream ML training.

We treat the whole wing as a single instance ("wing") of one category.
Each landmark id maps to a keypoint slot; visibility flag follows COCO:
    0 = not labeled (skipped/missing)
    1 = labeled but not visible (uncertain)
    2 = labeled and visible
Bbox is the tight box around placed landmarks (with small padding),
since we don't have an instance segmentation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from PIL import Image

from .profiles import get_profile
from .schema import WingAnnotation


def export_coco(
    annotations: Iterable[WingAnnotation],
    out_path: Path,
    image_root: Path,
    profile_name: str,
    padding: int = 20,
) -> None:
    profile = get_profile(profile_name)
    keypoint_names = [lm.label for lm in profile.landmarks]
    skeleton: List[List[int]] = []  # left empty — biologist can fill later

    images = []
    annotations_out = []
    next_ann_id = 1

    for img_id, ann in enumerate(annotations, start=1):
        img_path = image_root / ann.image
        if img_path.exists():
            w, h = ann.image_size
        else:
            w, h = ann.image_size

        images.append({
            "id": img_id,
            "file_name": ann.image,
            "width": w,
            "height": h,
        })

        flat: List[float] = []
        xs, ys = [], []
        n_visible = 0
        for spec in profile.landmarks:
            lm = ann.get_landmark(spec.id)
            if lm is None or lm.skipped or lm.x < 0 or lm.y < 0:
                flat.extend([0.0, 0.0, 0])
                continue
            v = 1 if lm.uncertain else 2
            flat.extend([float(lm.x), float(lm.y), v])
            xs.append(lm.x)
            ys.append(lm.y)
            n_visible += 1

        if xs:
            x0, x1 = max(0, min(xs) - padding), min(w, max(xs) + padding)
            y0, y1 = max(0, min(ys) - padding), min(h, max(ys) + padding)
            bbox = [x0, y0, x1 - x0, y1 - y0]
            area = (x1 - x0) * (y1 - y0)
        else:
            bbox = [0, 0, w, h]
            area = w * h

        annotations_out.append({
            "id": next_ann_id,
            "image_id": img_id,
            "category_id": 1,
            "keypoints": flat,
            "num_keypoints": n_visible,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
        })
        next_ann_id += 1

    coco = {
        "info": {"description": "BeeWings landmark annotations", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations_out,
        "categories": [
            {
                "id": 1,
                "name": "wing",
                "supercategory": "bee",
                "keypoints": keypoint_names,
                "skeleton": skeleton,
            }
        ],
    }
    out_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
