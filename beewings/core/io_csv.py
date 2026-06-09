"""CSV export: one row per landmark per image."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .schema import WingAnnotation


def export_csv(annotations: Iterable[WingAnnotation], out_path: Path) -> None:
    rows = []
    for ann in annotations:
        for lm in ann.landmarks:
            rows.append({
                "image": ann.image,
                "profile": ann.profile,
                "point_id": lm.id,
                "x": f"{lm.x:.3f}",
                "y": f"{lm.y:.3f}",
                "uncertain": int(lm.uncertain),
                "skipped": int(lm.skipped),
                "image_width": ann.image_size[0],
                "image_height": ann.image_size[1],
                "flipped": int(ann.flipped),
                "annotator": ann.annotator,
            })
    with out_path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            f.write("image,profile,point_id,x,y,uncertain,skipped,image_width,image_height,flipped,annotator\n")
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
