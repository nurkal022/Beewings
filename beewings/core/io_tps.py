"""TPS export — Rohlf's geometric morphometrics format.

Format (one specimen):
    LM=<n>
    x1 y1
    x2 y2
    ...
    IMAGE=filename.jpg
    ID=specimen_id

Skipped / missing landmarks are written as TPS convention (-1.000000 -1.000000)
but most downstream tools (MorphoJ) treat any LM=n as fully present, so we
only include actually placed landmarks and downsize LM accordingly. Specimens
with incomplete sets are still emitted (caller's responsibility to filter).

Y-axis convention: TPS expects image-coordinate Y flipped (origin bottom-left).
We invert using image height so output is compatible with tpsDig/tpsRelw.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .schema import WingAnnotation


def export_tps(annotations: Iterable[WingAnnotation], out_path: Path, flip_y: bool = True) -> None:
    lines: list[str] = []
    for idx, ann in enumerate(annotations, start=1):
        placed = [lm for lm in ann.landmarks if not lm.skipped and lm.x >= 0 and lm.y >= 0]
        if not placed:
            continue
        h = ann.image_size[1]
        lines.append(f"LM={len(placed)}")
        for lm in placed:
            y = (h - lm.y) if flip_y else lm.y
            lines.append(f"{lm.x:.5f} {y:.5f}")
        lines.append(f"IMAGE={ann.image}")
        lines.append(f"ID={idx}")
        lines.append("")  # blank separator
    out_path.write_text("\n".join(lines), encoding="utf-8")
