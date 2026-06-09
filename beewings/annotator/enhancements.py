"""Lightweight image enhancement applied for display only.

These operations never modify on-disk pixels — the original is preserved.
Output is a uint8 BGR ndarray ready for QImage conversion.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EnhanceParams:
    contrast: float = 1.0     # 1.0 = unchanged, >1 increases contrast
    brightness: int = 0       # additive, [-100, 100]
    gamma: float = 1.0        # 1.0 = unchanged
    clahe: bool = False       # apply CLAHE (adaptive histogram eq.)
    invert: bool = False
    flip_h: bool = False

    def is_identity(self) -> bool:
        return (
            self.contrast == 1.0
            and self.brightness == 0
            and self.gamma == 1.0
            and not self.clahe
            and not self.invert
            and not self.flip_h
        )


def apply(img: np.ndarray, p: EnhanceParams) -> np.ndarray:
    if p.is_identity():
        return img

    out = img
    if p.flip_h:
        out = cv2.flip(out, 1)

    if p.contrast != 1.0 or p.brightness != 0:
        out = cv2.convertScaleAbs(out, alpha=p.contrast, beta=p.brightness)

    if p.gamma != 1.0:
        inv = 1.0 / max(p.gamma, 1e-3)
        table = ((np.arange(256) / 255.0) ** inv * 255).astype(np.uint8)
        out = cv2.LUT(out, table)

    if p.clahe:
        if out.ndim == 3:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
            out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        else:
            out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(out)

    if p.invert:
        out = 255 - out

    return out
