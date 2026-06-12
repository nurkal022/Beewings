"""Unicode-path-safe OpenCV image IO.

cv2.imread / cv2.imwrite use the ANSI file API on Windows and silently fail on
paths containing non-ASCII characters (e.g. Cyrillic folder names), returning
None / False. These wrappers go through numpy file IO + imdecode / imencode,
which handle Unicode paths on every platform.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path, flags: int = cv2.IMREAD_COLOR):
    """Read an image; returns an ndarray (BGR by default), or None on failure."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path, img) -> bool:
    """Write an image, inferring the format from the file extension. Returns success."""
    path = Path(path)
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        buf.tofile(str(path))
    except OSError:
        return False
    return True
