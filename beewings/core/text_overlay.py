"""Render text on cv2 images with Cyrillic support via PIL.

cv2.putText uses Hershey fonts that don't include Cyrillic glyphs (they
get replaced with question marks). This helper uses PIL with a Unicode
TrueType font so Russian text renders correctly.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Reasonable cross-platform candidates. We try them in order.
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",       # Linux Ubuntu/Debian
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",                          # Windows
]


@lru_cache(maxsize=32)
def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for p in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def put_text(img: np.ndarray, text: str, org: Tuple[int, int],
             color: Tuple[int, int, int] = (240, 240, 240),
             size: int = 16,
             stroke_color: Tuple[int, int, int] | None = None,
             stroke_width: int = 0) -> np.ndarray:
    """Draw `text` onto `img` (BGR uint8) at `org` (top-left x, y).

    Modifies and returns img.
    """
    if img.ndim == 2:
        rgb = np.stack([img] * 3, axis=2)
    else:
        rgb = img[..., ::-1]                             # BGR -> RGB
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = _load_font(size)
    rgb_color = (color[2], color[1], color[0])           # BGR -> RGB
    kwargs = {}
    if stroke_color is not None and stroke_width > 0:
        kwargs["stroke_fill"] = (stroke_color[2], stroke_color[1], stroke_color[0])
        kwargs["stroke_width"] = stroke_width
    draw.text(org, text, fill=rgb_color, font=font, **kwargs)
    np.copyto(img, np.array(pil)[..., ::-1])             # RGB -> BGR
    return img


def text_size(text: str, size: int = 16) -> Tuple[int, int]:
    font = _load_font(size)
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]
