"""DrawWing / IdentiFly ``.dw.png`` export.

IdentiFly (and the related DrawWing) stores an annotated wing as an ordinary
PNG image of the wing with the landmark coordinates embedded in a compressed
text chunk. Re-opening the PNG in IdentiFly reads the chunk back and shows the
placed points. We reproduce that exact container so our exports open there.

Container (reverse-engineered from real IdentiFly samples):
    * 8-bit grayscale PNG (PNG colour type 0) of the wing, unscaled.
    * a ``zTXt`` chunk, keyword ``IdentiFly``, zlib-compressed, holding:
          landmarks:x1 y1 x2 y2 ... x19 y19;
      19 integer pairs in the image's own pixel space (origin top-left, NO
      Y-flip), each number formatted ``%3d`` and joined by single spaces, with a
      trailing ``;`` and no newline. For 19 points that is exactly 162 bytes.

Coordinates are taken verbatim from the stored annotation, which is already in
the saved image's pixel space (the same coordinates that render correctly when
the image is reloaded in the annotator).
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .schema import WingAnnotation

# IdentiFly's keyword for the embedded-landmarks text chunk.
DW_KEYWORD = "IdentiFly"

# Order in which landmark IDs are written into the chunk. IdentiFly expects the
# 19 Tofilski landmarks in a fixed sequence; our Tofilski-19 profile uses IDs
# 1..19 in that same sequence. Isolated here so a round-trip mismatch is a
# one-line fix rather than a logic change.
IDENTIFLY_ORDER: Sequence[int] = tuple(range(1, 20))


def _landmark_chunk_text(coords: Sequence[Tuple[int, int]]) -> str:
    """Build the exact ``landmarks:...;`` payload IdentiFly writes."""
    flat: List[int] = [v for xy in coords for v in xy]
    return "landmarks:" + " ".join("%3d" % v for v in flat) + ";"


def _ordered_int_coords(
    ann: WingAnnotation, order: Sequence[int]
) -> Optional[List[Tuple[int, int]]]:
    """Return integer (x, y) for each id in ``order``, or None if any missing."""
    by_id = {lm.id: lm for lm in ann.landmarks}
    out: List[Tuple[int, int]] = []
    for lm_id in order:
        lm = by_id.get(lm_id)
        if lm is None or lm.skipped or lm.x < 0 or lm.y < 0:
            return None
        out.append((int(round(lm.x)), int(round(lm.y))))
    return out


def export_dw_png(
    image_path: Path,
    ann: WingAnnotation,
    out_path: Path,
    order: Sequence[int] = IDENTIFLY_ORDER,
) -> bool:
    """Write an IdentiFly ``.dw.png`` for one annotated wing.

    Embeds the wing image as 8-bit grayscale plus the landmark chunk. Requires
    every id in ``order`` to be present (IdentiFly files are always full sets);
    if any is missing or skipped, writes nothing and returns ``False``.
    """
    coords = _ordered_int_coords(ann, order)
    if coords is None:
        return False

    img = Image.open(image_path).convert("L")  # 8-bit grayscale, PNG colortype 0
    meta = PngInfo()
    meta.add_text(DW_KEYWORD, _landmark_chunk_text(coords), zip=True)  # zTXt chunk
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG", pnginfo=meta)
    return True


def read_dw_png(path: Path) -> List[Tuple[int, int]]:
    """Read embedded landmarks back from a ``.dw.png`` (round-trip / re-import).

    Returns the list of (x, y) integer pairs, or [] if no IdentiFly chunk.
    """
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return []
    off = 8
    while off + 12 <= len(data):
        (length,) = struct.unpack(">I", data[off : off + 4])
        ctype = data[off + 4 : off + 8]
        body = data[off + 8 : off + 8 + length]
        if ctype == b"zTXt":
            keyword, rest = body.split(b"\x00", 1)
            if keyword.decode("latin-1") == DW_KEYWORD:
                # rest = <compression method byte><compressed datastream>
                text = zlib.decompress(rest[1:]).decode("latin-1")
                nums = [int(v) for v in text.split("landmarks:")[1].rstrip(";").split()]
                return list(zip(nums[0::2], nums[1::2]))
        off += 12 + length
        if ctype == b"IEND":
            break
    return []
