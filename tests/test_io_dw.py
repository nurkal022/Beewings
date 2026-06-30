from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from beewings.core.io_dw import (
    DW_KEYWORD,
    IDENTIFLY_ORDER,
    _landmark_chunk_text,
    export_dw_png,
    read_dw_png,
)
from beewings.core.schema import Landmark, WingAnnotation

SAMPLE = Path(__file__).resolve().parents[1] / "проба" / "11 6.dw.png"


def _full_ann(image="w.png", n=19):
    lms = [Landmark(id=i, x=float(i * 7), y=float(i * 3 + 2)) for i in range(1, n + 1)]
    return WingAnnotation(image=image, image_size=(200, 120),
                          profile="Тофильский 19 точек", landmarks=lms)


def _make_image(path: Path, w=200, h=120):
    Image.fromarray(np.full((h, w, 3), 180, np.uint8), "RGB").save(path)


def test_chunk_text_is_byte_exact_for_19_points():
    coords = [(i, i + 1) for i in range(19)]
    text = _landmark_chunk_text(coords)
    assert text.startswith("landmarks:") and text.endswith(";")
    assert "\n" not in text
    # 10 ("landmarks:") + 38*3 + 37 spaces + 1 (";") == 162, matching real samples.
    assert len(text.encode("latin-1")) == 162


def test_format_matches_real_identifly_sample():
    # The container the real IdentiFly export uses: grayscale + zTXt/IdentiFly.
    if not SAMPLE.exists():
        return
    im = Image.open(SAMPLE)
    assert im.mode == "L"  # 8-bit grayscale
    coords = read_dw_png(SAMPLE)
    assert len(coords) == 19
    assert all(isinstance(x, int) and isinstance(y, int) for x, y in coords)


def test_export_round_trips(tmp_path: Path):
    img = tmp_path / "w.png"
    _make_image(img)
    ann = _full_ann()
    out = tmp_path / "w.dw.png"
    assert export_dw_png(img, ann, out) is True

    coords = read_dw_png(out)
    expected = [(int(round(lm.x)), int(round(lm.y)))
                for lm in sorted(ann.landmarks, key=lambda l: IDENTIFLY_ORDER.index(l.id))]
    assert coords == expected


def test_exported_png_is_grayscale_with_identifly_chunk(tmp_path: Path):
    img = tmp_path / "w.png"
    _make_image(img)
    out = tmp_path / "w.dw.png"
    export_dw_png(img, _full_ann(), out)

    assert Image.open(out).mode == "L"
    # zTXt chunk with our keyword must be present and decompress cleanly.
    data = out.read_bytes()
    off, found = 8, False
    while off + 12 <= len(data):
        (length,) = struct.unpack(">I", data[off:off + 4])
        ctype = data[off + 4:off + 8]
        body = data[off + 8:off + 8 + length]
        if ctype == b"zTXt":
            kw, rest = body.split(b"\x00", 1)
            if kw.decode("latin-1") == DW_KEYWORD:
                found = True
                assert zlib.decompress(rest[1:]).decode("latin-1").startswith("landmarks:")
        off += 12 + length
    assert found


def test_incomplete_annotation_writes_nothing(tmp_path: Path):
    img = tmp_path / "w.png"
    _make_image(img)
    ann = _full_ann()
    ann.landmarks = ann.landmarks[:-1]  # drop one -> not a full 19-point set
    out = tmp_path / "w.dw.png"
    assert export_dw_png(img, ann, out) is False
    assert not out.exists()
