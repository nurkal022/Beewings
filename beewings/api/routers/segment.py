"""POST /segment — split a scan slide into individual wing crops.

output=inline  → boxes + base64 JPEG crops, nothing written to disk.
output=files   → calls segment.process_scan, writes crops to out_dir, returns paths.
"""
from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ...segment.cli import process_scan
from ...segment.crop import crop_wing
from ...segment.layout import reading_order
from ...segment.slide import detect_label_region, estimate_background
from ...segment.wings import find_wings, wing_mask
from ..io import decode_image, read_image, resolve_path
from ..schemas import SegmentJSONRequest, SegmentResponse, WingBox

router = APIRouter()

Box = Tuple[int, int, int, int]


def _resolve_label(img, bg: float, label_frac: Optional[float]):
    if label_frac is not None:
        h, w = img.shape[:2]
        return (0, 0, int(round(label_frac * w)), h)
    return detect_label_region(img, bg)


def _segment_inline(bgr, margin: float, min_area_frac: float,
                    rotate: bool, label_frac: Optional[float]) -> SegmentResponse:
    bg = estimate_background(bgr)
    label = _resolve_label(bgr, bg, label_frac)
    mask = wing_mask(bgr, bg, exclude=label)
    boxes = find_wings(mask, min_area_frac=min_area_frac)
    order = reading_order(boxes)

    out_boxes: List[WingBox] = []
    for n, idx in enumerate(order):
        x, y, w, h = boxes[idx]
        patch = crop_wing(bgr, boxes[idx], margin=margin, rotate=rotate, bg=bg)
        ok, buf = cv2.imencode(".jpg", patch)
        b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
        out_boxes.append(WingBox(index=n, x=int(x), y=int(y), w=int(w), h=int(h),
                                 crop_b64=b64))
    return SegmentResponse(n_wings=len(out_boxes), output="inline", boxes=out_boxes)


def _segment_files(scan_path: Path, out_dir: Path, margin: float,
                   min_area_frac: float, rotate: bool,
                   label_frac: Optional[float]) -> SegmentResponse:
    n = process_scan(scan_path, out_dir, margin=margin, min_area_frac=min_area_frac,
                     rotate=rotate, label_frac=label_frac)
    stem = scan_path.stem
    files = sorted(str(p) for p in out_dir.glob(f"{stem}_crop_*.jpg"))
    return SegmentResponse(n_wings=n, output="files", files=files)


@router.post("/segment", response_model=SegmentResponse)
async def segment(
    request: Request,
    scan: Optional[UploadFile] = File(default=None),
    margin: float = Form(default=0.10),
    min_area_frac: float = Form(default=0.0003),
    rotate: bool = Form(default=False),
    label_frac: Optional[float] = Form(default=None),
    output: str = Form(default="inline"),
    out_dir: Optional[str] = Form(default=None),
) -> SegmentResponse:
    # JSON body (server-side path).
    if scan is None and request.headers.get("content-type", "").startswith("application/json"):
        body = SegmentJSONRequest(**(await request.json()))
        scan_path = resolve_path(body.scan_path)
        if not scan_path.exists():
            raise HTTPException(status_code=400, detail=f"scan not found: {scan_path}")
        if body.output == "files":
            if not body.out_dir:
                raise HTTPException(status_code=400, detail="out_dir is required for output=files")
            return _segment_files(scan_path, resolve_path(body.out_dir), body.margin,
                                  body.min_area_frac, body.rotate, body.label_frac)
        bgr = cv2.imread(str(scan_path))
        if bgr is None:
            raise HTTPException(status_code=400, detail=f"cannot read scan: {scan_path}")
        return _segment_inline(bgr, body.margin, body.min_area_frac, body.rotate, body.label_frac)

    if scan is None:
        raise HTTPException(status_code=400,
                            detail="provide an uploaded 'scan' or a JSON body with scan_path")
    data = await scan.read()

    if output == "files":
        if not out_dir:
            raise HTTPException(status_code=400, detail="out_dir is required for output=files")
        # process_scan reads from disk, so stage the upload to a temp file.
        with tempfile.TemporaryDirectory() as td:
            suffix = Path(scan.filename or "scan.jpg").suffix or ".jpg"
            tmp = Path(td) / f"upload{suffix}"
            tmp.write_bytes(data)
            return _segment_files(tmp, resolve_path(out_dir), margin,
                                  min_area_frac, rotate, label_frac)

    bgr = decode_image(data)
    return _segment_inline(bgr, margin, min_area_frac, rotate, label_frac)
