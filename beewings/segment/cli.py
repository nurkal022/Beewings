"""beewings-crop: split a scan slide into individual wing images.

Usage:
    beewings-crop --input scan.jpg --out crops/ [--debug]
    beewings-crop --input scan_dir/ --out crops/ --recursive --debug
    beewings-crop --input scan.jpg --inplace
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from .crop import crop_box, crop_wing
from .debug import render_overlay
from .layout import reading_order
from .slide import detect_label_region, estimate_background
from .wings import find_wings, wing_mask

Box = Tuple[int, int, int, int]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _parse_label_box(text: str) -> Box:
    """Parse a "x,y,w,h" string into an int Box (argparse type)."""
    parts = text.replace(" ", "").split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected four comma-separated ints: X,Y,W,H")
    try:
        x, y, w, h = (int(v) for v in parts)
    except ValueError:
        raise argparse.ArgumentTypeError("X,Y,W,H must be integers")
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("W and H must be positive")
    return (x, y, w, h)


def _clamp_box(box: Box, w: int, h: int) -> Optional[Box]:
    """Clamp a box to image bounds; return None if it has no area."""
    x, y, bw, bh = box
    x0 = max(0, min(x, w))
    y0 = max(0, min(y, h))
    x1 = max(0, min(x + bw, w))
    y1 = max(0, min(y + bh, h))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def _resolve_label(
    img: "cv2.typing.MatLike",
    bg: float,
    label_box: Optional[Box],
    label_frac: Optional[float],
) -> Optional[Box]:
    """Pick the label region. A manual `label_box` (pixels) wins; else
    `label_frac` masks the left fraction of the width; else auto-detect."""
    h, w = img.shape[:2]
    if label_box is not None:
        return _clamp_box(label_box, w, h)
    if label_frac is not None:
        return _clamp_box((0, 0, int(round(label_frac * w)), h), w, h)
    return detect_label_region(img, bg)


def process_scan(
    scan_path: Path,
    out_dir: Path,
    margin: float = 0.10,
    min_area_frac: float = 0.0003,
    rotate: bool = False,
    debug: bool = False,
    dry_run: bool = False,
    label_box: Optional[Box] = None,
    label_frac: Optional[float] = None,
) -> int:
    """Crop one scan. Returns the number of wings found. Writes outputs to
    `out_dir` named `<stem>_crop_N.jpg`, `<stem>_label.jpg`, `<stem>_debug.jpg`.

    The label region is auto-detected unless `label_box` (pixels: x,y,w,h) or
    `label_frac` (mask the left fraction of the width) is given — useful for
    diagonal layouts where auto-detection cannot find the separating gap.
    """
    img = cv2.imread(str(scan_path))
    if img is None:
        raise ValueError(f"cannot read image: {scan_path}")

    stem = scan_path.stem
    bg = estimate_background(img)
    label = _resolve_label(img, bg, label_box, label_frac)
    mask = wing_mask(img, bg, exclude=label)
    boxes = find_wings(mask, min_area_frac=min_area_frac)
    order = reading_order(boxes)

    if not dry_run or debug:
        out_dir.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        for n, idx in enumerate(order):
            patch = crop_wing(img, boxes[idx], margin=margin, rotate=rotate, bg=bg)
            cv2.imwrite(str(out_dir / f"{stem}_crop_{n}.jpg"), patch)
        if label is not None:
            cv2.imwrite(str(out_dir / f"{stem}_label.jpg"), crop_box(img, label, 0.02))

    if debug:
        overlay = render_overlay(img, boxes, order, label_box=label)
        cv2.imwrite(str(out_dir / f"{stem}_debug.jpg"), overlay)

    return len(order)


def _iter_scans(root: Path, recursive: bool) -> List[Path]:
    if root.is_file():
        return [root]
    globber = root.rglob if recursive else root.glob
    return sorted(p for p in globber("*") if p.suffix.lower() in IMAGE_EXTS)


def _out_dir_for(scan: Path, input_root: Path, out: Optional[Path], inplace: bool) -> Path:
    if inplace:
        return scan.parent
    if out is None:
        raise ValueError("out directory is required when not inplace")
    if input_root.is_file():
        return out
    # Mirror the sub-folder structure under `out`.
    rel = scan.parent.relative_to(input_root)
    return out / rel


def main() -> int:
    p = argparse.ArgumentParser(description="Split a scan slide into per-wing crops.")
    p.add_argument("--input", type=Path, required=True, help="scan file or folder")
    p.add_argument("--out", type=Path, help="output folder (mirrors input tree)")
    p.add_argument("--inplace", action="store_true", help="write crops next to the scan")
    p.add_argument("--recursive", action="store_true", help="recurse into sub-folders")
    p.add_argument("--margin", type=float, default=0.10)
    p.add_argument("--min-area-frac", type=float, default=0.0003)
    p.add_argument("--rotate-canonical", action="store_true",
                   help="rotate wings to horizontal, base on the left (for ML)")
    p.add_argument("--debug", action="store_true", help="save <stem>_debug.jpg overlay")
    p.add_argument("--dry-run", action="store_true", help="only write debug overlay")
    p.add_argument("--label-box", type=_parse_label_box, default=None,
                   metavar="X,Y,W,H",
                   help="manual label rectangle in pixels (overrides auto-detect)")
    p.add_argument("--label-frac", type=float, default=None, metavar="F",
                   help="mask the left fraction F of the width as label "
                        "(overrides auto-detect; for diagonal layouts)")
    args = p.parse_args()

    if not args.inplace and args.out is None:
        p.error("either --out or --inplace is required")
    if args.label_box is not None and args.label_frac is not None:
        p.error("use either --label-box or --label-frac, not both")

    # Skip files we generate ourselves so re-runs are idempotent.
    skip = ("_crop_", "_label", "_debug")
    scans = [s for s in _iter_scans(args.input, args.recursive)
             if not any(t in s.stem for t in skip)]
    if not scans:
        print(f"no scans found in {args.input}", file=sys.stderr)
        return 1

    total = 0
    for scan in scans:
        out_dir = _out_dir_for(scan, args.input, args.out, args.inplace)
        try:
            n = process_scan(
                scan, out_dir,
                margin=args.margin, min_area_frac=args.min_area_frac,
                rotate=args.rotate_canonical, debug=args.debug, dry_run=args.dry_run,
                label_box=args.label_box, label_frac=args.label_frac,
            )
            print(f"{scan}: {n} wings")
            total += n
        except Exception as exc:  # keep going on the rest of the batch
            print(f"{scan}: ERROR {exc}", file=sys.stderr)

    print(f"done: {total} wings from {len(scans)} scan(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
