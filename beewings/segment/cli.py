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
from typing import List, Optional

import cv2

from .crop import crop_box, crop_wing
from .debug import render_overlay
from .layout import reading_order
from .slide import detect_label_region, estimate_background
from .wings import find_wings, wing_mask

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def process_scan(
    scan_path: Path,
    out_dir: Path,
    margin: float = 0.10,
    min_area_frac: float = 0.0003,
    rotate: bool = False,
    debug: bool = False,
    dry_run: bool = False,
) -> int:
    """Crop one scan. Returns the number of wings found. Writes outputs to
    `out_dir` named `<stem>_crop_N.jpg`, `<stem>_label.jpg`, `<stem>_debug.jpg`.
    """
    img = cv2.imread(str(scan_path))
    if img is None:
        raise ValueError(f"cannot read image: {scan_path}")

    stem = scan_path.stem
    bg = estimate_background(img)
    label = detect_label_region(img, bg)
    mask = wing_mask(img, bg, exclude=label)
    boxes = find_wings(mask, min_area_frac=min_area_frac)
    order = reading_order(boxes)

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
    assert out is not None
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
    args = p.parse_args()

    if not args.inplace and args.out is None:
        p.error("either --out or --inplace is required")

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
            )
            print(f"{scan}: {n} wings")
            total += n
        except Exception as exc:  # keep going on the rest of the batch
            print(f"{scan}: ERROR {exc}", file=sys.stderr)

    print(f"done: {total} wings from {len(scans)} scan(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
