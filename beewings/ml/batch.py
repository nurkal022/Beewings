"""Process a whole folder of wing images with a trained ML checkpoint.

Saves a WingAnnotation JSON per image into <folder>/annotations/, optionally
also a single CSV aggregating all predictions and a JSON report with
confidence statistics and computed indices.

Usage:
    beewings-ml-batch \\
        --checkpoint checkpoints/tofilski19.pt \\
        --images path/to/wings_folder \\
        [--tta] [--csv all_predictions.csv] [--no-save-json]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

import cv2
from tqdm import tqdm

from ..core.indices import compute_all_alpatov
from ..core.profiles import PROFILES, get_profile
from ..core.schema import Landmark, WingAnnotation, annotation_path, save_annotation
from .inference import load, predict


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _detect_profile_for_n(n: int) -> str:
    """Pick a profile name whose landmark count matches n. Used to label the
    output WingAnnotation consistently."""
    for name, prof in PROFILES.items():
        if len(prof.ids) == n:
            return name
    # fallback
    return f"{n}-point"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Batch-process a folder of wing images.")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--images", required=True, type=Path, help="Folder with .jpg/.png wings")
    p.add_argument("--profile", default=None,
                   help="Profile name override (default: auto from n_points)")
    p.add_argument("--device", default="auto")
    p.add_argument("--tta", action="store_true", help="Test-time augmentation (flip)")
    p.add_argument("--csv", type=Path, default=None,
                   help="Optional aggregated CSV path")
    p.add_argument("--report", type=Path, default=None,
                   help="Optional JSON report with confidence + indices")
    p.add_argument("--no-save-json", action="store_true",
                   help="Skip per-image JSON annotation files")
    p.add_argument("--confidence-min", type=float, default=0.3,
                   help="Skip landmarks whose heatmap peak < this threshold")
    p.add_argument("--include-confidence", action="store_true",
                   help="Include confidence column in --csv output")
    args = p.parse_args(argv)

    lm = load(args.checkpoint, device=args.device)
    print(f"Loaded model: n_points={lm.n_points}  device={lm.device}")

    profile_name = args.profile or _detect_profile_for_n(lm.n_points)
    try:
        profile = get_profile(profile_name)
    except KeyError:
        profile = None

    paths: List[Path] = sorted(
        p for p in args.images.iterdir()
        if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )
    if not paths:
        print(f"No images found in {args.images}", file=sys.stderr)
        return 2
    print(f"Processing {len(paths)} images...")

    csv_rows: List[dict] = []
    report: dict = {
        "checkpoint": str(args.checkpoint),
        "profile": profile_name,
        "n_points": lm.n_points,
        "tta": args.tta,
        "images": {},
    }

    for img_path in tqdm(paths):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            tqdm.write(f"  skip (unreadable): {img_path.name}")
            continue
        h, w = bgr.shape[:2]
        coords, confs = predict(lm, bgr, tta=args.tta, return_confidence=True)

        keep = {lid: xy for lid, xy in coords.items() if confs[lid] >= args.confidence_min}

        if not args.no_save_json:
            ann = WingAnnotation(
                image=img_path.name,
                image_size=(w, h),
                profile=profile_name,
                annotator="ml-batch",
                notes=f"Auto-detected: {len(keep)}/{lm.n_points} landmarks "
                      f"(threshold {args.confidence_min}, tta={args.tta})",
            )
            for lid, (x, y) in keep.items():
                ann.landmarks.append(Landmark(
                    id=lid, x=x, y=y,
                    uncertain=(confs[lid] < 0.5),  # flag low-confidence landmarks
                ))
            ann.landmarks.sort(key=lambda l: l.id)
            save_annotation(ann, annotation_path(img_path, args.images))

        # CSV row
        row = {"image": img_path.name, "image_w": w, "image_h": h,
               "n_predicted": len(keep), "min_confidence": min(confs.values())}
        for lid in range(1, lm.n_points + 1):
            x, y = keep.get(lid, (None, None))
            row[f"x{lid}"] = "" if x is None else f"{x:.3f}"
            row[f"y{lid}"] = "" if y is None else f"{y:.3f}"
            if args.include_confidence:
                row[f"c{lid}"] = f"{confs[lid]:.3f}"
        csv_rows.append(row)

        # Indices (Alpatov only)
        idx_block = None
        if lm.n_points == 12:
            results = compute_all_alpatov(keep)
            idx_block = {ir.name: ir.value for ir in results}

        report["images"][img_path.name] = {
            "n_predicted": len(keep),
            "confidences": confs,
            "indices": idx_block,
        }

    if args.csv:
        if csv_rows:
            with args.csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"Aggregated CSV: {args.csv}")

    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"Report JSON: {args.report}")

    print(f"\nDone. Processed {len(paths)} images.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
