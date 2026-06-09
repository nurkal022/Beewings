"""Compare detector predictions against ground-truth landmarks.

Usage:
    beewings-eval --csv landmarks.csv --images "wings (1)/original" --out debug_out

Outputs:
    - Per-image, per-landmark pixel error printed and saved to JSON.
    - Aggregate stats (mean / median / 90th-percentile pixel error per landmark id).
    - Overlay PNGs: ground truth in green circles, prediction in red dots.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .pipeline import Detector
from .shape_model import build_from_csv
from .visualize import landmarks_overlay


def _load_truth(csv_path: Path, profile: str = "19-point") -> Dict[str, Dict[int, Tuple[float, float]]]:
    truth: Dict[str, Dict[int, Tuple[float, float]]] = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["profile"] != profile or int(row["skipped"]):
                continue
            truth.setdefault(row["image"], {})[int(row["point_id"])] = (
                float(row["x"]), float(row["y"])
            )
    return truth


def evaluate(csv_path: Path, image_root: Path, out_dir: Path,
             gate_radius: float = 70.0, profile: str = "19-point") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] Training mean shape model from {csv_path} ...")
    model = build_from_csv(csv_path, image_root, profile=profile)
    print(f"      Used {model.n_specimens} specimens, {len(model.landmark_ids)} landmarks")
    det = Detector(model, gate_radius=gate_radius)

    truth = _load_truth(csv_path, profile=profile)
    per_lm_errors: Dict[int, List[float]] = {lid: [] for lid in model.landmark_ids}
    per_lm_missed: Dict[int, int] = {lid: 0 for lid in model.landmark_ids}
    per_image: Dict[str, dict] = {}

    print(f"[2/3] Detecting on {len(truth)} images ...")
    for img_name in sorted(truth):
        img_path = image_root / img_name
        if not img_path.exists():
            continue
        bgr = cv2.imread(str(img_path))
        result = det.detect(bgr, img_path)
        errs: Dict[int, float] = {}
        for lid in model.landmark_ids:
            gt = truth[img_name].get(lid)
            pred = result.predicted.get(lid)
            if gt is None:
                continue
            if pred is None:
                per_lm_missed[lid] += 1
                continue
            e = float(np.hypot(pred[0] - gt[0], pred[1] - gt[1]))
            errs[lid] = e
            per_lm_errors[lid].append(e)

        per_image[img_name] = {
            "errors_px": errs,
            "missed_ids": [lid for lid in model.landmark_ids
                           if truth[img_name].get(lid) is not None and lid not in result.predicted],
            "n_candidates": len(result.skeleton.junctions) + len(result.skeleton.endpoints),
        }

        # Overlay: ground truth green, predicted red with labels.
        overlay = landmarks_overlay(
            bgr,
            predicted=[(lid, xy) for lid, xy in result.predicted.items()],
            ground_truth=[(lid, xy) for lid, xy in truth[img_name].items()],
        )
        cv2.imwrite(str(out_dir / f"{Path(img_name).stem}_eval.jpg"),
                    overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # Aggregate report.
    print(f"[3/3] Aggregating results ...\n")
    report = {"per_landmark": {}, "overall": {}, "per_image": per_image}
    all_errors: List[float] = []
    for lid in model.landmark_ids:
        errs = per_lm_errors[lid]
        all_errors.extend(errs)
        report["per_landmark"][lid] = {
            "n": len(errs),
            "missed": per_lm_missed[lid],
            "mean_px": float(np.mean(errs)) if errs else None,
            "median_px": float(np.median(errs)) if errs else None,
            "p90_px": float(np.percentile(errs, 90)) if errs else None,
            "max_px": float(np.max(errs)) if errs else None,
        }
    if all_errors:
        report["overall"] = {
            "mean_px": float(np.mean(all_errors)),
            "median_px": float(np.median(all_errors)),
            "p90_px": float(np.percentile(all_errors, 90)),
            "n_predictions": len(all_errors),
            "n_missed_total": sum(per_lm_missed.values()),
        }

    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Pretty-print summary.
    print(f"{'ID':>3}  {'n':>3}  {'miss':>4}   {'mean':>6}  {'median':>6}  {'p90':>6}  {'max':>6}   px")
    print("-" * 60)
    for lid in model.landmark_ids:
        st = report["per_landmark"][lid]
        if st["mean_px"] is None:
            print(f"L{lid:>2}  {st['n']:>3}  {st['missed']:>4}      —       —       —       —")
        else:
            print(f"L{lid:>2}  {st['n']:>3}  {st['missed']:>4}   {st['mean_px']:>6.1f}  {st['median_px']:>6.1f}  {st['p90_px']:>6.1f}  {st['max_px']:>6.1f}")
    print("-" * 60)
    o = report["overall"]
    if o:
        print(f"OVERALL  mean={o['mean_px']:.2f}px  median={o['median_px']:.2f}px  "
              f"p90={o['p90_px']:.2f}px  predictions={o['n_predictions']}  missed={o['n_missed_total']}")
    print(f"\nOverlays + evaluation.json written to {out_dir}")
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate detector vs ground-truth CSV.")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True, help="Image folder")
    p.add_argument("--out", type=Path, default=Path("debug_out"))
    p.add_argument("--gate", type=float, default=70.0, help="Max pixel distance for a match")
    p.add_argument("--profile", default="19-point")
    args = p.parse_args(argv)
    evaluate(args.csv, args.images, args.out, gate_radius=args.gate, profile=args.profile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
