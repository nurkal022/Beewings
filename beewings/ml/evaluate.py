"""Evaluate a checkpoint on val/test split of the CSV.

Outputs per-landmark and overall pixel error in the ORIGINAL image space.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .inference import load, predict


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate ML checkpoint vs CSV ground truth.")
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--image-root", type=Path, default=None)
    p.add_argument("--split", default="val", choices=("train", "val", "test"))
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=None, help="Optional JSON report path")
    args = p.parse_args(argv)

    lm = load(args.checkpoint, device=args.device)
    n_points = lm.n_points

    per_lm: List[List[float]] = [[] for _ in range(n_points)]
    all_err: List[float] = []
    skipped = 0
    rows = []
    with args.csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != args.split:
                continue
            rows.append(r)
    print(f"{args.split}: {len(rows)} samples")

    for r in rows:
        ip = (args.image_root / r["image_path"]) if args.image_root else Path(r["image_path"])
        if not ip.exists():
            skipped += 1
            continue
        bgr = cv2.imread(str(ip))
        if bgr is None:
            skipped += 1
            continue
        pred = predict(lm, bgr)
        for i in range(1, n_points + 1):
            try:
                gx, gy = float(r[f"x{i}"]), float(r[f"y{i}"])
            except KeyError:
                continue
            if i not in pred:
                continue
            px, py = pred[i]
            d = float(np.hypot(px - gx, py - gy))
            per_lm[i - 1].append(d)
            all_err.append(d)

    report = {
        "split": args.split,
        "n_images": len(rows) - skipped,
        "skipped": skipped,
        "overall": {
            "n": len(all_err),
            "mean_px": float(np.mean(all_err)) if all_err else None,
            "median_px": float(np.median(all_err)) if all_err else None,
            "p90_px": float(np.percentile(all_err, 90)) if all_err else None,
        },
        "per_landmark": {},
    }
    for i, errs in enumerate(per_lm, start=1):
        report["per_landmark"][i] = {
            "n": len(errs),
            "mean_px": float(np.mean(errs)) if errs else None,
            "median_px": float(np.median(errs)) if errs else None,
            "p90_px": float(np.percentile(errs, 90)) if errs else None,
        }

    print(f"\n{'ID':>3}  {'n':>4}   {'mean':>6}  {'median':>6}  {'p90':>6}")
    print("-" * 36)
    for i in range(1, n_points + 1):
        s = report["per_landmark"][i]
        if s["mean_px"] is None:
            print(f"L{i:>2}  {s['n']:>4}      —       —       —")
        else:
            print(f"L{i:>2}  {s['n']:>4}   {s['mean_px']:>6.2f}  {s['median_px']:>6.2f}  {s['p90_px']:>6.2f}")
    o = report["overall"]
    if o["mean_px"] is not None:
        print("-" * 36)
        print(f"OVERALL  n={o['n']}  mean={o['mean_px']:.2f}px  median={o['median_px']:.2f}px  p90={o['p90_px']:.2f}px")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
