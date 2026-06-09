"""Batch subspecies classification by cubital index.

Runs the Alpatov 12pt model on a folder, computes CI for every wing under
all candidate ID-triples, then suggests subspecies based on reference
ranges. Designed for fast bulk screening of a beekeeping yard.

Usage:
    beewings-classify \\
        --checkpoint checkpoints/alpatov12.pt \\
        --images path/to/colony_yard \\
        --out yard_classification.csv

This is a PROVISIONAL classifier — final identification should be
confirmed by an expert based on the per-image overlay JSON.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ..core.indices import (
    CI_CANDIDATES, CI_REFERENCE, DEFAULT_CI,
    classify_by_ci, cubital_index_alpatov,
    discoidal_displacement, hantel_index, pribilski_index, radial_index,
)
from .inference import load, predict


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Batch subspecies classification by CI.")
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Alpatov 12pt model checkpoint")
    p.add_argument("--images", required=True, type=Path, help="Folder with wing images")
    p.add_argument("--out", type=Path, default=Path("classification.csv"))
    p.add_argument("--ci-variant", default=DEFAULT_CI, choices=list(CI_CANDIDATES),
                   help="Which CI formula to use (see indices.py)")
    p.add_argument("--device", default="auto")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--summary", action="store_true",
                   help="Print colony-level subspecies tally")
    args = p.parse_args(argv)

    lm = load(args.checkpoint, device=args.device)
    if lm.n_points != 12:
        print(f"WARNING: checkpoint is {lm.n_points}pt, not Alpatov 12pt — "
              f"results undefined.", file=sys.stderr)

    paths = sorted(p for p in args.images.iterdir()
                   if p.suffix.lower() in IMAGE_EXTS and p.is_file())
    print(f"Classifying {len(paths)} wings using CI variant {args.ci_variant}")

    rows = []
    tally: Counter = Counter()
    for ip in tqdm(paths):
        bgr = cv2.imread(str(ip))
        if bgr is None:
            continue
        pred, conf = predict(lm, bgr, tta=args.tta, return_confidence=True)
        ci_alp = cubital_index_alpatov(pred, variant=args.ci_variant).value
        dsa = discoidal_displacement(pred).value
        ri = radial_index(pred).value
        hantel = hantel_index(pred).value
        pi = pribilski_index(pred).value
        hits = classify_by_ci(ci_alp) if ci_alp is not None else []
        verdict = " / ".join(hits) if hits else "(вне известных диапазонов)"
        mean_conf = float(np.mean(list(conf.values()))) if conf else 0.0
        for hit in hits or ["(unmatched)"]:
            tally[hit] += 1
        rows.append({
            "image": ip.name,
            "CI": f"{ci_alp:.3f}" if ci_alp else "",
            "DsA": f"{dsa:.2f}" if dsa else "",
            "RI": f"{ri:.3f}" if ri else "",
            "Hantel": f"{hantel:.3f}" if hantel else "",
            "PI": f"{pi:.3f}" if pi else "",
            "predicted_subspecies": verdict,
            "confidence_mean": f"{mean_conf:.3f}",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nClassification CSV: {args.out}")

    if args.summary:
        print(f"\n=== Подвиды в выборке ({len(paths)} крыльев) ===")
        for subsp, count in tally.most_common():
            pct = 100 * count / sum(tally.values())
            print(f"  {count:>4}  ({pct:>5.1f}%)  {subsp}")
        # CI distribution stats
        ci_vals = [float(r["CI"]) for r in rows if r["CI"]]
        if ci_vals:
            print(f"\n=== CI ({args.ci_variant}) распределение ===")
            arr = np.array(ci_vals)
            print(f"  mean   = {arr.mean():.3f}")
            print(f"  median = {np.median(arr):.3f}")
            print(f"  std    = {arr.std():.3f}")
            print(f"  range  = [{arr.min():.2f}, {arr.max():.2f}]")
            print(f"\n=== Эталонные диапазоны CI ===")
            for name, (lo, hi) in CI_REFERENCE:
                print(f"  [{lo}, {hi}]  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
