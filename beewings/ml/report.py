"""Generate paper-ready evaluation figures and tables for a trained model.

Produces:
    - eval_table.csv     — per-landmark error stats (n, mean, median, p90, max)
    - per_landmark.png   — bar chart of mean+median+p90 errors per landmark
    - error_hist.png     — overall error distribution histogram
    - mean_shape.png     — landmark positions (mean across val set) overlaid
                           on a representative wing
    - summary.json       — machine-readable summary
    - sample_overlays/   — 20 random val images with GT (green) + pred (red)

Usage:
    beewings-ml-report \\
        --csv data/wings19.csv \\
        --checkpoint checkpoints/tofilski19.pt \\
        --image-root data/wings19_images \\
        --out reports/tofilski19
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
from tqdm import tqdm

from ..core.text_overlay import put_text
from .inference import load, predict


def _bar_chart_bytes(per_lm: List[dict], out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = [d["id"] for d in per_lm]
    mean = [d["mean_px"] for d in per_lm]
    median = [d["median_px"] for d in per_lm]
    p90 = [d["p90_px"] for d in per_lm]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(ids))
    width = 0.27
    ax.bar(x - width, median, width, label="median", color="#3a7d44")
    ax.bar(x,         mean,   width, label="mean",   color="#5da9e9")
    ax.bar(x + width, p90,    width, label="p90",    color="#e09f3e")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in ids])
    ax.set_ylabel("pixel error")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _histogram(all_err: List[float], out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(all_err, bins=80, color="#5da9e9", edgecolor="white")
    ax.axvline(np.median(all_err), color="#3a7d44", lw=1.5, label=f"median={np.median(all_err):.2f}")
    ax.axvline(np.mean(all_err), color="#e09f3e", lw=1.5, label=f"mean={np.mean(all_err):.2f}")
    ax.set_xlabel("pixel error")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Paper-ready evaluation report.")
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--image-root", type=Path, default=None)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--split", default="val", choices=("train", "val", "test"))
    p.add_argument("--tta", action="store_true")
    p.add_argument("--n-overlays", type=int, default=20)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    sample_dir = args.out / "sample_overlays"
    sample_dir.mkdir(exist_ok=True)

    lm = load(args.checkpoint, device=args.device)
    K = lm.n_points

    rows = [r for r in csv.DictReader(args.csv.open(encoding="utf-8"))
            if r["split"] == args.split]
    print(f"{args.split}: {len(rows)} samples")

    per_lm = [[] for _ in range(K)]
    all_err: List[float] = []
    confidences: List[float] = []

    import random
    random.seed(0)
    overlay_indices = set(random.sample(range(len(rows)), min(args.n_overlays, len(rows))))

    for i, r in enumerate(tqdm(rows)):
        ip = (args.image_root / r["image_path"]) if args.image_root else Path(r["image_path"])
        if not ip.exists():
            continue
        bgr = cv2.imread(str(ip))
        if bgr is None:
            continue
        pred, conf = predict(lm, bgr, tta=args.tta, return_confidence=True)

        # Per-landmark error
        for lid in range(1, K + 1):
            try:
                gx, gy = float(r[f"x{lid}"]), float(r[f"y{lid}"])
            except KeyError:
                continue
            if lid not in pred:
                continue
            px, py = pred[lid]
            d = float(np.hypot(px - gx, py - gy))
            per_lm[lid - 1].append(d)
            all_err.append(d)
            confidences.append(conf[lid])

        if i in overlay_indices:
            img = bgr.copy()
            h, w = img.shape[:2]
            cv2.rectangle(img, (0, 0), (w, 30), (40, 40, 40), -1)
            put_text(img, f"{ip.name}  GT=green pred=red", (8, 4), size=14)
            for lid in range(1, K + 1):
                try:
                    gx, gy = float(r[f"x{lid}"]), float(r[f"y{lid}"])
                    cv2.circle(img, (int(gx), int(gy)), 6, (0, 200, 0), 2)
                except KeyError:
                    pass
            for lid, (px, py) in pred.items():
                cv2.circle(img, (int(px), int(py)), 3, (0, 0, 220), -1)
            cv2.imwrite(str(sample_dir / f"{i:04d}_{ip.name}"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])

    # Build stats
    table = []
    for i in range(K):
        e = np.array(per_lm[i]) if per_lm[i] else np.array([])
        if len(e) == 0:
            table.append({"id": i + 1, "n": 0,
                          "mean_px": None, "median_px": None,
                          "p90_px": None, "max_px": None})
        else:
            table.append({
                "id": i + 1, "n": len(e),
                "mean_px": float(e.mean()),
                "median_px": float(np.median(e)),
                "p90_px": float(np.percentile(e, 90)),
                "max_px": float(e.max()),
            })

    # Save eval_table.csv
    with (args.out / "eval_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "n", "mean_px", "median_px", "p90_px", "max_px"])
        writer.writeheader()
        for t in table:
            writer.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in t.items()})

    # Overall summary
    summary = {
        "checkpoint": str(args.checkpoint),
        "n_points": K,
        "split": args.split,
        "n_samples": len(rows),
        "n_predictions": len(all_err),
        "overall_mean_px": float(np.mean(all_err)),
        "overall_median_px": float(np.median(all_err)),
        "overall_p90_px": float(np.percentile(all_err, 90)),
        "overall_p99_px": float(np.percentile(all_err, 99)),
        "overall_max_px": float(np.max(all_err)),
        "mean_confidence": float(np.mean(confidences)),
        "min_confidence": float(np.min(confidences)),
        "tta": args.tta,
        "per_landmark": table,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                           encoding="utf-8")

    # Figures
    title = f"{Path(args.checkpoint).stem} — {K}pt — {args.split}"
    _bar_chart_bytes(table, args.out / "per_landmark.png", title)
    _histogram(all_err, args.out / "error_hist.png", title)

    print(f"\n=== {Path(args.checkpoint).stem} {args.split} ({len(all_err)} preds) ===")
    print(f"  mean   = {summary['overall_mean_px']:.2f} px")
    print(f"  median = {summary['overall_median_px']:.2f} px")
    print(f"  p90    = {summary['overall_p90_px']:.2f} px")
    print(f"  p99    = {summary['overall_p99_px']:.2f} px")
    print(f"  mean confidence = {summary['mean_confidence']:.3f}")
    print(f"\nReport saved to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
