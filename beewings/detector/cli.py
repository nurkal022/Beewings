"""CLI: run the detector over a folder of wing images and save annotations.

Usage:
    beewings-detect --csv landmarks.csv --images "wings (1)/original"

Steps:
    1. Build (or load) shape model from --csv.
    2. For each image in --images: run detector -> save WingAnnotation JSON
       into <images>/annotations/ so the GUI can open them for review.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import cv2

from ..core.profiles import get_profile
from ..core.schema import Landmark, WingAnnotation, annotation_path, save_annotation
from .pipeline import Detector
from .shape_model import ShapeModel, build_from_csv

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _save_model_or_train(model_path: Path, csv_path: Path, images: Path) -> ShapeModel:
    if model_path.exists():
        return ShapeModel.load(model_path)
    model = build_from_csv(csv_path, images)
    model.save(model_path)
    return model


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run BeeWings classical detector over a folder.")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--csv", type=Path, help="CSV with ground-truth (for training the shape model)")
    p.add_argument("--model", type=Path, default=Path("shape_model.json"),
                   help="Where to store/load the trained shape model")
    p.add_argument("--gate", type=float, default=70.0)
    p.add_argument("--profile", default="19-point")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing annotation files")
    args = p.parse_args(argv)

    if not args.model.exists():
        if not args.csv:
            print("Error: --csv is required to train the model (or pass --model to load existing).",
                  file=sys.stderr)
            return 2
        print(f"Training shape model from {args.csv} ...")
    model = _save_model_or_train(args.model, args.csv or Path("landmarks.csv"), args.images)
    print(f"Loaded shape model with {model.n_specimens} specimens, {len(model.landmark_ids)} landmarks")

    det = Detector(model, gate_radius=args.gate)
    profile = get_profile(args.profile)

    paths: List[Path] = sorted([p for p in args.images.iterdir()
                                if p.suffix.lower() in IMAGE_EXTS and p.is_file()])
    print(f"Detecting on {len(paths)} images ...")
    for img_path in paths:
        out_path = annotation_path(img_path, args.images)
        if out_path.exists() and not args.overwrite:
            print(f"  skip (exists): {img_path.name}")
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  unreadable: {img_path.name}")
            continue
        result = det.detect(bgr, img_path)
        h, w = bgr.shape[:2]
        ann = WingAnnotation(
            image=img_path.name,
            image_size=(w, h),
            profile=profile.name,
            annotator="auto-detector",
            notes=f"Auto-detected via classical pipeline. "
                  f"{len(result.predicted)}/{len(model.landmark_ids)} matched.",
        )
        for lid, (x, y) in result.predicted.items():
            ann.landmarks.append(Landmark(id=lid, x=x, y=y))
        ann.landmarks.sort(key=lambda lm: lm.id)
        save_annotation(ann, out_path)
        print(f"  {img_path.name}: {len(result.predicted)}/{len(model.landmark_ids)} landmarks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
