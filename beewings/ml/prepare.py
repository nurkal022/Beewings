"""Walk a tree of TPS files (Rohlf format) and produce a unified training CSV.

Each output row corresponds to one wing image with its landmark coordinates.
Y is FLIPPED on read because TPS uses math-style bottom-up Y but we work
in image-style top-down Y everywhere downstream.

Output CSV schema:
    image_path,  # absolute path or relative-to-root
    n_points,    # 12 or 19 (or whatever the source declared)
    split,       # train / val / test, deterministic by image_path hash
    x1, y1, x2, y2, ..., xN, yN   # N == n_points

We never modify the input. We just build an index.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2


@dataclass
class TpsRecord:
    tps_file: Path
    image_path: Path
    n_points: int
    coords: List[Tuple[float, float]]   # already y-flipped to image coords
    image_size: Tuple[int, int]         # (w, h)


_LM_RE = re.compile(r"^LM=(\d+)", re.MULTILINE)
_IMAGE_RE = re.compile(r"^IMAGE=(.+)$", re.MULTILINE)


def _split_blocks(text: str) -> Iterator[str]:
    """Yield individual LM=N ... blocks from a TPS file."""
    parts = re.split(r"(?=^LM=)", text, flags=re.MULTILINE)
    for p in parts:
        if p.strip().startswith("LM="):
            yield p


def _parse_block(block: str) -> Optional[Tuple[int, List[Tuple[float, float]], str]]:
    m_lm = _LM_RE.match(block)
    if not m_lm:
        return None
    n = int(m_lm.group(1))
    if n <= 0:
        return None
    lines = block.splitlines()
    coords: List[Tuple[float, float]] = []
    for ln in lines[1:n + 1]:
        try:
            x_s, y_s = ln.strip().split()
            coords.append((float(x_s), float(y_s)))
        except ValueError:
            return None
    if len(coords) != n:
        return None
    m_img = _IMAGE_RE.search(block)
    if not m_img:
        return None
    return n, coords, m_img.group(1).strip()


def _resolve_image(tps_path: Path, image_field: str) -> Optional[Path]:
    """The IMAGE= field is usually a Windows path. We assume the JPG lives
    next to the TPS file under its basename."""
    basename = image_field.replace("\\", "/").split("/")[-1]
    candidate = tps_path.parent / basename
    if candidate.exists():
        return candidate
    # Some datasets place images in adjacent directories — fall back to a
    # case-insensitive scan of the TPS directory.
    target_lower = basename.lower()
    for p in tps_path.parent.iterdir():
        if p.is_file() and p.name.lower() == target_lower:
            return p
    return None


def parse_tree(root: Path, only_n: Optional[int] = None,
               verbose: bool = True) -> List[TpsRecord]:
    """Recurse through `root`, parse every .tps, return resolved records."""
    records: List[TpsRecord] = []
    tps_files = sorted(root.rglob("*.tps"))
    if verbose:
        print(f"Found {len(tps_files)} TPS files under {root}")

    stats = Counter()
    no_image = 0
    bad_size = 0
    for tps in tps_files:
        try:
            text = tps.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for block in _split_blocks(text):
            parsed = _parse_block(block)
            if parsed is None:
                stats["malformed"] += 1
                continue
            n, coords, image_field = parsed
            if only_n is not None and n != only_n:
                stats[f"skip_n={n}"] += 1
                continue
            img_path = _resolve_image(tps, image_field)
            if img_path is None:
                no_image += 1
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                bad_size += 1
                continue
            h, w = img.shape[:2]
            # Y-flip: TPS uses bottom-up convention.
            coords_flipped = [(x, h - y) for x, y in coords]
            # Sanity: drop points clearly outside the image (allow small slop).
            in_bounds = sum(1 for x, y in coords_flipped if -10 <= x <= w + 10 and -10 <= y <= h + 10)
            if in_bounds < len(coords_flipped):
                stats["partial_oob"] += 1
            records.append(TpsRecord(
                tps_file=tps,
                image_path=img_path,
                n_points=n,
                coords=coords_flipped,
                image_size=(w, h),
            ))
            stats[f"kept_n={n}"] += 1
    if verbose:
        print(f"  no image found: {no_image}")
        print(f"  unreadable image: {bad_size}")
        print(f"  block stats: {dict(stats)}")
        print(f"  records returned: {len(records)}")
    return records


def _deterministic_split(key: str, val_frac: float, test_frac: float) -> str:
    """Hash a split key -> deterministic train/val/test bucket."""
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    if bucket < test_frac:
        return "test"
    if bucket < test_frac + val_frac:
        return "val"
    return "train"


def _split_key(image_path: Path, split_by: str) -> str:
    """Key used for the deterministic split.

    "image"  — per-image (legacy): wings of the same colony may straddle
               train/val, inflating validation metrics because sibling wings
               are near-identical.
    "family" — per parent folder (colony/apiary): all wings of one colony land
               in the same split. Honest generalization estimate.
    """
    if split_by == "family":
        return str(image_path.parent)
    return str(image_path)


def write_csv(records: List[TpsRecord], out_csv: Path, n_points: int,
              val_frac: float = 0.1, test_frac: float = 0.05,
              relative_to: Optional[Path] = None,
              split_by: str = "image") -> Dict[str, int]:
    """Write filtered records (only those matching n_points) to CSV."""
    rows = []
    splits = Counter()
    rel_abs = relative_to.resolve() if relative_to else None
    for r in records:
        if r.n_points != n_points:
            continue
        abs_img = r.image_path.resolve()
        if rel_abs is not None:
            try:
                path_str = str(abs_img.relative_to(rel_abs))
            except ValueError:
                path_str = str(abs_img)
        else:
            path_str = str(abs_img)
        split = _deterministic_split(_split_key(r.image_path, split_by), val_frac, test_frac)
        splits[split] += 1
        row = {
            "image_path": path_str,
            "image_w": r.image_size[0],
            "image_h": r.image_size[1],
            "n_points": r.n_points,
            "split": split,
        }
        for i, (x, y) in enumerate(r.coords, start=1):
            row[f"x{i}"] = f"{x:.3f}"
            row[f"y{i}"] = f"{y:.3f}"
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No records with n_points={n_points} to write.")
    fieldnames = (["image_path", "image_w", "image_h", "n_points", "split"]
                  + [f"x{i}" for i in range(1, n_points + 1)
                     for _ in (0,)]
                  + [f"y{i}" for i in range(1, n_points + 1)
                     for _ in (0,)])
    # build correct alternating fieldnames
    fieldnames = ["image_path", "image_w", "image_h", "n_points", "split"]
    for i in range(1, n_points + 1):
        fieldnames += [f"x{i}", f"y{i}"]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return dict(splits)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Prepare unified training CSV from a TPS tree.")
    p.add_argument("--root", type=Path, required=True, help="Root of the TPS tree (e.g. realData/)")
    p.add_argument("--out", type=Path, required=True, help="Output CSV path")
    p.add_argument("--n-points", type=int, default=19, choices=(8, 12, 19))
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.05)
    p.add_argument("--relative-to", type=Path, default=None,
                   help="Make image_path relative to this directory (good for cross-machine training).")
    p.add_argument("--split-by", choices=("image", "family"), default="image",
                   help="'family' keeps all wings of one colony/apiary folder in the same "
                        "train/val/test split (avoids leakage); 'image' is the legacy per-image split.")
    args = p.parse_args(argv)

    records = parse_tree(args.root, only_n=args.n_points)
    splits = write_csv(records, args.out, n_points=args.n_points,
                       val_frac=args.val_frac, test_frac=args.test_frac,
                       relative_to=args.relative_to, split_by=args.split_by)
    print(f"\nWrote {sum(splits.values())} rows to {args.out}")
    print(f"Split: {splits}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
