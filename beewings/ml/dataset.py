"""PyTorch Dataset for wing landmark regression.

CSV schema produced by prepare.py:
    image_path, image_w, image_h, n_points, split, x1, y1, ..., xN, yN
We pad/resize images to a fixed input size (default 512x256 — wings are
horizontally elongated). All landmark coordinates are mapped through the
same transform so heatmap targets match the resized image.

Augmentations: small random rotation (±10°), brightness/contrast jitter,
random crop+resize jitter, blur. No horizontal flip — wings have no
left/right anatomical mirror, so flipping would permute landmark identities.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .heatmap import make_gaussian_heatmap


def _build_aug(input_hw: Tuple[int, int], train: bool) -> A.Compose:
    H, W = input_hw
    if train:
        return A.Compose([
            A.LongestMaxSize(max_size=max(H, W) + 64, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=H + 64, min_width=W + 64, border_mode=cv2.BORDER_CONSTANT,
                          fill=255),
            A.Affine(translate_percent={"x": (-0.04, 0.04), "y": (-0.04, 0.04)},
                     scale=(0.92, 1.08),
                     rotate=(-10, 10),
                     interpolation=cv2.INTER_LINEAR,
                     border_mode=cv2.BORDER_CONSTANT,
                     fill=255,
                     p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
            A.Blur(blur_limit=3, p=0.15),
            A.CenterCrop(height=H, width=W),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=max(H, W), interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=H, min_width=W, border_mode=cv2.BORDER_CONSTANT, fill=255),
            A.CenterCrop(height=H, width=W),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))


class WingDataset(Dataset):
    def __init__(self, csv_path: Path, split: str, n_points: int,
                 input_hw: Tuple[int, int] = (256, 512),
                 heatmap_hw: Tuple[int, int] = (64, 128),
                 sigma: float = 2.5,
                 image_root: Path = None,
                 augment: bool = None):
        super().__init__()
        self.csv_path = csv_path
        self.split = split
        self.n_points = n_points
        self.input_hw = input_hw
        self.heatmap_hw = heatmap_hw
        self.sigma = sigma
        self.image_root = image_root
        self.augment = (split == "train") if augment is None else augment

        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.rows = [r for r in reader if r["split"] == split]
        if not self.rows:
            raise RuntimeError(f"No rows for split={split} in {csv_path}")

        self.aug = _build_aug(input_hw, train=self.augment)

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_image(self, p: str) -> Path:
        if self.image_root:
            return self.image_root / p
        return Path(p)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img_path = self._resolve_image(row["image_path"])
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Source-frame landmarks
        kps: List[Tuple[float, float]] = []
        for i in range(1, self.n_points + 1):
            kps.append((float(row[f"x{i}"]), float(row[f"y{i}"])))

        out = self.aug(image=rgb, keypoints=kps)
        img_t = out["image"]                                # already normalized
        kps_t = np.array(out["keypoints"], dtype=np.float32)  # (K, 2) in input pixel coords

        # Scale to heatmap pixel coords.
        in_h, in_w = self.input_hw
        hm_h, hm_w = self.heatmap_hw
        scale_x = hm_w / in_w
        scale_y = hm_h / in_h
        kps_hm = kps_t.copy()
        kps_hm[:, 0] *= scale_x
        kps_hm[:, 1] *= scale_y

        # Visibility: drop keypoints that fell out of the input crop.
        vis = (
            (kps_t[:, 0] >= 0) & (kps_t[:, 0] < in_w) &
            (kps_t[:, 1] >= 0) & (kps_t[:, 1] < in_h)
        )

        heatmap = make_gaussian_heatmap((hm_h, hm_w), kps_hm, sigma=self.sigma, visible=vis)

        img_chw = torch.from_numpy(np.transpose(img_t, (2, 0, 1))).float()
        return {
            "image": img_chw,
            "heatmap": torch.from_numpy(heatmap).float(),
            "keypoints_input": torch.from_numpy(kps_t).float(),    # in input image px
            "keypoints_heatmap": torch.from_numpy(kps_hm).float(), # in heatmap px
            "visible": torch.from_numpy(vis.astype(np.float32)),
            "image_path": str(img_path),
        }
