"""Heatmap encoding/decoding for landmark regression.

The model outputs a (K, H, W) heatmap (K = number of landmarks). For each
landmark we put an isotropic Gaussian at the target location during training.
At inference time we decode by taking argmax of each channel and applying a
weighted-centroid refinement in a 3x3 window for sub-pixel precision.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def make_gaussian_heatmap(
    shape_hw: Tuple[int, int],
    centers_xy: np.ndarray,        # (K, 2) in pixel coords of `shape_hw`
    sigma: float = 2.0,
    visible: np.ndarray = None,    # (K,) bool
) -> np.ndarray:
    """Return (K, H, W) float32 with one Gaussian per landmark."""
    K = centers_xy.shape[0]
    H, W = shape_hw
    out = np.zeros((K, H, W), dtype=np.float32)
    radius = int(3 * sigma + 0.5)
    for i in range(K):
        if visible is not None and not visible[i]:
            continue
        cx, cy = float(centers_xy[i, 0]), float(centers_xy[i, 1])
        if not (-radius <= cx < W + radius and -radius <= cy < H + radius):
            continue
        x0, x1 = max(0, int(cx) - radius), min(W, int(cx) + radius + 1)
        y0, y1 = max(0, int(cy) - radius), min(H, int(cy) + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        out[i, y0:y1, x0:x1] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    return out


def decode_heatmap(
    heatmaps: torch.Tensor,        # (B, K, H, W) or (K, H, W)
    return_score: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode peaks to (B, K, 2) sub-pixel locations + (B, K) peak scores.

    Uses argmax + 3x3 weighted centroid refinement around the peak.
    """
    if heatmaps.ndim == 3:
        heatmaps = heatmaps.unsqueeze(0)
    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)
    scores, idx = flat.max(dim=2)
    ys = (idx // W).float()
    xs = (idx % W).float()

    # Sub-pixel refinement via 3x3 weighted centroid.
    coords = torch.stack([xs, ys], dim=-1)   # (B, K, 2)
    # Build padded heatmap for safe indexing.
    pad = torch.nn.functional.pad(heatmaps, (1, 1, 1, 1), mode="replicate")
    yy = ys.long() + 1
    xx = xs.long() + 1
    offsets = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            offsets.append((dy, dx))
    refined = torch.zeros_like(coords)
    weights_sum = torch.zeros(B, K, device=heatmaps.device)
    for dy, dx in offsets:
        # gather the 3x3 neighborhood values
        b_idx = torch.arange(B, device=heatmaps.device).view(B, 1).expand(B, K)
        k_idx = torch.arange(K, device=heatmaps.device).view(1, K).expand(B, K)
        v = pad[b_idx, k_idx, yy + dy, xx + dx]   # (B, K)
        v = torch.clamp(v, min=0)
        refined[..., 0] += v * (xs + dx)
        refined[..., 1] += v * (ys + dy)
        weights_sum += v
    weights_sum = torch.clamp(weights_sum, min=1e-6)
    refined = refined / weights_sum.unsqueeze(-1)
    if return_score:
        return refined, scores
    return refined, None
