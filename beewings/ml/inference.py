"""Run a trained UNet on a single image and return image-space landmark coords."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch

from .heatmap import decode_heatmap
from .model import UNet


@dataclass
class LoadedModel:
    model: UNet
    device: torch.device
    input_hw: Tuple[int, int]
    heatmap_hw: Tuple[int, int]
    n_points: int


def load(checkpoint_path: Path, device: str = "auto") -> LoadedModel:
    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        dev = torch.device("cuda")
    elif device == "mps" or (device == "auto" and torch.backends.mps.is_available()):
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    cfg = ckpt["config"]
    model = UNet(n_landmarks=cfg["n_points"], base_ch=cfg["base_channels"]).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return LoadedModel(
        model=model, device=dev,
        input_hw=(cfg["input_h"], cfg["input_w"]),
        heatmap_hw=(cfg["heatmap_h"], cfg["heatmap_w"]),
        n_points=cfg["n_points"],
    )


def _preprocess(bgr: np.ndarray, input_hw: Tuple[int, int]) -> Tuple[torch.Tensor, dict]:
    """Resize+pad to input_hw, return tensor and inverse mapping params."""
    H, W = input_hw
    h0, w0 = bgr.shape[:2]
    s = min(W / w0, H / h0)
    nw, nh = int(round(w0 * s)), int(round(h0 * s))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    padded = np.full((H, W, 3), 255, dtype=np.uint8)
    ox, oy = (W - nw) // 2, (H - nh) // 2
    padded[oy:oy + nh, ox:ox + nw] = resized
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    t = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).float()
    return t, {"scale": s, "ox": ox, "oy": oy, "w0": w0, "h0": h0}


def predict(lm: LoadedModel, bgr: np.ndarray,
            tta: bool = False,
            return_confidence: bool = False
            ):
    """Run a single image through the model.

    Args:
        tta: if True, average predictions from the original image plus a
             horizontally-flipped variant (then unflip) — usually ~5-10%
             accuracy boost at the cost of 2x compute. Note: flip-aug
             assumes the underlying biological landmarks are left/right
             symmetric in their image-space role. For bee wings this is
             true since both wings are mirror images.
        return_confidence: if True, also return per-landmark confidence
             score (peak amplitude in the heatmap, in [0, 1]).

    Returns:
        dict {id: (x, y)} (in original image pixel coords),
        or  ({id: (x, y)}, {id: confidence})  if return_confidence.
    """
    img_t, meta = _preprocess(bgr, lm.input_hw)
    img_t = img_t.to(lm.device, non_blocking=True)
    with torch.no_grad():
        pred = lm.model(img_t)                                # (1, K, h, w)
        if tta:
            # Flip input horizontally, predict, flip heatmap back. We do NOT
            # permute landmark IDs because here flipping is just an image
            # augmentation (the wing's structure is symmetric in the model's
            # learned representation).
            pred_flip = lm.model(torch.flip(img_t, dims=[3]))
            pred_flip = torch.flip(pred_flip, dims=[3])
            pred = 0.5 * (pred + pred_flip)

    coords_hm, scores = decode_heatmap(pred)                  # (1, K, 2), (1, K)
    coords = coords_hm[0].cpu().numpy()
    conf = scores[0].cpu().numpy()

    scale_x = lm.input_hw[1] / lm.heatmap_hw[1]
    scale_y = lm.input_hw[0] / lm.heatmap_hw[0]
    coords[:, 0] *= scale_x
    coords[:, 1] *= scale_y
    coords[:, 0] -= meta["ox"]
    coords[:, 1] -= meta["oy"]
    coords[:, 0] /= meta["scale"]
    coords[:, 1] /= meta["scale"]
    out: Dict[int, Tuple[float, float]] = {}
    out_conf: Dict[int, float] = {}
    for i, ((x, y), c) in enumerate(zip(coords, conf), start=1):
        out[i] = (float(x), float(y))
        out_conf[i] = float(c)
    if return_confidence:
        return out, out_conf
    return out


def predict_file(checkpoint_path: Path, image_path: Path,
                 device: str = "auto") -> Dict[int, Tuple[float, float]]:
    lm = load(checkpoint_path, device=device)
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(image_path)
    return predict(lm, bgr)
