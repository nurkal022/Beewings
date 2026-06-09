"""Training loop for landmark heatmap regression.

Usage:
    beewings-ml-train --csv data/wings19.csv --image-root /path/to/realData \
                      --out runs/unet19 --epochs 80 --batch 16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import WingDataset
from .heatmap import decode_heatmap
from .model import UNet


@dataclass
class TrainConfig:
    csv: str
    image_root: Optional[str]
    out: str
    n_points: int = 19
    input_h: int = 256
    input_w: int = 512
    heatmap_h: int = 64
    heatmap_w: int = 128
    sigma: float = 2.5
    epochs: int = 80
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    base_channels: int = 32
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"        # auto | cuda | mps | cpu
    amp: bool = True
    log_every: int = 50
    save_every_epochs: int = 5


def _pick_device(name: str) -> torch.device:
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if name == "mps" or (name == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def _val_pixel_error(model: nn.Module, loader: DataLoader, device: torch.device,
                     input_hw, heatmap_hw) -> dict:
    model.eval()
    all_err = []
    per_lm = {}
    scale_x = input_hw[1] / heatmap_hw[1]
    scale_y = input_hw[0] / heatmap_hw[0]
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device, non_blocking=True)
            gt_in = batch["keypoints_input"].numpy()         # (B, K, 2) input-px
            vis = batch["visible"].numpy()                    # (B, K)
            pred = model(img)
            coords_hm, _ = decode_heatmap(pred)
            coords_in = coords_hm.detach().cpu().numpy()
            coords_in[..., 0] *= scale_x
            coords_in[..., 1] *= scale_y
            for b in range(coords_in.shape[0]):
                for k in range(coords_in.shape[1]):
                    if vis[b, k] < 0.5:
                        continue
                    d = float(np.hypot(coords_in[b, k, 0] - gt_in[b, k, 0],
                                       coords_in[b, k, 1] - gt_in[b, k, 1]))
                    all_err.append(d)
                    per_lm.setdefault(k + 1, []).append(d)
    out = {
        "mean": float(np.mean(all_err)) if all_err else None,
        "median": float(np.median(all_err)) if all_err else None,
        "p90": float(np.percentile(all_err, 90)) if all_err else None,
        "n": len(all_err),
        "per_landmark_mean": {k: float(np.mean(v)) for k, v in per_lm.items()},
    }
    return out


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = _pick_device(cfg.device)
    print(f"Device: {device}")

    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    input_hw = (cfg.input_h, cfg.input_w)
    heatmap_hw = (cfg.heatmap_h, cfg.heatmap_w)
    image_root = Path(cfg.image_root) if cfg.image_root else None

    train_ds = WingDataset(Path(cfg.csv), "train", cfg.n_points,
                           input_hw=input_hw, heatmap_hw=heatmap_hw,
                           sigma=cfg.sigma, image_root=image_root, augment=True)
    val_ds = WingDataset(Path(cfg.csv), "val", cfg.n_points,
                         input_hw=input_hw, heatmap_hw=heatmap_hw,
                         sigma=cfg.sigma, image_root=image_root, augment=False)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True,
                            persistent_workers=cfg.num_workers > 0)

    model = UNet(n_landmarks=cfg.n_points, base_ch=cfg.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.MSELoss()

    best_med = float("inf")
    log_path = out_dir / "train.log"
    log_path.write_text("")

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}", leave=False)
        for step, batch in enumerate(pbar):
            img = batch["image"].to(device, non_blocking=True)
            target = batch["heatmap"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(img)
                loss = loss_fn(pred, target)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.item()) * img.size(0)
            seen += img.size(0)
            if step % cfg.log_every == 0:
                pbar.set_postfix(loss=f"{running/max(1,seen):.5f}")
        sched.step()
        train_loss = running / max(1, seen)

        val = _val_pixel_error(model, val_loader, device, input_hw, heatmap_hw)
        elapsed = time.time() - t0
        msg = (f"epoch {epoch+1}/{cfg.epochs}  train_loss={train_loss:.5f}  "
               f"val_mean={val['mean']:.2f}px  val_median={val['median']:.2f}px  "
               f"val_p90={val['p90']:.2f}px  elapsed={elapsed:.1f}s")
        print(msg)
        with log_path.open("a") as f:
            f.write(msg + "\n")
            f.write(json.dumps({"epoch": epoch + 1, "train_loss": train_loss, "val": val}) + "\n")

        if val["median"] < best_med:
            best_med = val["median"]
            ckpt = {
                "model": model.state_dict(),
                "config": asdict(cfg),
                "epoch": epoch + 1,
                "val": val,
            }
            torch.save(ckpt, out_dir / "best.pt")
        if (epoch + 1) % cfg.save_every_epochs == 0:
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch + 1},
                       out_dir / f"epoch_{epoch+1:03d}.pt")

    print(f"Done. Best val median: {best_med:.2f}px. Checkpoints in {out_dir}/")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train UNet heatmap landmark regressor.")
    p.add_argument("--csv", required=True)
    p.add_argument("--image-root", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--n-points", type=int, default=19)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=16, dest="batch_size")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4, dest="num_workers")
    p.add_argument("--input-h", type=int, default=256)
    p.add_argument("--input-w", type=int, default=512)
    p.add_argument("--heatmap-h", type=int, default=64)
    p.add_argument("--heatmap-w", type=int, default=128)
    p.add_argument("--sigma", type=float, default=2.5)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_false", dest="amp")
    args = vars(p.parse_args(argv))
    cfg = TrainConfig(**args)
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
