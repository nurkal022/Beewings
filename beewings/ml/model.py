"""UNet for landmark heatmap regression.

Plain encoder-decoder with skip connections. 5 down + 5 up. The output is
(K, H/4, W/4) heatmaps so the model produces a coarse map and we recover
sub-pixel precision with the heatmap decoder at inference.

We use GroupNorm instead of BatchNorm — works better with small batches and
mixed image sizes. ReLU activations. Output is raw (no sigmoid); we use MSE
loss against Gaussian targets in [0, 1].
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int) -> nn.GroupNorm:
    """GroupNorm with a group count that always divides `ch`."""
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return nn.GroupNorm(num_groups=g, num_channels=ch)
    return nn.GroupNorm(num_groups=1, num_channels=ch)


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        _gn(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        _gn(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """5-stage UNet. Input 3 channels, output K heatmap channels at 1/4 resolution.

    With input 256x512 -> heatmap output 64x128. Matches dataset.heatmap_hw default.
    """
    def __init__(self, n_landmarks: int, base_ch: int = 32):
        super().__init__()
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 8]
        self.enc1 = _conv_block(3, chs[0])
        self.enc2 = _conv_block(chs[0], chs[1])
        self.enc3 = _conv_block(chs[1], chs[2])
        self.enc4 = _conv_block(chs[2], chs[3])
        self.enc5 = _conv_block(chs[3], chs[4])

        self.up4 = nn.ConvTranspose2d(chs[4], chs[3], 2, stride=2)
        self.dec4 = _conv_block(chs[3] * 2, chs[3])
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec3 = _conv_block(chs[2] * 2, chs[2])
        # No further upsampling: we stop at 1/4 resolution for heatmap output.

        self.head = nn.Conv2d(chs[2], n_landmarks, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        e1 = self.enc1(x)                       # (B, 32, H, W)
        e2 = self.enc2(F.max_pool2d(e1, 2))     # (B, 64, H/2, W/2)
        e3 = self.enc3(F.max_pool2d(e2, 2))     # (B, 128, H/4, W/4)
        e4 = self.enc4(F.max_pool2d(e3, 2))     # (B, 256, H/8, W/8)
        e5 = self.enc5(F.max_pool2d(e4, 2))     # (B, 256, H/16, W/16)

        d4 = self.up4(e5)                       # (B, 256, H/8, W/8)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)                       # (B, 128, H/4, W/4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        return self.head(d3)                    # (B, K, H/4, W/4)
