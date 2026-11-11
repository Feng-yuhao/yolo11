"""DySample upsampler for the E5a existing-method control.

Algorithm: Liu et al., "Learning to Upsample by Learning to Sample", ICCV 2023.
Reference implementation: https://github.com/tiny-smart/dysample

E5a intentionally uses the standard low-resolution-prediction (``lp``),
four-group, no-dynamic-scope configuration.  This file does not contain the
proposed density-constrained extension planned for a later experiment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    """Lightweight learned point-sampling upsampler."""

    def __init__(self, channels, scale=2, style="lp", groups=4, dyscope=False):
        super().__init__()
        self.channels = int(channels)
        self.scale = int(scale)
        self.style = str(style)
        self.groups = int(groups)
        self.dyscope = bool(dyscope)
        if self.scale < 2:
            raise ValueError("DySample scale must be at least 2")
        if self.style not in {"lp", "pl"}:
            raise ValueError("DySample style must be 'lp' or 'pl'")
        if self.channels < self.groups or self.channels % self.groups:
            raise ValueError("DySample channels must be divisible by groups")
        if self.style == "pl" and (
            self.channels < self.scale**2 or self.channels % self.scale**2
        ):
            raise ValueError("Pixel-first DySample needs channels divisible by scale^2")

        predictor_channels = self.channels // self.scale**2 if self.style == "pl" else self.channels
        offset_channels = 2 * self.groups if self.style == "pl" else 2 * self.groups * self.scale**2
        self.offset = nn.Conv2d(predictor_channels, offset_channels, 1)
        nn.init.normal_(self.offset.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.offset.bias)
        if self.dyscope:
            self.scope = nn.Conv2d(predictor_channels, offset_channels, 1, bias=False)
            nn.init.zeros_(self.scope.weight)
        self.register_buffer("init_pos", self._initial_positions(), persistent=True)

    def _initial_positions(self):
        start = (-self.scale + 1) / (2 * self.scale)
        stop = (self.scale - 1) / (2 * self.scale)
        axis = torch.linspace(start, stop, steps=self.scale)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        base = torch.stack((xx, yy), dim=0)
        return base.unsqueeze(1).repeat(1, self.groups, 1, 1).reshape(1, -1, 1, 1)

    def _sample(self, x, offset):
        batch, _, height, width = offset.shape
        offset = offset.view(batch, 2, -1, height, width)
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=x.dtype, device=x.device) + 0.5,
            torch.arange(width, dtype=x.dtype, device=x.device) + 0.5,
            indexing="ij",
        )
        coords = torch.stack((xx, yy), dim=0).view(1, 2, 1, height, width)
        normalizer = x.new_tensor((width, height)).view(1, 2, 1, 1, 1)
        coords = 2.0 * (coords + offset) / normalizer - 1.0
        coords = F.pixel_shuffle(coords.reshape(batch, -1, height, width), self.scale)
        coords = coords.view(
            batch, 2, -1, self.scale * height, self.scale * width
        ).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        grouped = x.reshape(batch * self.groups, -1, height, width)
        sampled = F.grid_sample(
            grouped, coords, mode="bilinear", padding_mode="border", align_corners=False
        )
        return sampled.view(batch, -1, self.scale * height, self.scale * width)

    def _forward_lp(self, x):
        raw = self.offset(x)
        if self.dyscope:
            raw = raw * self.scope(x).sigmoid() * 0.5
        else:
            raw = raw * 0.25
        return self._sample(x, raw + self.init_pos.to(dtype=x.dtype))

    def _forward_pl(self, x):
        shuffled = F.pixel_shuffle(x, self.scale)
        raw = self.offset(shuffled)
        if self.dyscope:
            raw = raw * self.scope(shuffled).sigmoid() * 0.5
        else:
            raw = raw * 0.25
        raw = F.pixel_unshuffle(raw, self.scale)
        return self._sample(x, raw + self.init_pos.to(dtype=x.dtype))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"DySample expected BCHW with {self.channels} channels, got {tuple(x.shape)}"
            )
        return self._forward_pl(x) if self.style == "pl" else self._forward_lp(x)


__all__ = ("DySample",)
