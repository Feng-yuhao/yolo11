"""P3-guided adaptive low-pass filtering for VisDrone P2 features.

PGALP uses the fused high-resolution P2 feature as the detection feature and
the raw P3 feature only as a semantic guide.  It predicts a foreground
reliability map and per-location weights over fixed Gaussian 3/5/7 low-pass
filters.  Smoothing is applied mainly where foreground reliability is low.
No ground-truth mask or annotation is used by the module at train or test
time.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNAct(nn.Sequential):
    def __init__(self, c1, c2, kernel=1, padding=0, groups=1):
        super().__init__(
            nn.Conv2d(c1, c2, kernel, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )


def _gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    radius = kernel_size // 2
    coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.view(1, 1, kernel_size, kernel_size)


class PGALP(nn.Module):
    """P3-guided adaptive Gaussian low-pass module.

    Args:
        input_channels: ``[P2 channels, P3 channels]``, injected by parse_model.
        hidden_ratio: Channel ratio used by the lightweight routing network.
        kernels: Odd Gaussian kernel sizes.
        sigmas: Gaussian standard deviations corresponding to ``kernels``.
        layer_scale_init: Initial per-channel residual strength.
        reliability_margin: Keeps reliability away from exact 0/1 saturation.
    """

    def __init__(
        self,
        input_channels,
        hidden_ratio=0.25,
        kernels=(3, 5, 7),
        sigmas=(0.8, 1.2, 1.8),
        layer_scale_init=0.1,
        reliability_margin=0.05,
    ):
        super().__init__()
        if not isinstance(input_channels, (list, tuple)) or len(input_channels) != 2:
            raise ValueError("PGALP input_channels must be [P2 channels, P3 channels]")
        self.p2_channels, self.p3_channels = (int(v) for v in input_channels)
        self.hidden_ratio = float(hidden_ratio)
        self.kernels = tuple(int(v) for v in kernels)
        self.sigmas = tuple(float(v) for v in sigmas)
        self.reliability_margin = float(reliability_margin)
        if self.p2_channels < 8 or self.p3_channels < 8:
            raise ValueError("PGALP requires at least 8 channels per input")
        if self.hidden_ratio <= 0:
            raise ValueError("PGALP hidden_ratio must be positive")
        if len(self.kernels) < 2 or len(self.kernels) != len(self.sigmas):
            raise ValueError("PGALP kernels/sigmas must have the same length >= 2")
        if any(k < 3 or k % 2 == 0 for k in self.kernels):
            raise ValueError("PGALP Gaussian kernels must be odd and >= 3")
        if any(s <= 0 for s in self.sigmas):
            raise ValueError("PGALP Gaussian sigmas must be positive")
        if not 0.0 <= self.reliability_margin < 0.5:
            raise ValueError("PGALP reliability_margin must be in [0, 0.5)")
        if not 0.0 < float(layer_scale_init) <= 1.0:
            raise ValueError("PGALP layer_scale_init must be in (0, 1]")

        hidden = max(16, int(round(self.p2_channels * self.hidden_ratio / 8.0)) * 8)
        self.hidden = hidden
        self.p2_reduce = _ConvBNAct(self.p2_channels, hidden)
        self.p3_reduce = _ConvBNAct(self.p3_channels, hidden)
        guide_channels = 2 * hidden + 1
        self.router = nn.Sequential(
            _ConvBNAct(guide_channels, guide_channels, 3, padding=1, groups=guide_channels),
            _ConvBNAct(guide_channels, hidden),
            nn.Conv2d(hidden, 1 + len(self.kernels), 1, bias=True),
        )
        self.layer_scale = nn.Parameter(
            torch.full((1, self.p2_channels, 1, 1), float(layer_scale_init))
        )
        # Initial routing is neutral: reliability=0.5 and uniform filter weights.
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

        for index, (kernel, sigma) in enumerate(zip(self.kernels, self.sigmas)):
            self.register_buffer(
                f"gaussian_kernel_{index}", _gaussian_kernel(kernel, sigma), persistent=True
            )

    def _blur(self, x: torch.Tensor, index: int) -> torch.Tensor:
        kernel_size = self.kernels[index]
        kernel = getattr(self, f"gaussian_kernel_{index}").to(dtype=x.dtype, device=x.device)
        kernel = kernel.expand(self.p2_channels, 1, kernel_size, kernel_size)
        padding = kernel_size // 2
        return F.conv2d(F.pad(x, (padding,) * 4, mode="reflect"), kernel, groups=self.p2_channels)

    def routing(self, p2: torch.Tensor, p3: torch.Tensor, low_small: torch.Tensor):
        p3_aligned = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
        # One explicit high-frequency cue helps distinguish texture-rich locations.
        high_energy = (p2 - low_small).abs().mean(dim=1, keepdim=True)
        guide = torch.cat((self.p2_reduce(p2), self.p3_reduce(p3_aligned), high_energy), dim=1)
        logits = self.router(guide)
        margin = self.reliability_margin
        reliability = margin + (1.0 - 2.0 * margin) * torch.sigmoid(logits[:, :1])
        scale_weights = torch.softmax(logits[:, 1:], dim=1)
        return reliability, scale_weights

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("PGALP forward expects [P2, P3]")
        p2, p3 = inputs
        if p2.ndim != 4 or p3.ndim != 4:
            raise ValueError("PGALP inputs must be BCHW tensors")
        if p2.shape[1] != self.p2_channels or p3.shape[1] != self.p3_channels:
            raise ValueError(
                f"PGALP expected channels {self.p2_channels}/{self.p3_channels}, "
                f"got {p2.shape[1]}/{p3.shape[1]}"
            )
        if p2.shape[0] != p3.shape[0]:
            raise ValueError("PGALP P2/P3 batch sizes differ")
        if p2.shape[-2] != 2 * p3.shape[-2] or p2.shape[-1] != 2 * p3.shape[-1]:
            raise ValueError("PGALP expects P2 spatial size to be exactly 2x P3")

        candidates = [self._blur(p2, i) for i in range(len(self.kernels))]
        reliability, scale_weights = self.routing(p2, p3, candidates[0])
        smooth = torch.zeros_like(p2)
        for index, candidate in enumerate(candidates):
            smooth = smooth + scale_weights[:, index : index + 1] * candidate
        background = 1.0 - reliability
        return p2 + self.layer_scale * background * (smooth - p2)


__all__ = ("PGALP",)
