"""E34a: a short-lived but genuinely cross-stage high-resolution P2 state.

Only the new P2 state is expensive (stride 4). The original YOLO11s backbone,
neck, four Detect towers, TAL, and loss retain their original implementations.
There is one early feedback edge: H2 at the P3 stage changes the input from
which the later P4/P5 backbone features are computed.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _LocalBottleneck(nn.Module):
    """128-channel state, with all spatial processing at a narrow width."""

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.restore = nn.Sequential(
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.restore(self.spatial(self.reduce(x)))


class _SemanticProjection(nn.Module):
    """Project before upsampling so pointwise convolution runs at low resolution."""

    def __init__(self, source_channels: int, state_channels: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(source_channels, state_channels, 1, bias=False),
            nn.BatchNorm2d(state_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, source: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(self.project(source), size=size, mode="nearest")


class PersistentP2Stage1(nn.Module):
    """At the P3 stage, update the original P2 state with local and P3 evidence."""

    def __init__(self, channels: list[int], hidden: int = 32, gain: float = 0.1):
        super().__init__()
        if len(channels) != 2 or channels[0] <= 0 or channels[1] <= 0:
            raise ValueError(f"Expected [B2, B3] channels, got {channels}")
        if hidden <= 0 or not 0.0 < gain <= 0.25:
            raise ValueError("E34a hidden/gain outside the pre-registered range")
        self.local = _LocalBottleneck(channels[0], hidden)
        self.semantic = _SemanticProjection(channels[1], channels[0])
        self.gain = float(gain)  # fixed; no scalar gate that can zero the path

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        b2, b3 = inputs
        if b2.shape[0] != b3.shape[0] or b2.shape[-2:] != (2 * b3.shape[-2], 2 * b3.shape[-1]):
            raise ValueError(f"E34a stage1 expects stride-4/8 maps: {b2.shape}, {b3.shape}")
        return b2 + self.gain * (self.local(b2) + self.semantic(b3, b2.shape[-2:]))


class PersistentP2Feedback(nn.Module):
    """One early H2 -> deeper-backbone edge; original B3 lateral stays intact."""

    def __init__(self, channels: list[int], hidden: int = 32, gain: float = 0.1):
        super().__init__()
        if len(channels) != 2 or channels[0] <= 0 or channels[1] <= 0:
            raise ValueError(f"Expected [B3, H2] channels, got {channels}")
        if hidden <= 0 or not 0.0 < gain <= 0.25:
            raise ValueError("E34a hidden/gain outside the pre-registered range")
        self.reduce = nn.Sequential(
            nn.Conv2d(channels[1], hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden, channels[0], 1, bias=False),
            nn.BatchNorm2d(channels[0]),
        )
        self.gain = float(gain)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        b3, h2 = inputs
        if b3.shape[0] != h2.shape[0] or h2.shape[-2:] != (2 * b3.shape[-2], 2 * b3.shape[-1]):
            raise ValueError(f"E34a feedback expects stride-8/4 maps: {b3.shape}, {h2.shape}")
        detail = F.avg_pool2d(self.reduce(h2), kernel_size=2, stride=2)
        return b3 + self.gain * self.project(detail)


class PersistentP2Stage2(nn.Module):
    """At the P4 stage, finish the P2 state used in place of the old lateral."""

    def __init__(self, channels: list[int], hidden: int = 32, gain: float = 0.1):
        super().__init__()
        if len(channels) != 2 or channels[0] <= 0 or channels[1] <= 0:
            raise ValueError(f"Expected [H2, B4] channels, got {channels}")
        if hidden <= 0 or not 0.0 < gain <= 0.25:
            raise ValueError("E34a hidden/gain outside the pre-registered range")
        self.local = _LocalBottleneck(channels[0], hidden)
        self.semantic = _SemanticProjection(channels[1], channels[0])
        self.gain = float(gain)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        h2, b4 = inputs
        if h2.shape[0] != b4.shape[0] or h2.shape[-2:] != (4 * b4.shape[-2], 4 * b4.shape[-1]):
            raise ValueError(f"E34a stage2 expects stride-4/16 maps: {h2.shape}, {b4.shape}")
        return h2 + self.gain * (self.local(h2) + self.semantic(b4, h2.shape[-2:]))
