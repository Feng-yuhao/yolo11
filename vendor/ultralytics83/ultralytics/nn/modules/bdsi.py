"""Adjacent P2/P3 detail-semantic residual interaction for small objects."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BDSI(nn.Module):
    """One direction of a parallel bidirectional P2/P3 interaction.

    Inputs are ``[target, source]`` and the output has exactly the target shape.
    ``semantic_to_detail`` aligns P3 upward to refine P2.  The reverse direction
    uses a fixed low-pass filter before stride-2 sampling so P2 details can reach
    P3 without naive aliasing.  Spatial and channel gates form a signed residual.

    The two final gate projections are zero initialized.  Consequently the
    module is an exact identity at installation while its gate projections still
    receive gradients on the first optimizer step.
    """

    DIRECTIONS = {"semantic_to_detail", "detail_to_semantic"}

    def __init__(self, channels, direction, hidden_ratio=0.25, initial_strength=0.10):
        super().__init__()
        if not isinstance(channels, (list, tuple)) or len(channels) != 2:
            raise ValueError("BDSI expects [target_channels, source_channels]")
        target_channels, source_channels = map(int, channels)
        if target_channels <= 0 or source_channels <= 0:
            raise ValueError(f"Invalid BDSI channels: {channels}")
        if direction not in self.DIRECTIONS:
            raise ValueError(f"Unknown BDSI direction: {direction}")
        if not (0.0 < float(hidden_ratio) <= 1.0):
            raise ValueError(f"hidden_ratio must be in (0,1], got {hidden_ratio}")
        if not (0.0 < float(initial_strength) < 1.0):
            raise ValueError(f"initial_strength must be in (0,1), got {initial_strength}")

        hidden = max(16, int(round(target_channels * float(hidden_ratio))))
        self.target_channels = target_channels
        self.source_channels = source_channels
        self.direction = direction
        self.source_refine = nn.Sequential(
            nn.Conv2d(source_channels, source_channels, 3, 1, 1,
                      groups=source_channels, bias=False),
            nn.BatchNorm2d(source_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(source_channels, target_channels, 1, bias=False),
            nn.BatchNorm2d(target_channels),
            nn.SiLU(inplace=True),
        )
        self.target_context = nn.Sequential(
            nn.Conv2d(target_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.source_context = nn.Sequential(
            nn.Conv2d(target_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.spatial_gate = nn.Conv2d(hidden, 1, 1, bias=True)
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_gate = nn.Conv2d(hidden, target_channels, 1, bias=True)
        initial_logit = math.log(initial_strength / (1.0 - initial_strength))
        self.strength_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))

        kernel_1d = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32)
        kernel_2d = torch.outer(kernel_1d, kernel_1d) / 16.0
        self.register_buffer("blur_kernel", kernel_2d[None, None], persistent=True)
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.zeros_(self.spatial_gate.bias)
        nn.init.zeros_(self.channel_gate.weight)
        nn.init.zeros_(self.channel_gate.bias)

    @property
    def strength(self):
        return torch.sigmoid(self.strength_logit)

    def _align_source(self, source, size):
        if self.direction == "semantic_to_detail":
            source = self.source_refine(source)
            return F.interpolate(source, size=size, mode="nearest")

        kernel = self.blur_kernel.to(dtype=source.dtype).expand(
            self.source_channels, 1, 3, 3
        )
        source = F.conv2d(source, kernel, stride=2, padding=1,
                          groups=self.source_channels)
        source = self.source_refine(source)
        if source.shape[-2:] != size:
            source = F.interpolate(source, size=size, mode="nearest")
        return source

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("BDSI forward expects [target, source]")
        target, source = inputs
        if target.ndim != 4 or source.ndim != 4:
            raise ValueError("BDSI inputs must be BCHW tensors")
        if target.shape[0] != source.shape[0]:
            raise ValueError("BDSI inputs must have the same batch size")
        if target.shape[1] != self.target_channels or source.shape[1] != self.source_channels:
            raise ValueError(
                "BDSI runtime channels differ from construction: "
                f"target={target.shape[1]}/{self.target_channels}, "
                f"source={source.shape[1]}/{self.source_channels}"
            )

        aligned = self._align_source(source, target.shape[-2:])
        context = self.target_context(target) + self.source_context(aligned)
        context = F.silu(context, inplace=True)
        spatial = torch.sigmoid(self.spatial_gate(context))
        channel = torch.sigmoid(self.channel_gate(self.channel_pool(context)))
        signed_gate = spatial + channel - 1.0
        return target + self.strength * signed_gate * aligned

    def extra_repr(self):
        return (
            f"target_channels={self.target_channels}, source_channels={self.source_channels}, "
            f"direction={self.direction!r}, strength={float(self.strength.detach()):.4f}"
        )
