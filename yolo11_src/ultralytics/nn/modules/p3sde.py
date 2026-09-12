"""P3 detail-semantic enhancement for the clean E8b VisDrone experiment.

The module keeps the P3 tensor shape unchanged.  A local branch preserves
fine structures, a context branch increases the effective receptive field,
and spatial/channel gates select their contributions.  A small learnable
layer scale makes the initial module close to an identity mapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBNAct(nn.Sequential):
    def __init__(self, c1, c2, kernel=1, stride=1, padding=0, groups=1, dilation=1, act=True):
        layers = [
            nn.Conv2d(
                c1, c2, kernel, stride, padding, groups=groups,
                dilation=dilation, bias=False,
            ),
            nn.BatchNorm2d(c2),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class P3SDE(nn.Module):
    """Shape-preserving P3 selective detail/context enhancement.

    Args:
        channels: Input and output channel count, injected by parse_model.
        expansion: Hidden-channel ratio.  E8b-clean deliberately uses 1.0.
        context_kernel: Odd depthwise kernel for the context branch.
        context_dilation: Dilation of the second context depthwise layer.
        layer_scale_init: Initial residual scale; near-zero is identity-like.
    """

    def __init__(
        self,
        channels,
        expansion=1.0,
        context_kernel=5,
        context_dilation=2,
        layer_scale_init=1.0e-3,
    ):
        super().__init__()
        self.channels = int(channels)
        self.expansion = float(expansion)
        self.context_kernel = int(context_kernel)
        self.context_dilation = int(context_dilation)
        if self.channels < 8:
            raise ValueError("P3SDE requires at least 8 channels")
        if self.expansion <= 0:
            raise ValueError("P3SDE expansion must be positive")
        if self.context_kernel < 3 or self.context_kernel % 2 == 0:
            raise ValueError("P3SDE context_kernel must be odd and >= 3")
        if self.context_dilation < 1:
            raise ValueError("P3SDE context_dilation must be positive")

        hidden = max(8, int(round(self.channels * self.expansion / 8.0)) * 8)
        self.hidden = hidden
        self.pre = _ConvBNAct(self.channels, hidden, 1)

        self.detail = nn.Sequential(
            _ConvBNAct(hidden, hidden, 3, padding=1, groups=hidden),
            _ConvBNAct(hidden, hidden, 1),
        )
        context_padding = self.context_kernel // 2
        self.semantic = nn.Sequential(
            _ConvBNAct(
                hidden, hidden, self.context_kernel,
                padding=context_padding, groups=hidden,
            ),
            _ConvBNAct(
                hidden, hidden, 3,
                padding=self.context_dilation,
                groups=hidden, dilation=self.context_dilation,
            ),
            _ConvBNAct(hidden, hidden, 1),
        )

        gate_hidden = max(16, hidden // 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * hidden, gate_hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, 2 * hidden, 1, bias=True),
        )
        # Mean and max descriptors from both branches -> two spatial logits.
        self.spatial_gate = nn.Conv2d(4, 2, 3, padding=1, bias=True)
        self.project = _ConvBNAct(hidden, self.channels, 1, act=False)
        self.layer_scale = nn.Parameter(
            torch.full((1, self.channels, 1, 1), float(layer_scale_init))
        )

        nn.init.zeros_(self.channel_gate[-1].bias)
        nn.init.zeros_(self.spatial_gate.bias)

    @staticmethod
    def _descriptors(detail, semantic):
        return torch.cat(
            (
                detail.mean(dim=1, keepdim=True),
                detail.amax(dim=1, keepdim=True),
                semantic.mean(dim=1, keepdim=True),
                semantic.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"P3SDE expected BCHW with {self.channels} channels, got {tuple(x.shape)}"
            )
        base = self.pre(x)
        detail = base + self.detail(base)
        semantic = base + self.semantic(base)

        batch = x.shape[0]
        channel_logits = self.channel_gate(torch.cat((detail, semantic), dim=1))
        channel_logits = channel_logits.view(batch, 2, self.hidden, 1, 1)
        spatial_logits = self.spatial_gate(self._descriptors(detail, semantic)).unsqueeze(2)
        weights = torch.softmax(channel_logits + spatial_logits, dim=1)
        fused = weights[:, 0] * detail + weights[:, 1] * semantic
        return x + self.layer_scale * self.project(fused)


__all__ = ("P3SDE",)
