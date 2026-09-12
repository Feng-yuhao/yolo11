"""Phase-aware detail bypass for high-resolution small-object detection.

The module keeps the semantic P2 tensor as the identity/main path.  A P1 tensor
is rearranged into four stride-2 sampling phases, spatially mixed by a P2-
conditioned softmax gate, refined cheaply, and injected through a learnable
near-zero residual scale.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PADB(nn.Module):
    """P2-conditioned four-phase detail residual.

    Args:
        channels: ``[P1 channels, P2 channels]`` supplied by ``parse_model``.
        hidden: Hidden width of the spatial phase gate.
        gamma_init: Initial residual strength.  A small non-zero value lets both
            the residual scale and the detail branch learn from the first step.
    """

    def __init__(self, channels, hidden=16, gamma_init=0.01):
        super().__init__()
        if not isinstance(channels, (list, tuple)) or len(channels) != 2:
            raise ValueError(f"PADB expects [P1_channels, P2_channels], got {channels!r}")
        self.c_p1, self.c_p2 = (int(channels[0]), int(channels[1]))
        self.hidden = int(hidden)
        if min(self.c_p1, self.c_p2, self.hidden) <= 0:
            raise ValueError("PADB channel widths must be positive")

        self.phase_gate = nn.Sequential(
            nn.Conv2d(self.c_p2, self.hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.hidden, 4, 1, bias=True),
        )
        # Uniform phase mixing at initialization; content adaptation is learned.
        nn.init.zeros_(self.phase_gate[-1].weight)
        nn.init.zeros_(self.phase_gate[-1].bias)

        self.detail_refine = nn.Sequential(
            nn.Conv2d(self.c_p1, self.c_p1, 3, 1, 1, groups=self.c_p1, bias=False),
            nn.BatchNorm2d(self.c_p1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.c_p1, self.c_p2, 1, bias=False),
            nn.BatchNorm2d(self.c_p2),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def phase_weights(self, p2):
        """Return normalized spatial weights with shape ``B,4,H,W``."""
        return self.phase_gate(p2).softmax(dim=1)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("PADB forward expects [P1, P2]")
        p1, p2 = x
        if p1.ndim != 4 or p2.ndim != 4:
            raise ValueError("PADB inputs must be BCHW tensors")
        if p1.shape[1] != self.c_p1 or p2.shape[1] != self.c_p2:
            raise ValueError(
                f"PADB channel mismatch: got {p1.shape[1]}/{p2.shape[1]}, "
                f"expected {self.c_p1}/{self.c_p2}"
            )
        if p1.shape[-2] != 2 * p2.shape[-2] or p1.shape[-1] != 2 * p2.shape[-1]:
            raise ValueError(
                f"PADB expects P1 spatial size exactly 2x P2, got "
                f"{tuple(p1.shape[-2:])} and {tuple(p2.shape[-2:])}"
            )

        # Pixel-unshuffle preserves all four sub-pixel sampling phases.
        phases = F.pixel_unshuffle(p1, 2).reshape(
            p1.shape[0], self.c_p1, 4, p2.shape[-2], p2.shape[-1]
        )
        weights = self.phase_weights(p2).unsqueeze(1)
        mixed = (phases * weights).sum(dim=2)
        detail = self.detail_refine(mixed)
        return p2 + self.gamma.to(dtype=detail.dtype) * detail


__all__ = ("PADB",)
