"""E33a: P2-driven pyramid re-aggregation with a P3-delta return path.

The ordinary E1 detection towers, loss, and assignment are untouched.  Both
modules are exact identities at construction, including during model stride
initialisation.  They do not contain a fixed residual-gain ceiling.
"""

import torch
from torch import nn
from torch.nn import functional as F


class IRPUpdate(nn.Module):
    """Use newly formed P2 to update P3 before the original PAN P4/P5 path."""

    def __init__(self, channels):
        super().__init__()
        if len(channels) != 2 or channels[0] != channels[1]:
            raise ValueError(f"IRPUpdate needs [P3,P2] with equal channels, got {channels}")
        c = channels[0]
        hidden = max(32, c // 2)
        self.p2_project = nn.Conv2d(c, hidden, 1, bias=False)
        self.p3_project = nn.Conv2d(c, hidden, 1, bias=False)
        self.mix = nn.Sequential(
            nn.Conv2d(2 * hidden, 2 * hidden, 3, padding=1, groups=2 * hidden, bias=False),
            nn.SiLU(),
            nn.Conv2d(2 * hidden, hidden, 1, bias=False),
            nn.SiLU(),
        )
        self.out = nn.Conv2d(hidden, c, 1, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, inputs):
        p3, p2 = inputs
        if p2.shape[-2:] != (2 * p3.shape[-2], 2 * p3.shape[-1]):
            raise ValueError(f"IRPUpdate expects P2 at twice P3 resolution: {p2.shape}, {p3.shape}")
        detail = F.avg_pool2d(p2, kernel_size=2, stride=2)
        features = torch.cat((self.p3_project(p3), self.p2_project(detail)), dim=1)
        return p3 + self.out(self.mix(features))


class IRPReturn(nn.Module):
    """Return only the newly learned P3 increment to P2, never P3 itself."""

    def __init__(self, channels):
        super().__init__()
        if len(channels) != 3 or len(set(channels)) != 1:
            raise ValueError(f"IRPReturn needs [P2,P3_new,P3_old] of one width, got {channels}")
        c = channels[0]
        hidden = max(32, c // 2)
        # Bias-free throughout: a zero P3 increment *must* yield zero P2 update.
        self.project = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.Conv2d(hidden, c, 1, bias=False),
        )

    def forward(self, inputs):
        p2, new_p3, old_p3 = inputs
        if new_p3.shape != old_p3.shape:
            raise ValueError("IRPReturn P3 branches must have the same shape")
        delta = new_p3 - old_p3
        returned = F.interpolate(self.project(delta), size=p2.shape[-2:], mode="nearest")
        return p2 + returned
