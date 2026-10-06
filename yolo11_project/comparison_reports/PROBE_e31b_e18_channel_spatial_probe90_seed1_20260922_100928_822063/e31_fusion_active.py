"""E31b probe: channel-spatial compatibility FusionReadyAdapter for E18.

Relative to E31a, this adds a lightweight channel-compatibility path from pooled
top-down/lateral/difference descriptors.  The last channel-gate layer is zero
initialized and the sigmoid output is multiplied by 2, so the channel gate starts
at exactly 1.0.  Therefore E31b starts from the same residual magnitude as E31a.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .head import Detect

E31_VARIANT = "B_CHANNEL_SPATIAL"


class FusionReadyAdapter(nn.Module):
    """E31a plus a neutral-initialized channel compatibility gate."""

    def __init__(self, channels, hidden=32, residual_init=0.05):
        super().__init__()
        if len(channels) != 2 or min(channels) < 1:
            raise ValueError("Expected [top_down_channels, lateral_channels]")
        top_channels, lateral_channels = channels
        self.channels = tuple(channels)
        self.hidden = int(hidden)
        self.top_project = nn.Conv2d(top_channels, hidden, 1, bias=False)
        self.lateral_project = nn.Conv2d(lateral_channels, hidden, 1, bias=False)
        self.compatibility = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, 1),
        )
        nn.init.zeros_(self.compatibility[-1].weight)
        nn.init.zeros_(self.compatibility[-1].bias)

        self.delta = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2),
            nn.SiLU(),
            nn.Conv2d(hidden * 2, lateral_channels, 1),
        )

        # Pooled top/local/|top-local| descriptors -> per-output-channel gate.
        self.channel_compatibility = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, lateral_channels, 1, bias=True),
        )
        nn.init.zeros_(self.channel_compatibility[-1].weight)
        nn.init.zeros_(self.channel_compatibility[-1].bias)

        self.layer_scale = nn.Parameter(
            torch.full((1, lateral_channels, 1, 1), float(residual_init))
        )

    def forward(self, features):
        if len(features) != 2:
            raise ValueError("FusionReadyAdapter needs [top_down, lateral]")
        top_down, lateral = features
        if top_down.shape[0] != lateral.shape[0] or top_down.shape[-2:] != lateral.shape[-2:]:
            raise ValueError("Top-down and lateral feature grids must align")

        top = self.top_project(top_down)
        local = self.lateral_project(lateral)

        top_norm = F.normalize(top.float(), dim=1, eps=1e-6)
        local_norm = F.normalize(local.float(), dim=1, eps=1e-6)
        cosine = (top_norm * local_norm).sum(1, keepdim=True).to(top.dtype)
        difference_map = (top - local).abs()
        difference = difference_map.mean(1, keepdim=True)
        energy = local.abs().mean(1, keepdim=True)
        spatial_gate = self.compatibility(
            torch.cat((cosine, difference, energy), 1)
        ).sigmoid()

        channel_descriptor = torch.cat(
            (
                F.adaptive_avg_pool2d(top, 1),
                F.adaptive_avg_pool2d(local, 1),
                F.adaptive_avg_pool2d(difference_map, 1),
            ),
            1,
        )
        # Zero final layer -> sigmoid(0)=0.5 -> *2 = 1.0 at initialization.
        channel_gate = 2.0 * self.channel_compatibility(channel_descriptor).sigmoid()

        update = self.delta(torch.cat((top, local), 1))
        return lateral + self.layer_scale * spatial_gate * channel_gate * update


class E17Detect(Detect):
    """Unchanged E18/E17 training-only auxiliary head."""

    def __init__(self, nc=10, ch=()):
        if len(ch) != 6 or ch[4] != ch[5]:
            raise ValueError(
                "E17Detect expects [P2,P3,P4,P5,adapted_P4,adapted_P3] with equal lateral widths"
            )
        super().__init__(nc=nc, ch=ch[:4])
        hidden = min(64, ch[4])
        self.aux_head = nn.Sequential(
            nn.Conv2d(ch[4], hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, nc + 4, 1),
        )
        self._aux_logits = None

    def clear_prior_logits(self):
        self._aux_logits = None

    def take_aux_logits(self):
        logits, self._aux_logits = self._aux_logits, None
        return logits

    def forward(self, x):
        if len(x) != 6:
            raise ValueError("E17Detect must receive exactly six source features")
        if self.training:
            self._aux_logits = (self.aux_head(x[5]), self.aux_head(x[4]))
        else:
            self._aux_logits = None
        return super().forward(x[:4])
