"""E19a: reliability-conditioned local alignment before multiscale fusion.

The adapter keeps the E17a interface: it receives ``[top_down, lateral]`` and
returns a tensor with the lateral feature's shape.  Unlike E17a's value-only
adapter, it explicitly compares a 3x3 neighborhood of the top-down feature to
the lateral reference, softly aligns the projected top-down semantics, and
injects only a reliability-gated residual into the lateral branch.

No grid sampling, deformable convolution, replication padding, or cyclic
rolling is used. Local candidates use deterministic zero-border concatenation
and slicing; invalid border candidates are masked before softmax.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class RCFAAdapter(nn.Module):
    """Reliability-conditioned feature alignment with a bounded 3x3 search.

    Args:
        input_channels: ``[top_down_channels, lateral_channels]`` injected by
            ``parse_model``.
        hidden: Shared channel width used for correlation and aligned updates.
        residual_init: Initial residual strength applied to the lateral update.
        center_prior: Fixed logit prior for the zero-displacement candidate.
    """

    shifts = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    center_index = 4

    def __init__(self, input_channels, hidden=32, residual_init=0.05, center_prior=4.0):
        super().__init__()
        if not isinstance(input_channels, (list, tuple)) or len(input_channels) != 2:
            raise ValueError("RCFAAdapter input_channels must be [top_down, lateral]")
        top_channels, lateral_channels = (int(value) for value in input_channels)
        hidden = int(hidden)
        residual_init = float(residual_init)
        center_prior = float(center_prior)
        if min(top_channels, lateral_channels, hidden) < 1:
            raise ValueError("RCFAAdapter channel counts must be positive")
        if not 0.0 < residual_init <= 1.0:
            raise ValueError("residual_init must be in (0, 1]")
        if not math.isfinite(center_prior) or center_prior < 0.0:
            raise ValueError("center_prior must be finite and non-negative")

        self.channels = (top_channels, lateral_channels)
        self.hidden = hidden
        self.center_prior_value = center_prior
        self.top_project = nn.Conv2d(top_channels, hidden, 1, bias=False)
        self.lateral_project = nn.Conv2d(lateral_channels, hidden, 1, bias=False)

        # Nine local correlations plus same-location difference and local energy.
        self.shift_predictor = nn.Sequential(
            nn.Conv2d(11, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 9, 1),
        )
        nn.init.zeros_(self.shift_predictor[-1].weight)
        nn.init.zeros_(self.shift_predictor[-1].bias)

        # E17a-compatible reliability evidence.  Zero initialization makes the
        # initial sigmoid reliability exactly 0.5 without freezing gradients.
        self.reliability = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, 1),
        )
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.zeros_(self.reliability[-1].bias)

        self.update = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2),
            nn.SiLU(),
            nn.Conv2d(hidden * 2, lateral_channels, 1),
        )
        self.layer_scale = nn.Parameter(torch.tensor(residual_init))

        prior = torch.zeros(1, 9, 1, 1)
        prior[:, self.center_index] = center_prior
        self.register_buffer("center_bias", prior, persistent=True)

    @staticmethod
    def _validate_features(features):
        if not isinstance(features, (list, tuple)) or len(features) != 2:
            raise ValueError("RCFAAdapter expects [top_down, lateral]")
        top_down, lateral = features
        if top_down.ndim != 4 or lateral.ndim != 4:
            raise ValueError("RCFAAdapter inputs must be BCHW tensors")
        if top_down.shape[0] != lateral.shape[0] or top_down.shape[-2:] != lateral.shape[-2:]:
            raise ValueError("Top-down and lateral feature batches/grids must align")
        return top_down, lateral

    @classmethod
    def _shift_views(cls, feature):
        """Yield bounded local shifts without cyclic wraparound."""
        height, width = feature.shape[-2:]
        zero_column = feature.new_zeros((*feature.shape[:-1], 1))
        horizontal = torch.cat((zero_column, feature, zero_column), dim=-1)
        zero_row = feature.new_zeros((*horizontal.shape[:-2], 1, horizontal.shape[-1]))
        padded = torch.cat((zero_row, horizontal, zero_row), dim=-2)
        for dy, dx in cls.shifts:
            y0 = 1 + dy
            x0 = 1 + dx
            yield padded[..., y0 : y0 + height, x0 : x0 + width]

    def routing(self, features):
        """Return update, reliability and nine-way local alignment weights."""
        top_down, lateral = self._validate_features(features)
        if top_down.shape[1] != self.channels[0] or lateral.shape[1] != self.channels[1]:
            raise ValueError(
                f"RCFAAdapter expected channels {self.channels}, got "
                f"{(top_down.shape[1], lateral.shape[1])}"
            )

        top = self.top_project(top_down)
        local = self.lateral_project(lateral)
        top_norm = F.normalize(top.float(), dim=1, eps=1e-6)
        local_norm = F.normalize(local.float(), dim=1, eps=1e-6)

        correlations = [
            (shifted * local_norm).sum(dim=1, keepdim=True)
            for shifted in self._shift_views(top_norm)
        ]
        correlation = torch.cat(correlations, dim=1).to(dtype=top.dtype)
        center_cosine = correlation[:, self.center_index : self.center_index + 1]
        difference = (top - local).abs().mean(dim=1, keepdim=True)
        local_energy = local.abs().mean(dim=1, keepdim=True)

        shift_input = torch.cat((correlation, difference, local_energy), dim=1)
        logits = self.shift_predictor(shift_input) + self.center_bias.to(dtype=shift_input.dtype)
        validity_seed = top.new_ones((1, 1, top.shape[-2], top.shape[-1]))
        validity = torch.cat(list(self._shift_views(validity_seed)), dim=1).bool()
        logits = logits.masked_fill(~validity, torch.finfo(logits.dtype).min)
        shift_weights = logits.float().softmax(dim=1).to(dtype=top.dtype)

        # Accumulate candidates one-by-one instead of materializing Bx9xCxHxW.
        aligned_top = torch.zeros_like(top)
        for index, shifted in enumerate(self._shift_views(top)):
            aligned_top = aligned_top + shift_weights[:, index : index + 1] * shifted

        reliability_input = torch.cat((center_cosine, difference, local_energy), dim=1)
        reliability = torch.sigmoid(self.reliability(reliability_input))
        update = self.update(torch.cat((aligned_top, local), dim=1))
        return update, reliability, shift_weights

    def forward(self, features):
        _, lateral = self._validate_features(features)
        update, reliability, _ = self.routing(features)
        return lateral + self.layer_scale * reliability * update


__all__ = ("RCFAAdapter",)
