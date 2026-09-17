"""P1 semantic-selective detail injection for E15a.

The module preserves stride-2 P1 phase information with PixelUnshuffle,
projects it to the P2 channel space, and uses P2/P3 semantics to select
where the detail residual may enter P2. The per-channel bounded gain is
zero-initialized, making the initial module exactly identity with respect
to the P2 input.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNAct(nn.Sequential):
    def __init__(self, c1, c2, kernel=1, padding=0, groups=1, act=True):
        layers = [
            nn.Conv2d(c1, c2, kernel, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class P1SDI(nn.Module):
    """Inject phase-preserved P1 detail into P2 under compact semantic spatial gating."""

    def __init__(self, input_channels, gate_hidden=16, max_gain=0.25):
        super().__init__()
        if not isinstance(input_channels, (list, tuple)) or len(input_channels) != 3:
            raise ValueError("P1SDI input_channels must be [P1, P2_sem, P3_sem]")
        self.p1_channels, self.p2_channels, self.p3_channels = (int(v) for v in input_channels)
        self.gate_hidden = int(gate_hidden)
        self.max_gain = float(max_gain)
        if min(self.p1_channels, self.p2_channels, self.p3_channels) <= 0:
            raise ValueError("P1SDI channel counts must be positive")
        if self.gate_hidden <= 0:
            raise ValueError("gate_hidden must be positive")
        if not 0.0 < self.max_gain <= 1.0:
            raise ValueError("max_gain must be in (0, 1]")

        phase_channels = 4 * self.p1_channels
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.detail_project = nn.Sequential(
            _ConvBNAct(phase_channels, self.p2_channels, 1),
            _ConvBNAct(
                self.p2_channels,
                self.p2_channels,
                3,
                padding=1,
                groups=self.p2_channels,
            ),
            _ConvBNAct(self.p2_channels, self.p2_channels, 1, act=False),
        )
        self.gate_net = nn.Sequential(
            _ConvBNAct(6, self.gate_hidden, 3, padding=1),
            nn.Conv2d(self.gate_hidden, 1, 1, bias=True),
        )
        self.raw_gain = nn.Parameter(torch.zeros(1, self.p2_channels, 1, 1))

    @staticmethod
    def _descriptors(p2_sem, p1_detail, p3_up):
        return torch.cat(
            (
                p2_sem.mean(dim=1, keepdim=True),
                p2_sem.amax(dim=1, keepdim=True),
                p1_detail.mean(dim=1, keepdim=True),
                p1_detail.amax(dim=1, keepdim=True),
                p3_up.mean(dim=1, keepdim=True),
                p3_up.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )

    def routing(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
            raise ValueError("P1SDI forward expects [P1, P2_sem, P3_sem]")
        p1, p2_sem, p3_sem = inputs
        if any(t.ndim != 4 for t in inputs):
            raise ValueError("P1SDI inputs must be BCHW tensors")
        actual_channels = (p1.shape[1], p2_sem.shape[1], p3_sem.shape[1])
        expected_channels = (self.p1_channels, self.p2_channels, self.p3_channels)
        if actual_channels != expected_channels:
            raise ValueError(f"P1SDI expected channels {expected_channels}, got {actual_channels}")
        if not (p1.shape[0] == p2_sem.shape[0] == p3_sem.shape[0]):
            raise ValueError("P1SDI input batch sizes differ")
        if p1.shape[-2:] != (2 * p2_sem.shape[-2], 2 * p2_sem.shape[-1]):
            raise ValueError("P1SDI expects P1 to be exactly 2x P2 spatially")
        if p2_sem.shape[-2:] != (2 * p3_sem.shape[-2], 2 * p3_sem.shape[-1]):
            raise ValueError("P1SDI expects P2 to be exactly 2x P3 spatially")

        phase = self.pixel_unshuffle(p1)
        if phase.shape[-2:] != p2_sem.shape[-2:]:
            raise RuntimeError("PixelUnshuffle output does not match P2 spatial size")
        detail = self.detail_project(phase)
        p3_up = F.interpolate(p3_sem, size=p2_sem.shape[-2:], mode="nearest")
        gate = torch.sigmoid(self.gate_net(self._descriptors(p2_sem, detail, p3_up)))
        gain = self.max_gain * torch.tanh(self.raw_gain)
        return detail, gate, gain

    def forward(self, inputs):
        _, p2_sem, _ = inputs
        detail, gate, gain = self.routing(inputs)
        return p2_sem + gain * gate * detail


__all__ = ("P1SDI",)
