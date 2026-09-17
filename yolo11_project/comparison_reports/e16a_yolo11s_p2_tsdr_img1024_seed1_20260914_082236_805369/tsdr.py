"""Tiny-supervised decoupled detail routing head for E16a.

The P2 classification path is intentionally identical to E10a. Only the P2
box/DFL path may receive stride-2 P1 detail. A tiny-object router limits that
residual spatially; its logits are supervised by a training-only center
heatmap in :mod:`ultralytics.utils.tsdr_loss`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv
from .head import Detect


class TSDRDetect(Detect):
    """Detect head with a regression-only, tiny-routed P1 detail path at P2.

    YAML inputs are ``[P1, P2_sem, P3_sem, P4, P5]``. The inherited towers
    are built for ``[P2_sem, P3_sem, P4, P5]`` so their shapes match E10a.
    """

    def __init__(self, nc=80, gate_hidden=16, max_gain=0.25, ch=()):
        if not isinstance(ch, (list, tuple)) or len(ch) != 5:
            raise ValueError("TSDRDetect channels must be [P1, P2_sem, P3_sem, P4, P5]")
        p1_channels, p2_channels, p3_channels, p4_channels, p5_channels = (int(v) for v in ch)
        if min(ch) <= 0:
            raise ValueError("TSDRDetect channel counts must be positive")
        if int(gate_hidden) <= 0:
            raise ValueError("gate_hidden must be positive")
        if not 0.0 < float(max_gain) <= 1.0:
            raise ValueError("max_gain must be in (0, 1]")

        super().__init__(nc=nc, ch=(p2_channels, p3_channels, p4_channels, p5_channels))
        self.input_channels = (p1_channels, p2_channels, p3_channels, p4_channels, p5_channels)
        self.p1_channels = p1_channels
        self.p2_channels = p2_channels
        self.p3_channels = p3_channels
        self.gate_hidden = int(gate_hidden)
        self.max_gain = float(max_gain)

        phase_channels = 4 * p1_channels
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.detail_project = nn.Sequential(
            Conv(phase_channels, p2_channels, 1),
            Conv(p2_channels, p2_channels, 3, g=p2_channels),
            Conv(p2_channels, p2_channels, 1, act=False),
        )
        # P2 mean/max, P3 mean/max, P1-detail mean-abs/max-abs.
        self.gate_net = nn.Sequential(
            nn.Conv2d(6, self.gate_hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.gate_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                self.gate_hidden,
                self.gate_hidden,
                3,
                padding=2,
                dilation=2,
                groups=self.gate_hidden,
                bias=False,
            ),
            nn.BatchNorm2d(self.gate_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.gate_hidden, 1, 1, bias=True),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, math.log(0.1 / 0.9))
        self.raw_gain = nn.Parameter(torch.zeros(1, p2_channels, 1, 1))

        gaussian = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        ) / 16.0
        self.register_buffer("gaussian3", gaussian.view(1, 1, 3, 3), persistent=True)

        self._tsdr_gate_logits = None
        self._last_aux_gate_loss = None
        self.collect_diagnostics = False
        self._diagnostic = None

    def clear_prior_logits(self):
        """Clear the stride-build cache through DetectionModel's existing hook."""
        self._tsdr_gate_logits = None
        self._last_aux_gate_loss = None

    def pop_gate_logits(self):
        """Consume the current training forward's router logits exactly once."""
        value = self._tsdr_gate_logits
        self._tsdr_gate_logits = None
        return value

    def _high_pass(self, p1):
        kernel = self.gaussian3.to(device=p1.device, dtype=p1.dtype).expand(self.p1_channels, 1, 3, 3)
        padded = F.pad(p1, (1, 1, 1, 1), mode="replicate")
        low = F.conv2d(padded, kernel, groups=self.p1_channels)
        return p1 - low

    @staticmethod
    def _detached_descriptors(p2_sem, p3_up, detail):
        # Router supervision must not rewrite E10a's semantic representations.
        return torch.cat(
            (
                p2_sem.mean(1, keepdim=True),
                p2_sem.amax(1, keepdim=True),
                p3_up.mean(1, keepdim=True),
                p3_up.amax(1, keepdim=True),
                detail.abs().mean(1, keepdim=True),
                detail.abs().amax(1, keepdim=True),
            ),
            1,
        ).detach()

    def routing(self, p1, p2_sem, p3_sem):
        phase = self.pixel_unshuffle(self._high_pass(p1))
        detail = self.detail_project(phase)
        p3_up = F.interpolate(p3_sem, size=p2_sem.shape[-2:], mode="nearest")
        logits = self.gate_net(self._detached_descriptors(p2_sem, p3_up, detail))
        gate = logits.sigmoid()
        gain = self.max_gain * torch.tanh(self.raw_gain)
        return detail, logits, gate, gain

    def _validate_inputs(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 5:
            raise ValueError("TSDRDetect forward expects [P1, P2_sem, P3_sem, P4, P5]")
        if any(t.ndim != 4 for t in inputs):
            raise ValueError("TSDRDetect inputs must be BCHW tensors")
        channels = tuple(int(t.shape[1]) for t in inputs)
        if channels != self.input_channels:
            raise ValueError(f"TSDRDetect expected channels {self.input_channels}, got {channels}")
        if len({int(t.shape[0]) for t in inputs}) != 1:
            raise ValueError("TSDRDetect input batch sizes differ")
        p1, p2, p3, p4, p5 = inputs
        expected = (
            (2 * p2.shape[-2], 2 * p2.shape[-1]),
            (2 * p3.shape[-2], 2 * p3.shape[-1]),
            (2 * p4.shape[-2], 2 * p4.shape[-1]),
            (2 * p5.shape[-2], 2 * p5.shape[-1]),
        )
        actual = (p1.shape[-2:], p2.shape[-2:], p3.shape[-2:], p4.shape[-2:])
        if actual != expected:
            raise ValueError(f"TSDRDetect expects adjacent 2x pyramid levels, got {actual}")

    def forward(self, inputs):
        """Use routed P1 detail only in P2's box tower; classification is untouched."""
        self._validate_inputs(inputs)
        p1, p2_sem, p3_sem, p4, p5 = inputs
        detail, logits, gate, gain = self.routing(p1, p2_sem, p3_sem)
        p2_box = p2_sem + gain * gate * detail

        self._tsdr_gate_logits = logits if self.training else None
        if self.collect_diagnostics:
            with torch.no_grad():
                residual = p2_box - p2_sem
                self._diagnostic = {
                    "gate_mean": float(gate.float().mean().cpu()),
                    "gate_std": float(gate.float().std(unbiased=False).cpu()),
                    "gate_min": float(gate.float().min().cpu()),
                    "gate_max": float(gate.float().max().cpu()),
                    "detail_rms": float(detail.float().square().mean().sqrt().cpu()),
                    "residual_rms": float(residual.float().square().mean().sqrt().cpu()),
                    "p2_rms": float(p2_sem.float().square().mean().sqrt().cpu()),
                }

        features = [p2_sem, p3_sem, p4, p5]
        outputs = []
        for i, feature in enumerate(features):
            box_feature = p2_box if i == 0 else feature
            outputs.append(torch.cat((self.cv2[i](box_feature), self.cv3[i](feature)), 1))
        if self.training:
            return outputs
        y = self._inference(outputs)
        return y if self.export else (y, outputs)


__all__ = ("TSDRDetect",)
