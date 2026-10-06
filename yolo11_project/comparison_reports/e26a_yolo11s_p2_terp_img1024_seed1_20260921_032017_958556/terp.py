"""Task-exclusive reverse-pyramid residual routing for the E26a detection head.

Only the added cross-scale residual sources are task exclusive.  The P2-P5
base features, Detect prediction towers, decoding, loss and assignment remain
the standard E1 implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv
from .head import Detect

__all__ = ("TERPHead",)


class _LinearProject(nn.Module):
    """Bias-free 1x1 projection plus BN, deliberately without activation."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.project = Conv(c1, c2, k=1, s=1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x)


class _LearnableAADown(nn.Module):
    """Trainable depthwise stride-2 alignment with binomial low-pass initialization."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.down = nn.Conv2d(c1, c1, 3, stride=2, padding=1, groups=c1, bias=False)
        self.project = Conv(c1, c2, k=1, s=1, act=False)
        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=self.down.weight.dtype
        ).div_(16.0)
        with torch.no_grad():
            self.down.weight.copy_(kernel.view(1, 1, 3, 3).repeat(c1, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.down(x))


class _SpatialResidualRoute(nn.Module):
    """Bounded spatial residual route with a strict identity starting point."""

    def __init__(self, channels: int, max_gain: float = 0.2):
        super().__init__()
        if channels < 1 or max_gain <= 0:
            raise ValueError(f"Invalid residual route: channels={channels}, max_gain={max_gain}")
        self.channels = int(channels)
        self.max_gain = float(max_gain)
        descriptor_channels = 3 * self.channels
        self.alpha = nn.Parameter(torch.zeros(()))
        self.gate_dw = nn.Conv2d(
            descriptor_channels,
            descriptor_channels,
            3,
            padding=1,
            groups=descriptor_channels,
            bias=False,
        )
        self.gate_act = nn.SiLU(inplace=True)
        self.gate_out = nn.Conv2d(descriptor_channels, 1, 1, bias=True)
        # alpha=0, rather than a zero gate, provides the exact E1 identity.
        # A tiny nonzero output kernel lets the whole gate receive gradients on
        # the second staged backward after alpha leaves zero.
        nn.init.normal_(self.gate_out.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.gate_out.bias)

    @property
    def beta(self) -> torch.Tensor:
        return self.max_gain * torch.tanh(self.alpha)

    @staticmethod
    def _descriptor_norm(x: torch.Tensor) -> torch.Tensor:
        # Stateless and parameter-free.  This normalized copy is used only by
        # the gate; the residual always injects the unnormalized source.
        return F.group_norm(x, num_groups=1, weight=None, bias=None, eps=1.0e-5)

    def forward(self, target: torch.Tensor, aligned_source: torch.Tensor) -> torch.Tensor:
        if target.shape != aligned_source.shape:
            raise RuntimeError(
                f"TERP route shape mismatch: target={tuple(target.shape)}, source={tuple(aligned_source.shape)}"
            )
        target_norm = self._descriptor_norm(target)
        source_norm = self._descriptor_norm(aligned_source)
        descriptor = torch.cat((target_norm, source_norm, (target_norm - source_norm).abs()), dim=1)
        gate = torch.sigmoid(self.gate_out(self.gate_act(self.gate_dw(descriptor))))
        return target + self.beta * gate * aligned_source


class TERPHead(Detect):
    """E1 Detect towers with opposite-direction task-exclusive residual sources.

    Input order is [B1, B2, B3, B4, P2, P3, P4, P5].  The first four
    backbone tensors provide regression-only residual evidence.  The final
    four tensors are the unchanged E1 Detect pyramid and remain the shared
    bases for both prediction tasks.
    """

    route_names = ("cls_p2", "cls_p3", "cls_p4", "reg_p2", "reg_p3", "reg_p4", "reg_p5")

    def __init__(self, nc: int = 80, ch=(), max_gain: float = 0.2):
        channels = tuple(int(value) for value in ch)
        if len(channels) != 8:
            raise ValueError(f"TERPHead requires 8 input channels [B1..B4,P2..P5], received {channels}")
        source_channels, detect_channels = channels[:4], channels[4:]
        super().__init__(nc=nc, ch=detect_channels)
        if self.nl != 4:
            raise RuntimeError(f"TERPHead must retain four Detect levels, got nl={self.nl}")

        self.source_channels = source_channels
        self.detect_channels = detect_channels
        self.max_gain = float(max_gain)

        # Non-recursive adjacent semantic routes: raw P3->P2, P4->P3, P5->P4.
        self.cls_aligners = nn.ModuleList(
            _LinearProject(detect_channels[index + 1], detect_channels[index]) for index in range(3)
        )
        # Raw backbone routes: B1->P2, B2->P3, B3->P4, B4->P5.
        self.reg_aligners = nn.ModuleList(
            _LearnableAADown(source_channels[index], detect_channels[index]) for index in range(4)
        )
        self.cls_routes = nn.ModuleList(
            _SpatialResidualRoute(detect_channels[index], max_gain=self.max_gain) for index in range(3)
        )
        self.reg_routes = nn.ModuleList(
            _SpatialResidualRoute(detect_channels[index], max_gain=self.max_gain) for index in range(4)
        )

    def prepare_task_inputs(self, x):
        """Return regression/classification inputs without modifying caller-owned tensors."""
        if not isinstance(x, (list, tuple)) or len(x) != 8:
            raise ValueError(f"TERPHead forward requires 8 tensors, received {type(x).__name__} length={len(x)}")
        backbone = list(x[:4])
        base = list(x[4:])

        cls_inputs = list(base)
        for index in range(3):
            # Always use the original next-deeper base, never recursively routed cls features.
            aligned = self.cls_aligners[index](base[index + 1])
            aligned = F.interpolate(aligned, size=base[index].shape[-2:], mode="nearest")
            cls_inputs[index] = self.cls_routes[index](base[index], aligned)

        reg_inputs = []
        for index in range(4):
            aligned = self.reg_aligners[index](backbone[index])
            if aligned.shape != base[index].shape:
                raise RuntimeError(
                    f"TERP regression alignment {index} did not land on its Detect level: "
                    f"aligned={tuple(aligned.shape)}, target={tuple(base[index].shape)}"
                )
            reg_inputs.append(self.reg_routes[index](base[index], aligned))
        return reg_inputs, cls_inputs

    def route_gains(self) -> dict[str, torch.Tensor]:
        routes = list(self.cls_routes) + list(self.reg_routes)
        return {name: route.beta for name, route in zip(self.route_names, routes)}

    def forward(self, x):
        """Use task-specific inputs while preserving the standard Detect output contract."""
        if self.end2end:
            raise RuntimeError("E26a TERPHead is preregistered only for the standard non-end2end YOLO11 Detect path")
        reg_inputs, cls_inputs = self.prepare_task_inputs(x)
        raw = [
            torch.cat((self.cv2[index](reg_inputs[index]), self.cv3[index](cls_inputs[index])), dim=1)
            for index in range(self.nl)
        ]
        if self.training:
            return raw
        prediction = self._inference(raw)
        return prediction if self.export else (prediction, raw)
