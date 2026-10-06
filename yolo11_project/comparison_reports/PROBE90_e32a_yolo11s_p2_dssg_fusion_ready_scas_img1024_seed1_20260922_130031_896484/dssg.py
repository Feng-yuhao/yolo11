"""Deep-to-shallow semantic guidance modules for the E10a VisDrone study.

The two modules form one controlled mechanism. ``DeepSemanticGuide`` uses
P5/P4 sparse-response descriptors to calibrate P4 channels and injects the
selected semantics into P3 through a spatial gate. ``SemanticDetailGate``
then injects the enhanced P3 semantics into P2 only at locations supported
by P2 detail evidence. Both outputs preserve the target tensor shape and use
small residual layer scales so E10a starts close to the frozen E1 model.
"""

from __future__ import annotations

import math

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


def _round_channels(value: float, minimum: int = 16) -> int:
    return max(minimum, int(math.ceil(float(value) / 8.0) * 8))


class SparseResponsePool(nn.Module):
    """Blend global-average and strongest-response pooling per channel."""

    def __init__(self, topk_ratio=0.0625, sparse_mix=0.5):
        super().__init__()
        self.topk_ratio = float(topk_ratio)
        self.sparse_mix = float(sparse_mix)
        if not 0.0 < self.topk_ratio <= 1.0:
            raise ValueError("topk_ratio must be in (0, 1]")
        if not 0.0 <= self.sparse_mix <= 1.0:
            raise ValueError("sparse_mix must be in [0, 1]")

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("SparseResponsePool expects BCHW input")
        flat = x.flatten(2)
        count = max(1, int(math.ceil(flat.shape[-1] * self.topk_ratio)))
        average = flat.mean(dim=-1, keepdim=True)
        strongest = flat.topk(count, dim=-1, largest=True, sorted=False).values.mean(
            dim=-1, keepdim=True
        )
        descriptor = (1.0 - self.sparse_mix) * average + self.sparse_mix * strongest
        return descriptor.unsqueeze(-1)


class _SpatialGate(nn.Module):
    """Predict one cross-scale location gate from compact descriptors."""

    def __init__(self, descriptor_channels, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            _ConvBNAct(descriptor_channels, hidden, 3, padding=1),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, descriptors):
        return torch.sigmoid(self.net(descriptors))


class DeepSemanticGuide(nn.Module):
    """Calibrate P4 with sparse P4/P5 context and inject it into P3.

    Args:
        input_channels: ``[P3, P4, P5]`` channels, injected by parse_model.
        hidden_ratio: Hidden ratio for channel calibration.
        topk_ratio: Spatial fraction retained by strongest-response pooling.
        sparse_mix: Weight of strongest-response pooling versus global average.
        layer_scale_init: Initial residual strength.
    """

    def __init__(
        self,
        input_channels,
        hidden_ratio=0.25,
        topk_ratio=0.0625,
        sparse_mix=0.5,
        layer_scale_init=0.05,
    ):
        super().__init__()
        if not isinstance(input_channels, (list, tuple)) or len(input_channels) != 3:
            raise ValueError("DeepSemanticGuide input_channels must be [P3, P4, P5]")
        self.p3_channels, self.p4_channels, self.p5_channels = (int(v) for v in input_channels)
        self.hidden_ratio = float(hidden_ratio)
        self.topk_ratio = float(topk_ratio)
        self.sparse_mix = float(sparse_mix)
        if min(self.p3_channels, self.p4_channels, self.p5_channels) < 8:
            raise ValueError("DeepSemanticGuide requires at least 8 channels per input")
        if self.hidden_ratio <= 0.0:
            raise ValueError("hidden_ratio must be positive")
        if not 0.0 < float(layer_scale_init) <= 1.0:
            raise ValueError("layer_scale_init must be in (0, 1]")

        hidden = _round_channels(self.p3_channels * self.hidden_ratio)
        self.hidden = hidden
        self.pool = SparseResponsePool(self.topk_ratio, self.sparse_mix)
        self.p4_descriptor = nn.Conv2d(self.p4_channels, hidden, 1, bias=True)
        self.p5_descriptor = nn.Conv2d(self.p5_channels, hidden, 1, bias=True)
        self.channel_gate = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.p3_channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.p4_project = _ConvBNAct(self.p4_channels, self.p3_channels, 1)
        self.semantic_refine = nn.Sequential(
            _ConvBNAct(
                self.p3_channels,
                self.p3_channels,
                3,
                padding=1,
                groups=self.p3_channels,
            ),
            _ConvBNAct(self.p3_channels, self.p3_channels, 1, act=False),
        )
        self.spatial_gate = _SpatialGate(4, hidden=16)
        self.layer_scale = nn.Parameter(
            torch.full((1, self.p3_channels, 1, 1), float(layer_scale_init))
        )
        nn.init.zeros_(self.channel_gate[1].weight)
        nn.init.zeros_(self.channel_gate[1].bias)

    @staticmethod
    def _spatial_descriptors(local, semantic):
        return torch.cat(
            (
                local.mean(dim=1, keepdim=True),
                local.amax(dim=1, keepdim=True),
                semantic.mean(dim=1, keepdim=True),
                semantic.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )

    def channel_weights(self, p4, p5):
        descriptor = self.p4_descriptor(self.pool(p4)) + self.p5_descriptor(self.pool(p5))
        return self.channel_gate(descriptor)

    def routing(self, p3, p4, p5):
        channel = self.channel_weights(p4, p5)
        semantic = self.p4_project(p4) * (0.5 + channel)
        semantic = F.interpolate(semantic, size=p3.shape[-2:], mode="nearest")
        semantic = self.semantic_refine(semantic)
        spatial = self.spatial_gate(self._spatial_descriptors(p3, semantic))
        return semantic, channel, spatial

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
            raise ValueError("DeepSemanticGuide forward expects [P3, P4, P5]")
        p3, p4, p5 = inputs
        if any(tensor.ndim != 4 for tensor in inputs):
            raise ValueError("DeepSemanticGuide inputs must be BCHW tensors")
        expected = (self.p3_channels, self.p4_channels, self.p5_channels)
        actual = (p3.shape[1], p4.shape[1], p5.shape[1])
        if actual != expected:
            raise ValueError(f"DeepSemanticGuide expected channels {expected}, got {actual}")
        if p3.shape[0] != p4.shape[0] or p3.shape[0] != p5.shape[0]:
            raise ValueError("DeepSemanticGuide input batch sizes differ")
        if p3.shape[-2:] != (2 * p4.shape[-2], 2 * p4.shape[-1]):
            raise ValueError("DeepSemanticGuide expects P3 to be exactly 2x P4")
        if p4.shape[-2:] != (2 * p5.shape[-2], 2 * p5.shape[-1]):
            raise ValueError("DeepSemanticGuide expects P4 to be exactly 2x P5")
        semantic, _, spatial = self.routing(p3, p4, p5)
        return p3 + self.layer_scale * spatial * semantic


class SemanticDetailGate(nn.Module):
    """Inject enhanced P3 semantics into P2 under P2 detail control."""

    def __init__(self, input_channels, layer_scale_init=0.05):
        super().__init__()
        if not isinstance(input_channels, (list, tuple)) or len(input_channels) != 2:
            raise ValueError("SemanticDetailGate input_channels must be [P2, P3_sem]")
        self.p2_channels, self.p3_channels = (int(v) for v in input_channels)
        if min(self.p2_channels, self.p3_channels) < 8:
            raise ValueError("SemanticDetailGate requires at least 8 channels per input")
        if not 0.0 < float(layer_scale_init) <= 1.0:
            raise ValueError("layer_scale_init must be in (0, 1]")

        self.semantic_project = _ConvBNAct(self.p3_channels, self.p2_channels, 1)
        self.semantic_refine = nn.Sequential(
            _ConvBNAct(
                self.p2_channels,
                self.p2_channels,
                3,
                padding=1,
                groups=self.p2_channels,
            ),
            _ConvBNAct(self.p2_channels, self.p2_channels, 1, act=False),
        )
        self.spatial_gate = _SpatialGate(5, hidden=16)
        self.layer_scale = nn.Parameter(
            torch.full((1, self.p2_channels, 1, 1), float(layer_scale_init))
        )

    @staticmethod
    def _spatial_descriptors(p2, semantic):
        local_average = F.avg_pool2d(p2, kernel_size=3, stride=1, padding=1)
        high_frequency = (p2 - local_average).abs().mean(dim=1, keepdim=True)
        return torch.cat(
            (
                p2.mean(dim=1, keepdim=True),
                p2.amax(dim=1, keepdim=True),
                semantic.mean(dim=1, keepdim=True),
                semantic.amax(dim=1, keepdim=True),
                high_frequency,
            ),
            dim=1,
        )

    def routing(self, p2, p3_sem):
        semantic = self.semantic_project(p3_sem)
        semantic = F.interpolate(semantic, size=p2.shape[-2:], mode="nearest")
        semantic = self.semantic_refine(semantic)
        spatial = self.spatial_gate(self._spatial_descriptors(p2, semantic))
        return semantic, spatial

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("SemanticDetailGate forward expects [P2, P3_sem]")
        p2, p3_sem = inputs
        if p2.ndim != 4 or p3_sem.ndim != 4:
            raise ValueError("SemanticDetailGate inputs must be BCHW tensors")
        if p2.shape[1] != self.p2_channels or p3_sem.shape[1] != self.p3_channels:
            raise ValueError(
                f"SemanticDetailGate expected channels {self.p2_channels}/{self.p3_channels}, "
                f"got {p2.shape[1]}/{p3_sem.shape[1]}"
            )
        if p2.shape[0] != p3_sem.shape[0]:
            raise ValueError("SemanticDetailGate input batch sizes differ")
        if p2.shape[-2:] != (2 * p3_sem.shape[-2], 2 * p3_sem.shape[-1]):
            raise ValueError("SemanticDetailGate expects P2 to be exactly 2x P3_sem")
        semantic, spatial = self.routing(p2, p3_sem)
        return p2 + self.layer_scale * spatial * semantic


__all__ = ("DeepSemanticGuide", "SemanticDetailGate", "SparseResponsePool")
