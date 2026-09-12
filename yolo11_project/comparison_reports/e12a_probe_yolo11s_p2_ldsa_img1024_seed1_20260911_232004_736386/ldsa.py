"""Local discriminative semantic attention for the E12a VisDrone probe.

The module keeps E10a's learned P3 spatial gate and residual semantic path,
but removes the causally inactive global P4/P5 channel descriptor.  Instead,
P3 detail supplies local queries and P4 supplies semantic keys/values.  The
attention answers *what* local semantic evidence to retrieve; the existing
spatial gate independently answers *where/how much* to inject it.

The new attention branch uses an exact-zero residual gain at initialization.
Consequently the initial graph is the D10-tested neutral-channel E10a graph,
while gradients first update the gain and subsequently the attention weights.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNAct(nn.Sequential):
    """State-key-compatible copy of the E10a projection block."""

    def __init__(self, c1, c2, kernel=1, padding=0, groups=1, act=True):
        layers = [
            nn.Conv2d(c1, c2, kernel, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class _SpatialGate(nn.Module):
    """State-key-compatible copy of the effective E10a spatial gate."""

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


def _normalization_groups(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class LocalCrossScaleAttention(nn.Module):
    """Retrieve P4 semantics around locations supported by pooled P3 detail.

    Attention is evaluated at P4 resolution.  For a 1024 input this is only
    64x64, and each query attends to a 5x5 neighbourhood rather than the full
    image.  This preserves local class/context evidence without quadratic
    global-attention memory.
    """

    def __init__(
        self,
        channels: int,
        attention_channels: int,
        heads: int = 4,
        kernel_size: int = 5,
        gain_init: float = 0.0,
    ):
        super().__init__()
        channels = int(channels)
        attention_channels = int(attention_channels)
        heads = int(heads)
        kernel_size = int(kernel_size)
        if channels < 8 or attention_channels < heads or attention_channels % heads:
            raise ValueError("attention_channels must be positive and divisible by heads")
        if heads < 1:
            raise ValueError("heads must be positive")
        if kernel_size < 3 or kernel_size % 2 != 1:
            raise ValueError("kernel_size must be an odd integer >= 3")
        if not 0.0 <= float(gain_init) <= 0.25:
            raise ValueError("gain_init must be in [0, 0.25]")

        self.channels = channels
        self.attention_channels = attention_channels
        self.heads = heads
        self.head_channels = attention_channels // heads
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        groups = _normalization_groups(channels)
        self.query_norm = nn.GroupNorm(groups, channels)
        self.semantic_norm = nn.GroupNorm(groups, channels)
        self.query = nn.Conv2d(channels, attention_channels, 1, bias=False)
        self.key = nn.Conv2d(channels, attention_channels, 1, bias=False)
        self.value = nn.Conv2d(channels, attention_channels, 1, bias=False)
        self.output = nn.Conv2d(attention_channels, channels, 1, bias=False)
        self.relative_bias = nn.Parameter(torch.zeros(heads, kernel_size * kernel_size))
        self.gain = nn.Parameter(torch.full((1, channels, 1, 1), float(gain_init)))

    def forward(self, p3_query, p4_semantic):
        if p3_query.shape != p4_semantic.shape:
            raise ValueError(
                f"LocalCrossScaleAttention shape mismatch: {tuple(p3_query.shape)} "
                f"vs {tuple(p4_semantic.shape)}"
            )
        batch, channels, height, width = p4_semantic.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")
        locations = height * width
        neighbours = self.kernel_size * self.kernel_size

        query = self.query(self.query_norm(p3_query)).reshape(
            batch, self.heads, self.head_channels, locations
        )
        key = self.key(self.semantic_norm(p4_semantic))
        value = self.value(self.semantic_norm(p4_semantic))
        key = F.unfold(key, self.kernel_size, padding=self.padding).reshape(
            batch, self.heads, self.head_channels, neighbours, locations
        )
        value = F.unfold(value, self.kernel_size, padding=self.padding).reshape(
            batch, self.heads, self.head_channels, neighbours, locations
        )

        logits = (query.unsqueeze(3) * key).sum(dim=2) / math.sqrt(self.head_channels)
        logits = logits + self.relative_bias.view(1, self.heads, neighbours, 1)
        weights = torch.softmax(logits.float(), dim=2).to(dtype=value.dtype)
        context = (weights.unsqueeze(2) * value).sum(dim=3)
        context = context.reshape(batch, self.attention_channels, height, width)
        context = self.output(context)
        return p4_semantic + self.gain * context


class LocalDiscriminativeSemanticGuide(nn.Module):
    """Replace E10a's inactive global channel path with local P3-query attention.

    Args:
        input_channels: ``[P3, P4]`` channels injected by ``parse_model``.
        attention_ratio: Attention width relative to the P3 output channels.
        heads: Local-attention head count.
        kernel_size: P4-neighbourhood side length.
        attention_gain_init: Residual gain of the new attention branch.
        layer_scale_init: Used only for fresh formal initialization.  During the
            probe this tensor is copied exactly from frozen E10a.
    """

    def __init__(
        self,
        input_channels,
        attention_ratio=0.5,
        heads=4,
        kernel_size=5,
        attention_gain_init=0.0,
        layer_scale_init=0.05,
    ):
        super().__init__()
        if not isinstance(input_channels, (list, tuple)) or len(input_channels) != 2:
            raise ValueError("LocalDiscriminativeSemanticGuide input_channels must be [P3, P4]")
        self.p3_channels, self.p4_channels = (int(value) for value in input_channels)
        self.attention_ratio = float(attention_ratio)
        self.heads = int(heads)
        self.kernel_size = int(kernel_size)
        if min(self.p3_channels, self.p4_channels) < 8 or self.attention_ratio <= 0.0:
            raise ValueError("Invalid LDSA channels/attention_ratio")
        if not 0.0 < float(layer_scale_init) <= 1.0:
            raise ValueError("layer_scale_init must be in (0, 1]")

        raw_width = max(self.heads, int(round(self.p3_channels * self.attention_ratio)))
        attention_channels = int(math.ceil(raw_width / self.heads) * self.heads)
        self.attention_channels = attention_channels

        # Names/structures below intentionally match E10a for exact transfer.
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
        self.local_attention = LocalCrossScaleAttention(
            self.p3_channels,
            attention_channels,
            heads=self.heads,
            kernel_size=self.kernel_size,
            gain_init=float(attention_gain_init),
        )

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

    def routing(self, p3, p4):
        p4_semantic = self.p4_project(p4)
        # Average retains stable context; max preserves sparse tiny-target evidence.
        p3_query = 0.5 * (
            F.avg_pool2d(p3, kernel_size=2, stride=2)
            + F.max_pool2d(p3, kernel_size=2, stride=2)
        )
        p4_semantic = self.local_attention(p3_query, p4_semantic)
        semantic = F.interpolate(p4_semantic, size=p3.shape[-2:], mode="nearest")
        semantic = self.semantic_refine(semantic)
        spatial = self.spatial_gate(self._spatial_descriptors(p3, semantic))
        return semantic, spatial

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("LocalDiscriminativeSemanticGuide forward expects [P3, P4]")
        p3, p4 = inputs
        if p3.ndim != 4 or p4.ndim != 4:
            raise ValueError("LDSA inputs must be BCHW tensors")
        if p3.shape[1] != self.p3_channels or p4.shape[1] != self.p4_channels:
            raise ValueError(
                f"Expected channels {self.p3_channels}/{self.p4_channels}, "
                f"got {p3.shape[1]}/{p4.shape[1]}"
            )
        if p3.shape[0] != p4.shape[0]:
            raise ValueError("LDSA input batch sizes differ")
        if p3.shape[-2:] != (2 * p4.shape[-2], 2 * p4.shape[-1]):
            raise ValueError("LDSA expects P3 to be exactly 2x P4")
        semantic, spatial = self.routing(p3, p4)
        return p3 + self.layer_scale * spatial * semantic


__all__ = ("LocalCrossScaleAttention", "LocalDiscriminativeSemanticGuide")
