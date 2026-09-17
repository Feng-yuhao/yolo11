"""Clean hierarchical spatial semantic-detail guidance.

This module is the causal consolidation of E10a DSSG.  It deliberately
removes the P5 descriptor, sparse Top-K pooling, and channel gate that the
D10 frozen-checkpoint intervention found to be numerically inactive.  The
two spatially selective paths that remained measurable are retained:

    projected P4 semantics -> spatially gated residual into P3
    enhanced P3 semantics  -> detail-aware spatially gated residual into P2

The operators preserve feature shapes and start close to the E1 P2 model by
using zero-initialized gate logits and small learnable residual scales.
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


class _SpatialGate(nn.Module):
    """Predict a single spatial reliability map from compact descriptors."""

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


class P4P3SpatialGuide(nn.Module):
    """Inject projected P4 semantics into P3 under a cross-scale spatial gate."""

    def __init__(self, input_channels, layer_scale_init=0.05, gate_hidden=16):
        super().__init__()
        if not isinstance(input_channels, (list, tuple)) or len(input_channels) != 2:
            raise ValueError("P4P3SpatialGuide input_channels must be [P3, P4]")
        self.p3_channels, self.p4_channels = (int(v) for v in input_channels)
        if min(self.p3_channels, self.p4_channels) < 8:
            raise ValueError("P4P3SpatialGuide requires at least 8 channels per input")
        if not 0.0 < float(layer_scale_init) <= 1.0:
            raise ValueError("layer_scale_init must be in (0, 1]")
        if int(gate_hidden) < 4:
            raise ValueError("gate_hidden must be at least 4")

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
        self.spatial_gate = _SpatialGate(4, hidden=int(gate_hidden))
        self.layer_scale = nn.Parameter(
            torch.full((1, self.p3_channels, 1, 1), float(layer_scale_init))
        )

    @staticmethod
    def _descriptors(p3, semantic):
        return torch.cat(
            (
                p3.mean(dim=1, keepdim=True),
                p3.amax(dim=1, keepdim=True),
                semantic.mean(dim=1, keepdim=True),
                semantic.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )

    def routing(self, p3, p4):
        semantic = self.p4_project(p4)
        semantic = F.interpolate(semantic, size=p3.shape[-2:], mode="nearest")
        semantic = self.semantic_refine(semantic)
        spatial = self.spatial_gate(self._descriptors(p3, semantic))
        return semantic, spatial

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("P4P3SpatialGuide forward expects [P3, P4]")
        p3, p4 = inputs
        if p3.ndim != 4 or p4.ndim != 4:
            raise ValueError("P4P3SpatialGuide inputs must be BCHW tensors")
        if p3.shape[1] != self.p3_channels or p4.shape[1] != self.p4_channels:
            raise ValueError(
                f"P4P3SpatialGuide expected channels {self.p3_channels}/{self.p4_channels}, "
                f"got {p3.shape[1]}/{p4.shape[1]}"
            )
        if p3.shape[0] != p4.shape[0]:
            raise ValueError("P4P3SpatialGuide input batch sizes differ")
        if p3.shape[-2:] != (2 * p4.shape[-2], 2 * p4.shape[-1]):
            raise ValueError("P4P3SpatialGuide expects P3 to be exactly 2x P4")
        semantic, spatial = self.routing(p3, p4)
        return p3 + self.layer_scale * spatial * semantic


class P3P2DetailGuide(nn.Module):
    """Inject P3 semantics into P2 only where P2 detail supports the route."""

    def __init__(self, input_channels, layer_scale_init=0.05, gate_hidden=16):
        super().__init__()
        if not isinstance(input_channels, (list, tuple)) or len(input_channels) != 2:
            raise ValueError("P3P2DetailGuide input_channels must be [P2, P3_sem]")
        self.p2_channels, self.p3_channels = (int(v) for v in input_channels)
        if min(self.p2_channels, self.p3_channels) < 8:
            raise ValueError("P3P2DetailGuide requires at least 8 channels per input")
        if not 0.0 < float(layer_scale_init) <= 1.0:
            raise ValueError("layer_scale_init must be in (0, 1]")
        if int(gate_hidden) < 4:
            raise ValueError("gate_hidden must be at least 4")

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
        self.spatial_gate = _SpatialGate(5, hidden=int(gate_hidden))
        self.layer_scale = nn.Parameter(
            torch.full((1, self.p2_channels, 1, 1), float(layer_scale_init))
        )

    @staticmethod
    def _descriptors(p2, semantic):
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
        spatial = self.spatial_gate(self._descriptors(p2, semantic))
        return semantic, spatial

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("P3P2DetailGuide forward expects [P2, P3_sem]")
        p2, p3_sem = inputs
        if p2.ndim != 4 or p3_sem.ndim != 4:
            raise ValueError("P3P2DetailGuide inputs must be BCHW tensors")
        if p2.shape[1] != self.p2_channels or p3_sem.shape[1] != self.p3_channels:
            raise ValueError(
                f"P3P2DetailGuide expected channels {self.p2_channels}/{self.p3_channels}, "
                f"got {p2.shape[1]}/{p3_sem.shape[1]}"
            )
        if p2.shape[0] != p3_sem.shape[0]:
            raise ValueError("P3P2DetailGuide input batch sizes differ")
        if p2.shape[-2:] != (2 * p3_sem.shape[-2], 2 * p3_sem.shape[-1]):
            raise ValueError("P3P2DetailGuide expects P2 to be exactly 2x P3_sem")
        semantic, spatial = self.routing(p2, p3_sem)
        return p2 + self.layer_scale * spatial * semantic


__all__ = ("P4P3SpatialGuide", "P3P2DetailGuide")
