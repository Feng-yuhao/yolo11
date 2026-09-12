"""Contrastive-background spatial gates for the E11a DSSG probe."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dssg import DeepSemanticGuide, SemanticDetailGate, _ConvBNAct


class ContrastiveSpatialGate(nn.Module):
    """E10a-compatible gate with a learnable local-background contrast term.

    ``net`` intentionally keeps the same state keys and tensor shapes as the
    E10a spatial gate.  The two new scalar parameters start almost exactly at
    the E10a equation, so the probe starts from the frozen E10a checkpoint.
    """

    def __init__(self, descriptor_channels: int, hidden: int = 16, local_kernel: int = 7):
        super().__init__()
        if local_kernel < 3 or local_kernel % 2 == 0:
            raise ValueError("local_kernel must be an odd integer >= 3")
        self.local_kernel = int(local_kernel)
        self.net = nn.Sequential(
            _ConvBNAct(descriptor_channels, hidden, 3, padding=1),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # gain = 2*sigmoid(-8) ~= 0.00067: near-E10a but with nonzero gradient.
        self.contrast_logit = nn.Parameter(torch.tensor(-8.0))
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        self._cached_logits = None

    def contrast_gain(self):
        return 2.0 * torch.sigmoid(self.contrast_logit)

    def temperature(self):
        return self.log_temperature.clamp(-2.0, 2.0).exp()

    def forward(self, descriptors):
        raw_logits = self.net(descriptors)
        local_background = F.avg_pool2d(
            raw_logits,
            kernel_size=self.local_kernel,
            stride=1,
            padding=self.local_kernel // 2,
        )
        contrast = raw_logits - local_background
        logits = raw_logits + self.contrast_gain() * contrast / self.temperature()
        self._cached_logits = logits
        return torch.sigmoid(logits)

    def __getstate__(self):
        """Exclude the transient autograd cache from module deepcopy/pickling."""
        state = self.__dict__.copy()
        state["_cached_logits"] = None
        return state

    def pop_gate_logits(self):
        value = self._cached_logits
        self._cached_logits = None
        return value

    def clear_gate_logits(self):
        self._cached_logits = None


class CBSGDeepSemanticGuide(DeepSemanticGuide):
    """E10a P4->P3 semantic path with a fixed channel factor and CBSG gate."""

    def __init__(
        self,
        input_channels,
        hidden_ratio=0.25,
        topk_ratio=0.0625,
        sparse_mix=0.5,
        layer_scale_init=0.05,
        local_kernel=7,
    ):
        super().__init__(input_channels, hidden_ratio, topk_ratio, sparse_mix, layer_scale_init)
        # Keep the legacy descriptor parameters/state for an audited E10a
        # checkpoint transfer, but skip that proven-inactive computation.
        self.spatial_gate = ContrastiveSpatialGate(4, hidden=16, local_kernel=local_kernel)

    def channel_weights(self, p4, p5):
        del p5
        return p4.new_full((p4.shape[0], self.p3_channels, 1, 1), 0.5)


class CBSGDetailGate(SemanticDetailGate):
    """E10a P3->P2 detail path with the contrastive-background spatial gate."""

    def __init__(self, input_channels, layer_scale_init=0.05, local_kernel=9):
        super().__init__(input_channels, layer_scale_init)
        self.spatial_gate = ContrastiveSpatialGate(5, hidden=16, local_kernel=local_kernel)


__all__ = ("ContrastiveSpatialGate", "CBSGDeepSemanticGuide", "CBSGDetailGate")
