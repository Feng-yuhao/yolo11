"""Global semantic context gating for high-resolution detection features."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalSemanticContextGate(nn.Module):
    """Modulate a local high-resolution feature with an aligned deep feature.

    Inputs are ``[local, semantic]``.  The first tensor determines the output
    shape and channel count.  A spatial gate is built from aligned local/deep
    embeddings and a channel gate is built from pooled deep semantics.  Their
    geometric mean is centered around 0.5, allowing bounded suppression and
    enhancement of the local feature.

    Both final gate projections are zero-initialized, so the module is exactly
    identity at installation.  This keeps semantic transfer from the official
    YOLO11s checkpoint auditable while leaving non-zero gradients for the gate.
    """

    def __init__(
        self,
        channels,
        hidden_ratio=0.25,
        initial_strength=0.10,
    ):
        super().__init__()
        if not isinstance(channels, (list, tuple)) or len(channels) != 2:
            raise ValueError("GlobalSemanticContextGate expects [local_channels, semantic_channels]")
        local_channels, semantic_channels = map(int, channels)
        if local_channels <= 0 or semantic_channels <= 0:
            raise ValueError(f"Invalid channels: {channels}")
        if not (0.0 < float(hidden_ratio) <= 1.0):
            raise ValueError(f"hidden_ratio must be in (0,1], got {hidden_ratio}")
        if not (0.0 < float(initial_strength) < 1.0):
            raise ValueError(
                f"initial_strength must be in (0,1), got {initial_strength}"
            )
        hidden = max(16, int(round(local_channels * float(hidden_ratio))))
        self.local_channels = local_channels
        self.semantic_channels = semantic_channels
        self.local_projection = nn.Sequential(
            nn.Conv2d(local_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.semantic_projection = nn.Sequential(
            nn.Conv2d(semantic_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.spatial_gate = nn.Conv2d(hidden, 1, 1, bias=True)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(semantic_channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, local_channels, 1, bias=True),
        )
        initial_logit = math.log(initial_strength / (1.0 - initial_strength))
        self.strength_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.zeros_(self.spatial_gate.bias)
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)

    @property
    def strength(self):
        return torch.sigmoid(self.strength_logit)

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("GlobalSemanticContextGate forward expects [local, semantic]")
        local, semantic = inputs
        if local.ndim != 4 or semantic.ndim != 4:
            raise ValueError("GlobalSemanticContextGate inputs must be BCHW tensors")
        if local.shape[0] != semantic.shape[0]:
            raise ValueError("Local and semantic features must have the same batch size")
        if local.shape[1] != self.local_channels or semantic.shape[1] != self.semantic_channels:
            raise ValueError(
                "Runtime channels differ from construction: "
                f"local={local.shape[1]}/{self.local_channels}, "
                f"semantic={semantic.shape[1]}/{self.semantic_channels}"
            )
        local_embedding = self.local_projection(local)
        semantic_embedding = self.semantic_projection(semantic)
        semantic_embedding = F.interpolate(
            semantic_embedding, size=local.shape[-2:], mode="nearest"
        )
        spatial = torch.sigmoid(self.spatial_gate(local_embedding + semantic_embedding))
        channel = torch.sigmoid(self.channel_gate(semantic))
        joint = torch.sqrt((spatial * channel).clamp_min(1e-6))
        centered_gate = 2.0 * joint - 1.0
        return local * (1.0 + self.strength * centered_gate)

    def extra_repr(self):
        return (
            f"local_channels={self.local_channels}, "
            f"semantic_channels={self.semantic_channels}, "
            f"strength={float(self.strength.detach()):.4f}"
        )
