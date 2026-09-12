"""Single P4-to-P3 classification semantic adapter for the E13b probe."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .head import Detect


def _groups(channels: int, maximum: int = 8) -> int:
    for value in range(min(maximum, channels), 0, -1):
        if channels % value == 0:
            return value
    return 1


class P4P3ClassificationSemanticAdapter(nn.Module):
    """The unchanged E13a adapter, instantiated only for P4->P3."""

    def __init__(
        self,
        target_channels: int,
        source_channels: int,
        hidden_ratio: float = 0.25,
        topk_ratio: float = 0.0625,
        gain_init: float = 0.0,
    ):
        super().__init__()
        self.target_channels = int(target_channels)
        self.source_channels = int(source_channels)
        self.hidden_ratio = float(hidden_ratio)
        self.topk_ratio = float(topk_ratio)
        if min(self.target_channels, self.source_channels) < 8:
            raise ValueError("P4P3ClassificationSemanticAdapter requires at least 8 channels")
        if not 0.0 < self.hidden_ratio <= 1.0:
            raise ValueError("hidden_ratio must be in (0, 1]")
        if not 0.0 < self.topk_ratio <= 1.0:
            raise ValueError("topk_ratio must be in (0, 1]")
        if not 0.0 <= float(gain_init) <= 0.25:
            raise ValueError("gain_init must be in [0, 0.25]")

        hidden = max(16, int(math.ceil(self.target_channels * self.hidden_ratio / 8.0) * 8))
        self.hidden_channels = hidden
        self.source_project = nn.Sequential(
            nn.Conv2d(self.source_channels, self.target_channels, 1, bias=False),
            nn.GroupNorm(_groups(self.target_channels), self.target_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                self.target_channels,
                self.target_channels,
                3,
                padding=1,
                groups=self.target_channels,
                bias=False,
            ),
        )
        self.channel_selector = nn.Sequential(
            nn.Conv2d(self.source_channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.target_channels, 1, bias=True),
        )
        self.spatial_selector = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, 1, bias=True),
        )
        self.gain = nn.Parameter(
            torch.full((1, self.target_channels, 1, 1), float(gain_init))
        )
        nn.init.zeros_(self.channel_selector[-1].weight)
        nn.init.zeros_(self.channel_selector[-1].bias)
        nn.init.zeros_(self.spatial_selector[-1].weight)
        nn.init.zeros_(self.spatial_selector[-1].bias)

    def sparse_descriptor(self, source):
        flat = source.flatten(2)
        count = max(1, int(math.ceil(flat.shape[-1] * self.topk_ratio)))
        average = flat.mean(dim=-1, keepdim=True)
        strongest = flat.topk(count, dim=-1, largest=True, sorted=False).values.mean(
            dim=-1, keepdim=True
        )
        return (0.5 * average + 0.5 * strongest).unsqueeze(-1)

    def routing(self, target, source):
        semantic = self.source_project(source)
        semantic = F.interpolate(semantic, size=target.shape[-2:], mode="nearest")
        channel = 2.0 * torch.sigmoid(self.channel_selector(self.sparse_descriptor(source)))
        local_average = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
        high_frequency = (target - local_average).abs().mean(dim=1, keepdim=True)
        descriptors = torch.cat(
            (
                target.mean(dim=1, keepdim=True),
                target.amax(dim=1, keepdim=True),
                semantic.mean(dim=1, keepdim=True),
                semantic.amax(dim=1, keepdim=True),
                high_frequency,
            ),
            dim=1,
        )
        spatial = 2.0 * torch.sigmoid(self.spatial_selector(descriptors))
        return semantic, channel, spatial

    def forward(self, target, source):
        if target.ndim != 4 or source.ndim != 4:
            raise ValueError("P4P3ClassificationSemanticAdapter expects BCHW inputs")
        if target.shape[0] != source.shape[0]:
            raise ValueError("target/source batch sizes differ")
        if target.shape[1] != self.target_channels or source.shape[1] != self.source_channels:
            raise ValueError(
                f"Expected channels {self.target_channels}/{self.source_channels}, "
                f"got {target.shape[1]}/{source.shape[1]}"
            )
        if target.shape[-2:] != (2 * source.shape[-2], 2 * source.shape[-1]):
            raise ValueError("P3 target must be exactly 2x the P4 semantic source")
        semantic, channel, spatial = self.routing(target, source)
        return target + self.gain * channel * spatial * semantic


class P4P3SGCDDetect(Detect):
    """Detect head that modifies only the P3 classification input.

    Regression cv2 sees the original P2/P3/P4/P5 tensors. Classification cv3
    sees original P2/P4/P5 and P4-guided P3. This is the deletion-only E13b
    ablation requested after D13; no P3->P2 adapter is constructed.
    """

    def __init__(
        self,
        nc=80,
        hidden_ratio=0.25,
        topk_ratio=0.0625,
        gain_init=0.0,
        ch=(),
    ):
        super().__init__(nc=nc, ch=ch)
        if len(ch) != 4:
            raise ValueError("P4P3SGCDDetect requires P2/P3/P4/P5 inputs")
        self.hidden_ratio = float(hidden_ratio)
        self.topk_ratio = float(topk_ratio)
        self.semantic_adapter = P4P3ClassificationSemanticAdapter(
            ch[1], ch[2], hidden_ratio, topk_ratio, gain_init
        )

    def forward(self, x):
        if self.end2end:
            raise RuntimeError("E13b P4P3SGCDDetect does not support end-to-end mode")
        if not isinstance(x, list) or len(x) != 4:
            raise ValueError("P4P3SGCDDetect expects [P2, P3, P4, P5]")
        box_features = list(x)
        class_features = list(x)
        class_features[1] = self.semantic_adapter(x[1], x[2])
        outputs = [
            torch.cat((self.cv2[i](box_features[i]), self.cv3[i](class_features[i])), 1)
            for i in range(self.nl)
        ]
        if self.training:
            return outputs
        prediction = self._inference(outputs)
        return prediction if self.export else (prediction, outputs)


__all__ = ("P4P3ClassificationSemanticAdapter", "P4P3SGCDDetect")
