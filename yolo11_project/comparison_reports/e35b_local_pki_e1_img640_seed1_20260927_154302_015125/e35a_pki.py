"""PKINet-2024 operators adapted to YOLO11 C3k2; not a new claimed method.

Based on Cai et al., CVPR 2024, https://arxiv.org/abs/2403.06258.
Upstream: https://github.com/PKINet/PKINet (Apache-2.0).
This PyTorch implementation removes MMCV dependencies. See THIRD_PARTY_NOTICES.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from .block import Bottleneck, C3k, C3k2
from .conv import Conv


class E35CAA(nn.Module):
    """The linear CAA variant actually instantiated by InceptionBottleneck."""

    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AvgPool2d(7, stride=1, padding=3)
        self.project_in = nn.Conv2d(channels, channels, 1)
        self.horizontal = nn.Conv2d(channels, channels, (1, 11), padding=(0, 5), groups=channels)
        self.vertical = nn.Conv2d(channels, channels, (11, 1), padding=(5, 0), groups=channels)
        self.project_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        z = self.project_in(self.pool(x))
        return self.project_out(self.vertical(self.horizontal(z))).sigmoid()


class E35Inception(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre = Conv(channels, channels, 1)
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.context = nn.ModuleList(
            nn.Conv2d(channels, channels, k, padding=k // 2, groups=channels)
            for k in (5, 7, 9, 11)
        )
        self.mix = Conv(channels, channels, 1)
        self.caa = E35CAA(channels)
        self.post = Conv(channels, channels, 1)

    def forward(self, x):
        source = self.pre(x)
        local = self.local(source)
        # All four branches see the SAME local tensor, not a sequential cascade.
        multi = local
        for branch in self.context:
            multi = multi + branch(local)
        mixed = self.mix(multi)
        return self.post(mixed + mixed * self.caa(source))


class E35ConvFFN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Conv2d(channels, channels * 4, 1)
        self.depthwise = nn.Conv2d(channels * 4, channels * 4, 3, padding=1, groups=channels * 4)
        self.reduce = nn.Conv2d(channels * 4, channels, 1)

    def forward(self, x):
        # Channel LayerNorm, NOT GroupNorm and not spatial normalization.
        z = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        z = self.depthwise(self.expand(z))
        z = z * z.mean(dim=(2, 3), keepdim=True).sigmoid()  # GSiLU
        return self.reduce(z)


class E35PKIBlock(nn.Module):
    """Equal-channel PKIBlock, expansion=1, FFN=4, layer_scale=1, no drop path."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm1 = nn.BatchNorm2d(channels, eps=0.001, momentum=0.03)
        self.norm2 = nn.BatchNorm2d(channels, eps=0.001, momentum=0.03)
        self.inception = E35Inception(channels)
        self.ffn = E35ConvFFN(channels)
        self.gamma1 = nn.Parameter(torch.ones(channels))
        self.gamma2 = nn.Parameter(torch.ones(channels))
        self.reset_parameters()

    def reset_parameters(self):
        # Upstream Conv2d Kaiming-uniform, a=sqrt(5), fan_in; zero biases.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                if isinstance(module, nn.BatchNorm2d):
                    module.reset_running_stats()
        nn.init.ones_(self.gamma1)
        nn.init.ones_(self.gamma2)

    def forward(self, x):
        x = x + self.gamma1[None, :, None, None] * self.inception(self.norm1(x))
        return x + self.gamma2[None, :, None, None] * self.ffn(self.norm2(x))


class C3k2PKI(C3k2):
    """Keep the C3k2/C3k CSP shells; replace only their leaf Bottlenecks."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        if g != 1 or not shortcut:
            raise ValueError("E35a supports only the fixed E1 backbone configuration")
        for index, old in enumerate(self.m):
            if type(old) is Bottleneck:
                self.m[index] = E35PKIBlock(old.cv2.conv.out_channels)
            elif type(old) is C3k:
                for inner, leaf in enumerate(old.m):
                    if type(leaf) is not Bottleneck:
                        raise TypeError("Unexpected C3k leaf; refusing a silent topology change")
                    old.m[inner] = E35PKIBlock(leaf.cv2.conv.out_channels)
            else:
                raise TypeError(f"Unexpected C3k2 inner module: {type(old)}")
