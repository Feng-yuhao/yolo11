"""E41: image-conditioned task/channel/depth selection before Detect towers.

Inspired by task-aligned feature decomposition (TOOD); not a TOOD reproduction.
No auxiliary loss, box refinement, external sampling or new decoder.
"""
import torch
from torch import nn
from ultralytics.nn.modules.head import Detect

NEW_PREFIX = 'model.26.task_blocks.'
NEW_PARAMETERS = 199680


def encoder(cin, cout, kernel):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=kernel//2, bias=False),
                         nn.GroupNorm(6, cout), nn.SiLU())


class ChannelTaskBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = encoder(128, 48, 1)
        self.depths = nn.ModuleList(encoder(48, 48, 3) for _ in range(3))
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    nn.Conv2d(192, 32, 1, bias=False), nn.SiLU(),
                                    nn.Conv2d(32, 2 * 4 * 48, 1))
        self.out_cls = nn.Conv2d(48, 128, 1, bias=False)
        self.out_reg = nn.Conv2d(48, 128, 1, bias=False)
        nn.init.zeros_(self.out_cls.weight)
        nn.init.zeros_(self.out_reg.weight)

    def decompose(self, x):
        z = self.reduce(x)
        bank = [z]
        for layer in self.depths:
            z = layer(z)
            bank.append(z)
        b = x.shape[0]
        # Separate weights per task/channel, shared over spatial positions.
        # Pool only the routing descriptor: bank features retain full resolution.
        # Normalize DEPTH, never channels or tasks. Task 0=cls, task 1=reg.
        weights = self.router(torch.cat(bank, 1)).reshape(b, 2, 4, 48, 1, 1).softmax(2)
        stacked = torch.stack(bank, 1)
        cls = (stacked * weights[:, 0]).sum(1)
        reg = (stacked * weights[:, 1]).sum(1)
        return cls, reg, weights, bank

    def forward(self, x):
        cls, reg, _, _ = self.decompose(x)
        # No in-place mutation of the shared feature, and no detach.
        return x + self.out_cls(cls), x + self.out_reg(reg)


class ChannelTaskDetect(Detect):
    def forward(self, x):
        if self.end2end or self.nl != 4 or len(x) != 4:
            raise RuntimeError('E41 supports the pinned E1 four-scale Detect only')
        raw = []
        for i, base in enumerate(x):
            cls, reg = self.task_blocks[i](base) if self.enabled and i < 2 else (base, base)
            raw.append(torch.cat((self.cv2[i](reg), self.cv3[i](cls)), 1))
        if self.training:
            return raw
        value = self._inference(raw)  # original DFL, anchor coordinates and decode
        return value if self.export else (value, raw)


def attach(model):
    head = model.model[-1]
    if type(head) is not Detect or len(model.model) != 27 or head.nl != 4 or head.reg_max != 16:
        raise RuntimeError('Attach requires untouched E1 Detect model.26')
    if list(head.f) != [25, 16, 19, 22] or head.stride.tolist() != [4., 8., 16., 32.]:
        raise RuntimeError('Unexpected E1 feature sources/strides')
    if [tower[0].conv.in_channels for tower in head.cv2] != [128, 128, 256, 512]:
        raise RuntimeError('Unexpected original prediction tower channels')
    if next(head.parameters()).device.type != 'cpu':
        raise RuntimeError('Attach on CPU before moving to CUDA')
    with torch.random.fork_rng(devices=[]):
        # Do not call torch.manual_seed here: it also changes CUDA RNG.
        torch.random.default_generator.manual_seed(40001)
        blocks = nn.ModuleList(ChannelTaskBlock() for _ in range(2))
    head.__class__ = ChannelTaskDetect  # preserve every original tower parameter/buffer/key
    head.task_blocks = blocks
    head.enabled = True
    head.np = sum(p.numel() for p in head.parameters())
    head.type = 'channel_task_head.ChannelTaskDetect'
    if sum(p.numel() for p in blocks.parameters()) != NEW_PARAMETERS:
        raise RuntimeError('E41 topology/parameter count drift')
    return model


def branch(model):
    return model.model[-1].task_blocks


def original_to_target(name):
    return name


def is_new(name):
    return name.startswith(NEW_PREFIX)


def output_magnitudes(model):
    return [float(p.weight.detach().abs().sum()) for block in branch(model)
            for p in (block.out_cls, block.out_reg)]
