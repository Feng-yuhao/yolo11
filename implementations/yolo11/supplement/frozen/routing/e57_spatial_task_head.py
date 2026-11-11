"""E57: original E40 P2/P3 plus independent hidden-48 P4/P5 blocks.

Only the scope of task/depth preparation changes. Towers, DFL and decode
remain the pinned E1 implementation. No extra gate, detach or loss.
"""
import torch
from torch import nn
import parent_spatial_task_head as parent

CHANNELS = (128, 128, 256, 512)
NEW_PREFIX = parent.NEW_PREFIX
P4_P5_PARAMETERS = 249168
NEW_PARAMETERS = parent.NEW_PARAMETERS + P4_P5_PARAMETERS


class DeepScaleTaskBlock(parent.SpatialTaskBlock):
    """Same E40 operators, with only the first/last channel shapes adapted."""
    def __init__(self, cin):
        # Do not instantiate/discard the 128-channel parent: that consumes RNG.
        nn.Module.__init__(self)
        self.reduce = parent.encoder(cin, 48, 1)
        self.depths = nn.ModuleList(parent.encoder(48, 48, 3) for _ in range(3))
        self.router = nn.Sequential(nn.Conv2d(192, 32, 1, bias=False), nn.SiLU(),
                                    nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
                                    nn.SiLU(), nn.Conv2d(32, 8, 1))
        self.out_cls = nn.Conv2d(48, cin, 1, bias=False)
        self.out_reg = nn.Conv2d(48, cin, 1, bias=False)
        nn.init.zeros_(self.out_cls.weight)
        nn.init.zeros_(self.out_reg.weight)


class SpatialTaskDetect(parent.SpatialTaskDetect):
    def forward(self, x):
        if self.end2end or self.nl != 4 or len(x) != 4 or len(self.task_blocks) != 4:
            raise RuntimeError('E57 requires four independent P2/P3/P4/P5 task blocks')
        raw = []
        for i, base in enumerate(x):
            cls, reg = self.task_blocks[i](base) if self.enabled else (base, base)
            raw.append(torch.cat((self.cv2[i](reg), self.cv3[i](cls)), 1))
        if self.training:
            return raw
        value = self._inference(raw)
        return value if self.export else (value, raw)


def attach(model):
    # This exact, unmodified parent creates P2/P3 under its original 40001 seed.
    parent.attach(model)
    head = model.model[-1]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(57001)
        extra = nn.ModuleList(DeepScaleTaskBlock(c) for c in CHANNELS[2:])
    if sum(p.numel() for p in extra.parameters()) != P4_P5_PARAMETERS:
        raise RuntimeError('P4/P5 parameter count drift')
    head.task_blocks.extend(extra)
    head.__class__ = SpatialTaskDetect
    head.np = sum(p.numel() for p in head.parameters())
    head.type = 'spatial_task_head.SpatialTaskDetect'
    if sum(p.numel() for p in head.task_blocks.parameters()) != NEW_PARAMETERS:
        raise RuntimeError('E57 branch topology/parameter count drift')
    return model


def branch(model):
    return model.model[-1].task_blocks


def branch_groups():
    return [f'{i}.{part}.' for i in range(4)
            for part in ('reduce', 'depths.0', 'depths.1', 'depths.2',
                         'router', 'out_cls', 'out_reg')]


original_to_target = parent.original_to_target
is_new = parent.is_new
output_magnitudes = parent.output_magnitudes
