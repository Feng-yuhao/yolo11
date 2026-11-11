"""E58: E57 all-scale hidden96; E59: E57 + independent P2/P3 channel readout.

This is not E42's softmax(S+C), nor a guarantee of independent training effects.
Original Detect towers, feature channels, DFL, decode, TAL and loss stay intact.
"""
import torch
from torch import nn
from torch.nn import functional as F
import parent_spatial_task_head as parent
import e57_spatial_task_head as e57

CHANNELS = (128, 128, 256, 512)
COUNTS = {'e58': 1344672, 'e59': 486816}
VARIANT = 'e58'
NEW_PARAMETERS = COUNTS[VARIANT]
NEW_PREFIX = parent.NEW_PREFIX


def configure(variant):
    global VARIANT, NEW_PARAMETERS
    if variant not in COUNTS:
        raise ValueError('Choose e58 or e59')
    VARIANT, NEW_PARAMETERS = variant, COUNTS[variant]


class WideTaskBlock(parent.SpatialTaskBlock):
    def __init__(self, cin):
        nn.Module.__init__(self)
        self.reduce = parent.encoder(cin, 96, 1)
        self.depths = nn.ModuleList(parent.encoder(96, 96, 3) for _ in range(3))
        self.router = nn.Sequential(nn.Conv2d(384, 32, 1, bias=False), nn.SiLU(),
                                    nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
                                    nn.SiLU(), nn.Conv2d(32, 8, 1))
        self.out_cls = nn.Conv2d(96, cin, 1, bias=False)
        self.out_reg = nn.Conv2d(96, cin, 1, bias=False)
        nn.init.zeros_(self.out_cls.weight)
        nn.init.zeros_(self.out_reg.weight)


class DualReadoutTaskBlock(parent.SpatialTaskBlock):
    """Original spatial path plus separately normalized channel/depth readout."""
    def channel_decompose(self, bank):
        # Pool in FP32 BEFORE the small routing MLP. Functional half-weight
        # fallback preserves gradient flow and protects pure-FP16 EMA inference.
        with torch.autocast(device_type=bank[0].device.type, enabled=False):
            descriptor = self.channel_pool(torch.cat(bank, 1).float())
            if self.channel_router[0].weight.dtype == torch.float32:
                logits = self.channel_router(descriptor)
            else:
                first, _, last = self.channel_router
                hidden = F.silu(F.conv2d(descriptor, first.weight.float()))
                logits = F.conv2d(hidden, last.weight.float(), last.bias.float())
            weights = logits.reshape(len(bank[0]), 2, 4, 48, 1, 1).softmax(2)
            # No spatial x channel joint weight tensor. Accumulate each task
            # independently; restore feature dtype before output projection.
            values = tuple(sum(bank[k].float() * weights[:, task, k]
                               for k in range(4)).to(bank[0].dtype) for task in range(2))
        return values[0], values[1], weights

    def forward(self, x):
        spatial_cls, spatial_reg, _, bank = self.decompose(x)
        channel_cls, channel_reg, _ = self.channel_decompose(bank)
        return (x + self.out_cls(spatial_cls) + self.out_channel_cls(channel_cls),
                x + self.out_reg(spatial_reg) + self.out_channel_reg(channel_reg))


class SpatialTaskDetect(e57.SpatialTaskDetect):
    """Four blocks, still four detection scales. Forward inherited verbatim."""
    pass


def initialize_channel_readout(block):
    block.__class__ = DualReadoutTaskBlock
    block.channel_pool = nn.AdaptiveAvgPool2d(1)
    block.channel_router = nn.Sequential(nn.Conv2d(192, 32, 1, bias=False),
                                         nn.SiLU(), nn.Conv2d(32, 384, 1))
    block.out_channel_cls = nn.Conv2d(48, 128, 1, bias=False)
    block.out_channel_reg = nn.Conv2d(48, 128, 1, bias=False)
    nn.init.zeros_(block.out_channel_cls.weight)
    nn.init.zeros_(block.out_channel_reg.weight)
    return block


def attach(model):
    # Original E57 initialization completes FIRST, preserving every common
    # tensor for E59 (including P3 and P4/P5 RNG streams).
    e57.attach(model)
    head = model.model[-1]
    if VARIANT == 'e58':
        # Shape-changing width experiment cannot preserve hidden48 weights.
        # Seed rules remain deterministic and isolated from training RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(40001)
            shallow = [WideTaskBlock(c) for c in CHANNELS[:2]]
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(57001)
            deep = [WideTaskBlock(c) for c in CHANNELS[2:]]
        head.task_blocks = nn.ModuleList(shallow + deep)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(59001)
            for block in head.task_blocks[:2]:
                initialize_channel_readout(block)
    head.__class__ = SpatialTaskDetect
    head.experiment_variant = VARIANT
    head.np = sum(p.numel() for p in head.parameters())
    head.type = 'spatial_task_head.SpatialTaskDetect'
    if sum(p.numel() for p in head.task_blocks.parameters()) != NEW_PARAMETERS:
        raise RuntimeError('Branch parameter count drift for ' + VARIANT)
    return model


def branch(model):
    return model.model[-1].task_blocks


def branch_groups():
    groups = e57.branch_groups()
    if VARIANT == 'e59':
        groups += [f'{i}.{part}.' for i in range(2)
                   for part in ('channel_router', 'out_channel_cls', 'out_channel_reg')]
    return groups


def output_magnitudes(model):
    values = parent.output_magnitudes(model)
    for block in branch(model)[:2]:
        if isinstance(block, DualReadoutTaskBlock):
            values.extend(float(p.weight.detach().abs().sum())
                          for p in (block.out_channel_cls, block.out_channel_reg))
    return values


original_to_target = parent.original_to_target
is_new = parent.is_new
