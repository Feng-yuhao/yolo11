"""E60a: fixed E58 W96 + residual hierarchical group encoding at P2/P3.

Original depth-chain inputs are NOT overwritten. Only read-bank copies
Z1/Z2/Z3 receive a zero-initialized residual. Parent spatial routing, Detect
towers and loss are unchanged. Group hierarchy borrows Res2Net's idea.
"""
import torch
from torch import nn
import e58_spatial_task_head as e58

CHANNELS = e58.CHANNELS
VARIANT = 'e60'
ADDED_PARAMETERS = 149472
NEW_PARAMETERS = e58.COUNTS['e58'] + ADDED_PARAMETERS
NEW_PREFIX = e58.NEW_PREFIX


def configure(variant):
    if variant != VARIANT:
        raise ValueError('This package implements only E60a seed1')
    e58.configure('e58')


class GroupResidualEncoder(nn.Module):
    """Four 24-channel groups, FP32 sidecar inside the unchanged AMP parent.

    Nested zero-init projections can learn values below FP16's smallest
    representable value. Casting such weights before convolution silently
    removes the inner-kernel gradient. Keep this NEW sidecar in FP32, including
    when the original Trainer stores an EMA.half() checkpoint.
    """
    def __init__(self):
        super().__init__()
        self.kernels = nn.ModuleList(
            nn.Sequential(nn.Conv2d(24, 24, 3, padding=1, bias=False),
                          nn.GroupNorm(6, 24), nn.SiLU()) for _ in range(3))
        self.project = nn.Conv2d(96, 96, 1, bias=False)
        nn.init.zeros_(self.project.weight)

    def _apply(self, fn, recurse=True):
        def retain_precision(tensor):
            converted = fn(tensor)
            if tensor.is_floating_point() and converted.dtype != torch.float32:
                # Use the ORIGINAL tensor; converted.float() would retain
                # quantization damage from .half()/.bfloat16(). Device moves
                # still follow fn, including CPU checkpoint serialization.
                return tensor.to(device=converted.device, dtype=torch.float32)
            return converted
        return super()._apply(retain_precision, recurse=recurse)

    def delta(self, z):
        if z.shape[1] != 96:
            raise RuntimeError('Group residual encoder requires hidden96')
        with torch.autocast(device_type=z.device.type, enabled=False):
            u = z.float().chunk(4, dim=1)
            v = [u[0]]
            for j, kernel in enumerate(self.kernels):
                value = u[j+1] if j == 0 else u[j+1] + v[-1]
                v.append(kernel(value))
            return self.project(torch.cat(v, dim=1))

    def forward(self, z):
        # Preserve the E58 bank/router/tower dtype; do not promote the parent
        # feature path or change its spatial softmax precision.
        return z + self.delta(z).to(dtype=z.dtype)


class GroupResidualTaskBlock(e58.WideTaskBlock):
    def original_bank(self, x):
        bank = [self.reduce(x)]
        for layer in self.depths:
            bank.append(layer(bank[-1]))
        return bank

    def decompose(self, x):
        bank = self.original_bank(x)
        # Modified copies never feed the next ORIGINAL encoding depth.
        bank = [bank[0]] + [r(z) for r, z in zip(self.refinements, bank[1:])]
        b, _, h, w = x.shape
        logits = self.router(torch.cat(bank, 1)).reshape(b, 2, 4, h, w)
        weights = logits.softmax(dim=2)
        stacked = torch.stack(bank, dim=1)
        cls = (stacked * weights[:, 0, :, None]).sum(dim=1)
        reg = (stacked * weights[:, 1, :, None]).sum(dim=1)
        return cls, reg, weights, bank


class SpatialTaskDetect(e58.SpatialTaskDetect):
    """Four-scale E58 forward inherited verbatim."""
    pass


def attach(model):
    e58.configure('e58')
    e58.attach(model)
    head = model.model[-1]
    # Parent initialized FIRST. Isolated CPU RNG preserves training streams.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(60001)
        for block in head.task_blocks[:2]:
            block.__class__ = GroupResidualTaskBlock
            block.refinements = nn.ModuleList(GroupResidualEncoder() for _ in range(3))
    head.__class__ = SpatialTaskDetect
    head.experiment_variant = VARIANT
    head.np = sum(p.numel() for p in head.parameters())
    head.type = 'spatial_task_head.SpatialTaskDetect'
    if sum(p.numel() for p in head.task_blocks.parameters()) != NEW_PARAMETERS:
        raise RuntimeError('E60 branch parameter count drift')
    return model


def branch(model):
    return model.model[-1].task_blocks


def branch_groups():
    return e58.branch_groups() + [
        f'{i}.refinements.{d}.{part}.' for i in range(2) for d in range(3)
        for part in ('kernels.0', 'kernels.1', 'kernels.2', 'project')]


def output_magnitudes(model):
    return e58.output_magnitudes(model) + [
        float(r.project.weight.detach().abs().sum())
        for block in branch(model)[:2] for r in block.refinements]


original_to_target = e58.original_to_target
is_new = e58.is_new
