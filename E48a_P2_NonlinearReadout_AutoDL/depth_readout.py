"""Depth-specific linear readout expressed as zero-start depth contrasts.

The basis is a symmetric reparameterization, NOT a novel attention operator.
No new normalization, activation, router, auxiliary loss, or detach is added.
"""
import torch
from torch import nn
from parent_spatial_task_head import SpatialTaskBlock


CONTRAST_BASIS = ((.5, .5, .5), (.5, -.5, -.5),
                  (-.5, .5, -.5), (-.5, -.5, .5))
ADDED_PARAMETERS = 2 * 2 * 3 * 48 * 128


def fp32_project(layer, value):
    # Keep actual registered Conv modules/parameters (optimizer, hooks, THOP).
    # Stateless FP32 evaluation also supports half EMA without in-place casts.
    with torch.autocast(device_type=value.device.type, enabled=False):
        if layer.weight.dtype == torch.float32:
            return layer(value.float())
        parameters = {name: p.float() for name, p in layer.named_parameters()}
        return torch.func.functional_call(layer, parameters, (value.float(),), strict=True)


class DepthReadoutBlock(SpatialTaskBlock):
    def forward(self, x):
        cls, reg, weights, bank = self.decompose(x)  # original E40, exactly once
        outputs = (x + self.out_cls(cls), x + self.out_reg(reg))
        if not self.readout_enabled:
            return outputs
        stacked = torch.stack(bank, 1)
        result = []
        for task, layers in enumerate((self.readout_cls, self.readout_reg)):
            with torch.autocast(device_type=x.device.type, enabled=False):
                weighted = stacked.float() * weights[:, task, :, None].float()
                update = None
                for j, layer in enumerate(layers):
                    contrast = sum(self.contrast_basis[k, j].float() * weighted[:, k]
                                   for k in range(4))
                    delta = fp32_project(layer, contrast)
                    update = delta if update is None else update + delta
                result.append((outputs[task].float() + update).to(outputs[task].dtype))
        return tuple(result)


def install_readouts(block):
    if type(block) is not SpatialTaskBlock:
        raise RuntimeError('Install DRF on the untouched E40 block only')
    if next(block.parameters()).device.type != 'cpu':
        raise RuntimeError('Install on CPU, then move the WHOLE model to CUDA')
    block.__class__ = DepthReadoutBlock
    for task in ('cls', 'reg'):
        layers = nn.ModuleList(nn.Conv2d(48, 128, 1, bias=False) for _ in range(3))
        for layer in layers:
            nn.init.zeros_(layer.weight)
        setattr(block, 'readout_' + task, layers)
    block.register_buffer('contrast_basis', torch.tensor(CONTRAST_BASIS))
    block.readout_enabled = True


def added_conv_flops(imgsz=640):
    # Two tasks, three readouts per task, both P2/P3. MAC = two FLOPs.
    if imgsz % 32:
        raise ValueError('Profiling size must be divisible by 32')
    return sum(2 * (imgsz // stride)**2 * 2 * 3 * 48 * 128 for stride in (4, 8))
