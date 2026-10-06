"""P2-only nonlinear joint readout of the EXISTING E47 signed contrasts."""
import torch
from torch import nn
from depth_readout import DepthReadoutBlock, fp32_project

ADDED_PARAMETERS = 2 * (144 * 32 + 32 * 128)


class NonlinearReadoutBlock(DepthReadoutBlock):
    def forward(self, x):
        cls, reg, weights, bank = self.decompose(x)  # same E40 encoder/router, once
        outputs = (x + self.out_cls(cls), x + self.out_reg(reg))
        if not self.readout_enabled:
            return outputs  # E47 bypass semantics retained
        stacked = torch.stack(bank, 1)
        result = []
        for task, (layers, joint) in enumerate(((self.readout_cls, self.nonlinear_cls),
                                               (self.readout_reg, self.nonlinear_reg))):
            with torch.autocast(device_type=x.device.type, enabled=False):
                weighted = stacked.float() * weights[:, task, :, None].float()
                contrasts = []
                update = None
                for j, layer in enumerate(layers):
                    contrast = sum(self.contrast_basis[k, j].float() * weighted[:, k]
                                   for k in range(4))
                    contrasts.append(contrast)
                    delta = fp32_project(layer, contrast)
                    update = delta if update is None else update + delta
                # Round the original E47 sum exactly as its native path did.
                original = (outputs[task].float() + update).to(outputs[task].dtype)
                if self.nonlinear_enabled:
                    hidden = joint[1](fp32_project(joint[0], torch.cat(contrasts, 1)))
                    delta = fp32_project(joint[2], hidden)
                    original = (original.float() + delta).to(original.dtype)
                result.append(original)
        return tuple(result)


def install_nonlinear(block):
    if type(block) is not DepthReadoutBlock:
        raise RuntimeError('Install only on the untouched E47 P2 block')
    if next(block.parameters()).device.type != 'cpu':
        raise RuntimeError('Install on CPU before moving the whole model to CUDA')
    block.__class__ = NonlinearReadoutBlock
    for task in ('cls', 'reg'):
        layers = nn.Sequential(nn.Conv2d(144, 32, 1, bias=False), nn.SiLU(),
                               nn.Conv2d(32, 128, 1, bias=False))
        nn.init.zeros_(layers[2].weight)
        setattr(block, 'nonlinear_' + task, layers)
    block.nonlinear_enabled = True


def added_conv_flops(imgsz=640):
    if imgsz % 32:
        raise ValueError('Input size must be divisible by 32')
    return 2 * (imgsz // 4) ** 2 * ADDED_PARAMETERS
