"""Existing DCNv2 sampling, confined to one P2 regression Conv. No silent fallback."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import deform_conv2d

SITE = 'model.26.cv2.0.1'


class E36SamplingConv(nn.Module):
    def __init__(self, base, variant):
        super().__init__()
        c = base.conv
        if (variant not in ('a', 'b') or c.in_channels != 64 or c.out_channels != 64
                or c.kernel_size != (3, 3) or c.stride != (1, 1)
                or c.padding != (1, 1) or c.groups != 1 or c.dilation != (1, 1)):
            raise ValueError('Expected original P2 cv2[0][1] Conv64->64 3x3 stride1')
        self.base = base
        self.variant = variant
        self.offset_bound = .5
        self.descriptor = nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=True)
        self.mask_head = nn.Conv2d(64, 9, 1, bias=True)
        self.offset_head = nn.Conv2d(64, 18, 1, bias=True) if variant == 'b' else None
        self.reset_new_parameters()

    def reset_new_parameters(self):
        self.descriptor.reset_parameters()
        nn.init.zeros_(self.mask_head.weight)
        nn.init.zeros_(self.mask_head.bias)
        if self.offset_head is not None:
            nn.init.zeros_(self.offset_head.weight)
            nn.init.zeros_(self.offset_head.bias)

    def forward(self, x):
        # Both arms use the SAME FP32 operator island. This is not a claim of
        # bit-exact equivalence to E1's autocast convolution. Casts preserve grads.
        with torch.autocast(device_type=x.device.type, enabled=False):
            h = F.silu(F.conv2d(x.float(), self.descriptor.weight.float(),
                               self.descriptor.bias.float(), padding=1, groups=64))
            logits = F.conv2d(h, self.mask_head.weight.float(), self.mask_head.bias.float())
            mask = 2 * logits.sigmoid()
            if self.offset_head is None:
                offsets = h.new_zeros((h.shape[0], 18, h.shape[2], h.shape[3]))
            else:
                offsets = self.offset_bound * F.conv2d(
                    h, self.offset_head.weight.float(), self.offset_head.bias.float()).tanh()
            conv = self.base.conv
            y = deform_conv2d(x.float(), offsets, conv.weight.float(),
                              conv.bias.float() if conv.bias is not None else None,
                              padding=(1, 1), mask=mask)
        y = y.to(x.dtype)
        if hasattr(self.base, 'bn'):
            y = self.base.bn(y)
        return self.base.act(y)


def attach(model, variant):
    if any(isinstance(m, E36SamplingConv) for m in model.modules()):
        raise RuntimeError('Sampling wrapper already attached')
    tower = model.model[26].cv2[0]
    # Creating new branches must not advance the baseline augmentation RNG.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(36001)
        tower[1] = E36SamplingConv(tower[1], variant)
        # B constructs an extra head; reset the shared descriptor from the same
        # seed AFTER all constructors, so A/B new common tensors also match.
        torch.manual_seed(36001)
        tower[1].reset_new_parameters()
    return model


def original_to_target(key):
    prefix = SITE + '.'
    return prefix + 'base.' + key[len(prefix):] if key.startswith(prefix) else key


def is_new_key(key):
    return any(key.startswith(SITE + '.' + name + '.')
               for name in ('descriptor', 'mask_head', 'offset_head'))
