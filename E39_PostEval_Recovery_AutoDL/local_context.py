"""E39: localization context selection inside the original P2 regression tower.

An adaptation of selective-receptive-field ideas, not a reproduction of LSKNet.
No new box decoder, loss, sampling coordinates, auxiliary targets or P1 input.
"""
import torch
from torch import nn
from torch.nn import functional as F

SITE = 'model.26.cv2.0.1'


class LocalContext(nn.Module):
    def __init__(self, channels=64, hidden=32):
        super().__init__()
        self.reduce = nn.Conv2d(channels, hidden, 1, bias=False)
        self.norm = nn.GroupNorm(4, hidden)
        self.near = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.context = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden, bias=False)
        self.selector = nn.Conv2d(4, 2, 3, padding=1)
        self.project = nn.Conv2d(2*hidden, channels, 1, bias=False)
        nn.init.zeros_(self.selector.weight)
        nn.init.zeros_(self.selector.bias)
        nn.init.zeros_(self.project.weight)

    def forward(self, x):
        z = F.silu(self.norm(self.reduce(x)))
        near = F.silu(self.near(z))
        context = F.silu(self.context(near))  # 7x7 support relative to z
        descriptor = torch.cat((near.mean(1, keepdim=True), near.amax(1, keepdim=True),
                                context.mean(1, keepdim=True), context.amax(1, keepdim=True)), 1)
        gates = self.selector(descriptor).softmax(1)
        return self.project(torch.cat((near*gates[:, :1], context*gates[:, 1:]), 1))


class ContextConv(nn.Module):
    def __init__(self, base):
        super().__init__()
        c = base.conv
        if (c.in_channels, c.out_channels, c.kernel_size, c.stride, c.groups) != (64, 64, (3, 3), (1, 1), 1):
            raise ValueError('Expected E1 P2 second regression Conv64/64 3x3')
        self.base = base  # Original Conv/BN/SiLU, including pretrained tensors.
        self.local_context = LocalContext()
        self.enabled = True

    def forward(self, x):
        if self.enabled:
            x = x + self.local_context(x)  # out-of-place: shared classification input is untouched
        return self.base(x)


def attach(model):
    if any(isinstance(m, ContextConv) for m in model.modules()):
        raise RuntimeError('E39 already attached')
    head = model.model[-1]
    if len(model.model) != 27 or head.nl != 4 or head.reg_max != 16:
        raise RuntimeError('Requires original four-level E1')
    # Adding parameters does not advance the baseline training/augmentation RNG.
    if head.cv2[0][1].conv.weight.device.type != 'cpu':
        raise RuntimeError('Attach E39 on CPU before moving the model to CUDA')
    with torch.random.fork_rng(devices=[]):
        # torch.manual_seed also seeds CUDA; fork_rng(devices=[]) restores only CPU.
        torch.random.default_generator.manual_seed(39001)
        head.cv2[0][1] = ContextConv(head.cv2[0][1])
    return model


def original_to_target(name):
    prefix = SITE + '.'
    return prefix + 'base.' + name[len(prefix):] if name.startswith(prefix) else name


def branch(model):
    return model.model[-1].cv2[0][1].local_context


def is_new(name):
    return name.startswith(SITE + '.local_context.')
