"""E35b: preserve the original local Bottleneck, add a PKI delta.

This is an integration hypothesis, not a demonstrated root-cause repair.
PKI internals remain the exact frozen E35a adaptation of PKINet (Apache-2.0).
"""
from torch import nn
from .block import Bottleneck, C3k, C3k2
from .e35a_pki import E35PKIBlock


class E35LocalPKI(nn.Module):
    def __init__(self, base):
        super().__init__()
        if type(base) is not Bottleneck or not base.add:
            raise ValueError('E35b requires the original equal-channel residual Bottleneck')
        self.base = base
        self.channels = base.cv2.conv.out_channels
        self.pki = E35PKIBlock(self.channels)
        # No BN/activation/bias and no extra zero-valued multiplying gate.
        self.project = nn.Conv2d(self.channels, self.channels, 1, bias=False)
        self.reset_new_parameters()

    def reset_new_parameters(self):
        self.pki.reset_parameters()
        nn.init.zeros_(self.project.weight)

    def forward(self, x):
        return self.base(x) + self.project(self.pki(x) - x)


class C3k2LocalPKI(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        if g != 1 or not shortcut:
            raise ValueError('E35b supports only the audited E1 configuration')
        for i, old in enumerate(self.m):
            if type(old) is Bottleneck:
                self.m[i] = E35LocalPKI(old)
            elif type(old) is C3k:
                for j, leaf in enumerate(old.m):
                    if type(leaf) is not Bottleneck:
                        raise TypeError('Unexpected C3k leaf')
                    old.m[j] = E35LocalPKI(leaf)
            else:
                raise TypeError(f'Unexpected C3k2 member: {type(old)}')
