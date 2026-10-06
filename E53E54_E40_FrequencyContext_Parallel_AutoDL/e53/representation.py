"""Full E40 plus one zero-output residual; no new loss or detector algorithm.

The normalized undecimated Haar frame is defined explicitly here. Circular
shifts are applied ONLY after a 12-cell replicated halo; the final crop keeps
wraparound outside the receptive field. This is not a decimated WTConv.
"""
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.modules.block import C3k2

VARIANT = 'e53'  # mechanically specialized when building the two packages
SITE = 25 if VARIANT == 'e53' else 16
HIDDEN = 32 if VARIANT == 'e53' else 48
ADDED_PARAMETERS = 18694 if VARIANT == 'e53' else 16665
HALO = 12


def conv(m, x):
    return F.conv2d(x, m.weight.float(), None if m.bias is None else m.bias.float(),
                    m.stride, m.padding, m.dilation, m.groups)


def normalized(m, x):
    return F.group_norm(x, m.num_groups, m.weight.float(), m.bias.float(), m.eps)


def haar_analysis(x, distance):
    """LL, horizontal difference, vertical difference, diagonal difference."""
    right = torch.roll(x, -distance, -1)
    down = torch.roll(x, -distance, -2)
    diagonal = torch.roll(down, -distance, -1)
    return ((x + right + down + diagonal) * .25,
            (x - right + down - diagonal) * .25,
            (x + right - down - diagonal) * .25,
            (x - right - down + diagonal) * .25)


def haar_synthesis(bands, distance):
    """Adjoint of the normalized redundant tight frame: A^T A = I."""
    ll, horizontal, vertical, diagonal = bands
    a = ll + horizontal + vertical + diagonal
    b = ll - horizontal + vertical - diagonal
    c = ll + horizontal - vertical - diagonal
    d = ll - horizontal - vertical + diagonal
    return (a + torch.roll(b, distance, -1) + torch.roll(c, distance, -2) +
            torch.roll(torch.roll(d, distance, -2), distance, -1)) * .25


class FrequencyResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = nn.Conv2d(128, 32, 1, bias=False)
        self.norm = nn.GroupNorm(4, 32)
        self.local = nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False)
        self.low = nn.Conv2d(32, 32, 5, padding=4, dilation=2, groups=32, bias=False)
        self.high = nn.ModuleList(nn.Conv2d(96, 96, 5, padding=2*d, dilation=d,
                                           groups=96, bias=False) for d in (1, 2))
        self.selectors = nn.ModuleList(nn.Sequential(
            nn.Conv2d(128, 16, 1), nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16), nn.SiLU(),
            nn.Conv2d(16, 3, 1)) for _ in range(2))
        self.output = nn.Conv2d(32, 128, 1, bias=False)
        nn.init.zeros_(self.output.weight)
        self.enabled = True
        self.capture_stats = False
        self.last_statistics = None

    def features(self, source):
        z = F.silu(normalized(self.norm, conv(self.reduce, source)))
        padded = F.pad(z, (HALO,)*4, mode='replicate')
        levels = []
        low = padded
        weights = []
        for i, distance in enumerate((1, 2)):
            low, *details = haar_analysis(low, distance)
            descriptor = torch.cat((low, *(v.abs() for v in details)), 1)
            selector = self.selectors[i]
            gate = conv(selector[4], F.silu(conv(selector[2], F.silu(conv(selector[0], descriptor))))).sigmoid()
            processed = conv(self.high[i], torch.cat(details, 1)).chunk(3, 1)
            levels.append(tuple(v*gate[:,j:j+1] for j,v in enumerate(processed)))
            weights.append(gate)
        low = conv(self.low, low)
        for i in (1, 0):
            low = haar_synthesis((low, *levels[i]), 2**i)
        reconstructed = low[..., HALO:-HALO, HALO:-HALO]
        return reconstructed + conv(self.local, z), weights

    def forward(self, x):
        if not self.enabled: return x
        if x.ndim != 4 or x.shape[1] != 128: raise RuntimeError('Expected Bx128xHxW')
        with torch.autocast(device_type=x.device.type, enabled=False):
            feature, weights = self.features(x.float())
            delta = conv(self.output, feature)
            if self.capture_stats:
                self.last_statistics = dict(shape=list(x.shape),
                    delta_mean_abs=float(delta.detach().abs().mean()),
                    selection_means=[w.detach().mean((0,2,3)).tolist() for w in weights],
                    halo=HALO, levels=2, full_resolution=True)
                self.capture_stats = False
            return (x.float() + delta).to(x.dtype)


class ContextResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = nn.Conv2d(128, 48, 1, bias=False)
        self.norm = nn.GroupNorm(6, 48)
        self.local = nn.Conv2d(48, 48, 3, padding=1, groups=48, bias=False)
        self.near = nn.Conv2d(48, 48, 5, padding=2, groups=48, bias=False)
        self.far = nn.Conv2d(48, 48, 7, padding=9, dilation=3, groups=48, bias=False)
        self.selector = nn.Conv2d(2, 3, 7, padding=3)
        self.output = nn.Conv2d(48, 128, 1, bias=False)
        nn.init.zeros_(self.output.weight)
        self.enabled = True
        self.capture_stats = False
        self.last_statistics = None

    def features(self, source):
        z = F.silu(normalized(self.norm, conv(self.reduce, source)))
        near = conv(self.near, z)
        far = conv(self.far, near)
        context = torch.cat((near, far), 1)
        descriptor = torch.cat((context.mean(1, keepdim=True), context.amax(1, keepdim=True)), 1)
        weights = conv(self.selector, descriptor).softmax(1)  # bypass, near, far
        feature = conv(self.local, z) + weights[:,1:2]*near + weights[:,2:3]*far
        return feature, weights

    def forward(self, x):
        if not self.enabled: return x
        if x.ndim != 4 or x.shape[1] != 128: raise RuntimeError('Expected Bx128xHxW')
        with torch.autocast(device_type=x.device.type, enabled=False):
            feature, weights = self.features(x.float())
            delta = conv(self.output, feature)
            if self.capture_stats:
                self.last_statistics = dict(shape=list(x.shape),
                    delta_mean_abs=float(delta.detach().abs().mean()),
                    selection_means=weights.detach().mean((0,2,3)).tolist(),
                    selector_order=['bypass','near','far'])
                self.capture_stats = False
            return (x.float() + delta).to(x.dtype)


class ResidualC3k2(C3k2):
    def forward(self, x):
        return self.representation(super().forward(x))


def install(model):
    stage = model.model[SITE]
    if type(stage) is not C3k2 or stage.i != SITE or stage.f != -1 or stage.cv2.conv.out_channels != 128:
        raise RuntimeError('Wrong E1 C3k2 attachment site')
    if next(stage.parameters()).device.type != 'cpu': raise RuntimeError('Attach on CPU first')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(53001 if VARIANT == 'e53' else 54001)
        module = FrequencyResidual() if VARIANT == 'e53' else ContextResidual()
    if sum(p.numel() for p in module.parameters()) != ADDED_PARAMETERS:
        raise RuntimeError('New branch parameter arithmetic mismatch')
    stage.__class__ = ResidualC3k2
    stage.representation = module
    stage.type = 'representation.ResidualC3k2'
    stage.np = sum(p.numel() for p in stage.parameters())
    model.representation_topology = dict(variant=VARIANT, site=SITE, stride=4 if SITE==25 else 8,
        channels=128, hidden=HIDDEN, original_C3k2_preserved=True, complete_E40_preserved=True,
        branch_compute='FP32', only_output_zero=True)
    return model
