"""E52: a local-path-preserving stride-8 context residual.

Linear spatial reduction (full-resolution Q, pooled K/V) follows PVTv2.
Not a reproduction of the PVT backbone, not a claim of novel attention.
All branch arithmetic is FP32, including stored-half EMA inference. The only
zero initialization is the output projection; no second zero residual gain.
"""
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.modules.block import C3k2

CHANNELS, HIDDEN, HEADS, POOL = 256, 64, 4, 7


def conv(module, x):
    return F.conv2d(x, module.weight.float(),
                    None if module.bias is None else module.bias.float(),
                    module.stride, module.padding, module.dilation, module.groups)


def linear(module, x):
    return F.linear(x, module.weight.float(),
                    None if module.bias is None else module.bias.float())


def norm(module, x):
    return F.layer_norm(x, module.normalized_shape, module.weight.float(),
                        module.bias.float(), module.eps)


class GlobalContextResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = nn.Conv2d(CHANNELS, HIDDEN, 1, bias=False)
        self.position = nn.Conv2d(HIDDEN, HIDDEN, 3, padding=1, groups=HIDDEN, bias=False)
        self.query_norm = nn.LayerNorm(HIDDEN)
        self.query = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.pool_project = nn.Conv2d(HIDDEN, HIDDEN, 1, bias=False)
        self.pool_norm = nn.LayerNorm(HIDDEN)
        self.key_value = nn.Linear(HIDDEN, 2 * HIDDEN, bias=False)
        self.attn_project = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.ffn_norm = nn.LayerNorm(HIDDEN)
        self.ffn_in = nn.Linear(HIDDEN, 2 * HIDDEN, bias=False)
        self.ffn_local = nn.Conv2d(2 * HIDDEN, 2 * HIDDEN, 3, padding=1,
                                   groups=2 * HIDDEN, bias=False)
        self.ffn_out = nn.Linear(2 * HIDDEN, HIDDEN, bias=False)
        self.output = nn.Conv2d(HIDDEN, CHANNELS, 1, bias=False)
        nn.init.zeros_(self.output.weight)
        self.enabled = True
        self.capture_stats = False
        self.last_statistics = None

    def attention(self, tokens, h, w, return_weights=False):
        b, n, c = tokens.shape
        if (n, c) != (h*w, HIDDEN):
            raise RuntimeError('Incorrect spatial token layout')
        q = linear(self.query, norm(self.query_norm, tokens))
        pooled = F.adaptive_avg_pool2d(tokens.transpose(1, 2).reshape(b, c, h, w), (POOL, POOL))
        pooled = conv(self.pool_project, pooled).flatten(2).transpose(1, 2)
        pooled = F.gelu(norm(self.pool_norm, pooled))
        kv = linear(self.key_value, pooled).reshape(b, POOL*POOL, 2, HEADS, c//HEADS)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        q = q.reshape(b, n, HEADS, c//HEADS).transpose(1, 2)
        weights = ((q @ k.transpose(-1, -2)) * (c//HEADS)**-.5).softmax(-1)
        value = (weights @ v).transpose(1, 2).reshape(b, n, c)
        value = linear(self.attn_project, value)
        return (value, weights) if return_weights else value

    def forward(self, x):
        if not self.enabled:
            return x
        if x.ndim != 4 or x.shape[1] != CHANNELS:
            raise RuntimeError('E52 branch requires Bx256xHxW')
        with torch.autocast(device_type=x.device.type, enabled=False):
            source = x.float()
            z = conv(self.reduce, source)
            z = z + conv(self.position, z)
            b, c, h, w = z.shape
            tokens = z.flatten(2).transpose(1, 2)
            tokens = tokens + self.attention(tokens, h, w)
            feed = linear(self.ffn_in, norm(self.ffn_norm, tokens))
            feed = feed.transpose(1, 2).reshape(b, 2*c, h, w)
            feed = F.gelu(conv(self.ffn_local, feed)).flatten(2).transpose(1, 2)
            tokens = tokens + linear(self.ffn_out, feed)
            delta = conv(self.output, tokens.transpose(1, 2).reshape(b, c, h, w))
            result = (source + delta).to(x.dtype)
            if self.capture_stats:
                self.last_statistics = dict(input_mean_abs=float(source.detach().abs().mean()),
                                            delta_mean_abs=float(delta.detach().abs().mean()),
                                            shape=list(x.shape), context_tokens=POOL*POOL)
                self.capture_stats = False
            return result


class ContextC3k2(C3k2):
    def forward(self, x):
        return self.global_context(super().forward(x))


def install(model):
    stage = model.model[4]
    if type(stage) is not C3k2 or stage.i != 4 or stage.f != -1 or stage.cv2.conv.out_channels != CHANNELS:
        raise RuntimeError('Expected untouched E1 backbone model.4 C3k2, C256')
    if next(stage.parameters()).device.type != 'cpu':
        raise RuntimeError('Install on CPU before CUDA placement')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(52001)
        module = GlobalContextResidual()
    stage.__class__ = ContextC3k2
    stage.global_context = module
    stage.type = 'global_context.ContextC3k2'
    stage.np = sum(p.numel() for p in stage.parameters())
    model.e52_topology = dict(site=4, stride=8, channels=CHANNELS, hidden=HIDDEN,
                             heads=HEADS, pool=POOL, layers=1, original_C3k2_preserved=True,
                             complete_E40_preserved=True, branch_compute='FP32')
    return model


# A constant, independently checked by unit tests: imports must not consume RNG.
ADDED_PARAMETERS = 71744
