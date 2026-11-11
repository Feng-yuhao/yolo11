"""TOOD TaskDecomposition adaptation, NOT a complete TOOD detector.

TaskDecomposition math adapted from fcjian/TOOD, Apache-2.0, commit
1bda27c6f290c5b828c566eb95ea50fed75a53cb (author_source/tood_head.py).
The original sigmoid layer attention / learned channel recombination / GN32 /
ReLU and shared six-convolution feature bank are retained. YOLO-specific
interfaces: per-level 1x1 input adapters, width64, zero-initialized residual
output adapters into the UNCHANGED four native classification/regression towers.
No E60 group residual, spatial softmax router, TOOD alignment branch or new loss.
"""
import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.modules.head import Detect

CHANNELS = (128, 128, 256, 512)
WIDTH, DEPTHS = 64, 6
INIT_SEED = 64001
NEW_PREFIX = 'model.26.td_adapter.'


class ConvGNReLU(nn.Module):
    def __init__(self, cin, cout, kernel):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False)
        self.norm = nn.GroupNorm(32, cout)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activate(self.norm(self.conv(x)))


class TaskDecomposition(nn.Module):
    def __init__(self, feat_channels=WIDTH, stacked_convs=DEPTHS):
        super().__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = feat_channels * stacked_convs
        # Original TOODHead passes la_down_rate = stacked_convs * 8.
        hidden = self.in_channels // (stacked_convs * 8)
        self.la_conv1 = nn.Conv2d(self.in_channels, hidden, 1)
        self.relu = nn.ReLU(inplace=True)
        self.la_conv2 = nn.Conv2d(hidden, stacked_convs, 1)
        self.sigmoid = nn.Sigmoid()
        self.reduction_conv = ConvGNReLU(self.in_channels, feat_channels, 1)
        self.init_weights()

    def init_weights(self):
        for layer, std in ((self.la_conv1, .001), (self.la_conv2, .001),
                           (self.reduction_conv.conv, .01)):
            nn.init.normal_(layer.weight, std=std)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def layer_weights(self, feat, avg_feat=None):
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        return self.sigmoid(self.la_conv2(self.relu(self.la_conv1(avg_feat))))

    def forward(self, feat, avg_feat=None):
        b, c, h, w = feat.shape
        if c != self.in_channels:
            raise RuntimeError('TD feature bank channel mismatch')
        weight = self.layer_weights(feat, avg_feat)
        # Equivalent to weighting each depth then applying the dense 1x1
        # reduction. The reduction Conv2d is NOT called separately: count bmm.
        conv_weight = weight.reshape(b, 1, self.stacked_convs, 1) * \
            self.reduction_conv.conv.weight.reshape(1, self.feat_channels,
                                                     self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels, self.in_channels)
        value = torch.bmm(conv_weight, feat.reshape(b, c, h * w))
        value = value.reshape(b, self.feat_channels, h, w)
        return self.reduction_conv.activate(self.reduction_conv.norm(value))


class TDAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.ModuleList(ConvGNReLU(cin, WIDTH, 1) for cin in CHANNELS)
        # One shared set across all FPN levels, as in the author's TOODHead.
        self.inter = nn.ModuleList(ConvGNReLU(WIDTH, WIDTH, 3) for _ in range(DEPTHS))
        self.cls_td = TaskDecomposition()
        self.reg_td = TaskDecomposition()
        self.out_cls = nn.ModuleList(nn.Conv2d(WIDTH, cin, 1, bias=False) for cin in CHANNELS)
        self.out_reg = nn.ModuleList(nn.Conv2d(WIDTH, cin, 1, bias=False) for cin in CHANNELS)
        for module in list(self.pre) + list(self.inter):
            nn.init.normal_(module.conv.weight, std=.01)
        for module in list(self.out_cls) + list(self.out_reg):
            nn.init.zeros_(module.weight)

    def decompose(self, x, level):
        value = self.pre[level](x)
        bank = []
        for layer in self.inter:
            value = layer(value)
            bank.append(value)
        feat = torch.cat(bank, 1)  # SIX post-convolution depths, not raw z0.
        avg = F.adaptive_avg_pool2d(feat, (1, 1))
        return self.cls_td(feat, avg), self.reg_td(feat, avg)

    def forward(self, x, level):
        cls, reg = self.decompose(x, level)
        return x + self.out_cls[level](cls), x + self.out_reg[level](reg)


class TOODTDDetect(Detect):
    def forward(self, x):
        if self.end2end or self.nl != 4 or len(x) != 4:
            raise RuntimeError('Only pinned YOLO11s-P2 four-scale Detect is supported')
        raw = []
        for level, base in enumerate(x):
            cls, reg = self.td_adapter(base, level) if self.td_enabled else (base, base)
            raw.append(torch.cat((self.cv2[level](reg), self.cv3[level](cls)), 1))
        if self.training:
            return raw
        decoded = self._inference(raw)
        return decoded if self.export else (decoded, raw)


def attach(model):
    head = model.model[-1]
    if type(head) is not Detect or len(model.model) != 27 or head.nl != 4 or head.reg_max != 16:
        raise RuntimeError('Attach only to an untouched E1 model.26 Detect')
    if list(head.f) != [25, 16, 19, 22] or head.stride.tolist() != [4., 8., 16., 32.]:
        raise RuntimeError('E1 feature indices/strides changed')
    if tuple(v[0].conv.in_channels for v in head.cv2) != CHANNELS:
        raise RuntimeError('Original prediction tower input channels changed')
    if next(head.parameters()).device.type != 'cpu':
        raise RuntimeError('Attach before moving the model to CUDA')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(INIT_SEED)
        adapter = TDAdapter()
    head.__class__ = TOODTDDetect
    head.td_adapter = adapter
    head.td_enabled = True
    head.type = 'tood_td_head.TOODTDDetect'
    head.np = sum(p.numel() for p in head.parameters())
    return model


def branch(model):
    return model.model[-1].td_adapter


def branch_groups():
    return [f'pre.{i}.' for i in range(4)] + [f'inter.{i}.' for i in range(DEPTHS)] + \
        [f'{task}.{section}.' for task in ('cls_td', 'reg_td')
         for section in ('la_conv1', 'la_conv2', 'reduction_conv')] + \
        [f'{task}.{i}.' for task in ('out_cls', 'out_reg') for i in range(4)]


def output_magnitudes(model):
    adapter = branch(model)
    return [float(v.weight.detach().abs().sum()) for v in list(adapter.out_cls) + list(adapter.out_reg)]


def assert_topology(model):
    head = model.model[-1]
    if type(head) is not TOODTDDetect or not head.td_enabled or head.nc != 10 or head.nl != 4:
        raise RuntimeError('Checkpoint is not this enabled TOOD-TD adaptation')
    td = head.td_adapter
    if len(td.pre) != 4 or len(td.inter) != 6 or len(td.out_cls) != 4 or len(td.out_reg) != 4:
        raise RuntimeError('TD topology changed')
    if tuple(m.conv.in_channels for m in td.pre) != CHANNELS:
        raise RuntimeError('TD per-scale channel adapters changed')
    if any(m.conv.out_channels != WIDTH for m in list(td.pre) + list(td.inter)):
        raise RuntimeError('TD width changed')
    if any(m.norm.num_groups != 32 for m in td.modules() if isinstance(m, ConvGNReLU)):
        raise RuntimeError('TD GN32 changed')
    for task in (td.cls_td, td.reg_td):
        if task.in_channels != WIDTH * DEPTHS or task.la_conv1.out_channels != WIDTH // 8:
            raise RuntimeError('TD layer attention changed')
    return dict(width=WIDTH, depths=DEPTHS, channels=list(CHANNELS),
                shared_across_levels=True, gates='image_global_sigmoid_not_softmax',
                native_prediction_towers=True, new_parameters=sum(p.numel() for p in td.parameters()))
