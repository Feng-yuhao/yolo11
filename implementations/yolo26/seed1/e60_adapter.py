"""E60 feature preparation for pinned Ultralytics 8.4.177, NOT YOLO11 Detect.

The four W96 banks and six P2/P3 FP32 group refinements retain E60's
operators. Preparation happens before the native dual-head detach boundary:
one2many trains it, one2one reads detached prepared features during training.
Native towers, two losses, progressive weights, assignment and decode stay.
"""
import torch
from torch import nn
from ultralytics.nn.modules.head import Detect

CHANNELS = (64, 128, 256, 512)
NEW_PARAMETERS = 1475712


def encoder(cin, cout, kernel):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False),
                         nn.GroupNorm(6, cout), nn.SiLU())


class GroupResidualEncoder(nn.Module):
    """E60's FP32 sidecar, including original-value-preserving _apply."""
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
        return z + self.delta(z).to(dtype=z.dtype)


class E60Block(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.reduce = encoder(cin, 96, 1)
        self.depths = nn.ModuleList(encoder(96, 96, 3) for _ in range(3))
        self.router = nn.Sequential(nn.Conv2d(384, 32, 1, bias=False), nn.SiLU(),
                                    nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
                                    nn.SiLU(), nn.Conv2d(32, 8, 1))
        self.out_cls = nn.Conv2d(96, cin, 1, bias=False)
        self.out_reg = nn.Conv2d(96, cin, 1, bias=False)
        nn.init.zeros_(self.out_cls.weight)
        nn.init.zeros_(self.out_reg.weight)

    def original_bank(self, x):
        bank = [self.reduce(x)]
        for layer in self.depths:
            bank.append(layer(bank[-1]))
        return bank

    def decompose(self, x):
        bank = self.original_bank(x)
        if hasattr(self, 'refinements'):
            bank = [bank[0]] + [r(z) for r, z in zip(self.refinements, bank[1:])]
        b, _, h, w = x.shape
        weights = self.router(torch.cat(bank, 1)).reshape(b, 2, 4, h, w).softmax(2)
        stacked = torch.stack(bank, 1)
        cls = (stacked * weights[:, 0, :, None]).sum(1)
        reg = (stacked * weights[:, 1, :, None]).sum(1)
        return cls, reg, weights, bank

    def forward(self, x):
        cls, reg, _, _ = self.decompose(x)
        return x + self.out_cls(cls), x + self.out_reg(reg)


class E60Detect26(Detect):
    def _prepared_head(self, base, cls, reg, box_head, cls_head):
        if box_head is None or cls_head is None:
            return {}
        bs = base[0].shape[0]
        boxes = torch.cat([box_head[i](reg[i]).view(bs, 4 * self.reg_max, -1)
                           for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](cls[i]).view(bs, self.nc, -1)
                            for i in range(self.nl)], dim=-1)
        return {'boxes': boxes, 'scores': scores, 'feats': base}

    def forward(self, x):
        if not self.enabled:
            return super().forward(x)
        if len(x) != 4 or self.reg_max != 1:
            raise RuntimeError('Adapter requires the pinned four-scale YOLO26 head')
        prepared = [block(value) for block, value in zip(self.task_blocks, x)]
        cls, reg = [p[0] for p in prepared], [p[1] for p in prepared]
        preds = self._prepared_head(x, cls, reg, **self.one2many)
        if getattr(self, 'one2one_cv2', None) is not None:
            if self.training:
                aux_x = [t.detach() for t in x]
                aux_cls = [t.detach() for t in cls]
                aux_reg = [t.detach() for t in reg]
            else:
                aux_x, aux_cls, aux_reg = x, cls, reg
            auxiliary = self._prepared_head(aux_x, aux_cls, aux_reg, **self.one2one)
            preds = {'one2many': preds, 'one2one': auxiliary}
        if self.training:
            return preds
        result = self._inference(preds['one2one' if self.end2end else 'one2many']
                                 if 'one2one' in preds else preds)
        if self.end2end:
            result = self.postprocess(result.permute(0, 2, 1))
        return result if self.export else (result, preds)


def attach(model):
    head = model.model[-1]
    if (type(head) is not Detect or len(model.model) != 30 or head.nl != 4 or
            head.reg_max != 1 or list(head.f) != [19, 22, 25, 28] or
            head.stride.tolist() != [4., 8., 16., 32.]):
        raise RuntimeError('Not the pinned YOLO26s-P2 architecture')
    channels = tuple(layer[0].conv.in_channels for layer in head.cv2)
    if channels != CHANNELS or getattr(head, 'one2one_cv2', None) is None:
        raise RuntimeError('Channel or native dual-head topology drift')
    if next(model.parameters()).device.type != 'cpu':
        raise RuntimeError('Attach on CPU after common pretrained transfer')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(40001)
        shallow = [E60Block(c) for c in CHANNELS[:2]]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(57001)
        deep = [E60Block(c) for c in CHANNELS[2:]]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(60001)
        for block in shallow:
            block.refinements = nn.ModuleList(GroupResidualEncoder() for _ in range(3))
    head.task_blocks = nn.ModuleList(shallow + deep)
    head.__class__ = E60Detect26
    head.enabled = True
    head.type = 'e60_adapter.E60Detect26'
    head.np = sum(p.numel() for p in head.parameters())
    if sum(p.numel() for p in head.task_blocks.parameters()) != NEW_PARAMETERS:
        raise RuntimeError('Adapter parameter count drift')
    return model


def branch_groups():
    base = [f'model.29.task_blocks.{i}.{part}.' for i in range(4)
            for part in ('reduce', 'depths.0', 'depths.1', 'depths.2',
                         'router', 'out_cls', 'out_reg')]
    return base + [f'model.29.task_blocks.{i}.refinements.{d}.{part}.'
                   for i in range(2) for d in range(3)
                   for part in ('kernels.0', 'kernels.1', 'kernels.2', 'project')]
