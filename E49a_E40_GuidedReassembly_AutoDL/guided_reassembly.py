"""E49: shallow-guided residual reassembly of ONLY the P3 upsampling donor.

Inspired by CARAFE/FADE (see PLAN.md). Not a new upsampling principle.
No learned coordinates, direct shallow-value fusion, auxiliary loss or detach.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F

CHANNELS = 128
HIDDEN = 32
KERNEL = 3
ADDED_PARAMETERS = 8778


def reassemble(coarse, kernels):
    """B,C,H,W and B,9,2H,2W -> B,C,2H,2W.

    Row-major offsets: (-1,-1),(-1,0),..., (1,1). Replicated padding.
    Reuse low-grid patches for all four phases; never expand C*9 on P2.
    Accumulate in FP32 even for half EMA / AMP, without changing stored dtype.
    """
    b, c, h, w = coarse.shape
    if kernels.shape != (b, 9, 2*h, 2*w):
        raise ValueError('Reassembly kernel/input shape mismatch')
    with torch.autocast(device_type=coarse.device.type, enabled=False):
        patches = F.unfold(F.pad(coarse.float(), (1,1,1,1), mode='replicate'), 3)
        patches = patches.reshape(b, c, 9, h, w)
        weights = kernels.float().reshape(b, 9, h, 2, w, 2).permute(0,3,5,1,2,4)
        # Output ordering is coarse-row, fine-row-phase, coarse-col, fine-col-phase.
        result = torch.einsum('bckhw,bpqkhw->bchpwq', patches, weights)
        return result.reshape(b, c, 2*h, 2*w)


class GuidedReassembly(nn.Module):
    def __init__(self):
        super().__init__()
        self.coarse_project = nn.Sequential(nn.Conv2d(128,32,1,bias=False),
                                            nn.GroupNorm(4,32,affine=False))
        self.fine_project = nn.Sequential(nn.Conv2d(128,32,1,bias=False),
                                          nn.GroupNorm(4,32,affine=False))
        self.kernel = nn.Sequential(nn.Conv2d(32,32,3,padding=1,groups=32,bias=False),
                                    nn.SiLU(), nn.Conv2d(32,9,1))
        # Only this scalar is zero-initialized. Kernel network remains nonzero.
        # Signed tanh mixing is bounded, but the FULL operator is not a low-pass
        # convex combination when gain<0. Do not describe it as such in a paper.
        self.gain = nn.Parameter(torch.zeros(()))
        self.enabled = True
        self.capture_stats = False
        self.last_statistics = None

    def weights(self, coarse, fine):
        descriptor = F.silu(F.interpolate(self.coarse_project(coarse),
                                         size=fine.shape[-2:], mode='nearest')
                             + self.fine_project(fine))
        return self.kernel(descriptor).float().softmax(1)

    def forward(self, inputs):
        coarse, fine = inputs
        if (coarse.ndim != 4 or fine.ndim != 4 or coarse.shape[1]!=128 or fine.shape[1]!=128
                or coarse.shape[0]!=fine.shape[0]
                or fine.shape[-2:] != (coarse.shape[-2]*2,coarse.shape[-1]*2)
                or coarse.device != fine.device):
            raise RuntimeError('E49 requires P3 C128 and backbone-P2 C128 at exact 2x scale')
        nearest = F.interpolate(coarse, size=fine.shape[-2:], mode='nearest')
        if not self.enabled:
            return nearest
        rebuilt = reassemble(coarse, self.weights(coarse, fine))
        with torch.autocast(device_type=coarse.device.type, enabled=False):
            value = nearest.float() + self.gain.float().tanh() * (rebuilt-nearest.float())
        output = value.to(coarse.dtype)
        if self.capture_stats:
            # Optional first real batch per epoch. Store only Python floats,
            # never retain a feature tensor/graph in model/EMA/checkpoints.
            with torch.no_grad():
                base_rms=nearest.float().square().mean().sqrt().clamp_min(1e-12)
                self.last_statistics=dict(
                    proposed_delta_over_base_RMS=float((rebuilt-nearest.float()).square().mean().sqrt()/base_rms),
                    actual_output_change_over_base_RMS=float((output.float()-nearest.float()).square().mean().sqrt()/base_rms))
            self.capture_stats=False
        return output


def install(model):
    old = model.model[23]
    if (type(old) is not nn.Upsample or old.f != 16 or old.mode != 'nearest'
            or old.scale_factor != 2 or model.model[24].f != [-1,2]):
        raise RuntimeError('E49 requires the untouched E1 model23 -> Concat24 topology')
    if not {2,16}.issubset(model.save) or next(model.parameters()).device.type!='cpu':
        raise RuntimeError('Attach on CPU; both actual route sources must be saved')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(49001)
        module = GuidedReassembly()
    module.i, module.f = 23, [16,2]
    module.type = 'guided_reassembly.GuidedReassembly'
    module.np = sum(p.numel() for p in module.parameters())
    if module.np != ADDED_PARAMETERS:
        raise RuntimeError('GuidedReassembly parameter count drift')
    model.model[23] = module
    # Keep E1 YAML as the initialization recipe, and record the runtime edit
    # explicitly. Loading .pt uses the saved model, NOT parsing a fake YAML.
    model.e49_topology = dict(replaced_layer=23, sources=[16,2], preserved=[24,25,26],
                             E40_preserved=True, coarse_stride=8, fine_stride=4,
                             kernel=3, hidden=32, new_init_seed=49001)
    return model


def added_conv_flops(imgsz=640):
    low, high = imgsz//8, imgsz//4
    return 2*(128*32*low*low + 128*32*high*high
              +32*9*high*high +32*9*high*high)


def mixing_flops(imgsz=640):
    # Explicit approximation: nine weighted values (multiply+sum) plus
    # nearest residual subtract/multiply/add. Excludes softmax, tanh and copies.
    return (2*9+3)*128*(imgsz//4)**2
