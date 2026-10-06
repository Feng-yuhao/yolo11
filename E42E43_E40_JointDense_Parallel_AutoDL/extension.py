"""Two controlled E40 extensions. No backbone, loss or assignment change."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from spatial_task_head import SpatialTaskBlock, encoder


def channel_logits_fp32(router, descriptor):
    """Protect BOTH channel projections, not merely the final softmax.

    The zero-initialized last projection first learns tiny FP32 weights. CUDA
    autocast can round those weights to zero before the matmul and erase the
    first projection's gradient. Keep this small pooled MLP in FP32, including
    its weights/activations/backward. Existing Parameter objects stay registered;
    .float() is differentiable and also supports saved FP16 EMA weights.
    Pool with FP32 accumulation without materializing a full FP32 feature map.
    """
    front, last = router[1], router[3]
    with torch.autocast(device_type=descriptor.device.type, enabled=False):
        pooled = descriptor.mean(dim=(-2, -1), keepdim=True, dtype=torch.float32)
        hidden = F.silu(F.conv2d(pooled, front.weight.float(),
                                None if front.bias is None else front.bias.float()))
        return F.conv2d(hidden, last.weight.float(),
                        None if last.bias is None else last.bias.float())


def joint_mix(bank, spatial, channel, chunk=8):
    """Normalize depth (dimension 1 after selecting a task), never channels.

    FP32 avoids opposing-logit product underflow; cast back before projections.
    Chunking reduces temporary peaks, not a promise about saved backward memory.
    """
    z = torch.stack(bank, 1)
    results = []
    for task in range(2):
        pieces = []
        for start in range(0, z.shape[2], chunk):
            stop = min(start + chunk, z.shape[2])
            logits = spatial[:, task, :, None].float() + channel[:, task, :, start:stop].float()
            weights = logits.softmax(1)
            pieces.append((weights * z[:, :, start:stop].float()).sum(1))
        results.append(torch.cat(pieces, 1).to(z.dtype))
    return tuple(results)


class JointTaskBlock(SpatialTaskBlock):
    def decompose(self, x):
        z = self.reduce(x)
        bank = [z]
        for layer in self.depths:
            z = layer(z)
            bank.append(z)
        descriptor = torch.cat(bank, 1)
        b, _, h, w = x.shape
        spatial = self.router(descriptor).reshape(b, 2, 4, h, w)
        channel = channel_logits_fp32(self.channel_router, descriptor).reshape(b, 2, 4, 48, 1, 1)
        cls, reg = joint_mix(bank, spatial, channel)
        # Do not return/materialize the full six-dimensional weights at P2.
        return cls, reg, (spatial, channel), bank


class DenseTaskBlock(SpatialTaskBlock):
    def decompose(self, x):
        z0 = self.reduce(x)
        z1 = self.depths[0](z0)
        z2 = self.depths[1](self.fuse2(torch.cat((z0, z1), 1)))
        z3 = self.depths[2](self.fuse3(torch.cat((z0, z1, z2), 1)))
        bank = [z0, z1, z2, z3]
        b, _, h, w = x.shape
        weights = self.router(torch.cat(bank, 1)).reshape(b, 2, 4, h, w).softmax(2)
        z = torch.stack(bank, 1)
        return (z * weights[:, 0, :, None]).sum(1), (z * weights[:, 1, :, None]).sum(1), weights, bank


def install(blocks, variant):
    # Called only AFTER constructing BOTH original E40 blocks. Never consume
    # the original block-construction stream or the global training RNG.
    before = {n: p.clone() for n, p in blocks.state_dict().items()}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42001 if variant == 'joint' else 43001)
        for block in blocks:
            if variant == 'joint':
                block.__class__ = JointTaskBlock
                block.channel_router = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(192, 32, 1, bias=False), nn.SiLU(), nn.Conv2d(32, 384, 1))
                nn.init.zeros_(block.channel_router[-1].weight)
                nn.init.zeros_(block.channel_router[-1].bias)
            elif variant == 'dense':
                block.__class__ = DenseTaskBlock
                block.fuse2 = encoder(96, 48, 1)
                block.fuse3 = encoder(144, 48, 1)
            else:
                raise ValueError(variant)
    for n, p in before.items():
        torch.testing.assert_close(blocks.state_dict()[n], p, rtol=0, atol=0)
    return blocks


def branch_groups(variant):
    parts = ['reduce', 'depths.0', 'depths.1', 'depths.2', 'router', 'out_cls', 'out_reg']
    parts += ['channel_router.1', 'channel_router.3'] if variant == 'joint' else ['fuse2', 'fuse3']
    return [f'{i}.{part}.' for i in range(2) for part in parts]


def routing_audit(device, variant):
    # Oracle uses independent full tensors only on small maps, including gradients.
    from checks import require
    from spatial_task_head import VARIANT
    require(VARIANT == variant, 'Wrong variant imported')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(40001)
        originals = nn.ModuleList(SpatialTaskBlock() for _ in range(2))
    rng = torch.get_rng_state().clone()
    blocks = install(copy.deepcopy(originals), variant)
    require(torch.equal(rng, torch.get_rng_state()), 'Extension changed CPU RNG')
    for n, v in originals.state_dict().items():
        torch.testing.assert_close(v, blocks.state_dict()[n], rtol=0, atol=0)
    block = blocks[0].to(device).eval()
    x = torch.rand(2, 128, 9, 11, device=device, requires_grad=True)
    guard = x.detach().clone()
    with torch.no_grad():
        block.out_cls.weight.fill_(.001)
        block.out_reg.weight.fill_(.002)
    cls, reg, weights, bank = block.decompose(x)
    if variant == 'joint':
        reference = originals[0].to(device).eval()
        ref_cls, ref_reg, _, _ = reference.decompose(x)
        torch.testing.assert_close(cls, ref_cls, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(reg, ref_reg, rtol=1e-6, atol=1e-6)
        z = torch.randn(2, 4, 48, 3, 5, device=device, requires_grad=True)
        s = torch.randn(2, 2, 4, 3, 5, device=device, requires_grad=True)
        c = torch.randn(2, 2, 4, 48, 1, 1, device=device, requires_grad=True)
        w = (s[:, :, :, None] + c).softmax(2)
        oracle = (w * z[:, None]).sum(2)
        actual = torch.stack(joint_mix(list(z.unbind(1)), s, c), 1)
        torch.testing.assert_close(actual, oracle, rtol=1e-6, atol=1e-6)
        probe = torch.randn_like(actual)
        g1 = torch.autograd.grad((oracle * probe).sum(), (z, s, c), retain_graph=True)
        g2 = torch.autograd.grad((actual * probe).sum(), (z, s, c))
        for a, b in zip(g1, g2):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        sharp = torch.tensor([100., 0., -100., 0.], device=device).reshape(1, 1, 4, 1, 1)
        extreme = joint_mix([torch.ones(1, 48, 1, 1, device=device) * i for i in range(4)],
                            sharp.expand(1, 2, 4, 1, 1), -sharp[:, :, :, None].expand(1, 2, 4, 48, 1, 1))
        for v in extreme:
            torch.testing.assert_close(v, torch.full_like(v, 1.5))
        # Independent input routing: spatial selection for channels 0..23,
        # opposing channel selection overrides it for channels 24..47.
        s = torch.zeros(1, 2, 4, 2, 3, device=device)
        s[:, 0, 0] = 20; s[:, 1, 3] = 20
        c = torch.zeros(1, 2, 4, 48, 1, 1, device=device)
        c[:, 0, 2, 24:] = 60; c[:, 1, 1, 24:] = 60
        got = joint_mix([torch.full((1, 48, 2, 3), float(i), device=device) for i in range(4)], s, c)
        for task, (lo, hi) in enumerate(((0., 2.), (3., 1.))):
            torch.testing.assert_close(got[task][:, :24], torch.full_like(got[task][:, :24], lo), atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(got[task][:, 24:], torch.full_like(got[task][:, 24:], hi), atol=1e-6, rtol=1e-6)
    else:
        z0 = block.reduce(x); z1 = block.depths[0](z0)
        z2 = block.depths[1](block.fuse2(torch.cat([z0, z1], dim=1)))
        z3 = block.depths[2](block.fuse3(torch.cat([z0, z1, z2], dim=1)))
        for a, b in zip(bank, [z0, z1, z2, z3]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(weights.sum(2), torch.ones_like(weights[:, :, 0]), rtol=1e-6, atol=1e-6)
        for task, v in enumerate((cls, reg)):
            torch.testing.assert_close(v, sum(bank[k] * weights[:, task, k:k+1] for k in range(4)), rtol=1e-6, atol=1e-6)
    xc, xr = block(x)
    torch.testing.assert_close(xc, x + block.out_cls(cls), rtol=0, atol=0)
    torch.testing.assert_close(xr, x + block.out_reg(reg), rtol=0, atol=0)
    (xc.square().sum() + xr.square().sum()).backward()
    require(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0, 'Input detached')
    torch.testing.assert_close(x.detach(), guard, rtol=0, atol=0)
    return dict(common_E40_initialization_exact=True, independent_formula=True,
                full_chunk_forward_backward=variant == 'joint', no_detach=True, no_mutation=True)
