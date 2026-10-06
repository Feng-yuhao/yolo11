"""Revised E44 local calibration / E45 residual depth reuse; no loss changes."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from spatial_task_head import SpatialTaskBlock


def compare_gradients(actual, expected, label, *, exact=False):
    """Compare equivalent graphs, not bit patterns of different CUDA reductions.

    No fixed absolute floor: zero reference gradients must stay exactly zero.
    The absolute budget is 64 FP32 eps times this tensor's reference maximum;
    8 eps relative allowance handles normal nonzero entries. Never relax FP16
    gradients into this test, and report the measured rather than allowed error.
    """
    if actual is None or expected is None:
        if actual is not expected:
            raise AssertionError('Gradient presence mismatch: '+label)
        return dict(label=label, both_none=True)
    if actual.dtype != torch.float32 or expected.dtype != torch.float32:
        raise AssertionError('Expected FP32 gradients: '+label)
    if actual.shape != expected.shape or actual.device != expected.device:
        raise AssertionError('Gradient shape/device mismatch: '+label)
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError('Nonfinite gradients: '+label)
    scale=float(expected.detach().abs().max())
    eps=torch.finfo(torch.float32).eps
    atol=0. if exact else 64*eps*scale
    rtol=0. if exact else 8*eps
    torch.testing.assert_close(actual,expected,rtol=rtol,atol=atol,msg=label)
    error=float((actual.detach()-expected.detach()).abs().max())
    return dict(label=label,max_absolute_error=error,reference_maximum=scale,
                absolute_budget=atol,relative_budget=rtol,bitwise_equal=torch.equal(actual,expected))


def linear_fp32(layer, x):
    # Preserve tiny learned zero-start weights under autocast. Parameter identity
    # and gradients stay intact; use the correct device, NEVER a CPU tensor copy.
    with torch.autocast(device_type=x.device.type, enabled=False):
        if layer.weight.dtype == torch.float32:
            return layer(x.float())
        # EMA/half checkpoints: a stateless FP32 view avoids mutating module
        # dtype or optimizer Parameters. Calling the registered module also
        # retains forward hooks and Conv FLOPs profiling (unlike bare F.conv2d).
        parameters={'weight':layer.weight.float()}
        if layer.bias is not None: parameters['bias']=layer.bias.float()
        return torch.func.functional_call(layer,parameters,(x.float(),))


def local_calibration(route, u):
    # FP32 for this small extra branch only; original E40 encoding/router/towers
    # retain their original precision. No clamp/nan_to_num/detach or BN buffer.
    with torch.autocast(device_type=u.device.type, enabled=False):
        h = F.silu(linear_fp32(route[0], u))
        h = F.silu(linear_fp32(route[2], h))
        gate = 2 * linear_fp32(route[4], h).sigmoid()
        return (u.float() * gate).to(u.dtype)


class LocalCalibrationBlock(SpatialTaskBlock):
    def forward(self, x):
        cls, reg, _, _ = self.decompose(x)
        cls = local_calibration(self.calibrators[0], cls)
        reg = local_calibration(self.calibrators[1], reg)
        return x + self.out_cls(cls), x + self.out_reg(reg)


class ResidualReuseBlock(SpatialTaskBlock):
    def decompose(self, x):
        z0 = self.reduce(x)
        z1 = self.depths[0](z0)
        z2 = self.depths[1](z1 + linear_fp32(self.reuse2, torch.cat((z0, z1), 1)).to(z1.dtype))
        z3 = self.depths[2](z2 + linear_fp32(self.reuse3, torch.cat((z0, z1, z2), 1)).to(z2.dtype))
        bank = [z0, z1, z2, z3]
        b, _, h, w = x.shape
        weights = self.router(torch.cat(bank, 1)).reshape(b, 2, 4, h, w).softmax(2)
        z = torch.stack(bank, 1)
        return ((z * weights[:, 0, :, None]).sum(1),
                (z * weights[:, 1, :, None]).sum(1), weights, bank)


def install(blocks, variant):
    # Construct BOTH common E40 blocks first. CPU-only fork does not seed CUDA.
    before = {n: p.clone() for n, p in blocks.state_dict().items()}
    cpu_rng = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(44001 if variant == 'local' else 45001)
        for block in blocks:
            if variant == 'local':
                block.__class__ = LocalCalibrationBlock
                block.calibrators = nn.ModuleList([
                    nn.Sequential(nn.Conv2d(48, 16, 1, bias=False), nn.SiLU(),
                                  nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),
                                  nn.SiLU(), nn.Conv2d(16, 48, 1, bias=True))
                    for _ in range(2)])
                for route in block.calibrators:
                    nn.init.zeros_(route[4].weight); nn.init.zeros_(route[4].bias)
            elif variant == 'reuse':
                block.__class__ = ResidualReuseBlock
                block.reuse2 = nn.Conv2d(96, 48, 1, bias=False)
                block.reuse3 = nn.Conv2d(144, 48, 1, bias=False)
                nn.init.zeros_(block.reuse2.weight); nn.init.zeros_(block.reuse3.weight)
            else:
                raise ValueError(variant)
    assert torch.equal(cpu_rng, torch.get_rng_state()), 'Extension changed global RNG'
    for n, p in before.items():
        torch.testing.assert_close(blocks.state_dict()[n], p, rtol=0, atol=0)
    return blocks


def branch_groups(variant):
    parts = ['reduce', 'depths.0', 'depths.1', 'depths.2', 'router', 'out_cls', 'out_reg']
    if variant == 'local':
        parts += [f'calibrators.{task}.{layer}' for task in range(2) for layer in (0, 2, 4)]
    elif variant == 'reuse':
        parts += ['reuse2', 'reuse3']
    else:
        raise ValueError(variant)
    return [f'{i}.{part}.' for i in range(2) for part in parts]


def oracle(block, x, variant):
    # Independently written regular-FP32 oracle, not a second call to forward.
    z = [block.reduce(x)]
    z.append(block.depths[0](z[0]))
    for depth in (1, 2):
        previous = z[-1]
        if variant == 'reuse':
            layer = block.reuse2 if depth == 1 else block.reuse3
            previous = previous + F.conv2d(torch.cat(z, 1), layer.weight)
        z.append(block.depths[depth](previous))
    w = block.router(torch.cat(z, 1)).reshape(x.shape[0], 2, 4, *x.shape[-2:]).softmax(2)
    u = [sum(z[k] * w[:, t, k:k+1] for k in range(4)) for t in range(2)]
    if variant == 'local':
        u = [v + v * (2 * route(v).sigmoid() - 1) for v, route in zip(u, block.calibrators)]
    return [x + project(v) for v, project in zip(u, (block.out_cls, block.out_reg))]


def routing_audit(device, variant):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(40001)
        originals = nn.ModuleList([SpatialTaskBlock(), SpatialTaskBlock()]).eval()
        for block in originals:
            nn.init.normal_(block.out_cls.weight, std=.04)
            nn.init.normal_(block.out_reg.weight, std=.05)
    originals=originals.to(device)
    # Install on CPU as formal attach does, then transfer the entire module.
    blocks = install(copy.deepcopy(originals).cpu(), variant).to(device).eval()
    gradient_comparisons=[]
    for index,(original, block) in enumerate(zip(originals, blocks)):
        xr = torch.rand(2, 128, 9, 11, device=device, requires_grad=True)
        xt = xr.detach().clone().requires_grad_(True)
        guard = xt.detach().clone()
        for a, b in zip(original(xr), block(xt)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(original.decompose(xr)[3], block.decompose(xt)[3]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        sum(y.square().mean() for y in original(xr)).backward()
        sum(y.square().mean() for y in block(xt)).backward()
        gradient_comparisons.append(compare_gradients(xt.grad,xr.grad,f'level{index}.input'))
        actual = dict(block.named_parameters())
        for n, p in original.named_parameters():
            gradient_comparisons.append(compare_gradients(actual[n].grad,p.grad,f'level{index}.{n}'))
        torch.testing.assert_close(xt.detach(), guard, rtol=0, atol=0)
        with torch.no_grad():
            if variant == 'local':
                for route in block.calibrators:
                    nn.init.normal_(route[4].weight, std=.05)
                    nn.init.normal_(route[4].bias, std=.05)
            else:
                nn.init.normal_(block.reuse2.weight, std=.03)
                nn.init.normal_(block.reuse3.weight, std=.03)
        xa = xr.detach().clone().requires_grad_(True)
        xo = xr.detach().clone().requires_grad_(True)
        a, o = block(xa), oracle(block, xo, variant)
        for aa, oo in zip(a, o):
            torch.testing.assert_close(aa, oo, rtol=2e-5, atol=2e-6)
        ga = torch.autograd.grad(sum(v.square().mean() for v in a), xa)[0]
        go = torch.autograd.grad(sum(v.square().mean() for v in o), xo)[0]
        torch.testing.assert_close(ga, go, rtol=3e-5, atol=3e-6)
        # Inspect EVERY registered parameter, not only input gradients. Use
        # independent graphs so testing cannot silently reuse forward output.
        parameters=list(block.named_parameters())
        actual_outputs=block(xr.detach())
        oracle_outputs=oracle(block,xr.detach(),variant)
        actual_grads=torch.autograd.grad(sum(v.square().mean() for v in actual_outputs),
                                        [p for _,p in parameters],allow_unused=True)
        oracle_grads=torch.autograd.grad(sum(v.square().mean() for v in oracle_outputs),
                                        [p for _,p in parameters],allow_unused=True)
        for (name,_),ag,og in zip(parameters,actual_grads,oracle_grads):
            gradient_comparisons.append(compare_gradients(ag,og,f'activated.level{index}.{name}'))
        if variant == 'local':
            block.zero_grad(set_to_none=True)
            block(xr.detach())[0].square().mean().backward()
            assert all(p.grad is None for p in block.calibrators[1].parameters())
        # Independent loss-only update, with NO momentum/weight decay. The
        # real detector audit also checks natural zero-start staged gradients;
        # this activated module check prevents mistaking decay for learnability.
        block.zero_grad(set_to_none=True)
        initial={n:p.detach().clone() for n,p in block.named_parameters()}
        sum(v.square().mean() for v in block(xr.detach())).backward()
        groups=[g.split('.',1)[1] for g in branch_groups(variant) if g.startswith('0.')]
        for group in groups:
            grads=[p.grad for n,p in block.named_parameters() if n.startswith(group)]
            assert grads and any(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in grads),group
        torch.optim.SGD(block.parameters(),lr=.02,weight_decay=0,momentum=0).step()
        for group in groups:
            assert any(not torch.equal(p,initial[n]) for n,p in block.named_parameters() if n.startswith(group)),group
        # Save-EMA/pure-half numeric boundary, including ACTIVATED paths.
        half = copy.deepcopy(block).half().eval()
        hx = xr.detach().half()
        with torch.no_grad():
            fp32 = copy.deepcopy(half).float()(hx.float())
            for hh, ff in zip(half(hx), fp32):
                assert hh.dtype == torch.float16 and torch.isfinite(hh).all()
                torch.testing.assert_close(hh.float(), ff, rtol=.015, atol=.003)
    return dict(common_E40_hidden_and_nonzero_output_identity=True,
                initial_input_and_common_gradients_equal=True,
                gradient_comparison='FP32 tensor-scaled numerical equality; not bitwise identity',
                gradient_comparisons=gradient_comparisons,
                activated_independent_formula_and_gradient=True,
                activated_all_parameter_gradients_checked=True,
                task_scope_checked=variant == 'local',
                activated_half_value_checked=True,loss_only_updates_without_weight_decay=True,
                no_detach=True, no_mutation=True)
