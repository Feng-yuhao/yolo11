"""Independent depth-readout math, route/gradient/precision checks."""
import copy
import torch
from torch import nn
from torch.nn import functional as F


def compare_gradients(a, b, name, exact=False):
    torch.testing.assert_close(a, b, rtol=0 if exact else 3e-4,
                               atol=0 if exact else 1e-7, msg=name)


def branch_groups(unused=None):
    old = [f'{i}.{part}.' for i in range(2) for part in
           ('reduce', 'depths.0', 'depths.1', 'depths.2', 'router', 'out_cls', 'out_reg')]
    return old + [f'{i}.readout_{task}.{j}.' for i in range(2)
                  for task in ('cls', 'reg') for j in range(3)]


def independent_readout(block, x):
    """Oracle: transform each weighted depth with its full depth-specific matrix.

    Production aggregates contrast features BEFORE projections. This oracle
    projects each depth separately and uses an independently declared basis.
    """
    cls, reg, weights, bank = block.decompose(x)
    basis = x.new_tensor([[1, 1, 1], [1, -1, -1],
                          [-1, 1, -1], [-1, -1, 1]], dtype=torch.float32) / 2
    answer = []
    with torch.autocast(device_type=x.device.type, enabled=False):
        for t, (shared, layers) in enumerate(((block.out_cls, block.readout_cls),
                                              (block.out_reg, block.readout_reg))):
            value = x.float()
            for k in range(4):
                matrix = shared.weight.float() + sum(basis[k, j] * layer.weight.float()
                                                     for j, layer in enumerate(layers))
                value = value + F.conv2d(bank[k].float(), matrix) * weights[:, t, k, None].float()
            answer.append(value)
    return tuple(answer)


def new_block(device):
    from parent_spatial_task_head import SpatialTaskBlock
    from depth_readout import install_readouts
    block = SpatialTaskBlock()
    install_readouts(block)
    return block.to(device)


def routing_audit(device, unused=None):
    from parent_spatial_task_head import SpatialTaskBlock
    from depth_readout import CONTRAST_BASIS, install_readouts
    basis = torch.tensor(CONTRAST_BASIS, device=device)
    torch.testing.assert_close(basis.T @ basis, torch.eye(3, device=device), rtol=0, atol=0)
    torch.testing.assert_close(basis.sum(0), torch.zeros(3, device=device), rtol=0, atol=0)
    old = SpatialTaskBlock().eval()
    nn.init.normal_(old.out_cls.weight, std=.02)
    nn.init.normal_(old.out_reg.weight, std=.02)
    target = copy.deepcopy(old)
    install_readouts(target)
    old = old.to(device); target = target.to(device)
    x = torch.randn(2, 128, 13, 19, device=device)
    guard = x.clone()
    # Compare activated E40, not a trivial E1 identity with old outputs zero.
    for a, b in zip(old(x), target(x)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    xr = x.clone().requires_grad_(); xt = x.clone().requires_grad_()
    a = old(xr); b = target(xt)
    (a[0].square().mean() + .7*a[1].square().mean()).backward()
    (b[0].square().mean() + .7*b[1].square().mean()).backward()
    compare_gradients(xr.grad, xt.grad, 'activated E40 input gradient')
    params = dict(target.named_parameters())
    for name, p in old.named_parameters():
        compare_gradients(params[name].grad, p.grad, 'activated E40 ' + name)
    shapes = []
    for h, w in ((13, 19), (8, 8), (1, 1)):
        live = new_block(device).eval()
        for layers in (live.readout_cls, live.readout_reg):
            for layer in layers:
                nn.init.normal_(layer.weight, std=.015)
        nn.init.normal_(live.out_cls.weight, std=.02)
        nn.init.normal_(live.out_reg.weight, std=.02)
        # Both old and new readouts active: independent full-depth linear map.
        inp = torch.randn(2, 128, h, w, device=device)
        for actual, expected in zip(live(inp), independent_readout(live, inp)):
            torch.testing.assert_close(actual.float(), expected, rtol=3e-5, atol=2e-6)
        shapes.append([h, w])
    live = new_block(device).eval()
    for layer in live.readout_cls:
        nn.init.normal_(layer.weight, std=.02)
    dc, dr = live(x)
    torch.testing.assert_close(dr, x, rtol=0, atol=0)
    if torch.equal(dc, x): raise RuntimeError('Classification DRF route absent')
    dc.square().mean().backward()
    for layer in live.readout_reg:
        if layer.weight.grad is not None and layer.weight.grad.count_nonzero():
            raise RuntimeError('Classification loss directly updated regression readout')
    live.zero_grad(set_to_none=True)
    for layer in live.readout_cls:
        nn.init.zeros_(layer.weight)
    for layer in live.readout_reg:
        nn.init.normal_(layer.weight, std=.02)
    dc, dr = live(x)
    torch.testing.assert_close(dc, x, rtol=0, atol=0)
    if torch.equal(dr, x): raise RuntimeError('Regression DRF route absent')
    dr.square().mean().backward()
    for layer in live.readout_cls:
        if layer.weight.grad is not None and layer.weight.grad.count_nonzero():
            raise RuntimeError('Regression loss directly updated classification readout')
    torch.testing.assert_close(x, guard, rtol=0, atol=0)
    # Loss-only SGD, without weight decay/momentum: all new projections are live.
    live = new_block(device).eval()
    for layers in (live.readout_cls, live.readout_reg):
        for layer in layers: nn.init.normal_(layer.weight, std=.02)
    initial = {n: p.detach().clone() for n, p in live.named_parameters() if 'readout_' in n}
    optimizer = torch.optim.SGD(live.parameters(), lr=.1, weight_decay=0, momentum=0)
    dc, dr = live(x)
    (dc.square().mean() + .7*dr.square().mean()).backward()
    for n, p in live.named_parameters():
        if 'readout_' in n and (p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum()==0):
            raise RuntimeError('Dead depth readout loss gradient ' + n)
    optimizer.step()
    for n, p in live.named_parameters():
        if n in initial and torch.equal(p, initial[n]): raise RuntimeError('No loss-only update ' + n)
    # Bypass preserves activated E40 exactly even with nonzero readout matrices.
    live.readout_enabled = False
    cls, reg, _, _ = live.decompose(x)
    for a, b in zip(live(x), (x+live.out_cls(cls), x+live.out_reg(reg))):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    live.readout_enabled = True
    precision = {}
    dtype = torch.float16 if str(device) != 'cpu' else torch.bfloat16
    converted = copy.deepcopy(live).to(dtype)
    inp = x.to(dtype)
    # Test the added residual itself against a stored-dtype/promoted oracle;
    # original E40 operations keep their native AMP/half semantics.
    with torch.no_grad():
        _, _, weights, bank = converted.decompose(inp)
        promoted_bank = [value.float() for value in bank]
        for t, layers in enumerate((converted.readout_cls, converted.readout_reg)):
            from depth_readout import fp32_project
            weighted = torch.stack(promoted_bank, 1) * weights[:, t, :, None].float()
            for j, layer in enumerate(layers):
                contrast = sum(converted.contrast_basis[k, j].float()*weighted[:, k] for k in range(4))
                actual = fp32_project(layer, contrast.to(dtype))
                expected = F.conv2d(contrast.to(dtype).float(), layer.weight.float())
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if not all(p.dtype == dtype for p in converted.parameters()):
            raise RuntimeError('Precision check mutated stored parameter dtype')
        if not all(torch.isfinite(v).all() and v.dtype==dtype for v in converted(inp)):
            raise RuntimeError('Converted full block failed')
    precision[str(dtype)] = dict(FP32_added_readout=True, native_E40_precision_preserved=True)
    if str(device)!='cpu':
        live=new_block(device).eval()
        for layers in (live.readout_cls,live.readout_reg):
            for layer in layers: nn.init.normal_(layer.weight,std=.02)
        with torch.autocast(device_type='cuda',dtype=torch.float16):
            actual=live(x)
            expected=independent_readout(live,x)
            for a,b in zip(actual,expected):
                torch.testing.assert_close(a.float(),b,rtol=3e-5,atol=2e-6)
            objective=actual[0].square().mean()+.7*actual[1].square().mean()
        objective.backward()
        for name,param in live.named_parameters():
            if param.grad is None or not torch.isfinite(param.grad).all() or param.grad.abs().sum()==0:
                raise RuntimeError('AMP active branch gradient failed '+name)
        precision['autocast_float16']=dict(independent_oracle=True,all_branch_gradients_finite_and_nonzero=True)
    return dict(status='passed',orthonormal_zero_sum_basis=True,independent_full_depth_matrix_oracle=shapes,
                activated_E40_identity_and_gradients=True,separate_task_outputs_and_gradients=True,
                all_six_new_projection_loss_only_updates=True,caller_input_untouched=True,
                readout_bypass_exact_E40=True,precision=precision)
