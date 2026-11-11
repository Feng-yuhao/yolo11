"""Independent ablation semantics; no task-specific oracle reused for shared."""
import copy
import torch
from torch import nn
import spatial_task_head as target


def routing_unit_channels(device, cin):
    full = target.full.e58.WideTaskBlock(cin)
    if cin == 128:
        full.__class__ = target.full.GroupResidualTaskBlock
        full.refinements = nn.ModuleList(target.GroupResidualEncoder() for _ in range(3))
        with torch.no_grad():
            for r in full.refinements:
                r.project.weight.normal_(0, .01)
    parent = copy.deepcopy(full).to(device).eval()
    block = target.convert_block(full, target.VARIANT).to(device).eval()
    oracle = copy.deepcopy(block)
    x = torch.randn(2, cin, 9, 11, device=device, requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    original = x.detach().clone()
    caught, hooks = {}, []
    def capture(j):
        def hook(module, inputs):
            caught[j] = inputs[0].detach().clone()
        return hook
    for j, layer in enumerate(block.depths):
        hooks.append(layer.register_forward_pre_hook(capture(j)))
    try:
        cls, reg, weights, bank = block.decompose(x)
    finally:
        for hook in hooks:
            hook.remove()
    raw = [parent.reduce(x)]
    for layer in parent.depths:
        raw.append(layer(raw[-1]))
    for j in range(3):
        torch.testing.assert_close(caught[j], raw[j], rtol=0, atol=0)
    expected_bank = raw if cin != 128 else [raw[0]] + [r(z) for r,z in zip(parent.refinements, raw[1:])]
    for a,b in zip(bank, expected_bank):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert weights.shape == (2, 2, 4, 9, 11)
    torch.testing.assert_close(weights[:,0], weights[:,1], rtol=0, atol=0)
    torch.testing.assert_close(weights.sum(2), torch.ones_like(weights[:,:,0]), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(cls, reg, rtol=0, atol=0)
    # Independent, explicit sum oracle (different reduction order is allowed
    # only FP32 roundoff). Check input AND every active branch parameter gradient.
    ob = [oracle.reduce(xx)]
    for layer in oracle.depths:
        ob.append(layer(ob[-1]))
    if hasattr(oracle, 'refinements'):
        ob = [ob[0]] + [r(z) for r,z in zip(oracle.refinements, ob[1:])]
    if target.VARIANT == 'uniform':
        assert not hasattr(block, 'router')
        torch.testing.assert_close(weights, torch.full_like(weights, .25), rtol=0, atol=0)
        expected = sum(z*.25 for z in ob)
    else:
        assert block.router[-1].out_channels == 4
        logits = oracle.router(torch.cat(ob, 1))
        ew = torch.softmax(logits, 1)
        expected = sum(ob[d]*ew[:,d:d+1] for d in range(4))
        assert (weights[:,:,:,1:,:]-weights[:,:,:,:-1,:]).abs().max() > 0
    torch.testing.assert_close(cls, expected, rtol=1e-6, atol=1e-6)
    probe = torch.randn_like(cls)
    (cls*probe).sum().backward(); (expected*probe).sum().backward()
    torch.testing.assert_close(x.grad, xx.grad, rtol=2e-5, atol=2e-5)
    for name,p in block.named_parameters():
        if name.startswith('out_'):
            continue
        q = dict(oracle.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(x.detach(), original, rtol=0, atol=0)
    # Separate output projections MUST remain. Only depth weighting is shared.
    with torch.no_grad():
        block.out_cls.weight.fill_(.001); block.out_reg.weight.fill_(.002)
        if target.VARIANT == 'shared':
            block.router[-1].weight.zero_()
            block.router[-1].bias.copy_(torch.tensor([20.,-20.,-20.,-20.], device=device))
    c,r,w,b = block.decompose(x.detach())
    if target.VARIANT == 'shared':
        torch.testing.assert_close(c, b[0], rtol=1e-6, atol=1e-6)
    xc,xr = block(x.detach())
    torch.testing.assert_close(xc, x.detach()+block.out_cls(c), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(xr, x.detach()+block.out_reg(r), rtol=1e-6, atol=1e-6)
    assert not torch.equal(xc, xr)
    return dict(variant=target.VARIANT, weighted_sum_and_gradient_oracle=True,
                original_serial_inputs_unchanged=True, P2_P3_refiners_unchanged=cin==128,
                task_weights_identical=True, separate_output_projections=True,
                uniform_router_absent=target.VARIANT=='uniform', no_input_mutation=True)


def routing_unit(device):
    from group_checks import group_oracle
    from precision_checks import tiny_weight_audit, checkpoint_precision_audit
    from checks import strict_math
    # Two algebraically equivalent reductions can quantize differently when
    # fed into TF32 convolution backward. Independent FP32 oracle tests disable
    # TF32, just like the original E60 loss/gradient equality gate; formal
    # training backend precision is NOT changed by this temporary context.
    with strict_math():
        return dict(group_oracle=group_oracle(device),
                    tiny_AMP_weights=tiny_weight_audit(device, fixed=True),
                    checkpoint_precision=checkpoint_precision_audit(device),
                    levels={str(cin):routing_unit_channels(device, cin) for cin in (128,256,512)})


def count_conv_linear(model, size=640):
    counts, handles = {}, []
    def hook(name):
        def count(module, inputs, output):
            if isinstance(module, nn.Conv2d):
                macs = output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
            else:
                macs = output.numel()*module.in_features
            counts[name] = counts.get(name, 0)+macs
        return count
    was_training = model.training
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook(name)))
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1,3,size,size,device=next(model.parameters()).device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return dict(parameters=sum(p.numel() for p in model.parameters()),
                Conv_Linear_GMACs=sum(counts.values())/1e9,
                Conv_Linear_GFLOPs_2ops_per_MAC=2*sum(counts.values())/1e9,
                boundary='Conv2d/Linear only, not normalization/activation/softmax/weighted sums/complete FLOPs')


def profile_audit(model, base):
    parent = target.full.attach(copy.deepcopy(base))
    p, b, f = (count_conv_linear(m) for m in (model,base,parent))
    expected = {'uniform': .872576, 'shared': .008704}[target.VARIANT]
    actual = f['Conv_Linear_GFLOPs_2ops_per_MAC']-p['Conv_Linear_GFLOPs_2ops_per_MAC']
    if abs(actual-expected) > 1e-8:
        raise RuntimeError('Active router Conv-only delta drift: '+str((actual,expected)))
    if f['parameters']-p['parameters'] != {'uniform':51360,'shared':528}[target.VARIANT]:
        raise RuntimeError('Router parameter removal drift')
    return dict(candidate=p, E1=b, full_E60=f, removed_router_Conv_GFLOPs=actual)
