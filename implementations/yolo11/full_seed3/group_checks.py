"""Independent functional oracle + actual depth-chain/router path checks."""
import copy
import torch
from torch.nn import functional as F
from spatial_task_head import GroupResidualEncoder, GroupResidualTaskBlock
import e58_spatial_task_head as e58


def group_oracle(device):
    module=GroupResidualEncoder().to(device)
    z=torch.randn(2,96,7,9,device=device,requires_grad=True)
    original=z.detach().clone()
    torch.testing.assert_close(module(z),z,rtol=0,atol=0)
    with torch.no_grad(): module.project.weight.normal_(0,.01)
    # Independent functional implementation with separately cloned parameters.
    weights={n:p.detach().clone().requires_grad_() for n,p in module.named_parameters()}
    zz=z.detach().clone().requires_grad_()
    split=[zz[:,24*i:24*(i+1)] for i in range(4)]
    values=[split[0]]
    for j in range(3):
        argument=split[j+1] if j==0 else split[j+1]+values[-1]
        prefix=f'kernels.{j}.'
        c=F.conv2d(argument,weights[prefix+'0.weight'],padding=1)
        c=F.group_norm(c,6,weights[prefix+'1.weight'],weights[prefix+'1.bias'],eps=1e-5)
        values.append(F.silu(c))
    expected=zz+F.conv2d(torch.cat(values,dim=1),weights['project.weight'])
    actual=module(z)
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
    probe=torch.randn_like(actual)
    (actual*probe).sum().backward(); (expected*probe).sum().backward()
    torch.testing.assert_close(z.grad,zz.grad,rtol=1e-5,atol=1e-6)
    for name,p in module.named_parameters():
        torch.testing.assert_close(p.grad,weights[name].grad,rtol=1e-5,atol=1e-6)
        if not torch.isfinite(p.grad).all() or p.grad.abs().sum()==0:
            raise RuntimeError('Missing independent group gradient '+name)
    torch.testing.assert_close(z.detach(),original,rtol=0,atol=0)
    if sum(p.numel() for p in module.parameters())!=24912:
        raise RuntimeError('Wrong single refinement parameter count')
    half_error=None
    if str(device)!='cpu':
        # Force NONZERO projections: cannot pass half-value checks only because
        # the residual branch is initially or numerically zero.
        half=copy.deepcopy(module).half().eval()
        hx=z.detach().half()
        with torch.no_grad():
            ha=half(hx); hb=copy.deepcopy(half).float()(hx.float())
        if ha.dtype!=torch.float16 or not torch.isfinite(ha).all():
            raise RuntimeError('Active group branch pure-half dtype/value failed')
        torch.testing.assert_close(ha.float(),hb,rtol=.01,atol=.003)
        half_error=float((ha.float()-hb).abs().max())
    return dict(identity_at_zero_projection=True,independent_value_and_gradient_oracle=True,
                separate_group_kernels=True,no_detach_no_inplace=True,parameters_per_refiner=24912,
                active_pure_half_max_absolute_error=half_error)


def bank_path_audit(device):
    block=e58.WideTaskBlock(128)
    parent=copy.deepcopy(block).to(device).eval()
    block.__class__=GroupResidualTaskBlock
    block.refinements=torch.nn.ModuleList(GroupResidualEncoder() for _ in range(3))
    block=block.to(device).eval()
    x=torch.randn(2,128,9,11,device=device)
    original=x.clone()
    with torch.no_grad():
        target=block.decompose(x); base=parent.decompose(x)
        for a,b in zip(target[:3],base[:3]): torch.testing.assert_close(a,b,rtol=0,atol=0)
        for a,b in zip(target[3],base[3]): torch.testing.assert_close(a,b,rtol=0,atol=0)
        for r in block.refinements: r.project.weight.normal_(0,.01)
    seen={}; hooks=[]
    def capture(key):
        def hook(m,inputs): seen[key]=inputs[0].detach().clone()
        return hook
    for j,layer in enumerate(block.depths): hooks.append(layer.register_forward_pre_hook(capture(j)))
    hooks.append(block.router.register_forward_pre_hook(capture('router')))
    try:
        with torch.no_grad():
            original_bank=parent.decompose(x)[3]
            bank=block.decompose(x)[3]
        for j in range(3):
            torch.testing.assert_close(seen[j],original_bank[j],rtol=0,atol=0)
            expected=block.refinements[j](original_bank[j+1])
            torch.testing.assert_close(bank[j+1],expected,rtol=0,atol=0)
            if torch.equal(bank[j+1],original_bank[j+1]):
                raise RuntimeError('Refinement not entering read bank')
        torch.testing.assert_close(seen['router'],torch.cat(bank,1),rtol=0,atol=0)
        torch.testing.assert_close(x,original,rtol=0,atol=0)
    finally:
        for hook in hooks: hook.remove()
    half_error=None
    if str(device)!='cpu':
        half=copy.deepcopy(block).half().eval()
        hx=x.half()
        with torch.no_grad():
            a=half.decompose(hx); b=copy.deepcopy(half).float().decompose(hx.float())
        for aa,bb in zip(a[:3],b[:3]):
            torch.testing.assert_close(aa.float(),bb,rtol=.01,atol=.003)
        for aa,bb in zip(a[3],b[3]):
            torch.testing.assert_close(aa.float(),bb,rtol=.01,atol=.003)
        half_error=max(float((aa.float()-bb).abs().max()) for aa,bb in zip(a[3],b[3]))
    return dict(initial_E58_bank_route_exact=True,original_depth_inputs_unchanged=True,
                enhanced_bank_reaches_spatial_router=True,no_input_mutation=True,
                active_pure_half_bank_max_absolute_error=half_error)
