"""Small-branch precision/backward checks, CPU FP16 or actual CUDA AMP."""
import copy
import torch
from torch import nn
from extension import local_calibration, linear_fp32


def precision_audit(device, variant):
    from checks import require
    target = torch.device(device)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(723)
        if variant == 'local':
            route = nn.Sequential(nn.Conv2d(48,16,1,bias=False),nn.SiLU(),
                nn.Conv2d(16,16,3,padding=1,groups=16,bias=False),nn.SiLU(),nn.Conv2d(16,48,1))
            with torch.no_grad():
                route[4].weight.fill_(2e-8); route[4].bias.zero_()
            shape=(2,48,5,7)
            forward=lambda m,x: local_calibration(m,x)
        else:
            route=nn.Conv2d(96,48,1,bias=False)
            with torch.no_grad(): route.weight.fill_(2e-8)
            shape=(2,96,5,7)
            forward=lambda m,x: linear_fp32(m,x)
    route.to(target); reference=copy.deepcopy(route)
    generator=torch.Generator(device=target).manual_seed(37)
    x=torch.rand(shape,device=target,generator=generator).half().requires_grad_(True)
    xr=x.detach().float().requires_grad_(True)
    # For local, compare the SAME output dtype boundary in the oracle, so round
    # off at feature cast is not confused with weak weight/gradient underflow.
    if variant=='local':
        expected=(xr*(2*reference(xr).sigmoid())).half()
    else:
        expected=reference(xr)
    with torch.autocast(device_type=target.type,dtype=torch.float16,enabled=True):
        actual=forward(route,x)
    torch.testing.assert_close(actual.float(),expected.float(),rtol=2e-4,atol=2e-6)
    probe=torch.rand(actual.shape,device=target,generator=generator)
    (actual.float()*probe).sum().backward(); (expected.float()*probe).sum().backward()
    maxima={}
    for (n,p),(rn,rp) in zip(route.named_parameters(),reference.named_parameters()):
        require(n==rn and p.grad is not None and torch.isfinite(p.grad).all(),'Precision gradient missing '+n)
        require(p.grad.abs().sum()>0,'Weak path gradient erased '+n)
        torch.testing.assert_close(p.grad,rp.grad,rtol=5e-4,atol=1e-7)
        maxima[n]=float(p.grad.abs().max())
    return dict(device=str(target),weak_zero_start_weights=2e-8,
                AMP_oracle_forward_backward=True,loss_gradients=maxima,
                only_new_branch_fp32=True,synthetic_only=True)
