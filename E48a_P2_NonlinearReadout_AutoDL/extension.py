"""Independent E48 oracle, task routing, staged learning, precision checks."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from parent_e47_extension import compare_gradients


def branch_groups(unused=None):
    from parent_e47_extension import branch_groups as previous
    return previous() + [f'0.nonlinear_{task}.{layer}.'
                         for task in ('cls', 'reg') for layer in (0, 2)]


def new_block(device):
    from parent_e47_extension import new_block as previous
    from nonlinear_readout import install_nonlinear
    block = previous('cpu')
    install_nonlinear(block)
    return block.to(device)


def old_output(block, x):
    from depth_readout import DepthReadoutBlock
    return DepthReadoutBlock.forward(block, x)


def independent_joint(block, x):
    """Reconstruct weighted contrasts by einsum, explicit convs (not production helper)."""
    _, _, weights, bank = block.decompose(x)
    base = old_output(block, x)
    basis = torch.tensor([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],
                         device=x.device, dtype=torch.float32)/2
    result = []
    with torch.autocast(device_type=x.device.type, enabled=False):
        for task, layers in enumerate((block.nonlinear_cls, block.nonlinear_reg)):
            weighted = torch.stack(bank, 1).float() * weights[:,task,:,None].float()
            contrasts = torch.einsum('kj,bkchw->bjchw', basis, weighted)
            joined = contrasts.flatten(1,2)  # contrast-major, then original channel
            delta = F.conv2d(F.silu(F.conv2d(joined,layers[0].weight.float())),
                             layers[2].weight.float())
            result.append((base[task].float()+delta).to(base[task].dtype))
    return tuple(result)


def activate_parent(block):
    for name, param in block.named_parameters():
        if 'nonlinear_' not in name and ('out_' in name or 'readout_' in name):
            nn.init.normal_(param, std=.02)


def routing_audit(device, unused=None):
    from parent_e47_extension import routing_audit as previous_audit, new_block as previous_block
    from nonlinear_readout import install_nonlinear, ADDED_PARAMETERS
    parent_tests = previous_audit(device)
    old = previous_block('cpu').eval()
    activate_parent(old)
    target = copy.deepcopy(old); install_nonlinear(target)
    assert sum(p.numel() for p in target.parameters())-sum(p.numel() for p in old.parameters())==ADDED_PARAMETERS
    old=old.to(device); target=target.to(device)
    x=torch.randn(2,128,13,19,device=device)
    guard=x.clone()
    for a,b in zip(old(x),target(x)):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    xr=x.clone().requires_grad_(); xt=x.clone().requires_grad_()
    a=old(xr); b=target(xt)
    (a[0].square().mean()+.7*a[1].square().mean()).backward()
    (b[0].square().mean()+.7*b[1].square().mean()).backward()
    compare_gradients(xr.grad,xt.grad,'activated E47 input gradient')
    for name,param in old.named_parameters():
        compare_gradients(dict(target.named_parameters())[name].grad,param.grad,'E47.'+name)
    # First update must be driven by loss, NOT weight decay or momentum.
    first_before = [v[0].weight.detach().clone() for v in (target.nonlinear_cls,target.nonlinear_reg)]
    finals = [v[2].weight.detach().clone() for v in (target.nonlinear_cls,target.nonlinear_reg)]
    for layers in (target.nonlinear_cls,target.nonlinear_reg):
        assert layers[0].weight.grad is not None and layers[0].weight.grad.count_nonzero()==0
        assert layers[2].weight.grad is not None and torch.isfinite(layers[2].weight.grad).all()
        assert layers[2].weight.grad.abs().sum()>0
    opt=torch.optim.SGD(target.parameters(),lr=.1,momentum=0,weight_decay=0)
    opt.step(); opt.zero_grad(set_to_none=True)
    for layers,w1,w2 in zip((target.nonlinear_cls,target.nonlinear_reg),first_before,finals):
        torch.testing.assert_close(layers[0].weight,w1,rtol=0,atol=0)
        assert not torch.equal(layers[2].weight,w2)
    c,r=target(x); (c.square().mean()+.7*r.square().mean()).backward()
    for layers in (target.nonlinear_cls,target.nonlinear_reg):
        assert torch.isfinite(layers[0].weight.grad).all() and layers[0].weight.grad.abs().sum()>0
    opt.step()
    for layers,w1 in zip((target.nonlinear_cls,target.nonlinear_reg),first_before):
        assert not torch.equal(layers[0].weight,w1)
    shapes=[]
    for h,w in ((13,19),(8,8),(1,1)):
        live=new_block(device).eval(); activate_parent(live)
        for layers in (live.nonlinear_cls,live.nonlinear_reg):
            nn.init.normal_(layers[2].weight,std=.02)
        inp=torch.randn(2,128,h,w,device=device)
        for a,b in zip(live(inp),independent_joint(live,inp)):
            torch.testing.assert_close(a,b,rtol=3e-5,atol=2e-6)
        shapes.append([h,w])
    # Each new task readout is exclusive; shared encoders may still get both gradients.
    for task in (0,1):
        live=new_block(device).eval(); activate_parent(live)
        layers=(live.nonlinear_cls,live.nonlinear_reg)
        nn.init.normal_(layers[task][2].weight,std=.02)
        baseline=old_output(live,x); values=live(x)
        torch.testing.assert_close(values[1-task],baseline[1-task],rtol=0,atol=0)
        assert not torch.equal(values[task],baseline[task])
        values[task].square().mean().backward()
        assert all(p.grad is None or p.grad.count_nonzero()==0 for p in layers[1-task].parameters())
        live.nonlinear_enabled=False
        for a,b in zip(live(x),old_output(live,x)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    precision={}
    for dtype in ([torch.float16,torch.bfloat16] if str(device)!='cpu' else [torch.bfloat16]):
        converted=copy.deepcopy(target).to(dtype).eval()
        inp=x.to(dtype)
        with torch.no_grad():
            for a,b in zip(converted(inp),independent_joint(converted,inp)):
                torch.testing.assert_close(a,b,rtol=.003 if dtype==torch.float16 else .02,atol=.002)
                assert a.dtype==dtype and torch.isfinite(a).all()
        assert all(p.dtype==dtype for p in converted.parameters())
        precision[str(dtype)]=dict(stored_dtype_unchanged=True,FP32_joint_readout=True)
    if str(device)!='cpu':
        live=new_block(device).eval(); activate_parent(live)
        for layers in (live.nonlinear_cls,live.nonlinear_reg): nn.init.normal_(layers[2].weight,std=.02)
        with torch.autocast(device_type='cuda',dtype=torch.float16):
            a=live(x); b=independent_joint(live,x)
            for aa,bb in zip(a,b): torch.testing.assert_close(aa,bb,rtol=3e-5,atol=2e-6)
            objective=a[0].square().mean()+.7*a[1].square().mean()
        objective.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                   for n,p in live.named_parameters() if 'nonlinear_' in n)
        precision['AMP']=dict(active_new_loss_gradients=True)
    torch.testing.assert_close(x,guard,rtol=0,atol=0)
    return dict(status='passed',parent_E47=parent_tests,activated_E47_initial_identity=True,
                common_preclip_gradients_equal=True,independent_joint_oracle=shapes,
                task_exclusive_added_paths=True,loss_only_two_stage_learning=True,
                nonlinear_bypass_exact_E47=True,caller_untouched=True,precision=precision)
