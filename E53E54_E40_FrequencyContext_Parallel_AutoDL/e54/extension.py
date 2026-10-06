"""Independent Haar indexing/adjoint, selection and native two-step tests."""
import copy
import torch
from torch.nn import functional as F
from representation import VARIANT, HIDDEN, FrequencyResidual, ContextResidual, haar_analysis, haar_synthesis


def compare_gradients(a,b,name,exact=False):
    torch.testing.assert_close(a,b,rtol=0 if exact else 3e-4,atol=0 if exact else 1e-7,msg=name)


def branch_groups(unused=None):
    old=[f'task_blocks.{i}.{part}.' for i in range(2) for part in
         ('reduce','depths.0','depths.1','depths.2','router','out_cls','out_reg')]
    parts=('reduce','norm','local','low','high.0','high.1','selectors.0','selectors.1','output') if VARIANT=='e53' else (
           'reduce','norm','local','near','far','selector','output')
    return old+['representation.'+p+'.' for p in parts]


def shift_index(x,dy,dx):
    # Independent gather: no torch.roll or production transform calls.
    y=(torch.arange(x.shape[-2],device=x.device)+dy)%x.shape[-2]
    z=(torch.arange(x.shape[-1],device=x.device)+dx)%x.shape[-1]
    return x.index_select(-2,y).index_select(-1,z)


def analysis_oracle(x,d):
    return tuple(sum(shift_index(x,dy*d,dx*d)*coefficient for dy,dx,coefficient in terms)*.25 for terms in (
        ((0,0,1),(0,1,1),(1,0,1),(1,1,1)),
        ((0,0,1),(0,1,-1),(1,0,1),(1,1,-1)),
        ((0,0,1),(0,1,1),(1,0,-1),(1,1,-1)),
        ((0,0,1),(0,1,-1),(1,0,-1),(1,1,1))))


def synthesis_oracle(bands,d):
    signs=((1,1,1,1),(1,-1,1,-1),(1,1,-1,-1),(1,-1,-1,1))
    result=torch.zeros_like(bands[0])
    for band,coefficients in zip(bands,signs):
        for (dy,dx),v in zip(((0,0),(0,1),(1,0),(1,1)),coefficients):
            result=result+shift_index(band,-dy*d,-dx*d)*(.25*v)
    return result


def frequency_oracle(module,x):
    from representation import HALO
    z=F.silu(F.group_norm(F.conv2d(x,module.reduce.weight.float()),4,
                           module.norm.weight.float(),module.norm.bias.float(),module.norm.eps))
    # An independently extended domain with *different* halo validates that
    # production crop is insensitive to periodic wraps and outside padding.
    pad=HALO+8
    low=F.pad(z,(pad,)*4,mode='replicate'); levels=[]
    for i,d in enumerate((1,2)):
        low,*details=analysis_oracle(low,d)
        s=module.selectors[i]
        descriptor=torch.cat((low,*(v.abs() for v in details)),1)
        gate=F.conv2d(F.silu(F.conv2d(F.silu(F.conv2d(descriptor,s[0].weight.float(),s[0].bias.float())),
                s[2].weight.float(),s[2].bias.float(),padding=1,groups=16)),s[4].weight.float(),s[4].bias.float()).sigmoid()
        high=F.conv2d(torch.cat(details,1),module.high[i].weight.float(),padding=2*d,dilation=d,groups=96).chunk(3,1)
        levels.append(tuple(v*gate[:,j:j+1] for j,v in enumerate(high)))
    low=F.conv2d(low,module.low.weight.float(),padding=4,dilation=2,groups=32)
    for i in (1,0): low=synthesis_oracle((low,*levels[i]),2**i)
    return low[...,pad:-pad,pad:-pad]+F.conv2d(z,module.local.weight.float(),padding=1,groups=32)


def context_oracle(module,x):
    z=F.silu(F.group_norm(F.conv2d(x,module.reduce.weight.float()),6,
                           module.norm.weight.float(),module.norm.bias.float(),module.norm.eps))
    near=F.conv2d(z,module.near.weight.float(),padding=2,groups=48)
    far=F.conv2d(near,module.far.weight.float(),padding=9,dilation=3,groups=48)
    context=torch.cat((near,far),1)
    descriptor=torch.stack((context.mean(1),context.max(1)[0]),1)
    weights=F.conv2d(descriptor,module.selector.weight.float(),module.selector.bias.float(),padding=3).softmax(1)
    return F.conv2d(z,module.local.weight.float(),padding=1,groups=48)+weights[:,1:2]*near+weights[:,2:3]*far


def routing_audit(device,unused=None):
    from representation import ADDED_PARAMETERS
    layouts=((1,1),(3,5),(8,11),(17,19))
    for h,w in layouts:
        x=torch.randn(2,3,h,w,device=device,requires_grad=True)
        for d in (1,2):
            bands=haar_analysis(x,d); oracle=analysis_oracle(x,d)
            for a,b in zip(bands,oracle): torch.testing.assert_close(a,b,rtol=0,atol=0)
            reconstruction=haar_synthesis(bands,d)
            torch.testing.assert_close(reconstruction,x,rtol=2e-6,atol=5e-7)
            arbitrary=tuple(torch.randn_like(x) for _ in range(4))
            torch.testing.assert_close(haar_synthesis(arbitrary,d),synthesis_oracle(arbitrary,d),rtol=2e-6,atol=5e-7)
            probe=torch.randn_like(x)
            g1=torch.autograd.grad((reconstruction*probe).sum(),x,retain_graph=True)[0]
            g2=torch.autograd.grad((synthesis_oracle(analysis_oracle(x,d),d)*probe).sum(),x)[0]
            compare_gradients(g1,g2,'Haar adjoint gradient')
    module=(FrequencyResidual() if VARIANT=='e53' else ContextResidual()).to(device)
    if sum(p.numel() for p in module.parameters())!=ADDED_PARAMETERS: raise RuntimeError('Parameter count drift')
    oracle=frequency_oracle if VARIANT=='e53' else context_oracle
    for h,w in ((1,1),(3,5),(8,11)):
        x=torch.randn(2,128,h,w,device=device)
        actual,weights=module.features(x)
        torch.testing.assert_close(actual,oracle(module,x),rtol=3e-5,atol=3e-6)
        if VARIANT=='e53':
            if not all(bool(((v>0)&(v<1)).all()) for v in weights): raise RuntimeError('Not soft detail selection')
        else:
            torch.testing.assert_close(weights.sum(1),torch.ones_like(weights[:,0]),rtol=2e-6,atol=2e-6)
    if VARIANT=='e54':
        # Explicit null option: cannot remove the independent local branch.
        saved=copy.deepcopy(module.selector.state_dict())
        with torch.no_grad():
            module.selector.weight.zero_(); module.selector.bias.copy_(torch.tensor([80.,-80.,-80.],device=device))
        x=torch.randn(2,128,13,15,device=device)
        from representation import conv,normalized
        local=conv(module.local,F.silu(normalized(module.norm,conv(module.reduce,x))))
        torch.testing.assert_close(module.features(x)[0],local,rtol=0,atol=0)
        module.selector.load_state_dict(saved)
        # 5x5 then dilation3 7x7 covers every offset in a 23x23 square.
        support={a+3*b for a in range(-2,3) for b in range(-3,4)}
        if support!=set(range(-11,12)): raise RuntimeError('Sparse nominal context support')
    x=torch.randn(2,128,8,11,device=device,requires_grad=True); guard=x.detach().clone()
    torch.testing.assert_close(module(x),x,rtol=0,atol=0)
    target=torch.randn_like(x); optimizer=torch.optim.SGD(module.parameters(),lr=.05,weight_decay=0)
    before={n:p.detach().clone() for n,p in module.named_parameters()}
    scale=128.
    ((module(x)-target).square().mean()*scale).backward()
    for n,p in module.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all(): raise RuntimeError('Missing first gradient '+n)
        if (p.grad.abs().sum()>0)!=(n=='output.weight'): raise RuntimeError('Wrong zero-start gradient '+n)
    torch.testing.assert_close(x.grad,2*scale*(x.detach()-target)/x.numel(),rtol=2e-6,atol=1e-8)
    optimizer.step(); optimizer.zero_grad(set_to_none=True); x.grad=None
    ((module(x)-target).square().mean()*scale).backward()
    for n,p in module.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum()==0:
            raise RuntimeError('Missing second-step gradient '+n)
    optimizer.step()
    if any(torch.equal(p,before[n]) for n,p in module.named_parameters()): raise RuntimeError('Dead native two-step update')
    expected=x.detach()+F.conv2d(oracle(module,x.detach()),module.output.weight.float())
    torch.testing.assert_close(module(x.detach()),expected,rtol=3e-5,atol=3e-6)
    torch.testing.assert_close(x.detach(),guard,rtol=0,atol=0)
    module.enabled=False; torch.testing.assert_close(module(x),x,rtol=0,atol=0); module.enabled=True
    dtype=torch.bfloat16 if str(device)=='cpu' else torch.float16
    half=copy.deepcopy(module).to(dtype); hx=x.detach().to(dtype)
    torch.testing.assert_close(half(hx),copy.deepcopy(half).float()(hx.float()).to(dtype),rtol=0,atol=0)
    if str(device)!='cpu':
        module.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16): loss=module(x).square().mean()
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in module.parameters()): raise RuntimeError('AMP branch fail')
    return dict(status='passed',independent_transform_adjoint_gradients=True,layouts=layouts,
        zero_start_exact=True,native_two_step_updates=True,inputs_untouched=True,
        independent_full_branch_oracle=True,different_halo_crop_verified=VARIANT=='e53',
        stored_half_value_exact=True,context_bypass_verified=VARIANT=='e54')
