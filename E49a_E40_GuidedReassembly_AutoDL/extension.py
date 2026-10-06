"""Independent sampling/index/gradient/precision oracles for E49."""
import copy
import torch
from torch.nn import functional as F


def compare_gradients(a,b,name,exact=False):
    torch.testing.assert_close(a,b,rtol=0 if exact else 3e-4,
                               atol=0 if exact else 1e-7,msg=name)


def branch_groups(unused=None):
    old=[f'task_blocks.{i}.{part}.' for i in range(2) for part in
         ('reduce','depths.0','depths.1','depths.2','router','out_cls','out_reg')]
    return old+['reassembly.gain','reassembly.coarse_project.',
                'reassembly.fine_project.','reassembly.kernel.0.','reassembly.kernel.2.']


def independent_reassemble(coarse,kernels):
    """Slow direct fine-grid indexing; no unfold, reshape-phase or einsum."""
    b,c,h,w=coarse.shape
    answer=[]
    for y in range(2*h):
        row=[]
        for x in range(2*w):
            values=[]
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    yy=max(0,min(h-1,y//2+dy)); xx=max(0,min(w-1,x//2+dx))
                    values.append(coarse[:,:,yy,xx].float())
            row.append((torch.stack(values,2)*kernels[:,:,y,x].float()[:,None,:]).sum(2))
        answer.append(torch.stack(row,-1))
    return torch.stack(answer,-2)


def routing_audit(device,unused=None):
    from guided_reassembly import GuidedReassembly,reassemble
    tests=[]
    for h,w in ((2,3),(3,2),(1,1),(3,5)):
        coarse=torch.arange(2*3*h*w,device=device,dtype=torch.float32).reshape(2,3,h,w)
        kernels=torch.zeros(2,9,2*h,2*w,device=device)
        # All nine offsets across fine cells, not only a central identity test.
        for y in range(2*h):
            for x in range(2*w): kernels[:,(y*2*w+x)%9,y,x]=1
        actual=reassemble(coarse,kernels)
        torch.testing.assert_close(actual,independent_reassemble(coarse,kernels),rtol=0,atol=0)
        torch.testing.assert_close(reassemble(torch.ones_like(coarse),kernels),
                                   torch.ones_like(actual),rtol=0,atol=0)
        random=torch.rand_like(kernels).softmax(1)
        torch.testing.assert_close(reassemble(coarse,random),independent_reassemble(coarse,random),
                                   rtol=2e-6,atol=2e-5)
        tests.append([h,w])
    # An impulse independently catches swapped axes / mixed output phases.
    impulse=torch.zeros(1,1,3,5,device=device); impulse[:,:,1,2]=1
    ks=torch.zeros(1,9,6,10,device=device)
    for y in range(6):
        for x in range(10): ks[:,(y%2)*3+(x%2),y,x]=1
    torch.testing.assert_close(reassemble(impulse,ks),independent_reassemble(impulse,ks),rtol=0,atol=0)
    ca=torch.randn(2,3,3,5,device=device,requires_grad=True)
    ka=torch.rand(2,9,6,10,device=device,requires_grad=True)
    cb=ca.detach().clone().requires_grad_(); kb=ka.detach().clone().requires_grad_()
    probe=torch.randn(2,3,6,10,device=device)
    (reassemble(ca,ka)*probe).sum().backward()
    (independent_reassemble(cb,kb)*probe).sum().backward()
    compare_gradients(ca.grad,cb.grad,'coarse indexing gradient')
    compare_gradients(ka.grad,kb.grad,'kernel indexing gradient')

    module=GuidedReassembly().to(device)
    coarse=torch.randn(2,128,3,5,device=device)
    fine=torch.randn(2,128,6,10,device=device)
    guard=[coarse.clone(),fine.clone()]
    nearest=F.interpolate(coarse,scale_factor=2,mode='nearest')
    torch.testing.assert_close(module([coarse,fine]),nearest,rtol=0,atol=0)
    weights=module.weights(coarse,fine)
    torch.testing.assert_close(weights.sum(1),torch.ones_like(weights[:,0]),rtol=2e-6,atol=2e-6)
    if (weights[:,:,0::2,0::2]-weights[:,:,1::2,1::2]).abs().sum()==0:
        raise RuntimeError('Fine descriptors do not produce phase-specific kernels')
    optimizer=torch.optim.SGD(module.parameters(),lr=.1,weight_decay=0,momentum=0)
    objective=(module([coarse,fine])-torch.randn_like(nearest)).square().mean()
    objective.backward()
    if module.gain.grad is None or not torch.isfinite(module.gain.grad) or module.gain.grad.abs()==0:
        raise RuntimeError('Zero-start gain has no finite loss gradient')
    for n,p in module.named_parameters():
        if n!='gain' and (p.grad is None or p.grad.count_nonzero()!=0):
            raise RuntimeError('Wrong initial upstream gradient '+n)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    before={n:p.detach().clone() for n,p in module.named_parameters()}
    objective=(module([coarse,fine])-torch.randn_like(nearest)).square().mean()
    objective.backward()
    for n,p in module.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum()==0:
            raise RuntimeError('No second-step loss gradient '+n)
    optimizer.step()
    for n,p in module.named_parameters():
        if torch.equal(p,before[n]): raise RuntimeError('No loss-only parameter update '+n)
    active=module([coarse,fine])
    oracle=nearest+module.gain.tanh()*(independent_reassemble(coarse,module.weights(coarse,fine))-nearest)
    torch.testing.assert_close(active,oracle,rtol=3e-5,atol=2e-6)
    module.enabled=False
    torch.testing.assert_close(module([coarse,fine]),nearest,rtol=0,atol=0)
    module.enabled=True
    for a,b in zip((coarse,fine),guard): torch.testing.assert_close(a,b,rtol=0,atol=0)
    dtype=torch.float16 if str(device)!='cpu' else torch.bfloat16
    half=copy.deepcopy(module).to(dtype)
    hc,hf=coarse.to(dtype),fine.to(dtype)
    with torch.no_grad():
        kw=half.weights(hc,hf)
        actual=half([hc,hf])
        hn=F.interpolate(hc,scale_factor=2,mode='nearest').float()
        expected=(hn+half.gain.float().tanh()*(independent_reassemble(hc,kw)-hn)).to(dtype)
        # Independent scalar sums and einsum can round on opposite sides of a
        # stored-half boundary. Permit ~2 dtype ULP, not bitwise equality.
        torch.testing.assert_close(actual.float(),expected.float(),
                                   rtol=2*torch.finfo(dtype).eps,atol=2e-6)
        if actual.dtype!=dtype or not torch.isfinite(actual).all() or kw.dtype!=torch.float32:
            raise RuntimeError('Stored-half reassembly precision/dtype failed')
    if str(device)!='cpu':
        module.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            value=module([coarse,fine])
            loss=value.square().mean()
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in module.parameters()):
            raise RuntimeError('AMP active reconstruction gradient failed')
    return dict(status='passed',direct_index_oracle=tests,impulse_and_border=True,
                independent_gradients=True,phase_conditioning=True,weights_normalized=True,
                initial_nearest_exact=True,loss_only_two_step_updates=True,
                inputs_untouched=True,bypass_nearest_exact=True,stored_half_value_oracle=True,
                FP32_weight_normalization_and_reassembly=True)
