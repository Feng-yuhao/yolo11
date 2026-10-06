"""Independent loop/unfold value and gradient oracles; no causal claims."""
import copy
import torch
from torch.nn import functional as F
from guided_reassembly import VARIANT,GuidedReassembly,reassemble,local_average

def compare_gradients(a,b,name,exact=False):
    torch.testing.assert_close(a,b,rtol=0 if exact else 3e-4,atol=0 if exact else 1e-7,msg=name)
def branch_groups(unused=None):
    return [f'task_blocks.{i}.{part}.' for i in range(2) for part in
        ('reduce','depths.0','depths.1','depths.2','router','out_cls','out_reg')]+[
        'reassembly.gain','reassembly.coarse_project.','reassembly.fine_project.',
        'reassembly.kernel.0.','reassembly.kernel.2.']
def independent_reassemble(coarse,kernels):
    b,c,h,w=coarse.shape
    rows=[]
    for y in range(2*h):
        cells=[]
        for x in range(2*w):
            values=[coarse[:,:,max(0,min(h-1,y//2+dy)),max(0,min(w-1,x//2+dx))].float()
                for dy in (-1,0,1) for dx in (-1,0,1)]
            cells.append((torch.stack(values,2)*kernels[:,:,y,x].float()[:,None]).sum(2))
        rows.append(torch.stack(cells,-1))
    return torch.stack(rows,-2)
def independent_average(fine,kernels):
    b,c,h,w=fine.shape
    patches=F.unfold(F.pad(fine.float(),(1,1,1,1),mode='replicate'),3).reshape(b,c,9,h,w)
    return (patches*kernels.float()[:,None]).sum(2)
def inputs(c,f):
    return [c,f] if VARIANT=='e50' else [F.interpolate(c,scale_factor=2,mode='nearest'),f,c]
def active_oracle(module,c,f):
    w=module.weights(c,f)
    n=F.interpolate(c,scale_factor=2,mode='nearest')
    if VARIANT=='e50':
        return (n.float()+module.gain.float().tanh()*(independent_reassemble(c,w)-n.float())).to(c.dtype)
    v=(f.float()+module.gain.float().tanh()*(f.float()-independent_average(f,w))).to(f.dtype)
    return torch.cat((n,v),1)
def routing_audit(device,unused=None):
    tested=[]
    for h,w in ((2,3),(3,2),(1,1),(3,5)):
        c=torch.arange(2*3*h*w,device=device,dtype=torch.float32).reshape(2,3,h,w)
        k=torch.zeros(2,9,2*h,2*w,device=device)
        for y in range(2*h):
            for x in range(2*w): k[:,(y*2*w+x)%9,y,x]=1
        torch.testing.assert_close(reassemble(c,k),independent_reassemble(c,k),rtol=0,atol=0)
        f=torch.randn(2,3,2*h,2*w,device=device)
        torch.testing.assert_close(local_average(f,k),independent_average(f,k),rtol=0,atol=0)
        for value,operation,oracle in ((c,reassemble,independent_reassemble),(f,local_average,independent_average)):
            a=value.clone().requires_grad_(); b=value.clone().requires_grad_()
            ka=torch.rand_like(k).softmax(1).requires_grad_(); kb=ka.detach().clone().requires_grad_()
            probe=torch.randn(2,3,2*h,2*w,device=device)
            va=operation(a,ka); vb=oracle(b,kb)
            torch.testing.assert_close(va,vb,rtol=3e-6,atol=2e-5)
            (va*probe).sum().backward(); (vb*probe).sum().backward()
            compare_gradients(a.grad,b.grad,'value indexing'); compare_gradients(ka.grad,kb.grad,'kernel indexing')
            constant=torch.ones_like(value)
            torch.testing.assert_close(operation(constant,ka.detach()),torch.ones_like(va),rtol=2e-6,atol=2e-6)
        tested.append([h,w])
    module=GuidedReassembly().to(device)
    c=torch.randn(2,128,3,5,device=device); f=torch.randn(2,128,6,10,device=device)
    guard=[c.clone(),f.clone()]
    n=F.interpolate(c,scale_factor=2,mode='nearest')
    zero=n if VARIANT=='e50' else torch.cat((n,f),1)
    torch.testing.assert_close(module(inputs(c,f)),zero,rtol=0,atol=0)
    shapes=[]
    hook=module.kernel[0].register_forward_pre_hook(lambda m,x:shapes.append(list(x[0].shape[-2:])))
    weights=module.weights(c,f); hook.remove()
    if shapes!=[[3,5],[6,10]]: raise RuntimeError('DW not called at two native resolutions')
    torch.testing.assert_close(weights.sum(1),torch.ones_like(weights[:,0]),rtol=2e-6,atol=2e-6)
    # Independent descriptor arithmetic, including shared DW placement.
    cp=F.conv2d(F.silu(module.coarse_project(c)),module.kernel[0].weight,padding=1,groups=32)
    fp=F.conv2d(F.silu(module.fine_project(f)),module.kernel[0].weight,padding=1,groups=32)
    logits=F.conv2d(F.silu(fp+F.interpolate(cp,scale_factor=2,mode='nearest')),module.kernel[2].weight,module.kernel[2].bias)
    torch.testing.assert_close(weights,logits.float().softmax(1),rtol=0,atol=0)
    opt=torch.optim.SGD(module.parameters(),lr=.1,weight_decay=0)
    module(inputs(c,f)).square().mean().backward()
    if module.gain.grad is None or not torch.isfinite(module.gain.grad) or module.gain.grad.abs()==0:
        raise RuntimeError('Gain has no real first loss gradient')
    for name,p in module.named_parameters():
        if name!='gain' and (p.grad is None or p.grad.count_nonzero()!=0): raise RuntimeError('Wrong zero-start gradient '+name)
    opt.step(); opt.zero_grad(set_to_none=True)
    before={name:p.detach().clone() for name,p in module.named_parameters()}
    (module(inputs(c,f))-torch.randn_like(zero)).square().mean().backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum()==0 for p in module.parameters()):
        raise RuntimeError('Missing second-step gradient')
    opt.step()
    if any(torch.equal(p,before[name]) for name,p in module.named_parameters()): raise RuntimeError('No actual parameter update')
    torch.testing.assert_close(module(inputs(c,f)),active_oracle(module,c,f),rtol=3e-5,atol=2e-6)
    module.enabled=False; torch.testing.assert_close(module(inputs(c,f)),zero,rtol=0,atol=0); module.enabled=True
    for a,b in zip((c,f),guard): torch.testing.assert_close(a,b,rtol=0,atol=0)
    if VARIANT=='e51':
        constant=torch.ones_like(f)
        result=module(inputs(c,constant))
        torch.testing.assert_close(result[:,128:],constant,rtol=2e-6,atol=2e-6)
        torch.testing.assert_close(result[:,:128],n,rtol=0,atol=0)
    dtype=torch.float16 if str(device)!='cpu' else torch.bfloat16
    half=copy.deepcopy(module).to(dtype); hc,hf=c.to(dtype),f.to(dtype)
    with torch.no_grad():
        actual=half(inputs(hc,hf)); expected=active_oracle(half,hc,hf)
        torch.testing.assert_close(actual.float(),expected.float(),rtol=2*torch.finfo(dtype).eps,atol=2e-6)
        if actual.dtype!=dtype or not torch.isfinite(actual).all(): raise RuntimeError('Stored half value failure')
    if str(device)!='cpu':
        module.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16): loss=module(inputs(c,f)).square().mean()
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in module.parameters()): raise RuntimeError('AMP gradients fail')
    return dict(status='passed',direct_index_and_unfold_oracles=tested,independent_gradients=True,
        native_grid_shared_DW=True,zero_start_exact=True,loss_only_two_step_updates=True,
        inputs_untouched=True,constant_preservation=True,stored_half_value_oracle=True,
        FP32_weights_and_accumulation=True)
