"""Independent attention, gradient and precision oracles (not performance probes)."""
import copy
import torch
from torch.nn import functional as F


def compare_gradients(a,b,name,exact=False):
    torch.testing.assert_close(a,b,rtol=0 if exact else 3e-4,
                               atol=0 if exact else 1e-7,msg=name)


def branch_groups(unused=None):
    old=[f'task_blocks.{i}.{part}.' for i in range(2) for part in
         ('reduce','depths.0','depths.1','depths.2','router','out_cls','out_reg')]
    return old+['context.'+part+'.' for part in
                ('reduce','position','query_norm','query','pool_project','pool_norm',
                 'key_value','attn_project','ffn_norm','ffn_in','ffn_local','ffn_out','output')]


def independent_attention(module,tokens,h,w):
    """Per-batch/head reference, no production head reshape/permute machinery."""
    from global_context import HIDDEN,HEADS,POOL
    q=F.linear(F.layer_norm(tokens,(HIDDEN,),module.query_norm.weight.float(),
                           module.query_norm.bias.float(),module.query_norm.eps),module.query.weight.float())
    image=tokens.transpose(1,2).reshape(len(tokens),HIDDEN,h,w)
    # Independently reproduce adaptive pooling boundaries (including odd grids).
    import math
    rows=[]
    for y in range(POOL):
        cells=[]
        for x in range(POOL):
            cells.append(image[:,:,math.floor(y*h/POOL):math.ceil((y+1)*h/POOL),
                                math.floor(x*w/POOL):math.ceil((x+1)*w/POOL)].mean((-1,-2)))
        rows.append(torch.stack(cells,-1))
    pooled=torch.stack(rows,-2)
    pooled=F.conv2d(pooled,module.pool_project.weight.float()).flatten(2).transpose(1,2)
    pooled=F.gelu(F.layer_norm(pooled,(HIDDEN,),module.pool_norm.weight.float(),
                             module.pool_norm.bias.float(),module.pool_norm.eps))
    kv=F.linear(pooled,module.key_value.weight.float())
    batches=[]
    d=HIDDEN//HEADS
    for b in range(len(tokens)):
        heads=[]
        for head in range(HEADS):
            query=q[b,:,head*d:(head+1)*d]
            key=kv[b,:,head*d:(head+1)*d]
            value=kv[b,:,HIDDEN+head*d:HIDDEN+(head+1)*d]
            weights=(query @ key.T * d**-.5).softmax(1)
            heads.append(weights @ value)
        batches.append(torch.cat(heads,-1))
    return F.linear(torch.stack(batches),module.attn_project.weight.float())


def routing_audit(device,unused=None):
    from global_context import GlobalContextResidual,ADDED_PARAMETERS,HIDDEN
    module=GlobalContextResidual().to(device)
    if sum(p.numel() for p in module.parameters())!=ADDED_PARAMETERS:
        raise RuntimeError('Global context parameter arithmetic drift')
    layouts=[]
    for h,w in ((3,5),(8,11),(7,7),(1,1)):
        tokens=torch.randn(2,h*w,HIDDEN,device=device)
        actual,weights=module.attention(tokens,h,w,return_weights=True)
        expected=independent_attention(module,tokens,h,w)
        torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6)
        torch.testing.assert_close(weights.sum(-1),torch.ones_like(weights[...,0]),rtol=2e-6,atol=2e-6)
        if tuple(weights.shape)!=(2,4,h*w,49): raise RuntimeError('Incorrect query/context axes')
        layouts.append([h,w])
    tokens=torch.randn(2,8*11,HIDDEN,device=device,requires_grad=True)
    other=tokens.detach().clone().requires_grad_()
    probe=torch.randn_like(tokens)
    grads=torch.autograd.grad((module.attention(tokens,8,11)*probe).sum(),tokens)[0]
    refgrads=torch.autograd.grad((independent_attention(module,other,8,11)*probe).sum(),other)[0]
    compare_gradients(grads,refgrads,'independent attention input gradient')
    # Far-away feature change affects the first query: capability check, not AP.
    remote=tokens.detach().clone(); remote[:,-1,:]+=torch.randn(HIDDEN,device=device)*3
    if torch.equal(module.attention(tokens.detach(),8,11)[:,0],module.attention(remote,8,11)[:,0]):
        raise RuntimeError('Nonlocal context is not reaching queries')
    x=torch.randn(2,256,8,11,device=device,requires_grad=True)
    guard=x.detach().clone()
    torch.testing.assert_close(module(x),x,rtol=0,atol=0)
    target=torch.randn_like(x)
    optimizer=torch.optim.SGD(module.parameters(),lr=.05,weight_decay=0,momentum=0)
    before={n:p.detach().clone() for n,p in module.named_parameters()}
    # Scale the disposable unit objective so FP32 norm weights initialized at
    # exactly 1 move by more than one ULP on step2. Formal loss is NOT rescaled.
    objective_scale=128.
    ((module(x)-target).square().mean()*objective_scale).backward()
    for n,p in module.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all(): raise RuntimeError('Missing initial gradient '+n)
        if n=='output.weight':
            if p.grad.abs().sum()==0: raise RuntimeError('Dead zero output projection')
        elif p.grad.count_nonzero()!=0: raise RuntimeError('Wrong zero-start upstream gradient '+n)
    torch.testing.assert_close(x.grad,2*objective_scale*(x.detach()-target)/x.numel(),rtol=2e-6,atol=1e-8)
    optimizer.step(); optimizer.zero_grad(set_to_none=True); x.grad=None
    ((module(x)-target).square().mean()*objective_scale).backward()
    for n,p in module.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum()==0:
            raise RuntimeError('Missing second-step branch gradient '+n)
    optimizer.step()
    for n,p in module.named_parameters():
        if torch.equal(p,before[n]): raise RuntimeError('No loss-only parameter update '+n)
    torch.testing.assert_close(x.detach(),guard,rtol=0,atol=0)
    module.enabled=False
    torch.testing.assert_close(module(x),x,rtol=0,atol=0); module.enabled=True
    dtype=torch.bfloat16 if str(device)=='cpu' else torch.float16
    half=copy.deepcopy(module).to(dtype); hx=x.detach().to(dtype)
    expected=copy.deepcopy(half).float()(hx.float()).to(dtype)
    actual=half(hx)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    if not torch.isfinite(actual).all() or actual.dtype!=dtype:
        raise RuntimeError('Stored-half FP32 branch value failure')
    if str(device)!='cpu':
        module.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16): loss=module(x).square().mean()
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in module.parameters()):
            raise RuntimeError('AMP branch gradients failed')
    return dict(status='passed',attention_oracle_layouts=layouts,independent_input_gradients=True,
                global_context_reaches_remote_queries=True,initial_identity_exact=True,
                loss_only_two_step_updates=True,input_unmodified=True,
                stored_half_value_exact_to_quantized_FP32_oracle=True)
