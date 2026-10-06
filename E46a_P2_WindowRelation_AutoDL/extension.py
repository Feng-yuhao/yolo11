"""Independent relation/mask checks and explicit parameter-gradient groups."""
import copy
import torch
from torch import nn
from window_context import partition,reverse,window_mask,WindowBlock,WindowContext


def compare_gradients(a,b,name,exact=False):
    if exact:
        torch.testing.assert_close(a,b,rtol=0,atol=0,msg=name)
    else:
        torch.testing.assert_close(a,b,rtol=3e-4,atol=1e-7,msg=name)


def branch_groups(unused=None):
    old=[f'{i}.{part}.' for i in range(2) for part in
         ('reduce','depths.0','depths.1','depths.2','router','out_cls','out_reg')]
    return old+[f'0.context.{part}.' for part in ('reduce','blocks.0','blocks.1','out_cls','out_reg')]


def coordinate_mask_oracle(h,w,shift,device):
    """Independent noncyclic shifted-window group IDs in original coordinates."""
    hp=((h+7)//8)*8; wp=((w+7)//8)*8
    y,x=torch.meshgrid(torch.arange(hp,device=device),torch.arange(wp,device=device),indexing='ij')
    sy=(y+shift)%hp; sx=(x+shift)%wp
    # Each rolled window should not cross an original-image cyclic seam.
    gy=torch.div(sy-shift,8,rounding_mode='floor')
    gx=torch.div(sx-shift,8,rounding_mode='floor')
    ids=partition((gy*(wp//8+1)+gx)[None,:,:,None]).squeeze(-1)
    valid=partition(((sy<h)&(sx<w))[None,:,:,None]).squeeze(-1)
    allowed=(ids[:,:,None]==ids[:,None,:])&valid[:,None,:]
    empty=~allowed.any(-1)
    allowed=allowed|(empty[:,:,None]&torch.eye(64,device=device,dtype=torch.bool)[None])
    return allowed


def routing_audit(device,unused=None):
    from parent_spatial_task_head import SpatialTaskBlock
    from spatial_task_head import ContextTaskBlock,CONTEXT_PARAMETERS
    results=[]
    for h,w in ((16,24),(13,19),(1,1),(5,7),(8,8)):
        for shift in (0,4):
            mask,hp,wp=window_mask(h,w,shift,device)
            oracle=coordinate_mask_oracle(h,w,shift,device)
            if not torch.equal(mask,oracle): raise RuntimeError(f'Wrong cyclic/padding mask {h,w,shift}')
            a=torch.arange(hp*wp,device=device).reshape(1,hp,wp,1)
            torch.testing.assert_close(reverse(partition(a),1,hp,wp),a,rtol=0,atol=0)
            block=WindowBlock(shift).to(device).eval()
            image=torch.randn(1,h,w,64,device=device,requires_grad=True)
            value=block(image)
            value.square().mean().backward()
            if not torch.isfinite(value).all() or not torch.isfinite(image.grad).all():
                raise RuntimeError('Invalid padded attention forward/backward')
            # Independent per-window/head matrix product, including relative bias.
            tokens=torch.randn(len(mask),64,64,device=device)
            with torch.no_grad():
                actual=block.attention(tokens,mask,1)
                qkv=block.qkv(tokens).reshape(-1,64,3,4,16)
                expected=torch.zeros_like(tokens)
                bias=block.relative_bias[block.relative_index.flatten()].reshape(64,64,4)
                for win in range(len(mask)):
                    outputs=[]
                    for head in range(4):
                        q,k,v=(qkv[win,:,task,head] for task in range(3))
                        score=(q@k.T)*.25+bias[:,:,head]
                        score=score.masked_fill(~oracle[win],float('-inf'))
                        outputs.append(score.softmax(1)@v)
                    expected[win]=block.proj(torch.cat(outputs,1))
                torch.testing.assert_close(actual,expected,rtol=3e-5,atol=2e-6)
            results.append([h,w,shift])
    base=SpatialTaskBlock().to(device).eval()
    # Nonzero E40 outputs ensure identity isn't hidden behind E40's own zero init.
    nn.init.normal_(base.out_cls.weight,std=.02); nn.init.normal_(base.out_reg.weight,std=.02)
    target=copy.deepcopy(base); target.__class__=ContextTaskBlock
    target.context=WindowContext().to(device)
    if sum(p.numel() for p in target.context.parameters())!=CONTEXT_PARAMETERS: raise RuntimeError('Context count')
    x=torch.randn(2,128,13,19,device=device,requires_grad=True)
    with torch.no_grad():
        a=base(x); b=target(x)
        for aa,bb in zip(a,b): torch.testing.assert_close(aa,bb,rtol=0,atol=0)
    # Task-exclusive output projections; attention itself is SHARED across tasks.
    nn.init.normal_(target.context.out_cls.weight,std=.02)
    before=x.detach().clone()
    a=base(x); b=target(x)
    torch.testing.assert_close(b[1],a[1],rtol=0,atol=0)
    if torch.equal(b[0],a[0]): raise RuntimeError('Context classification route absent')
    torch.testing.assert_close(x.detach(),before,rtol=0,atol=0)
    target.context.out_cls.weight.data.zero_(); nn.init.normal_(target.context.out_reg.weight,std=.02)
    a=base(x); b=target(x)
    torch.testing.assert_close(b[0],a[0],rtol=0,atol=0)
    if torch.equal(b[1],a[1]): raise RuntimeError('Context regression route absent')
    # Capture both reductions to prove the original input reaches each branch.
    caught={}
    def capture(name):
        def hook(module,inputs): caught[name]=inputs[0].detach().clone()
        return hook
    hooks=[target.reduce.register_forward_pre_hook(capture('E40')),
           target.context.reduce.register_forward_pre_hook(capture('context'))]
    try:
        target(x)
        for value in caught.values(): torch.testing.assert_close(value,x.detach(),rtol=0,atol=0)
        if set(caught)!={'E40','context'}: raise RuntimeError('Missing parallel input')
    finally:
        for hook in hooks: hook.remove()
    # Softmax over KEY tokens means changing a neighbour value affects a query.
    attn=WindowBlock(0).to(device).eval()
    with torch.no_grad():
        attn.qkv.weight.zero_(); attn.qkv.bias.zero_()
        attn.qkv.weight[128:].copy_(torch.eye(64,device=device))
        attn.relative_bias.zero_(); attn.proj.weight.copy_(torch.eye(64,device=device)); attn.proj.bias.zero_()
        tokens=torch.zeros(1,64,64,device=device); tokens[0,1,0]=64
        mask=torch.ones(1,64,64,device=device,dtype=torch.bool)
        actual=attn.attention(tokens,mask,1)
        torch.testing.assert_close(actual[0,:,0],torch.ones(64,device=device),rtol=0,atol=0)
    precision={}
    # A loss-only active-branch update proves every context parameter is live;
    # no weight decay or momentum can manufacture apparent update evidence.
    live=WindowContext().to(device)
    nn.init.normal_(live.out_cls.weight,std=.02); nn.init.normal_(live.out_reg.weight,std=.02)
    initial={n:p.detach().clone() for n,p in live.named_parameters()}
    optimizer=torch.optim.SGD(live.parameters(),lr=.1,weight_decay=0,momentum=0)
    dc,dr=live(torch.randn(2,128,13,19,device=device))
    objective=dc.square().mean()+.7*dr.square().mean()
    objective.backward()
    for name,param in live.named_parameters():
        if param.grad is None or not torch.isfinite(param.grad).all() or param.grad.abs().sum()==0:
            raise RuntimeError('Inactive context loss gradient '+name)
    optimizer.step()
    for name,param in live.named_parameters():
        if torch.equal(param,initial[name]): raise RuntimeError('Inactive context loss-only update '+name)
    for dtype in ((torch.float16,) if str(device)!='cpu' else (torch.bfloat16,)):
        module=WindowContext().to(device).eval()
        nn.init.normal_(module.out_cls.weight,std=.01); nn.init.normal_(module.out_reg.weight,std=.01)
        converted=copy.deepcopy(module).to(dtype)
        inp=torch.randn(2,128,13,19,device=device,dtype=dtype)
        with torch.no_grad():
            expected=copy.deepcopy(converted).float()(inp.float())
            actual=converted(inp)
            for aa,bb in zip(actual,expected): torch.testing.assert_close(aa,bb,rtol=0,atol=0)
        precision[str(dtype)]=dict(FP32_context=True,no_parameter_mutation=all(p.dtype==dtype for p in converted.parameters()))
    return dict(status='passed',window_and_padding_masks=results,independent_QKV_oracle=True,
                key_softmax_and_cross_position_value=True,nonzero_E40_identity=True,
                separate_task_outputs=True,parallel_original_input=True,all_context_loss_only_updates=True,precision=precision)
