"""Profile at REAL 640 resolution: fixed pooled K/V costs must not scale by H*W.

Functional FP32 projections do not call leaf module THOP hooks. Register a
custom branch hook counting all convolution/linear/QK/AV multiply-accumulates.
Norm, GELU, softmax, pooling and memory traffic are excluded, not 'free'.
"""
import copy
import torch


def attention_mac_breakdown(h,w,batch=1):
    from global_context import CHANNELS,HIDDEN,HEADS,POOL
    n=h*w; k=POOL*POOL; c=HIDDEN
    return dict(reduce=batch*n*CHANNELS*c,position=batch*n*c*9,
                query=batch*n*c*c,pool_project=batch*k*c*c,key_value=batch*k*c*2*c,
                QK=batch*n*k*c,AV=batch*n*k*c,attention_project=batch*n*c*c,
                ffn_in=batch*n*c*2*c,ffn_local=batch*n*2*c*9,
                ffn_out=batch*n*2*c*c,output=batch*n*c*CHANNELS)


def context_profile(model,parent,device='cpu',imgsz=640):
    import thop
    from global_context import GlobalContextResidual
    target=copy.deepcopy(model).to(device).float().eval()
    reference=copy.deepcopy(parent).to(device).float().eval()
    traced=[]
    def count(module,inputs,output):
        x=inputs[0]; b,_,h,w=x.shape
        details=attention_mac_breakdown(h,w,b)
        module.total_ops += sum(details.values())
        traced.append(dict(shape=list(x.shape),Q_tokens=h*w,KV_tokens=49,MACs=details))
    image=torch.zeros(1,3,imgsz,imgsz,device=device)
    with torch.no_grad():
        actual,_=thop.profile(target,inputs=(image,),custom_ops={GlobalContextResidual:count},verbose=False)
        old,_=thop.profile(reference,inputs=(image,),verbose=False)
    expected=sum(attention_mac_breakdown(imgsz//8,imgsz//8).values())*2/1e9
    delta=(actual-old)*2/1e9
    if len(traced)!=1 or traced[0]['shape'][2:]!=[imgsz//8,imgsz//8] or abs(delta-expected)>1e-7:
        raise RuntimeError('Real-shape full-model profiling mismatch: '+str((delta,expected,traced)))
    # Independently use PyTorch's operator-level FLOPs (not the custom formula).
    module=copy.deepcopy(model.model[4].global_context).cpu().float().eval()
    x=torch.zeros(1,256,imgsz//8,imgsz//8)
    with torch.no_grad(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:
        module(x)
    operator_flops={event.key:int(event.flops) for event in prof.key_averages() if event.flops}
    major=sum(v for k,v in operator_flops.items() if k in ('aten::conv2d','aten::mm','aten::bmm','aten::addmm'))
    if abs(major/1e9-expected)>1e-7:
        raise RuntimeError('Independent operator FLOPs mismatch: '+str((major,expected,operator_flops)))
    return dict(GFLOPs640=actual*2/1e9,E40_GFLOPs640=old*2/1e9,
                added_major_GFLOPs640=delta,actual_trace=traced,
                torch_operator_flops=operator_flops,independent_major_flops_verified=True,
                boundary='2 FLOPs/MAC; actual640 shape; QK/AV and projections included; norm/softmax/pool/GELU/copies excluded; not latency.')
