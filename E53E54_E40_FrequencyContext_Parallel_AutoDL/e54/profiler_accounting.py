"""Actual 640 grid, including padded stationary-wavelet convolutions.

2 FLOPs/MAC for convolutions. Fixed Haar additions/scales, GN, sigmoid,
softmax, roll/crop and memory traffic are separately disclosed, not zero cost.
"""
import copy
import torch
from representation import VARIANT,SITE,HALO,FrequencyResidual,ContextResidual


def mac_breakdown(h,w,b=1):
    n=b*h*w
    if VARIANT=='e53':
        padded=b*(h+2*HALO)*(w+2*HALO)
        return dict(reduce=n*128*32,output=n*32*128,local=n*32*9,
            low=padded*32*25,high=padded*2*96*25,
            selector_project=padded*2*128*16,
            selector_dw=padded*2*16*9,selector_output=padded*2*16*3)
    return dict(reduce=n*128*48,output=n*48*128,local=n*48*9,
                near=n*48*25,far=n*48*49,selector=n*2*3*49)


def representation_profile(model,parent,device='cpu',imgsz=640):
    import thop
    target=copy.deepcopy(model).to(device).float().eval()
    reference=copy.deepcopy(parent).to(device).float().eval()
    traced=[]
    def count(module,inputs,output):
        b,_,h,w=inputs[0].shape
        details=mac_breakdown(h,w,b)
        module.total_ops+=sum(details.values())
        traced.append(dict(shape=list(inputs[0].shape),MACs=details))
    image=torch.zeros(1,3,imgsz,imgsz,device=device)
    with torch.no_grad():
        actual,_=thop.profile(target,inputs=(image,),custom_ops={FrequencyResidual:count,ContextResidual:count},verbose=False)
        old,_=thop.profile(reference,inputs=(image,),verbose=False)
    stride=4 if SITE==25 else 8
    expected=sum(mac_breakdown(imgsz//stride,imgsz//stride).values())*2/1e9
    delta=(actual-old)*2/1e9
    if len(traced)!=1 or abs(delta-expected)>1e-7:
        raise RuntimeError('Full graph/profile delta drift '+str((delta,expected,traced)))
    module=copy.deepcopy(model.model[SITE].representation).cpu().float().eval()
    x=torch.zeros(1,128,imgsz//stride,imgsz//stride)
    with torch.no_grad(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:
        module(x)
    ops={e.key:int(e.flops) for e in prof.key_averages() if e.flops}
    major=ops.get('aten::conv2d',0)
    if abs(major/1e9-expected)>1e-7: raise RuntimeError('Independent operator profile drift '+str((major,expected,ops)))
    return dict(GFLOPs640=actual*2/1e9,E40_GFLOPs640=old*2/1e9,
        added_major_GFLOPs640=delta,actual_trace=traced,torch_operator_flops=ops,
        independent_major_flops_verified=True,
        boundary='Conv 2FLOPs/MAC at actual640. Haar/normalization/selection/copies disclosed separately; not a latency guarantee.')
