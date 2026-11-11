"""Fixed many-to-one semantic max before NMS, not post-hoc score selection."""
from common import SPECS,require
def map_scores(raw,target):
    import torch
    require(raw.ndim==3 and raw.shape[1]==14 and raw.dtype==torch.float32,'Expected original decoded source10 FP32 BCN')
    require(bool(torch.isfinite(raw).all()) and bool(((raw[:,4:]>=0)&(raw[:,4:]<=1)).all()),'Invalid source sigmoid scores')
    scores=raw[:,[4+i for i in SPECS[target]['indices']]].amax(dim=1,keepdim=True)
    return torch.cat((raw[:,:4],scores),dim=1)
def mapped_nms(raw,target):
    from ultralytics.utils.ops import non_max_suppression
    return non_max_suppression(map_scores(raw,target),.001,.7,nc=1,multi_label=False,agnostic=False,max_det=500,max_nms=30000)
def inverse_boxes(rows,input_hw,original_hw):
    output=rows.clone();ih,iw=input_hw;oh,ow=original_hw;gain=min(ih/oh,iw/ow)
    px=round((iw-ow*gain)/2-.1);py=round((ih-oh*gain)/2-.1)
    output[:,:4]-=output.new_tensor([px,py,px,py]);output[:,:4]/=gain
    output[:,[0,2]]=output[:,[0,2]].clamp(0,ow);output[:,[1,3]]=output[:,[1,3]].clamp(0,oh)
    return output
