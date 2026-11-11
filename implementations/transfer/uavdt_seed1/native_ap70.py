"""Port of author overall evalRes(.7) + VOCap. Class-agnostic supplement."""
from collections import defaultdict
import numpy as np
from common import now, require


def frame_matches(gt, predictions, threshold=.7):
    """No ignore GT in author's overall call. On equal IoU, last GT wins."""
    g=np.asarray(gt,dtype=np.float64).reshape(-1,4)
    dt=sorted(predictions,key=lambda p:-p['score'])
    used=np.zeros(len(g),dtype=bool); flags=[]
    for p in dt:
        box=np.asarray(p['bbox'],dtype=np.float64)
        matched=False
        if len(g):
            wh=np.maximum(0,np.minimum(g[:,:2]+g[:,2:],box[:2]+box[2:])-np.maximum(g[:,:2],box[:2]))
            inter=wh.prod(axis=1); union=g[:,2:].prod(axis=1)+box[2:].prod()-inter
            overlap=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
            overlap[used]=-1
            best=overlap.max()
            if best>=threshold:
                index=np.flatnonzero(overlap==best)[-1]; used[index]=True; matched=True
        flags.append((p['score'],int(matched)))
    return flags


def integral_ap(flags, total_gt):
    if not flags: return 0.
    values=np.asarray(sorted(flags,key=lambda p:-p[0]),dtype=np.float64)
    tp=np.cumsum(values[:,1]==1); fp=np.cumsum(values[:,1]==0)
    rec=tp/max(1,total_gt); prec=tp/np.maximum(1,tp+fp)
    mr=np.r_[0,rec,1]; mp=np.r_[0,prec,0]
    mp=np.maximum.accumulate(mp[::-1])[::-1]
    idx=np.flatnonzero(mr[1:]!=mr[:-1])+1
    return float(np.sum((mr[idx]-mr[idx-1])*mp[idx]))*100


def score(gt, predictions, progress=None):
    by_gt=defaultdict(list); by_pred=defaultdict(list)
    for a in gt['annotations']: by_gt[a['image_id']].append(a['bbox'])
    for p in predictions: by_pred[p['image_id']].append(p)
    ids=[r['id'] for r in gt['images']]
    require(set(by_pred).issubset(ids),'Native AP image outside GT')
    flags=[]
    for i,image_id in enumerate(ids,1):
        flags.extend(frame_matches(by_gt[image_id],by_pred[image_id]))
        if progress and (i%1000==0 or i==len(ids)):
            progress(i,len(ids))
            print(f'{now()} NATIVE AP70 MATCH {i}/{len(ids)}',flush=True)
    return dict(AP70_overall_percent=integral_ap(flags,len(gt['annotations'])),
                images=len(ids),GT_boxes=len(gt['annotations']),detections=len(predictions),
                class_agnostic=True,IoU_threshold=.7,integral_VOC_AP=True,
                implementation='Python author-overall-math port; not MATLAB execution',
                predictions='Same fixed mapped three-class NMS/top500 predictions after strict ignore filter')
