"""Frozen VisDrone Matlab1.0.4 ignore-region protocol, NOT native YOLO AP."""
from __future__ import annotations
from collections import Counter
import gc, logging, math, time
from pathlib import Path
from common7 import *

def options(device):
    return dict(imgsz=640,device=device,conf=.001,iou=.7,max_det=500,half=False,
                augment=False,verbose=False,save=False,agnostic_nms=False,classes=None)

def audit_dataset(args,split,verify_pixels=True):
    import d27_official as official
    count=548 if split=='val' else 1610
    path=args.dataset if split=='val' else args.test_dataset
    old=official.EXPECTED_IMAGES
    try:
        official.EXPECTED_IMAGES=count
        records,meta=official.load_dataset(path)
    finally:official.EXPECTED_IMAGES=old
    ref=read(HERE/'references/reference_protocol.json')['official_'+('val' if split=='val' else 'test')+'_dataset']
    expected={k:v for k,v in ref.items() if k not in ['dataset_root','split','image_content_sha256']}
    require({k:v for k,v in meta.items() if k!='dataset_root'}==expected,'Original annotation/dimension/ignore mapping changed: '+split)
    if split=='test-dev':
        m=read(HERE/'references/td_seed1/testdev_input_manifest.json')
        require(len(records)==len(m['images'])==1610,'Wrong test-dev count')
        require(m['dataset']['image_content_sha256']==ref['image_content_sha256'],'Wrong cached image identity')
        for r,i in zip(records,m['images']):
            require((r.image_id,r.stem,r.width,r.height)==(i['id'],i['stem'],i['width'],i['height']),'Test image ID differs')
            if verify_pixels:require(sha(r.image_path)==i['sha256'],'Test image bytes differ: '+r.stem)
        meta['image_content_sha256']=ref['image_content_sha256']
        meta['pixel_hashes_checked_this_phase']=bool(verify_pixels)
    return records,meta

def convert(record,data):
    answer=[]
    for x1,y1,x2,y2,score,cls in data:
        require(all(math.isfinite(v) for v in [x1,y1,x2,y2,score,cls]),'Nonfinite prediction')
        require(cls==int(cls) and 0<=cls<10 and 0<=score<=1,'Invalid prediction class/score')
        require(-.001<=x1<=x2<=record.width+.001 and -.001<=y1<=y2<=record.height+.001,'Wrong original-image coordinates')
        answer.append(dict(image_id=record.image_id,category_id=int(cls)+1,bbox=[x1,y1,max(0.,x2-x1),max(0.,y2-y1)],score=score))
    return answer

def predict(args,weights,records,out):
    import torch,numpy as np
    import engine7
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils import LOGGER,ops
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(1,deterministic=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    mapping={str(r.image_path.resolve()):r for r in records}
    require(len(mapping)==len(records),'Duplicate image paths')
    seen=set();predictions=[];shapes=Counter();checked=0;max_error=0.
    class NMSGuard(logging.Handler):
        exceeded=False
        def emit(self,item):
            if 'NMS time limit' in item.getMessage():self.exceeded=True
    class Predictor(DetectionPredictor):
        def postprocess(self,preds,img,orig_imgs):
            nonlocal checked,max_error
            require(isinstance(preds,(tuple,list)) and len(preds)==2 and len(preds[1])==4,'Wrong four-scale head interface')
            require(img.dtype==preds[0].dtype==torch.float32 and torch.isfinite(preds[0]).all(),'Prediction must be finite FP32')
            shapes['x'.join(map(str,img.shape))]+=1
            raw=preds[0].detach().clone() if checked<8 else None
            results=super().postprocess(preds,img,orig_imgs)
            oracle=ops.non_max_suppression(raw.clone(),.001,.7,max_det=500) if raw is not None else None
            for i,result in enumerate(results):
                path=str(Path(result.path).resolve());require(path in mapping,'Unexpected predicted image')
                r=mapping[path]
                require(r.image_id not in seen and result.orig_shape==(r.height,r.width),'Repeated image or wrong dimensions')
                seen.add(r.image_id)
                if oracle is not None and checked<8:
                    expected=oracle[i].clone()
                    gain=min(img.shape[-2]/r.height,img.shape[-1]/r.width)
                    px=round((img.shape[-1]-r.width*gain)/2-.1);py=round((img.shape[-2]-r.height*gain)/2-.1)
                    expected[:,:4]-=expected.new_tensor([px,py,px,py]);expected[:,:4]/=gain
                    expected[:,[0,2]]=expected[:,[0,2]].clamp(0,r.width);expected[:,[1,3]]=expected[:,[1,3]].clamp(0,r.height)
                    require(expected.shape==result.boxes.data.shape,'NMS oracle count differs')
                    torch.testing.assert_close(expected[:,4:],result.boxes.data[:,4:],rtol=0,atol=0)
                    if len(expected):
                        err=float((expected[:,:4]-result.boxes.data[:,:4]).abs().max());max_error=max(max_error,err)
                        require(err<=.002,'Inverse letterbox oracle mismatch')
                    checked+=1
                predictions.extend(convert(r,result.boxes.data.cpu().tolist()))
            return results
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    model=YOLO(str(weights),task='detect');top=engine7.topology(model.model)
    require(all(torch.isfinite(p).all() for p in model.model.parameters()),'Nonfinite checkpoint weights')
    opts=options(args.device)
    predictor=Predictor(overrides={**model.overrides,**opts,'mode':'predict','batch':4},_callbacks=model.callbacks)
    guard=NMSGuard();LOGGER.addHandler(guard);start=time.monotonic()
    try:
        for first in range(0,len(records),4):
            chunk=records[first:first+4]
            model.predict([str(r.image_path) for r in chunk],batch=len(chunk),predictor=predictor,stream=False,**opts)
            require(not guard.exceeded,'NMS timeout; incomplete predictions rejected')
            done=first+len(chunk)
            if first==0 or done%80==0 or done==len(records):
                elapsed=time.monotonic()-start
                value=dict(status='predicting',key=args.key,images=done,total_images=len(records),elapsed_seconds=elapsed,
                           inference_ETA_seconds=(len(records)-done)*elapsed/done)
                write(out/'progress.json',value)
                print(f'{now()} {args.key} PREDICT {done}/{len(records)} elapsed={elapsed:.1f}s ETA={value["inference_ETA_seconds"]:.1f}s (scoring extra)',flush=True)
        require(seen=={r.image_id for r in records},'Not every image produced a result (including empty detections)')
        write(out/'area_predictions.json',predictions)
        manifest=dict(status='completed',seen_images=len(seen),predictions=len(predictions),sha256=sha(out/'area_predictions.json'),
                      topology=top,shapes=dict(shapes),elapsed_seconds=time.monotonic()-start,
                      coordinate_oracle_images=checked,coordinate_oracle_max_error=max_error,
                      backend=dict(matmul_allow_tf32=False,cudnn_allow_tf32=True))
        write(out/'prediction_manifest.json',manifest)
        return manifest
    finally:
        LOGGER.removeHandler(guard);del predictor,model;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()

def score_records(records,out):
    import d27_official as official
    write(Path(out)/'progress.json',dict(status='scoring',at=now(),images=len(records),stage='VisDrone Matlab1.0.4 ignore-region AP (CPU)'))
    filtered,ignore=official.filter_detections_and_audit(records,official.load_predictions(Path(out)/'area_predictions.json',len(records)))
    result=official.official_evaluate(records,filtered);finite_metrics(result)
    write(Path(out)/'ignore_audit.json',ignore)
    return result

def evaluate(args,out):
    import engine7
    import d27_official as official
    b=engine7.binding(args);job=JOBS[args.key]
    proof=read(args.report/'jobs'/args.key/'train/training_completion.json')
    require(proof['status']=='completed' and proof['key']==args.key and proof['seed']==job['seed'] and proof['completed_epochs']==200,'Formal200 training incomplete')
    require(proof['binding']==b,'Scoring/training bindings differ')
    run=args.project/'runs'/job['run'];csv_epochs(run/'results.csv')
    best=run/'weights/best.pt'
    require(sha(best)==proof['best_sha256'] and sha(run/'weights/last.pt')==proof['last_sha256'],'Only original val-selected best can be scored')
    records,meta=audit_dataset(args,args.split,verify_pixels=True)
    manifest=predict(args,best,records,out)
    require(sha(best)==proof['best_sha256'],'Checkpoint changed during prediction')
    print(f'{now()} {args.key} OFFICIAL {args.split} SCORING CPU',flush=True)
    metrics=score_records(records,out)
    binding=dict(key=args.key,seed=job['seed'],split=args.split,images=len(records),checkpoint_sha256=sha(best),
                 package_sha256=package_audit(),admission_id=b['admission'],protocol={**options(args.device),'batch':4},
                 dataset=meta,scorer_sha256=sha(Path(official.__file__)),smoke_only=False)
    result=dict(status='completed',key=args.key,seed=job['seed'],official=metrics,binding=binding,prediction_sha256=manifest['sha256'],
                no_test_selection=True,scorer='unchanged_D27_VisDrone_Matlab104_ignore_port',finished_at=now())
    write(out/'official_metrics.json',result);manifest['binding']=binding;write(out/'prediction_manifest.json',manifest)
    write(out/'progress.json',dict(status='completed',at=now(),images=len(records),official=metrics))
    print(f'{now()} {args.key} OFFICIAL {args.split} COMPLETED AP={metrics["AP50_95_percent"]:.6f}%',flush=True)
    return result
