"""Same frozen VisDrone2019 ignore scoring; independent fixed one2many+NMS export."""
import copy
import math
from pathlib import Path
import torch
import e60_adapter
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import nms, ops
import d27_official as official
from protocol import write, now, NAMES
from common import sha256


def check_checkpoint(model,variant):
    head=model.model[-1]
    if len(model.model)!=30 or head.nl!=4 or head.reg_max!=1 or list(head.f)!=[19,22,25,28]:
        raise RuntimeError('Wrong YOLO26s-P2 checkpoint topology')
    if model.yolo26_port_variant!=variant or list(model.names.values())!=NAMES:
        raise RuntimeError('Wrong variant/class mapping in scoring checkpoint')
    if (variant=='e60') != isinstance(head,e60_adapter.E60Detect26):
        raise RuntimeError('Wrong custom head in checkpoint')


class AuditedPredictor(DetectionPredictor):
    def postprocess(self,preds,img,orig_imgs,**kwargs):
        raw=preds[0] if isinstance(preds,(list,tuple)) else preds
        if raw.ndim!=3 or raw.shape[1]!=14 or raw.shape[2]<=500:
            raise RuntimeError('Unexpected capped one2one output instead of one2many dense candidates')
        if not torch.isfinite(raw).all(): raise RuntimeError('Nonfinite raw decoded predictions')
        guard=raw.detach().clone()
        cloned=raw.clone()
        # Independent original-coordinate oracle before Predictor's inverse transform.
        ref=nms.non_max_suppression(raw.clone(),.001,.7,max_det=500,nc=10,end2end=False)
        expected=[]
        for pred,orig in zip(ref,orig_imgs):
            original=pred[:,:4].clone().double()
            h,w=img.shape[2:]; oh,ow=orig.shape[:2]
            gain=min(h/oh,w/ow)
            resized_w=round(ow*gain); resized_h=round(oh*gain)
            padx=round((w-resized_w)/2-.1); pady=round((h-resized_h)/2-.1)
            original[:,[0,2]]=(original[:,[0,2]]-padx)/(resized_w/ow)
            original[:,[1,3]]=(original[:,[1,3]]-pady)/(resized_h/oh)
            # Native 8.4.177 inverts actual rounded x/y resize dimensions separately.
            # Advanced-index clamp does not write back; explicit assignment is required.
            original[:,0]=original[:,0].clamp(0,ow); original[:,2]=original[:,2].clamp(0,ow)
            original[:,1]=original[:,1].clamp(0,oh); original[:,3]=original[:,3].clamp(0,oh)
            expected.append((original,pred[:,4:].clone()))
        results=super().postprocess(cloned,img,orig_imgs,**kwargs)
        torch.testing.assert_close(raw,guard,rtol=0,atol=0)
        for value,(boxes,attributes) in zip(results,expected):
            actual=value.boxes.data
            if actual.shape[0]!=len(boxes): raise RuntimeError('Independent NMS selection mismatch')
            torch.testing.assert_close(actual[:,:4].double(),boxes,rtol=0,atol=.002)
            torch.testing.assert_close(actual[:,4:],attributes,rtol=0,atol=0)
        return results


def predict(args,variant,weights,out,records):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    model=YOLO(str(weights),task='detect')
    check_checkpoint(model.model,variant)
    model.model.end2end=False; model.model.model[-1].max_det=500
    lookup={str(r.image_path.resolve()):r for r in records}; seen=set(); predictions=[]
    options=dict(imgsz=640 if not args.local_test else args.local_size,batch=4,device='cuda:0',
                 quantize=None,nms=True,conf=.001,iou=.7,max_det=500,rect=True,
                 augment=False,agnostic_nms=False,verbose=False,save=False,channels_last=False)
    if args.local_test and not torch.cuda.is_available(): options['device']='cpu'
    started=__import__('time').monotonic()
    for start in range(0,len(records),4):
        chunk=records[start:start+4]
        results=model.predict([str(r.image_path) for r in chunk],predictor=AuditedPredictor,**options)
        for result in results:
            record=lookup[str(Path(result.path).resolve())]
            if record.image_id in seen or result.orig_shape!=(record.height,record.width):
                raise RuntimeError('Repeated image or original dimensions mismatch')
            seen.add(record.image_id)
            if len(result.boxes)>500: raise RuntimeError('NMS max_det limit drift')
            for x1,y1,x2,y2,score,cls in result.boxes.data.cpu().tolist():
                if not all(math.isfinite(v) for v in (x1,y1,x2,y2,score,cls)): raise RuntimeError('Invalid export')
                predictions.append(dict(image_id=record.image_id,category_id=int(cls)+1,
                                        bbox=[x1,y1,max(0,x2-x1),max(0,y2-y1)],score=score))
        if start==0 or (start+len(chunk))%80==0 or start+len(chunk)==len(records):
            print(f'{now()} {variant.upper()} OFFICIAL PREDICT {len(seen)}/{len(records)}',flush=True)
    if len(seen)!=len(records): raise RuntimeError('Incomplete official prediction')
    write(out/'area_predictions.json',predictions)
    write(out/'image_mapping.json',[dict(id=r.image_id,file_name=str(r.image_path),width=r.width,height=r.height) for r in records])
    return dict(images=len(seen),detections=len(predictions),seconds=__import__('time').monotonic()-started,
                one2many_NMS=True,FP32=True,batch=4,conf=.001,iou=.7,max_det=500,
                inverse_coordinates_independently_checked=True,raw_predictions_not_mutated=True,
                checkpoint_sha256=sha256(weights))


def evaluate(args,variant,weights,out):
    official.run_unit_tests()
    records,dataset=official.load_dataset(args.val_dataset)
    prediction=predict(args,variant,weights,out,records)
    filtered,ignore=official.filter_detections_and_audit(records,official.load_predictions(out/'area_predictions.json',len(records)))
    metrics=official.official_evaluate(records,filtered)
    result=dict(status='completed',variant=variant,seed=1,dataset=dataset,official_metrics=metrics,
                scorer_revision=official.REVISION,scorer_sha256=sha256(Path(official.__file__)),
                prediction=prediction,ignore=ignore,checkpoint=str(weights),checkpoint_sha256=sha256(weights),
                selection='fixed val-selected best; no test-dev tuning',finished_at=now())
    write(out/'official_metrics.json',result)
    print(f'{now()} {variant.upper()} OFFICIAL VAL AP={metrics["AP50_95_percent"]:.4f}%',flush=True)
    return result
