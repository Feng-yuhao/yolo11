"""Unmodified frozen detector, mapped native NMS, streamed original-coordinate predictions."""
import logging
from pathlib import Path
import time
from common import PREDICT,PROTOCOL,SOURCE_CLASSES,SPECS,now,require,sha,split_key,write
from mapping import inverse_boxes,map_scores,mapped_nms
from modeling import audit,fingerprint
from storage import PredictionWriter
from target_data import image_id_sha

def convert(record,rows):
    import math
    result=[]
    for x1,y1,x2,y2,score,category in rows:
        require(all(math.isfinite(v) for v in (x1,y1,x2,y2,score,category)),'Nonfinite prediction')
        require(category==0 and 0<=score<=1,'Wrong harmonized category/score')
        require(-.01<=x1<=x2<=record['width']+.01 and -.01<=y1<=y2<=record['height']+.01,'Wrong original-image coordinates')
        result.append(dict(image_id=record['id'],category_id=1,bbox=[x1,y1,max(0.,x2-x1),max(0.,y2-y1)],score=score))
    return result

def predict(args,weights,key,items,out):
    import torch,numpy as np
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils import LOGGER,ops
    target,source_key=split_key(key);require(not out.exists(),'Refuse overwrite of prior predictions')
    out.mkdir(parents=True)
    lookup={str(r['path'].resolve()):r for r in items};require(len(lookup)==len(items),'Duplicate target paths')
    audit_ids={r['id'] for r in items[:8]}|{r['id'] for r in items if r['height']>r['width']}
    seen=set();audited=set();shapes={};max_error=0.
    writer=PredictionWriter(out/'predictions.json.gz',args.report)
    class Guard(logging.Handler):
        exceeded=False
        def emit(self,r):
            if 'NMS time limit' in r.getMessage():self.exceeded=True
    class Predictor(DetectionPredictor):
        def postprocess(self,preds,img,orig_imgs):
            nonlocal max_error
            require(isinstance(preds,(tuple,list)) and img.dtype==torch.float32,'Unexpected original source forward')
            raw=preds[0];shape='x'.join(map(str,img.shape));shapes[shape]=shapes.get(shape,0)+1
            require(not any(m.training for m in self.model.model.modules()),'Training/BN adaptation forbidden')
            needed=any(lookup[str(Path(p).resolve())]['id'] in audit_ids for p in self.batch[0])
            saved=raw.detach().clone() if needed else None
            boxes=mapped_nms(raw,target);oracle=None
            if saved is not None:
                require(torch.equal(saved,raw),'Mapping/NMS mutated source output')
                array=saved.cpu().numpy()
                expected=np.concatenate((array[:,:4],np.max(array[:,[4+i for i in SPECS[target]['indices']]],axis=1,keepdims=True)),axis=1)
                np.testing.assert_array_equal(map_scores(saved,target).cpu().numpy(),expected)
                oracle=ops.non_max_suppression(torch.from_numpy(expected).to(raw.device).clone(),.001,.7,nc=1,multi_label=False,max_det=500,agnostic=False,max_nms=30000)
            if not isinstance(orig_imgs,list):orig_imgs=ops.convert_torch2numpy_batch(orig_imgs)
            results=[]
            for index,(pred,original,path) in enumerate(zip(boxes,orig_imgs,self.batch[0])):
                record=lookup[str(Path(path).resolve())]
                require(record['id'] not in seen and original.shape[:2]==(record['height'],record['width']),'Duplicate image or wrong orientation')
                pred[:,:4]=ops.scale_boxes(img.shape[2:],pred[:,:4],original.shape);seen.add(record['id'])
                if oracle is not None and record['id'] in audit_ids:
                    expected=inverse_boxes(oracle[index],img.shape[-2:],original.shape[:2])
                    require(expected.shape==pred.shape,'NMS count mismatch')
                    torch.testing.assert_close(expected[:,4:],pred[:,4:],rtol=0,atol=0)
                    error=float((expected[:,:4]-pred[:,:4]).abs().max()) if len(expected) else 0.
                    require(error<=.01,'Inverse-letterbox coordinate mismatch');max_error=max(max_error,error);audited.add(record['id'])
                writer.add(convert(record,pred.detach().cpu().tolist()))
                results.append(Results(original,path=path,names={0:SPECS[target]['category']},boxes=pred))
            return results
    checkpoint_hash=sha(weights);model=YOLO(str(weights),task='detect');model_info=audit(model.model,source_key)
    options=dict(PREDICT,device='cuda:0',project=str(out),name='unused',exist_ok=True,batch=4)
    predictor=Predictor(overrides={**model.overrides,**options,'mode':'predict'},_callbacks=model.callbacks)
    predictor.setup_model(model=model.model,verbose=False);model.predictor=predictor
    require(predictor.model.names==dict(enumerate(SOURCE_CLASSES)),'Original ten-class head changed')
    require(not any(m.training for m in predictor.model.model.modules()),'Model must be eval')
    require(not any(p.requires_grad for p in predictor.model.model.parameters()),'Backend gradients must be disabled')
    before=fingerprint(predictor.model.model);guard=Guard();LOGGER.addHandler(guard);started=time.monotonic()
    try:
        for first in range(0,len(items),4):
            block=items[first:first+4]
            model.predict([str(r['path']) for r in block],stream=False,**dict(options,batch=len(block)))
            require(not guard.exceeded,'NMS time limit: partial predictions rejected')
            done=first+len(block);elapsed=time.monotonic()-started
            progress=dict(key=key,done_images=done,total_images=len(items),elapsed_seconds=elapsed,ETA_seconds=elapsed*(len(items)-done)/done,at=now())
            write(out/'progress.json',progress)
            if done%200==0 or done==len(items):print(f'{now()} {key} PREDICT {done}/{len(items)} elapsed={elapsed:.1f}s ETA={progress["ETA_seconds"]:.1f}s',flush=True)
        require(seen=={r['id'] for r in items} and audited==audit_ids,'Incomplete image/coordinate oracle coverage')
        after=fingerprint(predictor.model.model)
        require(after==before and sha(weights)==checkpoint_hash,'Source parameters/BN/checkpoint changed')
        writer.close()
        proof=dict(status='completed',key=key,seen_images=len(seen),seen_image_ids=sorted(seen),image_ids_sha256=image_id_sha(seen),
                   prediction_count=writer.count,prediction_file='predictions.json.gz',prediction_sha256=sha(out/'predictions.json.gz'),
                   checkpoint_sha256=checkpoint_hash,model=model_info,coordinate_oracle_ids=sorted(audited),coordinate_oracle_max_error_px=max_error,
                   raw_mapping_numpy_oracle=True,parameters_and_BN_unchanged=True,inference_state_sha256_before=before,inference_state_sha256_after=after,
                   shapes=shapes,seconds=time.monotonic()-started,protocol=PROTOCOL)
        write(out/'prediction_manifest.json',proof);return proof
    finally:
        LOGGER.removeHandler(guard);writer.abort()
