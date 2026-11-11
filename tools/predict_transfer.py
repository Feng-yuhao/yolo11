"""Frozen ten-class source model; semantic max mapping before native class-aware NMS."""
import argparse
import json
import logging
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from st_mdrf.runtime import load_model,sha

SPECS={
 "songdo":{"classes":["car","bus","truck","motorcycle"],"groups":[[3,4],[8],[5],[9]],"images":1084},
 "dior":{"classes":["vehicle"],"groups":[[3,4,5,8]],"images":11738},
 "uavdt":{"classes":["car","truck","bus"],"groups":[[3,4],[5],[8]],"images":16592},
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target",choices=list(SPECS),required=True)
    p.add_argument("--model",choices=["yolo11s-p2","st-mdrf"],required=True);p.add_argument("--seed",type=int,choices=[1,2,3],required=True)
    p.add_argument("--weights",type=Path);p.add_argument("--dataset-root",type=Path,required=True)
    p.add_argument("--gt",type=Path,required=True,help="Original/prepared target COCO-format GT, preserving original image IDs")
    p.add_argument("--device",default="0");p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error("Output already exists")
    spec=SPECS[a.target];gt=json.loads(a.gt.read_text(encoding="utf-8"))
    images=sorted(gt["images"],key=lambda r:r["id"])
    if len(images)!=spec["images"] or len({r["id"] for r in images})!=len(images):raise RuntimeError("Use the complete official target test split")
    lookup={};paths=[]
    for item in images:
        choices=[a.dataset_root/item["file_name"],a.dataset_root/"images/test"/Path(item["file_name"]).name,a.dataset_root/"images"/Path(item["file_name"]).name]
        path=next((p.resolve() for p in choices if p.is_file()),None)
        if path is None:raise FileNotFoundError(item["file_name"])
        if str(path) in lookup:raise RuntimeError("Duplicate target image path")
        lookup[str(path)]=item;paths.append(str(path))
    wrapper,_=load_model(a.model,a.seed,a.weights,device="cpu")
    import torch
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.engine.results import Results
    from ultralytics.utils import LOGGER,ops
    class Guard(logging.Handler):
        exceeded=False
        def emit(self,record):
            if "NMS time limit" in record.getMessage():self.exceeded=True
    class Predictor(DetectionPredictor):
        def postprocess(self,preds,img,orig_imgs):
            if img.dtype!=torch.float32:raise RuntimeError("FP32 target inference required")
            raw=preds[0] if isinstance(preds,(list,tuple)) else preds
            scores=torch.stack([raw[:,[i+4 for i in group],:].amax(dim=1) for group in spec["groups"]],dim=1)
            mapped=torch.cat((raw[:,:4],scores),dim=1)
            boxes=ops.non_max_suppression(mapped.clone(),.001,.7,nc=len(spec["classes"]),multi_label=False,max_det=500,agnostic=False,max_nms=30000)
            if not isinstance(orig_imgs,list):orig_imgs=ops.convert_torch2numpy_batch(orig_imgs)
            result=[]
            for pred,original,path in zip(boxes,orig_imgs,self.batch[0]):
                record=lookup[str(Path(path).resolve())]
                if original.shape[:2]!=(record["height"],record["width"]):raise RuntimeError("Target orientation/dimensions changed")
                pred[:,:4]=ops.scale_boxes(img.shape[2:],pred[:,:4],original.shape)
                if not bool(torch.isfinite(pred).all()):raise RuntimeError("Nonfinite target prediction")
                result.append(Results(original,path=path,names=dict(enumerate(spec["classes"])),boxes=pred))
            return result
    options=dict(imgsz=640,batch=4,device=a.device,half=False,conf=.001,iou=.7,max_det=500,rect=False,augment=False,agnostic_nms=False,verbose=False,save=False)
    predictor=Predictor(overrides={**wrapper.overrides,**options,"mode":"predict"},_callbacks=wrapper.callbacks)
    predictor.setup_model(model=wrapper.model,verbose=False);wrapper.predictor=predictor
    def fingerprint():
        import hashlib
        digest=hashlib.sha256()
        for name,value in sorted(predictor.model.model.state_dict().items()):
            digest.update(name.encode());digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    before=fingerprint();guard=Guard();LOGGER.addHandler(guard);detections=0
    a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        with a.output.open("x",encoding="utf-8") as output:
            output.write("[");first_record=True
            for start in range(0,len(paths),4):
                chunk=paths[start:start+4]
                results=wrapper.predict(chunk,**dict(options,batch=len(chunk)))
                if guard.exceeded:raise RuntimeError("Native NMS time limit exceeded; partial predictions are invalid")
                for item,result in zip(images[start:start+4],results):
                    for x1,y1,x2,y2,score,category in result.boxes.data.detach().cpu().tolist():
                        row={"image_id":item["id"],"category_id":int(category)+1,"bbox":[x1,y1,max(0.,x2-x1),max(0.,y2-y1)],"score":score}
                        if not first_record:output.write(",")
                        output.write(json.dumps(row,allow_nan=False));first_record=False;detections+=1
                print(f"Predicted {start+len(chunk)}/{len(paths)}",flush=True)
            output.write("]")
        after=fingerprint()
        if before!=after:raise RuntimeError("Model parameters/BN statistics changed during frozen transfer")
        a.output.with_suffix(".manifest.json").write_text(json.dumps({"target":a.target,"model":a.model,"seed":a.seed,"images":len(paths),"detections":detections,"mapping":spec,"map_before_NMS":True,"training":False,"parameters_and_BN_unchanged":True,"inference_state_sha256_before":before,"inference_state_sha256_after":after,"prediction_sha256":sha(a.output)},indent=2),encoding="utf-8")
    finally:LOGGER.removeHandler(guard)


if __name__=="__main__":main()
