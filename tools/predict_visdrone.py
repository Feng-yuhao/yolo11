"""FP32 / max-side640 / batch4 / conf.001 / NMS.7 / top500 paper export."""
import argparse
import json
import logging
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from st_mdrf.runtime import load_model,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",required=True);p.add_argument("--seed",required=True,type=int,choices=[1,2,3])
    p.add_argument("--weights",type=Path);p.add_argument("--dataset-root",type=Path,required=True,help="Official split folder containing images/ and annotations/")
    p.add_argument("--split",choices=["val","test-dev"],required=True)
    p.add_argument("--device",default="0");p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error("Output already exists; use a new path")
    wrapper,spec=load_model(a.model,a.seed,a.weights,device="cpu")
    import torch
    from ultralytics.utils import LOGGER
    class Guard(logging.Handler):
        exceeded=False
        def emit(self,record):
            if "NMS time limit" in record.getMessage():self.exceeded=True
    guard=Guard();LOGGER.addHandler(guard)
    import d27_official as official
    official.EXPECTED_IMAGES=548 if a.split=="val" else 1610
    records,metadata=official.load_dataset(a.dataset_root)
    kwargs=dict(imgsz=640,batch=4,device=a.device,half=False,conf=.001,iou=.7,max_det=500,
                augment=False,agnostic_nms=False,save=False,verbose=False)
    if spec["family"]=="yolo26":
        from scoring26 import AuditedPredictor
        kwargs.update(predictor=AuditedPredictor,nms=True,rect=True,channels_last=False)
    else:kwargs["rect"]=False
    result=[]
    for first in range(0,len(records),4):
        part=records[first:first+4]
        predictions=wrapper.predict([str(r.image_path) for r in part],**kwargs)
        if guard.exceeded:raise RuntimeError("Native NMS timed out; partial predictions cannot be reported")
        for record,prediction in zip(part,predictions):
            if prediction.orig_shape!=(record.height,record.width):raise RuntimeError("Original dimensions changed")
            if len(prediction.boxes)>500 or not bool(torch.isfinite(prediction.boxes.data).all()):raise RuntimeError("Invalid prediction export")
            for x1,y1,x2,y2,score,category in prediction.boxes.data.detach().cpu().tolist():
                result.append({"image_id":record.image_id,"category_id":int(category)+1,"bbox":[x1,y1,max(0.,x2-x1),max(0.,y2-y1)],"score":score})
        print(f"Predicted {first+len(part)}/{len(records)}",flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,allow_nan=False),encoding="utf-8")
    LOGGER.removeHandler(guard)
    a.output.with_suffix(".manifest.json").write_text(json.dumps({"model":a.model,"seed":a.seed,"dataset":metadata,"images":len(records),"detections":len(result),"prediction_sha256":sha(a.output),"FP32":True,"max_side":640,"batch":4,"confidence":.001,"nms_iou":.7,"max_det":500,"YOLO26_one_to_many_NMS":spec["family"]=="yolo26"},indent=2),encoding="utf-8")


if __name__=="__main__":main()
