"""One FP32 four-image prediction pass supplies official AP and raw diagnostics."""
from pathlib import Path
import json
import numpy as np
import torch


def predict_records(args,best,out,records):
    from run_e40 import write
    import spatial_task_head  # required before unpickling the custom Detect subclass
    import d27_official as official
    import d38_common as common
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils import ops
    by_path={str(r.image_path.resolve()):r for r in records}
    seen=set(); predictions=[]; rows=[]
    class Predictor(DetectionPredictor):
        def postprocess(self,preds,img,orig_imgs):
            if not isinstance(preds,(tuple,list)) or len(preds)!=2 or len(preds[1])!=4:
                raise RuntimeError('Wrong four-level detector outputs')
            raw=preds[0].detach().clone()
            guard=raw.clone()
            n=preds[1][0].shape[-2]*preds[1][0].shape[-1]
            n3=preds[1][1].shape[-2]*preds[1][1].shape[-1]
            result=super().postprocess(preds,img,orig_imgs)
            torch.testing.assert_close(raw,guard,rtol=0,atol=0)
            for i,value in enumerate(result):
                record=by_path[str(Path(value.path).resolve())]
                if record.image_id in seen: raise RuntimeError('Repeated image')
                seen.add(record.image_id)
                for x1,y1,x2,y2,score,cls in value.boxes.data.cpu().tolist():
                    predictions.append(dict(image_id=record.image_id,category_id=int(cls)+1,
                                            bbox=[x1,y1,max(0,x2-x1),max(0,y2-y1)],score=score))
                boxes=ops.scale_boxes(img.shape[2:],ops.xywh2xyxy(raw[i,:4].T.clone()),value.orig_shape).cpu().numpy()
                gt=record.eval_gt
                gt=gt[(gt[:,4]==0)&(gt[:,5]>=1)&(gt[:,5]<=10)]
                target=gt[:,:4].copy(); target[:,2:]+=target[:,:2]
                any_iou=common.iou_max(target,boxes)
                p2_iou=common.iou_max(target,boxes[:n])
                p3_iou=common.iou_max(target,boxes[n:n+n3])
                labels=raw[i,4:].argmax(0).cpu().numpy()+1
                correct=common.iou_max(target,boxes,labels,gt[:,5])
                final=value.boxes.data.detach().cpu().numpy()
                final_any=common.iou_max(target,final[:,:4])
                final_correct=common.iou_max(target,final[:,:4],final[:,5]+1,gt[:,5])
                sizes=np.sqrt(gt[:,2]*gt[:,3])*640/max(record.width,record.height)
                for j in range(len(gt)):
                    rows.append(dict(image_id=record.image_id,gt_index=j,size=float(sizes[j]),
                                     cls=int(gt[j,5]),raw_any=float(any_iou[j]),raw_p2=float(p2_iou[j]),
                                     raw_p3=float(p3_iou[j]),raw_correct=float(correct[j]),
                                     final_any=float(final_any[j]),final_correct=float(final_correct[j])))
            return result
    model=YOLO(str(best),task='detect')
    options=common.prediction_options(args.device)
    predictor=Predictor(overrides={**model.overrides,**options,'mode':'predict'},_callbacks=model.callbacks)
    for start in range(0,len(records),4):
        chunk=records[start:start+4]
        model.predict([str(r.image_path) for r in chunk],batch=len(chunk),predictor=predictor,stream=False,**options)
        if start==0 or (start+len(chunk))%80==0 or start+len(chunk)==len(records):
            print(f'E40 EVAL {start+len(chunk)}/{len(records)}',flush=True)
    if len(seen)!=len(records): raise RuntimeError('Incomplete prediction pass')
    write(out/'area_predictions.json',predictions)
    write(out/'gt_rows.json',rows)
    return predictions,rows


def evaluate(args,best,out):
    from run_e40 import BASE_REPORT,write
    import d27_official as official
    import d38_common as common
    official.run_unit_tests()
    records,dataset=official.load_dataset(args.dataset)
    predictions,rows=predict_records(args,best,out,records)
    filtered,ignore=official.filter_detections_and_audit(records,official.load_predictions(out/'area_predictions.json',len(records)))
    candidate=official.official_evaluate(records,filtered)
    base_path=args.project/'comparison_reports'/BASE_REPORT
    base_meta=json.loads((base_path/'metrics.json').read_text())
    if base_meta['status']!='completed' or base_meta['seed']!=2 or base_meta['training']['completed_epochs']!=200:
        raise RuntimeError('E1 seed2 reference report is incomplete or mismatched')
    base_best=args.project/'runs/e1_yolo11s_p2add_img640_seed2/weights/best.pt'
    if not base_best.is_file() or official.sha256(base_best)!=base_meta['evaluation_weights']['sha256']:
        raise RuntimeError('E1 seed2 reference best.pt missing or hash changed')
    images=json.loads((base_path/'area_ground_truth.json').read_text())['images']
    if len(images)!=len(records): raise RuntimeError('Baseline mapping count differs')
    for record,image in zip(records,images):
        if (record.image_id,record.stem,record.width,record.height)!=(image['id'],Path(image['file_name']).stem,image['width'],image['height']):
            raise RuntimeError('Baseline image mapping differs')
    base_filtered,_=official.filter_detections_and_audit(records,official.load_predictions(base_path/'area_predictions.json',len(records)))
    baseline=official.official_evaluate(records,base_filtered)
    if not all(0.0<=baseline[key]<=100.0 for key in ('AP50_95_percent','AP50_percent','AP75_percent')):
        raise RuntimeError('Invalid E1 seed2 official score')
    keys=common.KEYS
    delta={k:candidate[k]-baseline[k] for k in keys}
    slices={}
    for lo,hi,name in ((0,float('inf'),'all'),(0,8,'lt8'),(8,16,'8_16'),(16,24,'16_24'),(24,32,'24_32'),(32,float('inf'),'ge32')):
        group=[r for r in rows if lo<=r['size']<hi]
        item=dict(gt_count=len(group))
        for key in ('raw_any','raw_p2','raw_p3','raw_correct','final_any','final_correct'):
            item[key+'75_percent']=100*sum(r[key]>=.75 for r in group)/len(group) if group else None
        slices[name]=item
    comparison=dict(status='completed',baseline=baseline,candidate=candidate,delta_pp=delta,
                    seed=2,imgsz=640,checkpoint=str(best),checkpoint_sha256=official.sha256(best),
                    baseline_checkpoint_sha256=base_meta['evaluation_weights']['sha256'],
                    dataset=dataset,ignore=ignore,geometry=slices,
                    screening_passed=delta['AP50_95_percent']>=.5 and delta['AP75_percent']>=.3 and delta['AP50_percent']>=0,
                    boundary='Seed2 paired with E1 seed2; size raw coverage has no seed2 reference and is not AP.')
    write(out/'comparison.json',comparison)
    write(out/'official_metrics.json',dict(status='completed',baseline=baseline,candidate=candidate,delta_pp=delta,
                                          dataset=dataset,checkpoint_sha256=comparison['checkpoint_sha256']))
    table='|Metric|E1|E40|Delta pp|\n|---|---:|---:|---:|\n'
    for key in keys:
        table+=f'|{key}|{baseline[key]:.4f}|{candidate[key]:.4f}|{delta[key]:+.4f}|\n'
    (out/'REPORT.md').write_text('# E40 formal seed2/640/200 epochs; paired E1 seed2\n\n'+table+
        '\nTwo-seed evidence must be judged with seed1; not a significance or publication guarantee. '+
        'Official VisDrone ignore rules and reference reproduction are unchanged. '+
        'Raw/final coverage is independent candidate existence per GT, not one-to-one recall or size AP. '+
        'P2/P3 coverage is raw-any; final coverage is before official ignore filtering. '+
        'No D38 raw-coverage delta is shown because D38 used E1 seed1. '+
        'See comparison.json/gt_rows.json for geometry; official_metrics.json for scores.\n',encoding='utf-8')
    print('E40 OFFICIAL RESULT '+json.dumps(delta)+' REPORT='+str(out),flush=True)
    return comparison
