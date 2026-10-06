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
            print(f'E46 EVAL {start+len(chunk)}/{len(records)}',flush=True)
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
    images=json.loads((base_path/'area_ground_truth.json').read_text())['images']
    if len(images)!=len(records): raise RuntimeError('Baseline mapping count differs')
    for record,image in zip(records,images):
        if (record.image_id,record.stem,record.width,record.height)!=(image['id'],Path(image['file_name']).stem,image['width'],image['height']):
            raise RuntimeError('Baseline image mapping differs')
    base_filtered,_=official.filter_detections_and_audit(records,official.load_predictions(base_path/'area_predictions.json',len(records)))
    baseline=official.official_evaluate(records,base_filtered)
    expected=dict(AP50_95_percent=25.63123996261998,AP50_percent=44.72545960536977,AP75_percent=25.02982351561637)
    for key,value in expected.items():
        if abs(baseline[key]-value)>1e-7: raise RuntimeError('E1 official reference did not reproduce '+key)
    from parent_reference import reference_e40
    parent=reference_e40(args,dataset)
    keys=common.KEYS
    parent_delta={k:candidate[k]-parent[k] for k in keys}
    delta={k:candidate[k]-baseline[k] for k in keys}
    slices={}
    d38_path=args.project/'comparison_reports/DIAG_D38_E1_RAW_STAGES_20260927_234535_900447/metrics.json'
    d38=json.loads(d38_path.read_text()) if d38_path.is_file() else None
    if d38 is not None and (d38['signature']['dataset']!=dataset or d38['signature']['weights']!=
                           json.loads((base_path/'metrics.json').read_text())['evaluation_weights']['sha256']):
        raise RuntimeError('D38 reference uses different dataset/weights')
    for lo,hi,name in ((0,float('inf'),'all'),(0,8,'lt8'),(8,16,'8_16'),(16,24,'16_24'),(24,32,'24_32'),(32,float('inf'),'ge32')):
        group=[r for r in rows if lo<=r['size']<hi]
        item=dict(gt_count=len(group))
        for key in ('raw_any','raw_p2','raw_p3','raw_correct','final_any','final_correct'):
            item[key+'75_percent']=100*sum(r[key]>=.75 for r in group)/len(group) if group else None
        if d38 is not None:
            old=d38['slices'][name]
            if old['gt_count']!=len(group): raise RuntimeError('D38 group mismatch '+name)
            item['raw_any75_delta_pp']=item['raw_any75_percent']-old['candidate_coverage_percent']['0.75']['raw_any']
        slices[name]=item
    comparison=dict(status='completed',baseline=baseline,candidate=candidate,delta_pp=delta,
                    parent_E40=parent,delta_vs_E40_pp=parent_delta,seed=1,imgsz=640,checkpoint=str(best),checkpoint_sha256=official.sha256(best),
                    dataset=dataset,ignore=ignore,geometry=slices,
                    screening_passed=parent_delta['AP50_95_percent']>=0.25 and parent_delta['AP75_percent']>=0 and parent_delta['AP50_percent']>=0,
                    boundary='Single seed screening; raw coverage is candidate existence, not AP or one-to-one recall.')
    write(out/'comparison.json',comparison)
    write(out/'official_metrics.json',dict(status='completed',baseline=baseline,candidate=candidate,delta_pp=delta,
                                          parent_E40=parent,delta_vs_E40_pp=parent_delta,dataset=dataset,checkpoint_sha256=comparison['checkpoint_sha256']))
    table='|Metric|E1|E46|Delta pp|\n|---|---:|---:|---:|\n'
    for key in keys:
        table+=f'|{key}|{baseline[key]:.4f}|{candidate[key]:.4f}|{delta[key]:+.4f}|\n'
    (out/'REPORT.md').write_text('# E46 formal seed1/640/200 epochs\n\n'+table+
        '\nDelta vs E40: '+json.dumps(parent_delta)+'\n\nSingle-seed screening only, not a significance or publication guarantee. '+
        'Official VisDrone ignore rules and reference reproduction are unchanged. '+
        'Raw/final coverage is independent candidate existence per GT, not one-to-one recall or size AP. '+
        'P2/P3 coverage is raw-any; final coverage is before official ignore filtering. '+
        'See comparison.json/gt_rows.json for geometry; official_metrics.json for scores.\n',encoding='utf-8')
    print('E46 OFFICIAL RESULT '+json.dumps(delta)+' REPORT='+str(out),flush=True)
    return comparison

