"""One FP32 four-image prediction pass supplies official AP and raw diagnostics."""
from pathlib import Path
import json
import numpy as np
import torch


def predict_records(args,best,out,records):
    from run_experiment import write
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
    head=model.model.model[-1]
    if (type(head) is not spatial_task_head.SpatialTaskDetect or len(head.task_blocks)!=4 or
            tuple(b.reduce[0].in_channels for b in head.task_blocks)!=spatial_task_head.CHANNELS or
            getattr(head,'experiment_variant',None)!=args.variant):
        raise RuntimeError('Evaluation checkpoint does not match E60 grouped residual head')
    if not (all(isinstance(b, spatial_task_head.GroupResidualTaskBlock) and len(b.refinements)==3
                for b in head.task_blocks[:2]) and
            all(not hasattr(b,'refinements') for b in head.task_blocks[2:])):
        raise RuntimeError('Wrong actual P2/P3 refinement topology in evaluation checkpoint')
    options=common.prediction_options(args.device)
    predictor=Predictor(overrides={**model.overrides,**options,'mode':'predict'},_callbacks=model.callbacks)
    for start in range(0,len(records),4):
        chunk=records[start:start+4]
        model.predict([str(r.image_path) for r in chunk],batch=len(chunk),predictor=predictor,stream=False,**options)
        if start==0 or (start+len(chunk))%80==0 or start+len(chunk)==len(records):
            print(f'{args.variant.upper()} EVAL {start+len(chunk)}/{len(records)}',flush=True)
    if len(seen)!=len(records): raise RuntimeError('Incomplete prediction pass')
    write(out/'area_predictions.json',predictions)
    write(out/'gt_rows.json',rows)
    return predictions,rows


def evaluate(args,best,out):
    from run_experiment import BASE_REPORT,write
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
    from parent_reference import E1_EXPECTED
    expected=E1_EXPECTED
    for key,value in expected.items():
        if abs(baseline[key]-value)>1e-7: raise RuntimeError('E1 official reference did not reproduce '+key)
    from parent_reference import reference_e40,reference_e58,rescore_reference,E40_REPORT,E58_REPORT
    parent=reference_e40(args,dataset)
    parent_report=json.loads((args.project/'comparison_reports'/E40_REPORT/'comparison.json').read_text())
    e58,e58_report=reference_e58(args,dataset)
    parent=rescore_reference(args,records,E40_REPORT,parent)
    e58=rescore_reference(args,records,E58_REPORT,e58)
    keys=common.KEYS
    parent_delta={k:candidate[k]-parent[k] for k in keys}
    e58_delta={k:candidate[k]-e58[k] for k in keys}
    delta={k:candidate[k]-baseline[k] for k in keys}
    slices={}
    for lo,hi,name in ((0,float('inf'),'all'),(0,8,'lt8'),(8,16,'8_16'),(16,24,'16_24'),(24,32,'24_32'),(32,float('inf'),'ge32')):
        group=[r for r in rows if lo<=r['size']<hi]
        item=dict(gt_count=len(group))
        for key in ('raw_any','raw_p2','raw_p3','raw_correct','final_any','final_correct'):
            item[key+'75_percent']=100*sum(r[key]>=.75 for r in group)/len(group) if group else None
        parent_group=parent_report['geometry'][name]
        if parent_group['gt_count']!=len(group): raise RuntimeError('E40 geometry group mismatch '+name)
        item['delta_vs_E40_pp']={key:item[key]-parent_group[key] for key in item
                                 if key.endswith('75_percent') and key in parent_group}
        e58_group=e58_report['geometry'][name]
        if e58_group['gt_count']!=len(group): raise RuntimeError('E58 geometry group mismatch '+name)
        item['delta_vs_E58_pp']={key:item[key]-e58_group[key] for key in item
                                 if key.endswith('75_percent') and key in e58_group}
        slices[name]=item
    comparison=dict(status='completed',baseline=baseline,candidate=candidate,delta_pp=delta,
                    parent_E40=parent,delta_vs_E40_pp=parent_delta,
                    parent_E58=e58,delta_vs_E58_pp=e58_delta,
                    variant=args.variant,seed=1,imgsz=640,checkpoint=str(best),checkpoint_sha256=official.sha256(best),
                    dataset=dataset,ignore=ignore,geometry=slices,
                    screening=dict(AP_gain_at_least_0p20_pp=e58_delta[keys[0]]>=.20,
                                   AP75_not_down=e58_delta['AP75_percent']>=0.,
                                   not_statistical_significance=True),
                    boundary='Seed1 exploratory comparison to E58; raw coverage is candidate existence, not AP or one-to-one recall.')
    write(out/'comparison.json',comparison)
    write(out/'official_metrics.json',dict(status='completed',baseline=baseline,candidate=candidate,delta_pp=delta,
                                          seed=1,parent_E40=parent,delta_vs_E40_pp=parent_delta,
                                          parent_E58=e58,delta_vs_E58_pp=e58_delta,
                                          dataset=dataset,checkpoint_sha256=comparison['checkpoint_sha256']))
    table='|Metric|E1 s1|E40 s1|E58 s1|E60 s1|Delta vs E58 pp|Delta vs E1 pp|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for key in keys:
        table+=f'|{key}|{baseline[key]:.4f}|{parent[key]:.4f}|{e58[key]:.4f}|{candidate[key]:.4f}|{e58_delta[key]:+.4f}|{delta[key]:+.4f}|\n'
    (out/'REPORT.md').write_text('# E60a grouped residual encoding / seed1 / 640\n\n'+table+
        '\nPrimary comparator: E58 seed1. New P2/P3 group encoding is the only design change. '+
        'All three cached official references were rescored with the same annotations/scorer. '+
        'Single-seed resource screening only, not a significance or publication guarantee. '+
        'Official VisDrone ignore rules and reference reproduction are unchanged. '+
        'Raw/final coverage is independent candidate existence per GT, not one-to-one recall or size AP. '+
        'P2/P3 coverage is raw-any; final coverage is before official ignore filtering. '+
        'See comparison.json/gt_rows.json for geometry; official_metrics.json for scores.\n',encoding='utf-8')
    print('E60 OFFICIAL RESULT vs_E58='+json.dumps(e58_delta)+' vs_E40='+json.dumps(parent_delta)+' vs_E1='+json.dumps(delta)+' REPORT='+str(out),flush=True)
    return comparison
