"""Seed3 results plus verified available paired seeds; retain every sign."""
import math
from pathlib import Path
import statistics
import common as c
from protocol import read, write, require, REF, VARIANTS, SEED

KEYS=('AP50_95_percent','AP50_percent','AP75_percent','AR500_percent')


def completed_pair(report,split,local=False,seed=SEED):
    result={v:read(Path(report)/(v+'_'+split.replace('-','_'))/'official_metrics.json') for v in VARIANTS}
    for variant,value in result.items():
        require(value['status']=='completed' and value['seed']==seed and value['variant']==variant
                and value['split']==split and value['local_fixture']==local,'Wrong paired seed/split score')
        require(all(math.isfinite(value['official_metrics'][k]) for k in KEYS),'Nonfinite official metric')
        if not local:
            require(value['dataset']['images']==(548 if split=='val' else 1610),'Incomplete evaluation')
    require(result['baseline']['dataset']==result['e60']['dataset'] and
            result['baseline']['protocol']==result['e60']['protocol'] and
            result['baseline']['scorer_sha256']==result['e60']['scorer_sha256'],'Pair protocol/data/scorer mismatch')
    return result


def validate_reference(value,current,variant,seed):
    require(value['status']=='completed' and value['seed']==seed and value['variant']==variant
            and not value.get('local_fixture',False),'Wrong reference seed/variant/private score')
    require(value['dataset']['annotation_dimension_sha256']==current['dataset']['annotation_dimension_sha256']
            and value['dataset']['images']==current['dataset']['images']
            and value['scorer_sha256']==current['scorer_sha256'],'Reference data/scorer differs')
    pred=value['prediction']
    require(pred['FP32'] and pred['one2many_NMS'] and pred['batch']==4 and pred['conf']==.001
            and pred['iou']==.7 and pred['max_det']==500,'Reference prediction settings differ')
    require(all(math.isfinite(value['official_metrics'][k]) for k in KEYS),'Nonfinite reference metric')


def inside(path,parent):
    path,parent=Path(path).resolve(),Path(parent).resolve()
    require(path!=parent and path.is_relative_to(parent),'Reference path escapes expected project folder')
    return path


def available_seed2(project,current,destination):
    """Read one exact completed controller index; no search/ranking by performance."""
    index=Path(project)/'logs/latest_yolo26_seed2_serial.json'
    if not index.is_file():
        return None,dict(status='not_available',reason='No seed2 index; no seed2 value invented')
    try:
        before=c.sha256(index);info=read(index)
        session=inside(info['session'],Path(project)/'logs')
        report=inside(info['report_root'],Path(project)/'comparison_reports')
        state=read(session/'state.json')
        require(state['seed']==2 and state['status']=='completed' and not state.get('local_fixture')
                and not state.get('child_active') and state['execution']=='serial','Seed2 pipeline incomplete/private')
        require(Path(state['session']).resolve()==session and Path(state['report_root']).resolve()==report,
                'Seed2 index/state paths disagree')
        sig=read(session/'signature.json')['payload']
        for key,expected in (('seed',2),('layout','serial'),('train_batch',16),('validation_batch',32),
                             ('prediction_batch',4),('imgsz',640),('epochs',200),('pretrained',c.WEIGHT_SHA)):
            require(sig[key]==expected,'Seed2 training protocol differs: '+key)
        require(sig['source']==c.sha256(c.HERE/'training_source_sha256.json'),'Seed2 source manifest differs')
        training=inside(state.get('training_report') or report,Path(project)/'comparison_reports')
        pairs={split:completed_pair(report,split,False,seed=2) for split in ('val','test-dev')}
        for variant in VARIANTS:
            completion=read(training/(variant+'_train')/'training_completion.json')
            best=Path(project)/'runs'/('yolo26s_p2_'+variant+'_img640_seed2_200e')/'weights/best.pt'
            require(completion['status']=='training_completed' and completion['epochs']==200
                    and completion['seed']==2 and completion['variant']==variant
                    and Path(completion['best']).resolve()==best.resolve(),'Wrong seed2 completed training')
            digest=c.sha256(best)
            require(digest==completion['best_sha256'],'Seed2 best changed after training')
            for split in pairs:
                value=pairs[split][variant]
                validate_reference(value,current[split][variant],variant,2)
                require(value['checkpoint_sha256']==digest,'Seed2 score/checkpoint mismatch')
                predictions=report/(variant+'_'+split.replace('-','_'))/'area_predictions.json'
                require(c.sha256(predictions)==value['prediction_sha256'],'Seed2 prediction file changed')
        require(c.sha256(index)==before,'Seed2 index changed during read')
        for split,pair in pairs.items():
            for variant,value in pair.items():
                write(destination/(variant+'_seed2_'+split.replace('-','_')+'.json'),value)
        write(destination/'seed2_controller_state.json',state)
        write(destination/'seed2_signature.json',dict(payload=sig))
        return pairs,dict(status='included',session=str(session),report_root=str(report),
                          index_sha256=before,selection='Exact fixed seed2; all signs retained, no performance selection')
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as error:
        return None,dict(status='not_included_requires_review',reason=str(error),index=str(index),
                         boundary='No replacement seed or fabricated mean; seed3 remains independently valid')


def summarize(report,local=False):
    report=Path(report)
    current={split:completed_pair(report,split,local) for split in ('val','test-dev')}
    originals={}
    if not local:
        for split in current:
            originals[split]={v:read(REF/(v+'_seed1_'+split.replace('-','_')+'.json')) for v in VARIANTS}
            for variant in VARIANTS:
                validate_reference(originals[split][variant],current[split][variant],variant,1)
    extra,extra_state=(None,dict(status='private_fixture_not_applicable')) if local else available_seed2(
        report.resolve().parents[1],current,report/'references_used')
    seeds=[SEED] if local else ([1,2,SEED] if extra is not None else [1,SEED])
    output=dict(status='completed',seeds=seeds,local_fixture=local,sample_standard_deviation=True,
                limited_n_not_significance=True,seed2_reference=extra_state,splits={})
    lines=['# YOLO26s-P2 baseline / +E60 / seed3 serial','',
           'Paired seeds included: '+', '.join(map(str,seeds))+'.',
           'Seed2 reference status: '+extra_state['status']+'.','']
    if extra is None and not local:
        lines+=['Seed2 is NOT silently counted. Reason: '+extra_state.get('reason','unavailable')+'.','']
    for split,fresh in current.items():
        rows={SEED:{v:fresh[v]['official_metrics'] for v in VARIANTS}}
        if not local: rows[1]={v:originals[split][v]['official_metrics'] for v in VARIANTS}
        if extra is not None: rows[2]={v:extra[split][v]['official_metrics'] for v in VARIANTS}
        aggregate={}
        for key in KEYS:
            values={v:[rows[s][v][key] for s in seeds] for v in VARIANTS}
            deltas=[rows[s]['e60'][key]-rows[s]['baseline'][key] for s in seeds]
            aggregate[key]=dict(baseline_mean=statistics.mean(values['baseline']),
                e60_mean=statistics.mean(values['e60']),delta_mean_pp=statistics.mean(deltas),
                baseline_sample_std=statistics.stdev(values['baseline']) if len(rows)>1 else None,
                e60_sample_std=statistics.stdev(values['e60']) if len(rows)>1 else None,
                paired_delta_sample_std=statistics.stdev(deltas) if len(rows)>1 else None,
                paired_delta_pp={str(s):rows[s]['e60'][key]-rows[s]['baseline'][key] for s in seeds})
        output['splits'][split]=dict(seed3=fresh,seed_rows={str(s):rows[s] for s in seeds},aggregate=aggregate)
        lines+=['## '+split,'','|Metric (%)|Baseline seed3|+E60 seed3|Seed3 delta pp|Mean paired delta pp|',
                '|---|---:|---:|---:|---:|']
        for key in KEYS:
            b,e=rows[SEED]['baseline'][key],rows[SEED]['e60'][key]
            lines.append(f'|{key}|{b:.4f}|{e:.4f}|{e-b:+.4f}|{aggregate[key]["delta_mean_pp"]:+.4f}|')
        lines+=['','|Seed|Baseline AP|+E60 AP|Paired AP delta pp|','|---|---:|---:|---:|']
        for seed in seeds:
            b,e=(rows[seed][v]['AP50_95_percent'] for v in VARIANTS)
            lines.append(f'|{seed}|{b:.4f}|{e:.4f}|{e-b:+.4f}|')
        if len(rows)>1:
            lines+=['','|Metric (%)|Baseline mean +/- sample SD|+E60 mean +/- sample SD|Mean paired delta pp|',
                    '|---|---:|---:|---:|']
            for key,a in aggregate.items():
                lines.append(f'|{key}|{a["baseline_mean"]:.4f} +/- {a["baseline_sample_std"]:.4f}|'
                             f'{a["e60_mean"]:.4f} +/- {a["e60_sample_std"]:.4f}|{a["delta_mean_pp"]:+.4f}|')
        lines+=['']
    lines+=['Both seed3 candidates train from the original YOLO26s checkpoint, not old best.pt.',
            'Frozen seed1/2 model construction, shared/isolated initialization, dual loss/STAL and 200-epoch protocol.',
            'Only global training seed3 and output paths change. Native mAP is not official AP.',
            'FP32/640/b4/one2many+NMS/conf.001/IoU.7/max500, fixed val-best before test-dev evaluation.',
            'All positive and negative paired seeds are retained. A small number of seeds is not proof of significance.','']
    write(report/'comparison.json',output)
    (report/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
