"""All three paired seeds; seed1 comes from a byte-bound completed report."""
import statistics
from common import HERE,KEYS,PROTOCOL,read,require,write


def summarize(report):
    ref=read(HERE/'references/seed1_reference.json')
    models=dict(ref['models']); native=dict(ref['native'])
    for k in KEYS:
        models[k]=read(report/'evaluation'/k/'coco_metrics.json')
        native[k]=read(report/'evaluation'/k/'native_ap70.json')
    fields=['AP_percent','AP50_percent','AP75_percent','AP_small_percent','AP_medium_percent','AP_large_percent']
    compared=['mapping','score_merge','postprocess','predict','batch','multi_label','max_nms','evaluator','maxDets','GT','ignored_regions']
    for k,v in models.items():
        require(v['status']=='completed' and v['images']==16592,'Incomplete model '+k)
        require(v['gt_sha256']==ref['models']['e1_seed1']['gt_sha256'],'Different target GT')
        require(all(v['protocol'][f]==ref['models']['e1_seed1']['protocol'][f] for f in compared),'Protocol differs '+k)
        require(native[k]['images']==16592 and native[k]['GT_boxes']==375884,'Incomplete native AP70')
    aggregates={}; paired={}
    for f in fields:
        a=[models[f'e1_seed{s}'][f] for s in (1,2,3)]
        b=[models[f'e60_seed{s}'][f] for s in (1,2,3)]
        if any(v is None for v in a+b): aggregates[f]=None;continue
        gains=[y-x for x,y in zip(a,b)];paired[f]=gains
        aggregates[f]=dict(e1_mean=statistics.mean(a),e1_SD=statistics.stdev(a),e60_mean=statistics.mean(b),
                           e60_SD=statistics.stdev(b),paired_gain_mean_pp=statistics.mean(gains),paired_gain_SD_pp=statistics.stdev(gains))
    result=dict(status='completed',models=models,native=native,paired_gain_pp=paired,aggregates=aggregates,protocol=PROTOCOL,
                seed1_reference=ref['origin'],new_evaluations=list(KEYS))
    write(report/'summary.json',result)
    lines=['# VisDrone -> UAVDT / all three paired source seeds','',
           'Source-only inference, no target training/adaptation/checkpoint selection.',
           'Seed1 is the frozen previously completed report; seeds2/3 are new evaluations.',
           'COCO-style AP after author ignored-region filtering; native class-agnostic AP70 is a separate Python math port.','',
           '|Seed|E1 AP|E60 AP|Delta pp|E1 AP50|E60 AP50|E1 AP75|E60 AP75|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in (1,2,3):
        a,b=models[f'e1_seed{s}'],models[f'e60_seed{s}']
        lines.append(f'|{s}|{a["AP_percent"]:.4f}|{b["AP_percent"]:.4f}|{b["AP_percent"]-a["AP_percent"]:+.4f}|'
                     f'{a["AP50_percent"]:.4f}|{b["AP50_percent"]:.4f}|{a["AP75_percent"]:.4f}|{b["AP75_percent"]:.4f}|')
    lines+=['','|Metric|E1 mean +/- SD|E60 mean +/- SD|Paired mean gain pp|','|---|---:|---:|---:|']
    for f,v in aggregates.items():
        if v is not None: lines.append(f'|{f}|{v["e1_mean"]:.4f} +/- {v["e1_SD"]:.4f}|{v["e60_mean"]:.4f} +/- {v["e60_SD"]:.4f}|{v["paired_gain_mean_pp"]:+.4f}|')
    lines+=['','|Seed|E1 native AP70|E60 native AP70|Delta pp|','|---|---:|---:|---:|']
    for s in (1,2,3):
        a,b=(native[f'{k}_seed{s}']['AP70_overall_percent'] for k in ('e1','e60'))
        lines.append(f'|{s}|{a:.4f}|{b:.4f}|{b-a:+.4f}|')
    lines+=['','All positive and negative results retained. These are repeated source checkpoints, not three independent target datasets.',
            'AP-small uses original image areas. Compact JSON changes whitespace only, not predictions or evaluation math.']
    (report/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
