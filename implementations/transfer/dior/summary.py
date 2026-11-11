"""Report every paired source seed, including losses; never rank/select datasets."""
import statistics
from common import KEYS,SPECS,PROTOCOL,read,require,write
def summarize(report):
    scores={k:read(report/'evaluation'/k/'metrics.json') for k in KEYS}
    require(all(s['status']=='completed' for s in scores.values()),'Twelve completed scores required')
    result=dict(status='completed',protocol=PROTOCOL,source_seeds=[1,2,3],targets={})
    lines=['# Frozen VisDrone -> DIOR / TinyPerson, source seeds1/2/3','',
           'Twelve serial evaluations, no target training or BN adaptation; all checkpoint and mapping choices fixed before target AP.','']
    for target,spec in SPECS.items():
        selected={k:scores[target+'_'+k] for k in ('e1_seed1','e60_seed1','e1_seed2','e60_seed2','e1_seed3','e60_seed3')}
        info=dict(models=selected,paired={},aggregates={});result['targets'][target]=info
        lines += ['## '+target,'',spec['evaluator'],spec['mapping'],'',
                  '|Source model|AP (%)|AP50 (%)|AP75 (%)|','|---|---:|---:|---:|']
        for k,v in selected.items():lines.append(f'|{k}|{v["AP_percent"]:.4f}|{v["AP50_percent"]:.4f}|{v["AP75_percent"]:.4f}|')
        lines += ['','|Metric|E1 mean ± sample SD|E60 mean ± sample SD|Mean paired gain (pp)|','|---|---:|---:|---:|']
        fields=['AP_percent','AP50_percent','AP75_percent','AP_small_percent','AP_medium_percent','AP_large_percent']
        if target=='tiny':fields+=['native:AP25_tiny_percent','native:AP50_tiny_percent','native:AP75_tiny_percent']
        for field in fields:
            def value(k):return selected[k]['native_tiny'][field[7:]] if field.startswith('native:') else selected[k][field]
            vals={kind:[value(f'{kind}_seed{s}') for s in (1,2,3)] for kind in ('e1','e60')}
            if any(x is None for a in vals.values() for x in a):info['aggregates'][field]=None;continue
            gain=[b-a for a,b in zip(vals['e1'],vals['e60'])];info['paired'][field]=gain
            row={kind:dict(mean=statistics.mean(a),sample_SD=statistics.stdev(a)) for kind,a in vals.items()}
            row.update(paired_gain_mean_pp=statistics.mean(gain),paired_gain_sample_SD_pp=statistics.stdev(gain));info['aggregates'][field]=row
            a,b=row['e1'],row['e60'];lines.append(f'|{field}|{a["mean"]:.4f} ± {a["sample_SD"]:.4f}|{b["mean"]:.4f} ± {b["sample_SD"]:.4f}|{statistics.mean(gain):+.4f}|')
        lines+=['','All paired gains, seed1/2/3: '+str(info['paired']),'']
    lines+=['Do not average the two datasets into a single AP. TinyPerson AP retains its author ignore/IOD matching and is NOT vanilla COCO AP.',
            'DIOR is vehicle-only, TinyPerson is merged-person; neither is a full-class source-head benchmark.',
            'Three source seeds do not create independent target scenes. Earlier Songdo/UAVDT/State-Air positive and negative results remain unchanged.']
    write(report/'summary.json',result);(report/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return result
