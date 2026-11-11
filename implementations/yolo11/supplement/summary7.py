"""Keep every seed, combine only like-for-like official protocols, sample SD."""
import csv,io,statistics
from common7 import *

def cached_rows():
    rows=[]
    def add(model,seed,split,metrics,path):
        finite_metrics(metrics)
        rows.append(dict(model=model,seed=seed,split=split,**{k:metrics[k] for k in KEYS},
                         origin='cached_original',source=path,source_sha256=sha(HERE/path)))
    main='references/official_main_per_seed.csv'
    with (HERE/main).open(encoding='utf-8-sig') as f:
        for value in csv.DictReader(f):
            if value['model'] in ['e1','e60']:
                add(value['model'],int(value['seed']),value['split'],{k:float(value[k]) for k in KEYS},main)
    route='references/routing_seed1_summary.json';d=read(HERE/route);require(d['status']=='completed','Cached routing incomplete')
    for kind in ['uniform','shared']:
        r=d['results'][kind];require(r['status']=='completed' and r['seed']==1,'Wrong cached routing seed')
        add(kind,1,'val',r['candidate'],route)
    for seed in [1,2]:
        path=f'references/e58_seed{seed}_comparison.json';d=read(HERE/path)
        require(d['status']=='completed' and d['seed']==seed and d['variant']=='e58','Cached E58 identity differs')
        add('e58',seed,'val',d['candidate'],path)
        path=f'references/e58_seed{seed}_test_official_metrics.json';d=read(HERE/path)
        pred=read(HERE/f'references/e58_seed{seed}_test_prediction_manifest.json')
        require(d['status']=='completed' and pred['status']=='completed' and pred['seen_images']==1610 and d['prediction_sha256']==pred['sha256'],'Cached E58 test incomplete')
        require(d['binding']['image_count']==1610 and d['binding']['dataset_annotation_sha256']==read(HERE/'references/reference_protocol.json')['official_test_dataset']['annotation_dimension_sha256'],'Cached E58 test protocol differs')
        add('e58',seed,'test-dev',d['official'],path)
    for split,name in [('val','val'),('test-dev','test')]:
        path=f'references/td_seed1_{name}_metrics.json';d=read(HERE/path)
        require(d['status']=='completed' and d['binding']['seed']==1,'Cached TD identity differs')
        add('tood_td',1,split,d['official'],path)
    require(len({(r['model'],r['seed'],r['split']) for r in rows})==len(rows),'Duplicated cached result')
    return rows

def csv_text(rows):
    handle=io.StringIO();w=csv.DictWriter(handle,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return handle.getvalue()

def aggregate(rows):
    groups={};paired=[]
    mapping={(r['model'],r['seed'],r['split']):r for r in rows}
    for split in ['val','test-dev']:
        for model in ['e1','e60','e58','uniform','shared','tood_td']:
            members=[mapping[(model,s,split)] for s in [1,2,3] if (model,s,split) in mapping]
            # Do not turn incomplete two-seed test scoring into a three-seed table.
            if len(members)!=3:continue
            groups[split+'/'+model]=dict(split=split,model=model,seeds=[1,2,3],n=3,
                metrics={k:dict(mean=statistics.mean(r[k] for r in members),sample_sd=statistics.stdev(r[k] for r in members)) for k in KEYS})
        for candidate,reference in [('e60','e1'),('e60','e58'),('e60','uniform'),('e60','shared'),('e60','tood_td'),('tood_td','e1')]:
            if not all((m,s,split) in mapping for m in [candidate,reference] for s in [1,2,3]):continue
            for metric in KEYS:
                diff=[mapping[(candidate,s,split)][metric]-mapping[(reference,s,split)][metric] for s in [1,2,3]]
                paired.append(dict(split=split,candidate=candidate,reference=reference,metric=metric,n=3,
                              seed1=diff[0],seed2=diff[1],seed3=diff[2],mean_pp=statistics.mean(diff),sample_sd_pp=statistics.stdev(diff),positive_seeds=sum(x>0 for x in diff)))
    return groups,paired

def summarize(project,report):
    report=Path(report);rows=cached_rows()
    for key,job in JOBS.items():
        for split in ['val','test-dev']:
            p=report/'jobs'/key/split/'official_metrics.json';d=read(p)
            require(d['status']=='completed' and d['key']==key and d['seed']==job['seed'],'Incomplete fresh score '+key)
            finite_metrics(d['official'])
            rows.append(dict(model=job['variant'],seed=job['seed'],split=split,**{k:d['official'][k] for k in KEYS},
                         origin='new_seven_run_batch',source=str(p),source_sha256=sha(p)))
    require(len({(r['model'],r['seed'],r['split']) for r in rows})==len(rows),'Fresh/cached result collision')
    groups,paired=aggregate(rows)
    require(all('val/'+m in groups for m in ['e1','e60','e58','uniform','shared','tood_td']),'Not all validation comparisons have three seeds')
    require('test-dev/tood_td' in groups and 'test-dev/e58' in groups,'Three-seed TD/E58 test missing')
    text(report/'per_seed.csv',csv_text(rows));text(report/'paired_differences.csv',csv_text(paired))
    write(report/'summary.json',dict(status='completed',at=now(),new_formal_runs=7,new_official_scores=14,
          per_seed=rows,aggregates=groups,paired_differences=paired,seed_selection=False,
          missing_old_tests=['uniform seed1 test-dev','shared seed1 test-dev'],
          boundary='Routing controls have all three validation seeds; their new seed2/3 tests are retained individually, not labelled a three-seed test mean. No test-driven tuning or favourable-seed selection.'))
    md='# ST-MDRF seven-run seed supplement\n\nOriginal YOLO11 / COCO initialization / 640 / batch16 / workers8 / SGD / 200 epochs.\n'
    md+='All seven checks, real individual smokes and permitted joint gates precede any formal training. Official val/test-dev use the original VisDrone Matlab1.0.4 ignore-region port, not native YOLO AP.\n\n'
    for split in ['val','test-dev']:
        md+='## '+split+' / mean ± sample SD / three paired seeds\n\n|Model|AP (%)|AP50 (%)|AP75 (%)|AR500 (%)|\n|---|---:|---:|---:|---:|\n'
        for model in ['e1','e60','e58','uniform','shared','tood_td']:
            group=groups.get(split+'/'+model)
            if group:md+='|'+model+'|'+ '|'.join(f'{group["metrics"][k]["mean"]:.4f} ± {group["metrics"][k]["sample_sd"]:.4f}' for k in KEYS)+'|\n'
        md+='\n|Paired AP difference|seed1|seed2|seed3|mean pp|sample SD pp|\n|---|---:|---:|---:|---:|---:|\n'
        for r in paired:
            if r['split']==split and r['metric']=='AP50_95_percent':md+=f'|{r["candidate"]} − {r["reference"]}|{r["seed1"]:+.4f}|{r["seed2"]:+.4f}|{r["seed3"]:+.4f}|{r["mean_pp"]:+.4f}|{r["sample_sd_pp"]:.4f}|\n'
        md+='\n'
    md+='Original cached evidence is bundled with source hashes; existing E1/E60 were not retrained. E58 means the no-group-readout control; TD means TaskDecomposition adaptation, not complete TOOD. Uniform and Shared overlap and their differences must not be added as independent contributions.\n\n'
    md+='Uniform/Shared seed1 test-dev was not in this batch: their seed2/3 full test results are in per_seed.csv, not misrepresented as n=3. Keep all positive and negative results. Native metrics and two-epoch engineering smoke are not paper results.\n'
    text(report/'REPORT.md',md)
    return groups
