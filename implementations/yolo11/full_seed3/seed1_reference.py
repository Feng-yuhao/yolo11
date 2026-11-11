"""Read-only completed E60 seeds1/2 identities and honest three-seed summaries."""
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

REPORT = 'E60_RUN_20261010_133029_103161'
SHA = 'bf38299c8723f5fe399e2f94fa4042df2ff6ae51dcb99fceece8bc85ed00dc17'
EXPECTED = dict(AP50_95_percent=26.969156363130686,
                AP50_percent=46.39520525911735,AP75_percent=26.567837427107293,
                AR500_percent=39.00411627684551)
FROZEN = ('spatial_task_head.py','e58_spatial_task_head.py','e57_spatial_task_head.py',
          'parent_spatial_task_head.py','e1_reference.yaml','e1_640_core.py',
          'p2_tal_chunked.py','d27_official.py','d38_common.py','precision_checks.py',
          'group_checks.py','optimizer_checks.py','validation_smoke.py','smoke_progress.py')


def verify_seed1(args,dataset,params=None):
    import d27_official as official
    root=args.project/'comparison_reports'/REPORT
    comparison=json.loads((root/'comparison.json').read_text())
    training=json.loads((root/'training.json').read_text())
    if (comparison['status']!='completed' or comparison['variant']!='e60' or
        comparison['seed']!=1 or comparison['imgsz']!=640 or
        comparison['checkpoint_sha256']!=SHA or training['status']!='completed' or
        training['completed_epochs']!=200 or training['params']['seed']!=1):
        raise RuntimeError('Missing/wrong completed E60 seed1 reference')
    if comparison['dataset']!=dataset:raise RuntimeError('E60 seed1 dataset differs')
    best=args.project/'runs/e60a_e58_group_residual_encoding_img640_seed1_200e/weights/best.pt'
    if not best.is_file() or official.sha256(best)!=SHA:
        raise RuntimeError('E60 seed1 checkpoint missing/changed: '+str(best))
    for key,value in EXPECTED.items():
        if abs(comparison['candidate'][key]-value)>1e-7:
            raise RuntimeError('E60 seed1 reference metric changed: '+key)
    if params is not None:
        ignored={'seed','project','name','model','data','device'}
        left={k:v for k,v in training['params'].items() if k not in ignored}
        right={k:v for k,v in params.items() if k not in ignored}
        if left!=right:raise RuntimeError('E60 seed1/3 protocol drift beyond seed/output paths')
    here=Path(__file__).resolve().parent
    with zipfile.ZipFile(root/'actual_source.zip') as archive:
        for name in FROZEN:
            if archive.read(name).replace(b'\r\n',b'\n')!=(here/name).read_bytes().replace(b'\r\n',b'\n'):
                raise RuntimeError('E60 actual seed1 implementation changed: '+name)
    return comparison


def verify_seed2(args,dataset,params=None):
    import d27_official as official
    from run_experiment import E60_SEED2_REPORT
    root=args.project/'comparison_reports'/E60_SEED2_REPORT
    comparison=json.loads((root/'comparison.json').read_text())
    training=json.loads((root/'training.json').read_text())
    sha='03e58654cd47412b4122e4c1fe0a7880844e37692931289ebf59f2e1d65130d0'
    expected=dict(AP50_95_percent=26.694683551773657,
                  AP50_percent=45.979007332797934,AP75_percent=26.347538442768755,
                  AR500_percent=38.873229001842915)
    if (comparison['status']!='completed' or comparison['variant']!='e60' or
        comparison['seed']!=2 or comparison['imgsz']!=640 or comparison['dataset']!=dataset or
        comparison['checkpoint_sha256']!=sha or training['status']!='completed' or
        training['completed_epochs']!=200 or training['params']['seed']!=2):
        raise RuntimeError('Missing/wrong completed E60 seed2 reference')
    best=args.project/'runs/e60a_e58_group_residual_encoding_img640_seed2_200e/weights/best.pt'
    if not best.is_file() or official.sha256(best)!=sha:
        raise RuntimeError('E60 seed2 checkpoint missing/changed: '+str(best))
    for key,value in expected.items():
        if abs(comparison['candidate'][key]-value)>1e-7:
            raise RuntimeError('E60 seed2 reference metric changed: '+key)
    if params is not None:
        ignored={'seed','project','name','model','data','device'}
        left={k:v for k,v in training['params'].items() if k not in ignored}
        right={k:v for k,v in params.items() if k not in ignored}
        if left!=right:raise RuntimeError('E60 seed2/3 protocol drift beyond seed/output paths')
    here=Path(__file__).resolve().parent
    with zipfile.ZipFile(root/'actual_source.zip') as archive:
        for name in FROZEN:
            if archive.read(name).replace(b'\r\n',b'\n')!=(here/name).read_bytes().replace(b'\r\n',b'\n'):
                raise RuntimeError('E60 actual seed2 implementation changed: '+name)
    return comparison


def summarize(first,second,third):
    """Exactly seeds1/2/3 for E1/E40/E60; E58 remains explicitly two-seed."""
    from d38_common import KEYS
    reports=(first,second,third)
    if ([r.get('seed') for r in reports]!=[1,2,3] or
        any(r.get('status')!='completed' or r.get('variant')!='e60' or r.get('imgsz')!=640
            or r.get('dataset')!=first.get('dataset') for r in reports)):
        raise RuntimeError('Cannot combine unmatched seed reports')
    result=dict(seeds=[1,2,3],split='validation',paired=True,
                boundary='Three fixed paired seeds describe repeatability, not a significance test, test-set performance or publication guarantee. E58 has only seeds1/2.',
                models={},paired_deltas={})
    for model,field in (('E1','baseline'),('E40','parent_E40'),('E58','parent_E58'),('E60','candidate')):
        selected=reports[:2] if model=='E58' else reports
        result['models'][model]={key:dict(seeds=[r['seed'] for r in selected],n=len(selected),
            values=[r[field][key] for r in selected],mean=statistics.mean(r[field][key] for r in selected),
            sample_SD=statistics.stdev(r[field][key] for r in selected)) for key in KEYS}
    for model,field in (('E1','baseline'),('E40','parent_E40'),('E58','parent_E58')):
        result['paired_deltas']['E60_vs_'+model]={}
        selected=reports[:2] if model=='E58' else reports
        for key in KEYS:
            values=[r['candidate'][key]-r[field][key] for r in selected]
            result['paired_deltas']['E60_vs_'+model][key]=dict(seeds=[r['seed'] for r in selected],n=len(selected),values=values,
                mean=statistics.mean(values),sample_SD=statistics.stdev(values),both_positive=all(v>0 for v in values))
    return result
