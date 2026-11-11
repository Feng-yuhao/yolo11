"""Reproducible release; no remote calls or experiment launches."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()


def build():
    old=ROOT/'PAPER_EXPERIMENTS/code/VisDrone_to_Songdo_SourceOnly_AutoDL'
    specs=json.loads((old/'references/checkpoints.json').read_text())
    chosen={f'{k}_seed{s}':specs[f'{k}_seed{s}'] for s in (2,3) for k in ('e1','e60')}
    for k,v in chosen.items():
        assert sha(ROOT/'PAPER_EXPERIMENTS/project'/v['relative_path'])==v['sha256'],k
    (HERE/'references/checkpoints.json').write_text(json.dumps(chosen,indent=2)+'\n',encoding='utf-8')
    prior=ROOT/'PAPER_EXPERIMENTS/project/comparison_reports/VISDRONE_TO_UAVDT_S1_SOURCE_ONLY_20261023_175120_1930'
    state=json.loads((prior/'state.json').read_text())
    assert state['status']=='completed' and sha(prior/'ARTIFACT_SHA256.json')==state['artifact_manifest_sha256']
    manifests=json.loads((prior/'ARTIFACT_SHA256.json').read_text())
    reference=dict(origin=dict(report_name=prior.name,artifact_manifest_sha256=state['artifact_manifest_sha256']),models={},native={})
    for k in ('e1_seed1','e60_seed1'):
        for field,filename in [('models','coco_metrics.json'),('native','native_ap70.json')]:
            rel=f'evaluation/{k}/{filename}';assert sha(prior/rel)==manifests[rel]
            reference[field][k]=json.loads((prior/rel).read_text())
    (HERE/'references/seed1_reference.json').write_text(json.dumps(reference,indent=2)+'\n',encoding='utf-8')
    for n in ('spatial_task_head.py','e58_spatial_task_head.py','e57_spatial_task_head.py','parent_spatial_task_head.py','modeling.py'):
        assert sha(HERE/n)==sha(old/n),'Frozen source model asset changed: '+n
    seed1=ROOT/'PAPER_EXPERIMENTS/code/VisDrone_to_UAVDT_Seed1_SourceOnly_AutoDL'
    for n in ('inference.py','mapping.py','modeling.py','coco_eval.py','native_ap70.py','uavdt_data.py','worker.py','admission.py'):
        assert sha(HERE/n)==sha(seed1/n),'Seed1 prediction/data/scoring code changed: '+n
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and
                 p.name not in ('manifest.json','songdo_author_README.md'))
    for p in files:
        if p.suffix=='.py': ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        if p.suffix=='.sh': assert b'\r' not in p.read_bytes(),'Bash requires LF'
    manifest=dict(revision='visdrone_to_uavdt_seeds23_source_only_v1',files={p.relative_to(HERE).as_posix():sha(p) for p in files})
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    destination=ROOT/'PAPER_EXPERIMENTS/deliverables'/(HERE.name+'.zip')
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in [*files,HERE/'manifest.json']:
            info=zipfile.ZipInfo(HERE.name+'/'+p.relative_to(HERE).as_posix(),date_time=(2026,10,23,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o644<<16; z.writestr(info,p.read_bytes())
    with zipfile.ZipFile(destination) as z: assert z.testzip() is None
    digest=sha(destination)
    destination.with_suffix('.zip.sha256').write_text(digest+'  '+destination.name+'\n',encoding='ascii',newline='\n')
    print(json.dumps(dict(path=str(destination),bytes=destination.stat().st_size,sha256=digest),indent=2))


if __name__=='__main__': build()
