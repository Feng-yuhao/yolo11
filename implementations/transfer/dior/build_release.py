"""Bind sources/checkpoints, syntax-check and make a deterministic portable ZIP."""
import ast
import json
from pathlib import Path
import zipfile
from common import HERE,REVISION,sha
def build():
    root=HERE.parents[2];old=root/'PAPER_EXPERIMENTS/code/VisDrone_to_StateAir_ThreeSeeds_SourceOnly_AutoDL'
    for n in ('spatial_task_head.py','e58_spatial_task_head.py','e57_spatial_task_head.py','parent_spatial_task_head.py','modeling.py'):
        assert sha(HERE/n)==sha(old/n),'Frozen source model changed: '+n
    for k,s in json.loads((HERE/'references/checkpoints.json').read_text(encoding='utf-8')).items():assert sha(root/'PAPER_EXPERIMENTS/project'/s['relative_path'])==s['sha256'],k
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json')
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        if p.suffix=='.sh':assert b'\r' not in p.read_bytes(),'Bash must use LF'
    with (HERE/'manifest.json').open('w',encoding='utf-8',newline='\n') as f:json.dump(dict(revision=REVISION,files={p.relative_to(HERE).as_posix():sha(p) for p in files}),f,indent=2);f.write('\n')
    dest=root/'PAPER_EXPERIMENTS/deliverables'/(HERE.name+'.zip')
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in [*files,HERE/'manifest.json']:
            info=zipfile.ZipInfo(HERE.name+'/'+p.relative_to(HERE).as_posix(),date_time=(2026,10,24,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16;z.writestr(info,p.read_bytes())
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    with dest.with_suffix('.zip.sha256').open('w',encoding='ascii',newline='\n') as f:f.write(sha(dest)+'  '+dest.name+'\n')
    print(json.dumps(dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest)),indent=2))
if __name__=='__main__':build()
