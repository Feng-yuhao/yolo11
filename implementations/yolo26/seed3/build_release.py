"""Build an independent seed3 upload package; never contacts a server."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]
PREVIOUS=HERE.parent/'YOLO26_P2_E60_Seed2_Serial_AutoDL'
FROZEN=('common.py','models26.py','e60_adapter.py','checks26.py','training26.py',
        'scoring26.py','d27_official.py','source_sha256.json','training_source_sha256.json',
        'LICENSE','seed1_protocol.py','process_lock.py','shutdown_safety.py','fixture_tools.py')

def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as file:
        for data in iter(lambda:file.read(1024*1024),b''): value.update(data)
    return value.hexdigest()

def write(path,value):
    with Path(path).open('w',encoding='utf-8',newline='\n') as file:
        json.dump(value,file,ensure_ascii=False,indent=2,allow_nan=False);file.write('\n')

def build():
    manifest=json.loads((PREVIOUS/'manifest.json').read_text(encoding='utf-8'))
    copied={}
    for name in FROZEN:
        source=PREVIOUS/name
        if sha(source)!=manifest[name]: raise RuntimeError('Frozen seed2 source drift: '+name)
        shutil.copyfile(source,HERE/name);copied[name]=sha(source)
    for directory in ('assets','weights','references'):
        for source in (PREVIOUS/directory).rglob('*'):
            if not source.is_file(): continue
            relative=source.relative_to(PREVIOUS).as_posix()
            if sha(source)!=manifest[relative]: raise RuntimeError('Seed1/2 reference/asset drift: '+relative)
            target=HERE/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    write(HERE/'references/frozen_seed2_files.json',copied)
    for path in HERE.glob('*.py'): ast.parse(path.read_text(encoding='utf-8'))
    for path in HERE.glob('*.sh'):
        if b'\r' in path.read_bytes(): raise RuntimeError('Shell script is not LF-only: '+str(path))
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                 and p.suffix!='.pyc' and p.name!='manifest.json')
    write(HERE/'manifest.json',{p.relative_to(HERE).as_posix():sha(p) for p in files})
    target=PAPER/'audit'/(HERE.name+'.zip');target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files+[HERE/'manifest.json']:
            info=zipfile.ZipInfo(HERE.name+'/'+path.relative_to(HERE).as_posix(),(2026,10,17,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            archive.writestr(info,path.read_bytes())
    with target.with_suffix('.zip.sha256').open('w',encoding='ascii',newline='\n') as file:
        file.write(sha(target)+'  '+target.name+'\n')
    print(json.dumps(dict(zip=str(target),sha256=sha(target),files=len(files)+1,bytes=target.stat().st_size),indent=2))

if __name__=='__main__': build()
