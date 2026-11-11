"""Mechanical copies of already validated unchanged modules/weights, manifest and ZIP."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]
PRE=HERE.parent/'YOLO26_P2_E60_Preflight_AutoDL'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def lf_write(path,text):
    with path.open('w',encoding='utf-8',newline='\n') as handle: handle.write(text)


def build():
    for name in ('common.py','models26.py','e60_adapter.py','checks26.py','source_sha256.json','LICENSE'):
        shutil.copyfile(PRE/name,HERE/name)
        assert sha(PRE/name)==sha(HERE/name)
    shutil.copyfile(HERE.parent/'E60_seed1/d27_official.py',HERE/'d27_official.py')
    (HERE/'weights').mkdir(exist_ok=True)
    shutil.copyfile(PRE/'weights/yolo26s.pt',HERE/'weights/yolo26s.pt')
    shutil.copyfile(PAPER/'audit/yolo26_port/vendor/yolo26n.pt',HERE/'weights/yolo26n.pt')
    assert sha(HERE/'weights/yolo26n.pt')=='9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef'
    source=PAPER/'audit/yolo26_port/vendor/ultralytics-8.4.177'
    source_files=sorted(p for p in (source/'ultralytics').rglob('*') if p.is_file() and p.suffix in ('.py','.yaml') and '__pycache__' not in p.parts)
    lf_write(HERE/'training_source_sha256.json',json.dumps({p.relative_to(source).as_posix():sha(p) for p in source_files},indent=2)+'\n')
    for path in HERE.glob('*.py'): ast.parse(path.read_text(encoding='utf-8'))
    for path in HERE.glob('*.sh'): lf_write(path,path.read_text())
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                 and p.suffix!='.pyc' and p.name!='manifest.json')
    lf_write(HERE/'manifest.json',json.dumps({p.relative_to(HERE).as_posix():sha(p) for p in files},indent=2)+'\n')
    target=PAPER/'audit'/('YOLO26_P2_E60_Seed1_AutoDL.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files+[HERE/'manifest.json']: archive.write(path,HERE.name+'/'+path.relative_to(HERE).as_posix())
    lf_write(target.with_suffix('.zip.sha256'),f'{sha(target)}  {target.name}\n')
    print(json.dumps(dict(zip=str(target),sha256=sha(target),files=len(files)+1),indent=2))


if __name__=='__main__': build()
