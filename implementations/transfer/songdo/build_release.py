"""Local deterministic packaging; no remote calls and no experiment launch."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def build():
    old = json.loads((ROOT/'PAPER_EXPERIMENTS/code/Paper_NoTrain_Supplement_AutoDL/inputs.json').read_text())
    frozen = {}
    for name in ('spatial_task_head.py','e58_spatial_task_head.py','e57_spatial_task_head.py','parent_spatial_task_head.py'):
        original = ROOT/'PAPER_EXPERIMENTS/code/evaluation_TESTDEV_E60_seed1'/name
        assert sha(HERE/name) == sha(original), 'Frozen source model changed: '+name
        frozen[name] = dict(sha256=sha(original),origin='evaluation_TESTDEV_E60_seed1/'+name)
    dump(HERE/'references/frozen_model_assets.json',frozen)
    specs = {}
    for seed in (1,2,3):
        for kind in ('e1','e60'):
            key = f'{kind}_seed{seed}'; relative = old['weights'][key]
            path = ROOT/'PAPER_EXPERIMENTS/project'/relative
            assert sha(path) == old['checkpoint_sha256'][key], key+' source checkpoint differs'
            specs[key] = dict(relative_path=relative,sha256=sha(path),source_dataset='VisDrone2019',
                              source_seed=seed,selection='source val-selected best, not target selected',
                              original_binding='Paper_NoTrain_Supplement_AutoDL/inputs.json')
    dump(HERE/'references/checkpoints.json',specs)
    files = sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and
                   p.name not in ('manifest.json',) and not p.name.endswith('.pyc'))
    for path in files:
        if path.suffix == '.py': ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
        if path.suffix == '.sh':
            assert b'\r' not in path.read_bytes(), 'Bash launcher must use LF'
    dump(HERE/'manifest.json',dict(revision='visdrone_to_songdo_source_only_v1',
                                  files={p.relative_to(HERE).as_posix():sha(p) for p in files}))
    destination = ROOT/'PAPER_EXPERIMENTS/deliverables'/f'{HERE.name}.zip'
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in [*files,HERE/'manifest.json']:
            info = zipfile.ZipInfo(HERE.name+'/'+path.relative_to(HERE).as_posix(),date_time=(2026,10,23,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o644 << 16
            archive.writestr(info,path.read_bytes())
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(files)+1
    checksum = sha(destination)
    destination.with_suffix('.zip.sha256').write_text(checksum+'  '+destination.name+'\n',encoding='ascii')
    print(json.dumps(dict(path=str(destination),bytes=destination.stat().st_size,sha256=checksum,files=len(files)+1),indent=2))


if __name__ == '__main__': build()
