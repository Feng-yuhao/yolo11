"""Seal/package only. Never train, connect to a server, or call shutdown."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent


def digest(path):
    raw = path.read_bytes()
    if path.suffix != '.pt':
        raw = raw.replace(b'\r\n',b'\n')
    return hashlib.sha256(raw).hexdigest()


def build(output=None, seal_only=False):
    names = []
    for path in HERE.rglob('*'):
        rel = path.relative_to(HERE)
        if (not path.is_file() or '__pycache__' in rel.parts or path.name == 'manifest.json' or
                path.suffix not in ('.py','.sh','.yaml','.json','.md','.txt','.pt')):
            continue
        if len(rel.parts)>1 and rel.parts[0] not in ('references','author_source'):
            continue
        names.append(path)
    names.sort()
    manifest = {p.relative_to(HERE).as_posix():digest(p) for p in names}
    (HERE / 'manifest.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode('utf-8'))
    if seal_only:
        return dict(sealed=True,files=len(names)+1)
    output = Path(output or HERE.parents[1] / 'deliverables')
    output.mkdir(parents=True,exist_ok=True)
    archive = output / (HERE.name + '.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as target:
        for path in names + [HERE / 'manifest.json']:
            data = path.read_bytes()
            if path.suffix != '.pt':
                data = data.replace(b'\r\n',b'\n')
            target.writestr(HERE.name + '/' + path.relative_to(HERE).as_posix(),data)
    with zipfile.ZipFile(archive) as target:
        if target.testzip() is not None:
            raise RuntimeError('Release CRC failure')
    value = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.zip.sha256').write_bytes((value+'  '+archive.name+'\n').encode('ascii'))
    return dict(archive=str(archive),sha256=value,files=len(names)+1,bytes=archive.stat().st_size)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path); p.add_argument('--seal-only',action='store_true')
    args=p.parse_args(); print(json.dumps(build(args.output,args.seal_only),indent=2))
