"""Build derived manifest/zip only; excludes tests' outputs, weights and receipts."""
from pathlib import Path
import hashlib
import json
import zipfile

HERE=Path(__file__).resolve().parent


def build():
    files=[p for p in sorted(HERE.iterdir()) if p.is_file() and p.suffix in ('.py','.sh','.yaml','.md','.txt')]
    manifest={p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in files}
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    archive=HERE.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as out:
        for p in files+[HERE/'manifest.json']:
            data=p.read_bytes()
            if p.suffix in ('.py','.sh','.yaml','.md','.txt','.json'): data=data.replace(b'\r\n',b'\n')
            out.writestr(HERE.name+'/'+p.name,data)
    print(archive)
    print('files=',len(files)+1,'bytes=',archive.stat().st_size)


if __name__=='__main__': build()
