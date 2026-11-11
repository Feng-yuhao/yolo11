"""Mechanical deterministic source packaging; never starts a training job."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
FROZEN = {name:name for name in ('e1_640_core.py','e1_reference.yaml','e57_spatial_task_head.py',
          'e58_spatial_task_head.py','parent_spatial_task_head.py','p2_tal_chunked.py',
          'd27_official.py','d38_common.py','group_checks.py','precision_checks.py',
          'optimizer_checks.py','smoke_progress.py')}
FROZEN.update({'e60_full_head.py':'spatial_task_head.py','e60_validation_smoke.py':'validation_smoke.py'})


def build():
    (HERE/'frozen_sources.json').write_bytes((json.dumps(FROZEN,indent=2)+'\n').encode('utf-8'))
    paths = [p for p in sorted(HERE.iterdir()) if p.is_file() and p.suffix in ('.py','.sh','.yaml','.json','.md','.txt')
             and p.name!='manifest.json']
    manifest = {}
    for path in paths:
        data = path.read_bytes().replace(b'\r\n',b'\n')
        if path.suffix == '.py': ast.parse(data.decode('utf-8'),filename=path.name)
        manifest[path.name] = hashlib.sha256(data).hexdigest()
    (HERE/'manifest.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode('utf-8'))
    output = HERE.parents[1]/'audit'/'E60_Routing_Ablations_Seed1_AutoDL.zip'
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in paths+[HERE/'manifest.json']:
            item = zipfile.ZipInfo(HERE.name+'/'+path.name,date_time=(2026,10,13,0,0,0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = (0o755 if path.suffix=='.sh' else 0o644)<<16
            archive.writestr(item,path.read_bytes().replace(b'\r\n',b'\n'))
    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    # LF bytes, not Windows CRLF; sha256sum filenames must not include '\r'.
    output.with_suffix('.zip.sha256').write_bytes((sha+'  '+output.name+'\n').encode('ascii'))
    print(output); print('SHA256='+sha); print('files='+str(len(paths)+1))


if __name__ == '__main__':
    build()
