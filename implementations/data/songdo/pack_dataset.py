"""Package only an independently verified dataset. ZIP64, no overwrite or deletion."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile


def pack(root, output):
    root=root.resolve(); output=output.resolve()
    verified=json.loads((root/'VERIFICATION.json').read_text(encoding='utf-8'))
    if verified.get('status')!='passed' or (root/'BUILD_FAILED.json').exists():
        raise RuntimeError('Independent verification must pass before packaging')
    if output.exists() or output.with_suffix('.zip.part').exists():
        raise FileExistsError('Refuse to overwrite existing archive/partial archive')
    output.parent.mkdir(parents=True,exist_ok=True)
    part=output.with_suffix('.zip.part')
    files=sorted(path for path in root.rglob('*') if path.is_file())
    started=time.monotonic()
    with zipfile.ZipFile(part,'x',allowZip64=True,compresslevel=6) as archive:
        for index,path in enumerate(files,1):
            if path.is_symlink(): raise ValueError('Dataset must not contain symlinks')
            # JPEG is already compressed. Store it without changing/reencoding
            # a single image byte; compress labels/JSON/XML/source normally.
            archive.write(path,root.name+'/'+path.relative_to(root).as_posix(),
                          compress_type=zipfile.ZIP_STORED if path.suffix.lower()=='.jpg' else zipfile.ZIP_DEFLATED)
            if index%750==0 or index==len(files):
                print(f'Package {index}/{len(files)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    md5=hashlib.md5(); sha=hashlib.sha256()
    with part.open('rb') as handle:
        for chunk in iter(lambda:handle.read(16*1024*1024),b''):
            md5.update(chunk); sha.update(chunk)
    with zipfile.ZipFile(part) as archive:
        if archive.testzip() is not None: raise RuntimeError('Release ZIP CRC test failed')
        names=archive.namelist()
        if len(names)!=len(files) or any(not n.startswith(root.name+'/') for n in names):
            raise RuntimeError('Release paths/counts mismatch')
    # Refuse a destination created by another process during the build.
    if output.exists(): raise FileExistsError('Output appeared during packaging')
    part.rename(output)
    output.with_suffix('.zip.sha256').write_bytes((sha.hexdigest()+'  '+output.name+'\n').encode('ascii'))
    payload=dict(status='passed',archive=str(output),bytes=output.stat().st_size,files=len(files),
                 sha256=sha.hexdigest(),md5=md5.hexdigest(),ZIP64=True,ZIP_CRC_passed=True,
                 seconds=time.monotonic()-started,original_images_unchanged=True)
    output.with_suffix('.zip.report.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(payload,indent=2),flush=True)
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); pack(a.root,a.output)
