"""Read-only original Songdo Vision v2 download audit; never changes archives."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import time
import zipfile

EXPECTED = {
    'train.zip': dict(md5='521fff93d9a2d0034d1133188cde8fb6', bytes=12900897193, images=4335, boxes=218662),
    'test.zip': dict(md5='4a84cebf5accd465e08e01bee6a3524e', bytes=3216450659, images=1084, boxes=55528),
}


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    temp.replace(path)


def inspect(root, output, full=True):
    from PIL import Image
    result = dict(dataset='Songdo Vision v2',source='https://zenodo.org/records/21728430',archives={})
    for name, expected in EXPECTED.items():
        path = root / name
        if path.stat().st_size != expected['bytes']:
            raise RuntimeError('Archive byte count mismatch: '+str(path))
        with zipfile.ZipFile(path) as archive:
            entries = [i for i in archive.infolist() if not i.is_dir()]
            names = [i.filename for i in entries]
            if len(names) != len(set(names)):
                raise RuntimeError('Duplicate ZIP names: '+name)
            for item in entries:
                p = PurePosixPath(item.filename)
                if p.is_absolute() or '..' in p.parts or '\\' in item.filename or ':' in item.filename:
                    raise RuntimeError('Unsafe member: '+item.filename)
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError('ZIP symlink unsupported')
            extensions = Counter(PurePosixPath(n).suffix.lower() for n in names)
            coco_names = [n for n in names if n.endswith('coco_annotations.json')]
            if len(coco_names) != 1:
                raise RuntimeError('Expected one original COCO annotation JSON')
            coco = json.loads(archive.read(coco_names[0]))
            if len(coco['images']) != expected['images'] or len(coco['annotations']) != expected['boxes']:
                raise RuntimeError('Original annotation counts differ')
            portrait = [i for i in coco['images'] if i['height']>i['width']]
            samples = []
            image_names = [n for n in names if n.lower().endswith('.jpg')]
            for filename in image_names[:2]:
                content = archive.read(filename)
                with Image.open(io.BytesIO(content)) as image:
                    image.load()
                    samples.append(dict(name=filename,dimensions=list(image.size),format=image.format,
                                        mode=image.mode,EXIF_orientation=image.getexif().get(274,1)))
            labels = [n for n in names if n.endswith('.txt')]
            xmls = [n for n in names if n.endswith('.xml')]
            info = dict(path=str(path),bytes=path.stat().st_size,files=len(entries),
                        uncompressed_bytes=sum(i.file_size for i in entries),extensions=dict(extensions),
                        first_names=names[:10],last_names=names[-4:],coco_member=coco_names[0],
                        categories=coco['categories'],image_keys=list(coco['images'][0]),
                        first_images=coco['images'][:2],first_annotations=coco['annotations'][:3],
                        portraits=portrait,COCO_info=coco.get('info'),image_samples=samples,
                        yolo_sample={n:archive.read(n).decode('utf-8-sig').splitlines()[:3] for n in labels[:2]},
                        XML_sample={n:archive.read(n).decode('utf-8-sig')[:1800] for n in xmls[:1]},
                        original_split_counts=expected)
            result['archives'][name] = info
            write(output / 'inventory.json',result)
            print(json.dumps({k:info[k] for k in ('files','uncompressed_bytes','extensions','categories','image_keys','first_annotations','portraits','yolo_sample')},ensure_ascii=False),flush=True)
        if full:
            started = time.monotonic(); md5 = hashlib.md5(); sha = hashlib.sha256(); done=0
            with path.open('rb') as handle:
                while True:
                    block = handle.read(16*1024*1024)
                    if not block:
                        break
                    md5.update(block); sha.update(block); done += len(block)
                    if done//(1024**3) != (done-len(block))//(1024**3):
                        print(f'{name} verify {done/1024**3:.1f}/{path.stat().st_size/1024**3:.1f}GiB elapsed={time.monotonic()-started:.1f}s',flush=True)
            info.update(md5=md5.hexdigest(),sha256=sha.hexdigest(),hash_seconds=time.monotonic()-started)
            info['MD5_matches_official'] = info['md5'] == expected['md5']
            write(output / 'inventory.json',result)
            if not info['MD5_matches_official']:
                raise RuntimeError('MD5 does NOT match official file: '+name)
            print(name+' OFFICIAL MD5 PASSED; SHA256='+info['sha256'],flush=True)
    result['status'] = 'hashes_passed' if full else 'inventory_only'
    write(output / 'inventory.json',result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--inventory-only',action='store_true')
    a = p.parse_args(); inspect(a.root,a.output,not a.inventory_only)
