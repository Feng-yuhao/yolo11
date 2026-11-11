"""Prepare byte-preserved Songdo Vision v2 YOLO data, never train or alter originals."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import statistics
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile
from PIL import Image

HERE = Path(__file__).resolve().parent
VERSION = 'SongdoVision_v2_YOLO_v1'
CLASSES = ('car','bus','truck','motorcycle')
SPLIT_SEED = 20261017
OFFICIAL = {
    'train':dict(images=4335,boxes=218662,classes=(196719,7052,11861,3030)),
    'test':dict(images=1084,boxes=55528,classes=(49862,1766,3080,820))}
DOC_MD5 = {'data.yaml':'d43bf0a3bad9600ef5fb6beb482a7f61',
           'names.txt':'94789b2db3f32c2dd652eb8c38833612',
           'LICENSE.txt':'527dc6cad772ccb187d5bfe5af738204',
           'README.md':'a3d634c6190a18c04db2aca9e4fc1875'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def dump(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def validation_ids(images):
    count=math.floor(len(images)*.10+.5)
    # No class/metric-dependent selection, no test images or model outputs.
    def rank(image):
        return hashlib.sha256(f'songdo-v2-holdout:{SPLIT_SEED}:{image["id"]}'.encode()).hexdigest()
    return {i['id'] for i in sorted(images,key=rank)[:count]}


def parse_label(data, width, height):
    rows=[]; seen=set()
    for line in data.decode('utf-8-sig').splitlines():
        if not line.strip(): continue
        values=line.split()
        if len(values)!=5: raise ValueError('YOLO detection row needs exactly5 fields')
        cls=int(values[0]); numbers=tuple(map(float,values[1:]))
        if str(cls)!=values[0] or not 0<=cls<4 or not all(math.isfinite(v) for v in numbers):
            raise ValueError('Nonfinite label or invalid class ID')
        cx,cy,bw,bh=numbers
        if not(0<=cx<=1 and 0<=cy<=1 and 0<bw<=1 and 0<bh<=1):
            raise ValueError('Invalid normalized YOLO box')
        xyxy=((cx-bw/2)*width,(cy-bh/2)*height,(cx+bw/2)*width,(cy+bh/2)*height)
        if min(xyxy)<-1e-8 or xyxy[2]>width+1e-8 or xyxy[3]>height+1e-8:
            raise ValueError('Original box exceeds image boundary')
        key=(cls,*numbers)
        if key in seen: raise ValueError('Exact duplicate original YOLO label')
        seen.add(key)
        rows.append((cls,xyxy))
    return rows


def validate_triplet(image, label_data, xml_data, annotations):
    width,height=image['width'],image['height']
    root=ET.fromstring(xml_data)
    stem=Path(image['file_name']).stem
    if ((int(root.findtext('size/width')),int(root.findtext('size/height'))) != (width,height) or
            Path(root.findtext('filename').replace('\\','/')).name != stem+'.jpg'):
        raise ValueError('XML image identity/dimensions differ: '+stem)
    labels=parse_label(label_data,width,height); objects=root.findall('object')
    if len(labels)!=len(objects) or len(labels)!=len(annotations):
        raise ValueError('COCO/VOC/YOLO object count differs: '+stem)
    maximum=0.
    for (cls,corners),obj,ann in zip(labels,objects,annotations):
        x,y,bw,bh=ann['bbox']; integer=(x,y,x+bw,y+bh)
        xml=tuple(int(obj.findtext('bndbox/'+name)) for name in ('xmin','ymin','xmax','ymax'))
        if obj.findtext('name')!=CLASSES[cls] or ann['category_id']!=cls+1 or xml!=integer:
            raise ValueError('Class/order/COCO/VOC corner mismatch: '+stem)
        if (ann['image_id']!=image['id'] or ann.get('iscrowd',0)!=0 or ann['area']!=bw*bh or
                bw<=0 or bh<=0 or x<0 or y<0 or x+bw>width or y+bh>height):
            raise ValueError('Invalid original COCO box or image binding: '+stem)
        error=max(abs(a-b) for a,b in zip(corners,integer)); maximum=max(maximum,error)
        # COCO/VOC round continuous corners; six-place YOLO itself has <=
        # .75e-6*W/H edge error. Do NOT re-round or regenerate the YOLO labels.
        if error > .5+.75e-6*max(width,height)+1e-8:
            raise ValueError('YOLO vs original integer corner mismatch: '+stem)
    return labels,maximum


def size_statistics(values):
    values=sorted(values)
    return dict(count=len(values),median=statistics.median(values),minimum=min(values),maximum=max(values),
                p10=values[int(.10*(len(values)-1))],p90=values[int(.90*(len(values)-1))],
                bins={name:sum(lo<=v<hi for v in values) for lo,hi,name in
                      ((0,8,'lt8'),(8,16,'8_16'),(16,32,'16_32'),(32,float('inf'),'ge32'))})


def build(source_root, output, audit_path, workers=4):
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    if audit['status']!='hashes_passed': raise RuntimeError('Full official archive checksum audit required')
    for split in ('train','test'):
        info=audit['archives'][split+'.zip']; path=source_root/(split+'.zip')
        if not info['MD5_matches_official'] or path.stat().st_size!=info['bytes']:
            raise RuntimeError('Archive changed or did not pass original MD5')
    if output.exists(): raise FileExistsError('Refuse to overwrite an existing dataset: '+str(output))
    output.mkdir(parents=True,exist_ok=False)
    dump(output/'provenance/build_started.json',dict(version=VERSION,source_root=str(source_root),
          archive_sha256={s:audit['archives'][s+'.zip']['sha256'] for s in ('train','test')},split_seed=SPLIT_SEED))
    manifest={}; samples=[]; all_ann_ids=set(); all_ids=set(); full_decode=0; jpeg_hashes={}
    sizes={s:[] for s in ('train','val','test')}; original_sizes={s:[] for s in sizes}
    counts={s:Counter() for s in sizes}; backgrounds=Counter(); portraits=Counter()
    max_corner=0.; started=time.monotonic()
    for split in ('train','val','test'):
        for parent in ('images','labels','annotations_voc'):
            (output/parent/split).mkdir(parents=True)
    for dirname in ('annotations_coco','splits','provenance/official','provenance/original_coco','preparation_code','qa'):
        (output/dirname).mkdir(parents=True,exist_ok=True)

    def store(relative, data):
        path=output/relative; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
        return dict(path=relative,bytes=len(data),sha256=digest(data))

    for original_split in ('train','test'):
        with zipfile.ZipFile(source_root/(original_split+'.zip')) as archive:
            raw_coco=archive.read(original_split+'/coco_annotations.json')
            coco=json.loads(raw_coco); groups=defaultdict(list)
            if [(c['id'],c['name']) for c in coco['categories']] != [(i+1,n) for i,n in enumerate(CLASSES)]:
                raise ValueError('Original categories do not match0-based YOLO mapping')
            for ann in coco['annotations']:
                if ann['id'] in all_ann_ids: raise ValueError('Global annotation ID repeated')
                all_ann_ids.add(ann['id']); groups[ann['image_id']].append(ann)
            image_ids={i['id'] for i in coco['images']}
            if len(image_ids)!=len(coco['images']) or image_ids & all_ids or set(groups)-image_ids:
                raise ValueError('Invalid global COCO image IDs or orphan annotations')
            all_ids.update(image_ids)
            selected=validation_ids(coco['images']) if original_split=='train' else set()
            if original_split=='train' and len(selected)!=434: raise ValueError('Holdout size drift')
            expected=OFFICIAL[original_split]
            if len(coco['images'])!=expected['images'] or len(coco['annotations'])!=expected['boxes']:
                raise ValueError('Original split counts differ')
            names=[n for n in archive.namelist() if not n.endswith('/')]
            desired={original_split+'/coco_annotations.json'}
            for image in coco['images']:
                stem=Path(image['file_name']).stem
                if image['file_name']!=stem+'.jpg' or not stem.isdigit():
                    raise ValueError('Unexpected source image name')
                desired.update(f'{original_split}/{kind}/{stem}{ext}' for kind,ext in
                               (('images','.jpg'),('labels','.txt'),('labels','.xml')))
            if set(names)!=desired or len(names)!=len(desired): raise ValueError('Unexpected/missing ZIP members')
            item=store('provenance/original_coco/'+original_split+'.json',raw_coco)
            manifest[item['path']]=item['sha256']
            for target in (('train','val') if original_split=='train' else ('test',)):
                ids={i['id'] for i in coco['images'] if ('val' if i['id'] in selected else original_split)==target}
                subset={**coco,'images':[i for i in coco['images'] if i['id'] in ids],
                        'annotations':[a for a in coco['annotations'] if a['image_id'] in ids]}
                # Keep original global image/annotation IDs and integer geometry.
                dump(output/'annotations_coco'/(target+'.json'),subset)
                (output/'splits'/(target+'.txt')).write_text(''.join(Path(i['file_name']).stem+'\n'
                    for i in sorted(subset['images'],key=lambda r:r['id'])),encoding='utf-8')
            (output/'splits'/('official_'+original_split+'.txt')).write_text(''.join(Path(i['file_name']).stem+'\n'
                for i in sorted(coco['images'],key=lambda r:r['id'])),encoding='utf-8')

            def process(image):
                stem=Path(image['file_name']).stem
                target='val' if image['id'] in selected else original_split
                image_data=archive.read(f'{original_split}/images/{stem}.jpg')
                label_data=archive.read(f'{original_split}/labels/{stem}.txt')
                xml_data=archive.read(f'{original_split}/labels/{stem}.xml')
                if not image_data.endswith(b'\xff\xd9'): raise ValueError('JPEG is missing end marker: '+stem)
                with Image.open(io.BytesIO(image_data)) as decoded:
                    if decoded.size!=(image['width'],image['height']) or decoded.getexif().get(274,1)!=1:
                        raise ValueError('Image dimensions/EXIF disagree with original annotations: '+stem)
                    if decoded.format!='JPEG' or decoded.mode!='RGB': raise ValueError('Unexpected image format/mode')
                    decoded.load()  # FULL pixel decoding; not header-only JPEG checking
                labels,error=validate_triplet(image,label_data,xml_data,groups[image['id']])
                files=[store(f'images/{target}/{stem}.jpg',image_data),
                       store(f'labels/{target}/{stem}.txt',label_data),
                       store(f'annotations_voc/{target}/{stem}.xml',xml_data)]
                cls=Counter(CLASSES[c] for c,_ in labels)
                gain=640/max(image['width'],image['height'])
                areas=[math.sqrt((b[2]-b[0])*(b[3]-b[1])) for _,b in labels]
                sample=dict(id=image['id'],stem=stem,original_split=original_split,split=target,
                            width=image['width'],height=image['height'],boxes=len(labels),classes=dict(cls),
                            image=files[0]['path'],image_sha256=files[0]['sha256'],
                            label=files[1]['path'],label_sha256=files[1]['sha256'],
                            xml=files[2]['path'],xml_sha256=files[2]['sha256'],
                            annotation_ids=[a['id'] for a in groups[image['id']]])
                return sample,files,cls,[v*gain for v in areas],areas,error

            ordered=sorted(coco['images'],key=lambda r:r['id'])
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for number,(sample,files,cls,scaled,areas,error) in enumerate(executor.map(process,ordered),1):
                    full_decode+=1; samples.append(sample); target=sample['split']; counts[target].update(cls)
                    sizes[target].extend(scaled); original_sizes[target].extend(areas)
                    backgrounds[target]+=sample['boxes']==0; portraits[target]+=sample['height']>sample['width']
                    max_corner=max(max_corner,error)
                    prior=jpeg_hashes.setdefault(sample['image_sha256'],sample['id'])
                    if prior!=sample['id']: raise ValueError('Exact duplicate original JPEG bytes: '+str((prior,sample['id'])))
                    for item in files: manifest[item['path']]=item['sha256']
                    if number%250==0 or number==len(ordered):
                        print(f'Prepare {original_split} {number}/{len(ordered)} FULL JPEG decode / CRC / labels '
                              f'elapsed={time.monotonic()-started:.1f}s',flush=True)
    for name,expected in DOC_MD5.items():
        data=(HERE/'official'/name).read_bytes()
        if hashlib.md5(data).hexdigest()!=expected: raise ValueError('Official source document checksum mismatch: '+name)
        store('provenance/official/'+name,data)
    shutil.copy2(HERE/'official/LICENSE.txt',output/'LICENSE.txt')
    shutil.copy2(HERE/'official/names.txt',output/'names.txt')
    shutil.copy2(HERE/'DATASET_README.md',output/'README.md')
    shutil.copy2(HERE/'data.yaml',output/'data.yaml')
    shutil.copy2(HERE/'verify_dataset.py',output/'verify_dataset.py')
    for path in HERE.glob('*.py'): shutil.copy2(path,output/'preparation_code'/path.name)
    (output/'provenance/samples.jsonl').write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in samples),encoding='utf-8')
    summary=dict(status='prepared',version=VERSION,source='https://zenodo.org/records/21728430',
        original_archive_hashes={s:audit['archives'][s+'.zip'] for s in ('train','test')},
        class_names=list(CLASSES),original_splits=OFFICIAL,split_seed=SPLIT_SEED,
        validation=dict(source='official train only',count=434,method='lowest SHA256 ranks',
                        exact_rank_string='songdo-v2-holdout:20261017:{original_integer_image_id}',
                        official_test_untouched=True,official_validation_set=False,group_disjoint_claim=False),
        images_unchanged=True,YOLO_labels_byte_preserved=True,no_annotation_filtering=True,
        full_image_decodes=full_decode,all_ZIP_entries_read_with_CRC=True,exact_duplicate_JPEGs=0,
        max_YOLO_vs_integer_corner_error_px=max_corner,
        geometry_note='YOLO subpixel labels preserved; original COCO/VOC share rounded integer corners, not regenerated from YOLO.',
        splits={s:dict(images=sum(r['split']==s for r in samples),boxes=sum(counts[s].values()),
                       background_images=backgrounds[s],portrait_images=portraits[s],
                       classes={n:counts[s][n] for n in CLASSES},
                       sqrt_box_area_at_input640=size_statistics(sizes[s]),
                       sqrt_box_area_original_pixels=size_statistics(original_sizes[s])) for s in ('train','val','test')},
        boundary='Fixed train-only frame holdout, not video/scene-disjoint. Original author test split is random frames. No performance/gain claim.',
        build_seconds=time.monotonic()-started)
    dump(output/'provenance/conversion_summary.json',summary)
    for path in output.rglob('*'):
        if path.is_file() and path.name not in ('SHA256SUMS.json','VERIFICATION.json','BUILD_FAILED.json'):
            rel=path.relative_to(output).as_posix()
            if rel not in manifest: manifest[rel]=file_sha(path)
    dump(output/'SHA256SUMS.json',dict(sorted(manifest.items())))
    print('PREPARED '+str(output),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4); a=p.parse_args()
    new_output=not a.output.exists()
    try:
        build(a.source_root.resolve(),a.output.resolve(),a.audit.resolve(),a.workers)
    except BaseException:
        if new_output and (a.output/'provenance/build_started.json').is_file():
            dump(a.output/'BUILD_FAILED.json',dict(status='failed',error=traceback.format_exc()))
        raise
