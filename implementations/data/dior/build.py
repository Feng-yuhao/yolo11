"""Build test-only portable datasets, preserving JPEG/source annotation bytes."""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from PIL import Image
from verify_dataset import sha,verify

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
RAW=ROOT/'PAPER_EXPERIMENTS/datasets/raw_targets_20261023'
OUT=ROOT/'PAPER_EXPERIMENTS/datasets'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def put(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(data)
def image(path,data,expected):
    with Image.open(io.BytesIO(data)) as im:
        im.load();assert im.size==expected,(path,im.size,expected)
        assert im.getexif().get(274,1)==1,'Unresolved EXIF orientation'
    put(path,data)
def label_file(root,stem,anns,width,height):
    lines=[]
    for a in anns:
        if a.get('ignore',False):continue
        x,y,w,h=a['bbox'];assert w>0 and h>0
        lines.append(f'0 {(x+w/2)/width:.12f} {(y+h/2)/height:.12f} {w/width:.12f} {h/height:.12f}')
    p=root/'labels_positive_only/test'/(stem+'.txt');p.parent.mkdir(parents=True,exist_ok=True);p.write_text('\n'.join(lines)+('\n' if lines else ''),encoding='ascii')
def finalize(root,gt,protocol,readme):
    write(root/protocol['canonical_annotation'],gt);write(root/'PROTOCOL.json',protocol)
    (root/'README.md').write_text(readme,encoding='utf-8')
    shutil.copy2(HERE/'verify_dataset.py',root/'verify_dataset.py')
    (root/'preparation_code').mkdir(exist_ok=True);shutil.copy2(HERE/'build.py',root/'preparation_code/build.py')
    shutil.copy2(HERE/'verify_dataset.py',root/'preparation_code/verify_dataset.py')
    write(root/'SHA256SUMS.json',{p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('SHA256SUMS.json','VERIFICATION.json')})
    result=verify(root,decode=False);result['full_decodes_during_build']=protocol['images']
    result['payload_bytes']=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    write(root/'VERIFICATION.json',result)
    # A separate archive is retained locally for easy backup. Server install streams it;
    # the remote host need not hold an archive and an extracted copy simultaneously.
    archive=ROOT/'PAPER_EXPERIMENTS/deliverables'/(root.name+'.tar')
    with tarfile.open(archive,'x',format=tarfile.PAX_FORMAT) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file():z.add(p,arcname=root.name+'/'+p.relative_to(root).as_posix(),recursive=False)
    result['archive']=str(archive);result['archive_bytes']=archive.stat().st_size;result['archive_sha256']=sha(archive)
    write(ROOT/'PAPER_EXPERIMENTS/deliverables'/(root.name+'.receipt.json'),result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

def dior():
    root=OUT/'DIOR_TestOnly_Vehicle_v1';root.mkdir(exist_ok=False)
    with zipfile.ZipFile(RAW/'DIOR/ImageSets.zip') as z:
        split=z.read(next(n for n in z.namelist() if n.endswith('/test.txt')))
    ids=split.decode().split();assert len(ids)==len(set(ids))==11738
    put(root/'splits/test.txt',split)
    gt=dict(info=dict(description='DIOR official test, vehicle-only continuous HBB'),licenses=[],
            categories=[dict(id=1,name='vehicle')],images=[],annotations=[])
    invalid=[];all_counts=Counter();source_hashes={};grouped=defaultdict(list)
    with zipfile.ZipFile(RAW/'DIOR/Annotations.zip') as z:
        for stem in ids:
            raw=z.read('Annotations/Horizontal Bounding Boxes/'+stem+'.xml');r=ET.fromstring(raw)
            put(root/'provenance/original_HBB_XML'/(stem+'.xml'),raw)
            w,h=int(r.findtext('size/width')),int(r.findtext('size/height'));assert (w,h)==(800,800)
            gt['images'].append(dict(id=int(stem),file_name='images/test/'+stem+'.jpg',width=w,height=h))
            for index,o in enumerate(r.findall('object')):
                name=o.findtext('name').strip();all_counts[name]+=1
                if name!='vehicle':continue
                node=o.find('bndbox');xy=[float(node.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]
                assert all(1<=v<=800 for v in xy),[stem,xy]
                x1,y1,x2,y2=[v-1 for v in xy];bw,bh=x2-x1,y2-y1;assert bw>=0 and bh>=0
                difficult=int(o.findtext('difficult','0'));ignored=bool(difficult or bw==0 or bh==0)
                a=dict(id=len(gt['annotations'])+1,image_id=int(stem),category_id=1,bbox=[x1,y1,bw,bh],
                       area=bw*bh,iscrowd=int(ignored),ignore=ignored,source_xml_xyxy=xy,source_object_index=index,source_difficult=difficult)
                if bw==0 or bh==0:invalid.append(dict(image_id=int(stem),object_index=index,source_xml_xyxy=xy,policy='retain raw XML and zero-area ignored canonical entry; no fabricated box'))
                gt['annotations'].append(a);grouped[int(stem)].append(a)
            label_file(root,stem,grouped[int(stem)],w,h)
    positive=sum(not a['ignore'] for a in gt['annotations']);assert len(gt['annotations'])==26640 and positive==26639
    with zipfile.ZipFile(RAW/'DIOR/JPEGImages-test.zip') as z:
        for index,stem in enumerate(ids,1):
            raw=z.read('JPEGImages-test/'+stem+'.jpg');image(root/'images/test'/(stem+'.jpg'),raw,(800,800))
            source_hashes[stem]=hashlib.sha256(raw).hexdigest()
            if index%1000==0:print('DIOR full decode/copy',index,'/',len(ids),flush=True)
    write(root/'provenance/original_image_SHA256.json',source_hashes)
    write(root/'provenance/invalid_source_boxes.json',invalid)
    write(root/'provenance/all20_original_class_counts.json',dict(all_counts))
    write(root/'provenance/source_archives.json',{n:json.loads((RAW/'DIOR'/(n+'.receipt.json')).read_text()) for n in ('Annotations.zip','ImageSets.zip','JPEGImages-test.zip')})
    protocol=dict(id='DIOR_OFFICIAL_TEST_VEHICLE_XML_MINUS1_CONTINUOUS_v1',images=11738,positive_boxes=26639,ignored_entries=1,
        original_vehicle_entries=26640,canonical_annotation='annotations_coco/test_vehicle.json',
        source='https://gcheng-nwpu.github.io/#Datasets',split='original official test; not validation',
        image_policy='original JPEG bytes; no crop, resize or re-encoding',
        class_policy='vehicle-only derived evaluation view; full20-category original test XML retained',
        geometry='All XML x/y corners minus1, width=xmax-xmin, height=ymax-ymin; no +1 width. Matches MMDetection2.28.2 XMLDataset continuous coordinate representation, not a claim to execute original DIOR VOC evaluator.',
        geometry_reference='https://raw.githubusercontent.com/open-mmlab/mmdetection/v2.28.2/mmdet/datasets/xml_style.py',
        zero_area='One original vehicle entry at image15906 is retained with ignore/iscrowd1, no fabricated size; excluded from positive YOLO labels.',
        ordinary_YOLO_val_is_not_authoritative=True,training=False)
    finalize(root,gt,protocol,'# DIOR official test / vehicle-only evaluation view\n\n11,738 original test images; all20-class original HBB XML preserved. 26,640 original vehicle entries, including one zero-area entry (image15906). It is explicitly retained as ignored, leaving26,639 valid positives.\n\nOriginal image bytes unchanged. COCO and normalized YOLO positive-only labels use the documented XML-minus1 continuous convention in PROTOCOL.json; do not silently add VOC inclusive widths. These are not full20-class benchmark scores.\n\nlabels_positive_only/test is a geometric interoperability view. PROTOCOL.json and annotations_coco/test_vehicle.json are authoritative; a plain YOLO val run does not prove the intended protocol. No training YAML is supplied.\n\nCPU verification: python verify_dataset.py --full-decode\n')

def tiny():
    root=OUT/'TinyPerson_TestOnly_Person_v1';root.mkdir(exist_ok=False)
    source=json.loads((RAW/'TinyPerson/tiny_set_test.json').read_text());assert len(source['images'])==786
    for n in ('tiny_set_test.json','tiny_set_test_with_dense.json'):
        put(root/'provenance'/n,(RAW/'TinyPerson'/n).read_bytes())
    gt=copy.deepcopy(source);gt['info']=dict(description='TinyPerson original normal test; merged person; author ignore semantics required')
    gt['licenses']=[];gt['categories']=[dict(id=1,name='person')]
    for im in gt['images']:
        im['source_file_name']=im['file_name'];im['file_name']='images/test/'+Path(im['file_name']).name
    assert len({i['file_name'] for i in gt['images']})==786
    grouped=defaultdict(list)
    for a in gt['annotations']:
        a['source_category_id']=a['category_id'];a['category_id']=1
        a['source_ignore']=a.get('ignore',False)
        a['ignore']=bool(a.get('ignore') or a.get('uncertain') or a.get('logo'))
        grouped[a['image_id']].append(a)
    for im in gt['images']:label_file(root,Path(im['file_name']).stem,grouped[im['id']],im['width'],im['height'])
    hashes={}
    with tarfile.open(RAW/'TinyPerson/test.tar.gz','r:gz') as z:
        members={m.name.removeprefix('./'):m for m in z.getmembers() if m.isfile()}
        for index,im in enumerate(gt['images'],1):
            raw=z.extractfile(members['test/'+im['source_file_name']]).read()
            image(root/im['file_name'],raw,(im['width'],im['height']));hashes[im['source_file_name']]=hashlib.sha256(raw).hexdigest()
            if index%200==0:print('Tiny full decode/copy',index,'/',786,flush=True)
    put(root/'splits/test.txt',('\n'.join(str(i['id']) for i in gt['images'])+'\n').encode())
    write(root/'provenance/original_image_SHA256.json',hashes)
    positive=sum(not a['ignore'] for a in gt['annotations']);ignored=len(gt['annotations'])-positive
    assert (positive,ignored)==(13687,4821)
    protocol=dict(id='TINYPERSON_ORIGINAL_TEST786_AUTHOR_IGNORE_PERSON_v1',images=786,positive_boxes=positive,ignored_entries=ignored,
        canonical_annotation='annotations/test_person_with_ignore.json',
        source='https://github.com/ucas-vg/PointTinyBenchmark/blob/master/dataset/TinyPerson_annotation.md',
        image_policy='only original normal-test786 image bytes; no erased images, resize, crop or dense-test images',
        geometry='original continuous COCO xywh, areas, image/annotation IDs unchanged; source-category retained; no ±1 shift',
        ignore_policy='ignore OR uncertain OR logo preserved as ignored; do NOT erase images or turn ignored boxes into positives; native TinyPerson evaluator needed',
        compatibility_warning='COCO-shaped annotation is not safe for vanilla COCOeval or YOLO val because their ignore behavior differs. Positive-only YOLO labels are not complete evaluation supervision.',
        documentation_count_discrepancy='Actual downloaded original JSON contains13687 positives by its flags, not13787 quoted in older README. No objects invented or removed to force old prose totals.',
        class_policy='sea/earth merged to person before evaluation; original two-class JSON retained unchanged',training=False)
    finalize(root,gt,protocol,'# TinyPerson original test786 / person\n\nOnly786 normal-test original images are included. Both original JSON versions are preserved under provenance; dense-test images are not installed. Images are not erased, recolored, cropped or re-encoded.\n\nThe original test JSON has18,508 entries:13,687 positive and4,821 ignored by original flags. An older author README quotes13,787 positives; this package follows the downloaded JSON, not a forced prose count.\n\nannotations/test_person_with_ignore.json merges sea/earth into person while preserving IDs, boxes and all ignore/uncertain/logo metadata. Native TinyPerson ignore handling is REQUIRED. Do not use ordinary YOLO val or vanilla COCOeval as equivalent scoring. labels_positive_only/test contains positives for coordinate interoperability only; it cannot represent ignore regions. No training YAML is supplied.\n\nCPU verification: python verify_dataset.py --full-decode\n')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=('tiny','dior','finish_tiny'));a=p.parse_args()
    if a.dataset=='finish_tiny':
        root=OUT/'TinyPerson_TestOnly_Person_v1';protocol=json.loads((root/'PROTOCOL.json').read_text(encoding='utf-8'))
        gt=json.loads((root/protocol['canonical_annotation']).read_text(encoding='utf-8'))
        finalize(root,gt,protocol,(root/'README.md').read_text(encoding='utf-8'))
    else:(tiny if a.dataset=='tiny' else dior)()
