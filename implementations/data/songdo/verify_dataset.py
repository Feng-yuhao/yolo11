"""Independent CPU verifier; stdlib+Pillow, no torch/GPU/network required."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import time
import xml.etree.ElementTree as ET

CLASSES=('car','bus','truck','motorcycle')
COUNTS={'train':3901,'val':434,'test':1084}


def require(value, message):
    if not value: raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def verify(root, decode_images=False):
    from PIL import Image
    started=time.monotonic(); root=root.resolve()
    require(not(root/'BUILD_FAILED.json').exists(),'Build has a failure marker; do not train')
    checks=json.loads((root/'SHA256SUMS.json').read_text(encoding='utf-8'))
    for index,(name,expected) in enumerate(checks.items(),1):
        relative=PurePosixPath(name)
        require(not relative.is_absolute() and '..' not in relative.parts and '\\' not in name,
                'Unsafe checksum path')
        path=root/name
        require(path.resolve().is_relative_to(root) and not path.is_symlink(), 'Escaping/symlink file')
        require(path.is_file() and sha(path)==expected,'SHA256 mismatch: '+name)
        if index%1500==0 or index==len(checks): print(f'Verify SHA256 {index}/{len(checks)}',flush=True)
    cfg=(root/'data.yaml').read_text(encoding='utf-8').splitlines()
    require(not any(line.startswith('path:') for line in cfg),'Nonportable YAML path override')
    for line in ('train: images/train','val: images/val','test: images/test','nc: 4'):
        require(line in cfg,'Incorrect data YAML: '+line)
    for category,name in enumerate(CLASSES): require(f'  {category}: {name}' in cfg,'Wrong class mapping')
    summary=json.loads((root/'provenance/conversion_summary.json').read_text(encoding='utf-8'))
    samples=[json.loads(line) for line in (root/'provenance/samples.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(samples)==5419 and len({r['id'] for r in samples})==5419,'Global image count/uniqueness mismatch')
    source={s:json.loads((root/'provenance/original_coco'/(s+'.json')).read_text(encoding='utf-8')) for s in ('train','test')}
    original_images={s:{r['id']:r for r in v['images']} for s,v in source.items()}
    original_ann={s:{r['id']:r for r in v['annotations']} for s,v in source.items()}
    require(len(original_images['train'])==4335 and len(original_images['test'])==1084,'Original counts differ')
    require(not(original_images['train'].keys()&original_images['test'].keys()),'Official IDs overlap')
    require(not(original_ann['train'].keys()&original_ann['test'].keys()),'Global annotation IDs overlap')
    require(len(original_ann['train'])==218662 and len(original_ann['test'])==55528,'Original box counts differ')
    # Independent recomputation of the predeclared holdout, not converter imports.
    ranked=sorted(original_images['train'],key=lambda ident:
                  hashlib.sha256(('songdo-v2-holdout:20261017:'+str(ident)).encode('utf-8')).digest())
    holdout=set(ranked[:434]); actual_images={}; actual_annotations={}
    all_hashes={}; checked={}; max_error=0.; portrait_count=0
    for split,count in COUNTS.items():
        records=[r for r in samples if r['split']==split]; ids={r['id'] for r in records}
        expected_ids=(holdout if split=='val' else set(original_images['train'])-holdout if split=='train'
                      else set(original_images['test']))
        require(ids==expected_ids and len(records)==count,'Changed/overlapping train-val-test split: '+split)
        original_split='test' if split=='test' else 'train'
        ids_txt=(root/'splits'/(split+'.txt')).read_text().splitlines()
        require(len(ids_txt)==count and set(ids_txt)=={r['stem'] for r in records},'Split list mismatch')
        for parent,suffix in (('images','.jpg'),('labels','.txt'),('annotations_voc','.xml')):
            require({p.stem for p in (root/parent/split).glob('*'+suffix)}=={r['stem'] for r in records},
                    'Image/label/XML pairing differs: '+split+'/'+parent)
        coco=json.loads((root/'annotations_coco'/(split+'.json')).read_text(encoding='utf-8'))
        require(coco['categories']==source[original_split]['categories'],'Original categories changed')
        actual_images[split]={r['id']:r for r in coco['images']}
        actual_annotations[split]={r['id']:r for r in coco['annotations']}
        require(len(actual_images[split])==len(coco['images'])==count and set(actual_images[split])==ids,
                'Derived COCO image count/IDs differ')
        require(len(actual_annotations[split])==len(coco['annotations']),'Repeated derived annotation IDs')
        for ident,image in actual_images[split].items():
            require(image==original_images[original_split][ident],'Original COCO image record changed')
        for ident,ann in actual_annotations[split].items():
            require(ann['image_id'] in ids and ann==original_ann[original_split][ident],
                    'Dropped/changed/invented original COCO annotation')
        grouped={ident:[] for ident in ids}
        for ann in coco['annotations']: grouped[ann['image_id']].append(ann)
        categories=Counter(); empty=0; portraits=0; boxes=0
        for number,record in enumerate(records,1):
            require(record['original_split']==original_split,'Official test moved into train/val')
            image=actual_images[split][record['id']]; width,height=image['width'],image['height']
            require((record['width'],record['height'])==(width,height),'Sample dimensions differ')
            require((width,height) in ((3840,2160),(2160,3840)),'Unexpected full-resolution image size')
            require(record['image']==f'images/{split}/{record["stem"]}.jpg' and
                    record['label']==f'labels/{split}/{record["stem"]}.txt' and
                    record['xml']==f'annotations_voc/{split}/{record["stem"]}.xml','Wrong sample paths')
            for field in ('image','label','xml'):
                require(checks[record[field]]==record[field+'_sha256'],'Source bytes not preserved in manifest')
            require(record['image_sha256'] not in all_hashes,'Duplicate exact image bytes')
            all_hashes[record['image_sha256']]=split
            with Image.open(root/record['image']) as decoded:
                require(decoded.size==(width,height) and decoded.getexif().get(274,1)==1,
                        'Actual image dimensions/EXIF differ')
                decoded.load() if decode_images else decoded.verify()
            portrait_count+=height>width; portraits+=height>width
            xml=ET.parse(root/record['xml']).getroot()
            require((int(xml.findtext('size/width')),int(xml.findtext('size/height')))==(width,height),
                    'Original XML dimension mismatch')
            require(Path(xml.findtext('filename').replace('\\','/')).name==image['file_name'],'Original XML filename mismatch')
            lines=[line for line in (root/record['label']).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
            require(len(lines)==len(xml.findall('object'))==len(grouped[record['id']])==record['boxes'],
                    'Object count differs between formats')
            require([a['id'] for a in grouped[record['id']]]==record['annotation_ids'],'Annotation order/IDs drift')
            seen=set(); class_counts=Counter()
            for line,obj,ann in zip(lines,xml.findall('object'),grouped[record['id']]):
                fields=line.split(); require(len(fields)==5,'YOLO row length invalid')
                category=int(fields[0]); vals=[float(v) for v in fields[1:]]
                require(str(category)==fields[0] and 0<=category<4 and all(math.isfinite(v) for v in vals),'Bad YOLO row')
                cx,cy,bw,bh=vals
                require(0<=cx<=1 and 0<=cy<=1 and 0<bw<=1 and 0<bh<=1,'Bad normalized box')
                corners=(cx*width-bw*width/2,cy*height-bh*height/2,cx*width+bw*width/2,cy*height+bh*height/2)
                require(corners[0]>=-1e-8 and corners[1]>=-1e-8 and corners[2]<=width+1e-8 and
                        corners[3]<=height+1e-8,'YOLO box out of image')
                key=(category,*vals); require(key not in seen,'Duplicate YOLO object'); seen.add(key)
                x,y,w,h=ann['bbox']; integer=(x,y,x+w,y+h)
                xml_box=tuple(int(obj.findtext('bndbox/'+k)) for k in ('xmin','ymin','xmax','ymax'))
                require(xml_box==integer and obj.findtext('name')==CLASSES[category] and
                        ann['category_id']==category+1,'Category/XML/COCO correspondence differs')
                require(w>0 and h>0 and ann['area']==w*h and ann['iscrowd']==0,'COCO area/crowd mismatch')
                error=max(abs(a-b) for a,b in zip(corners,integer)); max_error=max(max_error,error)
                require(error<=.5+.75e-6*max(width,height)+1e-8,'Subpixel/integer rounding contract differs')
                categories[CLASSES[category]]+=1; class_counts[CLASSES[category]]+=1
            require(dict(class_counts)==record['classes'],'Per-image class count drift')
            boxes+=len(lines); empty+=not lines
            if number%500==0 or number==count: print(f'Verify labels {split} {number}/{count}',flush=True)
        require(boxes==summary['splits'][split]['boxes'] and
                {n:categories[n] for n in CLASSES}==summary['splits'][split]['classes'],'Summary counts differ')
        checked[split]=dict(images=count,boxes=boxes,classes={n:categories[n] for n in CLASSES},
                            background_images=empty,portrait_images=portraits)
    require(actual_images['train']|actual_images['val']==original_images['train'],'Original train images not fully preserved')
    require(actual_annotations['train']|actual_annotations['val']==original_ann['train'],'Original train annotations not fully preserved')
    require(actual_images['test']==original_images['test'] and actual_annotations['test']==original_ann['test'],
            'Official test membership/annotations changed')
    require(portrait_count==8 and sum(r['boxes'] for r in checked.values())==274190,'Full dataset count mismatch')
    return dict(status='passed',root=str(root),splits=checked,full_decode_this_verification=decode_images,
                full_decodes_during_build=summary['full_image_decodes'],SHA256_checked_files=len(checks),
                global_image_ids=5419,global_annotation_ids=274190,portrait_images=portrait_count,
                max_YOLO_vs_integer_corner_error_px=max_error,original_test_untouched=True,
                train_only_holdout_recomputed=True,all_original_GT_objects_preserved=True,
                original_JPEG_YOLO_XML_byte_hashes=True,torch_imported=False,seconds=time.monotonic()-started)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--decode-images',action='store_true'); a=p.parse_args()
    try:
        result=verify(a.root,a.decode_images)
    except BaseException:
        import traceback
        (a.root/'VERIFICATION.json').write_text(json.dumps(dict(status='failed',error=traceback.format_exc()),indent=2)+'\n',encoding='utf-8')
        raise
    (a.root/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
