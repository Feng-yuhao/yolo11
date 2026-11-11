"""Original DET/ignore semantics; never score the legacy clipped YOLO labels."""
import hashlib
import json
from pathlib import Path
import re
from common import CLASSES, DATA_MANIFEST_SHA, now, read, require, sha, write


def definition(root):
    text = (root/'original_toolkit/utils/CalculateDetectionPR_overall.m').read_text()
    seqs = re.findall(r"'(M\d+)'", re.search(r'seqDirs\s*=\s*\{([^}]+)\}',text).group(1))
    lens = [int(v) for v in re.search(r'seqLens\s*=\s*\[([^]]+)\]',text).group(1).split(',')]
    ignored = re.findall(r"'(M\d+)'", re.search(r'pass\s*=\s*\{([^}]+)\}',text).group(1))
    require(len(seqs)==20 and sum(lens)==16592 and len(ignored)==18, 'Author test definition differs')
    return seqs,lens,ignored


def target_audit(root, full):
    require(sha(root/'manifest.json') == DATA_MANIFEST_SHA, 'Wrong test-only data manifest')
    entries = read(root/'manifest.json')['files']
    metadata = hashlib.sha256()
    for i,e in enumerate(entries,1):
        p = root/e['path']
        require(p.resolve().is_relative_to(root.resolve()) and not p.is_symlink(), 'Unsafe data path')
        st = p.stat()
        require(st.st_size == e['bytes'], 'Target size changed: '+e['path'])
        metadata.update(json.dumps([e['path'],st.st_size,st.st_mtime_ns]).encode())
        if full or not e['path'].startswith(('images/','labels/')):
            require(sha(p)==e['sha256'], 'Target SHA mismatch: '+e['path'])
        if full and i%3000==0: print(f'{now()} TARGET SHA {i}/{len(entries)}',flush=True)
    seqs,lens,ignored=definition(root)
    records=read(root/'test_images.json')
    expected={f'{s}_img{f:06d}.jpg' for s,n in zip(seqs,lens) for f in range(1,n+1)}
    require(len(records)==16592 and {Path(r['file_name']).name for r in records}==expected,'Incomplete author test split')
    require(all((r['width'],r['height'])==(1024,540) for r in records),'Unexpected native image dimensions')
    return dict(images=16592,boxes=375884,sequences=20,manifest_sha256=DATA_MANIFEST_SHA,
                test_file_stat_sha256=metadata.hexdigest(),test_images_byte_verified=full,
                target_train_or_val_used=False,ignore_sequences=ignored)


def records(root):
    result=[]
    for i,r in enumerate(read(root/'test_images.json'),1):
        p=Path(r['file_name'])
        require(p.parts[:2]==('images','test') and len(p.parts)==3,'Unsafe test image path')
        result.append(dict(id=i,width=r['width'],height=r['height'],path=root/p,
                           sequence=r['sequence'],frame=r['frame']))
    return result


def prepare_gt(root, destination):
    items=records(root); lookup={(r['sequence'],r['frame']):r for r in items}
    seqs,lens,ignore_seqs=definition(root)
    annotations=[]; ignored={}; by_image={r['id']:[] for r in items}
    for seq in seqs:
        for line in (root/f'original_toolkit/GT/{seq}_gt_whole.txt').read_text().splitlines():
            if not line.strip(): continue
            values=list(map(int,line.split(',')))
            require(len(values)==9,'Wrong DET GT format')
            frame,track,x,y,w,h,out,occ,cls=values
            require(cls in (1,2,3) and w>0 and h>0,'Invalid DET object')
            item=lookup[(seq,frame)]; box=[x,y,w,h]
            by_image[item['id']].append(box)
            annotations.append(dict(id=len(annotations)+1,image_id=item['id'],category_id=cls,bbox=box,
                                    area=w*h,iscrowd=0,out_of_view=out,occlusion=occ,track_id=track))
        if seq in ignore_seqs:
            for line in (root/f'original_toolkit/GT/{seq}_gt_ignore.txt').read_text().splitlines():
                if not line.strip(): continue
                v=list(map(int,line.split(','))); require(len(v)==9,'Wrong ignore format')
                frame,_,x,y,w,h,*_=v
                require(w>0 and h>0,'Invalid ignore area')
                ignored.setdefault(lookup[(seq,frame)]['id'],[]).append([x,y,w,h])
    require(len(annotations)==375884,'DET object coverage mismatch')
    gt=dict(info={'description':'Original UAVDT DET GT; custom COCO-style schema, not an official COCO release'},
            images=[dict(id=r['id'],file_name=r['path'].name,width=r['width'],height=r['height'],
                         sequence=r['sequence'],frame=r['frame']) for r in items],
            categories=[dict(id=i+1,name=n) for i,n in enumerate(CLASSES)],annotations=annotations)
    write(destination/'gt_coco_style.json',gt)
    write(destination/'ignore_by_image.json',ignored)
    return dict(gt_sha256=sha(destination/'gt_coco_style.json'),ignore_sha256=sha(destination/'ignore_by_image.json'),
                original_GT_boxes=375884,ignore_regions=sum(map(len,ignored.values())),
                geometry='Original unmodified continuous xywh; YOLO labels not used for scoring')


def filter_ignored(predictions, ignored):
    kept=[]; removed=[]
    for p in predictions:
        x,y,w,h=p['bbox']
        regions=ignored.get(str(p['image_id']),ignored.get(p['image_id'],[]))
        # Exact strict inequalities from the author's overall detection script.
        inside=any(x>rx and y>ry and x+w<rx+rw and y+h<ry+rh for rx,ry,rw,rh in regions)
        (removed if inside else kept).append(p)
    return kept,removed
