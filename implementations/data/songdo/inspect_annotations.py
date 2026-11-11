"""Audit every original label without changing coordinates or deleting boxes."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

CLASSES = ('car','bus','truck','motorcycle')


def inspect(root):
    all_ids=set(); all_ann_ids=set(); result={}
    for split in ('train','test'):
        counts=Counter(); edge_max=0.; corner_max=0.; bad=[]; duplicates=[]; rounding=Counter()
        with zipfile.ZipFile(root/(split+'.zip')) as source:
            coco=json.loads(source.read(split+'/coco_annotations.json'))
            groups=defaultdict(list)
            for ann in coco['annotations']:
                if ann['id'] in all_ann_ids: raise RuntimeError('Repeated global annotation ID')
                all_ann_ids.add(ann['id']); groups[ann['image_id']].append(ann)
            for image in coco['images']:
                if image['id'] in all_ids: raise RuntimeError('Repeated global image ID')
                all_ids.add(image['id'])
                stem=Path(image['file_name']).stem; w,h=image['width'],image['height']
                raw=source.read(f'{split}/labels/{stem}.txt').decode('utf-8-sig')
                rows=[list(map(float,line.split())) for line in raw.splitlines() if line.strip()]
                xml=ET.fromstring(source.read(f'{split}/labels/{stem}.xml'))
                objects=xml.findall('object')
                if len(rows)!=len(objects) or len(rows)!=len(groups[image['id']]):
                    raise RuntimeError('Format object counts differ: '+stem)
                seen=set()
                for index,(row,ann,obj) in enumerate(zip(rows,groups[image['id']],objects)):
                    if len(row)!=5 or not all(math.isfinite(v) for v in row): raise RuntimeError('Bad label')
                    cls,cx,cy,bw,bh=row
                    if cls!=int(cls) or not 0<=cls<4 or not(0<=cx<=1 and 0<=cy<=1 and 0<bw<=1 and 0<bh<=1):
                        raise RuntimeError('Bad normalized row: '+str(row))
                    if tuple(row) in seen: duplicates.append([split,stem,index])
                    seen.add(tuple(row))
                    corners=((cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h)
                    edge_max=max(edge_max,-corners[0],-corners[1],corners[2]-w,corners[3]-h)
                    left,top,width,height=ann['bbox']; icorners=(left,top,left+width,top+height)
                    xmlbox=tuple(int(obj.findtext('bndbox/'+v)) for v in ('xmin','ymin','xmax','ymax'))
                    if xmlbox!=icorners or obj.findtext('name')!=CLASSES[int(cls)] or ann['category_id']!=cls+1:
                        raise RuntimeError('Class/order/XML/COCO mismatch: '+stem)
                    error=max(abs(a-b) for a,b in zip(corners,icorners)); corner_max=max(corner_max,error)
                    if error>.505: bad.append([stem,index,error,corners,icorners])
                    for a,b in zip(corners,icorners): rounding[str(round(a)==b)]+=1
                    counts[CLASSES[int(cls)]]+=1
        result[split]=dict(images=len(coco['images']),boxes=sum(counts.values()),classes=dict(counts),
                           max_YOLO_vs_integer_corner_error_px=corner_max,max_edge_excess_px=edge_max,
                           exact_duplicate_rows=len(duplicates),duplicate_examples=duplicates[:5],
                           corner_errors_gt0p505px=len(bad),error_examples=bad[:4],python_round_matches=dict(rounding))
        print(json.dumps({split:result[split]},ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); result=inspect(a.root)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
