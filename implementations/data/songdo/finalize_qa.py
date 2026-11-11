"""Generate diagnostic bbox previews and seal supplemental QA/code files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
COLORS=('#39b9ff','#ff5252','#ffd740','#63e768')
CLASSES=('car','bus','truck','motorcycle')


def make_previews(root):
    samples=[json.loads(line) for line in (root/'provenance/samples.jsonl').read_text().splitlines()]
    chosen=[]
    for split in ('train','val','test'):
        chosen.append(next(r for r in samples if r['split']==split and r['width']>r['height']))
    chosen.extend(r for r in samples if r['height']>r['width'])
    try: font=ImageFont.load_default(size=16)
    except TypeError: font=ImageFont.load_default()
    canvas=Image.new('RGB',(1440,((len(chosen)+2)//3)*320),'#18232b'); draw=ImageDraw.Draw(canvas)
    for i,record in enumerate(chosen):
        with Image.open(root/record['image']) as original:
            tile=original.copy(); tile.thumbnail((472,284),Image.Resampling.LANCZOS)
        left=(i%3)*480+(480-tile.width)//2; top=(i//3)*320+27+(284-tile.height)//2
        canvas.paste(tile,(left,top)); sx=tile.width/record['width']; sy=tile.height/record['height']
        for row in (root/record['label']).read_text().splitlines():
            if not row.strip(): continue
            category,cx,cy,w,h=map(float,row.split())
            box=(left+(cx-w/2)*record['width']*sx,top+(cy-h/2)*record['height']*sy,
                 left+(cx+w/2)*record['width']*sx,top+(cy+h/2)*record['height']*sy)
            draw.rectangle(box,outline=COLORS[int(category)],width=1)
        draw.text(((i%3)*480+9,(i//3)*320+5),
                  f'{record["split"]} {record["stem"]} {record["width"]}x{record["height"]} boxes={record["boxes"]}',font=font,fill='white')
    canvas.save(root/'qa/bbox_overview.jpg',quality=92)
    # Close-ups of each class; deterministic first two occurrences, not
    # performance-driven example selection or edited training images.
    crops=Image.new('RGB',(1440,660),'#18232b'); cd=ImageDraw.Draw(crops)
    selected={i:[] for i in range(4)}
    for record in samples:
        if all(len(items)>=2 for items in selected.values()): break
        for row in (root/record['label']).read_text().splitlines():
            if not row.strip(): continue
            cat,cx,cy,w,h=map(float,row.split()); cat=int(cat)
            if len(selected[cat])>=2: continue
            if any(s[0]['id']==record['id'] for s in selected[cat]): continue
            selected[cat].append((record,cx,cy,w,h))
    for cat,items in selected.items():
        for j,(record,cx,cy,w,h) in enumerate(items):
            iw,ih=record['width'],record['height']; center=(cx*iw,cy*ih)
            side=max(w*iw,h*ih)*3.
            x1=max(0,center[0]-side/2); y1=max(0,center[1]-side/2)
            x2=min(iw,center[0]+side/2); y2=min(ih,center[1]+side/2)
            x1,y1,x2,y2=map(int,(x1,y1,x2,y2))
            with Image.open(root/record['image']) as original:
                crop=original.crop((x1,y1,x2,y2))
            crop.thumbnail((350,286),Image.Resampling.LANCZOS)
            left=cat*360+(360-crop.width)//2; top=j*330+31+(286-crop.height)//2
            crops.paste(crop,(left,top)); sx=crop.width/(x2-x1); sy=crop.height/(y2-y1)
            box=(left+((cx-w/2)*iw-x1)*sx,top+((cy-h/2)*ih-y1)*sy,
                 left+((cx+w/2)*iw-x1)*sx,top+((cy+h/2)*ih-y1)*sy)
            cd.rectangle(box,outline=COLORS[cat],width=2)
            cd.text((cat*360+9,j*330+7),f'{CLASSES[cat]} id={record["stem"]} ({record["split"]})',font=font,fill=COLORS[cat])
    crops.save(root/'qa/class_crops.jpg',quality=94)
    (root/'qa/PREVIEW_NOTE.txt').write_text('Diagnostic image/label correspondence previews only. Training JPEGs and labels are untouched.\n'
        'Colors: blue=car; red=bus; yellow=truck; green=motorcycle.\n'
        'First deterministic examples, not selected by model performance.\n',encoding='utf-8')


def seal(root):
    # Preserve all old sealed image/annotation hashes; only add newly-created
    # QA/provenance/code files. Fail instead of blessing changed original data.
    old=json.loads((root/'SHA256SUMS.json').read_text(encoding='utf-8'))
    for path in HERE.glob('*.py'):
        shutil.copy2(path,root/'preparation_code'/path.name)
    result={}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.name in ('SHA256SUMS.json','VERIFICATION.json') or '__pycache__' in path.parts:
            continue
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
        name=path.relative_to(root).as_posix(); actual=h.hexdigest()
        if name in old and name.startswith(('images/','labels/','annotations_','provenance/original_coco/','provenance/official/')):
            if actual!=old[name]: raise RuntimeError('Previously sealed source data changed: '+name)
        result[name]=actual
    (root/'SHA256SUMS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('Sealed files:',len(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    modes=p.add_mutually_exclusive_group(); modes.add_argument('--seal-only',action='store_true')
    modes.add_argument('--previews-only',action='store_true')
    a=p.parse_args()
    if not a.seal_only: make_previews(a.root.resolve())
    if not a.previews_only: seal(a.root.resolve())
