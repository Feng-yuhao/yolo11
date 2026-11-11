"""Portable CPU-only immutable payload, image and supervision audit."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def verify(root,decode=False):
    root=Path(root).resolve();manifest=json.loads((root/'SHA256SUMS.json').read_text(encoding='utf-8'));protocol=json.loads((root/'PROTOCOL.json').read_text(encoding='utf-8'))
    for n,d in manifest.items():
        p=root/n
        assert p.resolve().is_relative_to(root) and not p.is_symlink() and p.is_file(),n
        assert sha(p)==d,n
    gt=json.loads((root/protocol['canonical_annotation']).read_text(encoding='utf-8'))
    images={i['id']:i for i in gt['images']};assert len(images)==protocol['images']
    assert len({i['file_name'] for i in images.values()})==len(images)
    assert len({a['id'] for a in gt['annotations']})==len(gt['annotations'])
    positive=0;ignored=0;max_error=0
    from collections import defaultdict
    grouped=defaultdict(list)
    for a in gt['annotations']:
        im=images[a['image_id']];x,y,w,h=a['bbox']
        assert 0<=x and 0<=y and w>=0 and h>=0 and x+w<=im['width']+.001 and y+h<=im['height']+.001,a
        grouped[a['image_id']].append(a)
        if a.get('ignore',False):ignored+=1
        else:positive+=1;assert w>0 and h>0
    for i in images.values():
        path=root/i['file_name'];assert path.is_file()
        if decode:
            from PIL import Image
            with Image.open(path) as im:im.load();assert im.size==(i['width'],i['height'])
        labels=(root/'labels_positive_only/test'/(path.stem+'.txt')).read_text().splitlines()
        expected=[a for a in grouped[i['id']] if not a.get('ignore',False)]
        assert len(labels)==len(expected)
        for line,a in zip(labels,expected):
            c,cx,cy,w,h=map(float,line.split());x,y,bw,bh=a['bbox']
            assert c==0
            err=max(abs(cx*i['width']-(x+bw/2)),abs(cy*i['height']-(y+bh/2)),abs(w*i['width']-bw),abs(h*i['height']-bh))
            max_error=max(max_error,err);assert err<.00001
    assert positive==protocol['positive_boxes'] and ignored==protocol['ignored_entries']
    return dict(status='passed',root=str(root),files_SHA256_checked=len(manifest),images=len(images),
                positive_boxes=positive,ignored_entries=ignored,max_YOLO_inverse_error_px=max_error,
                full_pixel_decode=decode,torch_imported=False,training=False,evaluation=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--full-decode',action='store_true');a=p.parse_args()
    result=verify(a.root,a.full_decode)
    (a.root/'VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
