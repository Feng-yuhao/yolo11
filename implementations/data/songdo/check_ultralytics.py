"""Read-only actual YOLO11 data loader compatibility, no model/GPU/training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True); p.add_argument('--settings',type=Path,required=True)
    a=p.parse_args(); root=a.root.resolve()
    os.environ['YOLO_AUTOINSTALL']='false'; os.environ['YOLO_CONFIG_DIR']=str(a.settings.resolve())
    sys.path.insert(0,str(a.source.resolve()))
    import ultralytics
    from ultralytics.data import utils
    from PIL import Image
    if Path(ultralytics.__file__).resolve().parent!=a.source.resolve()/'ultralytics':
        raise RuntimeError('Wrong Ultralytics source')
    samples=[json.loads(line) for line in (root/'provenance/samples.jsonl').read_text().splitlines()]
    def refuse_save(*args,**kwargs): raise RuntimeError('QA forbids image repair/rewriting')
    with patch.object(utils,'check_font',lambda *args,**kwargs:None), patch.object(Image.Image,'save',refuse_save):
        resolved=utils.check_det_dataset(str(root/'data.yaml'),autodownload=False)
        for split in ('train','val','test'):
            if Path(resolved[split]).resolve()!=root/'images'/split: raise RuntimeError('YAML resolved wrong root')
        def check(row):
            image,label=root/row['image'],root/row['label']
            if utils.img2label_paths([str(image)])[0]!=str(label): raise RuntimeError('Wrong label pairing')
            result=utils.verify_image_label((str(image),str(label),'Songdo QA: ',False,4,0,0))
            filename,labels,shape,_,_,missing,found,empty,corrupt,message=result
            if not filename or missing or found!=1 or corrupt or message:
                raise RuntimeError('Native verify_image_label failed: '+str((image,message)))
            if len(labels)!=row['boxes'] or tuple(shape)!=(row['height'],row['width']):
                raise RuntimeError('Native YOLO dropped labels or changed portrait dimensions')
            if len(labels) and (labels[:,0].max()>=4 or labels[:,0].min()<0): raise RuntimeError('Wrong native classes')
            return len(labels)
        total=0
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i,n in enumerate(pool.map(check,samples),1):
                total+=n
                if i%500==0 or i==len(samples): print(f'Native YOLO data QA {i}/{len(samples)} boxes={total}',flush=True)
    result=dict(status='passed',ultralytics=ultralytics.__version__,images=len(samples),boxes=total,
                resolved_splits={s:str(resolved[s]) for s in ('train','val','test')},
                all_native_verify_image_label_passed=True,no_auto_repairs=True,
                font_download_only_disabled=True,training_started=False,GPU_used=False)
    (root/'provenance/ULTRALYTICS_COMPATIBILITY.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
