"""CPU-only bindings and fixed source-only protocol. No training entry point."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
REVISION='visdrone_to_dior_tinyperson_s123_source_only_v1'
SOURCE_CLASSES=('pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor')
SOURCE_KEYS=tuple(f'{kind}_seed{s}' for s in (1,2,3) for kind in ('e1','e60'))
TARGETS=('dior','tiny')
KEYS=tuple(t+'_'+k for t in TARGETS for k in SOURCE_KEYS)
PREFIX='VISDRONE_TO_DIOR_TINY_S123_SOURCE_ONLY_'
INDEX='latest_visdrone_to_dior_tiny_three_seeds_source_only.json'
LOCK='.visdrone_to_dior_tiny_three_seeds_source_only.lock'
CANCEL='KEEP_RUNNING_DIOR_TINY_S123'
PREDICT=dict(imgsz=640,conf=.001,iou=.7,max_det=500,half=False,augment=False,
             rect=False,agnostic_nms=False,save=False,save_txt=False,save_conf=False,verbose=False)
SPECS={
 'dior':dict(folder='DIOR_TestOnly_Vehicle_v1',images=11738,positive_boxes=26639,ignored_entries=1,
             category='vehicle',indices=[3,4,5,8],mapping='per-anchor max(car,van,truck,bus); other source classes discarded',
             GT='original official test HBB, XML corners minus1 continuous coordinates; all8430 vehicle-negative images retained',
             evaluator='standard pycocotools COCO AP .50:.05:.95/maxDets100; derived vehicle-only view, NOT DIOR20 benchmark'),
 'tiny':dict(folder='TinyPerson_TestOnly_Person_v1',images=786,positive_boxes=13687,ignored_entries=4821,
             category='person',indices=[0,1],mapping='per-anchor max(pedestrian,people); other source classes discarded',
             GT='original normal test786; sea/earth merged; ignore/uncertain/logo retained; annotation IDs offset+1 for COCO zero sentinel',
             evaluator='author TinyBenchmark v1 ignore+uncertain+IOD matcher; COCO-style .50:.05:.95/max100 plus native tiny AP25/50/75/max200')}
PROTOCOL=dict(id=REVISION,source='VisDrone2019 val-best',targets=SPECS,predict=PREDICT,batch=4,
              postprocess='semantic max before best-class single-label native NMS .7/top500; no added probabilities',
              preprocessing='unchanged 8.3.0 native LetterBox maximum side640; actual tensor shapes saved; no tiles',
              selection='six existing source val-best checkpoints, never target-selected',
              training=False,adaptation=False,target_BN_updates=False,execution='serial12',
              tiny_ignore_version='MIT author TinyBenchmark branch cocoeval.py SHA222b3173; IOD >= current IoU threshold; not later huicv transformed IOD threshold',
              storage='lossless gzip original predictions, no float rounding or prediction pruning',
              caveat='Declared common-category harmonization, not full-category benchmarks or target retraining. Three source seeds, not three independent target datasets.')

def require(ok,message):
    if not ok:raise RuntimeError(message)
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with tmp.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def split_key(key):
    target,source_key=key.split('_',1)
    require(target in TARGETS and source_key in SOURCE_KEYS,'Unknown fixed evaluation key')
    return target,source_key
def parser():
    p=argparse.ArgumentParser(description='VisDrone E1/E60 -> DIOR/TinyPerson three seeds, no training')
    p.add_argument('--project',type=Path,default=Path('/root/autodl-tmp/yolo11_project'))
    p.add_argument('--source',type=Path,default=Path('/root/autodl-tmp/yolo11_src'))
    p.add_argument('--data-parent',type=Path,default=Path('/root/autodl-tmp/project'))
    return p
def package_audit():
    m=read(HERE/'manifest.json');require(m['revision']==REVISION,'Wrong package revision')
    for n,d in m['files'].items():require(sha(HERE/n)==d,'Package changed: '+n)
    return sha(HERE/'manifest.json')
def source_audit(source):
    result={}
    for n,d in read(HERE/'references/source_bindings.json').items():
        actual=hashlib.sha256((source/'ultralytics'/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        require(n=='nn/tasks.py' or actual==d,'Pinned YOLO11 source differs: '+n);result[n]=actual
    return result
def weights_audit(project):
    specs=read(HERE/'references/checkpoints.json');require(set(specs)==set(SOURCE_KEYS),'Need all six source checkpoints')
    result={}
    for k,s in specs.items():
        p=project/s['relative_path'];require(p.is_file() and not p.is_symlink(),'Missing source best: '+str(p))
        require(sha(p)==s['sha256'],'Wrong/modified source checkpoint: '+k)
        result[k]=dict(s,path=str(p.resolve()),bytes=p.stat().st_size)
    return result
def boot(args,settings):
    import importlib.metadata as meta
    if not getattr(args,'local_engineering',False):
        require(sys.platform=='linux' and Path(sys.prefix).resolve()==Path('/root/miniconda3/envs/yolo11'),'Use yolo11 Python, never base/yolo26')
    os.environ.update(TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',YOLO_AUTOINSTALL='false',YOLO_VERBOSE='false',YOLO_CONFIG_DIR=str(settings))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.path.insert(0,str(args.source.resolve()))
    import torch,ultralytics,spatial_task_head
    spatial_task_head.configure('e60')
    require(ultralytics.__version__=='8.3.0' and Path(ultralytics.__file__).resolve().parent==args.source.resolve()/'ultralytics','Wrong YOLO11 source/version')
    if not getattr(args,'local_engineering',False):require(torch.__version__=='2.5.1+cu121' and torch.version.cuda=='12.1','Original server Torch/CUDA required')
    require(torch.cuda.is_available(),'CUDA unavailable: use GPU mode, no CPU fallback')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(1,deterministic=True)
    return dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
                ultralytics=ultralytics.__version__,source=source_audit(args.source),
                dependencies={k:meta.version(k) for k in ('numpy','Pillow','pycocotools','torchvision')},
                local_engineering=getattr(args,'local_engineering',False))
