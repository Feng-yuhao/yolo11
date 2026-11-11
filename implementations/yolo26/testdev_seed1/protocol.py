"""Fixed paired protocol; receipts bind source, package, data and device layout."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from datetime import datetime
import common as c

NAMES = ['pedestrian','people','bicycle','car','van','truck','tricycle',
         'awning-tricycle','bus','motor']
DATA = Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019.yaml')
N_SHA = '9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef'
VAL_SHA = '058c87e779a61a09933bebe35af1c05c807a444dc842b4be12124dd36134c787'
CAP_GIB = 28


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def options(args, name):
    return dict(model=str(args.project/'weights/yolo26s.pt'), data=str(args.data),
                project=str(args.project/'runs'), name=name, exist_ok=False,
                task='detect', epochs=200, imgsz=640, batch=16, device='cuda:0',
                workers=8, seed=1, deterministic=True, patience=0, optimizer='SGD',
                lr0=.01, lrf=.01, momentum=.937, weight_decay=.0005,
                amp=True, cos_lr=True, close_mosaic=10, max_det=500,
                nms=True, conf=.001, iou=.7, val=True, plots=True,
                save=True, save_period=-1, resume=False, pretrained=True,
                cache=False, rect=False, multi_scale=0., fraction=1., nbs=64,
                warmup_epochs=3., warmup_momentum=.8, warmup_bias_lr=.1,
                box=7.5, cls=.5, dfl=1.5, cls_pw=0., channels_last=False,
                quantize=None, compile=False, hsv_h=.015, hsv_s=.7, hsv_v=.4,
                degrees=0., translate=.1, scale=.5, shear=0., perspective=0.,
                flipud=0., fliplr=.5, mosaic=1., mixup=0., copy_paste=0.)


def run_name(variant):
    return 'yolo26s_p2_' + ('baseline' if variant=='baseline' else 'e60') + '_img640_seed1_200e'


def deploy_amp_weight(args):
    src = c.HERE/'weights/yolo26n.pt'
    if c.sha256(src) != N_SHA:
        raise RuntimeError('Bundled official AMP checkpoint SHA mismatch')
    target = args.project/'weights/yolo26n.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and c.sha256(target) != N_SHA:
        raise RuntimeError('Will not overwrite a different existing yolo26n.pt')
    if not target.exists():
        shutil.copyfile(src,target)
    from ultralytics.utils import SETTINGS
    SETTINGS.update({'weights_dir':str(target.parent), 'runs_dir':str(args.project/'runs')})
    import ultralytics.utils as utils
    utils.WEIGHTS_DIR=target.parent  # check_amp imports this constant at call time


def engineering_receipt(args):
    # The original preflight is a prerequisite, not a substitute for real smoke.
    preflight = args.project/'experiment_code/YOLO26_P2_E60_Preflight_AutoDL'
    for name in ('e60_adapter.py','models26.py','checks26.py','source_sha256.json'):
        if c.sha256(preflight/name) != c.sha256(c.HERE/name):
            raise RuntimeError('Frozen preflight implementation differs: '+name)
    for path in sorted((args.project/'comparison_reports').glob('YOLO26_P2_E60_ENGINEERING_*/check.json'),reverse=True):
        result = json.loads(path.read_text(encoding='utf-8'))
        if (result.get('status')=='engineering_passed' and result.get('cuda_executed')
                and result.get('local_engineering') is False
                and all(result[k].get('actual_update_groups')==52 for k in ('amp_updates','fp32_updates'))):
            return dict(path=str(path),sha256=c.sha256(path))
    raise RuntimeError('No completed server CUDA engineering check; do not bypass it')


def dataset_audit(args):
    import numpy as np
    from PIL import Image
    from ultralytics.data.utils import check_det_dataset, img2label_paths
    import d27_official as official
    data = check_det_dataset(str(args.data),autodownload=False)
    names = data['names']
    names = [names[i] for i in range(len(names))] if isinstance(names,dict) else names
    if names != NAMES:
        raise RuntimeError('Wrong VisDrone class mapping')
    result = {}; image_sets = {}
    for split, expected in (('train',6471),('val',548)):
        directory = Path(data[split])
        paths = sorted(p for p in directory.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'} and p.is_file())
        if not args.local_test and len(paths)!=expected:
            raise RuntimeError(f'{split}: wrong image count {len(paths)}')
        image_sets[split] = {p.stem for p in paths}
        digest = hashlib.sha256(); instances = 0; duplicates = 0
        for index,(image,label) in enumerate(zip(paths,img2label_paths([str(p) for p in paths])),1):
            label = Path(label)
            with Image.open(image) as im:
                if im.getexif().get(274,1) not in (None,1):
                    raise RuntimeError('EXIF rotated image: '+str(image))
                dimensions = im.size
            content = label.read_bytes()
            rows = []
            for line in content.decode('utf-8-sig').splitlines():
                if not line.strip(): continue
                row = tuple(float(v) for v in line.split())
                if (len(row)!=5 or not all(math.isfinite(v) for v in row) or
                        row[0]!=int(row[0]) or not 0<=row[0]<10 or
                        not all(0<=v<=1 for v in row[1:3]) or not all(0<v<=1 for v in row[3:])):
                    raise RuntimeError('Invalid YOLO training/val label: '+str(label))
                rows.append(row)
            instances += len(set(rows)); duplicates += len(rows)-len(set(rows))
            digest.update(image.name.encode()+b'\0'+c.sha256(image).encode()+b'\0'+content+b'\0'+str(dimensions).encode())
            if index%1500==0: print(f'{now()} DATA AUDIT {split} {index}/{len(paths)}',flush=True)
        result[split] = dict(images=len(paths),unique_instances=instances,duplicate_rows=duplicates,sha256=digest.hexdigest())
    if image_sets['train'] & image_sets['val']:
        raise RuntimeError('Image IDs overlap between train and val')
    if not args.local_test and result['val']['unique_instances']!=38759:
        raise RuntimeError('Converted validation labels differ from E1/E60 protocol')
    records, annotation_meta = official.load_dataset(args.val_dataset)
    if not args.local_test and annotation_meta['annotation_dimension_sha256']!=VAL_SHA:
        raise RuntimeError('Original val annotations differ from frozen E60 scoring')
    # Independent original-annotation -> YOLO inverse-box check, without ignore filtering.
    labels = img2label_paths([str(r.image_path) for r in records])
    for record,label in zip(records,labels):
        expected=[]
        for row in record.raw_gt:
            if row[4]>0 and 1<=row[5]<=10 and row[2]>0 and row[3]>0:
                x,y,w,h = row[:4]
                expected.append((row[5]-1,(x+w/2)/record.width,(y+h/2)/record.height,w/record.width,h/record.height))
        actual=[tuple(map(float,line.split())) for line in Path(label).read_text().splitlines() if line.strip()]
        a=np.asarray(sorted(set(actual)),dtype=np.float64).reshape(-1,5)
        b=np.asarray(sorted(set(expected)),dtype=np.float64).reshape(-1,5)
        if a.shape!=b.shape or not np.allclose(a,b,rtol=0,atol=2e-6):
            raise RuntimeError('YOLO/raw annotation coordinate or category mismatch: '+record.stem)
    result['official_val']=annotation_meta
    result['raw_annotation_yolo_coordinates_checked']=True
    return result


def signature(args):
    evidence = c.audit(args.source,args.local_test)
    audit_training_source(args.source)
    if not args.local_test: c.resource_guard(False)
    assets = json.loads((args.project/'environment_records/yolo26_p2_e60_assets.json').read_text()) if not args.local_test else {}
    if not args.local_test and (assets.get('data_sha256')!=c.sha256(args.data) or assets.get('pretrained_sha256')!=c.WEIGHT_SHA):
        raise RuntimeError('prepare receipt data/pretrain mismatch')
    engineering = engineering_receipt(args) if not args.local_test else {'local_fixture':True}
    if c.sha256(args.project/'weights/yolo26s.pt') != c.WEIGHT_SHA:
        raise RuntimeError('Official pretraining SHA changed')
    import torch
    if not torch.cuda.is_available() and not args.local_test:
        raise RuntimeError('CUDA required; no CPU fallback')
    deploy_amp_weight(args)
    dataset = dataset_audit(args)
    device = dict(name=torch.cuda.get_device_name(0),total_bytes=torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else {'name':'cpu'}
    value = dict(package=c.sha256(c.HERE/'manifest.json'),source=c.sha256(c.HERE/'source_sha256.json'),
                 versions=evidence['versions'],data_yaml=c.sha256(args.data),dataset=dataset,
                 weights=c.WEIGHT_SHA,seed=1,layout='serial',allocator_cap_GiB=CAP_GIB,
                 train_batch=16,val_batch=32,imgsz=640,epochs=200,nms=True,device=device,
                 local_fixture=args.local_test,engineering=engineering)
    fingerprint=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    return dict(fingerprint=fingerprint,payload=value)


def audit_training_source(source):
    expected=json.loads((c.HERE/'training_source_sha256.json').read_text())
    for relative,digest in expected.items():
        if c.sha256(Path(source)/relative)!=digest:
            raise RuntimeError('Training/inference source drift: '+relative)
    return len(expected)


def verify_gate(args, sig):
    receipt = json.loads((args.project/'comparison_reports/yolo26_p2_e60_seed1_smoke_gate.json').read_text())
    if receipt.get('fingerprint')!=sig['fingerprint'] or receipt.get('local_fixture') or receipt.get('status')!='passed':
        raise RuntimeError('Real smoke receipt is missing/stale/not server; rerun smoke, no bypass')
    for variant in ('baseline','e60'):
        path=Path(receipt['reports'][variant])/'smoke.json'
        if c.sha256(path)!=receipt['report_sha256'][variant] or json.loads(path.read_text())['status']!='passed':
            raise RuntimeError('Smoke report altered/incomplete: '+variant)
    return receipt
