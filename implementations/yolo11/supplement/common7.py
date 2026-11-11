"""CPU-only identities, durable receipts, and predeclared seven-run plan."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone, timedelta
import csv, hashlib, json, os, math, shutil, zipfile

HERE=Path(__file__).resolve().parent
DEFAULT_PROJECT=Path('/root/autodl-tmp/yolo11_project')
DEFAULT_SOURCE=Path('/root/autodl-tmp/yolo11_src')
DEFAULT_DATA=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019.yaml')
PYTHON=Path('/root/miniconda3/envs/yolo11/bin/python')
KEYS=('AP50_95_percent','AP50_percent','AP75_percent','AR500_percent')
JOBS={
 'shared_s2':dict(kind='routing',variant='shared',seed=2,run='e60_ablation_shared_img640_seed2_200e'),
 'e58_s3':dict(kind='e58',variant='e58',seed=3,run='e58_w96_p2345_img640_seed3_200e'),
 'uniform_s2':dict(kind='routing',variant='uniform',seed=2,run='e60_ablation_uniform_img640_seed2_200e'),
 'td_s2':dict(kind='td',variant='tood_td',seed=2,run='tood_td_e1_p2345_w64_d6_img640_seed2_200e'),
 'shared_s3':dict(kind='routing',variant='shared',seed=3,run='e60_ablation_shared_img640_seed3_200e'),
 'td_s3':dict(kind='td',variant='tood_td',seed=3,run='tood_td_e1_p2345_w64_d6_img640_seed3_200e'),
 'uniform_s3':dict(kind='routing',variant='uniform',seed=3,run='e60_ablation_uniform_img640_seed3_200e')}
WAVES=[['shared_s2','e58_s3'],['uniform_s2','td_s2'],['shared_s3','td_s3'],['uniform_s3']]

def now():return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
def require(value,message):
    if not value:raise RuntimeError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with temp.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,p)
    if os.name!='nt':
        fd=os.open(p.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
def text(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with t.open('w',encoding='utf-8',newline='\n') as f:f.write(value);f.flush();os.fsync(f.fileno())
    os.replace(t,p)
def sha(p,normalized=False):
    if normalized:return hashlib.sha256(Path(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def package_audit():
    p=HERE/'package_manifest.json';m=read(p)
    for name,expected in m['files'].items():
        target=HERE/name
        require(target.resolve().is_relative_to(HERE),'Package path escape')
        require(target.is_file() and sha(target)==expected,'Package file changed/missing: '+name)
    return sha(p)
def contained(p,parent):
    p=Path(p).resolve();parent=Path(parent).resolve()
    require(p!=parent and p.is_relative_to(parent),'Path is not inside expected parent: '+str(p))
    return p
def free_disk(project,minimum_gib=2):
    available=shutil.disk_usage(project).free/2**30
    require(available>=minimum_gib,f'Insufficient disk: {available:.2f} GiB free; need {minimum_gib:g} GiB. No automatic deletion.')
    return available
def record_receipt(out,name,binding,evidence):
    path=Path(out)/(name+'.json');write(path,evidence)
    receipt=dict(status='passed',binding=binding,evidence_sha256=sha(path),at=now())
    write(Path(out)/(name+'_receipt.json'),receipt)
    return receipt
def verify_receipt(out,name,binding):
    out=Path(out);r=read(out/(name+'_receipt.json'));value=read(out/(name+'.json'))
    require(r['status']=='passed' and r['binding']==binding,'Stale/missing '+name+' receipt: '+str(out))
    require(r['evidence_sha256']==sha(out/(name+'.json')) and value['status']=='passed','Invalid '+name+' evidence')
    require(not (out/'failure.json').exists(),'Failed stage cannot be a gate: '+str(out))
    return value
def csv_epochs(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        rows=[{k.strip():v.strip() for k,v in r.items()} for r in csv.DictReader(f)]
    require([int(r['epoch']) for r in rows]==list(range(1,201)),'Formal results.csv must contain exactly epochs 1..200')
    return rows
def checkpoint_check(path):
    path=Path(path);require(path.stat().st_size>=1024*1024,'Truncated checkpoint: '+str(path))
    with zipfile.ZipFile(path) as z:
        require(sum(x.file_size for x in z.infolist())<1024**3,'Unexpected checkpoint size')
        require(z.testzip() is None,'Checkpoint CRC failed')
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),zip_crc_valid=True)
def finite_metrics(metrics):
    for key in KEYS:require(isinstance(metrics[key],(float,int)) and math.isfinite(metrics[key]) and 0<=metrics[key]<=100,'Invalid official metric '+key)

def native_metrics_for_json(metrics):
    """Do not confuse the pinned FP16 validation-BCE sum overflow with AP/NaNs.

    The original validator is NOT modified. The archived validation precision
    oracle independently checks the same loss under AMP and finite raw logits.
    Only positive infinity in this known diagnostic field is representable;
    every AP/precision/recall and all other loss values must be finite.
    """
    result={};diagnostics=[]
    for key,value in metrics.items():
        value=float(value)
        if math.isfinite(value):result[key]=value
        else:
            require(key=='val/cls_loss' and value==math.inf,'Nonfinite native metric/loss: '+key)
            result[key]='inf';diagnostics.append('Original native FP16 validation cls-loss sum overflow; not a training loss or AP value')
    return result,diagnostics

def completed_inventory(project,report):
    """Strict success criteria, used again by the torch-free shutdown guard."""
    project,report=Path(project),Path(report);proof={}
    for key,job in JOBS.items():
        root=report/'jobs'/key;train=read(root/'train/training_completion.json')
        require(train['status']=='completed' and train['completed_epochs']==200 and train['seed']==job['seed'] and train['key']==key and train.get('official_scores_completed') is True,'Training/scoring incomplete: '+key)
        require(train['binding']['key']==key and train['binding']['job']==job and train['binding']['params']['seed']==job['seed'] and train['binding']['params']['epochs']==200 and train['binding']['params']['batch']==16,'Wrong training recipe binding: '+key)
        run=contained(project/'runs'/job['run'],project/'runs');csv_epochs(run/'results.csv')
        files={n:checkpoint_check(run/'weights'/n) for n in ['best.pt','last.pt']}
        require(files['best.pt']['sha256']==train['best_sha256'] and files['last.pt']['sha256']==train['last_sha256'],'Checkpoint changed: '+key)
        for split,count in [('val',548),('test-dev',1610)]:
            out=root/split;value=read(out/'official_metrics.json');pred=read(out/'prediction_manifest.json')
            require(value['status']=='completed' and value['key']==key and value['seed']==job['seed'],'Wrong score identity: '+key)
            b=value['binding'];require(b['checkpoint_sha256']==train['best_sha256'] and b['split']==split and b['images']==count,'Wrong score binding: '+key)
            require(pred['status']=='completed' and pred['binding']==b and pred['seen_images']==count and not b.get('smoke_only') and sha(out/'area_predictions.json')==value['prediction_sha256']==pred['sha256'],'Incomplete predictions: '+key+'/'+split)
            require(b['package_sha256']==package_audit(),'Scoring package differs: '+key)
            require(b['admission_id']==train['binding']['admission'] and b['seed']==job['seed'] and b['scorer_sha256']==sha(HERE/'frozen'/job['kind']/'d27_official.py'),'Wrong scorer/admission identity: '+key)
            expected=dict(imgsz=640,device='cuda:0',conf=.001,iou=.7,max_det=500,half=False,augment=False,
                          verbose=False,save=False,agnostic_nms=False,classes=None,batch=4)
            require(b['protocol']==expected,'Final inference protocol differs: '+key)
            finite_metrics(value['official'])
        proof[key]=dict(training=train,weights=files)
    require(read(report/'summary.json')['status']=='completed' and (report/'REPORT.md').is_file(),'Final summary missing')
    return proof
