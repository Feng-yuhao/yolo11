"""Fixed protocol and durable provenance; no training on import."""
import csv
from datetime import datetime
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REF = HERE / 'references'
PROJECT = Path('/root/autodl-tmp/yolo11_project')
SOURCE = Path('/root/autodl-tmp/yolo11_src')
DATA = Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019.yaml')
DATA_ROOT = DATA.parent
CAP_GIB = 13
REVISION = 'e0_s23_parallel_e60_testdev_v1'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.%s.tmp' % os.getpid())
    with tmp.open('w', encoding='utf-8') as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def sha(path, normalize=False):
    h = hashlib.sha256()
    if normalize:
        h.update(Path(path).read_bytes().replace(b'\r\n', b'\n'))
    else:
        with Path(path).open('rb') as f:
            for chunk in iter(lambda: f.read(4*1024**2), b''):
                h.update(chunk)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def package_audit():
    entries = read(HERE/'manifest.json')['sha256_normalized']
    for rel, expected in entries.items():
        require(sha(HERE/rel, True) == expected, 'Package changed: '+rel)
    return entries


def params(seed):
    import e0_core as core
    result = core.training_params()
    result.update(seed=seed, name='baseline_yolo11s_img640_seed%d' % seed)
    return result


def comparable(p):
    # Only architecture, random seed and machine/output addresses may differ.
    ignored = {'model', 'seed', 'name', 'project', 'data', 'device'}
    return {k:v for k,v in p.items() if k not in ignored}


def protocol_audit():
    original = read(REF/'e0_seed1_training.json')
    for seed in (2,3):
        require(comparable(params(seed)) == comparable(original['train_params']), 'E0 seed1 protocol drift')
        for other in (1,2,3):
            e60 = read(REF/('e60_seed%d_training.json' % other))
            require(comparable(params(seed)) == comparable(e60['params']), 'E60 comparison protocol drift')
    return original


def setup(seed=1, strict=True):
    sys.path.insert(0, str(SOURCE))
    os.environ['YOLO_CONFIG_DIR'] = str(PROJECT/'comparison_reports'/('e0_seed%d_pair_settings' % seed))
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    settings=Path(os.environ['YOLO_CONFIG_DIR'])
    if not (settings/'Arial.ttf').is_file():
        for font in sorted((PROJECT/'comparison_reports').glob('*settings/Arial.ttf')):
            if font.stat().st_size>100000:
                settings.mkdir(parents=True,exist_ok=True)
                shutil.copy2(font,settings/'Arial.ttf')
                break
    import torch
    import ultralytics
    import e0_core as core
    require(Path(ultralytics.__file__).resolve().parent == SOURCE/'ultralytics', 'Wrong Ultralytics installation')
    if strict:
        require(ultralytics.__version__ == '8.3.0' and torch.__version__ == '2.5.1+cu121', 'Pinned version mismatch')
        require(torch.cuda.is_available(), 'CUDA unavailable')
    core.ROOT, core.SOURCE, core.DATA = PROJECT, SOURCE, DATA
    core.PRETRAINED = PROJECT/'weights/yolo11s.pt'
    core.SEED, core.EXPERIMENT = seed, params(seed)['name']
    core.AUTO_SHUTDOWN = False  # Never power off one child while the other runs.
    core.load_dependencies()
    return torch, core


def cap(torch, gib=CAP_GIB):
    free, total = torch.cuda.mem_get_info(0)  # BYTES, not GiB.
    fraction = gib*1024**3/total
    require(0 < fraction <= 1, 'Invalid memory fraction')
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    print('Fixed %d GiB allocator cap; batch16 unchanged; total %.2f GiB' % (gib,total/1024**3), flush=True)


def signature(seed, torch):
    original = protocol_audit()
    require(sha(PROJECT/'weights/yolo11s.pt') == original['input_weights']['sha256'], 'Wrong original pretraining weights')
    require(sha(DATA) == original['data_yaml_sha256'], 'Training data YAML changed')
    source = {}
    expected = read(REF/'e60_seed1_training.json')['signature']['source']
    for rel, value in expected.items():
        source[rel] = sha(SOURCE/'ultralytics'/rel, True)
        if rel != 'nn/tasks.py':
            require(source[rel] == value, 'Pinned training source changed: '+rel)
    # Bind all further relevant training/inference modules, including additions
    # to tasks.py. The real raw topology is checked independently.
    for rel in ('nn/modules/block.py','models/yolo/detect/train.py','models/yolo/detect/val.py',
                'data/dataset.py','data/build.py','engine/model.py','engine/validator.py',
                'utils/torch_utils.py','utils/ops.py','engine/predictor.py',
                'nn/autobackend.py','models/yolo/detect/predict.py','cfg/default.yaml'):
        source[rel] = sha(SOURCE/'ultralytics'/rel, True)
    labels = {}
    for split, count in (('train',6471), ('val',548)):
        root = DATA_ROOT/('VisDrone2019-DET-'+split)
        images = sorted((root/'images').glob('*.jpg'))
        require(len(images) == count, 'Wrong '+split+' image count')
        h = hashlib.sha256()
        for image in images:
            label = root/'labels'/(image.stem+'.txt')
            require(label.is_file(), 'Missing converted label '+str(label))
            h.update(image.name.encode()+b'\0'+label.read_bytes()+b'\0')
        labels[split] = h.hexdigest()
    return dict(revision=REVISION, seed=seed, params=params(seed), cap_GiB=CAP_GIB,
                package=package_audit(), source=source, labels=labels,
                data=sha(DATA), pretrained=sha(PROJECT/'weights/yolo11s.pt'),
                torch=torch.__version__,cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0), total=torch.cuda.get_device_properties(0).total_memory)


def gates(seed, mode):
    return PROJECT/'comparison_reports/e0_s23_pair_gates'/('seed%d_%s.json' % (seed,mode))


def verify(seed, mode, binding):
    receipt = read(gates(seed,mode))
    require(receipt['status'] == 'passed' and receipt['signature'] == digest(binding), 'Missing/stale '+mode+' seed%d' % seed)
    return receipt


def completed_e0(seed):
    path = PROJECT/'comparison_reports/e0_s23_pair_gates'/('seed%d_formal_completed.json' % seed)
    result = read(path)
    require(result['status'] == 'completed' and result['completed_epochs'] == 200, 'E0 formal200 incomplete')
    require(result['seed'] == seed and comparable(result['params']) == comparable(params(seed)), 'Wrong E0 provenance')
    require(result['signature']['package'] == package_audit(), 'E0 package differs from the completed run')
    require(sha(result['weights']) == result['checkpoint_sha256'], 'Completed E0 val-best changed')
    with Path(result['run_dir'],'results.csv').open(encoding='utf-8-sig') as f:
        epochs = [int({k.strip():v.strip() for k,v in r.items()}['epoch']) for r in csv.DictReader(f)]
    require(epochs == list(range(1,201)), 'Not exactly 200 recorded epochs')
    return result


def e60_identity(seed, actual=True):
    ref = read(REF/('e60_seed%d_comparison.json' % seed))
    training = read(REF/('e60_seed%d_training.json' % seed))
    require(ref['status'] == training['status'] == 'completed' and ref['seed'] == seed and
            ref['variant'] == 'e60' and training['completed_epochs'] == 200, 'E60 incomplete/wrong seed')
    weights = PROJECT/('runs/e60a_e58_group_residual_encoding_img640_seed%d_200e/weights/best.pt' % seed)
    if actual:
        require(sha(weights) == ref['checkpoint_sha256'], 'Missing/changed E60 val-best seed%d' % seed)
    return dict(seed=seed, weights=str(weights), checkpoint_sha256=ref['checkpoint_sha256'],
                completed_epochs=200,params=training['params'])
