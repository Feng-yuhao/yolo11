"""CPU-only protocol, byte bindings and atomic receipts. No training entry."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REVISION = 'visdrone_to_songdo_source_only_v1'
SOURCE_CLASSES = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
                  'tricycle', 'awning-tricycle', 'bus', 'motor')
CLASSES = ('car', 'bus', 'truck', 'motorcycle')
COUNTS = {'test': 1084}
DATA_MANIFEST_SHA = '3c4a61ecbb782e8e2b0cda3303ee13f623f4a28b5377e352420fcdfc978c063f'
KEYS = tuple(f'{kind}_seed{seed}' for seed in (1, 2, 3) for kind in ('e1', 'e60'))
PREDICT = dict(imgsz=640, conf=.001, iou=.7, max_det=500, half=False,
               augment=False, rect=False, agnostic_nms=False, save=False,
               save_txt=False, save_conf=False, verbose=False)
PROTOCOL = dict(
    id='VisDrone10_to_Songdo4_source_only_v1', source='VisDrone2019', target='Songdo Vision v2 official test',
    source_classes=list(SOURCE_CLASSES), target_classes=list(CLASSES),
    mapping={'car': ['car', 'van'], 'bus': ['bus'], 'truck': ['truck'], 'motorcycle': ['motor']},
    score_merge='per anchor max(car,van); other mapped scores unchanged; never sum independent sigmoid scores',
    postprocess='map BEFORE best-class selection, class-aware NMS and top500; discard unmapped classes',
    predict=PREDICT, batch=4, multi_label=False, max_nms=30000,
    preprocessing='Pinned 8.3.0 native LetterBox: max side640, minimum stride padding for same-shape batches; '
                  'mixed orientations square-pad. Actual BCHW shapes recorded; rect flag does not override this old predictor.',
    evaluator='standard pycocotools COCO bbox AP@[.50:.05:.95]', maxDets=[1, 10, 100],
    GT='original author v2 integer COCO boxes/areas/IDs, all four classes',
    AP_small='original-image area<32^2, not the resized640 size',
    selection='all six existing source-val-selected best.pt, never target-selected',
    adaptation=False, training=False, target_BN_updates=False, shutdown=False,
    caveat='Source-only direct transfer, not novel-class/open-vocabulary zero-shot. Target retraining is a different experiment.')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def parser():
    p = argparse.ArgumentParser(description='Frozen VisDrone E1/E60 -> Songdo test; NO training or shutdown')
    p.add_argument('--project', type=Path, default=Path('/root/autodl-tmp/yolo11_project'))
    p.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/yolo11_src'))
    p.add_argument('--dataset', type=Path, default=Path('/root/autodl-tmp/project/SongdoVision_v2_YOLO_v1'))
    return p


def package_audit():
    manifest = read(HERE / 'manifest.json')
    require(manifest['revision'] == REVISION, 'Wrong package revision')
    for name, expected in manifest['files'].items():
        require(sha(HERE / name) == expected, 'Package changed: ' + name)
    return sha(HERE / 'manifest.json')


def source_audit(source):
    result = {}
    for name, expected in read(HERE / 'references/source_bindings.json').items():
        path = source / 'ultralytics' / name
        actual = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        # tasks.py acquired unrelated model registrations in the original project;
        # record its actual digest and additionally check loaded source topology.
        require(name == 'nn/tasks.py' or actual == expected, 'Pinned YOLO11 source differs: ' + name)
        result[name] = actual
    return result


def weights_audit(project):
    result = {}
    specs = read(HERE / 'references/checkpoints.json')
    require(set(specs) == set(KEYS), 'All three source pairs are required')
    for key in KEYS:
        spec = specs[key]
        path = project / spec['relative_path']
        require(path.is_file() and not path.is_symlink(), 'Missing source-val best: ' + str(path))
        require(sha(path) == spec['sha256'], 'Wrong/modified source checkpoint: ' + key)
        result[key] = dict(spec, path=str(path.resolve()), bytes=path.stat().st_size)
    return result


def target_audit(root, full):
    """Only test payload is required; target train/val are not read or used."""
    require(sha(root / 'SHA256SUMS.json') == DATA_MANIFEST_SHA, 'Wrong prepared Songdo manifest')
    checks = read(root / 'SHA256SUMS.json')
    names = sorted(n for n in checks if n.startswith('images/test/'))
    names += ['annotations_coco/test.json', 'splits/test.txt']
    require(len(names) == 1086, 'Wrong original test image count')
    gt = read(root / 'annotations_coco/test.json')
    require(gt['categories'] == [dict(id=i+1, name=n, supercategory='vehicle')
                                  for i, n in enumerate(CLASSES)], 'Target classes differ')
    require(len(gt['images']) == 1084 and len(gt['annotations']) == 55528, 'Wrong target GT count')
    require(len({r['id'] for r in gt['images']}) == 1084, 'Duplicate test IDs')
    metadata = hashlib.sha256()
    for index, name in enumerate(names, 1):
        path = root / name
        require(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(), 'Unsafe target path')
        stat = path.stat()
        metadata.update(json.dumps([name, stat.st_size, stat.st_mtime_ns]).encode())
        if full:
            require(sha(path) == checks[name], 'Target payload SHA mismatch: ' + name)
            if index % 200 == 0 or index == len(names):
                print(f'{now()} TARGET SHA256 {index}/{len(names)}', flush=True)
    require(sha(root / 'annotations_coco/test.json') == checks['annotations_coco/test.json'], 'GT changed')
    require(sha(root / 'splits/test.txt') == checks['splits/test.txt'], 'Test split changed')
    return dict(images=1084, boxes=55528, manifest_sha256=DATA_MANIFEST_SHA,
                gt_sha256=checks['annotations_coco/test.json'], test_file_stat_sha256=metadata.hexdigest(),
                test_images_byte_verified=bool(full), target_train_or_val_used=False)


def boot(args, settings):
    import importlib.metadata as metadata
    local = getattr(args, 'local_engineering', False)
    if not local:
        require(sys.platform == 'linux' and Path(sys.prefix).resolve() == Path('/root/miniconda3/envs/yolo11'),
                'Use /root/miniconda3/envs/yolo11/bin/python; never base or yolo26')
    os.environ.update(TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', YOLO_AUTOINSTALL='false',
                      YOLO_VERBOSE='false', YOLO_CONFIG_DIR=str(settings))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    import ultralytics
    import spatial_task_head
    spatial_task_head.configure('e60')
    require(ultralytics.__version__ == '8.3.0' and
            Path(ultralytics.__file__).resolve().parent == args.source.resolve() / 'ultralytics',
            'Wrong Ultralytics package/version')
    if not local:
        require(torch.__version__ == '2.5.1+cu121' and torch.version.cuda == '12.1', 'Original Torch/CUDA required')
    require(torch.cuda.is_available(), 'CUDA unavailable; no CPU fallback')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(1, deterministic=True)
    return dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                ultralytics=ultralytics.__version__, device=torch.cuda.get_device_name(0),
                dependencies={name:metadata.version(name) for name in ('torchvision','numpy','Pillow','pycocotools')},
                local_engineering=local, source=source_audit(args.source))
