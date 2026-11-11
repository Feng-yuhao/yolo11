#!/usr/bin/env python3
"""VisDrone YOLO11s P2 experiment with image-wise TAL, Ultralytics 8.3.0.

Requires companion p2_tal_chunked.py. Checks include real loss/gradient equivalence.

Default: original yolo11s.pt -> P2 semantic transfer -> 200 epochs -> evaluation -> shutdown.
Run --smoke2 first: 2 full epochs plus full evaluation, NEVER automatic shutdown.
--check-only checks the architecture/transfer on CPU as well as dataset/configuration.
--no-shutdown disables shutdown. --check-only checks configuration without training.
--eval-only /absolute/path/best.pt evaluates an existing checkpoint without training
and NEVER shuts down. Do not use smoke weights as formal baseline results.

Area evaluation reproduces evaluate_smoke.py's separate FP32 prediction protocol.
This is YOLO-converted VisDrone, NOT official VisDrone ignore-region evaluation.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path('/root/autodl-tmp/yolo11_project')
SOURCE = Path('/root/autodl-tmp/yolo11_src')
PRETRAINED = ROOT / 'weights/yolo11s.pt'
DATA = Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019.yaml')
EXPERIMENT = 'e1_yolo11s_p2add_img640_seed1'
SCRIPT_REVISION = 'p2_tal_imagewise_v3'
MATCHING_HELPER = Path(__file__).resolve().with_name('p2_tal_chunked.py')
MODEL_YAML = Path(__file__).resolve().with_name('yolo11s-p2-add.yaml')
PRETRAINED_SHA256 = '85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5'
SMOKE_GATE = ROOT / 'comparison_reports/e1_p2_640_smoke_passed.json'
IMGSZ, BATCH, EPOCHS, SEED = 640, 16, 200, 1
MAX_DET, AREA_BATCH = 500, 4
CONF, IOU = 0.001, 0.7
AUTO_SHUTDOWN = True
SHUTDOWN_ON_FAILURE = True
EXPECTED_CLASSES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
    'tricycle', 'awning-tricycle', 'bus', 'motor',
]
SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def now():
    return datetime.now().isoformat(timespec='seconds')


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def json_text(data):
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=str)


def write_json(path, data):
    atomic_text(path, json_text(data) + '\n')


def value_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command):
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=30)
        return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    except Exception as exc:
        return {'error': str(exc)}


def training_params():
    return dict(
        data=str(DATA), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
        device=0, workers=8, seed=SEED, deterministic=True, patience=0,
        optimizer='SGD', lr0=0.01, lrf=0.01, momentum=0.937,
        weight_decay=0.0005, amp=True, cos_lr=True, close_mosaic=10,
        max_det=MAX_DET, conf=CONF, iou=IOU,
        val=True, plots=True, save=True, save_period=-1,
        project=str(ROOT / 'runs'), name=EXPERIMENT,
        exist_ok=False, resume=False, pretrained=True,
        cache=False, rect=False, multi_scale=False, fraction=1.0,
        nbs=64, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
        box=7.5, cls=0.5, dfl=1.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0,
        translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0,
    )


def report_text(report):
    lines = [
        'YOLO11s VisDrone experiment report',
        f"status: {report['status']}",
        f"purpose: {report['purpose']}",
        f"report_dir: {report['report_dir']}",
        'P/R/AP/AR: fractions 0-1; multiply by 100 for percentages.',
        'One fixed seed only: no multi-seed mean/std or statistical significance claim.',
        '',
    ]
    for section in ('overall', 'area_metrics', 'model_profile', 'training', 'validation'):
        if section in report:
            lines.append(section + ':')
            lines.extend(f'  {k}: {v}' for k, v in report[section].items())
            lines.append('')
    if 'per_class' in report:
        lines += ['Per-class metrics:', 'class, instances, P, R, AP50, AP50-95']
        for name, row in report['per_class'].items():
            lines.append(', '.join([name] + [str(row.get(k)) for k in
                ('instances', 'Precision', 'Recall', 'AP50', 'AP50_95')]))
    lines += ['', 'Full metadata:', json_text(report)]
    return '\n'.join(lines) + '\n'


def save_report(report):
    report['updated_at'] = now()
    out = Path(report['report_dir'])
    write_json(out / 'metrics.json', report)
    atomic_text(out / 'metrics.txt', report_text(report))


def load_dependencies():
    # Keep imports inside the guarded main routine: import errors are logged too.
    global np, torch, ultralytics, Image, YOLO, check_det_dataset, img2label_paths
    global get_flops, COCO, COCOeval
    import numpy as np
    import torch
    import ultralytics
    from PIL import Image
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset, img2label_paths
    from ultralytics.utils.torch_utils import get_flops
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval


def memory_peak():
    return {
        'allocated_GiB': torch.cuda.max_memory_allocated(0) / 1024**3,
        'reserved_GiB': torch.cuda.max_memory_reserved(0) / 1024**3,
    }


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def prepare_dataset(report):
    dataset = check_det_dataset(str(DATA), autodownload=False)
    names = dataset['names']
    if isinstance(names, list):
        names = dict(enumerate(names))
    if [names.get(k) for k in range(10)] != EXPECTED_CLASSES or len(names) != 10:
        raise ValueError('Dataset class order differs from the agreed VisDrone 10 classes.')
    counts = {}
    for split, expected in [('train', 6471), ('val', 548)]:
        source = dataset[split]
        if not isinstance(source, str) or not Path(source).is_dir():
            raise ValueError(f'{split} must be a single images directory in this script.')
        paths = sorted(p.resolve() for p in Path(source).rglob('*')
                       if p.is_file() and p.suffix.lower() in SUFFIXES)
        if len(paths) != expected:
            raise ValueError(f'{split}: expected {expected} images, found {len(paths)}.')
        counts[split + '_images'] = len(paths)
        if split == 'val':
            images = paths

    gt = dict(info={'description': 'YOLO-converted VisDrone, original image areas'},
              licenses=[], images=[], annotations=[],
              categories=[{'id': k + 1, 'name': v} for k, v in names.items()])
    ids, per_class, duplicates = {}, {k: 0 for k in names}, 0
    labels = img2label_paths([str(p) for p in images])
    digest = hashlib.sha256()
    for image_id, (path, label) in enumerate(zip(images, labels), 1):
        label = Path(label)
        if not label.is_file():
            raise FileNotFoundError(f'Missing validation label: {label}')
        with Image.open(path) as image:
            width, height = image.size
            # The tested VisDrone images have no EXIF rotations. Do not silently
            # use wrong dimensions if a different dataset is substituted.
            if image.getexif().get(274, 1) not in (None, 1):
                raise ValueError(f'EXIF-rotated image needs protocol review: {path}')
        content = label.read_bytes()
        digest.update(str(path).encode() + b'\0' + content + b'\0')
        ids[str(path)] = image_id
        gt['images'].append(dict(id=image_id, file_name=str(path), width=width, height=height))
        seen = set()
        for line in content.decode('utf-8-sig').splitlines():
            if not line.strip():
                continue
            fields = tuple(map(float, line.split()))
            if len(fields) != 5 or not all(math.isfinite(x) for x in fields):
                raise ValueError(f'Invalid detection label: {label}')
            cls, xc, yc, bw, bh = fields
            if cls != int(cls) or int(cls) not in names:
                raise ValueError(f'Invalid class ID: {label}')
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                raise ValueError(f'Invalid normalized coordinates: {label}')
            if fields in seen:
                duplicates += 1
                continue
            seen.add(fields)
            w, h = bw * width, bh * height
            gt['annotations'].append(dict(
                id=len(gt['annotations']) + 1, image_id=image_id,
                category_id=int(cls) + 1,
                bbox=[xc * width - w / 2, yc * height - h / 2, w, h],
                area=w * h, iscrowd=0,
            ))
            per_class[int(cls)] += 1
    if len(gt['annotations']) != 38759:
        raise ValueError(f"Expected 38759 validation labels, found {len(gt['annotations'])}.")
    counts.update(validation_instances=len(gt['annotations']),
                  validation_label_sha256=digest.hexdigest(),
                  duplicate_validation_labels_removed=duplicates,
                  class_names=names, class_counts=per_class)
    report['dataset'] = counts
    write_json(Path(report['report_dir']) / 'area_ground_truth.json', gt)
    return images, ids, gt, names, per_class


def preflight(report, weights, require_gpu=True):
    load_dependencies()
    if ultralytics.__version__ != '8.3.0':
        raise RuntimeError(f'Expected Ultralytics 8.3.0, got {ultralytics.__version__}.')
    if torch.__version__ != '2.5.1+cu121':
        raise RuntimeError(f'Expected torch 2.5.1+cu121, got {torch.__version__}.')
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Start the AutoDL instance in GPU mode.')
    for path in (DATA, weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    report['environment'] = dict(
        python=platform.python_version(), ultralytics=ultralytics.__version__,
        torch=torch.__version__, cuda_runtime=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), source=ultralytics.__file__,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        platform=platform.platform(),
    )
    report['input_weights'] = dict(path=str(weights), sha256=sha256(weights))
    report['data_yaml_sha256'] = sha256(DATA)
    out = Path(report['report_dir'])
    shutil.copy2(DATA, out / 'dataset.yaml')
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    write_json(out / 'environment_details.json', {
        'pip_freeze': command_output([sys.executable, '-m', 'pip', 'freeze']),
        'source_commit': command_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD']),
        'source_status': command_output(['git', '-C', str(SOURCE), 'status', '--porcelain']),
        'source_diff': command_output(['git', '-C', str(SOURCE), 'diff', 'HEAD']),
        'nvidia_smi': command_output(['nvidia-smi']),
    })
    return prepare_dataset(report)


def transfer_key(source_key):
    """Map semantic P3/P4/P5 head branches, never align heads by shifted indices."""
    parts = source_key.split('.')
    if len(parts) < 3 or parts[0] != 'model' or not parts[1].isdigit():
        raise ValueError(f'Unexpected source key: {source_key}')
    index = int(parts[1])
    if 0 <= index <= 22:
        return source_key
    if index != 23:
        raise ValueError(f'Expected original YOLO11s Detect at layer 23: {source_key}')
    parts[1] = '26'
    if parts[2] == 'dfl':
        return '.'.join(parts)
    if parts[2] in ('cv2', 'cv3') and len(parts) > 4 and parts[3] in ('0', '1', '2'):
        parts[3] = str(int(parts[3]) + 1)
        return '.'.join(parts)
    raise ValueError(f'Unknown Detect state key: {source_key}')


def class_output_key(key, layer, branches):
    parts = key.split('.')
    return (len(parts) == 6 and parts[:3] == ['model', str(layer), 'cv3']
            and parts[3] in branches and parts[4] == '2'
            and parts[5] in ('weight', 'bias'))


def plan_transfer(source_state, target_state):
    """Only the six 80->10 class output tensors may fail shape matching."""
    mapped, skipped, mapping = {}, [], []
    for key, tensor in source_state.items():
        target_key = transfer_key(key)
        if target_key not in target_state:
            raise ValueError(f'Missing target tensor: {key} -> {target_key}')
        target = target_state[target_key]
        if tuple(tensor.shape) != tuple(target.shape):
            expected = (class_output_key(key, 23, ('0', '1', '2'))
                        and tensor.shape[0] == 80 and target.shape[0] == 10
                        and tuple(tensor.shape[1:]) == tuple(target.shape[1:]))
            if not expected:
                raise ValueError(f'Unexpected shape mismatch: {key} -> {target_key}: '
                                 f'{tuple(tensor.shape)} != {tuple(target.shape)}')
            skipped.append(dict(source=key, target=target_key, reason='80 COCO classes -> 10 VisDrone classes'))
            continue
        if target_key in mapped:
            raise ValueError(f'Duplicate target mapping: {target_key}')
        mapped[target_key] = tensor
        mapping.append(dict(source=key, target=target_key, shape=list(tensor.shape)))
    missing = sorted(set(target_state) - set(mapped))
    for key in missing:
        new_p2 = key.startswith(('model.25.', 'model.26.cv2.0.', 'model.26.cv3.0.'))
        new_class_output = class_output_key(key, 26, ('1', '2', '3'))
        if not (new_p2 or new_class_output):
            raise ValueError(f'Unexpected uninitialized tensor: {key}')
    if len(skipped) != 6:
        raise ValueError(f'Expected six class-output skips, found {len(skipped)}; wrong source/model?')
    groups = {}
    for name, prefixes in {
        'backbone': tuple(f'model.{i}.' for i in range(11)),
        'original_neck': tuple(f'model.{i}.' for i in range(11, 23)),
        'original_P3_P4_P5_detect_and_DFL': ('model.26.',),
    }.items():
        keys = [key for key in mapped if key.startswith(prefixes)]
        groups[name] = dict(tensors=len(keys), elements_including_buffers=sum(mapped[k].numel() for k in keys))
    return mapped, dict(
        rule='identity layers 0..22; Detect 23->26; old branches 0,1,2->new 1,2,3',
        source='original 80-class yolo11s.pt; never baseline best.pt',
        loaded_tensors=len(mapped), groups=groups, skipped_class_outputs=skipped,
        new_or_reinitialized_target_tensors=missing, tensor_mapping=mapping,
        note='P2 fusion and P2 Detect are random; 10-class output layers are reinitialized.',
    )


def load_p2_pretrained(target):
    original = YOLO(str(PRETRAINED)).model.float()
    if len(original.model) != 24 or list(original.stride.cpu().tolist()) != [8., 16., 32.]:
        raise ValueError('Pretrained checkpoint is not original P3/P4/P5 YOLO11s.')
    for i in range(23):
        if type(original.model[i]) is not type(target.model[i]) or original.model[i].f != target.model[i].f:
            raise ValueError(f'Original layer {i} changed; this is not P2-only.')
    mapping, audit = plan_transfer(original.state_dict(), target.state_dict())
    result = target.load_state_dict(mapping, strict=False)
    if result.unexpected_keys:
        raise ValueError(f'Unexpected loaded keys: {result.unexpected_keys}')
    actual = target.state_dict()
    if any(not torch.equal(actual[key].cpu(), value.cpu()) for key, value in mapping.items()):
        raise RuntimeError('Transferred tensors failed equality verification.')
    audit['verified_tensor_equality'] = True
    del mapping, original
    return audit


def assert_p2_model(model):
    if len(model.model) != 27 or model.yaml.get('scale') != 's':
        raise ValueError('Expected the 27-layer s-scale P2-addition architecture.')
    if model.model[-1].nc != 10 or model.model[-1].f != [25, 16, 19, 22]:
        raise ValueError('Expected 10 classes and Detect inputs P2/P3/P4/P5.')
    if list(model.stride.cpu().tolist()) != [4., 8., 16., 32.]:
        raise ValueError(f'Wrong strides: {model.stride}')
    if model.model[10].__class__.__name__ != 'C2PSA':
        raise ValueError('Original P5 C2PSA must be retained.')


def canonical_layers(layers):
    """Normalize only Upsample's unspecified size, without changing the config.

    PyYAML reads the official YAML spelling 'None' as a string. Ultralytics
    converts it with ast.literal_eval later in parse_model(). Our earlier
    preflight compared before that conversion and incorrectly rejected it.
    YAML null and Python None have the same intended size semantics here.
    """
    result = copy.deepcopy(layers)
    for layer in result:
        if (isinstance(layer, list) and len(layer) == 4
                and layer[2] == 'nn.Upsample'
                and isinstance(layer[3], list) and layer[3]
                and layer[3][0] == 'None'):
            layer[3][0] = None
    return result


def validate_p2_config(cfg, original_cfg):
    expected_added = [
        [16, 1, 'nn.Upsample', [None, 2, 'nearest']],
        [[-1, 2], 1, 'Concat', [1]],
        [-1, 2, 'C3k2', [256, False]],
        [[25, 16, 19, 22], 1, 'Detect', ['nc']],
    ]
    head = canonical_layers(cfg.get('head', []))
    checks = [
        ('backbone', canonical_layers(cfg.get('backbone', [])),
         canonical_layers(original_cfg['backbone'])),
        ('original_neck', head[:12], canonical_layers(original_cfg['head'][:12])),
        ('P2_extension', head[12:], expected_added),
        ('scale', cfg.get('scale'), 's'),
        ('nc', cfg.get('nc'), 10),
        ('s_scaling', cfg.get('scales', {}).get('s'), original_cfg['scales']['s']),
    ]
    differences = [(name, actual, expected) for name, actual, expected in checks if actual != expected]
    if differences:
        details = '; '.join(f'{name}: expected={expected!r}, actual={actual!r}'
                            for name, actual, expected in differences)
        raise ValueError('P2 YAML does not match the audited addition-only design. '
                         f'Loaded file={cfg.get("yaml_file", str(MODEL_YAML))}. ' + details)


def p2_preflight(report, eval_only=False):
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils import yaml_load
    from ultralytics.utils.torch_utils import init_seeds
    report['p2_script_revision'] = SCRIPT_REVISION
    print(f'P2 script revision: {SCRIPT_REVISION}', flush=True)
    if not MODEL_YAML.is_file():
        raise FileNotFoundError(f'Upload the companion YAML: {MODEL_YAML}')
    if Path(ultralytics.__file__).resolve().parent != (SOURCE / 'ultralytics').resolve():
        raise RuntimeError('Wrong ultralytics import location; activate the original yolo11 environment.')
    if not PRETRAINED.is_file() or sha256(PRETRAINED) != PRETRAINED_SHA256:
        raise RuntimeError('Original yolo11s.pt missing/changed; do not use baseline or smoke best.pt.')
    cfg = yaml_model_load(str(MODEL_YAML))
    original_cfg = yaml_load(SOURCE / 'ultralytics/cfg/models/11/yolo11.yaml')
    validate_p2_config(cfg, original_cfg)
    code_files = [
        'nn/tasks.py', 'nn/modules/block.py', 'nn/modules/head.py', 'nn/modules/conv.py',
        'engine/model.py', 'engine/trainer.py', 'models/yolo/detect/train.py',
        'utils/tal.py', 'utils/loss.py',
    ]
    code_hashes = {name: hashlib.sha256((SOURCE / 'ultralytics' / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                   for name in code_files}
    report['p2_architecture'] = dict(
        yaml=str(MODEL_YAML), yaml_sha256=sha256(MODEL_YAML),
        original_weights_sha256=sha256(PRETRAINED), strides=[4, 8, 16, 32],
        variant='addition-only top-down P2; original P3/P4/P5 paths retained; no P2-to-P3 feedback',
        p2_channels_at_s_scale=128, original_C2PSA_unchanged=True, source_code_sha256=code_hashes,
        initialization_note='Semantic remapping from original COCO weights; all shared compatible tensors retained.',
    )
    out = Path(report['report_dir'])
    shutil.copy2(MODEL_YAML, out / MODEL_YAML.name)
    init_seeds(SEED, deterministic=True)
    model = DetectionModel(cfg, nc=10, verbose=False)
    assert_p2_model(model)
    report['preflight_transfer'] = load_p2_pretrained(model)
    model.eval()
    with torch.inference_mode():
        _, raw = model(torch.zeros(1, 3, 256, 256))
    sizes = [list(x.shape[-2:]) for x in raw]
    if sizes != [[64, 64], [32, 32], [16, 16], [8, 8]]:
        raise RuntimeError(f'Unexpected P2 forward shapes: {sizes}')
    report['p2_architecture']['CPU_forward_256_feature_shapes'] = sizes
    report['p2_architecture']['preflight_parameters'] = sum(p.numel() for p in model.parameters())
    save_report(report)
    del model, raw
    gc.collect()
    print('P2 structure/semantic weight transfer/CPU forward checks passed. Strides: 4,8,16,32.', flush=True)
    if not MATCHING_HELPER.is_file():
        raise FileNotFoundError(f'Upload the matching helper beside this script: {MATCHING_HELPER}')
    from p2_tal_chunked import implementation_info, verify_equivalence, assert_patch_restoration
    report['p2_architecture']['matching_implementation'] = implementation_info()
    shutil.copy2(MATCHING_HELPER, out / MATCHING_HELPER.name)
    baseline_cfg = copy.deepcopy(original_cfg)
    baseline_cfg.update(scale='s', nc=10)
    check_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Checking original vs image-wise TAL outputs/loss/gradients on {check_device}.', flush=True)
    report['status'] = 'checking_matching_equivalence'
    save_report(report)
    try:
        report['matching_equivalence'] = verify_equivalence(
            copy.deepcopy(cfg), device=check_device, baseline_cfg=baseline_cfg)
        assert_patch_restoration()
        write_json(out / 'tal_equivalence.json', report['matching_equivalence'])
    except BaseException:
        write_json(out / 'tal_equivalence.json', dict(
            status='failed', device=check_device, error=traceback.format_exc()))
        raise
    finally:
        clear_gpu()
    save_report(report)
    print('TAL equivalence checks passed. This is not yet a 1024px memory test.', flush=True)


def smoke_signature(report):
    params = training_params()
    params.pop('name')
    params.pop('project')
    payload = dict(
        train_params=params, evaluation_protocol=report['evaluation_protocol'],
        architecture=report['p2_architecture'],
        data_yaml_sha256=report['data_yaml_sha256'],
        validation_label_sha256=report['dataset']['validation_label_sha256'],
        environment=report['environment'],
        script_sha256=sha256(Path(__file__)),
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def save_smoke_gate(report):
    write_json(SMOKE_GATE, dict(
        status='smoke_passed', signature=smoke_signature(report),
        report_dir=report['report_dir'], run_dir=report['run_dir'], finished_at=now(),
        note='Two full training epochs + builtin validation + full FP32/area evaluation. Not an accuracy result.',
    ))


def verify_smoke_gate(report):
    if not SMOKE_GATE.is_file():
        raise RuntimeError('Run --smoke2 first and inspect the complete report; formal training has not started.')
    gate = json.loads(SMOKE_GATE.read_text(encoding='utf-8'))
    if gate.get('status') != 'smoke_passed' or gate.get('signature') != smoke_signature(report):
        raise RuntimeError('Smoke gate does not match current code/config/environment. Run --smoke2 again.')
    report['smoke_gate'] = gate


def make_p2_trainer(report):
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.nn.tasks import DetectionModel

    class P2Trainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            # Ultralytics rebuilds the dataset-nc model in train(). Transfer HERE,
            # not only to the temporary YOLO wrapper that gets discarded.
            if weights is not None:
                raise RuntimeError('Unexpected model weights/resume; this experiment starts from original yolo11s.pt.')
            model = DetectionModel(cfg, nc=self.data['nc'], verbose=verbose)
            assert_p2_model(model)
            report['pretrained_transfer'] = load_p2_pretrained(model)
            write_json(Path(report['report_dir']) / 'pretrained_transfer.json', report['pretrained_transfer'])
            save_report(report)
            return model

    return P2Trainer


def train(report, smoke=False):
    from p2_tal_chunked import chunked_assignment
    # No smoke checkpoint, no resume, no weights from a previous seed.
    params = training_params()
    if smoke:
        params.update(epochs=2, name='smoke2_' + EXPERIMENT)
    target_epochs = params['epochs']
    report['train_params'] = params
    report['training'] = dict(completed_epochs=0, best_checkpoint_epoch=None,
                              best_epoch_by_map50_95=None,
                              best_checkpoint_rule='Ultralytics 8.3.0 validation fitness',
                              memory_note='PyTorch allocator peaks, including in-training validation; not whole-device usage.')
    best_map = -math.inf
    previous_epoch = 0
    started = time.perf_counter()
    clear_gpu()
    torch.cuda.reset_peak_memory_stats(0)
    model = YOLO(str(MODEL_YAML), task='detect')

    def setup(trainer):
        report['run_dir'] = str(Path(trainer.save_dir).resolve())
        report['training']['effective_amp'] = bool(trainer.amp)
        report['training']['parameters_before_fusion'] = sum(p.numel() for p in trainer.model.parameters())
        shutil.copy2(Path(trainer.save_dir) / 'args.yaml', Path(report['report_dir']) / 'train_args.yaml')
        shutil.copy2(MODEL_YAML, Path(trainer.save_dir) / MODEL_YAML.name)
        write_json(Path(trainer.save_dir) / 'pretrained_transfer.json', report['pretrained_transfer'])
        shutil.copy2(MATCHING_HELPER, Path(trainer.save_dir) / MATCHING_HELPER.name)
        write_json(Path(trainer.save_dir) / 'tal_equivalence.json', report['matching_equivalence'])
        save_report(report)

    def saved(trainer):
        # This event runs after best.pt is saved; matches ties/fitness exactly.
        if trainer.fitness == trainer.best_fitness:
            report['training']['best_checkpoint_epoch'] = int(trainer.epoch) + 1

    def epoch_end(trainer):
        nonlocal best_map, previous_epoch
        epoch = int(trainer.epoch) + 1
        # on_fit_epoch_end also fires once after final_eval; don't count twice.
        if epoch <= previous_epoch:
            return
        previous_epoch = epoch
        score = value_or_none(trainer.metrics.get('metrics/mAP50-95(B)', float('nan')))
        if score is not None and score > best_map:
            best_map = score
            report['training']['best_epoch_by_map50_95'] = epoch
        report['training'].update(completed_epochs=epoch,
                                  elapsed_seconds=time.perf_counter() - started,
                                  peak_memory=memory_peak())
        save_report(report)

    model.add_callback('on_train_start', setup)
    model.add_callback('on_model_save', saved)
    model.add_callback('on_fit_epoch_end', epoch_end)
    report['status'] = 'training'
    save_report(report)
    matching_stats = {}
    try:
        with chunked_assignment() as matching_stats:
            model.train(trainer=make_p2_trainer(report), **params)
        torch.cuda.synchronize()
        run_dir = Path(model.trainer.save_dir).resolve()
        report['run_dir'] = str(run_dir)
        report['training']['seconds_including_builtin_final_validation'] = time.perf_counter() - started
        report['training']['peak_memory'] = memory_peak()
        if report['training']['completed_epochs'] != target_epochs:
            raise RuntimeError(f'Not all {target_epochs} epochs completed.')
        weights = run_dir / 'weights/best.pt'
        if not weights.is_file():
            raise FileNotFoundError(weights)
        for filename in ('results.csv', 'args.yaml'):
            shutil.copy2(run_dir / filename, Path(report['report_dir']) / filename)
        report['status'] = 'training_completed'
        save_report(report)  # durable training record BEFORE any extra evaluation
        return weights
    finally:
        report['training']['matching_execution'] = dict(matching_stats)
        report['training']['comparison_note'] = (
            'Full forward batch unchanged; image-wise matching only. '
            'Training time/memory comparisons require the same matching implementation on both models.')
        report['training']['elapsed_seconds'] = time.perf_counter() - started
        report['training']['peak_memory'] = memory_peak()
        save_report(report)
        del model
        clear_gpu()


def summarize_area(evaluator):
    # Avoid COCOeval.summarize()'s default maxDets=100 AP entry when using 500.
    p = evaluator.params
    max_index = p.maxDets.index(MAX_DET)
    result = {}
    for name in ('all', 'small', 'medium', 'large'):
        a = p.areaRngLbl.index(name)
        precision = evaluator.eval['precision'][:, :, :, a, max_index]
        recall = evaluator.eval['recall'][:, :, a, max_index]
        for metric, values in [('AP', precision), ('AR', recall)]:
            valid = values[values > -1]
            result[f'{metric}_{name}'] = float(valid.mean()) if valid.size else None
    result['max_det'] = MAX_DET
    return result


def evaluate(report, weights, dataset):
    images, path_ids, gt, names, counts = dataset
    out = Path(report['report_dir'])
    report['evaluation_weights'] = dict(path=str(weights), sha256=sha256(weights))
    report['status'] = 'validating'
    save_report(report)
    model = YOLO(str(weights))
    assert_p2_model(model.model)
    if model.names != names:
        raise ValueError('Checkpoint classes do not match dataset classes.')
    clear_gpu()
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize()
    start = time.perf_counter()
    metrics = model.val(
        data=str(DATA), split='val', imgsz=IMGSZ, batch=BATCH, device=0,
        workers=8, conf=CONF, iou=IOU, max_det=MAX_DET,
        half=False, augment=False, plots=True, verbose=True,
        project=str(out), name='validation', exist_ok=False,
    )
    torch.cuda.synchronize()
    report['validation'] = dict(seconds=time.perf_counter() - start,
                                peak_memory=memory_peak(),
                                speed_ms_per_image={k: float(v) for k, v in metrics.speed.items()},
                                speed_note='Batched FP32 validation, NOT batch=1 deployment FPS.')
    box = metrics.box
    report['overall'] = dict(Precision=float(box.mp), Recall=float(box.mr),
                             mAP50=float(box.map50), mAP75=float(box.map75),
                             mAP50_95=float(box.map))
    report['per_class'] = {
        names[k]: dict(instances=counts[k], Precision=None, Recall=None, AP50=None, AP50_95=None)
        for k in names
    }
    for index, cls in enumerate(box.ap_class_index):
        report['per_class'][names[int(cls)]].update(
            Precision=float(box.p[index]), Recall=float(box.r[index]),
            AP50=float(box.ap50[index]), AP50_95=float(box.ap[index]))
    report['model_profile'] = dict(parameters=sum(p.numel() for p in model.model.parameters()),
                                   imgsz=IMGSZ, state='post-validation, fused')
    try:
        flops = float(get_flops(model.model, imgsz=IMGSZ))
        report['model_profile']['GFLOPs'] = flops if flops > 0 else None
    except Exception as exc:
        report['model_profile'].update(GFLOPs=None, profile_error=str(exc))
    save_report(report)
    del metrics, box, model
    clear_gpu()

    print('\nArea prediction: explicit chunks of at most 4 images.', flush=True)
    report['status'] = 'area_predicting'
    save_report(report)
    model = YOLO(str(weights))
    assert_p2_model(model.model)
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize()
    started = time.perf_counter()
    detections = []
    for start in range(0, len(images), AREA_BATCH):
        chunk = images[start:start + AREA_BATCH]
        results = model.predict(
            source=[str(p) for p in chunk], imgsz=IMGSZ, batch=len(chunk), device=0,
            conf=CONF, iou=IOU, max_det=MAX_DET, half=False, augment=False,
            verbose=False, save=False, stream=False,
        )
        for result in results:
            image_id = path_ids[str(Path(result.path).resolve())]
            if result.boxes is None:
                continue
            for x1, y1, x2, y2, score, cls in result.boxes.data.cpu().tolist():
                detections.append(dict(image_id=image_id, category_id=int(cls) + 1,
                                       bbox=[x1, y1, max(0, x2-x1), max(0, y2-y1)], score=score))
        del results
        done = min(start + AREA_BATCH, len(images))
        if start == 0 or done % 100 == 0 or done == len(images):
            print(f'Area prediction: {done}/{len(images)}', flush=True)
            report['area_progress_images'] = done
            save_report(report)
    torch.cuda.synchronize()
    report['area_prediction'] = dict(seconds=time.perf_counter() - started, peak_memory=memory_peak())
    write_json(out / 'area_predictions.json', detections)
    del model
    clear_gpu()
    report['status'] = 'area_accumulating_on_cpu'
    save_report(report)
    print('Computing area AP/AR on CPU; this may take several minutes.', flush=True)
    gt_coco = COCO()
    gt_coco.dataset = gt
    gt_coco.createIndex()
    if detections:
        dt_coco = gt_coco.loadRes(detections)
    else:
        dt_coco = COCO()
        dt_coco.dataset = dict(images=gt['images'], categories=gt['categories'], annotations=[])
        dt_coco.createIndex()
    evaluator = COCOeval(gt_coco, dt_coco, 'bbox')
    evaluator.params.maxDets = [1, 10, MAX_DET]
    evaluator.evaluate()
    evaluator.accumulate()
    report['area_metrics'] = summarize_area(evaluator)
    report['area_metrics']['prediction_count'] = len(detections)
    save_report(report)


def export_csv(report):
    out = Path(report['report_dir'])
    fields = ['class', 'instances', 'Precision', 'Recall', 'AP50', 'AP50_95']
    with (out / 'per_class.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, values in report['per_class'].items():
            writer.writerow({'class': name, **values})
    row = dict(experiment=EXPERIMENT, seed=SEED, purpose=report['purpose'],
               **report['overall'], **report['area_metrics'], **report['model_profile'])
    row.update(training_seconds=report.get('training', {}).get('seconds_including_builtin_final_validation'),
               training_peak_allocated_GiB=report.get('training', {}).get('peak_memory', {}).get('allocated_GiB'),
               validation_inference_ms_per_image=report['validation']['speed_ms_per_image'].get('inference'))
    with (out / 'seed_metrics.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def request_shutdown(out, enabled, reason):
    marker = Path(out) / 'shutdown_status.json'
    state = dict(time=now(), enabled=enabled, reason=reason,
                 command=['/bin/bash', '-c', '/usr/bin/shutdown'],
                 status='disabled' if not enabled else 'requesting')
    write_json(marker, state)
    if not enabled:
        print('Auto shutdown disabled; turn off the instance manually if idle.', flush=True)
        return
    # Guard against accidental execution on a local workstation.
    if platform.system() != 'Linux' or not Path('/root/autodl-tmp').is_dir():
        state['status'] = 'refused_non_autodl_path'
        write_json(marker, state)
        return
    if not Path('/usr/bin/shutdown').is_file():
        state['status'] = 'shutdown_command_missing'
        write_json(marker, state)
        return
    print('Reports saved. Requesting AutoDL shutdown through bash.', flush=True)
    if hasattr(os, 'sync'):
        os.sync()
    try:
        # bash handles AutoDL scripts without a valid shebang (Exec format error).
        completed = subprocess.run(state['command'], capture_output=True,
                                   text=True, timeout=30, check=False)
        state.update(returncode=completed.returncode, stdout=completed.stdout,
                     stderr=completed.stderr,
                     status='command_returned_unverified' if completed.returncode == 0 else 'request_failed')
    except Exception as exc:
        state.update(status='request_exception', error=str(exc))
    write_json(marker, state)
    if hasattr(os, 'sync'):
        os.sync()
    print('Shutdown command recorded. Actual power state must be checked in AutoDL console.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-shutdown', action='store_true')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only', action='store_true')
    modes.add_argument('--smoke2', action='store_true', help='2 full epochs + full evaluation; no shutdown; not formal result')
    modes.add_argument('--eval-only', type=Path, metavar='BEST_PT')
    args = parser.parse_args()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    prefix = 'CHECK_P2' if args.check_only else 'EVAL_ONLY_P2' if args.eval_only else 'SMOKE2_' + EXPERIMENT if args.smoke2 else EXPERIMENT
    out = ROOT / 'comparison_reports' / f'{prefix}_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    report = dict(
        status='initializing', purpose=prefix, experiment=EXPERIMENT, seed=SEED,
        seed_count=1, multi_seed_std=None, started_at=now(), report_dir=str(out),
        is_formal=not (args.check_only or args.eval_only or args.smoke2),
        evaluation_protocol=dict(
            split='val', imgsz=IMGSZ, batch=BATCH, area_batch=AREA_BATCH,
            conf=CONF, iou=IOU, max_det=MAX_DET, half=False,
            protocol_id='visdrone_yolo_labels_ultra830_fp32_area_separate_predict_v1',
            area_definition='COCO default original-image bbox area ranges, IoU 0.50:0.95',
            notes=[
                'COCO area evaluation on converted VisDrone labels, not official VisDrone ignore regions.',
                'Ultralytics AP and COCO AP_all are separate evaluators/prediction passes; do not mix them.',
                'Batch val may use rectangular padding; area predict uses each 4-image chunk padding.',
                'P/R are Ultralytics selected F1 operating point metrics, not P/R at conf=0.001.',
                'Training built-in validation may use FP16 and double batch; final report is standalone FP32.',
            ],
        ),
    )
    save_report(report)
    success = False
    started = time.perf_counter()
    try:
        weights = args.eval_only.resolve() if args.eval_only else PRETRAINED
        dataset = preflight(report, weights, require_gpu=not args.check_only)
        p2_preflight(report, eval_only=bool(args.eval_only))
        report['train_params'] = training_params()
        save_report(report)
        if args.check_only:
            report['status'] = 'check_passed'
            success = True
            print('Preflight passed. No training, no shutdown.', flush=True)
            return
        if not args.eval_only:
            if not args.smoke2:
                verify_smoke_gate(report)
            weights = train(report, smoke=args.smoke2)
        else:
            report['training'] = dict(note='Evaluation only; training duration/memory cannot be reconstructed.')
        evaluate(report, weights, dataset)
        export_csv(report)
        report.update(status='completed', finished_at=now(), total_seconds=time.perf_counter() - started)
        save_report(report)
        if not args.eval_only:
            actual_run = Path(report['run_dir'])
            for name in ('metrics.txt', 'metrics.json', 'per_class.csv', 'seed_metrics.csv'):
                shutil.copy2(out / name, actual_run / ('complete_' + name))
            if args.smoke2:
                save_smoke_gate(report)
            else:
                # Append formal runs only, never smoke/check/evaluation-only.
                with (ROOT / 'comparison_reports/ALL_EXPERIMENTS_COMPARISON.txt').open('a', encoding='utf-8') as handle:
                    handle.write('\n' + '=' * 72 + '\n' + report_text(report))
                    handle.flush()
                    os.fsync(handle.fileno())
        success = True
        print(f"\nCompleted. Report: {out / 'metrics.txt'}", flush=True)
    except BaseException:
        report.update(status='failed_or_interrupted', finished_at=now(),
                      total_seconds=time.perf_counter() - started, error=traceback.format_exc())
        save_report(report)
        atomic_text(out / 'error.log', report['error'])
        print(f'Failure saved: {out}', file=sys.stderr, flush=True)
        raise
    finally:
        save_report(report)
        enabled = (AUTO_SHUTDOWN and not args.no_shutdown and not args.check_only
                   and not args.eval_only and not args.smoke2 and (success or SHUTDOWN_ON_FAILURE))
        try:
            request_shutdown(out, enabled, report['status'])
        except Exception:
            # Reporting errors must never hide the original training exception.
            print('Could not finish shutdown handling:\n' + traceback.format_exc(), file=sys.stderr)


if __name__ == '__main__':
    main()
