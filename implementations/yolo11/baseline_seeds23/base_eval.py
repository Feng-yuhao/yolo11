#!/usr/bin/env python3
"""Frozen val-best E0/E58 seed1 -> official VisDrone test-dev. No training."""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
REF = HERE / 'references'
REVISION = 'testdev_e0_e58_seed1_v1'
TEST_ANN_SHA = '8ac85f9cf40c98b0373281ccf5d3c902b7bb502492e51a48537249b0a1b8dffd'
TEST_IMAGE_SHA = '09a2d2daa96dc1ad5deb2311fb132edece644e554fd2c7bc6f0418384654f20b'
SCORER_SHA = 'a9e0f50885f81e72ed85baf57a880800c6c51a51527a646b5c60184a1b76bf60'
GIB = 1024 ** 3
KEYS = ('AP50_95_percent', 'AP50_percent', 'AP75_percent', 'AR500_percent')
SPECS = {
    'e0_seed1': {
        'weights': 'runs/baseline_yolo11s_img640_seed1/weights/best.pt',
        'sha256': 'bc5d1304c7bf4e0c81db853d0d7a39dd46485940ec2b8c2da63df8c3eda934b8',
        'val_report': 'baseline_yolo11s_img640_seed1_20260926_073621_610152',
        'nl': 3, 'stride': (8, 16, 32),
    },
    'e58_seed1': {
        'weights': 'runs/e58a_e57_allscale_w96_img640_seed1_200e/weights/best.pt',
        'sha256': 'd4afe6bcbedb415f5551a09e3f48e6f0d436d6a49d0d8080a5b9f6394fb7319e',
        'val_report': 'E58_RUN_20261008_001519_559728',
        'nl': 4, 'stride': (4, 8, 16, 32),
    },
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def sha(path, normalized=False):
    if normalized:
        return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def options(device):
    # stream is a predict() argument; never pass it to get_cfg overrides.
    return dict(imgsz=640, device=device, conf=.001, iou=.7, max_det=500,
                half=False, augment=False, verbose=False, save=False,
                agnostic_nms=False, classes=None)


def protocol(device):
    return dict(**options(device), batch=4,
                letterbox='unchanged Ultralytics 8.3.0 predictor pre_transform',
                deterministic_algorithms=True, deterministic_warn_only=True,
                cudnn_deterministic=True, cudnn_benchmark=False)


def prepare(args, strict=True):
    # The pinned source MUST precede importing custom classes/unpickling.
    source = args.source.resolve()
    require((source / 'ultralytics/__init__.py').is_file(), 'Missing Ultralytics source: ' + str(source))
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(HERE))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_CONFIG_DIR'] = str(args.project / 'comparison_reports/testdev_e0_e58_settings')
    import torch
    import ultralytics
    from ultralytics.utils.torch_utils import init_seeds
    import spatial_task_head
    spatial_task_head.configure('e58')
    require(Path(ultralytics.__file__).resolve().parent == source / 'ultralytics',
            'Imported a different Ultralytics installation')
    if strict:
        original = read(REF / 'e0_val_metadata.json')['environment']
        require(ultralytics.__version__ == original['ultralytics'] == '8.3.0', 'Ultralytics version drift')
        require(torch.__version__ == original['torch'], 'Torch version differs from original evaluation')
        require(torch.version.cuda == original['cuda_runtime'], 'CUDA runtime version drift')
    init_seeds(1, deterministic=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    return torch


def audit_package():
    manifest = read(HERE / 'PACKAGE_MANIFEST.json')
    for rel, expected in manifest['sha256_normalized'].items():
        require(sha(HERE / rel, normalized=True) == expected, 'Package source/reference differs: ' + rel)
    require(sha(HERE / 'd27_official.py', True) == SCORER_SHA, 'Frozen official evaluator was modified')
    return manifest


def source_binding(args):
    original = read(REF / 'e1_val_metadata.json')['p2_architecture']['source_code_sha256']
    values = {}
    # Trainer/loss/TAL are not used by this read-only entry, so only bind the
    # inference dependencies that were pinned during the existing runs.
    for rel in ('nn/modules/block.py', 'nn/modules/head.py',
                'nn/modules/conv.py', 'engine/model.py'):
        actual = sha(args.source / 'ultralytics' / rel, True)
        require(actual == original[rel], 'Original inference source differs: ' + rel)
        values[rel] = actual
    # tasks.py accumulated unrelated YAML registrations after E1. Record its
    # actual SHA and bind the receipts; validate the real inference path by
    # reproducing the original val predictions, not by rejecting new names.
    for rel in ('nn/tasks.py', 'engine/predictor.py', 'models/yolo/detect/predict.py',
                'utils/ops.py', 'nn/autobackend.py', 'data/augment.py'):
        values[rel] = sha(args.source / 'ultralytics' / rel, True)
    return values


def load_dataset(root, count):
    import d27_official as official
    old = official.EXPECTED_IMAGES
    try:
        official.EXPECTED_IMAGES = count
        return official.load_dataset(Path(root))
    finally:
        official.EXPECTED_IMAGES = old


def audit_dataset(root):
    records, meta = load_dataset(root, 1610)
    frozen = read(REF / 'testdev_input_manifest.json')
    require(meta['annotation_dimension_sha256'] == TEST_ANN_SHA, 'Wrong test-dev original annotations/dimensions')
    require(frozen['dataset']['image_content_sha256'] == TEST_IMAGE_SHA, 'Frozen image reference drift')
    require(len(frozen['images']) == len(records) == 1610, 'Wrong test-dev image count')
    started = time.monotonic()
    for i, (record, ref) in enumerate(zip(records, frozen['images']), 1):
        require((record.image_id, record.stem, record.width, record.height) ==
                (ref['id'], ref['stem'], ref['width'], ref['height']), 'Image ID/stem/dimension mapping differs')
        require(sha(record.image_path) == ref['sha256'], 'Image content differs: ' + record.stem)
        if i % 200 == 0 or i == 1610:
            print(f'{now()} TEST-DEV data audit {i}/1610 elapsed={time.monotonic()-started:.1f}s', flush=True)
    meta.update(split='test-dev', image_content_sha256=TEST_IMAGE_SHA)
    return records, meta


def validated_cached_reference(key, p):
    metric = read(REF / (key + '_testdev_official.json'))
    pred = read(REF / (key + '_testdev_prediction_manifest.json'))
    binding = metric['binding']
    expected_weight = {
        'e1': 'f1a5ba6642a01445a99b2c7c80d8e79299a4402159bc37d2a08bb31ff365df12',
        'e40': '0a9fdac26c4fe7a9161900c82e879f584026afa569f870c0159d41feca031337',
    }[key]
    require(metric['status'] == pred['status'] == 'completed', 'Incomplete cached test result')
    require(binding == pred['binding'] and binding['key'] == key + '_seed1', 'Cached result binding differs')
    require(binding['checkpoint_sha256'] == expected_weight, 'Cached reference has different weights')
    require(binding['dataset_annotation_sha256'] == TEST_ANN_SHA and
            binding['dataset_image_sha256'] == TEST_IMAGE_SHA, 'Cached reference has different test data')
    require(binding['image_count'] == pred['seen_images'] == 1610 and not binding['smoke_only'],
            'Cached reference is not a full test-dev result')
    # GPU numbering is not a prediction hyperparameter; all other fields must match.
    expected_p = dict(p); expected_p['device'] = binding['protocol']['device']
    require(binding['protocol'] == expected_p, 'Cached reference prediction protocol differs')
    require(metric['prediction_sha256'] == pred['sha256'], 'Cached reference prediction SHA differs')
    return dict(official=metric['official'], provenance='frozen existing full test-dev result; not rerun',
                binding=binding, prediction_sha256=metric['prediction_sha256'])


def preflight(args, torch):
    package = audit_package()
    source = source_binding(args)
    e0 = read(REF / 'e0_val_metadata.json')
    e58 = read(REF / 'e58_training.json')
    comparison = read(REF / 'e58_val_comparison.json')
    require(e0['seed'] == 1 and e0['train_params']['imgsz'] == 640 and
            e0['training']['completed_epochs'] == 200 and e0['status'] == 'completed', 'Wrong E0 training provenance')
    require(e58['params']['seed'] == 1 and e58['params']['imgsz'] == 640 and
            e58['completed_epochs'] == 200 and e58['status'] == 'completed' and e58['variant'] == 'e58',
            'Wrong E58 training provenance')
    require(e0['evaluation_weights']['sha256'] == SPECS['e0_seed1']['sha256'] and
            comparison['checkpoint_sha256'] == SPECS['e58_seed1']['sha256'], 'Val-best provenance differs')
    weights = {}
    val_prediction_sha = {}
    for key, spec in SPECS.items():
        path = args.project / spec['weights']
        require(path.is_file(), 'Missing trained val-best checkpoint: ' + str(path))
        require(sha(path) == spec['sha256'], 'Checkpoint differs from completed val-best: ' + key)
        weights[key] = str(path.resolve())
        val_pred = args.project / 'comparison_reports' / spec['val_report'] / 'area_predictions.json'
        require(val_pred.is_file(), 'Missing original val predictions for reproduction: ' + str(val_pred))
        val_prediction_sha[key] = sha(val_pred)
    records, meta = audit_dataset(args.dataset)
    env = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(int(args.device))
               if torch.cuda.is_available() else None)
    binding = dict(revision=REVISION, package=package['sha256_normalized'], source=source,
                   checkpoint_sha256={k: v['sha256'] for k, v in SPECS.items()},
                   val_prediction_sha256=val_prediction_sha, protocol=protocol(args.device),
                   dataset_annotation_sha256=TEST_ANN_SHA, dataset_image_sha256=TEST_IMAGE_SHA,
                   image_count=1610, environment=env, allocator_cap_GiB=6)
    cached = {k + '_seed1': validated_cached_reference(k, binding['protocol']) for k in ('e1', 'e40')}
    print('E0/E58 val-best SHA / seed1 / 640 / 200 epochs / original test-dev / cached references PASSED', flush=True)
    return records, meta, weights, binding, cached


def model_audit(model, key):
    import torch
    import spatial_task_head
    from ultralytics.nn.modules.head import Detect
    head = model.model.model[-1]
    spec = SPECS[key]
    require(head.nc == 10 and head.nl == spec['nl'] and tuple(head.stride.tolist()) == spec['stride'],
            'Wrong trained detector nc/nl/stride: ' + key)
    require(not head.end2end, 'Unexpected end-to-end detector')
    if key == 'e0_seed1':
        require(type(head) is Detect, 'E0 is not the original Detect head')
    else:
        require(type(head) is spatial_task_head.SpatialTaskDetect and head.enabled and
                getattr(head, 'experiment_variant', None) == 'e58', 'Wrong E58 custom head or disabled route')
        require(len(head.task_blocks) == 4 and all(type(b) is spatial_task_head.WideTaskBlock for b in head.task_blocks),
                'Wrong E58 four-scale W96 blocks')
        require(tuple(b.reduce[0].in_channels for b in head.task_blocks) == (128, 128, 256, 512) and
                all(b.reduce[0].out_channels == 96 for b in head.task_blocks), 'E58 width/channel drift')
    for name, parameter in model.model.named_parameters():
        require(torch.isfinite(parameter).all().item(), 'Non-finite checkpoint tensor: ' + name)
    return dict(head=type(head).__module__ + '.' + type(head).__name__, nc=head.nc,
                detection_scales=head.nl, stride=list(spec['stride']),
                parameter_count=sum(p.numel() for p in model.model.parameters()))


def convert_detections(record, data):
    """Standard Results.boxes.xyxy are already original-image coordinates."""
    answer = []
    for x1, y1, x2, y2, score, cls in data:
        import math
        require(all(math.isfinite(v) for v in (x1, y1, x2, y2, score, cls)), 'Non-finite prediction')
        require(cls == int(cls) and 0 <= cls < 10 and 0 <= score <= 1, 'Invalid score/class')
        require(x2 >= x1 and y2 >= y1, 'Reversed XYXY box')
        require(-.001 <= x1 <= x2 <= record.width + .001 and
                -.001 <= y1 <= y2 <= record.height + .001, 'Boxes are not original-image/clipped coordinates')
        # Preserve zero extents, matching the original reference evaluator.
        answer.append(dict(image_id=record.image_id, category_id=int(cls) + 1,
                           bbox=[x1, y1, max(0, x2-x1), max(0, y2-y1)], score=score))
    return answer


def predict(args, key, weights, records, out):
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils import LOGGER
    import logging
    mapping = {str(r.image_path.resolve()): r for r in records}
    require(len(mapping) == len(records), 'Duplicate input image paths')
    seen = set(); predictions = []; shapes = Counter()
    class NMSGuard(logging.Handler):
        exceeded = False
        def emit(self, item):
            if 'NMS time limit' in item.getMessage():
                self.exceeded = True
    class Predictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs):
            require(isinstance(preds, (tuple, list)) and len(preds) == 2 and
                    len(preds[1]) == SPECS[key]['nl'], 'Wrong decoded/raw head outputs')
            require(img.dtype == preds[0].dtype == torch.float32, 'Prediction is not FP32')
            require(torch.isfinite(preds[0]).all().item(), 'Non-finite raw predictions')
            shapes['x'.join(map(str, img.shape))] += 1
            results = super().postprocess(preds, img, orig_imgs)
            for result in results:
                path = str(Path(result.path).resolve())
                require(path in mapping, 'Unexpected result image path')
                record = mapping[path]
                require(record.image_id not in seen, 'Image predicted more than once')
                require(result.orig_shape == (record.height, record.width), 'Original image size differs')
                seen.add(record.image_id)
                predictions.extend(convert_detections(record, result.boxes.data.cpu().tolist()))
            return results
    model = YOLO(str(weights), task='detect')
    topology = model_audit(model, key)
    opts = options(args.device)
    predictor = Predictor(overrides={**model.overrides, **opts, 'mode': 'predict'}, _callbacks=model.callbacks)
    guard = NMSGuard(); LOGGER.addHandler(guard)
    started = time.monotonic()
    try:
        for start in range(0, len(records), 4):
            chunk = records[start:start+4]
            model.predict([str(r.image_path) for r in chunk], batch=len(chunk),
                          predictor=predictor, stream=False, **opts)
            require(not guard.exceeded, 'NMS time limit exceeded; refusing incomplete predictions')
            done = start + len(chunk)
            if start == 0 or done % 80 == 0 or done == len(records):
                elapsed = time.monotonic() - started
                mem = torch.cuda.memory_reserved(int(args.device)) / GIB if args.device != 'cpu' else 0
                print(f'{now()} {key} PREDICT {done}/{len(records)} elapsed={elapsed:.1f}s '
                      f'ETA={(len(records)-done)*elapsed/done:.1f}s reserved={mem:.2f}GiB', flush=True)
        require(seen == {r.image_id for r in records}, 'Missing images, including empty-prediction images')
        out.mkdir(parents=True, exist_ok=True)
        write(out / 'area_predictions.json', predictions)
        metadata = dict(status='completed', seen_images=len(seen), predictions=len(predictions),
                        sha256=sha(out / 'area_predictions.json'), actual_tensor_shapes=dict(shapes),
                        elapsed_seconds=time.monotonic()-started, finished_at=now(), topology=topology,
                        backend=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                     cudnn_allow_tf32=torch.backends.cudnn.allow_tf32))
        write(out / 'prediction_manifest.json', metadata)
        return predictions, metadata
    finally:
        LOGGER.removeHandler(guard)
        del predictor, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def score(records, out):
    import d27_official as official
    predictions = official.load_predictions(out / 'area_predictions.json', len(records))
    filtered, audit = official.filter_detections_and_audit(records, predictions)
    return official.official_evaluate(records, filtered), audit


def compare_predictions(actual, expected):
    import numpy as np
    require(len(actual) == len(expected), f'Val reproduction detection count differs: {len(actual)} vs {len(expected)}')
    require([(v['image_id'], v['category_id']) for v in actual] ==
            [(v['image_id'], v['category_id']) for v in expected], 'Val reproduction ID/class/order differs')
    if not actual:
        return dict(count=0, max_coordinate_error=0, max_score_error=0)
    a = np.asarray([v['bbox'] + [v['score']] for v in actual], dtype=np.float64)
    b = np.asarray([v['bbox'] + [v['score']] for v in expected], dtype=np.float64)
    coord_error = float(np.abs(a[:, :4]-b[:, :4]).max())
    score_error = float(np.abs(a[:, 4]-b[:, 4]).max())
    # Absolute tolerances only: relative tolerance on 1000px coordinates would
    # otherwise hide a sizable regression. Class/order/count must be exact.
    require(coord_error <= .002 and score_error <= 2e-5,
            f'Original val predictions did not reproduce: box={coord_error}, score={score_error}')
    return dict(count=len(actual), max_coordinate_error=coord_error, max_score_error=score_error,
                coordinate_atol=.002, score_atol=2e-5)


def units():
    import d27_official as official
    from types import SimpleNamespace
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import ops
    import torch
    get_cfg(overrides={**options('cpu'), 'mode': 'predict'})
    record = SimpleNamespace(image_id=2, width=100, height=80)
    converted = convert_detections(record, [[3., 7., 13., 27., .5, 0.], [5., 4., 5., 4., .25, 9.]])
    require(converted == [dict(image_id=2, category_id=1, bbox=[3., 7., 10., 20.], score=.5),
                          dict(image_id=2, category_id=10, bbox=[5., 4., 0, 0], score=.25)], 'Independent XYXY->COCO oracle failed')
    for original, resized in (((360, 640), (384, 640)), ((333, 517), (416, 640)), ((640, 640), (640, 640))):
        gain = min(resized[0]/original[0], resized[1]/original[1])
        px = round((resized[1]-original[1]*gain)/2-.1)
        py = round((resized[0]-original[0]*gain)/2-.1)
        original_box = torch.tensor([[20., 15., 100., 80.]])
        network_box = original_box*gain + torch.tensor([px, py, px, py])
        torch.testing.assert_close(ops.scale_boxes(resized, network_box, original), original_box, rtol=1e-5, atol=1e-4)
    compare_predictions(converted, converted)
    try:
        compare_predictions(converted, converted[:-1])
    except RuntimeError:
        pass
    else:
        raise RuntimeError('Reproduction mismatch was not rejected')
    return dict(official_semantics=official.run_unit_tests(), coordinate_conversion=True,
                zero_extent_retained=True, reproduction_rejects_mismatch=True, no_invalid_stream_override=True)


def gate_path(args, mode):
    return args.project / 'comparison_reports/testdev_e0_e58_gates' / (mode + '.json')


def save_gate(args, mode, binding, out):
    write(gate_path(args, mode), dict(status='passed', signature=digest(binding), binding=binding,
                                    report=str(out.resolve()), finished_at=now()))


def verify_gates(args, binding):
    for mode in ('check', 'smoke'):
        path = gate_path(args, mode)
        require(path.is_file(), 'Run ' + mode + ' first; missing ' + str(path))
        receipt = read(path)
        require(receipt['status'] == 'passed' and receipt['signature'] == digest(binding),
                'Stale/different code, weights, data or environment: repeat ' + mode)
    if binding.get('revision') == REVISION:
        print('E0/E58 CHECK AND REAL 8-IMAGE CUDA SMOKE RECEIPTS VALID', flush=True)
    else:
        print('LOCAL SYNTHETIC signature-gate test passed (NOT a server smoke receipt)', flush=True)


def cuda_limit(args, torch):
    require(torch.cuda.is_available(), 'Server smoke/full evaluation requires CUDA')
    device = int(args.device)
    free, total = torch.cuda.mem_get_info(device)
    require(free >= 8 * GIB, 'Need at least 8 GiB free before the sequential batch4 inference')
    fraction = 6 * GIB / total  # total is BYTES, never GiB. Tested independently.
    require(0 < fraction <= 1, 'Invalid memory cap fraction')
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    print(f'Sequential FP32 batch4; cap=6GiB free={free/GIB:.2f}GiB; NO TRAINING', flush=True)


def smoke(args, records, weights, binding, out):
    val, val_meta = load_dataset(args.val_dataset, 548)
    require(val_meta['annotation_dimension_sha256'] ==
            '058c87e779a61a09933bebe35af1c05c807a444dc842b4be12124dd36134c787', 'Wrong val data for reproduction')
    mapping = read(args.project / 'comparison_reports' / SPECS['e0_seed1']['val_report'] / 'area_ground_truth.json')['images']
    require(len(mapping) == len(val), 'Original val mapping length differs')
    for record, ref in zip(val, mapping):
        require((record.image_id, record.stem, record.width, record.height) ==
                (ref['id'], Path(ref['file_name']).stem, ref['width'], ref['height']), 'Original val ID mapping differs')
    results = {}
    for key in SPECS:
        actual, _ = predict(args, key, weights[key], val[:8], out / key / 'val_reproduction')
        all_expected = read(args.project / 'comparison_reports' / SPECS[key]['val_report'] / 'area_predictions.json')
        expected = [v for v in all_expected if v['image_id'] <= 8]
        reproduction = compare_predictions(actual, expected)
        del all_expected
        predict(args, key, weights[key], records[:8], out / key / 'testdev8')
        small_score, audit = score(records[:8], out / key / 'testdev8')
        results[key] = dict(val_reproduction=reproduction, testdev8_official=small_score, ignore_audit=audit)
    write(out / 'smoke.json', results)
    save_gate(args, 'smoke', binding, out)
    print('E0/E58 CUDA 8-IMAGE SMOKE AND ORIGINAL VAL PREDICTION REPRODUCTION PASSED', flush=True)


def full(args, records, meta, weights, binding, cached, out):
    verify_gates(args, binding)
    results = {}
    for key in SPECS:
        folder = out / key
        _, prediction_meta = predict(args, key, weights[key], records, folder)
        print(f'{now()} {key} OFFICIAL SCORING (CPU, all 10 classes)', flush=True)
        metrics, audit = score(records, folder)
        require(sha(weights[key]) == SPECS[key]['sha256'], 'Checkpoint changed during evaluation')
        result = dict(status='completed', official=metrics, ignore_audit=audit,
                      checkpoint=str(weights[key]), checkpoint_sha256=SPECS[key]['sha256'],
                      prediction_sha256=prediction_meta['sha256'], binding=binding,
                      provenance='new full test-dev prediction from frozen val-best')
        write(folder / 'official_metrics.json', result)
        prediction_meta['binding'] = binding
        write(folder / 'prediction_manifest.json', prediction_meta)
        results[key] = result
    all_models = dict(results, **cached)
    deltas = {f'e58_minus_{key}_pp': {m: results['e58_seed1']['official'][m]-v['official'][m] for m in KEYS}
              for key, v in all_models.items() if key != 'e58_seed1'}
    summary = dict(status='completed', revision=REVISION, split='test-dev', seed=1,
                   dataset=meta, binding=binding, models=all_models, deltas=deltas,
                   finished_at=now(),
                   boundary='One seed, val-selected checkpoints. E58-E0 is total gain, not new module-only gain. '
                            'Test-dev is a held-out evaluation, not a hyperparameter/checkpoint selection set. '
                            'No AP-small is inferred from overall AP; no stability or acceptance guarantee.')
    write(out / 'summary.json', summary)
    table = '|Model (seed1)|Official AP|AP50|AP75|AR500|Source|\n|---|---:|---:|---:|---:|---|\n'
    for key in ('e0_seed1', 'e1_seed1', 'e40_seed1', 'e58_seed1'):
        item = all_models[key]; v = item['official']
        table += f"|{key}|{v[KEYS[0]]:.4f}|{v[KEYS[1]]:.4f}|{v[KEYS[2]]:.4f}|{v[KEYS[3]]:.4f}|{'new' if key in results else 'frozen reference'}|\n"
    detail = '\n'.join(f"- {key}: AP {value[KEYS[0]]:+.4f} pp; AP50 {value[KEYS[1]]:+.4f}; AP75 {value[KEYS[2]]:+.4f}."
                       for key, value in deltas.items())
    (out / 'REPORT.md').write_text('# E0/E58 seed1: official VisDrone test-dev\n\n' + table + '\n' + detail +
                                 '\n\n' + summary['boundary'] + '\n', encoding='utf-8')
    write(out / 'COMPLETED.json', dict(status='completed', finished_at=now(), summary_sha256=sha(out / 'summary.json')))
    print(f'{now()} ALL COMPLETED REPORT={out / "REPORT.md"}\n' + detail, flush=True)


@contextmanager
def exclusive(args):
    import fcntl
    path = args.project / 'logs/testdev_e0_e58.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another E0/E58 evaluation controller is active; do not launch twice')
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'smoke', 'verify-gates', 'run', 'all'))
    parser.add_argument('--project', type=Path, default=Path('/root/autodl-tmp/yolo11_project'))
    parser.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/yolo11_src'))
    parser.add_argument('--dataset', type=Path, default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-test-dev'))
    parser.add_argument('--val-dataset', type=Path, default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val'))
    parser.add_argument('--device', default='0', choices=('0',))
    args = parser.parse_args()
    args.project = args.project.resolve(); args.source = args.source.resolve()
    with exclusive(args):
        modes = ('check', 'smoke', 'run') if args.mode == 'all' else (args.mode,)
        torch = prepare(args)
        records, meta, weights, binding, cached = preflight(args, torch)
        for mode in modes:
            out = args.project / 'comparison_reports' / ('TESTDEV_E0_E58_' + mode.upper().replace('-', '_') + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            out.mkdir(parents=True, exist_ok=False)
            print(f'{now()} {mode.upper()} START REPORT={out}', flush=True)
            try:
                write(out / 'input_manifest.json', dict(binding=binding, weights=weights, dataset=meta))
                if mode == 'check':
                    from ultralytics import YOLO
                    checks = units()
                    for key in SPECS:
                        model = YOLO(weights[key], task='detect')
                        checks[key] = model_audit(model, key)
                        del model
                    write(out / 'checks.json', checks)
                    save_gate(args, 'check', binding, out)
                    print('E0/E58 PACKAGE / COORDINATE / OFFICIAL / CHECKPOINT TOPOLOGY CHECK PASSED', flush=True)
                elif mode == 'verify-gates':
                    verify_gates(args, binding)
                else:
                    cuda_limit(args, torch)
                    if mode == 'smoke':
                        require(gate_path(args, 'check').is_file() and
                                read(gate_path(args, 'check'))['signature'] == digest(binding), 'Run check before smoke')
                        smoke(args, records, weights, binding, out)
                    else:
                        write(args.project / 'logs/latest_testdev_e0_e58.json', dict(report_dir=str(out), started_at=now()))
                        full(args, records, meta, weights, binding, cached, out)
                print(f'{now()} {mode.upper()} FINISHED; automatic shutdown DISABLED', flush=True)
            except BaseException:
                write(out / 'FAILED.json', dict(status='failed', at=now(), traceback=traceback.format_exc()))
                raise


if __name__ == '__main__':
    main()
