"""Frozen YOLO26s-P2 seed1 pair -> VisDrone test-dev. NO TRAINING ENTRY."""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

HERE = Path(__file__).resolve().parent
REF = HERE / 'references'
PROJECT = Path('/root/autodl-tmp/yolo26_project')
SOURCE = Path('/root/autodl-tmp/yolo26_src')
DATA = Path('/root/autodl-tmp/project/VisDrone2019')
KEYS = ('AP50_95_percent', 'AP50_percent', 'AP75_percent', 'AR500_percent')
VARIANTS = ('baseline', 'e60')
VAL_SHA = '058c87e779a61a09933bebe35af1c05c807a444dc842b4be12124dd36134c787'
TEST_SHA = '8ac85f9cf40c98b0373281ccf5d3c902b7bb502492e51a48537249b0a1b8dffd'
IMAGE_SHA = '09a2d2daa96dc1ad5deb2311fb132edece644e554fd2c7bc6f0418384654f20b'
CAP_GIB = 6


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def protocol():
    return dict(imgsz=640, batch=4, half=False, conf=.001, iou=.7, max_det=500,
                rect=True, augment=False, agnostic_nms=False, nms=True,
                branch='one2many+NMS', end2end=False,
                matmul_tf32=False, cudnn_tf32=False, cudnn_benchmark=False,
                cudnn_deterministic=True, checkpoint_selection='fixed val-selected best.pt',
                scoring='VisDrone2019 DET Matlab1.0.4 semantic port, class0 ignore')


def runtime(args):
    # Select the explicit YOLO26 source BEFORE any torch/ultralytics import.
    require((args.source / 'ultralytics/__init__.py').is_file(), 'Missing yolo26_src')
    sys.path.insert(0, str(args.source.resolve()))
    settings = args.project / 'environment_records/ultralytics_settings'
    (settings / 'Ultralytics').mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(settings)
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_NUM_THREADS'] = '1'


def package_audit():
    manifest = read(HERE / 'manifest.json')
    for relative, digest in manifest.items():
        path = (HERE / relative).resolve()
        require(path.is_relative_to(HERE), 'Unsafe package manifest path')
        require(sha(path) == digest, 'Package file changed: ' + relative)
    return sha(HERE / 'manifest.json')


def environment_audit(args):
    runtime(args)
    package = package_audit()
    import common as common
    evidence = common.audit(args.source, local=False)
    common.resource_guard()
    from protocol import audit_training_source
    evidence['training_source_files_checked'] = audit_training_source(args.source)
    require(evidence['source'] == str((args.source / 'ultralytics/__init__.py').resolve()),
            'Wrong YOLO source imported')
    import importlib.metadata
    require(importlib.metadata.version('ultralytics') == '8.4.177', 'Wrong Ultralytics version')
    return dict(evidence, package_sha256=package, training_started=False)


def checkpoint_audit(project):
    specs = read(REF / 'checkpoint_specs.json')
    result = {}
    for variant in VARIANTS:
        spec = specs[variant]
        completed = read(REF / (variant + '_training_completion.json'))
        metric = read(REF / (variant + '_val_official_metrics.json'))
        require(completed['status'] == 'training_completed' and completed['epochs'] == 200
                and completed['variant'] == variant, 'Incomplete original formal training')
        require(metric['status'] == 'completed' and metric['variant'] == variant
                and metric['seed'] == 1, 'Wrong original official val result')
        require(completed['best_sha256'] == metric['checkpoint_sha256'] == spec['sha256'],
                'Checkpoint provenance mismatch: ' + variant)
        require(metric['dataset']['images'] == 548 and
                metric['dataset']['annotation_dimension_sha256'] == VAL_SHA,
                'Original result is not the fixed official val split')
        path = project / spec['relative_path']
        require(path.is_file(), 'Missing trained checkpoint: ' + str(path))
        require(sha(path) == spec['sha256'], 'Wrong/damaged val-best checkpoint: ' + variant)
        result[variant] = dict(path=str(path.resolve()), sha256=spec['sha256'])
    return result


def load_split(root, count):
    import d27_official as official
    old = official.EXPECTED_IMAGES
    try:
        official.EXPECTED_IMAGES = count
        return official.load_dataset(root)
    finally:
        official.EXPECTED_IMAGES = old


def datasets_audit(args):
    val, vm = load_split(args.val_dataset, 548)
    records, meta = load_split(args.dataset, 1610)
    require(vm['annotation_dimension_sha256'] == VAL_SHA, 'Original val annotations changed')
    require(meta['annotation_dimension_sha256'] == TEST_SHA, 'Original test-dev annotations changed')
    require(not {r.stem for r in val}.intersection(r.stem for r in records),
            'val/test-dev IDs overlap')
    frozen = read(REF / 'testdev_input_manifest.json')
    require(frozen['dataset']['image_content_sha256'] == IMAGE_SHA and
            len(frozen['images']) == len(records) == 1610, 'Frozen test-dev reference differs')
    for index, (record, expected) in enumerate(zip(records, frozen['images']), 1):
        require((record.image_id, record.stem, record.width, record.height) ==
                (expected['id'], expected['stem'], expected['width'], expected['height']),
                'Test-dev image mapping changed: ' + record.stem)
        require(sha(record.image_path) == expected['sha256'], 'Test-dev image bytes changed: ' + record.stem)
        if index % 400 == 0 or index == 1610:
            print(f'{now()} TEST-DEV SHA audit {index}/1610', flush=True)
    from PIL import Image
    for record in val[:8] + records:
        with Image.open(record.image_path) as image:
            require(image.getexif().get(274, 1) in (1, None), 'EXIF rotation: ' + record.stem)
    meta.update(split='test-dev', image_content_sha256=IMAGE_SHA)
    return val, records, dict(val=vm, testdev=meta)


def model_extra_audit(model, variant):
    import torch
    head = model.model[-1]
    require(head.stride.tolist() == [4., 8., 16., 32.], 'Wrong strides')
    require(hasattr(head, 'one2one_cv2') and head.one2one_cv2 is not None, 'Missing native dual head')
    if variant == 'e60':
        import e60_adapter as adapter
        require(head.enabled and tuple(b.reduce[0].in_channels for b in head.task_blocks) == adapter.CHANNELS,
                'Disabled or incompatible E60 head')
        require(len(head.task_blocks) == 4 and all(b.reduce[0].out_channels == 96 for b in head.task_blocks),
                'Wrong E60 widths/scales')
        for index, block in enumerate(head.task_blocks):
            require(len(block.depths) == 3, 'Wrong depth bank')
            if index < 2:
                require(len(block.refinements) == 3, 'Missing P2/P3 refinements')
                for refine in block.refinements:
                    require(type(refine) is adapter.GroupResidualEncoder and
                            len(refine.kernels) == 3 and all(p.dtype == torch.float32 for p in refine.parameters()),
                            'Group encoding topology/FP32 protection changed')
            else:
                require(not hasattr(block, 'refinements'), 'This is not the frozen E60 (P4/P5 refinements found)')
    for name, parameter in model.named_parameters():
        require(torch.isfinite(parameter).all().item(), 'Nonfinite checkpoint tensor: ' + name)
    return dict(head=type(head).__module__ + '.' + type(head).__name__,
                parameters=sum(p.numel() for p in model.parameters()), stride=head.stride.tolist())


def assert_val_predictions(actual, expected):
    """Strict class/order/count guard; small FP32 numeric tolerance is not an AP tolerance."""
    import numpy as np
    require(len(actual) == len(expected), 'Val reproduction detection count differs')
    largest = dict(box_abs=0., score_abs=0.)
    for new, old in zip(actual, expected):
        require((new['image_id'], new['category_id']) == (old['image_id'], old['category_id']),
                'Val reproduction image/class/order differs')
        boxes = np.asarray(new['bbox']) - np.asarray(old['bbox'])
        box_error = float(np.abs(boxes).max())
        score_error = abs(new['score'] - old['score'])
        require(box_error <= .002 and score_error <= 2e-6, 'Val prediction values differ from completed seed1')
        largest['box_abs'] = max(largest['box_abs'], box_error)
        largest['score_abs'] = max(largest['score_abs'], score_error)
    return dict(status='passed', detections=len(actual), max_errors=largest,
                box_atol=.002, score_atol=2e-6, class_count_order_exact=True)


def init_gpu():
    import torch
    require(torch.cuda.is_available(), 'CUDA GPU required; do not run this in no-GPU 2GiB mode')
    free, total = torch.cuda.mem_get_info(0)
    require(free >= CAP_GIB * 2**30 and total >= CAP_GIB * 2**30,
            f'Need at least {CAP_GIB}GiB free VRAM for evaluation; free={free / 2**30:.2f}GiB')
    torch.cuda.set_per_process_memory_fraction(CAP_GIB * 2**30 / total, 0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return dict(device=torch.cuda.get_device_name(0), allocator_cap_GiB=CAP_GIB,
                torch=torch.__version__, cuda=torch.version.cuda)


def predict_fixed(args, variant, weights, out, records):
    # No change to the original validated Predictor/NMS/round-trip coordinate audit.
    import scoring26
    import torch
    from ultralytics import YOLO
    audit_model = YOLO(str(weights), task='detect').model
    scoring26.check_checkpoint(audit_model, variant)
    topology = model_extra_audit(audit_model, variant)
    del audit_model
    gc.collect()
    result = scoring26.predict(args, variant, weights, out, records)
    result.update(topology=topology, half=False, end2end=False,
                  predictions_sha256=sha(Path(out) / 'area_predictions.json'))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def score_fixed(records, out):
    import d27_official as official
    out = Path(out)
    print(f'{now()} OFFICIAL SCORING CPU all10 classes; may be quiet between class messages', flush=True)
    grouped = official.load_predictions(out / 'area_predictions.json', len(records))
    filtered, ignore = official.filter_detections_and_audit(records, grouped)
    metrics = official.official_evaluate(records, filtered)
    require(all(math.isfinite(metrics[key]) for key in KEYS), 'Nonfinite official metric')
    return metrics, ignore


def worker(args):
    require(args.variant in VARIANTS and args.out is not None, 'Use controller for evaluation')
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        environment = environment_audit(args)
        checkpoints = checkpoint_audit(args.project)
        gpu = init_gpu()
        import d27_official as official
        official.run_unit_tests()
        weights = Path(checkpoints[args.variant]['path'])
        if args.worker_stage == 'smoke':
            val, vm = load_split(args.val_dataset, 548)
            require(vm['annotation_dimension_sha256'] == VAL_SHA, 'Wrong val split')
            predict_fixed(args, args.variant, weights, args.out / 'val_reproduction', val[:8])
            reproduction = assert_val_predictions(read(args.out / 'val_reproduction/area_predictions.json'),
                                                  read(REF / (args.variant + '_val_first8_predictions.json')))
            records, tm = load_split(args.dataset, 1610)
            require(tm['annotation_dimension_sha256'] == TEST_SHA, 'Wrong test-dev split')
            prediction = predict_fixed(args, args.variant, weights, args.out / 'testdev_first8', records[:8])
            smoke_metrics, _ = score_fixed(records[:8], args.out / 'testdev_first8')
            value = dict(status='passed', full_testdev=False, val_reproduction=reproduction,
                         testdev_smoke_images=8, prediction=prediction, smoke_only_metrics=smoke_metrics)
        else:
            records, meta = load_split(args.dataset, 1610)
            require(meta['annotation_dimension_sha256'] == TEST_SHA, 'Wrong test-dev split')
            meta.update(split='test-dev', image_content_sha256=IMAGE_SHA)
            prediction = predict_fixed(args, args.variant, weights, args.out, records)
            metrics, ignore = score_fixed(records, args.out)
            value = dict(status='completed', full_testdev=True, seed=1, variant=args.variant,
                         dataset=meta, protocol=protocol(), prediction=prediction, official_metrics=metrics,
                         ignore=ignore, scorer_revision=official.REVISION,
                         scorer_sha256=sha(HERE / 'd27_official.py'), checkpoint=checkpoints[args.variant],
                         environment=environment, gpu=gpu, finished_at=now(), training_started=False)
            write(args.out / 'official_metrics.json', value)
            print(f'{now()} {args.variant.upper()} TEST-DEV AP={metrics["AP50_95_percent"]:.4f}%', flush=True)
        value.update(finished_at=now(), environment=environment, gpu=gpu, training_started=False)
        write(args.out / 'completion.json', value)
    except BaseException:
        write(args.out / 'failure.json', dict(status='failed', error=traceback.format_exc(), finished_at=now()))
        raise


def summarize(report, dataset):
    results = {v: read(report / (v + '_testdev') / 'official_metrics.json') for v in VARIANTS}
    for variant, result in results.items():
        require(result['status'] == 'completed' and result['full_testdev'] and result['seed'] == 1
                and result['variant'] == variant and result['dataset']['images'] == 1610,
                'Not a complete same-seed test-dev pair')
    require(results['baseline']['dataset'] == results['e60']['dataset'] and
            results['baseline']['protocol'] == results['e60']['protocol'], 'Different pair data/protocol')
    delta = {k: results['e60']['official_metrics'][k] - results['baseline']['official_metrics'][k] for k in KEYS}
    write(report / 'comparison.json', dict(status='completed', split='test-dev', seed=1,
          models=results, delta_pp=delta, dataset=dataset, training_started=False,
          single_seed_not_stability=True, no_test_based_selection=True))
    lines = ['# YOLO26s-P2 / E60 / seed1 / official test-dev', '',
             '|Metric (%)|YOLO26s-P2 baseline|YOLO26s-P2 + E60|Delta pp|', '|---|---:|---:|---:|']
    for key in KEYS:
        lines.append(f'|{key}|{results["baseline"]["official_metrics"][key]:.4f}|'
                     f'{results["e60"]["official_metrics"][key]:.4f}|{delta[key]:+.4f}|')
    lines += ['', '1610 test-dev images; both fixed validation-selected best checkpoints; no retraining.',
              'FP32 / 640 / batch4 / one2many+NMS / conf=.001 / IoU=.7 / max_det500.',
              'Same frozen VisDrone Matlab1.0.4 semantic-port scorer as prior official validation.',
              'Each model reproduced the original first8 val predictions before full test-dev evaluation.',
              'All signed results retained. This one seed does not establish cross-seed stability.',
              'Do not select epochs, change thresholds or choose architecture using this test-dev result.',
              'Automatic shutdown DISABLED; no server training/environment modifications.', '']
    (report / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'{now()} DELTA pp={json.dumps(delta)}', flush=True)


def controller(args):
    args.project.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.project / '.yolo26_testdev_seed1.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Another seed1 test-dev controller is active; do not start twice')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    report = args.project / 'comparison_reports' / ('TESTDEV_YOLO26_P2_E60_S1_' + stamp)
    session = args.project / 'logs' / ('testdev_yolo26_p2_e60_s1_' + stamp)
    report.mkdir(parents=True, exist_ok=False)
    session.mkdir(parents=True, exist_ok=False)
    state = dict(status='preflight', stage='check', mode=args.mode, session=str(session), report=str(report),
                 started_at=now(), auto_shutdown=False, training_started=False, jobs={})
    write(args.project / 'logs/latest_yolo26_testdev_seed1.json', dict(session=str(session), report=str(report)))

    def save():
        write(session / 'state.json', state)
        write(report / 'state.json', state)

    def job(stage, variant):
        key = variant + ('_smoke' if stage == 'smoke' else '_testdev')
        log = session / (key + '.log')
        command = [sys.executable, '-u', str(HERE / 'run_eval.py'), args.mode,
                   '--worker-stage', stage, '--variant', variant, '--out', str(report / key),
                   '--project', str(args.project), '--source', str(args.source),
                   '--dataset', str(args.dataset), '--val-dataset', str(args.val_dataset)]
        state['stage'] = key
        print(f'{now()} START {key}; log={log}', flush=True)
        with log.open('w', encoding='utf-8') as handle:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding='utf-8', errors='replace', bufsize=1)
            state['jobs'][key] = dict(pid=proc.pid, status='running', log=str(log), started_at=now())
            save()
            try:
                for line in proc.stdout:
                    handle.write(line)
                    handle.flush()
                    print(line, end='', flush=True)
                code = proc.wait()
            except BaseException:
                if proc.poll() is None:
                    proc.terminate()
                proc.wait()
                raise
        state['jobs'][key].update(status='completed' if code == 0 else 'failed', exit_code=code, finished_at=now())
        save()
        require(code == 0, f'{key} failed (exit={code}); see {log}; no bypass or automatic retry')

    try:
        save()
        environment = environment_audit(args)
        checkpoints = checkpoint_audit(args.project)
        _, _, dataset = datasets_audit(args)
        write(report / 'preflight.json', dict(status='passed', environment=environment,
              checkpoints=checkpoints, dataset=dataset, protocol=protocol(), training_started=False))
        with zipfile.ZipFile(report / 'actual_source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in list(read(HERE / 'manifest.json')) + ['manifest.json']:
                archive.write(HERE / name, name)
        print(f'{now()} CHECK PASSED; fixed YOLO26 environment / val-best / original test-dev verified', flush=True)
        if args.mode != 'check':
            state['status'] = 'running'
            for variant in VARIANTS:
                job('smoke', variant)
            print(f'{now()} BOTH SMOKES PASSED; original val predictions reproduced', flush=True)
            if args.mode == 'all':
                for variant in VARIANTS:
                    job('eval', variant)
                summarize(report, dataset['testdev'])
        state.update(status='completed', stage='finished', finished_at=now())
        save()
        print(f'{now()} COMPLETED REPORT={report}; automatic shutdown DISABLED', flush=True)
    except BaseException:
        state.update(status='failed', error=traceback.format_exc(), finished_at=now())
        save()
        raise
    finally:
        import shutil
        for log in session.glob('*.log'):
            shutil.copyfile(log, report / log.name)
        lock.close()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'smoke', 'all'), nargs='?', default='all')
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--dataset', type=Path, default=DATA / 'VisDrone2019-DET-test-dev')
    parser.add_argument('--val-dataset', type=Path, default=DATA / 'VisDrone2019-DET-val')
    parser.add_argument('--worker-stage', choices=('smoke', 'eval'), help=argparse.SUPPRESS)
    parser.add_argument('--variant', choices=VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument('--out', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.local_test = False
    args.local_size = 640
    return args


if __name__ == '__main__':
    current = arguments()
    if current.worker_stage:
        worker(current)
    else:
        controller(current)
