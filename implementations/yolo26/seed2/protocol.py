"""Seed2 serial orchestration; original seed1 numerical protocol is frozen."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import seed1_protocol as frozen
import common as c

write, now, NAMES = frozen.write, frozen.now, frozen.NAMES
DATA, N_SHA, VAL_SHA = frozen.DATA, frozen.N_SHA, frozen.VAL_SHA
CAP_GIB = 28
EVAL_CAP_GIB = 6
SEED = 2
VARIANTS = ('baseline', 'e60')
TEST_SHA = '8ac85f9cf40c98b0373281ccf5d3c902b7bb502492e51a48537249b0a1b8dffd'
IMAGE_SHA = '09a2d2daa96dc1ad5deb2311fb132edece644e554fd2c7bc6f0418384654f20b'
REF = c.HERE / 'references'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def options(args, name):
    value = frozen.options(args, name)
    value['seed'] = SEED
    return value


def run_name(variant):
    require(variant in VARIANTS, 'Unknown variant')
    return 'yolo26s_p2_' + variant + '_img640_seed2_200e'


def actual_run_name(args, variant):
    return run_name(variant) if not args.local_test else 'PRIVATE_yolo26_' + variant + '_seed2_2e'


def runtime(args, label):
    import os
    import sys
    require((args.source / 'ultralytics/__init__.py').is_file(), 'Missing pinned yolo26_src')
    sys.path.insert(0, str(args.source.resolve()))
    folder = args.project / 'environment_records' / ('yolo26_seed2_' + label) / 'Ultralytics'
    folder.mkdir(parents=True, exist_ok=True)
    # New isolated settings directories otherwise trigger a remote font download
    # when the native plots=True branch first renders training/validation images.
    import shutil
    font = c.HERE / 'assets/Arial.ttf'
    require(font.is_file(), 'Bundled offline plotting font missing')
    target_font = folder / 'Arial.ttf'
    if target_font.exists():
        require(c.sha256(target_font) == c.sha256(font), 'Existing plotting font differs; will not overwrite')
    else:
        shutil.copyfile(font,target_font)
    os.environ['YOLO_CONFIG_DIR'] = str(folder.parent)
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_NUM_THREADS'] = '1'
    if args.local_test:
        require(args.project.resolve() != c.PROJECT.resolve(), 'Private tests cannot write into server project')


def load_split(root, count):
    import d27_official as official
    old = official.EXPECTED_IMAGES
    try:
        official.EXPECTED_IMAGES = count
        return official.load_dataset(root)
    finally:
        official.EXPECTED_IMAGES = old


def test_dataset(args):
    count = len(list((args.dataset / 'annotations').glob('*.txt'))) if args.local_test else 1610
    records, meta = load_split(args.dataset, count)
    if not args.local_test:
        require(meta['annotation_dimension_sha256'] == TEST_SHA, 'Wrong original test-dev annotations')
        reference = read(REF / 'testdev_input_manifest.json')
        require(reference['dataset']['image_content_sha256'] == IMAGE_SHA and
                len(reference['images']) == len(records) == 1610, 'Wrong test-dev reference')
        from PIL import Image
        for i, (record, ref) in enumerate(zip(records, reference['images']), 1):
            require((record.image_id, record.stem, record.width, record.height) ==
                    (ref['id'], ref['stem'], ref['width'], ref['height']), 'Test-dev image mapping differs')
            require(c.sha256(record.image_path) == ref['sha256'], 'Test-dev image content differs: ' + record.stem)
            with Image.open(record.image_path) as image:
                require(image.getexif().get(274, 1) in (1, None), 'EXIF rotated test-dev image')
            if i % 400 == 0 or i == 1610:
                print(f'{now()} TEST-DEV audit {i}/1610', flush=True)
        meta['image_content_sha256'] = IMAGE_SHA
    meta['split'] = 'test-dev'
    return records, meta


def source_audit(args):
    evidence = c.audit(args.source, args.local_test)
    evidence['training_source_files_checked'] = frozen.audit_training_source(args.source)
    if not args.local_test:
        import ultralytics
        c.resource_guard()
        require(ultralytics.__version__ == '8.4.177', 'Wrong Ultralytics version')
    return evidence


def signature(args):
    import torch
    evidence = source_audit(args)
    require(c.sha256(args.project / 'weights/yolo26s.pt') == c.WEIGHT_SHA, 'Wrong original COCO/Objects365 checkpoint')
    frozen.deploy_amp_weight(args)
    import d27_official as official
    old = official.EXPECTED_IMAGES
    try:
        if args.local_test:
            official.EXPECTED_IMAGES = len(list((args.val_dataset / 'annotations').glob('*.txt')))
        dataset = frozen.dataset_audit(args)
    finally:
        official.EXPECTED_IMAGES = old
    if not args.local_test:
        original = read(REF / 'seed1_signature.json')['payload']
        require(c.sha256(args.data) == original['data_yaml'], 'Training YAML differs from seed1')
        for split in ('train', 'val'):
            require(dataset[split] == original['dataset'][split], 'Training/val data differs from seed1: ' + split)
    test_records, test_meta = test_dataset(args)
    val_records, _ = load_split(args.val_dataset, dataset['official_val']['images'])
    require(not {r.stem for r in val_records}.intersection(r.stem for r in test_records), 'val/test-dev ID overlap')
    require(torch.cuda.is_available(), 'CUDA required; no CPU or no-GPU-mode fallback')
    value = dict(package=c.sha256(c.HERE / 'manifest.json'), source=c.sha256(c.HERE / 'training_source_sha256.json'),
                 versions=evidence['versions'], torch_cuda=torch.version.cuda, data_yaml=c.sha256(args.data),
                 dataset=dataset, testdev=test_meta, pretrained=c.WEIGHT_SHA,
                 seed=SEED, layout='serial', train_batch=16, validation_batch=32, prediction_batch=4,
                 allocator_cap_GiB=4 if args.local_test else CAP_GIB,
                 imgsz=args.local_size if args.local_test else 640, epochs=args.local_epochs if args.local_test else 200,
                 source_actual=str(args.source.resolve()), local_fixture=args.local_test,
                 device=dict(name=torch.cuda.get_device_name(0), total_bytes=torch.cuda.get_device_properties(0).total_memory))
    fingerprint = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return dict(fingerprint=fingerprint, payload=value)


def gate_path(args):
    prefix = 'PRIVATE_' if args.local_test else ''
    return args.project / 'comparison_reports' / (prefix + 'yolo26_p2_e60_seed2_serial_gate.json')


def issue_gate(args, sig, report):
    check = read(report / 'check/check.json')
    require(check['status'] == 'engineering_passed' and check['cuda_executed'], 'CUDA check incomplete')
    files = {'check': str(report / 'check/check.json')}
    for variant in VARIANTS:
        path = report / (variant + '_smoke') / 'smoke.json'
        smoke = read(path)
        require(smoke['status'] == 'passed' and smoke['variant'] == variant and
                smoke['local_fixture'] == args.local_test and smoke['successful_updates'] >= 8 and
                smoke['train_batch'] == 16 and smoke['validation_batch'] == 32 and
                smoke['validation_images'] == sig['payload']['dataset']['val']['images'], 'Real smoke incomplete: ' + variant)
        require(smoke['imgsz'] == sig['payload']['imgsz'] and smoke['native_validator_complete'], 'Wrong smoke shape/validation')
        if variant == 'e60':
            require(smoke['gradient_groups'] == smoke['actual_update_groups'] == 52, 'E60 did not update all52 groups')
        files[variant] = str(path)
    receipt = dict(status='passed', seed=SEED, layout='serial', local_fixture=args.local_test,
                   fingerprint=sig['fingerprint'], files=files,
                   sha256={k:c.sha256(v) for k,v in files.items()}, finished_at=now())
    write(gate_path(args), receipt)
    return receipt


def verify_gate(args, sig):
    value = read(gate_path(args))
    require(value['status'] == 'passed' and value['fingerprint'] == sig['fingerprint'] and
            value['seed'] == SEED and value['layout'] == 'serial' and
            value['local_fixture'] == args.local_test, 'Missing/stale/wrong-layout seed2 gates; rerun all, no bypass')
    require(not value['local_fixture'] or args.local_test, 'Private receipt cannot authorize formal training')
    for label, path in value['files'].items():
        require(c.sha256(path) == value['sha256'][label], 'Check/smoke receipt changed: ' + label)
    return value


def training_weights(args, train_report, variant):
    result = read(Path(train_report) / (variant + '_train') / 'training_completion.json')
    epochs = args.local_epochs if args.local_test else 200
    expected = args.project / 'runs' / actual_run_name(args, variant)
    require(result['status'] == 'training_completed' and result['epochs'] == epochs and
            result['variant'] == variant and result['seed'] == SEED and
            Path(result['run']).resolve() == expected.resolve(), 'Not the completed matching seed2 training')
    best = expected / 'weights/best.pt'
    require(Path(result['best']).resolve() == best.resolve() and c.sha256(best) == result['best_sha256'],
            'Selected seed2 best.pt changed')
    return best


def evaluate(args, variant, weights, out, split):
    import d27_official as official
    from scoring26 import predict
    official.run_unit_tests()
    if split == 'val':
        count = len(list((args.val_dataset / 'annotations').glob('*.txt'))) if args.local_test else 548
        records, meta = load_split(args.val_dataset, count)
        require(args.local_test or meta['annotation_dimension_sha256'] == VAL_SHA, 'Wrong validation annotations')
        meta['split'] = 'val'
    else:
        records, meta = test_dataset(args)
    before = c.sha256(weights)
    prediction = predict(args, variant, weights, out, records)
    require(c.sha256(weights) == before == prediction['checkpoint_sha256'], 'Checkpoint changed during inference')
    predictions_path = out / 'area_predictions.json'
    print(f'{now()} {variant.upper()} {split.upper()} OFFICIAL SCORING CPU all10 classes', flush=True)
    filtered, ignore = official.filter_detections_and_audit(records, official.load_predictions(predictions_path, len(records)))
    metrics = official.official_evaluate(records, filtered)
    require(all(math.isfinite(metrics[k]) for k in ('AP50_95_percent','AP50_percent','AP75_percent','AR500_percent')),
            'Nonfinite official metric')
    result = dict(status='completed', variant=variant, seed=SEED, split=split, dataset=meta,
                  official_metrics=metrics, prediction=prediction, prediction_sha256=c.sha256(predictions_path),
                  checkpoint=str(weights), checkpoint_sha256=before, ignore=ignore,
                  scorer_revision=official.REVISION, scorer_sha256=c.sha256(Path(official.__file__)),
                  selection='fixed val-best; no test-dev selection', local_fixture=args.local_test,
                  protocol=dict(imgsz=args.local_size if args.local_test else 640, batch=4, half=False,
                                one2many_NMS=True, conf=.001, iou=.7, max_det=500), finished_at=now())
    write(out / 'official_metrics.json', result)
    print(f'{now()} {variant.upper()} {split.upper()} AP={metrics["AP50_95_percent"]:.4f}%', flush=True)
    return result
