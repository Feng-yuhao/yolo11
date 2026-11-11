"""Fixed official VisDrone2019 ignore-region AP; val-best only, no test tuning."""
from __future__ import annotations
from collections import Counter
import csv
import gc
import logging
from pathlib import Path
import time
from shutdown_safety import read, write, sha, now

HERE = Path(__file__).resolve().parent
REF = HERE / 'references'
KEYS = ('AP50_95_percent','AP50_percent','AP75_percent','AR500_percent')
E1_SHA = 'f1a5ba6642a01445a99b2c7c80d8e79299a4402159bc37d2a08bb31ff365df12'
E60_SHA = 'bf38299c8723f5fe399e2f94fa4042df2ff6ae51dcb99fceece8bc85ed00dc17'
TEST_ANN = '8ac85f9cf40c98b0373281ccf5d3c902b7bb502492e51a48537249b0a1b8dffd'
TEST_IMAGES = '09a2d2daa96dc1ad5deb2311fb132edece644e554fd2c7bc6f0418384654f20b'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def options(device):
    return dict(imgsz=640, device=device, conf=.001, iou=.7, max_det=500,
                half=False, augment=False, verbose=False, save=False,
                agnostic_nms=False, classes=None)


def audit_dataset(args, split):
    import d27_official as official
    count = 548 if split == 'val' else 1610
    path = args.dataset if split == 'val' else args.test_dataset
    previous = official.EXPECTED_IMAGES
    try:
        official.EXPECTED_IMAGES = count
        records, meta = official.load_dataset(path)
    finally:
        official.EXPECTED_IMAGES = previous
    if split == 'val':
        expected = read(REF / 'val_e1_e60.json')['dataset']
        require({k:v for k,v in expected.items() if k != 'dataset_root'} ==
                {k:v for k,v in meta.items() if k != 'dataset_root'}, 'Original val annotation/dimension mapping changed')
    else:
        require(meta['annotation_dimension_sha256'] == TEST_ANN, 'Wrong test-dev original annotations')
        manifest = read(REF / 'testdev_input_manifest.json')
        require(manifest['dataset']['image_content_sha256'] == TEST_IMAGES, 'Cached test image fingerprint changed')
        require(len(records) == len(manifest['images']) == 1610, 'Wrong test-dev size')
        for record, image in zip(records, manifest['images']):
            require((record.image_id, record.stem, record.width, record.height) ==
                    (image['id'],image['stem'],image['width'],image['height']), 'Test image ID/dimension mapping differs')
            require(sha(record.image_path) == image['sha256'], 'Test image bytes differ: ' + record.stem)
        meta['image_content_sha256'] = TEST_IMAGES
    return records, meta


def cached_references(split):
    """Reuse original same-seed results with complete bindings; not new inference."""
    val = read(REF / 'val_e1_e60.json')
    base = read(REF / 'e1_training.json')
    parent = read(REF / 'e60_training.json')
    require(val['status'] == base['status'] == parent['status'] == 'completed', 'Incomplete cached formal training')
    require(base['seed'] == parent['params']['seed'] == val['seed'] == 1, 'Cached seeds differ')
    require(base['training']['completed_epochs'] == parent['completed_epochs'] == 200, 'Cached training is not formal200')
    require(val['checkpoint_sha256'] == E60_SHA and base['evaluation_weights']['sha256'] == E1_SHA,
            'Cached reference checkpoint changed')
    result = {}
    for name, weight in (('e1', E1_SHA), ('e60', E60_SHA)):
        if split == 'val':
            metrics = val['baseline' if name == 'e1' else 'candidate']
            provenance = dict(checkpoint_sha256=weight, original_report='references/val_e1_e60.json',
                              dataset=val['dataset'], scorer='unchanged_D27_Matlab104_ignore_port')
        else:
            metric = read(REF / (name + '_seed1_official_metrics.json'))
            pred = read(REF / (name + '_seed1_prediction_manifest.json'))
            binding = metric['binding']
            require(metric['status'] == pred['status'] == 'completed' and binding == pred['binding'],
                    'Cached test reference incomplete: ' + name)
            saved_weight = binding['checkpoint_sha256']
            if isinstance(saved_weight, dict):
                saved_weight = saved_weight[name + '_seed1']
            require(saved_weight == weight and binding['image_count'] == pred['seen_images'] == 1610 and
                    not binding.get('smoke_only', False), 'Cached test checkpoint/count differs: ' + name)
            require(binding['dataset_annotation_sha256'] == TEST_ANN and binding['dataset_image_sha256'] == TEST_IMAGES,
                    'Cached test dataset differs: ' + name)
            for key, expected in options('cpu').items():
                if key != 'device':
                    require(binding['protocol'][key] == expected, 'Cached test protocol differs: ' + key)
            require(binding['protocol']['batch'] == 4 and metric['prediction_sha256'] == pred['sha256'],
                    'Cached test predictions/batch differ')
            require(pred['backend']['matmul_allow_tf32'] is False and pred['backend']['cudnn_allow_tf32'] is True,
                    'Cached reference TF32 policy differs')
            metrics = metric['official']
            provenance = dict(binding=binding, prediction_sha256=metric['prediction_sha256'],
                              original_report='references/' + name + '_seed1_official_metrics.json')
        result[name] = dict(official=metrics, provenance=provenance,
                            inference_rerun=False, seed=1, checkpoint_sha256=weight)
    return result


def convert(record, data):
    import math
    answer = []
    for x1,y1,x2,y2,score,cls in data:
        require(all(math.isfinite(v) for v in (x1,y1,x2,y2,score,cls)), 'Nonfinite prediction')
        require(cls == int(cls) and 0 <= cls < 10 and 0 <= score <= 1, 'Invalid prediction class/score')
        require(-.001 <= x1 <= x2 <= record.width+.001 and -.001 <= y1 <= y2 <= record.height+.001,
                'Predictions are not clipped original-image coordinates')
        answer.append(dict(image_id=record.image_id, category_id=int(cls)+1,
                           bbox=[x1,y1,max(0.,x2-x1),max(0.,y2-y1)], score=score))
    return answer


def predict(args, weights, records, out, coordinate_check_images=8):
    import torch
    import numpy as np
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils import LOGGER, ops
    from ultralytics.utils.torch_utils import init_seeds
    import tood_td_head
    init_seeds(1, deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    mapping = {str(r.image_path.resolve()): r for r in records}
    require(len(mapping) == len(records), 'Duplicate image paths')
    seen, predictions, shapes = set(), [], Counter()
    checked, coordinate_error = 0, 0.

    class NMSGuard(logging.Handler):
        exceeded = False
        def emit(self, item):
            if 'NMS time limit' in item.getMessage():
                self.exceeded = True

    class Predictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs):
            nonlocal checked, coordinate_error
            require(isinstance(preds, (tuple,list)) and len(preds) == 2 and len(preds[1]) == 4,
                    'Wrong four-scale raw/decoded prediction interface')
            require(img.dtype == preds[0].dtype == torch.float32 and torch.isfinite(preds[0]).all(),
                    'Official prediction must be finite FP32')
            shapes['x'.join(map(str, img.shape))] += 1
            # Preserve raw input before the native NMS can mutate its box view.
            raw = preds[0].detach().clone() if checked < coordinate_check_images else None
            results = super().postprocess(preds, img, orig_imgs)
            oracle = ops.non_max_suppression(raw.clone(), .001, .7, max_det=500) if raw is not None else None
            for i, result in enumerate(results):
                path = str(Path(result.path).resolve())
                require(path in mapping, 'Unexpected image result')
                record = mapping[path]
                require(record.image_id not in seen and result.orig_shape == (record.height,record.width),
                        'Duplicate image or original dimensions differ')
                seen.add(record.image_id)
                if oracle is not None and checked < coordinate_check_images:
                    expected = oracle[i].clone()
                    # Independent inverse letterbox arithmetic, matching the
                    # pinned Predictor convention (not an assumed square pad).
                    gain = min(img.shape[-2]/record.height, img.shape[-1]/record.width)
                    px = round((img.shape[-1]-record.width*gain)/2-.1)
                    py = round((img.shape[-2]-record.height*gain)/2-.1)
                    expected[:,:4] -= expected.new_tensor([px,py,px,py])
                    expected[:,:4] /= gain
                    expected[:,[0,2]] = expected[:,[0,2]].clamp(0,record.width)
                    expected[:,[1,3]] = expected[:,[1,3]].clamp(0,record.height)
                    require(expected.shape == result.boxes.data.shape, 'NMS oracle count differs')
                    torch.testing.assert_close(expected[:,4:], result.boxes.data[:,4:], rtol=0, atol=0)
                    if len(expected):
                        err = float((expected[:,:4]-result.boxes.data[:,:4]).abs().max())
                        coordinate_error = max(coordinate_error,err)
                        require(err <= .002, 'Coordinate inverse oracle mismatch')
                    checked += 1
                predictions.extend(convert(record, result.boxes.data.cpu().tolist()))
            return results

    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights), task='detect')
    topology = tood_td_head.assert_topology(model.model)
    require(all(torch.isfinite(p).all() for p in model.model.parameters()), 'Nonfinite model weights')
    opts = options(args.device)
    predictor = Predictor(overrides={**model.overrides, **opts, 'mode':'predict'}, _callbacks=model.callbacks)
    guard = NMSGuard(); LOGGER.addHandler(guard)
    start = time.monotonic()
    try:
        for first in range(0,len(records),4):
            chunk = records[first:first+4]
            model.predict([str(r.image_path) for r in chunk], batch=len(chunk), predictor=predictor,
                          stream=False, **opts)
            require(not guard.exceeded, 'NMS time limit exceeded: incomplete predictions rejected')
            done = first+len(chunk)
            if first == 0 or done % 80 == 0 or done == len(records):
                elapsed = time.monotonic()-start
                print(f'{now()} TD PREDICT {done}/{len(records)} elapsed={elapsed:.1f}s '
                      f'ETA={(len(records)-done)*elapsed/done:.1f}s', flush=True)
        require(seen == {r.image_id for r in records}, 'Some images never predicted (including empty results)')
        write(out / 'area_predictions.json', predictions)
        manifest = dict(status='completed', seen_images=len(seen), predictions=len(predictions),
                        sha256=sha(out / 'area_predictions.json'), topology=topology,
                        shapes=dict(shapes), elapsed_seconds=time.monotonic()-start,
                        coordinate_oracle_images=checked, coordinate_oracle_max_error=coordinate_error,
                        backend=dict(matmul_allow_tf32=False,cudnn_allow_tf32=True))
        write(out / 'prediction_manifest.json', manifest)
        return manifest
    finally:
        LOGGER.removeHandler(guard)
        del predictor, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def training_proof(args):
    from run_experiment import EXPERIMENT, package_audit, source_binding
    path = args.report_root / 'train/training.json'
    if not path.is_file():
        index = read(args.project / 'logs/latest_tood_td_train.json')
        path = Path(index['report']) / 'training.json'
    train = read(path)
    require(train['status'] == 'completed' and train['completed_epochs'] == 200 and
            train['seed'] == 1 and train['variant'] == 'tood_td', 'Need a completed formal200 TD model')
    require(train['signature']['package'] == package_audit(), 'Training package and scoring package differ')
    require(train['signature']['source'] == source_binding(args), 'Training/scoring source binding differs')
    require(train['signature']['data'] == sha(args.data), 'Training dataset YAML changed')
    best = args.project / 'runs' / EXPERIMENT / 'weights/best.pt'
    if args.weights is not None:
        best = args.weights
    require(sha(best) == train['best_sha256'], 'Only the original val-selected best may be tested')
    run = args.project / 'runs' / EXPERIMENT
    with (run / 'results.csv').open(encoding='utf-8-sig') as handle:
        rows = [{k.strip():v.strip() for k,v in row.items()} for row in csv.DictReader(handle)]
    require([int(r['epoch']) for r in rows] == list(range(1,201)), 'CSV must contain exactly epochs1..200')
    require(sha(run / 'weights/last.pt') == train['last_sha256'], 'Final last.pt missing/changed')
    return best, train


def evaluate_stage(args, split):
    import d27_official as official
    best, train = training_proof(args)
    records, meta = audit_dataset(args, split)
    refs = cached_references(split)
    out = args.report_dir
    write(args.report_root / 'training_evidence.json', train)
    # The selected checkpoint SHA is frozen BEFORE both prediction passes.
    manifest = predict(args, best, records, out)
    require(sha(best) == train['best_sha256'], 'Checkpoint changed during inference')
    print(f'{now()} TD OFFICIAL {split} SCORING (CPU, all10 classes)', flush=True)
    filtered, ignore = official.filter_detections_and_audit(
        records, official.load_predictions(out / 'area_predictions.json', len(records)))
    metrics = official.official_evaluate(records, filtered)
    binding = dict(seed=1, split=split, checkpoint_sha256=sha(best),
                   protocol={**options(args.device),'batch':4}, dataset=meta,
                   scorer_sha256=sha(HERE / 'd27_official.py'), prediction_sha256=manifest['sha256'])
    result = dict(status='completed', official=metrics, binding=binding, ignore=ignore,
                  cached_references=refs, prediction_sha256=manifest['sha256'],
                  delta_vs_E1_pp={k:metrics[k]-refs['e1']['official'][k] for k in KEYS},
                  delta_vs_E60_pp={k:metrics[k]-refs['e60']['official'][k] for k in KEYS},
                  boundary='One predeclared seed1 public-module adaptation; not complete TOOD; no test tuning.')
    write(out / 'official_metrics.json', result)
    manifest['binding'] = binding
    write(out / 'prediction_manifest.json', manifest)
    table = '|Metric|E1 seed1|TD-adapt seed1|E60 seed1|TD−E1 pp|TD−E60 pp|\n|---|---:|---:|---:|---:|---:|\n'
    for key in KEYS:
        table += (f'|{key}|{refs["e1"]["official"][key]:.4f}|{metrics[key]:.4f}|'
                  f'{refs["e60"]["official"][key]:.4f}|{result["delta_vs_E1_pp"][key]:+.4f}|'
                  f'{result["delta_vs_E60_pp"][key]:+.4f}|\n')
    (out / 'REPORT.md').write_text('# Public TOOD-TD adaptation / ' + split + '\n\n' + table +
        '\nCached E1/E60 are original same-seed results with hash/protocol bindings, not new inference. '
        'Native YOLO mAP is separate from this frozen VisDrone2019 Matlab1.0.4 ignore-region port. '
        'All positive and negative results are retained; a single seed does not prove a stable win.\n', encoding='utf-8')
    print(f'{now()} TD OFFICIAL {split} COMPLETED REPORT={out}', flush=True)
    return result


def summarize(args):
    from run_experiment import snapshot
    best, train = training_proof(args)
    val = read(args.report_root / 'val/official_metrics.json')
    test = read(args.report_root / 'test/official_metrics.json')
    for name, result, count in (('val',val,548),('test-dev',test,1610)):
        require(result['status'] == 'completed' and result['binding']['split'] == name and
                result['binding']['checkpoint_sha256'] == sha(best) and
                result['binding']['dataset']['images'] == count, 'Incomplete/wrong final scoring')
    write(args.report_root / 'training_evidence.json', train)
    write(args.report_root / 'summary.json', dict(status='completed', seed=1,
          public_method='TOOD TaskDecomposition adaptation (not full TOOD)', val=val, test=test,
          checkpoint_sha256=sha(best), parameters=train['parameters'], new_parameters=train['new_parameters']))
    markdown = '# Public TOOD-TD / E1 / E60 / seed1\n\n'
    for section in ('val','test'):
        markdown += (args.report_root / section / 'REPORT.md').read_text(encoding='utf-8') + '\n'
    markdown += '\nTraining: original COCO initialization, 640/b16/workers8/SGD/200e. '
    markdown += 'Validation alone selects best. Test-dev does not select epochs, thresholds or seeds.\n'
    (args.report_root / 'REPORT.md').write_text(markdown, encoding='utf-8')
    snapshot(args, args.report_root)
    write(args.report_root / 'completion.json', dict(status='completed', seed=1, trained_epochs=200,
          official_val_completed=True, official_test_completed=True,
          checkpoint_sha256=sha(best), finished_at=now()))
    print(f'{now()} TD ALL COMPLETED REPORT={args.report_root / "REPORT.md"}', flush=True)
