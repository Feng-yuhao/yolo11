#!/usr/bin/env python3
"""D46: fixed E1/E40 val checkpoints, read-only instance-competition audit."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import competition as geometry
import d27_official as official

PACKAGE = Path(__file__).resolve().parent
REFS = {
    'e1_seed1': 'e1_yolo11s_p2add_img640_seed1_20260926_102750_718906',
    'e1_seed2': 'e1_yolo11s_p2add_img640_seed2_20260926_172907_911324',
    'e1_seed3': 'e1_yolo11s_p2add_img640_seed3_20260926_172932_190728',
    'e40_seed1': 'E40_RUN_20260929_205658_507620',
    'e40_seed2': 'E40S2_RUN_20260930_130808_314014',
    'e40_seed3': 'E40S3_RUN_20261002_223525_596764',
}
KEYS = ('AP50_95_percent', 'AP50_percent', 'AP75_percent', 'AR500_percent')
OPTIONS = dict(imgsz=640, conf=.001, iou=.7, max_det=500, half=False,
               augment=False, verbose=False, save=False, agnostic_nms=False, classes=None)
REVISION = 'd46_instance_competition_v1'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, value):
    p = Path(p)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, p)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def bind(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def trace_nms(raw):
    """Carry anchor IDs in a mask column; pinned NMS arithmetic unchanged.

    Output first 500 must reproduce the ordinary prediction. Float32 exactly
    represents these <2^24 anchor IDs. Explicit nc=10 prevents IDs becoming scores.
    """
    import torch
    from ultralytics.utils import ops
    require(raw.ndim == 2 and raw.shape[0] == 14, 'Expected [14,anchors] ten-class decoded tensor')
    count = raw.shape[1]
    require(count < 2 ** 24 and raw.dtype == torch.float32, 'ID precision/dtype differs')
    guard = raw.clone()
    ids = torch.arange(count, device=raw.device, dtype=raw.dtype)[None]
    enriched = torch.cat((raw, ids), 0)[None]
    kept = ops.non_max_suppression(enriched, .001, .7, nc=10, max_det=30000,
                                   max_time_img=60, in_place=False)[0]
    score, label = raw[4:].max(0)
    pre = torch.where(score > .001)[0]
    if len(pre) > 30000:
        pre = pre[score[pre].argsort(descending=True)[:30000]]
    torch.testing.assert_close(raw, guard, rtol=0, atol=0)
    require(torch.equal(kept[:, 6], kept[:, 6].round()), 'Noninteger carried candidate ID')
    return dict(kept=kept, kept_ids=kept[:, 6].long().cpu().numpy(),
                pre_ids=pre.cpu().numpy(), scores=score.cpu().numpy(), labels=label.cpu().numpy()+1,
                network_boxes=ops.xywh2xyxy(raw[:4].T).cpu().numpy())


def score(records, path):
    filtered, ignore = official.filter_detections_and_audit(records, official.load_predictions(path, 548))
    return official.official_evaluate(records, filtered), ignore


def reference_predictions(spec, args):
    return args.project / 'comparison_reports' / REFS[spec['key']] / 'area_predictions.json'


def validate_model(model, spec):
    from ultralytics.nn.modules.head import Detect
    from spatial_task_head import SpatialTaskDetect, NEW_PARAMETERS
    h = model.model.model[-1]
    require(h.nc == 10 and h.nl == 4 and h.reg_max == 16 and len(model.model.model) == 27, 'Wrong E1 topology')
    require(list(h.f) == [25, 16, 19, 22] and h.stride.tolist() == [4., 8., 16., 32.], 'Wrong feature order/strides')
    require(model.names == {i-1: n for i, n in official.CLASS_NAMES.items()}, 'Wrong class names')
    if spec['model'] == 'E40':
        require(type(h) is SpatialTaskDetect and h.enabled and len(h.task_blocks) == 2, 'Wrong E40 implementation')
        require(sum(p.numel() for p in h.task_blocks.parameters()) == NEW_PARAMETERS, 'E40 parameters differ')
    else:
        require(type(h) is Detect and not hasattr(h, 'task_blocks'), 'E1 is not original Detect')


def setup(args):
    os.environ['YOLO_CONFIG_DIR'] = str(args.project / 'comparison_reports' / 'd46_settings')
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    import ultralytics
    require(Path(ultralytics.__file__).resolve().is_relative_to(args.source.resolve()), 'Wrong Ultralytics source')
    require(ultralytics.__version__ == '8.3.0', 'Pinned project version must be 8.3.0')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.device == '0':
        require(torch.cuda.is_available(), 'CUDA unavailable')
        free, total = torch.cuda.mem_get_info(0)
        require(args.allow_busy_gpu or free/total >= .90, 'GPU busy; wait or explicitly use --allow-busy-gpu')
        require(free >= (args.memory_cap_gib+2)*1024**3, 'Need allocator cap + 2 GiB free')
        torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib*1024**3/total, 0)
    return torch


def preflight(args):
    torch = setup(args)
    manifest = read(PACKAGE / 'models.json')
    for name, sha in manifest['vendor_sha256'].items():
        require(official.sha256(PACKAGE / name) == sha, 'Changed frozen vendor: ' + name)
    specs = [s for s in manifest['models'] if s['seed'] in args.seeds]
    require(len(specs) == len(args.seeds)*2, 'Need two models per seed')
    records, dataset = official.load_dataset(args.dataset)
    require(dataset['annotation_dimension_sha256'] == '058c87e779a61a09933bebe35af1c05c807a444dc842b4be12124dd36134c787',
            'Not the original val annotation/image dimension mapping')
    for spec in specs:
        weight = args.project / spec['weights']
        require(weight.is_file() and official.sha256(weight) == spec['sha256'], 'Checkpoint missing/changed: ' + str(weight))
        rp = reference_predictions(spec, args)
        require(rp.is_file(), 'Missing original reference predictions: ' + str(rp))
        ref_report = args.project / 'comparison_reports' / REFS['e40_seed'+str(spec['seed'])]
        ref = read(ref_report / 'official_metrics.json')
        require(ref['dataset']['annotation_dimension_sha256'] == dataset['annotation_dimension_sha256'], 'Reference dataset drift')
        spec['expected'] = {k: ref['candidate' if spec['model'] == 'E40' else 'baseline'][k] for k in KEYS}
        spec['reference_sha256'] = official.sha256(rp)
        print(f"D46 verified {spec['key']} checkpoint/reference", flush=True)
    payload = dict(revision=REVISION, models=specs, dataset=dataset,
                   protocol=dict(**OPTIONS, batch=4, device=args.device),
                   cap_gib=args.memory_cap_gib, source=str(args.source.resolve()),
                   torch=torch.__version__,
                   source_sha={p: official.sha256(args.source/'ultralytics'/p) for p in
                               ('nn/tasks.py', 'nn/modules/head.py', 'nn/modules/block.py', 'nn/modules/conv.py',
                                'utils/ops.py', 'models/yolo/detect/predict.py', 'engine/predictor.py')},
                   package_sha={p.name: official.sha256(p) for p in PACKAGE.iterdir()
                                if p.suffix in ('.py', '.json', '.sh')},
                   image_content_sha=bind([(r.stem, official.sha256(r.image_path)) for r in records]))
    return records, specs, payload


def gate(args, name):
    return args.project / 'comparison_reports' / ('d46_' + name + '_passed.json')


def gates(args, payload, names):
    for name in names:
        path = gate(args, name)
        require(path.is_file() and read(path).get('binding') == bind(payload),
                f'Missing or stale {name} receipt. Run {name} with the SAME seeds/device/cap.')


def compare_saved(reference, predictions, selected):
    by_id = {}
    for p in reference:
        by_id.setdefault(p['image_id'], []).append(p)
    actual = {}
    for p in predictions:
        actual.setdefault(p['image_id'], []).append(p)
    max_box_error, max_score_error = 0., 0.
    for record in selected:
        left, right = by_id.get(record.image_id, []), actual.get(record.image_id, [])
        require(len(left) == len(right), f'Prediction count did not reproduce image {record.image_id}')
        # Strict score-ranked order; no nearest-box matching can hide missing detections.
        for a, b in zip(left, right):
            require(a['category_id'] == b['category_id'], 'Saved class/order reproduction failed')
            be = float(np.max(np.abs(np.array(a['bbox'])-np.array(b['bbox']))))
            se = abs(a['score']-b['score'])
            max_box_error, max_score_error = max(max_box_error, be), max(max_score_error, se)
            require(be <= .002 and se <= 1e-6, 'Saved coordinate/score reproduction failed')
    return dict(images=len(selected), max_box_error=max_box_error, max_score_error=max_score_error)


def predict(args, spec, selected, out):
    torch = setup(args)
    import spatial_task_head  # checkpoint pickle class, before YOLO load
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils import ops
    model = YOLO(str(args.project/spec['weights']), task='detect')
    validate_model(model, spec)
    selected_by_path = {str(r.image_path.resolve()): r for r in selected}
    rows, image_stats, events, predictions, seen = [], [], [], [], set()
    started = time.monotonic()

    class Predictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs):
            require(isinstance(preds, (tuple, list)) and len(preds) == 2 and len(preds[1]) == 4, 'Wrong raw outputs')
            raw = preds[0].detach().clone()
            guard = raw.clone()
            result = super().postprocess(preds, img, orig_imgs)
            torch.testing.assert_close(raw, guard, rtol=0, atol=0)
            counts = [x.shape[-2]*x.shape[-1] for x in preds[1]]
            require(sum(counts) == raw.shape[-1], 'Anchor/scale identity mismatch')
            level = np.repeat([4, 8, 16, 32], counts)
            for i, value in enumerate(result):
                record = selected_by_path[str(Path(value.path).resolve())]
                require(record.image_id not in seen, 'Repeated image')
                seen.add(record.image_id)
                traced = trace_nms(raw[i])
                # Verify exactly against normal predictor AFTER the same coordinate scaling.
                decoded = traced['kept'][:500, :6].clone()
                decoded[:, :4] = ops.scale_boxes(img.shape[2:], decoded[:, :4], value.orig_shape)
                torch.testing.assert_close(decoded, value.boxes.data, rtol=0, atol=0)
                boxes = ops.scale_boxes(img.shape[2:], ops.xywh2xyxy(raw[i, :4].T.clone()), value.orig_shape).cpu().numpy()
                gt0 = record.eval_gt
                gt0 = gt0[(gt0[:, 4] == 0) & (gt0[:, 5] >= 1) & (gt0[:, 5] <= 10)]
                gt = gt0[:, :4].copy(); gt[:, 2:] += gt[:, :2]
                gc = gt0[:, 5].astype(int)
                scale = 640/max(record.width, record.height)
                gt_rows, adj, adj_any = geometry.raw_geometry(gt, gc, boxes, traced['labels'], traced['scores'], scale, level)
                conf_ids = np.flatnonzero(traced['scores'] > .001)
                matching = geometry.stage_match(gt_rows, adj, adj_any,
                    dict(raw=None, confidence=conf_ids, pre_nms=traced['pre_ids'],
                         post_nms=traced['kept_ids'], final=traced['kept_ids'][:500]))
                witnesses = geometry.nms_witnesses(traced['network_boxes'], traced['labels'], traced['scores'],
                                                   traced['pre_ids'], traced['kept_ids'])
                ev = geometry.witness_events(gt, gc, boxes, traced['scores'], traced['labels'], witnesses)
                for e in ev:
                    e['image_id'] = record.image_id
                    e['gt_has_surviving75'] = gt_rows[e['gt']]['post_nms_independent75']
                    e['gt_has_final75'] = gt_rows[e['gt']]['final_independent75']
                for r in gt_rows:
                    r['image_id'] = record.image_id
                    r['stem'] = record.stem
                rows.extend(gt_rows); events.extend(ev)
                image_stats.append(dict(image_id=record.image_id, matching=matching,
                                        raw=len(boxes), confidence=len(conf_ids), pre_nms=len(traced['pre_ids']),
                                        nms_survivors=len(traced['kept_ids']), final=len(decoded),
                                        witness_unresolved=sum(w['witness'] < 0 for w in witnesses.values()),
                                        witness_tie_or_boundary=sum(w['ambiguous'] for w in witnesses.values())))
                for x1, y1, x2, y2, score_value, cls in value.boxes.data.cpu().tolist():
                    predictions.append(dict(image_id=record.image_id, category_id=int(cls)+1,
                                            bbox=[x1, y1, max(0, x2-x1), max(0, y2-y1)], score=score_value))
            return result

    predictor = Predictor(overrides={**model.overrides, **OPTIONS, 'device': args.device, 'mode': 'predict'},
                          _callbacks=model.callbacks)
    for start in range(0, len(selected), 4):
        part = selected[start:start+4]
        model.predict([str(r.image_path) for r in part], batch=len(part), stream=False,
                      predictor=predictor, device=args.device, **OPTIONS)
        done = start+len(part)
        elapsed = time.monotonic()-started
        print(f"D46 {spec['key']} {done}/{len(selected)} images elapsed={elapsed:.1f}s "
              f"ETA={(len(selected)-done)*elapsed/done:.1f}s events={len(events)}", flush=True)
        if done % 80 == 0 or done == len(selected):
            write(out/'progress.json', dict(images=done, total=len(selected), elapsed_seconds=elapsed, timestamp=now()))
    require(len(seen) == len(selected), 'Incomplete images')
    require(official.sha256(args.project/spec['weights']) == spec['sha256'], 'Disk checkpoint changed')
    require(not model.model.training and all(p.grad is None for p in model.model.parameters()), 'Unexpected training/gradients')
    reproduction = compare_saved(read(reference_predictions(spec, args)), predictions, selected)
    write(out/'predictions.json', predictions)
    write(out/'gt_rows.json', rows)
    write(out/'nms_events.json', events)
    write(out/'image_stats.json', image_stats)
    summary = geometry.summarize(rows, image_stats, events)
    summary.update(key=spec['key'], seed=spec['seed'], reproduction=reproduction,
                   checkpoint_sha256=spec['sha256'], subset_only=len(selected) != 548,
                   weights_unchanged=True, no_gradients=True, elapsed_seconds=time.monotonic()-started)
    write(out/'summary.json', summary)
    del model, predictor
    gc.collect()
    if args.device == '0':
        torch.cuda.empty_cache()
    return summary


def report(out, specs):
    summaries = {s['key']: read(out/s['key']/'summary.json') for s in specs}
    table = '|Model|Official AP|raw same-class independent75|raw maximum one-to-one75|post-NMS maximum one-to-one75|\n|---|---:|---:|---:|---:|\n'
    for key, s in summaries.items():
        m = s['matching']; n = m['raw']['gt']
        table += f"|{key}|{s['official']['AP50_95_percent']:.4f}|{100*m['raw']['independent']/max(n,1):.3f}|{100*m['raw']['maximum_one_to_one']/max(n,1):.3f}|{100*m['post_nms']['maximum_one_to_one']/max(n,1):.3f}|\n"
    # No automatic train recommendation from signs or arbitrary correlation cutoffs.
    decision = dict(status='completed', reproduction_passed=True, training_authorized=False,
                    next_step='Review paired seeds, size/class-conditioned neighbor errors and distinct-instance NMS witnesses before proposing training.',
                    unsupported_claims=['crowding is the causal root', 'witness events equal recoverable AP',
                                        'maximum matching can be achieved by the trained model', 'a relation module will improve AP'])
    write(out/'decision.json', decision)
    write(out/'all_models.json', summaries)
    (out/'REPORT.md').write_text('# D46 E1/E40 instance-competition audit\n\n'+table+
        '\nAll original predictions and official scores reproduced before interpretation. No training.\n\n'+
        '## Read these files\n\n'+
        '- `all_models.json`: paired-seed summaries and size/class/density cells.\n'+
        '- `<model>/gt_rows.json`: signed center/size errors, same-class nearest-neighbor direction, surviving alternatives.\n'+
        '- `<model>/nms_events.json`: good-candidate overlap witnesses, GT association and ties/unresolved cases.\n'+
        '- `<model>/image_stats.json`: independent versus exact maximum one-to-one coverage at all stages.\n\n'+
        '## Scientific boundaries\n\n'+ '\n'.join('- '+b for b in next(iter(summaries.values()))['boundaries'])+
        '\n- Matched IDs, neighbor direction and NMS witnesses do not identify a training-time root cause.\n'+
        '- Geometry candidates are pre-official-ignore; official metrics retain original ignore rules.\n'+
        '- No test-dev evaluation, GT-oracle score rewriting, threshold search, checkpoint tuning or new training.\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['check', 'smoke', 'run', 'verify-gates', 'self-test'])
    p.add_argument('--project', type=Path, default=Path('/root/autodl-tmp/yolo11_project'))
    p.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/yolo11_src'))
    p.add_argument('--dataset', type=Path, default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val'))
    p.add_argument('--seeds', type=int, nargs='+', choices=[1, 2, 3], default=[1, 2])
    p.add_argument('--device', default='0', choices=['0', 'cpu'])
    p.add_argument('--memory-cap-gib', type=float, default=4.)
    p.add_argument('--allow-busy-gpu', action='store_true')
    a = p.parse_args()
    a.seeds = sorted(set(a.seeds))
    require(1 <= a.memory_cap_gib <= 8, 'Allocator cap must be 1..8 GiB')
    setup(a)
    from audit_unit_checks import unit_tests
    checked = unit_tests(a.device)
    if a.mode == 'self-test':
        print(json.dumps(checked, indent=2)); return
    records, specs, payload = preflight(a)
    if a.mode == 'verify-gates':
        gates(a, payload, ['check', 'smoke'])
        print('D46 CHECK AND EIGHT-IMAGE SMOKE RECEIPTS VALID', flush=True); return
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = a.project/'comparison_reports'/('D46_'+a.mode.upper()+'_'+stamp)
    out.mkdir(parents=True, exist_ok=False)
    write(out/'manifest.json', payload)
    write(out/'tests.json', checked)
    write(out/'status.json', dict(status='running', started_at=now(), mode=a.mode))
    print('D46 START '+now()+' REPORT='+str(out), flush=True)
    try:
        if a.mode == 'check':
            import spatial_task_head
            from ultralytics import YOLO
            for s in specs:
                model = YOLO(str(a.project/s['weights']), task='detect')
                validate_model(model, s)
                del model
            official.run_unit_tests()
            write(gate(a, 'check'), dict(status='passed', binding=bind(payload), report=str(out)))
            print('D46 CHECK PASSED', flush=True)
        else:
            gates(a, payload, ['check'] if a.mode == 'smoke' else ['check', 'smoke'])
            selected = records[:8] if a.mode == 'smoke' else records
            for s in specs:
                sub = out/s['key']; sub.mkdir()
                summary = predict(a, s, selected, sub)
                actual, ignore = score(selected, sub/'predictions.json')
                reference, _ = score(selected, reference_predictions(s, a))
                for k in KEYS:
                    require(abs(actual[k]-reference[k]) < 1e-6, 'Official reference reproduction failed: '+s['key']+' '+k)
                    if a.mode == 'run':
                        require(abs(actual[k]-s['expected'][k]) < 1e-6, 'Original formal metric drift: '+s['key']+' '+k)
                summary.update(official=actual, official_reference=reference, ignore_audit=ignore,
                               official_reproduction_passed=True)
                write(sub/'summary.json', summary)
                print(f"D46 {s['key']} COMPLETE AP={actual['AP50_95_percent']:.6f}", flush=True)
            if a.mode == 'smoke':
                write(gate(a, 'smoke'), dict(status='passed', binding=bind(payload), report=str(out)))
                print('D46 EIGHT-IMAGE SMOKE PASSED FOR ALL SELECTED MODELS', flush=True)
            else:
                report(out, specs)
        write(out/'status.json', dict(status='completed', finished_at=now(), mode=a.mode))
        print('D46 FINISHED '+now()+' REPORT='+str(out), flush=True)
    except BaseException as error:
        write(out/'status.json', dict(status='failed', failed_at=now(), mode=a.mode, error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
