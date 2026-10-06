#!/usr/bin/env python3
"""One frozen E47 checkpoint, six scoped inference states; no optimizer."""
import argparse
from datetime import datetime
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from io_utils import HERE, KEYS, package_signature, read, resolve_reports, sha, write

SLICES = ((0, float('inf'), 'all'), (0, 8, 'lt8'), (8, 16, '8_16'),
          (16, 24, '16_24'), (24, 32, '24_32'), (32, float('inf'), 'ge32'))
FIELDS = ('raw_any', 'raw_p2', 'raw_p3', 'raw_correct', 'final_any', 'final_correct',
          'official_final_any', 'official_final_correct')
BOUNDARY = ('Frozen-checkpoint intervention, not training-time causal proof. '
            'Coverage is candidate existence per GT, not AP or one-to-one recall. '
            'all_readout_off preserves E47-coadapted E40 weights, not a separately trained E40.')


def setup(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    os.environ['YOLO_CONFIG_DIR'] = str(args.project / 'comparison_reports/d45_settings')
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    import ultralytics
    import spatial_task_head  # original checkpoint classes, not an installed extension
    if (ultralytics.__version__ != '8.3.0' or
            not Path(ultralytics.__file__).resolve().is_relative_to(args.source.resolve())):
        raise RuntimeError('Requires your Ultralytics 8.3.0 source; specify --source.')
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(args.seed, deterministic=True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    if args.device != 'cpu':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no implicit CPU fallback.')
        args.device = 'cuda:' + args.device if args.device.isdigit() else args.device
        torch.cuda.set_device(args.device)
        if not args.allow_busy_gpu:
            query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, timeout=15)
            if query.returncode:
                raise RuntimeError('Cannot establish GPU idleness. Use offline now, or explicitly --allow-busy-gpu after checking resources.')
            other = [v.strip() for v in query.stdout.splitlines() if v.strip().isdigit() and int(v.strip()) != os.getpid()]
            if other:
                raise RuntimeError('Other CUDA processes detected: ' + ','.join(other) +
                                   '. Run offline while seed2 trains; wait for GPU release for full diagnosis. '
                                   '--allow-busy-gpu is explicit opt-in; no processes are killed.')
        free, total = torch.cuda.mem_get_info()
        cap = int(args.memory_cap_gib * 2**30)
        if cap >= total or free < cap + 2**30:
            raise RuntimeError(f'Need {args.memory_cap_gib} GiB allocator cap + 1 GiB headroom; '
                               f'free={free/2**30:.2f} GiB. No batch reduction or training interruption.')
        torch.cuda.set_per_process_memory_fraction(cap / total)
    return torch


def sources(args):
    e47, e40 = resolve_reports(args.project, args.seed, args.e47_report, args.e40_report)
    comparison = read(e47 / 'comparison.json')
    name = f'e47a_p2_depth_readout_img640_seed{args.seed}_200e'
    weights = args.project / 'runs' / name / 'weights/best.pt'
    if sha(weights) != comparison['checkpoint_sha256']:
        raise RuntimeError('Checkpoint hash differs from completed report: ' + str(weights))
    baseline_name = {1: 'e1_yolo11s_p2add_img640_seed1_20260926_102750_718906',
                     2: 'e1_yolo11s_p2add_img640_seed2_20260926_172907_911324'}[args.seed]
    baseline_dir = args.project / 'comparison_reports' / baseline_name
    baseline = read(baseline_dir / 'metrics.json')
    audit = {}
    for name, expected in baseline['p2_architecture']['source_code_sha256'].items():
        if name.startswith('nn/'):
            actual = sha(args.source / 'ultralytics' / name, True)
            audit[name] = dict(expected=expected, actual=actual, matches=actual == expected)
            if name != 'nn/tasks.py' and actual != expected:
                raise RuntimeError('Original network implementation changed: ' + name)
    return weights, e47, e40, comparison, baseline_dir, audit


def make_predictor(args, model, records, options):
    import numpy as np
    import torch
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils import ops
    import d38_common as common
    from interventions import STATES, intervention, state_digest
    from readout_stats import ReadoutStats
    by_path = {str(rec.image_path.resolve()): rec for rec in records}

    class Predictor(DetectionPredictor):
        def setup_model(self, model, verbose=False):
            super().setup_model(model, verbose=verbose)
            if any(m.training for m in self.model.model.modules()):
                raise RuntimeError('Every diagnostic module must be in eval mode.')
            self.head = self.model.model.model[-1]
            if (self.head.nl, self.head.reg_max, list(self.head.f)) != (4, 16, [25, 16, 19, 22]):
                raise RuntimeError('Wrong four-scale Detect topology.')
            if (not self.head.enabled or len(self.head.task_blocks) != 2 or
                    any(not getattr(block, 'readout_enabled', False) for block in self.head.task_blocks)):
                raise RuntimeError('Saved E47 readout is missing/disabled; do not reattach or reinitialize it.')
            self.capture = None
            self.capturing = True
            def cache(module, inputs):
                if self.capturing:
                    self.capture = tuple(x.detach().clone() for x in inputs[0])
            self.cache_handle = self.head.register_forward_pre_hook(cache)
            self.rows = {state: [] for state in STATES}
            self.predictions = {state: [] for state in STATES}
            self.d45_seen = {state: set() for state in STATES}
            self.stats = ReadoutStats()
            self.state_before = state_digest(self.model.model)
            self.first = True

        def inference(self, im, *unused, **kwargs):
            if im.dtype != torch.float32:
                raise RuntimeError('Diagnosis pins FP32.')
            context = [by_path[str(Path(p).resolve())] for p in self.batch[0]]
            self.stats.set_context(context, im.shape[2:])
            self.capture = None
            with intervention(self.head, 'normal', observer=self.stats):
                normal = super().inference(im, *unused, **kwargs)
            if self.capture is None or len(self.capture) != 4:
                raise RuntimeError('Missing four-level head feature cache.')
            features = self.capture
            self.capturing = False
            results = {'normal': normal}
            try:
                with torch.inference_mode():
                    if self.first:
                        replay = self.head([x.clone() for x in features])
                        torch.testing.assert_close(replay[0], normal[0], rtol=0, atol=1e-5)
                        for x, y in zip(replay[1], normal[1]):
                            torch.testing.assert_close(x, y, rtol=0, atol=0)
                    for state in STATES[1:]:
                        inputs = [x.clone() for x in features]
                        with intervention(self.head, state):
                            value = self.head(inputs)
                        if self.first:
                            for x, y in zip(inputs, features):
                                torch.testing.assert_close(x, y, rtol=0, atol=0)
                            for level in (2, 3):  # P4/P5 are NEVER changed
                                torch.testing.assert_close(value[1][level], normal[1][level], rtol=0, atol=0)
                            if state != 'all_readout_off':
                                level = 0 if state.startswith('p2') else 1
                                other = 1 - level
                                torch.testing.assert_close(value[1][other], normal[1][other], rtol=0, atol=0)
                                unaffected = slice(0, 64) if '_cls_' in state else slice(64, None)
                                torch.testing.assert_close(value[1][level][:, unaffected], normal[1][level][:, unaffected], rtol=0, atol=0)
                        results[state] = value
                    if self.first:
                        restored = self.head([x.clone() for x in features])
                        torch.testing.assert_close(restored[0], normal[0], rtol=0, atol=1e-5)
                        self.first = False
            finally:
                self.capturing = True
            self.pending = results
            return normal

        def postprocess(self, preds, img, orig_imgs):
            answer = None
            normal_boxes = {}
            for state in STATES:
                value = self.pending[state]
                if (not isinstance(value, (tuple, list)) or len(value) != 2 or len(value[1]) != 4 or
                        value[0].ndim != 3 or value[0].shape[1] != 14 or not torch.isfinite(value[0]).all()):
                    raise RuntimeError('Invalid detector output: ' + state)
                raw = value[0].detach().clone()
                # NMS mutates decoded xywh in this source: NEVER pass cached value[0].
                result = super().postprocess((raw.clone(), value[1]), img, orig_imgs)
                n2 = value[1][0].shape[-2] * value[1][0].shape[-1]
                n3 = value[1][1].shape[-2] * value[1][1].shape[-1]
                for i, item in enumerate(result):
                    rec = by_path[str(Path(item.path).resolve())]
                    if rec.image_id in self.d45_seen[state]:
                        raise RuntimeError('Repeated diagnostic image.')
                    self.d45_seen[state].add(rec.image_id)
                    final = item.boxes.data.detach().cpu().numpy()
                    for x1, y1, x2, y2, score, cls in final.tolist():
                        self.predictions[state].append(dict(image_id=rec.image_id, category_id=int(cls) + 1,
                            bbox=[x1, y1, max(0, x2-x1), max(0, y2-y1)], score=score))
                    gt = rec.eval_gt
                    gt = gt[(gt[:, 4] == 0) & (gt[:, 5] >= 1) & (gt[:, 5] <= 10)]
                    target = gt[:, :4].copy()
                    target[:, 2:] += target[:, :2]
                    boxes = ops.scale_boxes(img.shape[2:], ops.xywh2xyxy(raw[i, :4].T.clone()), item.orig_shape).cpu().numpy()
                    labels = raw[i, 4:].argmax(0).cpu().numpy() + 1
                    if state in ('p2_cls_off', 'p3_cls_off'):
                        # Classification-only interventions leave all decoded geometry EXACTLY equal.
                        np.testing.assert_array_equal(boxes, normal_boxes[rec.image_id][0])
                        any_iou, p2_iou, p3_iou = normal_boxes[rec.image_id][1:]
                    else:
                        any_iou = common.iou_max(target, boxes)
                        p2_iou = common.iou_max(target, boxes[:n2])
                        p3_iou = common.iou_max(target, boxes[n2:n2+n3])
                        if state == 'normal':
                            normal_boxes[rec.image_id] = (boxes, any_iou, p2_iou, p3_iou)
                    correct = common.iou_max(target, boxes, labels, gt[:, 5])
                    final_any = common.iou_max(target, final[:, :4])
                    final_correct = common.iou_max(target, final[:, :4], final[:, 5] + 1, gt[:, 5])
                    sizes = np.sqrt(gt[:, 2] * gt[:, 3]) * 640 / max(rec.height, rec.width)
                    for j in range(len(gt)):
                        self.rows[state].append(dict(image_id=rec.image_id, gt_index=j, cls=int(gt[j, 5]),
                            size=float(sizes[j]), gt_xywh=gt[j, :4].tolist(), raw_any=float(any_iou[j]),
                            raw_p2=float(p2_iou[j]), raw_p3=float(p3_iou[j]), raw_correct=float(correct[j]),
                            final_any=float(final_any[j]), final_correct=float(final_correct[j])))
                torch.testing.assert_close(value[0], raw, rtol=0, atol=0)
                if state == 'normal':
                    answer = result
            self.pending = None
            self.capture = None
            return answer

        def finish(self):
            self.cache_handle.remove()
            expected = {rec.image_id for rec in records}
            for state in STATES:
                if self.d45_seen[state] != expected:
                    raise RuntimeError('Missing images for ' + state)
            if state_digest(self.model.model) != self.state_before:
                raise RuntimeError('Diagnosis modified registered weights/buffers.')
            if any(p.grad is not None for p in self.model.model.parameters()):
                raise RuntimeError('Diagnosis unexpectedly generated gradients.')
            return self.predictions, self.rows, self.stats

    return Predictor(overrides={**model.overrides, **options, 'mode': 'predict'}, _callbacks=model.callbacks)


def fixture_test(args, weights):
    import contextlib
    import io
    import numpy as np
    from ultralytics import YOLO
    from PIL import Image
    import d27_official as official
    import d38_common as common
    images = [np.random.default_rng(450+i).integers(0, 256, (h, w, 3), dtype=np.uint8)
              for i, (h, w) in enumerate(((97, 131), (85, 127), (111, 93), (81, 123)))]
    fixture_dir = args.out / 'fixture_images'
    fixture_dir.mkdir(parents=True, exist_ok=False)
    for i, image in enumerate(images):
        Image.fromarray(image).save(fixture_dir / f'image{i}.png')
    records = [official.ImageRecord(i+1, f'image{i}', fixture_dir / f'image{i}.png', Path('unused.txt'),
               im.shape[1], im.shape[0], np.zeros((0, 8)), np.array([[15, 20, 20, 18, 0, 1]], dtype=np.float64),
               np.zeros((0, 4)), np.zeros((0, 4))) for i, im in enumerate(images)]
    model = YOLO(str(weights), task='detect')
    opts = dict(common.prediction_options(args.device), imgsz=128)
    predictor = make_predictor(args, model, records, opts)
    for start in (0, 2):
        # The real runner reuses one Predictor across many model.predict calls.
        model.predict([str(r.image_path) for r in records[start:start+2]], batch=2,
                      predictor=predictor, stream=False, **opts)
    predictions, rows, stats = predictor.finish()
    if len(stats.rows) != 8 or any(len(items) != 4 for items in rows.values()):
        raise RuntimeError('Synthetic GT/observer integration failed.')
    statistics = stats.summary(rows['normal'])
    if statistics['P2/all']['gt_count'] != 4:
        raise RuntimeError('Real flat geometry/statistics join failed.')
    fixture_scores = {}
    for state, values in predictions.items():
        grouped = [[] for _ in records]
        for index, item in enumerate(values):
            grouped[item['image_id']-1].append([*item['bbox'], item['score'], item['category_id'], index])
        raw = [np.asarray(g, dtype=np.float64).reshape(-1, 7) for g in grouped]
        filtered, _ = official.filter_detections_and_audit(records, raw)
        filtered_geometry(records, filtered, rows[state])
        coverage = geometry_summary(rows[state], rows['normal'] if state != 'normal' else None)
        if coverage['all']['gt_count'] != 4:
            raise RuntimeError('Real filtered geometry summary failed.')
        with contextlib.redirect_stdout(io.StringIO()):
            fixture_scores[state] = official.official_evaluate(records, filtered)
    return dict(actual_checkpoint=True, cached_head_replay=True, nms_clone=True,
                P4_P5_and_untouched_task_exact=True, states=len(rows), images=4, engine_batches=2,
                flat_statistics_join=True, official_ignore_and_scoring_pipeline=True,
                boundary='Synthetic engine integration, not real validation/CUDA memory proof.')


def geometry_summary(rows, reference=None):
    result = {}
    saved = {(r['image_id'], r['gt_index']): r for r in reference} if reference is not None else None
    for lo, hi, name in SLICES:
        group = [r for r in rows if lo <= r['size'] < hi]
        item = dict(gt_count=len(group))
        for key in FIELDS:
            item[key + '75_percent'] = 100 * sum(r[key] >= .75 for r in group) / len(group) if group else None
            if reference is not None:
                improved = sum(r[key] >= .75 and saved[(r['image_id'], r['gt_index'])][key] < .75 for r in group)
                lost = sum(r[key] < .75 and saved[(r['image_id'], r['gt_index'])][key] >= .75 for r in group)
                item[key + '_paired_counts'] = dict(improved=improved, lost=lost, net=improved-lost)
        result[name] = item
    return result


def filtered_geometry(records, filtered, rows):
    import d38_common as common
    index = {(r['image_id'], r['gt_index']): r for r in rows}
    for rec in records:
        gt = rec.eval_gt
        gt = gt[(gt[:, 4] == 0) & (gt[:, 5] >= 1) & (gt[:, 5] <= 10)]
        target = gt[:, :4].copy()
        target[:, 2:] += target[:, :2]
        values = filtered[rec.image_id-1]
        boxes = values[:, :4].copy()
        boxes[:, 2:] += boxes[:, :2]
        any_iou = common.iou_max(target, boxes)
        correct = common.iou_max(target, boxes, values[:, 5], gt[:, 5])
        for j in range(len(gt)):
            index[(rec.image_id, j)].update(official_final_any=float(any_iou[j]), official_final_correct=float(correct[j]))


def signature(args, comparison, dataset, report):
    import torch
    import d38_common as common
    names = ('nn/tasks.py', 'nn/modules/head.py', 'nn/modules/block.py', 'nn/modules/conv.py',
             'utils/ops.py', 'engine/predictor.py', 'models/yolo/detect/predict.py', 'nn/autobackend.py')
    gpu = None
    if args.device != 'cpu':
        prop = torch.cuda.get_device_properties(args.device)
        gpu = dict(name=prop.name, total_memory=prop.total_memory, capability=list(torch.cuda.get_device_capability(args.device)))
    return dict(package=package_signature(), seed=args.seed, weights=comparison['checkpoint_sha256'],
                dataset=dataset['annotation_dimension_sha256'], saved_predictions=sha(report / 'area_predictions.json'),
                source={name: sha(args.source / 'ultralytics' / name, True) for name in names},
                torch=torch.__version__, cuda=torch.version.cuda, gpu=gpu, device=args.device,
                protocol=common.prediction_options(args.device), batch=4, memory_cap_gib=args.memory_cap_gib,
                cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32)


def run(args):
    torch = setup(args)
    import numpy as np
    from ultralytics import YOLO
    import d27_official as official
    import d38_common as common
    from interventions import STATES, audit_interventions
    weights, original, parent, comparison, baseline_dir, source_audit = sources(args)
    audit_model = YOLO(str(weights), task='detect')
    audit_model.model.float().eval()
    checks = audit_interventions(audit_model.model.model[-1], device=args.device)
    del audit_model
    checks['official'] = official.run_unit_tests()
    checks['coordinates_and_nms'] = common.tests()
    checks['checkpoint_fixture'] = fixture_test(args, weights)
    write(args.out / 'check.json', dict(status='passed', device=args.device, checks=checks, source=source_audit))
    if args.mode == 'check':
        print(f'D45 CHECK PASSED device={args.device}; no training.', flush=True)
        return
    gc.collect()
    if args.device != 'cpu':
        torch.cuda.empty_cache()
    if args.device == 'cpu':
        raise RuntimeError('Smoke/full inference requires CUDA; CPU check is engineering only.')
    records, dataset = official.load_dataset(args.dataset)
    if dataset['annotation_dimension_sha256'] != comparison['dataset']['annotation_dimension_sha256']:
        raise RuntimeError('Original validation annotation/dimension fingerprint differs.')
    images = read(baseline_dir / 'area_ground_truth.json')['images']
    if len(images) != len(records) or any((r.image_id, r.stem, r.width, r.height) !=
            (im['id'], Path(im['file_name']).stem, im['width'], im['height']) for r, im in zip(records, images)):
        raise RuntimeError('Original dataset image ID mapping differs.')
    sig = signature(args, comparison, dataset, original)
    gate = args.project / 'comparison_reports' / f'd45_e47_seed{args.seed}_smoke_gate.json'
    if args.mode == 'run':
        receipt = read(gate) if gate.is_file() else {}
        if receipt.get('signature') != sig:
            raise RuntimeError('Matching CUDA 8-image smoke missing/stale; run smoke first.')
        smoke = read(Path(receipt['report']) / 'metrics.json')
        check = read(Path(receipt['report']) / 'check.json')
        if (smoke.get('status') != 'completed' or smoke.get('image_count') != 8 or
                smoke.get('signature') != sig or not smoke.get('normal_prediction_reproduced') or
                not smoke.get('normal_geometry_reproduced') or check.get('status') != 'passed' or
                check.get('device') != args.device or (Path(receipt['report']) / 'failure.json').exists()):
            raise RuntimeError('Smoke receipt evidence invalid.')
    selected = records[:8] if args.mode == 'smoke' else records
    started = time.perf_counter()
    metrics = dict(status='running', seed=args.seed, zero_training=True, subset_only=args.mode == 'smoke',
                   image_count=len(selected), signature=sig, boundary=BOUNDARY,
                   original_report=str(original), parent_report=str(parent), checkpoint=str(weights),
                   started_at=datetime.now().astimezone().isoformat())
    write(args.out / 'metrics.json', metrics)
    opts = common.prediction_options(args.device)
    model = YOLO(str(weights), task='detect')
    predictor = make_predictor(args, model, selected, opts)
    for start in range(0, len(selected), 4):
        chunk = selected[start:start+4]
        model.predict([str(r.image_path) for r in chunk], batch=len(chunk), predictor=predictor, stream=False, **opts)
        done = start + len(chunk)
        if start == 0 or done % 40 == 0 or done == len(selected):
            elapsed = time.perf_counter() - started
            eta = elapsed / done * (len(selected)-done)
            print(f'D45 IMAGES {done}/{len(selected)} six head states; elapsed={elapsed/60:.1f}min '
                  f'inference_remaining~{eta/60:.1f}min reserved={torch.cuda.memory_reserved()/2**30:.2f}GiB', flush=True)
    predictions, rows, stats = predictor.finish()
    write(args.out / 'readout_rows.json', stats.rows)
    write(args.out / 'readout_summary.json', stats.summary(rows['normal']))
    for state in STATES:
        write(args.out / (state + '_predictions.json'), predictions[state])
    del predictor, model, predictions
    gc.collect()
    torch.cuda.empty_cache()
    saved = official.load_predictions(original / 'area_predictions.json', len(records))
    normal = official.load_predictions(args.out / 'normal_predictions.json', len(records))
    for rec in selected:
        np.testing.assert_allclose(normal[rec.image_id-1][:, :6], saved[rec.image_id-1][:, :6], rtol=0, atol=1e-5,
                                   err_msg='Normal saved prediction reproduction failed: ' + rec.stem)
    saved_rows = {(r['image_id'], r['gt_index']): r for r in read(original / 'gt_rows.json')}
    for row in rows['normal']:
        old = saved_rows[(row['image_id'], row['gt_index'])]
        if row['cls'] != old['cls'] or abs(row['size']-old['size']) > 1e-6:
            raise RuntimeError('Normal GT indexing/size differs.')
        for key in FIELDS[:6]:
            if abs(row[key]-old[key]) > 1e-5:
                raise RuntimeError('Normal raw/final geometry reproduction failed: ' + key)
    metrics['normal_prediction_reproduced'] = True
    metrics['normal_geometry_reproduced'] = True
    scores = {}
    for state in STATES:
        print(f'D45 CPU {"OFFICIAL SCORING" if args.mode == "run" else "IGNORE/GEOMETRY CHECK"} {state}; '
              'MODEL TENSORS RELEASED (CUDA context remains)', flush=True)
        raw = normal if state == 'normal' else official.load_predictions(args.out / (state + '_predictions.json'), len(records))
        filtered, ignore = official.filter_detections_and_audit(records, raw)
        filtered_geometry(selected, filtered, rows[state])
        write(args.out / (state + '_gt_rows.json'), rows[state])
        write(args.out / (state + '_geometry.json'), geometry_summary(rows[state], rows['normal'] if state != 'normal' else None))
        if args.mode == 'run':
            score = official.official_evaluate(records, filtered)
            if state == 'normal':
                delta = {k: score[k]-comparison['candidate'][k] for k in KEYS}
                if max(abs(v) for v in delta.values()) > .002:
                    raise RuntimeError('Normal official AP reproduction failed: ' + str(delta))
                metrics['official_reproduction_delta_pp'] = delta
            versus = {k: score[k]-scores['normal']['official'][k] for k in KEYS} if state != 'normal' else {k: 0. for k in KEYS}
            class_delta = {k: {metric: value[metric]-scores['normal']['official']['per_class'][k][metric]
                           for metric in KEYS[:3]} for k, value in score['per_class'].items()} if state != 'normal' else {}
            scores[state] = dict(official=score, delta_vs_normal_pp=versus, per_class_delta_pp=class_delta, ignore=ignore)
            write(args.out / (state + '_official.json'), scores[state])
            print(f'D45 {state} AP={score[KEYS[0]]:.6f} delta_vs_normal={versus}', flush=True)
        del raw, filtered
        gc.collect()
    if sha(weights) != comparison['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed during diagnosis.')
    if signature(args, comparison, dataset, original) != sig:
        raise RuntimeError('Package/source/data reference or runtime signature changed during diagnosis.')
    metrics.update(status='completed', scores=scores, no_gradients=True, weights_unchanged=True,
                   finished_at=datetime.now().astimezone().isoformat(), elapsed_seconds=time.perf_counter()-started)
    write(args.out / 'metrics.json', metrics)
    if args.mode == 'smoke':
        write(gate, dict(signature=sig, report=str(args.out)))
    print(f'D45 {args.mode.upper()} FINISHED REPORT={args.out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check', 'smoke', 'run'), required=True)
    parser.add_argument('--seed', type=int, choices=(1, 2), default=1)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--e47-report', type=Path)
    parser.add_argument('--e40-report', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--memory-cap-gib', type=float, default=6.)
    parser.add_argument('--allow-busy-gpu', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.memory_cap_gib <= 12:
        raise ValueError('Allocator cap must be 2..12 GiB.')
    args.project = args.project.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    print(f'D45 {args.mode} START {datetime.now().astimezone().isoformat()} REPORT={args.out}', flush=True)
    try:
        package_signature()
        run(args)
    except BaseException:
        write(args.out / 'failure.json', dict(status='failed', zero_training=True, error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
