#!/usr/bin/env python3
"""Read-only E1-640 raw-candidate stage audit. Never trains or patches source."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime

KEYS = ('AP50_95_percent', 'AP50_percent', 'AP75_percent', 'AR500_percent')
STAGES = ('raw_any', 'raw_argmax_correct', 'confidence', 'max_nms', 'nms_unlimited', 'top500')
BOUNDARY = [
    'Coverage is independent GT candidate existence, not one-to-one recall or AP.',
    'Raw coverage uses decoded boxes, not a causal test of backbone, TAL or regression loss.',
    'Raw any-class versus argmax-correct gap is not isolated classification error; nearby instances can explain it.',
    'NMS/score/budget losses locate postprocessing removal, not proof that disabling that operation improves AP.',
    'Continuous XYXY diagnostic IoU is separate from frozen official MATLAB-style AP.',
    'Only nonignored class 1..10 official GT are included; raw candidates are not ignore-filtered to preserve the actual NMS path.',
    'Size is sqrt(area)*640/max(original H,W), not official AP-small; density is conditioned on size but remains confounded.',
    'No new training or module is authorized by this report. No test-dev is used.',
]

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def sha(p, normalized=False):
    b = Path(p).read_bytes()
    return hashlib.sha256(b.replace(b'\r\n', b'\n') if normalized else b).hexdigest()

def prediction_options(device):
    # stream is a model.predict call argument, NOT a get_cfg override.
    return dict(imgsz=640, device=device, conf=.001, iou=.7, max_det=500,
                half=False, augment=False, verbose=False, save=False,
                agnostic_nms=False, classes=None)

def iou_max(gt, boxes, labels=None, gt_classes=None):
    import numpy as np
    answer = np.zeros(len(gt), dtype=np.float32)
    # Bounded CPU chunks, no giant GT-by-all-anchor GPU allocation.
    for start in range(0, len(boxes), 2048):
        b = boxes[start:start+2048]
        lo = np.maximum(gt[:, None, :2], b[None, :, :2])
        hi = np.minimum(gt[:, None, 2:], b[None, :, 2:])
        inter = np.maximum(hi-lo, 0).prod(2)
        ga = np.maximum(gt[:, 2:]-gt[:, :2], 0).prod(1)
        ba = np.maximum(b[:, 2:]-b[:, :2], 0).prod(1)
        v = inter / np.maximum(ga[:, None]+ba[None, :]-inter, 1e-9)
        if labels is not None:
            v *= gt_classes[:, None] == labels[None, start:start+2048]
        if len(b):
            answer = np.maximum(answer, v.max(1))
    return answer

def tests():
    import numpy as np
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import ops
    get_cfg(overrides={**prediction_options('cpu'), 'mode':'predict'})
    gt = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    assert (iou_max(gt, gt[:1]) == 1).all()  # explicitly NOT one-to-one recall
    assert (iou_max(gt, gt[:0]) == 0).all()
    assert len(iou_max(gt[:0], gt[:0])) == 0
    assert (iou_max(gt, gt[:1], np.array([2]), np.array([1, 1])) == 0).all()
    for original, resized in [((360, 640), (384, 640)), ((333, 517), (416, 640)), ((640, 640), (640, 640))]:
        # Test exactly the scale_boxes convention used by DetectionPredictor,
        # including rounded asymmetric padding, not an assumed square input.
        gain = min(resized[0]/original[0], resized[1]/original[1])
        px = round((resized[1]-original[1]*gain)/2-.1)
        py = round((resized[0]-original[0]*gain)/2-.1)
        b = torch.tensor([[20., 15., 100., 80.]])
        scaled = b*gain + torch.tensor([px, py, px, py])
        torch.testing.assert_close(ops.scale_boxes(resized, scaled, original), b, rtol=1e-5, atol=1e-4)
    # Overlapping same-class boxes suppressed, different class survives;
    # unlimited NMS prefix must equal normal max_det output.
    p = torch.zeros(1, 14, 3)
    p[0, :4, :] = torch.tensor([[10., 10., 10.], [10., 10., 10.], [8., 8., 8.], [8., 8., 8.]])
    p[0, 4, :2] = torch.tensor([.9, .8]); p[0, 5, 2] = .7
    full = ops.non_max_suppression(p.clone(), .001, .7, max_det=30000, max_time_img=60)[0]
    short = ops.non_max_suppression(p.clone(), .001, .7, max_det=1)[0]
    assert len(full) == 2
    torch.testing.assert_close(full[:1], short, rtol=0, atol=0)
    return dict(empty_and_wrong_class=True, multiple_gt_one_box_explicit=True,
                rectangular_coordinate_conversion=True, class_aware_nms_and_prefix=True,
                actual_predictor_configuration_valid=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=Path('/root/autodl-tmp/yolo11_project'))
    p.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/yolo11_src'))
    p.add_argument('--dataset', type=Path, default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val'))
    p.add_argument('--seed', type=int, choices=(1, 2, 3), default=1)
    p.add_argument('--report', type=Path)
    p.add_argument('--device', default='0')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--smoke-images', type=int, default=0)
    a = p.parse_args()
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    settings_root = a.project if a.project.is_dir() else Path(__file__).resolve().parent
    os.environ.setdefault('YOLO_CONFIG_DIR', str(settings_root/'comparison_reports/d38_settings'))
    sys.path.insert(0, str(a.source.resolve()))
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils import ops
    if not Path(ultralytics.__file__).resolve().is_relative_to(a.source.resolve()):
        raise RuntimeError('Wrong Ultralytics source. Specify --source explicitly.')
    checked = tests()
    if a.self_test:
        print(json.dumps(checked, indent=2)); return
    pattern = f'e1_yolo11s_p2add_img640_seed{a.seed}_*'
    paths = [a.report] if a.report else list((a.project/'comparison_reports').glob(pattern))
    valid = []
    for path in paths:
        if (path/'metrics.json').is_file():
            m = read(path/'metrics.json')
            if m.get('status') == 'completed' and m.get('is_formal') and m.get('seed') == a.seed and m.get('training', {}).get('completed_epochs') == 200:
                valid.append((path, m))
    if len(valid) != 1:
        raise RuntimeError('Need one formal E1-640 report; use --report to resolve ambiguity.')
    directory, m = valid[0]
    expected = dict(imgsz=640, area_batch=4, conf=.001, iou=.7, max_det=500, half=False)
    for k, v in expected.items():
        if m['evaluation_protocol'].get(k) != v:
            raise RuntimeError(f'Protocol differs: {k}')
    source_hashes = m['p2_architecture']['source_code_sha256']
    # Checkpoint loading does not call the YAML parser. tasks.py also contains
    # registrations for unrelated later experiments: record any drift rather
    # than equate whole-file registration changes with an E1 forward change.
    # Core layer implementations stay strict; real predictions are gated below.
    source_audit = {}
    for rel, digest in source_hashes.items():
        if rel.startswith('nn/'):
            actual = sha(a.source/'ultralytics'/rel, True)
            source_audit[rel] = dict(expected=digest, actual=actual, matches=actual==digest)
            if rel != 'nn/tasks.py' and actual != digest:
                raise RuntimeError(f'Network layer implementation drift: {rel}. Do not bypass this check.')
    if not source_audit['nn/tasks.py']['matches']:
        print('tasks.py whole-file drift recorded; parser is unused. Real prediction reproduction is mandatory before conclusions.', flush=True)
    weights = a.project/'runs'/f'e1_yolo11s_p2add_img640_seed{a.seed}'/'weights/best.pt'
    if sha(weights) != m['evaluation_weights']['sha256']:
        raise RuntimeError('Checkpoint hash mismatch.')
    evaluator = Path(__file__).with_name('d27_official.py')
    if sha(evaluator, True) != 'a9e0f50885f81e72ed85baf57a880800c6c51a51527a646b5c60184a1b76bf60':
        raise RuntimeError('Frozen official evaluator hash mismatch.')
    spec = importlib.util.spec_from_file_location('d38_d27', evaluator)
    d27 = importlib.util.module_from_spec(spec); sys.modules[spec.name] = d27; spec.loader.exec_module(d27)
    checked['official'] = d27.run_unit_tests()
    if a.check_only:
        print(json.dumps(checked, indent=2)); print('D38 check passed; no training. Next --smoke-images 8.'); return
    if a.device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.')
    records, dataset = d27.load_dataset(a.dataset)
    images = read(directory/'area_ground_truth.json')['images']
    if len(images) != len(records) or any((r.image_id, r.stem, r.width, r.height) != (im['id'], Path(im['file_name']).stem, im['width'], im['height']) for r, im in zip(records, images)):
        raise RuntimeError('Dataset mapping mismatch.')
    if a.smoke_images and (a.smoke_images < 4 or a.smoke_images % 4 or a.smoke_images >= len(records)):
        raise ValueError('Smoke size must be a multiple of 4, at least 4, below 548.')
    signature = dict(script=sha(__file__), evaluator=sha(evaluator, True), weights=sha(weights),
                     saved_predictions=sha(directory/'area_predictions.json'), dataset=dataset,
                     source={rel: sha(a.source/'ultralytics'/rel, True) for rel in ('nn/tasks.py','nn/modules/head.py','nn/modules/block.py','nn/modules/conv.py','utils/ops.py','models/yolo/detect/predict.py','engine/predictor.py')},
                     torch=torch.__version__, ultralytics=ultralytics.__version__, device=a.device, protocol=expected)
    gate = a.project/'comparison_reports'/f'd38_seed{a.seed}_smoke_passed.json'
    if not a.smoke_images and (not gate.is_file() or read(gate).get('signature') != signature):
        raise RuntimeError('Matching smoke gate missing. Run --smoke-images 8 first.')
    selected = records[:a.smoke_images] if a.smoke_images else records
    by_path = {str(r.image_path.resolve()): r for r in selected}
    output = a.project/'comparison_reports'/('DIAG_D38_E1_RAW_STAGES_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', signature=signature, tests=checked, seed=a.seed,
                  subset_only=bool(a.smoke_images), boundaries=BOUNDARY, source_audit=source_audit)
    write(output/'metrics.json', report)
    rows, predictions, counts = [], [], []

    class AuditPredictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs):
            if not isinstance(preds, (tuple, list)) or len(preds) != 2 or not isinstance(preds[1], list):
                raise RuntimeError('Expected ordinary Detect (decoded, four raw maps).')
            raw = preds[0].detach().clone()
            if not torch.isfinite(raw).all():
                raise RuntimeError('Nonfinite raw predictions.')
            maps = preds[1]
            if raw.shape[1] != 14 or len(maps) != 4 or sum(z.shape[-2]*z.shape[-1] for z in maps) != raw.shape[2]:
                raise RuntimeError('E1 head must be nc=10, nl=4, consistent anchor count.')
            strides = [int(img.shape[-2]/z.shape[-2]) for z in maps]
            if strides != [4, 8, 16, 32]:
                raise RuntimeError(f'Unexpected strides: {strides}')
            level = np.concatenate([np.full(z.shape[-2]*z.shape[-1], s) for z, s in zip(maps, strides)])
            # Same actual NMS routine, only remove final budget for diagnosis.
            all_nms = ops.non_max_suppression(raw.clone(), self.args.conf, self.args.iou,
                max_det=30000, max_nms=30000, max_time_img=60)
            results = super().postprocess(preds, img, orig_imgs)
            for index, (result, kept) in enumerate(zip(results, all_nms)):
                rec = by_path[str(Path(result.path).resolve())]
                normal = result.boxes.data.detach()
                kept[:, :4] = ops.scale_boxes(img.shape[2:], kept[:, :4], result.orig_shape)
                torch.testing.assert_close(kept[:500], normal, rtol=0, atol=0)
                b = ops.xywh2xyxy(raw[index, :4].T.clone())
                b = ops.scale_boxes(img.shape[2:], b, result.orig_shape).cpu().numpy()
                scores, cls = raw[index, 4:].max(0)
                scores = scores.cpu().numpy(); cls = cls.cpu().numpy()+1
                positive = scores > .001
                ids = np.flatnonzero(positive)
                if len(ids) > 30000:
                    ids = np.flatnonzero(positive)[torch.tensor(scores[positive], device=raw.device).argsort(descending=True)[:30000].cpu().numpy()]
                g = rec.eval_gt
                g = g[(g[:, 4] == 0) & (g[:, 5] >= 1) & (g[:, 5] <= 10)]
                gt = g[:, :4].copy(); gt[:, 2:] += gt[:, :2]
                gc = g[:, 5].astype(int)
                coverage = {
                    'raw_any': iou_max(gt, b),
                    'raw_argmax_correct': iou_max(gt, b, cls, gc),
                    'confidence': iou_max(gt, b[positive], cls[positive], gc),
                    'max_nms': iou_max(gt, b[ids], cls[ids], gc),
                    'nms_unlimited': iou_max(gt, kept[:, :4].cpu().numpy(), kept[:, 5].cpu().numpy()+1, gc),
                    'top500': iou_max(gt, normal[:, :4].cpu().numpy(), normal[:, 5].cpu().numpy()+1, gc),
                }
                per_level = {str(s): iou_max(gt, b[level == s], cls[level == s], gc) for s in strides}
                size = np.sqrt(g[:, 2]*g[:, 3])*640/max(rec.width, rec.height)
                centers = (gt[:, :2]+gt[:, 2:])/2*640/max(rec.width, rec.height)
                distance = np.linalg.norm(centers[:, None]-centers[None], axis=2)
                np.fill_diagonal(distance, np.inf)
                density = (distance <= 16).sum(1)
                for j in range(len(g)):
                    rows.append(dict(image_id=rec.image_id, gt_index=j, cls=int(gc[j]), size=float(size[j]),
                                     density=int(density[j]), iou={k: float(v[j]) for k,v in coverage.items()},
                                     raw_correct_per_stride={k: float(v[j]) for k,v in per_level.items()}))
                counts.append(dict(image_id=rec.image_id, raw=len(b), confidence=int(positive.sum()),
                                   after_max_nms=len(ids), after_nms=len(kept), top500=len(normal), input_shape=list(img.shape[2:])))
                for x1,y1,x2,y2,score,c in normal.cpu().tolist():
                    predictions.append(dict(image_id=rec.image_id, category_id=int(c)+1,
                        bbox=[x1,y1,max(0.,x2-x1),max(0.,y2-y1)],score=score))
            return results

    started = time.perf_counter()
    model = YOLO(str(weights))
    opts = prediction_options(a.device)
    try:
        # This project's Ultralytics 8.3.0 accepts a predictor INSTANCE,
        # unlike newer APIs that accept a class. Keep the local API semantics.
        predictor = AuditPredictor(overrides={**model.overrides, **opts, 'mode':'predict'}, _callbacks=model.callbacks)
        # No warm-up image enters statistics: AutoBackend warmup does not postprocess.
        for start in range(0, len(selected), 4):
            chunk = selected[start:start+4]
            model.predict([str(r.image_path) for r in chunk], batch=len(chunk), predictor=predictor, stream=False, **opts)
            if start == 0 or (start+len(chunk)) % 40 == 0 or start+len(chunk) == len(selected):
                print(f'D38 {start+len(chunk)}/{len(selected)} images', flush=True)
        write(output/'normal_predictions.json', predictions)
        if len(counts) != len(selected) or {z['image_id'] for z in counts} != {r.image_id for r in selected}:
            raise RuntimeError('Diagnostic missed or duplicated images.')
        # Exact imagewise numerical reproduction, including smoke subset.
        saved = d27.load_predictions(directory/'area_predictions.json', len(records))
        current = d27.load_predictions(output/'normal_predictions.json', len(records))
        for rec in selected:
            # Column 7 is file-order metadata, not a prediction attribute.
            np.testing.assert_allclose(current[rec.image_id-1][:, :6], saved[rec.image_id-1][:, :6], rtol=0, atol=1e-5)
        report['normal_saved_prediction_reproduction'] = True
        if not a.smoke_images:
            filtered, audit = d27.filter_detections_and_audit(records, current)
            sf, _ = d27.filter_detections_and_audit(records, saved)
            official, reference = d27.official_evaluate(records, filtered), d27.official_evaluate(records, sf)
            delta = {k: official[k]-reference[k] for k in KEYS}
            if max(abs(v) for v in delta.values()) > .002:
                raise RuntimeError(f'Official reproduction failed: {delta}')
            report.update(official=official, official_delta_pp=delta, ignore_audit=audit)
        def summarize(items):
            result = dict(gt_count=len(items), candidate_coverage_percent={}, sequential_loss_counts={}, raw_per_stride_percent={})
            if not items: return result
            for threshold in (.5, .75):
                key = str(threshold)
                result['candidate_coverage_percent'][key] = {s: 100*sum(r['iou'][s]>=threshold for r in items)/len(items) for s in STAGES}
                result['sequential_loss_counts'][key] = {f'{u}->{v}': sum(r['iou'][u]>=threshold and r['iou'][v]<threshold for r in items) for u,v in zip(STAGES, STAGES[1:])}
                result['raw_per_stride_percent'][key] = {s:100*sum(r['raw_correct_per_stride'][s]>=threshold for r in items)/len(items) for s in ('4','8','16','32')}
            return result
        slices = {'all':summarize(rows)}
        for lo,hi,name in [(0,8,'lt8'),(8,16,'8_16'),(16,24,'16_24'),(24,32,'24_32'),(32,float('inf'),'ge32')]:
            group = [r for r in rows if lo<=r['size']<hi]
            slices[name] = summarize(group)
            for dn,condition in [('0',lambda n:n==0),('1_3',lambda n:1<=n<=3),('ge4',lambda n:n>=4)]:
                slices[name+'/density_'+dn] = summarize([r for r in group if condition(r['density'])])
            for c in range(1,11):
                slices[name+'/class_'+str(c)] = summarize([r for r in group if r['cls']==c])
        report.update(status='completed', seconds=time.perf_counter()-started, slices=slices, image_counts=counts)
        write(output/'gt_stage_rows.json', rows)
        write(output/'metrics.json', report)
        lines = ['# D38 E1-640 raw-stage audit', '', 'Normal saved predictions reproduced. No training.', '',
                 '|size|GT|raw any @.75|raw correct @.75|conf @.75|max_nms @.75|NMS @.75|top500 @.75|', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name in ('all','lt8','8_16','16_24','24_32','ge32'):
            z=slices[name]; v=z['candidate_coverage_percent'].get('.75', z['candidate_coverage_percent'].get('0.75',{}))
            lines.append('|'+name+'|'+str(z['gt_count'])+'|'+ '|'.join(f'{v.get(k,0):.3f}' for k in STAGES)+'|')
        lines += ['', 'Detailed density/class/stride breakdown: metrics.json; per-GT evidence: gt_stage_rows.json.', '', 'Interpretation boundaries:', ''] + ['- '+b for b in BOUNDARY]
        (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
        if sha(weights) != signature['weights']:
            raise RuntimeError('Checkpoint changed during audit.')
        if a.smoke_images:
            write(gate, dict(signature=signature, report=str(output)))
        print(f'D38 completed: {output}', flush=True)
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        write(output/'metrics.json', report)
        raise

if __name__ == '__main__':
    main()
