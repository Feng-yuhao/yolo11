#!/usr/bin/env python3
"""D47: frozen E40 candidates -> fixed endpoint quality fitting -> official val AP."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
import numpy as np
import d27_official as official

PACKAGE = Path(__file__).resolve().parent
REFS = {1: 'E40_RUN_20260929_205658_507620', 2: 'E40S2_RUN_20260930_130808_314014'}
KEYS = ('AP50_95_percent', 'AP50_percent', 'AP75_percent', 'AR500_percent')
OPTIONS = dict(imgsz=640, conf=.001, iou=.7, max_det=500, half=False, augment=False,
               verbose=False, save=False, agnostic_nms=False, classes=None)
TRAIN_COUNT = 6471
VAL_HASH = '058c87e779a61a09933bebe35af1c05c807a444dc842b4be12124dd36134c787'


def require(ok, text):
    if not ok:
        raise RuntimeError(text)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def bind(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def setup(args):
    sys.path.insert(0, str(args.source.resolve()))
    os.environ['YOLO_CONFIG_DIR'] = str(args.project/'comparison_reports'/'d47_settings')
    import torch
    import ultralytics
    require(Path(ultralytics.__file__).resolve().is_relative_to(args.source.resolve()), 'Wrong Ultralytics source')
    require(ultralytics.__version__ == '8.3.0', 'Requires pinned project Ultralytics 8.3.0')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.device == '0':
        require(torch.cuda.is_available(), 'CUDA unavailable')
        free, total = torch.cuda.mem_get_info()
        require(args.allow_busy_gpu or free/total >= .9, 'GPU busy; wait or explicitly allow it')
        require(free >= (args.memory_cap_gib+2)*1024**3, 'Need allocator cap + 2 GiB free')
        torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib*1024**3/total)
    return torch


def load_train(path):
    from train_annotations import load_training_dataset
    return load_training_dataset(path, expected_images=TRAIN_COUNT)


def preflight(args):
    print('D47 preflight: checkpoint and original train/val annotation audit', flush=True)
    specs = [s for s in read(PACKAGE/'models.json')['models'] if s['model'] == 'E40' and s['seed'] in args.seeds]
    require(len(specs) == len(args.seeds), 'Missing pinned E40 seed')
    for name, sha in read(PACKAGE/'models.json')['vendor_sha256'].items():
        require(official.sha256(PACKAGE/name) == sha, 'Frozen dependency changed: '+name)
    train, train_meta = load_train(args.train_dataset)
    val, val_meta = official.load_dataset(args.val_dataset)
    require(val_meta['annotation_dimension_sha256'] == VAL_HASH, 'Wrong original val data')
    require(not set(r.stem for r in train).intersection(r.stem for r in val), 'Train/val stems overlap')
    for spec in specs:
        w = args.project/spec['weights']
        require(w.is_file() and official.sha256(w) == spec['sha256'], 'Wrong/missing checkpoint '+str(w))
        ref = args.project/'comparison_reports'/REFS[spec['seed']]
        require((ref/'area_predictions.json').is_file(), 'Missing original E40 reference predictions')
        metrics = read(ref/'official_metrics.json')
        spec['expected'] = {k: metrics['candidate'][k] for k in KEYS}
        spec['reference_sha256'] = official.sha256(ref/'area_predictions.json')
    from ranking import REVISION
    # Image size/mtime fingerprint detects changed local data across gate/cache use;
    # annotation bytes/dimensions have their own content digest.
    images = [(str(r.image_path.resolve()), r.image_path.stat().st_size, r.image_path.stat().st_mtime_ns)
              for r in train+val]
    payload = dict(revision=REVISION, specs=specs, train=train_meta, val=val_meta,
        image_stat_sha256=bind(images), protocol=dict(**OPTIONS, batch=4, device=args.device),
        memory_cap_gib=args.memory_cap_gib, fit=dict(epochs=10, images_per_step=8, lr=.001,
        weight_decay=.0001, rank_weight=1., bound=2., foreground_budget=256, negative_budget=256),
        source_sha={p: official.sha256(args.source/'ultralytics'/p) for p in
                    ('nn/tasks.py', 'nn/modules/head.py', 'utils/ops.py', 'engine/predictor.py')},
        code_sha={p.name: official.sha256(p) for p in PACKAGE.iterdir() if p.suffix in ('.py', '.json', '.sh')})
    import torch
    payload['torch'] = torch.__version__
    payload['gpu'] = torch.cuda.get_device_name() if args.device == '0' else 'cpu'
    return train, val, specs, payload


def receipt(args, mode):
    return args.project/'comparison_reports'/('d47_'+mode+'_passed.json')


def verify(args, payload, modes):
    for mode in modes:
        path = receipt(args, mode)
        require(path.is_file() and read(path)['binding'] == bind(payload), 'Missing/stale '+mode+' gate')


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def tapped_detector(args, spec):
    import torch
    import spatial_task_head
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from spatial_task_head import SpatialTaskDetect, NEW_PARAMETERS
    model = YOLO(str(args.project/spec['weights']), task='detect')
    h = model.model.model[-1]
    require(type(h) is SpatialTaskDetect and h.enabled and len(h.task_blocks) == 2, 'Not E40')
    require(h.nl == 4 and h.nc == 10 and h.reg_max == 16 and list(h.f) == [25, 16, 19, 22], 'Wrong Detect layout')
    require(h.stride.tolist() == [4., 8., 16., 32.], 'Wrong strides')
    require(sum(p.numel() for p in h.task_blocks.parameters()) == NEW_PARAMETERS, 'E40 branch drift')
    require(model.names == {i-1: n for i, n in official.CLASS_NAMES.items()}, 'Wrong class names')
    model.model.eval().requires_grad_(False)
    predictor = DetectionPredictor(overrides={**model.overrides, **OPTIONS,
        'device': args.device, 'batch': 4, 'mode': 'predict'}, _callbacks=model.callbacks)
    predictor.setup_model(model=model.model, verbose=False)  # pinned AutoBackend fuse/eval
    model.predictor = predictor  # prevent model.predict from constructing/fusing a second backend
    network = predictor.model.model
    head = network.model[-1]
    stash = {}
    handles = []
    def capture(key):
        def hook(module, inputs):
            stash[key] = inputs[0].detach()
        return hook
    for i in range(4):
        handles.append(head.cv2[i][-1].register_forward_pre_hook(capture(('reg', i))))
        handles.append(head.cv3[i][-1].register_forward_pre_hook(capture(('cls', i))))
    before = frozen_digest(network)
    try:
        yield model, predictor, stash
        require(before == frozen_digest(network), 'Frozen E40 parameters/BN buffers changed')
        require(not network.training and all(p.grad is None and not p.requires_grad for p in network.parameters()),
                'E40 received gradient or was put into training mode')
    finally:
        for handle in handles:
            handle.remove()
        del model, predictor, network, stash
        gc.collect()
        if args.device == '0':
            torch.cuda.empty_cache()


def independent_nms(raw, network_shape, original_shape):
    from ultralytics.utils import ops
    kept = ops.non_max_suppression(raw.clone()[None], .001, .7, nc=10, max_det=500,
                                    max_time_img=60, in_place=False)[0]
    kept[:, :4] = ops.scale_boxes(network_shape, kept[:, :4], original_shape)
    return kept


def prediction_rows(tensor, image_id):
    return [dict(image_id=image_id, category_id=int(c)+1,
                 bbox=[x1, y1, max(0., x2-x1), max(0., y2-y1)], score=s)
            for x1, y1, x2, y2, s, c in tensor.cpu().tolist()]


def forward_records(args, spec, records, callback):
    import torch
    with tapped_detector(args, spec) as (model, predictor, stash):
        original = predictor.postprocess
        by_path = {str(r.image_path.resolve()): r for r in records}
        seen = set()
        def postprocess(preds, img, orig_imgs):
            require(isinstance(preds, (tuple, list)) and len(preds[1]) == 4, 'Raw tensors missing')
            raw = preds[0].detach().clone()
            maps = [(stash[('reg', i)], stash[('cls', i)], preds[1][i]) for i in range(4)]
            # DetectionPredictor NMS mutates its input; NEVER feed the original raw
            # object subsequently used for ranker extraction/re-scoring.
            baseline = original(raw.clone(), img, orig_imgs)
            for i, result in enumerate(baseline):
                record = by_path[str(Path(result.path).resolve())]
                require(record.stem not in seen, 'Repeated image')
                seen.add(record.stem)
                exact = independent_nms(raw[i], img.shape[2:], result.orig_shape)
                torch.testing.assert_close(exact, result.boxes.data, rtol=0, atol=0)
                callback(record, raw[i], maps, i, img.shape[2:], result)
            return baseline
        predictor.postprocess = postprocess
        started = time.monotonic()
        for start in range(0, len(records), 4):
            part = records[start:start+4]
            model.predict([str(r.image_path) for r in part], predictor=predictor,
                          batch=len(part), stream=False, device=args.device, **OPTIONS)
            done = start+len(part)
            if done % 40 == 0 or done == len(records):
                elapsed = time.monotonic()-started
                print(f"D47 {spec['key']} {done}/{len(records)} images elapsed={elapsed:.1f}s "
                      f"ETA={(len(records)-done)*elapsed/max(done,1)/60:.1f}min", flush=True)
        require(len(seen) == len(records), 'Incomplete frozen inference')
    require(official.sha256(args.project/spec['weights']) == spec['sha256'], 'Disk checkpoint changed')


def excluded(record, boxes_xyxy, labels, quality=None):
    xywh = boxes_xyxy.copy(); xywh[:, 2:] -= xywh[:, :2]
    mask = np.array([official.official_region_coverage(b, record.ignore_rects, record.width, record.height) >= .5
                     for b in xywh], dtype=bool)
    for cls in range(1, 11):
        ignored = record.eval_gt[(record.eval_gt[:, 4] == 1) & (record.eval_gt[:, 5] == cls)]
        ci = np.flatnonzero(labels == cls)
        if len(ignored) and len(ci):
            # Use original evaluator's ignored-GT IoF, not ordinary IoU.
            coverage = official.overlap_matrix(xywh[ci], ignored[:, :4], np.ones(len(ignored)))
            # Official matching prefers regular GT over score-zero ignored GT.
            # Do not discard a regular-positive candidate just because it also
            # overlaps an ignored object. Region filtering above remains unconditional.
            regular_positive = np.zeros(len(ci), bool) if quality is None else quality[ci] >= .5
            mask[ci] |= (coverage.max(1) >= .5) & ~regular_positive
    return mask


def cache_candidates(args, spec, records, root, payload, partial=False):
    import torch
    from ultralytics.utils import ops
    import ranking as rank
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root/'cache.json'
    binding = bind(dict(payload=payload, key=spec['key'], subset=[r.stem for r in records], partial=partial))
    if manifest_path.is_file():
        m = read(manifest_path)
        require(m['binding'] == binding and m['completed'], 'Cache provenance differs')
        for item in m['shards']:
            require(official.sha256(root/item['file']) == item['sha256'], 'Cache shard corrupt')
        print('D47 reused verified cache '+str(root), flush=True)
        return m
    shards = []
    def save(record, raw, maps, index, shape, result):
        score, labels = raw[4:].max(0)
        ids = torch.where(score > rank.SUPPORT)[0]
        labels = labels[ids].cpu().numpy()+1
        boxes = ops.scale_boxes(shape, ops.xywh2xyxy(raw[:4, ids].T.clone()), result.orig_shape).cpu().numpy()
        q, owner = rank.targets(boxes, labels, record.eval_gt)
        scores = score[ids].cpu().numpy()
        rng = np.random.RandomState(int(hashlib.sha256(record.stem.encode()).hexdigest()[:8], 16))
        sampled = rank.sample_candidates(q, owner, scores, rng)
        ignore = excluded(record, boxes[sampled], labels[sampled], q[sampled])
        sampled = sampled[~ignore]
        q, owner, scores = q[sampled], owner[sampled], scores[sampled]
        all_features = rank.features(raw, maps, index)
        x = all_features[ids[sampled]].cpu().numpy().astype(np.float32)
        require(np.isfinite(x).all(), 'Invalid features')
        filename = f'{record.image_id:05d}.npz'
        temp = root/(filename+'.tmp')
        with temp.open('wb') as handle:
            # Audit-only coordinates/classes/IDs are NEVER consumed as fitted inputs.
            np.savez_compressed(handle, x=x, q=q, score=scores, pair=rank.pairs(q, owner), owner=owner,
                                original_xyxy=boxes[sampled], predicted_class=labels[sampled],
                                candidate_id=ids[sampled].cpu().numpy())
        os.replace(temp, root/filename)
        shards.append(dict(file=filename, stem=record.stem, image_id=record.image_id,
                           original_width=record.width, original_height=record.height,
                           heldout=rank.holdout(record.stem),
                           rows=len(x), pairs=len(rank.pairs(q, owner)), ignored=int(ignore.sum()),
                           positives=int((q >= .5).sum()), sha256=official.sha256(root/filename)))
    forward_records(args, spec, records, save)
    m = dict(binding=binding, completed=True, partial=partial, shards=shards, checkpoint_sha=spec['sha256'],
             train_only=True, features_dtype='float32', selection='256 positive / 128 hard + up to 128 random negative',
             scope='raw pre-NMS candidates above bounded-correction support floor')
    write(manifest_path, m)
    print('D47 CACHE COMPLETE '+str(root)+' rows='+str(sum(s['rows'] for s in shards)), flush=True)
    return m


def load_shard(root, item):
    with np.load(root/item['file'], allow_pickle=False) as value:
        return {key: value[key].copy() for key in ('x', 'q', 'score', 'pair')}


def train_normalizer(root, shards):
    from ranking import FEATURE_DIM
    total = np.zeros(FEATURE_DIM, np.float64); square = total.copy(); count = 0
    for item in shards:
        x = load_shard(root, item)['x'].astype(np.float64)
        total += x.sum(0); square += (x*x).sum(0); count += len(x)
    require(count > 0, 'No fit-partition rows')
    mean = total/count
    std = np.sqrt(np.maximum(square/count-mean*mean, 0)).clip(.01)
    return mean.astype(np.float32), std.astype(np.float32)


def minibatch(root, items, device):
    import torch
    values = [load_shard(root, item) for item in items]
    offset, pair = 0, []
    for v in values:
        pair.append(v['pair']+offset); offset += len(v['x'])
    require(offset > 0, 'Empty supervised batch')
    return (torch.from_numpy(np.concatenate([v['x'] for v in values])).to(device),
            torch.from_numpy(np.concatenate([v['score'] for v in values])).to(device),
            torch.from_numpy(np.concatenate([v['q'] for v in values])).to(device),
            torch.from_numpy(np.concatenate(pair)).to(device))


def fit_rankers(args, cache_root, cache, destination, smoke=False):
    import torch
    import ranking as rank
    destination.mkdir(parents=True, exist_ok=True)
    fit = [s for s in cache['shards'] if not s['heldout'] and s['rows']]
    held = [s for s in cache['shards'] if s['heldout'] and s['rows']]
    if smoke:
        fit = [s for s in cache['shards'] if s['rows']]
        held = []
    require(fit and (smoke or held), 'Missing grouped train fit/holdout partition')
    mean, std = train_normalizer(cache_root, fit)
    device = 'cuda:0' if args.device == '0' else 'cpu'
    models = {v: rank.Ranker(v, mean, std).to(device) for v in ('dfl', 'full')}
    optimizers = {v: torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.0001) for v, m in models.items()}
    epochs = 1 if smoke else 10
    histories = []
    started = time.monotonic()
    for epoch in range(epochs):
        order = np.random.RandomState(47000+epoch).permutation(len(fit))
        accum = {v: dict(loss=0., pairs=0, correct=0.) for v in models}
        steps = 0
        for start in range(0, len(order), 8):
            items = [fit[i] for i in order[start:start+8]]
            x, score, q, pair = minibatch(cache_root, items, device)
            for v, m in models.items():
                m.train(); optimizers[v].zero_grad(set_to_none=True)
                loss, metrics = rank.objective(m(x), score, q, pair)
                require(torch.isfinite(loss).item(), 'Nonfinite ranker loss')
                loss.backward()
                require(all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None),
                        'Nonfinite ranker gradient')
                torch.nn.utils.clip_grad_norm_(m.parameters(), 10.)
                optimizers[v].step()
                accum[v]['loss'] += float(loss.detach())
                accum[v]['pairs'] += metrics['pairs']
                accum[v]['correct'] += float(metrics['pair_accuracy'])*metrics['pairs']
            steps += 1
            if steps % 100 == 0:
                print(f'D47 FIT epoch={epoch+1}/{epochs} step={steps}/{(len(fit)+7)//8} elapsed={(time.monotonic()-started)/60:.1f}min', flush=True)
        held_metrics = {}
        with torch.no_grad():
            for v, m in models.items():
                m.eval(); count = correct = base_correct = 0; loss_sum = held_steps = 0
                for start in range(0, len(held), 8):
                    x, score, q, pair = minibatch(cache_root, held[start:start+8], device)
                    loss, met = rank.objective(m(x), score, q, pair)
                    count += met['pairs']; correct += float(met['pair_accuracy'])*met['pairs']
                    if len(pair):
                        base_correct += int((score[pair[:, 0]] > score[pair[:, 1]]).sum())
                    loss_sum += float(loss); held_steps += 1
                held_metrics[v] = dict(pairs=count, adjusted_accuracy=correct/max(1,count),
                    original_accuracy=base_correct/max(1,count), loss=loss_sum/max(1,held_steps))
        hist = dict(epoch=epoch+1, train={v: dict(loss=a['loss']/steps,
                    pair_accuracy=a['correct']/max(1,a['pairs']), pairs=a['pairs']) for v,a in accum.items()},
                    train_group_holdout=held_metrics, elapsed_seconds=time.monotonic()-started)
        histories.append(hist); write(destination/'history.json', histories)
        print('D47 FIT '+json.dumps(hist, ensure_ascii=False), flush=True)
        # Save every last endpoint for interruption recovery; NO best selection.
        for v,m in models.items():
            temp = destination/(v+'.pt.tmp')
            torch.save(dict(state={k:t.detach().cpu() for k,t in m.state_dict().items()}, variant=v,
                mean=mean, std=std, epoch=epoch+1, frozen_checkpoint_sha=cache['checkpoint_sha'],
                cache_binding=cache['binding'], optimizer=optimizers[v].state_dict(),
                normalizer_fit_rows=sum(s['rows'] for s in fit), train_only=True), temp)
            os.replace(temp, destination/(v+'.pt'))
    # A tiny fit is engineering smoke only; cannot enter formal eval automatically.
    write(destination/'fit.json', dict(completed=True, epochs=epochs, smoke=smoke,
        cache_binding=cache['binding'], fixed_endpoint=True, validation_used_for_fitting=False,
        fit_images=len(fit), heldout_images=len(held)))
    for m in models.values():
        m.eval().requires_grad_(False)
    return models


def compare_original(args, spec, records, predictions):
    ref = read(args.project/'comparison_reports'/REFS[spec['seed']]/'area_predictions.json')
    def grouped(rows):
        d = {}
        for r in rows: d.setdefault(r['image_id'], []).append(r)
        return d
    a,b = grouped(predictions), grouped(ref)
    for r in records:
        left,right = a.get(r.image_id, []),b.get(r.image_id, [])
        require(len(left) == len(right), 'Original prediction count drift')
        for x,y in zip(left,right):
            require(x['category_id'] == y['category_id'] and abs(x['score']-y['score']) <= 1e-6 and
                    np.max(np.abs(np.array(x['bbox'])-np.array(y['bbox']))) <= .002,
                    'Original E40 prediction/order/coordinate reproduction failed')


def evaluate(args, spec, records, models, destination, partial=False):
    import torch
    import ranking as rank
    destination.mkdir(parents=True, exist_ok=True)
    results = {v: [] for v in ('baseline', 'dfl', 'full')}
    def inspect(record, raw, maps, index, shape, original):
        x = rank.features(raw, maps, index)
        results['baseline'].extend(prediction_rows(original.boxes.data, record.image_id))
        for v, m in models.items():
            adjusted = rank.calibrate(raw, x, m)
            kept = independent_nms(adjusted, shape, original.orig_shape)
            results[v].extend(prediction_rows(kept, record.image_id))
    with torch.no_grad():
        forward_records(args, spec, records, inspect)
    compare_original(args, spec, records, results['baseline'])
    metrics = {}
    for v, predictions in results.items():
        path = destination/(v+'_predictions.json'); write(path, predictions)
        filtered, ignore = official.filter_detections_and_audit(records, official.load_predictions(path, 548))
        metrics[v] = official.official_evaluate(records, filtered)
        write(destination/(v+'_official.json'), dict(metrics=metrics[v], ignore=ignore))
        print(f'D47 {spec["key"]} {v} Official AP={metrics[v]["AP50_95_percent"]:.6f}', flush=True)
    if not partial:
        require(all(abs(metrics['baseline'][k]-spec['expected'][k]) < 1e-6 for k in KEYS), 'Official baseline failed to reproduce')
    result = dict(metrics=metrics, deltas={v:{k:metrics[v][k]-metrics['baseline'][k] for k in KEYS}
        for v in ('dfl','full')}, baseline_reproduced=True, subset_only=partial,
        box_coordinates_unchanged=True, predicted_class_unchanged=True, frozen_weights_unchanged=True)
    write(destination/'comparison.json', result)
    return result


def load_rankers(args, destination, spec, cache):
    import torch
    from ranking import Ranker
    result = {}
    fit = read(destination/'fit.json')
    require(fit['completed'] and fit['epochs'] == 10 and not fit['smoke'] and fit['cache_binding'] == cache['binding'],
            'Need full fixed ten-epoch endpoint, not smoke/partial/best')
    for v in ('dfl', 'full'):
        saved = torch.load(destination/(v+'.pt'), map_location='cpu', weights_only=False)
        require(saved['frozen_checkpoint_sha'] == spec['sha256'] and saved['cache_binding'] == cache['binding'], 'Wrong ranker provenance')
        m = Ranker(v, saved['mean'], saved['std'])
        m.load_state_dict(saved['state'], strict=True)
        result[v] = m.to('cuda:0' if args.device == '0' else 'cpu').eval().requires_grad_(False)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['check','smoke','verify-gates','cache','fit','eval','run','self-test'])
    p.add_argument('--project', type=Path, default=Path('/root/autodl-tmp/yolo11_project'))
    p.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/yolo11_src'))
    p.add_argument('--train-dataset', type=Path, default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-train'))
    p.add_argument('--val-dataset', type=Path, default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val'))
    p.add_argument('--seeds', type=int, nargs='+', choices=[1,2], default=[1,2])
    p.add_argument('--device', choices=['cpu','0'], default='0')
    p.add_argument('--memory-cap-gib', type=float, default=6.)
    p.add_argument('--allow-busy-gpu', action='store_true')
    args = p.parse_args(); args.seeds = sorted(set(args.seeds))
    require(2 <= args.memory_cap_gib <= 12, 'Cap must be 2..12 GiB')
    setup(args)
    from unit_checks import run_checks
    if args.mode == 'self-test':
        print(json.dumps(run_checks(args.device), indent=2)); return
    train,val,specs,payload = preflight(args)
    if args.mode == 'verify-gates':
        verify(args, payload, ['check','smoke']); print('D47 CHECK AND REAL SMOKE VALID'); return
    out = args.project/'comparison_reports'/('D47_'+args.mode.upper()+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    write(out/'manifest.json', payload)
    write(out/'status.json', dict(status='running', started_at=now(), mode=args.mode))
    print('D47 START '+now()+' REPORT='+str(out), flush=True)
    cache_base = args.project/'d47_candidate_cache'/bind(payload)[:16]
    fit_base = args.project/'runs'/'d47_frozen_quality'/bind(payload)[:16]
    try:
        if args.mode == 'check':
            tests = run_checks(args.device); official.run_unit_tests(); write(out/'tests.json', tests)
            for spec in specs:
                # This check exercises actual pickle load/fuse/head hooks, not fitting.
                with tapped_detector(args, spec): pass
            write(receipt(args, 'check'), dict(binding=bind(payload), report=str(out), passed=True))
            print('D47 CHECK PASSED', flush=True)
        elif args.mode == 'smoke':
            verify(args, payload, ['check'])
            for spec in specs:
                root = out/spec['key']; cache_root = root/'cache'
                cache = cache_candidates(args, spec, train[:8], cache_root, payload, partial=True)
                models = fit_rankers(args, cache_root, cache, root/'fit', smoke=True)
                evaluate(args, spec, val[:8], models, root/'eval', partial=True)
                del models; gc.collect()
            write(receipt(args,'smoke'), dict(binding=bind(payload), report=str(out), passed=True))
            print('D47 EIGHT-TRAIN / EIGHT-VAL IMAGE SMOKE PASSED', flush=True)
        else:
            verify(args, payload, ['check','smoke'])
            require(shutil.disk_usage(args.project).free >= 8*1024**3, 'Need >=8 GiB free disk for two candidate caches')
            summary = {}
            for spec in specs:
                cache_root = cache_base/spec['key']; destination = fit_base/spec['key']
                if args.mode in ('cache','run'):
                    cache = cache_candidates(args, spec, train, cache_root, payload)
                else:
                    cache = read(cache_root/'cache.json')
                    expected_binding = bind(dict(payload=payload,key=spec['key'],subset=[r.stem for r in train],partial=False))
                    require(cache['binding'] == expected_binding and cache['completed'] and not cache['partial'], 'Wrong cache')
                    for shard in cache['shards']:
                        require(official.sha256(cache_root/shard['file']) == shard['sha256'], 'Corrupt cache')
                if args.mode in ('fit','run'):
                    fit_rankers(args, cache_root, cache, destination)
                if args.mode in ('eval','run'):
                    models = load_rankers(args, destination, spec, cache)
                    summary[spec['key']] = evaluate(args, spec, val, models, out/spec['key'])
                    del models; gc.collect()
            if summary:
                write(out/'all_models.json', summary)
                table = '# D47 fixed-endpoint frozen-candidate ranking\n\n|Model|Baseline AP|DFL AP|Full AP|Full delta pp|\n|---|---:|---:|---:|---:|\n'
                for key,s in summary.items():
                    table += '|{}|{:.4f}|{:.4f}|{:.4f}|{:+.4f}|\n'.format(key,
                        s['metrics']['baseline'][KEYS[0]], s['metrics']['dfl'][KEYS[0]],
                        s['metrics']['full'][KEYS[0]], s['deltas']['full'][KEYS[0]])
                table += '\nTraining-only grouped fit/holdout; no val-based endpoint or threshold selection.\n'
                table += 'Frozen existing val-best E40; this is NOT original-initialization 200-epoch training.\n'
                table += 'No GT at inference. No test-dev. Ranking loss/accuracy is not AP.\n'
                table += 'No automatic authorization of end-to-end training; inspect AP/AP50/AP75 and false positives.\n'
                (out/'REPORT.md').write_text(table, encoding='utf-8')
            write(out/'locations.json', dict(cache=str(cache_base), fit=str(fit_base)))
        write(out/'status.json', dict(status='completed', finished_at=now(), mode=args.mode))
        print('D47 FINISHED '+now()+' REPORT='+str(out), flush=True)
    except BaseException as error:
        write(out/'status.json', dict(status='failed', failed_at=now(), error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
