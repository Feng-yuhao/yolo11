"""Validation batch32 and EMA FP16 stress, retaining the training optimizer/model."""
import copy
import json
from pathlib import Path
import time
import torch


def tensor_summary(value):
    value = value.detach()
    finite = torch.isfinite(value)
    result = dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                  finite=bool(finite.all()), nan=int(torch.isnan(value).sum()),
                  inf=int(torch.isinf(value).sum()))
    if value.numel() and result['finite']:
        result.update(min=float(value.min()), max=float(value.max()))
    return result


def store_report(report_dir, name, payload):
    if report_dir is not None:
        path = Path(report_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def fail(stage, value, report_dir=None, context=None):
    payload = dict(stage=stage, tensor=tensor_summary(value), context=context or {})
    store_report(report_dir, 'VALIDATION_FAILURE.json', payload)
    print('VALIDATION FAILED ' + json.dumps(payload, allow_nan=False), flush=True)
    raise RuntimeError('Nonfinite validation ' + stage + '; see VALIDATION_FAILURE.json')


def require_finite_predictions(pred, report_dir=None, context=None):
    # Check raw tower logits too: finite decoded scores alone do not prove that
    # the criterion received finite inputs. Do not clamp/nan_to_num any output.
    if not torch.isfinite(pred[0]).all():
        fail('decoded_boxes_or_scores', pred[0], report_dir, context)
    for level, raw in enumerate(pred[1]):
        if not torch.isfinite(raw).all():
            fail('raw_P' + str(level + 2) + '_logits', raw, report_dir, context)


def validation_loss(model, batch, pred, report_dir=None, context=None, audit_native=False):
    # The model still performs a pure FP16 validation forward. Run the original
    # criterion in AMP, as in training: BCE/CE get FP32 accumulation. Calling
    # BCE(...).sum() on millions of FP16 elements outside autocast can overflow
    # even when EVERY logit is finite. No loss formula/weights are replaced.
    require_finite_predictions(pred, report_dir, context)
    native = None
    if audit_native:
        with torch.autocast(device_type=batch['img'].device.type, enabled=False):
            native_total, native_items = model.loss(batch, pred)
        native = dict(total=tensor_summary(native_total), items=tensor_summary(native_items),
                      item_values=[float(v) if torch.isfinite(v) else str(float(v))
                                   for v in native_items])
    with torch.autocast(device_type=batch['img'].device.type, dtype=torch.float16):
        total, items = model.loss(batch, pred)
    if not torch.isfinite(total).all():
        fail('AMP_total_loss', total, report_dir, context)
    if not torch.isfinite(items).all():
        fail('AMP_box_cls_dfl_loss', items, report_dir, context)
    if audit_native:
        audit = dict(native_FP16=native, AMP=dict(total=tensor_summary(total),
                     items=tensor_summary(items), item_values=items.tolist()),
                     context=context or {}, raw_and_decoded_finite=True)
        store_report(report_dir, 'VALIDATION_PRECISION.json', audit)
        print('VALIDATION LOSS PRECISION native_FP16=' + str(native['item_values']) +
              ' AMP=' + str(items.tolist()) + ' raw/decoded=finite', flush=True)
    return total, items


def validation_stress(model, args, data, params, report_dir=None):
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset, build_dataloader
    from ultralytics.utils import ops
    from p2_tal_chunked import chunked_assignment
    from pair_memory import check_headroom
    # Same lifetime as Trainer: original FP32 model and optimizer stay on GPU
    # while a disposable copy has EMA-like lifetime/dtype (not EMA averaging).
    # Prediction stays FP16; only the loss computation uses autocast. No train
    # BN mutation. This does not edit the pinned formal validator or criterion.
    ema = copy.deepcopy(model).eval().half()
    ema.criterion = None
    opts = get_cfg(overrides={**params, 'batch':32, 'rect':True})
    from parallel_safety import dataset_build_lock
    with dataset_build_lock(args):
        dataset = build_yolo_dataset(opts, data['val'], 32, data, mode='val', rect=True, stride=32)
    if len(dataset) != 548:
        raise RuntimeError('Validation split changed')
    loader = build_dataloader(dataset, 32, args.workers, shuffle=False, rank=-1)
    count, shapes = 0, set()
    with torch.no_grad(), chunked_assignment():
        # Worst-case square image test, independent of the current rectangular split.
        square = torch.zeros(32, 3, 640, 640, device=args.device, dtype=torch.float16)
        prediction = ema(square)
        require_finite_predictions(prediction, report_dir, dict(test='square_batch32'))
        del square, prediction
        for i, batch in enumerate(loader):
            check_headroom(torch.cuda, args.device)
            batch['img'] = batch['img'].to(args.device).half() / 255
            context = dict(batch_index=i + 1, image_shape=list(batch['img'].shape),
                           GT=len(batch['cls']), image_files=list(batch.get('im_file', ()))[:4])
            if i == 0:
                print('VALIDATION ENTER ' + json.dumps(context), flush=True)
            pred = ema(batch['img'])
            _, items = validation_loss(ema, batch, pred, report_dir, context, audit_native=i == 0)
            detections = ops.non_max_suppression(pred[0].clone(), conf_thres=.001, iou_thres=.7, max_det=500)
            for detection in detections:
                if not torch.isfinite(detection).all():
                    fail('NMS', detection, report_dir, context)
            count += len(batch['img']); shapes.add(tuple(batch['img'].shape))
            print(f'VALIDATION SMOKE {i+1}/{len(loader)} batch={len(batch["img"])} '
                  f'reserved={torch.cuda.memory_reserved()/2**30:.2f} GiB', flush=True)
        if count != 548:
            raise RuntimeError('Incomplete validation smoke')
    del ema, loader
    return dict(parallel_pair=True,validation_batch=32,validation_images=count,EMA_half=True,validation_loss_and_NMS=True,
                square_batch32=True,raw_logits_and_decoded_finite=True,
                loss_precision='autocast_float16',shapes=[list(v) for v in sorted(shapes)])
