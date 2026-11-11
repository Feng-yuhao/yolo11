"""Frozen source forward -> semantic mapping -> native NMS -> original COCO coordinates."""
import logging
from pathlib import Path
import time
from common import CLASSES, PREDICT, PROTOCOL, now, require, sha, write
from mapping import inverse_boxes, map_scores, mapped_nms
from coco_eval import convert
from modeling import audit, fingerprint


def predict(args, weights, key, items, out):
    import torch
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils import LOGGER, ops
    import numpy as np
    require(not out.exists(), 'Refuse to overwrite inference output')
    out.mkdir(parents=True)
    lookup = {str(r['path'].resolve()):r for r in items}
    require(len(lookup) == len(items), 'Duplicate target paths')
    audit_ids = {r['id'] for r in items[:8]} | {r['id'] for r in items if r['height'] > r['width']}
    seen, audited, detections, shapes = set(), set(), [], {}
    max_error = 0.

    class Guard(logging.Handler):
        exceeded = False
        def emit(self, record):
            if 'NMS time limit' in record.getMessage():
                self.exceeded = True

    class Predictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs):
            nonlocal max_error
            require(isinstance(preds,(tuple,list)) and img.dtype == torch.float32, 'Unexpected source forward format')
            raw = preds[0]
            shape = 'x'.join(map(str,img.shape)); shapes[shape] = shapes.get(shape,0)+1
            require(not any(m.training for m in self.model.model.modules()), 'Training/BN update mode forbidden')
            needed = any(lookup[str(Path(p).resolve())]['id'] in audit_ids for p in self.batch[0])
            saved = raw.detach().clone() if needed else None
            boxes = mapped_nms(raw)
            if saved is not None:
                require(torch.equal(saved,raw), 'Mapping/NMS mutated source predictions')
                # Independent NumPy indexing checks every raw anchor, not only retained detections.
                array = saved.cpu().numpy()
                expected = np.concatenate((array[:,:4],np.stack((np.maximum(array[:,7],array[:,8]),
                           array[:,12],array[:,9],array[:,13]),axis=1)),axis=1)
                np.testing.assert_array_equal(map_scores(saved).cpu().numpy(),expected)
                oracle = ops.non_max_suppression(torch.from_numpy(expected).to(raw.device).clone(),
                           .001,.7,nc=4,multi_label=False,max_det=500,agnostic=False,max_nms=30000)
            else:
                oracle = None
            if not isinstance(orig_imgs,list):
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
            result = []
            for index,(pred,original,path) in enumerate(zip(boxes,orig_imgs,self.batch[0])):
                record = lookup[str(Path(path).resolve())]
                require(record['id'] not in seen and original.shape[:2] == (record['height'],record['width']),
                        'Repeated image or wrong orientation/dimensions')
                pred[:,:4] = ops.scale_boxes(img.shape[2:],pred[:,:4],original.shape)
                seen.add(record['id'])
                if oracle is not None and record['id'] in audit_ids:
                    expected = inverse_boxes(oracle[index],img.shape[-2:],original.shape[:2])
                    require(expected.shape == pred.shape, 'NMS count oracle mismatch')
                    torch.testing.assert_close(expected[:,4:],pred[:,4:],rtol=0,atol=0)
                    error = float((expected[:,:4]-pred[:,:4]).abs().max()) if len(expected) else 0.
                    max_error = max(error,max_error)
                    require(error <= .01, 'Inverse-letterbox coordinate oracle failed')
                    audited.add(record['id'])
                detections.extend(convert(record,pred.detach().cpu().tolist()))
                result.append(Results(original,path=path,names=dict(enumerate(CLASSES)),boxes=pred))
            return result

    checkpoint_hash = sha(weights)
    model = YOLO(str(weights),task='detect')
    model_info = audit(model.model,key)
    options = dict(PREDICT,device='cuda:0',project=str(out),name='unused',exist_ok=True,batch=4)
    predictor = Predictor(overrides={**model.overrides,**options,'mode':'predict'},_callbacks=model.callbacks)
    predictor.setup_model(model=model.model,verbose=False)
    model.predictor = predictor
    require(predictor.model.names == dict(enumerate(__import__('common').SOURCE_CLASSES)), 'Source head names changed')
    require(not any(m.training for m in predictor.model.model.modules()), 'Model must be eval/frozen')
    require(not any(p.requires_grad for p in predictor.model.model.parameters()), 'Backend must disable parameter gradients')
    frozen_before = fingerprint(predictor.model.model)
    guard = Guard(); LOGGER.addHandler(guard)
    started = time.monotonic()
    try:
        for first in range(0,len(items),4):
            block = items[first:first+4]
            # Each call consumes exactly this batch; no hidden shuffle/drop_last.
            model.predict([str(r['path']) for r in block],stream=False,**dict(options,batch=len(block)))
            require(not guard.exceeded, 'NMS time limit exceeded; partial predictions rejected')
            done = first+len(block); elapsed = time.monotonic()-started
            progress = dict(key=key,done_images=done,total_images=len(items),elapsed_seconds=elapsed,
                            ETA_seconds=elapsed*(len(items)-done)/done,at=now())
            write(out/'progress.json',progress)
            if done % 80 == 0 or done == len(items):
                print(f'{now()} {key} PREDICT {done}/{len(items)} elapsed={elapsed:.1f}s '
                      f'ETA={progress["ETA_seconds"]:.1f}s',flush=True)
        require(seen == {r['id'] for r in items} and audited == audit_ids, 'Incomplete prediction/coordinate coverage')
        frozen_after = fingerprint(predictor.model.model)
        require(frozen_after == frozen_before and sha(weights) == checkpoint_hash,
                'Parameters, BN statistics or source checkpoint changed during target inference')
        write(out/'coco_predictions.json',detections)
        proof = dict(status='completed',key=key,seen_images=len(seen),seen_image_ids=sorted(seen),
                     prediction_count=len(detections),prediction_sha256=sha(out/'coco_predictions.json'),
                     checkpoint_sha256=checkpoint_hash,model=model_info,coordinate_oracle_ids=sorted(audited),
                     coordinate_oracle_max_error_px=max_error,raw_mapping_numpy_oracle=True,
                     inference_state_sha256_before=frozen_before,inference_state_sha256_after=frozen_after,
                     parameters_and_BN_unchanged=True,shapes=shapes,seconds=time.monotonic()-started,protocol=PROTOCOL)
        write(out/'prediction_manifest.json',proof)
        return detections,proof
    finally:
        LOGGER.removeHandler(guard)
