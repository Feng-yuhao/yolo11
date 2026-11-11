"""Standard COCO math retained from the completed Songdo evaluation package."""
import contextlib
import io
import math
from common import CLASSES, PROTOCOL, read, require, sha


def records(root):
    from uavdt_data import records as load
    return load(root)


def convert(record, rows):
    result = []
    for x1, y1, x2, y2, score, category in rows:
        require(all(math.isfinite(v) for v in (x1,y1,x2,y2,score,category)), 'Nonfinite prediction')
        require(category == int(category) and 0 <= category < 3 and 0 <= score <= 1, 'Wrong mapped category/score')
        require(-.01 <= x1 <= x2 <= record['width']+.01 and
                -.01 <= y1 <= y2 <= record['height']+.01, 'Wrong original-image coordinates')
        result.append(dict(image_id=record['id'], category_id=int(category)+1,
                           bbox=[x1,y1,max(0.,x2-x1),max(0.,y2-y1)], score=score))
    return result


def coco_score(gt_path, predictions, out, image_ids=None):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO(str(gt_path))
        selected = sorted(truth.getImgIds() if image_ids is None else image_ids)
        selected_set = set(selected)
        require(len(selected_set) == len(selected) and selected_set.issubset(truth.imgs), 'Scoring IDs missing/duplicate in original GT')
        require(set(truth.getCatIds()) == {1,2,3}, 'Wrong COCO category IDs')
        require(all(p['image_id'] in selected_set and p['category_id'] in (1,2,3)
                    for p in predictions), 'Prediction IDs outside scoring population')
        if predictions:
            candidate = truth.loadRes(predictions)
        else:
            candidate = COCO()
            candidate.dataset = dict(images=truth.dataset['images'], categories=truth.dataset['categories'], annotations=[])
            candidate.createIndex()
    ev = COCOeval(truth, candidate, 'bbox')
    ev.params.imgIds, ev.params.catIds, ev.params.maxDets = selected, [1,2,3], [1,10,100]
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        ev.evaluate(); ev.accumulate(); ev.summarize()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'coco_stdout.txt').write_text(captured.getvalue(), encoding='utf-8')
    valid = lambda value: None if value < 0 else float(value)*100
    names = ('AP','AP50','AP75','AP_small','AP_medium','AP_large',
             'AR1','AR10','AR100','AR_small','AR_medium','AR_large')
    result = {name+'_percent': valid(value) for name,value in zip(names,ev.stats)}
    per_class, gt_counts = {}, {}
    from collections import Counter
    counts = Counter(a['category_id'] for a in truth.dataset['annotations'] if a['image_id'] in selected_set)
    for index,name in enumerate(CLASSES):
        values = ev.eval['precision'][:,:,index,0,-1]
        values = values[values > -1]
        per_class[name] = float(np.mean(values))*100 if values.size else None
        gt_counts[name] = counts[index+1]
    result.update(per_class_AP_percent=per_class, GT_counts=gt_counts, images=len(selected),
                  gt_sha256=sha(gt_path), maxDets=[1,10,100], protocol=PROTOCOL)
    return result
