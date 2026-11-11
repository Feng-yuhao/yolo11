"""DIOR COCO math; licensed original TinyBenchmark v1 ignore/IOD math."""
import contextlib
import copy
import io
import types
from common import HERE,read,require,sha,write,now

def tiny_module():
    path=HERE/'author_eval/tiny_cocoeval_original.py'
    require(sha(path)==read(HERE/'references/author_evaluator.json')['original_sha256'],'Author evaluator changed')
    source=path.read_text(encoding='utf-8')
    # Compatibility only: np.float was the builtin float, and linspace now
    # requires an integer sample count. No threshold/matching/math changes.
    source=source.replace('dtype=np.float)','dtype=float)')
    for text in ('np.round((1.00 - .0) / .01) + 1','np.round((0.95 - .5) / .05) + 1'):
        source=source.replace(text,'int('+text[:-4]+') + 1')
    module=types.ModuleType('bound_author_tiny_v1');exec(compile(source,str(path),'exec'),module.__dict__)
    return module

def evaluate(gt_path,predictions,target,out,image_ids=None):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        truth=COCO(str(gt_path))
        selected=sorted(truth.getImgIds() if image_ids is None else image_ids);chosen=set(selected)
        require(len(chosen)==len(selected) and chosen.issubset(truth.imgs),'Invalid evaluation population')
        require(set(truth.getCatIds())=={1},'Wrong single mapped class')
        require(all(p['image_id'] in chosen and p['category_id']==1 for p in predictions),'Prediction outside evaluation population')
    module=tiny_module() if target=='tiny' else None
    def run(native=False):
        with contextlib.redirect_stdout(io.StringIO()):
            gt=copy.deepcopy(truth)
            if predictions:dt=gt.loadRes(copy.deepcopy(predictions))
            else:
                dt=COCO();dt.dataset=dict(images=gt.dataset['images'],categories=gt.dataset['categories'],annotations=[]);dt.createIndex()
        cls=module.COCOeval if module else COCOeval
        class Progress(cls):
            calls=0
            def evaluateImg(self,*args,**kwargs):
                result=super().evaluateImg(*args,**kwargs);self.calls+=1
                if self.calls%500==0:
                    write(out/'scoring_progress.json',dict(stage='Tiny native AP' if native else 'COCO-style AP',
                          done_image_area_pairs=self.calls,total_image_area_pairs=len(selected)*len(self.params.areaRng),at=now()))
                return result
        if module:
            module.Params.EVAL_STRANDARD='tiny'
            ev=Progress(gt,dt,'bbox',ignore_uncertain=True,use_iod_for_ignore=True,use_ignore_attr=True)
        else:ev=Progress(gt,dt,'bbox')
        ev.params.imgIds=selected;ev.params.catIds=[1]
        if not native:
            ev.params.iouThrs=np.linspace(.5,.95,10);ev.params.maxDets=[1,10,100]
            ev.params.areaRng=[[0,1e10],[0,32**2],[32**2,96**2],[96**2,1e10]]
            ev.params.areaRngLbl=['all','small','medium','large']
        with contextlib.redirect_stdout(io.StringIO()) as capture:
            ev.evaluate();ev.accumulate()
            if native:ev.summarize()
        (out/('tiny_native_stdout.txt' if native else 'coco_style_stdout.txt')).write_text(capture.getvalue(),encoding='utf-8')
        def mean(iou=None,area=0):
            values=ev.eval['precision'][:,:,:,area,-1]
            if iou is not None:values=values[np.isclose(ev.params.iouThrs,iou)]
            values=values[values>-1];return float(np.mean(values))*100 if values.size else None
        if native:
            return {f'AP{int(t*100)}_{name}_percent':mean(t,i) for t in (.25,.5,.75) for i,name in enumerate(ev.params.areaRngLbl)}
        return dict(AP_percent=mean(),AP50_percent=mean(.5),AP75_percent=mean(.75),
                    AP_small_percent=mean(area=1),AP_medium_percent=mean(area=2),AP_large_percent=mean(area=3))
    metrics=run()
    if target=='tiny':metrics['native_tiny']=run(native=True)
    annotations=[a for a in truth.dataset['annotations'] if a['image_id'] in chosen]
    metrics.update(images=len(selected),positive_boxes=sum(not a.get('ignore',False) for a in annotations),
                   ignored_entries=sum(bool(a.get('ignore',False)) for a in annotations),gt_sha256=sha(gt_path),
                   evaluator='author TinyBenchmark v1 ignore-aware COCO-style AP' if module else 'standard pycocotools COCO AP',
                   maxDets=[1,10,100],native_maxDets=[200] if module else None)
    return metrics
