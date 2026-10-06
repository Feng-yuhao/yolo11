"""Real pinned checkpoints on isolated synthetic images. Never a CUDA smoke receipt."""
import argparse
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import audit
import d27_official as official


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','0'],default='cpu')
    a=p.parse_args()
    real_project=a.project
    a.memory_cap_gib=4; a.allow_busy_gpu=True
    audit.setup(a)
    import spatial_task_head
    from ultralytics import YOLO
    specs=[s for s in audit.read(audit.PACKAGE/'models.json')['models'] if s['seed']==1]
    completed_specs=[]
    fixture_root=audit.PACKAGE/'local_fixture'
    fixture_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(fixture_root)) as temporary:
        a.project=Path(temporary)
        records=[]
        rng=np.random.RandomState(46)
        for i,(h,w) in enumerate([(333,517),(640,640),(517,333),(360,640)],1):
            image=rng.randint(0,256,(h,w,3),dtype=np.uint8)
            image[20:90,30:100]=[30,220,40]
            path=a.project/f'{i}.png'; Image.fromarray(image).save(path)
            gt=np.array([[30,20,70,70,0,4],[42,20,70,70,0,4]],dtype=float) if i<4 else np.empty((0,6))
            records.append(official.ImageRecord(i,str(i),path,a.project/f'{i}.txt',w,h,
                           np.empty((0,8)),gt,np.empty((0,4)),np.empty((0,4))))
        for s in specs:
            s={**s,'weights':str((real_project/s['weights']).resolve())}
            baseline=YOLO(s['weights'],task='detect')
            audit.validate_model(baseline,s)
            reference=[]
            values=baseline.predict([str(r.image_path) for r in records],batch=4,device=a.device,stream=False,**audit.OPTIONS)
            for r,v in zip(records,values):
                for x1,y1,x2,y2,score,cls in v.boxes.data.cpu().tolist():
                    reference.append(dict(image_id=r.image_id,category_id=int(cls)+1,bbox=[x1,y1,x2-x1,y2-y1],score=score))
            dest=a.project/'comparison_reports'/audit.REFS[s['key']]
            dest.mkdir(parents=True)
            audit.write(dest/'area_predictions.json',reference)
            del baseline, values
            import gc
            gc.collect()
            if a.device=='0':
                import torch
                torch.cuda.empty_cache()
            out=a.project/s['key']; out.mkdir()
            summary=audit.predict(a,s,records,out)
            assert summary['reproduction']['images']==4 and summary['image_count']==4
            assert len(audit.read(out/'gt_rows.json'))==6
            actual,_=audit.score(records,out/'predictions.json')
            expected,_=audit.score(records,dest/'area_predictions.json')
            assert all(abs(actual[k]-expected[k])<1e-6 for k in audit.KEYS)
            summary['official']=actual
            audit.write(out/'summary.json',summary)
            completed_specs.append(s)
            print('D46 REAL CHECKPOINT SYNTHETIC INTEGRATION PASSED '+s['key'],flush=True)
        audit.report(a.project,completed_specs)
        assert (a.project/'REPORT.md').is_file()
        assert audit.read(a.project/'decision.json')['training_authorized'] is False
    print(a.device+' synthetic integration only; no real val-image or server gate claim.',flush=True)


if __name__=='__main__':
    main()
