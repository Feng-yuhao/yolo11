"""Real E40 checkpoint / synthetic image integration. NOT a real-data gate."""
import argparse
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import run_d47 as runner
import d27_official as official


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu','0'], default='cpu')
    parser.add_argument('--seed', choices=[1,2], type=int, default=1)
    args = parser.parse_args(); real_project = args.project
    args.memory_cap_gib = 4.; args.allow_busy_gpu = True
    runner.setup(args)
    import torch
    import spatial_task_head
    from ultralytics import YOLO
    import ranking
    spec = next(s for s in runner.read(runner.PACKAGE/'models.json')['models'] if s['key']==f'e40_seed{args.seed}')
    spec = {**spec,'weights':str((real_project/spec['weights']).resolve())}
    fixture = runner.PACKAGE/'local_fixture'; fixture.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(fixture)) as directory:
        args.project = Path(directory)
        paths = []
        rng = np.random.RandomState(47)
        for i,(h,w) in enumerate([(333,517),(640,640),(517,333),(360,640)],1):
            group = i if i<4 else next(j for j in range(20,1000) if ranking.holdout(f'{j:07d}_00000_d_000001'))
            path = args.project/f'{group:07d}_00000_d_000001.png'
            pixels = rng.randint(0,256,(h,w,3),dtype=np.uint8)
            pixels[20:90,30:100]=[30,220,40]
            Image.fromarray(pixels).save(path); paths.append(path)
        baseline = YOLO(spec['weights'],task='detect')
        values = baseline.predict([str(p) for p in paths],batch=4,device=args.device,stream=False,**runner.OPTIONS)
        reference,records = [],[]
        for i,(path,value) in enumerate(zip(paths,values),1):
            reference.extend(runner.prediction_rows(value.boxes.data,i))
            # Construct synthetic GT ONLY for unit/integration coverage of target paths.
            # These are not natural data, and cannot support an AP improvement claim.
            gt = []
            for box in value.boxes.data.cpu().tolist()[:4]:
                x1,y1,x2,y2,score,cls = box
                if x2>x1 and y2>y1:
                    gt.append([x1,y1,x2-x1,y2-y1,0,int(cls)+1])
            if not gt:
                gt = [[30,20,70,70,0,4]]
            if i==4:
                gt=[]
            h,w = value.orig_shape
            records.append(official.ImageRecord(i,path.stem,path,args.project/f'{i}.txt',w,h,
                np.empty((0,8)),np.array(gt,dtype=float).reshape(-1,6),np.empty((0,4)),np.empty((0,4))))
        ref = args.project/'comparison_reports'/runner.REFS[args.seed]
        runner.write(ref/'area_predictions.json',reference)
        del values,baseline
        import gc
        gc.collect()
        if args.device=='0': torch.cuda.empty_cache()
        # Exact identity from zero output, all classes and box coordinates checked.
        mean,std = np.zeros(225,np.float32),np.ones(225,np.float32)
        models = {v:ranking.Ranker(v,mean,std).to('cuda:0' if args.device=='0' else 'cpu').eval()
                  for v in ('dfl','full')}
        identity = runner.evaluate(args,spec,records,models,args.project/'identity',partial=True)
        assert all(identity['deltas'][v][k]==0 for v in models for k in runner.KEYS)
        del models
        cache = runner.cache_candidates(args,spec,records,args.project/'cache',dict(fixture=True),partial=True)
        assert cache['train_only'] and len(cache['shards'])==4
        models = runner.fit_rankers(args,args.project/'cache',cache,args.project/'fit',smoke=True)
        restored={}
        for v,m in models.items():
            saved=torch.load(args.project/'fit'/(v+'.pt'),map_location='cpu',weights_only=False)
            other=ranking.Ranker(v,saved['mean'],saved['std']).to('cuda:0' if args.device=='0' else 'cpu')
            other.load_state_dict(saved['state'],strict=True); other.eval().requires_grad_(False)
            x=torch.randn(20,225,device=next(m.parameters()).device)
            torch.testing.assert_close(m(x),other(x),rtol=0,atol=0)
            restored[v]=other
        actual = runner.evaluate(args,spec,records,restored,args.project/'adjusted',partial=True)
        assert actual['baseline_reproduced'] and actual['box_coordinates_unchanged']
        assert runner.read(args.project/'fit'/'fit.json')['smoke']
        try:
            runner.load_rankers(args,args.project/'fit',spec,cache)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Smoke fit was incorrectly accepted as full fit')
        # Exercise the full fixed endpoint, grouped holdout, normalizer and eval
        # provenance too (tiny synthetic cache, never a real training result).
        runner.fit_rankers(args,args.project/'cache',cache,args.project/'fullfit',smoke=False)
        full_models=runner.load_rankers(args,args.project/'fullfit',spec,cache)
        assert runner.read(args.project/'fullfit'/'fit.json')['epochs']==10
        assert set(full_models)=={'dfl','full'}
        sums=runner.read(args.project/'fullfit'/'history.json')
        assert len(sums)==10 and sums[-1]['epoch']==10
        # Replay completed cache instead of silently regenerating it.
        again=runner.cache_candidates(args,spec,records,args.project/'cache',dict(fixture=True),partial=True)
        assert again['binding']==cache['binding']
    print('D47 '+args.device+' REAL E40 SYNTHETIC INTEGRATION PASSED; not a real-data receipt',flush=True)


if __name__=='__main__':
    main()
