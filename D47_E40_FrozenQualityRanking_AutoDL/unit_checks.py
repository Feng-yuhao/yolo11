"""Independent numerical and semantic checks; never a server smoke receipt."""
import tempfile
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
import ranking as rank


def run_checks(device='cpu'):
    from train_annotation_checks import run_checks as annotation_checks
    annotation_checks()
    d = 'cuda:0' if device == '0' else 'cpu'
    # Continuous xyxy analytic oracle, matching class, background and duplicate GT.
    gt = np.array([[10, 20, 10, 10, 0, 1], [50, 20, 10, 10, 0, 2],
                   [10, 20, 10, 10, 1, 1]], float)
    boxes = np.array([[10,20,20,30],[12,20,22,30],[50,20,60,30],[100,100,105,105]], float)
    q, owner = rank.targets(boxes, np.array([1,1,1,1]), gt)
    np.testing.assert_allclose(q, [1, 2/3, 0, 0], atol=1e-7)
    assert owner.tolist() == [0,0,-1,-1]
    duplicate = np.vstack((gt,gt[0]))
    assert rank.targets(boxes, np.array([1,1,1,1]), duplicate)[1][0] == -1
    np.testing.assert_array_equal(rank.pairs(q,owner), [[0,1]])
    # Pixel-coordinate scale/translation invariant; not xywh or stride units.
    transformed_boxes = boxes*2 + np.array([5,7,5,7])
    transformed_gt = gt.copy(); transformed_gt[:,:2] = gt[:,:2]*2+[5,7]; transformed_gt[:,2:4] *= 2
    np.testing.assert_allclose(rank.targets(transformed_boxes, np.array([1,1,1,1]), transformed_gt)[0], q)
    assert rank.holdout('0000001_00001_d_0000001') == rank.holdout('0000001_00001_d_0000002')

    rng = np.random.RandomState(47)
    qs = rng.rand(1000).astype(np.float32)
    owners = np.where(qs >= .5, np.arange(1000)//10, -1)
    chosen = rank.sample_candidates(qs, owners, rng.rand(1000), rng)
    assert len(chosen) <= 512 and len(chosen) == len(np.unique(chosen))
    assert (qs[chosen]>=.5).sum() <= 256 and (qs[chosen]<.5).sum() <= 256
    assert len(rank.sample_candidates(np.empty(0),np.empty(0,int),np.empty(0),rng)) == 0
    assert rank.pairs(np.empty(0),np.empty(0,int)).shape == (0,2)

    # Feature channels and flattened anchor identity: independently inspect each site.
    maps, raw = [], torch.zeros(14, 32, device=d)
    for level in range(4):
        r = torch.arange(64*8, device=d).reshape(1,64,2,4).float()+1000*level
        c = torch.arange(128*8, device=d).reshape(1,128,2,4).float()+10000*level
        logits = torch.zeros(1,74,2,4, device=d)
        maps.append((r,c,logits))
        raw[:4,level*8:(level+1)*8] = torch.tensor([30,40,8,16],device=d)[:,None]
        raw[4+level,level*8:(level+1)*8] = .2
    x = rank.features(raw, maps, 0)
    for level in range(4):
        for site in range(8):
            torch.testing.assert_close(x[level*8+site,:64],maps[level][0][0,:,site//4,site%4],rtol=0,atol=0)
            torch.testing.assert_close(x[level*8+site,64:192],maps[level][1][0,:,site//4,site%4],rtol=0,atol=0)
    np.testing.assert_allclose(x[0,192:208].cpu().numpy().reshape(4,4)[:,2],1.,atol=1e-6)
    assert x.shape == (32,225) and not x.requires_grad

    # Identity -> nonzero gradients -> real optimization -> serialization.
    mean,std = np.zeros(225,np.float32),np.ones(225,np.float32)
    for variant in ('dfl','full'):
        m = rank.Ranker(variant, mean, std).to(d)
        zero = rank.calibrate(raw,x,m)
        torch.testing.assert_close(zero,raw,rtol=0,atol=0)
        inputs = torch.randn(8,225,device=d)
        scores = torch.tensor([.1,.3,.2,.4,.01,.5,.1,.2],device=d)
        quality = torch.tensor([.95,.55,.8,.5,0,0,.1,.3],device=d)
        pair = torch.tensor([[0,1],[2,3]],device=d)
        before = m.net[-1].weight.detach().clone()
        optimizer = torch.optim.AdamW(m.parameters(),lr=.001)
        for step in range(2):
            optimizer.zero_grad()
            loss,metrics = rank.objective(m(inputs),scores,quality,pair)
            assert torch.isfinite(loss)
            loss.backward(); optimizer.step()
        assert not torch.equal(before,m.net[-1].weight)
        assert m.net[0].weight.grad is not None and m.net[0].weight.grad.abs().sum()>0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            torch.save(m.state_dict(),path)
            restored = rank.Ranker(variant,mean,std).to(d)
            restored.load_state_dict(torch.load(path,map_location=d,weights_only=True))
            torch.testing.assert_close(restored(inputs),m(inputs),rtol=0,atol=0)
            # Independent interpreter, no captured classes/closures from this run.
            cpu_model=rank.Ranker(variant,mean,std)
            cpu_model.load_state_dict({k:v.detach().cpu() for k,v in m.state_dict().items()})
            cpu_input=inputs.detach().cpu()
            torch.save(dict(variant=variant,mean=torch.from_numpy(mean),std=torch.from_numpy(std),
                       state=cpu_model.state_dict(),input=cpu_input,output=cpu_model(cpu_input).detach()),path)
            child=subprocess.run([sys.executable,str(Path(__file__).with_name('reload_check.py')),str(path)],
                                  capture_output=True,text=True,timeout=90)
            assert child.returncode==0, child.stdout+child.stderr
        corrected = rank.calibrate(raw,x,m)
        torch.testing.assert_close(corrected[:4],raw[:4],rtol=0,atol=0)
        assert torch.equal(corrected[4:].argmax(0),raw[4:].argmax(0))

    s = torch.logspace(-7,-.01,100,device=d)
    delta = torch.linspace(-2,2,100,device=d)
    value = rank.corrected_score(s,delta)
    torch.testing.assert_close(value,torch.sigmoid(torch.logit(s)+delta),rtol=3e-6,atol=1e-7)
    floor = torch.tensor(rank.SUPPORT*.999,device=d)
    assert rank.corrected_score(floor,torch.tensor(2.,device=d)) <= .001
    assert torch.equal(rank.corrected_score(s,torch.zeros_like(s)),s)
    # NMS input mutation and scale conversion; changed rank must choose another box.
    from ultralytics.utils import ops
    from run_d47 import independent_nms, excluded
    import d27_official as official
    raw2 = torch.zeros(14,8,device=d)
    raw2[:4,0] = torch.tensor([50,50,10,10],device=d)
    raw2[:4,1] = torch.tensor([51,50,10,10],device=d)
    raw2[4,0] = .5; raw2[4,1] = .4
    copy = raw2.clone()
    kept = independent_nms(raw2,(640,640),(640,640))
    assert len(kept)==1 and kept[0,0]==45
    torch.testing.assert_close(raw2,copy,rtol=0,atol=0)
    raw2[4,0]=.2; raw2[4,1]=.6
    assert independent_nms(raw2,(640,640),(640,640))[0,0]==46
    # Official raster ignore and score-zero IoF (not IoU) independently tested.
    record = official.ImageRecord(1,'fixture',Path('image.png'),Path('ann.txt'),640,640,
        np.empty((0,8)),np.array([[10,10,100,100,1,1]],float),np.array([[200,200,30,30]],float),np.empty((0,4)))
    mask = excluded(record,np.array([[12,12,20,20],[12,12,20,20],[202,202,220,220]],float),np.array([1,2,2]))
    assert mask.tolist()==[True,False,True]
    positive_mask=excluded(record,np.array([[12,12,20,20]],float),np.array([1]),np.array([.8]))
    assert positive_mask.tolist()==[False]
    return dict(device=device, analytic_quality_targets=True, grouped_split=True,
        independent_feature_anchor_order=True, identity=True, two_step_update=True,
        fresh_process_state_reload=True, bounded_score_support=True, original_nms=True,
        background_wrong_class_and_ignore=True, no_GT_in_features=True)


if __name__ == '__main__':
    print(run_checks())
