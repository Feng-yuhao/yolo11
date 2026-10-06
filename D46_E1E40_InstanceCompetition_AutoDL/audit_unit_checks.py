"""Independent fixtures for geometry/matching/ID-carrying original NMS."""
import numpy as np
import competition as c


def brute(adj):
    def visit(g, used):
        if g == len(adj):
            return 0
        return max([visit(g+1, used)] + [1+visit(g+1, used | {x}) for x in adj[g] if x not in used])
    return visit(0, set())


def unit_tests(device='cpu'):
    import torch
    from ultralytics.utils import ops
    from audit import trace_nms
    # Exhaust every bipartite graph of <=3x3, plus explicit augmenting-chain fixture.
    checked = 0
    for mask in range(512):
        adj = [[j for j in range(3) if mask & (1 << (i*3+j))] for i in range(3)]
        got = c.maximum_matching(adj)
        assert len(got) == brute(adj)
        assert len(set(got.values())) == len(got)
        assert all(j in adj[i] for i,j in got.items())
        checked += 1
    assert c.maximum_matching([]) == {}
    assert len(c.maximum_matching([[1, 2], [1]])) == 2
    assert c.iou(np.zeros((0,4)), np.zeros((0,4))).shape == (0,0)
    np.testing.assert_allclose(c.iou([[0,0,10,10]], [[0,0,10,10], [20,20,30,30]]), [[1,0]])
    gt = np.array([[0,0,10,10], [0,0,10,10]], dtype=float)
    b = gt[:1]
    r, a, aa = c.raw_geometry(gt, np.array([1,1]), b, np.array([1]), np.array([.8]), 1, np.array([4]))
    s = c.stage_match(r,a,aa,dict(raw=None,final=[0]))
    assert s['raw']['independent'] == 2 and s['raw']['maximum_one_to_one'] == 1
    r,a,aa = c.raw_geometry(gt, np.array([2,2]), b, np.array([1]), np.array([.8]), 1, np.array([4]))
    assert not any(a) and all(aa)
    # Known +x shift toward nearest same-class neighbor; width correct.
    gt = np.array([[0,0,10,10], [12,0,22,10]], dtype=float)
    boxes = np.array([[2,0,12,10], [12,0,22,10]], dtype=float)
    r,a,aa = c.raw_geometry(gt,np.array([1,1]),boxes,np.array([1,1]),np.array([.8,.7]),1,np.array([4,4]))
    assert r[0]['center_dx_norm'] == .2 and r[0]['same_class_neighbor_cosine'] == 1
    assert r[0]['log_width_error'] == 0 and r[0]['neighborward_projection'] > 0
    toward_boxes=np.array([[2,0,12,10],[10,0,20,10]],dtype=float)
    toward,_,_=c.raw_geometry(gt,np.array([1,1]),toward_boxes,np.array([1,1]),np.array([.8,.7]),1,np.array([4,4]))
    for row in toward:
        row['image_id']=1
    control=c.direction_control(toward)
    assert control['exchangeable_near_misses']==2 and control['observed_mean_cosine']==1
    assert abs(control['permuted_mean_cosine']) < .5
    assert c.direction_control(toward[:1])['exchangeable_near_misses']==0
    dev = 'cuda:0' if device == '0' else 'cpu'
    if device == '0':
        assert torch.cuda.is_available(), 'CUDA unavailable'
    generator = torch.Generator().manual_seed(46)
    for n in (0, 3, 17, 1001):
        raw = torch.rand(14,n,generator=generator, dtype=torch.float32)
        raw[:2] *= 100; raw[2:4] *= 20
        # Distinct scores; add overlapping same/different-class fixture.
        raw[4:] *= .8
        if n >= 3:
            raw[:, :3] = 0
            raw[:4, :3] = torch.tensor([[10,10,10],[10,10,10],[8,8,8],[8,8,8]])
            raw[4,0] = .99; raw[4,1] = .95; raw[5,2] = .97
        raw = raw.to(dev)
        before = raw.clone()
        traced = trace_nms(raw)
        ordinary = ops.non_max_suppression(raw[None].clone(), .001,.7,max_det=500,max_time_img=60)[0]
        torch.testing.assert_close(traced['kept'][:500,:6],ordinary,rtol=0,atol=0)
        torch.testing.assert_close(raw,before,rtol=0,atol=0)
        assert len(traced['kept_ids']) == len(set(traced['kept_ids']))
        witnesses = c.nms_witnesses(traced['network_boxes'],traced['labels'],traced['scores'],traced['pre_ids'],traced['kept_ids'])
        if n >= 3:
            assert witnesses[1]['witness'] == 0 and not witnesses[1]['ambiguous']
            assert 2 in traced['kept_ids']
    # A score-tie must never be claimed as a uniquely reconstructed suppressor.
    w = c.nms_witnesses(np.array([[0,0,10,10],[0,0,10,10]],dtype=np.float32),np.array([1,1]),np.array([.8,.8]),[0,1],[0])
    assert w[1]['ambiguous']
    assert c.nms_witnesses(np.zeros((0,4),dtype=np.float32),np.zeros(0,dtype=int),np.zeros(0),[],[]) == {}
    # Distinct-GT ownership witness and a worse same-instance witness; alternatives
    # must not be miscounted as absent detections. This is geometry, not AP.
    gt=np.array([[0,0,10,10],[1,0,11,10]],dtype=float)
    b=gt.copy()
    witness={0:dict(witness=1,ambiguous=False,reason='strict_overlap',overlap=.81,score_gap=.1)}
    ev=c.witness_events(gt,np.array([1,1]),b,np.array([.7,.8]),np.array([1,1]),witness)
    assert ev[0]['kind']=='cross_instance' and ev[0]['lower_quality_on_lost_gt']
    assert ev[0]['multi_gt_good']==2  # overlapping GT ownership is visible, not hidden
    tie={0:{**witness[0],'ambiguous':True}}
    assert c.witness_events(gt,np.array([1,1]),b,np.array([.7,.8]),np.array([1,1]),tie)[0]['kind']=='ambiguous'
    assert c.witness_events(np.empty((0,4)),np.empty(0,dtype=int),b,np.array([.7,.8]),np.array([1,1]),witness)==[]
    # Budget: identity survives masks, and top500 equals the ordinary prefix.
    n=30005
    raw=torch.zeros(14,n,device=dev)
    raw[0]=torch.arange(n,device=dev)%170*3
    raw[1]=(torch.arange(n,device=dev)//170)*3
    raw[2:4]=1
    raw[4]=torch.linspace(.9,.1,n,device=dev)
    t=trace_nms(raw)
    assert len(t['pre_ids']) == 30000 and len(t['kept_ids']) == 30000
    baseline=ops.non_max_suppression(raw[None].clone(),.001,.7,max_det=500,max_time_img=60)[0]
    torch.testing.assert_close(t['kept'][:500,:6],baseline,rtol=0,atol=0)
    return dict(exhaustive_maximum_matching_graphs=checked, empty_and_wrong_class=True,
                independent_vs_unique=True, signed_center_oracle=True,
                witness_gt_ownership_and_ties=True,
                descriptive_direction_permutation_and_singletons=True,
                original_nms_identity_and_budget=True, tie_ambiguity=True,
                raw_input_unchanged=True, device=dev)
