"""Tiny independent engineering fixtures; target AP never used to choose settings."""
from pathlib import Path
import tempfile
import unittest
import io
from common import HERE, CLASSES, KEYS, boot, parser, require, write


class ReleaseTests(unittest.TestCase):
    temp_root=HERE
    def fixture(self,n=7):
        import torch
        r=torch.zeros(1,14,n,dtype=torch.float32)
        r[:,0]=torch.arange(n)*30+20; r[:,1]=20; r[:,2:4]=10
        return r

    def test_mapping_and_immutability(self):
        import torch
        from mapping import map_scores,mapped_nms
        r=self.fixture(); r[:,7]=.2; r[:,8]=.8; r[:,9]=.4; r[:,12]=.6; r[:,13]=.99
        before=r.clone(); a=map_scores(r)
        torch.testing.assert_close(a[:,4:,0],torch.tensor([[.8,.4,.6]]),rtol=0,atol=0)
        a.zero_(); mapped_nms(r); self.assertTrue(torch.equal(before,r))

    def test_map_before_bestclass_and_car_van_dedup(self):
        from mapping import mapped_nms
        r=self.fixture(); r[:,:,1]=r[:,:,0]; r[:,4]=.999
        r[0,7,0]=.9; r[0,8,1]=.8
        out=mapped_nms(r)[0]; self.assertEqual(len(out),1); self.assertEqual(int(out[0,5]),0)

    def test_threeclass_order_and_overlap(self):
        from mapping import mapped_nms
        r=self.fixture(); r[:,:,1]=r[:,:,0]; r[:,:,2]=r[:,:,0]
        r[0,7,0]=.9; r[0,9,1]=.8; r[0,12,2]=.7
        self.assertEqual(mapped_nms(r)[0][:,5].tolist(),[0.,1.,2.])

    def test_unmapped_before_top500(self):
        from mapping import mapped_nms
        r=self.fixture(620); r[0,4,:610]=.99; r[0,7,619]=.2
        self.assertEqual(len(mapped_nms(r)[0]),1)

    def test_reject_invalid(self):
        from mapping import map_scores
        r=self.fixture(); r[0,7,0]=float('nan')
        with self.assertRaises(RuntimeError): map_scores(r)

    def test_coordinate_oracle(self):
        import torch
        from mapping import inverse_boxes
        # Native stride padding at 1024x540 -> resized 640x338 + 7/7 top/bottom.
        row=torch.tensor([[62.5,132.,125.,194.5,.9,0.]])
        expected=torch.tensor([[100.,200.,200.,300.,.9,0.]])
        torch.testing.assert_close(inverse_boxes(row,(352,640),(540,1024)),expected,rtol=0,atol=.001)

    def test_ignore_strict_edges_and_partial_overlap(self):
        from uavdt_data import filter_ignored
        make=lambda box:dict(image_id=1,bbox=box,category_id=1,score=.9)
        inside=make([1,1,8,8]); same=make([0,0,10,10]); partial=make([9,1,3,3]); edge=make([0,1,8,8])
        kept,removed=filter_ignored([inside,same,partial,edge],{'1':[[0,0,10,10]]})
        self.assertEqual(removed,[inside]); self.assertEqual(kept,[same,partial,edge])

    def test_native_perfect_duplicate_wrong_geometry_and_class_agnostic(self):
        from native_ap70 import frame_matches,integral_ap
        p=lambda box,score=.9,cls=1:dict(bbox=box,score=score,category_id=cls)
        gt=[[0,0,10,10]]
        flags=frame_matches(gt,[p([0,0,10,10],.9,3),p([0,0,10,10],.8)])
        self.assertEqual(flags,[(.9,1),(.8,0)])
        self.assertAlmostEqual(integral_ap(flags,1),100.)
        self.assertEqual(integral_ap(frame_matches(gt,[p([50,50,10,10])]),1),0.)
        self.assertEqual(integral_ap([],1),0.)

    def test_native_boundary_last_tie(self):
        from native_ap70 import frame_matches
        ps=[dict(bbox=[0,0,10,7],score=.9),dict(bbox=[0,0,10,10],score=.8)]
        self.assertEqual([v[1] for v in frame_matches([[0,0,10,10]],ps)],[1,0])
        self.assertEqual([v[1] for v in frame_matches([[0,0,10,10]]*2,ps)],[1,1])

    def test_native_independent_scalar_oracle(self):
        import random
        from native_ap70 import frame_matches
        rng=random.Random(771)
        for _ in range(80):
            gt=[[rng.randrange(20),rng.randrange(20),10,10] for _ in range(5)]
            ps=[dict(bbox=[rng.randrange(20),rng.randrange(20),10,10],score=rng.random()) for _ in range(9)]
            used=set(); expected=[]
            for p in sorted(ps,key=lambda v:-v['score']):
                x,y,w,h=p['bbox']; best=.7; selected=None
                for i,(gx,gy,gw,gh) in enumerate(gt):
                    if i in used: continue
                    inter=max(0,min(x+w,gx+gw)-max(x,gx))*max(0,min(y+h,gy+gh)-max(y,gy))
                    iou=inter/(w*h+gw*gh-inter)
                    if iou>=best: best=iou; selected=i
                if selected is not None: used.add(selected)
                expected.append((p['score'],int(selected is not None)))
            self.assertEqual(frame_matches(gt,ps),expected)

    def test_coco_oracles(self):
        from coco_eval import coco_score
        with tempfile.TemporaryDirectory(dir=self.temp_root) as tmp:
            root=Path(tmp)
            gt=dict(info={},images=[dict(id=i,width=100,height=100) for i in (1,2,3)],
                    categories=[dict(id=i+1,name=n) for i,n in enumerate(CLASSES)],
                    annotations=[dict(id=i,image_id=i,category_id=i,bbox=[10,10,20,20],area=400,iscrowd=0) for i in (1,2,3)])
            write(root/'gt.json',gt)
            ps=[dict(image_id=i,category_id=i,bbox=[10,10,20,20],score=.9) for i in (1,2,3)]
            self.assertAlmostEqual(coco_score(root/'gt.json',ps,root/'perfect')['AP_percent'],100.)
            self.assertEqual(coco_score(root/'gt.json',[dict(p,category_id=p['category_id']%3+1) for p in ps],root/'wrong')['AP_percent'],0.)
            self.assertEqual(coco_score(root/'gt.json',[],root/'empty')['AP_percent'],0.)

    def test_busy_gpu_and_bn_fingerprint(self):
        import torch
        from unittest.mock import patch
        from admission import check_idle
        from modeling import fingerprint
        with patch('admission.capture',side_effect=['32760,20000\n','1234\n']):
            with self.assertRaises(RuntimeError): check_idle()
        m=torch.nn.BatchNorm2d(4).eval(); before=fingerprint(m)
        with torch.no_grad(): m.running_mean.add_(1)
        self.assertNotEqual(fingerprint(m),before)

    def test_summary_keeps_negative(self):
        from controller import summarize
        with tempfile.TemporaryDirectory(dir=self.temp_root) as tmp:
            root=Path(tmp)
            for i,k in enumerate(KEYS):
                v=10-i
                fields={f+'_percent':v for f in ('AP','AP50','AP75','AP_small','AP_medium','AP_large')}
                write(root/'evaluation'/k/'coco_metrics.json',dict(fields,status='completed',images=16592))
                write(root/'evaluation'/k/'native_ap70.json',dict(AP70_overall_percent=v))
            self.assertEqual(summarize(root)['paired_gain_pp']['AP_percent'],-1)

    def test_disk_admission_no_cleanup(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from controller import disk_admission
        with patch('controller.shutil.disk_usage',return_value=SimpleNamespace(free=8*1024**3)):
            with self.assertRaisesRegex(RuntimeError,'Need >=9GiB'): disk_admission(HERE)
        with patch('controller.shutil.disk_usage',return_value=SimpleNamespace(free=9*1024**3)):
            self.assertEqual(disk_admission(HERE)['free_bytes'],9*1024**3)


def run_tests(out):
    out.mkdir(parents=True,exist_ok=True); ReleaseTests.temp_root=out
    with io.StringIO() as captured:
        result=unittest.TextTestRunner(stream=captured,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReleaseTests))
        text=captured.getvalue()
    print(text,flush=True)
    proof=dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,errors=len(result.errors),failures=len(result.failures))
    write(out/'unit_checks.json',proof); (out/'unit_checks.txt').write_text(text,encoding='utf-8')
    require(result.wasSuccessful(),'Engineering fixture tests failed'); return proof


if __name__=='__main__':
    p=parser(); p.add_argument('--report',type=Path,required=True); args=p.parse_args()
    write(args.report/'check_environment.json',boot(args,args.report/'settings/check'))
    run_tests(args.report)
