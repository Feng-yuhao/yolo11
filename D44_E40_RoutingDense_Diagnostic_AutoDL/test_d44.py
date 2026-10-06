"""CPU tests of statistics coordinates, metadata, and report behavior (no receipts)."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from torch import nn
from routing_stats import RoutingStats, js
from diagnose_d44 import summary, KEYS
from diagnose_worker import package_signature, SPECS


class UnitTests(unittest.TestCase):
    def test_manifest_and_model_namespaces(self):
        hashes=package_signature()
        self.assertEqual(hashes['d27_official.py'],'a9e0f50885f81e72ed85baf57a880800c6c51a51527a646b5c60184a1b76bf60')
        self.assertEqual(SPECS['e40s2'][0],'e40')
        self.assertEqual(SPECS['e40s2'][3],2)
        self.assertNotEqual(hashes['models/e40/spatial_task_head.py'],hashes['models/e42/spatial_task_head.py'])

    def test_JS_depth_axis(self):
        p=torch.tensor([[1.,.25],[0.,.25],[0.,.25],[0.,.25]])
        q=torch.tensor([[0.,.25],[1.,.25],[0.,.25],[0.,.25]])
        got=js(p,q)
        torch.testing.assert_close(got,torch.tensor([np.log(2),0.],dtype=torch.float32),rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(js(p,p),torch.zeros(2),rtol=0,atol=0)

    def test_GT_projection_asymmetric_rectangular_padding(self):
        # Original 517x333 -> actual tensor 640x416; P2 map 160x104.
        # Independent expected pixel: gain=640/517, vertical rounded pad=2.
        rec=SimpleNamespace(image_id=1,width=517,height=333,
                            eval_gt=np.array([[100,80,20,16,0,1],[0,0,10,10,1,2]],dtype=np.float64))
        gain=640/517; xx=int((110*gain)/4); yy=int((88*gain+2)/4)
        x=torch.ones(1,128,104,160)
        weights=torch.full((1,2,4,104,160),.25)
        weights[0,0,:,yy,xx]=torch.tensor([1.,0.,0.,0.])
        block=SimpleNamespace(out_cls=nn.Identity(),out_reg=nn.Identity())
        stats=RoutingStats('e40'); stats.set_context([rec],(416,640))
        with torch.inference_mode(): stats(0,block,x,(x*2,x*3,weights,[x]*4))
        self.assertEqual(len(stats.rows),1)  # score-zero ignored target not sampled
        row=stats.rows[0]
        self.assertEqual(row['cls_depth_weights'],[1.,0.,0.,0.])
        self.assertAlmostEqual(row['cls_ratio'],2.,places=6)
        self.assertAlmostEqual(row['reg_ratio'],3.,places=6)
        self.assertAlmostEqual(row['size'],np.sqrt(320)*640/517,places=6)

    def test_report_no_automatic_training(self):
        reports={}
        for model,modes in (('e40',('normal','uniform_depth','shared_task')),
                            ('e40s2',('normal','uniform_depth','shared_task')),
                            ('e42',('normal','channel_off_p2','channel_off_p3')),
                            ('e43',('normal','dense_bypass_p2','dense_bypass_p3'))):
            values={}
            for index,mode in enumerate(modes):
                delta={k:0. if index==0 else (.1 if index==1 else -.1) for k in KEYS}
                values[mode]=dict(official={k:26.+delta[k] for k in KEYS},
                                  delta_vs_same_checkpoint_normal_pp=delta)
            reports[model]=dict(scores=values)
        with TemporaryDirectory(prefix='d44_unit_') as directory:
            out=Path(directory); summary(out,reports)
            decision=json.loads((out/'decision.json').read_text(encoding='utf-8'))
            self.assertTrue(decision['directional_hints']['E45']['observed_P2_bypass_rescue_and_P3_bypass_not_rescue'])
            self.assertIn('no automatic training',decision['next_action'])
            self.assertIn('off', (out/'REPORT.md').read_text(encoding='utf-8').lower())


if __name__=='__main__': unittest.main(verbosity=2)
