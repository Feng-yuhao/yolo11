"""Read-only observer and synthetic-fixture regression tests; no training/shutdown."""
import sys, unittest, types
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import td_coverage_check as coverage

class CoverageTests(unittest.TestCase):
    def test_fixture_shapes_sizes_and_independent_rng(self):
        torch.manual_seed(2);state=torch.get_rng_state().clone();bank=coverage.fixtures('cpu')
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        torch.manual_seed(3);other=coverage.fixtures('cpu')
        self.assertTrue(torch.equal(bank[0]['img'],other[0]['img']))
        self.assertEqual(len(bank),4)
        for batch,size in zip(bank,[.025,.10,.35,.90]):
            self.assertEqual(list(batch['img'].shape),[1,3,640,640])
            self.assertAlmostEqual(float(batch['bboxes'][0,2]),size,places=6)
            self.assertEqual(batch['cls'].tolist(),[[3.]])
        if torch.cuda.is_available():
            gpu=coverage.fixtures('cuda:0')
            self.assertTrue(torch.equal(bank[0]['img'],gpu[0]['img'].cpu()))

    def test_install_always_restores_original(self):
        original=object();module=types.SimpleNamespace(synthetic_updates=original)
        with coverage.install(module):self.assertIs(module.synthetic_updates,coverage.synthetic_updates)
        self.assertIs(module.synthetic_updates,original)
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with coverage.install(module):raise RuntimeError('intentional')
        self.assertIs(module.synthetic_updates,original)

    def test_observer_is_read_only_and_removable(self):
        class Head(torch.nn.Module):
            def forward(self,x):return [torch.zeros(1,4,n,1) for n in [4,3,2,1]]
        class Assigner(torch.nn.Module):
            def forward(self,x):return x
        class Model:
            def __init__(self):self.model=[Head()];self.criterion=types.SimpleNamespace(assigner=Assigner())
        model=Model();audit=coverage.AssignmentAudit(model)
        head_outputs=model.model[-1](None)
        scores=torch.ones(1,10,2);scores[:,4:7]=0
        fg=torch.ones(1,10,dtype=torch.bool);out=(None,None,scores,fg,None)
        snapshots=[v.clone() if isinstance(v,torch.Tensor) else v for v in out]
        self.assertIs(model.criterion.assigner(out),out)
        self.assertEqual(audit.fg,[4,3,2,1]);self.assertEqual(audit.weighted,[4,0,2,1])
        for now,before in zip(out,snapshots):
            if isinstance(now,torch.Tensor):self.assertTrue(torch.equal(now,before))
        audit.close();self.assertEqual(len(model.model[-1]._forward_hooks),0)
        self.assertEqual(len(model.criterion.assigner._forward_hooks),0)
        self.assertEqual(len(head_outputs),4)

if __name__=='__main__':unittest.main(verbosity=2)
