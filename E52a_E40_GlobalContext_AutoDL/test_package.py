"""Local CPU/synthetic engineering tests. Never issues real CUDA smoke gates."""
from pathlib import Path
import copy
import unittest
import torch

HERE=Path(__file__).resolve().parent


class PackageTests(unittest.TestCase):
    def test_attention_gradient_half_oracles(self):
        from extension import routing_audit
        self.assertEqual(routing_audit('cpu')['status'],'passed')

    def test_parameter_arithmetic(self):
        from global_context import GlobalContextResidual,ADDED_PARAMETERS
        self.assertEqual(sum(p.numel() for p in GlobalContextResidual().parameters()),71744)
        self.assertEqual(ADDED_PARAMETERS,71744)

    def test_original_parent_sources(self):
        root=HERE.parent; parent=root/'E40a_SpatialTaskHead_AutoDL'
        if not parent.is_dir(): self.skipTest('Original local parent absent')
        normalize=lambda p:p.read_bytes().replace(b'\r\n',b'\n')
        self.assertEqual(normalize(HERE/'parent_spatial_task_head.py'),normalize(parent/'spatial_task_head.py'))
        for name in ('d27_official.py','d38_common.py','p2_tal_chunked.py','e1_reference.yaml'):
            self.assertEqual(normalize(HERE/name),normalize(root/'E49a_E40_GuidedReassembly_AutoDL'/name))

    def test_initialization_graph_groups_and_profiles(self):
        from ultralytics.nn.tasks import DetectionModel,yaml_model_load
        import parent_spatial_task_head as parent_head
        from spatial_task_head import attach,branch,NEW_PARAMETERS
        from extension import branch_groups
        from profiler_accounting import context_profile
        cfg=yaml_model_load(str(HERE/'e1_reference.yaml')); cfg['scale']='s'
        base=DetectionModel(cfg,nc=10,verbose=False)
        parent=parent_head.attach(copy.deepcopy(base)).eval()
        rng=torch.get_rng_state().clone()
        model=attach(copy.deepcopy(base)).eval()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(NEW_PARAMETERS,247184)
        for n,p in parent.state_dict().items(): self.assertTrue(torch.equal(model.state_dict()[n],p),n)
        self.assertEqual(len(model.model),27)
        self.assertEqual(model.model[4].i,4)
        self.assertEqual(model.model[15].f,[-1,4])
        self.assertEqual(model.model[26].f,[25,16,19,22])
        groups=branch_groups(); self.assertEqual(len(groups),27)
        for n,p in branch(model).named_parameters():
            self.assertEqual(sum(n.startswith(g) for g in groups),1,n)
        for h,w in ((128,160),(160,128),(128,128)):
            x=torch.rand(1,3,h,w)
            with torch.no_grad(): torch.testing.assert_close(model(x)[0],parent(x)[0],rtol=0,atol=0)
        profile=context_profile(model,parent,imgsz=640)
        self.assertAlmostEqual(profile['added_major_GFLOPs640'],.837607424,places=9)
        self.assertTrue(profile['independent_major_flops_verified'])
        print('Real640 profile:',profile,flush=True)

    def test_gate_entry_and_reference(self):
        shell=(HERE/'run_e52.sh').read_text()
        self.assertIn('set -euo pipefail',shell)
        self.assertIn('for STAGE in check smoke run',shell)
        self.assertIn('flock -n 9',shell)
        runner=(HERE/'run_e52.py').read_text()
        self.assertIn('e52_cuda_check_gate.json',runner)
        self.assertIn('e52_smoke_gate.json',runner)
        self.assertIn("if report['completed_epochs']!=200",runner)
        self.assertIn('model[4].global_context.capture_stats=True',runner)
        self.assertNotIn('model[23].capture_stats',runner)
        import d27_official as official
        import d38_common as common
        self.assertTrue(official.run_unit_tests()); self.assertTrue(common.tests())


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
