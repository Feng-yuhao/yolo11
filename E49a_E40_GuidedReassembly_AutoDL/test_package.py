"""Local synthetic unit tests; do not issue CUDA server smoke receipts."""
import importlib.util
from pathlib import Path
import unittest
import torch

HERE=Path(__file__).resolve().parent


class PackageTests(unittest.TestCase):
    def test_index_gradient_and_initialization(self):
        from extension import routing_audit
        self.assertEqual(routing_audit('cpu')['status'],'passed')

    def test_parameter_and_conv_budget(self):
        from guided_reassembly import GuidedReassembly,ADDED_PARAMETERS,added_conv_flops
        self.assertEqual(sum(p.numel() for p in GuidedReassembly().parameters()),ADDED_PARAMETERS)
        self.assertEqual(ADDED_PARAMETERS,8778)
        self.assertAlmostEqual(added_conv_flops(640)/1e9,.2916352)

    def test_original_parent_and_official_code(self):
        import hashlib
        root=HERE.parent
        original=root/'E40a_SpatialTaskHead_AutoDL'
        if not original.is_dir(): self.skipTest('Original local parent absent on server')
        def normalized(p): return p.read_bytes().replace(b'\r\n',b'\n')
        self.assertEqual(normalized(HERE/'parent_spatial_task_head.py'),normalized(original/'spatial_task_head.py'))
        for name in ('d27_official.py','d38_common.py','p2_tal_chunked.py','e1_reference.yaml'):
            self.assertEqual(normalized(HERE/name),normalized(root/'E47a_E40_DRF_AutoDL'/name))

    def test_profiler_hook_policy_variations(self):
        import copy
        import thop
        from unittest.mock import patch
        from guided_reassembly import GuidedReassembly
        from profiler_accounting import reassembly_profile_delta
        from ultralytics.nn.tasks import DetectionModel,yaml_model_load
        from ultralytics.utils.torch_utils import get_flops
        import parent_spatial_task_head as parent_head
        from spatial_task_head import attach
        cfg=yaml_model_load(str(HERE/'e1_reference.yaml')); cfg['scale']='s'
        base=DetectionModel(cfg,nc=10,verbose=False)
        parent=parent_head.attach(copy.deepcopy(base)).eval()
        target=attach(copy.deepcopy(base)).eval()
        original_profile=thop.profile

        def count_norm(module,inputs,output):
            module.total_ops += 2*inputs[0].numel()
        def zero(module,inputs,output):
            pass
        def nearest(module,inputs,output):
            module.total_ops += output.numel()

        # Same public profiler API, four realistic hook-policy combinations.
        # No dependency edits and no artificial change to model forward.
        for norm_counted in (False,True):
            for up_counted in (False,True):
                policy={torch.nn.GroupNorm:count_norm if norm_counted else zero,
                        torch.nn.Upsample:nearest if up_counted else zero}
                def profiling(model,*args,**kwargs):
                    kwargs['custom_ops']={**kwargs.get('custom_ops',{}),**policy}
                    return original_profile(model,*args,**kwargs)
                with patch.object(thop,'profile',side_effect=profiling):
                    accounting=reassembly_profile_delta(target.model[23],parent.model[23])
                    parent_flops=get_flops(parent,imgsz=640)
                    actual=get_flops(target,imgsz=640)
                    expected=parent_flops+accounting['expected_whole_model_delta_GFLOPs640']
                    self.assertAlmostEqual(actual,expected,places=8)
                    self.assertAlmostEqual(accounting['independent_conv_GFLOPs640'],.2916352,places=10)
                    self.assertAlmostEqual(accounting['profiler_nonconv_GFLOPs640'],.004096 if norm_counted else 0.,places=10)
                    self.assertAlmostEqual(accounting['old_nn_upsample_profiler_GFLOPs640'],.0065536 if up_counted else 0.,places=10)
                    old_formula=parent_flops+.2916352-accounting['old_nn_upsample_profiler_GFLOPs640']
                    self.assertAlmostEqual(actual-old_formula,.004096 if norm_counted else 0.,places=8)
                    print(f'Profiler regression: GroupNorm={norm_counted}, nearest={up_counted}, '
                          f'old_formula_error={actual-old_formula:.7f}, corrected_error={actual-expected:.7f}',flush=True)

    def test_groups_unique_and_complete(self):
        from extension import branch_groups
        from spatial_task_head import branch,attach,NEW_PARAMETERS
        from ultralytics.nn.tasks import DetectionModel,yaml_model_load
        cfg=yaml_model_load(str(HERE/'e1_reference.yaml')); cfg['scale']='s'
        base=DetectionModel(cfg,nc=10,verbose=False)
        rng=torch.get_rng_state().clone()
        target=attach(base)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        groups=branch_groups(); self.assertEqual(len(groups),19)
        for n,p in branch(target).named_parameters():
            self.assertEqual(sum(n.startswith(g) for g in groups),1,n)
        self.assertEqual(sum(p.numel() for p in branch(target).parameters()),NEW_PARAMETERS)
        self.assertEqual(target.model[23].f,[16,2])
        self.assertEqual(target.model[24].f,[-1,2])
        self.assertEqual(target.model[26].f,[25,16,19,22])

    def test_shell_and_gate_entry(self):
        shell=(HERE/'run_e49.sh').read_text()
        self.assertIn('set -euo pipefail',shell)
        self.assertIn('for STAGE in check smoke run',shell)
        self.assertIn('/root/miniconda3/envs/yolo11/bin/python',shell)
        self.assertIn('flock -n 9',shell)
        runner=(HERE/'run_e49.py').read_text()
        self.assertNotIn('from depth_readout',runner)
        self.assertIn('e49_cuda_check_gate.json',runner)
        self.assertIn('e49_smoke_gate.json',runner)
        self.assertIn("if report['completed_epochs']!=200",runner)

    def test_reference_metrics_and_official_units(self):
        import d27_official as official
        import d38_common as common
        self.assertTrue(official.run_unit_tests())
        self.assertTrue(common.tests())


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
