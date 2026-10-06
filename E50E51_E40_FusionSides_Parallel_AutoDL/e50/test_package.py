"""Synthetic-only tests; never creates real CUDA smoke gates."""
import copy, hashlib, unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from guided_reassembly import SITE,VARIANT,GuidedReassembly,ADDED_PARAMETERS,added_conv_flops
HERE=Path(__file__).resolve().parent

class PackageTests(unittest.TestCase):
    def test_independent_index_and_precision(self):
        from extension import routing_audit
        self.assertEqual(routing_audit('cpu')['status'],'passed')
    def models(self):
        from ultralytics.nn.tasks import DetectionModel,yaml_model_load
        from spatial_task_head import attach
        import parent_spatial_task_head as parent
        cfg=yaml_model_load(str(HERE/'e1_reference.yaml')); cfg['scale']='s'
        base=DetectionModel(cfg,nc=10,verbose=False)
        rng=torch.get_rng_state().clone()
        target=attach(copy.deepcopy(base)).eval()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        return base,parent.attach(copy.deepcopy(base)).eval(),target
    def test_graph_common_state_and_groups(self):
        from spatial_task_head import branch,NEW_PARAMETERS
        from extension import branch_groups
        base,parent,target=self.models()
        for name,value in parent.state_dict().items():
            torch.testing.assert_close(value,target.state_dict()[name],rtol=0,atol=0)
        groups=branch_groups()
        for name,p in branch(target).named_parameters():
            self.assertEqual(sum(name.startswith(g) for g in groups),1,name)
        self.assertEqual(sum(p.numel() for p in branch(target).parameters()),NEW_PARAMETERS)
        self.assertEqual(sum(p.numel() for p in target.parameters())-sum(p.numel() for p in parent.parameters()),8778)
        self.assertEqual(target.model[26].f,[25,16,19,22])
        self.assertEqual(target.model[SITE].f,[16,2] if SITE==23 else [-1,2,16])
        if SITE==24:
            self.assertIs(type(target.model[23]),nn.Upsample)
            captured={}
            def original(m,x,y): captured['B']=y; captured['guard']=y.clone()
            def concat(m,x,y):
                torch.testing.assert_close(captured['B'],captured['guard'],rtol=0,atol=0)
                self.assertIs(x[0][1],captured['B'])
            h1=target.model[2].register_forward_hook(original)
            h2=target.model[24].register_forward_hook(concat)
            with torch.no_grad(): target(torch.randn(1,3,128,160))
            h1.remove();h2.remove()
    def test_profile_policies_shared_DW_twice(self):
        import thop
        from ultralytics.utils.torch_utils import get_flops
        from profiler_accounting import reassembly_profile_delta
        _,parent,target=self.models()
        original=thop.profile
        def norm(m,x,y): m.total_ops+=2*x[0].numel()
        def zero(m,x,y): pass
        def up(m,x,y): m.total_ops+=y.numel()
        for count_norm in (False,True):
            for count_up in (False,True):
                def profiling(m,*a,**kw):
                    kw['custom_ops']={nn.GroupNorm:norm if count_norm else zero,nn.Upsample:up if count_up else zero}
                    return original(m,*a,**kw)
                with patch.object(thop,'profile',side_effect=profiling):
                    accounting=reassembly_profile_delta(target.model[SITE],parent.model[SITE])
                    self.assertAlmostEqual(accounting['independent_conv_GFLOPs640'],added_conv_flops()/1e9,places=10)
                    actual=get_flops(target,imgsz=640); old=get_flops(parent,imgsz=640)
                    self.assertAlmostEqual(actual,old+accounting['expected_whole_model_delta_GFLOPs640'],places=8)
                    self.assertEqual(len(accounting['observed_convolutions']),5)
    def test_preserved_parent_and_evaluation(self):
        original=HERE.parents[1]/'E49a_E40_GuidedReassembly_AutoDL'
        for name in ('parent_spatial_task_head.py','e1_640_core.py','p2_tal_chunked.py','d27_official.py','d38_common.py','e1_reference.yaml'):
            self.assertEqual((HERE/name).read_bytes().replace(b'\r\n',b'\n'),(original/name).read_bytes().replace(b'\r\n',b'\n'))
        import d27_official as official,d38_common as common
        self.assertTrue(official.run_unit_tests());self.assertTrue(common.tests())
    def test_controller_cap_and_synchronized_validation(self):
        runner=(HERE/'run_e49.py').read_text()
        self.assertIn("synchronize('train_ready')",runner);self.assertIn("synchronize('validation_done')",runner)
        self.assertIn('13.*2**30/total',runner)
        self.assertIn("if report['completed_epochs']!=200",runner)
        self.assertNotIn("requires an idle >=30",runner)
        self.assertEqual(sum(p.numel() for p in GuidedReassembly().parameters()),ADDED_PARAMETERS)
if __name__=='__main__':
    torch.set_num_threads(4); unittest.main(verbosity=2)
