"""CPU fixtures and optional small local CUDA units, not server smoke."""
import copy
from pathlib import Path
import unittest
import torch
from representation import VARIANT,SITE,ADDED_PARAMETERS
HERE=Path(__file__).resolve().parent


class Tests(unittest.TestCase):
    def test_independent_unit(self):
        from extension import routing_audit
        self.assertEqual(routing_audit('cpu')['status'],'passed')

    def test_initialization_graph_profile(self):
        from ultralytics.nn.tasks import DetectionModel,yaml_model_load
        import parent_spatial_task_head as parent_head
        from spatial_task_head import attach,branch,NEW_PARAMETERS
        from extension import branch_groups
        from profiler_accounting import representation_profile
        cfg=yaml_model_load(str(HERE/'e1_reference.yaml')); cfg['scale']='s'
        base=DetectionModel(cfg,nc=10,verbose=False)
        parent=parent_head.attach(copy.deepcopy(base)).eval()
        rng=torch.get_rng_state().clone(); model=attach(copy.deepcopy(base)).eval()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for n,p in parent.state_dict().items(): self.assertTrue(torch.equal(model.state_dict()[n],p),n)
        self.assertEqual(len(model.model),27)
        self.assertEqual(sum(p.numel() for p in model.parameters())-sum(p.numel() for p in parent.parameters()),ADDED_PARAMETERS)
        self.assertEqual(sum(p.numel() for p in branch(model).parameters()),NEW_PARAMETERS)
        for n,p in branch(model).named_parameters(): self.assertEqual(sum(n.startswith(g) for g in branch_groups()),1,n)
        for h,w in ((128,160),(160,128),(128,128)):
            x=torch.rand(1,3,h,w)
            with torch.no_grad(): torch.testing.assert_close(model(x)[0],parent(x)[0],rtol=0,atol=0)
        profile=representation_profile(model,parent,imgsz=640)
        self.assertLess(profile['GFLOPs640'],profile['E40_GFLOPs640']*1.15)
        print('PROFILE',VARIANT,profile,flush=True)

    def test_parent_source_and_gates(self):
        root=HERE.parents[1]
        normalize=lambda p:p.read_bytes().replace(b'\r\n',b'\n')
        self.assertEqual(normalize(HERE/'parent_spatial_task_head.py'),normalize(root/'E40a_SpatialTaskHead_AutoDL/spatial_task_head.py'))
        for n in ('d27_official.py','d38_common.py','p2_tal_chunked.py','e1_reference.yaml'):
            self.assertEqual(normalize(HERE/n),normalize(root/'E49a_E40_GuidedReassembly_AutoDL'/n))
        runner=(HERE/'run_experiment.py').read_text()
        self.assertIn("synchronize('train_ready')",runner)
        self.assertIn("synchronize('validation_done')",runner)
        self.assertIn("if report['completed_epochs']!=200",runner)
        self.assertIn(f'model[{SITE}].representation.capture_stats=True',runner)
        self.assertNotIn('28.',runner)
        from run_experiment import validate_data_yaml
        import hashlib,tempfile
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'data.yaml'; path.write_bytes(b'train: images/train\r\n')
            validate_data_yaml(path,hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(RuntimeError): validate_data_yaml(path,'0'*64)


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
