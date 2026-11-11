"""Local E60 regression tests. Shutdown runner ALWAYS mocked."""
import copy
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'yolo11_src'))
os.environ.setdefault('YOLO_CONFIG_DIR',str(HERE.parent/'tmp/e60_test_settings'))
os.environ.setdefault('YOLO_AUTOINSTALL','false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
torch.set_num_threads(2)
import spatial_task_head as head
import e58_spatial_task_head as parent
import run_experiment as run
import shutdown_safety as safety


class ReleaseTests(unittest.TestCase):
    def test_frozen_dependencies(self):
        original=HERE.parent/'E58E59_E57_W96_DualReadout_Parallel_AutoDL'
        for name in ('e57_spatial_task_head.py','parent_spatial_task_head.py','e1_640_core.py',
                     'e1_reference.yaml','p2_tal_chunked.py','d27_official.py','d38_common.py',
                     'validation_smoke.py','smoke_progress.py'):
            self.assertEqual(run.digest(HERE/name),run.digest(original/name),name)
        self.assertEqual(run.digest(HERE/'e58_spatial_task_head.py'),run.digest(original/'spatial_task_head.py'))

    def test_model_parent_weights_rng_counts(self):
        from ultralytics.nn.tasks import DetectionModel,yaml_model_load
        cfg=yaml_model_load(str(HERE/'e1_reference.yaml'));cfg['scale']='s'
        base=DetectionModel(cfg,nc=10,verbose=False)
        head.configure('e60'); parent.configure('e58')
        state=torch.get_rng_state().clone()
        target=head.attach(copy.deepcopy(base)); reference=parent.attach(copy.deepcopy(base))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        for name,value in reference.state_dict().items():
            self.assertTrue(torch.equal(value,target.state_dict()[name]),name)
        self.assertEqual(sum(p.numel() for p in target.parameters()),11173112)
        self.assertEqual(sum(p.numel() for p in head.branch(target).parameters()),1494144)
        self.assertEqual(len(head.branch_groups()),52)
        self.assertEqual([len(b.refinements) if hasattr(b,'refinements') else 0 for b in head.branch(target)],[3,3,0,0])
        half=copy.deepcopy(target).half()
        for name,p in half.named_parameters():
            expected=torch.float32 if '.refinements.' in name else torch.float16
            self.assertEqual(p.dtype,expected,name)
        # One hot refiner must not change another original depth's parameters.
        for i in (2,3): self.assertIs(type(head.branch(target)[i]),parent.WideTaskBlock)
        torch.random.default_generator.manual_seed(1234)
        other=head.attach(copy.deepcopy(base))
        for name,value in head.branch(target).state_dict().items():
            self.assertTrue(torch.equal(value,head.branch(other).state_dict()[name]),name)

    def test_protocol_and_references(self):
        args=SimpleNamespace(project=HERE.parent/'yolo11_project',source=HERE.parent/'yolo11_src',
            data=HERE.parent/'tmp/no_local_data.yaml',dataset=HERE.parent/'tmp/no_local_images',
            device='cpu',workers=8,variant='e60',mode='check')
        core=run.setup(args); base=run.protocol_check(core,args); params=run.config(core,args)
        self.assertEqual(base['seed'],1)
        self.assertEqual((params['seed'],params['epochs'],params['batch'],params['imgsz'],params['workers']),(1,200,16,640,8))
        self.assertEqual(params['name'],'e60a_e58_group_residual_encoding_img640_seed1_200e')
        self.assertFalse(core.AUTO_SHUTDOWN or core.SHUTDOWN_ON_FAILURE)

    def test_group_cpu_oracles(self):
        from checks import strict_math
        from group_checks import group_oracle,bank_path_audit
        with strict_math(): group_oracle('cpu'); bank_path_audit('cpu')

    def test_tiny_checkpoint_weights_retained(self):
        from precision_checks import checkpoint_precision_audit
        checkpoint_precision_audit('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required for FP16 autocast')
    def test_tiny_autocast_gradient_and_checkpoint(self):
        from checks import strict_math
        from precision_checks import tiny_weight_audit, checkpoint_precision_audit
        with strict_math():
            legacy=tiny_weight_audit('cuda:0',fixed=False)
            fixed=tiny_weight_audit('cuda:0',fixed=True)
            checkpoint_precision_audit('cuda:0')
        self.assertEqual(legacy['project_FP16_nonzero'],0)
        self.assertFalse(any(legacy['ambient_AMP_inner_gradient_max'].values()))
        self.assertTrue(all(fixed['ambient_AMP_inner_gradient_max'].values()))
        self.assertTrue(fixed['all_inner_groups_really_updated'])

    def test_memory_byte_units(self):
        import smoke_memory
        smoke_memory.configure('e60')
        cuda=mock.Mock();cuda.mem_get_info.return_value=(30*2**30,32229*2**20)
        with smoke_memory.bounded_smoke(cuda,'cuda:0'): pass
        fraction,device=cuda.set_per_process_memory_fraction.call_args.args
        self.assertAlmostEqual(fraction,28/(32229/1024)); self.assertTrue(0<fraction<1)

    def test_sgd_rounding_and_rejection(self):
        from optimizer_checks import SGDUpdateAudit
        for device in (['cpu','cuda:0'] if torch.cuda.is_available() else ['cpu']):
            p=torch.nn.Parameter(torch.ones(3,device=device))
            optimizer=torch.optim.SGD([p],lr=.001,momentum=.937)
            audit=SGDUpdateAudit({'router.weight':p},['router.'])
            for unused in range(8):
                p.grad=torch.full_like(p,1e-10)
                audit.capture(optimizer);optimizer.step();audit.verify(optimizer)
            self.assertEqual(audit.subulp_groups(),{'router.'})
            self.assertFalse(audit.stats['router.']['ever_changed'])
            self.assertEqual(audit.steps,8)
            # A representable update MUST happen; a no-op step is rejected.
            p.grad=torch.ones_like(p);audit.capture(optimizer)
            with self.assertRaisesRegex(RuntimeError,'disagrees'):audit.verify(optimizer)
            optimizer.step();audit.verify(optimizer)
            self.assertTrue(audit.stats['router.']['ever_changed'])
            self.assertFalse(audit.subulp_groups())
            absent=torch.optim.SGD([torch.nn.Parameter(torch.ones(1,device=device))],lr=.001)
            with self.assertRaisesRegex(RuntimeError,'absent'):audit.capture(absent)
            # Zero-gradient directions cannot be waved through as sub-ULP.
            q=torch.nn.Parameter(torch.ones(3,device=device)); opt=torch.optim.SGD([q],lr=.001)
            zero=SGDUpdateAudit({'router.weight':q},['router.']);q.grad=torch.zeros_like(q)
            zero.capture(opt);opt.step();zero.verify(opt)
            self.assertFalse(zero.subulp_groups())

    def test_smoke_receipt_rejection(self):
        from single_safety import validate_smoke_evidence
        groups=head.branch_groups()
        evidence=dict(status='passed',variant='e60',seed=1,single_job=True,
            successful_optimizer_updates=8,branch_updated=True,updated_groups=groups,
            expected_groups=groups,allocator_cap_GiB=28.,peak_reserved_GiB=20.,post_update_inference=True,
            validation=dict(validation_images=548,validation_batch=32,EMA_half=True,square_batch32=True,
                            validation_loss_and_NMS=True,parallel_pair=False))
        validate_smoke_evidence(evidence,'e60')
        for changed in ({'seed':2},{'variant':'e58'},{'updated_groups':groups[:-1]},
                        {'successful_optimizer_updates':7},{'allocator_cap_GiB':15.},
                        {'peak_reserved_GiB':29.},{'validation':{}}):
            with self.assertRaises(RuntimeError): validate_smoke_evidence({**evidence,**changed},'e60')

    def test_cached_official_reference_reproduction_guard(self):
        import parent_reference as ref
        import d27_official as official
        args=SimpleNamespace(project=HERE.parent/'yolo11_project')
        scores=dict(AP50_95_percent=26.,AP50_percent=45.,AP75_percent=26.,AR500_percent=38.)
        with mock.patch.object(official,'load_predictions',return_value=[]), \
             mock.patch.object(official,'filter_detections_and_audit',return_value=([],{})), \
             mock.patch.object(official,'official_evaluate',return_value=scores):
            self.assertEqual(ref.rescore_reference(args,[],ref.E58_REPORT,scores),scores)
            with self.assertRaises(RuntimeError):
                ref.rescore_reference(args,[],ref.E58_REPORT,{**scores,'AP75_percent':26.01})

    def test_shutdown_binary_shebang_and_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=HERE.parent/'tmp') as directory:
            root=Path(directory); command=root/'platform_cmd'
            runner=mock.Mock(return_value=SimpleNamespace(returncode=0,stdout='',stderr=''))
            good=dict(total_MiB=32229,free_MiB=32229,compute_pids=[])
            kwargs=dict(probe=lambda:good,jobs=lambda:[],runner=runner,sleeper=lambda s:None,
                        delay=0,platform_name='linux',command=command)
            for header in (b'#!/bin/bash\necho ok\n',b'\x7fELFmockfixture'):
                command.write_bytes(header)
                result=safety.auto_shutdown(True,root,root/'cancel',**kwargs)
                self.assertEqual(result['status'],'shutdown_command_returned_zero')
                self.assertEqual(runner.call_args.args[0],[str(command)])
            runner.reset_mock()
            kwargs['probe']=lambda:{**good,'compute_pids':[321]}
            self.assertEqual(safety.auto_shutdown(True,root,root/'cancel',**kwargs)['status'],'skipped_safety_check_or_command_error')
            runner.assert_not_called()
            kwargs['probe']=lambda:{**good,'free_MiB':12000}
            self.assertEqual(safety.auto_shutdown(True,root,root/'cancel',**kwargs)['status'],'skipped_safety_check_or_command_error')
            runner.assert_not_called()
            kwargs['probe']=lambda:good
            runner.return_value=SimpleNamespace(returncode=1,stdout='',stderr='mock failure')
            self.assertEqual(safety.auto_shutdown(True,root,root/'cancel',**kwargs)['status'],'shutdown_command_failed')

    def test_shutdown_ascii_bash_and_guards(self):
        with tempfile.TemporaryDirectory(dir=HERE.parent/'tmp') as directory:
            root=Path(directory); cmd=root/'shutdown'; cmd.write_text('echo platform_shutdown\n')
            runner=mock.Mock(return_value=SimpleNamespace(returncode=0,stdout='accepted',stderr=''))
            snapshot=lambda:dict(total_MiB=32229,free_MiB=32229,compute_pids=[])
            kwargs=dict(probe=snapshot,jobs=lambda:[],runner=runner,sleeper=lambda s:None,
                        delay=0,platform_name='linux',command=cmd)
            result=safety.auto_shutdown(True,root,root/'cancel',**kwargs)
            self.assertEqual(result['status'],'shutdown_command_returned_zero')
            self.assertEqual(runner.call_args.args[0],['/bin/bash',str(cmd)])
            runner.reset_mock(); (root/'cancel').touch()
            self.assertEqual(safety.auto_shutdown(True,root,root/'cancel',**kwargs)['status'],'cancelled_by_KEEP_RUNNING_file')
            runner.assert_not_called(); (root/'cancel').unlink()
            kwargs['jobs']=lambda:[{'pid':987,'command':'other training'}]
            self.assertEqual(safety.auto_shutdown(True,root,root/'cancel',**kwargs)['status'],'skipped_safety_check_or_command_error')
            runner.assert_not_called()
            kwargs['jobs']=lambda:[]; kwargs['probe']=mock.Mock(side_effect=RuntimeError('GPU unavailable'))
            self.assertEqual(safety.auto_shutdown(True,root,root/'cancel',**kwargs)['status'],'skipped_safety_check_or_command_error')
            runner.assert_not_called()
            self.assertEqual(safety.auto_shutdown(False,root,root/'cancel',**kwargs)['status'],'disabled')
            runner.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
