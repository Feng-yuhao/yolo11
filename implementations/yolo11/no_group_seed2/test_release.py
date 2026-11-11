"""Local unit tests. Shutdown is ALWAYS mocked, never executed."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'yolo11_src'))
os.environ.setdefault('YOLO_CONFIG_DIR',str(HERE.parent/'tmp/e58_seed2_test_settings'))
os.environ.setdefault('YOLO_AUTOINSTALL','false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
torch.set_num_threads(2)
import spatial_task_head as head
import run_experiment as run
import shutdown_safety as safety
from single_safety import validate_smoke_evidence, CAPS


class ReplicationTests(unittest.TestCase):
    def test_frozen_core_source_identity(self):
        original=HERE.parent/'E58E59_E57_W96_DualReadout_Parallel_AutoDL'
        for name in ('spatial_task_head.py','e57_spatial_task_head.py','parent_spatial_task_head.py',
                     'e1_640_core.py','e1_reference.yaml','p2_tal_chunked.py','d27_official.py',
                     'd38_common.py','validation_smoke.py','smoke_progress.py'):
            self.assertEqual(run.digest(HERE/name),run.digest(original/name),name)

    def test_paired_seed2_protocol_and_references(self):
        args=SimpleNamespace(project=HERE.parent/'yolo11_project',source=HERE.parent/'yolo11_src',
            data=HERE.parent/'missing_local_dataset.yaml',dataset=HERE.parent/'missing_local_images',
            device='cpu',workers=8,variant='e58',mode='check')
        core=run.setup(args)
        base=run.protocol_check(core,args)
        config=run.config(core,args)
        self.assertEqual(base['seed'],2)
        self.assertEqual(config['seed'],2)
        self.assertEqual((config['epochs'],config['batch'],config['imgsz'],config['workers']),(200,16,640,8))
        self.assertEqual(config['name'],'e58a_e57_allscale_w96_img640_seed2_200e')
        self.assertFalse(core.AUTO_SHUTDOWN or core.SHUTDOWN_ON_FAILURE)

    def test_width_initialization_and_rng(self):
        from ultralytics.nn.tasks import DetectionModel,yaml_model_load
        cfg=yaml_model_load(str(HERE/'e1_reference.yaml'));cfg['scale']='s'
        base=DetectionModel(cfg,nc=10,verbose=False)
        head.configure('e58')
        rng=torch.get_rng_state().clone()
        model=head.attach(copy.deepcopy(base))
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        blocks=head.branch(model)
        self.assertEqual([b.reduce[0].out_channels for b in blocks],[96]*4)
        self.assertEqual([b.reduce[0].in_channels for b in blocks],[128,128,256,512])
        self.assertEqual(sum(p.numel() for p in blocks.parameters()),1344672)
        self.assertEqual(sum(p.numel() for p in model.parameters()),11023640)
        for name,value in base.state_dict().items():
            self.assertTrue(torch.equal(model.state_dict()[name],value),name)
        # Changing the global training seed does not change isolated E58 branch initialization.
        torch.manual_seed(2)
        other=head.attach(copy.deepcopy(base))
        for name,value in blocks.state_dict().items():
            self.assertTrue(torch.equal(value,head.branch(other).state_dict()[name]),name)

    def test_memory_units(self):
        import smoke_memory
        smoke_memory.configure('e58')
        cuda=mock.Mock();cuda.mem_get_info.return_value=(30*2**30,32229*2**20)
        with smoke_memory.bounded_smoke(cuda,'cuda:0'): pass
        fraction,_=cuda.set_per_process_memory_fraction.call_args.args
        self.assertAlmostEqual(fraction,15/(32229/1024))
        self.assertTrue(0<fraction<1)

    def test_smoke_rejects_old_seed_and_missing_groups(self):
        head.configure('e58')
        groups=head.branch_groups()
        value=dict(status='passed',variant='e58',seed=2,single_job=True,
                   successful_optimizer_updates=8,branch_updated=True,updated_groups=groups,
                   expected_groups=groups,allocator_cap_GiB=15.,peak_reserved_GiB=13.,post_update_inference=True,
                   validation=dict(validation_images=548,validation_batch=32,EMA_half=True,
                                   square_batch32=True,validation_loss_and_NMS=True,parallel_pair=False))
        validate_smoke_evidence(value,'e58')
        for key,changed in (('seed',1),('successful_optimizer_updates',7),('updated_groups',[]),
                            ('allocator_cap_GiB',28),('peak_reserved_GiB',16)):
            bad=copy.deepcopy(value);bad[key]=changed
            with self.assertRaises(RuntimeError):validate_smoke_evidence(bad,'e58')
        bad=copy.deepcopy(value);bad['validation']['validation_images']=8
        with self.assertRaises(RuntimeError):validate_smoke_evidence(bad,'e58')


IDLE=dict(total_MiB=32229,free_MiB=32229,compute_pids=[])


class ShutdownTests(unittest.TestCase):
    def fixture(self,temp,**changes):
        root=Path(temp)
        command=root/'fake_shutdown';command.touch()  # never executed: runner is mocked
        options=dict(enabled=True,state_dir=root,cancel_file=root/'KEEP_RUNNING',delay=5,
                     platform_name='linux',command=command,probe=mock.Mock(return_value=IDLE),
                     jobs=mock.Mock(return_value=[]),runner=mock.Mock(return_value=SimpleNamespace(
                         returncode=0,stdout='fake accepted',stderr='')),sleeper=mock.Mock())
        options.update(changes)
        return options

    def test_success_saved_then_only_native_command(self):
        with tempfile.TemporaryDirectory() as temp:
            options=self.fixture(temp)
            result=safety.auto_shutdown(**options)
            self.assertEqual(result['status'],'shutdown_command_returned_zero')
            self.assertEqual(options['probe'].call_count,2)
            self.assertEqual(options['runner'].call_args.args[0],[str(options['command'])])
            self.assertTrue((Path(temp)/'shutdown_status.json').is_file())

    def test_busy_gpu_before_and_after_countdown(self):
        for sequence in ((dict(IDLE,compute_pids=[99]),),(IDLE,dict(IDLE,compute_pids=[99]))):
            with tempfile.TemporaryDirectory() as temp:
                options=self.fixture(temp,probe=mock.Mock(side_effect=sequence))
                result=safety.auto_shutdown(**options)
                self.assertEqual(result['status'],'skipped_safety_check_or_command_error')
                options['runner'].assert_not_called()

    def test_disabled_or_cancelled_never_execute(self):
        for disabled,cancelled in ((True,False),(False,True)):
            with tempfile.TemporaryDirectory() as temp:
                options=self.fixture(temp,enabled=not disabled)
                if cancelled: options['cancel_file'].touch()
                safety.auto_shutdown(**options)
                options['runner'].assert_not_called()
                options['probe'].assert_not_called()

    def test_unknown_state_or_cpu_job_prevents_shutdown(self):
        for changes in (dict(probe=mock.Mock(side_effect=RuntimeError('no nvidia-smi'))),
                        dict(jobs=mock.Mock(return_value=[dict(pid=123,command='train.py')])),
                        dict(platform_name='win32')):
            with tempfile.TemporaryDirectory() as temp:
                options=self.fixture(temp,**changes)
                safety.auto_shutdown(**options)
                options['runner'].assert_not_called()

    def test_command_failure_is_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            options=self.fixture(temp,runner=mock.Mock(return_value=SimpleNamespace(
                returncode=1,stdout='',stderr='fixture failure')))
            self.assertEqual(safety.auto_shutdown(**options)['status'],'shutdown_command_failed')

    def test_admission_requires_exclusive_gpu(self):
        safety.require_idle(IDLE,[],admission=True)
        for state in (dict(IDLE,compute_pids=[1]),dict(IDLE,free_MiB=20000),dict(IDLE,total_MiB=16000)):
            with self.assertRaises(RuntimeError):safety.require_idle(state,[],admission=True)

    def test_controller_order_failure_and_duplicate_are_safe(self):
        import controller
        cases=('success','child_error','admission_error','duplicate')
        for case in cases:
            with tempfile.TemporaryDirectory() as temp:
                fake_fcntl=SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=mock.Mock())
                if case=='duplicate':fake_fcntl.flock.side_effect=BlockingIOError()
                child=SimpleNamespace(pid=12345,stdout=io.StringIO('fixture live output\n'),
                                      wait=mock.Mock(return_value=1 if case=='child_error' else 0))
                # Fresh stdout per stage.
                def popen(*args,**kwargs):
                    child.stdout=io.StringIO('fixture live output\n')
                    return child
                with mock.patch.dict(sys.modules,{'fcntl':fake_fcntl}), \
                     mock.patch.object(sys,'platform','linux'), \
                     mock.patch.object(sys,'argv',['controller.py','all','--auto-shutdown','--project',temp]), \
                     mock.patch.object(controller,'gpu_snapshot',return_value=IDLE), \
                     mock.patch.object(controller,'active_python_jobs',return_value=[]), \
                     mock.patch.object(controller,'require_idle',side_effect=RuntimeError('busy') if case=='admission_error' else None), \
                     mock.patch.object(controller.signal,'signal'), \
                     mock.patch.object(controller.subprocess,'Popen',side_effect=popen) as launch, \
                     mock.patch.object(controller,'auto_shutdown',return_value=dict(status='disabled')) as shutdown:
                    if case=='duplicate':
                        with self.assertRaises(RuntimeError):controller.main()
                        shutdown.assert_not_called();launch.assert_not_called()
                    else:
                        code=controller.main()
                        self.assertEqual(code,0 if case=='success' else 1)
                        self.assertEqual(shutdown.call_args.args[0],case!='admission_error')
                        modes=[call.args[0][call.args[0].index('--mode')+1] for call in launch.call_args_list]
                        self.assertEqual(modes,list(controller.MODES['all']) if case=='success' else ['check'] if case=='child_error' else [])


if __name__=='__main__': unittest.main(verbosity=2)
