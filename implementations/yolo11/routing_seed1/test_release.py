"""Local tests only. Never execute an actual platform shutdown or server job."""
import copy
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
sys.path.insert(0,str(PAPER/'code/server_received_20261013/yolo11_src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
os.environ['YOLO_CONFIG_DIR'] = str(PAPER/'audit/local_ablation_tests/settings')
import torch
torch.set_num_threads(2)
import common as c
import run_experiment as run
import spatial_task_head as head
TEST_TEMP = PAPER/'audit/local_ablation_tests/temp'
TEST_TEMP.mkdir(parents=True,exist_ok=True)


def arguments(variant='uniform'):
    return SimpleNamespace(variant=variant,execution='serial',mode='local-check',project=PAPER/'project',
        source=PAPER/'code/server_received_20261013/yolo11_src',data=PAPER/'audit/missing_local_data.yaml',
        dataset=PAPER/'audit/missing_local_images',device='cpu',workers=8)


class ReleaseTests(unittest.TestCase):
    def test_frozen_actual_source(self):
        c.package_audit(); c.frozen_audit(PAPER/'project')

    def test_counts_rng_initialization_scope(self):
        for variant, total, groups in (('uniform',11121752,48),('shared',11172584,52)):
            core = run.setup(arguments(variant))
            model, base, _ = run.build_model(core)
            self.assertEqual(sum(p.numel() for p in model.parameters()),total)
            self.assertEqual(len(head.branch_groups()),groups)
            head.assert_topology(model,variant)
            saved = copy.deepcopy(model).half()
            for name,p in saved.named_parameters():
                self.assertEqual(p.dtype,torch.float32 if '.refinements.' in name else torch.float16)

    def test_routing_oracles_all_channels(self):
        from routing_checks import routing_unit_channels
        for variant in c.VARIANTS:
            head.configure(variant)
            for cin in (128,256,512):
                self.assertTrue(routing_unit_channels('cpu',cin)['weighted_sum_and_gradient_oracle'])

    def test_active_computation_counts(self):
        from routing_checks import profile_audit
        for variant in c.VARIANTS:
            core = run.setup(arguments(variant))
            model, base, _ = run.build_model(core)
            self.assertGreater(profile_audit(model,base)['candidate']['Conv_Linear_GMACs'],0)

    def test_serial_parallel_caps_and_gates_different(self):
        a = arguments()
        x = c.gate_path(a,'check'); a.execution='parallel'
        self.assertNotEqual(x,c.gate_path(a,'check'))
        import smoke_memory as memory
        memory.configure('parallel')
        cuda = mock.Mock(); cuda.mem_get_info.return_value=(31*2**30,32*2**30)
        with memory.bounded_smoke(cuda,'cuda:0'): pass
        cuda.set_per_process_memory_fraction.assert_called_once_with(15/32,'cuda:0')
        memory.configure('serial')
        with memory.bounded_smoke(cuda,'cuda:0'): pass
        self.assertEqual(cuda.set_per_process_memory_fraction.call_args.args[0],28/32)

    def test_stale_and_cpu_gate_rejected(self):
        a = arguments()
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            a.project=Path(folder)
            out=a.project/'evidence'; out.mkdir()
            c.write(out/'check.json',dict(status='passed',cuda_executed=False))
            c.write(c.gate_path(a,'check'),dict(signature={'x':1},report=str(out)))
            with self.assertRaises(RuntimeError): c.verify_gate(a,'check',{'x':2})
            with self.assertRaises(RuntimeError): c.verify_gate(a,'check',{'x':1})

    def test_final_barrier_allows_exited_successful_peer(self):
        import pair_safety as pair
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            a=arguments(); a.execution='parallel'; a.pair=Path(folder)
            jobs={'uniform':dict(pid=os.getpid(),start_ticks=10),'shared':dict(pid=999999,start_ticks=20)}
            c.write(a.pair/'children.json',jobs)
            directory=a.pair/'barriers/both_smoke_receipts_saved'; directory.mkdir(parents=True)
            c.write(directory/'shared.json',dict(pid=999999,start_ticks=20))
            with mock.patch.object(pair,'start_ticks',side_effect=lambda pid:10 if pid==os.getpid() else (_ for _ in ()).throw(RuntimeError('gone'))):
                pair.barrier(a,'both_smoke_receipts_saved',timeout=1)

    def test_shutdown_ascii_and_busy_mocked(self):
        import shutdown_safety as safety
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            root=Path(folder); command=root/'shutdown'
            command.write_text('echo platform shutdown\n')
            runner=mock.Mock(return_value=SimpleNamespace(returncode=0,stdout='',stderr=''))
            idle=lambda:dict(total_MiB=32768,free_MiB=32768,compute_pids=[])
            result=safety.auto_shutdown(True,root,root/'KEEP',delay=0,probe=idle,jobs=lambda:[],runner=runner,
                                       sleeper=lambda x:None,platform_name='linux',command=command)
            self.assertEqual(result['status'],'shutdown_command_returned_zero')
            self.assertEqual(runner.call_args.args[0],['/bin/bash',str(command)])
            runner.reset_mock()
            safety.auto_shutdown(True,root,root/'KEEP',delay=0,probe=idle,
                                 jobs=lambda:[{'pid':123,'command':'train.py'}],runner=runner,
                                 sleeper=lambda x:None,platform_name='linux',command=command)
            runner.assert_not_called()

    def test_controller_execution_commands(self):
        import controller
        a=arguments()
        command=controller.command(a,'uniform','run',Path('report'))
        self.assertIn('--execution',command)
        self.assertEqual(command[command.index('--execution')+1],'serial')
        self.assertNotIn('--weights',command)
        self.assertEqual(command[command.index('--variant')+1],'uniform')

    def test_serial_training_failure_does_not_skip_sibling(self):
        import controller
        a=arguments()
        with mock.patch.object(controller,'wait_released'), \
             mock.patch.object(controller,'launch',side_effect=[{'log':Path('u.log')},{'log':Path('s.log')}]) as spawn, \
             mock.patch.object(controller,'wait_jobs',side_effect=[{'uniform':1},{'shared':0}]):
            result=controller.run_stage(a,'run',Path('session'),Path('reports'),{})
        self.assertEqual(result,{'uniform':1,'shared':0})
        self.assertEqual(spawn.call_count,2)

    def test_parallel_training_failure_keeps_successful_sibling(self):
        import controller
        a=arguments(); a.execution='parallel'
        with mock.patch.object(controller,'launch',side_effect=[{'variant':'uniform'},{'variant':'shared'}]), \
             mock.patch.object(controller,'wait_jobs',return_value={'uniform':1,'shared':0}):
            result=controller.run_stage(a,'run',Path('session'),Path('reports'),{})
        self.assertEqual(result,{'uniform':1,'shared':0})


if __name__ == '__main__':
    unittest.main(verbosity=2)
