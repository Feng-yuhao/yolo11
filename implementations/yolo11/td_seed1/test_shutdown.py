"""Shutdown regression tests. Every shutdown runner/probe/lock is fake."""
import copy
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile
import shutdown_safety as safety
import shutdown_after as after
from single_safety import EXPERIMENTS

HERE=Path(__file__).resolve().parent
FREE=dict(total_MiB=32760,free_MiB=32230,compute_pids=[])
EMPTY=dict(jobs=[],ignored_services=[])


class Clock:
    def __init__(self): self.value=0.
    def __call__(self): return self.value
    def sleep(self,seconds): self.value+=seconds


class ShutdownTests(unittest.TestCase):
    def setUp(self):
        scratch=(HERE.parent.parent/'tood_td_test_scratch').resolve()
        scratch.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix='tood_td_shutdown_test_',dir=str(scratch))
        self.root=Path(self.temporary.name).resolve()
        self.project=self.root/'project'; self.session=self.project/'logs/tood_td_all_test'
        self.report=self.project/'comparison_reports/TOOD_TD_ALL_test'
        self.session.mkdir(parents=True); self.report.mkdir(parents=True)
        self.command=self.root/'shutdown'
        self.command.write_text('# user execute shutdown command in container\n# /proc/1/fd/1 supervisord\nexit 0\n')
        self.clock=Clock(); self.runner=mock.Mock(return_value=SimpleNamespace(returncode=0,stdout='accepted',stderr=''))
        self.state=dict(mode='all',auto_shutdown=True,admitted=True,status='failed',child_active=False,
                        controller_identity=dict(pid=100,start_ticks=123),report_root=str(self.report),
                        shutdown_command=safety.shutdown_invocation(self.command))
        safety.write(self.session/'state.json',self.state)
        safety.write(self.project/'logs/latest_tood_td_controller.json',dict(session=str(self.session)))

    def tearDown(self): self.temporary.cleanup()

    def run_watch(self,**extra):
        options=dict(platform_name='linux',probe=lambda:copy.deepcopy(FREE),processes=lambda:copy.deepcopy(EMPTY),
                     identity=lambda pid:None,locks=lambda path:True,runner=self.runner,
                     sleeper=self.clock.sleep,clock=self.clock,command=self.command,wait_seconds=120,delay=10)
        options.update(extra)
        return after.run_watch(self.session,self.project,**options)

    def update(self,**extra):
        self.state.update(extra); safety.write(self.session/'state.json',self.state)

    def test_no_shebang_always_bash_not_direct_exec(self):
        result=self.run_watch()
        self.assertEqual(result['status'],'shutdown_command_returned_zero')
        self.assertEqual(self.runner.call_args.args[0],['/bin/bash',str(self.command)])
        self.assertTrue(result['platform_poweroff_and_billing_stop_not_confirmed'])

    def test_known_shebang_platform_script(self):
        self.command.write_text('#!/bin/bash\n# user execute shutdown command in container\n# /proc/1/fd/1 supervisord\nexit 0\n')
        self.update(shutdown_command=safety.shutdown_invocation(self.command))
        self.assertEqual(self.run_watch()['status'],'shutdown_command_returned_zero')

    def test_unknown_helper_no_fallback(self):
        self.command.write_text('exit 0\n')
        with self.assertRaises(RuntimeError): safety.shutdown_invocation(self.command)
        self.assertEqual(self.run_watch()['status'],'shutdown_watcher_error')
        self.runner.assert_not_called()

    def test_tensorboard_python_u_and_module_are_readonly(self):
        for argv in (['python','-u','/env/bin/tensorboard','--logdir','runs'],
                     ['python','-I','-m','tensorboard.main'],['python','-m','jupyterlab']):
            self.assertEqual(safety.classify(argv,HERE)['policy'],'ignored_readonly_frontend')
        self.assertEqual(safety.classify(['python','-u',str(HERE/'watch_progress.py')],HERE)['policy'],
                         'ignored_readonly_frontend')

    def test_unknown_python_and_notebook_kernel_block(self):
        for argv in (['python','-u','train.py'],['python','unknown.py'],['python','-m','ipykernel_launcher'],
                     ['python','-c','print(1)'],['python','queue_next.py'],['python','watch_shutdown.py']):
            self.assertEqual(safety.classify(argv,HERE)['policy'],'blocks_shutdown')

    def test_gpu_pid_blocks_even_tensorboard_is_exempt(self):
        with self.assertRaises(RuntimeError): safety.require_idle({**FREE,'compute_pids':[854]},EMPTY)
        result=self.run_watch(probe=lambda:{**FREE,'compute_pids':[854]})
        self.assertEqual(result['status'],'blocked_idle_wait_timeout'); self.runner.assert_not_called()

    def test_partial_failure_saved_before_request(self):
        (self.report/'failure.json').write_text('{"error":"smoke failed"}')
        def request(*args,**kwargs):
            saved=safety.read(self.session/'PRESERVATION.json')
            self.assertTrue(saved['completed_output_errors'])
            self.assertEqual(saved['terminal_status'],'failed')
            self.assertEqual(safety.read(self.session/'shutdown_status.json')['status'],'requesting_shutdown')
            self.assertFalse(saved['run_present'])
            return SimpleNamespace(returncode=0,stdout='',stderr='')
        result=self.run_watch(runner=request)
        self.assertEqual(result['status'],'shutdown_command_returned_zero')

    def test_disabled_cancelled_not_admitted_never_shutdown(self):
        for extra in ({'auto_shutdown':False},{'status':'cancelled'},{'admitted':False},{'child_active':True}):
            old=copy.deepcopy(self.state); self.update(**extra)
            self.assertEqual(self.run_watch()['status'],'not_authorized_or_not_terminal')
            self.state=old; self.update()
        self.runner.assert_not_called()

    def test_keep_running_before_countdown(self):
        (self.project/'KEEP_RUNNING_TOOD_TD').touch()
        self.assertEqual(self.run_watch()['status'],'cancelled_by_KEEP_RUNNING_file')
        self.runner.assert_not_called()

    def test_keep_running_during_countdown(self):
        def sleep(seconds):
            self.clock.sleep(seconds); (self.project/'KEEP_RUNNING_TOOD_TD').touch()
        self.assertEqual(self.run_watch(sleeper=sleep)['status'],'cancelled_by_KEEP_RUNNING_file')
        self.runner.assert_not_called()

    def test_waits_for_own_parent_identity(self):
        result=self.run_watch(identity=lambda pid:self.state['controller_identity'] if self.clock()<6 else None)
        self.assertEqual(result['status'],'shutdown_command_returned_zero')
        self.assertGreaterEqual(self.clock(),16)

    def test_parent_never_exits(self):
        result=self.run_watch(identity=lambda pid:self.state['controller_identity'])
        self.assertEqual(result['status'],'blocked_parent_still_active'); self.runner.assert_not_called()

    def test_gpu_release_is_waited_for(self):
        result=self.run_watch(probe=lambda:{**FREE,'compute_pids':[200]} if self.clock()<20 else copy.deepcopy(FREE))
        self.assertEqual(result['status'],'shutdown_command_returned_zero')
        self.assertGreaterEqual(self.clock(),30)

    def test_other_eval_starts_during_countdown(self):
        busy={'jobs':[{'pid':222,'target':'eval.py'}],'ignored_services':[]}
        result=self.run_watch(processes=lambda:busy if 5<=self.clock()<20 else copy.deepcopy(EMPTY))
        self.assertEqual(result['status'],'shutdown_command_returned_zero')
        self.assertGreaterEqual(self.clock(),30)

    def test_known_readonly_tensorboard_does_not_block_idle(self):
        result=self.run_watch(processes=lambda:dict(jobs=[],ignored_services=[{'pid':854,'target':'tensorboard'}]))
        self.assertEqual(result['status'],'shutdown_command_returned_zero')

    def test_new_session_cancels_old_poweroff(self):
        safety.write(self.project/'logs/latest_tood_td_controller.json',dict(session=str(self.project/'logs/other')))
        self.assertEqual(self.run_watch()['status'],'cancelled_new_TOOD_TD_session'); self.runner.assert_not_called()

    def test_controller_lock_is_never_bypassed(self):
        self.assertEqual(self.run_watch(locks=lambda path:False)['status'],'blocked_active_TOOD_TD_controller_lock')
        self.runner.assert_not_called()

    def test_platform_helper_change_blocks(self):
        self.command.write_text(self.command.read_text()+'# changed\n')
        self.assertEqual(self.run_watch()['status'],'blocked_platform_helper_changed'); self.runner.assert_not_called()

    def test_nonzero_command_is_not_success(self):
        self.runner.return_value=SimpleNamespace(returncode=3,stdout='',stderr='failure')
        self.assertEqual(self.run_watch()['status'],'shutdown_command_failed')

    def test_duplicate_command_is_not_reissued(self):
        safety.write(self.session/'shutdown_status.json',dict(status='requesting_shutdown'))
        self.assertEqual(self.run_watch()['status'],'requesting_shutdown'); self.runner.assert_not_called()

    def test_no_linux_guard(self):
        self.assertEqual(self.run_watch(platform_name='win32')['status'],'blocked_non_linux')
        self.runner.assert_not_called()

    def test_completed_without_outputs_does_not_poweroff(self):
        self.update(status='completed')
        self.assertEqual(self.run_watch()['status'],'blocked_incomplete_completed_result'); self.runner.assert_not_called()

    def _completion_fixture(self):
        run=self.project/'runs'/EXPERIMENTS['tood_td']; (run/'weights').mkdir(parents=True)
        report=self.report
        for name in ('best.pt','last.pt'):
            with zipfile.ZipFile(run/'weights'/name,'w',compression=zipfile.ZIP_STORED) as archive:
                archive.writestr('fixture/data',b'x'*(1024*1024+1))
        with (run/'results.csv').open('w',newline='') as handle:
            writer=csv.writer(handle); writer.writerow(['epoch']); writer.writerows([[i] for i in range(1,201)])
        best_sha=safety.sha(run/'weights/best.pt')
        safety.write(report/'training_evidence.json',dict(status='completed',completed_epochs=200,best_sha256=best_sha))
        safety.write(report/'completion.json',dict(status='completed',trained_epochs=200,
                     official_val_completed=True,official_test_completed=True,checkpoint_sha256=best_sha))
        for split,count in (('val',548),('test',1610)):
            folder=report/split; folder.mkdir()
            (folder/'area_predictions.json').write_text('[]')
            pred_sha=safety.sha(folder/'area_predictions.json')
            result=dict(status='completed',binding=dict(seed=1,checkpoint_sha256=best_sha,
                        dataset={'images':count}),prediction_sha256=pred_sha)
            safety.write(folder/'official_metrics.json',result)
            safety.write(folder/'prediction_manifest.json',dict(seen_images=count,sha256=pred_sha))
        (report/'REPORT.md').write_text('test'); safety.write(report/'summary.json',{'status':'completed'})
        return run

    def test_real_completion_file_checks_and_crc(self):
        run=self._completion_fixture()
        self.assertEqual(after.verify_completed(self.project,self.report),[])
        self.update(status='completed')
        self.assertEqual(self.run_watch()['status'],'shutdown_command_returned_zero')
        (run/'weights/last.pt').write_bytes(b'truncated')
        self.assertTrue(after.verify_completed(self.project,self.report))

    def test_val_alone_never_counts_as_whole_pipeline(self):
        self._completion_fixture()
        (self.report/'test/official_metrics.json').unlink()
        self.update(status='completed')
        self.assertEqual(self.run_watch()['status'],'blocked_incomplete_completed_result')
        self.runner.assert_not_called()

    def test_changed_prediction_file_blocks_shutdown(self):
        self._completion_fixture()
        (self.report/'test/area_predictions.json').write_text('[{}]')
        self.update(status='completed')
        self.assertEqual(self.run_watch()['status'],'blocked_incomplete_completed_result')
        self.runner.assert_not_called()


if __name__=='__main__': unittest.main()
