"""Control-flow tests are CPU/stdlib only; ALL jobs/poweroff probes are mocked."""
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import unittest
from unittest import mock
import controller
import shutdown_safety


class PipelineTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[2] / 'audit/TOOD_TD_test_scratch'
        scratch.mkdir(parents=True,exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(dir=str(scratch))
        self.project = Path(self.scratch.name) / 'project'
        self.project.mkdir()

    def tearDown(self):
        self.scratch.cleanup()

    def invoke(self, fail=None, auto=True):
        jobs = []; helpers = []
        project = self.project / ('case_' + str(len(list(self.project.iterdir()))))
        project.mkdir()
        class Child:
            pid = 10000
            stdout = io.StringIO('fixture stage output\n')
            def wait(self,**unused):
                return 1 if fail == self.stage else 0
            def poll(self):
                return self.wait()
        def launch(command,**kwargs):
            if 'shutdown_after.py' in command[2]:
                helpers.append(command)
                return SimpleNamespace(pid=11000)
            stage = command[command.index('--mode')+1]
            jobs.append(stage)
            out = Path(command[command.index('--report-dir')+1])
            out.mkdir(parents=True,exist_ok=True)
            if stage == 'summary':
                out.parent.joinpath('REPORT.md').write_text('fixture report')
            child = Child(); child.stage = stage; child.stdout = io.StringIO('fixture ' + stage + '\n')
            return child
        fcntl = SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=mock.Mock())
        argv = ['controller.py','all','--project',str(project)] + (['--auto-shutdown'] if auto else [])
        with mock.patch.object(sys,'argv',argv), mock.patch.object(sys,'platform','linux'), \
             mock.patch.dict(sys.modules,{'fcntl':fcntl}), mock.patch.object(controller.signal,'signal'), \
             mock.patch.object(controller,'gpu_snapshot',return_value={'total_MiB':32760,'free_MiB':32230,'compute_pids':[]}), \
             mock.patch.object(controller,'process_snapshot',return_value={'jobs':[],'ignored_services':[]}), \
             mock.patch.object(controller,'process_identity',return_value={'pid':123,'start_ticks':1}), \
             mock.patch.object(controller,'shutdown_invocation',return_value={'sha256':'fake','args':['/bin/bash','fake']}), \
             mock.patch.object(controller.subprocess,'Popen',side_effect=launch), \
             mock.patch('shutdown_after.verify_completed',return_value=[]):
            code = controller.main()
        index = json.loads((project / 'logs/latest_tood_td_controller.json').read_text())
        state = json.loads((Path(index['session']) / 'state.json').read_text())
        return code,jobs,helpers,state

    def test_whole_order_includes_test_before_summary_shutdown(self):
        code,jobs,helpers,state = self.invoke()
        self.assertEqual(code,0)
        self.assertEqual(jobs,['check','smoke','verify-gates','train','val','test','summary'])
        self.assertEqual(state['status'],'completed')
        self.assertEqual(len(helpers),1)
        self.assertEqual(state['shutdown']['status'],'queued_after_parent_exit')

    def test_each_failure_stops_and_saves_then_only_queues_helper(self):
        for stage in ('check','smoke','verify-gates','train','val','test','summary'):
            with self.subTest(stage=stage):
                code,jobs,helpers,state = self.invoke(fail=stage)
                self.assertEqual(code,1)
                self.assertEqual(jobs[-1],stage)
                self.assertEqual(state['status'],'failed')
                self.assertEqual(state['stage_exit_codes'][stage],1)
                self.assertEqual(len(helpers),1)  # actual shutdown still waits for parent + idle

    def test_power_off_is_opt_in_only(self):
        code,jobs,helpers,state = self.invoke(auto=False)
        self.assertEqual(code,0); self.assertEqual(helpers,[])
        self.assertFalse(state['shutdown']['enabled'])

    def test_explicit_launcher_passes_auto_shutdown(self):
        text = Path(__file__).with_name('start_all_auto_shutdown.sh').read_text()
        self.assertIn('all --auto-shutdown',text)

    def test_readonly_progress_not_ignored_for_other_package(self):
        package = Path(__file__).resolve().parent
        self.assertEqual(shutdown_safety.classify(['python','-u',str(package/'watch_progress.py')],package)['policy'],
                         'ignored_readonly_frontend')
        self.assertEqual(shutdown_safety.classify(['python','-u',str(package/'unknown.py')],package)['policy'],
                         'blocks_shutdown')


if __name__ == '__main__':
    unittest.main()
