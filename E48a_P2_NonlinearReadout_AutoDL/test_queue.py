"""Failure-tolerant queue tests; mocked processes/GPU, no real launch or kill."""
import argparse
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import queue_after_e47 as q
import smoke_memory as sm


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def process(root, pid, ppid, ticks, script, mode='run', state='S'):
    path = root/str(pid); path.mkdir()
    fields = [state, str(ppid)] + ['0']*17 + [str(ticks)] + ['0']*5
    (path/'stat').write_text(f'{pid} (worker with spaces) ' + ' '.join(fields))
    (path/'cmdline').write_bytes(b'python\0'+str(script).encode()+b'\0--mode\0'+mode.encode()+b'\0')


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.project = self.root/'project'
        self.package = self.root/'E48'; self.package.mkdir()
        self.e47 = self.root/'E47'; self.e47.mkdir()
        self.args = argparse.Namespace(project=self.project, source=self.root/'src',
            data=self.root/'data.yaml', dataset=self.root/'dataset', e47_package=self.e47,
            poll_seconds=15, timeout_hours=1.)
        self.fcntl = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *x: None)
        self.idle = dict(idle=True, compute_pids=[], free_GiB=30.)

    def tearDown(self):
        self.tmp.cleanup()

    def simulate(self, roots=None, gpu=None, code=0):
        child = types.SimpleNamespace(pid=888, wait=lambda: code)
        with patch.object(q, 'HERE', self.package), \
             patch.object(q, 'e47_roots', side_effect=roots or [[], [], []]), \
             patch.object(q, 'gpu_status', side_effect=gpu or [self.idle]*3), \
             patch.object(q.subprocess, 'run') as verify, \
             patch.object(q.subprocess, 'Popen', return_value=child) as launch, \
             patch.object(q.time, 'sleep'):
            q.queue(self.args, self.fcntl)
        return verify, launch

    def test_workers_do_not_count_as_independent_jobs(self):
        proc = self.root/'proc'; proc.mkdir()
        script = self.e47/'run_e47.py'
        process(proc, 100, 99, 500, script)
        for i in range(101, 125):
            process(proc, i, 100, 600+i, script)
        self.assertEqual([p['pid'] for p in q.e47_roots(self.e47,self.project,proc)], [100])
        process(proc, 200, 1, 1000, script)
        self.assertEqual([p['pid'] for p in q.e47_roots(self.e47,self.project,proc)], [100,200])

    def test_check_ignored_eval_waited_zombie_ignored_pid_reuse(self):
        proc = self.root/'proc'; proc.mkdir(); script = self.e47/'run_e47.py'
        process(proc, 1, 0, 100, script, 'check')
        process(proc, 2, 0, 200, script, 'eval')
        process(proc, 3, 0, 300, script, state='Z')
        self.assertEqual([p['pid'] for p in q.e47_roots(self.e47,self.project,proc)], [2])
        old = q.identity(2, proc)
        stat = proc/'2/stat'; stat.write_text(stat.read_text().replace('200','201'))
        self.assertNotEqual(old, q.identity(2, proc))

    def test_all_steps_controller_excludes_child_and_forked_workers(self):
        proc=self.root/'proc'; proc.mkdir()
        process(proc,100,1,500,self.e47/'all_steps.py')
        process(proc,101,100,510,self.e47/'run_e47.py')
        for pid in range(102,125): process(proc,pid,101,520+pid,self.e47/'run_e47.py')
        found=q.e47_roots(self.e47,self.project,proc)
        self.assertEqual([v['pid'] for v in found],[100])
        self.assertEqual(found[0]['mode'],'all')

    def test_orphaned_worker_is_still_waited(self):
        proc=self.root/'proc'; proc.mkdir()
        process(proc,101,1,600,self.e47/'run_e47.py')
        self.assertEqual([v['pid'] for v in q.e47_roots(self.e47,self.project,proc)],[101])

    def test_e47_failure_still_launches_once_after_gpu_release(self):
        report = self.project/'comparison_reports/E47S2_RUN_test'
        dump(report/'failure.json', dict(error='evaluation import failed'))
        dump(report/'training.json', dict(status='completed',completed_epochs=200))
        busy = dict(idle=False,compute_pids=[100],free_GiB=20.)
        verify, launch = self.simulate(roots=[[dict(pid=100)],[],[],[],[]],
            gpu=[busy,self.idle,self.idle,self.idle])
        self.assertEqual(verify.call_count, 2)
        self.assertTrue(all('--mode' in c.args[0] and 'verify-gates' in c.args[0] for c in verify.call_args_list))
        launch.assert_called_once()
        state = json.loads(next((self.project/'logs').glob('queue_e48_*/state.json')).read_text())
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(state['e47']['status'], 'failed')
        self.assertIn('evaluation import failed', state['e47']['error'])

    def test_e47_training_failure_also_does_not_block(self):
        dump(self.project/'comparison_reports/E47S2_RUN_test/failure.json', dict(error='CUDA OOM'))
        dump(self.project/'comparison_reports/E47S2_RUN_test/training.json', dict(status='training',completed_epochs=5))
        self.simulate()

    def test_absent_report_and_absent_old_lock_allow_idle_start(self):
        verify, launch = self.simulate()
        launch.assert_called_once()
        self.assertFalse((self.project/'.e47_seed2_execution.lock').exists())

    def test_stale_gate_prevents_monitor_and_training(self):
        with patch.object(q,'HERE',self.package), \
             patch.object(q.subprocess,'run',side_effect=RuntimeError('stale gate')), \
             patch.object(q.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError,'stale gate'):
                q.queue(self.args,self.fcntl)
            launch.assert_not_called()

    def test_existing_run_never_overwritten(self):
        (self.project/'runs'/q.EXPERIMENT).mkdir(parents=True)
        with patch.object(q,'HERE',self.package),patch.object(q.subprocess,'run'),patch.object(q.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError,'already exists'):
                q.queue(self.args,self.fcntl)
            launch.assert_not_called()

    def test_unknown_gpu_response_never_assumed_idle(self):
        result=types.SimpleNamespace(returncode=0,stdout='[Not Supported]',stderr='')
        with patch.object(q.subprocess,'run',return_value=result):
            with self.assertRaisesRegex(RuntimeError,'Cannot determine'):
                q.gpu_status()

    def test_idle_memory_guard_matches_formal(self):
        empty=types.SimpleNamespace(returncode=0,stdout='',stderr='')
        low=types.SimpleNamespace(returncode=0,stdout=str(29*1024),stderr='')
        high=types.SimpleNamespace(returncode=0,stdout=str(31*1024),stderr='')
        with patch.object(q.subprocess,'run',side_effect=[empty,low]):
            self.assertFalse(q.gpu_status()['idle'])
        with patch.object(q.subprocess,'run',side_effect=[empty,high]):
            self.assertTrue(q.gpu_status()['idle'])
        nan=types.SimpleNamespace(returncode=0,stdout='nan',stderr='')
        with patch.object(q.subprocess,'run',side_effect=[empty,nan]):
            with self.assertRaisesRegex(RuntimeError,'nonfinite'): q.gpu_status()

    def test_transient_gpu_query_failure_waits_and_recovers(self):
        child=types.SimpleNamespace(pid=888,wait=lambda:0)
        with patch.object(q,'HERE',self.package),patch.object(q,'e47_roots',return_value=[]), \
             patch.object(q,'gpu_status',side_effect=[RuntimeError('temporary query failure'),self.idle,self.idle,self.idle]), \
             patch.object(q.subprocess,'run'),patch.object(q.subprocess,'Popen',return_value=child) as launch, \
             patch.object(q.time,'sleep'):
            q.queue(self.args,self.fcntl)
        launch.assert_called_once()

    def test_new_job_between_final_checks_keeps_waiting(self):
        verify, launch = self.simulate(roots=[[],[],[dict(pid=99)],[],[],[]],
            gpu=[self.idle]*5)
        self.assertEqual(verify.call_count, 3)
        launch.assert_called_once()

    def test_old_execution_lock_busy_keeps_waiting(self):
        self.project.mkdir(parents=True,exist_ok=True)
        (self.project/'.e47_seed2_execution.lock').touch()
        calls=[0]
        def flock(*args):
            calls[0]+=1
            if calls[0]==1:
                raise BlockingIOError('E47 controller still holds lock')
        lock_api=types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=flock)
        with patch.object(q,'HERE',self.package),patch.object(q,'e47_roots',return_value=[]), \
             patch.object(q,'gpu_status',return_value=self.idle),patch.object(q.subprocess,'run'), \
             patch.object(q.subprocess,'Popen',return_value=types.SimpleNamespace(pid=9,wait=lambda:0)) as launch, \
             patch.object(q.time,'sleep'):
            q.queue(self.args,lock_api)
        self.assertEqual(calls[0],2)
        launch.assert_called_once()

    def test_gate_changed_while_waiting_prevents_launch(self):
        with patch.object(q,'HERE',self.package),patch.object(q,'e47_roots',return_value=[]), \
             patch.object(q,'gpu_status',return_value=self.idle), \
             patch.object(q.subprocess,'run',side_effect=[None,RuntimeError('gate changed')]), \
             patch.object(q.subprocess,'Popen') as launch,patch.object(q.time,'sleep'):
            with self.assertRaisesRegex(RuntimeError,'gate changed'):
                q.queue(self.args,self.fcntl)
            launch.assert_not_called()

    def test_e48_failure_reported_without_second_launch(self):
        with self.assertRaisesRegex(RuntimeError,'E48 exited 1'):
            self.simulate(code=1)
        state=json.loads(next((self.project/'logs').glob('queue_e48_*/state.json')).read_text())
        self.assertEqual(state['status'],'failed')
        self.assertEqual(state['exit_code'],1)

    def test_timeout_never_launches(self):
        with patch.object(q,'HERE',self.package),patch.object(q.subprocess,'run'), \
             patch.object(q,'e47_roots',return_value=[dict(pid=1)]), \
             patch.object(q.time,'monotonic',side_effect=[0.,0.,0.,4000.]), \
             patch.object(q.subprocess,'Popen') as launch:
            with self.assertRaises(TimeoutError):
                q.queue(self.args,self.fcntl)
            launch.assert_not_called()

    def test_smoke_cap_matches_formal_and_busy_entry_guard(self):
        values=dict(free=19*sm.GIB,fraction=1.)
        cuda=types.SimpleNamespace(mem_get_info=lambda device:(values['free'],32*sm.GIB),
            get_per_process_memory_fraction=lambda device:values['fraction'],
            set_per_process_memory_fraction=lambda fraction,device:values.update(fraction=fraction))
        with self.assertRaises(ValueError):
            with sm.bounded_smoke(cuda,'cuda:0',allow_busy=True):
                self.assertEqual(values['fraction'],13/32)
                raise ValueError('fixture')
        self.assertEqual(values['fraction'],13/32)
        values['free']=16*sm.GIB
        with self.assertRaisesRegex(RuntimeError,'16.5'):
            with sm.bounded_smoke(cuda,'cuda:0',allow_busy=True):
                self.fail('must not allocate')
        values['free']=1*sm.GIB
        with self.assertRaisesRegex(RuntimeError,'headroom'):
            sm.check_headroom(cuda,'cuda:0')


if __name__ == '__main__':
    unittest.main(verbosity=2)
