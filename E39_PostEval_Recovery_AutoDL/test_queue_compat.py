"""Local unit tests; never call a real GPU, training, or process kill."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import queue_after_e39 as q
import smoke_memory as sm


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def proc(root,pid,parent,ticks,script,mode='run'):
    d=root/str(pid); d.mkdir()
    fields=['S',str(parent)]+['0']*17+[str(ticks)]+['0']*5
    (d/'stat').write_text(f'{pid} (worker with spaces) '+' '.join(fields))
    (d/'cmdline').write_bytes(b'python\0'+str(script).encode()+b'\0--mode\0'+mode.encode()+b'\0')


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name); self.project=self.root/'project'
        self.package=self.root/'E39'; self.package.mkdir()
        (self.package/'.e39_queue.lock').touch()
        self.report=self.project/'comparison_reports/E39_RUN_20260929_000000'
        self.best=self.project/'runs'/q.E39_NAME/'weights/best.pt'
        self.best.parent.mkdir(parents=True); self.best.write_bytes(b'test-only checkpoint bytes')
        self.training=dict(status='completed',completed_epochs=200,best=str(self.best),
                           params=dict(name=q.E39_NAME,epochs=200,imgsz=640,seed=1))
        self.metrics=dict(AP50_95_percent=25.,AP50_percent=45.,AP75_percent=24.,AR500_percent=37.)
        self.comparison=dict(status='completed',candidate=self.metrics,delta_pp=self.metrics,
                             seed=1,imgsz=640,checkpoint=str(self.best),dataset=dict(images=548),
                             checkpoint_sha256=hashlib.sha256(self.best.read_bytes()).hexdigest())
        dump(self.report/'training.json',self.training); dump(self.report/'comparison.json',self.comparison)
        (self.report/'REPORT.md').write_text('test')
        dump(self.report/'area_predictions.json',[]); dump(self.report/'gt_rows.json',[])

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_receipt(self):
        self.assertTrue(q.e39_receipt(self.report,self.project)['complete'])

    def test_failure_prevents_start(self):
        dump(self.report/'failure.json',dict(error='failed'))
        with self.assertRaisesRegex(RuntimeError,'failed'): q.e39_receipt(self.report,self.project)

    def test_early_stop_prevents_start(self):
        self.training['completed_epochs']=120; dump(self.report/'training.json',self.training)
        with self.assertRaisesRegex(RuntimeError,'before 200'): q.e39_receipt(self.report,self.project)

    def test_eval_still_running_waits(self):
        (self.report/'comparison.json').unlink()
        self.assertFalse(q.e39_receipt(self.report,self.project)['complete'])

    def test_wrong_artifact_or_split_prevents_start(self):
        self.best.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'changed'): q.e39_receipt(self.report,self.project)
        self.comparison['checkpoint_sha256']=hashlib.sha256(self.best.read_bytes()).hexdigest()
        self.comparison['dataset']['images']=8; dump(self.report/'comparison.json',self.comparison)
        with self.assertRaisesRegex(RuntimeError,'548'): q.e39_receipt(self.report,self.project)

    def test_nonfinite_metric_prevents_start(self):
        self.comparison['candidate']['AP75_percent']=float('nan')
        dump(self.report/'comparison.json',self.comparison)
        with self.assertRaisesRegex(RuntimeError,'nonfinite'): q.e39_receipt(self.report,self.project)

    def test_missing_artifact_prevents_start(self):
        (self.report/'gt_rows.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'artifact'): q.e39_receipt(self.report,self.project)

    def test_multiple_reports_require_explicit_selection(self):
        other=self.project/'comparison_reports/E39_RUN_20260929_010000'
        dump(other/'training.json',self.training)
        with self.assertRaisesRegex(RuntimeError,'Multiple'): q.choose_report(self.project)
        self.assertEqual(q.choose_report(self.project,self.report),self.report.resolve())

    def test_workers_are_not_independent_jobs(self):
        proot=self.root/'proc'; proot.mkdir(); script=self.package/'run_e39.py'
        proc(proot,100,99,500,script)
        for i in range(101,125): proc(proot,i,100,501+i,script)
        roots=q.find_e39_processes(self.package,self.project,proot)
        self.assertEqual([p['pid'] for p in roots],[100])
        proc(proot,150,99,800,script)
        self.assertEqual([p['pid'] for p in q.find_e39_processes(self.package,self.project,proot)],[100,150])

    def test_check_process_ignored_and_pid_reuse_detected(self):
        proot=self.root/'proc'; proot.mkdir(); script=self.package/'run_e39.py'
        proc(proot,200,1,500,script,'check')
        self.assertEqual(q.find_e39_processes(self.package,self.project,proot),[])
        first=q.process_identity(200,proot)
        stat=proot/'200/stat'
        stat.write_text(stat.read_text().replace('500','600'))
        self.assertNotEqual(first,q.process_identity(200,proot))
        stat.write_text(stat.read_text().replace(') S ',') Z '))
        self.assertIsNone(q.process_identity(200,proot))

    def test_queue_stable_idle_then_one_launch_and_gate_recheck(self):
        dest=self.root/'E40'; dest.mkdir()
        args=argparse.Namespace(project=self.project,source=self.root/'src',data=self.root/'data.yaml',
            dataset=self.root/'dataset',e39_package=self.package,e39_report=self.report,
            poll_seconds=15,timeout_hours=1.)
        fcntl=types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,LOCK_UN=4,flock=lambda *x:None)
        gpu=[dict(idle=False,compute_pids=[999],free_GiB=20.),
             dict(idle=True,compute_pids=[],free_GiB=30.),dict(idle=True,compute_pids=[],free_GiB=30.)]
        child=types.SimpleNamespace(pid=888,wait=lambda:0)
        with patch.object(q,'HERE',dest),patch.object(q,'find_e39_processes',return_value=[]),\
             patch.object(q,'gpu_idle',side_effect=gpu) as idle,\
             patch.object(q.subprocess,'run') as verify,patch.object(q.subprocess,'Popen',return_value=child) as launch,\
             patch.object(q.time,'sleep'):
            q._queue(args,fcntl)
        self.assertEqual(idle.call_count,3)
        self.assertEqual(verify.call_count,2)
        launch.assert_called_once()
        self.assertEqual(launch.call_args.args[0][:3],['bash',str(dest/'run_e40.sh'),'run'])
        state=json.loads(next((self.project/'logs').glob('queue_e40_*/state.json')).read_text())
        self.assertEqual(state['status'],'completed')

    def test_no_live_process_with_incomplete_receipt_blocks(self):
        self.training.update(status='training',completed_epochs=80); dump(self.report/'training.json',self.training)
        dest=self.root/'E40'; dest.mkdir()
        args=argparse.Namespace(project=self.project,source=self.root/'src',data=self.root/'data.yaml',
            dataset=self.root/'dataset',e39_package=self.package,e39_report=self.report,
            poll_seconds=15,timeout_hours=1.)
        with patch.object(q,'HERE',dest),patch.object(q,'find_e39_processes',return_value=[]),\
             patch.object(q.subprocess,'run'),patch.object(q.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError,'No live'):
                q._queue(args,None)
            launch.assert_not_called()

    def test_missing_lock_waits_for_all_completion_guards(self):
        for complete,roots,alive,controllers,unlocked in (
            (False,[],False,False,True),
            (True,[{'pid':100}],False,False,True),
            (True,[],True,False,True),
            (True,[],False,True,True),
            (True,[],False,False,False),
        ):
            self.assertFalse(q.ready_to_start(complete,roots,alive,controllers,unlocked))
        self.assertTrue(q.ready_to_start(True,[],False,False,True))

    def test_no_lock_completed_run_launches_once_without_creating_fake_lock(self):
        lock=self.package/'.e39_queue.lock'; lock.unlink()
        dest=self.root/'E40'; dest.mkdir()
        args=argparse.Namespace(project=self.project,source=self.root/'src',data=self.root/'data.yaml',
            dataset=self.root/'dataset',e39_package=self.package,e39_report=self.report,
            poll_seconds=15,timeout_hours=1.)
        fcntl=types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,LOCK_UN=4,flock=lambda *x:None)
        child=types.SimpleNamespace(pid=888,wait=lambda:0)
        with patch.object(q,'HERE',dest),patch.object(q,'find_e39_processes',return_value=[]),\
             patch.object(q,'gpu_idle',return_value=dict(idle=True,compute_pids=[],free_GiB=30.)) as idle,\
             patch.object(q.subprocess,'run') as verify,patch.object(q.subprocess,'Popen',return_value=child) as launch,\
             patch.object(q.time,'sleep'):
            q._queue(args,fcntl)
        self.assertEqual(idle.call_count,2); self.assertEqual(verify.call_count,2)
        launch.assert_called_once(); self.assertFalse(lock.exists())

    def test_controller_ancestor_binding_excludes_interactive_shell(self):
        proot=self.root/'proc'; proot.mkdir()
        proc(proot,10,1,100,Path('/bin/interactive_shell'),'run')
        proc(proot,20,10,200,Path('/different/package/queue_after_e38.py'),'run')
        proc(proot,30,20,300,self.package/'run_e39.py','run')
        self.assertEqual(q.controller_ancestors(dict(pid=30,start_ticks=300),proot),[dict(pid=20,start_ticks=200)])
        self.assertEqual(q.controller_ancestors(None,proot),[])

    def test_no_lock_incomplete_receipt_cannot_launch(self):
        (self.package/'.e39_queue.lock').unlink()
        self.training.update(status='training',completed_epochs=120); dump(self.report/'training.json',self.training)
        dest=self.root/'E40'; dest.mkdir()
        args=argparse.Namespace(project=self.project,source=self.root/'src',data=self.root/'data.yaml',
            dataset=self.root/'dataset',e39_package=self.package,e39_report=self.report,
            poll_seconds=15,timeout_hours=1.)
        with patch.object(q,'HERE',dest),patch.object(q,'find_e39_processes',return_value=[]),\
             patch.object(q.subprocess,'run'),patch.object(q.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError,'No live'): q._queue(args,None)
            launch.assert_not_called()

    def test_gpu_response_does_not_assume_idle(self):
        responses=[types.SimpleNamespace(returncode=0,stdout='123\n',stderr=''),
                   types.SimpleNamespace(returncode=0,stdout='30000\n',stderr='')]
        with patch.object(q.subprocess,'run',side_effect=responses): self.assertFalse(q.gpu_idle()['idle'])
        with patch.object(q.subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout='[Not Supported]',stderr='')):
            with self.assertRaisesRegex(RuntimeError,'Unexpected'): q.gpu_idle()

    def test_smoke_budget_and_restoration(self):
        values=dict(free=23*sm.GIB,fraction=1.)
        cuda=types.SimpleNamespace(mem_get_info=lambda device:(values['free'],32*sm.GIB),
            get_per_process_memory_fraction=lambda device:values['fraction'],
            set_per_process_memory_fraction=lambda fraction,device:values.update(fraction=fraction))
        with self.assertRaisesRegex(ValueError,'intentional'):
            with sm.bounded_smoke(cuda,'cuda:0'):
                self.assertEqual(values['fraction'],14/32)
                raise ValueError('intentional')
        self.assertEqual(values['fraction'],1.)
        values['free']=20*sm.GIB
        with self.assertRaisesRegex(RuntimeError,'20.5'): 
            with sm.bounded_smoke(cuda,'cuda:0'): self.fail('must not start')
        values['free']=5*sm.GIB
        with self.assertRaisesRegex(RuntimeError,'headroom'): sm.check_headroom(cuda,'cuda:0')


if __name__=='__main__':
    unittest.main(verbosity=2)
