"""Mocked process/barrier tests, not Linux GPU concurrency certification."""
import importlib.util,json,os,tempfile,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run_pair as controller
class Job:
    def __init__(self,pid,values): self.pid=pid;self.values=iter(values);self.last=None
    def poll(self):
        try: self.last=next(self.values)
        except StopIteration: pass
        return self.last
    def wait(self,timeout=None): return self.last
class ControllerTests(unittest.TestCase):
    def test_manifest_paths_portable(self):
        self.assertTrue(all('\\' not in key for key in controller.code_signature()))
    def test_failed_formal_job_does_not_kill_other(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            args=SimpleNamespace(project=root,source=root,data=root,dataset=root)
            jobs=[Job(123,[None,1,1]),Job(124,[None,None,0])]
            with patch.object(controller.subprocess,'Popen',side_effect=jobs),patch.object(controller,'gpu_free',return_value=12000),patch.object(controller.time,'sleep'),patch.object(controller.os,'killpg',create=True) as kill:
                with self.assertRaises(RuntimeError): controller.launch(args,'run',root)
                kill.assert_not_called()
            self.assertEqual(json.loads((root/'run_exits.json').read_text()),{'e50':1,'e51':0})
    def test_command_paths_isolated(self):
        args=SimpleNamespace(project=Path('/project'),source=Path('/source'),data=Path('/data'),dataset=Path('/val'))
        a,b=[controller.command(args,v,'run') for v in controller.VARIANTS]
        self.assertNotEqual(a[1],b[1]);self.assertEqual(a[2:],b[2:])
    def test_barrier_waits_for_both(self):
        path=controller.HERE/'e50/pair_barrier.py'
        spec=importlib.util.spec_from_file_location('pair_barrier_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary,patch.dict(os.environ,PAIR_SMOKE_DIR=temporary,PAIR_VARIANT='e50'):
            errors=[]
            def worker():
                try: module.synchronize('train_ready')
                except BaseException as e: errors.append(e)
            thread=threading.Thread(target=worker);thread.start()
            deadline=time.monotonic()+2
            while not (Path(temporary)/'e50_train_ready').exists() and time.monotonic()<deadline: time.sleep(.01)
            self.assertTrue(thread.is_alive())
            (Path(temporary)/'e51_train_ready').write_text('fixture')
            thread.join(2)
            self.assertFalse(thread.is_alive());self.assertFalse(errors)
if __name__=='__main__': unittest.main(verbosity=2)
