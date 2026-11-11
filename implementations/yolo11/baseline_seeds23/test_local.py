"""Local regression tests, CPU only. Shutdown is ALWAYS mocked."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import common as c
ROOT=c.HERE.parent
c.SOURCE=ROOT/'yolo11_src'
c.PROJECT=ROOT/'yolo11_project'
TEST_TMP=ROOT/'tmp/e0_s23_pair_local'
TEST_TMP.mkdir(parents=True,exist_ok=True)
tempfile.tempdir=str(TEST_TMP)
os.environ['YOLO_CONFIG_DIR']=str(TEST_TMP/'settings')
os.environ['YOLO_AUTOINSTALL']='false'
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
sys.path.insert(0,str(c.SOURCE))
import torch
torch.set_num_threads(2)
import evaluate_all as ev
import train_e0
import shutdown_safety as shutdown


class Tests(unittest.TestCase):
    def test_package_original_code_and_protocol(self):
        c.package_audit();c.protocol_audit()
        original=ROOT/'yolo11_project/comparison_reports/baseline_yolo11s_img640_seed1_20260926_073621_610152/yolo11s_img640_seed1.py'
        self.assertEqual(c.sha(original,True),c.sha(c.HERE/'e0_core.py',True))
        for seed in (2,3):
            self.assertEqual(c.params(seed)['seed'],seed)
            self.assertEqual(c.params(seed)['batch'],16)
            self.assertEqual(c.params(seed)['epochs'],200)
            self.assertFalse(c.params(seed)['resume'])

    def test_e0_real_original_load_fuse_fresh_process(self):
        with tempfile.TemporaryDirectory() as out:
            value=train_e0.checkpoint_unit(torch,Path(out))
        self.assertTrue(value['original_detect'])

    def test_e60_actual_three_seed_code_and_checkpoint(self):
        import spatial_task_head
        from ultralytics import YOLO
        spatial_task_head.configure('e60')
        for seed in (1,2,3):
            identity=c.e60_identity(seed)
            model=YOLO(identity['weights'],task='detect')
            value=ev.e60.model_audit(model,ev.e60.KEY)
            self.assertTrue(value['fp32_sidecars'])

    def test_original_prediction_units_and_cached_provenance(self):
        ev.base.units()
        for key in ('e0_seed1','e60_seed1','e1_seed1','e1_seed2','e1_seed3'):
            result=ev.cached(key,ev.base.protocol('cpu'))
            self.assertEqual(result['status'],'completed')
        # A reference mismatch cannot silently turn into a new score or gate.
        real=c.read
        def changed(path):
            value=real(path)
            if Path(path).name=='e1_seed2_testdev_metrics.json':
                value['binding']['image_count']=8
            return value
        with patch.object(c,'read',side_effect=changed):
            with self.assertRaises(RuntimeError):ev.cached('e1_seed2',ev.base.protocol('cpu'))

    def test_summary_paired_deltas_no_seed_dropping(self):
        models={}
        for kind,values in (('e0',(10,11,12)),('e1',(12,13,14)),('e60',(13,12,15))):
            for seed,value in enumerate(values,1):
                models['%s_seed%d'%(kind,seed)]=dict(official={m:value for m in ev.KEYS})
        result=ev.summarize(models)
        delta=result['paired_deltas']['e60_minus_e1'][ev.KEYS[0]]
        self.assertEqual(delta['values_pp'],[1,-1,1])
        self.assertAlmostEqual(delta['mean_pp'],1/3)
        self.assertAlmostEqual(result['statistics']['e0'][ev.KEYS[0]]['sample_std'],1)
        del models['e60_seed3']
        with self.assertRaises(KeyError):ev.summarize(models)

    def test_controller_order_parallel_and_serial(self):
        import controller
        calls=[]
        with tempfile.TemporaryDirectory() as out:
            session=Path(out)
            mocks=dict(package_audit=unittest.mock.DEFAULT,protocol_audit=unittest.mock.DEFAULT,
                       completed_e0=unittest.mock.DEFAULT,e60_identity=unittest.mock.DEFAULT)
            with patch.multiple(c,**mocks),patch.object(controller,'admission'),\
                 patch.object(controller,'parallel',side_effect=lambda mode,*_:calls.append(('parallel',mode))),\
                 patch.object(controller,'serial',side_effect=lambda script,mode,*a: calls.append(('serial',script,mode)) or session):
                controller.run(SimpleNamespace(mode='eval'),session,{})
            self.assertEqual(calls,[('serial','evaluate_all.py','run')])
        self.assertEqual(ev.ORDER,('e0_seed2','e0_seed3','e60_seed2','e60_seed3'))

    def test_actual_serial_predictor_matches_independent_standard(self):
        from PIL import Image
        import numpy as np
        from ultralytics import YOLO
        import spatial_task_head
        spatial_task_head.configure('e60')
        args=SimpleNamespace(device='cpu')
        old_options=ev.base.options
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);records=[]
            for index in range(1,7):
                w,h=(131+index,77+index) if index<5 else (88,123)
                path=out/('%02d.jpg'%index)
                rng=np.random.default_rng(index)
                Image.fromarray(rng.integers(0,255,(h,w,3),dtype=np.uint8)).save(path)
                records.append(SimpleNamespace(image_id=index,width=w,height=h,image_path=path))
            for kind in ('e0','e60'):
                key=kind+'_seed1'
                weight=(c.PROJECT/'runs/baseline_yolo11s_img640_seed1/weights/best.pt' if kind=='e0'
                        else Path(c.e60_identity(1)['weights']))
                ev.base.SPECS={key:dict(nl=3 if kind=='e0' else 4)}
                def audit(model,unused):
                    return train_e0.audit_model(model.model) if kind=='e0' else ev.e60.model_audit(model,ev.e60.KEY)
                with patch.object(ev.base,'options',side_effect=lambda device:dict(old_options(device),imgsz=128)),\
                     patch.object(ev.base,'model_audit',side_effect=audit):
                    actual,_=ev.base.predict(args,key,weight,records,out/key)
                standard=YOLO(str(weight),task='detect');expected=[]
                for start in (0,4):
                    chunk=records[start:start+4]
                    result=standard.predict([str(r.image_path) for r in chunk],batch=len(chunk),
                                            **dict(old_options('cpu'),imgsz=128))
                    for record,pred in zip(chunk,result):
                        expected.extend(ev.base.convert_detections(record,pred.boxes.data.tolist()))
                self.assertEqual(actual,expected)

    def test_original_formal_train_delegation_not_short_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            core_mock=SimpleNamespace(EXPERIMENT='baseline_yolo11s_img640_seed2',YOLO=object())
            def train(report):
                report['training']=dict(best_checkpoint_epoch=87)
                report['run_dir']=str(root/'run')
                return root/'best.pt'
            core_mock.train=unittest.mock.Mock(side_effect=train)
            binding={'fixed_schedule':200}
            with patch.object(c,'PROJECT',root),patch.object(c,'verify'),\
                 patch.object(c,'read',return_value={'status':'passed','signatures':{'2':c.digest(binding)}}),\
                 patch.object(c,'sha',return_value='fake'),patch.object(c,'write') as writer,patch.object(c,'cap'):
                train_e0.formal(torch,core_mock,2,root/'report',binding)
            core_mock.train.assert_called_once()
            completion=writer.call_args[0][1]
            self.assertEqual(completion['completed_epochs'],200)
            self.assertEqual(completion['params']['epochs'],200)
            self.assertEqual(completion['seed'],2)

    def test_failure_does_not_kill_successful_sibling(self):
        import controller
        first=SimpleNamespace(pid=101,poll=unittest.mock.Mock(side_effect=[1,1]))
        second=SimpleNamespace(pid=102,poll=unittest.mock.Mock(side_effect=[None,0]))
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(controller.subprocess,'Popen',side_effect=[first,second]),\
                 patch.object(controller,'gpu_snapshot',return_value={'free_MiB':32229}),\
                 patch.object(controller.time,'sleep'),patch.object(c,'write'):
                with self.assertRaises(RuntimeError):controller.parallel('run',Path(tmp),{})
        self.assertEqual(second.poll.call_count,2)
        self.assertFalse(hasattr(first,'terminate'))

    def test_gates_stale_and_no_complete200(self):
        with tempfile.TemporaryDirectory() as out,patch.object(c,'PROJECT',Path(out)):
            signature={'seed':2,'batch':16}
            c.write(c.gates(2,'check'),dict(status='passed',signature=c.digest(signature)))
            c.verify(2,'check',signature)
            with self.assertRaises(RuntimeError):c.verify(2,'check',dict(signature,batch=8))
            c.write(c.PROJECT/'comparison_reports/e0_s23_pair_gates/seed2_formal_completed.json',
                    dict(status='completed',completed_epochs=199))
            with self.assertRaises(RuntimeError):c.completed_e0(2)

    def test_shutdown_ascii_bash_and_busy_cancel(self):
        idle=dict(total_MiB=32229,free_MiB=32229,compute_pids=[])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);command=root/'shutdown'
            command.write_text('echo mock platform shutdown\n')
            runner=unittest.mock.Mock(return_value=SimpleNamespace(returncode=0,stdout='mock',stderr=''))
            result=shutdown.auto_shutdown(True,root,root/'KEEP',delay=0,probe=lambda:idle,jobs=lambda:[],
                       runner=runner,sleeper=lambda _:None,platform_name='linux',command=command)
            self.assertEqual(result['status'],'shutdown_command_returned_zero')
            self.assertEqual(runner.call_args[0][0],['/bin/bash',str(command)])
            runner.reset_mock()
            busy=dict(idle,compute_pids=[123])
            result=shutdown.auto_shutdown(True,root,root/'KEEP',delay=0,probe=lambda:busy,jobs=lambda:[],
                       runner=runner,sleeper=lambda _:None,platform_name='linux',command=command)
            self.assertEqual(result['status'],'skipped_safety_check_or_command_error')
            runner.assert_not_called()
            (root/'KEEP').touch()
            result=shutdown.auto_shutdown(True,root,root/'KEEP',delay=0,probe=lambda:idle,jobs=lambda:[],
                       runner=runner,sleeper=lambda _:None,platform_name='linux',command=command)
            self.assertEqual(result['status'],'cancelled_by_KEEP_RUNNING_file')


if __name__=='__main__':unittest.main(verbosity=2)
