"""CPU release regression tests; never train a server or issue CUDA receipts."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parent
BASE_NAME=ROOT.name[:-3] if ROOT.name.endswith('_v2') else ROOT.name


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class ReleaseTests(unittest.TestCase):
    def test_core_forward_unchanged_from_first_delivery(self):
        import zipfile
        archive=ROOT.parent/(BASE_NAME+'.zip')
        self.assertTrue(archive.is_file(),'Original delivery needed for local audit')
        with zipfile.ZipFile(archive) as z:
            old=ast.parse(z.read(BASE_NAME+'/extension.py').decode('utf-8'))
            new=ast.parse((ROOT/'extension.py').read_text(encoding='utf-8'))
            for name in ('linear_fp32','local_calibration','LocalCalibrationBlock',
                         'ResidualReuseBlock','install','branch_groups','oracle'):
                a=next(n for n in old.body if getattr(n,'name',None)==name)
                b=next(n for n in new.body if getattr(n,'name',None)==name)
                self.assertEqual(ast.dump(a),ast.dump(b),name+' changed training semantics')
            for variant in ('e44','e45'):
                for file in ('spatial_task_head.py','e1_640_core.py','p2_tal_chunked.py',
                             'evaluate.py','d27_official.py','d38_common.py','run_e40.py'):
                    before=z.read(BASE_NAME+'/'+variant+'/'+file).replace(b'\r\n',b'\n')
                    after=(ROOT/variant/file).read_bytes().replace(b'\r\n',b'\n')
                    self.assertEqual(before,after,variant+'/'+file)

    def test_gradient_budget_rejects_real_errors(self):
        code='''
import torch
from extension import compare_gradients
ref=torch.tensor([0.,1e-4,-2e-4],dtype=torch.float32)
rounded=ref.clone(); rounded[0]=2.91e-11
assert not compare_gradients(rounded,ref,'roundoff')['bitwise_equal']
for bad in (ref+1e-7, ref*1.001, torch.full_like(ref,float('nan'))):
    try: compare_gradients(bad,ref,'intentional corrupt gradient')
    except AssertionError: pass
    else: raise AssertionError('real error was accepted')
for actual,expected in ((torch.ones(3)*1e-30,torch.zeros(3)),(None,ref)):
    try: compare_gradients(actual,expected,'zero/missing gradient')
    except AssertionError: pass
    else: raise AssertionError('zero/missing error accepted')
print('roundoff accepted; sign/scale/missing/nonfinite errors rejected')
'''
        env=os.environ.copy()
        env['YOLO_CONFIG_DIR']=str(ROOT.parent/'tmp/e44_e45_test_settings')
        env['YOLO_AUTOINSTALL']='false'
        p=subprocess.run([sys.executable,'-c',code],cwd=ROOT/'e44',env=env,capture_output=True,text=True,timeout=90)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)

    def test_all_python_parses(self):
        for path in ROOT.rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'),filename=str(path))

    def test_pinned_sources_and_formal_protocol_unchanged(self):
        parent=ROOT.parent/'E40a_SpatialTaskHead_AutoDL'
        template=ROOT.parent/'E42E43_E40_JointDense_Parallel_AutoDL/e43/run_e40.py'
        before=ast.parse(template.read_text(encoding='utf-8'))
        for name in ('e44','e45'):
            for file in ('e1_640_core.py','e1_reference.yaml','p2_tal_chunked.py','d27_official.py','d38_common.py'):
                self.assertEqual((ROOT/name/file).read_bytes(),(parent/file).read_bytes())
            core=load(name+'_core',ROOT/name/'e1_640_core.py')
            params=core.training_params()
            self.assertEqual((params['epochs'],params['batch'],params['imgsz'],params['seed']), (200,16,640,1))
            self.assertEqual((params['close_mosaic'],params['patience'],params['workers']), (10,0,8))
            self.assertTrue(params['amp'])
            text=(ROOT/name/'run_e40.py').read_text(encoding='utf-8')
            after=ast.parse(text.replace(name.upper(),'E43').replace(name+'_','e43_'))
            for fn in ('config','build_model','trainer_class'):
                a=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name==fn)
                b=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name==fn)
                self.assertEqual(ast.dump(a),ast.dump(b),name+':'+fn)
            self.assertIn("status='evaluating'",text)
            self.assertIn("out/'completion.json'",text)
            self.assertIn('seen_loss_gradients',text)
            self.assertIn("get('validation_batch')!=32",text)
            self.assertIn("result.get('successful_optimizer_updates',0)<8",text)
            self.assertNotIn("channel_router.1.",(ROOT/name/'checks.py').read_text())

    def test_both_formulas_initialization_weak_precision_and_half(self):
        code='''
import torch
torch.set_num_threads(2)
from spatial_task_head import VARIANT,NEW_PARAMETERS
from extension import routing_audit,install,branch_groups
from precision_audit import precision_audit
from spatial_task_head import SpatialTaskBlock
from torch import nn
blocks=install(nn.ModuleList([SpatialTaskBlock(),SpatialTaskBlock()]),VARIANT)
assert sum(p.numel() for p in blocks.parameters())==NEW_PARAMETERS
groups=branch_groups(VARIANT)
names=[n for n,p in blocks.named_parameters()]
assert all(sum(n.startswith(g) for g in groups)==1 for n in names)
assert all(any(n.startswith(g) for n in names) for g in groups)
print(VARIANT,routing_audit('cpu',VARIANT),precision_audit('cpu',VARIANT),flush=True)
'''
        env=os.environ.copy(); env['PYTHONIOENCODING']='utf-8'
        env['YOLO_AUTOINSTALL']='false'
        env['YOLO_CONFIG_DIR']=str(ROOT.parent/'tmp/e44_e45_test_settings')
        for name in ('e44','e45'):
            p=subprocess.run([sys.executable,'-c',code],cwd=ROOT/name,env=env,
                             capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            print(name+': independent formula / all parameter gradients / precision / half passed',flush=True)

    def test_real_AMP_skip_requires_successful_updates(self):
        import torch
        progress_module=load('release_progress',ROOT/'e44/smoke_progress.py')
        checks=load('release_checks',ROOT/'e44/checks.py')
        progress=progress_module.SmokeProgress()
        model=torch.nn.Linear(4,1)
        opt=torch.optim.SGD(model.parameters(),lr=.001)
        scaler=torch.amp.GradScaler('cpu',enabled=True)
        for _ in range(32):
            with torch.autocast('cpu',dtype=torch.float16):
                loss=model(torch.ones(4,4)).float().square().mean()*1024
            scaler.scale(loss).backward()
            progress.observe(checks.amp_step(model,opt,scaler))
            if progress.ready: break
        progress.require_success()
        self.assertGreater(progress.skipped,0)
        self.assertGreaterEqual(progress.successful,8)
        bad=progress_module.SmokeProgress()
        for _ in range(32): bad.observe(dict(updated=False,status='overflow_skip',bad_parameter_count=1))
        with self.assertRaises(RuntimeError): bad.require_success()

    def test_pair_receipts_reject_mixed_or_failed_smokes(self):
        helper=load('pair_helper',ROOT/'pair_receipts.py')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); project=root/'project'; (project/'comparison_reports').mkdir(parents=True)
            barrier=root/'barrier'; barrier.mkdir()
            for name in ('e44','e45'):
                report=root/name; report.mkdir()
                (report/'smoke.json').write_text(json.dumps(dict(status='passed',validation=dict(parallel_pair=True,barrier_directory=str(barrier)))))
                (project/'comparison_reports'/f'{name}_smoke_gate.json').write_text(json.dumps(dict(report=str(report))))
                (barrier/(name+'.ready')).touch()
            state=barrier/'state.json'
            state.write_text(json.dumps(dict(mode='smoke',exit_codes={'e44':0,'e45':0})))
            self.assertEqual(helper.require_parallel_smoke(project),str(barrier))
            state.write_text(json.dumps(dict(mode='smoke',exit_codes={'e44':0,'e45':1})))
            with self.assertRaises(RuntimeError): helper.require_parallel_smoke(project)
            state.write_text(json.dumps(dict(mode='smoke',exit_codes={'e44':0,'e45':0})))
            other=root/'other'; other.mkdir()
            (root/'e45/smoke.json').write_text(json.dumps(dict(status='passed',validation=dict(parallel_pair=True,barrier_directory=str(other)))))
            with self.assertRaises(RuntimeError): helper.require_parallel_smoke(project)

    def test_controller_safety(self):
        text=(ROOT/'run_pair.py').read_text()
        self.assertNotIn('.kill(',text); self.assertNotIn('.terminate(',text)
        self.assertIn("all(v is not None for v in codes.values())",text)
        self.assertIn('require_parallel_smoke(args.project)',text)
        self.assertIn('Existing formal run; refusing overwrite',text)
        self.assertIn("(logs/'abort').touch()",text)
        self.assertIn("e44e45_latest_",text)
        for name in ('e44','e45'):
            self.assertIn('.'+name+'_execution.lock',(ROOT/name/'run.sh').read_text())
            self.assertIn('exec 9>"$PROJECT/',(ROOT/name/'run.sh').read_text())
            evaltext=(ROOT/name/'evaluate.py').read_text()
            self.assertIn("screening_passed=candidate['AP50_95_percent']>=26.44",evaltext)
            self.assertIn('delta_vs_E40_pp',evaltext)
            self.assertIn('preds[0].detach().clone()',evaltext)
            self.assertNotIn("'stream'",evaltext)

    def test_actual_child_failure_does_not_kill_peer(self):
        # Real disposable child processes, mocked GPU query/Linux lock import.
        # Does not claim actual CUDA concurrency or a Linux flock validation.
        with mock.patch.dict(sys.modules,{'fcntl':types.SimpleNamespace()}):
            controller=load('pair_controller_fixture',ROOT/'run_pair.py')
        controller.gpu=lambda: dict(total_MiB=32229,free_MiB=32229,pids=[])
        controller.progress=lambda *unused: 'fixture-only'
        controller.say=lambda *unused: None
        controller.command=lambda name,mode,args: [sys.executable,'-c',
            "import time,sys; print('child '+sys.argv[1],flush=True); time.sleep(0.1 if sys.argv[1]=='e44' else 0.4); sys.exit(1 if sys.argv[1]=='e44' else 0)",name]
        original_sleep=controller.time.sleep
        with tempfile.TemporaryDirectory() as temp:
            args=types.SimpleNamespace(project=Path(temp))
            with mock.patch.object(controller.time,'sleep',side_effect=lambda delay: original_sleep(.03)):
                self.assertEqual(controller.launch('run',args),1)
            latest=json.loads((args.project/'logs/e44e45_latest_run.json').read_text())
            directory=Path(latest['directory'])
            state=json.loads((directory/'state.json').read_text())
            self.assertEqual(state['exit_codes'],{'e44':1,'e45':0})
            self.assertFalse((directory/'abort').exists())
            self.assertIn('child e45',Path(latest['logs']['e45']).read_text())

    def test_archive_hashes_if_released(self):
        archive=ROOT.parent/(BASE_NAME+'_v2.zip')
        if not archive.exists(): return
        import zipfile
        with zipfile.ZipFile(archive) as z:
            self.assertIsNone(z.testzip())
            prefix=BASE_NAME+'_v2'
            manifest=json.loads(z.read(prefix+'/manifest.json'))
            for path,expected in manifest.items():
                self.assertEqual(hashlib.sha256(z.read(prefix+'/'+path)).hexdigest(),expected)


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    unittest.main(verbosity=2)
