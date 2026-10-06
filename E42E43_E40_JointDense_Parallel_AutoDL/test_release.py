"""Local safety/independent numeric checks. No server or dataset training."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import zipfile

ROOT=Path(__file__).resolve().parent
from smoke_progress import SmokeProgress,stress_batches


class ReleaseTests(unittest.TestCase):
    def test_reproduced_AMP_history_requires_more_real_batches(self):
        progress=SmokeProgress()
        # Both pasted real logs: skips on 1..5 and 7; updates on 6 and 8.
        for success in (False,False,False,False,False,True,False,True):
            progress.observe(dict(updated=success,status='finite_update' if success else 'overflow_skip',bad_parameter_count=0 if success else 2))
        self.assertEqual((progress.attempts,progress.successful,progress.skipped),(8,2,6))
        self.assertFalse(progress.ready)
        for _ in range(6):
            progress.observe(dict(updated=True,status='finite_update',bad_parameter_count=0))
        progress.require_success()
        self.assertEqual(progress.attempts,14)

    def test_persistent_overflow_stops_without_receipt(self):
        progress=SmokeProgress()
        for _,_ in stress_batches(range(809)):
            progress.observe(dict(updated=False,status='overflow_skip',bad_parameter_count=1))
        self.assertEqual(progress.attempts,32)
        with self.assertRaisesRegex(RuntimeError,'0/8 successful optimizer updates'):
            progress.require_success()

    def test_AMP_update_claim_requires_finite_gradient_evidence(self):
        with self.assertRaisesRegex(RuntimeError,'Invalid finite-update'):
            SmokeProgress().observe(dict(updated=True,status='finite_update',bad_parameter_count=1))

    def test_original_release_formal_training_and_common_model_unchanged(self):
        old=ROOT.parent/'E42E43_E40_JointDense_Parallel_AutoDL.zip'
        if not old.is_file(): return
        with zipfile.ZipFile(old) as archive:
            for name in ('e42','e43'):
                # E42's new channel MLP intentionally gets an FP32 precision
                # island; the matched E40/Dense topology and formal protocol do
                # not change. Check additions are also intentional.
                for file in ('spatial_task_head.py','e1_640_core.py','p2_tal_chunked.py','evaluate.py'):
                    self.assertEqual((ROOT/name/file).read_bytes().replace(b'\r\n',b'\n'),archive.read('E42E43_E40_JointDense_Parallel_AutoDL/'+name+'/'+file))
                before=ast.parse(archive.read('E42E43_E40_JointDense_Parallel_AutoDL/'+name+'/run_e40.py').decode())
                after=ast.parse((ROOT/name/'run_e40.py').read_text())
                for fn in ('config','build_model','trainer_class','train'):
                    a=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name==fn)
                    b=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name==fn)
                    self.assertEqual(ast.dump(a),ast.dump(b),fn)

    def test_weak_channel_gradients_match_FP32_under_AMP(self):
        code="""
import json,torch
torch.set_num_threads(2)
from precision_audit import channel_precision_audit
print(json.dumps(channel_precision_audit('cpu')))
if torch.cuda.is_available():
    print(json.dumps(channel_precision_audit('cuda:0')))
"""
        env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['YOLO_AUTOINSTALL']='false'
        env['YOLO_CONFIG_DIR']=str(ROOT.parent/'tmp/e42e43_test_settings')
        proc=subprocess.run([sys.executable,'-c',code],cwd=ROOT/'e42',capture_output=True,text=True,
                            encoding='utf-8',errors='replace',env=env,timeout=120)
        self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
        print(proc.stdout.strip())

    def test_v3_core_and_formal_training_unchanged(self):
        archive_path = ROOT.parent / (ROOT.name + '_v3.zip')
        self.assertTrue(archive_path.is_file())
        with zipfile.ZipFile(archive_path) as archive:
            for name in ('e42', 'e43'):
                for file in ('spatial_task_head.py', 'extension.py', 'precision_audit.py',
                             'amp_branch_audit.py', 'checks.py', 'p2_tal_chunked.py',
                             'e1_640_core.py', 'evaluate.py'):
                    self.assertEqual((ROOT/name/file).read_bytes().replace(b'\r\n',b'\n'),
                                     archive.read(ROOT.name+'/'+name+'/'+file))

    def test_validation_precision_and_failure_guards(self):
        # Full batch32/640 raw tensor count, original pinned E1 criterion.
        # No synthetic-module proxy for BCE reduction. Finite logits must not
        # be mistaken for nonfinite inference because a native FP16 sum fails.
        code = '''
import json, torch
from types import SimpleNamespace
from ultralytics.utils.loss import v8DetectionLoss
from validation_smoke import validation_loss, require_finite_predictions
torch.set_num_threads(2)
device='cuda:0' if torch.cuda.is_available() else 'cpu'
m=torch.nn.Module()
m.register_parameter('p',torch.nn.Parameter(torch.zeros(1,device=device)))
m.args=SimpleNamespace(box=7.5,cls=.5,dfl=1.5)
m.model=[SimpleNamespace(stride=torch.tensor([4,8,16,32],device=device),nc=10,reg_max=16)]
criterion=v8DetectionLoss(m)
class OriginalLoss:
    def loss(self,batch,pred): return criterion(pred,batch)
raw=[torch.zeros(32,74,h,h,device=device,dtype=torch.float16) for h in (160,80,40,20)]
for x in raw: x[:,64:].fill_(-4.)
decoded=torch.zeros(32,14,34000,device=device,dtype=torch.float16)
batch=dict(img=torch.empty(32,3,1,1,device=device,dtype=torch.float16),
           batch_idx=torch.empty(0,device=device),cls=torch.empty(0,1,device=device),
           bboxes=torch.empty(0,4,device=device))
pred=(decoded,raw)
_,native=criterion(pred,batch)
assert not torch.isfinite(native).all() and all(torch.isfinite(x).all() for x in raw)
total,items=validation_loss(OriginalLoss(),batch,pred,audit_native=True)
assert torch.isfinite(total) and torch.isfinite(items).all()
from torch.nn import functional as F
reference=sum(F.binary_cross_entropy_with_logits(x[:,64:].float(),torch.zeros_like(x[:,64:],dtype=torch.float32),reduction='sum') for x in raw)*.5
torch.testing.assert_close(items[1],reference,rtol=1e-6,atol=1e-3)
# Nonempty GTs also use the SAME original TAL/CIoU/DFL criterion.
batch.update(batch_idx=torch.arange(32,device=device,dtype=torch.float32),
             cls=torch.ones(32,1,device=device),
             bboxes=torch.tensor([[.5,.5,.1,.1]],device=device).repeat(32,1))
total,items=validation_loss(OriginalLoss(),batch,pred)
assert torch.isfinite(total) and torch.isfinite(items).all()
bad=decoded.clone(); bad[0,0,0]=float('nan')
try: require_finite_predictions((bad,raw))
except RuntimeError as e: assert 'decoded_boxes_or_scores' in str(e)
else: raise AssertionError('Nonfinite decoded output incorrectly passed')
raw[0][0,0,0,0]=float('inf')
try: require_finite_predictions(pred)
except RuntimeError as e: assert 'raw_P2_logits' in str(e)
else: raise AssertionError('Nonfinite raw logit incorrectly passed')
raw[0][0,0,0,0]=0
class NonfiniteLoss:
    def loss(self,batch,pred): return torch.tensor(float('inf'),device=device),torch.ones(3,device=device)
try: validation_loss(NonfiniteLoss(),batch,pred)
except RuntimeError as e: assert 'AMP_total_loss' in str(e)
else: raise AssertionError('Nonfinite AMP loss incorrectly passed')
print('Validation regression: batch32 finite raw + native FP16 inf reproduced; AMP original criterion finite; all failure guards reject')
'''
        env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['YOLO_AUTOINSTALL']='false'
        env['YOLO_CONFIG_DIR']=str(ROOT.parent/'tmp/e42e43_test_settings')
        proc=subprocess.run([sys.executable,'-c',code],cwd=ROOT/'e42',capture_output=True,text=True,
                            encoding='utf-8',errors='replace',env=env,timeout=120)
        self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
        print(proc.stdout.strip())

    def test_real_cpu_AMP_skip_then_successful_steps(self):
        import torch
        spec=importlib.util.spec_from_file_location('amp_checks',ROOT/'e42/checks.py')
        checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)
        torch.manual_seed(7)
        model=torch.nn.Linear(4,1)
        optimizer=torch.optim.SGD(model.parameters(),lr=.001)
        scaler=torch.amp.GradScaler('cpu',enabled=True)
        progress=SmokeProgress()
        for _,_ in stress_batches(range(100)):
            with torch.autocast('cpu',dtype=torch.float16):
                loss=model(torch.ones(4,4)).float().square().mean()*1024
            scaler.scale(loss).backward()
            update=checks.amp_step(model,optimizer,scaler)
            progress.observe(update)
            if progress.ready:break
        progress.require_success()
        self.assertGreater(progress.skipped,0)
        self.assertGreater(progress.attempts,8)

    def test_all_python_compiles(self):
        for p in ROOT.rglob('*.py'):
            ast.parse(p.read_text(encoding='utf-8'),filename=str(p))

    def test_fixed_protocol_and_independent_paths(self):
        for name in ('e42','e43'):
            spec=importlib.util.spec_from_file_location(name+'_runner',ROOT/name/'run_e40.py')
            runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
            spec=importlib.util.spec_from_file_location(name+'_core',ROOT/name/'e1_640_core.py')
            core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
            params=core.training_params()
            self.assertEqual((params['epochs'],params['batch'],params['imgsz'],params['seed']), (200,16,640,1))
            self.assertEqual((params['patience'],params['close_mosaic']), (0,10))
            self.assertTrue(params['amp'])
            self.assertIn(name,runner.EXPERIMENT)
            self.assertIn('seed1',runner.BASE_REPORT)
            text=(ROOT/name/'run_e40.py').read_text()
            self.assertIn("args.mode in ('smoke','run')",text)
            self.assertIn('seen_loss_gradients',text)
            self.assertIn("get('validation_batch')!=32",text)
            self.assertIn("get('successful_optimizer_updates',0)<8",text)
            self.assertIn('stress_batches(loader)',text)
            self.assertNotIn('for i,batch in smoke_batches(loader)',text)
            self.assertNotIn('queue_after_e39',text)

    def test_original_common_source_immutable(self):
        old=ROOT.parent/'E40a_SpatialTaskHead_AutoDL'
        if not old.is_dir(): return
        for name in ('e42','e43'):
            for file in ('e1_640_core.py','e1_reference.yaml','p2_tal_chunked.py','d27_official.py','d38_common.py'):
                self.assertEqual((ROOT/name/file).read_bytes(),(old/file).read_bytes())

    def test_half_blocks_and_finite_extreme_gradients(self):
        for name in ('e42','e43'):
            code="""
import copy,torch
torch.set_num_threads(2)
from torch import nn
from spatial_task_head import SpatialTaskBlock,VARIANT
from extension import install,joint_mix
blocks=install(nn.ModuleList([SpatialTaskBlock(),SpatialTaskBlock()]),VARIANT)
block=blocks[0].half().eval()
with torch.no_grad():
    block.out_cls.weight.fill_(.001); block.out_reg.weight.fill_(.002)
    x=torch.rand(2,128,9,11,dtype=torch.float16)
    actual=block(x); expected=copy.deepcopy(block).float()(x.float())
    for a,b in zip(actual,expected):
        assert a.dtype==torch.float16 and torch.isfinite(a).all()
        torch.testing.assert_close(a.float(),b,rtol=.01,atol=.002)
print(VARIANT,'CPU FP16 block passed')
"""
            env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['YOLO_AUTOINSTALL']='false'
            env['YOLO_CONFIG_DIR']=str(ROOT.parent/'tmp/e42e43_test_settings')
            proc=subprocess.run([sys.executable,'-c',code],cwd=ROOT/name,capture_output=True,text=True,encoding='utf-8',errors='replace',env=env,timeout=120)
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
            print(name+' FP16 block passed')

    def test_control_flow_no_peer_kill(self):
        text=(ROOT/'run_pair.py').read_text()
        self.assertNotIn('.kill(',text); self.assertNotIn('.terminate(',text)
        self.assertIn("all(v is not None for v in codes.values())",text)
        self.assertIn("mode=='smoke'",text)
        self.assertIn("(logs/'abort').touch()",text)
        self.assertIn('Existing formal run; refusing overwrite',text)


if __name__=='__main__':
    unittest.main(verbosity=2)
