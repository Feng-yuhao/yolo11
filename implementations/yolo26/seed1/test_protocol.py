"""Fast negative-gate, coordinate, native schedule and checkpoint regression tests."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import common as c
TEST_ROOT=Path(__file__).resolve().parents[2]/'audit/yolo26_port/unit_tests'
TEST_ROOT.mkdir(parents=True,exist_ok=True)
c.runtime(TEST_ROOT)
import torch
import protocol as p
import controller
from training26 import PairedTrainer


class Tests(unittest.TestCase):
    def test_fixed_pair_protocol(self):
        args=SimpleNamespace(project=Path('/root/autodl-tmp/yolo26_project'),data=p.DATA)
        opts=p.options(args,'baseline')
        self.assertEqual((opts['epochs'],opts['batch'],opts['imgsz'],opts['seed']), (200,16,640,1))
        self.assertTrue(opts['nms']); self.assertTrue(opts['amp']); self.assertFalse(opts['channels_last'])
        self.assertEqual(opts['nbs'],64); self.assertEqual(opts['optimizer'],'SGD')

    def test_source_drift_refused(self):
        with patch('protocol.c.sha256',return_value='bad'):
            with self.assertRaisesRegex(RuntimeError,'source drift'): p.audit_training_source(Path('/bad'))

    def test_local_and_stale_gates_refused(self):
        with tempfile.TemporaryDirectory(dir=str(TEST_ROOT)) as tmp:
            root=Path(tmp); (root/'comparison_reports').mkdir()
            args=SimpleNamespace(project=root)
            target=root/'comparison_reports/yolo26_p2_e60_seed1_smoke_gate.json'
            for item in (dict(status='passed',local_fixture=True,fingerprint='good'),dict(status='passed',local_fixture=False,fingerprint='old')):
                target.write_text(json.dumps(item))
                with self.assertRaisesRegex(RuntimeError,'smoke receipt'): p.verify_gate(args,{'fingerprint':'good'})

    def test_receipt_reports_required_unchanged(self):
        with tempfile.TemporaryDirectory(dir=str(TEST_ROOT)) as tmp:
            root=Path(tmp); (root/'comparison_reports').mkdir()
            reports={}
            for variant in ('baseline','e60'):
                path=root/variant; path.mkdir(); (path/'smoke.json').write_text(json.dumps(dict(status='passed'))); reports[variant]=str(path)
            receipt=dict(status='passed',local_fixture=False,fingerprint='good',reports=reports,
                         report_sha256={v:c.sha256(Path(path)/'smoke.json') for v,path in reports.items()})
            (root/'comparison_reports/yolo26_p2_e60_seed1_smoke_gate.json').write_text(json.dumps(receipt))
            p.verify_gate(SimpleNamespace(project=root),{'fingerprint':'good'})
            (root/'e60/smoke.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'altered'): p.verify_gate(SimpleNamespace(project=root),{'fingerprint':'good'})

    def test_batch_reduction_refused_before_native_pipeline(self):
        dummy=SimpleNamespace(batch_size=8,args=SimpleNamespace(batch=8))
        with self.assertRaisesRegex(RuntimeError,'batch size'): PairedTrainer._build_train_pipeline(dummy)

    def test_nonfinite_save_refused(self):
        model=torch.nn.Linear(1,1); model.weight.data.fill_(float('nan'))
        dummy=SimpleNamespace(model=model)
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'): PairedTrainer.save_model(dummy)

    def test_nonfinite_loss_no_recovery(self):
        dummy=SimpleNamespace(loss=torch.tensor(float('inf')),loss_items={},fitness=0.)
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'): PairedTrainer._handle_nan_recovery(dummy,0)

    def test_inverse_resize_rounding_differs_from_single_gain(self):
        from ultralytics.utils.ops import scale_boxes
        boxes=torch.tensor([[11.,17.,125.,103.]])
        expected=boxes.double().clone()
        h,w=128,128; oh,ow=128,160
        gain=min(h/oh,w/ow); rh,rw=round(oh*gain),round(ow*gain)
        px,py=round((w-rw)/2-.1),round((h-rh)/2-.1)
        expected[:,[0,2]]=(expected[:,[0,2]]-px)/(rw/ow)
        expected[:,[1,3]]=(expected[:,[1,3]]-py)/(rh/oh)
        torch.testing.assert_close(scale_boxes((h,w),boxes,(oh,ow)).double(),expected,atol=.0001,rtol=0)

    def test_controller_whole_flow_without_gpu_or_real_jobs(self):
        from unittest.mock import MagicMock
        import sys
        for mode,expected in (('smoke',['smoke','smoke']),('run',['train','eval','train','eval'])):
            with tempfile.TemporaryDirectory(dir=str(TEST_ROOT)) as tmp:
                root=Path(tmp); (root/'logs').mkdir(); (root/'comparison_reports').mkdir()
                called=[]
                class FakeProcess:
                    def __init__(self,command,**kwargs):
                        self.pid=123; self.stdout=iter(['PRIVATE MOCK ONLY\n'])
                        stage=command[3]; called.append(stage)
                        variant=command[command.index('--variant')+1]
                        out=Path(command[command.index('--report')+1]); out.mkdir()
                        if stage=='smoke': (out/'smoke.json').write_text(json.dumps(dict(status='passed')))
                        elif stage=='train': (out/'training_completion.json').write_text(json.dumps(dict(best=str(root/'mock.pt'))))
                        else:
                            value=dict(dataset={'private_mock':True},official_metrics={k:0. for k in ('AP50_95_percent','AP50_percent','AP75_percent','AR500_percent')})
                            (out/'official_metrics.json').write_text(json.dumps(value))
                    def wait(self): return 0
                with patch.object(c,'PROJECT',root),patch('controller.signature',return_value={'fingerprint':'private_mock'}),\
                     patch('controller.verify_gate',return_value={}),patch('controller.subprocess.Popen',FakeProcess),\
                     patch.dict(sys.modules,{'fcntl':MagicMock()}),patch.object(sys,'argv',['controller.py',mode]):
                    controller.main()
                self.assertEqual(called,expected)
                index=json.loads((root/'logs/latest_yolo26_p2_e60.json').read_text())
                state=json.loads((Path(index['session'])/'state.json').read_text())
                self.assertEqual(state['status'],'completed'); self.assertFalse(state['auto_shutdown'])


if __name__=='__main__': unittest.main()
