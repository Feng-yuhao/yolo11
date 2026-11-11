"""PRIVATE protocol/controller/shutdown mocks, not server results or receipts."""
import copy
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import common as c
import protocol as p

ROOT = c.HERE.parents[1] / 'audit/yolo26_seed2_serial/unit'
ROOT.mkdir(parents=True,exist_ok=True)


class ProtocolTests(unittest.TestCase):
    def test_only_training_seed_changes(self):
        args = SimpleNamespace(project=Path('/project'),data=Path('/data.yaml'))
        old,new = p.frozen.options(args,'example'),p.options(args,'example')
        self.assertEqual(old.pop('seed'),1)
        self.assertEqual(new.pop('seed'),2)
        self.assertEqual(old,new)
        self.assertEqual((new['epochs'],new['batch'],new['workers'],new['imgsz']),(200,16,8,640))

    def test_output_names_never_seed1(self):
        for variant in p.VARIANTS:
            self.assertIn('seed2',p.run_name(variant))
            self.assertNotEqual(p.run_name(variant),p.frozen.run_name(variant))

    def test_frozen_implementation_files(self):
        for name,digest in p.read(p.REF / 'frozen_seed1_files.json').items():
            self.assertEqual(c.sha256(c.HERE / name),digest,name)

    def test_new_settings_get_offline_plotting_font(self):
        import os
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp,patch.dict(os.environ),patch.object(sys,'path',list(sys.path)):
            root = Path(tmp); source = root / 'source'
            (source / 'ultralytics').mkdir(parents=True)
            (source / 'ultralytics/__init__.py').write_text('')
            args = SimpleNamespace(project=root,source=source,local_test=True)
            p.runtime(args,'font_test')
            target = root / 'environment_records/yolo26_seed2_font_test/Ultralytics/Arial.ttf'
            self.assertEqual(c.sha256(target),c.sha256(c.HERE / 'assets/Arial.ttf'))

    def test_expected_image_count_restored(self):
        import d27_official as official
        previous = official.EXPECTED_IMAGES
        with patch.object(official,'load_dataset',side_effect=ValueError('private')):
            with self.assertRaises(ValueError):
                p.load_split(Path('/none'),9)
        self.assertEqual(official.EXPECTED_IMAGES,previous)

    def test_stale_local_wrong_layout_gates_refused(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            args = SimpleNamespace(project=Path(tmp),local_test=False)
            value = dict(status='passed',fingerprint='good',seed=2,layout='serial',local_fixture=False,files={},sha256={})
            for key,bad in (('fingerprint','stale'),('seed',1),('layout','parallel'),('local_fixture',True)):
                item = dict(value); item[key] = bad
                p.write(p.gate_path(args),item)
                with self.assertRaises(RuntimeError):
                    p.verify_gate(args,dict(fingerprint='good'))

    def test_edited_smoke_refused(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp); args = SimpleNamespace(project=root,local_test=False)
            receipt = root / 'smoke.json'; p.write(receipt,dict(status='passed'))
            value = dict(status='passed',fingerprint='good',seed=2,layout='serial',local_fixture=False,
                         files={'e60':str(receipt)},sha256={'e60':c.sha256(receipt)})
            p.write(p.gate_path(args),value)
            p.verify_gate(args,dict(fingerprint='good'))
            p.write(receipt,{})
            with self.assertRaisesRegex(RuntimeError,'receipt changed'):
                p.verify_gate(args,dict(fingerprint='good'))

    def test_issue_gate_requires_actual_updates(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp); args = SimpleNamespace(project=root,local_test=False)
            p.write(root / 'check/check.json',dict(status='engineering_passed',cuda_executed=True))
            sig = dict(fingerprint='private',payload=dict(dataset=dict(val=dict(images=548)),imgsz=640))
            for variant in p.VARIANTS:
                value = dict(status='passed',variant=variant,local_fixture=False,successful_updates=8,
                             train_batch=16,validation_batch=32,validation_images=548,imgsz=640,
                             native_validator_complete=True,gradient_groups=52,actual_update_groups=51)
                p.write(root / (variant + '_smoke') / 'smoke.json',value)
            with self.assertRaisesRegex(RuntimeError,'all52'):
                p.issue_gate(args,sig,root)

    def test_checkpoint_not_replaceable_after_training(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp); args = SimpleNamespace(project=root,local_test=False)
            run = root / 'runs' / p.run_name('baseline')
            best = run / 'weights/best.pt'; best.parent.mkdir(parents=True); best.write_bytes(b'private model fixture')
            report = root / 'report'
            value = dict(status='training_completed',epochs=200,variant='baseline',seed=2,
                         run=str(run),best=str(best),best_sha256=c.sha256(best))
            p.write(report / 'baseline_train/training_completion.json',value)
            self.assertEqual(p.training_weights(args,report,'baseline'),best)
            best.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'changed'):
                p.training_weights(args,report,'baseline')

    def test_negative_seed2_retained_in_mean(self):
        from summarize import summarize,KEYS
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            for split in ('val','test-dev'):
                for variant in p.VARIANTS:
                    old = p.read(p.REF / (variant + '_seed1_' + split.replace('-','_') + '.json'))
                    base = p.read(p.REF / ('baseline_seed1_' + split.replace('-','_') + '.json'))
                    value = copy.deepcopy(old)
                    value.update(seed=2,variant=variant,split=split,local_fixture=False,
                                 protocol=dict(private_unit_mock=True))
                    for key in KEYS:
                        value['official_metrics'][key] = base['official_metrics'][key] + (0. if variant == 'baseline' else -.2)
                    p.write(root / (variant + '_' + split.replace('-','_')) / 'official_metrics.json',value)
            summarize(root)
            result = p.read(root / 'comparison.json')
            for split in ('val','test-dev'):
                stats = result['splits'][split]['aggregate']['AP50_95_percent']
                self.assertAlmostEqual(stats['paired_delta_pp']['2'],-.2)
                self.assertAlmostEqual(stats['delta_mean_pp'],(stats['paired_delta_pp']['1']-.2)/2)

    def test_controller_serial_flow_and_partial_failure(self):
        import controller
        for mode,failure,expected in (
                ('check',None,['check']),
                ('smoke',None,['check','baseline_smoke','e60_smoke']),
                ('all',None,['check','baseline_smoke','e60_smoke','baseline_train','e60_train',
                             'baseline_val','baseline_test_dev','e60_val','e60_test_dev']),
                ('all','baseline_train',['check','baseline_smoke','e60_smoke','baseline_train','e60_train',
                                         'e60_val','e60_test_dev'])):
            with self.subTest(mode=mode,failure=failure), tempfile.TemporaryDirectory(dir=ROOT) as tmp:
                root = Path(tmp)
                args = SimpleNamespace(mode=mode,auto_shutdown=False,local_test=True,local_size=128,local_epochs=2,
                      project=root,source=root / 'source',data=root / 'data.yaml',val_dataset=root / 'val',
                      dataset=root / 'test',training_report=None)
                calls = []
                class MockProcess:
                    def __init__(self,command,**kwargs):
                        self.pid = 123; self.stdout = iter(['PRIVATE MOCK ONLY\n'])
                        stage = command[3]; variant = command[command.index('--variant')+1]
                        split = command[command.index('--split')+1] if '--split' in command else None
                        self.key = 'check' if stage == 'check' else variant + '_' + (split.replace('-','_') if split else stage)
                        calls.append(self.key)
                    def wait(self):
                        return 1 if self.key == failure else 0
                    def poll(self):
                        return self.wait()
                with patch.object(controller,'arguments',return_value=args),patch.object(controller,'runtime'), \
                     patch.object(controller,'signature',return_value=dict(fingerprint='private',payload={})), \
                     patch.object(controller,'issue_gate'),patch.object(controller,'verify_gate'), \
                     patch.object(controller.subprocess,'Popen',MockProcess),patch('summarize.summarize') as summary:
                    if failure:
                        with self.assertRaisesRegex(RuntimeError,'Some stages failed'):
                            controller.main()
                    else:
                        controller.main()
                    self.assertEqual(calls,expected)
                    self.assertEqual(summary.call_count,int(mode == 'all' and not failure))
                index = p.read(root / 'logs/PRIVATE_latest_yolo26_seed2_serial.json')
                state = p.read(Path(index['report_root']) / 'state.json')
                self.assertEqual(state['status'],'failed' if failure else 'completed')
                self.assertFalse(state['auto_shutdown']); self.assertFalse(state['child_active'])


class ShutdownTests(unittest.TestCase):
    def test_watch_never_calls_runner_without_authorization(self):
        import shutdown_after as s
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            project = Path(tmp); session = project / 'logs/session'; session.mkdir(parents=True)
            p.write(session / 'state.json',dict(auto_shutdown=False))
            calls = []
            result = s.run_watch(session,project,platform_name='linux',runner=lambda *a,**k:calls.append(a))
            self.assertEqual(result['status'],'not_authorized_or_not_terminal'); self.assertEqual(calls,[])

    def test_text_platform_helper_uses_bash(self):
        from shutdown_safety import shutdown_invocation
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / 'shutdown'
            path.write_text('echo "user execute shutdown command in container" > /proc/1/fd/1\n# supervisord\n')
            self.assertEqual(shutdown_invocation(path)['args'],['/bin/bash',str(path)])

    def test_watch_success_cancel_busy_and_new_session(self):
        import shutdown_after as s
        for scenario in ('success','cancel','busy','new_session'):
            with self.subTest(scenario=scenario),tempfile.TemporaryDirectory(dir=ROOT) as tmp:
                project = Path(tmp); session = project / 'logs/session'; session.mkdir(parents=True)
                report = project / 'comparison_reports/YOLO26_P2_E60_S2_ALL_PRIVATE'; report.mkdir(parents=True)
                command = project / 'shutdown'
                command.write_text('echo "user execute shutdown command in container" > /proc/1/fd/1\n# supervisord\n')
                from shutdown_safety import shutdown_invocation
                state = dict(auto_shutdown=True,admitted=True,local_fixture=False,status='failed',child_active=False,
                             report_root=str(report),controller_identity=dict(pid=123,start_ticks=456),
                             shutdown_command=shutdown_invocation(command))
                p.write(session / 'state.json',state)
                p.write(project / 'logs/latest_yolo26_seed2_serial.json',dict(session=str(session)))
                if scenario == 'cancel':
                    (project / 'KEEP_RUNNING_YOLO26_SEED2').touch()
                if scenario == 'new_session':
                    p.write(project / 'logs/latest_yolo26_seed2_serial.json',dict(session=str(project / 'logs/new')))
                calls = []; elapsed = [0.]
                def sleep(seconds): elapsed[0] += seconds * (1000 if scenario == 'busy' else 1)
                def runner(*args,**kwargs):
                    calls.append(args[0]); return SimpleNamespace(returncode=0)
                result = s.run_watch(session,project,platform_name='linux',runner=runner,sleeper=sleep,
                    clock=lambda:elapsed[0],identity=lambda pid:None,locks=lambda path:True,command=command,
                    gpu=lambda:dict(total_MiB=32760,free_MiB=32760,compute_pids=[999] if scenario == 'busy' else []),
                    processes=lambda:dict(jobs=[],ignored_services=[]),delay=5)
                self.assertEqual(len(calls),int(scenario == 'success'))
                self.assertEqual(result['status'],dict(success='shutdown_command_returned_zero',
                    cancel='cancelled_by_KEEP_RUNNING',busy='blocked_idle_wait_timeout',
                    new_session='cancelled_new_session_or_active_controller')[scenario])


if __name__ == '__main__':
    unittest.main()
