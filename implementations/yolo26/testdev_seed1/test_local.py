"""Read-only protocol regressions and PRIVATE controller mocks, not server receipts."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import run_eval as r

SCRATCH = r.HERE.parents[1] / 'audit/yolo26_testdev_s1/unit'
SCRATCH.mkdir(parents=True, exist_ok=True)


class Tests(unittest.TestCase):
    def test_fixed_protocol(self):
        p = r.protocol()
        self.assertEqual((p['imgsz'], p['batch'], p['half'], p['conf'], p['iou'], p['max_det']),
                         (640, 4, False, .001, .7, 500))
        self.assertEqual(p['branch'], 'one2many+NMS')
        self.assertFalse(p['end2end'])

    def test_wrong_checkpoint_bytes_refused(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            project = Path(tmp)
            spec = r.read(r.REF / 'checkpoint_specs.json')['baseline']
            path = project / spec['relative_path']
            path.parent.mkdir(parents=True)
            path.write_bytes(b'wrong checkpoint')
            with self.assertRaisesRegex(RuntimeError, 'Wrong/damaged'):
                r.checkpoint_audit(project)

    def test_missing_checkpoint_refused(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            with self.assertRaisesRegex(RuntimeError, 'Missing trained'):
                r.checkpoint_audit(Path(tmp))

    def test_wrong_provenance_refused(self):
        original = r.read
        def changed(path):
            obj = original(path)
            if Path(path).name == 'baseline_training_completion.json':
                obj['epochs'] = 199
            return obj
        with patch.object(r, 'read', changed):
            with self.assertRaisesRegex(RuntimeError, 'Incomplete original'):
                r.checkpoint_audit(Path('/none'))

    def test_prediction_reproduction(self):
        pred = dict(image_id=1, category_id=4, bbox=[1., 2., 3., 4.], score=.8)
        self.assertEqual(r.assert_val_predictions([pred], [pred])['status'], 'passed')
        for field, value in (('image_id', 2), ('category_id', 3), ('bbox', [1., 2., 3., 5.]), ('score', .79)):
            changed = copy.deepcopy(pred)
            changed[field] = value
            with self.assertRaises(RuntimeError):
                r.assert_val_predictions([changed], [pred])
        with self.assertRaisesRegex(RuntimeError, 'count'):
            r.assert_val_predictions([], [pred])

    def test_official_expected_count_restored_on_failure(self):
        import d27_official as official
        count = official.EXPECTED_IMAGES
        with patch.object(official, 'load_dataset', side_effect=ValueError('private expected failure')):
            with self.assertRaises(ValueError):
                r.load_split(Path('/none'), 1610)
        self.assertEqual(official.EXPECTED_IMAGES, count)

    def test_packaged_files(self):
        self.assertEqual(len(r.package_audit()), 64)

    def test_no_training_commands(self):
        with patch.object(sys, 'argv', ['run_eval.py', 'train']):
            with self.assertRaises(SystemExit):
                r.arguments()
        with patch.object(sys, 'argv', ['run_eval.py', 'all']):
            self.assertEqual(r.arguments().mode, 'all')

    def test_controller_serial_flow_and_failure(self):
        for mode, failure, expected in (
                ('check', None, []), ('smoke', None, [('smoke', 'baseline'), ('smoke', 'e60')]),
                ('all', None, [('smoke', 'baseline'), ('smoke', 'e60'), ('eval', 'baseline'), ('eval', 'e60')]),
                ('all', 'baseline_smoke', [('smoke', 'baseline')])):
            with self.subTest(mode=mode, failure=failure), tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
                root = Path(tmp)
                args = SimpleNamespace(mode=mode, project=root, source=root / 'source',
                                       dataset=root / 'test', val_dataset=root / 'val')
                called = []
                class MockProcess:
                    def __init__(self, command, **kwargs):
                        self.pid = 123
                        self.stdout = iter(['PRIVATE CONTROLLER MOCK ONLY\n'])
                        stage = command[command.index('--worker-stage') + 1]
                        variant = command[command.index('--variant') + 1]
                        self.key = variant + '_' + stage
                        called.append((stage, variant))
                    def wait(self):
                        return 1 if self.key == failure else 0
                    def poll(self):
                        return self.wait()
                with patch.dict(sys.modules, {'fcntl': MagicMock()}), \
                     patch.object(r, 'environment_audit', return_value={'private_mock': True}), \
                     patch.object(r, 'checkpoint_audit', return_value={'private_mock': True}), \
                     patch.object(r, 'datasets_audit', return_value=([], [], {'testdev': {'private_mock': True}})), \
                     patch.object(r.subprocess, 'Popen', MockProcess), patch.object(r, 'summarize') as summary:
                    if failure:
                        with self.assertRaisesRegex(RuntimeError, 'failed'):
                            r.controller(args)
                    else:
                        r.controller(args)
                    self.assertEqual(called, expected)
                    self.assertEqual(summary.call_count, int(mode == 'all' and not failure))
                index = r.read(root / 'logs/latest_yolo26_testdev_seed1.json')
                state = r.read(Path(index['report']) / 'state.json')
                self.assertEqual(state['status'], 'failed' if failure else 'completed')
                self.assertFalse(state['training_started'])
                self.assertFalse(state['auto_shutdown'])


if __name__ == '__main__':
    unittest.main()
