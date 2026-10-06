"""Offline replication/config/pipeline tests; no server jobs or smoke receipts."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import all_steps
import references as refs

HERE = Path(__file__).resolve().parent


class ReplicationTests(unittest.TestCase):
    def test_fixed_same_seed_references(self):
        self.assertEqual(refs.SEED, 2)
        self.assertIn('seed2', refs.BASE_REPORT)
        self.assertTrue(refs.PARENT_REPORT.startswith('E40S2_RUN_'))
        self.assertEqual(refs.EXPERIMENT, 'e47a_p2_depth_readout_img640_seed2_200e')
        refs.require_scores(refs.BASE_EXPECTED, refs.BASE_EXPECTED, 'fixture')
        with self.assertRaises(RuntimeError):
            refs.require_scores({**refs.BASE_EXPECTED, 'AP50_95_percent': 25.63123996261998},
                                refs.BASE_EXPECTED, 'wrong seed1')

    def test_all_pipeline_success_order_and_live_output(self):
        calls = []
        def runner(command, **kwargs):
            calls.append((command, kwargs))
            return SimpleNamespace(returncode=0)
        self.assertEqual(all_steps.run_all(['--workers', '8'], runner), 0)
        self.assertEqual([cmd[cmd.index('--mode') + 1] for cmd, _ in calls], list(all_steps.STAGES))
        for command, kwargs in calls:
            self.assertEqual(command[-2:], ['--workers', '8'])
            self.assertFalse(kwargs['check'])
            self.assertNotIn('stdout', kwargs)
            self.assertNotIn('capture_output', kwargs)

    def test_failed_stage_never_launches_following_stages(self):
        for failed in all_steps.STAGES:
            calls = []
            def runner(command, **kwargs):
                stage = command[command.index('--mode') + 1]
                calls.append(stage)
                return SimpleNamespace(returncode=7 if stage == failed else 0)
            with self.subTest(failed=failed):
                self.assertEqual(all_steps.run_all([], runner), 7)
                self.assertEqual(calls, list(all_steps.STAGES[:all_steps.STAGES.index(failed) + 1]))
        with self.assertRaises(ValueError):
            all_steps.run_all(['--mode', 'run'])

    def test_static_package_and_manifest(self):
        manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
        for path in HERE.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            self.assertIn(path.name, manifest)
        for name, expected in manifest.items():
            actual = hashlib.sha256((HERE / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            self.assertEqual(actual, expected, name)
        text = (HERE / 'run_e47.py').read_text()
        self.assertIn('core.SEED = SEED', text)
        self.assertIn('init_seeds(SEED, deterministic=True)', text)
        self.assertNotIn('E40_RUN_20260929_205658_507620', text)
        self.assertNotIn('e47_cuda_check_gate.json', text)
        self.assertNotIn('DIAG_D38_E1_RAW_STAGES_', (HERE / 'evaluate.py').read_text())


if __name__ == '__main__':
    unittest.main(verbosity=2)
