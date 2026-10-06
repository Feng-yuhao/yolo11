"""Recovery safety fixtures. No GPU, real training or remote processes."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import recovery_common as r
import recover_e39 as runner
import queue_after_e39 as q


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.project = Path(self.tmp.name)/'project'
        self.report = self.project/'comparison_reports/E39_RUN_fixture'
        self.out = self.project/'comparison_reports/E39_RECOVERY_fixture'
        self.best = self.project/'runs'/r.E39_NAME/'weights/best.pt'
        self.best.parent.mkdir(parents=True)
        self.best.write_bytes(b'checkpoint fixture')
        self.best.with_name('last.pt').write_bytes(b'last fixture')
        self.training = dict(status='completed', completed_epochs=200, best=str(self.best),
            last=str(self.best.with_name('last.pt')), params=dict(name=r.E39_NAME, epochs=200, imgsz=640, batch=16, seed=1))
        r.write(self.report/'training.json', self.training)
        r.write(self.report/'failure.json', dict(error='Traceback original\n'+r.MISSING_EVALUATE+'\n'))
        self.original_failure = (self.report/'failure.json').read_bytes()
        self.baseline = dict(AP50_95_percent=25.63123996261998, AP50_percent=44.72545960536977,
                             AP75_percent=25.02982351561637, AR500_percent=40.)
        self.candidate = {k: v+0.1 for k, v in self.baseline.items()}
        self.comparison = dict(status='completed', candidate=self.candidate, baseline=self.baseline,
            delta_pp={k: 0.1 for k in r.KEYS}, checkpoint=str(self.best), checkpoint_sha256=r.sha(self.best),
            seed=1, imgsz=640, dataset=dict(images=548))
        r.write(self.out/'comparison.json', self.comparison)
        r.write(self.out/'official_metrics.json', self.candidate)
        r.write(self.out/'area_predictions.json', [])
        r.write(self.out/'gt_rows.json', [])
        (self.out/'REPORT.md').write_text('Fixture evaluation')
        self.receipt = dict(status='completed', original_report=str(self.report.resolve()),
            evaluation_report=str(self.out.resolve()), failure_sha256=r.sha(self.report/'failure.json'),
            training_sha256=r.sha(self.report/'training.json'), checkpoint_sha256=r.sha(self.best),
            artifact_sha256={name: r.sha(self.out/name) for name in
                ('comparison.json', 'REPORT.md', 'area_predictions.json', 'gt_rows.json', 'official_metrics.json')})
        r.write(self.report/'evaluation_recovery.json', self.receipt)

    def tearDown(self):
        self.tmp.cleanup()

    def test_recovery_accepts_only_completed_matching_evaluation_and_preserves_failure(self):
        self.assertTrue(q.e39_receipt(self.report, self.project)['complete'])
        self.assertEqual((self.report/'failure.json').read_bytes(), self.original_failure)

    def test_without_recovery_blocked(self):
        (self.report/'evaluation_recovery.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            q.e39_receipt(self.report, self.project)

    def test_training_failure_cannot_be_recovered(self):
        r.write(self.report/'failure.json', dict(error='RuntimeError: CUDA out of memory'))
        with self.assertRaisesRegex(RuntimeError, 'documented'):
            q.e39_receipt(self.report, self.project)

    def test_incomplete_training_blocked(self):
        self.training['completed_epochs'] = 199
        r.write(self.report/'training.json', self.training)
        with self.assertRaisesRegex(RuntimeError, '200'):
            q.e39_receipt(self.report, self.project)

    def test_checkpoint_change_blocked(self):
        self.best.write_bytes(b'different checkpoint')
        with self.assertRaisesRegex(RuntimeError, 'checkpoint changed'):
            q.e39_receipt(self.report, self.project)

    def test_modified_official_artifacts_blocked(self):
        r.write(self.out/'comparison.json', dict(status='completed'))
        with self.assertRaisesRegex(RuntimeError, 'artifact changed'):
            q.e39_receipt(self.report, self.project)

    def test_changed_original_failure_blocked(self):
        r.write(self.report/'failure.json', dict(error='Extra evidence\n'+r.MISSING_EVALUATE))
        with self.assertRaisesRegex(RuntimeError, 'evidence changed'):
            q.e39_receipt(self.report, self.project)

    def test_wrong_baseline_reproduction_blocked(self):
        self.comparison['baseline']['AP75_percent'] += 1
        r.write(self.out/'comparison.json', self.comparison)
        with self.assertRaisesRegex(RuntimeError, 'reference did not reproduce'):
            r.validate_comparison(self.out, self.best)

    def test_incomplete_dataset_blocked(self):
        self.comparison['dataset']['images'] = 8
        r.write(self.out/'comparison.json', self.comparison)
        with self.assertRaisesRegex(RuntimeError, '548'):
            r.validate_comparison(self.out, self.best)

    def test_failed_recovery_blocked(self):
        r.write(self.out/'failure.json', dict(error='evaluation failed'))
        with self.assertRaisesRegex(RuntimeError, 'also failed'):
            q.e39_receipt(self.report, self.project)

    def test_evidence_check_requires_real_200_row_training_history(self):
        src = self.project/'source'
        data = self.project/'data.yaml'
        data.write_bytes(b'fixture dataset')
        source_hashes = {}
        for name in ('utils/tal.py', 'utils/loss.py', 'engine/trainer.py', 'nn/modules/head.py'):
            file = src/'ultralytics'/name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b'fixture source')
            source_hashes[name] = r.sha(file)
        base = dict(status='completed', seed=1, training=dict(completed_epochs=200),
                    train_params=self.training['params'], data_yaml_sha256=r.sha(data),
                    p2_architecture=dict(source_code_sha256=source_hashes))
        r.write(self.project/'comparison_reports'/r.BASE_REPORT/'metrics.json', base)
        csv = self.best.parent.parent/'results.csv'
        csv.write_text('epoch, box_loss\n'+''.join(f'{i}, 1\n' for i in range(1, 201)))
        args = argparse.Namespace(project=self.project, source=src, data=data, run_report=self.report)
        self.assertEqual(runner.evidence_check(args)[1], self.best.resolve())
        csv.write_text('epoch, box_loss\n1, 1\n')
        with self.assertRaisesRegex(RuntimeError, 'epochs 1..200'):
            runner.evidence_check(args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
