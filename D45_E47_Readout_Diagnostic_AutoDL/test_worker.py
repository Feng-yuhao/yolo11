"""Fast invariants for the diagnostic aggregation and report selection."""
import json
from pathlib import Path
import tempfile
import unittest
from diagnose_worker import FIELDS, geometry_summary
from io_utils import resolve_reports


class WorkerTests(unittest.TestCase):
    def row(self, index, good):
        return dict(image_id=1, gt_index=index, size=12., cls=1,
                    **{k: .8 if good else .6 for k in FIELDS})

    def test_pair_counts_and_empty(self):
        a = [self.row(0, True), self.row(1, False)]
        b = [self.row(0, False), self.row(1, True)]
        value = geometry_summary(a, b)
        self.assertEqual(value['all']['raw_any_paired_counts'], dict(improved=1, lost=1, net=0))
        self.assertEqual(value['all']['raw_any75_percent'], 50.)
        self.assertIsNone(value['lt8']['final_correct75_percent'])

    def test_report_incomplete_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for prefix in ('E47S2_RUN_a', 'E40S2_RUN_a'):
                report = root / 'comparison_reports' / prefix
                report.mkdir(parents=True)
                (report / 'comparison.json').write_text(json.dumps(dict(status='completed', seed=2, imgsz=640)))
                (report / 'training.json').write_text(json.dumps(dict(status='running', completed_epochs=100)))
            with self.assertRaisesRegex(RuntimeError, 'Need one completed'):
                resolve_reports(root, 2)

    def test_report_ambiguous_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for prefix in ('E47S2_RUN_a', 'E47S2_RUN_b', 'E40S2_RUN_a'):
                report = root / 'comparison_reports' / prefix
                report.mkdir(parents=True)
                (report / 'comparison.json').write_text(json.dumps(dict(status='completed', seed=2, imgsz=640)))
                (report / 'training.json').write_text(json.dumps(dict(status='completed', completed_epochs=200)))
            with self.assertRaisesRegex(RuntimeError, 'found 2'):
                resolve_reports(root, 2)


if __name__ == '__main__':
    unittest.main()
