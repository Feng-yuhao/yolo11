"""Stdlib synthetic tests: strict pairing, tradeoffs, stage ordering, no-source-write."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import offline_analysis as audit


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.a, self.b, self.out = self.root / "e47", self.root / "e40", self.root / "out"
        self.a.mkdir(); self.b.mkdir()
        self.rows_b = [self.row(0, .7, .7), self.row(1, .9, .9), self.row(2, .9, .7), self.row(3, .7, .7)]
        self.rows_a = [self.row(0, .9, .9), self.row(1, .7, .7), self.row(2, .9, .9), self.row(3, .9, .7)]
        self.dataset = {"annotation_dimension_sha256": "a" * 64, "images": 1, "evaluated_gt_rows": 4,
                        "raw_class_counts": {"1": 4}, "score_zero_counts": {"1": 0},
                        "class0_ignore_regions": 0, "class11_others": 0, "gt_rows_dropped_by_class0_region": 0}
        self.comp_b = self.comparison(20)
        self.comp_a = self.comparison(21)
        self.save()

    @staticmethod
    def row(index, raw, final):
        return {"image_id": 1, "gt_index": index, "cls": 1, "size": 10., "raw_any": raw,
                "raw_p2": raw, "raw_p3": raw, "raw_correct": raw,
                "final_any": final, "final_correct": final}

    def comparison(self, score):
        candidate = {key: float(score) for key in audit.AP_KEYS}
        candidate["per_class"] = {str(i): {"name": "class_" + str(i), "images_with_class_weight": 1,
                                          **{key: float(score) for key in audit.AP_KEYS}} for i in range(1, 11)}
        candidate["class_image_weights"] = {str(i): 1 for i in range(1, 11)}
        candidate["aggregation_note"] = "official image availability weighting"
        return {"status": "completed", "seed": 1, "imgsz": 640, "dataset": copy.deepcopy(self.dataset),
                "ignore": {"fp_audit_note": "exact convention"}, "checkpoint_sha256": "b" * 64,
                "candidate": candidate}

    def save(self):
        for directory, comp, rows in ((self.a, self.comp_a, self.rows_a), (self.b, self.comp_b, self.rows_b)):
            (directory / "comparison.json").write_text(json.dumps(comp), encoding="utf-8")
            (directory / "official_metrics.json").write_text(json.dumps({"status": "completed", "candidate": comp["candidate"], "dataset": comp["dataset"]}), encoding="utf-8")
            (directory / "gt_rows.json").write_text(json.dumps(rows), encoding="utf-8")

    def run_pair(self):
        return audit.summarize_pair(self.a, self.b, self.out)

    def test_tradeoffs_and_preservation(self):
        original = (self.a / "gt_rows.json").read_bytes()
        result = self.run_pair()
        final = result["overall"]["coverage"]["final_correct"]
        self.assertEqual((final["improved_count"], final["lost_count"], final["net_count"]), (2, 1, 1))
        self.assertEqual(final["delta_pp"], 25.)
        self.assertEqual(result["official"]["delta_pp"]["AP50_95_percent"], 1.)
        self.assertEqual((self.a / "gt_rows.json").read_bytes(), original)
        self.assertFalse(result["pairing"]["per_row_coordinates_checked"])
        self.assertFalse(result["pairing"]["density_available_in_both_reports"])
        self.assertEqual(sum(result["overall"]["failure_stage_ordering"]["e47_counts"].values()), 4)
        self.assertTrue((self.out / "REPORT_OFFLINE.md").is_file())
        self.assertEqual(list(self.out.glob("*.tmp")), [])

    def test_order_independent_pairing(self):
        self.rows_a.reverse(); self.save()
        self.assertEqual(self.run_pair()["overall"]["coverage"]["final_correct"]["net_count"], 1)

    def test_missing_key_not_inner_join(self):
        self.rows_a[-1]["gt_index"] = 99; self.save()
        with self.assertRaisesRegex(ValueError, "no inner join"):
            self.run_pair()
        self.assertFalse(self.out.exists())

    def test_duplicate_key_rejected(self):
        self.rows_a[-1] = copy.deepcopy(self.rows_a[0]); self.save()
        with self.assertRaisesRegex(ValueError, "duplicate GT key"):
            self.run_pair()

    def test_missing_metric_not_zero(self):
        del self.rows_a[0]["raw_any"]; self.save()
        with self.assertRaisesRegex(ValueError, "missing GT field raw_any"):
            self.run_pair()

    def test_class_and_size_rejected(self):
        self.rows_a[0]["cls"] = 2; self.save()
        with self.assertRaisesRegex(ValueError, "GT class differs"):
            self.run_pair()
        self.rows_a[0]["cls"] = 1
        self.rows_a[0]["size"] = 11; self.save()
        with self.assertRaisesRegex(ValueError, "GT size differs"):
            self.run_pair()

    def test_coordinates_verified_when_present(self):
        for row in self.rows_a + self.rows_b:
            row["gt_xywh"] = [10., 20., 10., 10.]
        self.save()
        self.assertTrue(self.run_pair()["pairing"]["per_row_coordinates_checked"])
        self.rows_a[0]["gt_xywh"][0] = 11; self.save()
        with self.assertRaisesRegex(ValueError, "GT coordinates differ"):
            self.run_pair()

    def test_metadata_checks(self):
        for field, value, expression in (("seed", 2, "matched seed"), ("imgsz", 1024, "imgsz")):
            original = self.comp_a[field]
            self.comp_a[field] = value; self.save()
            with self.assertRaisesRegex(ValueError, expression):
                self.run_pair()
            self.comp_a[field] = original
        self.comp_a["dataset"]["annotation_dimension_sha256"] = "c" * 64; self.save()
        with self.assertRaisesRegex(ValueError, "Dataset / ignored-GT mismatch"):
            self.run_pair()

    def test_non_nested_final_success_priority(self):
        self.rows_a[0]["raw_any"] = .7
        self.rows_a[0]["raw_correct"] = .7
        self.rows_a[0]["final_any"] = .9
        self.rows_a[0]["final_correct"] = .9
        self.save()
        stage = self.run_pair()["overall"]["failure_stage_ordering"]
        self.assertEqual(stage["e47_counts"]["final_correct_present"], 2)
        self.assertEqual(stage["non_nested_threshold_counts"]["e47"]["final_correct_without_raw_argmax_correct"], 1)

    def test_output_source_protection(self):
        with self.assertRaisesRegex(ValueError, "inside an original report"):
            audit.summarize_pair(self.a, self.b, self.a / "new_output")

    def test_absent_file_rejected(self):
        (self.a / "official_metrics.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_pair()

    def test_empty_slice_is_null_not_zero(self):
        self.run_pair()
        slices = json.loads((self.out / "paired_slices.json").read_text(encoding="utf-8"))["slices"]
        self.assertEqual(slices["lt8"]["gt_count"], 0)
        self.assertIsNone(slices["lt8"]["coverage"]["raw_any"]["delta_pp"])

    def test_official_metrics_mismatch_rejected(self):
        official_path = self.a / "official_metrics.json"
        value = json.loads(official_path.read_text())
        value["candidate"]["AP75_percent"] = 99
        official_path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "official/comparison metric mismatch"):
            self.run_pair()

    def test_invalid_and_nonfinite_iou_rejected(self):
        self.rows_a[0]["raw_any"] = 1.1; self.save()
        with self.assertRaisesRegex(ValueError, "Out-of-range"):
            self.run_pair()
        self.rows_a[0]["raw_any"] = float("nan"); self.save()
        with self.assertRaisesRegex(ValueError, "Invalid JSON number"):
            self.run_pair()

    def test_matched_density_is_checked_not_imputed(self):
        for row in self.rows_a + self.rows_b:
            row["density"] = 3
        self.save()
        self.assertTrue(self.run_pair()["pairing"]["density_available_in_both_reports"])
        self.rows_a[0]["density"] = 4; self.save()
        with self.assertRaisesRegex(ValueError, "GT density differs"):
            self.run_pair()

    def test_failed_input_does_not_overwrite_previous_output(self):
        self.run_pair()
        previous = (self.out / "offline_summary.json").read_bytes()
        self.rows_a[-1]["gt_index"] = 99; self.save()
        with self.assertRaisesRegex(ValueError, "no inner join"):
            self.run_pair()
        self.assertEqual((self.out / "offline_summary.json").read_bytes(), previous)


if __name__ == "__main__":
    unittest.main(verbosity=2)
