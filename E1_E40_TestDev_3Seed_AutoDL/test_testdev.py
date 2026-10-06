"""Engineering tests only. No benchmark conclusions or trained weight changes."""
import contextlib
import copy
import io
import json
import logging
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import d27_official as official
import testdev as app


class TestPipeline(unittest.TestCase):
    def test_vendor_pins_and_seed_pairs(self):
        self.assertEqual(len(app.vendor_check()), 6)

    def test_original_scorer(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = official.run_unit_tests()
        self.assertTrue(all(value == "passed" for value in result.values()))

    def test_split_mapping_and_content_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "annotations").mkdir()
            for stem in ("b", "a"):
                Image.new("RGB", (64, 48)).save(root / "images" / (stem + ".jpg"))
                (root / "annotations" / (stem + ".txt")).write_text("10,10,10,10,1,1,0,0\n", encoding="utf-8")
            previous = official.EXPECTED_IMAGES
            with contextlib.redirect_stdout(io.StringIO()):
                records, summary, table = app.load_records(root, expected=2)
            self.assertEqual(official.EXPECTED_IMAGES, previous)
            self.assertEqual([r.stem for r in records], ["a", "b"])
            self.assertEqual([r.image_id for r in records], [1, 2])
            self.assertEqual([row["id"] for row in table], [1, 2])
            with self.assertRaises(ValueError):
                app.load_records(root)  # A val/smoke-sized split cannot masquerade as test-dev.
            Image.new("RGB", (64, 48), (255, 0, 0)).save(root / "images" / "a.jpg")
            with contextlib.redirect_stdout(io.StringIO()):
                _, changed, _ = app.load_records(root, expected=2)
            self.assertEqual(summary["annotation_dimension_sha256"], changed["annotation_dimension_sha256"])
            self.assertNotEqual(summary["image_content_sha256"], changed["image_content_sha256"])
            Image.new("RGB", (64, 48)).save(root / "images" / "a.png")
            with self.assertRaises(RuntimeError):
                app.load_records(root, expected=2)

    def test_light_filter_equals_original_filter(self):
        rng = np.random.default_rng(12)
        record = official.ImageRecord(1, "fixture", Path("x.jpg"), Path("x.txt"), 100, 100,
                    np.array([[1,1,20,20,0,0,0,0], [30,30,10,10,1,1,0,0]], dtype=float),
                    np.array([[30,30,10,10,0,1]], dtype=float),
                    np.array([[1,1,20,20]], dtype=float), np.zeros((0,4)))
        rows = np.column_stack([rng.uniform(0,90,(40,2)), rng.uniform(1,20,(40,2)),
                                rng.uniform(0,1,40), rng.integers(1,11,40), np.arange(40)])
        rows = np.concatenate([np.array([[2,2,5,5,.99,1,40]]), rows])
        rows = rows[np.argsort(-rows[:,4], kind="mergesort")]
        with contextlib.redirect_stdout(io.StringIO()):
            ref, audit = official.filter_detections_and_audit([record], [rows])
            actual, new_audit = app.filter_for_score([record], [rows])
        np.testing.assert_array_equal(ref[0], actual[0])
        self.assertEqual(audit["removed_by_class0_ignore"], new_audit["removed_by_class0"])

    def test_prediction_cache_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            binding = dict(image_count=1, split="test-dev", checkpoint_sha256="a")
            self.assertFalse(app.prediction_valid(out, binding))
            app.write(out / "predictions.json", [])
            app.write(out / "prediction_manifest.json", dict(status="completed", binding=binding,
                        seen_images=1, sha256=official.sha256(out / "predictions.json")))
            self.assertTrue(app.prediction_valid(out, binding))
            with self.assertRaises(RuntimeError):
                app.prediction_valid(out, {**binding, "split": "val"})
            app.write(out / "predictions.json", [dict(image_id=1)])
            with self.assertRaises(RuntimeError):
                app.prediction_valid(out, binding)

    def test_manifest_rejects_wrong_split_protocol_and_seed(self):
        payload = dict(revision=app.REVISION, dataset=dict(images=1610), models=app.vendor_check(),
                       images=[dict(id=i+1, stem=f"img{i}") for i in range(1610)],
                       protocol=app.inference_protocol())
        app.validate_manifest(payload)
        wrong = copy.deepcopy(payload)
        wrong["models"][0]["sha256"] = "not-val-selected"
        with self.assertRaises(RuntimeError):
            app.validate_manifest(wrong)
        wrong = copy.deepcopy(payload)
        wrong["protocol"]["half"] = True
        with self.assertRaises(RuntimeError):
            app.validate_manifest(wrong)
        wrong = copy.deepcopy(payload)
        wrong["dataset"]["images"] = 548
        with self.assertRaises(RuntimeError):
            app.validate_manifest(wrong)

    def test_interrupted_writer_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.json"
            writer = app.PredictionWriter(path)
            writer.add({"image_id": 1})
            writer.close()
            self.assertFalse(path.exists())
            writer = app.PredictionWriter(path)
            writer.add({"image_id": 2})
            writer.complete()
            self.assertEqual(app.read(path), [{"image_id": 2}])

    def test_timeout_does_not_produce_score(self):
        handler = app.NmsTimeoutGuard()
        record = logging.LogRecord("test", logging.WARNING, "x", 1,
                                   "WARNING NMS time limit 2.2s exceeded", (), None)
        with self.assertRaises(RuntimeError):
            handler.emit(record)

    def test_paired_statistics_not_best_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            app.write(out / "input_manifest.json", {"fixture": True})
            results = {}
            for seed, delta in enumerate([.1, .5, .3], 1):
                for model, extra in (("e1", 0), ("e40", delta)):
                    results[f"{model}_seed{seed}"] = dict(official={k: 20 + seed + extra for k in app.KEYS})
            app.save_summary(out, {}, results)
            summary = app.read(out / "summary.json")
            self.assertAlmostEqual(summary["statistics"]["paired_delta_pp"][app.KEYS[0]]["mean"], .3)
            self.assertTrue((out / "COMPLETED.json").exists())


def run_tests():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestPipeline))
    if not result.wasSuccessful():
        raise RuntimeError("Test-dev engineering tests failed")
    return result


if __name__ == "__main__":
    run_tests()
