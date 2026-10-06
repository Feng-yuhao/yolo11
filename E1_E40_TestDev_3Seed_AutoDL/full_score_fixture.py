"""Synthetic 1610-image CLI recovery test; NOT a real test-dev evaluation."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image
import testdev as app


def main():
    with tempfile.TemporaryDirectory(prefix="testdev_CPU_score_fixture_") as tmp:
        root = Path(tmp)
        dataset, report = root / "fixture_data", root / "FIXTURE_ONLY_REPORT"
        (dataset / "images").mkdir(parents=True)
        (dataset / "annotations").mkdir()
        stream = io.BytesIO()
        Image.new("RGB", (64,48)).save(stream, format="JPEG")
        image_bytes = stream.getvalue()
        for i in range(1610):
            stem = f"{i:05d}"
            (dataset / "images" / (stem + ".jpg")).write_bytes(image_bytes)
            (dataset / "annotations" / (stem + ".txt")).write_text("10,10,10,10,1,1,0,0\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            _, summary, table = app.load_records(dataset)
        payload = dict(revision=app.REVISION, fixture_only=True, dataset=summary, images=table,
                       models=app.vendor_check(), protocol=app.inference_protocol(),
                       environment=dict(source_sha256="fixture_only_no_real_inference"))
        app.write(report / "input_manifest.json", payload)
        for spec in payload["models"]:
            out = report / spec["key"]
            predictions = [dict(image_id=i, category_id=1, bbox=[10,10,10,10],score=.9)
                           for i in range(1,1611) if spec["model"] == "E40" or i > 161]
            app.write(out / "predictions.json", predictions)
            app.write(out / "prediction_manifest.json", dict(status="completed", seen_images=1610,
                      binding=app.binding_for(payload,spec), sha256=app.official.sha256(out / "predictions.json")))
        command = [sys.executable,"-u",str(app.PACKAGE / "testdev.py"),"--mode","score",
                   "--dataset",str(dataset),"--report",str(report),"--source",str(root / "NO_SOURCE_OR_PT")]
        completed = subprocess.run(command, capture_output=True, text=True)
        app.require(completed.returncode == 0, completed.stdout + completed.stderr)
        value = app.read(report / "summary.json")
        for seed in (1,2,3):
            for model, expected in (("e1",90.), ("e40",100.)):
                actual = value["models"][f"{model}_seed{seed}"]["official"]["AP50_95_percent"]
                app.require(abs(actual-expected)<1e-9, f"Wrong synthetic AP: {actual}")
        app.require(abs(value["statistics"]["paired_delta_pp"]["AP50_95_percent"]["mean"]-10)<1e-9,
                    "Wrong paired summary")
        app.require((report / "COMPLETED.json").is_file(), "No complete marker")
        # Re-running recovery reuses all six verified CPU scores.
        again = subprocess.run(command, capture_output=True, text=True)
        app.require(again.returncode == 0 and again.stdout.count("Reusing verified CPU score") == 6,
                    "CPU recovery did not correctly reuse complete scores")
    print("SYNTHETIC 1610-image six-model CPU-score CLI + resume PASSED; no real test-dev was evaluated")


if __name__ == "__main__":
    main()
