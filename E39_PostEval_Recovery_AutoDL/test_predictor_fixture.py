"""Actual CPU E39 checkpoint reload -> prediction -> raw geometry fixture.

Run from the workspace using the existing E39 two-epoch integration checkpoint.
This does NOT train or evaluate the real validation set and does not alter weights.
"""
from pathlib import Path
import os
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'tmp/e39_recovery_test_settings'))
import numpy as np
import torch
from PIL import Image
import local_context
import d27_official as official
import d38_common as common
from saved_e39_evaluator import predict_records
from ultralytics import YOLO
from recovery_common import sha

torch.set_num_threads(4)
weights = ROOT/'yolo11_project/comparison_reports/E39_LOCAL-INTEGRATION_20260929_072731_487350/runs/integration/weights/best.pt'
before = sha(weights)
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temporary:
    out = Path(temporary)
    os.environ['YOLO_CONFIG_DIR'] = str(out/'settings')
    records = []
    for i in range(4):
        image = out/f'{i}.jpg'
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (128, 128, 3), dtype=np.uint8)).save(image)
        gt = np.asarray([[63., 63., 2., 2., 0., 1.]])
        records.append(official.ImageRecord(i+1, str(i), image, out/f'{i}.txt', 128, 128,
            np.asarray([[63., 63., 2., 2., 1., 1., 0., 0.]]), gt, np.empty((0, 4)), np.empty((0, 4))))
    print('Frozen official evaluator tests: '+str(official.run_unit_tests()), flush=True)
    predictions, rows = predict_records(SimpleNamespace(device='cpu'), weights, out, records)
    assert len(rows) == 4 and {r['image_id'] for r in rows} == {1, 2, 3, 4}
    assert all(np.isfinite(r[k]) and 0 <= r[k] <= 1 for r in rows for k in ('raw_any', 'raw_p2', 'raw_correct'))
    plain = YOLO(str(weights), task='detect')
    assert any(isinstance(m, local_context.ContextConv) for m in plain.model.modules())
    results = plain.predict([str(r.image_path) for r in records], batch=4, stream=False, **common.prediction_options('cpu'))
    expected = []
    for record, value in zip(records, results):
        for x1, y1, x2, y2, score, cls in value.boxes.data.cpu().tolist():
            expected.append(dict(image_id=record.image_id, category_id=int(cls)+1,
                bbox=[x1, y1, max(0, x2-x1), max(0, y2-y1)], score=score))
    assert predictions == expected, 'Audit predictor changed final predictions'
    filtered, audit = official.filter_detections_and_audit(records, official.load_predictions(out/'area_predictions.json', 4))
    metrics = official.official_evaluate(records, filtered)
    assert all(np.isfinite(metrics[k]) for k in common.KEYS)
    assert before == sha(weights), 'Checkpoint altered during evaluation'
    print('E39 actual checkpoint reload/fuse -> exact plain-prediction reproduction -> raw geometry -> official scoring: PASSED', flush=True)
