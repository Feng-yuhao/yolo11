"""Pinned E1 seed3 artifacts; scores are reproduced from its existing predictions."""
import hashlib
import json

BASE_REPORT = 'e1_yolo11s_p2add_img640_seed3_20260926_172932_190728'
ARTIFACTS = {
    'area_ground_truth.json': '0bf644d08a7d26de58cdff26b5f3f0f16c1e0e2731fa7ba5315a22c717679b05',
    'area_predictions.json': '8d2a0f18d8e5cd9003c1640cf86aa38b32b6d08c3586f74ca4aaf01fb6cac34e',
}
WEIGHTS = '11e656c7cdeb9112d303bae0d68c32a874031a7734ad88fda1f37fa9c88778eb'
EXPECTED_OFFICIAL = dict(AP50_95_percent=25.47334717171393,
                        AP50_percent=44.337464035543746,
                        AP75_percent=24.84501913155771,
                        AR500_percent=37.64210458739394)


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def check_reference_artifacts(args, meta=None):
    root = args.project / 'comparison_reports' / BASE_REPORT
    if meta is None:
        meta = json.loads((root / 'metrics.json').read_text(encoding='utf-8'))
    if (meta.get('status') != 'completed' or meta.get('seed') != 3 or
            meta.get('training', {}).get('completed_epochs') != 200):
        raise RuntimeError('E1 seed3 reference is not formal 640/200e')
    if meta.get('evaluation_weights', {}).get('sha256') != WEIGHTS:
        raise RuntimeError('E1 seed3 reference checkpoint identity differs')
    best = args.project / 'runs/e1_yolo11s_p2add_img640_seed3/weights/best.pt'
    if not best.is_file() or sha256(best) != WEIGHTS:
        raise RuntimeError('E1 seed3 best.pt missing or changed: ' + str(best))
    for name, expected in ARTIFACTS.items():
        path = root / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError('E1 seed3 archived prediction/mapping changed: ' + str(path))
    return meta


def require_reproduction(value):
    for key, expected in EXPECTED_OFFICIAL.items():
        if abs(value[key] - expected) > 1e-7:
            raise RuntimeError('E1 seed3 official reference did not reproduce: ' + key)

