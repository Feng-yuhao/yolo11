"""Atomic report IO; no training imports or checkpoint modifications."""
import hashlib
import json
import math
from pathlib import Path

BASE_REPORT = 'e1_yolo11s_p2add_img640_seed1_20260926_102750_718906'
E39_NAME = 'e39a_p2_local_context_img640_seed1_200e'
MISSING_EVALUATE = "ModuleNotFoundError: No module named 'evaluate'"
KEYS = ('AP50_95_percent', 'AP50_percent', 'AP75_percent', 'AR500_percent')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def completed_training(report, project):
    report, project = Path(report).resolve(), Path(project).resolve()
    if report.parent != (project/'comparison_reports').resolve() or not report.name.startswith('E39_RUN_'):
        raise RuntimeError('Expected the original E39_RUN report inside this project')
    failure = read(report/'failure.json')
    # Recover only this documented post-training import error, never a training
    # failure, OOM, metric assertion or truncated run.
    if str(failure.get('error', '')).strip().splitlines()[-1:] != [MISSING_EVALUATE]:
        raise RuntimeError('Not the documented missing-evaluate post-training failure')
    training = read(report/'training.json')
    params = training.get('params', {})
    if training.get('status') != 'completed' or training.get('completed_epochs') != 200:
        raise RuntimeError('E39 training is not completed at 200 epochs')
    for key, value in dict(name=E39_NAME, epochs=200, imgsz=640, batch=16, seed=1).items():
        if params.get(key) != value:
            raise RuntimeError('Wrong E39 training protocol: '+key)
    best = project/'runs'/E39_NAME/'weights/best.pt'
    last = best.with_name('last.pt')
    if Path(training.get('best', '')).resolve() != best or not best.is_file() or not last.is_file():
        raise RuntimeError('Matching saved E39 best/last checkpoints are missing')
    if Path(training.get('last', '')).resolve() != last:
        raise RuntimeError('Wrong last checkpoint in training receipt')
    return training, best


def validate_comparison(out, best):
    out, best = Path(out).resolve(), Path(best).resolve()
    comparison = read(out/'comparison.json')
    if comparison.get('status') != 'completed' or comparison.get('dataset', {}).get('images') != 548:
        raise RuntimeError('Official evaluation must complete all 548 images')
    if (comparison.get('seed') != 1 or comparison.get('imgsz') != 640 or
            Path(comparison.get('checkpoint', '')).resolve() != best or
            comparison.get('checkpoint_sha256') != sha(best)):
        raise RuntimeError('Official evaluation checkpoint/protocol mismatch')
    for table in ('baseline', 'candidate', 'delta_pp'):
        for key in KEYS:
            value = comparison.get(table, {}).get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError('Missing/nonfinite official metric '+table+'.'+key)
    for key, expected in dict(AP50_95_percent=25.63123996261998,
                             AP50_percent=44.72545960536977,
                             AP75_percent=25.02982351561637).items():
        if abs(comparison['baseline'][key]-expected) > 1e-7:
            raise RuntimeError('E1 official reference did not reproduce '+key)
    for key in KEYS:
        if abs(comparison['delta_pp'][key] - (comparison['candidate'][key]-comparison['baseline'][key])) > 1e-7:
            raise RuntimeError('Official metric delta mismatch '+key)
    for name in ('REPORT.md', 'area_predictions.json', 'gt_rows.json', 'official_metrics.json'):
        if not (out/name).is_file():
            raise RuntimeError('Missing recovery artifact '+name)
    if read(out/'official_metrics.json') != comparison['candidate']:
        raise RuntimeError('Official metric summary differs from scored comparison')
    return comparison


def recovered_evaluation(report, project):
    """A completed recovery supplements, but never deletes, failure evidence."""
    training, best = completed_training(report, project)
    report = Path(report).resolve()
    receipt = read(report/'evaluation_recovery.json')
    out = Path(receipt.get('evaluation_report', '')).resolve()
    if out.parent != report.parent or not out.name.startswith('E39_RECOVERY_'):
        raise RuntimeError('Invalid recovery report location')
    if receipt.get('status') != 'completed' or receipt.get('original_report') != str(report):
        raise RuntimeError('Invalid recovery completion receipt')
    if receipt.get('failure_sha256') != sha(report/'failure.json') or receipt.get('training_sha256') != sha(report/'training.json'):
        raise RuntimeError('Original failure/training evidence changed since recovery')
    if receipt.get('checkpoint_sha256') != sha(best):
        raise RuntimeError('E39 checkpoint changed since recovery')
    artifacts = receipt.get('artifact_sha256', {})
    for name in ('comparison.json', 'REPORT.md', 'area_predictions.json', 'gt_rows.json', 'official_metrics.json'):
        if artifacts.get(name) != sha(out/name):
            raise RuntimeError('Recovery evaluation artifact changed: '+name)
    if (out/'failure.json').exists():
        raise RuntimeError('Recovery evaluation also failed')
    validate_comparison(out, best)
    return out
