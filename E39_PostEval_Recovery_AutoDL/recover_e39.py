#!/usr/bin/env python3
"""Recover the completed E39 run's missing post-training evaluator only."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from recovery_common import BASE_REPORT, E39_NAME, completed_training, read, sha, write
from recovery_common import validate_comparison, recovered_evaluation

HERE = Path(__file__).resolve().parent


def package_check():
    manifest = read(HERE/'manifest.json')
    for name, expected in manifest.items():
        actual = hashlib.sha256((HERE/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if actual != expected:
            raise RuntimeError('Incomplete/modified recovery package: '+name)


def evidence_check(args):
    training, best = completed_training(args.run_report, args.project)
    csv_path = best.parent.parent/'results.csv'
    with csv_path.open(newline='', encoding='utf-8-sig') as handle:
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(handle)]
    if [int(row['epoch']) for row in rows] != list(range(1, 201)):
        raise RuntimeError('results.csv must contain exactly epochs 1..200')
    baseline = read(args.project/'comparison_reports'/BASE_REPORT/'metrics.json')
    if (baseline.get('status') != 'completed' or baseline.get('seed') != 1 or
            baseline.get('training', {}).get('completed_epochs') != 200):
        raise RuntimeError('Wrong completed E1 reference')
    ignore = {'project', 'name', 'model', 'data', 'device'}
    left = {k: v for k, v in baseline['train_params'].items() if k not in ignore}
    right = {k: v for k, v in training['params'].items() if k not in ignore}
    if left != right:
        raise RuntimeError('Saved E39 training protocol differs from E1')
    if sha(args.data) != baseline['data_yaml_sha256']:
        raise RuntimeError('Dataset YAML differs from E1; do not overwrite it')
    sources = {}
    for name in ('utils/tal.py', 'utils/loss.py', 'engine/trainer.py', 'nn/modules/head.py'):
        sources[name] = hashlib.sha256((args.source/'ultralytics'/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if sources[name] != baseline['p2_architecture']['source_code_sha256'][name]:
            raise RuntimeError('Pinned source differs from E1: '+name)
    return training, best, sources


def recover(args):
    package_check()
    training, best, sources = evidence_check(args)
    if (args.run_report/'evaluation_recovery.json').exists():
        out = recovered_evaluation(args.run_report, args.project)
        print('E39 EVALUATION ALREADY RECOVERED: '+str(out), flush=True)
        return out
    # Linux training roots exclude inherited DataLoader worker command lines.
    if sys.platform == 'linux':
        from queue_after_e39 import find_e39_processes
        if find_e39_processes(args.e39_package, args.project):
            raise RuntimeError('E39 training is still running; no concurrent recovery')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_CONFIG_DIR'] = str(args.project/'comparison_reports/e39_recovery_settings')
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    import ultralytics
    import local_context
    if not Path(ultralytics.__file__).resolve().is_relative_to(args.source.resolve()) or ultralytics.__version__ != '8.3.0':
        raise RuntimeError('Requires the original source checkout / Ultralytics 8.3.0')
    torch.set_num_threads(4)
    if args.device != 'cpu':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no CPU fallback')
        torch.cuda.set_device(args.device)
        if torch.cuda.mem_get_info()[0]/2**30 < 6:
            raise RuntimeError('Need at least 6 GiB free for the fixed batch4 evaluation')
    failure_hash, training_hash, best_hash = sha(args.run_report/'failure.json'), sha(args.run_report/'training.json'), sha(best)
    out = args.project/'comparison_reports'/('E39_RECOVERY_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    print('E39 POST-EVAL RECOVERY START '+datetime.now().isoformat()+' REPORT='+str(out), flush=True)
    print('Saved best checkpoint only; NO TRAINING; original failure.json preserved', flush=True)
    try:
        from saved_e39_evaluator import evaluate
        evaluate(args, best, out)
        comparison = read(out/'comparison.json')
        write(out/'official_metrics.json', comparison['candidate'])
        validate_comparison(out, best)
        if sha(best) != best_hash or sha(args.run_report/'failure.json') != failure_hash or sha(args.run_report/'training.json') != training_hash:
            raise RuntimeError('Original checkpoint/training/failure changed during evaluation')
        receipt = dict(status='completed', original_report=str(args.run_report.resolve()),
                       evaluation_report=str(out.resolve()), checkpoint_sha256=best_hash,
                       failure_sha256=failure_hash, training_sha256=training_hash,
                       preserved_original_failure=True, no_retraining=True,
                       source_sha256=sources, recovered_at=datetime.now().isoformat(),
                       artifact_sha256={name: sha(out/name) for name in
                           ('comparison.json', 'REPORT.md', 'area_predictions.json', 'gt_rows.json', 'official_metrics.json')})
        write(out/'recovery.json', receipt)
        # Publish only AFTER every evaluation/reproduction check succeeds.
        write(args.run_report/'evaluation_recovery.json', receipt)
        recovered_evaluation(args.run_report, args.project)
        print('E39 RECOVERY COMPLETED '+datetime.now().isoformat(), flush=True)
        print('Official AP/AP50/AP75: '+json.dumps({k: comparison['candidate'][k] for k in
              ('AP50_95_percent', 'AP50_percent', 'AP75_percent')}), flush=True)
        return out
    except BaseException:
        write(out/'failure.json', dict(status='failed', error=traceback.format_exc()))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=Path('/root/autodl-tmp/yolo11_project'))
    p.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/yolo11_src'))
    p.add_argument('--data', type=Path, default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019.yaml'))
    p.add_argument('--dataset', type=Path, default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val'))
    p.add_argument('--e39-package', type=Path, default=Path('/root/autodl-tmp/E39a_P2_LocalContext_AutoDL'))
    p.add_argument('--run-report', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--inspect', action='store_true', help='Read-only report/protocol checks; no predictions/training')
    args = p.parse_args()
    args.project, args.source, args.run_report = args.project.resolve(), args.source.resolve(), args.run_report.resolve()
    if args.inspect:
        package_check()
        training, best, _ = evidence_check(args)
        print(json.dumps(dict(training_completed_epochs=training['completed_epochs'], best=str(best),
              best_sha256=sha(best), last_exists=best.with_name('last.pt').is_file(),
              original_evaluate_exists=(args.e39_package/'evaluate.py').is_file(),
              original_failure_preserved=True), indent=2), flush=True)
        return
    if sys.platform == 'linux':
        import fcntl
        with (args.run_report/'.post_eval_recovery.lock').open('a+') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another E39 recovery process is already running')
            recover(args)
    else:
        recover(args)


if __name__ == '__main__':
    main()
