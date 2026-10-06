#!/usr/bin/env python3
"""Fixed E47 seed2 + seed3 controller; no downloads or remote operations."""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import time
import uuid

import parallel_safety as safety

ROOT = Path(__file__).resolve().parent
PREFIXES = {'e47s2': 'E47S2_', 'e47s3': 'E47S3_'}


def say(message):
    print(f'[{datetime.now().isoformat(timespec="seconds")}] {message}', flush=True)


def worker_command(name, mode, args):
    return ['bash', str(ROOT / name / 'run_e47.sh'), mode,
            '--project', str(args.project), '--source', str(args.source), '--data', str(args.data),
            '--dataset', str(args.dataset), '--device', args.device, '--workers', str(args.workers)]


def run_one(name, mode, args, runner=subprocess.run):
    say(f'START {name} {mode}; output follows live')
    result = runner(worker_command(name, mode, args), check=False)
    say(f'END {name} {mode}: exit={result.returncode}')
    return result.returncode


def progress(project, name, mode='run'):
    reports = project / 'comparison_reports'
    candidates = sorted(reports.glob(PREFIXES[name] + mode.upper() + '_*'), reverse=True)
    for directory in candidates:
        path = directory / 'training.json'
        if not path.exists():
            if (directory / 'failure.json').exists():
                return dict(status='failed', completed_epochs=0, report=str(directory))
            continue
        try:
            value = safety.read_json(path)
        except (OSError, ValueError):
            return dict(status='writing', report=str(directory))
        epoch = value.get('completed_epochs', 0)
        elapsed = value.get('elapsed_seconds', 0)
        remaining = elapsed / epoch * (200 - epoch) if epoch else None
        return dict(status=value.get('status'), completed_epochs=epoch, total_epochs=200,
                    elapsed_seconds=elapsed, ETA_seconds=remaining, report=str(directory),
                    training_json=str(path))
    return dict(status='starting' if mode == 'run' else mode, completed_epochs=0)


def check_run_destinations(args):
    for name, experiment in safety.EXPERIMENTS.items():
        path = args.project / 'runs' / experiment
        if path.exists():
            raise RuntimeError(f'{name}: existing formal run (including independent E47); refusing overwrite: {path}')


def launch_pair(mode, args, gpu, popen=subprocess.Popen, sleep=time.sleep, poll_interval=None):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_dir = args.project / 'logs' / f'pair_e47_seed2_seed3_{mode}_{stamp}'
    log_dir.mkdir(parents=True, exist_ok=False)
    invocation = uuid.uuid4().hex
    context = safety.make_context(args, mode, log_dir, gpu, invocation)
    environment = os.environ.copy()
    environment.update({safety.ENV_PREFIX + 'PAIR': '1', safety.ENV_PREFIX + 'PHASE': mode,
                        safety.ENV_PREFIX + 'TOKEN': context['token'],
                        safety.ENV_PREFIX + 'CONTEXT': str(log_dir / 'pair_context.json')})
    state = dict(schema=safety.SCHEMA, phase=mode, invocation=invocation, started_at=datetime.now().isoformat(),
                 controller=context['controller'], context=str(log_dir / 'pair_context.json'),
                 logs={name: str(log_dir / f'{name}_{mode}.log') for name in safety.NAMES},
                 pids={}, exit_codes={}, progress={})
    safety.write_json(log_dir / 'state.json', state)
    say(f'{mode.upper()} controller state: {log_dir / "state.json"}')
    children, handles = {}, {}
    try:
        for name in safety.NAMES:
            handles[name] = Path(state['logs'][name]).open('w', encoding='utf-8', buffering=1)
            try:
                children[name] = popen(worker_command(name, mode, args), stdout=handles[name],
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       start_new_session=True, env=environment)
                state['pids'][name] = children[name].pid
                say(f'{name}: PID={children[name].pid}; log={state["logs"][name]}')
            except Exception as exc:
                state['exit_codes'][name] = 127
                safety.write_json(log_dir / f'{name}_failed.json', dict(error=repr(exc), exit_code=127))
                say(f'{name} launch failed: {exc}; other task continues')
        interval = poll_interval if poll_interval is not None else (30 if mode == 'run' else 5)
        while True:
            for name, child in children.items():
                code = child.poll()
                state['exit_codes'][name] = code
                if code is not None and code != 0:
                    safety.write_json(log_dir / f'{name}_failed.json', dict(exit_code=code, detected_at=time.time()))
            try:
                state['GPU'] = safety.gpu_state()
            except Exception as exc:
                state['gpu_error'] = repr(exc)
            state['progress'] = {name: progress(args.project, name, mode) for name in safety.NAMES}
            state['updated_at'] = datetime.now().isoformat()
            safety.write_json(log_dir / 'state.json', state)
            say(' | '.join(f'{name}: {state["progress"][name].get("completed_epochs", 0)}/200 '
                          f'status={state["progress"][name]["status"]} '
                          f'ETA_h={round(state["progress"][name]["ETA_seconds"]/3600, 2) if state["progress"][name].get("ETA_seconds") is not None else "?"} '
                          f'exit={state["exit_codes"].get(name)}' for name in safety.NAMES)
                + f' | GPU free={state.get("GPU", {}).get("free_MiB", "?")} MiB')
            if all(state['exit_codes'].get(name) is not None for name in safety.NAMES):
                break
            sleep(interval)
        success = all(state['exit_codes'][name] == 0 for name in safety.NAMES)
        if mode == 'smoke' and success:
            try:
                receipt = safety.build_pair_receipt(args, context)
                receipt_path = args.project / 'comparison_reports' / safety.RECEIPT_NAME
                safety.write_json(receipt_path, receipt)
                state['pair_receipt'] = str(receipt_path)
                say(f'PARALLEL CUDA SMOKE PASSED: {receipt_path}; overlap={receipt["overlap_seconds"]}')
            except Exception as exc:
                success = False
                state['receipt_error'] = repr(exc)
                say('No parallel smoke receipt: ' + repr(exc))
        if mode == 'run' and success:
            try:
                from result_summary import summarize
                reports = {name: state['progress'][name]['report'] for name in safety.NAMES}
                state['summary'] = summarize(args.project, reports)
                say('BOTH TRAINING AND OFFICIAL EVALUATION COMPLETE; summary=' + str(state['summary']))
            except Exception as exc:
                success = False
                state['summary_error'] = repr(exc)
                say('Training results preserved; final summary failed: ' + repr(exc))
        state.update(status='completed' if success else 'failed', finished_at=datetime.now().isoformat())
        safety.write_json(log_dir / 'state.json', state)
        for name in safety.NAMES:
            say(f'FINISHED {name} {mode}: exit={state["exit_codes"][name]}; log={state["logs"][name]}')
        return 0 if success else 1
    except BaseException as exc:
        state.update(status='controller_interrupted', interrupted_at=datetime.now().isoformat(),
                     controller_error=repr(exc),
                     worker_notice='Workers were not terminated; inspect the listed PID/log before restarting.',
                     exit_codes={name: child.poll() for name, child in children.items()})
        safety.write_json(log_dir / 'state.json', state)
        say('Controller interrupted. Worker processes may continue; PID/log are recorded in ' + str(log_dir / 'state.json'))
        raise
    finally:
        # Never terminate a child, another training process, or the machine.
        for handle in handles.values():
            handle.close()


def execute(args):
    safety.package_check()
    args.project = args.project.resolve()
    args.source = args.source.resolve()
    args.data = args.data.resolve()
    args.dataset = args.dataset.resolve()
    if not args.project.is_dir():
        raise RuntimeError('Missing project: ' + str(args.project))
    if args.workers != 8 or args.device != 'cuda:0':
        raise RuntimeError('Pair protocol fixes workers=8 and device=cuda:0')
    import fcntl  # Server entry is Linux; mock tests can import this module on Windows.
    lock = (args.project / '.e40s3_e47s1_pair_execution.lock').open('a+')
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another E40 seed3 / E47 seed1 pair controller is active') from exc
        if args.mode == 'check':
            codes = [run_one(name, 'check', args) for name in safety.NAMES]
            return 0 if all(code == 0 for code in codes) else 1
        if args.mode in ('verify', 'run'):
            codes = [run_one(name, 'verify-gates', args) for name in safety.NAMES]
            if any(code != 0 for code in codes):
                raise RuntimeError('CUDA check/smoke gates failed current worker signature verification')
            receipt = safety.verify_pair_receipt(args)
            say(f'CURRENT PARALLEL SMOKE RECEIPT VALID; overlap={receipt["overlap_seconds"]}')
            if args.mode == 'verify':
                return 0
        check_run_destinations(args)
        gpu = safety.gpu_state()
        safety.require_idle_gpu(gpu)
        return launch_pair(args.mode, args, gpu)
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'smoke', 'verify', 'run', 'monitor'))
    parser.add_argument('--project', type=Path, default=Path('/root/autodl-tmp/yolo11_project'))
    parser.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/yolo11_src'))
    parser.add_argument('--data', type=Path, default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019.yaml'))
    parser.add_argument('--dataset', type=Path, default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.mode == 'monitor':
        logs = sorted((args.project / 'logs').glob('pair_e40s3_e47s1_*/state.json'),
                      key=lambda p: p.stat().st_mtime_ns, reverse=True)
        if not logs:
            raise RuntimeError('No pair controller state found')
        print(__import__('json').dumps(safety.read_json(logs[0]), indent=2, ensure_ascii=False), flush=True)
        return 0
    return execute(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        say('FAILED: ' + repr(exc))
        raise
