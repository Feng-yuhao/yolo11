#!/usr/bin/env python3
"""Start checked E48 after E47 seed2 exits and GPU is released, regardless of outcome.

This is an explicitly authorized failure-tolerant queue. It records E47 seed2 errors,
never declares E47 seed2 successful, never kills processes, and never runs check/smoke.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EXPERIMENT = 'e48a_p2_nonlinear_readout_img640_seed1_200e'


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def identity(pid, proc_root=Path('/proc')):
    try:
        text = (proc_root / str(pid) / 'stat').read_text()
        fields = text[text.rfind(')') + 2:].split()
        if fields[0] in ('Z', 'X'):
            return None
        return dict(pid=int(pid), ppid=int(fields[1]), start_ticks=int(fields[19]))
    except (OSError, ValueError, IndexError):
        return None


def e47_roots(package, project, proc_root=Path('/proc')):
    expected = {(package / name).resolve() for name in ('run_e47.py','all_steps.py')}
    matched = {}
    for item in proc_root.iterdir():
        if not item.name.isdigit():
            continue
        try:
            tokens = [x.decode() for x in (item / 'cmdline').read_bytes().split(b'\0') if x]
            def resolve_token(token):
                path = Path(token)
                return path.resolve() if path.is_absolute() else ((item/'cwd').resolve()/path).resolve()
            scripts = {resolve_token(t) for t in tokens[1:] if t.endswith('.py')}
            if not scripts.intersection(expected):
                continue
            mode = tokens[tokens.index('--mode') + 1] if '--mode' in tokens else None
            if (package/'all_steps.py').resolve() in scripts:
                mode = 'all'  # controller holds old execution lock even between stages
            if mode not in ('run', 'eval', 'all'):
                continue
            if '--project' in tokens and resolve_token(tokens[tokens.index('--project')+1]) != project.resolve():
                continue
            info = identity(item.name, proc_root)
            if info is not None:
                matched[info['pid']] = {**info, 'mode': mode}
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError, OSError, UnicodeDecodeError):
            continue
    roots = []
    for info in matched.values():
        parent, seen, child = info['ppid'], {info['pid']}, False
        while parent > 1 and parent not in seen:
            seen.add(parent)
            if parent in matched and matched[parent]['start_ticks'] <= info['start_ticks']:
                child = True
                break
            value = identity(parent, proc_root)
            if value is None:
                break
            parent = value['ppid']
        if not child:
            roots.append(info)
    return sorted(roots, key=lambda x: x['pid'])


def gpu_status():
    p = subprocess.run(['nvidia-smi', '-i', '0', '--query-compute-apps=pid',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
    if p.returncode:
        raise RuntimeError('GPU process query failed: ' + p.stderr)
    pids = []
    for line in p.stdout.splitlines():
        value = line.strip()
        if value.isdigit():
            pids.append(int(value))
        elif value and 'No running processes found' not in value:
            raise RuntimeError('Cannot determine GPU processes: ' + value)
    m = subprocess.run(['nvidia-smi', '-i', '0', '--query-gpu=memory.free',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
    if m.returncode:
        raise RuntimeError('GPU memory query failed: ' + m.stderr)
    try:
        free = float(m.stdout.strip()) / 1024
    except ValueError:
        raise RuntimeError('Invalid GPU memory response: ' + m.stdout)
    if not math.isfinite(free) or free < 0:
        raise RuntimeError('Invalid/nonfinite GPU free memory')
    return dict(idle=not pids and free >= 30., compute_pids=pids, free_GiB=free)


def observe_gpu():
    # A transient SMI/driver query failure is not permission to start and need
    # not discard an overnight queue. Record unknown/busy, then try next poll.
    try:
        return gpu_status()
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        return dict(idle=False,compute_pids=None,free_GiB=None,query_error=str(error))


def prior_outcome(project):
    """Read-only observation; missing/failed receipts never block this queue."""
    reports = sorted((project/'comparison_reports').glob('E47S2_RUN_*'))
    if not reports:
        return dict(report=None, status='no_report_yet')
    report = reports[-1]
    answer = dict(report=str(report), status='unknown')
    for name in ('training.json', 'failure.json', 'comparison.json'):
        path = report/name
        if not path.exists():
            continue
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
            if name == 'training.json':
                answer.update(training_status=value.get('status'), epochs=value.get('completed_epochs'))
            elif name == 'failure.json':
                answer.update(status='failed', error=str(value.get('error', value))[-2500:])
            else:
                answer['evaluation_status'] = value.get('status')
        except (ValueError, OSError) as error:
            answer[name] = 'unreadable: ' + str(error)
    if answer['status'] != 'failed':
        answer['status'] = answer.get('training_status', 'unknown')
    return answer


def queue(args, fcntl):
    common = ['--project', str(args.project), '--source', str(args.source),
              '--data', str(args.data), '--dataset', str(args.dataset), '--device', 'cuda:0', '--workers', '8']
    verify = [sys.executable, '-u', str(HERE/'run_e48.py'), '--mode', 'verify-gates', *common]
    subprocess.run(verify, check=True)  # does not create or repair any gate
    run = args.project/'runs'/EXPERIMENT
    if run.exists():
        raise RuntimeError('E48 run already exists; no duplicate/overwrite: ' + str(run))
    out = args.project/'logs'/('queue_e48_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    state = dict(status='waiting', started_at=datetime.now().isoformat(),
                 policy='E47 seed2 success OR failure -> process exit + GPU idle -> E48',
                 e48_checks_and_smoke_valid=True, report_dir=str(out))
    write(out/'state.json', state)
    print('E48 QUEUE ACTIVE: E47 seed2 failure does not block E48. STATE=' + str(out/'state.json'), flush=True)
    start, last_print, stable = time.monotonic(), -60., 0
    prior_lock = None
    try:
        while True:
            roots = e47_roots(args.e47_package, args.project)
            gpu = observe_gpu() if not roots else None
            released = True
            # Optional old execution lock: absent lock is not an error.
            lock_path = args.project/'.e47_seed2_execution.lock'
            if not roots and lock_path.is_file() and prior_lock is None:
                prior_lock = open(lock_path, 'r+')
                try:
                    fcntl.flock(prior_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    prior_lock.close(); prior_lock = None
                    released = False
            ready = not roots and released and gpu is not None and gpu['idle']
            stable = stable+1 if ready else 0
            if not ready and prior_lock is not None:
                prior_lock.close(); prior_lock = None
            state.update(e47=prior_outcome(args.project), root_jobs=roots, gpu=gpu,
                         stable_idle_polls=stable, updated_at=datetime.now().isoformat())
            write(out/'state.json', state)
            if time.monotonic()-last_print >= 30 or ready:
                print(f'E48 WAIT {state["updated_at"]} E47 seed2={state["e47"]} '
                      f'roots={roots} GPU={gpu} idle_checks={stable}/2', flush=True)
                last_print = time.monotonic()
            if stable >= 2:
                subprocess.run(verify, check=True)  # detect changes while waiting
                if run.exists():
                    raise RuntimeError('E48 appeared during wait; no second launch')
                current = e47_roots(args.e47_package, args.project)
                final_gpu = observe_gpu() if not current else None
                if not current and final_gpu['idle']:
                    break
                # A transient context/restarted job is a reason to keep waiting,
                # not abandon the user's overnight queue.
                print('E48 final GPU check is busy; continuing to wait', flush=True)
                stable = 0
                if prior_lock is not None:
                    prior_lock.close(); prior_lock = None
            if time.monotonic()-start > args.timeout_hours*3600:
                raise TimeoutError('Queue timeout; no E48 training launched')
            time.sleep(args.poll_seconds)
        state.update(status='running', training_started_at=datetime.now().isoformat())
        write(out/'state.json', state)
        print('E47 seed2 EXITED AND GPU RELEASED. STARTING E48 FORMAL 200 EPOCHS ' + state['training_started_at'], flush=True)
        child = subprocess.Popen(['bash', str(HERE/'run_e48.sh'), 'run', *common])
        state['e48_pid'] = child.pid; write(out/'state.json', state)
        code = child.wait()
        state.update(status='completed' if code == 0 else 'failed', exit_code=code,
                     finished_at=datetime.now().isoformat())
        write(out/'state.json', state)
        if code:
            raise RuntimeError(f'E48 exited {code}; read the same launcher log; do not overwrite its run')
        print('E48 TRAINING AND OFFICIAL EVALUATION COMPLETED '+state['finished_at'], flush=True)
    except BaseException as error:
        state.update(status='failed', error=str(error), failed_at=datetime.now().isoformat())
        write(out/'state.json', state)
        raise
    finally:
        if prior_lock is not None:
            prior_lock.close()


def main():
    if sys.platform != 'linux':
        raise RuntimeError('AutoDL Linux queue; local tests do not launch training')
    import fcntl
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=Path('/root/autodl-tmp/yolo11_project'))
    p.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/yolo11_src'))
    p.add_argument('--data', type=Path, default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019.yaml'))
    p.add_argument('--dataset', type=Path, default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val'))
    p.add_argument('--e47-package', type=Path, default=Path('/root/autodl-tmp/E47_Seed2_AutoDL'))
    p.add_argument('--poll-seconds', type=int, default=15)
    p.add_argument('--timeout-hours', type=float, default=36.)
    p.add_argument('--inspect', action='store_true')
    args = p.parse_args()
    if args.poll_seconds < 5 or args.timeout_hours <= 0:
        raise ValueError('Invalid polling/timeout setting')
    if args.inspect:
        print(json.dumps(dict(E47_seed2=e47_roots(args.e47_package, args.project),
                              outcome=prior_outcome(args.project), GPU=gpu_status()), indent=2), flush=True)
        return
    with open(HERE/'.e48_queue.lock', 'a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('An E48 queue is already active')
        queue(args, fcntl)


if __name__ == '__main__':
    main()
