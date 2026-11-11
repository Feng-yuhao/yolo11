"""Opt-in AutoDL shutdown: saved reports, exited child, two idle checks.
Tests inject fake probes/runners: no tests may execute a real shutdown.
"""
import csv
from datetime import datetime, timezone, timedelta
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def capture(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError('GPU state unavailable: '+result.stderr.strip())
    return result.stdout


def gpu_snapshot():
    rows = list(csv.reader(io.StringIO(capture([
        'nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits']))))
    if len(rows) != 1 or len(rows[0]) != 2:
        raise RuntimeError('This entry supports exactly one server GPU')
    total, free = (int(x.strip()) for x in rows[0])
    values = capture(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    pids = []
    for line in values.splitlines():
        if not line.strip():
            continue
        if not line.strip().isdigit():
            raise RuntimeError('Cannot reliably enumerate GPU PIDs: '+line)
        pids.append(int(line.strip()))
    return dict(total_MiB=total, free_MiB=free, compute_pids=sorted(set(pids)))


def active_python_jobs():
    """Catch training/eval still starting, before CUDA PIDs become visible."""
    if not Path('/proc').is_dir():
        raise RuntimeError('Linux /proc unavailable; cannot safely shut down')
    jobs = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            raw = (proc/'cmdline').read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        # Incomplete scans (e.g. permissions) fail closed, not "idle".
        argv = [x.decode('utf-8', errors='replace') for x in raw.split(b'\0') if x]
        if not argv or 'python' not in Path(argv[0]).name.lower():
            continue
        text = ' '.join(argv[1:])
        if re.search(r'(train[^ /]*\.py|yolo[^ /]*\.py|run_[^ /]*\.py|formal_[^ /]*\.py|eval[^ /]*\.py|'
                     r'diagnos[^ /]*\.py|queue[^ /]*\.py|controller\.py|torchrun|--mode[ =]+(?:run|train|eval))', text):
            jobs.append(dict(pid=int(proc.name), command=text[:500]))
    return jobs


def require_idle(snapshot, jobs, admission=False):
    if snapshot['compute_pids'] or jobs:
        raise RuntimeError('Other GPU/training/evaluation jobs active: '+str((snapshot, jobs)))
    if snapshot['total_MiB'] < 30*1024:
        raise RuntimeError('Requires the existing 32GiB server GPU')
    needed = 28*1024 if admission else snapshot['total_MiB']-2048
    if snapshot['free_MiB'] < needed:
        raise RuntimeError('GPU not released/idle: '+str(snapshot))


def auto_shutdown(enabled, state_dir, cancel_file, delay=60, probe=gpu_snapshot,
                  jobs=active_python_jobs, runner=subprocess.run, sleeper=time.sleep,
                  platform_name=None, command=Path('/usr/bin/shutdown')):
    state_dir, cancel_file = Path(state_dir), Path(cancel_file)
    platform_name = sys.platform if platform_name is None else platform_name
    status = dict(enabled=enabled, at=now(), command=[str(command)], reason='job_finished_or_failed')
    marker = state_dir/'shutdown_status.json'

    def save(value, **extra):
        status.update(status=value, **extra, at=now())
        write(marker, status)
        print('AUTO SHUTDOWN '+value+' '+json.dumps(extra, ensure_ascii=False), flush=True)
        return dict(status)

    if not enabled:
        return save('disabled')
    if platform_name != 'linux' or not command.is_file():
        return save('skipped_unsupported_platform_or_missing_autodl_command')
    try:
        # AutoDL sometimes installs executable ASCII WITHOUT a shebang. Passing
        # that file directly to exec gives Errno8; explicitly use bash only for
        # this platform-owned script. Never invoke a fallback destructive command.
        header=command.read_bytes()[:128]
        if header.startswith((b'#!',b'\x7fELF')):
            invocation=[str(command)]
        elif header and b'\0' not in header:
            invocation=['/bin/bash',str(command)]
        else:
            return save('skipped_unrecognized_shutdown_command')
        status['command']=invocation
        if cancel_file.exists():
            return save('cancelled_by_KEEP_RUNNING_file')
        require_idle(probe(), jobs())
        save('countdown', seconds=delay, cancel_file=str(cancel_file))
        for remaining in range(delay, 0, -5):
            if cancel_file.exists():
                return save('cancelled_by_KEEP_RUNNING_file')
            sleeper(min(5, remaining))
        if cancel_file.exists():
            return save('cancelled_by_KEEP_RUNNING_file')
        require_idle(probe(), jobs())
        save('requesting_shutdown')
        sys.stdout.flush()
        sys.stderr.flush()
        if hasattr(os, 'sync'):
            os.sync()
        result = runner(invocation, capture_output=True, text=True, timeout=30)
        if result.returncode:
            return save('shutdown_command_failed', returncode=result.returncode,
                        stdout=result.stdout, stderr=result.stderr)
        # Request accepted does not prove that platform billing has stopped.
        return save('shutdown_command_returned_zero', stdout=result.stdout)
    except Exception as error:
        return save('skipped_safety_check_or_command_error', error=str(error))
