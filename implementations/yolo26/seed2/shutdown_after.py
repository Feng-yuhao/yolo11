"""Opt-in AutoDL shutdown AFTER parent exits. Local tests only inject fake calls."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile
from shutdown_safety import (now,read,write,sha,process_identity,process_snapshot,gpu_snapshot,
                             require_idle,shutdown_invocation)

HERE = Path(__file__).resolve().parent
RUNS = {v:'yolo26s_p2_' + v + '_img640_seed2_200e' for v in ('baseline','e60')}


def contained(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path == root or not path.is_relative_to(root):
        raise RuntimeError('Unexpected result path: ' + str(path))
    return path


def lock_free(path):
    import fcntl
    with Path(path).open('r+b') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle, fcntl.LOCK_UN)
    return True


def verify_completed(project, report, train_report=None):
    errors = []
    train_report = Path(train_report) if train_report else Path(report)
    for variant, name in RUNS.items():
        try:
            run = Path(project) / 'runs' / name
            completion = read(train_report / (variant + '_train') / 'training_completion.json')
            if completion['status'] != 'training_completed' or completion['epochs'] != 200 or completion['seed'] != 2:
                raise RuntimeError('Not a completed200 seed2 run')
            with (run / 'results.csv').open(encoding='utf-8-sig') as handle:
                rows = [{k.strip():v.strip() for k,v in row.items()} for row in csv.DictReader(handle)]
            if [int(row['epoch']) for row in rows] != list(range(1,201)):
                raise RuntimeError('CSV epochs are not exactly1..200')
            for filename in ('best.pt','last.pt'):
                weights = run / 'weights' / filename
                with zipfile.ZipFile(weights) as archive:
                    if archive.testzip() is not None:
                        raise RuntimeError('Checkpoint ZIP CRC failed')
            digest = sha(run / 'weights/best.pt')
            if completion['best_sha256'] != digest:
                raise RuntimeError('Best checkpoint changed')
            for split, count in (('val',548),('test_dev',1610)):
                metric = read(Path(report) / (variant + '_' + split) / 'official_metrics.json')
                if (metric['status'] != 'completed' or metric['seed'] != 2 or metric['variant'] != variant
                        or metric['dataset']['images'] != count or metric['checkpoint_sha256'] != digest):
                    raise RuntimeError('Incomplete/mismatched official score: ' + split)
            if read(Path(report) / 'comparison.json')['status'] != 'completed':
                raise RuntimeError('Comparison missing')
        except (OSError,ValueError,KeyError,RuntimeError,zipfile.BadZipFile) as error:
            errors.append(variant + ': ' + str(error))
    return errors


def preserve(project, session, report, terminal):
    paths = []
    for root in [report] + [project / 'runs' / name for name in RUNS.values()]:
        if root.is_dir():
            paths.extend(p for p in root.rglob('*') if p.is_file())
    paths += [p for p in session.glob('*.log') if p.name != 'shutdown_watch.log']
    files = []
    for path in sorted(set(paths)):
        contained(path,project)
        before = path.stat()
        digest = sha(path)
        after = path.stat()
        if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
            raise RuntimeError('File still changing: ' + str(path))
        files.append(dict(path=str(path),bytes=after.st_size,sha256=digest))
    result = dict(status='available_artifacts_preserved',at=now(),terminal_status=terminal,files=files,
                  boundary='Only already saved outputs retained; no missing weights or scores synthesized.')
    write(session / 'PRESERVATION.json',result)
    return result


def run_watch(session, project, *, runner=subprocess.run, sleeper=time.sleep, clock=time.monotonic,
              identity=process_identity, gpu=gpu_snapshot, processes=None, locks=lock_free,
              platform_name=None, command=Path('/usr/bin/shutdown'), delay=60):
    project = Path(project).resolve()
    session = contained(session,project / 'logs')
    marker = session / 'shutdown_status.json'
    status = dict(enabled=True,at=now())
    def save(label,**extra):
        status.update(status=label,at=now(),**extra)
        write(marker,status)
        print('AUTO SHUTDOWN ' + label,flush=True)
        return dict(status)
    try:
        if (sys.platform if platform_name is None else platform_name) != 'linux':
            return save('blocked_non_linux')
        state = read(session / 'state.json')
        if not (state.get('auto_shutdown') and state.get('admitted') and not state.get('local_fixture')
                and state.get('status') in ('completed','failed') and not state.get('child_active')):
            return save('not_authorized_or_not_terminal')
        if marker.is_file() and read(marker).get('status') in ('requesting_shutdown','shutdown_command_returned_zero'):
            return read(marker)
        report = contained(state['report_root'],project / 'comparison_reports')
        if not report.name.startswith('YOLO26_P2_E60_S2_'):
            raise RuntimeError('Not a seed2 report')
        cancel = project / 'KEEP_RUNNING_YOLO26_SEED2'
        expected = state['controller_identity']
        save('waiting_parent_exit')
        deadline = clock() + 60
        while identity(expected['pid']) == expected:
            if clock() >= deadline:
                return save('blocked_parent_still_active')
            sleeper(2)
        lock = project / 'logs/.yolo26_seed2_serial.lock'
        def bound():
            index = read(project / 'logs/latest_yolo26_seed2_serial.json')
            return Path(index['session']).resolve() == session and locks(lock)
        if not bound():
            return save('cancelled_new_session_or_active_controller')
        if state['status'] == 'completed':
            errors = verify_completed(project,report,state.get('training_report'))
            if errors:
                return save('blocked_incomplete_completed_outputs',errors=errors)
        info = shutdown_invocation(command)
        if info['sha256'] != state['shutdown_command']['sha256']:
            return save('blocked_platform_helper_changed')
        kept = preserve(project,session,report,state['status'])
        save('preserved_waiting_idle',preserved_files=len(kept['files']))
        processes = processes or (lambda:process_snapshot(HERE))
        deadline = clock() + 1800
        while clock() < deadline:
            if cancel.exists():
                return save('cancelled_by_KEEP_RUNNING')
            if not bound():
                return save('cancelled_new_session')
            try:
                require_idle(gpu(),processes())
            except RuntimeError as error:
                save('waiting_other_jobs_or_GPU_release',detail=str(error)); sleeper(10); continue
            save('countdown',seconds=delay)
            ready = True
            for left in range(delay,0,-5):
                sleeper(min(5,left))
                if cancel.exists():
                    return save('cancelled_by_KEEP_RUNNING')
                if not bound():
                    return save('cancelled_new_session')
                try:
                    require_idle(gpu(),processes())
                except RuntimeError:
                    ready = False; break
            if not ready:
                continue
            if shutdown_invocation(command)['sha256'] != info['sha256']:
                return save('blocked_platform_helper_changed')
            # The production lock closes the same-launcher admission race.
            import contextlib
            guard = contextlib.nullcontext()
            if platform_name is None:
                from process_lock import acquire
                guard = acquire(lock)
            with guard:
                if cancel.exists() or Path(read(project / 'logs/latest_yolo26_seed2_serial.json')['session']).resolve() != session:
                    return save('cancelled_new_session_or_KEEP_RUNNING')
                require_idle(gpu(),processes())
                if shutdown_invocation(command)['sha256'] != info['sha256']:
                    return save('blocked_platform_helper_changed')
                save('requesting_shutdown',request_saved_before_invocation=True)
                sys.stdout.flush(); sys.stderr.flush()
                if platform_name is None and hasattr(os,'sync'):
                    os.sync()
                result = runner(info['args'],capture_output=True,text=True,timeout=30)
            return save('shutdown_command_returned_zero' if result.returncode == 0 else 'shutdown_command_failed',
                        returncode=result.returncode,platform_poweroff_and_billing_stop_not_confirmed=True)
        return save('blocked_idle_wait_timeout')
    except BaseException:
        return save('shutdown_watcher_error',error=traceback.format_exc())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--project',type=Path,required=True)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Server-only; local tests must inject a fake runner')
    import fcntl
    with (args.session / 'shutdown_watcher.lock').open('a+') as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(1)
        run_watch(args.session,args.project)
