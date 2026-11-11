"""Serial seed2 baseline -> E60 -> official val/test-dev; no remote launch."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile
import common as c
from protocol import (runtime, signature, issue_gate, verify_gate, gate_path, actual_run_name,
                      read, write, now, require, DATA, VARIANTS)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('all','check','smoke','verify-gates','run','eval'), nargs='?', default='all')
    parser.add_argument('--project', type=Path, default=c.PROJECT)
    parser.add_argument('--source', type=Path, default=c.SOURCE)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--val-dataset', type=Path, default=DATA.parent / 'VisDrone2019-DET-val')
    parser.add_argument('--dataset', type=Path, default=DATA.parent / 'VisDrone2019-DET-test-dev')
    parser.add_argument('--training-report', type=Path, help='eval-only recovery: original complete seed2 train report root')
    parser.add_argument('--auto-shutdown', action='store_true', help='Explicit opt-in AFTER terminal state and idle checks')
    parser.add_argument('--local-test', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--local-size', type=int, default=128, help=argparse.SUPPRESS)
    parser.add_argument('--local-epochs', type=int, default=2, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = arguments()
    require(not args.auto_shutdown or args.mode in ('all','run','eval'), 'check/smoke never enable shutdown')
    require(not args.local_test or not args.auto_shutdown, 'Private fixtures cannot request shutdown')
    runtime(args, 'controller')
    from process_lock import acquire
    (args.project / 'logs').mkdir(parents=True, exist_ok=True)
    lock_path = args.project / 'logs/.yolo26_seed2_serial.lock'
    try:
        lock = acquire(lock_path,args.local_test)
    except (BlockingIOError,PermissionError):
        raise RuntimeError('Another seed2 serial controller is active; do not start twice')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    prefix = 'PRIVATE_' if args.local_test else ''
    report = args.project / 'comparison_reports' / (prefix + 'YOLO26_P2_E60_S2_' + args.mode.upper() + '_' + stamp)
    session = args.project / 'logs' / (prefix + 'yolo26_seed2_' + args.mode + '_' + stamp)
    report.mkdir(parents=True, exist_ok=False)
    session.mkdir(parents=True, exist_ok=False)
    state = dict(status='preflight', stage='check', mode=args.mode, seed=2, execution='serial',
                 session=str(session), report_root=str(report), started_at=now(), jobs={},
                 training_report=str(args.training_report) if args.mode == 'eval' else str(report),
                 auto_shutdown=args.auto_shutdown, local_fixture=args.local_test, child_active=False, admitted=False)
    index_path = args.project / 'logs' / (prefix + 'latest_yolo26_seed2_serial.json')
    write(index_path, dict(session=str(session), report_root=str(report)))
    errors = []
    def save():
        write(session / 'state.json', state)
        write(report / 'state.json', state)

    def job(mode, variant='e60', split=None, train_report=None, fatal=True):
        key = 'check' if mode == 'check' else variant + '_' + (split.replace('-','_') if mode == 'eval' else mode)
        destination = report / key
        command = [sys.executable, '-u', str(c.HERE / 'worker.py'), mode, '--variant', variant,
                   '--project', str(args.project), '--source', str(args.source), '--data', str(args.data),
                   '--val-dataset', str(args.val_dataset), '--dataset', str(args.dataset),
                   '--report', str(destination), '--signature', str(session / 'signature.json')]
        if split:
            command += ['--split', split, '--training-report', str(train_report)]
        if args.local_test:
            command += ['--local-test','--local-size',str(args.local_size),'--local-epochs',str(args.local_epochs)]
        log = session / (key + '.log')
        print(f'{now()} START {key}; live output below; log={log}', flush=True)
        state['stage'] = key
        with log.open('w', encoding='utf-8') as handle:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding='utf-8', errors='replace', bufsize=1)
            state['child_active'] = True
            state['jobs'][key] = dict(pid=proc.pid, status='running', started_at=now(),
                                      report=str(destination), log=str(log))
            save()
            try:
                for line in proc.stdout:
                    handle.write(line)
                    handle.flush()
                    print(line, end='', flush=True)
                code = proc.wait()
            except BaseException:
                if proc.poll() is None:
                    proc.terminate()
                proc.wait()
                raise
            finally:
                state['child_active'] = False
        state['jobs'][key].update(status='completed' if code == 0 else 'failed', exit_code=code, finished_at=now())
        save()
        if code:
            message = f'{key} failed exit={code}; log={log}; no automatic retry/batch reduction'
            errors.append(message)
            if fatal:
                raise RuntimeError(message)
        return code == 0

    try:
        save()
        if args.auto_shutdown:
            from shutdown_safety import readiness, process_identity
            checked = readiness(c.HERE)
            state.update(shutdown_command=checked['command'], controller_identity=process_identity(os.getpid()))
            save()
        if args.mode in ('all','run'):
            for variant in VARIANTS:
                require(not (args.project / 'runs' / actual_run_name(args,variant)).exists(),
                        'Existing seed2 run: refusing overwrite/restart; use eval for scoring recovery')
        if args.mode == 'eval':
            require(args.training_report is not None, 'eval recovery requires --training-report')
        state['admitted'] = True
        sig = signature(args)
        write(session / 'signature.json', sig)
        write(report / 'preflight.json', sig)
        with zipfile.ZipFile(report / 'actual_source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in list(read(c.HERE / 'manifest.json')) + ['manifest.json']:
                if not name.startswith('weights/'):
                    archive.write(c.HERE / name, name)
        if args.mode in ('all','check','smoke'):
            state['status'] = 'checking'; save()
            job('check')
            if args.mode != 'check':
                state['status'] = 'smoking'; save()
                for variant in VARIANTS:
                    job('smoke',variant)
                issue_gate(args,sig,report)
                print(f'{now()} BOTH CUDA CHECK / REAL SMOKE GATES PASSED; batch16/val32 unchanged', flush=True)
        if args.mode in ('all','verify-gates','run'):
            verify_gate(args,sig)
            print(f'{now()} SEED2 GATES VALID', flush=True)
        if args.mode in ('all','run'):
            state['status'] = 'training'; save()
            for variant in VARIANTS:
                job('train', variant, fatal=False)
        if args.mode in ('all','run','eval'):
            state['status'] = 'evaluating'; save()
            train_root = args.training_report if args.mode == 'eval' else report
            for variant in VARIANTS:
                if args.mode != 'eval' and state['jobs'][variant + '_train']['exit_code'] != 0:
                    continue
                for split in ('val','test-dev'):
                    job('eval', variant, split, train_root, fatal=False)
            if not errors:
                from summarize import summarize
                summarize(report,args.local_test)
        if errors:
            write(report / 'PARTIAL_RESULTS.json',dict(status='failed',errors=errors,completed_jobs=state['jobs']))
            raise RuntimeError('Some stages failed; valid partial outputs retained: ' + ' | '.join(errors))
        state.update(status='completed', stage='finished', finished_at=now()); save()
        print(f'{now()} ALL COMPLETED REPORT={report}; auto_shutdown={args.auto_shutdown}', flush=True)
    except BaseException:
        state.update(status='failed', error=traceback.format_exc(), finished_at=now()); save()
        raise
    finally:
        for path in session.glob('*.log'):
            if path.name != 'shutdown_watch.log':
                shutil.copyfile(path,report / path.name)
        if args.auto_shutdown and state['admitted'] and not state['child_active']:
            with (session / 'shutdown_watch.log').open('a',encoding='utf-8') as handle:
                subprocess.Popen([sys.executable,'-u',str(c.HERE / 'shutdown_after.py'),
                                  '--session',str(session),'--project',str(args.project)],
                                 stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            print('AUTO SHUTDOWN watcher queued; waits for controller exit, preserved outputs and idle GPU.', flush=True)
        lock.close()


if __name__ == '__main__':
    main()
