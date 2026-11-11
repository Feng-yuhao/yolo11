"""Single job with live logs; opt-in shutdown on success/error, never on cancel."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
from shutdown_safety import now, write, gpu_snapshot, active_python_jobs, require_idle, auto_shutdown

HERE = Path(__file__).resolve().parent
MODES = {'all': ('check', 'smoke', 'verify-gates', 'run'),
         'run': ('verify-gates', 'run'), 'check': ('check',), 'smoke': ('smoke',),
         'verify-gates': ('verify-gates',), 'eval': ('eval',)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=MODES)
    parser.add_argument('--auto-shutdown', action='store_true', help='all/run success OR failure: save state, shutdown if idle')
    parser.add_argument('--project', type=Path, default=Path('/root/autodl-tmp/yolo11_project'))
    parser.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/yolo11_src'))
    parser.add_argument('--data', type=Path, default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019.yaml'))
    parser.add_argument('--dataset', type=Path, default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val'))
    parser.add_argument('--weights', type=Path)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('AutoDL controller is Linux-only; local tests use run_experiment.py')
    if args.auto_shutdown and args.mode not in ('all', 'run'):
        parser.error('--auto-shutdown only allowed with all/run; check/smoke/eval never shut down')
    if args.mode == 'eval' and args.weights is None:
        parser.error('eval requires --weights /absolute/path/best.pt')
    import fcntl
    lock_path = args.project/'logs/.e60_controller.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # Duplicate launch must NEVER shut down the active controller's job.
        lock.close()
        raise RuntimeError('E60 controller already active; no shutdown requested')
    session = args.project/'logs'/('e60_'+args.mode+'_'+now().replace(':', '').replace('+', '_'))
    session.mkdir(parents=True, exist_ok=False)
    state = dict(status='admission', mode=args.mode, pid=os.getpid(), started_at=now(),
                 auto_shutdown=args.auto_shutdown, stage_exit_codes={}, logs_dir=str(session))
    write(session/'state.json', state)
    write(args.project/'logs/latest_e60_controller.json', dict(session=str(session), pid=os.getpid()))
    cancel_file = args.project/'KEEP_RUNNING_E60'
    print(f'E60 controller logs: {session}\nAuto shutdown requested: {args.auto_shutdown}\n'
          f'Cancel shutdown at any time: touch {cancel_file}', flush=True)
    child = None
    admitted = False
    cancelled = False
    exit_code = 1

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Controller received signal {signum}')

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        require_idle(gpu_snapshot(), active_python_jobs(), admission=True)
        admitted = True
        for mode in MODES[args.mode]:
            state.update(status='running', stage=mode)
            write(session/'state.json', state)
            command = [sys.executable, '-u', str(HERE/'run_experiment.py'), '--mode', mode,
                       '--project', str(args.project), '--source', str(args.source),
                       '--data', str(args.data), '--dataset', str(args.dataset)]
            if args.weights is not None:
                command.extend(['--weights', str(args.weights)])
            log = session/(mode+'.log')
            print(f'[{now()}] START {mode}; live output below; log={log}', flush=True)
            with log.open('w', encoding='utf-8', buffering=1) as handle:
                child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, bufsize=1, start_new_session=True)
                state['child_pid'] = child.pid
                write(session/'state.json', state)
                for line in child.stdout:
                    handle.write(line)
                    print(line, end='', flush=True)
                code = child.wait()
                handle.flush()
                os.fsync(handle.fileno())
            child = None
            state['stage_exit_codes'][mode] = code
            write(session/'state.json', state)
            if code:
                raise RuntimeError(f'{mode} failed (exit={code}); log: {log}; no gate bypass/retraining')
            print(f'[{now()}] END {mode}: exit=0', flush=True)
        state['status'] = 'completed'
        exit_code = 0
    except KeyboardInterrupt:
        cancelled = True
        state.update(status='cancelled', error=traceback.format_exc())
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                state['owned_child_still_active'] = child.pid
        exit_code = 130
    except Exception:
        state.update(status='failed', error=traceback.format_exc())
        print(state['error'], file=sys.stderr, flush=True)
    finally:
        state['finished_at'] = now()
        write(session/'state.json', state)
        print(f'[{now()}] CONTROLLER {state["status"]}; state={session/"state.json"}', flush=True)
        enabled = args.auto_shutdown and admitted and not cancelled
        status = auto_shutdown(enabled, session, cancel_file)
        state['shutdown'] = status
        write(session/'state.json', state)
        if enabled and status['status'] not in ('shutdown_command_returned_zero', 'requesting_shutdown'):
            print('IMPORTANT: shutdown not confirmed; check AutoDL console and shut down manually '
                  'when safe. Hourly billing may continue.', flush=True)
        lock.close()
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
