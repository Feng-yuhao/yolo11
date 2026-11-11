"""One TOOD_TD seed1 pipeline; opt-in power-off only after final artifacts/exit."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
from shutdown_safety import now,read,write,gpu_snapshot,process_snapshot,require_idle,shutdown_invocation,process_identity,readiness

HERE=Path(__file__).resolve().parent
MODES={'all':('check','smoke','verify-gates','train','val','test','summary'),
       'run':('verify-gates','train','val','test','summary'),
       'check':('check',),'smoke':('smoke',),'verify-gates':('verify-gates',),
       'eval':('val','test','summary')}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=tuple(MODES)+('shutdown-check',))
    parser.add_argument('--auto-shutdown',action='store_true')
    parser.add_argument('--project',type=Path,default=Path('/root/autodl-tmp/yolo11_project'))
    parser.add_argument('--source',type=Path,default=Path('/root/autodl-tmp/yolo11_src'))
    parser.add_argument('--data',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019.yaml'))
    parser.add_argument('--dataset',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val'))
    parser.add_argument('--test-dataset',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-test-dev'))
    parser.add_argument('--weights',type=Path)
    parser.add_argument('--shutdown-wait-seconds',type=int,default=1800)
    args=parser.parse_args()
    for name in ('project','source','data','dataset','test_dataset'):
        setattr(args,name,getattr(args,name).resolve())
    if args.weights is not None: args.weights=args.weights.resolve()
    if sys.platform!='linux': parser.error('Controller is Linux-only; use local_validate.py for developer fixtures')
    if args.auto_shutdown and args.mode not in ('all','run'):
        parser.error('--auto-shutdown is explicitly allowed only with all/run')
    if args.shutdown_wait_seconds<60: parser.error('Shutdown idle wait must be at least60 seconds')
    if args.mode=='shutdown-check':
        import json
        print(json.dumps(readiness(HERE),indent=2,ensure_ascii=False),flush=True)
        return 0
    if args.mode=='eval' and args.weights is None: parser.error('eval requires --weights')
    if args.weights is not None and args.mode!='eval': parser.error('--weights is only for scoring recovery, never formal initialization')
    import fcntl
    lock_path=args.project/'logs/.tood_td_controller.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    lock=lock_path.open('a+')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Another TOOD_TD controller is active; duplicate NEVER schedules shutdown')
    stamp=now().replace(':','').replace('+','_')+'_'+str(os.getpid())
    session=args.project/'logs'/('tood_td_'+args.mode+'_'+stamp)
    report_root=args.project/'comparison_reports'/('TOOD_TD_'+args.mode.upper()+'_'+stamp)
    session.mkdir(parents=True,exist_ok=False); report_root.mkdir(parents=True,exist_ok=False)
    state=dict(status='admission',mode=args.mode,pid=os.getpid(),started_at=now(),auto_shutdown=args.auto_shutdown,
               admitted=False,stage_exit_codes={},reports={},session=str(session),report_root=str(report_root),
               controller_identity=process_identity(os.getpid()),child_active=False)
    write(session/'state.json',state)
    write(args.project/'logs/latest_tood_td_controller.json',dict(session=str(session),report_root=str(report_root),pid=os.getpid()))
    cancel=args.project/'KEEP_RUNNING_TOOD_TD'
    print(f'TOOD_TD logs: {session}\nTOOD_TD report root: {report_root}\nAUTO SHUTDOWN enabled={args.auto_shutdown}\n'
          f'Cancel power-off: touch {cancel}',flush=True)
    child=None; cancelled=False; exit_code=1
    def interrupt(signum,frame): raise KeyboardInterrupt(f'Controller received {signum}')
    signal.signal(signal.SIGTERM,interrupt); signal.signal(signal.SIGINT,interrupt)
    try:
        # Check before model allocation, so an unavailable helper is caught now.
        if args.auto_shutdown: state['shutdown_command']=shutdown_invocation()
        gpu=gpu_snapshot(); processes=process_snapshot(HERE)
        require_idle(gpu,processes,admission=True)
        state.update(admitted=True,admission_gpu=gpu,ignored_services=processes['ignored_services'])
        for mode in MODES[args.mode]:
            out=report_root/mode
            state.update(status='running',stage=mode)
            state['reports'][mode]=str(out); write(session/'state.json',state)
            command=[sys.executable,'-u',str(HERE/'run_experiment.py'),'--mode',mode,
                     '--project',str(args.project),'--source',str(args.source),'--data',str(args.data),
                     '--dataset',str(args.dataset),'--test-dataset',str(args.test_dataset),
                     '--report-dir',str(out),'--report-root',str(report_root)]
            if args.weights is not None: command+=['--weights',str(args.weights)]
            log=session/(mode+'.log')
            print(f'[{now()}] START {mode}; log={log}',flush=True)
            with log.open('w',encoding='utf-8',buffering=1) as handle:
                child=subprocess.Popen(command,cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                       text=True,bufsize=1,start_new_session=True)
                state.update(child_pid=child.pid,child_active=True); write(session/'state.json',state)
                for line in child.stdout:
                    handle.write(line); print(line,end='',flush=True)
                code=child.wait(); handle.flush(); os.fsync(handle.fileno())
            child=None
            state.update(child_active=False); state['stage_exit_codes'][mode]=code
            write(session/'state.json',state)
            if code: raise RuntimeError(f'{mode} failed exit={code}; no gate bypass or automatic retry; {log}')
            print(f'[{now()}] END {mode}: exit=0',flush=True)
        if args.mode in ('all','run','eval'):
            from shutdown_after import verify_completed
            errors=verify_completed(args.project,report_root)
            if errors: raise RuntimeError('Final result proof incomplete: '+str(errors))
            state['summary']=str(report_root/'summary.json')
            with (report_root/'REPORT.md').open('a',encoding='utf-8') as handle:
                handle.write('\nController logs: '+str(session)+'\n')
                handle.flush(); os.fsync(handle.fileno())
        state.update(status='completed'); exit_code=0
    except KeyboardInterrupt:
        cancelled=True; state.update(status='cancelled',error=traceback.format_exc()); exit_code=130
    except BaseException:
        state.update(status='failed',error=traceback.format_exc())
        print(state['error'],file=sys.stderr,flush=True)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: state['owned_child_still_active']=child.pid
        state['child_active']=bool(child is not None and child.poll() is None)
        state['finished_at']=now(); write(session/'state.json',state)
        enabled=args.auto_shutdown and state['admitted'] and not cancelled and not state['child_active']
        if enabled:
            # Detached helper waits for THIS controller identity + lock to exit.
            # Thus its own parent is not a false-positive "other Python job".
            state['shutdown']=dict(status='queued_after_parent_exit',enabled=True,reason='job_finished_or_failed')
            write(session/'state.json',state)
            try:
                with (session/'shutdown_watch.log').open('w',encoding='utf-8') as handle:
                    helper=subprocess.Popen([sys.executable,'-u',str(HERE/'shutdown_after.py'),
                        '--session',str(session),'--project',str(args.project),
                        '--wait-seconds',str(args.shutdown_wait_seconds)],cwd=HERE,
                        stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                state['shutdown']['watch_pid']=helper.pid
                print(f'AUTO SHUTDOWN queued AFTER parent exit; helper PID={helper.pid}; log={session/"shutdown_watch.log"}',flush=True)
            except Exception:
                state['shutdown']=dict(status='helper_launch_failed',enabled=True,error=traceback.format_exc())
                print('AUTO SHUTDOWN helper could not launch; check AutoDL console when safe. Billing may continue.',flush=True)
        else:
            state['shutdown']=dict(enabled=False,status='disabled_or_not_admitted_or_cancelled',
                                   requested=args.auto_shutdown,cancelled=cancelled)
            print('AUTO SHUTDOWN not armed: '+str(state['shutdown']),flush=True)
        write(session/'state.json',state)
        if state['shutdown']['status']=='helper_launch_failed':
            write(session/'shutdown_status.json',state['shutdown'])
        print(f'[{now()}] CONTROLLER {state["status"]}; state={session/"state.json"}',flush=True)
        lock.close()
    return exit_code


if __name__=='__main__': sys.exit(main())
