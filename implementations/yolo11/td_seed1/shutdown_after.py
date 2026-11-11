"""Detached, opt-in single-job shutdown. All local tests inject a fake runner."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile
from shutdown_safety import now,read,write,sha,process_identity,process_snapshot,gpu_snapshot,require_idle,shutdown_invocation
from single_safety import EXPERIMENTS

HERE=Path(__file__).resolve().parent


def contained(path,root):
    path=Path(path).resolve(); root=Path(root).resolve()
    if path==root or not path.is_relative_to(root):
        raise RuntimeError('Result path outside its explicit parent: '+str(path))
    return path


def lock_free(path):
    import fcntl
    if not Path(path).is_file(): raise RuntimeError('Controller lock missing')
    with Path(path).open('r+b') as handle:
        try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return False
        fcntl.flock(handle,fcntl.LOCK_UN)
    return True


def checkpoint_crc(path):
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist())>1024**3:
            raise RuntimeError('Unexpected oversized checkpoint container')
        member=archive.testzip()
        if member: raise RuntimeError('Checkpoint ZIP CRC failure: '+member)
    return dict(zip_crc_valid=True,not_a_torch_model_reload_test=True)


def verify_completed(project,report_root):
    """No torch/GPU allocation; distinguish trained200 from scored/completed."""
    errors=[]; project=Path(project); report=Path(report_root)
    run=project/'runs'/EXPERIMENTS['tood_td']
    try:
        train=read(report/'training_evidence.json'); complete=read(report/'completion.json')
        if (train['status']!='completed' or train['completed_epochs']!=200 or
                complete['status']!='completed' or complete['trained_epochs']!=200 or
                not complete['official_val_completed'] or not complete['official_test_completed']):
            raise RuntimeError('Training or official scoring not completed')
        with (run/'results.csv').open(encoding='utf-8-sig') as handle:
            rows=[{k.strip():v.strip() for k,v in row.items()} for row in csv.DictReader(handle)]
        if [int(row['epoch']) for row in rows]!=list(range(1,201)):
            raise RuntimeError('results.csv is not exactly epochs1..200')
        for name in ('best.pt','last.pt'):
            path=contained(run/'weights'/name,run)
            if path.stat().st_size<1024*1024: raise RuntimeError('Missing/truncated '+name)
            checkpoint_crc(path)
        best=run/'weights/best.pt'; digest=sha(best)
        if train['best_sha256']!=digest or complete['checkpoint_sha256']!=digest:
            raise RuntimeError('Selected best checkpoint binding differs')
        for split,count in (('val',548),('test',1610)):
            value=read(report/split/'official_metrics.json')
            binding=value['binding']; pred=read(report/split/'prediction_manifest.json')
            predictions=report/split/'area_predictions.json'
            if (value['status']!='completed' or binding['seed']!=1 or
                    binding['checkpoint_sha256']!=digest or binding['dataset']['images']!=count or
                    pred['seen_images']!=count or pred['sha256']!=value['prediction_sha256'] or
                    sha(predictions)!=pred['sha256']):
                raise RuntimeError('Official metrics do not match completed TD best: '+split)
        if not (report/'summary.json').is_file() or not (report/'REPORT.md').is_file():
            raise RuntimeError('Final official summary/report missing')
    except (OSError,ValueError,KeyError,RuntimeError,zipfile.BadZipFile) as error:
        errors.append(str(error))
    return errors


def preserve_available(project,session,report_root,terminal_status):
    """Keep/hash every available result. Never invent missing checkpoint/score."""
    project,session,report_root=Path(project),Path(session),Path(report_root)
    run=project/'runs'/EXPERIMENTS['tood_td']
    paths=[]
    for root in (report_root,run):
        if root.is_dir(): paths.extend(path for path in root.rglob('*') if path.is_file())
    # Controller/stage logs are closed before this helper starts its idle test.
    paths.extend(path for path in session.glob('*.log') if path.name!='shutdown_watch.log')
    if (session/'state.json').is_file(): paths.append(session/'state.json')
    inventory=[dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)) for path in sorted(set(paths))]
    payload=dict(status='available_artifacts_preserved',at=now(),terminal_status=terminal_status,
                 files=inventory,run_present=run.is_dir(),
                 completed_output_errors=verify_completed(project,report_root),
                 boundary='Failure preserves partial files only; missing weights/scores are NOT synthesized.')
    write(session/'PRESERVATION.json',payload)
    return payload


def run_watch(session,project,wait_seconds=1800,delay=60,platform_name=None,
              probe=gpu_snapshot,processes=None,identity=process_identity,locks=lock_free,
              runner=subprocess.run,sleeper=time.sleep,clock=time.monotonic,command=Path('/usr/bin/shutdown')):
    session=contained(session,Path(project)/'logs'); project=Path(project).resolve()
    processes=processes or (lambda:process_snapshot(HERE))
    marker=session/'shutdown_status.json'; status=dict(enabled=True,at=now(),pid=os.getpid())
    def save(value,**extra):
        status.update(status=value,at=now(),**extra); write(marker,status)
        print('AUTO SHUTDOWN '+value+' '+str(extra),flush=True)
        return dict(status)
    try:
        if (sys.platform if platform_name is None else platform_name)!='linux':
            return save('blocked_non_linux')
        state=read(session/'state.json')
        if (state.get('mode') not in ('all','run') or not state.get('auto_shutdown') or
                not state.get('admitted') or state.get('status') not in ('completed','failed') or
                state.get('child_active')):
            return save('not_authorized_or_not_terminal')
        report_root=contained(state['report_root'],project/'comparison_reports')
        if not report_root.name.startswith('TOOD_TD_'): raise RuntimeError('Not an TOOD_TD report')
        cancel=project/'KEEP_RUNNING_TOOD_TD'
        if cancel.exists(): return save('cancelled_by_KEEP_RUNNING_file')
        previous=read(marker) if marker.is_file() else {}
        if previous.get('status') in ('requesting_shutdown','shutdown_command_returned_zero'):
            return dict(previous)  # no duplicate request after an ambiguous response
        expected=state['controller_identity']
        save('waiting_parent_exit',parent=expected)
        deadline=clock()+60
        while identity(expected['pid'])==expected:
            if clock()>=deadline: return save('blocked_parent_still_active')
            sleeper(2)
        lock_path=project/'logs/.tood_td_controller.lock'
        if not locks(lock_path): return save('blocked_active_TOOD_TD_controller_lock')
        def still_bound():
            index=read(project/'logs/latest_tood_td_controller.json')
            return Path(index['session']).resolve()==session and locks(lock_path)
        if not still_bound(): return save('cancelled_new_TOOD_TD_session')
        if state['status']=='completed':
            errors=verify_completed(project,report_root)
            if errors: return save('blocked_incomplete_completed_result',errors=errors)
        preservation=preserve_available(project,session,report_root,state['status'])
        info=shutdown_invocation(command)
        if info['sha256']!=state['shutdown_command']['sha256']:
            return save('blocked_platform_helper_changed')
        status.update(command=info,terminal_status=state['status'],
                      preservation_file=str(session/'PRESERVATION.json'),preserved_files=len(preservation['files']))
        deadline=clock()+wait_seconds
        last_detail=None
        while clock()<deadline:
            if cancel.exists(): return save('cancelled_by_KEEP_RUNNING_file')
            if not still_bound(): return save('cancelled_new_TOOD_TD_session')
            gpu=probe(); snapshot=processes()
            detail=dict(gpu=gpu,processes=snapshot)
            try: require_idle(gpu,snapshot)
            except RuntimeError:
                if detail!=last_detail: save('waiting_other_jobs_or_GPU_release',**detail)
                last_detail=detail; sleeper(10); continue
            save('countdown',seconds=delay,cancel_file=str(cancel),**detail)
            uninterrupted=True
            for remaining in range(delay,0,-5):
                if cancel.exists(): return save('cancelled_by_KEEP_RUNNING_file')
                if not still_bound(): return save('cancelled_new_TOOD_TD_session')
                sleeper(min(5,remaining))
                gpu=probe(); snapshot=processes()
                try: require_idle(gpu,snapshot)
                except RuntimeError:
                    save('countdown_reset_other_task_started',gpu=gpu,processes=snapshot)
                    uninterrupted=False; break
            if not uninterrupted: continue
            if cancel.exists(): return save('cancelled_by_KEEP_RUNNING_file')
            if not still_bound(): return save('cancelled_new_TOOD_TD_session')
            if shutdown_invocation(command)['sha256']!=info['sha256']:
                return save('blocked_platform_helper_changed')
            # Save the request BEFORE the platform can terminate this process.
            save('requesting_shutdown',request_saved_before_invocation=True)
            sys.stdout.flush(); sys.stderr.flush()
            if platform_name is None and hasattr(os,'sync'): os.sync()
            result=runner(info['args'],capture_output=True,text=True,timeout=30)
            if result.returncode:
                return save('shutdown_command_failed',returncode=result.returncode,
                            stdout=result.stdout,stderr=result.stderr)
            return save('shutdown_command_returned_zero',stdout=result.stdout,
                        platform_poweroff_and_billing_stop_not_confirmed=True)
        return save('blocked_idle_wait_timeout',wait_seconds=wait_seconds,
                    manual_action='Inspect listed blockers/AutoDL console; this script kills no other job.')
    except BaseException:
        return save('shutdown_watcher_error',error=traceback.format_exc(),
                    manual_action='Check AutoDL console when safe; billing may continue.')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--session',required=True,type=Path)
    p.add_argument('--project',required=True,type=Path); p.add_argument('--wait-seconds',type=int,default=1800)
    args=p.parse_args()
    if sys.platform!='linux': p.error('Server-only; local tests must use a mocked runner')
    import fcntl
    lock=(args.session/'shutdown_watcher.lock').open('a+')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: return 1
    result=run_watch(args.session,args.project,args.wait_seconds)
    return 0 if result['status'] in ('shutdown_command_returned_zero','cancelled_by_KEEP_RUNNING_file') else 1


if __name__=='__main__': sys.exit(main())
