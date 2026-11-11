"""Explicit success-only shutdown; local tests inject runner, never call real shutdown."""
import os
from pathlib import Path
import subprocess
import sys
import time
from common import HERE,KEYS,read,write,sha,require,now
from shutdown_safety import process_identity,process_snapshot,gpu_snapshot,require_idle,shutdown_invocation


def lock_free(path):
    import fcntl
    with Path(path).open('r+b') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return False
        fcntl.flock(f,fcntl.LOCK_UN);return True


def verify_outputs(report):
    state=read(report/'state.json')
    require(state['status']=='completed' and state['auto_shutdown'],'Only completed opt-in run may shut down')
    require(all(state['jobs'].get('evaluation_'+k,{}).get('exit_code')==0 for k in KEYS),'Four evaluations not successful')
    require(sha(report/'ARTIFACT_SHA256.json')==state['artifact_manifest_sha256'],'Artifact index changed')
    for name,digest in read(report/'ARTIFACT_SHA256.json').items():
        p=report/name
        require(p.resolve().is_relative_to(report.resolve()) and sha(p)==digest,'Saved result changed: '+name)
    specs=read(HERE/'references/checkpoints.json')
    for k in KEYS:
        base=report/'evaluation'/k;p=read(base/'prediction_manifest.json');s=read(base/'coco_metrics.json')
        require(p['seen_images']==16592 and len(set(p['seen_image_ids']))==16592 and p['parameters_and_BN_unchanged'],'Incomplete prediction proof')
        require(p['checkpoint_sha256']==s['checkpoint_sha256']==specs[k]['sha256'],'Wrong checkpoint proof')
        require(s['status']=='completed' and s['images']==16592 and sum(s['GT_counts'].values())==375884,'Incomplete COCO score')
        require(read(base/'native_ap70.json')['images']==16592,'Incomplete native AP70')
    require(read(report/'summary.json')['status']=='completed' and (report/'REPORT.md').is_file(),'Summary missing')


def run_watch(report,project,*,platform=None,identity=process_identity,locks=lock_free,probe=gpu_snapshot,
              processes=None,runner=subprocess.run,sleep=time.sleep,clock=time.monotonic,command=Path('/usr/bin/shutdown'),delay=60):
    report=Path(report).resolve();project=Path(project).resolve()
    marker=report/'shutdown_status.json'
    def save(status,**kw):
        v=dict(status=status,at=now(),**kw);write(marker,v);print(v,flush=True);return v
    try:
        require((sys.platform if platform is None else platform)=='linux','Linux only')
        require(report.is_relative_to(project/'comparison_reports') and report.name.startswith('VISDRONE_TO_UAVDT_S23_SOURCE_ONLY_'),'Wrong report')
        state=read(report/'state.json')
        if state.get('status')!='completed' or not state.get('auto_shutdown'):return save('not_authorized_or_not_complete')
        previous=read(marker) if marker.exists() else {}
        if previous.get('status') in ('requesting_shutdown','shutdown_command_returned_zero'):return previous
        cancel=project/'KEEP_RUNNING_UAVDT_S23'
        lock=project/'logs/.visdrone_to_uavdt_seeds23_source_only.lock'
        def bound():
            return Path(read(project/'logs/latest_visdrone_to_uavdt_seeds23_source_only.json')['report']).resolve()==report and locks(lock)
        save('waiting_controller_exit');deadline=clock()+60
        while identity(state['controller_identity']['pid'])==state['controller_identity']:
            if clock()>=deadline:return save('blocked_controller_still_alive')
            sleep(2)
        if cancel.exists():return save('cancelled_by_KEEP_RUNNING_file')
        require(bound(),'New session or active controller lock')
        save('verifying_saved_results');verify_outputs(report)
        info=shutdown_invocation(command)
        require(info['sha256']==state['shutdown_command']['sha256'],'Platform helper changed')
        processes=processes or (lambda:process_snapshot(HERE))
        deadline=clock()+1800
        while clock()<deadline:
            if cancel.exists():return save('cancelled_by_KEEP_RUNNING_file')
            require(bound(),'New session/active lock')
            try:require_idle(probe(),processes())
            except RuntimeError as e:
                save('waiting_other_tasks',detail=str(e));sleep(10);continue
            save('countdown',seconds=delay,cancel_file=str(cancel));clean=True
            for _ in range(0,delay,5):
                sleep(min(5,delay))
                if cancel.exists():return save('cancelled_by_KEEP_RUNNING_file')
                require(bound(),'New session/active lock')
                try:require_idle(probe(),processes())
                except RuntimeError:clean=False;break
            if not clean:continue
            require(shutdown_invocation(command)['sha256']==info['sha256'],'Platform helper changed')
            require(read(report/'state.json')['status']=='completed','Completion revoked')
            save('requesting_shutdown')
            if platform is None and hasattr(os,'sync'):os.sync()
            result=runner(info['args'],capture_output=True,text=True,timeout=30)
            return save('shutdown_command_returned_zero' if result.returncode==0 else 'shutdown_command_failed',
                        returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
                        billing_stop_not_confirmed=True)
        return save('blocked_idle_wait_timeout')
    except BaseException:
        import traceback
        return save('blocked_shutdown_error',error=traceback.format_exc())


if __name__=='__main__':
    import argparse,fcntl
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--project',type=Path,required=True);a=p.parse_args()
    with (a.report/'shutdown_watcher.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);run_watch(a.report,a.project)
