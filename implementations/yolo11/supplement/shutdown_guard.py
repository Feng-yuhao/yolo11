#!/usr/bin/env python3
"""Detached success-only AutoDL guard; never shut down on failure or partial results."""
from __future__ import annotations
import argparse, os, subprocess, sys, time, traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common7 import *
import platform_safety as safety
from preservation7 import verify_seal

def eligible(state,index,session,report,parent_live,cancelled):
    require(not parent_live,'Controller has not exited')
    require(not cancelled,'User cancelled automatic shutdown')
    require(state.get('status')=='completed' and state.get('stage')=='completed','Pipeline is not successfully completed')
    require(state.get('auto_shutdown') is True and state.get('all_7_formal200_saved') is True and state.get('all_14_official_scores_completed') is True,'Incomplete success flags')
    require(Path(index['session']).resolve()==Path(session).resolve() and Path(index['report']).resolve()==Path(report).resolve(),'A newer/different pipeline owns the project')
    require(Path(state['session']).resolve()==Path(session).resolve() and Path(state['report']).resolve()==Path(report).resolve(),'State/session identity changed')
    require(state.get('completion_seal_sha256'),'Missing preservation seal')
    require(all(v.get('status') in ['completed','failed'] for v in state['jobs'].values()),'Workers have not exited')
    # Failed joint smokes are permitted ONLY when the plan explicitly downgraded
    # the entire formal schedule. All formal/check/single-smoke/score exits are 0.
    for job in state['jobs'].values():
        if job.get('exit_code')==0:continue
        require(job['phase']=='smoke' and 'pair_gates' in Path(job['out']).parts and job.get('category') in ['cuda_oom','gpu_headroom','peer_failed'],
                'Unexpected failed task in completion state')
    return True

def assert_lock_free(project):
    import fcntl
    with (Path(project)/'logs/.stmdrf_supp7.lock').open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(handle,fcntl.LOCK_UN)

def idle_now():
    gpu=safety.gpu_snapshot();processes=safety.process_snapshot(HERE)
    safety.require_idle(gpu,processes,admission=False)
    return dict(gpu=gpu,processes=processes)

def invoke_verified(command,runner=subprocess.run):
    require(sys.platform=='linux','Shutdown is Linux-only')
    current=safety.shutdown_invocation()
    require(current['args']==command['args'] and current['sha256']==command['sha256'] and current['bytes']==command['bytes'],'Platform helper changed')
    result=runner(current['args'],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30)
    require(result.returncode==0,'AutoDL shutdown helper returned '+str(result.returncode)+': '+result.stdout[-2000:])
    return dict(status='shutdown_command_returned_zero',at=now(),returncode=result.returncode,output=result.stdout[-2000:],
                note='Container command accepted; this receipt is not independent confirmation that billing has stopped')

def run(project,session,report):
    project,session,report=Path(project).resolve(),Path(session).resolve(),Path(report).resolve()
    contained(session,project/'logs');contained(report,project/'comparison_reports')
    state=read(session/'state.json');parent=state['controller_identity'];expected=safety.shutdown_invocation()
    require(state['auto_shutdown'] is True and state['admission']['shutdown_invocation']['sha256']==expected['sha256'],'Guard does not match admitted shutdown helper')
    status=session/'shutdown_status.json'
    def update(kind,**kw):
        value=dict(status=kind,at=now(),session=str(session),report=str(report),guard_identity=safety.process_identity(os.getpid()),**kw)
        write(status,value);print(now()+' SHUTDOWN '+kind+' '+str(kw),flush=True)
    update('armed',condition='Only seven successful formal200 runs + fourteen full scores + preservation seal; wait for controller exit')
    while safety.process_identity(parent['pid'])==parent:time.sleep(10)
    state=read(session/'state.json')
    if state.get('status')!='completed':
        update('disarmed_incomplete_or_failed',terminal_status=state.get('status'),shutdown_called=False);return
    cancel=project/'KEEP_RUNNING_ST_MDRF_SUPP7'
    idle_start=None;seal_checked=False;last_notice=None
    while True:
        if cancel.exists():update('cancelled_by_user',cancel_file=str(cancel),shutdown_called=False);return
        try:
            state=read(session/'state.json');idx=read(project/'logs/latest_stmdrf_supp7.json')
            eligible(state,idx,session,report,safety.process_identity(parent['pid'])==parent,False)
            assert_lock_free(project)
            if not seal_checked:
                verify_seal(project,report,state['completion_seal_sha256']);seal_checked=True
            probes=idle_now()
            current=safety.shutdown_invocation()
            require(current['sha256']==expected['sha256'],'Platform helper changed since admission')
        except BaseException as exc:
            message=str(exc);idle_start=None
            # A changed identity or missing/corrupted result cannot be repaired by
            # waiting. Idle/busy checks may legitimately clear later.
            if not seal_checked or 'pipeline' in message.lower() or 'helper changed' in message.lower():
                update('blocked_safety_check',error=message,shutdown_called=False);return
            if message!=last_notice:update('waiting_idle',error=message,shutdown_called=False);last_notice=message
            time.sleep(30);continue
        if idle_start is None:
            idle_start=time.monotonic();update('countdown',seconds=120,cancel_file=str(cancel),shutdown_called=False)
        if time.monotonic()-idle_start>=120:break
        time.sleep(10)
    # Independent final re-read, full byte verification, idle proof, and sync.
    state=read(session/'state.json');idx=read(project/'logs/latest_stmdrf_supp7.json')
    eligible(state,idx,session,report,safety.process_identity(parent['pid'])==parent,cancel.exists())
    verify_seal(project,report,state['completion_seal_sha256']);assert_lock_free(project);probes=idle_now()
    require(not cancel.exists(),'User cancelled shutdown')
    update('requesting_shutdown',terminal_status='completed',preservation_seal_sha256=state['completion_seal_sha256'],probes=probes,shutdown_called=False)
    os.sync()
    # Recheck after sync, which may take time on a busy disk.
    require(not cancel.exists(),'User cancelled shutdown');idle_now()
    result=invoke_verified(expected)
    write(status,{**result,'terminal_status':'completed','preservation_seal_sha256':state['completion_seal_sha256']})
    os.sync()

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--session',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    try:run(a.project,a.session,a.report);return 0
    except BaseException:
        value=dict(status='blocked_safety_check',at=now(),shutdown_called=False,error=traceback.format_exc())
        write(a.session/'shutdown_status.json',value);print(value['error'],file=sys.stderr,flush=True);return 1

if __name__=='__main__':sys.exit(main())
