#!/usr/bin/env python3
"""Live two-job controller. No training until separate check + joint smoke pass."""
from datetime import datetime
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from pair_safety import CAPS, EXPERIMENTS, write_json, validate_smoke_evidence

HERE=Path(__file__).resolve().parent
NAMES=tuple(CAPS)


def say(value):
    print(f'[{datetime.now().isoformat(timespec="seconds")}] {value}',flush=True)


def package_check():
    manifest=json.loads((HERE/'manifest.json').read_text())
    for relative,expected in manifest.items():
        path=HERE/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:
            raise RuntimeError('Package incomplete/changed: '+relative)


def gpu_state():
    proc=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.free,utilization.gpu',
                         '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    rows=[v.strip() for v in proc.stdout.splitlines() if v.strip()]
    if len(rows)!=1: raise RuntimeError('This package requires exactly one GPU')
    total,free,util=(int(v.strip()) for v in rows[0].split(','))
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                        capture_output=True,text=True,check=True,timeout=15)
    pids=[int(v.strip()) for v in apps.stdout.splitlines() if v.strip().isdigit()]
    return dict(total_MiB=total,free_MiB=free,utilization_percent=util,compute_pids=pids)


def require_idle(state):
    if state['total_MiB']<30*1024 or state['free_MiB']<29*1024 or state['compute_pids']:
        raise RuntimeError('Pair smoke/formal requires idle 32GB GPU with >=29GiB free: '+str(state))


def command(args,name,mode,session=None):
    cmd=[sys.executable,'-u',str(HERE/'run_experiment.py'),'--variant',name,'--mode',mode,
         '--project',str(args.project),'--source',str(args.source),'--data',str(args.data),
         '--dataset',str(args.dataset),'--device',args.device,'--workers','8']
    if session is not None: cmd+=['--pair-session',str(session)]
    return cmd


def serial(args,mode):
    for name in NAMES:
        say(f'START {name} {mode}; live output follows')
        code=subprocess.run(command(args,name,mode),check=False).returncode
        say(f'END {name} {mode}: exit={code}')
        if code: raise RuntimeError(f'{name} {mode} failed; no formal jobs started')


def progress(session,name):
    pointer=session/f'{name}_report.json'
    if not pointer.is_file(): return 'initializing'
    try:
        report=Path(json.loads(pointer.read_text())['report'])
        path=report/'training.json'
        if not path.is_file(): return 'preflight'
        data=json.loads(path.read_text())
        seconds=data.get('last_epoch_seconds',0)
        return f'{data.get("completed_epochs",0)}/200 {data.get("status","?")} last_epoch={seconds:.1f}s'
    except (OSError,ValueError,KeyError):
        return 'writing report'


def launch(args,mode):
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    session=args.project/'logs'/f'pair_e58_e59_{mode}_{stamp}'
    session.mkdir(parents=True,exist_ok=False)
    logs={name:str(session/f'{name}_{mode}.log') for name in NAMES}
    write_json(args.project/'logs'/f'latest_e58_e59_{mode}.json',dict(session=str(session),logs=logs))
    if mode=='run': write_json(session/'formal_authorized.json',dict(controller_pid=os.getpid()))
    handles=[];children={}
    state=dict(mode=mode,session=str(session),logs=logs,started_at=datetime.now().isoformat(),exit_codes={},pids={})
    say(f'{mode.upper()} logs: {session}')
    try:
        for name in NAMES:
            handle=open(logs[name],'w',encoding='utf-8',buffering=1);handles.append(handle)
            proc=subprocess.Popen(command(args,name,mode,session),stdout=handle,stderr=subprocess.STDOUT,
                                  start_new_session=True,close_fds=True)
            children[name]=proc
            state['pids'][name]=proc.pid
            write_json(session/'state.json',state)
            say(f'{name}: PID={proc.pid}; log={logs[name]}')
        while True:
            state['exit_codes']={name:proc.poll() for name,proc in children.items()}
            state['updated_at']=datetime.now().isoformat()
            try: state['gpu']=gpu_state()
            except Exception as error: state['gpu_error']=repr(error)
            if mode=='smoke' and any(code not in (None,0) for code in state['exit_codes'].values()):
                write_json(session/'abort.json',dict(error='Peer smoke failed',exit_codes=state['exit_codes']))
            state['progress']={name:progress(session,name) if mode=='run' else 'smoke' for name in NAMES}
            write_json(session/'state.json',state)
            say(' | '.join(f'{name}: {state["progress"][name]} exit={state["exit_codes"][name]}' for name in NAMES)+
                f' | free={state.get("gpu",{}).get("free_MiB","?")}MiB')
            if all(code is not None for code in state['exit_codes'].values()): break
            time.sleep(30 if mode=='run' else 5)
        state['finished_at']=datetime.now().isoformat()
        write_json(session/'state.json',state)
        if state['exit_codes']!={'e58':0,'e59':0}:
            raise RuntimeError(f'{mode} finished with failures: {state["exit_codes"]}; logs={session}')
        if mode=='smoke':
            reports={};signatures={}
            for name in NAMES:
                receipt=json.loads((args.project/'comparison_reports'/f'{name}_smoke_gate.json').read_text())
                reports[name]=receipt['report'];signatures[name]=receipt['signature']
                value=json.loads((Path(receipt['report'])/'smoke.json').read_text())
                validate_smoke_evidence(value,name,session)
            write_json(args.project/'comparison_reports/e58_e59_pair_smoke_gate.json',
                       dict(status='passed',session=str(session),reports=reports,signatures=signatures))
            say('JOINT CUDA SMOKE PASSED: both real batch16 updates and complete EMA-half batch32 validation')
        say(f'COMPLETED {mode}; logs={session}; auto shutdown disabled')
        return session
    except BaseException as error:
        state.update(controller_error=repr(error),updated_at=datetime.now().isoformat())
        write_json(session/'state.json',state)
        if mode=='smoke': write_json(session/'abort.json',dict(error=repr(error)))
        say('Controller stopped. Existing child jobs are NOT killed; inspect listed PIDs/logs before relaunching.')
        raise
    finally:
        for handle in handles: handle.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('check','smoke','verify-gates','run','all'))
    p.add_argument('--project',type=Path,default=Path('/root/autodl-tmp/yolo11_project'))
    p.add_argument('--source',type=Path,default=Path('/root/autodl-tmp/yolo11_src'))
    p.add_argument('--data',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019.yaml'))
    p.add_argument('--dataset',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val'))
    p.add_argument('--device',choices=('cuda:0',),default='cuda:0')
    args=p.parse_args();package_check()
    import fcntl
    path=args.project/'logs/.e58_e59_controller.lock';path.parent.mkdir(parents=True,exist_ok=True)
    lock=path.open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.mode in ('check','all'): serial(args,'check')
    if args.mode in ('smoke','all'):
        require_idle(gpu_state());launch(args,'smoke')
    if args.mode in ('verify-gates','run','all'): serial(args,'verify-gates')
    if args.mode in ('run','all'):
        for name in NAMES:
            path=args.project/'runs'/EXPERIMENTS[name]
            if path.exists(): raise RuntimeError('Refusing to overwrite existing run: '+str(path))
        require_idle(gpu_state());launch(args,'run')


if __name__=='__main__':
    main()
