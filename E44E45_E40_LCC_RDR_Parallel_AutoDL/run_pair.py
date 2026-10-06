"""Independent formal200 pair. A failure NEVER kills the other formal job."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
NAMES=('e44','e45')
RUNS={'e44':'e44a_local_channel_calibration_img640_seed1_200e',
      'e45':'e45a_residual_depth_reuse_img640_seed1_200e'}


def say(text):
    print('['+datetime.now().isoformat(timespec='seconds')+'] '+text,flush=True)


def write(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8'); tmp.replace(path)


def gpu():
    p=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.free,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    rows=p.stdout.strip().splitlines()
    if len(rows)!=1: raise RuntimeError('Exactly one GPU required')
    total,free,used=map(lambda v:int(v.strip()),rows[0].split(','))
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    return dict(total_MiB=total,free_MiB=free,used_MiB=used,pids=[int(v) for v in p.stdout.splitlines() if v.strip().isdigit()])


def idle():
    state=gpu()
    if state['free_MiB']<28*1024 or state['pids']:
        raise RuntimeError('Parallel mode requires idle GPU >=28 GiB free: '+str(state))
    return state


def command(name,mode,args):
    return ['bash',str(ROOT/name/'run.sh'),mode,'--project',str(args.project),'--source',str(args.source)]


def progress(args,name,started):
    reports=sorted((args.project/'comparison_reports').glob(name.upper()+'_RUN_*'),reverse=True)
    for p in reports:
        f=p/'training.json'
        if f.exists() and f.stat().st_mtime>=started:
            try:
                v=json.loads(f.read_text()); return f"{v.get('completed_epochs',0)}/200 {v.get('status')}"
            except (OSError,json.JSONDecodeError): return 'writing status'
    return 'initializing'


def launch(mode,args):
    logs=args.project/'logs'/('pair_e44_e45_'+mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    logs.mkdir(parents=True)
    say('Logs: '+str(logs))
    write(args.project/'logs'/('e44e45_latest_'+mode+'.json'),dict(directory=str(logs),logs={n:str(logs/(n+'_'+mode+'.log')) for n in NAMES}))
    children={}; handles=[]; started=time.time(); shown_failures=set()
    try:
        for name in NAMES:
            handle=(logs/(name+'_'+mode+'.log')).open('w',encoding='utf-8',buffering=1); handles.append(handle)
            env=os.environ.copy()
            if mode=='smoke': env['E44E45_SMOKE_BARRIER']=str(logs)
            children[name]=subprocess.Popen(command(name,mode,args),stdout=handle,stderr=subprocess.STDOUT,
                                             env=env,start_new_session=True)
            say(f'{name}: PID={children[name].pid}; log={handle.name}')
        while True:
            codes={n:c.poll() for n,c in children.items()}
            for name,code in codes.items():
                if code not in (None,0) and name not in shown_failures:
                    shown_failures.add(name)
                    path=logs/(name+'_'+mode+'.log')
                    say(f'{name} FAILED exit={code}; last traceback from {path}:')
                    print('\n'.join(path.read_text(encoding='utf-8',errors='replace').splitlines()[-28:]),flush=True)
            if mode=='smoke' and any(v not in (None,0) for v in codes.values()):
                (logs/'abort').touch()  # unblock peer's smoke barrier, NOT kill a job
            state=dict(mode=mode,logs={n:str(logs/(n+'_'+mode+'.log')) for n in NAMES},
                       pids={n:c.pid for n,c in children.items()},exit_codes=codes,updated_at=datetime.now().isoformat())
            try: state['gpu']=gpu()
            except Exception as exc: state['gpu_error']=repr(exc)
            if mode=='run': state['epochs']={n:progress(args,n,started) for n in NAMES}
            write(logs/'state.json',state)
            say(' | '.join(f'{n}: {state.get("epochs",{}).get(n,mode)} exit={codes[n]}' for n in NAMES)+
                f' | free={state.get("gpu",{}).get("free_MiB","?")} MiB')
            if all(v is not None for v in codes.values()): break
            time.sleep(30 if mode=='run' else 5)
        say('FINISHED '+mode+': '+str(codes)+'; auto shutdown disabled')
        return 0 if all(v==0 for v in codes.values()) else 1
    finally:
        for h in handles: h.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('check','smoke','verify','run','serial'))
    p.add_argument('--project',type=Path,default=Path('/root/autodl-tmp/yolo11_project'))
    p.add_argument('--source',type=Path,default=Path('/root/autodl-tmp/yolo11_src'))
    args=p.parse_args()
    for name,expected in json.loads((ROOT/'manifest.json').read_text()).items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:
            raise RuntimeError('Release changed: '+name)
    lock=(ROOT/'.pair.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.mode in ('check','verify'):
        codes=[]
        for name in NAMES:
            say('START '+name+' '+args.mode+' (live output follows)')
            codes.append(subprocess.run(command(name,'verify-gates' if args.mode=='verify' else 'check',args)).returncode)
        return int(any(codes))
    if args.mode=='smoke':
        idle(); return launch('smoke',args)
    for name in NAMES:
        if subprocess.run(command(name,'verify-gates',args)).returncode:
            raise RuntimeError('Missing/stale CUDA check or full real smoke: '+name)
        if (args.project/'runs'/RUNS[name]).exists():
            raise RuntimeError('Existing formal run; refusing overwrite: '+name)
    if args.mode=='run':
        from pair_receipts import require_parallel_smoke
        require_parallel_smoke(args.project)
    idle()
    if args.mode=='serial':
        codes=[subprocess.run(command(n,'run',args)).returncode for n in NAMES]
        return int(any(codes))
    return launch('run',args)


if __name__=='__main__':
    raise SystemExit(main())
