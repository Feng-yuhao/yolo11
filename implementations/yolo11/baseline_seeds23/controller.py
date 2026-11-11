"""Parallel independent E0 formal runs -> SERIAL evaluation -> opt-in shutdown."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import common as c
from shutdown_safety import auto_shutdown,gpu_snapshot,active_python_jobs,require_idle


@contextmanager
def exclusive():
    import fcntl
    path=c.PROJECT/'logs/e0_s23_e60_execution.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Controller already running; do not launch twice')
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def child_command(script,mode,out,seed=None,pair=None):
    command=[sys.executable,'-u',str(c.HERE/script),mode,'--out',str(out)]
    if seed is not None:command+=['--seed',str(seed)]
    if pair is not None:command+=['--pair',str(pair)]
    return command


def serial(script,mode,session,label,seed=None):
    log=session/(label+'.log')
    out=c.PROJECT/'comparison_reports'/(label.upper()+'_'+session.name)
    print(c.now()+' START '+label+' log='+str(log),flush=True)
    command=child_command(script,mode,out,seed)
    # Tee line-by-line so short checks are visible; training uses own log files.
    with log.open('w',encoding='utf-8') as handle:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in process.stdout:
                handle.write(line);handle.flush();print(line,end='',flush=True)
            code=process.wait()
        except BaseException:
            # Do not kill an orphaned child; auto_shutdown safety detects it.
            raise
    c.require(code==0,label+' failed; read '+str(log))
    return out


def parallel(mode,session,state):
    jobs={};handles=[]
    try:
        for seed in (2,3):
            label='e0_seed%d_%s'%(seed,mode)
            log=session/(label+'.log')
            out=c.PROJECT/'comparison_reports'/(label.upper()+'_'+session.name)
            handle=log.open('w',encoding='utf-8');handles.append(handle)
            process=subprocess.Popen(child_command('train_e0.py',mode,out,seed,session if mode=='smoke' else None),
                                     stdout=handle,stderr=subprocess.STDOUT)
            jobs[str(seed)]=dict(process=process,log=str(log),out=str(out))
            print('%s %s PID=%d LOG=%s'%(c.now(),label,process.pid,log),flush=True)
        state['children']={k:dict(pid=v['process'].pid,log=v['log'],report=v['out']) for k,v in jobs.items()}
        c.write(session/'state.json',state)
        while True:
            exits={k:v['process'].poll() for k,v in jobs.items()}
            # Also show actual last saved epoch/ETA without tqdm clutter.
            snapshots={}
            for key,value in jobs.items():
                metrics=Path(value['out'])/'metrics.json'
                if metrics.is_file():
                    snapshots[key]=c.read(metrics).get('training',{}).get('completed_epochs',0)
            gpu=gpu_snapshot()
            print('%s %s exits=%s epochs=%s GPU_free=%dMiB'%(c.now(),mode,exits,snapshots,gpu['free_MiB']),flush=True)
            state.update(stage=mode,exit_codes=exits,epochs=snapshots,updated_at=c.now())
            c.write(session/'state.json',state)
            if all(value is not None for value in exits.values()):break
            time.sleep(15)
        # A failing job NEVER forcibly kills the other independent formal run.
        c.require(all(value==0 for value in exits.values()),'Parallel '+mode+' failed '+str(exits)+'; successful sibling retained')
        if mode=='smoke':
            bindings={str(s):c.read(c.gates(s,'smoke')) for s in (2,3)}
            c.write(c.PROJECT/'comparison_reports/e0_s23_pair_gates/parallel.json',dict(status='passed',
                    signatures={k:v['signature'] for k,v in bindings.items()},session=str(session),finished_at=c.now()))
        return exits
    finally:
        for handle in handles:handle.close()


def admission():
    # Parent process is excluded by active_python_jobs. No other train/eval job
    # may be starting, including a process not yet visible to nvidia-smi.
    require_idle(gpu_snapshot(),active_python_jobs(),admission=True)


def run(args,session,state):
    c.package_audit();c.protocol_audit()
    admission()
    if args.mode in ('check','all'):
        for seed in (2,3):serial('train_e0.py','check',session,'e0_seed%d_check'%seed,seed)
        serial('evaluate_all.py','check',session,'testdev_engineering_check')
    if args.mode in ('smoke','all'):
        parallel('smoke',session,state)
    if args.mode in ('run','all'):
        # Validate BOTH gates BEFORE spawning either formal process.
        # No E60 initialization/training and no silent retries from scratch.
        for seed in (2,3):
            _,core=c.setup(seed)
            import torch
            binding=c.signature(seed,torch)
            c.verify(seed,'check',binding);c.verify(seed,'smoke',binding)
            pair=c.read(c.PROJECT/'comparison_reports/e0_s23_pair_gates/parallel.json')
            c.require(pair['status']=='passed' and pair['signatures'][str(seed)]==c.digest(binding),'Stale parallel smoke')
            c.require(not (c.PROJECT/'runs'/c.params(seed)['name']).exists(),'Run already exists; use eval after completion, not rerun')
        # Parent queried CUDA only for gate/environment checks; release its
        # tiny context allocations, but never builds a training model.
        parallel('run',session,state)
    if args.mode in ('run','all','eval'):
        for seed in (2,3):c.completed_e0(seed);c.e60_identity(seed)
        # Tiny parent CUDA context is acceptable to inference, but it is gone
        # before shutdown. Full inference is one subprocess, serial internally.
        report=serial('evaluate_all.py','run',session,'testdev_e0_e60_s23')
        state['final_report']=str(report/'REPORT.md')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('check','smoke','run','all','eval','status'))
    parser.add_argument('--auto-shutdown',action='store_true')
    args=parser.parse_args()
    index=c.PROJECT/'logs/latest_e0_s23_e60.json'
    if args.mode=='status':
        info=c.read(index);state=c.read(Path(info['session'])/'state.json')
        print(__import__('json').dumps(state,ensure_ascii=False,indent=2));return
    with exclusive():
        session=c.PROJECT/'logs'/('e0_s23_e60_'+args.mode+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
        session.mkdir(parents=True,exist_ok=False)
        state=dict(status='running',mode=args.mode,started_at=c.now(),session=str(session),
                   auto_shutdown=args.auto_shutdown)
        c.write(index,dict(session=str(session)));c.write(session/'state.json',state)
        print('SESSION='+str(session)+'; E0 seeds23 parallel; test-dev serial',flush=True)
        admitted=False
        try:
            # Only a genuinely admitted run may shut down after any failure.
            admission();admitted=True
            run(args,session,state)
            state.update(status='completed',finished_at=c.now())
        except BaseException:
            state.update(status='failed_or_interrupted',error=traceback.format_exc(),finished_at=c.now())
            raise
        finally:
            c.write(session/'state.json',state)
            # Use a fresh CPU-only subprocess: this parent may own a harmless
            # CUDA context and must EXIT before the idle/GPU check can pass.
            enabled=args.auto_shutdown and admitted and args.mode in ('all','run','eval')
            if enabled:
                log=(session/'shutdown.log').open('w',encoding='utf-8')
                process=subprocess.Popen([sys.executable,'-u',str(c.HERE/'shutdown_after.py'),
                        '--parent',str(os.getpid()),'--session',str(session)],stdout=log,stderr=subprocess.STDOUT,
                        start_new_session=True)
                state['shutdown']=dict(status='helper_started',pid=process.pid,log=str(session/'shutdown.log'))
                c.write(session/'state.json',state);log.close()
                print('AUTO SHUTDOWN queued AFTER parent exits; 60s idle-checked countdown',flush=True)
            else:print('AUTO SHUTDOWN disabled for this stage / admission failure',flush=True)


if __name__=='__main__':main()
