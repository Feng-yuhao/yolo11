"""Independent E50/E51 jobs. Concurrent real smoke required before double run."""
from __future__ import annotations
import argparse, hashlib, json, os, signal, subprocess, sys, time
from datetime import datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent
VARIANTS=('e50','e51')
def log(value): print('['+datetime.now().isoformat(timespec='seconds')+'] '+value,flush=True)
def write(path,data):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(data,indent=2),encoding='utf-8'); tmp.replace(path)
def digest(path): return hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def code_signature():
    return {p.relative_to(HERE).as_posix():digest(p) for p in sorted(HERE.rglob('*'))
            if p.is_file() and p.suffix in ('.py','.sh','.yaml') and '__pycache__' not in p.parts}
def receipts(project):
    return {v:{kind:digest(project/'comparison_reports'/f'{v}_{kind}_gate.json')
            for kind in ('cuda_check','smoke')} for v in VARIANTS}
def gpu_free():
    r=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                     capture_output=True,text=True,check=True,timeout=15)
    return int(r.stdout.splitlines()[0])
def command(args,v,mode):
    return ['bash',str(HERE/v/'run_e49.sh'),mode,'--project',str(args.project),
            '--source',str(args.source),'--data',str(args.data),'--dataset',str(args.dataset)]
def launch(args,mode,directory,parallel=True,barrier=None):
    variants=VARIANTS if parallel else (args.single_variant,)
    jobs={}; handles={}; offsets={}; statuses={}
    try:
        for v in variants:
            path=directory/(v+'_'+mode+'.log'); handles[v]=path.open('w',encoding='utf-8')
            env=os.environ.copy()
            if barrier: env.update(PAIR_SMOKE_DIR=str(barrier),PAIR_VARIANT=v)
            jobs[v]=subprocess.Popen(command(args,v,mode),stdout=handles[v],
                stderr=subprocess.STDOUT,env=env,start_new_session=True)
            offsets[v]=0
            log(f'{v}: PID={jobs[v].pid}; log={path}')
        started=time.monotonic(); last=0
        while True:
            statuses={v:p.poll() for v,p in jobs.items()}
            if not parallel:
                path=directory/(variants[0]+'_'+mode+'.log')
                with path.open(encoding='utf-8',errors='replace') as reader:
                    reader.seek(offsets[variants[0]]); value=reader.read(); offsets[variants[0]]=reader.tell()
                if value: print(value,end='',flush=True)
            if barrier and any(code not in (None,0) for code in statuses.values()):
                (barrier/'aborted').touch()
            if time.monotonic()-last>10:
                log(f'{mode}: exits={statuses}, elapsed={(time.monotonic()-started)/60:.1f} min, GPU free={gpu_free()} MiB')
                last=time.monotonic()
            if all(code is not None for code in statuses.values()): break
            time.sleep(1)
        write(directory/(mode+'_exits.json'),statuses)
        log(f'FINISHED {mode}: {statuses}; auto shutdown disabled')
        if any(code!=0 for code in statuses.values()):
            raise RuntimeError(f'{mode} failed: {statuses}; inspect the corresponding log. No training gate issued.')
        return statuses
    except BaseException:
        # Natural formal-job failures are handled AFTER both jobs finish above.
        # Interruption/controller error cleans only this controller's children.
        for p in jobs.values():
            if p.poll() is None: os.killpg(p.pid,signal.SIGTERM)
        for p in jobs.values():
            try: p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL); p.wait()
        raise
    finally:
        for f in handles.values(): f.close()
def verify(args,directory):
    for v in VARIANTS:
        args.single_variant=v; launch(args,'verify-gates',directory,parallel=False)
    path=args.project/'comparison_reports/e50_e51_parallel_smoke_gate.json'
    if not path.is_file(): raise RuntimeError('Missing genuine concurrent smoke receipt')
    gate=json.loads(path.read_text())
    if gate.get('status')!='passed' or gate.get('code')!=code_signature() or gate.get('receipts')!=receipts(args.project):
        raise RuntimeError('Stale parallel receipt; repeat concurrent smoke')
    barrier=Path(gate['barrier'])
    if (barrier/'aborted').exists() or not all((barrier/(v+'_'+phase)).is_file()
            for v in VARIANTS for phase in ('train_ready','validation_done')):
        raise RuntimeError('Incomplete concurrent training/validation barrier')
    if json.loads((Path(gate['logs'])/'smoke_exits.json').read_text())!={'e50':0,'e51':0}:
        raise RuntimeError('Parallel smoke exit evidence failed')
    log('CURRENT CUDA CHECK + CONCURRENT REAL SMOKE RECEIPTS VALID')
def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('check','smoke','verify-gates','run','all'))
    p.add_argument('--project',type=Path,default=Path('/root/autodl-tmp/yolo11_project'))
    p.add_argument('--source',type=Path,default=Path('/root/autodl-tmp/yolo11_src'))
    p.add_argument('--data',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019.yaml'))
    p.add_argument('--dataset',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val'))
    args=p.parse_args()
    import fcntl
    with (args.project/'.e50_e51_controller.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('E50/E51 controller already active')
        directory=args.project/'logs'/('pair_e50_e51_'+args.mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        directory.mkdir(parents=True); log('Logs: '+str(directory))
        manifest=json.loads((HERE/'manifest.json').read_text())
        if manifest!=code_signature(): raise RuntimeError('Root package code changed; use the verified complete package')
        stages=('check','smoke','run') if args.mode=='all' else (args.mode,)
        for stage in stages:
            if stage=='check':
                for v in VARIANTS:
                    args.single_variant=v; launch(args,'check',directory,parallel=False)
            elif stage=='smoke':
                if gpu_free()<30*1024: raise RuntimeError('Concurrent smoke must start on idle >=30 GiB free GPU')
                barrier=directory/'barrier'; barrier.mkdir()
                launch(args,'smoke',directory,barrier=barrier)
                if not all((barrier/(v+'_'+phase)).is_file() for v in VARIANTS
                           for phase in ('train_ready','validation_done')):
                    raise RuntimeError('Incomplete simultaneous smoke phases')
                write(args.project/'comparison_reports/e50_e51_parallel_smoke_gate.json',
                    dict(status='passed',code=code_signature(),receipts=receipts(args.project),
                         barrier=str(barrier),logs=str(directory),allocator_cap_GiB_each=13.))
            else:
                verify(args,directory)
                if stage=='run':
                    if gpu_free()<30*1024: raise RuntimeError('Formal pair must start with >=30 GiB free GPU')
                    launch(args,'run',directory)
        log('COMPLETED controller '+args.mode+'; logs='+str(directory))
if __name__=='__main__': main()
