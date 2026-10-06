#!/usr/bin/env python3
"""Require prior E40 CUDA check/smoke, then wait for one completed E39 run.

No source patches, process killing, log-string-only success detection or batch
fallbacks. E39 failures never trigger the next training automatically.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import time

# Independent from the active E39 package. This E40 controller is included in
# the E40 safety signature; never patch E39 or silently reuse stale E40 gates.
HERE=Path(__file__).resolve().parent
E39_NAME='e39a_p2_local_context_img640_seed1_200e'
E40_NAME='e40a_spatial_task_head_img640_seed1_200e'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def process_identity(pid,proc_root=Path('/proc')):
    try:
        text=(proc_root/str(pid)/'stat').read_text()
        fields=text[text.rfind(')')+2:].split()
        # remainder begins at field 3; starttime is field 22
        if fields[0] in ('Z','X'): return None
        return dict(pid=int(pid),start_ticks=int(fields[19]))
    except (FileNotFoundError,ProcessLookupError,PermissionError):
        return None


def process_parent(pid,proc_root):
    try:
        text=(proc_root/str(pid)/'stat').read_text()
        fields=text[text.rfind(')')+2:].split()
        return int(fields[1])  # Linux stat field 4: PPID
    except (FileNotFoundError,ProcessLookupError,PermissionError):
        return None


def training_roots(processes,proc_root):
    """Forked DataLoader workers inherit parent's cmdline, but are not jobs.

    Exclude only descendants of another live matching process. Two independent
    roots remain two jobs; orphan workers remain blockers rather than success.
    """
    by_pid={p['pid']:p for p in processes}
    roots=[]
    for process in processes:
        pid=process['pid']; seen={pid}; is_child=False
        while True:
            parent=process_parent(pid,proc_root)
            if parent is None or parent<=1 or parent in seen:
                break
            seen.add(parent)
            if parent in by_pid:
                ancestor=by_pid[parent]
                current=process_identity(parent,proc_root)
                if (current=={'pid':parent,'start_ticks':ancestor['start_ticks']} and
                        ancestor['start_ticks']<=process['start_ticks']):
                    is_child=True
                break
            pid=parent
        if not is_child:
            roots.append(process)
    return sorted(roots,key=lambda p:p['pid'])


def find_e39_processes(package,project,proc_root=Path('/proc')):
    expected=(package/'run_e39.py').resolve()
    answer=[]
    for item in proc_root.iterdir():
        if not item.name.isdigit(): continue
        try:
            tokens=[x.decode() for x in (item/'cmdline').read_bytes().split(b'\0') if x]
        except (FileNotFoundError,PermissionError,ProcessLookupError):
            continue
        matching=False
        for token in tokens[1:]:
            try:
                if Path(token).resolve()==expected: matching=True
            except (ValueError,OSError):
                pass
        if not matching: continue
        if '--project' in tokens and Path(tokens[tokens.index('--project')+1]).resolve()!=project.resolve():
            continue
        mode=tokens[tokens.index('--mode')+1] if '--mode' in tokens else None
        identity=process_identity(item.name,proc_root)
        if identity is not None and mode=='run':
            answer.append({**identity,'mode':mode})
    return training_roots(answer,proc_root)


def choose_report(project,explicit=None):
    if explicit is not None:
        report=explicit.resolve()
        if not report.is_dir(): raise RuntimeError('E39 report directory does not exist')
        return report
    reports=[]
    for path in (project/'comparison_reports').glob('E39_RUN_*'):
        training=path/'training.json'
        if not training.is_file(): continue
        data=read_json(training)
        if data.get('params',{}).get('name')==E39_NAME:
            reports.append(path.resolve())
    if not reports:
        raise RuntimeError('No E39 formal training report yet; queue after its first completed epoch or specify --e39-report')
    if len(reports)>1:
        raise RuntimeError('Multiple E39 run reports: specify --e39-report instead of silently switching runs')
    return reports[0]


def e39_receipt(report,project):
    failure=report/'failure.json'
    if failure.exists():
        data=read_json(failure)
        raise RuntimeError('E39 failed; next training is blocked. '+str(data.get('error',''))[-1200:])
    training_path=report/'training.json'
    if not training_path.exists(): return dict(complete=False,epochs=0,status='waiting_for_training_report')
    training=read_json(training_path)
    params=training.get('params',{})
    if (params.get('name')!=E39_NAME or params.get('epochs')!=200 or params.get('imgsz')!=640 or params.get('seed')!=1):
        raise RuntimeError('E39 receipt is from the wrong experiment/protocol')
    epochs=training.get('completed_epochs',0)
    if training.get('status')=='failed': raise RuntimeError('E39 training receipt reports failure')
    if training.get('status')!='completed':
        return dict(complete=False,epochs=epochs,status=training.get('status','unknown'))
    if epochs!=200: raise RuntimeError('E39 stopped before 200 epochs; will not interpret this as success')
    expected=project/'runs'/E39_NAME/'weights/best.pt'
    best=Path(training.get('best','')).resolve()
    if best!=expected.resolve() or not best.is_file():
        raise RuntimeError('Completed E39 receipt has no matching best checkpoint')
    comparison_path=report/'comparison.json'
    if not comparison_path.exists(): return dict(complete=False,epochs=epochs,status='waiting_for_official_evaluation')
    comparison=read_json(comparison_path)
    if comparison.get('status')!='completed': raise RuntimeError('E39 official evaluation did not complete')
    if not (report/'REPORT.md').is_file(): return dict(complete=False,epochs=epochs,status='waiting_for_final_report')
    candidate=comparison.get('candidate',{})
    if not isinstance(candidate,dict) or not comparison.get('delta_pp'):
        raise RuntimeError('Incomplete E39 official comparison payload')
    for key in ('AP50_95_percent','AP50_percent','AP75_percent','AR500_percent'):
        for table in (candidate,comparison['delta_pp']):
            if not isinstance(table.get(key),(int,float)) or not math.isfinite(table[key]):
                raise RuntimeError('E39 official receipt has missing/nonfinite metric '+key)
    if comparison.get('dataset',{}).get('images')!=548:
        raise RuntimeError('E39 official receipt did not evaluate the complete 548-image split')
    if comparison.get('seed')!=1 or comparison.get('imgsz')!=640 or Path(comparison.get('checkpoint','')).resolve()!=best:
        raise RuntimeError('E39 comparison protocol/checkpoint does not match training')
    actual=hashlib.sha256(best.read_bytes()).hexdigest()
    if comparison.get('checkpoint_sha256')!=actual:
        raise RuntimeError('E39 best checkpoint changed after evaluation')
    for name in ('area_predictions.json','gt_rows.json'):
        if not (report/name).is_file(): raise RuntimeError('Missing completed E39 artifact '+name)
    return dict(complete=True,epochs=epochs,status='training_and_official_evaluation_completed')


def gpu_idle(gpu_index=0):
    command=['nvidia-smi','-i',str(gpu_index),'--query-compute-apps=pid','--format=csv,noheader,nounits']
    result=subprocess.run(command,capture_output=True,text=True,timeout=20)
    if result.returncode: raise RuntimeError('Cannot inspect GPU processes: '+result.stderr)
    pids=[]
    for line in result.stdout.splitlines():
        text=line.strip()
        if text.isdigit(): pids.append(int(text))
        elif text and 'No running processes found' not in text:
            raise RuntimeError('Unexpected nvidia-smi process response: '+text)
    memory=subprocess.run(['nvidia-smi','-i',str(gpu_index),'--query-gpu=memory.free','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,timeout=20)
    if memory.returncode: raise RuntimeError('Cannot inspect GPU memory: '+memory.stderr)
    free=float(memory.stdout.strip())/1024
    return dict(idle=not pids and free>=24.,compute_pids=pids,free_GiB=free)


def main():
    global HERE
    if sys.platform!='linux': raise RuntimeError('Queue is for the AutoDL Linux host; local tests only exercise its decision functions')
    import fcntl
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path('/root/autodl-tmp/yolo11_project'))
    p.add_argument('--source',type=Path,default=Path('/root/autodl-tmp/yolo11_src'))
    p.add_argument('--data',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019.yaml'))
    p.add_argument('--dataset',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val'))
    p.add_argument('--e39-package',type=Path,default=Path('/root/autodl-tmp/E39a_P2_LocalContext_AutoDL'))
    p.add_argument('--e39-report',type=Path)
    p.add_argument('--e40-package',type=Path,default=HERE)
    p.add_argument('--inspect',action='store_true',help='Read-only list of E39 root jobs; no training or queue launch')
    p.add_argument('--poll-seconds',type=int,default=15)
    p.add_argument('--timeout-hours',type=float,default=36.)
    args=p.parse_args()
    HERE=args.e40_package.resolve()
    if args.poll_seconds<5: raise ValueError('poll-seconds must be at least 5')
    if args.inspect:
        processes=find_e39_processes(args.e39_package,args.project)
        print(json.dumps(dict(root_jobs=len(processes),processes=processes),indent=2),flush=True)
        return
    lock=open(HERE/'.e40_queue.lock','a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('An E40 controller/queue is already running; do not launch a second one')
    try:
        _queue(args,fcntl)
    finally:
        lock.close()


def _queue(args,fcntl):
    common=['--project',str(args.project),'--source',str(args.source),'--data',str(args.data),
            '--dataset',str(args.dataset),'--device','cuda:0','--workers','8']
    verify=[sys.executable,'-u',str(HERE/'run_e40.py'),'--mode','verify-gates',*common]
    print('E40 checking existing CUDA check and smoke receipts; this queue does NOT run missing tests',flush=True)
    subprocess.run(verify,check=True)
    if (args.project/'runs'/E40_NAME).exists(): raise RuntimeError('E40 run already exists; refusing duplicate training')
    report=choose_report(args.project,args.e39_report)
    processes=find_e39_processes(args.e39_package,args.project)
    print(f'E39 root training jobs: {processes} (forked workers excluded)',flush=True)
    if len(processes)>1: raise RuntimeError('Multiple E39 jobs found; choose/resolve them before queuing')
    if processes and processes[0]['mode']!='run': raise RuntimeError('E39 is not in formal run mode yet; cannot bind queue to this training')
    watched=processes[0] if processes else None
    if watched is None and not e39_receipt(report,args.project)['complete']:
        raise RuntimeError('No live E39 formal process and no completed receipt; refusing to queue an unbound job')
    if watched is not None:
        # Verify the selected receipt belongs to the running process by timing.
        boot=int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime ')))
        started=boot+watched['start_ticks']/os.sysconf('SC_CLK_TCK')
        recorded=datetime.fromisoformat(read_json(report/'training.json')['started_at']).timestamp()
        if abs(started-recorded)>300:
            raise RuntimeError('Selected report does not match current E39 process start time; specify the correct --e39-report')
    out=args.project/'logs'/('queue_e40_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    state=dict(status='waiting',e39_report=str(report),watched_process=watched,started_at=datetime.now().isoformat(),
               e40_checks_already_passed=True,report_dir=str(out))
    print(f'E40 QUEUE watching {report} PID={watched} STATE={out}/state.json',flush=True)
    e39_lock_path=args.e39_package/'.e39_queue.lock'
    if not e39_lock_path.is_file(): raise RuntimeError('Missing E39 controller lock; cannot verify exclusive completion')
    controller_lock=open(e39_lock_path,'a+')
    started=time.monotonic(); last_print=-60.; stable_idle=0
    try:
        while True:
            receipt=e39_receipt(report,args.project)
            current=find_e39_processes(args.e39_package,args.project)
            if watched is not None:
                watched_alive=process_identity(watched['pid'])=={'pid':watched['pid'],'start_ticks':watched['start_ticks']}
            else:
                watched_alive=False
            exclusive=False
            try:
                fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                exclusive=True
            except BlockingIOError:
                pass
            if current and watched is not None and any((p['pid'],p['start_ticks'])!=(watched['pid'],watched['start_ticks']) for p in current):
                raise RuntimeError('A different E39 process appeared; queue refuses to switch experiments')
            if not watched_alive and not current and exclusive and not receipt['complete']:
                raise RuntimeError('E39 process/controller exited without complete 200-epoch and official receipts; next training is blocked')
            ready=receipt['complete'] and not current and not watched_alive and exclusive
            gpu=gpu_idle() if ready else None
            stable_idle=stable_idle+1 if ready and gpu['idle'] else 0
            state.update(e39=receipt,gpu=gpu,stable_idle_polls=stable_idle,updated_at=datetime.now().isoformat())
            write_json(out/'state.json',state)
            if time.monotonic()-last_print>=60 or ready:
                print(f'E40 WAIT {datetime.now().isoformat()} E39={receipt["epochs"]}/200 '
                      f'status={receipt["status"]} process_alive={bool(current) or watched_alive} '
                      f'controller_released={exclusive} GPU={gpu}',flush=True)
                last_print=time.monotonic()
            if stable_idle>=2: break
            if exclusive: fcntl.flock(controller_lock,fcntl.LOCK_UN)
            if time.monotonic()-started>args.timeout_hours*3600: raise TimeoutError('Queue timeout; no next training launched')
            time.sleep(args.poll_seconds)
        # Retain E39's exclusive controller lock until E40 completes to prevent
        # someone restarting another E39 in the same GPU while E40 is running.
        subprocess.run(verify,check=True)  # source/config may have changed during wait
        if (args.project/'runs'/E40_NAME).exists(): raise RuntimeError('E40 appeared during wait; refusing overwrite')
        state.update(status='running',training_started_at=datetime.now().isoformat())
        write_json(out/'state.json',state)
        print('E39 COMPLETE + GPU IDLE + E40 CHECK/SMOKE VALID. STARTING E40 FORMAL 200 EPOCHS '+datetime.now().isoformat(),flush=True)
        # A separate execution lock also protects against manual E40 starts.
        process=subprocess.Popen(['bash',str(HERE/'run_e40.sh'),'run',*common])
        state['e40_pid']=process.pid; write_json(out/'state.json',state)
        code=process.wait()
        if code: raise RuntimeError(f'E40 failed with exit code {code}; read the same launcher log')
        state.update(status='completed',finished_at=datetime.now().isoformat())
        write_json(out/'state.json',state)
        print('E40 QUEUE AND TRAINING COMPLETED '+datetime.now().isoformat(),flush=True)
    except BaseException as error:
        state.update(status='failed',error=str(error),failed_at=datetime.now().isoformat())
        write_json(out/'state.json',state)
        raise
    finally:
        controller_lock.close()


if __name__=='__main__':
    main()
