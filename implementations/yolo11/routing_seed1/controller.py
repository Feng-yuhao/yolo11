"""Two fixed seed1 ablations: serial by default; optional gated parallel."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import common as c
from shutdown_safety import gpu_snapshot, active_python_jobs, require_idle


@contextmanager
def exclusive(project):
    import fcntl
    path = project/'logs'/'e60_routing_ablations_controller.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def admission():
    require_idle(gpu_snapshot(),active_python_jobs(),admission=True)


def wait_released():
    # A process exit and nvidia-smi's context cleanup need not be simultaneous.
    deadline = time.monotonic()+180
    while True:
        try:
            admission(); return
        except RuntimeError as error:
            if time.monotonic()>=deadline:
                raise
            print(c.now()+' waiting for exited job/GPU release: '+str(error),flush=True)
            time.sleep(5)


def command(args,variant,stage,out,pair=None):
    result = [sys.executable,'-u',str(c.HERE/'run_experiment.py'),'--variant',variant,'--mode',stage,
              '--out',str(out),'--execution',args.execution,'--project',str(args.project),
              '--source',str(args.source),'--data',str(args.data),'--dataset',str(args.dataset),
              '--device','cuda:0','--workers','8']
    if pair is not None:
        result += ['--pair',str(pair)]
    if stage == 'eval':
        result += ['--weights',str(args.project/'runs'/c.EXPERIMENTS[variant]/'weights/best.pt')]
    return result


def launch(args,variant,stage,session,report_root,state,tee=False,pair=None):
    log, out = session/(variant+'_'+stage+'.log'), report_root/(variant+'_'+stage)
    handle = log.open('w',encoding='utf-8')
    process = subprocess.Popen(command(args,variant,stage,out,pair),stdout=subprocess.PIPE if tee else handle,
                               stderr=subprocess.STDOUT,text=True,bufsize=1)
    job = dict(process=process,handle=handle,variant=variant,stage=stage,log=log,out=out)
    state.setdefault('jobs',{})[variant+'_'+stage] = dict(pid=process.pid,log=str(log),report=str(out),exit_code=None)
    c.write(session/'state.json',state)
    print(c.now()+f' {variant} {stage} PID={process.pid} LOG={log}',flush=True)
    if tee:
        for line in process.stdout:
            handle.write(line); handle.flush(); print(line,end='',flush=True)
    return job


def wait_jobs(jobs,session,state,smoke_pair=False):
    while True:
        exits = {job['variant']:job['process'].poll() for job in jobs}
        snapshots = {}
        for job in jobs:
            entry = state['jobs'][job['variant']+'_'+job['stage']]
            entry['exit_code'] = exits[job['variant']]
            path = job['out']/'training.json'
            if path.is_file():
                value = c.read(path)
                snapshots[job['variant']] = dict(epoch=value.get('completed_epochs',0),
                    last_seconds=value.get('epoch_seconds'),status=value.get('status'))
        state.update(updated_at=c.now(),snapshots=snapshots)
        c.write(session/'state.json',state)
        print(c.now()+' '+jobs[0]['stage']+' exits='+str(exits)+' progress='+str(snapshots),flush=True)
        if smoke_pair and any(code is not None and code != 0 for code in exits.values()):
            c.write(session/'abort.json',dict(error='Joint smoke peer failed',exit_codes=exits))
        if all(code is not None for code in exits.values()):
            break
        time.sleep(20)
    for job in jobs:
        job['handle'].close()
    return exits


def run_stage(args,stage,session,report_root,state):
    if stage in ('smoke','run') and args.execution == 'parallel':
        from pair_safety import start_ticks
        jobs = [launch(args,v,stage,session,report_root,state,pair=session if stage=='smoke' else None)
                for v in c.VARIANTS]
        if stage == 'smoke':
            c.write(session/'children.json',{job['variant']:dict(pid=job['process'].pid,
                      start_ticks=start_ticks(job['process'].pid)) for job in jobs})
        exits = wait_jobs(jobs,session,state,smoke_pair=stage=='smoke')
        if stage == 'smoke':
            c.require(all(code==0 for code in exits.values()), 'Joint smoke failed; do not lower batch or start training')
            receipts = {v:c.read(c.gate_path(arg_variant(args,v),'smoke')) for v in c.VARIANTS}
            c.write(c.pair_gate(args.project),dict(status='passed',session=str(session),
                    signatures={v:r['signature'] for v,r in receipts.items()},finished_at=c.now()))
        return exits
    exits = {}
    for variant in c.VARIANTS:
        # A failed independent training does not prevent the other prevalidated
        # ablation. Wait for the failed process/GPU context to exit first.
        wait_released()
        job = launch(args,variant,stage,session,report_root,state,tee=stage in ('check','verify-gates'))
        code = wait_jobs([job],session,state)[variant]
        exits[variant] = code
        if stage not in ('run','eval'):
            c.require(code==0,variant+' '+stage+' failed; see '+str(job['log']))
    return exits


def arg_variant(args,variant):
    from types import SimpleNamespace
    return SimpleNamespace(project=args.project,variant=variant,execution=args.execution)


def completed(args,variant):
    run = args.project/'runs'/c.EXPERIMENTS[variant]
    index = c.read(args.project/'logs'/('latest_e60_ablation_'+variant+'_run.json'))
    report = Path(index['report'])
    marker = c.read(report/'training_completion.json')
    c.require(marker.get('trained_epochs')==200 and (run/'weights/best.pt').is_file(),
              'No complete 200-epoch checkpoint for '+variant)
    return report


def summarize(args,report_root,state):
    table = '|Metric|E1 s1|Full E60 s1|Uniform s1|Shared s1|Uniform minus E60 pp|Shared minus E60 pp|\n|---|---:|---:|---:|---:|---:|---:|\n'
    results = {}
    for variant in c.VARIANTS:
        path = report_root/(variant+'_eval')/'comparison.json'
        if path.is_file():
            results[variant] = c.read(path)
    if results:
        reference = next(iter(results.values()))['references']
        for key in c.KEYS:
            values = [reference['E1'][key],reference['E60'][key]]
            values += [results[v]['candidate'][key] if v in results else None for v in c.VARIANTS]
            values += [results[v]['delta_pp']['E60'][key] if v in results else None for v in c.VARIANTS]
            table += '|'+key+'|'+'|'.join('NA' if x is None else f'{x:.4f}' for x in values)+'|\n'
    boundary = ('Uniform removes the complete adaptive router (both position-dependent and task-dependent depth weights). '
                'Shared preserves spatial routing but removes task-specific depth weights. Neither alters the original Detect towers. '
                'The differences overlap and must not be added as independent contributions. Fixed seed1 only; all results retained.')
    c.write(report_root/'summary.json',dict(status=state['status'],results=results,boundary=boundary,session=state['session']))
    (report_root/'REPORT.md').write_text('# E60 routing ablations / seed1\n\n'+table+'\n'+boundary+'\n\n'+
        'Execution: '+args.execution+'. Detailed training, check, smoke and official evaluation states are retained in the logs session.\n',encoding='utf-8')
    print(c.now()+' SUMMARY='+str(report_root/'REPORT.md'),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('check','smoke','run','all','eval','status'))
    parser.add_argument('--execution',choices=('serial','parallel'),default='serial')
    parser.add_argument('--project',type=Path,default=Path('/root/autodl-tmp/yolo11_project'))
    parser.add_argument('--source',type=Path,default=Path('/root/autodl-tmp/yolo11_src'))
    parser.add_argument('--data',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019.yaml'))
    parser.add_argument('--dataset',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val'))
    parser.add_argument('--auto-shutdown',action='store_true')
    args = parser.parse_args()
    index = args.project/'logs'/'latest_e60_routing_ablations.json'
    if args.mode == 'status':
        print(__import__('json').dumps(c.read(Path(c.read(index)['session'])/'state.json'),ensure_ascii=False,indent=2)); return
    with exclusive(args.project):
        c.package_audit(); admission()
        stamp = time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
        session = args.project/'logs'/('e60_routing_ablations_'+args.mode+'_'+stamp)
        report_root = args.project/'comparison_reports'/('E60_ROUTING_ABLATIONS_'+args.mode.upper()+'_'+stamp)
        session.mkdir(parents=True,exist_ok=False); report_root.mkdir(parents=True,exist_ok=False)
        state = dict(status='running',mode=args.mode,execution=args.execution,started_at=c.now(),
                     session=str(session),report_root=str(report_root),auto_shutdown=args.auto_shutdown)
        c.write(index,dict(session=str(session),report_root=str(report_root))); c.write(session/'state.json',state)
        print(c.now()+' SESSION='+str(session)+' EXECUTION='+args.execution,flush=True)
        try:
            if args.mode in ('check','all'):
                run_stage(args,'check',session,report_root,state)
            if args.mode in ('smoke','all'):
                run_stage(args,'smoke',session,report_root,state)
            if args.mode in ('run','all'):
                run_stage(args,'verify-gates',session,report_root,state)
                for variant in c.VARIANTS:
                    c.require(not (args.project/'runs'/c.EXPERIMENTS[variant]).exists(),
                              'Existing run; never retrain/overwrite: '+variant+' (use eval for recovery)')
                state['training_exit_codes'] = run_stage(args,'run',session,report_root,state)
            if args.mode in ('run','all','eval'):
                # Release training contexts before every official inference job.
                failures = {}
                for variant in c.VARIANTS:
                    try:
                        completed(args,variant); wait_released()
                        job = launch(args,variant,'eval',session,report_root,state)
                        code = wait_jobs([job],session,state)[variant]
                        if code != 0:
                            failures[variant] = 'official evaluation failed'
                        else:
                            training_root = completed(args,variant)
                            training = c.read(training_root/'training.json')
                            training.update(status='completed',official_evaluation_report=str(job['out']),finished_at=c.now())
                            c.write(training_root/'training.json',training)
                            c.write(training_root/'completion.json',dict(status='completed',trained_epochs=200,
                                    official_evaluation_completed=True,finished_at=c.now()))
                    except Exception as error:
                        failures[variant] = str(error)
                state['evaluation_errors'] = failures
                training_failed = any(code != 0 for code in state.get('training_exit_codes',{}).values())
                state['status'] = 'completed' if not failures and not training_failed else 'completed_with_failures'
                summarize(args,report_root,state)
            else:
                state['status'] = 'completed'
            state['finished_at'] = c.now()
        except BaseException:
            state.update(status='failed_or_interrupted',error=traceback.format_exc(),finished_at=c.now())
            c.write(session/'abort.json',dict(error=state['error']))
            raise
        finally:
            c.write(session/'state.json',state)
            if args.auto_shutdown and args.mode in ('run','all','eval'):
                with (session/'shutdown.log').open('w',encoding='utf-8') as handle:
                    helper = subprocess.Popen([sys.executable,'-u',str(c.HERE/'shutdown_after.py'),
                              '--parent',str(os.getpid()),'--session',str(session),'--project',str(args.project)],
                              stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                state['shutdown_helper_pid'] = helper.pid
                c.write(session/'state.json',state)
                print(c.now()+' optional shutdown helper queued; it waits for parent exit and checks idle twice',flush=True)
            else:
                print(c.now()+' AUTO SHUTDOWN disabled',flush=True)
        if state['status'] == 'completed_with_failures':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
