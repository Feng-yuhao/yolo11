"""Seed2/3 pairs: checks, four CUDA smoke passes, serial full tests."""
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from common import HERE, KEYS, PROTOCOL, REVISION, now, package_audit, parser, read, require, sha, source_audit, target_audit, weights_audit, write


def prepare(args, full=True):
    return dict(revision=REVISION,package_sha256=package_audit(),source=source_audit(args.source),
                weights=weights_audit(args.project),target=target_audit(args.dataset,full),protocol=PROTOCOL)


def disk_admission(project):
    free=shutil.disk_usage(project).free
    require(free>=6*1024**3,'Need >=6GiB free after dataset extraction for four compact prediction sets; no files deleted')
    return dict(free_bytes=free,minimum_free_bytes=6*1024**3)




def main(args):
    args.project,args.source,args.dataset=(p.resolve() for p in (args.project,args.source,args.dataset))
    package_audit()
    require(sys.platform=='linux' and Path(sys.prefix).resolve()==Path('/root/miniconda3/envs/yolo11'),
            'Use YOLO11 environment, not base/YOLO26')
    if args.mode=='prepare':
        print(__import__('json').dumps(prepare(args),indent=2)); return
    import fcntl
    logs=args.project/'logs'; logs.mkdir(parents=True,exist_ok=True)
    with (logs/'.visdrone_to_uavdt_seeds23_source_only.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        from admission import check_idle
        check_idle()
        disk_admission(args.project)
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S')+f'_{os.getpid()}'
        report=args.project/'comparison_reports'/('VISDRONE_TO_UAVDT_S23_SOURCE_ONLY_'+stamp)
        report.mkdir(parents=True,exist_ok=False)
        for name in [*read(HERE/'manifest.json')['files'],'manifest.json']:
            dest=report/'actual_code'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(HERE/name,dest)
        write(logs/'latest_visdrone_to_uavdt_seeds23_source_only.json',dict(report=str(report),at=now()))
        from shutdown_safety import process_identity, shutdown_invocation
        command=shutdown_invocation() if args.auto_shutdown else None
        state=dict(status='running',stage='prepare',started_at=now(),PID=os.getpid(),training=False,
                   auto_shutdown=args.auto_shutdown,controller_identity=process_identity(os.getpid()),shutdown_command=command,jobs={})
        write(report/'state.json',state); started=time.monotonic()
        try:
            binding=prepare(args)
            from uavdt_data import prepare_gt
            binding['generated_GT']=prepare_gt(args.dataset,report/'target')
            write(report/'binding.json',binding); write(report/'protocol.json',PROTOCOL)
            for phase,key in [('check',None)]+[(p,k) for p in ('smoke','evaluation') for k in KEYS]:
                name=phase if key is None else phase+'_'+key
                state.update(stage=name); write(report/'state.json',state)
                write(report/'admission'/(name+'.json'),check_idle())
                script='test_release.py' if phase=='check' else 'worker.py'
                command=[sys.executable,'-u',str(HERE/script),'--project',str(args.project),'--source',str(args.source),
                         '--dataset',str(args.dataset),'--report',str(report)]
                if key: command+=['--phase',phase,'--key',key]
                log=report/'logs'/(name+'.log'); log.parent.mkdir(parents=True,exist_ok=True)
                print(f'{now()} START {name} LOG={log}',flush=True)
                env=os.environ.copy(); env.pop('PYTHONPATH',None); env.pop('PYTHONHOME',None); env['PYTHONUTF8']='1'
                with log.open('w',encoding='utf-8') as handle:
                    child=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
                    state['jobs'][name]=dict(pid=child.pid,log=str(log),status='running',started_at=now()); write(report/'state.json',state)
                    code=child.wait()
                state['jobs'][name].update(exit_code=code,status='completed' if code==0 else 'failed',finished_at=now())
                write(report/'state.json',state); require(code==0,name+' failed; read '+str(log))
            from summary import summarize
            summarize(report)
            require(weights_audit(args.project)==binding['weights'],'Source weights changed')
            require(target_audit(args.dataset,False)['test_file_stat_sha256']==binding['target']['test_file_stat_sha256'],'Target data changed')
            state.update(status='finalizing',stage='hashing_results'); write(report/'state.json',state)
            write(report/'ARTIFACT_SHA256.json',{p.relative_to(report).as_posix():sha(p) for p in sorted(report.rglob('*'))
                                               if p.is_file() and p.name not in ('ARTIFACT_SHA256.json','state.json')})
            state.update(status='completed',stage='completed',finished_at=now(),seconds=time.monotonic()-started,
                         artifact_manifest_sha256=sha(report/'ARTIFACT_SHA256.json')); write(report/'state.json',state)
            print(f'{now()} ALL FOUR SEED2/3 EVALUATIONS COMPLETED REPORT={report}',flush=True)
            print((report/'REPORT.md').read_text(encoding='utf-8'),flush=True)
            if args.auto_shutdown:
                log=report/'shutdown_launcher.log'
                with log.open('a',encoding='utf-8') as out:
                    subprocess.Popen([sys.executable,'-u',str(HERE/'shutdown_after.py'),'--report',str(report),
                                      '--project',str(args.project)],stdout=out,stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
        except BaseException:
            state.update(status='failed',error=traceback.format_exc(),finished_at=now(),seconds=time.monotonic()-started)
            write(report/'state.json',state); print(state['error'],flush=True)
            print(f'Partial results retained: {report}. Server left running.',flush=True); raise


if __name__=='__main__':
    p=parser(); p.add_argument('mode',choices=('prepare','all')); p.add_argument('--auto-shutdown',action='store_true'); main(p.parse_args())
