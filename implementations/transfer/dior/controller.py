"""CPU preparation, checks, twelve CUDA smokes, twelve serial tests, then opt-in shutdown."""
from datetime import datetime,timedelta,timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from common import HERE,INDEX,KEYS,LOCK,PREFIX,PROTOCOL,REVISION,TARGETS,now,package_audit,parser,read,require,sha,source_audit,weights_audit,write
from target_data import audit,prepare_gt,root_for

def prepare(args,full=True):
    return dict(revision=REVISION,package_sha256=package_audit(),source=source_audit(args.source),weights=weights_audit(args.project),
                datasets={t:audit(root_for(args.data_parent,t),t,full) for t in TARGETS},protocol=PROTOCOL)
def disk_admission(project):
    free=shutil.disk_usage(project).free
    require(free>=int(4.5*1024**3),'Need >=4.5GiB free for12 losslessly compressed outputs +1GiB reserve; no automatic deletion')
    return dict(free_bytes=free,minimum_free_bytes=int(4.5*1024**3))
def main(args):
    args.project,args.source,args.data_parent=(p.resolve() for p in (args.project,args.source,args.data_parent));package_audit()
    require(sys.platform=='linux' and Path(sys.prefix).resolve()==Path('/root/miniconda3/envs/yolo11'),'Use /root/miniconda3/envs/yolo11/bin/python')
    if args.mode=='prepare':print(__import__('json').dumps(prepare(args),indent=2));return
    import fcntl
    from admission import check_idle
    from storage import disk_guard
    logs=args.project/'logs';logs.mkdir(parents=True,exist_ok=True)
    with (logs/LOCK).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_idle();disk_admission(args.project)
        stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')+f'_{os.getpid()}'
        report=args.project/'comparison_reports'/(PREFIX+stamp);report.mkdir(parents=True,exist_ok=False)
        for name in [*read(HERE/'manifest.json')['files'],'manifest.json']:
            dest=report/'actual_code'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(HERE/name,dest)
        write(logs/INDEX,dict(report=str(report),at=now()))
        from shutdown_safety import process_identity,shutdown_invocation
        state=dict(status='running',stage='prepare',started_at=now(),PID=os.getpid(),training=False,execution='serial12',
                   auto_shutdown=args.auto_shutdown,controller_identity=process_identity(os.getpid()),
                   shutdown_command=None,jobs={})
        write(report/'state.json',state);started=time.monotonic()
        try:
            # Bind the platform helper, never call it during prepare/check/smoke.
            if args.auto_shutdown:state['shutdown_command']=shutdown_invocation();write(report/'state.json',state)
            binding=prepare(args)
            binding['generated_GT']={t:prepare_gt(root_for(args.data_parent,t),t,report/'target'/t) for t in TARGETS}
            write(report/'binding.json',binding);write(report/'protocol.json',PROTOCOL)
            stages=[('check',None)]+[(phase,k) for phase in ('smoke','evaluation') for k in KEYS]
            for phase,key in stages:
                name=phase if key is None else phase+'_'+key
                state.update(stage=name);write(report/'state.json',state);disk_guard(args.project)
                write(report/'admission'/(name+'.json'),check_idle())
                script='test_release.py' if phase=='check' else 'worker.py'
                command=[sys.executable,'-u',str(HERE/script),'--project',str(args.project),'--source',str(args.source),
                         '--data-parent',str(args.data_parent),'--report',str(report)]
                if key:command+=['--phase',phase,'--key',key]
                log=report/'logs'/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                print(f'{now()} START {name} LOG={log}',flush=True)
                env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None);env['PYTHONUTF8']='1'
                with log.open('w',encoding='utf-8') as output:
                    child=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
                    state['jobs'][name]=dict(pid=child.pid,log=str(log),started_at=now(),status='running');write(report/'state.json',state)
                    code=child.wait()
                state['jobs'][name].update(exit_code=code,status='completed' if code==0 else 'failed',finished_at=now());write(report/'state.json',state)
                require(code==0,name+' failed; inspect '+str(log))
            from summary import summarize
            summarize(report);require(weights_audit(args.project)==binding['weights'],'Source checkpoints changed')
            for t in TARGETS:require(audit(root_for(args.data_parent,t),t)['file_stat_sha256']==binding['datasets'][t]['file_stat_sha256'],'Target changed')
            state.update(status='finalizing',stage='hashing_results');write(report/'state.json',state)
            write(report/'ARTIFACT_SHA256.json',{p.relative_to(report).as_posix():sha(p) for p in sorted(report.rglob('*')) if p.is_file() and p.name not in ('state.json','ARTIFACT_SHA256.json')})
            state.update(status='completed',stage='verifying_saved_results',finished_at=now(),seconds=time.monotonic()-started,
                         artifact_manifest_sha256=sha(report/'ARTIFACT_SHA256.json'));write(report/'state.json',state)
            from completion import verify_outputs
            verify_outputs(report,require_optin=False)
            state['stage']='completed';write(report/'state.json',state)
            print(f'{now()} ALL12 EVALUATIONS AND SCORING COMPLETED REPORT={report}',flush=True)
            print((report/'REPORT.md').read_text(encoding='utf-8'),flush=True)
            if args.auto_shutdown:
                with (report/'shutdown_launcher.log').open('a',encoding='utf-8') as out:
                    subprocess.Popen([sys.executable,'-u',str(HERE/'shutdown_after.py'),'--report',str(report),'--project',str(args.project)],
                                     stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
        except BaseException:
            state.update(status='failed',error=traceback.format_exc(),finished_at=now(),seconds=time.monotonic()-started);write(report/'state.json',state)
            print(state['error'],flush=True);print(f'Partial results retained at {report}. No shutdown on failure.',flush=True);raise
if __name__=='__main__':
    p=parser();p.add_argument('mode',choices=('prepare','all'));p.add_argument('--auto-shutdown',action='store_true');main(p.parse_args())
