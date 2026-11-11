#!/usr/bin/env python3
"""All seven gates FIRST, then at most two formal workers, serial scoring."""
from __future__ import annotations
import argparse, os, signal, subprocess, sys, time, traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common7 import *
import platform_safety as safety

MEMORY_FAILURES={'cuda_oom','gpu_headroom'}

def pair_decision(results):
    """Only authoritative CUDA memory failures + cancelled peers allow fallback."""
    failed=[r for r in results if r['exit_code']!=0]
    if not failed:return 'parallel'
    cats={r.get('category','error') for r in failed}
    if cats & MEMORY_FAILURES and cats <= MEMORY_FAILURES|{'peer_failed'}:return 'serial'
    raise RuntimeError('Non-memory joint smoke failure; stopping without shutdown: '+str(failed))

def planned_waves(mode):return [list(w) for w in WAVES] if mode=='parallel' else [[key] for key in JOBS]

def gates_ready(report):
    admission=read(Path(report)/'admission.json')
    for key,job in JOBS.items():
        for phase in ['check','smoke']:
            out=Path(report)/'jobs'/key/phase;r=read(out/(phase+'_receipt.json'))
            require(r['binding']['admission']==admission['binding_id'] and r['binding']['key']==key and r['binding']['job']==job,'Wrong/stale gate: '+key)
            verify_receipt(out,phase,r['binding'])
    return admission['binding_id']

def ram_available():
    values={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
    available=values['MemAvailable'];limit=None
    for lp,up in [('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        if Path(lp).is_file() and Path(up).is_file():
            raw=Path(lp).read_text().strip()
            if raw.isdigit() and int(raw)<2**60:
                limit=int(raw);available=min(available,max(0,limit-int(Path(up).read_text())))
            break
    require(available>=12*2**30,'Less than 12GiB effective available RAM; use GPU mode with sufficient RAM, not 2GiB no-card mode')
    return dict(available_bytes=available,cgroup_limit_bytes=limit)

def admission(args):
    require(sys.platform=='linux' and Path(sys.prefix).resolve()==PYTHON.parent.parent.resolve(),'Run only with the YOLO11 environment on Linux')
    package=package_audit();disk=free_disk(args.project,8)
    gpu=safety.gpu_snapshot();processes=safety.process_snapshot(HERE)
    safety.require_idle(gpu,processes,admission=True)
    ram=ram_available()
    command=safety.shutdown_invocation() if args.enable_shutdown else None
    for job in JOBS.values():require(not (args.project/'runs'/job['run']).exists(),'Target run already exists; refusing overwrite: '+job['run'])
    return dict(package_sha256=package,free_disk_GiB=disk,gpu=gpu,processes=processes,RAM=ram,shutdown_invocation=command)

class Controller:
    def __init__(self,args,session,report,state):
        self.args,self.session,self.report,self.state=args,session,report,state
        self.live={};self.log_handles={};self.guard=None
    def save(self):
        self.state['updated_at']=now();write(self.session/'state.json',self.state);write(self.report/'state.json',self.state)
    def stage(self,name):
        self.state['stage']=name;self.save();print(now()+' STAGE '+name,flush=True)
    def guard_alive(self):
        if self.guard is not None:require(self.guard.poll() is None,'Shutdown guard exited unexpectedly; stopping new work, no shutdown')
    def arm_guard(self):
        if not self.args.enable_shutdown:return
        handle=(self.session/'shutdown_guard.log').open('ab',buffering=0)
        cmd=[str(PYTHON),'-I','-u',str(HERE/'shutdown_guard.py'),'--project',str(self.args.project),'--session',str(self.session),'--report',str(self.report)]
        try:self.guard=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        finally:handle.close()
        self.state['shutdown_guard']=dict(pid=self.guard.pid,identity=safety.process_identity(self.guard.pid),log=str(self.session/'shutdown_guard.log'));self.save()
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            self.guard_alive()
            p=self.session/'shutdown_status.json'
            if p.is_file() and read(p).get('status')=='armed':return
            time.sleep(.2)
        raise RuntimeError('Shutdown guard did not acknowledge arming')
    def start(self,key,phase,out,cap=28.,pair=None,split=None):
        self.guard_alive();require(len(self.live)<2,'More than two workers forbidden')
        tag=key+'_'+phase+('_'+split if split else '')+('_'+pair.name if pair else '')
        require(tag not in self.state['jobs'],'Duplicate job stage')
        log=self.session/(tag+'.log');handle=log.open('ab',buffering=0)
        cmd=[str(PYTHON),'-I','-u',str(HERE/'worker.py'),'--key',key,'--phase',phase,
             '--project',str(self.args.project),'--source',str(self.args.source),'--data',str(self.args.data),
             '--dataset',str(self.args.dataset),'--test-dataset',str(self.args.test_dataset),
             '--report',str(self.report),'--out',str(out),'--cap-gib',str(cap)]
        if pair:cmd+=['--pair',str(pair)]
        if split:cmd+=['--split',split]
        try:p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        except BaseException:handle.close();raise
        self.live[tag]=p;self.log_handles[tag]=handle
        self.state['jobs'][tag]=dict(key=key,phase=phase,split=split,status='running',pid=p.pid,identity=safety.process_identity(p.pid),started_at=now(),log=str(log),out=str(out),cap_gib=cap)
        self.save();print(now()+' START '+tag+' LOG='+str(log),flush=True)
        return tag
    def wait(self,tags):
        while any(self.live[t].poll() is None for t in tags):
            self.save();time.sleep(5)
        results=[]
        for tag in tags:
            p=self.live.pop(tag);rc=p.wait();self.log_handles.pop(tag).close()
            job=self.state['jobs'][tag];job.update(exit_code=rc,status='completed' if rc==0 else 'failed',finished_at=now())
            fail=Path(job['out'])/'failure.json'
            if fail.is_file():job['category']=read(fail).get('category','error')
            elif rc:job['category']='error'
            results.append(dict(job));print(now()+' EXIT '+tag+' code='+str(rc),flush=True)
        self.save();return results
    def single(self,key,phase,out,split=None):
        result=self.wait([self.start(key,phase,out,split=split)])[0]
        require(result['exit_code']==0,'Stage failed; logs retained, no shutdown: '+result['log'])
    def preflight(self):
        self.stage('prepare_all_inputs');self.single(next(iter(JOBS)),'prepare',self.report/'prepare')
        for key in JOBS:
            self.stage('check_'+key);self.single(key,'check',self.report/'jobs'/key/'check')
        for key in JOBS:
            self.stage('individual_smoke_'+key);self.single(key,'smoke',self.report/'jobs'/key/'smoke')
        gates_ready(self.report)
        cap=min(15.,(self.state['admission']['gpu']['total_MiB']/1024-2.)/2.)
        mode='parallel';parallel_jobs={};joint=[]
        if self.args.serial:mode='serial'
        else:
            for i,keys in enumerate(WAVES):
                if len(keys)<2:continue
                self.stage('parallel_smoke_'+str(i+1))
                pair=self.report/'pair_gates'/('wave'+str(i+1));pair.mkdir(parents=True,exist_ok=False)
                write(pair/'members.json',dict(keys=keys,cap_gib=cap,binding=read(self.report/'admission.json')['binding_id']))
                tags=[self.start(k,'smoke',pair/k,cap,pair=pair) for k in keys]
                results=self.wait(tags);choice=pair_decision(results)
                joint.append(dict(wave=i+1,keys=keys,decision=choice,results=results))
                if choice=='serial':
                    mode='serial';parallel_jobs={}
                    print(now()+' JOINT CUDA MEMORY LIMIT: use serial, batch16/200e unchanged',flush=True)
                    break
                for k in keys:parallel_jobs[k]=dict(out=str(pair/k),pair=str(pair),cap_gib=cap)
        binding=gates_ready(self.report)
        plan=dict(status='all_gates_passed',binding=binding,execution=mode,waves=planned_waves(mode),
                  parallel_jobs=parallel_jobs,joint_smokes=joint,at=now(),unchanged_batch=16,unchanged_epochs=200)
        write(self.report/'execution_plan.json',plan)
        self.state.update(status='training',execution=mode,all_7_checks_passed=True,all_7_individual_smokes_passed=True);self.save()
        return plan
    def execute(self):
        self.arm_guard();plan=self.preflight()
        for i,wave in enumerate(plan['waves']):
            self.stage('formal_wave_'+str(i+1))
            cap=plan['parallel_jobs'][wave[0]]['cap_gib'] if len(wave)==2 else 28.
            results=self.wait([self.start(k,'train',self.report/'jobs'/k/'train',cap) for k in wave])
            # Let both members finish/save; never kill a healthy formal peer.
            # Score healthy finished workers even when the other one failed.
            for r in results:
                if r['exit_code']!=0:continue
                for split in ['val','test-dev']:
                    self.stage('official_'+r['key']+'_'+split)
                    self.single(r['key'],'score',self.report/'jobs'/r['key']/split,split=split)
                self.mark_scored(r['key'])
            require(all(r['exit_code']==0 for r in results),'Formal wave failed; healthy peer saved/scored, no new waves, no shutdown')
            self.guard_alive()
        self.stage('summarize_and_verify')
        from summary7 import summarize
        from preservation7 import seal_results
        summarize(self.args.project,self.report)
        proof=completed_inventory(self.args.project,self.report)
        seal=seal_results(self.args.project,self.report,proof)
        self.state.update(status='completed',stage='completed',finished_at=now(),completion_seal_sha256=sha(self.report/'COMPLETION_SEAL.json'),
                          all_7_formal200_saved=True,all_14_official_scores_completed=True)
        self.save()
        if hasattr(os,'sync'):os.sync()
        print(now()+' ALL SEVEN COMPLETED / FOURTEEN OFFICIAL SCORES / SAVED AND VERIFIED',flush=True)
        if self.args.enable_shutdown:print('Shutdown guard armed: begins 120-second cancellable countdown after controller exits.',flush=True)
    def mark_scored(self,key):
        out=self.report/'jobs'/key/'train'
        p=out/'training_completion.json';t=read(p);t.update(official_scores_completed=True,official_scores_finished_at=now());write(p,t)
        p=out/'training.json';t=read(p);t.update(status='completed',official_scores_completed=True,official_scores_finished_at=now());write(p,t)

def parse():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['all'])
    p.add_argument('--project',type=Path,default=DEFAULT_PROJECT);p.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--data',type=Path,default=DEFAULT_DATA)
    p.add_argument('--dataset',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val'))
    p.add_argument('--test-dataset',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-test-dev'))
    p.add_argument('--enable-shutdown',action='store_true');p.add_argument('--serial',action='store_true')
    a=p.parse_args()
    for name in ['project','source','data','dataset','test_dataset']:setattr(a,name,getattr(a,name).resolve())
    return a

def main():
    args=parse();controller=None
    try:
        import fcntl
        logs=args.project/'logs';logs.mkdir(parents=True,exist_ok=True)
        lock=(logs/'.stmdrf_supp7.lock').open('a+')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pre=admission(args)
        stamp=__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
        session=logs/('stmdrf_supp7_'+stamp);session.mkdir(exist_ok=False)
        report=args.project/'comparison_reports'/('ST_MDRF_SUPP7_'+stamp);report.mkdir(parents=True,exist_ok=False)
        state=dict(status='preflight',stage='admission_passed',started_at=now(),PID=os.getpid(),controller_identity=safety.process_identity(os.getpid()),
                   auto_shutdown=args.enable_shutdown,admission=pre,jobs={},session=str(session),report=str(report),
                   cancel_file=str(args.project/'KEEP_RUNNING_ST_MDRF_SUPP7'))
        controller=Controller(args,session,report,state);controller.save()
        write(logs/'latest_stmdrf_supp7.json',dict(session=str(session),report=str(report),package=str(HERE),PID=os.getpid(),at=now()))
        def interrupted(signum,frame):raise RuntimeError('Controller interrupted by signal '+str(signum))
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        controller.execute();return 0
    except BaseException:
        error=traceback.format_exc();print(error,file=sys.stderr,flush=True)
        if controller is not None:
            # A terminal state is not written until our own still-running children exit.
            # In particular, no finally block ever calls shutdown.
            if controller.live:
                print(now()+' STOPPING NEW WORK; waiting for existing owned workers to save/exit',flush=True)
                try:controller.wait(list(controller.live))
                except BaseException:pass
            controller.state.update(status='failed',finished_at=now(),error=error,stage=controller.state['stage']);controller.save()
        return 1

if __name__=='__main__':sys.exit(main())
