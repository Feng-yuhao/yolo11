"""Seed1 pair only: checks, two CUDA smoke passes, serial full tests."""
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
    require(free>=9*1024**3,'Need >=9GiB free AFTER dataset extraction for both raw/filtered prediction sets; no files were deleted')
    return dict(free_bytes=free,minimum_free_bytes=9*1024**3)


def summarize(report):
    scores={k:read(report/'evaluation'/k/'coco_metrics.json') for k in KEYS}
    native={k:read(report/'evaluation'/k/'native_ap70.json') for k in KEYS}
    require(all(s['status']=='completed' and s['images']==16592 for s in scores.values()),'Missing complete pair')
    fields=['AP_percent','AP50_percent','AP75_percent','AP_small_percent','AP_medium_percent','AP_large_percent']
    gains={f:None if any(scores[k][f] is None for k in KEYS) else scores['e60_seed1'][f]-scores['e1_seed1'][f] for f in fields}
    summary=dict(protocol=PROTOCOL,source_seed=1,models=scores,paired_gain_pp=gains,native_overall=native,
                 native_gain_pp=native['e60_seed1']['AP70_overall_percent']-native['e1_seed1']['AP70_overall_percent'])
    write(report/'summary.json',summary)
    lines=['# VisDrone -> UAVDT / seed1 direct transfer','',
           'No target training, BN update, checkpoint or threshold selection.',
           'COCO-style three-class AP uses original DET GT and author strict ignore-area prediction filtering.',
           'Not an official UAVDT COCO score. Native overall AP70 is a separate author-math port.',
           'Fixed mapping: max(car,van), truck, bus; before class-aware NMS/top500.','',
           '|Metric (%)|E1 seed1|E60 seed1|E60 minus E1 (pp)|','|---|---:|---:|---:|']
    for f in fields:
        a,b=scores['e1_seed1'][f],scores['e60_seed1'][f]
        lines.append(f'|{f}|'+ ('NA|NA|NA|' if gains[f] is None else f'{a:.4f}|{b:.4f}|{gains[f]:+.4f}|'))
    a,b=(native[k]['AP70_overall_percent'] for k in KEYS)
    lines+= [f'|Native-math overall AP70 (class-agnostic)|{a:.4f}|{b:.4f}|{b-a:+.4f}|','',
             'All positive and negative results retained. One paired source seed is descriptive, not multi-seed stability.',
             'AP-small uses ORIGINAL 1024x540 GT area. Raw and ignore-filtered predictions saved.',
             'Direct transfer differs from target retraining. No shutdown entry exists.']
    (report/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def main(args):
    args.project,args.source,args.dataset=(p.resolve() for p in (args.project,args.source,args.dataset))
    package_audit()
    require(sys.platform=='linux' and Path(sys.prefix).resolve()==Path('/root/miniconda3/envs/yolo11'),
            'Use YOLO11 environment, not base/YOLO26')
    if args.mode=='prepare':
        print(__import__('json').dumps(prepare(args),indent=2)); return
    import fcntl
    logs=args.project/'logs'; logs.mkdir(parents=True,exist_ok=True)
    with (logs/'.visdrone_to_uavdt_seed1_source_only.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        from admission import check_idle
        check_idle()
        disk_admission(args.project)
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S')+f'_{os.getpid()}'
        report=args.project/'comparison_reports'/('VISDRONE_TO_UAVDT_S1_SOURCE_ONLY_'+stamp)
        report.mkdir(parents=True,exist_ok=False)
        for name in [*read(HERE/'manifest.json')['files'],'manifest.json']:
            dest=report/'actual_code'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(HERE/name,dest)
        write(logs/'latest_visdrone_to_uavdt_seed1_source_only.json',dict(report=str(report),at=now()))
        state=dict(status='running',stage='prepare',started_at=now(),PID=os.getpid(),training=False,auto_shutdown=False,jobs={})
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
            summarize(report)
            require(weights_audit(args.project)==binding['weights'],'Source weights changed')
            require(target_audit(args.dataset,False)['test_file_stat_sha256']==binding['target']['test_file_stat_sha256'],'Target data changed')
            state.update(status='finalizing',stage='hashing_results'); write(report/'state.json',state)
            write(report/'ARTIFACT_SHA256.json',{p.relative_to(report).as_posix():sha(p) for p in sorted(report.rglob('*'))
                                               if p.is_file() and p.name not in ('ARTIFACT_SHA256.json','state.json')})
            state.update(status='completed',stage='completed',finished_at=now(),seconds=time.monotonic()-started,
                         artifact_manifest_sha256=sha(report/'ARTIFACT_SHA256.json')); write(report/'state.json',state)
            print(f'{now()} BOTH SEED1 EVALUATIONS COMPLETED REPORT={report}',flush=True)
            print((report/'REPORT.md').read_text(encoding='utf-8'),flush=True)
        except BaseException:
            state.update(status='failed',error=traceback.format_exc(),finished_at=now(),seconds=time.monotonic()-started)
            write(report/'state.json',state); print(state['error'],flush=True)
            print(f'Partial results retained: {report}. Server left running.',flush=True); raise


if __name__=='__main__':
    p=parser(); p.add_argument('mode',choices=('prepare','all')); main(p.parse_args())
