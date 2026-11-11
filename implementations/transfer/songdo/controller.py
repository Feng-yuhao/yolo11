"""CPU controller: prepare -> unit checks -> six smoke passes -> six serial evaluations."""
from datetime import datetime
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from common import HERE, KEYS, PROTOCOL, REVISION, now, package_audit, parser, read, require, sha, source_audit, target_audit, weights_audit, write


def prepare(args, full=True):
    return dict(revision=REVISION,package_sha256=package_audit(),source=source_audit(args.source),
                weights=weights_audit(args.project),target=target_audit(args.dataset,full),protocol=PROTOCOL)


def summarize(report):
    scores = {key:read(report/'evaluation'/key/'coco_metrics.json') for key in KEYS}
    require(all(v['status'] == 'completed' and v['images'] == 1084 for v in scores.values()), 'Missing full results')
    fields = ['AP_percent','AP50_percent','AP75_percent','AP_small_percent','AP_medium_percent','AP_large_percent']
    summary = dict(protocol=PROTOCOL,source_seeds=[1,2,3],models=scores,paired={},aggregates={})
    for field in fields:
        values = {kind:[scores[f'{kind}_seed{s}'][field] for s in (1,2,3)] for kind in ('e1','e60')}
        if any(v is None for group in values.values() for v in group):
            summary['aggregates'][field] = None
            continue
        gains = [b-a for a,b in zip(values['e1'],values['e60'])]
        summary['paired'][field] = gains
        summary['aggregates'][field] = dict(
            **{kind:dict(mean=statistics.mean(v),sample_SD=statistics.stdev(v)) for kind,v in values.items()},
            paired_gain_mean_pp=statistics.mean(gains),paired_gain_sample_SD_pp=statistics.stdev(gains))
    write(report/'summary.json',summary)
    lines = ['# VisDrone -> Songdo / frozen source-only direct transfer','',
             'No Songdo training, adaptation, BN updates, checkpoint or threshold selection.',
             'All six fixed source-val best checkpoints; standard COCO bbox AP, maxDets100.',
             'Car includes source car+van (maximum score per anchor, mapped before NMS).','',
             '|Source model / seed|AP (%)|AP50 (%)|AP75 (%)|', '|---|---:|---:|---:|']
    for key in KEYS:
        v = scores[key]
        lines.append(f'|{key}|{v["AP_percent"]:.4f}|{v["AP50_percent"]:.4f}|{v["AP75_percent"]:.4f}|')
    lines += ['', '|Paired source seed|E60 minus E1 AP (pp)|','|---|---:|']
    for seed,gain in zip((1,2,3),summary['paired']['AP_percent']):
        lines.append(f'|{seed}|{gain:+.4f}|')
    lines += ['', '|Metric|E1 mean +/- sample SD|E60 mean +/- sample SD|Paired gain (pp)|',
              '|---|---:|---:|---:|']
    for field in fields:
        row = summary['aggregates'][field]
        if row is not None:
            a,b = row['e1'],row['e60']
            lines.append(f'|{field}|{a["mean"]:.4f} +/- {a["sample_SD"]:.4f}|'
                         f'{b["mean"]:.4f} +/- {b["sample_SD"]:.4f}|{row["paired_gain_mean_pp"]:+.4f}|')
    lines += ['', 'AP-small uses ORIGINAL-image areas. Target-retrained Songdo results are a different setting;',
              'do not mix their AP gains with this source-only evaluation. All negative results are retained.',
              'Three source checkpoints describe repeatability, not independent target datasets or a significance guarantee.',
              'The source models have shared pretraining. This is not unseen-class/open-vocabulary zero-shot detection.',
              'No shutdown command exists in this package.']
    (report/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def main(args):
    args.project,args.source,args.dataset = (p.resolve() for p in (args.project,args.source,args.dataset))
    package_audit()
    require(sys.platform == 'linux' and Path(sys.prefix).resolve() == Path('/root/miniconda3/envs/yolo11'),
            'Use the YOLO11 environment, not base/YOLO26')
    if args.mode == 'prepare':
        print(__import__('json').dumps(prepare(args),ensure_ascii=False,indent=2))
        return
    import fcntl
    logs = args.project/'logs'; logs.mkdir(parents=True,exist_ok=True)
    with (logs/'.visdrone_to_songdo_source_only.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        from admission import check_idle
        check_idle()  # Fail before creating a run if another training is active.
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')+f'_{os.getpid()}'
        report = args.project/'comparison_reports'/('VISDRONE_TO_SONGDO_SOURCE_ONLY_'+stamp)
        report.mkdir(parents=True,exist_ok=False)
        for name in [*read(HERE/'manifest.json')['files'],'manifest.json']:
            target = report/'actual_code'/name
            target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(HERE/name,target)
        write(logs/'latest_visdrone_to_songdo_source_only.json',dict(report=str(report),at=now()))
        state = dict(status='running',stage='prepare',started_at=now(),PID=os.getpid(),
                     training=False,auto_shutdown=False,jobs={})
        write(report/'state.json',state)
        started = time.monotonic()
        try:
            binding = prepare(args)
            write(report/'binding.json',binding)
            write(report/'protocol.json',PROTOCOL)
            commands = [('check',None)] + [(phase,key) for phase in ('smoke','evaluation') for key in KEYS]
            for phase,key in commands:
                name = phase if key is None else phase+'_'+key
                state.update(stage=name); write(report/'state.json',state)
                admission = check_idle()
                write(report/'admission'/(name+'.json'),admission)
                script = 'test_release.py' if phase == 'check' else 'worker.py'
                command = [sys.executable,'-u',str(HERE/script),'--project',str(args.project),
                           '--source',str(args.source),'--dataset',str(args.dataset),'--report',str(report)]
                if key is not None:
                    command += ['--phase',phase,'--key',key]
                log = report/'logs'/(name+'.log'); log.parent.mkdir(parents=True,exist_ok=True)
                print(f'{now()} START {name} LOG={log}',flush=True)
                env = os.environ.copy()
                env.pop('PYTHONPATH',None); env.pop('PYTHONHOME',None)
                env['PYTHONUTF8'] = '1'
                with log.open('w',encoding='utf-8') as handle:
                    child = subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
                    state['jobs'][name] = dict(pid=child.pid,log=str(log),status='running',started_at=now())
                    write(report/'state.json',state)
                    code = child.wait()
                state['jobs'][name].update(exit_code=code,status='completed' if code == 0 else 'failed',finished_at=now())
                write(report/'state.json',state)
                require(code == 0, name+' failed; read '+str(log)+'. No shutdown or process kill was requested.')
            summarize(report)
            require(weights_audit(args.project) == binding['weights'], 'Source weights changed by another task')
            require(target_audit(args.dataset,False)['test_file_stat_sha256'] == binding['target']['test_file_stat_sha256'],
                    'Target data changed during evaluation')
            state.update(status='finalizing',stage='hashing_results')
            write(report/'state.json',state)
            write(report/'ARTIFACT_SHA256.json',{str(p.relative_to(report).as_posix()):sha(p)
                  for p in sorted(report.rglob('*')) if p.is_file() and p.name not in ('ARTIFACT_SHA256.json','state.json')})
            state.update(status='completed',stage='completed',finished_at=now(),seconds=time.monotonic()-started,
                         artifact_manifest_sha256=sha(report/'ARTIFACT_SHA256.json'))
            write(report/'state.json',state)
            print(f'{now()} ALL SIX EVALUATIONS COMPLETED REPORT={report}',flush=True)
            print((report/'REPORT.md').read_text(encoding='utf-8'),flush=True)
        except BaseException:
            state.update(status='failed',error=traceback.format_exc(),finished_at=now(),seconds=time.monotonic()-started)
            write(report/'state.json',state)
            print(state['error'],flush=True)
            print(f'Partial results preserved: {report}; server left running.',flush=True)
            raise


if __name__ == '__main__':
    p = parser(); p.add_argument('mode',choices=('prepare','all')); main(p.parse_args())
