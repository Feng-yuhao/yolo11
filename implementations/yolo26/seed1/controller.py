"""Serial pair, complete state/logs; no training on smoke/check commands."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import common as c
from protocol import write,now,signature,verify_gate,run_name,DATA


def summarize(root):
    results={name:json.loads((root/(name+'_eval')/'official_metrics.json').read_text()) for name in ('baseline','e60')}
    if results['baseline']['dataset']!=results['e60']['dataset']:
        raise RuntimeError('Pair scored on different original annotations')
    keys=('AP50_95_percent','AP50_percent','AP75_percent','AR500_percent')
    base=results['baseline']['official_metrics']; enhanced=results['e60']['official_metrics']
    delta={key:enhanced[key]-base[key] for key in keys}
    write(root/'comparison.json',dict(status='completed',baseline=results['baseline'],e60=results['e60'],delta_pp=delta,
          seed=1,architecture_transfer_only=True,single_seed_not_stability=True))
    lines=['# YOLO26s-P2 / E60 migration / seed1','',
           '|Metric|YOLO26s-P2 baseline|YOLO26s-P2 + E60|Delta pp|','|---|---:|---:|---:|']
    for key in keys: lines.append(f'|{key}|{base[key]:.4f}|{enhanced[key]:.4f}|{delta[key]:+.4f}|')
    lines.extend(['','Both candidates: original progressive dual-head loss, inference one2many+NMS, max_det500.',
       'Weights initialized from the same explicit official YOLO26s -> P2 mapping; not old YOLO11 weights.',
       'Official VisDrone2019 DET Matlab1.0.4 semantic port; YOLO mAP is separately recorded.',
       'This is one-seed cross-architecture evidence, not a stock NMS-free benchmark or proof of stable generalization.',
       'No test-dev was used to select architecture, checkpoint or thresholds. Negative results are retained.',''])
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('PAIR OFFICIAL DELTA pp='+json.dumps(delta)+' REPORT='+str(root/'REPORT.md'),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('smoke','verify-gates','run'))
    args=parser.parse_args()
    args.project=c.PROJECT; args.source=c.SOURCE; args.data=DATA
    args.val_dataset=DATA.parent/'VisDrone2019-DET-val'; args.local_test=False
    c.runtime(args.project)
    import fcntl
    lock=args.project/'.yolo26_p2_e60_seed1_pipeline.lock'
    handle=lock.open('a+')
    try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('Another YOLO26 pair controller is already active')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    report=args.project/'comparison_reports'/('YOLO26_P2_E60_'+args.mode.upper()+'_'+stamp)
    session=args.project/'logs'/('yolo26_p2_e60_'+args.mode+'_'+stamp)
    report.mkdir(parents=True); session.mkdir(parents=True)
    state=dict(status='preflight',mode=args.mode,execution='serial',auto_shutdown=False,session=str(session),
               report_root=str(report),started_at=now(),jobs={})
    write(args.project/'logs'/('latest_yolo26_p2_e60_'+args.mode+'.json'),dict(session=str(session),report_root=str(report)))
    write(args.project/'logs/latest_yolo26_p2_e60.json',dict(session=str(session),report_root=str(report),mode=args.mode))
    def save(): write(session/'state.json',state)
    def job(mode,variant,weights=None):
        key=variant+'_'+mode
        command=[sys.executable,'-u',str(c.HERE/'worker.py'),mode,'--variant',variant,
                 '--report',str(report/key),'--signature',str(session/'signature.json')]
        if weights is not None: command+=['--weights',str(weights)]
        log=session/(key+'.log')
        print(f'{now()} START {key}; live output follows; log={log}',flush=True)
        with log.open('w',encoding='utf-8') as file:
            proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            state['jobs'][key]=dict(pid=proc.pid,status='running',log=str(log),report=str(report/key),started_at=now()); save()
            for line in proc.stdout:
                file.write(line); file.flush(); print(line,end='',flush=True)
            code=proc.wait()
        state['jobs'][key].update(exit_code=code,status='completed' if code==0 else 'failed',finished_at=now()); save()
        if code: raise RuntimeError(f'{key} failed exit={code}; see {log}; no automatic retry or gate bypass')
        return report/key
    try:
        save(); sig=signature(args); write(session/'signature.json',sig)
        if args.mode=='smoke':
            state['status']='smoking'; save()
            reports={variant:str(job('smoke',variant)) for variant in ('baseline','e60')}
            receipt=dict(status='passed',local_fixture=False,fingerprint=sig['fingerprint'],signature=sig,
                         reports=reports,report_sha256={v:c.sha256(Path(p)/'smoke.json') for v,p in reports.items()},
                         layout='serial',finished_at=now())
            write(args.project/'comparison_reports/yolo26_p2_e60_seed1_smoke_gate.json',receipt)
            print('BOTH REAL 640/B16 SMOKES PASSED; no formal training was started',flush=True)
        else:
            verify_gate(args,sig)
            if args.mode=='verify-gates': print('REAL SMOKE GATES VALID',flush=True)
            else:
                for variant in ('baseline','e60'):
                    if (args.project/'runs'/run_name(variant)).exists():
                        raise RuntimeError('Refusing to overwrite/restart existing formal run: '+run_name(variant))
                state['status']='training'; save()
                for variant in ('baseline','e60'):
                    directory=job('train',variant)
                    completed=json.loads((directory/'training_completion.json').read_text())
                    job('eval',variant,Path(completed['best']))
                summarize(report)
        state.update(status='completed',finished_at=now()); save()
        print(f'{now()} CONTROLLER COMPLETED; REPORT={report}; auto shutdown DISABLED',flush=True)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc(),finished_at=now()); save()
        raise
    finally: handle.close()


if __name__=='__main__': main()
