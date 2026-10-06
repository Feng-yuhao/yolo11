#!/usr/bin/env python3
"""Sequential inference audit controller, with live child logs. Never trains."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
MODELS=('e40','e40s2','e42','e43')
OUT=None
KEYS=('AP50_95_percent','AP50_percent','AP75_percent','AR500_percent')


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def summary(out, reports):
    lines=['# D44: frozen routing / Dense inference interventions','',
           'ZERO TRAINING. No checkpoint/source overwrite. No automatic shutdown.',
           'Off-manifold sensitivity, not training-time causal proof or a new trained model.','',
           '|Model|Mode|AP|AP50|AP75|AR500|AP delta vs SAME checkpoint normal|',
           '|---|---|---:|---:|---:|---:|---:|']
    for model,report in reports.items():
        for mode,item in report.get('scores',{}).items():
            values=[item['official'][key] for key in KEYS]
            lines.append('|'+model+'|'+mode+'|'+ '|'.join(f'{v:.6f}' for v in values)+
                         f"|{item['delta_vs_same_checkpoint_normal_pp'][KEYS[0]]:+.6f}|")
    lines += ['', '## Interpretation boundaries', '',
              '- Compare intervention with its OWN checkpoint default, not E40 with different trained weights.',
              '- Turning an operation off at validation is not its matched retraining ablation.',
              '- A rescue supports a follow-up hypothesis; a drop can be distribution mismatch/co-adaptation.',
              '- Small AP differences can be numerical/seed effects; do not declare a root cause.',
              '- Do not select the best intervention as a final paper model or automatically launch E44/E45.',
              '- routing_rows.json uses GT centers / 3x3 feature neighborhoods. Entropy/JS are descriptive.',
              '- Input sqrt(area) groups are NOT official small-object AP. No test-dev is used.', '']
    # Limited directional hints, not experimental approval/significance thresholds.
    hints={}
    def delta(model,mode):
        return reports.get(model,{}).get('scores',{}).get(mode,{}).get('delta_vs_same_checkpoint_normal_pp',{})
    c2,c3=delta('e42','channel_off_p2'),delta('e42','channel_off_p3')
    if c2 and c3:
        hints['E44']=dict(
            frozen_rescue_at_levels=[level for level,d in (('P2',c2),('P3',c3))
                                    if d[KEYS[0]]>0 and d[KEYS[2]]>0],
            boundary='Not proof that S+C competition caused training degradation; independent compensation still needs controlled training.')
    d2,d3=delta('e43','dense_bypass_p2'),delta('e43','dense_bypass_p3')
    if d2 and d3:
        hints['E45']=dict(
            observed_P2_bypass_rescue_and_P3_bypass_not_rescue=(d2[KEYS[0]]>0 and d2[KEYS[2]]>0 and
                                                              d3[KEYS[0]]<=0 and d3[KEYS[2]]<=0),
            boundary='Even this pattern does not isolate training causality or guarantee P3-only Dense will help.')
    if 'e40' in reports and 'e40s2' in reports:
        for mode in ('uniform_depth','shared_task'):
            a,b=delta('e40',mode),delta('e40s2',mode)
            if a and b:
                hints['E40_'+mode]=dict(AP_delta_seed1=a[KEYS[0]],AP_delta_seed2=b[KEYS[0]],
                    both_negative=a[KEYS[0]]<0 and b[KEYS[0]]<0,
                    boundary='Measures trained model dependence, not which component originally created the training gain.')
    write(out/'decision.json',dict(status='completed',directional_hints=hints,
          next_action='Review full AP/AP50/AP75 and routing evidence before finalizing TWO training designs; no automatic training authorization.'))
    lines += ['## Directional hints (not causal conclusions)','', '```json',json.dumps(hints,indent=2,ensure_ascii=False),'```','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    global OUT
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('check','smoke','run'),required=True)
    p.add_argument('--models',default=','.join(MODELS))
    p.add_argument('--project',type=Path,default=Path('/root/autodl-tmp/yolo11_project'))
    p.add_argument('--source',type=Path,default=Path('/root/autodl-tmp/yolo11_src'))
    p.add_argument('--dataset',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--memory-cap-gib',type=float,default=6.)
    a=p.parse_args()
    models=a.models.split(',')
    if len(set(models))!=len(models) or not models or any(v not in MODELS for v in models):
        raise ValueError('Unique --models subset of '+','.join(MODELS)+' required')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out=a.project/'comparison_reports'/f'DIAG_D44_{a.mode.upper()}_{stamp}'
    out.mkdir(parents=True,exist_ok=False)
    OUT=out
    state=dict(status='running',mode=a.mode,zero_training=True,models=models,completed=[],
               started_at=datetime.now().astimezone().isoformat(),report_dir=str(out),automatic_shutdown=False)
    write(out/'controller.json',state)
    print(f'D44 CONTROLLER mode={a.mode} REPORT={out} device={a.device}; sequential inference, no training',flush=True)
    started=time.perf_counter(); reports={}
    for index,model in enumerate(models,1):
        cmd=[sys.executable,'-u',str(HERE/'diagnose_worker.py'),'--mode',a.mode,'--model',model,
             '--project',str(a.project),'--source',str(a.source),'--dataset',str(a.dataset),
             '--device',a.device,'--memory-cap-gib',str(a.memory_cap_gib),'--out',str(out/model)]
        log=out/(model+'.log')
        print(f'D44 [{index}/{len(models)}] {model} START log={log}',flush=True)
        with log.open('w',encoding='utf-8') as handle:
            child_env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONUTF8':'1'}
            job=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=child_env,
                                 text=True,encoding='utf-8',errors='replace',bufsize=1)
            try:
                for line in job.stdout:
                    handle.write(line); handle.flush()
                    print(line,end='',flush=True)
                code=job.wait()
            except BaseException:
                job.terminate()
                try: job.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    print(f'Child {job.pid} still exiting; no unrelated job was killed',flush=True)
                raise
        if code!=0:
            state.update(status='failed',failed_model=model,exit_code=code,log=str(log),
                         finished_at=datetime.now().astimezone().isoformat())
            write(out/'controller.json',state)
            raise RuntimeError(f'D44 {model} failed; real-time details above / {log}; no training started')
        state['completed'].append(model); write(out/'controller.json',state)
        if a.mode!='check': reports[model]=json.loads((out/model/'metrics.json').read_text(encoding='utf-8'))
        print(f'D44 [{index}/{len(models)}] {model} {a.mode} PASSED',flush=True)
    if a.mode=='run': summary(out,reports)
    state.update(status='completed',finished_at=datetime.now().astimezone().isoformat(),elapsed_seconds=time.perf_counter()-started)
    write(out/'controller.json',state)
    print(f'D44 {a.mode.upper()} COMPLETED {state["finished_at"]} elapsed={state["elapsed_seconds"]/60:.1f}min REPORT={out}',flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    sys.stderr.reconfigure(encoding='utf-8',errors='replace')
    try:
        main()
    except BaseException:
        import traceback
        if OUT is not None:
            state=json.loads((OUT/'controller.json').read_text(encoding='utf-8'))
            state.update(status='failed',finished_at=datetime.now().astimezone().isoformat(),error=traceback.format_exc())
            write(OUT/'controller.json',state)
        raise
