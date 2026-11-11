#!/usr/bin/env python3
"""Read-only monitor; prints changes, never imports torch or starts a run."""
import argparse,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common7 import *

def snapshot(project):
    index=project/'logs/latest_stmdrf_supp7.json'
    if not index.is_file():return 'Not admitted yet. Inspect launcher_stmdrf_supp7_v2.log; no training is implied.'
    idx=read(index);session,report=Path(idx['session']),Path(idx['report']);s=read(session/'state.json')
    lines=[f'SUPP7 status={s["status"]} stage={s["stage"]} execution={s.get("execution","pending_joint_smoke")} auto_shutdown={s["auto_shutdown"]}']
    counts={p:sum(1 for v in s['jobs'].values() if v['phase']==p and v['status']=='completed' and 'pair_gates' not in Path(v['out']).parts) for p in ['check','smoke','train','score']}
    lines+=['Passed: checks={check}/7 individual_smokes={smoke}/7 formal200={train}/7 official_scores={score}/14'.format(**counts)]
    for key in JOBS:
        r=report/'jobs'/key/'train/training.json'
        if r.is_file():
            t=read(r);native=t.get('native_metrics',{}).get('metrics/mAP50-95(B)')
            eta=t.get('ETA_hours');eta=f'{eta:.2f}h' if eta is not None else 'initializing'
            lines.append(f'{key}: {t["status"]} epoch={t.get("completed_epochs",0)}/200 current={t.get("current_epoch")} batch={t.get("current_batch")} native_AP={native} best={t.get("best_native_ap")} ETA={eta}')
        else:
            records=[v for v in s['jobs'].values() if v['key']==key]
            recent=records[-1] if records else None
            lines.append(key+': '+(recent['phase']+'='+recent['status'] if recent else 'not started'))
    for value in s['jobs'].values():
        if value['status']=='running' and value['phase']=='score':
            p=Path(value['out'])/'progress.json'
            if p.is_file():lines.append('Scoring '+value['key']+' '+str(value['split'])+': '+str(read(p)))
    p=session/'shutdown_status.json'
    if p.is_file():lines.append('shutdown='+read(p)['status'])
    if (project/'KEEP_RUNNING_ST_MDRF_SUPP7').exists():lines.append('Automatic shutdown CANCEL file exists (training is not cancelled).')
    lines+=['logs='+str(session),'report='+str(report)]
    if s.get('error'):lines.append('ERROR: '+s['error'].strip().splitlines()[-1])
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=DEFAULT_PROJECT);p.add_argument('--follow',action='store_true');p.add_argument('--interval',type=float,default=30.)
    a=p.parse_args();previous=None
    try:
        while True:
            try:current=snapshot(a.project)
            except (FileNotFoundError,ValueError) as e:current='State update in progress: '+str(e)
            if current!=previous:print(now()+'\n'+current,flush=True);previous=current
            if not a.follow:break
            time.sleep(max(5.,a.interval))
    except KeyboardInterrupt:return
if __name__=='__main__':main()
