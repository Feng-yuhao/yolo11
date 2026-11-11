"""Read-only changed-state monitor. Ctrl+C does not stop evaluation."""
from pathlib import Path
import time
from common import INDEX,KEYS,now,parser,read
def snapshot(project):
    p=project/'logs'/INDEX
    if not p.exists():return 'Not started or admission refused: inspect launcher log.'
    report=Path(read(p)['report']);state=read(report/'state.json');stage=state['stage']
    lines=[f'status={state["status"]} stage={stage} serial=True training=False auto_shutdown={state["auto_shutdown"]}',f'report={report}']
    if stage.startswith(('smoke_','evaluation_')):
        phase,key=stage.split('_',1);out=report/phase/key
        if (out/'progress.json').exists():
            d=read(out/'progress.json');lines.append(f'{key}: {d["done_images"]}/{d["total_images"]} images; elapsed={d["elapsed_seconds"]/60:.1f}min; inference_ETA={d["ETA_seconds"]/60:.1f}min (scoring extra)')
        if (out/'scoring_progress.json').exists():lines.append('scoring: '+str(read(out/'scoring_progress.json')))
    for phase in ('smoke','evaluation'):
        done=sum(state['jobs'].get(phase+'_'+k,{}).get('exit_code')==0 for k in KEYS);lines.append(f'{phase} successful: {done}/12')
    if (report/'shutdown_status.json').exists():lines.append('shutdown='+read(report/'shutdown_status.json')['status'])
    if state.get('error'):lines.append(state['error'].splitlines()[-1])
    return '\n'.join(lines)
if __name__=='__main__':
    p=parser();p.add_argument('--follow',action='store_true');args=p.parse_args();previous=None
    while True:
        current=snapshot(args.project)
        if current!=previous:print(now()+'\n'+current,flush=True);previous=current
        if not args.follow or 'status=failed ' in current or 'status=completed stage=completed ' in current:break
        time.sleep(15)
