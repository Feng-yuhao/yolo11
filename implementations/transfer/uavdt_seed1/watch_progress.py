"""Read-only, print changes only. Ctrl+C stops this viewer, not inference."""
import time
from common import now, parser, read


def snapshot(project):
    index = project/'logs/latest_visdrone_to_uavdt_seed1_source_only.json'
    if not index.is_file():
        return 'Source-only evaluation has not started (or admission failed; inspect launcher log).'
    from pathlib import Path
    report = Path(read(index)['report']); state = read(report/'state.json')
    lines = [f'status={state["status"]} stage={state["stage"]} training=False auto_shutdown=False',f'report={report}']
    stage = state['stage']
    if stage.startswith(('smoke_','evaluation_')):
        phase,key = stage.split('_',1)
        path = report/phase/key/'progress.json'
        if path.is_file():
            v = read(path)
            lines.append(f'{key}: {v["done_images"]}/{v["total_images"]} images '
                         f'elapsed={v["elapsed_seconds"]/60:.1f}min inference_ETA={v["ETA_seconds"]/60:.1f}min (scoring extra)')
        scoring = report/phase/key/'scoring_progress.json'
        if scoring.is_file():
            v = read(scoring); lines.append('scoring: '+str(v))
    done = sum(name.startswith('evaluation_') and job.get('exit_code') == 0 for name,job in state['jobs'].items())
    lines.append(f'Full seed1 evaluations completed: {done}/2')
    if state.get('error'):
        lines.append(state['error'].splitlines()[-1])
    return '\n'.join(lines)


if __name__ == '__main__':
    p = parser(); p.add_argument('--follow',action='store_true'); args = p.parse_args(); previous = None
    while True:
        current = snapshot(args.project)
        if current != previous:
            print(now()+'\n'+current,flush=True); previous = current
        if not args.follow or 'status=completed ' in current or 'status=failed ' in current:
            break
        time.sleep(10)
