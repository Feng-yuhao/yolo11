"""Read-only monitor, one compact update only when progress changes."""
import argparse
import json
from pathlib import Path
import time
from datetime import datetime


def load(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,ValueError):
        return {}


def snapshot(project):
    index = load(project / 'logs/latest_tood_td_controller.json')
    if not index:
        return 'TD controller has not started (or project path differs).'
    session = Path(index['session'])
    state = load(session / 'state.json')
    report = Path(index['report_root'])
    training = load(report / 'train/training.json')
    lines = [f'TD status={state.get("status")} stage={state.get("stage")} '
             f'auto_shutdown={state.get("auto_shutdown")}']
    if training:
        lines.append(f'epoch={training.get("completed_epochs",0)}/200 '
            f'batch={training.get("current_batch")} '
            f'native_AP={training.get("native_metrics",{}).get("metrics/mAP50-95(B)")} '
            f'best={training.get("best_native_ap")} best_epoch={training.get("best_native_ap_epoch")} '
            f'last_epoch={training.get("last_epoch_seconds")}s '
            f'elapsed={training.get("elapsed_seconds",0)/3600:.2f}h '
            f'ETA={training.get("ETA_hours")}h')
    for split in ('val','test'):
        score = load(report / split / 'official_metrics.json')
        if score:
            lines.append(f'{split} official AP={score.get("official",{}).get("AP50_95_percent")}% '
                         f'TD−E1={score.get("delta_vs_E1_pp",{}).get("AP50_95_percent")}pp '
                         f'TD−E60={score.get("delta_vs_E60_pp",{}).get("AP50_95_percent")}pp')
    power = load(session / 'shutdown_status.json') or state.get('shutdown',{})
    if power:
        lines.append('shutdown=' + str(power.get('status')))
    if state.get('error'):
        lines.append('ERROR: ' + state['error'].splitlines()[-1])
    lines += ['logs=' + str(session),'report=' + str(report)]
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project',type=Path,default=Path('/root/autodl-tmp/yolo11_project'))
    p.add_argument('--follow',action='store_true')
    args = p.parse_args()
    previous = None
    while True:
        value = snapshot(args.project)
        if value != previous:
            print(datetime.now().isoformat(timespec='seconds') + '\n' + value,flush=True)
            previous = value
        if not args.follow:
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
