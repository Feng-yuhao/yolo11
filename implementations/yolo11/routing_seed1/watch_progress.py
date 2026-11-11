"""Read-only concise progress, no tqdm lines; Ctrl+C affects only this viewer."""
import argparse
from pathlib import Path
import time
import common as c

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--project',type=Path,default=Path('/root/autodl-tmp/yolo11_project'))
    p.add_argument('--variant',choices=('both','uniform','shared'),default='both')
    p.add_argument('--follow',action='store_true')
    a = p.parse_args(); previous = None
    try:
        while True:
            result = []
            for v in c.VARIANTS if a.variant=='both' else (a.variant,):
                index = a.project/'logs'/('latest_e60_ablation_'+v+'_run.json')
                if not index.is_file():
                    result.append(v+': not started'); continue
                report = Path(c.read(index)['report'])
                path = report/'training.json'
                if not path.is_file():
                    result.append(v+': initializing'); continue
                value = c.read(path); epoch = value.get('completed_epochs',0)
                ap = value.get('metrics',{}).get('metrics/mAP50-95(B)')
                result.append(v+f': epoch={epoch}/200 AP={ap} last_seconds={value.get("epoch_seconds")} status={value.get("status")}')
            text = '\n'.join(result)
            if text != previous:
                print(c.now()+'\n'+text,flush=True); previous = text
            if not a.follow: break
            time.sleep(10)
    except KeyboardInterrupt:
        print('Stopped progress viewer; training was not signalled.',flush=True)
