"""Read-only compact monitor; no torch import or GPU allocation."""
import argparse
from pathlib import Path
import time
from datetime import datetime
import json

PROJECT = Path('/root/autodl-tmp/yolo26_project')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--interval', type=int, default=30)
    args = parser.parse_args()
    while True:
        try:
            index = read(PROJECT / 'logs/latest_yolo26_seed3_serial.json')
            state = read(Path(index['session']) / 'state.json')
            print(f'{datetime.now():%Y-%m-%d %H:%M:%S} seed3 status={state["status"]} '
                  f'stage={state["stage"]} execution={state["execution"]} auto_shutdown={state["auto_shutdown"]}',flush=True)
            for variant in ('baseline','e60'):
                job = state.get('jobs',{}).get(variant + '_train')
                progress_path = Path(index['report_root']) / (variant + '_train') / 'progress.json'
                if not job:
                    print(variant + ': not started',flush=True)
                elif progress_path.is_file():
                    p = read(progress_path)
                    print(f'{variant}: {job["status"]}; epoch={p["epoch"]}/{p["epochs"]}; '
                          f'native_mAP50-95={p.get("mAP50_95")}; last_epoch={p["epoch_seconds"]:.1f}s; '
                          f'elapsed={p["elapsed_hours"]:.2f}h; ETA={p["remaining_hours"]:.2f}h',flush=True)
                else:
                    print(f'{variant}: {job["status"]}; initializing; log={job["log"]}',flush=True)
            marker = Path(index['session']) / 'shutdown_status.json'
            if marker.is_file():
                print('shutdown=' + str(read(marker).get('status')),flush=True)
            print('report=' + index['report_root'],flush=True)
            if state['status'] in ('completed','failed'):
                break
        except (OSError,ValueError,KeyError) as error:
            print('Waiting for valid controller state: ' + str(error),flush=True)
        if not args.follow:
            break
        time.sleep(max(5,args.interval))


if __name__ == '__main__':
    main()
