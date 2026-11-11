"""Private2-epoch native integration, NOT a server gate or accuracy experiment."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = args.root.resolve()
    root.mkdir(parents=True,exist_ok=False)
    args.source = args.source.resolve()
    sys.path.insert(0,str(args.source))
    from fixture_tools import fixture
    data = fixture(root / 'data')
    dataset = root / 'data/test-dev'
    for folder in ('images','annotations'):
        (dataset / folder).mkdir(parents=True)
    for i in range(9):
        source = f'val_{i:04d}'
        target = f'test_{i:04d}'
        shutil.copyfile(root / 'data/val/images' / (source + '.jpg'),dataset / 'images' / (target + '.jpg'))
        shutil.copyfile(root / 'data/val/annotations' / (source + '.txt'),dataset / 'annotations' / (target + '.txt'))
    project = root / 'project'
    (project / 'weights').mkdir(parents=True)
    shutil.copyfile(here / 'weights/yolo26s.pt',project / 'weights/yolo26s.pt')
    command = [sys.executable,'-u',str(here / 'controller.py'),'all',
               '--source',str(args.source),'--project',str(project),'--data',str(data),
               '--val-dataset',str(root / 'data/val'),'--dataset',str(dataset),
               '--local-test','--local-size','128','--local-epochs','2']
    result = subprocess.run(command)
    if result.returncode:
        raise RuntimeError('Private integration failed; outputs retained in ' + str(root))
    index = json.loads((project / 'logs/PRIVATE_latest_yolo26_seed2_serial.json').read_text())
    report = Path(index['report_root'])
    state = json.loads((report / 'state.json').read_text())
    assert state['status'] == 'completed' and not state['auto_shutdown'] and state['local_fixture']
    expected = ['check','baseline_smoke','e60_smoke','baseline_train','e60_train',
                'baseline_val','baseline_test_dev','e60_val','e60_test_dev']
    assert list(state['jobs']) == expected
    assert all(item['status'] == 'completed' and item['exit_code'] == 0 for item in state['jobs'].values())
    (root / 'PRIVATE_LOCAL_VALIDATION.json').write_text(json.dumps(dict(status='passed',report=str(report),
        server_training_started=False,server_gate_issued=False,real_visdrone_tested=False,
        boundary='Private128px/b16, 48train/35val/9test, two complete native epochs for each model'),indent=2),encoding='utf-8')
    print('PRIVATE FULL SERIAL INTEGRATION PASSED')


if __name__ == '__main__':
    main()
