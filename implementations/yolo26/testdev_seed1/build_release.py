"""Copy frozen validated inference code/reference extracts; no model/server mutation."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
ORIGINAL = HERE.parent / 'YOLO26_P2_E60_Seed1_AutoDL'
RESULT = PAPER / 'yolo26_project/comparison_reports/YOLO26_P2_E60_RUN_20261014_135537_497769'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def build():
    for name in ('e60_adapter.py', 'scoring26.py', 'protocol.py', 'common.py', 'd27_official.py',
                 'source_sha256.json', 'training_source_sha256.json', 'LICENSE'):
        shutil.copyfile(ORIGINAL / name, HERE / name)
        assert sha(HERE / name) == sha(ORIGINAL / name)
    ref = HERE / 'references'
    ref.mkdir(exist_ok=True)
    shutil.copyfile(HERE.parent / 'evaluation_TESTDEV_E60_seed1/references/testdev_input_manifest.json',
                    ref / 'testdev_input_manifest.json')
    specs = {}
    for variant in ('baseline', 'e60'):
        train = json.loads((RESULT / (variant + '_train') / 'training_completion.json').read_text())
        metrics = json.loads((RESULT / (variant + '_eval') / 'official_metrics.json').read_text())
        checkpoint = PAPER / 'yolo26_project/runs' / Path(train['run']).name / 'weights/best.pt'
        # Windows Path does not split Linux paths as native drive separators; '/' does here.
        assert sha(checkpoint) == train['best_sha256'] == metrics['checkpoint_sha256']
        specs[variant] = dict(relative_path='runs/' + Path(train['run']).name + '/weights/best.pt',
                              sha256=train['best_sha256'])
        shutil.copyfile(RESULT / (variant + '_train') / 'training_completion.json',
                        ref / (variant + '_training_completion.json'))
        shutil.copyfile(RESULT / (variant + '_eval') / 'official_metrics.json',
                        ref / (variant + '_val_official_metrics.json'))
        preds = json.loads((RESULT / (variant + '_eval') / 'area_predictions.json').read_text())
        first8 = [p for p in preds if p['image_id'] <= 8]
        write(ref / (variant + '_val_first8_predictions.json'), first8)
    write(ref / 'checkpoint_specs.json', specs)
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'))
    for path in HERE.glob('*.sh'):
        content = path.read_text(encoding='utf-8')
        with path.open('w', encoding='utf-8', newline='\n') as handle:
            handle.write(content.replace('\r\n', '\n'))
    files = sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                   and p.suffix != '.pyc' and p.name != 'manifest.json')
    write(HERE / 'manifest.json', {p.relative_to(HERE).as_posix(): sha(p) for p in files})
    target = PAPER / 'audit' / (HERE.name + '.zip')
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files + [HERE / 'manifest.json']:
            archive.write(path, HERE.name + '/' + path.relative_to(HERE).as_posix())
    with target.with_suffix('.zip.sha256').open('w', encoding='ascii', newline='\n') as handle:
        handle.write(sha(target) + '  ' + target.name + '\n')
    print(json.dumps(dict(zip=str(target), sha256=sha(target), files=len(files) + 1,
                          bytes=target.stat().st_size), indent=2))


if __name__ == '__main__':
    build()
