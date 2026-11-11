"""Freeze copies of the actual seed1 implementation, references and offline assets."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
OLD = HERE.parent / 'YOLO26_P2_E60_Seed1_AutoDL'
SERVER = PAPER / 'yolo26_project/comparison_reports/YOLO26_P2_E60_RUN_20261014_135537_497769'
TEST = PAPER / 'yolo26_project/comparison_reports/TESTDEV_YOLO26_P2_E60_S1_20261016_140349_779106'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('w',encoding='utf-8',newline='\n') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write('\n')


def build():
    frozen = {}
    for name in ('common.py','models26.py','e60_adapter.py','checks26.py','training26.py','scoring26.py',
                 'd27_official.py','source_sha256.json','training_source_sha256.json','LICENSE'):
        source = SERVER / 'baseline_eval/actual_source' / name
        assert sha(source) == sha(OLD / name), 'Local historical source differs from actual server: ' + name
        shutil.copyfile(source,HERE / name)
        frozen[name] = sha(source)
    shutil.copyfile(SERVER / 'baseline_eval/actual_source/protocol.py', HERE / 'seed1_protocol.py')
    frozen['seed1_protocol.py'] = sha(HERE / 'seed1_protocol.py')
    shutil.copyfile(OLD / 'local_validation.py',HERE / 'fixture_tools.py')
    shutil.copyfile(HERE.parent / 'E63_E60_P45W192_Seed1_AutoDL/shutdown_safety.py',HERE / 'shutdown_safety.py')
    (HERE / 'assets').mkdir(exist_ok=True)
    shutil.copyfile(PAPER / 'project/comparison_reports/e63_settings/Arial.ttf',HERE / 'assets/Arial.ttf')
    ref = HERE / 'references'
    ref.mkdir(exist_ok=True)
    write(ref / 'frozen_seed1_files.json',frozen)
    shutil.copyfile(PAPER / 'yolo26_project/logs/yolo26_p2_e60_run_20261014_135537_497769/signature.json',
                    ref / 'seed1_signature.json')
    shutil.copyfile(HERE.parent / 'YOLO26_P2_E60_Testdev_Seed1_AutoDL/references/testdev_input_manifest.json',
                    ref / 'testdev_input_manifest.json')
    for variant in ('baseline','e60'):
        shutil.copyfile(SERVER / (variant + '_eval') / 'official_metrics.json',ref / (variant + '_seed1_val.json'))
        shutil.copyfile(TEST / (variant + '_testdev') / 'official_metrics.json',ref / (variant + '_seed1_test_dev.json'))
    (HERE / 'weights').mkdir(exist_ok=True)
    for name in ('yolo26s.pt','yolo26n.pt'):
        shutil.copyfile(OLD / 'weights' / name,HERE / 'weights' / name)
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'))
    for path in HERE.glob('*.sh'):
        content = path.read_text(encoding='utf-8')
        with path.open('w',encoding='utf-8',newline='\n') as handle:
            handle.write(content.replace('\r\n','\n'))
    files = sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                   and p.suffix != '.pyc' and p.name != 'manifest.json')
    write(HERE / 'manifest.json',{p.relative_to(HERE).as_posix():sha(p) for p in files})
    target = PAPER / 'audit' / (HERE.name + '.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files + [HERE / 'manifest.json']:
            archive.write(path,HERE.name + '/' + path.relative_to(HERE).as_posix())
    with target.with_suffix('.zip.sha256').open('w',encoding='ascii',newline='\n') as handle:
        handle.write(sha(target) + '  ' + target.name + '\n')
    print(json.dumps(dict(zip=str(target),sha256=sha(target),files=len(files)+1,bytes=target.stat().st_size),indent=2))


if __name__ == '__main__':
    build()
