"""Lightweight preparation: no torch import, no model construction, no GPU use."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
VERSION = '8.4.177'
PROJECT = Path('/root/autodl-tmp/yolo26_project')
SOURCE = Path('/root/autodl-tmp/yolo26_src')
WEIGHT_SHA = '646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def audit(source=SOURCE, local=False):
    source = Path(source).resolve()
    if not local and Path(sys.prefix).resolve() != Path('/root/miniconda3/envs/yolo26').resolve():
        raise RuntimeError('Wrong Python environment; use the yolo26 absolute executable')
    actual = Path(importlib.util.find_spec('ultralytics').origin).resolve()
    if not actual.is_relative_to(source):
        raise RuntimeError('Imported wrong source: ' + str(actual))
    expected = json.loads((HERE/'source_sha256.json').read_text(encoding='utf-8'))
    for relative, digest in expected.items():
        if sha256(source/relative) != digest:
            raise RuntimeError('Pinned source drift: ' + relative)
    manifest = json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
    for relative, digest in manifest.items():
        if sha256(HERE/relative) != digest:
            raise RuntimeError('Preflight package drift: ' + relative)
    versions = {name: importlib.metadata.version(name) for name in ('torch','torchvision')}
    if not local and versions != dict(torch='2.5.1+cu121', torchvision='0.20.1+cu121'):
        raise RuntimeError('Wrong torch/torchvision versions: ' + str(versions))
    return dict(source=str(actual), versions=versions, source_files_checked=len(expected),
                package_files_checked=len(manifest), local_engineering=local)


def prepare(project=PROJECT, source=SOURCE,
            data=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019.yaml')):
    evidence = audit(source)
    import yaml
    data = Path(data)
    cfg = yaml.safe_load(data.read_text(encoding='utf-8'))
    names = cfg['names']
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else names
    if names != ['pedestrian','people','bicycle','car','van','truck','tricycle',
                 'awning-tricycle','bus','motor']:
        raise RuntimeError('Unexpected VisDrone class order')
    supplied = HERE/'weights/yolo26s.pt'
    if sha256(supplied) != WEIGHT_SHA:
        raise RuntimeError('Bundled official checkpoint SHA mismatch')
    target = Path(project)/'weights/yolo26s.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256(target) != WEIGHT_SHA:
        raise RuntimeError('Existing different checkpoint; will not overwrite it')
    if not target.exists():
        shutil.copyfile(supplied, target)
    if sha256(target) != WEIGHT_SHA:
        raise RuntimeError('Copied checkpoint SHA mismatch')
    for name in ('runs','logs','comparison_reports','environment_records'):
        (Path(project)/name).mkdir(parents=True, exist_ok=True)
    value = dict(evidence, status='assets_verified', version=VERSION,
                 pretrained=str(target), pretrained_sha256=WEIGHT_SHA,
                 data=str(data), data_sha256=sha256(data),
                 model_constructed=False, cuda_executed=False, training_started=False,
                 formal_training_authorized_by_this_receipt=False)
    path = Path(project)/'environment_records/yolo26_p2_e60_assets.json'
    write(path, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return value


def resource_guard(local=False):
    if local:
        return
    limits = (Path('/sys/fs/cgroup/memory.max'),
              Path('/sys/fs/cgroup/memory/memory.limit_in_bytes'))
    for path in limits:
        if path.is_file():
            value = path.read_text().strip()
            if value.isdigit() and int(value) < 6 * 1024**3:
                raise RuntimeError('Need at least 6 GiB container RAM for full engineering checks; '
                                   'current no-GPU 2 GiB mode is not enough. Do not reinstall or retry this mode.')


def runtime(project=PROJECT):
    directory = Path(project)/'environment_records/ultralytics_settings/Ultralytics'
    directory.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(directory.parent)
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_NUM_THREADS'] = '1'
