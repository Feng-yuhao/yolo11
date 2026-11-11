"""CPU-only orchestration helpers and read-only provenance checks."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import zipfile

HERE = Path(__file__).resolve().parent
VARIANTS = ('uniform', 'shared')
LABELS = {'uniform': 'E60_UNIFORM_S1', 'shared': 'E60_SHARED_S1'}
EXPERIMENTS = {v: 'e60_ablation_' + v + '_img640_seed1_200e' for v in VARIANTS}
CAPS_GIB = {'serial': 28., 'parallel': 15.}
E1_REPORT = 'e1_yolo11s_p2add_img640_seed1_20260926_102750_718906'
E60_REPORT = 'E60_RUN_20261010_133029_103161'
E60_SHA = 'bf38299c8723f5fe399e2f94fa4042df2ff6ae51dcb99fceece8bc85ed00dc17'
KEYS = ('AP50_95_percent', 'AP50_percent', 'AP75_percent', 'AR500_percent')
E60_EXPECTED = (26.969156363130686, 46.39520525911735, 26.567837427107293, 39.00411627684551)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def sha(path, normalized=False):
    value = Path(path).read_bytes()
    return hashlib.sha256(value.replace(b'\r\n', b'\n') if normalized else value).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def package_audit():
    manifest = read(HERE/'manifest.json')
    for name, expected in manifest.items():
        require(sha(HERE/name, True) == expected, 'Package changed: ' + name)
    return sha(HERE/'manifest.json')


def frozen_audit(project):
    mapping = read(HERE/'frozen_sources.json')
    with zipfile.ZipFile(project/'comparison_reports'/E60_REPORT/'actual_source.zip') as archive:
        for name, original in mapping.items():
            require((HERE/name).read_bytes().replace(b'\r\n', b'\n') ==
                    archive.read(original).replace(b'\r\n', b'\n'),
                    'Frozen actual E60 source differs: ' + name)


def protocol_audit(args, core, allow_missing_data=False):
    base = args.project/'comparison_reports'/E1_REPORT
    parent = args.project/'comparison_reports'/E60_REPORT
    e1, e60, scores = read(base/'metrics.json'), read(parent/'training.json'), read(parent/'comparison.json')
    require(e1['status'] == e60['status'] == scores['status'] == 'completed', 'Incomplete E1/E60 reference')
    require(e1['seed'] == e60['seed'] == scores['seed'] == 1, 'Wrong reference seed')
    require(e1['training']['completed_epochs'] == e60['completed_epochs'] == 200, 'Wrong reference epochs')
    ignored = {'project', 'name', 'model', 'data', 'device'}
    params = core.training_params()
    params.update(device=args.device, workers=args.workers, model=str(core.MODEL_YAML))
    expected = {k:v for k,v in params.items() if k not in ignored}
    for description, saved in (('E1', e1['train_params']), ('E60', e60['params'])):
        actual = {k:v for k,v in saved.items() if k not in ignored}
        require(actual == expected, description + ' training protocol differs: ' +
                str({k:(actual.get(k), expected.get(k)) for k in actual.keys() | expected.keys()
                     if actual.get(k) != expected.get(k)}))
    for rel, expected_sha in e60['signature']['source'].items():
        require(sha(args.source/'ultralytics'/rel, True) == expected_sha,
                'Pinned actual E60 Ultralytics source differs: ' + rel)
    if args.data.is_file():
        require(sha(args.data) == e1['data_yaml_sha256'], 'Data YAML raw SHA differs from E1')
    else:
        require(allow_missing_data, 'Missing real dataset YAML: ' + str(args.data))
    require(scores['checkpoint_sha256'] == E60_SHA, 'Wrong completed E60 identity')
    for key, value in zip(KEYS, E60_EXPECTED):
        require(abs(scores['candidate'][key] - value) < 1e-7, 'E60 saved metric changed: ' + key)
    frozen_audit(args.project)
    for path in (base/'area_ground_truth.json', base/'area_predictions.json', parent/'area_predictions.json'):
        require(path.is_file(), 'Missing official comparison input: ' + str(path))
    return e1, scores


def gate_path(args, stage):
    return args.project/'comparison_reports'/'e60_ablation_gates'/(args.variant + '_' + args.execution + '_' + stage + '.json')


def pair_gate(project):
    return project/'comparison_reports'/'e60_ablation_gates'/'parallel.json'


def signature(args, core):
    import torch
    params = core.training_params()
    params.update(model=str(core.MODEL_YAML), device=args.device, workers=args.workers)
    gpu = None
    if args.device != 'cpu':
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu = dict(name=props.name, total_bytes=props.total_memory,
                   capability=list(torch.cuda.get_device_capability()))
    return dict(seed=1, variant=args.variant, execution=args.execution, params=params, allocator_cap_GiB=CAPS_GIB[args.execution],
                package=package_audit(), torch=torch.__version__, cuda=torch.version.cuda, GPU=gpu,
                dataset=str(args.dataset.resolve()), data_sha=sha(args.data) if args.data.is_file() else None,
                source={name:sha(args.source/'ultralytics'/name, True) for name in
                        ('nn/tasks.py','nn/modules/head.py','nn/modules/conv.py','utils/ops.py',
                         'utils/loss.py','utils/tal.py','engine/trainer.py','data/augment.py')},
                references={report+'/'+name:sha(args.project/'comparison_reports'/report/name) for report, name in
                            ((E1_REPORT,'metrics.json'), (E1_REPORT,'area_predictions.json'),
                             (E60_REPORT,'comparison.json'), (E60_REPORT,'training.json'),
                             (E60_REPORT,'actual_source.zip'), (E60_REPORT,'area_predictions.json'))},
                pretrained=core.sha256(core.PRETRAINED))


def verify_gate(args, stage, binding):
    receipt = read(gate_path(args, stage))
    require(receipt['signature'] == binding, 'Stale ' + stage + ' gate: ' + args.variant)
    evidence = Path(receipt['report'])/(stage + '.json')
    value = read(evidence)
    require(value.get('status') == 'passed' and not (evidence.parent/'failure.json').exists(),
            'Failed/incomplete gate evidence: ' + str(evidence))
    if stage == 'check':
        require(value.get('cuda_executed'), 'CPU engineering check is not a CUDA gate')
    return receipt, value
