"""Standard-library evidence and live-controller checks for the fixed parallel pair.

Hash binding prevents accidental stale/mixed evidence; Linux process ancestry prevents
enabling the worker entry by merely setting the PAIR environment variable. This is
an engineering guard, not a security boundary against a user editing the package.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import subprocess
import time
import re

ROOT = Path(__file__).resolve().parent
NAMES = ('e40', 'e47')
SCHEMA = 'e40s3_e47s1_parallel_v1'
ENV_PREFIX = 'E40S3_E47S1_'
RECEIPT_NAME = 'e40s3_e47s1_pair_smoke_receipt.json'
GATES = {'e40': ('e40_seed3_cuda_check_gate.json', 'e40_seed3_smoke_gate.json'),
         'e47': ('e47p_cuda_check_gate.json', 'e47p_smoke_gate.json')}
EXPERIMENTS = {'e40': 'e40a_spatial_task_head_img640_seed3_200e',
               'e47': 'e47a_p2_depth_readout_img640_seed1_200e'}
SOURCE_FILES = ('nn/tasks.py', 'nn/modules/head.py', 'nn/modules/conv.py',
                'utils/loss.py', 'utils/tal.py', 'engine/trainer.py', 'engine/validator.py',
                'models/yolo/detect/train.py', 'models/yolo/detect/val.py',
                'data/augment.py', 'data/build.py', 'data/dataset.py', 'data/utils.py')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def value_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path, normalized=False):
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b'\r\n', b'\n') if normalized else data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def memory_policy():
    return dict(allocator_cap_GiB=13.0, per_process_cap_bytes=13 * 2**30,
                other_job_reserve_GiB=2.0, combined_allocator_cap_GiB=26.0,
                controller_idle_free_MiB=30 * 1024, minimum_gpu_total_MiB=30 * 1024,
                automatic_batch_reduction=False)


@contextmanager
def dataset_build_lock(args):
    """Serialize shared labels.cache construction without changing Ultralytics."""
    if os.name != 'posix':
        if str(args.device) != 'cpu':
            raise RuntimeError('Real paired dataset construction requires Linux flock')
        yield  # local CPU fixtures have no concurrent real dataset writers
        return
    import fcntl
    args.project.mkdir(parents=True, exist_ok=True)
    with (args.project / '.e40s3_e47s1_dataset_build.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def dataset_fingerprint(args):
    """Bind labels byte-for-byte and images by path, size and mtime (not pixels).

    The fixed release uses scalar image-directory splits, so unsupported YAML
    forms fail instead of guessing a path different from check_det_dataset().
    Cache files are deliberately excluded: both workers may rebuild them.
    """
    if str(args.device) == 'cpu' and (not args.data.is_file() or not (args.dataset / 'annotations').is_dir()):
        return dict(status='unavailable_local_cpu', CUDA_gate_eligible=False,
                    reason='Real train/validation files are not available to this CPU fixture')
    if not args.data.is_file():
        raise FileNotFoundError('Missing dataset YAML: ' + str(args.data))
    settings = {}
    for line in args.data.read_text(encoding='utf-8').splitlines():
        match = re.match(r'^(path|train|val):\s*(.*?)\s*$', line)
        if match:
            value = match[2].split(' #', 1)[0].strip().strip('\"\'')
            if not value or value.startswith(('[', '{', '&', '*', '|', '>')):
                raise RuntimeError('Pair dataset fingerprint needs scalar image-directory split: ' + line)
            settings[match[1]] = value
    root = Path(settings.get('path', str(args.data.resolve().parent)))
    if not root.is_absolute():
        raise RuntimeError('Pair requires absolute dataset root to bind the same Ultralytics paths')
    result = dict(image_binding='relative path + byte size + mtime_ns; NOT image-content SHA256', splits={})
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.dng', '.mpo', '.pfm', '.heic'}
    for split, expected in (('train', 6471), ('val', 548)):
        if split not in settings:
            raise RuntimeError('Missing dataset split: ' + split)
        image_dir = (root / settings[split]).resolve()
        if not image_dir.exists() and settings[split].startswith('../'):
            image_dir = (root / settings[split][3:]).resolve()
        if not image_dir.is_dir():
            raise RuntimeError('Pair fingerprint requires image-directory split: ' + str(image_dir))
        inventory, labels = {}, {}
        images = sorted(p for p in image_dir.rglob('*') if p.is_file() and p.suffix.lower() in extensions)
        if len(images) != expected:
            raise RuntimeError(f'{split} image count differs: expected={expected}, actual={len(images)}')
        for image in images:
            stat = image.stat()
            relative = image.relative_to(image_dir).as_posix()
            inventory[relative] = [stat.st_size, stat.st_mtime_ns]
            text = str(image)
            separator = os.sep + 'images' + os.sep
            if separator not in text:
                raise RuntimeError('Image split path has no /images/ component: ' + text)
            prefix, suffix = text.rsplit(separator, 1)
            label = Path(prefix + os.sep + 'labels' + os.sep + suffix).with_suffix('.txt')
            labels[relative] = file_hash(label) if label.exists() else 'MISSING'
        result['splits'][split] = dict(images=str(image_dir), count=len(images),
                                      image_inventory_sha256=value_hash(inventory),
                                      labels_content_sha256=value_hash(labels))
    annotations = args.dataset / 'annotations'
    if not annotations.is_dir():
        raise RuntimeError('Missing original official validation annotations: ' + str(annotations))
    files = {p.relative_to(annotations).as_posix(): file_hash(p) for p in sorted(annotations.rglob('*.txt'))}
    if len(files) != 548:
        raise RuntimeError('Official annotation count differs from 548')
    result['official_annotations'] = dict(path=str(annotations.resolve()), count=len(files), content_sha256=value_hash(files))
    return result


def protocol():
    return dict(epochs=200, imgsz=640, batch=16, workers=8, validation_batch=32,
                validation_images=548, successful_smoke_updates=8,
                seeds={'e40': 3, 'e47': 1})


def expected_groups(name):
    groups = [f'{i}.{part}.' for i in range(2) for part in
              ('reduce', 'depths.0', 'depths.1', 'depths.2', 'router', 'out_cls', 'out_reg')]
    if name == 'e47':
        groups += [f'{i}.readout_{task}.{j}.' for i in range(2) for task in ('cls', 'reg') for j in range(3)]
    return groups


def package_digest():
    files = {}
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if path.is_file() and path.suffix in ('.py', '.sh', '.yaml', '.yml') and not any(
                part.startswith('.') or part == '__pycache__' for part in relative.parts):
            files[relative.as_posix()] = file_hash(path, normalized=True)
    if not files or 'parallel_safety.py' not in files or 'run_pair.py' not in files:
        raise RuntimeError('Incomplete parallel controller code')
    return value_hash(files)


def package_check():
    manifest_path = ROOT / 'manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError('Missing parallel release manifest')
    manifest = read_json(manifest_path)
    for relative, expected in manifest.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise RuntimeError('Missing/unsafe release file: ' + relative)
        if file_hash(path, normalized=True) != expected:
            raise RuntimeError('Changed parallel release file: ' + relative)
    required = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
                if p.is_file() and p.suffix in ('.py', '.sh', '.yaml', '.yml')
                and not any(q.startswith('.') or q == '__pycache__' for q in p.relative_to(ROOT).parts)}
    if required - set(manifest):
        raise RuntimeError('Code missing from release manifest: ' + repr(sorted(required - set(manifest))))
    return manifest


def gpu_state():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=15)
    rows = [x.strip() for x in result.stdout.splitlines() if x.strip()]
    if len(rows) != 1:
        raise RuntimeError('Pair requires exactly one visible server GPU: ' + repr(rows))
    row = [x.strip() for x in rows[0].split(',')]
    if len(row) != 7:
        raise RuntimeError('Invalid nvidia-smi GPU identity: ' + rows[0])
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=15)
    pids = []
    for line in result.stdout.splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) == 2 and parts[0].isdigit() and parts[1] == row[1]:
            pids.append(int(parts[0]))
    return dict(index=int(row[0]), uuid=row[1], name=row[2], total_MiB=int(row[3]),
                free_MiB=int(row[4]), used_MiB=int(row[5]), utilization_percent=int(row[6]),
                compute_pids=sorted(pids))


def gpu_identity(state):
    return {key: state[key] for key in ('index', 'uuid', 'name', 'total_MiB')}


def require_idle_gpu(state):
    policy = memory_policy()
    if (state['total_MiB'] < policy['minimum_gpu_total_MiB'] or
            state['free_MiB'] < policy['controller_idle_free_MiB'] or state['compute_pids']):
        raise RuntimeError('Pair smoke/run requires idle GPU, >=30 GiB free and no compute PID: ' + repr(state))


def runtime_binding(args, state=None):
    if str(args.device) != 'cuda:0' or args.workers != 8:
        raise RuntimeError('Fixed pair requires cuda:0 and workers=8')
    return dict(root=str(ROOT.resolve()), project=str(args.project.resolve()), source=str(args.source.resolve()),
                data=str(args.data.resolve()), dataset=str(args.dataset.resolve()), device=str(args.device),
                workers=args.workers, protocol=protocol(), memory_policy=memory_policy(),
                package_digest=package_digest(), GPU=gpu_identity(state or gpu_state()),
                source_files={name: file_hash(args.source / 'ultralytics' / name, normalized=True) for name in SOURCE_FILES},
                data_sha256=file_hash(args.data), dataset_fingerprint=dataset_fingerprint(args))


def process_identity(pid):
    # /proc stat field 22 is starttime; command names can contain spaces/parentheses.
    data = (Path('/proc') / str(pid) / 'stat').read_text()
    fields = data[data.rfind(')') + 2:].split()
    if fields[0] == 'Z':
        raise RuntimeError('Process is a zombie: ' + str(pid))
    return dict(pid=int(pid), start_ticks=int(fields[19]), ppid=int(fields[1]))


def is_descendant(pid, ancestor):
    for _ in range(64):
        if pid == ancestor:
            return True
        if pid <= 1:
            return False
        pid = process_identity(pid)['ppid']
    return False


def context_payload(context):
    return {key: value for key, value in context.items() if key != 'token'}


def make_context(args, phase, log_dir, state, invocation):
    if phase not in ('smoke', 'run'):
        raise ValueError('Invalid paired phase')
    identity = process_identity(os.getpid())
    context = dict(schema=SCHEMA, phase=phase, invocation=invocation,
                   controller={k: identity[k] for k in ('pid', 'start_ticks')},
                   binding=runtime_binding(args, state), created_at=time.time(),
                   log_dir=str(Path(log_dir).resolve()))
    if phase == 'run':
        receipt = args.project / 'comparison_reports' / RECEIPT_NAME
        context['pair_receipt'] = dict(path=str(receipt.resolve()), sha256=file_hash(receipt))
    context['token'] = value_hash(context_payload(context))
    write_json(Path(log_dir) / 'pair_context.json', context)
    return context


def validate_worker_context(args, name, phase):
    if name not in NAMES or phase not in ('smoke', 'run'):
        raise RuntimeError('Invalid paired worker/phase')
    if (os.environ.get(ENV_PREFIX + 'PAIR') != '1' or
            os.environ.get(ENV_PREFIX + 'PHASE') != phase):
        raise RuntimeError('Smoke/run must be started by run_pair.sh; PAIR flag alone is insufficient')
    path = Path(os.environ.get(ENV_PREFIX + 'CONTEXT', '')).resolve()
    context = read_json(path)
    token = value_hash(context_payload(context))
    if (context.get('schema') != SCHEMA or context.get('phase') != phase or context.get('token') != token or
            os.environ.get(ENV_PREFIX + 'TOKEN') != token):
        raise RuntimeError('Pair context phase/token/hash mismatch')
    logs = args.project.resolve() / 'logs'
    if (not path.is_relative_to(logs) or path.name != 'pair_context.json' or
            str(path.parent) != context.get('log_dir')):
        raise RuntimeError('Pair context must be stored in this project controller log directory')
    controller = process_identity(context['controller']['pid'])
    if controller['start_ticks'] != context['controller']['start_ticks'] or not is_descendant(os.getpid(), controller['pid']):
        raise RuntimeError('Pair controller no longer owns this worker PID/start_ticks')
    if context.get('binding') != runtime_binding(args):
        raise RuntimeError('Pair context paths/protocol/GPU/source/package changed')
    if phase == 'run':
        receipt = args.project / 'comparison_reports' / RECEIPT_NAME
        reference = context.get('pair_receipt', {})
        if reference != dict(path=str(receipt.resolve()), sha256=file_hash(receipt)):
            raise RuntimeError('Parallel smoke receipt changed after run launch')
        verify_pair_receipt(args)
        # The controller checks both destinations once, before launching either
        # child. Each worker's train() checks only its own run under its own lock.
        # A peer may already have created its directory during staggered startup.
    return context


def barrier(args, name, label, timeout=900):
    if label not in ('training_ready', 'before_validation', 'complete'):
        raise ValueError('Unknown parallel smoke barrier')
    context = validate_worker_context(args, name, 'smoke')
    folder = Path(context['log_dir']) / 'barriers'
    path = folder / f'{name}_{label}.json'
    if path.exists():
        raise RuntimeError('Repeated smoke barrier: ' + str(path))
    identity = process_identity(os.getpid())
    marker = dict(name=name, label=label, token=context['token'], invocation=context['invocation'],
                  pid=identity['pid'], start_ticks=identity['start_ticks'], entered_monotonic=time.monotonic())
    write_json(path, marker)
    started = time.monotonic()
    announced = started
    peer = next(n for n in NAMES if n != name)
    peer_path = folder / f'{peer}_{label}.json'
    while not peer_path.exists():
        if (Path(context['log_dir']) / f'{peer}_failed.json').exists():
            raise RuntimeError(f'Parallel smoke peer {peer} failed before {label}; no receipt')
        controller = process_identity(context['controller']['pid'])
        if controller['start_ticks'] != context['controller']['start_ticks']:
            raise RuntimeError('Parallel controller exited/reused during barrier')
        if time.monotonic() - started > timeout:
            raise RuntimeError(f'Parallel smoke barrier {label} timed out after {timeout}s')
        if time.monotonic() - announced >= 30:
            print(f'{name} paired smoke waiting for {peer}: {label}', flush=True)
            announced = time.monotonic()
        time.sleep(.25)
    other = read_json(peer_path)
    if other.get('token') != context['token'] or other.get('invocation') != context['invocation']:
        raise RuntimeError('Smoke barrier from another controller invocation')
    marker['released_monotonic'] = time.monotonic()
    write_json(path, marker)
    return marker


def smoke_evidence(args, name):
    context = validate_worker_context(args, name, 'smoke')
    stages = {n: {stage: read_json(Path(context['log_dir']) / 'barriers' / f'{n}_{stage}.json')
                  for stage in ('training_ready', 'before_validation', 'complete')} for n in NAMES}
    return dict(token=context['token'], controller=context['controller'], invocation=context['invocation'],
                context=str(Path(context['log_dir']) / 'pair_context.json'),
                context_sha256=file_hash(Path(context['log_dir']) / 'pair_context.json'),
                phase='smoke', binding=context['binding'], worker=name, stages=stages)


def interval_overlap(stages, start_stage, end_stage):
    starts = [stages[n][start_stage]['released_monotonic'] for n in NAMES]
    ends = [stages[n][end_stage]['entered_monotonic'] for n in NAMES]
    return min(ends) - max(starts)


def gate_bundle(args, name):
    reports = args.project / 'comparison_reports'
    result = {}
    for phase, filename in zip(('check', 'smoke'), GATES[name]):
        gate_path = reports / filename
        gate = read_json(gate_path)
        evidence = Path(gate['report']) / (phase + '.json')
        if not evidence.resolve().is_relative_to(reports.resolve()):
            raise RuntimeError('Gate report outside project reports')
        value = read_json(evidence)
        if value.get('status') != 'passed' or (evidence.parent / 'failure.json').exists():
            raise RuntimeError('Failed/incomplete gate evidence: ' + str(evidence))
        if phase == 'check' and value.get('cuda_executed') is not True:
            raise RuntimeError('CPU evidence cannot replace CUDA check')
        if phase == 'smoke':
            validation = value.get('validation', {})
            updates = value.get('updates', [])
            peak = value.get('peak_reserved_GiB', float('inf'))
            if (value.get('successful_optimizer_updates', 0) < 8 or not 8 <= value.get('real_batches', 0) <= 32 or
                    len(updates) != value.get('real_batches') or sum(v.get('updated') is True for v in updates) < 8 or
                    value.get('branch_updated') is not True or set(value.get('updated_groups', [])) != set(expected_groups(name)) or
                    value.get('allocator_cap_GiB') != 13 or value.get('other_job_reserve_GiB') != 2 or
                    not math.isfinite(peak) or not 0 < peak <= 13 or value.get('post_update_inference') is not True or
                    validation.get('validation_images') != 548 or validation.get('validation_batch') != 32 or
                    validation.get('loss_precision') != 'autocast_float16' or
                    any(validation.get(flag) is not True for flag in ('parallel_pair', 'EMA_half',
                        'validation_loss_and_NMS', 'square_batch32', 'raw_logits_and_decoded_finite'))):
                raise RuntimeError('Incomplete real parallel smoke: ' + name)
        result[phase] = dict(gate=str(gate_path.resolve()), gate_sha256=file_hash(gate_path),
                             signature=gate.get('signature'), evidence=str(evidence.resolve()),
                             evidence_sha256=file_hash(evidence), value=value)
    if result['check']['signature'] != result['smoke']['signature']:
        raise RuntimeError('Check/smoke signatures differ: ' + name)
    return result


def build_pair_receipt(args, context):
    binding = runtime_binding(args)
    if context.get('phase') != 'smoke' or context.get('binding') != binding:
        raise RuntimeError('Smoke context differs from current release/protocol/GPU')
    bundles = {name: gate_bundle(args, name) for name in NAMES}
    stages = None
    for name, bundle in bundles.items():
        evidence = bundle['smoke']['value'].get('pair_context', {})
        if (evidence.get('token') != context['token'] or evidence.get('invocation') != context['invocation'] or
                evidence.get('controller') != context['controller'] or evidence.get('binding') != binding or
                evidence.get('worker') != name or evidence.get('phase') != 'smoke'):
            raise RuntimeError('Two standalone/mixed smoke gates cannot issue pair receipt: ' + name)
        context_path = Path(context['log_dir']) / 'pair_context.json'
        if (evidence.get('context') != str(context_path) or
                evidence.get('context_sha256') != file_hash(context_path)):
            raise RuntimeError('Smoke context evidence hash changed')
        current_stages = evidence.get('stages')
        if stages is not None and current_stages != stages:
            # complete released timestamps need not have been written by both at first read.
            for worker in NAMES:
                for stage in ('training_ready', 'before_validation', 'complete'):
                    left = {k: v for k, v in stages[worker][stage].items() if k != 'released_monotonic'}
                    right = {k: v for k, v in current_stages[worker][stage].items() if k != 'released_monotonic'}
                    if left != right:
                        raise RuntimeError('Smoke barrier marker mismatch')
        stages = current_stages
        for worker in NAMES:
            for label, marker in stages[worker].items():
                if (marker.get('token') != context['token'] or marker.get('name') != worker or
                        marker.get('label') != label or marker.get('invocation') != context['invocation']):
                    raise RuntimeError('Invalid parallel smoke stage binding')
    train_overlap = interval_overlap(stages, 'training_ready', 'before_validation')
    validation_overlap = interval_overlap(stages, 'before_validation', 'complete')
    if min(train_overlap, validation_overlap) <= 0:
        raise RuntimeError('Smoke did not have overlapping real training AND validation windows')
    return dict(schema=SCHEMA, status='passed', issued_at=time.time(), binding=binding,
                smoke_context=context, stages=stages,
                overlap_seconds=dict(training=train_overlap, validation=validation_overlap),
                gates={name: {phase: {k: v for k, v in item.items() if k != 'value'}
                              for phase, item in bundle.items()} for name, bundle in bundles.items()})


def verify_pair_receipt(args):
    receipt = read_json(args.project / 'comparison_reports' / RECEIPT_NAME)
    if receipt.get('schema') != SCHEMA or receipt.get('status') != 'passed':
        raise RuntimeError('Missing/invalid parallel smoke receipt')
    rebuilt = build_pair_receipt(args, receipt['smoke_context'])
    for key in ('binding', 'gates', 'overlap_seconds'):
        if receipt.get(key) != rebuilt.get(key):
            raise RuntimeError('Parallel smoke receipt is stale/changed: ' + key)
    return receipt
