"""Pair barriers and receipts. CPU unit-testable; no remote side effects."""
import hashlib
import json
import os
from pathlib import Path
import time

CAPS = {'e58': 15., 'e59': 13.}
EXPERIMENTS = {'e58': 'e58a_e57_allscale_w96_img640_seed1_200e',
               'e59': 'e59a_e57_dual_readout_img640_seed1_200e'}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def source_manifest(root):
    return {p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for p in sorted(Path(root).iterdir()) if p.suffix in ('.py', '.sh', '.yaml')}


def barrier(args, stage, timeout=600):
    if args.pair_session is None:
        raise RuntimeError('Real smoke must be launched by run_pair.sh smoke')
    root = args.pair_session
    write_json(root / f'{args.variant}_{stage}.json',
               dict(variant=args.variant, stage=stage, pid=os.getpid(), at=time.time()))
    deadline = time.monotonic() + timeout
    other = 'e59' if args.variant == 'e58' else 'e58'
    last_print = 0.
    while not (root / f'{other}_{stage}.json').is_file():
        if (root / 'abort.json').is_file():
            raise RuntimeError('Parallel smoke peer failed; no pair gate issued')
        if time.monotonic() >= deadline:
            raise RuntimeError('Parallel smoke barrier timeout: ' + stage)
        if time.monotonic() - last_print >= 15:
            print(f'{args.variant.upper()} SMOKE waiting for {other} at {stage}', flush=True)
            last_print = time.monotonic()
        time.sleep(.5)


def validate_smoke_evidence(value, variant, session):
    if (value.get('status') != 'passed' or value.get('variant') != variant or
            value.get('pair_session') != str(session) or
            value.get('successful_optimizer_updates', 0) < 8 or
            not value.get('branch_updated') or
            set(value.get('updated_groups', [])) != set(value.get('expected_groups', [])) or
            len(value.get('updated_groups', [])) != (28 if variant == 'e58' else 34) or
            value.get('allocator_cap_GiB') != CAPS[variant] or
            value.get('peak_reserved_GiB', 100) > CAPS[variant] + .01):
        raise RuntimeError('Incomplete real smoke evidence for ' + variant)
    val = value.get('validation', {})
    if (val.get('validation_images') != 548 or val.get('validation_batch') != 32 or
            not val.get('EMA_half') or not val.get('square_batch32') or
            not val.get('validation_loss_and_NMS') or not val.get('parallel_pair')):
        raise RuntimeError('Incomplete parallel validation smoke for ' + variant)


def check_pair_receipt(project, expected):
    path = Path(project) / 'comparison_reports/e58_e59_pair_smoke_gate.json'
    receipt = json.loads(path.read_text())
    session = Path(receipt['session'])
    if receipt.get('status') != 'passed' or receipt.get('signatures') != expected:
        raise RuntimeError('Missing/stale joint smoke signatures; rerun pair smoke')
    state = json.loads((session / 'state.json').read_text())
    if state.get('exit_codes') != {'e58': 0, 'e59': 0} or (session / 'abort.json').exists():
        raise RuntimeError('Joint smoke did not complete successfully')
    for name in CAPS:
        report = Path(receipt['reports'][name])
        pointer = json.loads((session / f'{name}_report.json').read_text())
        individual = json.loads((Path(project) / 'comparison_reports' / f'{name}_smoke_gate.json').read_text())
        if (pointer.get('report') != str(report) or individual.get('report') != str(report) or
                individual.get('signature') != expected[name]):
            raise RuntimeError('Joint/individual smoke report mismatch: ' + name)
        if (report / 'failure.json').exists():
            raise RuntimeError('Failed smoke report: ' + str(report))
        evidence = json.loads((report / 'smoke.json').read_text())
        validate_smoke_evidence(evidence, name, session)
        for stage in ('training_done', 'validation_start', 'validation_done'):
            if not (session / f'{name}_{stage}.json').is_file():
                raise RuntimeError('Missing simultaneous smoke barrier: ' + name + '/' + stage)
    return receipt
