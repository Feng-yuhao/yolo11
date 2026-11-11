#!/usr/bin/env python3
"""Isolated YOLO11-P2 + TOOD-TD seed1. Never imports the YOLO26 environment."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import zipfile
from shutdown_safety import write, read, sha, now
from single_safety import EXPERIMENTS, validate_smoke_evidence

HERE = Path(__file__).resolve().parent
EXPERIMENT = EXPERIMENTS['tood_td']
LABEL = 'TOOD_TD'
REPORT = None
if __name__ == '__main__':
    sys.modules['run_experiment'] = sys.modules[__name__]


def normalized_sha(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def package_audit():
    manifest = read(HERE / 'manifest.json')
    for name, expected in manifest.items():
        path = HERE / name
        actual = sha(path) if path.suffix == '.pt' else normalized_sha(path)
        if actual != expected:
            raise RuntimeError('Package file changed: ' + name)
    if manifest.get('d27_official.py') != 'a9e0f50885f81e72ed85baf57a880800c6c51a51527a646b5c60184a1b76bf60':
        raise RuntimeError('Frozen VisDrone evaluator changed')
    return manifest


def source_binding(args):
    expected = dict(read(HERE / 'references/e1_training.json')['p2_architecture']['source_code_sha256'])
    expected.update(read(HERE / 'references/e60_seed1_official_metrics.json')['binding']['source'])
    result = {}
    for name,value in expected.items():
        actual = normalized_sha(args.source / 'ultralytics' / name)
        if name != 'nn/tasks.py' and actual != value:
            raise RuntimeError('Original training/inference source changed: ' + name)
        result[name] = actual
    return result


def setup(args, local=False):
    if not local and (sys.platform != 'linux' or Path(sys.prefix).resolve() !=
                      Path('/root/miniconda3/envs/yolo11').resolve()):
        raise RuntimeError('Use /root/miniconda3/envs/yolo11/bin/python; no base/yolo26 mixing')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    settings = args.project / 'comparison_reports/tood_td_settings'
    os.environ['YOLO_CONFIG_DIR'] = str(settings)
    from smoke_progress import seed_existing_font
    seed_existing_font(args.project, settings)
    if not (settings / 'Arial.ttf').is_file():
        for path in sorted((args.project / 'comparison_reports').glob('*settings/Arial.ttf')):
            if path.stat().st_size > 100_000:
                settings.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, settings / 'Arial.ttf')
                break
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    import ultralytics
    if not local:
        try:
            import pandas
        except ImportError as error:
            raise RuntimeError('Existing yolo11 environment needs pandas for native checkpoint saving; stop before training') from error
    from ultralytics import YOLO
    import e1_640_core as core
    import tood_td_head  # configure original source BEFORE importing this class
    if (Path(ultralytics.__file__).resolve().parent != args.source.resolve() / 'ultralytics' or
            ultralytics.__version__ != '8.3.0'):
        raise RuntimeError('Requires the actual isolated YOLO11 Ultralytics8.3.0 source')
    if not local and (torch.__version__ != '2.5.1+cu121' or torch.version.cuda != '12.1'):
        raise RuntimeError('Torch/CUDA differs from original E1/E60: use existing yolo11 env')
    package_audit()
    source_binding(args)
    os.chdir(HERE)  # genuine bundled yolov8n.pt used ONLY by original AMP check
    if sha(HERE / 'yolov8n.pt') != 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36':
        raise RuntimeError('Missing/changed original offline AMP auxiliary checkpoint')
    if not local and args.mode in ('smoke', 'train') and not (settings / 'Arial.ttf').is_file():
        raise RuntimeError('Restore a genuine cached Arial.ttf in comparison_reports/tood_td_settings; no online wait')
    ultralytics.utils.VERBOSE = False  # display only; epoch/batch callbacks remain
    core.ROOT, core.SOURCE, core.DATA = args.project, args.source, args.data
    core.PRETRAINED, core.MODEL_YAML = args.project / 'weights/yolo11s.pt', HERE / 'e1_reference.yaml'
    core.EXPERIMENT, core.SEED, core.torch, core.YOLO = EXPERIMENT, 1, torch, YOLO
    core.AUTO_SHUTDOWN = core.SHUTDOWN_ON_FAILURE = False
    if sha(core.PRETRAINED) != core.PRETRAINED_SHA256:
        raise RuntimeError('Must initialize from the original COCO yolo11s.pt, not E1/E60 best')
    torch.set_num_threads(4)
    if args.device != 'cpu':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no CPU fallback')
        torch.cuda.set_device(args.device)
        if not local:
            from smoke_memory import set_limit
            set_limit(torch.cuda, args.device)
    return core


def config(core, args):
    return {**core.training_params(), 'model': str(core.MODEL_YAML),
            'device': args.device, 'workers': args.workers}


def protocol_check(core, args, out, local=False):
    base = read(HERE / 'references/e1_training.json')
    parent = read(HERE / 'references/e60_training.json')
    ignore = {'project', 'name', 'model', 'data', 'device'}
    current = {k: v for k, v in config(core, args).items() if k not in ignore}
    for source in (base['train_params'], parent['params']):
        if current != {k: v for k, v in source.items() if k not in ignore}:
            raise RuntimeError('Training protocol differs from paired E1/E60 seed1')
    # Old tasks.py may contain unrelated registrations; bind its actual SHA
    # and check the real E1 topology. Other training/inference files are exact.
    bindings = source_binding(args)
    if not local and sha(args.data) != base['data_yaml_sha256']:
        raise RuntimeError('Original VisDrone training YAML raw SHA256 changed')
    if not local:
        core.load_dependencies()
        dataset_report = {'report_dir': str(out)}
        core.prepare_dataset(dataset_report)
        from evaluation import audit_dataset
        _, val = audit_dataset(args, 'val')
        _, test = audit_dataset(args, 'test-dev')
        dataset_report.update(official_val=val, official_test=test)
        # Bind actual normalized train labels as well as image names/sizes.
        # This is a staleness check, not a claim of original train pixel hashes.
        from ultralytics.data.utils import check_det_dataset, img2label_paths
        ds = check_det_dataset(str(args.data), autodownload=False)
        images = sorted(p for p in Path(ds['train']).rglob('*') if p.suffix.lower() in core.SUFFIXES)
        train_digest = hashlib.sha256()
        for image, label in zip(images, img2label_paths([str(v) for v in images])):
            train_digest.update(image.name.encode() + b'\0' + str(image.stat().st_size).encode() + b'\0')
            train_digest.update(Path(label).read_bytes() + b'\0')
        dataset_report['train_names_sizes_labels_sha256'] = train_digest.hexdigest()
        write(out / 'dataset_audit.json', dataset_report)
    else:
        dataset_report = {'local_only': True}
    import torch
    gpu = None
    if args.device != 'cpu':
        prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu = dict(name=prop.name, bytes=prop.total_memory,
                   capability=list(torch.cuda.get_device_capability()))
    binding = dict(package=package_audit(), source=bindings, pretrained=sha(core.PRETRAINED),
                   params=config(core, args), torch=torch.__version__, cuda=torch.version.cuda,
                   GPU=gpu, seed=1, allocator_cap_GiB=28., local_only=local,
                   data=None if local else sha(args.data),
                   train_labels=None if local else dataset_report['train_names_sizes_labels_sha256'],
                   val_annotation=None if local else val['annotation_dimension_sha256'],
                   test_annotation=None if local else test['annotation_dimension_sha256'])
    write(out / 'protocol.json', binding)
    return binding


def build_model(core):
    import torch
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils.torch_utils import init_seeds
    from tood_td_head import attach, NEW_PREFIX
    init_seeds(1, deterministic=True)
    cfg = yaml_model_load(str(core.MODEL_YAML)); cfg['scale'] = 's'
    reference = DetectionModel(copy.deepcopy(cfg), nc=10, verbose=False)
    core.assert_p2_model(reference)
    audit = core.load_p2_pretrained(reference)
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if torch.cuda.is_initialized() else None
    target = attach(copy.deepcopy(reference))
    if not torch.equal(rng, torch.get_rng_state()) or (cuda_rng is not None and
            not torch.equal(cuda_rng, torch.cuda.get_rng_state())):
        raise RuntimeError('New module initialization changed the training RNG')
    for name, value in reference.state_dict().items():
        if not torch.equal(value, target.state_dict()[name]):
            raise RuntimeError('Common E1 pretrained tensor changed: ' + name)
    if any(not k.startswith(NEW_PREFIX) for k in set(target.state_dict()) - set(reference.state_dict())):
        raise RuntimeError('Unexpected non-TD tensor additions')
    audit.update(public_module='TOOD TaskDecomposition adaptation', initialization_seed=64001,
                 common_E1_tensors_exact=True, from_trained_best=False)
    return target, reference, audit


def trainer_class(core, report):
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import de_parallel
    from tood_td_head import branch, output_magnitudes

    class TDTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if weights is not None:
                raise RuntimeError('Formal entry does not resume or fine-tune trained best')
            model, reference, audit = build_model(core)
            del reference
            write(Path(report['report_dir']) / 'transfer.json', audit)
            return model

        def _setup_train(self, world_size):
            super()._setup_train(world_size)
            model = de_parallel(self.model)
            parameters = [p for group in self.optimizer.param_groups for p in group['params']]
            if len(parameters) != len({id(p) for p in parameters}) or \
                    {id(p) for p in parameters} != {id(p) for p in model.parameters()}:
                raise RuntimeError('Original Trainer did not cover every model parameter exactly once')
            if self.device.type == 'cuda' and not self.amp:
                raise RuntimeError('AMP disabled unexpectedly; do not change comparison protocol')
            report.update(parameters=sum(p.numel() for p in model.parameters()),
                          new_parameters=sum(p.numel() for p in branch(model).parameters()))

        def run_callbacks(self, event):
            super().run_callbacks(event)
            if event == 'on_train_epoch_start':
                self._td_batch = 0
            elif event == 'on_train_batch_end':
                self._td_batch = getattr(self, '_td_batch', 0) + 1
                if self._td_batch % 80 == 0:
                    report.update(current_batch=self._td_batch, current_epoch=self.epoch + 1,
                                  batches_per_epoch=len(self.train_loader), heartbeat_at=now())
                    print(f'{now()} TD epoch={self.epoch+1}/{self.epochs} batch={self._td_batch}/'
                          f'{len(self.train_loader)} loss={self.loss_items.tolist()}', flush=True)
                    save_training(report)
            elif event == 'on_fit_epoch_end':
                if report.get('completed_epochs') == self.epoch + 1:
                    return  # original final_eval callback is not a second epoch
                elapsed = time.perf_counter() - report['_clock']
                previous = report.get('elapsed_seconds', 0.)
                metrics = {k: float(v) for k, v in self.metrics.items()}
                ap = metrics.get('metrics/mAP50-95(B)')
                report.update(completed_epochs=self.epoch + 1, current_batch=None,
                              elapsed_seconds=elapsed, last_epoch_seconds=elapsed-previous,
                              native_metrics=metrics, heartbeat_at=now(),
                              output_L1=output_magnitudes(de_parallel(self.model)))
                if ap is not None and ap > report.get('best_native_ap', -1.):
                    report.update(best_native_ap=ap, best_native_ap_epoch=self.epoch+1)
                eta = elapsed / (self.epoch+1) * (self.epochs-self.epoch-1)
                report['ETA_hours'] = eta / 3600
                print(f'{now()} TD epoch {self.epoch+1}/{self.epochs} native_mAP50-95={ap} '
                      f'best={report.get("best_native_ap")} best_epoch={report.get("best_native_ap_epoch")} '
                      f'epoch_seconds={elapsed-previous:.1f} elapsed={elapsed/3600:.2f}h ETA={eta/3600:.2f}h', flush=True)
                save_training(report)
                if any(v == 0 for v in report['output_L1']):
                    raise RuntimeError('A TD output adapter remained exactly zero after a full epoch')
    return TDTrainer


def save_training(report):
    write(Path(report['report_dir']) / 'training.json', {k: v for k, v in report.items() if k != '_clock'})


def snapshot(args, out):
    manifest = package_audit()
    with zipfile.ZipFile(out / 'actual_experiment_source.zip', 'w', zipfile.ZIP_DEFLATED) as target:
        for name in list(manifest) + ['manifest.json']:
            target.write(HERE / name, name)
    with zipfile.ZipFile(out / 'actual_ultralytics_source.zip', 'w', zipfile.ZIP_DEFLATED) as target:
        for path in sorted((args.source / 'ultralytics').rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.yml'):
                target.write(path, path.relative_to(args.source).as_posix())


def train(core, args, out, binding):
    from p2_tal_chunked import chunked_assignment
    import torch
    run = args.project / 'runs' / EXPERIMENT
    if run.exists():
        raise RuntimeError('Formal run already exists; never overwrite/retry silently: ' + str(run))
    report = dict(status='training', seed=1, variant='tood_td', report_dir=str(out),
                  params=config(core, args), signature=binding, completed_epochs=0,
                  started_at=now(), _clock=time.perf_counter())
    save_training(report)
    snapshot(args, out)
    trainer = trainer_class(core, report)(overrides=config(core, args))
    with chunked_assignment() as stats:
        trainer.train()
    if report['completed_epochs'] != 200:
        raise RuntimeError('Formal training did not complete exactly200 epochs')
    best, last = Path(trainer.best), Path(trainer.last)
    report.update(status='completed', finished_at=now(), best=str(best), last=str(last), matching=stats,
                  best_sha256=sha(best), last_sha256=sha(last))
    save_training(report)
    write(out / 'training_completion.json', dict(status='completed', trained_epochs=200,
          seed=1, variant='tood_td', val_selected_best=str(best), best_sha256=sha(best),
          last=str(last), last_sha256=sha(last), official_evaluation_completed=False))
    write(args.project / 'logs/latest_tood_td_train.json',dict(report=str(out),best=str(best),best_sha256=sha(best)))
    del trainer
    gc.collect(); torch.cuda.empty_cache()
    print(f'{now()} TD FORMAL200 COMPLETED; controller next scores official val then test-dev', flush=True)


def verify_gates(args, binding):
    verified = {}
    for mode in ('check', 'smoke'):
        gate = read(args.project / 'comparison_reports' / ('tood_td_' + mode + '_gate.json'))
        if gate.get('signature') != binding or gate.get('status') != 'passed':
            raise RuntimeError('Missing/stale TD gate: ' + mode)
        evidence_path = Path(gate['report']) / (mode + '.json')
        evidence = read(evidence_path)
        if sha(evidence_path) != gate['evidence_sha256'] or evidence.get('status') != 'passed' or \
                (evidence_path.parent / 'failure.json').exists():
            raise RuntimeError('Invalid TD gate evidence: ' + mode)
        if mode == 'check' and not evidence.get('cuda_executed'):
            raise RuntimeError('CPU engineering checks cannot authorize real server training')
        if mode == 'smoke':
            validate_smoke_evidence(evidence)
        verified[mode] = gate
    print('TD CUDA CHECK / REAL640-B16 / FULL548-VAL-B32 SMOKE GATES VALID', flush=True)
    return verified


def main():
    global REPORT
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True, choices=('check','smoke','verify-gates','train','val','test','summary'))
    p.add_argument('--project', type=Path, default=Path('/root/autodl-tmp/yolo11_project'))
    p.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/yolo11_src'))
    p.add_argument('--data', type=Path, default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019.yaml'))
    p.add_argument('--dataset', type=Path, default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val'))
    p.add_argument('--test-dataset', type=Path, default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-test-dev'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--weights', type=Path)
    p.add_argument('--report-dir', type=Path, required=True)
    p.add_argument('--report-root', type=Path, required=True)
    args = p.parse_args()
    for name in ('project','source','data','dataset','test_dataset','report_dir','report_root'):
        setattr(args, name, getattr(args, name).resolve())
    if args.weights is not None:
        args.weights = args.weights.resolve()
    if args.device != 'cuda:0' or args.workers != 8:
        p.error('Formal comparison is fixed cuda:0/workers8; developer fixtures use local_validate.py')
    REPORT = args.report_dir
    if not REPORT.is_relative_to(args.project / 'comparison_reports'):
        raise RuntimeError('Result must be inside this project/comparison_reports')
    REPORT.mkdir(parents=True, exist_ok=False)
    import fcntl
    lock_path = args.project / 'logs/.tood_td_execution.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print(f'{now()} TD {args.mode} START REPORT={REPORT}', flush=True)
    core = setup(args)
    if args.mode in ('val','test','summary'):
        from evaluation import evaluate_stage, summarize
        if args.mode == 'summary':
            summarize(args)
        else:
            evaluate_stage(args, 'val' if args.mode == 'val' else 'test-dev')
    else:
        binding = protocol_check(core, args, REPORT)
        if args.mode == 'check':
            from checks import check_all
            result = check_all(core, args, REPORT)
            write(REPORT / 'check.json', result)
        elif args.mode == 'smoke':
            gate = read(args.project / 'comparison_reports/tood_td_check_gate.json')
            if gate.get('signature') != binding or gate.get('status') != 'passed' or \
                    sha(Path(gate['report']) / 'check.json') != gate['evidence_sha256']:
                raise RuntimeError('Fresh CUDA check must pass before real smoke')
            check = read(Path(gate['report']) / 'check.json')
            if not check.get('cuda_executed') or check.get('status') != 'passed' or \
                    (Path(gate['report']) / 'failure.json').exists():
                raise RuntimeError('Real smoke requires successful CUDA, not CPU/stale/failed evidence')
            from checks import real_smoke
            result = real_smoke(core, args, REPORT)
            validate_smoke_evidence(result)
            write(REPORT / 'smoke.json', result)
        else:
            verified = verify_gates(args, binding)
            write(REPORT / 'verified_gates.json', verified)
            if args.mode == 'train':
                train(core, args, REPORT, binding)
        if args.mode in ('check','smoke'):
            path = REPORT / (args.mode + '.json')
            write(args.project / 'comparison_reports' / ('tood_td_' + args.mode + '_gate.json'),
                  dict(status='passed', signature=binding, report=str(REPORT), evidence_sha256=sha(path)))
            print('TD ' + args.mode.upper() + ' PASSED', flush=True)
    print(f'{now()} TD {args.mode} FINISHED', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        if REPORT is not None:
            write(REPORT / 'failure.json', dict(status='failed', at=now(), error=traceback.format_exc()))
        raise
