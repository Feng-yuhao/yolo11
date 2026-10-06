"""Common from-scratch training implementation, isolated runtime per arm."""
from __future__ import annotations
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def run(variant, entry):
    here = Path(entry).resolve().parent
    runtime = here / f'e36{variant}_runtime'
    if not (runtime / 'install_receipt.json').is_file():
        raise RuntimeError('Install E36 package first; do not modify shared source')
    os.environ.setdefault('YOLO_CONFIG_DIR', str(here / f'e36{variant}_yolo_settings'))
    sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(runtime / 'source'))
    import torch
    import e1_640_core as core
    from ultralytics.nn.modules.e36_sampling import attach, SITE, original_to_target, is_new_key
    from e36_checks import assert_architecture, equivalence, gradient_audit, serialization_audit, local_operator_audit
    base_load, base_assert = core.load_p2_pretrained, core.assert_p2_model
    base_train, base_evaluate, base_save = core.train, core.evaluate, core.save_report
    experiment = f'e36{variant}_p2_sampling_e1_img640_seed1'
    cfg_path = runtime / 'yolo11s-p2-e36.yaml'
    last_report = [None]

    def arm_assert(model):
        result = assert_architecture(model)
        if result['variant'] != variant:
            raise RuntimeError(f'Checkpoint/model belongs to arm {result["variant"]}, not requested {variant}')
        return result

    def verify_snapshot():
        receipt = json.loads((runtime / 'install_receipt.json').read_text(encoding='utf-8'))
        for rel, expected in receipt['runtime_hashes'].items():
            if digest(runtime / rel) != expected:
                raise RuntimeError(f'Private runtime changed: {rel}')
        manifest = json.loads((runtime / 'package_manifest.json').read_text(encoding='utf-8'))
        for name in ('e36_runner.py', f'yolo11s_p2_e36{variant}_img640_seed1.py',
                     'evaluate_e36_640_official.py', 'run_e36_pair.py', 'run_e36_pair.sh'):
            if digest(here / name) != manifest['package_hashes']['payload/' + name]:
                raise RuntimeError(f'Installed entry changed: {name}. Rerun checks/smoke after deliberate version change.')
        return receipt

    def reference_e1():
        from ultralytics.nn.tasks import DetectionModel, yaml_model_load
        from ultralytics.utils import RANK
        from ultralytics.utils.torch_utils import init_seeds
        if RANK != -1:
            raise RuntimeError('Single GPU/process only, not DDP')
        init_seeds(core.SEED + 1 + RANK, deterministic=True)
        cfg = yaml_model_load(str(cfg_path))
        cfg['scale'] = 's'
        model = DetectionModel(copy.deepcopy(cfg), nc=10, verbose=False)
        base_assert(model)
        transfer = base_load(model)
        return model, transfer

    def load_pretrained(target):
        reference, transfer = reference_e1()
        source, destination = reference.state_dict(), target.state_dict()
        shared = {original_to_target(k): v for k, v in source.items()}
        if set(shared) != {k for k in destination if not is_new_key(k)}:
            raise RuntimeError('Incomplete strict original E1 transfer; unaccounted keys')
        for key, value in shared.items():
            if destination[key].shape != value.shape:
                raise RuntimeError(f'Transfer shape drift: {key}')
        merged = dict(destination)
        merged.update(shared)
        target.load_state_dict(merged, strict=True)
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(36001)
            target.get_submodule(SITE).reset_new_parameters()
        if any(not torch.equal(target.state_dict()[k].cpu(), v.cpu()) for k, v in shared.items()):
            raise RuntimeError('Original E1 tensors changed during migration')
        return dict(original_e1_transfer=transfer, common_tensors_exact=True,
                    common_tensor_count=len(shared), seed=1, new_branch_seed=36001,
                    initialization='original yolo11s.pt -> deterministic E1; NOT best.pt or smoke weights',
                    new_keys=sorted(k for k in destination if is_new_key(k)),
                    rng_policy='restore post-reference E1 RNG after new-branch initialization')

    def save_report(report):
        last_report[0] = report
        base_save(report)

    def make_trainer(report):
        from ultralytics.models.yolo.detect import DetectionTrainer
        from ultralytics.nn.tasks import DetectionModel

        class SamplingTrainer(DetectionTrainer):
            def get_model(self, cfg=None, weights=None, verbose=True):
                if weights is not None:
                    raise RuntimeError('No resume/fine-tuning; original-weight controlled run only')
                model = attach(DetectionModel(cfg, nc=self.data['nc'], verbose=verbose), variant)
                assert_architecture(model)
                report['pretrained_transfer'] = load_pretrained(model)
                save_report(report)
                return model

            def build_optimizer(self, model, *args, **kwargs):
                optimizer = super().build_optimizer(model, *args, **kwargs)
                expected = {id(p) for n, p in model.named_parameters() if is_new_key(n)}
                actual = [id(p) for g in optimizer.param_groups for p in g['params']]
                if not expected or any(actual.count(p) != 1 for p in expected):
                    raise RuntimeError('Real training optimizer omitted/duplicated sampling parameters')
                report['sampling_optimizer_audit'] = dict(passed=True, parameter_tensors=len(expected))
                save_report(report)
                return optimizer
        return SamplingTrainer

    def preflight(report, eval_only=False):
        import torchvision
        from ultralytics.nn.tasks import DetectionModel, yaml_model_load
        from p2_tal_chunked import verify_equivalence, assert_patch_restoration, implementation_info
        receipt = verify_snapshot()
        report['environment']['torchvision'] = torchvision.__version__
        if Path(core.ultralytics.__file__).resolve().parent != (core.SOURCE / 'ultralytics').resolve():
            raise RuntimeError('Wrong shared/private Ultralytics import')
        if core.sha256(core.PRETRAINED) != core.PRETRAINED_SHA256:
            raise RuntimeError('Original yolo11s.pt changed')
        torch.set_num_threads(min(4, torch.get_num_threads()))
        cfg = yaml_model_load(str(cfg_path))
        cfg['scale'] = 's'
        model = attach(DetectionModel(copy.deepcopy(cfg), nc=10, verbose=False), variant)
        report['preflight_transfer'] = load_pretrained(model)
        reference, _ = reference_e1()
        report['initial_equivalence'] = equivalence(model, reference)
        report['cpu_sampling_gradients'] = gradient_audit(model, reference)
        out = Path(report['report_dir'])
        report['serialization'] = serialization_audit(model, out)
        report['cpu_operator'] = local_operator_audit()
        if torch.cuda.is_available():
            # Compare true FP32 arithmetic, not cuDNN TF32 vs DCN's FP32 GEMM.
            # These flags are restored even on failure; formal policy is untouched.
            old_tf32 = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cuda.matmul.allow_tf32 = False
                report['cuda_operator_fp32'] = local_operator_audit('cuda:0')
                report['cuda_operator_amp'] = local_operator_audit('cuda:0', amp=True)
                report['cuda_detection_fp32'] = gradient_audit(model, reference, 'cuda:0')
                report['cuda_detection_amp'] = gradient_audit(model, reference, 'cuda:0', amp=True)
                # Built-in validation casts the EMA module itself to FP16, rather
                # than merely entering autocast. Exercise that path separately.
                half_model = copy.deepcopy(model).to('cuda:0').half().eval()
                with torch.no_grad():
                    half_result = half_model(torch.rand(1, 3, 128, 160, device='cuda:0').half())[0]
                if not torch.isfinite(half_result).all():
                    raise RuntimeError('Pure-half EMA validation path failed')
                report['cuda_half_EMA_forward'] = dict(passed=True, rectangular=True)
                report['cuda_comparison_policy'] = 'TF32 disabled for audits only; formal flags restored'
                del half_model, half_result
            finally:
                torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old_tf32
        else:
            raise RuntimeError('Server preflight requires CUDA; installer CPU test is not sufficient')
        counts = sum(p.numel() for p in model.parameters()), sum(p.numel() for p in reference.parameters())
        if counts[0] > counts[1] * 1.15:
            raise RuntimeError('Parameter overhead exceeds +15% resource ceiling')
        report['p2_architecture'] = dict(
            **assert_architecture(model), module_sha256=digest(runtime / 'source/ultralytics/nn/modules/e36_sampling.py'),
            yaml_sha256=digest(cfg_path), frozen_core_sha256=digest(runtime / 'e1_640_core.py'),
            snapshot_receipt_sha256=digest(runtime / 'install_receipt.json'),
            source_code_sha256={name: digest(core.SOURCE / 'ultralytics' / name) for name in (
                'nn/tasks.py', 'nn/modules/block.py', 'nn/modules/head.py', 'utils/loss.py',
                'utils/tal.py', 'engine/trainer.py')},
            preflight_parameters=counts[0], e1_parameters=counts[1],
            theoretical_extra_conv_MACs_640=(64*9 + 64*9 + (64*18 if variant == 'b' else 0))*160*160,
            FLOPs_boundary='DCN sampling/interpolation not reliably counted by THOP; do not publish naive GFLOPs',
            matching_implementation=implementation_info(), shared_source_was_modified=False,
            actual_TAL_assignment_may_change_after_learning=True)
        report['p2_script_revision'] = 'e36_p2_sampling_640_seed1_v1'
        original_cfg = copy.deepcopy(cfg)
        # Standard E1 graph validates TAL/loss code equivalence. Sampling-specific
        # integration is covered above; no registration/parser changes exist.
        from ultralytics.utils import yaml_load
        baseline_cfg = yaml_load(core.SOURCE / 'ultralytics/cfg/models/11/yolo11.yaml')
        baseline_cfg.update(scale='s', nc=10)
        report['matching_equivalence'] = verify_equivalence(original_cfg, device='cuda:0', baseline_cfg=baseline_cfg)
        assert_patch_restoration()
        for p in (cfg_path, here / 'e36_runner.py', runtime / 'e1_640_core.py',
                  core.MATCHING_HELPER, runtime / 'e36_checks.py', runtime / 'source/ultralytics/nn/modules/e36_sampling.py',
                  runtime / 'package_manifest.json', runtime / 'install_receipt.json'):
            shutil.copy2(p, out / p.name)
        save_report(report)
        del model, reference
        gc.collect()
        core.clear_gpu()
        print(f'E36{variant}: exact E1 weights, initial numerical equivalence, CPU/CUDA/AMP gradients, reload/fuse and TAL checks passed.', flush=True)

    def train(report, smoke=False):
        if not smoke and (here / 'runs' / experiment).exists():
            raise RuntimeError('Formal output already exists. Refusing duplicate/restart; inspect results first.')
        return base_train(report, smoke=smoke)

    def evaluate(report, weights, dataset):
        result = base_evaluate(report, weights, dataset)
        # Stock THOP ignores deformable interpolation and our functional convs.
        profile = report.get('model_profile', {})
        profile['uncorrected_THOP_GFLOPs'] = profile.pop('GFLOPs', None)
        profile['GFLOPs'] = None
        profile['GFLOPs_note'] = 'Unsupported DCN sampling profiler; use measured memory/time and analytic extra conv MACs, not naive THOP total.'
        save_report(report)
        return result

    core.ROOT = here
    core.SOURCE = runtime / 'source'
    core.PRETRAINED = here / 'weights/yolo11s.pt'
    core.MODEL_YAML = cfg_path
    core.MATCHING_HELPER = runtime / 'p2_tal_chunked.py'
    core.EXPERIMENT = experiment
    core.SCRIPT_REVISION = 'e36_p2_sampling_640_seed1_v1'
    core.SEED = 1
    core.SMOKE_GATE = here / f'comparison_reports/e36{variant}_640_seed1_smoke_passed.json'
    core.AUTO_SHUTDOWN = core.SHUTDOWN_ON_FAILURE = False
    core.__file__ = str(Path(entry).resolve())
    core.assert_p2_model = arm_assert
    core.load_p2_pretrained = load_pretrained
    core.p2_preflight = preflight
    core.make_p2_trainer = make_trainer
    core.train, core.evaluate, core.save_report = train, evaluate, save_report
    manifest = json.loads((runtime / 'package_manifest.json').read_text(encoding='utf-8'))
    ignored = {'data', 'name', 'project'}
    expected = {k:v for k,v in manifest['baseline_train_params'].items() if k not in ignored}
    actual = {k:v for k,v in core.training_params().items() if k not in ignored}
    if expected != actual:
        raise RuntimeError('Training parameters drifted from actual E1-640 seed1')
    if sys.argv[1:] == ['--local-integration-test']:
        verify_snapshot()
        from ultralytics import YOLO
        # This CPU-only factory test doesn't import the GPU evaluation stack.
        core.torch, core.YOLO = torch, YOLO
        plain = YOLO(str(cfg_path), task='detect').model
        base_assert(plain)
        report = {'report_dir': str(here / 'comparison_reports' / ('LOCAL_E36' + variant)),
                  'status': 'local_factory_test', 'purpose': 'CPU factory/transfer/optimizer test, not training'}
        Path(report['report_dir']).mkdir(parents=True, exist_ok=True)
        cls = make_trainer(report)
        trainer = cls.__new__(cls)
        trainer.data = {'nc': 10}
        model = trainer.get_model(cfg=plain.yaml, weights=None, verbose=False)
        optimizer = trainer.build_optimizer(model, name='SGD', lr=.01, momentum=.937, decay=.0005, iterations=1000)
        reference, _ = reference_e1()
        report['equivalence'] = equivalence(model, reference)
        save_report(report)
        print(json.dumps(dict(status='passed', variant=variant,
                              common_tensors=report['pretrained_transfer']['common_tensor_count'],
                              optimizer=report['sampling_optimizer_audit'], equivalence=report['equivalence']), indent=2), flush=True)
        return
    # Fresh-process resource probe uses REAL augmented DataLoader batches, not
    # zero/single-box synthetic data; it does not save/continue any probe weights.
    if sys.argv[1:] == ['--resource-probe']:
        verify_snapshot()
        core.load_dependencies()
        from p2_tal_chunked import chunked_assignment
        from datetime import datetime
        params = core.training_params()
        params.update(model=str(cfg_path), name=f'resource_e36{variant}_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'), plots=False)
        report = {'report_dir': str(here / 'comparison_reports' / params['name']),
                  'status': 'resource_probe', 'purpose': 'Concurrent resource screening only, not accuracy'}
        Path(report['report_dir']).mkdir(parents=True, exist_ok=False)
        trainer = make_trainer(report)(overrides=params)
        with chunked_assignment():
            trainer._setup_train(world_size=1)
            if not trainer.amp:
                raise RuntimeError('Resource probe AMP was disabled unexpectedly')
            trainer.model.train()
            barrier_value = os.environ.get('E36_RESOURCE_BARRIER')
            barrier = Path(barrier_value) if barrier_value else None
            if barrier:
                (barrier / (variant + '.ready')).write_text('ready', encoding='utf-8')
                deadline = time.monotonic() + 240
                while not (barrier / 'start').exists():
                    if time.monotonic() > deadline:
                        raise RuntimeError('Parallel resource probe barrier timed out')
                    time.sleep(.2)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            rows = []
            for i, raw in enumerate(trainer.train_loader):
                if i == 8:
                    break
                data = trainer.preprocess_batch(raw)
                if barrier:
                    (barrier / f'{variant}.batch{i}').write_text('batch-ready', encoding='utf-8')
                    other = 'b' if variant == 'a' else 'a'
                    deadline = time.monotonic() + 240
                    while not (barrier / f'{other}.batch{i}').exists():
                        if time.monotonic() > deadline:
                            raise RuntimeError(f'Concurrent real-batch barrier timed out at batch {i}')
                        time.sleep(.1)
                trainer.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', enabled=trainer.amp):
                    loss, items = trainer.model(data)
                if not torch.isfinite(loss).all():
                    raise RuntimeError('Nonfinite resource-probe loss')
                trainer.scaler.scale(loss.sum()).backward()
                trainer.scaler.unscale_(trainer.optimizer)
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=10.)
                trainer.scaler.step(trainer.optimizer)
                trainer.scaler.update()
                rows.append({'batch': i+1, 'GT': len(data['cls']), 'loss': float(loss.sum().detach())})
            torch.cuda.synchronize()
            result = dict(status='completed', variant=variant, batches=rows,
                          seconds=time.perf_counter()-start,
                          reserved_GiB=torch.cuda.max_memory_reserved()/1024**3,
                          allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
                          note='Eight augmented training batches; finite sampling, NOT an OOM guarantee or training result')
            (Path(report['report_dir']) / 'resource.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)
        return
    core.main()
    if last_report[0] and last_report[0].get('status') == 'completed' and last_report[0].get('is_formal'):
        subprocess.run([sys.executable, str(here / 'evaluate_e36_640_official.py'), '--variant', variant,
                        '--candidate-report', last_report[0]['report_dir']], check=True)
