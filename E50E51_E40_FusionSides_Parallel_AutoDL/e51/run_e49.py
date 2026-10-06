#!/usr/bin/env python3
"""Standalone E51 seed1/640/200e. No remote edits, queue or parallel controller."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
EXPERIMENT = 'e51a_e40_local_contrast_img640_seed1_200e'
BASE_REPORT = 'e1_yolo11s_p2add_img640_seed1_20260926_102750_718906'
REPORT = None


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def validate_data_yaml(path,expected):
    """E1 records SHA256 of raw bytes, NOT normalized source-text hashes."""
    actual=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual!=expected:
        raise RuntimeError(f'Data YAML raw SHA256 differs from E1: expected={expected}, actual={actual}, file={path}. '
                           'Do not overwrite the dataset or bypass this check; compare with the archived E1 dataset.yaml.')
    return actual


def setup(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['YOLO_CONFIG_DIR'] = str(args.project/'comparison_reports/e51_settings')
    from smoke_progress import seed_existing_font
    seed_existing_font(args.project,os.environ['YOLO_CONFIG_DIR'])
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    import ultralytics
    from ultralytics import YOLO
    import e1_640_core as core
    if not Path(ultralytics.__file__).resolve().is_relative_to(args.source.resolve()):
        raise RuntimeError('Wrong Ultralytics import')
    if ultralytics.__version__ != '8.3.0':
        raise RuntimeError('Requires pinned Ultralytics 8.3.0')
    manifest = HERE/'manifest.json'
    if not manifest.is_file(): raise FileNotFoundError('Incomplete package: missing manifest.json')
    for name, expected in json.loads(manifest.read_text()).items():
        if digest(HERE/name) != expected:
            raise RuntimeError('Package changed: '+name)
    core.ROOT, core.SOURCE, core.DATA = args.project, args.source, args.data
    core.MODEL_YAML, core.PRETRAINED = HERE/'e1_reference.yaml', args.project/'weights/yolo11s.pt'
    core.EXPERIMENT = EXPERIMENT
    core.torch = torch
    core.YOLO = YOLO
    core.AUTO_SHUTDOWN = core.SHUTDOWN_ON_FAILURE = False
    if core.sha256(core.PRETRAINED) != core.PRETRAINED_SHA256:
        raise RuntimeError('Original initialization weights differ')
    torch.set_num_threads(4)
    if args.device != 'cpu':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no automatic CPU fallback')
        torch.cuda.set_device(args.device)
        if args.mode in ('smoke','run'):
            total=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            if total<30.*2**30:
                raise RuntimeError('Real smoke/formal require the 32GiB server GPU; no local 8GiB fallback')
            torch.cuda.set_per_process_memory_fraction(13.*2**30/total)
            print('Fixed 13 GiB allocator cap for smoke AND formal; batch16 unchanged',flush=True)
    return core


def config(core, args):
    result = core.training_params()
    result.update(model=str(core.MODEL_YAML), device=args.device, workers=args.workers)
    return result


def signature(core, args):
    import torch
    gpu=None
    if args.device!='cpu':
        properties=torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu=dict(name=properties.name,total_memory=properties.total_memory,
                 capability=list(torch.cuda.get_device_capability()))
    return dict(params=config(core, args), torch=torch.__version__, cuda=torch.version.cuda,
                GPU=gpu,allocator_cap_GiB=13.,
                baseline_parent=digest(args.project/'comparison_reports/E40_RUN_20260929_205658_507620/comparison.json'),
                evaluation_dataset=str(args.dataset.resolve()),
                source={name:digest(args.source/'ultralytics'/name) for name in
                        ('nn/tasks.py','nn/modules/head.py','nn/modules/conv.py','utils/loss.py',
                         'utils/tal.py','engine/trainer.py','data/augment.py')},
                package={p.name:digest(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.sh','.yaml')},
                yaml=digest(core.MODEL_YAML), pretrained=core.sha256(core.PRETRAINED),
                baseline=digest(args.project/'comparison_reports'/BASE_REPORT/'metrics.json'),
                data=core.sha256(args.data) if args.data.exists() else None)


def protocol_check(core, args):
    baseline = json.loads((args.project/'comparison_reports'/BASE_REPORT/'metrics.json').read_text())
    if (baseline['status'] != 'completed' or baseline['seed'] != 1 or
            baseline['training']['completed_epochs'] != 200):
        raise RuntimeError('Wrong baseline report')
    ignore = {'project','name','model','data','device'}
    left = {k:v for k,v in baseline['train_params'].items() if k not in ignore}
    right = {k:v for k,v in config(core,args).items() if k not in ignore}
    if left != right:
        raise RuntimeError('E1 training protocol differs: '+str({k:(left.get(k),right.get(k)) for k in left.keys()|right.keys() if left.get(k)!=right.get(k)}))
    for name in ('utils/tal.py','utils/loss.py','engine/trainer.py','nn/modules/head.py'):
        if digest(args.source/'ultralytics'/name) != baseline['p2_architecture']['source_code_sha256'][name]:
            raise RuntimeError('Baseline source changed: '+name)
    if args.data.is_file():
        validate_data_yaml(args.data,baseline['data_yaml_sha256'])
    elif args.device!='cpu':
        raise FileNotFoundError('Missing data YAML: '+str(args.data))
    # Fail before training if any reference/evaluation artifact is absent.
    base_path=args.project/'comparison_reports'/BASE_REPORT
    for path in (base_path/'area_ground_truth.json',base_path/'area_predictions.json'):
        if not path.is_file(): raise FileNotFoundError('Missing official comparison input: '+str(path))
    parent_path=args.project/'comparison_reports/E40_RUN_20260929_205658_507620/comparison.json'
    parent=json.loads(parent_path.read_text())
    from parent_reference import reference_e40
    reference_e40(args,parent['dataset'])  # completed parent / checkpoint SHA
    if args.device!='cpu':
        import d27_official as official
        records,fingerprint=official.load_dataset(args.dataset)
        reference_e40(args,fingerprint)
        images=json.loads((base_path/'area_ground_truth.json').read_text())['images']
        if len(images)!=len(records): raise RuntimeError('E1 original-image mapping count drift')
        for record,image in zip(records,images):
            if (record.image_id,record.stem,record.width,record.height)!=(image['id'],Path(image['file_name']).stem,image['width'],image['height']):
                raise RuntimeError('E1 official image ID/name/size mapping drift')
        official.load_predictions(base_path/'area_predictions.json',len(records))
        print('E51 official evaluation files / original annotations / E1 mapping / E40 SHA preflight passed',flush=True)
    # Parent training protocol must match too, not only the E1 protocol.
    parent_training=json.loads((parent_path.parent/'training.json').read_text())
    parent_params={k:v for k,v in parent_training['params'].items() if k not in ignore}
    if parent_params!=right:
        raise RuntimeError('E40 parent training protocol differs')
    return baseline


def build_model(core):
    import torch
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils.torch_utils import init_seeds
    from spatial_task_head import attach, original_to_target
    init_seeds(1, deterministic=True)
    cfg = yaml_model_load(str(core.MODEL_YAML)); cfg['scale'] = 's'
    reference = DetectionModel(copy.deepcopy(cfg), nc=10, verbose=False)
    core.assert_p2_model(reference)
    audit = core.load_p2_pretrained(reference)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if torch.cuda.is_initialized() else None
    target = attach(copy.deepcopy(reference))
    import parent_spatial_task_head as parent_head
    parent = parent_head.attach(copy.deepcopy(reference))
    for name, value in parent.state_dict().items():
        if not torch.equal(value, target.state_dict()[name]):
            raise RuntimeError('E40 common initialization drift: '+name)
    del parent
    if not torch.equal(cpu_rng, torch.get_rng_state()):
        raise RuntimeError('E51 attach changed CPU training RNG')
    if cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
        raise RuntimeError('E51 attach changed CUDA training RNG')
    for name, value in reference.state_dict().items():
        if not torch.equal(value, target.state_dict()[original_to_target(name)]):
            raise RuntimeError('E1 initialization drift: '+name)
    return target, reference, audit


def trainer_class(core, report):
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import de_parallel
    from spatial_task_head import branch, output_magnitudes, reconstruction_state
    class E51Trainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if weights is not None:
                raise RuntimeError('No fine-tuning/resume via formal entry')
            model, reference, audit = build_model(core)
            del reference
            report['initialization'] = audit
            write(Path(report['report_dir'])/'transfer.json',audit)
            return model

        def _setup_train(self, world_size):
            super()._setup_train(world_size)
            import torch
            model = de_parallel(self.model)
            parameters = [p for g in self.optimizer.param_groups for p in g['params']]
            if len(parameters)!=len({id(p) for p in parameters}) or {id(p) for p in parameters}!={id(p) for p in model.parameters()}:
                raise RuntimeError('Optimizer parameter coverage mismatch')
            if self.device.type == 'cuda' and not self.amp:
                raise RuntimeError('AMP unexpectedly disabled; do not change the comparison protocol')
            report['new_parameters'] = sum(p.numel() for p in branch(model).parameters())
            report['parameters'] = sum(p.numel() for p in model.parameters())
            report['runtime_topology'] = model.e49_topology

        def run_callbacks(self, event):
            super().run_callbacks(event)
            if event == 'on_train_epoch_start':
                # One actual input/output amplitude audit per epoch, no tensor
                # retention and no per-batch GPU synchronization overhead.
                de_parallel(self.model).model[24].capture_stats=True
            if event == 'on_fit_epoch_end':
                if report.get('completed_epochs') == self.epoch+1:
                    return  # final_eval invokes this callback again; not a new epoch
                model = de_parallel(self.model)
                magnitude = output_magnitudes(model)
                elapsed = time.perf_counter()-report['_clock']
                previous=report.get('elapsed_seconds',0.)
                report.update(completed_epochs=self.epoch+1, elapsed_seconds=elapsed,
                              last_epoch_seconds=elapsed-previous,task_output_weights_L1=magnitude,
                              reassembly=reconstruction_state(model))
                eta = elapsed/(self.epoch+1)*(self.epochs-self.epoch-1)
                print(f'E51 epoch {self.epoch+1}/{self.epochs} {datetime.now().isoformat()} '
                      f'epoch_seconds={elapsed-previous:.1f} elapsed={elapsed/3600:.2f}h '
                      f'remaining={eta/3600:.2f}h outputs_L1={magnitude} '
                      f'reassembly={report["reassembly"]}', flush=True)
                write(Path(report['report_dir'])/'training.json', {k:v for k,v in report.items() if k!='_clock'})
                if any(v == 0 for v in magnitude):
                    raise RuntimeError('An E40 task residual remained zero after a complete epoch')
    return E51Trainer


def smoke(core, args, out):
    import torch
    from smoke_memory import bounded_smoke
    if args.device == 'cpu':
        raise RuntimeError('Real batch16/640 smoke requires CUDA')
    try:
        with bounded_smoke(torch.cuda, args.device):
            _smoke_batches(core, args, out)
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError('E51 smoke exhausted its bounded memory budget; no smoke gate issued. '
                           'No smoke receipt issued; do not start formal training. '
                           'Do not lower batch or bypass the gate.') from error


def _smoke_batches(core, args, out):
    import torch
    import numpy as np
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset, build_dataloader
    from ultralytics.data.utils import check_det_dataset
    from p2_tal_chunked import chunked_assignment
    from checks import amp_step, smoke_batches, smoke_inference
    from spatial_task_head import branch, output_magnitudes
    from smoke_memory import check_headroom, SMOKE_CAP_GIB
    data = check_det_dataset(str(args.data), autodownload=False)
    params = config(core,args)
    model, reference, _ = build_model(core); del reference
    model.args = get_cfg(overrides=params)
    model = model.to(args.device).train()
    dataset = build_yolo_dataset(model.args, data['train'], 16, data, mode='train', rect=False, stride=32)
    if len(dataset)!=6471:
        raise RuntimeError('Training split size changed')
    loader = build_dataloader(dataset,16,args.workers,shuffle=True,rank=-1)
    factory = trainer_class(core,dict(report_dir=str(out),_clock=time.perf_counter()))
    trainer = factory.__new__(factory); trainer.args=model.args
    optimizer = trainer.build_optimizer(model,name='SGD',lr=.01,momentum=.937,decay=.0005,iterations=200*len(loader))
    scaler = torch.amp.GradScaler('cuda',enabled=True)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    branch_before={n:p.detach().clone() for n,p in branch(model).named_parameters()}
    from smoke_progress import SmokeProgress,stress_batches,MAX_ATTEMPTS,TARGET_UPDATES
    from extension import branch_groups
    from spatial_task_head import VARIANT
    groups=branch_groups(VARIANT)
    progress=SmokeProgress()
    start = time.perf_counter(); updates=[]; seen_loss_gradients=set()
    with chunked_assignment():
        for i,batch in stress_batches(loader):
            check_headroom(torch.cuda,args.device)
            batch['img']=batch['img'].to(args.device).float()/255
            # Engineering-only short stress test: fixed SGD lr=.01 rather than
            # eight almost-zero warmup iterations. Formal Trainer STILL uses
            # the unchanged E1 warmup/accumulation/200-epoch cosine schedule.
            with torch.autocast(device_type='cuda',enabled=True):
                loss,items=model(batch)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite real smoke loss')
            scaler.scale(loss).backward()
            from extension import branch_groups
            from spatial_task_head import VARIANT
            nonzero={g for g in branch_groups(VARIANT) if any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for n,p in branch(model).named_parameters() if n.startswith(g))}
            updates.append(amp_step(model,optimizer,scaler))
            progress.observe(updates[-1])
            if updates[-1]['updated']: seen_loss_gradients.update(nonzero)
            free,_=check_headroom(torch.cuda,args.device)
            print(f'E51 SMOKE attempt {i+1}/{MAX_ATTEMPTS} successful={progress.successful}/{TARGET_UPDATES} loss={items.tolist()} {updates[-1]} '
                  f'reserved={torch.cuda.memory_reserved()/2**30:.2f} GiB free={free:.2f} GiB '
                  f'elapsed={time.perf_counter()-start:.1f}s',flush=True)
            all_changed=all(any(not torch.equal(p,branch_before[n]) for n,p in branch(model).named_parameters() if n.startswith(g)) for g in groups)
            if progress.ready and set(groups)==seen_loss_gradients and all_changed:
                break
    progress.require_success()
    if any(v == 0 for v in output_magnitudes(model)):
        raise RuntimeError('At least one task output did not update during real smoke')
    from extension import branch_groups
    from spatial_task_head import VARIANT
    groups=branch_groups(VARIANT)
    if set(groups)!=seen_loss_gradients: raise RuntimeError('No nonzero successful-step LOSS gradient in '+str(set(groups)-seen_loss_gradients))
    for group in groups:
        if not any(not torch.equal(p,branch_before[n]) for n,p in branch(model).named_parameters() if n.startswith(group)):
            raise RuntimeError('No actual real-smoke parameter update in '+group)
    peak=torch.cuda.max_memory_reserved()/2**30
    if peak > SMOKE_CAP_GIB:
        raise RuntimeError('Smoke exceeds memory budget; no auto batch reduction')
    # Retain the real train model/optimizer while testing EMA FP16 val batch32.
    from validation_smoke import validation_stress
    from pair_barrier import synchronize
    synchronize('train_ready')
    validation=validation_stress(model,args,data,params,out)
    synchronize('validation_done')
    # Real forward after updates, not just a synthetic/checkpoint-only test.
    prediction=smoke_inference(model,batch['img'])
    check_headroom(torch.cuda,args.device)
    peak=torch.cuda.max_memory_reserved()/2**30
    result=dict(validation=validation,status='passed',real_batches=progress.attempts,successful_optimizer_updates=progress.successful,AMP_overflow_skips=progress.skipped,max_real_batches=MAX_ATTEMPTS,updates=updates,peak_reserved_GiB=peak,
                allocator_cap_GiB=SMOKE_CAP_GIB,other_job_reserve_GiB=2.,
                post_update_inference=True,inference_device=str(prediction.device),
                branch_updated=True,updated_groups=groups,task_output_weights_L1=output_magnitudes(model),seconds=time.perf_counter()-start)
    result['engineering_optimizer']='SGD lr=.01, one step per real batch; not the formal schedule'
    write(out/'smoke.json',result)
    write(args.project/'comparison_reports/e51_smoke_gate.json',dict(signature=signature(core,args),report=str(out)))
    print('E51 CUDA SMOKE PASSED',flush=True)


def train(core,args,out):
    from p2_tal_chunked import chunked_assignment
    import torch
    if torch.cuda.mem_get_info()[0]/2**30<2:
        raise RuntimeError('Need at least 2 GiB physical GPU headroom')
    run=args.project/'runs'/EXPERIMENT
    if run.exists(): raise RuntimeError('Run already exists; refusing overwrite: '+str(run))
    report=dict(status='training',report_dir=str(out),started_at=datetime.now().isoformat(),
                params=config(core,args),completed_epochs=0,_clock=time.perf_counter())
    trainer=trainer_class(core,report)(overrides=config(core,args))
    with chunked_assignment() as stats:
        trainer.train()
    if report['completed_epochs']!=200: raise RuntimeError('Did not complete 200 epochs')
    report.update(status='evaluating',training_finished_at=datetime.now().isoformat(),
                  best=str(trainer.best),last=str(trainer.last),matching=stats)
    write(out/'training.json',{k:v for k,v in report.items() if k!='_clock'})
    best=Path(trainer.best)
    del trainer
    import gc,torch
    gc.collect(); torch.cuda.empty_cache()
    from evaluate import evaluate
    evaluate(args,best,out)
    report.update(status='completed',finished_at=datetime.now().isoformat())
    write(out/'training.json',{k:v for k,v in report.items() if k!='_clock'})
    write(out/'completion.json',dict(status='completed',trained_epochs=200,official_evaluation_completed=True,finished_at=report['finished_at']))


def main():
    global REPORT
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('check','smoke','run','eval','local-integration','verify-gates'),required=True)
    p.add_argument('--project',type=Path,default=Path('/root/autodl-tmp/yolo11_project'))
    p.add_argument('--source',type=Path,default=Path('/root/autodl-tmp/yolo11_src'))
    p.add_argument('--data',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019.yaml'))
    p.add_argument('--dataset',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--weights',type=Path)
    args=p.parse_args(); core=setup(args)
    REPORT=args.project/'comparison_reports'/('E51_'+args.mode.upper()+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    REPORT.mkdir(parents=True)
    print(f'E51 {args.mode} START {datetime.now().isoformat()} REPORT={REPORT}',flush=True)
    if args.mode=='local-integration':
        from checks import integration
        integration(core,args,REPORT)
    elif args.mode=='eval':
        protocol_check(core,args)
        from evaluate import evaluate
        if args.weights is None: raise ValueError('eval requires --weights')
        evaluate(args,args.weights,REPORT)
    else:
        protocol_check(core,args)
        if args.mode=='check':
            import torch
            if args.device!='cpu' and torch.cuda.mem_get_info()[0]/2**30<8:
                raise RuntimeError('Need at least 8 GiB free for CUDA check; do not run CUDA check alongside other GPU jobs')
            from checks import check_all
            result=check_all(core,args,REPORT)
            write(REPORT/'check.json',result)
            gate='e51_cpu_check_gate.json' if args.device=='cpu' else 'e51_cuda_check_gate.json'
            write(args.project/'comparison_reports'/gate,dict(signature=signature(core,args),report=str(REPORT)))
            print('E51 '+('CPU' if args.device=='cpu' else 'CUDA')+' CHECK PASSED',flush=True)
        else:
            gates=['e51_cuda_check_gate.json']+(['e51_smoke_gate.json'] if args.mode in ('run','verify-gates') else [])
            for gate in gates:
                path=args.project/'comparison_reports'/gate
                if not path.exists() or json.loads(path.read_text())['signature']!=signature(core,args):
                    raise RuntimeError('Missing/stale gate: '+gate)
                receipt=json.loads(path.read_text())
                evidence=Path(receipt['report'])/('check.json' if 'check' in gate else 'smoke.json')
                result=json.loads(evidence.read_text())
                if result['status']!='passed' or (evidence.parent/'failure.json').exists():
                    raise RuntimeError('Invalid gate evidence: '+str(evidence))
                if 'check' in gate and not result.get('cuda_executed'):
                    raise RuntimeError('CPU check cannot replace CUDA check')
                from extension import branch_groups
                if 'smoke' in gate and (result.get('allocator_cap_GiB')!=13. or result.get('peak_reserved_GiB',float('inf'))>13. or not (8<=result.get('real_batches',0)<=32) or result.get('successful_optimizer_updates',0)<8 or not result.get('branch_updated') or set(result.get('updated_groups',[]))!=set(branch_groups()) or result.get('validation',{}).get('validation_images')!=548 or result.get('validation',{}).get('validation_batch')!=32 or result.get('validation',{}).get('loss_precision')!='autocast_float16' or result.get('validation',{}).get('raw_logits_and_decoded_finite') is not True):
                    raise RuntimeError('Smoke is incomplete or new branch did not update')
            if args.mode=='verify-gates':
                print('E51 CUDA CHECK AND REAL SMOKE RECEIPTS VALID',flush=True)
            else:
                (smoke if args.mode=='smoke' else train)(core,args,REPORT)
    print(f'E51 {args.mode} FINISHED {datetime.now().isoformat()}',flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException:
        import traceback
        if REPORT is not None: write(REPORT/'failure.json',dict(status='failed',error=traceback.format_exc()))
        raise
