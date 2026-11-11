#!/usr/bin/env python3
"""Fixed seed1 routing ablations; original E60 training recipe and official val."""
import argparse
import copy
import gc
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile
import common as c

SEED = 1
BASE_REPORT = c.E1_REPORT
REPORT = None
if __name__ == '__main__':
    sys.modules['run_experiment'] = sys.modules[__name__]
write = c.write


def setup(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    os.environ['YOLO_CONFIG_DIR'] = str(args.project/'comparison_reports'/('e60_ablation_'+args.variant+'_settings'))
    from smoke_progress import seed_existing_font
    seed_existing_font(args.project, os.environ['YOLO_CONFIG_DIR'])
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    import ultralytics
    from ultralytics import YOLO
    import pandas
    import e1_640_core as core
    import spatial_task_head as head
    import smoke_memory
    c.package_audit()
    c.require(Path(ultralytics.__file__).resolve().is_relative_to(args.source.resolve()), 'Wrong Ultralytics import')
    c.require(ultralytics.__version__ == '8.3.0', 'Requires pinned Ultralytics 8.3.0')
    head.configure(args.variant)
    smoke_memory.configure(args.execution)
    core.ROOT, core.SOURCE, core.DATA = args.project, args.source, args.data
    core.MODEL_YAML, core.PRETRAINED = c.HERE/'e1_reference.yaml', args.project/'weights/yolo11s.pt'
    core.EXPERIMENT, core.SEED = c.EXPERIMENTS[args.variant], SEED
    core.variant, core.torch, core.YOLO = args.variant, torch, YOLO
    core.AUTO_SHUTDOWN = core.SHUTDOWN_ON_FAILURE = False
    c.require(core.sha256(core.PRETRAINED) == core.PRETRAINED_SHA256, 'Original COCO initialization weights differ')
    torch.set_num_threads(4)
    if args.device != 'cpu':
        c.require(torch.cuda.is_available(), 'CUDA unavailable; no CPU training fallback')
        torch.cuda.set_device(args.device)
        if args.mode in ('smoke','run'):
            total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            c.require(total >= 30*2**30, 'Joint smoke/formal requires existing 32GiB GPU')
            cap = c.CAPS_GIB[args.execution]
            torch.cuda.set_per_process_memory_fraction(cap*2**30/total)
            print(c.now()+' '+args.variant+f' allocator={cap:g}GiB; original batch16/val32 unchanged', flush=True)
    c.protocol_audit(args, core, allow_missing_data=args.mode.startswith('local-'))
    if not args.mode.startswith('local-'):
        import d27_official as official
        records, fingerprint = official.load_dataset(args.dataset)
        saved = c.read(args.project/'comparison_reports'/c.E60_REPORT/'comparison.json')
        c.require(fingerprint == saved['dataset'], 'Official original annotation/image fingerprint changed')
        images = c.read(args.project/'comparison_reports'/c.E1_REPORT/'area_ground_truth.json')['images']
        c.require(len(records)==len(images)==548, 'Official validation split changed')
        for record, image in zip(records,images):
            c.require((record.image_id,record.stem,record.width,record.height) ==
                      (image['id'],Path(image['file_name']).stem,image['width'],image['height']), 'Official ID/size mapping differs')
        print(c.now()+' '+args.variant+' frozen E60 / original E1 seed1 / annotations / protocol passed', flush=True)
    return core


def config(core, args):
    params = core.training_params()
    params.update(model=str(core.MODEL_YAML), device=args.device, workers=args.workers)
    return params


def build_model(core):
    import torch
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils.torch_utils import init_seeds
    import spatial_task_head as head
    init_seeds(SEED, deterministic=True)
    cfg = yaml_model_load(str(core.MODEL_YAML)); cfg['scale'] = 's'
    reference = DetectionModel(copy.deepcopy(cfg), nc=10, verbose=False)
    core.assert_p2_model(reference)
    audit = core.load_p2_pretrained(reference)
    before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state().clone() if torch.cuda.is_initialized() else None
    target = head.attach(copy.deepcopy(reference))
    full = head.full.attach(copy.deepcopy(reference))
    head.assert_common_initialization(full,target,head.VARIANT)
    del full
    c.require(torch.equal(before, torch.get_rng_state()), 'Attach changed CPU training RNG')
    if cuda_before is not None:
        c.require(torch.equal(cuda_before,torch.cuda.get_rng_state()), 'Attach changed CUDA training RNG')
    for name, value in reference.state_dict().items():
        c.require(torch.equal(value,target.state_dict()[name]), 'Original E1 parameter initialization changed: '+name)
    return target, reference, audit


def trainer_class(core, report):
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import de_parallel
    from spatial_task_head import branch, output_magnitudes
    class AblationTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            c.require(weights is None, 'No best-weight fine tuning or implicit resume')
            model, reference, audit = build_model(core); del reference
            report['initialization'] = audit
            write(Path(report['report_dir'])/'transfer.json', audit)
            return model

        def _setup_train(self, world_size):
            super()._setup_train(world_size)
            model = de_parallel(self.model)
            covered = [p for group in self.optimizer.param_groups for p in group['params']]
            c.require(len(covered)==len({id(p) for p in covered}) and
                      {id(p) for p in covered}=={id(p) for p in model.parameters()}, 'Optimizer coverage mismatch')
            c.require(self.device.type != 'cuda' or self.amp, 'Formal AMP unexpectedly disabled')
            report['parameters'] = sum(p.numel() for p in model.parameters())
            report['new_parameters'] = sum(p.numel() for p in branch(model).parameters())

        def run_callbacks(self, event):
            super().run_callbacks(event)
            if event != 'on_fit_epoch_end' or report.get('completed_epochs')==self.epoch+1:
                return
            elapsed = time.perf_counter()-report['_clock']
            seconds = elapsed-report.get('elapsed_seconds',0.)
            magnitude = output_magnitudes(de_parallel(self.model))
            report.update(completed_epochs=self.epoch+1, elapsed_seconds=elapsed,
                          epoch_seconds=seconds, metrics=dict(self.metrics), outputs_L1=magnitude)
            eta = elapsed/(self.epoch+1)*(self.epochs-self.epoch-1)
            line = (f'{c.now()} {core.variant.upper()} epoch {self.epoch+1}/{self.epochs} '
                    f'mAP50-95={self.metrics.get("metrics/mAP50-95(B)",float("nan")):.5f} '
                    f'epoch_seconds={seconds:.1f} elapsed={elapsed/3600:.2f}h ETA={eta/3600:.2f}h')
            print(line, flush=True)
            with (Path(report['report_dir'])/'progress.log').open('a',encoding='utf-8') as handle:
                handle.write(line+'\n')
            write(Path(report['report_dir'])/'training.json', {k:v for k,v in report.items() if k!='_clock'})
            c.require(all(v>0 for v in magnitude), 'A retained output/refinement remained zero after an epoch')
    return AblationTrainer


def validate_smoke(value, variant, execution):
    import spatial_task_head as head
    groups = head.branch_groups()
    c.require(value.get('status')=='passed' and value.get('variant')==variant and value.get('seed')==1,
              'Invalid smoke identity')
    c.require(value.get('successful_optimizer_updates',0)>=8 and value.get('branch_updated') and
              set(value.get('expected_groups',[]))==set(groups) and
              set(value.get('updated_groups',[]))==set(groups) and
              value.get('execution')==execution and value.get('allocator_cap_GiB')==c.CAPS_GIB[execution] and
              value.get('peak_reserved_GiB',100)<=c.CAPS_GIB[execution]+.01,
              'Missing real retained-branch updates/memory evidence')
    v = value.get('validation',{})
    c.require(v.get('validation_batch')==32 and v.get('validation_images')==548 and
              v.get('EMA_half') and v.get('square_batch32') and v.get('validation_loss_and_NMS') and
              value.get('training_model_and_optimizer_retained'), 'Incomplete validation evidence')
    if execution == 'parallel':
        c.require(value.get('both_training_models_and_optimizers_retained') and
                  value.get('both_EMA_copies_retained_through_validation'), 'Incomplete joint residency evidence')


def smoke(core,args,out):
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset, build_dataloader
    from ultralytics.data.utils import check_det_dataset
    from p2_tal_chunked import chunked_assignment
    from checks import amp_step, smoke_inference
    from spatial_task_head import branch, branch_groups, output_magnitudes
    from smoke_progress import SmokeProgress, stress_batches
    from smoke_memory import bounded_smoke, check_headroom
    from pair_safety import barrier
    from validation_smoke import validation_stress
    c.require(args.device!='cpu' and (args.execution=='serial' or args.pair is not None), 'Real smoke only via controller')
    c.verify_gate(args,'check',c.signature(args,core))
    with bounded_smoke(torch.cuda,args.device):
        data = check_det_dataset(str(args.data), autodownload=False)
        model, reference, _ = build_model(core); del reference
        params = config(core,args)
        model.args = get_cfg(overrides=params)
        model = model.to(args.device).train()
        # Serialize cache creation only; do not hold the lock at a rendezvous.
        import fcntl
        with (args.project/'logs'/'e60_ablation_dataset_build.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            dataset = build_yolo_dataset(model.args,data['train'],16,data,mode='train',rect=False,stride=32)
            c.require(len(dataset)==6471, 'Real training split size changed')
            loader = build_dataloader(dataset,16,args.workers,shuffle=True,rank=-1)
        factory = trainer_class(core,dict(report_dir=str(out),_clock=time.perf_counter()))
        dummy = factory.__new__(factory); dummy.args = model.args
        optimizer = dummy.build_optimizer(model,name='SGD',lr=.01,momentum=.937,decay=.0005,iterations=200*len(loader))
        scaler = torch.amp.GradScaler('cuda',enabled=True)
        before = {n:p.detach().clone() for n,p in branch(model).named_parameters()}
        groups, seen, progress, history = branch_groups(), set(), SmokeProgress(), []
        torch.cuda.reset_peak_memory_stats()
        barrier(args,'training_models_and_optimizers_ready')
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        with chunked_assignment():
            for i,batch in stress_batches(loader):
                check_headroom(torch.cuda,args.device)
                batch['img'] = batch['img'].to(args.device).float()/255
                with torch.autocast('cuda',dtype=torch.float16):
                    loss, items = model(batch)
                c.require(torch.isfinite(loss), 'Nonfinite real smoke loss')
                scaler.scale(loss).backward()
                active = {g for g in groups if any(p.grad is not None and torch.isfinite(p.grad).all()
                          and p.grad.count_nonzero()>0 for n,p in branch(model).named_parameters() if n.startswith(g))}
                history.append(amp_step(model,optimizer,scaler)); progress.observe(history[-1])
                if history[-1]['updated']:
                    seen.update(active)
                changed = {g for g in groups if any(not torch.equal(p,before[n])
                           for n,p in branch(model).named_parameters() if n.startswith(g))}
                write(out/'branch_gradient_audit.json',dict(attempts=i+1,successful=progress.successful,
                      expected_groups=groups,gradient_groups=sorted(seen),changed_groups=sorted(changed)))
                print(f'{c.now()} {args.variant} SMOKE attempt={i+1}/32 updates={progress.successful}/8 '
                      f'groups={len(changed)}/{len(groups)} loss={items.tolist()} {history[-1]} '
                      f'reserved={torch.cuda.memory_reserved()/2**30:.2f}GiB', flush=True)
                if progress.ready and seen==changed==set(groups):
                    break
        progress.require_success()
        c.require(seen==changed==set(groups), 'Real smoke missing retained group gradient/update')
        c.require(all(v>0 for v in output_magnitudes(model)), 'Real smoke missing retained output/refinement update')
        del before, loss
        # Match formal lifetime: both FP32 models and optimizer state remain;
        # also create both EMA copies BEFORE either worker starts validation.
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        ema = copy.deepcopy(model).eval().half(); ema.criterion = None
        barrier(args,'both_EMA_copies_ready')
        validation = validation_stress(model,args,data,params,out,ema_model=ema,keep_ema=True)
        validation['parallel_pair'] = args.execution == 'parallel'
        prediction = smoke_inference(model,batch['img'])
        check_headroom(torch.cuda,args.device)
        barrier(args,'full_validation_done',dict(images=validation['validation_images']))
        result = dict(status='passed',variant=args.variant,seed=1,
             successful_optimizer_updates=progress.successful, AMP_overflow_skips=progress.skipped,
             expected_groups=groups,updated_groups=sorted(changed),branch_updated=True,
             updates=history,validation=validation,execution=args.execution,allocator_cap_GiB=c.CAPS_GIB[args.execution],
             peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
             training_model_and_optimizer_retained=True,
             both_training_models_and_optimizers_retained=args.execution=='parallel',
             both_EMA_copies_retained_through_validation=args.execution=='parallel',
             joint_session=str(args.pair) if args.pair else None,post_update_inference=True,
             seconds=time.perf_counter()-started,precision_policy='original E60 FP32 refiners / unchanged AMP parent')
        validate_smoke(result,args.variant,args.execution)
        write(out/'smoke.json',result)
        write(c.gate_path(args,'smoke'),dict(signature=c.signature(args,core),report=str(out)))
        barrier(args,'both_smoke_receipts_saved')
        print(c.now()+' '+args.variant+' '+args.execution.upper()+' CUDA SMOKE PASSED', flush=True)


def verify_gates(core,args):
    binding = c.signature(args,core)
    c.verify_gate(args,'check',binding)
    _, value = c.verify_gate(args,'smoke',binding); validate_smoke(value,args.variant,args.execution)
    if args.execution == 'parallel':
        pair = c.read(c.pair_gate(args.project))
        c.require(pair['status']=='passed' and pair['signatures'][args.variant]==binding and
                  pair['session']==value['joint_session'], 'Matching joint smoke gate missing/stale')
    print(c.now()+' '+args.variant+' CUDA CHECK AND '+args.execution.upper()+' SMOKE RECEIPTS VALID', flush=True)


def train(core,args,out):
    import torch
    from p2_tal_chunked import chunked_assignment
    from smoke_memory import check_headroom
    verify_gates(core,args); check_headroom(torch.cuda,args.device)
    run = args.project/'runs'/core.EXPERIMENT
    c.require(not run.exists(), 'Refusing overwrite/restart: '+str(run))
    report = dict(status='training',variant=args.variant,seed=1,report_dir=str(out),started_at=c.now(),
                  params=config(core,args),completed_epochs=0,_clock=time.perf_counter(),
                  execution=args.execution,allocator_cap_GiB=c.CAPS_GIB[args.execution],signature=c.signature(args,core))
    write(out/'training.json',{k:v for k,v in report.items() if k!='_clock'})
    with zipfile.ZipFile(out/'actual_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in c.HERE.iterdir():
            if path.is_file() and path.suffix in ('.py','.sh','.yaml','.json','.md','.txt'):
                archive.write(path,path.name)
    trainer = trainer_class(core,report)(overrides=config(core,args))
    with chunked_assignment() as stats:
        trainer.train()
    c.require(report['completed_epochs']==200, 'Formal training did not complete 200 epochs')
    report.update(status='trained_pending_official_eval',training_finished_at=c.now(),
                  best=str(trainer.best),last=str(trainer.last),matching=stats)
    write(out/'training.json',{k:v for k,v in report.items() if k!='_clock'})
    write(out/'training_completion.json',dict(trained_epochs=200,best=str(trainer.best),last=str(trainer.last)))
    print(c.now()+' '+args.variant+' FORMAL200 COMPLETED; official val will run serially in controller', flush=True)


def main():
    global REPORT
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant',choices=c.VARIANTS,required=True)
    parser.add_argument('--mode',choices=('check','smoke','run','eval','verify-gates','local-check','local-integration'),required=True)
    parser.add_argument('--project',type=Path,default=Path('/root/autodl-tmp/yolo11_project'))
    parser.add_argument('--source',type=Path,default=Path('/root/autodl-tmp/yolo11_src'))
    parser.add_argument('--data',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019.yaml'))
    parser.add_argument('--dataset',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val'))
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--execution',choices=('serial','parallel'),default='serial')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--pair',type=Path)
    parser.add_argument('--weights',type=Path)
    args = parser.parse_args()
    REPORT = args.out
    REPORT.mkdir(parents=True,exist_ok=False)
    lock = None
    if os.name != 'nt':
        import fcntl
        lock_path = args.project/'logs'/('e60_ablation_'+args.variant+'.lock')
        lock_path.parent.mkdir(parents=True,exist_ok=True)
        lock = lock_path.open('a+')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    core = setup(args)
    write(args.project/'logs'/('latest_e60_ablation_'+args.variant+'_'+args.mode+'.json'),dict(report=str(REPORT)))
    print(c.now()+' '+args.variant+' '+args.mode+' START REPORT='+str(REPORT), flush=True)
    if args.mode in ('check','local-check'):
        c.require(args.mode=='local-check' or args.device!='cpu', 'Server check requires CUDA')
        from checks import check_all
        result = check_all(core,args,REPORT)
        write(REPORT/'check.json',result)
        if args.mode == 'check':
            write(c.gate_path(args,'check'),dict(signature=c.signature(args,core),report=str(REPORT)))
    elif args.mode == 'local-integration':
        from checks import integration
        integration(core,args,REPORT)
    elif args.mode == 'verify-gates':
        verify_gates(core,args)
    elif args.mode == 'eval':
        c.require(args.weights is not None, 'Evaluation needs --weights; never retrains')
        from evaluate import evaluate
        evaluate(args,args.weights,REPORT)
    else:
        (smoke if args.mode=='smoke' else train)(core,args,REPORT)
    print(c.now()+' '+args.variant+' '+args.mode+' FINISHED', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        if REPORT is not None:
            write(REPORT/'failure.json',dict(status='failed_or_interrupted',error=traceback.format_exc()))
        raise
