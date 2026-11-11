"""Native fixed-version trainer; only model construction, audits and safe failures are specialized."""
import copy
import math
import time
import warnings
from pathlib import Path
import numpy as np
import torch
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import TQDM
from ultralytics.utils.loss import E2ELoss
from models26 import build_pair
from e60_adapter import branch_groups, GroupResidualEncoder
from protocol import write, now, options


def finite(value,where):
    if isinstance(value,torch.Tensor):
        if not torch.isfinite(value).all(): raise RuntimeError('Nonfinite '+where)
    elif isinstance(value,dict):
        for key,item in value.items(): finite(item,where+'/'+str(key))
    elif isinstance(value,(tuple,list)):
        for i,item in enumerate(value): finite(item,where+'/'+str(i))


class CheckedValidator(DetectionValidator):
    def get_model(self,model,trainer=None):
        value=super().get_model(model,trainer)
        if trainer is not None:
            original_loss=value.loss
            def checked_loss(batch,preds=None):
                finite(preds,'validation predictions')
                result=original_loss(batch,preds)
                finite(result,'validation loss')
                return result
            self._original_loss=(value,original_loss)
            value.loss=checked_loss
        return value

    def __call__(self,*args,**kwargs):
        try:
            result=super().__call__(*args,**kwargs)
            if result is not None and not all(math.isfinite(float(v)) for v in result.values()):
                raise RuntimeError('Nonfinite native validation metrics')
            return result
        finally:
            if hasattr(self,'_original_loss'):
                value,loss=self._original_loss
                del value.loss  # restore class method, not a bound method stored in future checkpoint
                del self._original_loss


class PairedTrainer(DetectionTrainer):
    def __init__(self,task_args,variant,report,overrides):
        self.task_args=task_args; self.variant=variant; self.report=Path(report)
        self.real_smoke=False
        super().__init__(overrides=overrides)
        self.add_callback('on_train_start',self.audit_setup)
        self.add_callback('on_train_epoch_start',self.epoch_start)
        self.add_callback('on_train_batch_end',self.batch_end)
        self.add_callback('on_fit_epoch_end',self.epoch_end)

    def get_model(self,cfg=None,weights=None,verbose=True):
        baseline,enhanced,audit=build_pair(self.task_args.project/'weights/yolo26s.pt')
        result=baseline if self.variant=='baseline' else enhanced
        del baseline,enhanced
        write(self.report/'initialization.json',audit)
        return result

    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        expected=16 if mode=='train' else 32
        if batch_size!=expected: raise RuntimeError('Automatic batch reduction is forbidden')
        return super().get_dataloader(dataset_path,batch_size,rank,mode)

    def _build_train_pipeline(self):
        if self.batch_size!=16 or self.args.batch!=16:
            raise RuntimeError('OOM changed batch size; stopping, not retraining with a smaller batch')
        return super()._build_train_pipeline()

    def get_validator(self):
        return CheckedValidator(self.test_loader,save_dir=self.save_dir,args=copy.copy(self.args),_callbacks=self.callbacks)

    def _handle_nan_recovery(self,epoch):
        finite(self.loss,'train loss'); finite(self.loss_items,'train loss items')
        if self.fitness is not None and not math.isfinite(float(self.fitness)):
            raise RuntimeError('Nonfinite fitness; no silent checkpoint recovery')
        return False

    def save_model(self):
        finite(self.model.state_dict(),'model before save')
        finite(self.ema.ema.state_dict(),'EMA before save')
        snapshot=copy.deepcopy(self.ema.ema).half()
        finite(snapshot.state_dict(),'serialized FP16 model before native save')
        del snapshot
        return super().save_model()

    def audit_setup(self,trainer=None):
        if not self.task_args.local_test:
            if len(self.train_loader.dataset)!=6471 or len(self.test_loader.dataset)!=548:
                raise RuntimeError('Trainer actually loaded a different dataset split')
        if not self.amp and self.device.type=='cuda': raise RuntimeError('AMP unexpectedly disabled')
        if getattr(self.model.model[-1],'one2one_cv2',None) is None or self.model.end2end:
            raise RuntimeError('Native dual training/NMS inference topology changed')
        assert self.test_loader.batch_size==32 and self.train_loader.batch_size==16
        write(self.report/'trainer_setup.json',dict(variant=self.variant,train_images=len(self.train_loader.dataset),
              val_images=len(self.test_loader.dataset),epochs=self.epochs,train_batch=16,val_batch=32,
              amp=self.amp,layout='serial',native_trainer=True,original_tal=True,original_dual_loss=True,
              nms=True,args=vars(self.args)))
        self.started=time.monotonic()

    def epoch_start(self,trainer=None):
        self.epoch_started=time.monotonic()
        self.batch_count=0

    def batch_end(self,trainer=None):
        finite(self.loss,'train loss'); finite(self.loss_items,'train loss items')
        self.batch_count+=1
        if self.batch_count%40==0:
            print(f'{now()} {self.variant.upper()} epoch={self.epoch+1}/{self.epochs} '
                  f'batch={self.batch_count}/{len(self.train_loader)} '
                  f'loss='+str({k:float(v) for k,v in self.loss_items.items()}),flush=True)

    def epoch_end(self,trainer=None):
        epoch=min(self.epoch+1,self.epochs)
        elapsed=time.monotonic()-self.started
        secs=time.monotonic()-self.epoch_started
        score=self.metrics.get('metrics/mAP50-95(B)') if self.metrics else None
        payload=dict(variant=self.variant,epoch=epoch,epochs=self.epochs,epoch_seconds=secs,
                     elapsed_hours=elapsed/3600,remaining_hours=elapsed/max(epoch,1)*(self.epochs-epoch)/3600,
                     mAP50_95=score,at=now(),status='training',final_best_validation=self.epoch>=self.epochs)
        criterion=getattr(self.model,'criterion',None)
        if isinstance(criterion,E2ELoss):
            payload['native_progressive_loss']=dict(one2many_gain=criterion.o2m,one2one_gain=criterion.o2o,
                                                    epoch_updates=criterion.updates)
        write(self.report/'progress.json',payload)
        print(f'{now()} {self.variant.upper()} epoch {epoch}/{self.epochs} mAP50-95={score} '
              f'epoch_seconds={secs:.1f} elapsed={elapsed/3600:.2f}h ETA={payload["remaining_hours"]:.2f}h',flush=True)

    def optimizer_step(self):
        if not self.real_smoke: return super().optimizer_step()
        # Same native CUDA step/clip/zero/EMA order, with read-only step auditing.
        self.scaler.unscale_(self.optimizer)
        params=dict(self.model.named_parameters())
        finite_grads=all(torch.isfinite(p.grad).all() for p in params.values() if p.grad is not None)
        observed=set()
        if finite_grads:
            observed={group for group in self.groups if any(p.grad is not None and torch.count_nonzero(p.grad).item()>0
                      for name,p in params.items() if name.startswith(group))}
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=10.)
        if finite_grads and not torch.isfinite(norm):
            raise RuntimeError('Finite elements but nonfinite gradient norm in smoke')
        before=float(self.scaler.get_scale())
        self.scaler.step(self.optimizer); self.scaler.update()
        after=float(self.scaler.get_scale())
        updated=finite_grads and after>=before
        if updated:
            self.successful += 1; self.grad_groups |= observed
            self.changed_groups |= {group for group in self.groups if any(not torch.equal(p.detach(),self.initial[name])
                                    for name,p in params.items() if name.startswith(group))}
        self.optimizer.zero_grad()
        if self.ema: self.ema.update(self.model)
        finite(self.model.state_dict(),'smoke parameters after step')
        self.step_records.append(dict(updated=bool(updated),scale_before=before,scale_after=after,
                                 finite_gradients=bool(finite_grads),observed_groups=len(observed)))


def quiet_progress():
    # Logging-only override, no iterator, data or numerical changes.
    import ultralytics.engine.trainer as trainer_module
    import ultralytics.engine.validator as validator_module
    class QuietTQDM(TQDM):
        def __init__(self,*args,**kwargs):
            kwargs['disable']=True
            super().__init__(*args,**kwargs)
    trainer_module.TQDM=QuietTQDM
    validator_module.TQDM=QuietTQDM


def make_trainer(args,variant,out,name):
    opts=options(args,name)
    if args.local_test:
        opts.update(imgsz=args.local_size,workers=0,plots=False,epochs=args.local_epochs,close_mosaic=0)
    trainer=PairedTrainer(args,variant,out,opts)
    return trainer


def smoke(args,variant,out):
    trainer=make_trainer(args,variant,out,'yolo26_smoke_'+variant+'_'+out.parent.name)
    try:
        trainer._setup_train(); trainer.epoch=trainer.start_epoch
        trainer.audit_setup()
        trainer._model_train(); trainer.real_smoke=True
        trainer.groups=branch_groups() if variant=='e60' else []
        trainer.initial={n:p.detach().clone() for n,p in trainer.model.named_parameters() if any(n.startswith(g) for g in trainer.groups)}
        trainer.grad_groups=set(); trainer.changed_groups=set(); trainer.successful=0; trainer.step_records=[]
        trainer.optimizer.zero_grad()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); trainer.scheduler.step()
        nw=trainer._get_warmup_iterations(len(trainer.train_loader)); last=-1; records=[]
        iterator=iter(trainer.train_loader)
        for ni in range(128):
            try: batch=next(iterator)
            except StopIteration: iterator=iter(trainer.train_loader); batch=next(iterator)
            if ni<nw:
                trainer.accumulate=max(1,int(np.interp(ni,[0,nw],[1,trainer.args.nbs/trainer.batch_size]).round()))
                for group in trainer.optimizer.param_groups:
                    group['lr']=float(np.interp(ni,[0,nw],[trainer.args.warmup_bias_lr if group.get('param_group')=='bias' else 0.,group['initial_lr']*trainer.lf(0)]))
                    if 'momentum' in group: group['momentum']=float(np.interp(ni,[0,nw],[trainer.args.warmup_momentum,trainer.args.momentum]))
            with torch.autocast('cuda',enabled=trainer.amp):
                batch=trainer.preprocess_batch(batch)
                trainer.loss,trainer.loss_items=trainer.model(batch)
                trainer.loss=trainer.loss.sum()
            finite((trainer.loss,trainer.loss_items),'real smoke loss')
            if not trainer.loss_names: trainer.loss_names=tuple(trainer.loss_items)
            trainer.scaler.scale(trainer.loss).backward()
            if ni-last>=trainer.accumulate:
                trainer.optimizer_step(); last=ni
            item=dict(attempt=ni+1,successful=trainer.successful,grad_groups=len(trainer.grad_groups),
                      updated_groups=len(trainer.changed_groups),loss={k:float(v) for k,v in trainer.loss_items.items()},
                      shape=list(batch['img'].shape),reserved_GiB=torch.cuda.memory_reserved()/2**30)
            records.append(item); write(out/'smoke_steps.json',dict(batches=records,steps=trainer.step_records))
            print(f'{now()} {variant.upper()} SMOKE {item}',flush=True)
            if trainer.successful>=8 and len(trainer.grad_groups)==len(trainer.groups) and len(trainer.changed_groups)==len(trainer.groups): break
        else: raise RuntimeError('Real smoke did not reach all finite gradient/actual-update requirements; no bypass')
        del batch
        if not isinstance(trainer.model.criterion,E2ELoss): raise RuntimeError('Wrong original YOLO26 loss')
        # Retain model+optimizer+EMA while validating, exactly as native Trainer.
        trainer.ema.update_attr(trainer.model,include=['yaml','nc','args','names','stride','class_weights'])
        square=torch.zeros(32,3,trainer.args.imgsz,trainer.args.imgsz,device=trainer.device,dtype=torch.float16)
        with torch.no_grad(),torch.autocast('cuda',enabled=trainer.amp): finite(trainer.ema.ema.eval()(square),'square validation batch32')
        del square
        trainer.validator.args.plots=False
        metrics=trainer.validator(trainer)
        finite(trainer.ema.ema.state_dict(),'EMA after complete native validation')
        checked_images=trainer.validator.seen
        if checked_images!=len(trainer.test_loader.dataset): raise RuntimeError('Incomplete full validation smoke')
        checkpoint=out/'smoke_checkpoint.pt'
        snapshot=copy.deepcopy(trainer.ema.ema).cpu().half()
        finite(snapshot.state_dict(),'half-save smoke snapshot')
        snapshot.criterion=None
        torch.save(dict(model=snapshot,ema=None,train_args=vars(trainer.args),epoch=0),checkpoint)
        del snapshot
        from scoring26 import predict
        import d27_official as official
        original=official.EXPECTED_IMAGES
        if args.local_test: official.EXPECTED_IMAGES=len(trainer.test_loader.dataset)
        val_records,_=official.load_dataset(args.val_dataset)
        prediction_summary=predict(args,variant,checkpoint,out/'prediction_smoke',val_records[:8])
        official.EXPECTED_IMAGES=original
        result=dict(status='passed',variant=variant,local_fixture=args.local_test,successful_updates=trainer.successful,
                    gradient_groups=len(trainer.grad_groups),actual_update_groups=len(trainer.changed_groups),
                    train_batch=16,validation_batch=32,imgsz=trainer.args.imgsz,validation_images=checked_images,
                    native_validator_complete=True,validation_metrics=metrics,square_batch32=True,
                    original_progressive_dual_loss=True,validation_precision='native FP32 EMA under autocast',
                    standalone_checkpoint_prediction=prediction_summary,
                    max_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,finished_at=now())
        write(out/'smoke.json',result)
        print(f'{now()} {variant.upper()} REAL SMOKE PASSED',flush=True)
        return result
    finally:
        for name in ('train_loader','test_loader'):
            loader=getattr(trainer,name,None)
            if hasattr(loader,'close'): loader.close()


def train(args,variant,out):
    trainer=make_trainer(args,variant,out,run_name_for(args,variant))
    trainer.train()
    import csv
    with trainer.csv.open() as handle: rows=list(csv.DictReader(handle))
    if len(rows)!=trainer.epochs or not trainer.best.is_file() or not trainer.last.is_file():
        raise RuntimeError('Formal training did not complete all epochs and checkpoints')
    value=dict(status='training_completed',variant=variant,epochs=trainer.epochs,run=str(trainer.save_dir),
               best=str(trainer.best),best_sha256=__import__('common').sha256(trainer.best),
               last=str(trainer.last),finished_at=now(),official_scoring_completed=False)
    write(out/'training_completion.json',value)
    return value


def run_name_for(args,variant):
    from protocol import run_name
    return run_name(variant) if not args.local_test else 'local_integration_'+variant+'_'+str(args.local_epochs)+'e'
