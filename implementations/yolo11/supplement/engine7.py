"""Seed-aware driver around byte-identical archived architectures and recipe."""
from __future__ import annotations
import copy, csv, gc, hashlib, importlib, os, shutil, sys, time, zipfile,contextlib
from pathlib import Path
from common7 import *

ARGS=None; CORE=None; HEAD=None; JOB=None; BINDING=None

@contextlib.contextmanager
def dataset_lock(args):
    import fcntl
    lock=args.project/'logs/.stmdrf_supp7_dataset.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)

def setup(args,local=False):
    global ARGS,CORE,HEAD,JOB
    ARGS=args;JOB=JOBS[args.key]
    if not local:
        require(sys.platform=='linux' and Path(sys.prefix).resolve()==PYTHON.parent.parent.resolve(),
                'Use only /root/miniconda3/envs/yolo11/bin/python, not base/yolo26')
        require(args.device=='cuda:0','Server phases require cuda:0; no silent CPU fallback')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ['YOLO_AUTOINSTALL']='false';os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    settings=args.report/'settings'/args.key;settings.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(HERE/'assets/Arial.ttf',settings/'Arial.ttf')
    os.environ['YOLO_CONFIG_DIR']=str(settings)
    frozen=HERE/'frozen'/JOB['kind']
    sys.path[:0]=[str(args.source.resolve()),str(frozen),str(HERE)]
    import torch, ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS
    # Only these new per-job settings are touched. Avoid optional third-party
    # login prompts/uploads; keep local TensorBoard and native files available.
    SETTINGS.update({k:False for k in ['sync','clearml','comet','dvc','hub','mlflow','neptune','raytune','wandb']})
    require(Path(ultralytics.__file__).resolve().parent==args.source.resolve()/'ultralytics',
            'Wrong Ultralytics source import: '+str(ultralytics.__file__))
    require(ultralytics.__version__=='8.3.0','Requires pinned Ultralytics 8.3.0')
    if not local:
        require(torch.__version__=='2.5.1+cu121' and torch.version.cuda=='12.1','Torch/CUDA differs from original YOLO11 recipe')
        import pandas, scipy, seaborn,torchvision  # fail before any expensive formal job
        require(torchvision.__version__=='0.20.1+cu121','Torchvision differs from original YOLO11 environment')
        require(torch.cuda.is_available(),'CUDA unavailable')
    package_audit()
    core=importlib.import_module('e1_640_core')
    head=importlib.import_module('tood_td_head' if JOB['kind']=='td' else 'spatial_task_head')
    if JOB['kind']!='td':head.configure(JOB['variant'])
    core.ROOT,core.SOURCE,core.DATA=args.project,args.source,args.data
    core.PRETRAINED=args.project/'weights/yolo11s.pt';core.MODEL_YAML=frozen/'e1_reference.yaml'
    core.EXPERIMENT,core.SEED=JOB['run'],JOB['seed']
    core.variant=JOB['variant'];core.torch,core.YOLO=torch,YOLO
    core.AUTO_SHUTDOWN=core.SHUTDOWN_ON_FAILURE=False
    require(sha(core.PRETRAINED)==core.PRETRAINED_SHA256,'Only the original COCO yolo11s.pt can initialize formal training')
    torch.set_num_threads(4)
    if args.device!='cpu':
        torch.cuda.set_device(args.device)
        total=torch.cuda.get_device_properties(0).total_memory
        require(args.cap_gib*2**30<total,'Allocator cap exceeds device memory')
        torch.cuda.set_per_process_memory_fraction(args.cap_gib*2**30/total,0)
    import smoke_memory
    smoke_memory.configure(args.cap_gib)
    # Native Ultralytics AMP check uses this trusted local auxiliary model only.
    # It must not initiate a network download or alter formal initialization.
    os.chdir(HERE/'assets')
    CORE,HEAD=core,head
    return core

def config(core,args):
    p=core.training_params();p.update(model=str(core.MODEL_YAML),device=args.device,workers=8)
    expected=read(HERE/'references/reference_protocol.json')['original_e60'][str(JOB['seed'])]
    ignored={'project','name','model','data','device'}
    actual={k:v for k,v in p.items() if k not in ignored}
    old={k:v for k,v in expected.items() if k not in ignored}
    require(actual==old,'Original same-seed formal recipe differs: '+str({k:(actual.get(k),old.get(k)) for k in actual.keys()|old.keys() if actual.get(k)!=old.get(k)}))
    require(p['seed']==JOB['seed'] and p['batch']==16 and p['epochs']==200,'Seed/batch/epochs changed')
    return p

def build_model(core):
    import torch
    from ultralytics.nn.tasks import DetectionModel,yaml_model_load
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(JOB['seed'],deterministic=True)
    cfg=yaml_model_load(str(core.MODEL_YAML));cfg['scale']='s'
    base=DetectionModel(copy.deepcopy(cfg),nc=10,verbose=False)
    core.assert_p2_model(base)
    audit=core.load_p2_pretrained(base)
    state=torch.get_rng_state().clone();cuda_state=torch.cuda.get_rng_state().clone() if torch.cuda.is_initialized() else None
    model=HEAD.attach(copy.deepcopy(base))
    if JOB['kind']=='routing':
        full=HEAD.full.attach(copy.deepcopy(base));HEAD.assert_common_initialization(full,model,JOB['variant']);del full
    require(torch.equal(state,torch.get_rng_state()),'Attach changed CPU training RNG')
    if cuda_state is not None:require(torch.equal(cuda_state,torch.cuda.get_rng_state()),'Attach changed CUDA training RNG')
    target=model.state_dict()
    for name,value in base.state_dict().items():require(torch.equal(value,target[name]),'Original E1 tensor changed: '+name)
    audit.update(training_seed=JOB['seed'],variant=JOB['variant'],COCO_initialization_only=True,
                 module_construction_rule='original isolated construction seeds preserved',
                 common_E1_tensors_exact=True,from_trained_best=False)
    return model,base,audit

def topology(model):
    if JOB['kind']=='td':return HEAD.assert_topology(model)
    if JOB['kind']=='routing':
        HEAD.assert_topology(model,JOB['variant'])
    else:
        h=model.model[-1]
        require(type(h) is HEAD.SpatialTaskDetect and h.experiment_variant=='e58','Wrong E58 saved head')
        require(len(h.task_blocks)==4 and not any(hasattr(b,'refinements') for b in h.task_blocks),'E58 unexpectedly has group readouts')
        require(sum(p.numel() for p in h.task_blocks.parameters())==HEAD.COUNTS['e58'],'Wrong E58 width/capacity')
    return dict(variant=JOB['variant'],seed=JOB['seed'],new_parameters=sum(p.numel() for p in HEAD.branch(model).parameters()),
                original_four_prediction_towers=True)

def source_binding(args):
    pins=read(HERE/'references/source_pins.json');result={}
    for name,expected in pins.items():
        p=args.source/'ultralytics'/name;actual=sha(p,True)
        if name!='nn/tasks.py':require(actual==expected,'Pinned source changed: '+name)
        result[name]=actual
    return result

def stat_snapshot(args,data):
    from ultralytics.data.utils import img2label_paths
    entries=[];label_digest=hashlib.sha256()
    for split,count in [('train',6471),('val',548)]:
        # Preserve the same logical images->labels mapping as native YOLO;
        # resolving a dataset symlink first can point at a different label tree.
        images=sorted(p.absolute() for p in Path(data[split]).rglob('*') if p.is_file() and p.suffix.lower() in CORE.SUFFIXES)
        require(len(images)==count,'Wrong training/val member count: '+split)
        for image,label in zip(images,img2label_paths([str(p) for p in images])):
            label=Path(label);require(label.is_file(),'Missing YOLO label: '+str(label))
            for p in [image,label]:
                stat=p.stat();entries.append(dict(path=str(p),resolved=str(p.resolve()),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
            if split=='train':label_digest.update(image.name.encode()+b'\0'+label.read_bytes()+b'\0')
    return entries,label_digest.hexdigest()

def prepare(args,out):
    from ultralytics.data.utils import check_det_dataset
    from scoring7 import audit_dataset
    import torch
    ref=read(HERE/'references/reference_protocol.json')
    require(sha(args.data)==ref['data_yaml_sha256'],'Original VisDrone YAML bytes differ')
    data=check_det_dataset(str(args.data),autodownload=False)
    names=data['names'];names=[names[i] for i in range(10)] if isinstance(names,dict) else names
    require(names==CORE.EXPECTED_CLASSES,'Wrong VisDrone ten-class order')
    entries,labels=stat_snapshot(args,data)
    write(args.report/'input_files.json',entries)
    _,val=audit_dataset(args,'val',verify_pixels=False)
    _,test=audit_dataset(args,'test-dev',verify_pixels=True)
    for seed in [1,2,3]:
        p=args.project/f'runs/e1_yolo11s_p2add_img640_seed{seed}/weights/best.pt'
        require(p.is_file() and sha(p)==ref['baseline_checkpoints'][str(seed)],'Missing/changed E1 serialization fixture: '+str(p))
    for key,job in JOBS.items():require(not (args.project/'runs'/job['run']).exists(),'Run already exists; no overwrite: '+job['run'])
    source=source_binding(args)
    p=config(CORE,args)
    prop=torch.cuda.get_device_properties(0)
    import importlib.metadata as metadata
    versions={name:metadata.version(name) for name in ['torch','torchvision','numpy','pandas','scipy','seaborn','Pillow','ultralytics-thop']}
    binding=dict(package_sha256=package_audit(),source=source,data_yaml=sha(args.data),pretrained=sha(CORE.PRETRAINED),
                 train_names_and_label_sha256=labels,input_files_sha256=sha(args.report/'input_files.json'),
                 val=val,test=test,torch=torch.__version__,cuda=torch.version.cuda,versions=versions,
                 GPU=dict(name=prop.name,total_bytes=prop.total_memory,capability=list(torch.cuda.get_device_capability())),
                 formal_recipe='Original same-seed E1/E60; 640/b16/workers8/SGD/200e; native val32',
                 tasks_py_binding_policy='Actual SHA recorded; unrelated registrations allowed; DetectionModel topology/tensors checked')
    write(args.report/'admission.json',dict(status='passed',at=now(),binding=binding,binding_id=canonical(binding),training_started=False))
    write(out/'prepare.json',dict(status='passed',binding_id=canonical(binding),GPU=binding['GPU'],training_images=6471,val_images=548,test_images=1610))
    print(now()+' ALL SEVEN INPUTS/RECIPES/DATASETS VERIFIED; no training started',flush=True)

def binding(args):
    receipt=read(args.report/'admission.json');b=receipt['binding']
    require(receipt['status']=='passed' and receipt['binding_id']==canonical(b),'Invalid admission receipt')
    require(b['package_sha256']==package_audit() and b['source']==source_binding(args),'Source/package changed after admission')
    require(b['data_yaml']==sha(args.data) and b['pretrained']==sha(CORE.PRETRAINED),'Data YAML/pretrained changed')
    require(b['input_files_sha256']==sha(args.report/'input_files.json'),'Input inventory changed')
    for e in read(args.report/'input_files.json'):
        st=Path(e['path']).stat()
        require(st.st_size==e['bytes'] and st.st_mtime_ns==e['mtime_ns'] and str(Path(e['path']).resolve())==e['resolved'],'Input changed after admission: '+e['path'])
    return dict(admission=receipt['binding_id'],key=args.key,job=JOB,params=config(CORE,args))

def rendezvous(args,name):
    if args.pair is None:return
    folder=args.pair/name;folder.mkdir(parents=True,exist_ok=True)
    write(folder/(args.key+'.json'),dict(key=args.key,at=now(),pid=os.getpid()))
    members=read(args.pair/'members.json')['keys'];deadline=time.monotonic()+1200
    while time.monotonic()<deadline:
        failures=list((args.pair/'failures').glob('*.json')) if (args.pair/'failures').exists() else []
        require(not failures,'Peer failed during pair smoke; inspect pair failures')
        if all((folder/(k+'.json')).is_file() for k in members):return
        time.sleep(1)
    raise RuntimeError('Pair smoke rendezvous timed out: '+name)

def trainer_class(core,report):
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import de_parallel
    class SupplementTrainer(DetectionTrainer):
        def get_model(self,cfg=None,weights=None,verbose=True):
            require(weights is None,'No implicit fine-tuning/resume from best.pt')
            model,base,audit=build_model(core);del base
            write(Path(report['report_dir'])/'transfer.json',audit)
            return model
        def get_validator(self):
            # Add fail-fast finite checks without changing native validation
            # precision, loss, NMS, fitness, or model selection.
            validator=super().get_validator()
            original=validator.postprocess
            def checked_postprocess(preds):
                from validation_smoke import require_finite_predictions
                require_finite_predictions(preds,Path(report['report_dir']),dict(phase='native_Trainer_validation'))
                results=original(preds)
                require(all(CORE.torch.isfinite(v).all() for v in results),'Nonfinite native NMS output')
                return results
            validator.postprocess=checked_postprocess
            return validator
        def _setup_train(self,world_size):
            # The native two-epoch fraction smoke may rebuild labels.cache.
            # Serialize setup/cache writes only, never hold this across a pair
            # rendezvous or across epochs; formal training stays concurrent.
            with dataset_lock(ARGS):super()._setup_train(world_size)
            model=de_parallel(self.model)
            covered=[p for g in self.optimizer.param_groups for p in g['params']]
            require(len(covered)==len({id(p) for p in covered}) and {id(p) for p in covered}=={id(p) for p in model.parameters()},'Optimizer does not cover every model parameter exactly once')
            require(self.device.type!='cuda' or self.amp,'Native AMP check failed; no protocol change')
            report['parameters']=sum(p.numel() for p in model.parameters())
            self._successful_steps=0
            def stepped(*unused):self._successful_steps+=1
            self._step_hook=self.optimizer.register_step_post_hook(stepped)
        def run_callbacks(self,event):
            super().run_callbacks(event)
            if event=='on_train_start' and report.get('paired_native'):
                rendezvous(ARGS,'native_models_optimizers_EMAs_ready')
            if event=='on_train_epoch_start':
                report['current_batch']=0;report['current_epoch']=self.epoch+1
            elif event=='on_train_batch_end':
                require(CORE.torch.isfinite(self.loss).all() and CORE.torch.isfinite(self.loss_items).all(),'Nonfinite native training loss')
                report['current_batch']=report.get('current_batch',0)+1
                if report['current_batch']%40==0:
                    report['heartbeat_at']=now();report['batches_per_epoch']=len(self.train_loader)
                    print(f'{now()} {ARGS.key} epoch={self.epoch+1}/{self.epochs} batch={report["current_batch"]}/{len(self.train_loader)} loss={self.loss_items.tolist()}',flush=True)
                    save_training(report)
            elif event=='on_train_epoch_end' and report.get('paired_native'):
                rendezvous(ARGS,f'native_epoch{self.epoch+1}_before_validation')
            elif event=='on_fit_epoch_end':
                if report.get('completed_epochs')==self.epoch+1:return
                elapsed=time.perf_counter()-report['_clock'];previous=report.get('elapsed_seconds',0.)
                metrics,diagnostics=native_metrics_for_json(self.metrics)
                report.update(completed_epochs=self.epoch+1,elapsed_seconds=elapsed,last_epoch_seconds=elapsed-previous,
                              current_batch=None,native_metrics=metrics,native_validation_diagnostics=diagnostics,heartbeat_at=now(),successful_optimizer_steps=self._successful_steps,
                              outputs_L1=HEAD.output_magnitudes(de_parallel(self.model)))
                ap=metrics.get('metrics/mAP50-95(B)')
                if ap is not None and ap>report.get('best_native_ap',-1):report.update(best_native_ap=ap,best_native_epoch=self.epoch+1)
                if float(self.fitness)==float(self.best_fitness):
                    report.update(val_selected_fitness=float(self.fitness),val_selected_epoch=self.epoch+1,
                                  checkpoint_selection='Unmodified native validation fitness, not test AP or best native AP alone')
                report['ETA_hours']=elapsed/(self.epoch+1)*(self.epochs-self.epoch-1)/3600
                save_training(report)
                print(f'{now()} {ARGS.key} epoch {self.epoch+1}/{self.epochs} native_AP={ap} best={report.get("best_native_ap")} epoch_seconds={elapsed-previous:.1f} ETA={report["ETA_hours"]:.2f}h',flush=True)
                free_disk(ARGS.project,2)
    return SupplementTrainer

def save_training(report):write(Path(report['report_dir'])/'training.json',{k:v for k,v in report.items() if k!='_clock'})

def check(args,out):
    b=binding(args)
    # The archived suites import these three interfaces, not the old seed1/2 drivers.
    import types
    shim=types.ModuleType('run_experiment')
    for n,v in dict(build_model=build_model,config=config,trainer_class=trainer_class,write=write,SEED=JOB['seed'],EXPERIMENT=JOB['run']).items():setattr(shim,n,v)
    sys.modules['run_experiment']=shim
    checks=importlib.import_module('checks')
    if JOB['kind']=='td':
        from td_coverage_check import install
        with install(checks):result=checks.check_all(CORE,args,out)
    else:result=checks.check_all(CORE,args,out)
    require(result['status']=='passed' and result.get('cuda_executed'),'CUDA engineering check not passed')
    result.update(key=args.key,seed=JOB['seed'],variant=JOB['variant'],formal_training_started=False)
    record_receipt(out,'check',b,result)

def branch_state(model):
    params=dict(HEAD.branch(model).named_parameters());groups=HEAD.branch_groups()
    require(all(sum(n.startswith(g) for g in groups)==1 for n in params),'Some branch parameters untracked or multiply tracked')
    require(all(any(n.startswith(g) for n in params) for g in groups),'Empty branch group')
    return params,groups

def amp_step(model,opt,scaler):
    import torch
    scaler.unscale_(opt)
    bad=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    before=float(scaler.get_scale());calls=[0]
    if not bad:torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
    hook=opt.register_step_post_hook(lambda *unused:calls.__setitem__(0,calls[0]+1))
    try:scaler.step(opt);scaler.update()
    finally:hook.remove()
    after=float(scaler.get_scale())
    require((calls[0]==0 and after<before) if bad else calls[0]==1,'Unexpected AMP step/skip')
    opt.zero_grad(set_to_none=True)
    return dict(updated=bool(calls[0]),bad_parameter_count=len(bad),scale_before=before,scale_after=after)

def real_gradient_smoke(args,out):
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset,build_dataloader
    from ultralytics.data.utils import check_det_dataset
    from p2_tal_chunked import chunked_assignment
    from validation_smoke import validation_stress
    import smoke_memory
    data=check_det_dataset(str(args.data),autodownload=False)
    model,base,_=build_model(CORE);del base
    params=config(CORE,args);model.args=get_cfg(overrides=params);model.to(args.device).train()
    # Single-job smoke creates caches first, so native pair workers only read them.
    with dataset_lock(args):
        dataset=build_yolo_dataset(model.args,data['train'],16,data,mode='train',rect=False,stride=32)
        require(len(dataset)==6471,'Wrong full training set in smoke')
        loader=build_dataloader(dataset,16,8,shuffle=True,rank=-1)
    cls=trainer_class(CORE,dict(report_dir=str(out),_clock=time.perf_counter()))
    dummy=cls.__new__(cls);dummy.args=model.args
    opt=dummy.build_optimizer(model,name='SGD',lr=.01,momentum=.937,decay=.0005,iterations=200*len(loader))
    scaler=torch.amp.GradScaler('cuda',enabled=True)
    bp,groups=branch_state(model);before={n:p.detach().clone() for n,p in bp.items()}
    seen=set();successful=0;history=[];changed=set()
    rendezvous(args,'real_models_and_optimizers_ready')
    with chunked_assignment():
        for i,batch in enumerate(loader):
            if i>=64:break
            smoke_memory.check_headroom(torch.cuda,args.device)
            batch['img']=batch['img'].to(args.device).float()/255
            with torch.autocast('cuda',dtype=torch.float16):loss,items=model(batch)
            require(torch.isfinite(loss),'Nonfinite real-data smoke loss')
            scaler.scale(loss).backward()
            active={g for g in groups if any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero()>0 for n,p in bp.items() if n.startswith(g))}
            item=amp_step(model,opt,scaler);history.append(item)
            if item['updated']:successful+=1;seen.update(active)
            changed={g for g in groups if any(not torch.equal(p,before[n]) for n,p in bp.items() if n.startswith(g))}
            write(out/'branch_updates.json',dict(successful=successful,attempts=i+1,expected=groups,gradient_groups=sorted(seen),updated_groups=sorted(changed)))
            print(f'{now()} {args.key} REAL_SMOKE batch={i+1}/64 finite_updates={successful}/8 groups={len(changed)}/{len(groups)} reserved={torch.cuda.memory_reserved()/2**30:.2f}GiB',flush=True)
            if successful>=8 and seen==changed==set(groups):break
    require(successful>=8 and seen==changed==set(groups),'Real smoke did not update all retained parameter groups')
    del before,loss
    opt.zero_grad(set_to_none=True);model.eval()
    ema=copy.deepcopy(model).eval().half();ema.criterion=None
    rendezvous(args,'real_training_models_optimizers_and_EMA_copies_ready')
    valid=validation_stress(model,args,data,params,out,ema_model=ema,keep_ema=True)
    rendezvous(args,'real_full_val32_complete_models_still_resident')
    result=dict(status='passed',successful_optimizer_updates=successful,attempts=len(history),expected_groups=groups,
                gradient_groups=sorted(seen),updated_groups=sorted(changed),validation=valid,
                training_model_optimizer_EMA_retained=True,parallel=args.pair is not None,
                precision_policy='unchanged original AMP parent; original FP32 group sidecars',engineering_SGD_per_batch=True)
    del ema,model,opt,loader,dataset;gc.collect();torch.cuda.empty_cache()
    return result

def reload_checkpoint(path,args):
    import torch
    value=torch.load(path,map_location='cpu',weights_only=False)
    require(value['train_args']['seed']==JOB['seed'] and value['train_args']['batch']==16 and value['train_args']['imgsz']==640,'Saved training seed/batch/imgsz differs')
    model=value.get('ema') if value.get('ema') is not None else value['model']
    if JOB['kind']=='routing':
        for block in HEAD.branch(model)[:2]:
            require(all(p.dtype==torch.float32 for r in block.refinements for p in r.parameters()),'Group branch was quantized in native checkpoint')
    topology(model)
    model=model.float().to(args.device).eval();model.criterion=None
    with torch.no_grad():require(torch.isfinite(model(torch.zeros(1,3,128,160,device=args.device))[0]).all(),'Native checkpoint reload inference failed')
    del model,value;gc.collect();torch.cuda.empty_cache()

def native_smoke(args,out):
    import torch
    from p2_tal_chunked import chunked_assignment
    from scoring7 import audit_dataset,predict,score_records
    p=config(CORE,args)
    overrides=dict(epochs=2,fraction=.02,nbs=16,warmup_epochs=0.,close_mosaic=0,
                   project=str(out/'native_runs'),name='smoke_only',exist_ok=False)
    p.update(overrides)
    r=dict(status='native_smoke',key=args.key,seed=JOB['seed'],purpose='engineering_only_not_formal_results',report_dir=str(out),
           paired_native=args.pair is not None,completed_epochs=0,_clock=time.perf_counter(),params=p)
    trainer=trainer_class(CORE,r)(overrides=p)
    with chunked_assignment():trainer.train()
    require(r['completed_epochs']==2 and trainer._successful_steps>=1,'Native Trainer smoke had no finite optimizer update')
    require(trainer.best.is_file() and trainer.last.is_file(),'Native Trainer failed to save best/last')
    rendezvous(args,'native_train_finalval_save_plot_complete')
    r.update(status='passed',best=str(trainer.best),last=str(trainer.last),successful_optimizer_steps=trainer._successful_steps)
    save_training(r);best=Path(trainer.best);last=Path(trainer.last)
    del trainer;gc.collect();torch.cuda.empty_cache()
    reload_checkpoint(best,args);reload_checkpoint(last,args)
    predictions={}
    for split in ['val','test-dev']:
        records,meta=audit_dataset(args,split,verify_pixels=False)
        directory=out/('scoring_smoke_'+split)
        pred=predict(args,best,records[:8],directory)
        metrics=score_records(records[:8],directory)
        predictions[split]=dict(images=8,prediction=pred,official=metrics)
    rendezvous(args,'native_reload_predict_scoring_smoke_complete')
    return dict(status='passed',epochs=2,training_fraction=.02,formal_batch=16,imgsz=640,native_validation_images=548,
                native_validation_batch=32,smoke_only_overrides=overrides,successful_optimizer_steps=r['successful_optimizer_steps'],
                native_save_last_best=True,native_final_validation=True,native_plots=True,
                both_checkpoints_reloaded=True,predict_and_official_scoring_smoke=predictions,
                boundary='Smoke checkpoint is never used for formal initialization or epoch selection')

def smoke(args,out):
    import torch
    b=binding(args);checkdir=args.report/'jobs'/args.key/'check';verify_receipt(checkdir,'check',b)
    torch.cuda.reset_peak_memory_stats()
    real=real_gradient_smoke(args,out/'real')
    native=native_smoke(args,out/'native')
    value=dict(status='passed',key=args.key,seed=JOB['seed'],variant=JOB['variant'],cuda_executed=True,
               cap_gib=args.cap_gib,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
               pair=str(args.pair) if args.pair else None,real=real,native=native)
    record_receipt(out,'smoke',b,value)
    print(now()+' '+args.key+' CHECK/REAL/NATIVE/SAVE/SCORER SMOKE PASSED',flush=True)

def train(args,out):
    import torch
    from p2_tal_chunked import chunked_assignment
    b=binding(args);verify_receipt(args.report/'jobs'/args.key/'check','check',b)
    verify_receipt(args.report/'jobs'/args.key/'smoke','smoke',b)
    plan=read(args.report/'execution_plan.json');require(plan['status']=='all_gates_passed','All seven gates must pass before ANY formal training')
    require(plan['binding']==read(args.report/'admission.json')['binding_id'],'Stale execution plan')
    if args.key in plan['parallel_jobs']:
        gate=plan['parallel_jobs'][args.key]
        value=verify_receipt(Path(gate['out']),'smoke',b)
        require(value['pair']==gate['pair'] and value['cap_gib']==args.cap_gib,'Wrong joint memory gate')
    else:require(args.cap_gib==28.,'Serial formal cap differs from tested single-job smoke')
    run=args.project/'runs'/JOB['run'];require(not run.exists(),'Formal run already exists; no automatic overwrite/resume')
    free_disk(args.project,4)
    report=dict(status='training',key=args.key,seed=JOB['seed'],variant=JOB['variant'],params=config(CORE,args),
                report_dir=str(out),binding=b,completed_epochs=0,started_at=now(),_clock=time.perf_counter(),
                cap_gib=args.cap_gib,purpose='formal200')
    save_training(report)
    with zipfile.ZipFile(out/'actual_experiment_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in list(read(HERE/'package_manifest.json')['files'])+['package_manifest.json']:z.write(HERE/name,name)
    with zipfile.ZipFile(out/'actual_ultralytics_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted((args.source/'ultralytics').rglob('*')):
            if p.is_file() and p.suffix in ['.py','.yaml','.yml']:z.write(p,p.relative_to(args.source).as_posix())
    trainer=trainer_class(CORE,report)(overrides=config(CORE,args))
    with chunked_assignment() as assignment:trainer.train()
    require(report['completed_epochs']==200,'Training did not finish 200 epochs')
    require(trainer._successful_steps>0,'No optimizer steps')
    csv_epochs(run/'results.csv')
    best,last=Path(trainer.best),Path(trainer.last)
    report.update(status='trained_pending_official_scores',finished_at=now(),best=str(best),last=str(last),
                  best_sha256=sha(best),last_sha256=sha(last),assignment=assignment)
    save_training(report)
    del trainer;gc.collect();torch.cuda.empty_cache()
    reload_checkpoint(best,args);reload_checkpoint(last,args)
    checkpoint_check(best);checkpoint_check(last)
    write(out/'training_completion.json',dict(status='completed',key=args.key,seed=JOB['seed'],variant=JOB['variant'],
          completed_epochs=200,best=str(best),last=str(last),best_sha256=sha(best),last_sha256=sha(last),
          checkpoint_reload_passed=True,binding=b,official_scores_completed=False,
          val_selected_epoch=report.get('val_selected_epoch'),val_selected_fitness=report.get('val_selected_fitness'),
          checkpoint_selection='Original native validation fitness; test not used'))
    print(now()+' '+args.key+' FORMAL200 SAVED/RELOADED; official scores remain',flush=True)
