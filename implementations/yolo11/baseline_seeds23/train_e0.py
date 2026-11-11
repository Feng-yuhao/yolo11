"""Wrap frozen actual E0 seed1 code; original train() is NOT reimplemented."""
import argparse
import copy
import gc
from pathlib import Path
import subprocess
import sys
import time
import traceback

import common as c


def audit_model(model):
    from ultralytics.nn.modules.head import Detect
    import torch
    head = model.model[-1]
    c.require(type(head) is Detect and head.nc == 10 and head.nl == 3 and
              tuple(head.stride.tolist()) == (8,16,32), 'E0 must be original three-scale YOLO11s')
    c.require(not hasattr(head,'task_blocks'), 'Unexpected added branch')
    c.require(all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite E0 parameter')
    return dict(parameters=sum(p.numel() for p in model.parameters()), stride=[8,16,32],original_detect=True)


def checkpoint_unit(torch, out):
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.nn.tasks import attempt_load_one_weight
    trainer = DetectionTrainer.__new__(DetectionTrainer)
    trainer.data = dict(nc=10)
    base,_ = attempt_load_one_weight(str(c.PROJECT/'weights/yolo11s.pt'), device='cpu', fuse=False)
    model = trainer.get_model(base.yaml, base, verbose=False).eval()
    result = audit_model(model)
    images = [torch.rand(1,3,96,128), torch.rand(1,3,128,128)]
    with torch.no_grad():
        expected = [model(x)[0] for x in images]
    path=out/'serialization_fp32.pt'
    torch.save(dict(model=copy.deepcopy(model).float(), train_args={}),path)
    code = """import sys,torch
sys.path.insert(0,sys.argv[1])
from ultralytics.nn.tasks import attempt_load_one_weight
d=torch.load(sys.argv[3],map_location='cpu',weights_only=False)
torch.set_num_threads(d['threads'])
m,_=attempt_load_one_weight(sys.argv[2],device='cpu',fuse=False)
for k,v in d['state'].items(): assert torch.equal(m.state_dict()[k],v), 'Reload tensor changed: '+k
with torch.no_grad():
 for x,y in zip(d['images'],d['predictions']): torch.testing.assert_close(m(x)[0],y,rtol=0,atol=0)
 m.fuse(verbose=False)
 for x,y in zip(d['images'],d['predictions']): torch.testing.assert_close(m(x)[0],y,rtol=2e-5,atol=.002)
print('Fresh process FP32 reload/fuse passed')
"""
    torch.save(dict(images=images,predictions=expected,threads=torch.get_num_threads(),
                    state=model.state_dict()),out/'oracle.pt')
    proc = subprocess.run([sys.executable,'-c',code,str(c.SOURCE),str(path),str(out/'oracle.pt')],
                          capture_output=True,text=True,timeout=180)
    c.require(proc.returncode == 0, 'Fresh-process original E0 reload/fuse failed:\n'+proc.stdout+proc.stderr)
    print(proc.stdout,flush=True)
    return result


def barrier(out, name, seed, timeout=240):
    c.write(out/('%s_%d.json' % (name,seed)),dict(at=c.now()))
    start=time.monotonic()
    while not all((out/('%s_%d.json' % (name,s))).is_file() for s in (2,3)):
        c.require(not any((out/('FAILED_%d.json' % s)).exists() for s in (2,3)), 'Other smoke failed')
        c.require(time.monotonic()-start < timeout, 'Parallel smoke barrier timed out: '+name)
        time.sleep(1)


def smoke(torch, core, seed, out, pair):
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils import ops
    from ultralytics.nn.tasks import attempt_load_one_weight
    from ultralytics.utils.torch_utils import init_seeds
    # Engineering stress only. Neither this model nor its optimizer survives
    # into formal training. Formal uses the untouched frozen E0 core.train().
    init_seeds(seed,deterministic=True)
    c.cap(torch)
    p=c.params(seed)
    p.update(model=str(core.PRETRAINED), project=str(out), name='scratch_setup',plots=False)
    trainer=DetectionTrainer(overrides=p)
    trainer._setup_train(1)
    c.require(trainer.amp,'AMP disabled unexpectedly')
    audit_model(trainer.model)
    barrier(pair,'model_ready',seed)
    torch.cuda.reset_peak_memory_stats()
    before=trainer.model.model[-1].cv2[0][-1].weight.detach().clone()
    successful=0
    for attempt,batch in enumerate(trainer.train_loader,1):
        trainer.model.train()
        batch=trainer.preprocess_batch(batch)
        with torch.autocast('cuda',dtype=torch.float16):
            loss,items=trainer.model(batch)
        c.require(torch.isfinite(loss).all() and torch.isfinite(items).all(),'Nonfinite real loss')
        trainer.scaler.scale(loss).backward()
        trainer.scaler.unscale_(trainer.optimizer)
        good=all(p.grad is None or torch.isfinite(p.grad).all() for p in trainer.model.parameters())
        scale=float(trainer.scaler.get_scale())
        if good:
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(),10.)
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
        trainer.optimizer.zero_grad(set_to_none=True)
        if good:
            trainer.ema.update(trainer.model)
            successful+=1
        print('E0 seed%d SMOKE %d/32 finite_updates=%d/8 loss=%s scale=%g->%g reserved=%.2fGiB' %
              (seed,attempt,successful,items.tolist(),scale,trainer.scaler.get_scale(),
               torch.cuda.memory_reserved()/1024**3),flush=True)
        if successful>=8:break
        if attempt>=32:break
    c.require(successful>=8 and not torch.equal(before,trainer.model.model[-1].cv2[0][-1].weight),
              'Need eight actual finite AMP steps and a real parameter change')
    trainer.model.eval()
    ema=copy.deepcopy(trainer.ema.ema).eval().half()
    barrier(pair,'validation_ready',seed)
    with torch.no_grad():
        square=torch.zeros(32,3,640,640,device='cuda',dtype=torch.float16)
        pred=ema(square)
        c.require(torch.isfinite(pred[0]).all(),'Nonfinite batch32 square inference')
        del square,pred
        count=0
        for i,batch in enumerate(trainer.test_loader,1):
            images=batch['img'].to('cuda').half()/255
            pred=ema(images)
            c.require(torch.isfinite(pred[0]).all() and all(torch.isfinite(x).all() for x in pred[1]),
                      'Nonfinite EMA validation raw/decoded values')
            batch['img']=images
            # Same original criterion, AMP accumulation avoids FP16 sum overflow.
            with torch.autocast('cuda',dtype=torch.float16):
                total,items=ema.loss(batch,pred)
            c.require(torch.isfinite(total).all() and torch.isfinite(items).all(),'Nonfinite validation loss')
            detections=ops.non_max_suppression(pred[0].clone(),.001,.7,max_det=500)
            c.require(all(torch.isfinite(d).all() for d in detections),'Nonfinite NMS')
            count+=len(images)
            print('E0 seed%d VALIDATION SMOKE %d/%d images=%d/548 reserved=%.2fGiB' %
                  (seed,i,len(trainer.test_loader),count,torch.cuda.memory_reserved()/1024**3),flush=True)
    c.require(count==548,'Incomplete full validation smoke')
    # Keep BOTH train+optimizer+EMA residents until both full validations finish.
    barrier(pair,'validation_done',seed,timeout=600)
    c.write(out/'smoke.json',dict(status='passed',successful_updates=successful,train_batch=16,
            full_val_images=548,val_batch=32,EMA_half=True,simultaneous_models=True,
            peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3))


def formal(torch,core,seed,out,binding):
    c.verify(seed,'check',binding)
    c.verify(seed,'smoke',binding)
    pair=c.read(c.PROJECT/'comparison_reports/e0_s23_pair_gates/parallel.json')
    c.require(pair['status']=='passed' and pair['signatures'][str(seed)]==c.digest(binding),'No matching real parallel smoke')
    run_dir=c.PROJECT/'runs'/c.params(seed)['name']
    c.require(not run_dir.exists(),'Refusing overwrite/auto-increment/resume: '+str(run_dir))
    c.cap(torch)
    report=dict(status='initializing',purpose='formal200',report_dir=str(out),
                seed=seed,experiment=core.EXPERIMENT,started_at=c.now())
    original_yolo=core.YOLO
    clock=time.monotonic();previous=[0,0.]
    def yolo(*args,**kwargs):
        model=original_yolo(*args,**kwargs)
        def progress(trainer):
            epoch=trainer.epoch+1
            if epoch<=previous[0]:return
            elapsed=time.monotonic()-clock
            print('E0 seed%d epoch %d/200 %s epoch_seconds=%.1f elapsed=%.2fh ETA=%.2fh mAP50-95=%.5f' %
                  (seed,epoch,c.now(),elapsed-previous[1],elapsed/3600,elapsed/epoch*(200-epoch)/3600,
                   trainer.metrics.get('metrics/mAP50-95(B)',float('nan'))),flush=True)
            previous[:]=[epoch,elapsed]
        model.add_callback('on_fit_epoch_end',progress)
        return model
    core.YOLO=yolo
    try:
        weights=core.train(report)  # EXACT ORIGINAL CODE, same SGD/warmup/LR/selection.
    finally:
        core.YOLO=original_yolo
    result=dict(status='completed',seed=seed,completed_epochs=200,params=c.params(seed),
                weights=str(weights),checkpoint_sha256=c.sha(weights),run_dir=report['run_dir'],
                best_checkpoint_epoch=report['training']['best_checkpoint_epoch'],
                selection='original Ultralytics validation fitness; never selected by test',
                signature=binding,report_dir=str(out),finished_at=c.now())
    c.write(c.PROJECT/'comparison_reports/e0_s23_pair_gates'/('seed%d_formal_completed.json' % seed),result)
    c.write(out/'training_completion.json',result)
    print('E0 seed%d FORMAL200 COMPLETED: %s' % (seed,weights),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('check','smoke','run'))
    parser.add_argument('--seed',type=int,choices=(2,3),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--pair',type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:
        torch,core=c.setup(args.seed)
        binding=c.signature(args.seed,torch)
        c.write(args.out/'input_manifest.json',binding)
        if args.mode=='check':
            report=dict(status='check',purpose='check',report_dir=str(args.out))
            core.preflight(report,core.PRETRAINED)
            result=checkpoint_unit(torch,args.out)
            # Build label caches ONCE serially before parallel DataLoaders.
            from ultralytics.cfg import get_cfg
            from ultralytics.data import build_yolo_dataset
            from ultralytics.data.utils import check_det_dataset
            data=check_det_dataset(str(c.DATA),autodownload=False)
            for split,batch in (('train',16),('val',32)):
                dataset=build_yolo_dataset(get_cfg(overrides=c.params(args.seed)),data[split],batch,data,
                        mode=split,rect=split=='val',stride=32)
                c.require(len(dataset)==(6471 if split=='train' else 548),'Wrong warmed cache split')
                del dataset
            result['label_caches_built_serially']=True
            c.write(args.out/'checks.json',result)
        elif args.mode=='smoke':
            c.verify(args.seed,'check',binding)
            c.require(args.pair is not None,'Smoke must be launched by parallel controller')
            smoke(torch,core,args.seed,args.out,args.pair)
        else:
            formal(torch,core,args.seed,args.out,binding)
        if args.mode!='run':
            c.write(c.gates(args.seed,args.mode),dict(status='passed',signature=c.digest(binding),
                    binding=binding,report=str(args.out),finished_at=c.now()))
    except BaseException:
        error=dict(status='failed',error=traceback.format_exc(),at=c.now())
        c.write(args.out/'FAILED.json',error)
        if args.pair:c.write(args.pair/('FAILED_%d.json' % args.seed),error)
        raise


if __name__=='__main__':main()
