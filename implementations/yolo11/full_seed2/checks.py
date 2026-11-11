"""Engineering gates: routing semantics, real main loss/updates, save, fuse, AMP."""
from __future__ import annotations
import contextlib
import copy
from itertools import islice
import os
from pathlib import Path
import subprocess
import sys
import time
import torch


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def smoke_batches(loader):
    return enumerate(islice(loader, 8))


def smoke_inference(model, image):
    require({p.device for p in model.parameters()} == {image.device}, 'Smoke model/image device mismatch')
    model.eval()
    with torch.no_grad():
        prediction = model(image[:1])[0]
    require(torch.isfinite(prediction).all(), 'Nonfinite smoke inference')
    return prediction


@contextlib.contextmanager
def strict_math():
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = old


def amp_step(model, optimizer, scaler, before_step=None):
    scaler.unscale_(optimizer)
    bad = [n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    before = float(scaler.get_scale())
    require(not bad or scaler.is_enabled(), 'Nonfinite FP32 gradients: '+str(bad[:5]))
    if not bad:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        if before_step is not None: before_step(optimizer)
    count = [0]
    def stepped(*unused):
        count[0] += 1
    hook = optimizer.register_step_post_hook(stepped)
    try:
        scaler.step(optimizer)
        scaler.update()
    finally:
        hook.remove()
    after = float(scaler.get_scale())
    require((count[0] == 0 and after < before) if bad else count[0] == 1, 'Incorrect AMP step/overflow behavior')
    optimizer.zero_grad(set_to_none=True)
    return dict(updated=bool(count[0]), scale_before=before, scale_after=after,
                status='overflow_skip' if bad else 'finite_update', bad_parameter_count=len(bad))


def routing_unit(device):
    from group_checks import group_oracle, bank_path_audit
    from precision_checks import tiny_weight_audit, checkpoint_precision_audit
    return dict(group_oracle=group_oracle(device),bank_path=bank_path_audit(device),
                tiny_AMP_weights=tiny_weight_audit(device,fixed=True),
                checkpoint_precision=checkpoint_precision_audit(device),
                spatial_router={str(cin): routing_unit_channels(device,cin) for cin in (128,256,512)})


def routing_unit_channels(device,cin):
    """Independent weighted-sum oracle, exact task routing, non-mutation, input gradients."""
    import spatial_task_head as target
    import e58_spatial_task_head as e58
    block=e58.WideTaskBlock(cin)
    if cin==128:
        block.__class__=target.GroupResidualTaskBlock
        block.refinements=torch.nn.ModuleList(target.GroupResidualEncoder() for _ in range(3))
        with torch.no_grad():
            for r in block.refinements: r.project.weight.normal_(0,.01)
    block=block.to(device).eval()
    x = torch.rand(2,cin,9,11, device=device, requires_grad=True)
    original = x.detach().clone()
    cls,reg,weights,bank = block.decompose(x)
    require(tuple(weights.shape)==(2,2,4,9,11), 'Wrong router shape')
    torch.testing.assert_close(weights.sum(2), torch.ones_like(weights[:,:,0]), rtol=1e-6,atol=1e-6)
    for task, actual in enumerate((cls, reg)):
        expected = sum(bank[d]*weights[:,task,d:d+1] for d in range(4))
        # Independent sum uses a different reduction order on CUDA. Allow only
        # FP32 roundoff here; initial E1 predictions/loss/gradients remain exact.
        torch.testing.assert_close(actual, expected, rtol=1e-6,atol=1e-6)
    # Hand-controlled logits: cls selects shallow depth, reg selects deep depth.
    with torch.no_grad():
        block.router[-1].weight.zero_()
        block.router[-1].bias.copy_(torch.tensor([20.,-20.,-20.,-20.,-20.,-20.,-20.,20.],device=device))
        block.out_cls.weight.fill_(.001)
        block.out_reg.weight.fill_(.002)
    c,r,w,bank = block.decompose(x)
    torch.testing.assert_close(c,bank[0],rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(r,bank[3],rtol=1e-6,atol=1e-6)
    xc,xr = block(x)
    torch.testing.assert_close(xc, x+block.out_cls(bank[0]),rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(xr, x+block.out_reg(bank[3]),rtol=1e-6,atol=1e-6)
    (xc.sum()+xr.sum()).backward()
    require(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0,'Input gradient detached')
    torch.testing.assert_close(x.detach(),original,rtol=0,atol=0)
    return dict(depth_softmax=True,independent_task_oracle=True,no_detach=True,no_input_mutation=True)


def head_route_oracle(head,device):
    """Capture actual cv2/cv3 inputs; compare with independent block outputs."""
    head.eval()
    x = [torch.rand(1,c,h,w,device=device) for c,h,w in ((128,12,16),(128,6,8),(256,3,4),(512,2,2))]
    original = [v.clone() for v in x]
    expected = [head.task_blocks[i](v) for i,v in enumerate(x)]
    caught = {}
    hooks = []
    for i in range(4):
        for task, tower in (('cls',head.cv3[i]),('reg',head.cv2[i])):
            def capture(module, inputs, key=(i,task)):
                caught[key] = inputs[0].detach().clone()
            hooks.append(tower.register_forward_pre_hook(capture))
    try:
        with torch.no_grad():
            _,on = head(x)
        for i in range(4):
            torch.testing.assert_close(caught[i,'cls'],expected[i][0],rtol=0,atol=0)
            torch.testing.assert_close(caught[i,'reg'],expected[i][1],rtol=0,atol=0)
            torch.testing.assert_close(x[i],original[i],rtol=0,atol=0)
        head.enabled=False
        with torch.no_grad():
            _,off=head(x)
        for i in range(4):
            torch.testing.assert_close(caught[i,'cls'],x[i],rtol=0,atol=0)
            torch.testing.assert_close(caught[i,'reg'],x[i],rtol=0,atol=0)
            for section in (slice(None,64),slice(64,None)):
                require((on[i][:,section]-off[i][:,section]).abs().sum()>0, f'Task effect missing at scale {i}')
    finally:
        head.enabled=True
        for hook in hooks: hook.remove()
    return dict(cv2_reg_cv3_cls_correct=True,all_four_levels_executed=True,caller_list_untouched=True)


def cold_start_amp_audit(core,args,out=None):
    """Actual original detection loss from nested zero init under CUDA AMP.

    Small synthetic fixture only. In particular, do NOT warm the new kernels
    with FP32 optimizer steps before testing the original AMP zero-init path.
    """
    from run_experiment import build_model,config
    from ultralytics.cfg import get_cfg
    from p2_tal_chunked import chunked_assignment
    from spatial_task_head import branch,branch_groups
    from optimizer_checks import SGDUpdateAudit
    model,reference,_=build_model(core)
    del reference
    model=model.to(args.device).train()
    model.args=get_cfg(overrides=config(core,args))
    image=torch.rand(1,3,256,320,device=args.device).repeat(4,1,1,1)
    # Coverage fixture, NOT a VisDrone scale-distribution claim: target sizes
    # overlap the broad initial DFL boxes at each pyramid level. Extremely small
    # synthetic GT may yield zero P2 positive alignment in the unchanged TAL.
    sizes=(.18,.22,.26,.36,.6,.9)
    indices=(0.,0.,0.,1.,2.,3.)
    centers=((.25,.25),(.5,.5),(.75,.75),(.5,.5),(.5,.5),(.5,.5))
    batch=dict(img=image,batch_idx=torch.tensor(indices,device=args.device),
               cls=torch.full((len(sizes),1),3.,device=args.device),
               bboxes=torch.tensor([[*c,s,s] for c,s in zip(centers,sizes)],device=args.device))
    parameters=dict(branch(model).named_parameters())
    before={n:p.detach().clone() for n,p in parameters.items()}
    groups=branch_groups(); seen=set(); successful=0; history=[]
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.001,momentum=.937)
    sgd_audit=SGDUpdateAudit(parameters,groups)
    scaler=torch.amp.GradScaler('cuda',enabled=True)
    with strict_math(),chunked_assignment():
        for attempt in range(40):
            with torch.autocast('cuda',dtype=torch.float16): loss,_=model(batch)
            require(torch.isfinite(loss),'Cold-start AMP forward failed')
            scaler.scale(loss).backward()
            active={g for g in groups if any(p.grad is not None and torch.isfinite(p.grad).all()
                    and p.grad.count_nonzero()>0 for n,p in parameters.items() if n.startswith(g))}
            step=amp_step(model,optimizer,scaler,before_step=sgd_audit.capture); history.append(step)
            if step['updated']:
                successful+=1; seen.update(active)
                sgd_audit.verify(optimizer)
            changed={g for g in groups if any(not torch.equal(p,before[n])
                     for n,p in parameters.items() if n.startswith(g))}
            if successful>=8 and seen==set(groups) and changed|sgd_audit.subulp_groups()==set(groups): break
    if seen!=set(groups):
        print('Cold AMP original E58 output projection ranges: '+str({n:dict(max=float(p.detach().abs().max()),
              nonzero_half=int(p.detach().half().count_nonzero())) for n,p in parameters.items() if '.out_' in n}),flush=True)
    # A tiny synthetic gradient is not proof of a visible FP32 update. Permit
    # ONLY independently verified <0.5-ULP rounding, not missing gradients,
    # optimizer omission or an unapplied representable update. This applies to
    # this synthetic fixture ONLY; the real-data smoke still requires 52/52
    # actual changed groups with the original criterion.
    numerical=sgd_audit.report()
    if out is not None:
        from run_experiment import write
        write(out/'cold_start_amp_audit.json',dict(attempts=len(history),successful_updates=successful,
              gradient_groups=sorted(seen),changed_groups=sorted(changed),optimizer=numerical))
    require(seen==set(groups),'Cold AMP missing gradient groups '+str(set(groups)-seen))
    require(changed|sgd_audit.subulp_groups()==set(groups),
            'Cold AMP unaccounted missing updates '+str(set(groups)-changed-sgd_audit.subulp_groups()))
    require(successful>=8,'Cold AMP insufficient successful updates')
    result=dict(synthetic_only=True,AMP_from_original_nested_zero_init=True,
                attempts=len(history),successful_updates=successful,groups_with_gradient=len(seen),
                groups_with_real_update=len(changed),expected_groups=len(groups),updates=history,
                optimizer_numerics=numerical,
                boundary='Sub-ULP synthetic updates are not counted as actual changes; real smoke still requires 52/52.')
    summary={k:result[k] for k in ('attempts','successful_updates','groups_with_gradient',
                                  'groups_with_real_update','expected_groups')}
    summary['verified_subULP_groups']=numerical['subULP_groups']
    summary['all_SGD_steps_verified']=True
    print('E60 CHECK cold-start AMP: '+str(summary),flush=True)
    return result


def check_all(core,args,out):
    from run_experiment import build_model,config
    from spatial_task_head import SpatialTaskDetect,branch,is_new,NEW_PARAMETERS,branch_groups
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.torch_utils import get_flops
    from p2_tal_chunked import chunked_assignment,verify_equivalence
    unit = routing_unit(args.device)
    model,reference,audit = build_model(core)
    require(type(model.model[-1]) is SpatialTaskDetect,'Wrong actual head')
    require(sum(p.numel() for p in branch(model).parameters())==NEW_PARAMETERS,'New parameter count drift')
    model=model.to(args.device); reference=reference.to(args.device)
    model.args=get_cfg(overrides=config(core,args)); reference.args=copy.deepcopy(model.args)
    image=torch.rand(1,3,256,320,device=args.device).repeat(4,1,1,1)
    amp_updates=[]
    with strict_math():
        model.eval(); reference.eval()
        with torch.no_grad():
            for value in (image,image[:,:,:128,:128]):
                a,am=model(value); b,bm=reference(value)
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                for aa,bb in zip(am,bm): torch.testing.assert_close(aa,bb,rtol=0,atol=0)
        print(f'{args.variant.upper()} CHECK initial raw/decoded predictions exactly equal E1 (square/rectangle)',flush=True)
        # Include small, medium and large GT so the unmodified TAL can assign
        # positives to coarse scales. Never manufacture assignments or change loss.
        batch=dict(img=image,batch_idx=torch.tensor([0.,1.,2.,3.],device=args.device),
                   cls=torch.tensor([[3.],[3.],[3.],[3.]],device=args.device),
                   bboxes=torch.tensor([[.5,.5,s,s] for s in (.08,.36,.6,.9)],device=args.device))
        model.train(); reference.train()
        with chunked_assignment():
            loss,items=model(batch); base,base_items=reference(batch)
            require(type(model.criterion) is v8DetectionLoss and len(items)==3,'Main loss changed')
            torch.testing.assert_close(loss,base,rtol=0,atol=0)
            torch.testing.assert_close(items,base_items,rtol=0,atol=0)
            loss.backward(); base.backward()
        params=dict(model.named_parameters())
        for n,p in reference.named_parameters():
            q=params[n]
            if p.grad is None: require(q.grad is None,'Extra frozen base gradient '+n)
            else: torch.testing.assert_close(q.grad,p.grad,rtol=0,atol=0)
        new=branch(model)
        before={n:p.detach().clone() for n,p in new.named_parameters()}
        for n,p in new.named_parameters():
            require(p.grad is not None and torch.isfinite(p.grad).all(),'Missing first gradient '+n)
            if '.out_' in n:
                require(p.grad.abs().sum()>0,'No first output gradient '+n)
            else:
                require(p.grad.count_nonzero()==0,'Nonzero upstream gradient at zero output init '+n)
        # Small engineering step keeps this fixed synthetic assignment fixture
        # populated across scales; it does not alter the formal SGD schedule.
        optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.001,momentum=.937)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        groups=branch_groups()
        seen_gradients=set(); actual_updates=1
        with chunked_assignment():
            # Small first upstream gradients can be below FP32 parameter ULP.
            # Require true gradients/changes across a bounded sequence, not
            # every scale having a positive regression anchor in every batch.
            for step in range(12):
                loss,_=model(batch); require(torch.isfinite(loss),'Nonfinite actual loss')
                loss.backward()
                for group in groups:
                    selected=[p for n,p in new.named_parameters() if n.startswith(group)]
                    require(selected and all(p.grad is not None and torch.isfinite(p.grad).all() for p in selected),'Missing group gradient '+group)
                    if any(p.grad.abs().sum()>0 for p in selected):
                        seen_gradients.add(group)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                actual_updates+=1
                changed=all(any(not torch.equal(p,before[n]) for n,p in new.named_parameters() if n.startswith(g)) for g in groups)
                if changed and seen_gradients==set(groups):
                    break
            require(seen_gradients==set(groups),'Missing detection-loss gradient groups '+str(set(groups)-seen_gradients))
            empty={**batch,'batch_idx':torch.empty(0,device=args.device),'cls':torch.empty(0,1,device=args.device),'bboxes':torch.empty(0,4,device=args.device)}
            empty_loss,_=model(empty); require(torch.isfinite(empty_loss),'Empty label loss failed')
        for group in groups:
            require(any(not torch.equal(p,before[n]) for n,p in new.named_parameters() if n.startswith(group)), 'No actual group update '+group)
        print(f'{args.variant.upper()} CHECK main loss/base gradients exact at start; all {len(groups)} branch groups updated',flush=True)
        routes=head_route_oracle(model.model[-1],args.device)
        # Trained E1 tensors are a disposable numeric serialization fixture only.
        ckpt=torch.load(args.project/'runs/e1_yolo11s_p2add_img640_seed2/weights/best.pt',map_location=args.device,weights_only=False)
        parent=ckpt.get('ema') if ckpt.get('ema') is not None else ckpt['model']
        result=model.load_state_dict(parent.float().state_dict(),strict=False)
        require(not result.unexpected_keys and all(is_new(n) for n in result.missing_keys),'Fixture transfer failed')
        model.eval(); model.criterion=None
        smoke_inference(model,image)
        path=out/'reload.pt'
        torch.save(dict(model=copy.deepcopy(model).cpu().eval(),ema=None,train_args={'task':'detect'}),path)
        restored=torch.load(path,map_location=args.device,weights_only=False)['model'].float().eval()
        with torch.no_grad():
            before_fuse=restored(image)[0]
            torch.testing.assert_close(before_fuse,model(image)[0],rtol=0,atol=0)
            restored.fuse(verbose=False)
            torch.testing.assert_close(restored(image)[0],before_fuse,rtol=2e-4,atol=2e-4)
        if args.device!='cpu':
            half=torch.load(path,map_location=args.device,weights_only=False)['model'].half().eval()
            with torch.no_grad():
                value=half(image.half())[0]
                require(torch.isfinite(value).all() and value.dtype==torch.float16,'Half inference dtype/value failed')
                for block in branch(half):
                    hx=torch.rand(2,block.reduce[0].in_channels,12,16,device=args.device,dtype=torch.float16)
                    actual=block(hx)
                    expected=copy.deepcopy(block).float()(hx.float())
                    for a,b in zip(actual,expected):
                        require(a.dtype==torch.float16,'Task block silently changed half dtype')
                        torch.testing.assert_close(a.float(),b,rtol=.01,atol=.002)
            del half
            model.train(); model.zero_grad(set_to_none=True)
            opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.001)
            scaler=torch.amp.GradScaler('cuda',enabled=True)
            with chunked_assignment():
                for attempt in range(32):
                    with torch.autocast(device_type='cuda',enabled=True): loss,_=model(batch)
                    require(torch.isfinite(loss),'AMP forward failed')
                    scaler.scale(loss).backward(); amp_updates.append(amp_step(model,opt,scaler))
                    print(f'{args.variant.upper()} CHECK AMP attempt {attempt+1}/32 '+str(amp_updates[-1]),flush=True)
                    if amp_updates[-1]['updated']: break
            require(any(v['updated'] for v in amp_updates),'AMP never reached a finite update: '+str(amp_updates))
            model.criterion=None
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join((str(args.source),str(Path(__file__).resolve().parent),env.get('PYTHONPATH','')))
    code="import torch,spatial_task_head,sys; torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; m=torch.load(sys.argv[1],map_location='cpu',weights_only=False)['model'].float().eval(); x=torch.rand(1,3,128,160); a=m(x)[0]; m.fuse(verbose=False); torch.testing.assert_close(a,m(x)[0],rtol=2e-4,atol=2e-4); print('fresh reload/fuse passed')"
    proc=subprocess.run([sys.executable,'-c',code,str(path)],env=env,capture_output=True,text=True,timeout=180)
    require(proc.returncode==0,proc.stdout+proc.stderr); print(proc.stdout.strip(),flush=True)
    from ultralytics import YOLO
    import numpy as np
    result=YOLO(str(path),task='detect').predict(np.zeros((128,160,3),dtype=np.uint8),imgsz=160,device=args.device,half=False,save=False,verbose=False)
    require(len(result)==1,'Standard Predictor failed')
    import d27_official as official
    import d38_common as common
    official_tests=official.run_unit_tests(); geometry_tests=common.tests()
    from ultralytics.nn.tasks import yaml_model_load
    tal_cfg=yaml_model_load(str(core.MODEL_YAML)); tal_cfg['scale']='s'
    tal=verify_equivalence(tal_cfg,device=args.device)
    fresh,base,_=build_model(core)
    fresh.eval(); base.eval()
    flops=get_flops(fresh,imgsz=640); base_flops=get_flops(base,imgsz=640)
    ratio_limit=1.80
    require(base_flops>0 and flops>base_flops and flops<=base_flops*ratio_limit,'Missing/excessive same-tool FLOPs '+str((flops,base_flops)))
    import e58_spatial_task_head as parent_head
    parent=parent_head.attach(copy.deepcopy(base))
    parent_flops=get_flops(parent,imgsz=640)
    # Conv-only analytic delta; THOP may additionally count normalization.
    delta_conv=6*(3*24*24*9+96*96)*2*(160*160+80*80)/2/1e9
    require(abs(delta_conv-4.755456)<1e-10,'Analytic group conv budget drift')
    require(4.74 <= flops-parent_flops <= 4.85,'Unexpected actual group profiler delta: '+str(flops-parent_flops))
    cold_amp=cold_start_amp_audit(core,args,out) if args.device!='cpu' else dict(cuda_executed=False)
    return dict(status='passed',device=args.device,cuda_executed=args.device!='cpu',initialization=audit,
                original_loss_and_gradients_equal_at_start=True,actual_optimizer_updates=actual_updates,
                branch_groups=groups,routing=routes,routing_unit=unit,empty_GT=True,AMP_updates=amp_updates,
                fresh_reload_fuse=True,standard_predictor=True,official=official_tests,geometry=geometry_tests,TAL=tal,
                trained_E1_fixture_only=True,new_parameters=NEW_PARAMETERS,
                parameters=sum(p.numel() for p in fresh.parameters()),GFLOPs640=flops,E1_GFLOPs640=base_flops,
                E58_GFLOPs640=parent_flops,added_conv_GFLOPs640=delta_conv,
                parent_E58_initialization_preserved=True,cold_start_AMP=cold_amp)


def integration(core,args,out):
    from PIL import Image
    import numpy as np
    from ultralytics.utils import yaml_save
    from run_experiment import trainer_class,config,write
    from p2_tal_chunked import chunked_assignment
    from spatial_task_head import output_magnitudes, SpatialTaskDetect, branch, GroupResidualTaskBlock
    fixture=out/'fixture'
    for split in ('train','val'):
        (fixture/'images'/split).mkdir(parents=True)
        (fixture/'labels'/split).mkdir(parents=True)
        for i in range(4):
            Image.fromarray(np.random.default_rng(i).integers(0,255,(128,160,3),dtype=np.uint8)).save(fixture/'images'/split/f'{i}.jpg')
            (fixture/'labels'/split/f'{i}.txt').write_text('3 0.5 0.5 0.9 0.9\n3 0.5 0.5 0.4 0.4\n0 0.15 0.15 0.08 0.09\n')
    yaml=fixture/'data.yaml'
    yaml_save(yaml,dict(path=str(fixture.resolve()),train='images/train',val='images/val',nc=10,names=['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']))
    report=dict(report_dir=str(out),_clock=time.perf_counter())
    params=config(core,args)
    params.update(data=str(yaml),device='cpu',epochs=2,imgsz=160,batch=2,workers=0,amp=False,
                  project=str(out/'runs'),name='integration',plots=False,mosaic=0.,close_mosaic=0,warmup_epochs=0.,nbs=2)
    trainer=trainer_class(core,report)(overrides=params)
    with chunked_assignment(): trainer.train()
    require(trainer.best.is_file() and trainer.last.is_file(),'Missing Trainer checkpoints')
    from ultralytics.utils.torch_utils import de_parallel
    live=de_parallel(trainer.model)
    require(all(v>0 for v in output_magnitudes(live)),'Live FP32 projection failed to learn')
    saved_magnitudes={}
    for path in (trainer.best,trainer.last):
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        model=ckpt.get('ema') if ckpt.get('ema') is not None else ckpt['model']
        require(all(p.dtype==torch.float32 for block in branch(model)[:2]
                    for r in block.refinements for p in r.parameters()),
                'New refiners quantized in Trainer EMA checkpoint')
        model.float().eval()
        require(type(model.model[-1]) is SpatialTaskDetect,'Saved head class lost')
        require(all(isinstance(b,GroupResidualTaskBlock) and len(b.refinements)==3 for b in branch(model)[:2]),'Saved new paths lost')
        magnitudes=output_magnitudes(model)
        require(all(v>0 for v in magnitudes),'Projection underflow/loss in saved EMA')
        saved_magnitudes[path.name]=magnitudes
        with torch.no_grad(): require(torch.isfinite(model(torch.rand(1,3,128,160))[0]).all(),'Saved Trainer inference failed')
    import shutil
    import d27_official as official
    from evaluate import predict_records
    of=out/'official_fixture'; (of/'images').mkdir(parents=True); (of/'annotations').mkdir()
    for i in range(4):
        shutil.copy2(fixture/'images/val'/f'{i}.jpg',of/'images'/f'{i}.jpg')
        (of/'annotations'/f'{i}.txt').write_text('64,51,32,26,1,4,0,0\n')
    count=official.EXPECTED_IMAGES
    try:
        official.EXPECTED_IMAGES=4
        records,_=official.load_dataset(of)
    finally: official.EXPECTED_IMAGES=count
    _,rows=predict_records(args,trainer.best,out,records)
    require(len(rows)==4,'Incomplete rectangular prediction fixture')
    require(all(0<=r[k]<=1 for r in rows for k in ('raw_any','raw_p2','raw_p3','raw_correct','final_any','final_correct')),'Invalid raw/final IoU')
    filtered,_=official.filter_detections_and_audit(records,official.load_predictions(out/'area_predictions.json',4))
    result=dict(status='passed',synthetic_only=True,epochs=2,standard_trainer_EMA_save_reload=True,
                live_FP32_all_projections_learned=True,saved_EMA_projection_magnitudes=saved_magnitudes,
                toy_EMA_half_projection_underflow=any(v==0 for m in saved_magnitudes.values() for v in m[8:]),
                new_FP32_refiners_preserved_in_EMA_half_checkpoint=True,
                boundary='Four-step toy integration only; not a server smoke or performance result.',
                four_image_rectangular_raw_official_predictor=True,fixture_metrics=official.official_evaluate(records,filtered))
    write(out/'integration.json',result)
    print(f'{args.variant.upper()} STANDARD TRAINER / EMA / CHECKPOINT INTEGRATION PASSED',flush=True)
