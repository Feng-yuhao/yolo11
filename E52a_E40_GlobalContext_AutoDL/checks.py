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


def amp_step(model, optimizer, scaler):
    scaler.unscale_(optimizer)
    bad = [n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    before = float(scaler.get_scale())
    require(not bad or scaler.is_enabled(), 'Nonfinite FP32 gradients: '+str(bad[:5]))
    if not bad:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
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
    from extension import routing_audit
    from spatial_task_head import VARIANT
    return routing_audit(device,VARIANT)


def head_route_oracle(head,device):
    """Capture actual cv2/cv3 inputs; compare with independent block outputs."""
    head.eval()
    x = [torch.rand(1,c,h,w,device=device) for c,h,w in ((128,12,16),(128,6,8),(256,3,4),(512,2,2))]
    original = [v.clone() for v in x]
    expected = [head.task_blocks[i](v) if i<2 else (v,v) for i,v in enumerate(x)]
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
            if i<2:
                for section in (slice(None,64),slice(64,None)):
                    require((on[i][:,section]-off[i][:,section]).abs().sum()>0, f'Task effect missing at scale {i}')
            else:
                torch.testing.assert_close(on[i],off[i],rtol=0,atol=0)
    finally:
        head.enabled=True
        for hook in hooks: hook.remove()
    return dict(cv2_reg_cv3_cls_correct=True,P4_P5_unchanged=True,caller_list_untouched=True)


def integration(core,args,out):
    from PIL import Image
    import numpy as np
    from ultralytics.utils import yaml_save
    from run_e52 import trainer_class,config,write
    from p2_tal_chunked import chunked_assignment
    from spatial_task_head import output_magnitudes
    fixture=out/'fixture'
    for split in ('train','val'):
        (fixture/'images'/split).mkdir(parents=True)
        (fixture/'labels'/split).mkdir(parents=True)
        for i in range(4):
            Image.fromarray(np.random.default_rng(i).integers(0,255,(128,160,3),dtype=np.uint8)).save(fixture/'images'/split/f'{i}.jpg')
            (fixture/'labels'/split/f'{i}.txt').write_text('3 0.5 0.5 0.2 0.2\n')
    yaml=fixture/'data.yaml'
    yaml_save(yaml,dict(path=str(fixture.resolve()),train='images/train',val='images/val',nc=10,names=['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']))
    report=dict(report_dir=str(out),_clock=time.perf_counter())
    params=config(core,args)
    params.update(data=str(yaml),device='cpu',epochs=2,imgsz=160,batch=2,workers=0,amp=False,
                  project=str(out/'runs'),name='integration',plots=False,mosaic=0.,close_mosaic=0,warmup_epochs=0.,nbs=2)
    trainer=trainer_class(core,report)(overrides=params)
    with chunked_assignment(): trainer.train()
    require(trainer.best.is_file() and trainer.last.is_file(),'Missing Trainer checkpoints')
    for path in (trainer.best,trainer.last):
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        model=ckpt.get('ema') if ckpt.get('ema') is not None else ckpt['model']
        model.float().eval()
        require(all(v>0 for v in output_magnitudes(model)),'Untrained task output in saved EMA')
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
                four_image_rectangular_raw_official_predictor=True,fixture_metrics=official.official_evaluate(records,filtered))
    write(out/'integration.json',result)
    print('E52 STANDARD TRAINER / EMA / CHECKPOINT INTEGRATION PASSED',flush=True)


def check_all(core,args,out):
    from run_e52 import build_model,config
    from spatial_task_head import SpatialTaskDetect,branch,NEW_PARAMETERS
    from global_context import ContextC3k2,ADDED_PARAMETERS
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.loss import v8DetectionLoss
    from p2_tal_chunked import chunked_assignment,verify_equivalence
    from extension import branch_groups,compare_gradients
    with strict_math(): unit=routing_unit(args.device)
    print('E52 CHECK independent attention / pooling / gradients / half values passed',flush=True)
    model,reference,audit=build_model(core)
    require(type(model.model[-1]) is SpatialTaskDetect,'Original E40 head not preserved')
    require(type(model.model[4]) is ContextC3k2 and model.model[4].f==-1,'Incorrect context site')
    require(len(model.model)==27 and model.model[15].f==[-1,4] and
            model.model[23].f==16 and model.model[24].f==[-1,2] and
            model.model[26].f==[25,16,19,22],'Original graph/save references changed')
    require(sum(p.numel() for p in branch(model).parameters())==NEW_PARAMETERS,'New parameter count drift')
    model=model.to(args.device); reference=reference.to(args.device)
    model.args=get_cfg(overrides=config(core,args)); reference.args=copy.deepcopy(model.args)
    image=torch.rand(2,3,128,160,device=args.device)
    with strict_math():
        model.eval(); reference.eval()
        with torch.no_grad():
            for value in (image,image[:,:,:128,:128]):
                a,raw=model(value); b,base_raw=reference(value)
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                for x,y in zip(raw,base_raw): torch.testing.assert_close(x,y,rtol=0,atol=0)
        batch=dict(img=image,batch_idx=torch.tensor([0.,1.],device=args.device),
                   cls=torch.tensor([[3.],[0.]],device=args.device),
                   bboxes=torch.tensor([[.5,.5,.2,.2],[.45,.45,.1,.12]],device=args.device))
        model.train(); reference.train()
        with chunked_assignment():
            loss,items=model(batch); base,base_items=reference(batch)
            require(type(model.criterion) is v8DetectionLoss and len(items)==3,'Main criterion changed')
            torch.testing.assert_close(loss,base,rtol=0,atol=0)
            torch.testing.assert_close(items,base_items,rtol=0,atol=0)
            loss.backward(); base.backward()
        params=dict(model.named_parameters())
        for name,p in reference.named_parameters():
            if p.grad is None: require(params[name].grad is None,'Extra base gradient '+name)
            else: compare_gradients(params[name].grad,p.grad,'base.'+name)
        new=branch(model); before={n:p.detach().clone() for n,p in new.named_parameters()}
        for n,p in new.named_parameters():
            require(p.grad is not None and torch.isfinite(p.grad).all(),'Missing first gradient '+n)
            live=('.out_' in n or n=='context.output.weight')
            require(p.grad.abs().sum()>0 if live else p.grad.count_nonzero()==0,
                    'Zero-start gradient semantics failed '+n)
        optimizer=torch.optim.SGD(new.parameters(),lr=.01,weight_decay=0,momentum=0)
        optimizer.step(); model.zero_grad(set_to_none=True)
        require(model.model[4].global_context.output.weight.abs().sum()>0,'Native first output update failed')
        # Disposable amplified fixture AFTER native zero-start update. Unit audit
        # above separately proves native two-step updates for EVERY context param.
        # This fixture only tests whole-graph connectivity, NEVER initializes train.
        with torch.no_grad(): model.model[4].global_context.output.weight.normal_(0,.01)
        groups=branch_groups(); seen=set()
        with chunked_assignment():
            for step in range(12):
                loss,_=model(batch); require(torch.isfinite(loss),'Nonfinite original main loss')
                loss.backward()
                for group in groups:
                    selected=[p for n,p in new.named_parameters() if n.startswith(group)]
                    require(selected and all(p.grad is not None and torch.isfinite(p.grad).all() for p in selected),
                            'Missing group gradient '+group)
                    if any(p.grad.abs().sum()>0 for p in selected): seen.add(group)
                optimizer.step(); model.zero_grad(set_to_none=True)
                changed=all(any(not torch.equal(p,before[n]) for n,p in new.named_parameters() if n.startswith(g)) for g in groups)
                if step>=1 and seen==set(groups) and changed: break
            require(seen==set(groups) and changed,'Whole-graph loss-only branch update failed')
            empty={**batch,'batch_idx':torch.empty(0,device=args.device),'cls':torch.empty(0,1,device=args.device),
                   'bboxes':torch.empty(0,4,device=args.device)}
            require(torch.isfinite(model(empty)[0]),'Empty GT failed')
        print('E52 CHECK original loss / initial common gradients / all branch groups updated',flush=True)
        model.eval(); context=model.model[4].global_context; calls=[]
        hook=context.register_forward_pre_hook(lambda m,inputs:calls.append(tuple(inputs[0].shape)))
        try:
            with torch.no_grad(): on,raw_on=model(image)
            context.enabled=False
            with torch.no_grad(): off,raw_off=model(image)
            require(calls==[(2,256,16,20),(2,256,16,20)],'Actual model4 stride8 routing failed')
            require(all((a-b).abs().sum()>0 for a,b in zip(raw_on,raw_off)),
                    'Upstream branch not reaching all original detection scales')
        finally:
            context.enabled=True; hook.remove()
        routes=head_route_oracle(model.model[-1],args.device)
        # A nontrivial E40 checkpoint is a serialization fixture only.
        checkpoint=torch.load(args.project/'runs/e40a_spatial_task_head_img640_seed1_200e/weights/best.pt',
                              map_location='cpu',weights_only=False)
        fixture=checkpoint.get('ema') if checkpoint.get('ema') is not None else checkpoint['model']
        result=model.load_state_dict(fixture.float().state_dict(),strict=False)
        require(not result.unexpected_keys and result.missing_keys and
                all(n.startswith('model.4.global_context.') for n in result.missing_keys),'E40 fixture transfer drift')
        import parent_spatial_task_head as parent_head
        parent=parent_head.attach(copy.deepcopy(reference).cpu()).to(args.device)
        parent.load_state_dict(fixture.state_dict(),strict=True)
        del checkpoint,fixture
        model.eval(); model.criterion=None; parent.eval(); parent.criterion=None
        with torch.no_grad():
            context.enabled=False
            torch.testing.assert_close(model(image)[0],parent(image)[0],rtol=0,atol=0)
            context.enabled=True
        del parent
        path=out/'reload.pt'
        torch.save(dict(model=copy.deepcopy(model).cpu().eval(),ema=None,train_args={'task':'detect'}),path)
        restored=torch.load(path,map_location=args.device,weights_only=False)['model'].float().eval()
        with torch.no_grad():
            original=model(image)[0]; restored_value=restored(image)[0]
            torch.testing.assert_close(restored_value,original,rtol=0,atol=0)
            restored.fuse(verbose=False)
            torch.testing.assert_close(restored(image)[0],original,rtol=2e-4,atol=2e-4)
        if args.device!='cpu':
            half=torch.load(path,map_location=args.device,weights_only=False)['model'].half().eval()
            with torch.no_grad():
                value=half(image.half())[0]
                require(torch.isfinite(value).all() and value.dtype==torch.float16,'Whole-model EMA-half failure')
            del half
            model.train(); model.zero_grad(set_to_none=True)
            opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.001)
            scaler=torch.amp.GradScaler('cuda',enabled=True); updates=[]
            with chunked_assignment():
                for _ in range(12):
                    with torch.autocast('cuda',enabled=True): loss,_=model(batch)
                    require(torch.isfinite(loss),'AMP original loss forward failed')
                    scaler.scale(loss).backward(); updates.append(amp_step(model,opt,scaler))
                    if updates[-1]['updated']: break
            require(any(v['updated'] for v in updates),'AMP did not reach a finite update')
            model.criterion=None
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join((str(args.source),str(Path(__file__).resolve().parent),env.get('PYTHONPATH','')))
    code="import torch,spatial_task_head,global_context,sys; torch.set_num_threads(2); m=torch.load(sys.argv[1],map_location='cpu',weights_only=False)['model'].float().eval(); x=torch.rand(1,3,128,160); a=m(x)[0]; m.fuse(verbose=False); torch.testing.assert_close(a,m(x)[0],rtol=2e-4,atol=2e-4); print('fresh reload/fuse passed')"
    proc=subprocess.run([sys.executable,'-c',code,str(path)],env=env,capture_output=True,text=True,timeout=180)
    require(proc.returncode==0,proc.stdout+proc.stderr); print(proc.stdout.strip(),flush=True)
    from ultralytics import YOLO
    import numpy as np
    prediction=YOLO(str(path),task='detect').predict(np.zeros((128,160,3),dtype=np.uint8),
                                                  imgsz=160,device=args.device,half=False,save=False,verbose=False)
    require(len(prediction)==1,'Standard Predictor failed')
    import d27_official as official
    import d38_common as common
    official_tests=official.run_unit_tests(); geometry_tests=common.tests()
    from ultralytics.nn.tasks import yaml_model_load
    cfg=yaml_model_load(str(core.MODEL_YAML)); cfg['scale']='s'
    tal=verify_equivalence(cfg,device=args.device)
    fresh,base,_=build_model(core)
    parent=parent_head.attach(copy.deepcopy(base)).eval()
    from profiler_accounting import context_profile
    accounting=context_profile(fresh,parent,device='cpu',imgsz=640)
    require(accounting['GFLOPs640']<accounting['E40_GFLOPs640']*1.10,'Context budget exceeds +10%')
    require(sum(p.numel() for p in fresh.parameters())-sum(p.numel() for p in parent.parameters())==ADDED_PARAMETERS,
            'Parameter delta drift')
    print('E52 CHECK real640 profiler including QK/AV: '+str(accounting['GFLOPs640']),flush=True)
    return dict(status='passed',device=args.device,cuda_executed=args.device!='cpu',initialization=audit,
                original_loss_and_common_gradients_equal_at_start=True,updated_groups=groups,
                context=unit,actual_model4_stride8_graph_verified=True,all_four_scales_reached=True,
                E40_heads_unchanged=routes,activated_E40_bypass_identity=True,empty_GT=True,
                fresh_reload_fuse=True,standard_predictor=True,official=official_tests,geometry=geometry_tests,TAL=tal,
                new_parameters=NEW_PARAMETERS,added_context_parameters=ADDED_PARAMETERS,
                parameters=sum(p.numel() for p in fresh.parameters()),profile=accounting)
