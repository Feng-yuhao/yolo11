"""Independent TD oracles, original loss/TAL, all24 groups, EMA/save/fuse."""
import ast
import contextlib
import copy
from pathlib import Path
import subprocess
import sys
import time
import os
import torch
from shutdown_safety import write, sha

HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise RuntimeError(message)


@contextlib.contextmanager
def strict_math():
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
           torch.backends.cudnn.benchmark)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, \
            torch.backends.cudnn.benchmark = old


def amp_step(model, optimizer, scaler):
    scaler.unscale_(optimizer)
    bad = [n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    before = float(scaler.get_scale())
    require(not bad or scaler.is_enabled(), 'Nonfinite FP32 gradient: ' + str(bad[:5]))
    if not bad:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
    calls = [0]
    def record(*unused):
        calls[0] += 1
    hook = optimizer.register_step_post_hook(record)
    try:
        scaler.step(optimizer); scaler.update()
    finally:
        hook.remove()
    after = float(scaler.get_scale())
    require((calls[0] == 0 and after < before) if bad else calls[0] == 1,
            'AMP skipped/stepped behavior inconsistent with gradient finiteness')
    optimizer.zero_grad(set_to_none=True)
    return dict(updated=bool(calls[0]), scale_before=before, scale_after=after,
                status='overflow_skip' if bad else 'finite_update', bad_parameter_count=len(bad))


def td_oracles(device):
    """Double-precision forward AND gradients vs author's class and explicit conv."""
    import torch.nn.functional as F
    from tood_td_head import TaskDecomposition, ConvGNReLU
    tree = ast.parse((HERE / 'author_source/tood_head.py').read_text(encoding='utf-8'))
    original = next(v for v in tree.body if isinstance(v, ast.ClassDef) and v.name == 'TaskDecomposition')
    # Execute ONLY the pinned TD class, not imports/full detector/MMDetection.
    def conv_module(cin, cout, kernel, **unused):
        return ConvGNReLU(cin, cout, kernel)
    def normal_init(layer, std):
        torch.nn.init.normal_(layer.weight, std=std)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)
    scope = dict(torch=torch, nn=torch.nn, F=F, ConvModule=conv_module, normal_init=normal_init)
    exec(compile(ast.Module(body=[original], type_ignores=[]), '<pinned-author-TD>', 'exec'), scope)
    ours = TaskDecomposition().to(device).double()
    author = scope['TaskDecomposition'](64,6,48,norm_cfg={'type':'GN','num_groups':32}).to(device).double()
    author.load_state_dict(ours.state_dict(), strict=True)
    x = torch.randn(2,384,5,7,device=device,dtype=torch.float64,requires_grad=True)
    xa, xb = x.detach().clone().requires_grad_(True), x.detach().clone().requires_grad_(True)
    actual = ours(x)
    expected_author = author(xa)
    gates = ours.layer_weights(xb)
    weighted = (xb.reshape(2,6,64,5,7) * gates.reshape(2,6,1,1,1)).reshape(2,384,5,7)
    expected_conv = F.relu(F.group_norm(F.conv2d(weighted,ours.reduction_conv.conv.weight),32,
                          ours.reduction_conv.norm.weight,ours.reduction_conv.norm.bias))
    torch.testing.assert_close(actual,expected_author,rtol=1e-11,atol=1e-11)
    torch.testing.assert_close(actual,expected_conv,rtol=1e-10,atol=1e-10)
    g = torch.randn_like(actual)
    ga = torch.autograd.grad((actual*g).sum(),[x]+list(ours.parameters()),retain_graph=True)
    gb = torch.autograd.grad((expected_author*g).sum(),[xa]+list(author.parameters()))
    gc = torch.autograd.grad((expected_conv*g).sum(),[xb]+list(ours.parameters()))
    for a,b,c in zip(ga,gb,gc):
        torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)
        torch.testing.assert_close(a,c,rtol=1e-9,atol=1e-10)
    require(gates.shape == (2,6,1,1) and bool(((gates>0)&(gates<1)).all()), 'Not global sigmoid gates')
    require(not torch.allclose(gates.sum(1),torch.ones_like(gates[:,0]),atol=.01),
            'TD incorrectly normalizes sigmoid gates to depth-softmax')
    # Same per-channel image averages, different positions -> same layer gates.
    permuted = x.detach().flatten(2).flip(2).reshape_as(x)
    torch.testing.assert_close(ours.layer_weights(x.detach()),ours.layer_weights(permuted),rtol=1e-12,atol=1e-12)
    return dict(pinned_author_TD_forward_gradients=True, explicit_gated_conv_forward_gradients=True,
                global_sigmoid_not_softmax=True, spatial_permutation_invariant_gates=True,
                author_commit='1bda27c6f290c5b828c566eb95ea50fed75a53cb')


def fixture_batch(device):
    image = torch.rand(4,3,128,160,device=device)
    sizes = (.15,.25,.45,.7,.9,.35)
    return dict(img=image, batch_idx=torch.tensor([0.,0.,1.,2.,3.,3.],device=device),
                cls=torch.full((6,1),3.,device=device),
                bboxes=torch.tensor([[.5,.5,s,s] for s in sizes],device=device))


def group_state(model):
    from tood_td_head import branch, branch_groups
    parameters = dict(branch(model).named_parameters())
    groups = branch_groups()
    require(len(groups) == 24 and all(any(n.startswith(g) for n in parameters) for g in groups),
            'Expected exactly24 nonempty TD parameter groups')
    require(all(sum(n.startswith(g) for g in groups) == 1 for n in parameters), 'Some TD parameters untracked/duplicated')
    return parameters, groups


def group_gradients(parameters, groups):
    return {g for g in groups if any(p.grad is not None and torch.isfinite(p.grad).all() and
            p.grad.detach().abs().sum()>0 for n,p in parameters.items() if n.startswith(g))}


def changed_groups(parameters, before, groups):
    return {g for g in groups if any(not torch.equal(p,before[n]) for n,p in parameters.items() if n.startswith(g))}


def synthetic_updates(core, args, out, amp):
    from run_experiment import build_model, config
    from ultralytics.cfg import get_cfg
    from p2_tal_chunked import chunked_assignment
    model, base, _ = build_model(core); del base
    model = model.to(args.device).train(); model.args = get_cfg(overrides=config(core,args))
    parameters, groups = group_state(model)
    before = {n:p.detach().clone() for n,p in parameters.items()}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01,momentum=.937)
    scaler = torch.amp.GradScaler('cuda',enabled=amp)
    history, seen, successful = [], set(), 0
    batch = fixture_batch(args.device)
    with strict_math(), chunked_assignment():
        for attempt in range(40):
            with torch.autocast(device_type='cuda' if amp else 'cpu',enabled=amp):
                loss, items = model(batch)
            require(bool(torch.isfinite(loss)), 'Nonfinite synthetic detection loss')
            scaler.scale(loss).backward()
            grads = group_gradients(parameters,groups)
            update = amp_step(model,optimizer,scaler); history.append(update)
            if update['updated']:
                successful += 1; seen.update(grads)
            changed = changed_groups(parameters,before,groups)
            print(f'TD CHECK AMP={amp} attempt={attempt+1} successful={successful} '
                  f'grad_groups={len(seen)}/24 updated_groups={len(changed)}/24',flush=True)
            write(out / ('synthetic_AMP' + str(amp) + '.json'), dict(successful=successful,history=history,
                  gradient_groups=sorted(seen),changed_groups=sorted(changed),local_or_synthetic_only=True))
            if successful >= 8 and seen == changed == set(groups):
                break
    require(successful >= 8 and seen == changed == set(groups),
            'Synthetic TD group evidence incomplete: gradients=' + str(set(groups)-seen) +
            ', updates=' + str(set(groups)-changed))
    model.criterion = None
    return model, dict(successful_optimizer_updates=successful,gradient_groups=sorted(seen),
                       updated_groups=sorted(changed),AMP=amp,attempts=len(history))


def profile(model, base, device):
    """Count executed Conv/Linear MACs + dynamic TD bmm (THOP misses this)."""
    from tood_td_head import TaskDecomposition, WIDTH, DEPTHS, CHANNELS
    def count(network):
        counters = dict(conv_linear=0,TD_bmm=0); hooks=[]
        def conv(layer, inputs, output):
            counters['conv_linear'] += output.numel() * layer.in_channels // layer.groups * layer.kernel_size[0]*layer.kernel_size[1]
        def linear(layer, inputs, output):
            counters['conv_linear'] += output.numel()*layer.in_features
        def td(layer, inputs, output):
            b,_,h,w = inputs[0].shape
            counters['TD_bmm'] += b*h*w*layer.feat_channels*layer.in_channels
        for module in network.modules():
            if isinstance(module,torch.nn.Conv2d): hooks.append(module.register_forward_hook(conv))
            elif isinstance(module,torch.nn.Linear): hooks.append(module.register_forward_hook(linear))
            elif isinstance(module,TaskDecomposition): hooks.append(module.register_forward_hook(td))
        try:
            with torch.no_grad(): network.eval()(torch.zeros(1,3,640,640,device=device))
        finally:
            for hook in hooks: hook.remove()
        return {**counters,'GMACs':sum(counters.values())/1e9}
    actual, reference = count(model), count(base)
    # First adapters, six shared 3x3 applications at every level, two dense TD
    # bmm reductions, two output projections and the two pooled attention MLPs.
    expected = sum((3*cin*WIDTH + (9*DEPTHS+2*DEPTHS)*WIDTH**2)*hw**2 +
                   2*(DEPTHS*WIDTH*(WIDTH//8)+(WIDTH//8)*DEPTHS)
                   for cin,hw in zip(CHANNELS,(160,80,40,20)))
    require(abs((actual['GMACs']-reference['GMACs'])*1e9-expected)<1., 'TD dynamic compute count drift')
    return dict(TD_adapt=actual,E1=reference,added_GMACs=expected/1e9,
                method='executed Conv2d/Linear + explicit dynamic TD bmm, batch1 unfused FP32 640',
                not_full_operator_FLOPs=True,
                excluded='GN/ReLU/sigmoid/pooling/gate multiply/add/decode/NMS; report as MACs, not complete FLOPs')


def actual_tower_routing(model, device):
    head = model.model[-1].eval()
    x = [torch.rand(2,c,h,w,device=device) for c,h,w in
         ((128,12,16),(128,6,8),(256,3,4),(512,2,2))]
    original = [v.clone() for v in x]
    caught, hooks = {}, []
    with torch.no_grad():
        expected = [head.td_adapter(v,i) for i,v in enumerate(x)]
    for i in range(4):
        for task,tower in (('cls',head.cv3[i]),('reg',head.cv2[i])):
            def capture(module, inputs, key=(i,task)):
                caught[key] = inputs[0].detach().clone()
            hooks.append(tower.register_forward_pre_hook(capture))
    try:
        with torch.no_grad():
            head(x)
        for i in range(4):
            torch.testing.assert_close(caught[i,'cls'],expected[i][0],rtol=0,atol=0)
            torch.testing.assert_close(caught[i,'reg'],expected[i][1],rtol=0,atol=0)
            torch.testing.assert_close(x[i],original[i],rtol=0,atol=0)
        head.td_enabled = False
        with torch.no_grad():
            head(x)
        for i in range(4):
            torch.testing.assert_close(caught[i,'cls'],x[i],rtol=0,atol=0)
            torch.testing.assert_close(caught[i,'reg'],x[i],rtol=0,atol=0)
    finally:
        head.td_enabled = True
        for hook in hooks:
            hook.remove()
    return dict(actual_cv2_reg_cv3_cls_inputs_correct=True,caller_features_not_mutated=True,
                four_levels_reached=True,bypass_returns_original_features=True)


def check_all(core, args, out):
    from run_experiment import build_model, config
    from p2_tal_chunked import chunked_assignment, verify_equivalence, assert_patch_restoration
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import yaml_model_load
    from tood_td_head import NEW_PREFIX, assert_topology, branch
    unit = td_oracles(args.device)
    model, base, audit = build_model(core)
    model, base = model.to(args.device), base.to(args.device)
    with strict_math():
        model.eval(); base.eval()
        with torch.no_grad():
            for shape in ((1,3,128,128),(1,3,128,160)):
                x = torch.rand(*shape,device=args.device)
                a, b = model(x), base(x)
                for left,right in zip([a[0]]+a[1],[b[0]]+b[1]):
                    torch.testing.assert_close(left,right,rtol=0,atol=0)
        model.train(); base.train()
        model.args = base.args = get_cfg(overrides=config(core,args))
        batch = fixture_batch(args.device)
        with chunked_assignment():
            la,ia = model(batch); lb,ib = base(batch)
            torch.testing.assert_close(la,lb,rtol=0,atol=0)
            torch.testing.assert_close(ia,ib,rtol=0,atol=0)
            la.backward(); lb.backward()
            target = dict(model.named_parameters())
            for name,p in base.named_parameters():
                q = target[name]
                require((p.grad is None) == (q.grad is None), 'Initial gradient presence differs: ' + name)
                if p.grad is not None:
                    torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
            model.zero_grad(set_to_none=True)
            empty = dict(batch,batch_idx=torch.zeros(0,device=args.device),
                         cls=torch.zeros(0,1,device=args.device), bboxes=torch.zeros(0,4,device=args.device))
            require(torch.isfinite(model(empty)[0]), 'All-empty GT loss failed')
    print('TD initial raw/decoded/loss/common gradients EXACTLY equal E1; empty GT passed',flush=True)
    topology = assert_topology(model)
    counts = profile(copy.deepcopy(model).eval(),copy.deepcopy(base).eval(),args.device)
    del model, base
    learned, fp32 = synthetic_updates(core,args,out,False)
    if args.device != 'cpu':
        del learned
        learned, amp = synthetic_updates(core,args,out,True)
    else:
        amp = dict(cuda_executed=False)
    routing = actual_tower_routing(learned,args.device)
    # Test learned nonzero TD paths while using the trained E1's common buffers
    # as a numeric serialization fixture; NEVER used in formal initialization.
    fixture = args.project / 'runs/e1_yolo11s_p2add_img640_seed1/weights/best.pt'
    if fixture.is_file():
        from evaluation import E1_SHA
        require(sha(fixture) == E1_SHA, 'E1 serialization fixture SHA differs')
        ckpt = torch.load(fixture,map_location='cpu',weights_only=False)
        old = ckpt['ema'] if ckpt.get('ema') is not None else ckpt['model']
        missing = learned.load_state_dict(old.float().state_dict(),strict=False)
        require(not missing.unexpected_keys and all(n.startswith(NEW_PREFIX) for n in missing.missing_keys),
                'Trained common numeric fixture load is invalid')
        del old, ckpt
    learned = learned.eval(); learned.criterion = None
    x = torch.rand(1,3,128,160,device=args.device)
    checkpoint = out / 'reload.pt'
    torch.save(dict(model=copy.deepcopy(learned).cpu().eval(),ema=None,train_args={'task':'detect'}),checkpoint)
    restored = torch.load(checkpoint,map_location=args.device,weights_only=False)['model'].float().eval()
    with strict_math(),torch.no_grad():
        before = learned(x)[0]
        torch.testing.assert_close(before,restored(x)[0],rtol=0,atol=0)
        restored.fuse(verbose=False)
        torch.testing.assert_close(before,restored(x)[0],rtol=3e-4,atol=3e-4)
        if args.device != 'cpu':
            half = torch.load(checkpoint,map_location=args.device,weights_only=False)['model'].half().eval()
            value = half(x.half())[0]
            require(value.dtype == torch.float16 and torch.isfinite(value).all(), 'EMA half inference failed')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(args.source),str(HERE),env.get('PYTHONPATH','')))
    code = "import torch,tood_td_head,sys;torch.set_num_threads(2);m=torch.load(sys.argv[1],map_location='cpu',weights_only=False)['model'].float().eval();tood_td_head.assert_topology(m);x=torch.rand(1,3,128,160);a=m(x)[0];m.fuse(verbose=False);torch.testing.assert_close(a,m(x)[0],rtol=3e-4,atol=3e-4);print('fresh TD reload/fuse passed')"
    proc = subprocess.run([sys.executable,'-c',code,str(checkpoint)],env=env,capture_output=True,text=True,timeout=180)
    require(proc.returncode == 0, proc.stdout+proc.stderr)
    print(proc.stdout.strip(),flush=True)
    import numpy as np
    from ultralytics import YOLO
    result = YOLO(str(checkpoint),task='detect').predict(np.zeros((128,160,3),dtype=np.uint8),
             imgsz=160,device=args.device,half=False,save=False,verbose=False)
    require(len(result) == 1, 'Standard Predictor failed')
    import d27_official as official
    scoring = official.run_unit_tests()
    cfg = yaml_model_load(str(core.MODEL_YAML)); cfg['scale']='s'
    tal = verify_equivalence(cfg,device=args.device)
    assert_patch_restoration()
    return dict(status='passed',cuda_executed=args.device!='cpu',topology=topology,initialization=audit,
                initial_E1_predictions_loss_common_gradients_exact=True,TD_oracles=unit,
                synthetic_FP32=fp32,synthetic_AMP=amp,empty_GT=True,
                original_TAL_equivalence=tal,official_unit_tests=scoring,
                save_reload_fuse_fresh_process=True,standard_predictor=True,
                actual_tower_routing=routing,
                compute=counts,parameters=9678968+topology['new_parameters'],
                boundary='Engineering/synthetic evidence, not real server640/b16 smoke or performance.')


def real_smoke(core,args,out):
    from run_experiment import build_model, config, trainer_class
    from p2_tal_chunked import chunked_assignment
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset, build_dataloader
    from ultralytics.data.utils import check_det_dataset
    from smoke_memory import check_headroom
    from validation_smoke import validation_stress
    from evaluation import audit_dataset, predict
    from tood_td_head import output_magnitudes
    data = check_det_dataset(str(args.data),autodownload=False)
    params = config(core,args)
    model, base, _ = build_model(core); del base
    model = model.to(args.device).train(); model.args = get_cfg(overrides=params)
    dataset = build_yolo_dataset(model.args,data['train'],16,data,mode='train',rect=False,stride=32)
    require(len(dataset) == 6471, 'Real training image count differs')
    loader = build_dataloader(dataset,16,args.workers,shuffle=True,rank=-1)
    cls = trainer_class(core,{'report_dir':str(out),'_clock':time.perf_counter()})
    trainer = cls.__new__(cls); trainer.args = model.args
    optimizer = trainer.build_optimizer(model,name='SGD',lr=.01,momentum=.937,decay=.0005,iterations=200*len(loader))
    scaler = torch.amp.GradScaler('cuda',enabled=True)
    parameters, groups = group_state(model)
    before = {n:p.detach().clone() for n,p in parameters.items()}
    seen, history, successful = set(), [], 0
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    with chunked_assignment():
        for attempt,batch in enumerate(loader):
            if attempt >= 32:
                break
            check_headroom(torch.cuda,args.device)
            batch['img'] = batch['img'].to(args.device).float()/255
            with torch.autocast('cuda',dtype=torch.float16):
                loss,items = model(batch)
            require(torch.isfinite(loss), 'Nonfinite real smoke loss')
            scaler.scale(loss).backward()
            grads = group_gradients(parameters,groups)
            update = amp_step(model,optimizer,scaler); history.append(update)
            if update['updated']:
                successful += 1; seen.update(grads)
            changed = changed_groups(parameters,before,groups)
            write(out / 'branch_gradient_audit.json',dict(attempts=attempt+1,successful=successful,
                  gradient_groups=sorted(seen),changed_groups=sorted(changed),expected_groups=groups,updates=history))
            print(f'TD SMOKE attempt={attempt+1}/32 updates={successful}/8 '
                  f'grad_groups={len(seen)}/24 changed_groups={len(changed)}/24 '
                  f'reserved={torch.cuda.memory_reserved()/2**30:.2f}GiB loss={items.tolist()}',flush=True)
            if successful >= 8 and seen == changed == set(groups):
                break
    require(successful >= 8 and seen == changed == set(groups),
            'Real smoke failed actual gradient/update groups: ' + str(set(groups)-seen) + ' / ' + str(set(groups)-changed))
    validation = validation_stress(model,args,data,params,out)
    path = out / 'smoke_only.pt'
    model.eval(); model.criterion = None
    torch.save(dict(model=copy.deepcopy(model).cpu().half(),ema=None,train_args={'task':'detect'}),path)
    del loader, optimizer, model
    torch.cuda.empty_cache()
    predictions = {}
    for split in ('val','test-dev'):
        records,_ = audit_dataset(args,split)
        predictions[split] = predict(args,path,records[:8],out / ('predict_' + split))
    peak = torch.cuda.max_memory_reserved()/2**30
    require(peak <= 28.01, 'Real smoke exceeded the fixed28GiB cap')
    return dict(status='passed',variant='tood_td',seed=1,
                successful_optimizer_updates=successful,attempts=len(history),updates=history,
                gradient_groups=sorted(seen),updated_groups=sorted(changed),validation=validation,
                allocator_cap_GiB=28.,peak_reserved_GiB=peak,post_update_inference=True,
                standard_predictor_val_test8=True,prediction_manifests=predictions,
                seconds=time.monotonic()-start,
                engineering_optimizer='SGD lr=.01 per real batch only in smoke; formal E1 schedule unchanged')
