"""CUDA-only full-detector AMP path test at 640/batch2.

No dataset, performance selection, formal run or production smoke receipt.
Both zero initializations stay intact; all ordinary model parameters optimize.
"""
import gc
import torch


def audit_detector_amp(core, args):
    from ultralytics.cfg import get_cfg
    from p2_tal_chunked import chunked_assignment
    from run_e40 import build_model, config
    from spatial_task_head import branch, VARIANT
    from extension import branch_groups
    from checks import amp_step
    if str(args.device) == 'cpu':
        raise RuntimeError('The detector AMP audit requires actual CUDA')
    model, reference, _ = build_model(core)
    del reference
    model.args = get_cfg(overrides=config(core, args))
    model.to(args.device).train()
    generator = torch.Generator(device=args.device).manual_seed(773)
    batch = dict(img=torch.rand(2,3,640,640,device=args.device,generator=generator),
                 batch_idx=torch.tensor([0.,0.,1.,1.],device=args.device),
                 cls=torch.tensor([[3.],[0.],[4.],[1.]],device=args.device),
                 bboxes=torch.tensor([[.3,.4,.15,.2],[.6,.6,.03,.05],
                                      [.45,.45,.2,.18],[.7,.7,.025,.04]],device=args.device))
    blocks = branch(model)
    groups = branch_groups(VARIANT)
    initial = {n:p.detach().clone() for n,p in blocks.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(),lr=.01,momentum=.937,weight_decay=.0005)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    seen = set()
    updates, gradient_records = [], []
    optimizer.zero_grad(set_to_none=True)
    with chunked_assignment():
        for attempt in range(32):
            with torch.autocast('cuda', dtype=torch.float16):
                loss, _ = model(batch)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite actual detection loss in staged AMP audit')
            scaler.scale(loss).backward()
            maxima = {}
            for group in groups:
                gradients = [p.grad for n,p in blocks.named_parameters() if n.startswith(group)]
                if not gradients or any(g is None for g in gradients):
                    raise RuntimeError('Detached/unregistered AMP branch: '+group)
                maxima[group] = max(float(g.abs().max()) for g in gradients)
            update = amp_step(model,optimizer,scaler)
            updates.append(update)
            if update['updated']:
                finite = {g for g,v in maxima.items() if 0 < v < float('inf')}
                seen.update(finite)
                gradient_records.append(maxima)
                # Double zero-init is intentional. The front channel MLP must
                # wait until its last projection has learned a nonzero weight.
                if VARIANT == 'joint' and len(gradient_records)<=2:
                    for index in range(2):
                        if maxima[f'{index}.channel_router.1.'] != 0:
                            raise RuntimeError('Unexpected early channel-front gradient')
            changed = all(any(not torch.equal(p,initial[n]) for n,p in blocks.named_parameters()
                              if n.startswith(group)) for group in groups)
            if attempt==0 or (attempt+1)%4==0:
                print(f'{VARIANT} CUDA branch check attempt={attempt+1}/32 '
                      f'successful={len(gradient_records)}/8 missing_groups={sorted(set(groups)-seen)}',flush=True)
            if len(gradient_records)>=8 and seen==set(groups) and changed:
                break
    if len(gradient_records)<8 or seen!=set(groups):
        raise RuntimeError('Staged detection AMP failed: successful='+str(len(gradient_records))+
                           ', missing LOSS gradients='+str(set(groups)-seen))
    if not changed:
        raise RuntimeError('Staged detection AMP has an unchanged parameter group')
    print(f'{VARIANT} actual CUDA detection AMP: {len(gradient_records)} finite updates, '
          f'all {len(groups)} groups have nonzero loss gradients and parameter updates',flush=True)
    result = dict(device=str(args.device),imgsz=640,batch=2,synthetic_only=True,
                  successful_updates=len(gradient_records),attempts=len(updates),
                  scaled_gradient_maxima=gradient_records,updates=updates,
                  all_groups_loss_gradients=True,all_groups_updated=True,
                  unchanged_zero_initialization=True)
    del model,optimizer,scaler,blocks,batch,initial,loss
    gc.collect()
    torch.cuda.empty_cache()
    return result
