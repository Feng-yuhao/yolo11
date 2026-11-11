"""Engineering checks only. No real-data smoke and no 200-epoch entry point."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import torch
from torch.nn import functional as F
from common import HERE, write
from e60_adapter import GroupResidualEncoder, branch_groups, NEW_PARAMETERS
from models26 import build_pair


def batch(device, size=128):
    generator = torch.Generator().manual_seed(260601)
    image = torch.rand(2, 3, size, size, generator=generator).to(device)
    classes, indices, boxes = [], [], []
    for i in range(2):
        for j in range(10):
            indices.append(i); classes.append(j)
            boxes.append([.12 + (j % 5)*.17, .25 + (j//5)*.45,
                          .06 + (j % 3)*.025, .07 + (j % 4)*.025])
        # Exercise coarse-scale regression too; a tiny-only fixture need not
        # assign any P5 positives and cannot audit that output projection.
        for category, box in ((3, [.50,.50,.84,.82]), (9,[.50,.50,.60,.62])):
            indices.append(i); classes.append(category); boxes.append(box)
    return dict(img=image, batch_idx=torch.tensor(indices, device=device).float(),
                cls=torch.tensor(classes, device=device).float()[:, None],
                bboxes=torch.tensor(boxes, device=device).float())


def group_oracle(device):
    block = GroupResidualEncoder().to(device)
    with torch.no_grad():
        block.project.weight.uniform_(-.03, .03)
    z = torch.randn(2, 96, 7, 11, device=device, requires_grad=True)
    actual = block(z)
    pieces = z.chunk(4, 1)
    values = [pieces[0]]
    for j in range(3):
        layer, norm, _ = block.kernels[j]
        x = pieces[j+1] if j == 0 else pieces[j+1] + values[-1]
        x = F.conv2d(x, layer.weight, padding=1)
        x = F.group_norm(x, 6, norm.weight, norm.bias, norm.eps)
        values.append(F.silu(x))
    expected = z + F.conv2d(torch.cat(values, 1), block.project.weight)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    parameters = (z, *block.parameters())
    g1 = torch.autograd.grad(actual.square().mean(), parameters)
    g2 = torch.autograd.grad(expected.square().mean(), parameters)
    for a, b in zip(g1, g2):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-7)
    before = {k:v.detach().clone() for k,v in block.state_dict().items()}
    block.half()
    for key, value in block.state_dict().items():
        assert value.dtype == torch.float32 and torch.equal(value, before[key]), key
    return dict(independent_values=True, independent_input_and_parameter_gradients=True,
                half_original_values_preserved=True)


def initial_equivalence(base, enhanced, device):
    for shape in ((128,128),(128,160)):
        image = torch.rand(2,3,*shape,device=device)
        base.eval(); enhanced.eval()
        with torch.no_grad():
            before, after = base(image), enhanced(image)
        torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
        for branch in ('one2many','one2one'):
            for key in ('boxes','scores'):
                torch.testing.assert_close(before[1][branch][key], after[1][branch][key],rtol=0,atol=0)
    data = batch(device)
    base.train(); enhanced.train()
    base.zero_grad(set_to_none=True); enhanced.zero_grad(set_to_none=True)
    base_loss, _ = base.loss(copy.deepcopy(data))
    full_loss, _ = enhanced.loss(copy.deepcopy(data))
    torch.testing.assert_close(base_loss, full_loss, rtol=0, atol=0)
    base_loss.sum().backward(); full_loss.sum().backward()
    params = dict(enhanced.named_parameters())
    compared = 0
    for key, value in base.named_parameters():
        other = params[key]
        if value.grad is None:
            assert other.grad is None, key
        else:
            torch.testing.assert_close(value.grad, other.grad, rtol=0, atol=0, msg=key)
            compared += 1
    assert type(enhanced.criterion).__name__ == 'E2ELoss'
    return dict(square_and_rectangle_raw_decoded_exact=True,
                original_dual_loss_exact=True, common_gradients_exact=compared,
                criterion='original E2ELoss', primary_inference='one2many + NMS')


def detach_audit(model, device):
    model.train(); model.zero_grad(set_to_none=True)
    preds = model(batch(device)['img'])
    auxiliary = preds['one2one']
    (auxiliary['boxes'].square().mean() + auxiliary['scores'].square().mean()).backward()
    new = [(n,p) for n,p in model.named_parameters() if '.task_blocks.' in n]
    assert all(p.grad is None for _,p in new)
    assert all(p.grad is None for p in model.model[0].parameters())
    head = model.model[-1]
    assert all(p.grad is None for p in head.cv2.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in head.one2one_cv2.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in head.one2one_cv3.parameters())
    return dict(detach_boundary='AFTER shared E60 feature preparation',
                one2one_does_not_update_shared_bank_or_backbone=True,
                original_one2one_towers_receive_gradients=True)


def updates(model, device, amp=False):
    model.train()
    params = dict(model.named_parameters())
    groups = branch_groups()
    snapshots = {n:p.detach().clone() for n,p in params.items() if '.task_blocks.' in n}
    touched, changed = set(), set()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.937, weight_decay=.0005)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    success = 0
    data = batch(device)
    for attempt in range(32):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            loss, items = model.loss(copy.deepcopy(data))
        assert torch.isfinite(loss).all(), (loss, items)
        scaler.scale(loss.sum()).backward()
        scaler.unscale_(optimizer)
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in params.values())
        scale = scaler.get_scale()
        if finite:
            for group in groups:
                if any(n.startswith(group) and p.grad is not None and p.grad.abs().sum()>0
                       for n,p in params.items()):
                    touched.add(group)
            # Same max_norm as the pinned native Trainer.optimizer_step.
            # Unclipped toy SGD is not a faithful stability check of that path.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer); scaler.update()
        if finite and scaler.get_scale() >= scale:
            success += 1
            for group in groups:
                if any(n.startswith(group) and not torch.equal(p.detach(), snapshots[n])
                       for n,p in params.items() if n in snapshots):
                    changed.add(group)
        print(f'CHECK synthetic AMP={amp} attempt={attempt+1} finite_updates={success}/8 '
              f'grad_groups={len(touched)}/52 updated_groups={len(changed)}/52',flush=True)
        if success >= 8:
            break
    assert success == 8, 'Not enough finite synthetic updates'
    assert touched == set(groups), 'Missing synthetic gradients: '+str(set(groups)-touched)
    assert changed == set(groups), 'Missing actual synthetic updates: '+str(set(groups)-changed)
    return dict(successful_updates=success, gradient_groups=len(touched), actual_update_groups=len(changed),
                synthetic=True, image_size=128, batch=2,
                real_visdrone_smoke=False, original_formal_schedule_executed=False)


def serialize_fuse(model, device, out):
    model.eval()
    image = batch(device)['img']
    with torch.no_grad():
        expected = model(image)[0]
    path = out/'engineering_model.pt'
    torch.save(model.cpu(), path)
    restored = torch.load(path, map_location='cpu', weights_only=False).to(device).eval()
    with torch.no_grad():
        actual = restored(image)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    restored.fuse(verbose=False)
    assert restored.model[-1].one2one_cv2 is None
    with torch.no_grad():
        fused = restored(image)[0]
    # Decode can subtract almost equal distances/anchors near coordinate zero;
    # a relative tolerance on that final coordinate is not meaningful. Audit
    # the algebra independently at FP64, then bound actual FP32 pixel error.
    precise = copy.deepcopy(model).to(device).double().eval()
    with torch.no_grad():
        double_before = precise(image.double())[0]
    precise.fuse(verbose=False)
    with torch.no_grad():
        double_after = precise(image.double())[0]
    torch.testing.assert_close(double_after,double_before,rtol=2e-8,atol=2e-8)
    del precise
    box_error = float((fused[:,:4]-expected[:,:4]).abs().max())
    score_error = float((fused[:,4:]-expected[:,4:]).abs().max())
    assert box_error < .002, f'FP32 fuse pixel error too large: {box_error}'
    torch.testing.assert_close(fused[:,4:],expected[:,4:],rtol=2e-4,atol=2e-6)
    # Probe sub-half-ULP refiners independently of the already trained model.
    probe = GroupResidualEncoder()
    with torch.no_grad():
        probe.project.weight.fill_(1e-8)
    saved = probe.project.weight.detach().clone()
    probe.half()
    assert probe.project.weight.dtype == torch.float32 and torch.equal(saved,probe.project.weight)
    # Fresh process imports the same package/source but no parent's Python cache.
    snippet = (
        'import sys,torch; from pathlib import Path; '
        f'sys.path.insert(0,{str(HERE)!r}); '
        'import e60_adapter; '
        f'm=torch.load({str(path)!r},map_location="cpu",weights_only=False); '
        'm.eval(); torch.set_num_threads(1); '
        'y=m(torch.zeros(1,3,128,128))[0]; '
        'assert torch.isfinite(y).all(); print("fresh-process checkpoint passed")'
    )
    process = subprocess.run([sys.executable,'-c',snippet],text=True,capture_output=True)
    if process.returncode:
        raise RuntimeError(process.stdout+process.stderr)
    if device.type=='cuda':
        half = copy.deepcopy(model).to(device).half().eval()
        with torch.no_grad():
            output = half(image.half())[0]
        assert torch.isfinite(output).all()
        assert all(p.dtype==torch.float32 for n,p in half.named_parameters() if '.refinements.' in n)
    return dict(exact_reload=True, fused_nms_branch_close=True, fresh_process=True,
                fp64_fuse_algebra_close=True,fp32_fuse_max_box_error_pixels=box_error,
                fp32_fuse_max_score_error=score_error,
                fp32_sidecar_value_preserved=True, cuda_half_executed=device.type=='cuda',
                checkpoint_is_engineering_only_not_for_accuracy=True)


def check(weights, device, out, local=False):
    device = torch.device(device)
    torch.set_num_threads(2)
    torch.manual_seed(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.type=='cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; do not silently fall back to CPU')
        free, total = torch.cuda.mem_get_info(device)
        minimum, cap = (3,4) if local else (8,16)
        if free < minimum*1024**3:
            raise RuntimeError(f'Need at least {minimum} GiB free for this engineering check')
        torch.cuda.set_per_process_memory_fraction(min(cap*1024**3/total,1.),device)
    result = dict(status='running', stage='engineering_only',local_engineering=local,
                  formal_training_gate=False, real_data_smoke_completed=False)
    write(out/'check.json',result)
    base, model, transfer = build_pair(weights)
    base.to(device); model.to(device)
    result['initialization'] = transfer
    result['parameters'] = dict(baseline=sum(p.numel() for p in base.parameters()),
                                enhanced=sum(p.numel() for p in model.parameters()),
                                added=NEW_PARAMETERS)
    result['group_oracle'] = group_oracle(device)
    print('Group hierarchy and FP32 value preservation passed',flush=True)
    result['initial_equivalence'] = initial_equivalence(base,model,device)
    print('Both native heads / original loss / common gradients exact at start',flush=True)
    result['detach'] = detach_audit(model,device)
    del base
    result['fp32_updates'] = updates(model,device,False)
    result['serialization'] = serialize_fuse(model,device,out)
    if device.type=='cuda':
        _, model, _ = build_pair(weights)
        model.to(device)
        result['amp_updates'] = updates(model,device,True)
    result.update(status='engineering_passed',cuda_executed=device.type=='cuda',
                  original_dual_head_training_preserved=True)
    write(out/'check.json',result)
    print('ENGINEERING CHECK PASSED. Real 640/b16 smoke and formal training are NOT executed.',flush=True)
    return result
