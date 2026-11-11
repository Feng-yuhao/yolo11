"""Regression fixtures for tiny nested residual weights; not a performance test."""
import copy
import io
import torch


def legacy_delta(module, z):
    """E60 v1 formula, intentionally subject to ambient autocast."""
    split = z.chunk(4, 1)
    values = [split[0]]
    for j, kernel in enumerate(module.kernels):
        argument = split[j+1] if j == 0 else split[j+1] + values[-1]
        values.append(kernel(argument))
    return module.project(torch.cat(values, 1))


def tiny_weight_audit(device, fixed=False):
    from spatial_task_head import GroupResidualEncoder
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(60602)
        encoder = GroupResidualEncoder().to(device)
    with torch.no_grad():
        encoder.project.weight.fill_(1e-9)
    reference = copy.deepcopy(encoder)
    x = torch.linspace(-2, 2, 2*96*7*9, device=device).reshape(2, 96, 7, 9).half()
    probe = torch.linspace(-.5, 1, x.numel(), device=device).reshape(x.shape)
    before = {n:p.detach().clone() for n,p in encoder.named_parameters()}
    if str(device) == 'cpu':
        return dict(cuda_fixture_executed=False)
    with torch.autocast('cuda', dtype=torch.float16):
        actual = encoder(x) if fixed else x + legacy_delta(encoder, x)
        loss = (actual.float() * probe).sum() / x.numel()
    (loss * 1024).backward()
    with torch.autocast('cuda', enabled=False):
        expected = x + legacy_delta(reference, x.float()).to(x.dtype)
        reference_loss = (expected.float() * probe).sum() / x.numel()
    (reference_loss * 1024).backward()
    groups = [f'kernels.{i}.' for i in range(3)]
    def max_gradient(module, group):
        return max(float(p.grad.detach().abs().max()) for n,p in module.named_parameters()
                   if n.startswith(group) and p.grad is not None)
    actual_grad = {g:max_gradient(encoder,g) for g in groups}
    expected_grad = {g:max_gradient(reference,g) for g in groups}
    if not all(v > 0 for v in expected_grad.values()):
        raise RuntimeError('FP32 tiny-weight control has no inner gradient')
    if fixed:
        for name, p in encoder.named_parameters():
            if not torch.isfinite(p.grad).all():
                raise RuntimeError('Nonfinite tiny-weight gradient '+name)
            torch.testing.assert_close(p.grad, dict(reference.named_parameters())[name].grad,
                                       rtol=1e-5, atol=1e-12)
        optimizer = torch.optim.SGD(encoder.parameters(), lr=.01)
        optimizer.step()
        for group in groups:
            if not any(not torch.equal(p,before[n]) for n,p in encoder.named_parameters()
                       if n.startswith(group)):
                raise RuntimeError('No real tiny-weight inner update '+group)
    elif any(actual_grad.values()):
        raise RuntimeError('Legacy fixture did not reproduce FP16 inner-gradient loss')
    return dict(cuda_fixture_executed=True, fixed=fixed,
                project_FP32_nonzero=int(before['project.weight'].count_nonzero()),
                project_FP16_nonzero=int(before['project.weight'].half().count_nonzero()),
                ambient_AMP_inner_gradient_max=actual_grad,
                FP32_control_inner_gradient_max=expected_grad,
                all_inner_groups_really_updated=fixed)


def checkpoint_precision_audit(device):
    from spatial_task_head import GroupResidualEncoder
    encoder = GroupResidualEncoder().to(device)
    with torch.no_grad():
        encoder.project.weight.fill_(1e-9)
    before = {n:p.detach().clone() for n,p in encoder.named_parameters()}
    for conversion in ('half','bfloat16','float'):
        clone = getattr(copy.deepcopy(encoder),conversion)()
        for name,p in clone.named_parameters():
            if p.dtype != torch.float32:
                raise RuntimeError('Protected parameter converted from FP32 '+name)
            torch.testing.assert_close(p,before[name],rtol=0,atol=0)
    half = copy.deepcopy(encoder).half()
    data = io.BytesIO()
    torch.save(half.cpu(),data); data.seek(0)
    restored = torch.load(data,map_location=device,weights_only=False)
    for name,p in restored.named_parameters():
        torch.testing.assert_close(p,before[name],rtol=0,atol=0)
        if p.device != torch.device(device):
            raise RuntimeError('Protected parameter ignored destination device')
    if str(device) != 'cpu':
        x = torch.randn(1,96,7,9,device=device,dtype=torch.float16,requires_grad=True)
        out = restored(x)
        if out.dtype != x.dtype or not torch.isfinite(out).all():
            raise RuntimeError('Mixed precision output dtype/value failed')
        (out.float().square().mean()*1024).backward()
        for i in range(3):
            selected = [p for n,p in restored.named_parameters() if n.startswith(f'kernels.{i}.')]
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in selected) or not any(p.grad.count_nonzero() for p in selected):
                raise RuntimeError('Saved tiny refiner lost inner gradients')
    return dict(tiny_FP32_weights_survive_half_bfloat16_save_reload=True,
                device_transfer_preserved=True,output_dtype_preserved=True)


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--fixed',action='store_true')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = tiny_weight_audit(args.device,args.fixed)
    if args.fixed: result['checkpoint'] = checkpoint_precision_audit(args.device)
    print(json.dumps(result,indent=2))
