"""Independent weak-gradient/value oracle for the new E42 channel router.

This is an engineering test, not a performance probe or a CUDA smoke receipt.
The input/projection scales intentionally exercise FP16's representability limit.
"""
import copy
import torch


def _fixture(device):
    from torch import nn
    from spatial_task_head import SpatialTaskBlock
    from extension import install
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(991)
        block = install(nn.ModuleList([SpatialTaskBlock(), SpatialTaskBlock()]), 'joint')[0]
    block = block.to(device)
    # Nonzero FP32 master weights below FP16's minimum subnormal. Such weights
    # can arise when the ZERO-initialized output projection first learns.
    with torch.no_grad():
        values = torch.linspace(-1., 1., block.channel_router[3].weight.numel(), device=device)
        block.channel_router[3].weight.copy_(values.reshape_as(block.channel_router[3].weight) * 1.e-9)
        block.channel_router[3].bias.zero_()
    descriptor = torch.linspace(-.75, 1.25, 2*192*3*5, device=device).reshape(2,192,3,5).half()
    cotangent = torch.linspace(-.8, 1.1, 2*384, device=device).reshape(2,384,1,1)
    return block, descriptor, cotangent


def channel_precision_audit(device):
    from extension import channel_logits_fp32
    block, descriptor, cotangent = _fixture(device)
    reference = copy.deepcopy(block.channel_router).float()
    expected = reference(descriptor.float())
    reference_gradient = torch.autograd.grad((expected * cotangent).sum()*256., reference[1].weight)[0]
    with torch.autocast(device_type=descriptor.device.type, dtype=torch.float16, enabled=True):
        # Legacy implementation: casting logits AFTER this call is too late.
        legacy = block.channel_router(descriptor)
        actual = channel_logits_fp32(block.channel_router, descriptor)
    legacy_gradient = torch.autograd.grad((legacy.float()*cotangent).sum()*256.,
                                          block.channel_router[1].weight, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad((actual*cotangent).sum()*256., block.channel_router[1].weight)[0]
    if legacy_gradient.count_nonzero() != 0:
        raise RuntimeError('Weak-gradient regression fixture did not reproduce legacy FP16 loss')
    if actual.dtype != torch.float32 or not torch.isfinite(actual_gradient).all() or actual_gradient.count_nonzero() == 0:
        raise RuntimeError('FP32 channel router still loses finite detection-side gradient')
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-12)
    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=3e-5, atol=2e-10)
    # The normal FP32 master parameters must receive gradients themselves;
    # there must not be a copied/unregistered parameter set.
    block.zero_grad(set_to_none=True)
    before = block.channel_router[1].weight.detach().clone()
    optimizer = torch.optim.SGD(block.channel_router.parameters(), lr=.01, weight_decay=0.)
    with torch.autocast(device_type=descriptor.device.type, dtype=torch.float16, enabled=True):
        logits = channel_logits_fp32(block.channel_router, descriptor)
    (logits*cotangent).sum().mul(256.).backward()
    optimizer.step()
    if torch.equal(before, block.channel_router[1].weight):
        raise RuntimeError('FP32 channel front has no real LOSS-only SGD update')
    # EMA/serialized half weights remain supported; no in-place dtype mutation.
    half = copy.deepcopy(block.channel_router).half()
    state_before = {n:v.clone() for n,v in half.state_dict().items()}
    with torch.no_grad():
        result = channel_logits_fp32(half, descriptor)
        expected_half = copy.deepcopy(half).float()(descriptor.float())
    torch.testing.assert_close(result, expected_half, rtol=2e-5, atol=2e-12)
    for n,v in half.state_dict().items():
        torch.testing.assert_close(v, state_before[n], rtol=0, atol=0)
    return dict(device=str(device),legacy_AMP_front_gradient_nonzero=int(legacy_gradient.count_nonzero()),
                FP32_front_gradient_nonzero=int(actual_gradient.count_nonzero()),
                forward_and_gradient_match_FP32_oracle=True,LOSS_only_SGD_update=True,
                half_parameters_unchanged=True)


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    print(json.dumps(channel_precision_audit(args.device), indent=2))
