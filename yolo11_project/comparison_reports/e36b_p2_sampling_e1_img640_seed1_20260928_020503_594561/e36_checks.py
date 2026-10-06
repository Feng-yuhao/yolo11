"""Executable regression checks. CPU tests are NOT CUDA certification."""
from __future__ import annotations
import copy
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import torch
from ultralytics.nn.modules.e36_sampling import E36SamplingConv, SITE, attach, original_to_target, is_new_key


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def assert_architecture(model):
    from ultralytics.nn.modules import Detect
    require(len(model.model) == 27 and model.yaml.get('scale') == 's', 'E1 topology/scale drift')
    head = model.model[-1]
    require(type(head) is Detect and head.nc == 10 and head.nl == 4, 'Detect changed')
    require(head.f == [25, 16, 19, 22] and model.stride.tolist() == [4., 8., 16., 32.], 'Pyramid changed')
    wrappers = {n: m for n, m in model.named_modules() if isinstance(m, E36SamplingConv)}
    require(list(wrappers) == [SITE], f'Wrong sampling sites: {list(wrappers)}')
    block = wrappers[SITE]
    require(block.offset_bound == .5 and ((block.offset_head is None) == (block.variant == 'a')), 'Offset policy changed')
    return dict(sampling_site=SITE, variant=block.variant, offset_bound_cells=.5,
                sampling_precision='FP32 island in both arms; output cast to incoming feature dtype',
                dcn_is_existing_method=True, original_detect=True)


def batch(device='cpu', side=128, count=2, objects=4):
    n = count * objects
    sizes = .03 + .2 * torch.rand(n, 2, device=device)
    centers = .15 + .7 * torch.rand(n, 2, device=device)
    return dict(img=torch.rand(count, 3, side, side, device=device),
                batch_idx=torch.arange(count, device=device).repeat_interleave(objects).float(),
                cls=torch.arange(n, device=device).remainder(10).float()[:, None],
                bboxes=torch.cat((centers, sizes), 1))


def equivalence(model, reference):
    m, r = copy.deepcopy(model).cpu().float().eval(), copy.deepcopy(reference).cpu().float().eval()
    for key, value in r.state_dict().items():
        require(torch.equal(m.state_dict()[original_to_target(key)], value), f'E1 tensor differs: {key}')
    block = m.get_submodule(SITE)
    require(not block.mask_head.weight.count_nonzero() and not block.mask_head.bias.count_nonzero(), 'Mask not unit-initialized')
    if block.offset_head is not None:
        require(not block.offset_head.weight.count_nonzero() and not block.offset_head.bias.count_nonzero(), 'Offsets not zero')
    errors = []
    with torch.no_grad():
        for h, w in ((128, 128), (128, 160)):
            x = torch.rand(2, 3, h, w)
            actual, expected = m(x), r(x)
            for a, b in zip(actual[1], expected[1]):
                torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
            torch.testing.assert_close(actual[0], expected[0], rtol=2e-4, atol=2e-4)
            errors.append(float((actual[0] - expected[0]).abs().max()))
    return dict(common_tensors_exact=True, fp32_prediction_numerical_equivalence=True,
                max_decoded_absolute_errors=errors, AMP_bit_exact_equivalence_claimed=False)


def gradient_audit(source, reference, device='cpu', amp=False):
    m = copy.deepcopy(source).float().to(device).train()
    r = copy.deepcopy(reference).float().to(device).train()
    for model in (m, r):
        model.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)
    from ultralytics.utils.loss import v8DetectionLoss
    require(type(m.init_criterion()) is v8DetectionLoss, 'Loss algorithm changed')
    optimizer = torch.optim.SGD(m.parameters(), lr=.01, momentum=.937)
    # An unscaled FP16 backward can falsely flag small new branches as blocked.
    # Conservative scale is for this audit only; real trainer keeps its default.
    scaler = torch.amp.GradScaler('cuda', init_scale=128., enabled=amp)
    records = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        data = batch(device)
        with torch.autocast(device_type=torch.device(device).type, enabled=amp):
            loss, items = m(data)
        require(torch.isfinite(loss).all(), 'Non-finite detection loss')
        scaler.scale(loss.sum()).backward()
        scaler.unscale_(optimizer)
        if step == 0 and not amp:
            other, other_items = r(data)
            other.sum().backward()
            torch.testing.assert_close(items, other_items, rtol=3e-4, atol=3e-4)
            # Full-model shared gradients can be sensitive to TAL ties. Preserve
            # weights exactly and test local Conv gradient separately below.
        norms = {}
        for n, p in m.named_parameters():
            if is_new_key(n):
                require(p.grad is not None and torch.isfinite(p.grad).all(), f'New gradient missing/nonfinite: {n}')
                norm = float(p.grad.norm())
                require(norm == 0 if step == 0 and '.descriptor.' in n else norm > 0,
                        f'Blocked/unexpected gradient at step {step + 1}: {n}={norm}')
                norms[n] = norm
        for n, p in m.named_parameters():
            if p.grad is not None:
                require(torch.isfinite(p.grad).all(), f'Nonfinite original gradient: {n}')
        old = {n: p.detach().clone() for n, p in m.named_parameters() if is_new_key(n)}
        scaler.step(optimizer)
        scaler.update()
        for n, p in m.named_parameters():
            if n in old:
                # Compare to the actual SGD direction, including momentum.
                # Nonzero gradients may produce no representable FP32 change.
                direction = optimizer.state[p].get('momentum_buffer', p.grad)
                expected_parameter = old[n].clone().add_(direction, alpha=-optimizer.param_groups[0]['lr'])
                torch.testing.assert_close(p.detach(), expected_parameter, rtol=1e-7, atol=0,
                                           msg=f'SGD update mismatch: {n}')
                if step == 0 and '.descriptor.' in n:
                    require(torch.equal(p, old[n]), f'Descriptor changed before head activation: {n}')
                elif not torch.equal(expected_parameter, old[n]):
                    require(not torch.equal(p, old[n]), f'Representable SGD update missing: {n}')
                norms[n + ':parameter_max_change'] = float((p.detach() - old[n]).abs().max())
                norms[n + ':sgd_direction_norm'] = float(direction.norm())
        records.append(dict(step=step + 1, loss=float(loss.sum().detach()), new_gradient_norms=norms))
    return dict(device=device, amp=amp, steps=records)


def local_operator_audit(device='cpu', amp=False):
    from ultralytics.nn.modules.conv import Conv
    base = Conv(64, 64, 3).to(device).eval()
    for variant in ('a', 'b'):
        block = E36SamplingConv(copy.deepcopy(base), variant).to(device).eval()
        x = torch.rand(2, 64, 16, 20, device=device, requires_grad=True)
        if not amp:
            a, b = block(x), base(x)
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
            (a.square().mean()).backward()
            ga = block.base.conv.weight.grad.clone()
            base.zero_grad(set_to_none=True)
            (base(x).square().mean()).backward()
            torch.testing.assert_close(ga, base.conv.weight.grad, rtol=2e-4, atol=1e-6)
        else:
            with torch.autocast(device_type='cuda'):
                y = block(x.half())
                loss = y.float().square().mean()
            loss.backward()
            require(torch.isfinite(y).all() and torch.isfinite(block.base.conv.weight.grad).all(), 'AMP operator failure')
    return dict(device=device, amp=amp, passed=True)


def serialization_audit(model, directory):
    m = copy.deepcopy(model).cpu().float().eval()
    x = torch.rand(1, 3, 128, 160)
    with torch.no_grad():
        expected = m(x)[0]
    path = Path(directory) / ('e36_' + m.get_submodule(SITE).variant + '_roundtrip.pt')
    torch.save({'model': m, 'input': x, 'expected': expected}, path)
    # Fresh interpreter imports the actual installed private module path.
    child_env = os.environ.copy()
    child_env['PYTHONPATH'] = os.pathsep.join(str(p) for p in sys.path if p)
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--roundtrip', str(path)],
                            text=True, capture_output=True, timeout=180, env=child_env)
    require(result.returncode == 0, 'Fresh-process reload/fuse failed:\n' + result.stdout + result.stderr)
    path.unlink()
    return dict(fresh_process_reload=True, fusion_checked=True)


def roundtrip(path):
    torch.set_num_threads(4)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model, x, expected = ckpt['model'], ckpt['input'], ckpt['expected']
    assert_architecture(model)
    with torch.no_grad():
        torch.testing.assert_close(model(x)[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(model.fuse(verbose=False)(x)[0], expected, rtol=3e-4, atol=3e-4)
        # Saving HALF precision modules is standard Ultralytics checkpoint behavior.
        half = copy.deepcopy(ckpt['model']).half().float()
        require(torch.isfinite(half(x)[0]).all(), 'Half-checkpoint conversion failure')


def install_test(runtime):
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    torch.set_num_threads(4)
    cfg = yaml_model_load(str(runtime / 'yolo11s-p2-e36.yaml'))
    require(cfg.get('scale') == 's', 'Actual YAML entry filename must select s scale')
    common_new = None
    for variant in ('a', 'b'):
        torch.manual_seed(1)
        base = DetectionModel(copy.deepcopy(cfg), nc=10, verbose=False)
        m = attach(copy.deepcopy(base), variant)
        new_common = {n: p.detach().clone() for n, p in m.named_parameters()
                      if is_new_key(n) and '.offset_head.' not in n}
        if common_new is None:
            common_new = new_common
        else:
            require(all(torch.equal(new_common[n], p) for n, p in common_new.items()),
                    'A/B descriptor/mask initial weights differ')
        result = dict(architecture=assert_architecture(m), equivalence=equivalence(m, base),
                      gradients=gradient_audit(m, base), operator=local_operator_audit())
        with tempfile.TemporaryDirectory() as directory:
            result['serialization'] = serialization_audit(m, directory)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('Use --install-test RUNTIME or --roundtrip CHECKPOINT')
    if sys.argv[1] == '--roundtrip':
        roundtrip(Path(sys.argv[2]))
    elif sys.argv[1] == '--install-test':
        install_test(Path(sys.argv[2]))
    else:
        raise SystemExit('Unknown check mode')
