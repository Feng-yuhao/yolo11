"""Inference-only, task/level-specific E47 readout bypasses.

The production E47 forward always runs first. Hooks capture its existing
decomposition and convolution outputs, then replace only a selected task's
new DRF addition with the captured original E40 output. No layer, parameter,
router, encoder, E40 residual, or checkpoint is modified.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import math
from types import MethodType

import torch


STATES = ('normal', 'p2_cls_off', 'p2_reg_off', 'p3_cls_off',
          'p3_reg_off', 'all_readout_off')


def _disabled(state, level, task):
    return (state == 'all_readout_off' or
            state == f'p{level + 2}_{("cls", "reg")[task]}_off')


def state_digest(module):
    """Registered tensor state only; Detect's ephemeral anchor cache is excluded."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _rms(value):
    return float(value.detach().float().square().mean().sqrt())


def summarize_observation(level, block, x, decomposition, base_outputs,
                          readout_updates):
    """Per-batch magnitude/routing summaries, not evidence of a causal defect.

    Observer API: observer(level, block, x, decomposition, base_outputs,
    readout_updates). decomposition is the production (cls, reg, weights,
    bank) result. All tensors are inference-only and must not be mutated.
    Use weights directly for GT-position statistics in the calling worker.
    """
    weights = decomposition[2].detach().float()
    if weights.shape[1:3] != (2, 4):
        raise RuntimeError('Expected two tasks and four encoder depths')
    p = weights.clamp_min(1e-12)
    middle = (p[:, 0] + p[:, 1]) * .5
    js = .5 * ((p[:, 0] * (p[:, 0].log() - middle.log())).sum(1) +
               (p[:, 1] * (p[:, 1].log() - middle.log())).sum(1))
    input_rms = _rms(x)
    tasks = {}
    for task, name in enumerate(('cls', 'reg')):
        entropy = -(p[:, task] * p[:, task].log()).sum(1)
        old_rms = _rms(base_outputs[task] - x)
        new_rms = _rms(readout_updates[task])
        tasks[name] = dict(
            input_rms=input_rms,
            e40_residual_rms=old_rms,
            new_readout_rms=new_rms,
            e40_residual_to_input_rms=old_rms / max(input_rms, 1e-12),
            new_readout_to_input_rms=new_rms / max(input_rms, 1e-12),
            router_entropy_nats=float(entropy.mean()),
            router_entropy_normalized=float(entropy.mean()) / math.log(4),
            depth_weight_mean=[float(v) for v in weights[:, task].mean((0, 2, 3))],
        )
    return dict(level=f'p{level + 2}', batch_size=int(x.shape[0]),
                height=int(x.shape[-2]), width=int(x.shape[-1]), tasks=tasks,
                task_router_js_nats=float(js.mean()),
                task_router_js_normalized=float(js.mean()) / math.log(2))


@contextmanager
def intervention(head, state, observer=None):
    """Scoped six-state bypass with guaranteed restoration on exceptions.

    This deliberately supports FP32/eval only. It does not claim the resulting
    co-adapted, bypassed E47 checkpoint equals a separately trained E40.
    Normal returns the untouched production output tuple, including its exact
    floating-point values. A disabled task uses the original captured E40
    sum directly, avoiding floating-point cancellation in output - update.
    """
    if state not in STATES:
        raise ValueError(f'Unknown E47 intervention: {state}')
    if head.training or not getattr(head, 'enabled', False):
        raise RuntimeError('E47 interventions require an enabled eval head')
    if len(head.task_blocks) != 2:
        raise RuntimeError('Expected exactly the P2/P3 E47 task blocks')
    handles, originals = [], []
    try:
        for level, block in enumerate(head.task_blocks):
            if block.training or not getattr(block, 'readout_enabled', False):
                raise RuntimeError('Need the complete, enabled E47 eval block')
            if any(p.dtype != torch.float32 for p in block.parameters()):
                raise RuntimeError('Call model.float().eval() before D45 diagnosis')
            if getattr(block, '_d45_intervention_active', False):
                raise RuntimeError('Nested E47 interventions are not supported')
            if tuple(block.contrast_basis.shape) != (4, 3):
                raise RuntimeError('E47 contrast basis shape drift')
            if len(block.readout_cls) != 3 or len(block.readout_reg) != 3:
                raise RuntimeError('Expected three readout projections per task')
            present = 'decompose' in block.__dict__
            old_instance = block.__dict__.get('decompose')
            original = block.decompose
            frame = {}
            originals.append((block, present, old_instance))
            block._d45_intervention_active = True

            def capture_decomposition(self, x, original=original, frame=frame):
                if x.dtype != torch.float32:
                    raise RuntimeError('D45 input must be FP32; AMP is disabled')
                result = original(x)
                frame.clear()
                frame.update(x=x, decomposition=result, base={}, readouts={0: [], 1: []})
                return result

            block.decompose = MethodType(capture_decomposition, block)
            for task, layer in enumerate((block.out_cls, block.out_reg)):
                def capture_base(module, args, output, task=task, frame=frame):
                    frame['base'][task] = frame['x'] + output
                handles.append(layer.register_forward_hook(capture_base))
            for task, layers in enumerate((block.readout_cls, block.readout_reg)):
                for layer in layers:
                    def capture_readout(module, args, output, task=task, frame=frame):
                        frame['readouts'][task].append(output)
                    handles.append(layer.register_forward_hook(capture_readout))

            def finish(module, args, output, level=level, frame=frame):
                if not isinstance(output, tuple) or len(output) != 2:
                    raise RuntimeError('E47 block must return (classification, regression)')
                if set(frame['base']) != {0, 1} or any(
                        len(frame['readouts'][t]) != 3 for t in range(2)):
                    raise RuntimeError('Production E47 did not execute all expected paths')
                base = tuple(frame['base'][t] for t in range(2))
                updates = tuple(frame['readouts'][t][0] + frame['readouts'][t][1] +
                                frame['readouts'][t][2] for t in range(2))
                if observer is not None:
                    observer(level, module, frame['x'], frame['decomposition'], base, updates)
                if state == 'normal':
                    return output
                return tuple(base[t] if _disabled(state, level, t) else output[t]
                             for t in range(2))

            handles.append(block.register_forward_hook(finish))
        with torch.inference_mode():
            yield
    finally:
        for handle in reversed(handles):
            handle.remove()
        for block, present, old_instance in reversed(originals):
            if present:
                block.decompose = old_instance
            else:
                del block.decompose
            del block._d45_intervention_active


def _independent_outputs(block, x):
    """Independent formula; deliberately does not call decompose/forward."""
    z = block.reduce(x)
    bank = [z]
    for layer in block.depths:
        z = layer(z)
        bank.append(z)
    b, _, h, w = x.shape
    weights = block.router(torch.cat(bank, 1)).reshape(b, 2, 4, h, w).softmax(2)
    bases, updates = [], []
    for task, (projection, layers) in enumerate(
            ((block.out_cls, block.readout_cls), (block.out_reg, block.readout_reg))):
        mixed = sum(bank[k] * weights[:, task, k:k + 1] for k in range(4))
        bases.append(x + projection(mixed))
        contrast_updates = []
        for j, layer in enumerate(layers):
            contrast = sum(block.contrast_basis[k, j] *
                           bank[k] * weights[:, task, k:k + 1] for k in range(4))
            contrast_updates.append(torch.nn.functional.conv2d(contrast, layer.weight))
        updates.append(sum(contrast_updates))
    return tuple(bases), tuple(updates)


def audit_interventions(head, device='cpu'):
    """Nonzero synthetic fixture using a copy of the actual four-scale head.

    Does not initialize or modify the caller's trained checkpoint. Reports the
    actual device used; a CPU pass cannot be relabeled as a CUDA pass.
    """
    before_original = state_digest(head)
    fixture = copy.deepcopy(head).float().to(device).eval()
    fixture.stride = fixture.stride.to(device)
    fixture.enabled = True
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(45001)
        # Initialize on CPU to avoid perturbing an unrelated CUDA RNG stream.
        for block in fixture.task_blocks:
            for layer in (block.out_cls, block.out_reg,
                          *block.readout_cls, *block.readout_reg):
                value = torch.randn(layer.weight.shape) * .025
                layer.weight.data.copy_(value.to(device))
        inputs = [torch.randn(2, c, h, w).to(device) for c, h, w in
                  ((128, 8, 12), (128, 4, 6), (256, 2, 3), (512, 1, 2))]
    fixture.zero_grad(set_to_none=True)
    guard_inputs = [x.clone() for x in inputs]
    guard_state = state_digest(fixture)
    hook_counts = [(len(b._forward_hooks), len(b.out_cls._forward_hooks),
                    len(b.out_reg._forward_hooks),
                    tuple(len(l._forward_hooks) for l in (*b.readout_cls, *b.readout_reg)))
                   for b in fixture.task_blocks]
    observed = []
    with torch.inference_mode():
        reference_blocks = [b(x) for b, x in zip(fixture.task_blocks, inputs)]
        reference_prediction, reference_raw = fixture([x.clone() for x in inputs])
        oracles = [_independent_outputs(b, x) for b, x in zip(fixture.task_blocks, inputs)]
        for state in STATES:
            def observe(*args):
                observed.append(summarize_observation(*args))
            with intervention(fixture, state, observer=observe):
                outputs = [b(x) for b, x in zip(fixture.task_blocks, inputs)]
                prediction, raw = fixture([x.clone() for x in inputs])
                for level in range(2):
                    for task in range(2):
                        if _disabled(state, level, task):
                            torch.testing.assert_close(outputs[level][task], oracles[level][0][task],
                                                       rtol=3e-5, atol=3e-6)
                            if torch.equal(outputs[level][task], reference_blocks[level][task]):
                                raise RuntimeError('Degenerate fixture: bypass did not change selected path')
                        else:
                            torch.testing.assert_close(outputs[level][task], reference_blocks[level][task],
                                                       rtol=0, atol=0)
                            channels = slice(64, None) if task == 0 else slice(0, 64)
                            torch.testing.assert_close(raw[level][:, channels],
                                                       reference_raw[level][:, channels], rtol=0, atol=0)
                for level in (2, 3):
                    torch.testing.assert_close(raw[level], reference_raw[level], rtol=0, atol=0)
                if state == 'normal':
                    torch.testing.assert_close(prediction, reference_prediction, rtol=0, atol=0)
            for level, block in enumerate(fixture.task_blocks):
                for a, b in zip(block(inputs[level]), reference_blocks[level]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
        def broken_observer(*args):
            raise RuntimeError('intentional observer restoration test')
        try:
            with intervention(fixture, 'p2_cls_off', observer=broken_observer):
                fixture([x.clone() for x in inputs])
        except RuntimeError as exc:
            if str(exc) != 'intentional observer restoration test':
                raise
        else:
            raise RuntimeError('Exception restoration fixture did not raise')
        after_prediction, after_raw = fixture([x.clone() for x in inputs])
        torch.testing.assert_close(after_prediction, reference_prediction, rtol=0, atol=0)
        for a, b in zip(after_raw, reference_raw):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    for level, block in enumerate(fixture.task_blocks):
        current_counts = (len(block._forward_hooks), len(block.out_cls._forward_hooks),
                          len(block.out_reg._forward_hooks),
                          tuple(len(l._forward_hooks) for l in (*block.readout_cls, *block.readout_reg)))
        if current_counts != hook_counts[level] or 'decompose' in block.__dict__ or hasattr(
                block, '_d45_intervention_active'):
            raise RuntimeError('Intervention hook/method leaked after scope exit')
    for a, b in zip(inputs, guard_inputs):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    if state_digest(fixture) != guard_state or state_digest(head) != before_original:
        raise RuntimeError('Intervention modified registered model state')
    if any(p.grad is not None for p in fixture.parameters()):
        raise RuntimeError('Inference diagnosis generated parameter gradients')
    return dict(status='passed', device=str(device), synthetic_only=True,
                states=list(STATES), production_normal_exact=True,
                independent_e40_bypass_formula=True, nonzero_readout_fixture=True,
                untouched_tasks_exact=True, p4_p5_predictions_exact=True,
                normal_restore_exact=True, observer_exception_restored=True,
                inputs_and_state_unchanged=True, no_gradients=True,
                observer_statistics_records=len(observed))
