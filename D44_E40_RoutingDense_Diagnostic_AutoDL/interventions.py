"""Scoped in-memory inference interventions. No parameter replacement or training."""
from contextlib import contextmanager
from types import MethodType
import torch

MODES = {
    'e40': ('normal', 'uniform_depth', 'shared_task'),
    'e40s2': ('normal', 'uniform_depth', 'shared_task'),
    'e42': ('normal', 'channel_off_p2', 'channel_off_p3'),
    'e43': ('normal', 'dense_bypass_p2', 'dense_bypass_p3'),
}


def sequential_bank(block, x):
    z = block.reduce(x)
    bank = [z]
    for layer in block.depths:
        z = layer(z)
        bank.append(z)
    return bank


def spatial_mix(block, x, bank, mode):
    b, _, h, w = x.shape
    weights = block.router(torch.cat(bank, 1)).reshape(b, 2, 4, h, w).softmax(2)
    if mode == 'uniform_depth':
        weights = torch.full_like(weights, .25)
    elif mode == 'shared_task':
        weights = weights.mean(1, keepdim=True).expand(-1, 2, -1, -1, -1)
    z = torch.stack(bank, 1)
    return (z * weights[:, 0, :, None]).sum(1), (z * weights[:, 1, :, None]).sum(1), weights, bank


def replacement(block, x, kind, mode):
    bank = sequential_bank(block, x)
    if kind == 'e42':
        from extension import joint_mix, channel_logits_fp32
        descriptor = torch.cat(bank, 1)
        b, _, h, w = x.shape
        spatial = block.router(descriptor).reshape(b, 2, 4, h, w)
        channel = channel_logits_fp32(block.channel_router, descriptor).reshape(b, 2, 4, 48, 1, 1)
        channel = torch.zeros_like(channel)  # only the selected level is intercepted
        cls, reg = joint_mix(bank, spatial, channel)
        return cls, reg, (spatial, channel), bank
    return spatial_mix(block, x, bank, mode)


@contextmanager
def intervene(head, kind, mode, observer=None):
    if mode not in MODES[kind]:
        raise ValueError(f'Unknown intervention: {kind}/{mode}')
    originals = []
    for level, block in enumerate(head.task_blocks):
        present = 'decompose' in block.__dict__
        old_instance = block.__dict__.get('decompose')
        original = block.decompose
        originals.append((block, present, old_instance))
        selected = ((kind in ('e40', 'e40s2') and mode != 'normal') or
                    (kind == 'e42' and mode == f'channel_off_p{level+2}') or
                    (kind == 'e43' and mode == f'dense_bypass_p{level+2}'))

        def wrapped(self, x, original=original, selected=selected, level=level):
            result = replacement(self, x, kind, mode) if selected else original(x)
            if observer is not None:
                observer(level, self, x, result)
            return result
        block.decompose = MethodType(wrapped, block)
    try:
        yield
    finally:
        for block, present, old_instance in originals:
            if present:
                block.decompose = old_instance
            else:
                del block.decompose


def state_digest(model):
    """Exclude ephemeral Detect anchors/shape, which are not registered state."""
    import hashlib
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        t = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(t.shape), t.dtype)).encode())
        digest.update(t.numpy().tobytes())
    return digest.hexdigest()


def self_test(kind, device):
    """Nonzero output projections prevent a vacuous zero-residual equivalence test."""
    import copy
    from torch import nn
    from spatial_task_head import SpatialTaskBlock
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(40001)
        blocks = nn.ModuleList(SpatialTaskBlock() for _ in range(2))
        if kind in ('e42', 'e43'):
            from extension import install
            blocks = install(blocks, 'joint' if kind == 'e42' else 'dense')
        # Controlled, nondegenerate trained-like values ONLY in synthetic fixtures.
        for block in blocks:
            nn.init.normal_(block.out_cls.weight, std=.02)
            nn.init.normal_(block.out_reg.weight, std=.03)
            if kind == 'e42':
                nn.init.normal_(block.channel_router[-1].weight, std=.15)
                nn.init.normal_(block.channel_router[-1].bias, std=.15)
        inputs = [torch.randn(2, 128, 9, 11), torch.randn(2, 128, 5, 7)]
    head = nn.Module(); head.task_blocks = blocks.to(device).eval()
    head.eval()
    inputs = [x.to(device) for x in inputs]
    guards = [x.clone() for x in inputs]
    before = state_digest(head)
    with torch.inference_mode():
        reference = [block(x) for block, x in zip(blocks, inputs)]
        for mode in MODES[kind]:
            with intervene(head, kind, mode):
                actual = [block(x) for block, x in zip(blocks, inputs)]
                for level, (block, x) in enumerate(zip(blocks, inputs)):
                    selected = (kind in ('e40','e40s2') and mode != 'normal') or mode.endswith(f'_p{level+2}')
                    if not selected:
                        for a, b in zip(actual[level], reference[level]):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                        continue
                    bank = sequential_bank(block, x)
                    b, _, h, w = x.shape
                    s = block.router(torch.cat(bank, 1)).reshape(b, 2, 4, h, w)
                    weights = s.softmax(2)
                    if mode == 'uniform_depth': weights = torch.full_like(weights, .25)
                    if mode == 'shared_task': weights = weights.mean(1, keepdim=True).expand_as(weights)
                    # Independent formula uses explicit list/sum, not replacement().
                    mixtures = [sum(bank[k] * weights[:, t, k:k+1] for k in range(4)) for t in range(2)]
                    oracle = (x + block.out_cls(mixtures[0]), x + block.out_reg(mixtures[1]))
                    for a, b in zip(actual[level], oracle):
                        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
                    if all(torch.equal(a, b) for a, b in zip(actual[level], reference[level])):
                        raise RuntimeError('Intervention did not alter the nondegenerate fixture')
            for level, block in enumerate(blocks):
                for a, b in zip(block(inputs[level]), reference[level]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            for a, b in zip(inputs, guards):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        try:
            with intervene(head, kind, MODES[kind][1]):
                raise RuntimeError('intentional restoration test')
        except RuntimeError as exc:
            if str(exc) != 'intentional restoration test': raise
    if state_digest(head) != before or any(p.grad is not None for p in head.parameters()):
        raise RuntimeError('Intervention changed state or generated gradients')
    if any('decompose' in block.__dict__ for block in blocks):
        raise RuntimeError('Intervention method leaked after scope exit')
    return dict(independent_formula=True, nonzero_output_fixture=True,
                unaffected_scale_exact=True, restored_exact=True, exception_restoration=True,
                inputs_and_state_unchanged=True, no_gradients=True, device=str(device))
