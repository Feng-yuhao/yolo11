"""Two predeclared E60 ablations, NOT two new optimized heads.

uniform: remove every router and average the four enhanced depth features.
shared: one spatial depth distribution serves both tasks; towers/projections stay.
All original E60 encoders, FP32 refiners, initialization and raw depth inputs stay.
"""
import copy
import torch
from torch import nn
import e60_full_head as full

CHANNELS = full.CHANNELS
NEW_PREFIX = full.NEW_PREFIX
GroupResidualEncoder = full.GroupResidualEncoder
VARIANT = None
COUNTS = {'uniform': 1442784, 'shared': 1493616}
NEW_PARAMETERS = None


def configure(variant):
    global VARIANT, NEW_PARAMETERS
    if variant not in COUNTS:
        raise ValueError('Choose uniform or shared')
    VARIANT, NEW_PARAMETERS = variant, COUNTS[variant]
    full.configure('e60')


class AblationTaskBlock(nn.Module):
    # Adopt a fully initialized E60 block, rather than reconstruct its encoders.
    original_bank = full.GroupResidualTaskBlock.original_bank
    forward = full.e58.WideTaskBlock.forward

    def enhanced_bank(self, x):
        bank = self.original_bank(x)
        if hasattr(self, 'refinements'):
            bank = [bank[0]] + [r(z) for r, z in zip(self.refinements, bank[1:])]
        return bank

    def decompose(self, x):
        bank = self.enhanced_bank(x)
        stacked = torch.stack(bank, dim=1)
        b, _, h, w = x.shape
        if self.ablation == 'uniform':
            feature = stacked.mean(dim=1)
            weights = stacked.new_full((b, 2, 4, h, w), .25)
        elif self.ablation == 'shared':
            # Exactly one softmax per location, normalized over DEPTH, not tasks.
            one = self.router(torch.cat(bank, dim=1)).softmax(dim=1)
            feature = (stacked * one[:, :, None]).sum(dim=1)
            weights = one[:, None].expand(-1, 2, -1, -1, -1)
        else:
            raise RuntimeError('Checkpoint contains an unknown ablation')
        # Distinct out_cls/out_reg projections and original cv3/cv2 towers follow.
        return feature, feature, weights, bank


# Compatibility for frozen group/precision tests and checkpoint topology checks.
GroupResidualTaskBlock = AblationTaskBlock


class SpatialTaskDetect(full.SpatialTaskDetect):
    pass  # Inherit the exact four-scale E57/E60 Detect forward.


def convert_block(block, variant):
    block.__class__ = AblationTaskBlock
    block.ablation = variant
    if variant == 'uniform':
        del block.router  # No unused/dummy parameters in the optimizer or profile.
    elif variant == 'shared':
        old = block.router[-1]
        # Reduce only the final 8-logit output to 4. Preserve the router trunk.
        # Initialize with the mean of the full E60 task logits; consume no RNG.
        with torch.random.fork_rng(devices=[]):
            layer = nn.Conv2d(old.in_channels, 4, 1, bias=True)
        with torch.no_grad():
            layer.weight.copy_((old.weight[:4] + old.weight[4:]) * .5)
            layer.bias.copy_((old.bias[:4] + old.bias[4:]) * .5)
        block.router[-1] = layer
    else:
        raise ValueError(variant)
    return block


def attach(model):
    if VARIANT is None:
        raise RuntimeError('Configure the ablation before attach')
    full.attach(model)
    head = model.model[-1]
    for block in head.task_blocks:
        convert_block(block, VARIANT)
    head.__class__ = SpatialTaskDetect
    head.experiment_variant = VARIANT
    head.np = sum(p.numel() for p in head.parameters())
    head.type = 'spatial_task_head.SpatialTaskDetect'
    assert_topology(model, VARIANT)
    return model


def assert_topology(model, variant):
    head = model.model[-1]
    if (type(head) is not SpatialTaskDetect or head.experiment_variant != variant or
            len(head.task_blocks) != 4 or head.nl != 4 or head.reg_max != 16):
        raise RuntimeError('Wrong saved ablation head')
    for i, block in enumerate(head.task_blocks):
        if (type(block) is not AblationTaskBlock or block.ablation != variant or
                block.reduce[0].in_channels != CHANNELS[i] or len(block.depths) != 3):
            raise RuntimeError('Wrong scale/block/encoder topology')
        if (hasattr(block, 'refinements') != (i < 2) or
                (i < 2 and len(block.refinements) != 3)):
            raise RuntimeError('P2/P3 refinement scope drift')
        if variant == 'uniform' and hasattr(block, 'router'):
            raise RuntimeError('Uniform ablation retained a router')
        if variant == 'shared' and block.router[-1].out_channels != 4:
            raise RuntimeError('Task-shared router must output exactly four logits')
        if block.out_cls is block.out_reg or head.cv2[i] is head.cv3[i]:
            raise RuntimeError('Original tasks/projections must remain separate')
    if sum(p.numel() for p in head.task_blocks.parameters()) != COUNTS[variant]:
        raise RuntimeError('Active branch parameter count drift')


def assert_common_initialization(full_model, candidate, variant):
    original, actual = full_model.state_dict(), candidate.state_dict()
    removed = set(original) - set(actual)
    if set(actual) - set(original):
        raise RuntimeError('Unexpected extra initialization keys')
    for name, value in original.items():
        router_key = name.startswith(NEW_PREFIX) and '.router.' in name
        final_key = router_key and (name.endswith('.4.weight') or name.endswith('.4.bias'))
        if variant == 'uniform' and router_key:
            if name not in removed:
                raise RuntimeError('Uniform router was not completely removed')
            continue
        expected = (value[:4] + value[4:]) * .5 if variant == 'shared' and final_key else value
        if name not in actual or not torch.equal(expected, actual[name]):
            raise RuntimeError('Common E60 initialization drift: ' + name)
    if any(not (n.startswith(NEW_PREFIX) and '.router.' in n) for n in removed):
        raise RuntimeError('Removed non-router parameters')


def branch(model):
    return model.model[-1].task_blocks


def branch_groups():
    groups = full.branch_groups()
    return [g for g in groups if not (VARIANT == 'uniform' and '.router.' in g)]


original_to_target = full.original_to_target
is_new = full.is_new
output_magnitudes = full.output_magnitudes
