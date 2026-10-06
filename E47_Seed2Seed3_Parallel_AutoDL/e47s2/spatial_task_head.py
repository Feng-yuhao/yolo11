"""E47a: untouched E40 encoders/router + task-specific depth-contrast readout."""
import torch
import parent_spatial_task_head as parent
from parent_spatial_task_head import SpatialTaskDetect, original_to_target, is_new
from depth_readout import DepthReadoutBlock, install_readouts, ADDED_PARAMETERS

VARIANT = 'depth_readout'
NEW_PARAMETERS = parent.NEW_PARAMETERS + ADDED_PARAMETERS


def attach(model):
    parent.attach(model)  # Create BOTH original E40 blocks first, seed40001.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(47001)
        for block in parent.branch(model):
            install_readouts(block)
    if sum(p.numel() for p in branch(model).parameters()) != NEW_PARAMETERS:
        raise RuntimeError('E47 depth-readout topology drift')
    head = model.model[-1]
    head.np = sum(p.numel() for p in head.parameters())
    return model


def branch(model):
    return parent.branch(model)


def output_magnitudes(model):
    result = parent.output_magnitudes(model)
    for block in branch(model):
        for layers in (block.readout_cls, block.readout_reg):
            result.extend(float(layer.weight.detach().abs().sum()) for layer in layers)
    return result
