"""E48: complete E47, plus zero-start P2 nonlinear task readouts only."""
import torch
import parent_e47_spatial_task_head as parent
from parent_spatial_task_head import SpatialTaskDetect, original_to_target, is_new
from nonlinear_readout import install_nonlinear, ADDED_PARAMETERS

VARIANT = 'p2_nonlinear_readout'
NEW_PARAMETERS = parent.NEW_PARAMETERS + ADDED_PARAMETERS


def attach(model):
    parent.attach(model)  # untouched CPU forks 40001 and 47001
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(48001)
        install_nonlinear(parent.branch(model)[0])
    if sum(p.numel() for p in branch(model).parameters()) != NEW_PARAMETERS:
        raise RuntimeError('E48 parameter/topology drift')
    head = model.model[-1]
    head.np = sum(p.numel() for p in head.parameters())
    return model


def branch(model):
    return parent.branch(model)


def output_magnitudes(model):
    result = parent.output_magnitudes(model)
    block = branch(model)[0]
    result.extend(float(layer[2].weight.detach().abs().sum())
                  for layer in (block.nonlinear_cls, block.nonlinear_reg))
    return result
