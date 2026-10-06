"""Preserve the original E40 implementation; add one representation residual at the selected neck stage."""
from torch import nn
import parent_spatial_task_head as parent
from parent_spatial_task_head import SpatialTaskBlock, SpatialTaskDetect
from representation import install, ADDED_PARAMETERS

from representation import VARIANT, SITE
NEW_PARAMETERS = parent.NEW_PARAMETERS + ADDED_PARAMETERS


def attach(model):
    return install(parent.attach(model))


def branch(model):
    # Inspection container only; it is never registered in the actual model.
    return nn.ModuleDict(dict(task_blocks=model.model[-1].task_blocks,
                              representation=model.model[SITE].representation))


def is_new(name):
    return parent.is_new(name) or name.startswith(f'model.{SITE}.representation.')


def original_to_target(name):
    return name


def output_magnitudes(model):
    return parent.output_magnitudes(model)


def reconstruction_state(model):
    module=model.model[SITE].representation
    return dict(output_weight_L1=float(module.output.weight.detach().float().abs().sum()),
                first_batch_statistics=module.last_statistics)
