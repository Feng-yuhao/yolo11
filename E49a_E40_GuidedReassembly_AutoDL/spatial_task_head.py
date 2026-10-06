"""E49 assembly; original E40 head source is preserved byte-for-byte separately."""
import torch
from torch import nn
import parent_spatial_task_head as parent
from parent_spatial_task_head import SpatialTaskBlock, SpatialTaskDetect
from guided_reassembly import install, ADDED_PARAMETERS

VARIANT = 'guided_reassembly'
NEW_PARAMETERS = parent.NEW_PARAMETERS + ADDED_PARAMETERS


def attach(model):
    return install(parent.attach(model))


def branch(model):
    # Temporary inspection container only. Both modules are already registered
    # under model23/model26. It is never inserted into the training model.
    return nn.ModuleDict(dict(task_blocks=model.model[-1].task_blocks,
                              reassembly=model.model[23]))


def is_new(name):
    return parent.is_new(name) or name.startswith('model.23.')


def original_to_target(name):
    return name


def output_magnitudes(model):
    return parent.output_magnitudes(model)


def reconstruction_state(model):
    gain=model.model[23].gain.detach().float()
    return dict(raw_gain=float(gain), mixing=float(gain.tanh()),
                first_batch_statistics=model.model[23].last_statistics)
