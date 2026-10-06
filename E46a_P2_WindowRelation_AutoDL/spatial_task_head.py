"""Exact original E40 bank/routing plus a parallel context branch at P2 only."""
import torch
from torch import nn
import parent_spatial_task_head as parent
from parent_spatial_task_head import SpatialTaskDetect,SpatialTaskBlock,original_to_target,is_new
from window_context import WindowContext

VARIANT='window'


class ContextTaskBlock(SpatialTaskBlock):
    def forward(self,x):
        # BOTH paths read the original x. Attention never alters the E40 bank.
        cls,reg=super().forward(x)
        if self.context.enabled:
            dc,dr=self.context(x)
            cls=(cls.float()+dc).to(cls.dtype)
            reg=(reg.float()+dr).to(reg.dtype)
        return cls,reg


def attach(model):
    parent.attach(model)  # creates both E40 blocks with seed40001, unchanged
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(46001)
        context=WindowContext()
    block=model.model[-1].task_blocks[0]
    block.__class__=ContextTaskBlock
    block.context=context
    if sum(p.numel() for p in branch(model).parameters())!=NEW_PARAMETERS:
        raise RuntimeError('E46 parameter topology drift')
    head=model.model[-1]
    head.np=sum(p.numel() for p in head.parameters())
    return model


def branch(model):
    return parent.branch(model)


def output_magnitudes(model):
    values=parent.output_magnitudes(model)
    c=branch(model)[0].context
    return values+[float(c.out_cls.weight.detach().abs().sum()),float(c.out_reg.weight.detach().abs().sum())]


# Explicit formula: projection + two transformer blocks + two task projections.
CONTEXT_PARAMETERS=128*64+2*(2*64*2+64*192+192+225*4+64*64+64+64*128+128+128*64+64)+2*64*128
NEW_PARAMETERS=parent.NEW_PARAMETERS+CONTEXT_PARAMETERS
