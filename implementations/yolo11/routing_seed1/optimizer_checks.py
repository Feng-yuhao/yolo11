"""Independent SGD-state/rounding audit for synthetic AMP engineering tests.

Does not alter parameters, gradients, optimizer state or the training protocol.
Nonzero gradients below FP32 update resolution are NOT reported as real updates.
"""
import torch


class SGDUpdateAudit:
    def __init__(self, parameters, groups):
        self.parameters=dict(parameters)
        self.groups=list(groups)
        self.pending=None
        self.steps=0
        self.stats={g:dict(nonzero_direction=False,ever_changed=False,
                           max_clipped_gradient=0.,max_proposed_delta=0.,max_delta_over_ULP=0.)
                    for g in self.groups}

    def capture(self, optimizer):
        """Observe AFTER unscale/clip, immediately BEFORE the real SGD step."""
        mapping={id(p):group for group in optimizer.param_groups for p in group['params']}
        pending={}
        for name,p in self.parameters.items():
            if id(p) not in mapping:
                raise RuntimeError('SGD audit: parameter absent from optimizer '+name)
            if p.dtype!=torch.float32:
                raise RuntimeError('SGD audit requires FP32 master parameters '+name)
            group=mapping[id(p)]
            if group.get('weight_decay',0.)!=0 or group.get('dampening',0.)!=0 or group.get('nesterov',False) or group.get('maximize',False):
                raise RuntimeError('Unexpected synthetic SGD protocol')
            old=p.detach().clone()
            if p.grad is None:
                pending[name]=(old,old,None,0.)
                continue
            grad=p.grad.detach().clone()
            if not torch.isfinite(grad).all():
                raise RuntimeError('Nonfinite clipped gradient '+name)
            momentum=group.get('momentum',0.)
            if momentum:
                buffer=optimizer.state.get(p,{}).get('momentum_buffer')
                # Match SGD's specified two FP32 momentum operations, using
                # independent copies. Never edit the live optimizer buffer.
                direction=grad.clone() if buffer is None else buffer.detach().clone().mul_(momentum).add_(grad)
            else:
                direction=grad
            lr=float(group['lr'])
            expected=old.clone().add_(direction,alpha=-lr)
            delta=direction.double()*(-lr)
            towards=torch.where(delta>=0,torch.full_like(old,float('inf')),torch.full_like(old,float('-inf')))
            spacing=(torch.nextafter(old,towards).double()-old.double()).abs()
            ratio=torch.where(delta!=0,delta.abs()/spacing,torch.zeros_like(delta))
            for prefix,stats in self.stats.items():
                if name.startswith(prefix):
                    stats['nonzero_direction'] |= bool(direction.count_nonzero())
                    stats['max_clipped_gradient']=max(stats['max_clipped_gradient'],float(grad.abs().max()))
                    stats['max_proposed_delta']=max(stats['max_proposed_delta'],float(delta.abs().max()))
                    stats['max_delta_over_ULP']=max(stats['max_delta_over_ULP'],float(ratio.max()))
            pending[name]=(old,expected,direction,momentum)
        self.pending=pending

    def verify(self, optimizer):
        if self.pending is None:
            raise RuntimeError('Missing before-step SGD observation')
        for name,p in self.parameters.items():
            old,expected,direction,momentum=self.pending[name]
            if not torch.equal(p.detach(),expected):
                raise RuntimeError('SGD update disagrees with independent FP32 reference '+name)
            if direction is not None and momentum:
                buffer=optimizer.state.get(p,{}).get('momentum_buffer')
                if buffer is None or not torch.equal(buffer,direction):
                    raise RuntimeError('SGD momentum disagrees with independent reference '+name)
            if not torch.equal(p.detach(),old):
                for prefix,stats in self.stats.items():
                    if name.startswith(prefix): stats['ever_changed']=True
        self.pending=None
        self.steps+=1

    def subulp_groups(self):
        return {g for g,s in self.stats.items() if s['nonzero_direction'] and not s['ever_changed']
                and 0<s['max_delta_over_ULP']<.5}

    def report(self):
        return dict(successful_steps_verified=self.steps,all_actual_steps_match_SGD_reference=True,
                    subULP_groups=sorted(self.subulp_groups()),group_numerics=self.stats)
