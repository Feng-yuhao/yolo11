"""TD engineering-only coverage test, with original loss/TAL and all24 groups.

V1 repeated one 128x160 fixture. It can assign no P5 regression positives,
so a healthy P5 reg output receives no regression gradient. This test uses a
predeclared four-size 640 fixture bank and records the ORIGINAL assignment.
No target assignment, label, loss, optimizer, model or formal recipe is patched.
"""
import contextlib
from pathlib import Path
import torch

REVISION='td_synthetic_scale_coverage_640_v2'

def fixtures(device):
    # CPU RNG makes the test inputs identical across CPU/CUDA and job seeds.
    # This independent generator never replaces a formal training RNG/seed.
    generator=torch.Generator(device='cpu').manual_seed(640128)
    image=torch.rand(1,3,640,640,generator=generator).to(device)
    sizes=(.025,.10,.35,.90)
    return [dict(img=image,batch_idx=torch.tensor([0.],device=device),
                 cls=torch.tensor([[3.]],device=device),
                 bboxes=torch.tensor([[.5,.5,s,s]],device=device)) for s in sizes]

class AssignmentAudit:
    """Read-only hooks: observe real TAL foreground + positive quality weights."""
    def __init__(self,model):
        if getattr(model,'criterion',None) is None:model.criterion=model.init_criterion()
        self.shapes=[];self.fg=[0]*4;self.weighted=[0]*4
        self.handles=[model.model[-1].register_forward_hook(self.head_hook),
                      model.criterion.assigner.register_forward_hook(self.assigner_hook)]
    def head_hook(self,module,inputs,output):
        assert isinstance(output,list) and len(output)==4,'Expected original four training outputs'
        self.shapes=[int(v.shape[-2]*v.shape[-1]) for v in output]
    def assigner_hook(self,module,inputs,output):
        fg=output[3];weighted=fg & (output[2].float().sum(-1)>0)
        assert sum(self.shapes)==fg.shape[1],'TAL/head level layout mismatch'
        self.fg=[int(v.sum()) for v in fg.split(self.shapes,dim=1)]
        self.weighted=[int(v.sum()) for v in weighted.split(self.shapes,dim=1)]
    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[]

def synthetic_updates(core,args,out,amp):
    from run_experiment import build_model,config
    from ultralytics.cfg import get_cfg
    from p2_tal_chunked import chunked_assignment
    from checks import strict_math,group_state,group_gradients,changed_groups,amp_step,require
    from common7 import write
    model,base,_=build_model(core);del base
    model=model.to(args.device).train();model.args=get_cfg(overrides=config(core,args))
    parameters,groups=group_state(model)
    before={n:p.detach().clone() for n,p in parameters.items()}
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01,momentum=.937)
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    bank=fixtures(args.device);audit=AssignmentAudit(model)
    seen=set();successful=0;history=[];weighted_total=[0]*4
    try:
        with strict_math(),chunked_assignment():
            for attempt in range(40):
                batch=bank[attempt%len(bank)]
                with torch.autocast(device_type='cuda' if amp else 'cpu',enabled=amp):
                    loss,items=model(batch)
                require(bool(torch.isfinite(loss)),'Nonfinite synthetic detection loss')
                scaler.scale(loss).backward()
                grads=group_gradients(parameters,groups)
                p5=[p.grad for n,p in parameters.items() if n.startswith('out_reg.3.') and p.grad is not None]
                p5_grad=sum(float(g.detach().float().abs().sum()) for g in p5)/scaler.get_scale()
                update=amp_step(model,optimizer,scaler)
                if update['updated']:
                    successful+=1;seen.update(grads)
                    weighted_total=[x+y for x,y in zip(weighted_total,audit.weighted)]
                changed=changed_groups(parameters,before,groups)
                step=dict(attempt=attempt+1,fixture=attempt%len(bank),image_shape=list(batch['img'].shape),
                          box_size=float(batch['bboxes'][0,2]),foreground_by_P2_P5=list(audit.fg),
                          positive_weight_by_P2_P5=list(audit.weighted),
                          P5_reg_unscaled_grad_L1=p5_grad if __import__('math').isfinite(p5_grad) else 'overflow',**update)
                history.append(step)
                print(f'TD COVERAGE AMP={amp} attempt={attempt+1} successful={successful}/8 '
                      f'TAL_positive={audit.weighted} grad_groups={len(seen)}/24 updated_groups={len(changed)}/24',flush=True)
                write(Path(out)/('synthetic_AMP'+str(amp)+'.json'),dict(revision=REVISION,
                    successful=successful,history=history,gradient_groups=sorted(seen),changed_groups=sorted(changed),
                    original_TAL_positive_weight_totals=weighted_total,local_or_synthetic_only=True,
                    original_TAL_and_detection_loss_unchanged=True))
                if successful>=8 and seen==changed==set(groups) and all(v>0 for v in weighted_total):break
        require(successful>=8 and seen==changed==set(groups) and all(v>0 for v in weighted_total),
            'TD scale coverage incomplete: gradients='+str(set(groups)-seen)+', updates='+str(set(groups)-changed)+
            ', original_TAL_positive_weights='+str(weighted_total))
    finally:audit.close()
    model.criterion=None
    return model,dict(successful_optimizer_updates=successful,gradient_groups=sorted(seen),updated_groups=sorted(changed),
        AMP=amp,attempts=len(history),revision=REVISION,original_TAL_positive_weight_totals=weighted_total,
        fixture_shapes=[1,3,640,640],normalized_GT_sizes=[.025,.10,.35,.90],
        original_TAL_and_detection_loss_unchanged=True,all24_groups_still_required=True,
        engineering_only=True,formal_training_recipe_changed=False)

@contextlib.contextmanager
def install(checks_module):
    original=checks_module.synthetic_updates
    checks_module.synthetic_updates=synthetic_updates
    try:yield
    finally:checks_module.synthetic_updates=original
