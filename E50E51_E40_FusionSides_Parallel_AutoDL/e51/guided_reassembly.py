"""Native-grid joint kernel: E50 coarse reassembly; E51 fine contrast."""
import torch
from torch import nn
from torch.nn import functional as F
VARIANT='e51'
SITE=23 if VARIANT=='e50' else 24
ADDED_PARAMETERS=8778

def neighborhood(value,kernels,scale=1):
    b,c,h,w=value.shape
    if kernels.shape!=(b,9,h*scale,w*scale): raise ValueError('Kernel shape mismatch')
    with torch.autocast(device_type=value.device.type,enabled=False):
        padded=F.pad(value.float(),(1,1,1,1),mode='replicate')
        result=torch.zeros(b,c,h*scale,w*scale,device=value.device,dtype=torch.float32)
        for k,(y,x) in enumerate(( (y,x) for y in range(3) for x in range(3) )):
            v=padded[:,:,y:y+h,x:x+w]
            if scale!=1: v=F.interpolate(v,scale_factor=scale,mode='nearest')
            result=result+v*kernels[:,k:k+1].float()
        return result
def reassemble(coarse,kernels): return neighborhood(coarse,kernels,2)
def local_average(fine,kernels): return neighborhood(fine,kernels,1)

class GuidedReassembly(nn.Module):
    def __init__(self):
        super().__init__()
        self.coarse_project=nn.Sequential(nn.Conv2d(128,32,1,bias=False),nn.GroupNorm(4,32,affine=False))
        self.fine_project=nn.Sequential(nn.Conv2d(128,32,1,bias=False),nn.GroupNorm(4,32,affine=False))
        self.kernel=nn.Sequential(nn.Conv2d(32,32,3,padding=1,groups=32,bias=False),nn.SiLU(),nn.Conv2d(32,9,1))
        self.gain=nn.Parameter(torch.zeros(()))
        self.enabled=True
        self.capture_stats=False
        self.last_statistics=None
    def weights(self,coarse,fine):
        # Shared DW parameters, two calls BEFORE resolution alignment.
        cp=self.kernel[0](F.silu(self.coarse_project(coarse)))
        fp=self.kernel[0](F.silu(self.fine_project(fine)))
        logits=self.kernel[2](F.silu(fp+F.interpolate(cp,size=fine.shape[-2:],mode='nearest')))
        with torch.autocast(device_type=coarse.device.type,enabled=False):
            return logits.float().softmax(1)
    def forward(self,inputs):
        if VARIANT=='e50':
            coarse,fine=inputs
            donor=F.interpolate(coarse,size=fine.shape[-2:],mode='nearest')
        else: donor,fine,coarse=inputs
        if (coarse.shape[1]!=128 or fine.shape[1]!=128 or fine.shape[-2:]!=(2*coarse.shape[-2],2*coarse.shape[-1]) or coarse.shape[0]!=fine.shape[0] or coarse.device!=fine.device):
            raise RuntimeError('Actual C128 stride8/stride4 2x sources required')
        if not self.enabled: return donor if VARIANT=='e50' else torch.cat((donor,fine),1)
        weights=self.weights(coarse,fine)
        with torch.autocast(device_type=fine.device.type,enabled=False):
            base=donor.float() if VARIANT=='e50' else fine.float()
            delta=reassemble(coarse,weights)-base if VARIANT=='e50' else base-local_average(fine,weights)
            value=base+self.gain.float().tanh()*delta
        output=value.to(coarse.dtype if VARIANT=='e50' else fine.dtype)
        if self.capture_stats:
            with torch.no_grad():
                denominator=base.square().mean().sqrt().clamp_min(1e-12)
                self.last_statistics=dict(proposed_delta_over_base_RMS=float(delta.square().mean().sqrt()/denominator),
                    actual_output_change_over_base_RMS=float((output.float()-base).square().mean().sqrt()/denominator))
            self.capture_stats=False
        # E51 returns a NEW concatenated tensor; original backbone B stays intact.
        return output if VARIANT=='e50' else torch.cat((donor,output),1)

def install(model):
    old=model.model[23]
    if type(old) is not nn.Upsample or old.f!=16 or old.mode!='nearest' or old.scale_factor!=2 or model.model[24].f!=[-1,2]:
        raise RuntimeError('Untouched E1 model23/Concat24 required')
    if not {2,16}.issubset(model.save) or next(model.parameters()).device.type!='cpu':
        raise RuntimeError('Attach on CPU with saved model2/model16')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(49001)
        module=GuidedReassembly()
    module.i=SITE
    module.f=[16,2] if VARIANT=='e50' else [-1,2,16]
    module.type='guided_reassembly.GuidedReassembly'
    module.np=sum(p.numel() for p in module.parameters())
    if module.np!=ADDED_PARAMETERS: raise RuntimeError('Parameter count drift')
    model.model[SITE]=module
    model.e49_topology=dict(variant=VARIANT,replaced_layer=SITE,sources=module.f,E40_preserved=True,
        kernel=3,hidden=32,new_init_seed=49001,native_grid_shared_DW=True,original_B_not_mutated=True)
    return model
def added_conv_flops(imgsz=640):
    lo,hi=imgsz//8,imgsz//4
    return 2*(128*32*(lo*lo+hi*hi)+32*9*(lo*lo+hi*hi)+32*9*hi*hi)
def mixing_flops(imgsz=640): return (2*9+3)*128*(imgsz//4)**2
