"""P2-only parallel relation branch; Swin-style windows, not a Swin backbone.

References: https://arxiv.org/abs/2103.14030 and microsoft/Swin-Transformer.
Two pre-LN attention/FFN blocks: width64, heads4, window8, shifts0/4,
FFN expansion2. No downsampling, dropout, DropPath or extra loss.
The small branch computes in FP32 in AMP and EMA-half inference. This is
intentional; the original E40 and prediction towers retain their precision.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call


def partition(x, size=8):
    b,h,w,c=x.shape
    return x.reshape(b,h//size,size,w//size,size,c).permute(0,1,3,2,4,5).reshape(-1,size*size,c)


def reverse(x, b, h, w, size=8):
    return x.reshape(b,h//size,w//size,size,size,-1).permute(0,1,3,2,4,5).reshape(b,h,w,-1)


def window_mask(h,w,shift,device,size=8):
    """Allowed key positions, including padding exclusion and cyclic-wrap mask.

    Padded query rows without an allowed key get a harmless diagonal entry;
    these rows are never propagated to valid pixels, and are cropped away.
    """
    hp=math.ceil(h/size)*size; wp=math.ceil(w/size)*size
    labels=torch.zeros(1,hp,wp,1,device=device)
    if shift:
        regions=(slice(0,-size),slice(-size,-shift),slice(-shift,None))
        number=0
        for row in regions:
            for col in regions:
                labels[:,row,col]=number
                number+=1
    regions=partition(labels,size).squeeze(-1)
    allowed=regions[:,:,None]==regions[:,None,:]
    valid=torch.zeros(1,hp,wp,1,device=device,dtype=torch.bool)
    valid[:,:h,:w]=True
    if shift: valid=torch.roll(valid,(-shift,-shift),(1,2))
    valid=partition(valid,size).squeeze(-1)
    allowed=allowed & valid[:,None,:]
    empty=~allowed.any(-1)
    allowed=allowed | (empty[:,:,None] & torch.eye(size*size,device=device,dtype=torch.bool)[None])
    return allowed,hp,wp


class WindowBlock(nn.Module):
    def __init__(self,shift):
        super().__init__()
        self.shift=shift
        self.norm1=nn.LayerNorm(64)
        self.qkv=nn.Linear(64,192,bias=True)
        self.relative_bias=nn.Parameter(torch.empty(225,4))
        nn.init.trunc_normal_(self.relative_bias,std=.02)
        coords=torch.stack(torch.meshgrid(torch.arange(8),torch.arange(8),indexing='ij')).flatten(1)
        delta=(coords[:,:,None]-coords[:,None,:]).permute(1,2,0)+7
        index=delta[:,:,0]*15+delta[:,:,1]
        self.register_buffer('relative_index',index,persistent=False)
        self.proj=nn.Linear(64,64)
        self.norm2=nn.LayerNorm(64)
        self.ffn=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,64))

    def attention(self,tokens,allowed,batch):
        count,n,c=tokens.shape
        q,k,v=self.qkv(tokens).reshape(count,n,3,4,16).permute(2,0,3,1,4).unbind(0)
        logits=(q*.25) @ k.transpose(-2,-1)
        bias=self.relative_bias[self.relative_index.reshape(-1)].reshape(n,n,4).permute(2,0,1)
        logits=logits+bias[None]
        logits=logits.reshape(batch,-1,4,n,n).masked_fill(~allowed[None,:,None],float('-inf'))
        weights=logits.softmax(-1).reshape(count,4,n,n)
        out=(weights @ v).transpose(1,2).reshape(count,n,c)
        return self.proj(out)

    def forward(self,x):
        b,h,w,c=x.shape
        allowed,hp,wp=window_mask(h,w,self.shift,x.device)
        normalized=self.norm1(x)
        normalized=F.pad(normalized,(0,0,0,wp-w,0,hp-h))
        if self.shift: normalized=torch.roll(normalized,(-self.shift,-self.shift),(1,2))
        attended=self.attention(partition(normalized),allowed,b)
        attended=reverse(attended,b,hp,wp)
        if self.shift: attended=torch.roll(attended,(self.shift,self.shift),(1,2))
        x=x+attended[:,:h,:w]
        return x+self.ffn(self.norm2(x))


class WindowContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce=nn.Conv2d(128,64,1,bias=False)
        self.blocks=nn.ModuleList([WindowBlock(0),WindowBlock(4)])
        self.out_cls=nn.Conv2d(64,128,1,bias=False)
        self.out_reg=nn.Conv2d(64,128,1,bias=False)
        nn.init.zeros_(self.out_cls.weight)
        nn.init.zeros_(self.out_reg.weight)
        self.enabled=True

    def forward(self,x):
        with torch.autocast(device_type=x.device.type,enabled=False):
            if self.reduce.weight.dtype!=torch.float32:
                # No in-place dtype conversion of live parameters or EMA buffers.
                values={n:p.float() for n,p in self.named_parameters()}
                values.update({n:b for n,b in self.named_buffers()})
                return functional_call(self,values,(x.float(),),strict=True)
            z=self.reduce(x.float()).permute(0,2,3,1)
            for block in self.blocks: z=block(z)
            z=z.permute(0,3,1,2)
            return self.out_cls(z),self.out_reg(z)


def context_flops(h,w):
    """Two operations/MAC; excludes LN, GELU, softmax, bias additions.

    Tensor matmuls are explicitly counted, unlike default Conv/Linear THOP.
    Windows include padding but projections outside blocks use actual pixels.
    """
    padded=math.ceil(h/8)*8*math.ceil(w/8)*8
    actual=h*w
    return dict(reduce=2*actual*128*64,
                attention_linear=2*2*padded*4*64*64,
                QK_AV=2*2*2*padded*64*64,
                FFN=2*2*padded*2*64*128,
                task_outputs=2*2*actual*64*128)
