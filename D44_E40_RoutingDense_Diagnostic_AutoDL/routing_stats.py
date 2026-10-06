"""GT-center / 3x3-neighborhood measurements. Descriptive, not error causes."""
import math
import numpy as np
import torch
from torch.nn import functional as F


def js(p, q):
    m = (p + q) * .5
    return .5 * ((p * (p.clamp_min(1e-12).log()-m.clamp_min(1e-12).log())).sum(0) +
                 (q * (q.clamp_min(1e-12).log()-m.clamp_min(1e-12).log())).sum(0))


class RoutingStats:
    def __init__(self, kind):
        self.kind = kind
        self.context = None
        self.rows = []

    def set_context(self, records, shape):
        self.context = (records, tuple(shape))

    def __call__(self, level, block, x, result):
        cls, reg, weights, bank = result
        records, image_shape = self.context
        if len(records) != len(x): raise RuntimeError('GT routing batch mismatch')
        rms = lambda t: F.avg_pool2d(t.float().square().mean(1, keepdim=True), 3, 1, 1,
                                    count_include_pad=False).clamp_min(0).sqrt()[:, 0]
        base = rms(x)
        cr = rms(block.out_cls(cls)); rr = rms(block.out_reg(reg))
        for index, rec in enumerate(records):
            gt = rec.eval_gt
            gt = gt[(gt[:,4] == 0) & (gt[:,5] >= 1) & (gt[:,5] <= 10)]
            if not len(gt): continue
            gain = min(image_shape[0]/rec.height, image_shape[1]/rec.width)
            px = round((image_shape[1]-rec.width*gain)/2-.1)
            py = round((image_shape[0]-rec.height*gain)/2-.1)
            cx = (gt[:,0]+gt[:,2]*.5)*gain+px
            cy = (gt[:,1]+gt[:,3]*.5)*gain+py
            xx = torch.as_tensor(np.floor(cx * x.shape[-1]/image_shape[1]), device=x.device,
                                 dtype=torch.long).clamp(0, x.shape[-1]-1)
            yy = torch.as_tensor(np.floor(cy * x.shape[-2]/image_shape[0]), device=x.device,
                                 dtype=torch.long).clamp(0, x.shape[-2]-1)
            if self.kind == 'e42':
                spatial, channel = weights
                s = spatial[index, :, :, yy, xx].float()  # task, depth, GT
                c = channel[index, :, :, :, 0, 0].float()  # task, depth, channel
                # Only GT centers, not a full [B,T,K,C,H,W] allocation.
                joint = (s[:, :, None, :] + c[:, :, :, None]).softmax(1)
                effective = joint.mean(2)
                spatial_weights = s.softmax(1)
                difference = (joint-spatial_weights[:, :, None, :]).abs().sum(1).mean((0,1))
            else:
                effective = weights[index, :, :, yy, xx].float()
                difference = None
            if effective.shape != (2,4,len(gt)):
                raise RuntimeError('Wrong depth/task axis in GT routing statistics')
            torch.testing.assert_close(effective.sum(1), torch.ones_like(effective[:,0]), rtol=1e-5, atol=1e-6)
            divergence = js(effective[0], effective[1]).cpu().numpy()
            entropy = -(effective*effective.clamp_min(1e-12).log()).sum(1).cpu().numpy()
            values = effective.cpu().numpy()
            a, b, c = (v[index, yy, xx].cpu().numpy() for v in (base, cr, rr))
            sizes = np.sqrt(gt[:,2]*gt[:,3])*640/max(rec.height,rec.width)
            changes = difference.cpu().numpy() if difference is not None else None
            for j in range(len(gt)):
                row = dict(image_id=rec.image_id, gt_index=j, level=f'P{level+2}',
                           cls=int(gt[j,5]), size=float(sizes[j]), task_JS_nats=float(divergence[j]),
                           cls_entropy_nats=float(entropy[0,j]), reg_entropy_nats=float(entropy[1,j]),
                           base_RMS=float(a[j]), cls_delta_RMS=float(b[j]), reg_delta_RMS=float(c[j]),
                           cls_ratio=float(b[j]/max(a[j],1e-12)), reg_ratio=float(c[j]/max(a[j],1e-12)),
                           cls_depth_weights=values[0,:,j].tolist(), reg_depth_weights=values[1,:,j].tolist())
                if changes is not None: row['joint_vs_spatial_depth_L1']=float(changes[j])
                self.rows.append(row)

    def summary(self):
        output = {}
        def summarize(items):
            keys = ('task_JS_nats','cls_entropy_nats','reg_entropy_nats','base_RMS',
                    'cls_delta_RMS','reg_delta_RMS','cls_ratio','reg_ratio','joint_vs_spatial_depth_L1')
            item = dict(gt_count=len(items))
            for k in keys:
                v = [r[k] for r in items if k in r]
                if v: item[k] = dict(mean=float(np.mean(v)), median=float(np.median(v)))
            return item
        for level in ('P2','P3'):
            for lo, hi, name in ((0,math.inf,'all'),(0,8,'lt8'),(8,16,'8_16'),(16,24,'16_24'),
                                 (24,32,'24_32'),(32,math.inf,'ge32')):
                items = [r for r in self.rows if r['level']==level and lo<=r['size']<hi]
                output[level+'/'+name]=summarize(items)
            for cls in range(1,11):
                output[level+'/class_'+str(cls)]=summarize([r for r in self.rows if r['level']==level and r['cls']==cls])
        return output
