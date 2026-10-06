"""Check loss algebra, gradient scope and real Detect integration."""
from __future__ import annotations
import copy
from types import SimpleNamespace
import torch
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from e37_loss import MixedNWDBboxLoss, MixedNWDDetectionLoss, nwd_loss, FIXED, REVISION


def require(value, message):
    if not value:
        raise RuntimeError(message)


def algebra_audit(device='cpu'):
    torch.manual_seed(37001)
    boxes = torch.tensor([[0.,0.,8.,8.], [30.,30.,38.,38.], [1.,1.,9.,9.]], device=device, requires_grad=True)
    targets = torch.tensor([[0.,0.,8.,8.], [0.,0.,8.,8.], [0.,0.,8.,8.]], device=device)
    loss = nwd_loss(boxes, targets, 8.)
    require(loss[0] == 0 and loss[1] > loss[2] > 0, 'NWD equality/disjoint ordering failed')
    loss.sum().backward()
    require(torch.isfinite(boxes.grad).all() and boxes.grad[0].abs().sum() == 0,
            'NWD equal-box gradient is not finite/zero')
    # Four P2 anchors plus one anchor for each deeper scale. Same input-pixel
    # boxes at P3 are deliberately included: only P2 can receive the change.
    counts, strides = (4,1,1,1), (4.,8.,16.,32.)
    stride = torch.tensor([4.,4.,4.,4.,8.,16.,32.], device=device)
    pixel_gt = torch.tensor([[8.,8.,16.,16.], [8.,8.,24.,24.], [8.,8.,16.,16.],
                             [8.,8.,20.,20.], [8.,8.,16.,16.], [8.,8.,16.,16.], [8.,8.,16.,16.]], device=device)
    pred = (pixel_gt + torch.tensor([.5,-.5,.5,-.5], device=device))/stride[:,None]
    gt = (pixel_gt/stride[:,None]).unsqueeze(0)
    pred = pred.unsqueeze(0).requires_grad_()
    points = (gt[0,:,:2]+gt[0,:,2:])*.5
    dist = torch.randn(1,7,64,device=device,requires_grad=True)
    scores = torch.zeros(1,7,10,device=device); scores[:,:,0] = .8
    fg = torch.tensor([[True,True,False,True,True,True,True]],device=device)
    denominator = scores.sum().clamp_min(1)
    stock = BboxLoss(16).to(device)
    mixed = MixedNWDBboxLoss(16,8.).to(device)
    mixed.level_counts, mixed.level_strides = counts, strides
    args = (dist,pred,points,gt,scores,denominator,fg)
    old, old_dfl = stock(*args)
    new, new_dfl = mixed(*args)
    require(mixed.last['selected'] == 2, 'P2/tiny/strict-16px/foreground routing failed')
    torch.testing.assert_close(new_dfl,old_dfl,rtol=0,atol=0)
    old_grad = torch.autograd.grad(old,pred,retain_graph=True)[0]
    new_grad = torch.autograd.grad(new,pred,retain_graph=True)[0]
    difference = new_grad-old_grad
    require(difference[:,[1,2,4,5,6]].abs().max() == 0, 'Nonselected box gradients changed')
    require(difference[:,[0,3]].abs().sum() > 0, 'Selected boxes received no changed gradient')
    from ultralytics.utils.metrics import bbox_iou
    select = torch.tensor([0,3],device=device)
    ciou = 1-bbox_iou(pred[0,select],gt[0,select],xywh=False,CIoU=True).view(-1)
    nwd = nwd_loss(pred[0,select]*stride[select,None], pixel_gt[select],8.)
    expected = old + .5*((nwd-ciou)*.8).sum()/denominator
    torch.testing.assert_close(new,expected)
    mixed.mix = 0
    disabled, disabled_dfl = mixed(*args)
    torch.testing.assert_close(disabled,old,rtol=0,atol=0)
    torch.testing.assert_close(disabled_dfl,old_dfl,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(disabled,pred)[0],old_grad,rtol=0,atol=0)
    mixed.mix = .5
    # No eligible P2 GT: bypass must be exact, not merely approximately zero.
    other = gt.clone(); other[:, :4, 2:] = other[:,:4,:2] + 5
    original_bypass = stock(dist,pred,points,other,scores,denominator,fg)
    mixed_bypass = mixed(dist,pred,points,other,scores,denominator,fg)
    require(mixed.last['selected'] == 0, 'Large P2 targets did not bypass')
    for a,b in zip(original_bypass,mixed_bypass): torch.testing.assert_close(a,b,rtol=0,atol=0)
    return dict(status='passed', device=device, gaussian_variance='size**2/4', identical_boxes_finite=True,
                disjoint_boxes_finite=True, exact_disabled_loss_gradient=True,
                pixel_units=True, selected_only_box_gradient=True, DFL_exact=True, strict_16px=True)


def detection_audit(reference, constant, device='cpu', amp=False):
    model = copy.deepcopy(reference).float().to(device).train()
    model.args = copy.deepcopy(DEFAULT_CFG)
    model.yaml['e37_nwd'] = dict(revision=REVISION, constant_px=constant, **FIXED)
    base, custom = v8DetectionLoss(model), MixedNWDDetectionLoss(model)
    img = torch.rand(2,3,128,160,device=device)
    batch = dict(img=img,batch_idx=torch.tensor([0.,1.],device=device),
                 cls=torch.tensor([[0.],[3.]],device=device),
                 bboxes=torch.tensor([[.51,.49,.05,.0625],[.51,.49,.05,.0625]],device=device))
    with torch.autocast(device_type=torch.device(device).type,enabled=amp):
        maps = model(img)
    detached = [x.detach().clone().requires_grad_() for x in maps]
    old, old_items = base(detached,batch)
    custom.bbox_loss.mix = 0
    off, off_items = custom(detached,batch)
    torch.testing.assert_close(off,old,rtol=0,atol=0)
    torch.testing.assert_close(off_items,old_items,rtol=0,atol=0)
    grad_old = torch.autograd.grad(old,detached,retain_graph=True)
    grad_off = torch.autograd.grad(off,detached,retain_graph=True)
    for a,b in zip(grad_old,grad_off): torch.testing.assert_close(a,b,rtol=0,atol=0)
    custom.bbox_loss.mix = .5
    active, active_items = custom(detached,batch)
    require(custom.bbox_loss.last.get('selected',0)>0,'Real TAL/Detect integration selected no P2-tiny positives')
    torch.testing.assert_close(active_items[1:],old_items[1:],rtol=0,atol=0)
    grad_active = torch.autograd.grad(active,detached)
    for level,(a,b) in enumerate(zip(grad_active,grad_old)):
        require(torch.isfinite(a).all(),'Nonfinite prediction gradient')
        torch.testing.assert_close(a[:,64:],b[:,64:],rtol=0,atol=0)
        if level>0: torch.testing.assert_close(a,b,rtol=0,atol=0)
    require((grad_active[0][:,:64]-grad_old[0][:,:64]).abs().sum()>0,'P2 regression gradient did not change')
    model.zero_grad(set_to_none=True)
    criterion = MixedNWDDetectionLoss(model)
    optimizer = torch.optim.SGD(model.parameters(),lr=.001)
    before = model.model[26].cv2[0][-1].weight.detach().clone()
    with torch.autocast(device_type=torch.device(device).type,enabled=amp):
        actual_loss,_ = criterion(model(img),batch)
    actual_loss.backward()
    require(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),'Nonfinite real model gradient')
    optimizer.step()
    require(not torch.equal(before,model.model[26].cv2[0][-1].weight),'Real box tower did not update')
    # Empty GT and all-large GT must leave ordinary detection loss intact.
    blank = {**batch,'batch_idx':batch['batch_idx'][:0],'cls':batch['cls'][:0],'bboxes':batch['bboxes'][:0]}
    with torch.no_grad():
        a,ai=base(maps,blank); b,bi=custom(maps,blank)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(ai,bi,rtol=0,atol=0)
    return dict(status='passed',device=device,amp=amp, initial_prediction_topology='E1',
                disabled_loss_and_raw_gradient_exact=True, classification_and_DFL_exact=True,
                P3_P4_P5_raw_gradient_exact=True, real_P2_update=True, empty_batch=True,
                scope_note='Shared feature gradients/actual TAL assignments may change after optimization.')
