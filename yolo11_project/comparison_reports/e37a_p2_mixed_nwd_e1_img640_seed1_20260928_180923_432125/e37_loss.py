"""E37a: replace half of the P2-tiny CIoU term with pixel-space NWD.

No auxiliary classifier, score calibration, label-assignment change, or extra
DFL weighting. The standard detector constructs the exact same TAL targets.
"""
from __future__ import annotations

import math
import torch
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou

REVISION = 'e37a_p2_tiny_mixed_nwd_v1'
FIXED = dict(mix=0.5, tiny_side_px=16.0, distance_epsilon_px=1e-6)


def nwd_loss(box1, box2, constant):
    """Aligned XYXY boxes in INPUT PIXELS; Gaussian std=(w/2,h/2).

    sqrt(d2+eps**2)-eps preserves zero loss/finite gradients for equal boxes.
    /4 is the Gaussian box convention in NWD, not the uniform-box /12 used
    by the old SDQC quality-target implementation.
    """
    if not math.isfinite(float(constant)) or constant <= 0:
        raise ValueError('NWD constant must be finite and positive')
    a, b = box1.float(), box2.float()
    centers = ((a[..., :2] + a[..., 2:]) - (b[..., :2] + b[..., 2:])) * 0.5
    sizes = (a[..., 2:] - a[..., :2]) - (b[..., 2:] - b[..., :2])
    distance2 = centers.square().sum(-1) + sizes.square().sum(-1) * 0.25
    eps = FIXED['distance_epsilon_px']
    distance = (distance2 + eps * eps).sqrt() - eps
    return -torch.expm1(-distance / float(constant))


class MixedNWDBboxLoss(BboxLoss):
    def __init__(self, reg_max, constant, mix=0.5):
        super().__init__(reg_max)
        self.constant = float(constant)
        self.mix = float(mix)
        if not math.isfinite(self.constant) or self.constant <= 0 or not 0 <= self.mix <= 1:
            raise ValueError('Invalid mixed-NWD configuration')
        self.level_counts = None
        self.level_strides = None
        self.last = {}
        self.totals = dict(calls=0, positives=0, selected=0, selected_score_sum=0.0,
                           selected_ciou_weighted_sum=0.0, selected_nwd_weighted_sum=0.0)

    def set_levels(self, maps, strides):
        self.level_counts = tuple(int(x.shape[-2] * x.shape[-1]) for x in maps)
        self.level_strides = tuple(float(s) for s in strides)
        if len(self.level_counts) != 4 or self.level_strides != (4., 8., 16., 32.):
            raise RuntimeError('E37a requires the original four-level E1 head')

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask):
        original, dfl = super().forward(pred_dist, pred_bboxes, anchor_points,
                                      target_bboxes, target_scores, target_scores_sum, fg_mask)
        if self.mix == 0:
            self.last = dict(selected=0, disabled=True)
            return original, dfl
        if self.level_counts is None or sum(self.level_counts) != pred_bboxes.shape[1]:
            raise RuntimeError('Missing/inconsistent actual detection-map dimensions')
        indices = fg_mask.nonzero(as_tuple=False)[:, 1]
        stride_all = torch.cat([pred_bboxes.new_full((n,), s)
                                for n, s in zip(self.level_counts, self.level_strides)])
        strides = stride_all[indices]
        predicted = pred_bboxes[fg_mask].float() * strides[:, None]
        assigned = target_bboxes[fg_mask].float() * strides[:, None]
        sides = assigned[:, 2:] - assigned[:, :2]
        size = sides.clamp_min(0).prod(-1).sqrt()
        selected = ((indices < self.level_counts[0]) & (sides > 0).all(-1)
                    & (size < FIXED['tiny_side_px']))
        count = int(selected.sum().detach())
        weight = target_scores.sum(-1)[fg_mask].float()[selected]
        self.last = dict(positives=int(indices.numel()), selected=count)
        self.totals['calls'] += 1
        self.totals['positives'] += int(indices.numel())
        self.totals['selected'] += count
        if not count:
            return original, dfl
        # Use cell-space CIoU, exactly as the original criterion does. NWD is
        # computed in pixels; target selection is GT-based and has no gradient.
        ciou = 1 - bbox_iou(pred_bboxes[fg_mask][selected],
                            target_bboxes[fg_mask][selected], xywh=False, CIoU=True).view(-1)
        nwd = nwd_loss(predicted[selected], assigned[selected], self.constant)
        correction = self.mix * ((nwd - ciou) * weight).sum() / target_scores_sum
        if not torch.isfinite(correction):
            raise FloatingPointError('Nonfinite mixed-NWD correction')
        self.last.update(ciou=float((ciou.detach().float()*weight).sum()),
                         nwd=float((nwd.detach()*weight).sum()),
                         correction=float(correction.detach()))
        self.totals['selected_score_sum'] += float(weight.sum().detach())
        self.totals['selected_ciou_weighted_sum'] += self.last['ciou']
        self.totals['selected_nwd_weighted_sum'] += self.last['nwd']
        return original + correction, dfl


class MixedNWDDetectionLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        config = model.yaml.get('e37_nwd')
        if not config or config.get('revision') != REVISION:
            raise RuntimeError('Missing E37a training-set calibration in model metadata')
        self.bbox_loss = MixedNWDBboxLoss(model.model[-1].reg_max,
                                        config['constant_px'], config['mix']).to(self.device)

    def __call__(self, preds, batch):
        maps = preds[1] if isinstance(preds, tuple) else preds
        self.bbox_loss.set_levels(maps, self.stride)
        return super().__call__(preds, batch)
