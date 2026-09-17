"""Training objective for E20a scale-adaptive distribution-quality calibration."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

__all__ = ("v8SDQCDetectionLoss",)


class v8SDQCDetectionLoss(v8DetectionLoss):
    """Keep TAL/DFL intact and add positive-only scale-adaptive quality learning."""

    def __init__(self, model, tal_topk=10):
        super().__init__(model, tal_topk=tal_topk)
        self.head = model.model[-1]
        config = model.yaml.get("sdqc", {})
        self.quality_gain = float(config.get("quality_gain", 0.25))
        self.nwd_constant = float(config.get("nwd_constant", 12.8))
        self.scale_center = float(config.get("scale_center", 32.0))
        self.scale_temperature = float(config.get("scale_temperature", 8.0))
        if self.quality_gain <= 0 or self.nwd_constant <= 0 or self.scale_temperature <= 0:
            raise ValueError("SDQC loss constants must be positive")
        self.no = self.nc + self.reg_max * 4 + 1
        self.last_diagnostics = {}

    @staticmethod
    def _aligned_iou(box1, box2):
        """Aligned IoU for two Nx4 xyxy tensors."""
        left_top = torch.maximum(box1[:, :2], box2[:, :2])
        right_bottom = torch.minimum(box1[:, 2:], box2[:, 2:])
        intersection = (right_bottom - left_top).clamp_min(0).prod(1)
        area1 = (box1[:, 2:] - box1[:, :2]).clamp_min(0).prod(1)
        area2 = (box2[:, 2:] - box2[:, :2]).clamp_min(0).prod(1)
        return intersection / (area1 + area2 - intersection).clamp_min(1e-7)

    def _aligned_nwd(self, box1, box2):
        """Normalized Gaussian-Wasserstein similarity for aligned xyxy boxes."""
        center1 = (box1[:, :2] + box1[:, 2:]) * 0.5
        center2 = (box2[:, :2] + box2[:, 2:]) * 0.5
        size1 = (box1[:, 2:] - box1[:, :2]).clamp_min(0)
        size2 = (box2[:, 2:] - box2[:, :2]).clamp_min(0)
        wasserstein2 = (center1 - center2).square().sum(1) + (size1 - size2).square().sum(1) / 12.0
        return torch.exp(-wasserstein2.clamp_min(0).sqrt() / self.nwd_constant)

    def _quality_targets(self, pred_boxes_px, target_boxes_px, fg_mask):
        target = pred_boxes_px.new_zeros(fg_mask.shape)
        if not fg_mask.any():
            return target
        predicted = pred_boxes_px[fg_mask].detach().float()
        assigned = target_boxes_px[fg_mask].detach().float()
        iou = self._aligned_iou(predicted, assigned)
        nwd = self._aligned_nwd(predicted, assigned)
        size = (assigned[:, 2:] - assigned[:, :2]).clamp_min(0).prod(1).sqrt()
        mix = torch.sigmoid((self.scale_center - size) / self.scale_temperature)
        quality = ((1.0 - mix) * iou + mix * nwd).clamp_(0.0, 1.0)
        target[fg_mask] = quality.to(target.dtype)
        return target

    def __call__(self, preds, batch):
        """Return the standard three logged components with quality included in cls."""
        loss = torch.zeros(3, device=self.device, dtype=torch.float32)  # box, cls+quality, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        concatenated = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2)
        pred_distri, pred_scores, pred_quality = concatenated.split((self.reg_max * 4, self.nc, 1), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_quality = pred_quality.permute(0, 2, 1).contiguous().squeeze(-1)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        pred_bboxes_px = pred_bboxes.detach() * stride_tensor
        _, target_bboxes_px, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            pred_bboxes_px.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = target_scores.sum().clamp_min(1.0)

        quality_target = self._quality_targets(pred_bboxes_px, target_bboxes_px, fg_mask)
        quality_probability = pred_quality.float().sigmoid()
        gamma = self.head.quality_gamma.float()

        # The quality value is detached in the class objective: the classifier
        # learns with the same calibrated score used at inference, while the
        # quality predictor remains a pure localization-quality estimator.
        class_probability = pred_scores.float().sigmoid()
        joint_probability = class_probability * quality_probability.detach().unsqueeze(-1).pow(gamma)
        joint_probability = joint_probability.clamp(1e-6, 1.0 - 1e-6)
        joint_logit = torch.logit(joint_probability)
        classification = F.binary_cross_entropy_with_logits(
            joint_logit, target_scores.float(), reduction="sum"
        ) / target_scores_sum

        if fg_mask.any():
            quality = F.binary_cross_entropy_with_logits(
                pred_quality.float()[fg_mask], quality_target.float()[fg_mask], reduction="mean"
            )
        else:
            quality = pred_quality.float().sum() * 0.0

        if fg_mask.any():
            target_bboxes = target_bboxes_px / stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box
        loss[1] = classification * self.hyp.cls + quality * self.quality_gain
        loss[2] *= self.hyp.dfl
        self.last_diagnostics = {
            "positive_count": int(fg_mask.sum().detach()),
            "quality_loss": float(quality.detach()),
            "quality_target_mean": float(quality_target[fg_mask].mean().detach()) if fg_mask.any() else None,
            "quality_prediction_mean": float(quality_probability[fg_mask].mean().detach()) if fg_mask.any() else None,
            "gamma": float(gamma.detach()),
        }
        return loss.sum() * batch_size, loss.detach()
