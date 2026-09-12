# Ultralytics YOLO, AGPL-3.0 license
"""TAL-assignment-guided reliability supervision for E9c AGPGALP."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

__all__ = ("v8AGPGALPDetectionLoss",)


class v8AGPGALPDetectionLoss(v8DetectionLoss):
    """Standard detection loss plus reliability supervision from P2 TAL positives."""

    def __init__(self, model, tal_topk=10):
        # DetectionTrainer injects these before the first training loss. Direct
        # installer/preflight models need the exact frozen protocol gains here.
        if not hasattr(model, "args"):
            model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        super().__init__(model, tal_topk=tal_topk)
        config = model.yaml.get("ag_pgalp", {})
        if not isinstance(config, dict) or not config.get("enabled", False):
            raise ValueError("v8AGPGALPDetectionLoss requires ag_pgalp.enabled=true")
        modules = [m for m in model.modules() if m.__class__.__name__ == "AGPGALP"]
        if len(modules) != 1:
            raise ValueError(f"Expected exactly one AGPGALP, found {len(modules)}")
        self.reliability_module = modules[0]
        self.reliability_weight = float(config.get("loss_weight", 0.05))
        self.focal_gamma = float(config.get("focal_gamma", 2.0))
        self.dilation = int(config.get("dilation", 3))
        if self.reliability_weight <= 0 or self.focal_gamma < 0:
            raise ValueError("Invalid AGPGALP reliability loss configuration")
        if self.dilation < 1 or self.dilation % 2 == 0:
            raise ValueError("ag_pgalp.dilation must be a positive odd integer")
        self.last_reliability_stats = {}

    def reliability_loss(self, logits, target):
        probability = logits.float().sigmoid().clamp(1e-6, 1.0 - 1e-6)
        positive = target > 0.5
        negative = ~positive
        pos_term = -((1.0 - probability).pow(self.focal_gamma) * probability.log())
        neg_term = -(probability.pow(self.focal_gamma) * (1.0 - probability).log())
        pos_loss = (pos_term * positive).sum() / positive.sum().clamp_min(1)
        neg_loss = (neg_term * negative).sum() / negative.sum().clamp_min(1)
        return pos_loss + neg_loss

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        logits = self.reliability_module.pop_reliability_logits()
        if logits is None:
            raise RuntimeError("AGPGALP did not expose reliability logits for this loss forward")
        height, width = feats[0].shape[-2:]
        p2_count = height * width
        if logits.shape != (batch_size, 1, height, width):
            raise RuntimeError(f"Reliability/P2 shape mismatch: logits={tuple(logits.shape)}, P2={(height, width)}")
        target = fg_mask[:, :p2_count].reshape(batch_size, 1, height, width).float().detach()
        if self.dilation > 1:
            target = F.max_pool2d(target, self.dilation, stride=1, padding=self.dilation // 2)
        raw_reliability = self.reliability_loss(logits, target)
        weighted_reliability = raw_reliability * self.reliability_weight
        with torch.no_grad():
            probability = logits.float().sigmoid()
            positive = target > 0.5
            negative = ~positive
            self.last_reliability_stats = {
                "positive_cells": positive.sum().detach(),
                "positive_fraction": positive.float().mean().detach(),
                "positive_probability": probability[positive].mean().detach() if positive.any() else probability.new_tensor(0.0),
                "negative_probability": probability[negative].mean().detach() if negative.any() else probability.new_tensor(0.0),
                "raw_loss": raw_reliability.detach(),
                "weighted_loss": weighted_reliability.detach(),
            }
        total = loss.sum() * batch_size + weighted_reliability * batch_size
        items = torch.cat((loss.detach(), weighted_reliability.detach().reshape(1)))
        return total, items

