"""Scale-weighted target-centre supervision for E11a CBSG gates."""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss


class v8CBSGDetectionLoss(v8DetectionLoss):
    """Standard YOLO detection loss plus two lightweight gate losses.

    Targets are built directly from augmented training boxes.  Each object
    contributes a compact centre neighbourhood; boxes below 32x32 pixels at
    the 1024 training canvas receive higher positive weight.  No auxiliary
    target or loss exists during deployment.
    """

    def __init__(self, model, tal_topk=10):
        if not hasattr(model, "args"):
            model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        super().__init__(model, tal_topk=tal_topk)
        config = model.yaml.get("cbsg", {})
        if not isinstance(config, dict) or not config.get("enabled", False):
            raise ValueError("v8CBSGDetectionLoss requires cbsg.enabled=true")
        deep = [module for module in model.modules() if module.__class__.__name__ == "CBSGDeepSemanticGuide"]
        detail = [module for module in model.modules() if module.__class__.__name__ == "CBSGDetailGate"]
        if len(deep) != 1 or len(detail) != 1:
            raise ValueError(f"Expected one CBSG deep/detail module, found {len(deep)}/{len(detail)}")
        self.p3_gate = deep[0].spatial_gate
        self.p2_gate = detail[0].spatial_gate
        self.p3_weight = float(config.get("p3_loss_weight", 0.02))
        self.p2_weight = float(config.get("p2_loss_weight", 0.04))
        self.focal_gamma = float(config.get("focal_gamma", 2.0))
        self.negative_balance = float(config.get("negative_balance", 0.25))
        self.small_boost = float(config.get("small_boost", 1.5))
        self.p3_dilation = int(config.get("p3_dilation", 3))
        self.p2_dilation = int(config.get("p2_dilation", 5))
        if min(self.p3_weight, self.p2_weight, self.negative_balance) <= 0:
            raise ValueError("CBSG loss weights must be positive")
        if self.focal_gamma < 0 or self.small_boost < 0:
            raise ValueError("Invalid CBSG focal_gamma/small_boost")
        if any(value < 1 or value % 2 == 0 for value in (self.p3_dilation, self.p2_dilation)):
            raise ValueError("CBSG dilations must be positive odd integers")
        self.last_gate_stats = {}

    def _centre_target(self, logits, batch, stride, dilation):
        batch_size, _, height, width = logits.shape
        device = logits.device
        dtype = logits.dtype
        target_flat = torch.zeros(batch_size * height * width, device=device, dtype=torch.float32)
        weight_flat = torch.zeros_like(target_flat)
        boxes = batch["bboxes"].to(device=device, dtype=torch.float32)
        indices = batch["batch_idx"].to(device=device, dtype=torch.long).view(-1)
        if boxes.numel():
            cx = (boxes[:, 0] * width).long().clamp_(0, width - 1)
            cy = (boxes[:, 1] * height).long().clamp_(0, height - 1)
            flat_index = indices * (height * width) + cy * width + cx
            input_h, input_w = height * stride, width * stride
            side = (boxes[:, 2] * input_w * boxes[:, 3] * input_h).clamp_min(1e-6).sqrt()
            tiny_emphasis = 1.0 + self.small_boost * (1.0 - side / 32.0).clamp_(0.0, 1.0)
            target_flat.scatter_reduce_(0, flat_index, torch.ones_like(tiny_emphasis), reduce="amax", include_self=True)
            weight_flat.scatter_reduce_(0, flat_index, tiny_emphasis, reduce="amax", include_self=True)
        target = target_flat.view(batch_size, 1, height, width)
        positive_weight = weight_flat.view(batch_size, 1, height, width)
        if dilation > 1:
            target = F.max_pool2d(target, dilation, stride=1, padding=dilation // 2)
            positive_weight = F.max_pool2d(positive_weight, dilation, stride=1, padding=dilation // 2)
        positive_weight = torch.where(target > 0, positive_weight.clamp_min(1.0), torch.zeros_like(positive_weight))
        return target.to(dtype=dtype), positive_weight.to(dtype=dtype)

    def _balanced_focal(self, logits, target, positive_weight):
        logits = logits.float()
        target = target.float()
        positive_weight = positive_weight.float()
        probability = logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
        positive = target > 0.5
        negative = ~positive
        pos_term = F.softplus(-logits) * (1.0 - probability).pow(self.focal_gamma) * positive_weight
        neg_term = F.softplus(logits) * probability.pow(self.focal_gamma)
        pos_loss = pos_term.sum() / positive_weight.sum().clamp_min(1.0)
        neg_loss = (neg_term * negative).sum() / negative.sum().clamp_min(1)
        return pos_loss + self.negative_balance * neg_loss

    def __call__(self, preds, batch):
        total, items = super().__call__(preds, batch)
        p3_logits = self.p3_gate.pop_gate_logits()
        p2_logits = self.p2_gate.pop_gate_logits()
        if p3_logits is None or p2_logits is None:
            raise RuntimeError("CBSG forward did not expose both spatial-gate logits")
        p3_target, p3_positive_weight = self._centre_target(p3_logits, batch, stride=8, dilation=self.p3_dilation)
        p2_target, p2_positive_weight = self._centre_target(p2_logits, batch, stride=4, dilation=self.p2_dilation)
        p3_raw = self._balanced_focal(p3_logits, p3_target, p3_positive_weight)
        p2_raw = self._balanced_focal(p2_logits, p2_target, p2_positive_weight)
        auxiliary = self.p3_weight * p3_raw + self.p2_weight * p2_raw
        batch_size = int(p2_logits.shape[0])
        with torch.no_grad():
            self.last_gate_stats = {
                "p3_positive_fraction": p3_target.mean().detach(),
                "p2_positive_fraction": p2_target.mean().detach(),
                "p3_positive_probability": p3_logits.sigmoid()[p3_target > 0.5].mean().detach(),
                "p3_negative_probability": p3_logits.sigmoid()[p3_target <= 0.5].mean().detach(),
                "p2_positive_probability": p2_logits.sigmoid()[p2_target > 0.5].mean().detach(),
                "p2_negative_probability": p2_logits.sigmoid()[p2_target <= 0.5].mean().detach(),
                "p3_raw_loss": p3_raw.detach(), "p2_raw_loss": p2_raw.detach(),
                "weighted_auxiliary": auxiliary.detach(),
            }
        # Keep the standard 3-item loss vector so Ultralytics logging and final
        # validation remain directly comparable. Auxiliary loss affects total.
        return total + auxiliary * batch_size, items


__all__ = ("v8CBSGDetectionLoss",)
