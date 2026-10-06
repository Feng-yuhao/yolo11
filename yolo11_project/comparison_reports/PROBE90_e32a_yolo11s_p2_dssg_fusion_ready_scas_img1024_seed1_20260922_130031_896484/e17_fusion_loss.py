"""Training-only scale-valid auxiliary supervision for E17a lateral features."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from ultralytics.utils.loss import v8DetectionLoss


class v8E17DetectionLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        settings = model.yaml.get("e17_aux", {})
        self.gain = float(settings.get("gain", 0.1))
        self.minimum_side = float(settings.get("min_feature_side", 1.5))
        self.negative_weight = float(settings.get("negative_weight", 0.1))
        self.box_weight = float(settings.get("box_weight", 0.25))
        if not 0 <= self.gain <= 1 or self.minimum_side <= 0:
            raise ValueError("Invalid E17a auxiliary settings")
        self.head = model.model[-1]

    def __call__(self, preds, batch):
        logits = self.head.take_aux_logits()
        if logits is None and not self.head.training:
            # Ultralytics also calls the criterion during eval-mode validation.
            # Auxiliary predictions are intentionally absent there: report only
            # the unchanged main box/cls/dfl loss, with no auxiliary gradients.
            return super().__call__(preds, batch)
        try:
            main, items = super().__call__(preds, batch)
            if logits is None:
                raise RuntimeError("Training E17Detect did not provide auxiliary logits")
            image_size = batch["img"].shape[-2:]
            auxiliary = sum(
                self._level_loss(level, batch, image_size) for level in logits
            ) / len(logits)
            if not torch.isfinite(auxiliary):
                raise FloatingPointError("Non-finite E17 auxiliary loss")
            self.head.last_aux_loss = auxiliary.detach()
            return main + self.gain * auxiliary * batch["img"].shape[0], items
        finally:
            self.head.clear_prior_logits()

    def _level_loss(self, logits, batch, image_size):
        b, channels, grid_h, grid_w = logits.shape
        nc = self.nc
        if channels != nc + 4:
            raise ValueError("Unexpected E17 auxiliary channel count")
        del image_size
        gt_xywh = batch["bboxes"].to(logits.device, dtype=torch.float32)
        gt_class = batch["cls"].reshape(-1).to(logits.device, dtype=torch.long)
        gt_image = batch["batch_idx"].reshape(-1).to(logits.device, dtype=torch.long)
        valid = (
            (gt_xywh[:, 2] * grid_w >= self.minimum_side)
            & (gt_xywh[:, 3] * grid_h >= self.minimum_side)
            & (gt_xywh[:, 2] > 0) & (gt_xywh[:, 3] > 0)
            & torch.isfinite(gt_xywh).all(1)
            & (gt_class >= 0) & (gt_class < nc)
            & (gt_image >= 0) & (gt_image < b)
        )
        gt_xywh, gt_class, gt_image = gt_xywh[valid], gt_class[valid], gt_image[valid]
        # Targets use the augmented-image normalized coordinates already consumed by main YOLO loss.
        if len(gt_xywh):
            x = (gt_xywh[:, 0] * grid_w).floor().clamp(0, grid_w - 1).long()
            y = (gt_xywh[:, 1] * grid_h).floor().clamp(0, grid_h - 1).long()
            positive = F.softplus(-logits[gt_image, gt_class, y, x]).mean()
        else:
            positive = logits[:, :nc].sum() * 0.0

        # Do not train the center's immediate neighborhood as background.
        allowed_negative = torch.ones((b, 1, grid_h, grid_w), device=logits.device, dtype=torch.bool)
        if len(gt_xywh):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    allowed_negative[gt_image, 0, (y + dy).clamp(0, grid_h - 1), (x + dx).clamp(0, grid_w - 1)] = False
        negative_values = F.softplus(logits[:, :nc])[allowed_negative.expand(-1, nc, -1, -1)]
        negative = negative_values.mean() if negative_values.numel() else logits[:, :nc].sum() * 0.0

        regression = logits[:, nc:].sum() * 0.0
        if len(gt_xywh):
            # Same-cell crowds are common in VisDrone. Use each cell's largest scale-valid box
            # only for auxiliary regression; all centers still supervise classification.
            keys = (gt_image * grid_h + y) * grid_w + x
            _, inverse = torch.unique(keys, return_inverse=True)
            groups = int(inverse.max().item()) + 1
            areas = gt_xywh[:, 2] * gt_xywh[:, 3]
            largest = torch.full((groups,), -1.0, device=logits.device)
            largest.scatter_reduce_(0, inverse, areas, reduce="amax", include_self=True)
            candidate = areas == largest[inverse]
            ranks = torch.arange(len(keys), device=logits.device)
            ranks = torch.where(candidate, ranks, len(keys))
            take = torch.full((groups,), len(keys), device=logits.device, dtype=torch.long)
            take.scatter_reduce_(0, inverse, ranks, reduce="amin", include_self=True)
            cx = (x[take].float() + 0.5) / grid_w
            cy = (y[take].float() + 0.5) / grid_h
            boxes = gt_xywh[take]
            distances = torch.stack((
                (cx - boxes[:, 0] + boxes[:, 2] / 2) * grid_w,
                (cy - boxes[:, 1] + boxes[:, 3] / 2) * grid_h,
                (boxes[:, 0] + boxes[:, 2] / 2 - cx) * grid_w,
                (boxes[:, 1] + boxes[:, 3] / 2 - cy) * grid_h,
            ), dim=1).clamp_min(0)
            target = torch.log1p(distances)
            predicted = logits[gt_image[take], nc:, y[take], x[take]]
            regression = F.smooth_l1_loss(predicted.float(), target)
        return positive + self.negative_weight * negative + self.box_weight * regression
