"""Combined E16a tiny-router supervision and E20 SDQC objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils.sdqc_loss import v8SDQCDetectionLoss

__all__ = ("v8TSDRSDQCDetectionLoss",)


class v8TSDRSDQCDetectionLoss(v8SDQCDetectionLoss):
    def __init__(self, model, gate_gain=0.05, tiny_equivalent_side=32.0):
        super().__init__(model)
        self.gate_gain = float(gate_gain)
        self.tiny_equivalent_side = float(tiny_equivalent_side)
        if self.head.__class__.__name__ != "TSDRSDQCDetect":
            raise TypeError("Combined loss requires TSDRSDQCDetect")
        if not 0.0 < self.gate_gain <= 1.0 or self.tiny_equivalent_side <= 0:
            raise ValueError("Invalid E20b TSDR supervision constants")

    @torch.no_grad()
    def _positive_mask(self, logits, batch):
        batch_size, _, height, width = logits.shape
        boxes = batch["bboxes"].detach().to(device=logits.device, dtype=logits.dtype)
        batch_idx = batch["batch_idx"].detach().view(-1).to(device=logits.device, dtype=torch.long)
        image_h, image_w = (int(value) for value in batch["img"].shape[-2:])
        mask = torch.zeros(batch_size * height * width, device=logits.device, dtype=torch.bool)
        if boxes.numel() == 0:
            return mask.view(batch_size, 1, height, width), 0
        side = torch.sqrt((boxes[:, 2] * image_w).clamp_min(0) * (boxes[:, 3] * image_h).clamp_min(0))
        tiny = side <= self.tiny_equivalent_side
        if not bool(tiny.any()):
            return mask.view(batch_size, 1, height, width), 0
        selected, selected_batch = boxes[tiny], batch_idx[tiny]
        center_x = (selected[:, 0] * width).long().clamp_(0, width - 1)
        center_y = (selected[:, 1] * height).long().clamp_(0, height - 1)
        offsets = torch.tensor(
            [[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]],
            device=logits.device, dtype=torch.long,
        )
        xs = (center_x[:, None] + offsets[None, :, 0]).clamp_(0, width - 1)
        ys = (center_y[:, None] + offsets[None, :, 1]).clamp_(0, height - 1)
        linear = selected_batch[:, None] * (height * width) + ys * width + xs
        mask[linear.reshape(-1)] = True
        return mask.view(batch_size, 1, height, width), int(tiny.sum())

    @staticmethod
    def _router_loss(logits, positive_mask):
        positive = logits[positive_mask]
        negative = logits[~positive_mask]
        positive_loss = (
            ((1.0 - positive.sigmoid()).pow(2) * F.softplus(-positive)).mean()
            if positive.numel() else logits.sum() * 0.0
        )
        negative_loss = (
            (negative.sigmoid().pow(2) * F.softplus(negative)).mean()
            if negative.numel() else logits.sum() * 0.0
        )
        return positive_loss + 0.25 * negative_loss

    def __call__(self, preds, batch):
        base_total, base_items = super().__call__(preds, batch)
        logits = self.head.pop_gate_logits()
        if logits is None:
            self.head._last_aux_gate_loss = None
            return base_total, base_items
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise RuntimeError(f"E20b TSDR router logits must be Bx1xHxW, got {tuple(logits.shape)}")
        positive_mask, tiny_count = self._positive_mask(logits, batch)
        gate_loss = self._router_loss(logits, positive_mask)
        if not torch.isfinite(gate_loss):
            raise RuntimeError("E20b TSDR router loss is non-finite")
        self.head._last_aux_gate_loss = {
            "loss": gate_loss.detach(),
            "tiny_objects": tiny_count,
            "positive_locations": int(positive_mask.sum()),
        }
        return base_total + self.gate_gain * gate_loss * int(logits.shape[0]), base_items
