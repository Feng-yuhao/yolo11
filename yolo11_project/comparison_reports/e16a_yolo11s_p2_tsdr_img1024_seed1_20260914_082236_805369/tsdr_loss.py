"""Training-only tiny-center supervision for the E16a TSDR router."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss


class v8TSDRDetectionLoss(v8DetectionLoss):
    """Standard TAL2 loss plus an isolated, vectorized tiny-router loss.

    Tiny centers and their 3x3 P2 neighborhoods are positive. All remaining
    positions receive focal background supervision. The target uses boxes
    after the current geometric augmentation and never modifies TAL matching.
    """

    def __init__(self, model, gate_gain=0.05, tiny_equivalent_side=32.0):
        super().__init__(model)
        self.head = model.model[-1]
        self.gate_gain = float(gate_gain)
        self.tiny_equivalent_side = float(tiny_equivalent_side)
        if self.head.__class__.__name__ != "TSDRDetect":
            raise TypeError("v8TSDRDetectionLoss requires TSDRDetect as the final layer")
        if not 0.0 < self.gate_gain <= 1.0:
            raise ValueError("TSDR gate loss gain must be in (0, 1]")
        if self.tiny_equivalent_side <= 0.0:
            raise ValueError("tiny_equivalent_side must be positive")

    @torch.no_grad()
    def _positive_mask(self, logits, batch):
        batch_size, _, height, width = logits.shape
        boxes = batch["bboxes"].detach().to(device=logits.device, dtype=logits.dtype)
        batch_idx = batch["batch_idx"].detach().view(-1).to(device=logits.device, dtype=torch.long)
        image_h, image_w = (int(v) for v in batch["img"].shape[-2:])
        flat_mask = torch.zeros(batch_size * height * width, device=logits.device, dtype=torch.bool)
        if boxes.numel() == 0:
            return flat_mask.view(batch_size, 1, height, width), 0

        equivalent_side = torch.sqrt(
            (boxes[:, 2] * image_w).clamp_min(0) * (boxes[:, 3] * image_h).clamp_min(0)
        )
        tiny = equivalent_side <= self.tiny_equivalent_side
        if not bool(tiny.any()):
            return flat_mask.view(batch_size, 1, height, width), 0

        selected = boxes[tiny]
        selected_batch = batch_idx[tiny]
        if bool(((selected_batch < 0) | (selected_batch >= batch_size)).any()):
            raise RuntimeError("TSDR target contains an invalid batch index")
        center_x = (selected[:, 0] * width).long().clamp_(0, width - 1)
        center_y = (selected[:, 1] * height).long().clamp_(0, height - 1)

        offsets = torch.tensor(
            [[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]],
            device=logits.device,
            dtype=torch.long,
        )
        xs = (center_x[:, None] + offsets[None, :, 0]).clamp_(0, width - 1)
        ys = (center_y[:, None] + offsets[None, :, 1]).clamp_(0, height - 1)
        linear = selected_batch[:, None] * (height * width) + ys * width + xs
        flat_mask[linear.reshape(-1)] = True
        return flat_mask.view(batch_size, 1, height, width), int(tiny.sum())

    @staticmethod
    def _router_loss(logits, positive_mask):
        positive_logits = logits[positive_mask]
        negative_logits = logits[~positive_mask]
        if positive_logits.numel():
            positive_loss = ((1.0 - positive_logits.sigmoid()).pow(2) * F.softplus(-positive_logits)).mean()
        else:
            positive_loss = logits.sum() * 0.0
        if negative_logits.numel():
            negative_loss = (negative_logits.sigmoid().pow(2) * F.softplus(negative_logits)).mean()
        else:
            negative_loss = logits.sum() * 0.0
        # Separate means prevent the dense background from numerically drowning positives.
        return positive_loss + 0.25 * negative_loss

    def __call__(self, preds, batch):
        base_total, base_items = super().__call__(preds, batch)
        logits = self.head.pop_gate_logits()
        if logits is None:
            self.head._last_aux_gate_loss = None
            return base_total, base_items
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise RuntimeError(f"TSDR router logits must be Bx1xHxW, got {tuple(logits.shape)}")
        positive_mask, tiny_count = self._positive_mask(logits, batch)
        gate_loss = self._router_loss(logits, positive_mask)
        if not torch.isfinite(gate_loss):
            raise RuntimeError("TSDR auxiliary gate loss is non-finite")
        self.head._last_aux_gate_loss = {
            "loss": gate_loss.detach(),
            "tiny_objects": tiny_count,
            "positive_locations": int(positive_mask.sum()),
        }
        batch_size = int(logits.shape[0])
        total = base_total + self.gate_gain * gate_loss * batch_size
        return total, base_items


__all__ = ("v8TSDRDetectionLoss",)
