# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Explicit tiny-object prior supervision used only by the VisDrone E3b ablation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .loss import v8DetectionLoss

__all__ = ("v8TinyPriorDetectionLoss",)


class v8TinyPriorDetectionLoss(v8DetectionLoss):
    """YOLO detection loss plus a scale-aware P2 center-prior auxiliary loss.

    Boxes come from ``batch['bboxes']`` after Ultralytics has applied all current
    image augmentations.  A box is tiny when its area in the current training
    image is below ``max_area_px`` (default 32**2).  Tiny centers are split into
    three size groups and expanded with different fixed Gaussian kernels.  The
    target therefore encodes both center location and approximate object scale.

    The auxiliary term supervises only the one-channel router already present in
    E3a TASD.  It adds no inference parameters or inference-only operations.
    """

    def __init__(self, model, tal_topk=10):
        super().__init__(model, tal_topk=tal_topk)
        config = model.yaml.get("tiny_prior", {})
        if not isinstance(config, dict) or not config.get("enabled", False):
            raise ValueError("v8TinyPriorDetectionLoss requires yaml tiny_prior.enabled=true.")

        modules = [m for m in model.modules() if m.__class__.__name__ == "TASDPriorFuse"]
        if len(modules) != 1:
            raise ValueError(f"Expected exactly one TASDPriorFuse, found {len(modules)}.")
        self.prior_module = modules[0]
        self.prior_weight = float(config.get("loss_weight", 0.25))
        self.max_area_px = float(config.get("max_area_px", 1024.0))
        self.prior_stride = int(config.get("stride", 4))
        self.bucket_edges_px = tuple(float(x) for x in config.get("bucket_edges_px", (12.0, 22.0)))
        self.sigmas = tuple(float(x) for x in config.get("sigmas", (0.75, 1.25, 2.0)))
        self.focal_gamma = float(config.get("focal_gamma", 2.0))
        self.negative_beta = float(config.get("negative_beta", 4.0))

        if self.prior_weight <= 0 or self.max_area_px <= 0 or self.prior_stride <= 0:
            raise ValueError("Tiny-prior loss weight, max area, and stride must be positive.")
        if len(self.bucket_edges_px) != 2 or not 0 < self.bucket_edges_px[0] < self.bucket_edges_px[1]:
            raise ValueError("tiny_prior.bucket_edges_px must contain two increasing positive values.")
        if len(self.sigmas) != 3 or any(s <= 0 for s in self.sigmas):
            raise ValueError("tiny_prior.sigmas must contain three positive values.")

        self.kernel_size = 11
        self.gaussian_kernels = self._make_gaussian_kernels(self.sigmas, self.kernel_size, self.device)
        self.last_prior_stats = {}

    @staticmethod
    def _make_gaussian_kernels(sigmas, kernel_size, device):
        radius = kernel_size // 2
        axis = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        kernels = []
        for sigma in sigmas:
            kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma**2))
            kernels.append(kernel / kernel.max())
        return torch.stack(kernels, 0).unsqueeze(1)

    @torch.no_grad()
    def build_target(self, batch, logits):
        """Build a deterministic Bx1xH/4xW/4 scale-aware center heatmap."""
        batch_size, channels, height, width = logits.shape
        if channels != 1:
            raise ValueError(f"Tiny-prior logits must have one channel, got {tuple(logits.shape)}.")
        image_height, image_width = batch["img"].shape[-2:]
        if (image_height, image_width) != (height * self.prior_stride, width * self.prior_stride):
            raise ValueError(
                "Tiny-prior map/input mismatch: "
                f"input={(image_height, image_width)}, prior={tuple(logits.shape[-2:])}, "
                f"stride={self.prior_stride}."
            )

        seeds = torch.zeros((batch_size, 3, height, width), device=logits.device, dtype=torch.float32)
        boxes = batch["bboxes"].to(device=logits.device, dtype=torch.float32)
        indices = batch["batch_idx"].view(-1).to(device=logits.device, dtype=torch.long)
        if boxes.numel() == 0:
            return seeds[:, :1], torch.zeros((), device=logits.device, dtype=torch.long)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.shape[0] != indices.numel():
            raise ValueError("Expected normalized xywh boxes aligned with batch_idx.")

        widths_px = boxes[:, 2] * float(image_width)
        heights_px = boxes[:, 3] * float(image_height)
        area_px = widths_px * heights_px
        valid = (
            (indices >= 0)
            & (indices < batch_size)
            & (boxes[:, 0] >= 0)
            & (boxes[:, 0] <= 1)
            & (boxes[:, 1] >= 0)
            & (boxes[:, 1] <= 1)
            & (widths_px > 0)
            & (heights_px > 0)
            & (area_px < self.max_area_px)
        )
        boxes = boxes[valid]
        indices = indices[valid]
        size_px = area_px[valid].sqrt()
        x = (boxes[:, 0] * width).floor().long().clamp_(0, width - 1)
        y = (boxes[:, 1] * height).floor().long().clamp_(0, height - 1)
        buckets = torch.zeros_like(indices)
        buckets[size_px > self.bucket_edges_px[0]] = 1
        buckets[size_px > self.bucket_edges_px[1]] = 2

        # Unique linear indices make the write deterministic even when two GT
        # centers fall into the same P2 cell and deterministic algorithms=True.
        linear = ((indices * 3 + buckets) * height + y) * width + x
        seeds.view(-1)[torch.unique(linear)] = 1.0
        kernels = self.gaussian_kernels.to(device=logits.device)
        heatmaps = F.conv2d(seeds, kernels, padding=self.kernel_size // 2, groups=3)
        target = heatmaps.amax(dim=1, keepdim=True).clamp_(0.0, 1.0)
        return target, valid.sum().detach()

    def prior_loss(self, logits, target):
        """Balanced soft CenterNet-style focal loss for a sparse heatmap."""
        probability = logits.float().sigmoid().clamp(1e-6, 1.0 - 1e-6)
        target = target.float()
        positive_weight = target
        negative_weight = (1.0 - target).pow(self.negative_beta)
        positive = -positive_weight * (1.0 - probability).pow(self.focal_gamma) * probability.log()
        negative = -negative_weight * probability.pow(self.focal_gamma) * (1.0 - probability).log()
        positive_loss = positive.sum() / positive_weight.sum().clamp_min(1.0)
        negative_loss = negative.sum() / negative_weight.sum().clamp_min(1.0)
        return positive_loss + negative_loss

    def __call__(self, preds, batch):
        logits = self.prior_module.pop_prior_logits()
        if logits is None:
            raise RuntimeError(
                "TASDPriorFuse did not expose router logits for this forward pass; "
                "the registered E3b module/source is inconsistent."
            )
        detection_total, detection_items = super().__call__(preds, batch)
        target, tiny_count = self.build_target(batch, logits)
        raw_prior = self.prior_loss(logits, target)
        weighted_prior = raw_prior * self.prior_weight
        self.last_prior_stats = {
            "tiny_instances": tiny_count,
            "raw_loss": raw_prior.detach(),
            "weighted_loss": weighted_prior.detach(),
        }
        total = detection_total + weighted_prior * logits.shape[0]
        items = torch.cat((detection_items, weighted_prior.detach().reshape(1)))
        return total, items
