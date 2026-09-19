"""Training-only Relative Small-aware Gate guidance for E21a RSG-DSSG.

Important:
- Inference topology is identical to E10a.
- No foreground/background mask supervision is used.
- Only the pre-sigmoid logits of the existing P2 SemanticDetailGate spatial gate are observed.
- Medium/large objects are not negatives.
- No box-exterior region is forced toward zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import weakref

import torch
import torch.nn.functional as F

from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8DetectionLoss


# Pickle-safe capture registry.
# The model stores only the module-level hook function. The criterion reference
# lives in this weak dictionary and is NOT serialized into checkpoints.
_RSG_CAPTURE_TARGETS = weakref.WeakKeyDictionary()


def _capture_rsg_pre_sigmoid(module, _inputs, output):
    criterion = _RSG_CAPTURE_TARGETS.get(module)
    if criterion is not None:
        criterion._latest_gate_logits = output if criterion.detail.training else None



@dataclass(frozen=True)
class RSGSettings:
    enabled: bool = True
    gain: float = 0.02
    warmup_epochs: float = 20.0
    batches_per_epoch: int = 809  # ceil(6471 / 8), fixed formal protocol
    max_aux_ratio: float = 0.05
    tiny_equivalent_side: float = 32.0
    margin: float = 0.25
    sigma_scale: float = 0.35
    sigma_min: float = 0.75
    sigma_max: float = 2.50
    radius: int = 6
    log_interval: int = 1000

    @classmethod
    def from_yaml(cls, cfg: dict | None) -> "RSGSettings":
        cfg = cfg or {}
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            gain=float(cfg.get("gain", 0.02)),
            warmup_epochs=float(cfg.get("warmup_epochs", 20.0)),
            batches_per_epoch=int(cfg.get("batches_per_epoch", 809)),
            max_aux_ratio=float(cfg.get("max_aux_ratio", 0.05)),
            tiny_equivalent_side=float(cfg.get("tiny_equivalent_side", 32.0)),
            margin=float(cfg.get("margin", 0.25)),
            sigma_scale=float(cfg.get("sigma_scale", 0.35)),
            sigma_min=float(cfg.get("sigma_min", 0.75)),
            sigma_max=float(cfg.get("sigma_max", 2.50)),
            radius=int(cfg.get("radius", 6)),
            log_interval=int(cfg.get("log_interval", 1000)),
        )

    def validate(self) -> None:
        if not self.enabled:
            raise ValueError("RSGSettings.validate() called while disabled")
        if not 0.0 < self.gain <= 1.0:
            raise ValueError("rsg.gain must be in (0, 1]")
        if self.warmup_epochs < 0:
            raise ValueError("rsg.warmup_epochs must be >= 0")
        if self.batches_per_epoch < 1:
            raise ValueError("rsg.batches_per_epoch must be positive")
        if not 0.0 < self.max_aux_ratio <= 0.25:
            raise ValueError("rsg.max_aux_ratio must be in (0, 0.25]")
        if self.tiny_equivalent_side <= 0:
            raise ValueError("rsg.tiny_equivalent_side must be positive")
        if self.margin < 0:
            raise ValueError("rsg.margin must be non-negative")
        if not 0 < self.sigma_min <= self.sigma_max:
            raise ValueError("Require 0 < sigma_min <= sigma_max")
        if self.sigma_scale <= 0:
            raise ValueError("rsg.sigma_scale must be positive")
        if self.radius < 1:
            raise ValueError("rsg.radius must be >= 1")


def relative_small_gate_loss(logits: torch.Tensor, batch: dict, settings: RSGSettings):
    """Relative ranking loss on existing P2 gate logits.

    For every small GT in the current augmented image:
      1) sample a size-adaptive anisotropic Gaussian neighborhood on P2;
      2) compute the Gaussian-weighted mean gate logit;
      3) require that mean to exceed the stop-gradient image-wide mean
         by a finite margin.

    No background position is assigned a negative target.
    Medium/large GTs do not enter this auxiliary objective.
    """
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"Expected Bx1xHxW gate logits, got {tuple(logits.shape)}")

    work = logits.float()
    bsz, _, gh, gw = work.shape
    device = work.device

    boxes = batch["bboxes"].detach().to(device=device, dtype=torch.float32)
    image_index = batch["batch_idx"].detach().reshape(-1).to(device=device, dtype=torch.long)
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"Expected normalized xywh boxes Nx4, got {tuple(boxes.shape)}")
    if boxes.shape[0] != image_index.shape[0]:
        raise ValueError("batch['bboxes'] and batch['batch_idx'] length mismatch")

    image_h, image_w = (int(v) for v in batch["img"].shape[-2:])
    valid = (
        torch.isfinite(boxes).all(dim=1)
        & (boxes[:, 2] > 0)
        & (boxes[:, 3] > 0)
        & (image_index >= 0)
        & (image_index < bsz)
    )
    if not bool(valid.any()):
        zero = work.sum() * 0.0
        return zero, {"small_objects": 0}

    boxes = boxes[valid]
    image_index = image_index[valid]
    equivalent_side = torch.sqrt(
        (boxes[:, 2] * image_w).clamp_min(0.0)
        * (boxes[:, 3] * image_h).clamp_min(0.0)
    )
    small = equivalent_side <= settings.tiny_equivalent_side
    if not bool(small.any()):
        zero = work.sum() * 0.0
        return zero, {"small_objects": 0}

    boxes = boxes[small]
    image_index = image_index[small]
    n = boxes.shape[0]

    cx = boxes[:, 0] * gw
    cy = boxes[:, 1] * gh
    bw = boxes[:, 2] * gw
    bh = boxes[:, 3] * gh
    sigma_x = (bw * settings.sigma_scale).clamp(settings.sigma_min, settings.sigma_max)
    sigma_y = (bh * settings.sigma_scale).clamp(settings.sigma_min, settings.sigma_max)

    offsets = torch.arange(-settings.radius, settings.radius + 1, device=device, dtype=torch.long)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    ox = ox.reshape(1, -1)
    oy = oy.reshape(1, -1)

    base_x = torch.floor(cx).long().reshape(-1, 1)
    base_y = torch.floor(cy).long().reshape(-1, 1)
    xi = base_x + ox
    yi = base_y + oy

    valid_xy = (xi >= 0) & (xi < gw) & (yi >= 0) & (yi < gh)
    xi_safe = xi.clamp(0, gw - 1)
    yi_safe = yi.clamp(0, gh - 1)

    px = xi.to(torch.float32) + 0.5
    py = yi.to(torch.float32) + 0.5
    dx = (px - cx.reshape(-1, 1)) / sigma_x.reshape(-1, 1)
    dy = (py - cy.reshape(-1, 1)) / sigma_y.reshape(-1, 1)
    weight = torch.exp(-0.5 * (dx.square() + dy.square()))
    weight = weight * valid_xy.to(weight.dtype)
    weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-8)

    flat = work[:, 0].reshape(-1)
    linear = image_index.reshape(-1, 1) * (gh * gw) + yi_safe * gw + xi_safe
    samples = flat[linear]
    small_mean = (samples * weight).sum(dim=1)

    map_mean = work[:, 0].mean(dim=(1, 2)).detach()
    relative = small_mean - map_mean[image_index]
    loss = F.softplus(settings.margin - relative).mean()

    return loss, {
        "small_objects": int(n),
        "relative_mean": relative.detach().mean(),
        "small_logit_mean": small_mean.detach().mean(),
        "map_logit_mean": map_mean[image_index].detach().mean(),
    }


class v8RSGDetectionLoss(v8DetectionLoss):
    """Standard YOLO detection loss + training-only RSG auxiliary objective."""

    def __init__(self, model):
        super().__init__(model)
        self.settings = RSGSettings.from_yaml(model.yaml.get("rsg", {}))
        self.settings.validate()
        if len(model.model) != 29:
            raise TypeError("RSG-DSSG expects the 29-layer E10a topology")
        self.detail = model.model[27]
        if self.detail.__class__.__name__ != "SemanticDetailGate":
            raise TypeError("RSG-DSSG requires SemanticDetailGate at model[27]")

        gate = self.detail.spatial_gate
        self._latest_gate_logits = None
        self._training_batches_seen = 0

        self._capture_module = gate.net[-1]
        _RSG_CAPTURE_TARGETS[self._capture_module] = self
        self._hook_handle = self._capture_module.register_forward_hook(_capture_rsg_pre_sigmoid)

    def _current_gain(self) -> float:
        warmup_batches = int(round(self.settings.warmup_epochs * self.settings.batches_per_epoch))
        if warmup_batches <= 0:
            return self.settings.gain
        progress = min(1.0, self._training_batches_seen / warmup_batches)
        return self.settings.gain * progress

    def __call__(self, preds, batch):
        base_total, base_items = super().__call__(preds, batch)
        if not self.detail.training:
            self._latest_gate_logits = None
            return base_total, base_items

        logits = self._latest_gate_logits
        self._latest_gate_logits = None
        self._training_batches_seen += 1
        if logits is None:
            raise RuntimeError("RSG did not capture P2 spatial-gate logits")

        rsg_loss, stats = relative_small_gate_loss(logits, batch, self.settings)
        gain = self._current_gain()
        batch_size = int(batch["img"].shape[0])
        weighted_aux = rsg_loss * (gain * batch_size)

        cap = base_total.detach().abs() * self.settings.max_aux_ratio
        denom = weighted_aux.detach().abs().clamp_min(1e-12)
        scale = torch.minimum(torch.ones_like(cap), cap / denom)
        weighted_aux = weighted_aux * scale
        if not torch.isfinite(weighted_aux):
            raise FloatingPointError("Non-finite RSG auxiliary loss")

        if self.settings.log_interval > 0 and self._training_batches_seen % self.settings.log_interval == 0:
            rel = stats.get("relative_mean")
            rel = float(rel.cpu()) if isinstance(rel, torch.Tensor) else float("nan")
            LOGGER.info(
                "RSG: batch=%d gain=%.6f small=%d rel=%.4f raw=%.5f weighted=%.5f cap_ratio<=%.3f",
                self._training_batches_seen,
                gain,
                int(stats.get("small_objects", 0)),
                rel,
                float(rsg_loss.detach().cpu()),
                float(weighted_aux.detach().cpu()),
                self.settings.max_aux_ratio,
            )

        return base_total + weighted_aux, base_items

    def __del__(self):
        capture_module = getattr(self, "_capture_module", None)
        try:
            if capture_module is not None:
                _RSG_CAPTURE_TARGETS.pop(capture_module, None)
        except Exception:
            pass
        handle = getattr(self, "_hook_handle", None)
        try:
            if handle is not None:
                handle.remove()
        except Exception:
            pass


__all__ = ("RSGSettings", "relative_small_gate_loss", "v8RSGDetectionLoss")
