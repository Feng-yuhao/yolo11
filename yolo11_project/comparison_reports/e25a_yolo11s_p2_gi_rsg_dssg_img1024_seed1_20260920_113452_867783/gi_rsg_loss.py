"""E25a training-only Gradient-Isolated Relative Small-aware Gate loss.

E25a keeps the E10a inference graph and the complete E21a RSG objective.
The sole intervention is the direct auxiliary-gradient scope:

    captured 5-channel descriptor -> detach -> existing spatial_gate.net
                                      (functional call with cloned BN buffers)

Consequently, the RSG auxiliary loss has direct gradients only for the
existing ``model[27].spatial_gate`` parameters.  This is direct-gradient
isolation, not complete causal compartmentalization: updated gate parameters
can still alter later normal detection-loss trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
import weakref

import torch
import torch.nn.functional as F
from torch.func import functional_call

from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8DetectionLoss


# Pickle-safe capture registry.  Checkpoints retain only the module-level hook;
# criterion references stay outside the serialized model in this weak map.
_GI_RSG_CAPTURE_TARGETS = weakref.WeakKeyDictionary()


def _capture_gi_rsg_descriptor(module, inputs):
    """Capture the normal-forward spatial-gate input without changing it."""
    criterion = _GI_RSG_CAPTURE_TARGETS.get(module)
    if criterion is None:
        return
    if not criterion.detail.training:
        criterion._latest_descriptor = None
        return
    if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
        raise RuntimeError("GI-RSG expected one tensor input to the P2 spatial gate")
    descriptor = inputs[0]
    if descriptor.ndim != 4 or descriptor.shape[1] != 5:
        raise RuntimeError(f"GI-RSG expected Bx5xHxW descriptor, got {tuple(descriptor.shape)}")
    criterion._latest_descriptor = descriptor


@dataclass(frozen=True)
class GIRSGSettings:
    """Frozen E21a settings; E25a must not tune any of these values."""

    enabled: bool = True
    gain: float = 0.02
    warmup_epochs: float = 20.0
    batches_per_epoch: int = 809
    max_aux_ratio: float = 0.05
    tiny_equivalent_side: float = 32.0
    margin: float = 0.25
    sigma_scale: float = 0.35
    sigma_min: float = 0.75
    sigma_max: float = 2.50
    radius: int = 6
    log_interval: int = 1000

    @classmethod
    def from_yaml(cls, cfg: dict | None) -> "GIRSGSettings":
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
            raise ValueError("GIRSGSettings.validate() called while disabled")
        if not 0.0 < self.gain <= 1.0:
            raise ValueError("gi_rsg.gain must be in (0, 1]")
        if self.warmup_epochs < 0:
            raise ValueError("gi_rsg.warmup_epochs must be >= 0")
        if self.batches_per_epoch < 1:
            raise ValueError("gi_rsg.batches_per_epoch must be positive")
        if not 0.0 < self.max_aux_ratio <= 0.25:
            raise ValueError("gi_rsg.max_aux_ratio must be in (0, 0.25]")
        if self.tiny_equivalent_side <= 0:
            raise ValueError("gi_rsg.tiny_equivalent_side must be positive")
        if self.margin < 0:
            raise ValueError("gi_rsg.margin must be non-negative")
        if not 0 < self.sigma_min <= self.sigma_max:
            raise ValueError("Require 0 < sigma_min <= sigma_max")
        if self.sigma_scale <= 0:
            raise ValueError("gi_rsg.sigma_scale must be positive")
        if self.radius < 1:
            raise ValueError("gi_rsg.radius must be >= 1")


def relative_small_gate_loss(logits: torch.Tensor, batch: dict, settings: GIRSGSettings):
    """The unchanged E21a relative small-aware gate objective."""
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
        return work.sum() * 0.0, {"small_objects": 0}

    boxes = boxes[valid]
    image_index = image_index[valid]
    equivalent_side = torch.sqrt(
        (boxes[:, 2] * image_w).clamp_min(0.0)
        * (boxes[:, 3] * image_h).clamp_min(0.0)
    )
    small = equivalent_side <= settings.tiny_equivalent_side
    if not bool(small.any()):
        return work.sum() * 0.0, {"small_objects": 0}

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
    ox, oy = ox.reshape(1, -1), oy.reshape(1, -1)
    base_x = torch.floor(cx).long().reshape(-1, 1)
    base_y = torch.floor(cy).long().reshape(-1, 1)
    xi, yi = base_x + ox, base_y + oy
    valid_xy = (xi >= 0) & (xi < gw) & (yi >= 0) & (yi < gh)
    xi_safe, yi_safe = xi.clamp(0, gw - 1), yi.clamp(0, gh - 1)

    px, py = xi.to(torch.float32) + 0.5, yi.to(torch.float32) + 0.5
    dx = (px - cx.reshape(-1, 1)) / sigma_x.reshape(-1, 1)
    dy = (py - cy.reshape(-1, 1)) / sigma_y.reshape(-1, 1)
    weight = torch.exp(-0.5 * (dx.square() + dy.square()))
    weight = weight * valid_xy.to(weight.dtype)
    weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-8)

    flat = work[:, 0].reshape(-1)
    linear = image_index.reshape(-1, 1) * (gh * gw) + yi_safe * gw + xi_safe
    small_mean = (flat[linear] * weight).sum(dim=1)
    map_mean = work[:, 0].mean(dim=(1, 2)).detach()
    relative = small_mean - map_mean[image_index]
    loss = F.softplus(settings.margin - relative).mean()
    return loss, {
        "small_objects": int(n),
        "relative_mean": relative.detach().mean(),
        "small_logit_mean": small_mean.detach().mean(),
        "map_logit_mean": map_mean[image_index].detach().mean(),
    }


def isolated_gate_logits(gate_net, descriptor: torch.Tensor) -> torch.Tensor:
    """Recompute pre-sigmoid logits with shared parameters and private BN buffers."""
    if descriptor.ndim != 4 or descriptor.shape[1] != 5:
        raise ValueError(f"Expected Bx5xHxW descriptor, got {tuple(descriptor.shape)}")
    parameters = dict(gate_net.named_parameters())
    # Fresh clones are essential: training-mode BatchNorm may mutate these.
    buffers = {name: value.detach().clone() for name, value in gate_net.named_buffers()}
    return functional_call(
        gate_net,
        (parameters, buffers),
        (descriptor.detach(),),
        strict=True,
    )


class v8GIRSGDetectionLoss(v8DetectionLoss):
    """Standard YOLO loss plus E21a RSG with isolated direct gradients."""

    def __init__(self, model):
        super().__init__(model)
        self.settings = GIRSGSettings.from_yaml(model.yaml.get("gi_rsg", {}))
        self.settings.validate()
        if len(model.model) != 29:
            raise TypeError("GI-RSG-DSSG expects the 29-layer E10a topology")
        self.detail = model.model[27]
        if self.detail.__class__.__name__ != "SemanticDetailGate":
            raise TypeError("GI-RSG-DSSG requires SemanticDetailGate at model[27]")

        self.gate = self.detail.spatial_gate
        self.gate_net = self.gate.net
        self._latest_descriptor = None
        self._training_batches_seen = 0
        self.aux_enabled = True
        self.last_aux_stats = None

        self._capture_module = self.gate
        _GI_RSG_CAPTURE_TARGETS[self._capture_module] = self
        self._hook_handle = self._capture_module.register_forward_pre_hook(_capture_gi_rsg_descriptor)

    def _current_gain(self) -> float:
        warmup_batches = int(round(self.settings.warmup_epochs * self.settings.batches_per_epoch))
        if warmup_batches <= 0:
            return self.settings.gain
        progress = min(1.0, self._training_batches_seen / warmup_batches)
        return self.settings.gain * progress

    def auxiliary_loss(self, batch: dict, base_total: torch.Tensor):
        """Compute isolated RSG and return its capped weighted contribution."""
        descriptor = self._latest_descriptor
        self._latest_descriptor = None
        if descriptor is None:
            raise RuntimeError("GI-RSG did not capture the P2 five-channel gate descriptor")

        logits = isolated_gate_logits(self.gate_net, descriptor)
        rsg_loss, stats = relative_small_gate_loss(logits, batch, self.settings)
        gain = self._current_gain()
        batch_size = int(batch["img"].shape[0])
        weighted_aux = rsg_loss * (gain * batch_size)
        cap = base_total.detach().abs() * self.settings.max_aux_ratio
        denom = weighted_aux.detach().abs().clamp_min(1e-12)
        scale = torch.minimum(torch.ones_like(cap), cap / denom)
        weighted_aux = weighted_aux * scale
        if not torch.isfinite(weighted_aux):
            raise FloatingPointError("Non-finite GI-RSG auxiliary loss")
        self.last_aux_stats = {
            **stats,
            "gain": gain,
            "raw_loss": rsg_loss.detach(),
            "weighted_loss": weighted_aux.detach(),
        }
        return weighted_aux

    def __call__(self, preds, batch):
        base_total, base_items = super().__call__(preds, batch)
        if not self.detail.training or not self.aux_enabled:
            self._latest_descriptor = None
            return base_total, base_items

        self._training_batches_seen += 1
        weighted_aux = self.auxiliary_loss(batch, base_total)
        if self.settings.log_interval > 0 and self._training_batches_seen % self.settings.log_interval == 0:
            stats = self.last_aux_stats or {}
            rel = stats.get("relative_mean")
            rel = float(rel.cpu()) if isinstance(rel, torch.Tensor) else float("nan")
            LOGGER.info(
                "GI-RSG: batch=%d gain=%.6f small=%d rel=%.4f raw=%.5f weighted=%.5f cap_ratio<=%.3f",
                self._training_batches_seen,
                float(stats.get("gain", 0.0)),
                int(stats.get("small_objects", 0)),
                rel,
                float(stats["raw_loss"].cpu()),
                float(stats["weighted_loss"].cpu()),
                self.settings.max_aux_ratio,
            )
        return base_total + weighted_aux, base_items

    def close(self):
        capture_module = getattr(self, "_capture_module", None)
        try:
            if capture_module is not None:
                _GI_RSG_CAPTURE_TARGETS.pop(capture_module, None)
        except Exception:
            pass
        handle = getattr(self, "_hook_handle", None)
        try:
            if handle is not None:
                handle.remove()
        except Exception:
            pass
        self._hook_handle = None

    def __del__(self):
        self.close()


__all__ = (
    "GIRSGSettings",
    "isolated_gate_logits",
    "relative_small_gate_loss",
    "v8GIRSGDetectionLoss",
)
