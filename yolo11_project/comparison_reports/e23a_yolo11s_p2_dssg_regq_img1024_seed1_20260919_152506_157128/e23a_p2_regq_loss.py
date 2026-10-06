"""E23a training-only P2-local regression-quality intervention.

Scientific intent
-----------------
D23/D23.1 localized E10a's high-IoU small-object bottleneck to raw box geometry,
predominantly at P2.  D23.1 did NOT support strong TAL positive starvation after
controlling class/size/density/anchor eligibility.

E23a therefore leaves:
- architecture/inference topology unchanged,
- TAL assignment unchanged,
- classification targets/loss unchanged,
- P3/P4/P5 regression branches unchanged,

and adds ONE controlled intervention:
increase the existing CIoU + DFL supervision for TAL-assigned SMALL positives on
the P2 detection level.

Crucially, the *extra* gradient is surgically restricted to P2-exclusive trainable
modules:
  model[25] : fused P2 C3k2
  model[27] : SemanticDetailGate (P2 output only)
  Detect.cv2[0] : P2 regression branch

The extra gradient does NOT enter:
  model[26] P3_sem / shared P3 semantic guide,
  P3/P4/P5 regression branches,
  any classification branch,
  the backbone / original P3-P5 neck.

The ordinary E10a loss still backpropagates everywhere exactly as before.

This is a first-order gradient-isolated causal test.  It does not add inference
parameters or FLOPs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Dict, Iterable, List, Tuple

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist, make_anchors


IMPLEMENTATION_REVISION = "e23a_regq_loss_v1_5_ema_validation_safe"


RUNTIME_AUDIT = {
    "batches": 0,
    "batches_with_aux": 0,
    "fg_anchors": 0,
    "p2_small_positive_anchors": 0,
    "target_score_sum": 0.0,
    "p2_small_target_score_sum": 0.0,
    "raw_aux_iou_sum": 0.0,
    "raw_aux_dfl_sum": 0.0,
    "weighted_aux_sum": 0.0,
    "base_total_sum_abs": 0.0,
    "max_aux_to_base_abs_ratio": 0.0,
    "final_gain": 0.0,
}


def reset_runtime_audit():
    for k in list(RUNTIME_AUDIT):
        if k in ("batches", "batches_with_aux", "fg_anchors", "p2_small_positive_anchors"):
            RUNTIME_AUDIT[k] = 0
        else:
            RUNTIME_AUDIT[k] = 0.0


def runtime_audit_snapshot():
    out = dict(RUNTIME_AUDIT)
    b = max(int(out["batches"]), 1)
    bw = max(int(out["batches_with_aux"]), 1)
    out["mean_p2_small_positive_anchors_per_batch"] = out["p2_small_positive_anchors"] / b
    out["mean_fg_anchors_per_batch"] = out["fg_anchors"] / b
    out["mean_raw_aux_iou_when_active"] = out["raw_aux_iou_sum"] / bw
    out["mean_raw_aux_dfl_when_active"] = out["raw_aux_dfl_sum"] / bw
    out["mean_weighted_aux_when_active"] = out["weighted_aux_sum"] / bw
    out["aggregate_aux_to_base_abs_ratio"] = (
        out["weighted_aux_sum"] / max(out["base_total_sum_abs"], 1e-12)
    )
    return out


@dataclass(frozen=True)
class P2RegQSettings:
    enabled: bool = True
    extra_gain: float = 0.50
    warmup_epochs: float = 20.0
    batches_per_epoch: int = 809
    small_equivalent_side: float = 32.0
    log_interval: int = 1000

    @classmethod
    def from_yaml(cls, cfg: dict | None) -> "P2RegQSettings":
        cfg = cfg or {}
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            extra_gain=float(cfg.get("extra_gain", 0.50)),
            warmup_epochs=float(cfg.get("warmup_epochs", 20.0)),
            batches_per_epoch=int(cfg.get("batches_per_epoch", 809)),
            small_equivalent_side=float(cfg.get("small_equivalent_side", 32.0)),
            log_interval=int(cfg.get("log_interval", 1000)),
        )

    def validate(self) -> None:
        if not self.enabled:
            raise ValueError("P2RegQSettings.validate() called while disabled")
        if not 0.0 < self.extra_gain <= 1.0:
            raise ValueError("p2_regq.extra_gain must be in (0, 1]")
        if self.warmup_epochs < 0:
            raise ValueError("p2_regq.warmup_epochs must be >= 0")
        if self.batches_per_epoch < 1:
            raise ValueError("p2_regq.batches_per_epoch must be positive")
        if not 0.0 < self.small_equivalent_side <= 64.0:
            raise ValueError("p2_regq.small_equivalent_side must be in (0, 64]")
        if self.log_interval < 0:
            raise ValueError("p2_regq.log_interval must be >= 0")


def _unique_trainable_parameters(modules: Iterable[torch.nn.Module]) -> List[torch.nn.Parameter]:
    params = []
    seen = set()
    for module in modules:
        for p in module.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                params.append(p)
    return params


def _grad_proxy(
    aux_scalar: torch.Tensor,
    allowed_params: List[torch.nn.Parameter],
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """Return a value-preserving scalar whose gradient exists only on allowed_params.

    `aux_scalar` is differentiated once with autograd.grad.  Those first-order
    gradients are detached and re-injected through a zero-valued linear surrogate.

    Forward value:
        proxy == aux_scalar.detach()

    Backward:
        d proxy / d p == d aux_scalar / d p for allowed p
        d proxy / d everything_else == 0

    The later AMP GradScaler scales this proxy together with the ordinary base loss,
    so its injected gradients receive the same optimizer-side scaling.
    """
    grads = torch.autograd.grad(
        aux_scalar,
        allowed_params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    proxy = aux_scalar.detach()
    grad_map: Dict[int, torch.Tensor] = {}
    for p, g in zip(allowed_params, grads):
        if g is None:
            continue
        if not torch.isfinite(g).all():
            raise FloatingPointError("Non-finite E23a isolated auxiliary gradient")
        gd = g.detach()
        grad_map[id(p)] = gd
        term = (p * gd).sum()
        proxy = proxy + term - term.detach()
    return proxy, grad_map


class v8P2RegQDetectionLoss(v8DetectionLoss):
    """E10a detection loss + P2-small regression auxiliary with gradient isolation."""

    def __init__(self, model):
        super().__init__(model)
        self.settings = P2RegQSettings.from_yaml(model.yaml.get("p2_regq", {}))
        self.settings.validate()

        if len(model.model) != 29:
            raise TypeError("E23a requires the 29-layer E10a topology")
        if model.model[25].__class__.__name__ != "C3k2":
            raise TypeError("E23a expects fused P2 C3k2 at model[25]")
        if model.model[26].__class__.__name__ != "DeepSemanticGuide":
            raise TypeError("E23a expects DeepSemanticGuide at model[26]")
        if model.model[27].__class__.__name__ != "SemanticDetailGate":
            raise TypeError("E23a expects SemanticDetailGate at model[27]")

        self.model25 = model.model[25]
        self.model26 = model.model[26]  # forbidden to auxiliary gradient; kept for audit
        self.model27 = model.model[27]
        self.head = model.model[28]
        if self.head.__class__.__name__ != "Detect" or self.head.nl != 4:
            raise TypeError("E23a expects standard four-level Detect")
        if tuple(float(x) for x in self.head.stride.detach().cpu().tolist()) != (4.0, 8.0, 16.0, 32.0):
            raise TypeError("E23a requires P2/P3/P4/P5 strides 4/8/16/32")

        # P2-exclusive trainable modules.  Do NOT include model[26], backbone,
        # classification branches, or P3/P4/P5 box branches.
        self.allowed_modules = (self.model25, self.model27, self.head.cv2[0])
        self.allowed_params = _unique_trainable_parameters(self.allowed_modules)

        # IMPORTANT:
        # During in-training validation Ultralytics evaluates the EMA model.
        # ModelEMA deliberately sets every EMA parameter requires_grad=False.
        # The validator also sets model.eval().  In that path the criterion is
        # needed only to compute the ordinary validation loss; the E23a auxiliary
        # must be inactive and an empty allowed-parameter list is expected.
        #
        # Conversely, in real TRAINING mode an empty allowed list is a hard error:
        # otherwise E23a would silently stop applying its intended intervention.
        self._ema_or_eval_no_trainable_params = not bool(self.allowed_params)

        self._training_batches_seen = 0
        self.audit_capture = False
        self._last_proxy_graph = None
        self._last_raw_aux_graph = None
        self._last_base_total_graph = None
        self._last_stats = None

    def _current_gain(self) -> float:
        warmup_batches = int(round(self.settings.warmup_epochs * self.settings.batches_per_epoch))
        if warmup_batches <= 0:
            return self.settings.extra_gain
        progress = min(1.0, self._training_batches_seen / warmup_batches)
        return self.settings.extra_gain * progress

    def _p2_small_aux(
        self,
        feats,
        pred_distri,
        pred_bboxes,
        anchor_points,
        stride_tensor,
        target_bboxes_px,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        # Same assigned GT / same TAL.  Only select the subset whose assigned target
        # is SMALL in the current augmented input and whose anchor belongs to P2.
        p2_count = int(feats[0].shape[-2] * feats[0].shape[-1])
        p2_anchor = torch.zeros(
            pred_bboxes.shape[1], dtype=torch.bool, device=pred_bboxes.device
        )
        p2_anchor[:p2_count] = True

        wh = (target_bboxes_px[..., 2:4] - target_bboxes_px[..., 0:2]).clamp_min(0.0)
        equiv_side = torch.sqrt((wh[..., 0] * wh[..., 1]).clamp_min(0.0))
        small = equiv_side <= self.settings.small_equivalent_side
        aux_mask = fg_mask & p2_anchor.unsqueeze(0) & small

        if not bool(aux_mask.any()):
            zero = pred_distri.sum() * 0.0
            return zero, zero, {
                "fg_anchors": int(fg_mask.sum().item()),
                "p2_small_positive_anchors": 0,
                "p2_small_target_score_sum": 0.0,
                "target_score_sum": float(target_scores_sum.detach().cpu())
                if isinstance(target_scores_sum, torch.Tensor) else float(target_scores_sum),
            }

        target_bboxes_grid = target_bboxes_px / stride_tensor
        weight = target_scores.sum(-1)[aux_mask].unsqueeze(-1)

        iou = bbox_iou(
            pred_bboxes[aux_mask],
            target_bboxes_grid[aux_mask],
            xywh=False,
            CIoU=True,
        )
        aux_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.bbox_loss.dfl_loss is not None:
            target_ltrb = bbox2dist(
                anchor_points,
                target_bboxes_grid,
                self.bbox_loss.dfl_loss.reg_max - 1,
            )
            dfl = self.bbox_loss.dfl_loss(
                pred_distri[aux_mask].view(-1, self.bbox_loss.dfl_loss.reg_max),
                target_ltrb[aux_mask],
            ) * weight
            aux_dfl = dfl.sum() / target_scores_sum
        else:
            aux_dfl = pred_distri.sum() * 0.0

        return aux_iou, aux_dfl, {
            "fg_anchors": int(fg_mask.sum().item()),
            "p2_small_positive_anchors": int(aux_mask.sum().item()),
            "p2_small_target_score_sum": float(weight.detach().sum().cpu()),
            "target_score_sum": float(target_scores_sum.detach().cpu())
            if isinstance(target_scores_sum, torch.Tensor) else float(target_scores_sum),
        }

    def __call__(self, preds, batch):
        # This is intentionally the Ultralytics 8.3.0 v8DetectionLoss path copied
        # verbatim in ordering, with one auxiliary block inserted after the base
        # bbox loss.  No second model forward and no second TAL assignment.
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(
            feats[0].shape[2:], device=self.device, dtype=dtype
        ) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(
            targets.to(self.device),
            batch_size,
            scale_tensor=imgsz[[1, 0, 1, 0]],
        )
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

        # Classification remains exactly E10a.
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        aux_proxy = pred_distri.sum() * 0.0
        raw_aux_graph = pred_distri.sum() * 0.0
        stats = {
            "fg_anchors": int(fg_mask.sum().item()),
            "p2_small_positive_anchors": 0,
            "p2_small_target_score_sum": 0.0,
            "target_score_sum": float(target_scores_sum.detach().cpu())
            if isinstance(target_scores_sum, torch.Tensor) else float(target_scores_sum),
        }
        gain = 0.0

        if self.head.training:
            if not self.allowed_params:
                raise RuntimeError(
                    "E23a criterion entered TRAINING mode with no trainable P2-local parameters. "
                    "An empty allowed set is valid only for eval/EMA validation."
                )
            self._training_batches_seen += 1
            gain = self._current_gain()

        if fg_mask.sum():
            target_bboxes_px = target_bboxes
            target_bboxes_grid = target_bboxes_px / stride_tensor

            # Ordinary E10a box + DFL loss, unchanged.
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes_grid,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

            if self.head.training:
                aux_iou, aux_dfl, stats = self._p2_small_aux(
                    feats,
                    pred_distri,
                    pred_bboxes,
                    anchor_points,
                    stride_tensor,
                    target_bboxes_px,
                    target_scores,
                    target_scores_sum,
                    fg_mask,
                )

                # Preserve standard box:dfl global gains.  Numerically this is a
                # +gain copy of the existing loss contributions from P2-small positives.
                raw_aux_graph = (
                    aux_iou * self.hyp.box + aux_dfl * self.hyp.dfl
                ) * (gain * batch_size)

                if gain > 0.0 and stats["p2_small_positive_anchors"] > 0:
                    aux_proxy, _ = _grad_proxy(raw_aux_graph, self.allowed_params)

                if self.audit_capture:
                    self._last_proxy_graph = aux_proxy
                    self._last_raw_aux_graph = raw_aux_graph
                    self._last_stats = dict(stats, gain=gain)

                # Lightweight runtime audit; no tensor graph is retained.
                RUNTIME_AUDIT["batches"] += 1
                RUNTIME_AUDIT["fg_anchors"] += stats["fg_anchors"]
                RUNTIME_AUDIT["p2_small_positive_anchors"] += stats["p2_small_positive_anchors"]
                RUNTIME_AUDIT["target_score_sum"] += stats["target_score_sum"]
                RUNTIME_AUDIT["p2_small_target_score_sum"] += stats["p2_small_target_score_sum"]
                RUNTIME_AUDIT["final_gain"] = gain
                if stats["p2_small_positive_anchors"] > 0:
                    RUNTIME_AUDIT["batches_with_aux"] += 1
                    RUNTIME_AUDIT["raw_aux_iou_sum"] += float(aux_iou.detach().cpu())
                    RUNTIME_AUDIT["raw_aux_dfl_sum"] += float(aux_dfl.detach().cpu())
                    RUNTIME_AUDIT["weighted_aux_sum"] += float(raw_aux_graph.detach().abs().cpu())

        if self.head.training and not fg_mask.sum():
            RUNTIME_AUDIT["batches"] += 1
            RUNTIME_AUDIT["final_gain"] = gain

        # Standard E10a gains.  The GRAPH below is base-only.
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        base_total = loss.sum() * batch_size
        if self.audit_capture:
            # Expose the exact base subgraph used inside `total` so the preflight
            # can audit the actual incremental objective without numerical
            # cancellation between two independently-built base-loss graphs.
            self._last_base_total_graph = base_total

        # Add the auxiliary ONLY through the gradient-isolated proxy.
        # This is the critical causal-isolation property: there is no ordinary
        # aux graph left that could leak into P3/shared/backbone parameters.
        if self.head.training and gain > 0.0 and stats["p2_small_positive_anchors"] > 0:
            total = base_total + aux_proxy
        else:
            total = base_total

        # loss_items are reporting-only. Add detached auxiliary contributions to
        # box/dfl so logs reflect optimization magnitude without creating gradients.
        display_loss = loss.detach().clone()
        if self.head.training and gain > 0.0 and stats["p2_small_positive_anchors"] > 0:
            display_loss[0] += (aux_iou.detach() * gain * self.hyp.box)
            display_loss[2] += (aux_dfl.detach() * gain * self.hyp.dfl)

        # Runtime ratio uses the standard-base magnitude.
        if self.head.training:
            base_abs = float(base_total.detach().abs().cpu())
            RUNTIME_AUDIT["base_total_sum_abs"] += base_abs
            if stats["p2_small_positive_anchors"] > 0 and base_abs > 0:
                ratio = float(aux_proxy.detach().abs().cpu()) / base_abs
                RUNTIME_AUDIT["max_aux_to_base_abs_ratio"] = max(
                    RUNTIME_AUDIT["max_aux_to_base_abs_ratio"], ratio
                )

            if (
                self.settings.log_interval > 0
                and self._training_batches_seen > 0
                and self._training_batches_seen % self.settings.log_interval == 0
            ):
                LOGGER.info(
                    "E23a P2-RegQ: batch=%d gain=%.4f fg=%d p2small=%d "
                    "aux/base=%.4f",
                    self._training_batches_seen,
                    gain,
                    stats["fg_anchors"],
                    stats["p2_small_positive_anchors"],
                    float(aux_proxy.detach().abs().cpu()) / max(base_abs, 1e-12),
                )

        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite E23a total loss")

        return total, display_loss


def module_gradient_abs_sum(module: torch.nn.Module, scalar: torch.Tensor) -> float:
    params = [p for p in module.parameters() if p.requires_grad]
    if not params:
        return 0.0
    grads = torch.autograd.grad(
        scalar, params, retain_graph=True, create_graph=False, allow_unused=True
    )
    return float(
        sum(
            g.detach().abs().sum().cpu()
            for g in grads
            if g is not None
        )
    )


__all__ = (
    "P2RegQSettings",
    "v8P2RegQDetectionLoss",
    "reset_runtime_audit",
    "runtime_audit_snapshot",
    "module_gradient_abs_sum",
)
