"""E32a training-only Scale-Consistent Auxiliary Supervision (SCAS).

E32a changes only the target construction used by E18a's shared P3/P4
auxiliary head.  The inference graph, main Detect loss and TAL assignment are
untouched.

The two auxiliary levels retain E18a's original equal level weighting.  What
changes is target ownership:

* classification: every valid GT owns exactly one level (P4 if P4-resolvable,
  otherwise P3);
* regression: a GT owns P4 if P4-resolvable, otherwise P3 if P3-resolvable,
  otherwise neither level;
* negative classification: every real GT is ignored in a 3x3 neighbourhood
  on both levels, including non-owner levels.

Identical (image, class, cell) classification targets are coalesced because a
single class logit cannot represent instance multiplicity.  Class-agnostic
regression keeps E18a's existing largest-box-per-cell rule.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

import torch
from torch.nn import functional as F

from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8DetectionLoss


# Survives criterion/model cleanup long enough for the runner to archive the
# augmented-batch routing evidence after training/probing.
LATEST_E32_ONLINE_STATS: dict[str, Any] = {}


def _zero(logits: torch.Tensor) -> torch.Tensor:
    return logits.sum() * 0.0


def _first_index_per_key(keys: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return deterministic first indices and the number of merged rows."""
    if keys.numel() == 0:
        return keys.new_empty((0,), dtype=torch.long), 0
    _, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    groups = int(inverse.max().item()) + 1
    rows = torch.arange(keys.numel(), device=keys.device, dtype=torch.long)
    first = torch.full((groups,), keys.numel(), device=keys.device, dtype=torch.long)
    first.scatter_reduce_(0, inverse, rows, reduce="amin", include_self=True)
    return first, int(keys.numel() - first.numel())


def _largest_index_per_key(keys: torch.Tensor, areas: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Select one deterministic largest-area target per class-agnostic cell."""
    if keys.numel() == 0:
        return keys.new_empty((0,), dtype=torch.long), 0
    _, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    groups = int(inverse.max().item()) + 1
    largest = torch.full((groups,), -1.0, device=areas.device, dtype=areas.dtype)
    largest.scatter_reduce_(0, inverse, areas, reduce="amax", include_self=True)
    candidate = areas == largest[inverse]
    rows = torch.arange(keys.numel(), device=keys.device, dtype=torch.long)
    rows = torch.where(candidate, rows, keys.numel())
    take = torch.full((groups,), keys.numel(), device=keys.device, dtype=torch.long)
    take.scatter_reduce_(0, inverse, rows, reduce="amin", include_self=True)
    return take, int(keys.numel() - take.numel())


class v8E32DetectionLoss(v8DetectionLoss):
    """Standard YOLO loss plus E18's auxiliary head with SCAS targets."""

    def __init__(self, model):
        super().__init__(model)
        base = model.yaml.get("e17_aux", {})
        cfg = model.yaml.get("e32_aux", {})
        if not isinstance(base, dict) or not isinstance(cfg, dict):
            raise TypeError("E32a requires e17_aux and e32_aux dictionaries")
        if not cfg.get("enabled", False):
            raise ValueError("v8E32DetectionLoss constructed while e32_aux is disabled")

        self.gain = float(base.get("gain", 0.1))
        self.minimum_side = float(base.get("min_feature_side", 1.5))
        self.negative_weight = float(base.get("negative_weight", 0.1))
        self.box_weight = float(base.get("box_weight", 0.25))
        if not 0.0 < self.gain <= 1.0:
            raise ValueError("e17_aux.gain must remain in (0, 1]")
        if self.minimum_side <= 0.0 or self.negative_weight < 0.0 or self.box_weight < 0.0:
            raise ValueError("Invalid frozen E18 auxiliary settings")
        if cfg.get("classification_owner") != "p4_if_valid_else_p3":
            raise ValueError("Unexpected E32 classification routing")
        if cfg.get("regression_owner") != "coarsest_valid_else_none":
            raise ValueError("Unexpected E32 regression routing")
        if cfg.get("negative_ignore") != "all_gt_both_levels_3x3":
            raise ValueError("Unexpected E32 negative-ignore policy")
        if cfg.get("level_aggregation") != "equal_level_mean":
            raise ValueError("E32 must preserve E18 equal P3/P4 level weighting")

        self.head = model.model[-1]
        if self.head.__class__.__name__ != "E17Detect":
            raise TypeError("E32a requires E18a's E17Detect auxiliary head")
        self._online: defaultdict[str, float] = defaultdict(float)
        self._batches = 0
        self.last_aux_stats: dict[str, Any] | None = None

    @staticmethod
    def _valid_gt(batch: dict, batch_size: int, nc: int, device: torch.device):
        boxes = batch["bboxes"].detach().to(device=device, dtype=torch.float32)
        classes = batch["cls"].detach().reshape(-1).to(device=device, dtype=torch.long)
        images = batch["batch_idx"].detach().reshape(-1).to(device=device, dtype=torch.long)
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError(f"Expected normalized xywh boxes Nx4, got {tuple(boxes.shape)}")
        if boxes.shape[0] != classes.shape[0] or boxes.shape[0] != images.shape[0]:
            raise ValueError("E32 GT tensors have inconsistent lengths")
        valid = (
            torch.isfinite(boxes).all(1)
            & (boxes[:, 2] > 0)
            & (boxes[:, 3] > 0)
            & (classes >= 0)
            & (classes < nc)
            & (images >= 0)
            & (images < batch_size)
        )
        return boxes[valid], classes[valid], images[valid]

    @staticmethod
    def _centres(boxes: torch.Tensor, grid_h: int, grid_w: int):
        x = (boxes[:, 0] * grid_w).floor().clamp(0, grid_w - 1).long()
        y = (boxes[:, 1] * grid_h).floor().clamp(0, grid_h - 1).long()
        return x, y

    def _classification_loss(
        self,
        logits: torch.Tensor,
        boxes: torch.Tensor,
        classes: torch.Tensor,
        images: torch.Tensor,
        owner: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        _, _, grid_h, grid_w = logits.shape
        selected = owner.nonzero(as_tuple=False).flatten()
        if selected.numel() == 0:
            return _zero(logits[:, : self.nc]), {"classification_rows": 0, "classification_collisions": 0}
        x, y = self._centres(boxes[selected], grid_h, grid_w)
        cls = classes[selected]
        img = images[selected]
        keys = (((img * self.nc + cls) * grid_h + y) * grid_w + x)
        take, collisions = _first_index_per_key(keys)
        loss = F.softplus(-logits[img[take], cls[take], y[take], x[take]]).mean()
        return loss, {
            "classification_rows": int(selected.numel()),
            "classification_unique": int(take.numel()),
            "classification_collisions": collisions,
        }

    def _negative_loss(
        self,
        logits: torch.Tensor,
        boxes: torch.Tensor,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        batch_size, _, grid_h, grid_w = logits.shape
        allowed = torch.ones((batch_size, 1, grid_h, grid_w), device=logits.device, dtype=torch.bool)
        if boxes.numel():
            x, y = self._centres(boxes, grid_h, grid_w)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    allowed[images, 0, (y + dy).clamp(0, grid_h - 1), (x + dx).clamp(0, grid_w - 1)] = False
        values = F.softplus(logits[:, : self.nc])[allowed.expand(-1, self.nc, -1, -1)]
        loss = values.mean() if values.numel() else _zero(logits[:, : self.nc])
        ignored = int((~allowed).sum().item())
        return loss, {
            "ignored_cells": ignored,
            "total_cells": int(allowed.numel()),
            "negative_logits": int(values.numel()),
        }

    def _regression_loss(
        self,
        logits: torch.Tensor,
        boxes: torch.Tensor,
        images: torch.Tensor,
        owner: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        _, _, grid_h, grid_w = logits.shape
        selected = owner.nonzero(as_tuple=False).flatten()
        if selected.numel() == 0:
            return _zero(logits[:, self.nc :]), {"regression_rows": 0, "regression_collisions": 0}
        selected_boxes = boxes[selected]
        img = images[selected]
        x, y = self._centres(selected_boxes, grid_h, grid_w)
        keys = (img * grid_h + y) * grid_w + x
        areas = selected_boxes[:, 2] * selected_boxes[:, 3]
        take, collisions = _largest_index_per_key(keys, areas)

        chosen = selected_boxes[take]
        cx = (x[take].float() + 0.5) / grid_w
        cy = (y[take].float() + 0.5) / grid_h
        distances = torch.stack(
            (
                (cx - chosen[:, 0] + chosen[:, 2] / 2) * grid_w,
                (cy - chosen[:, 1] + chosen[:, 3] / 2) * grid_h,
                (chosen[:, 0] + chosen[:, 2] / 2 - cx) * grid_w,
                (chosen[:, 1] + chosen[:, 3] / 2 - cy) * grid_h,
            ),
            dim=1,
        ).clamp_min(0)
        target = torch.log1p(distances)
        predicted = logits[img[take], self.nc :, y[take], x[take]]
        loss = F.smooth_l1_loss(predicted.float(), target)
        return loss, {
            "regression_rows": int(selected.numel()),
            "regression_unique": int(take.numel()),
            "regression_collisions": collisions,
        }

    def _level_loss(
        self,
        name: str,
        logits: torch.Tensor,
        boxes: torch.Tensor,
        classes: torch.Tensor,
        images: torch.Tensor,
        classification_owner: torch.Tensor,
        regression_owner: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        positive, positive_stats = self._classification_loss(
            logits, boxes, classes, images, classification_owner
        )
        negative, negative_stats = self._negative_loss(logits, boxes, images)
        regression, regression_stats = self._regression_loss(logits, boxes, images, regression_owner)
        total = positive + self.negative_weight * negative + self.box_weight * regression
        stats: dict[str, Any] = {
            "level": name,
            **positive_stats,
            **negative_stats,
            **regression_stats,
            "positive_loss": positive.detach(),
            "negative_loss": negative.detach(),
            "regression_loss": regression.detach(),
            "level_loss": total.detach(),
        }
        return total, stats

    def auxiliary_loss(self, logits, batch: dict) -> tuple[torch.Tensor, dict[str, Any]]:
        if not isinstance(logits, (tuple, list)) or len(logits) != 2:
            raise ValueError("E32 expects auxiliary logits ordered as (P3, P4)")
        p3, p4 = logits
        if p3.ndim != 4 or p4.ndim != 4 or p3.shape[0] != p4.shape[0]:
            raise ValueError("Invalid E32 auxiliary logits")
        if p3.shape[1] != self.nc + 4 or p4.shape[1] != self.nc + 4:
            raise ValueError("Unexpected E32 auxiliary channel count")
        if p3.shape[-2:] != (2 * p4.shape[-2], 2 * p4.shape[-1]):
            raise ValueError("E32 expects P3 auxiliary grid to be exactly 2x P4")

        boxes, classes, images = self._valid_gt(batch, p3.shape[0], self.nc, p3.device)
        p3_h, p3_w = p3.shape[-2:]
        p4_h, p4_w = p4.shape[-2:]
        p3_valid = (
            (boxes[:, 2] * p3_w >= self.minimum_side)
            & (boxes[:, 3] * p3_h >= self.minimum_side)
        )
        p4_valid = (
            (boxes[:, 2] * p4_w >= self.minimum_side)
            & (boxes[:, 3] * p4_h >= self.minimum_side)
        )
        if bool((p4_valid & ~p3_valid).any()):
            raise RuntimeError("P4-valid GT must also be P3-valid")

        cls_p4 = p4_valid
        cls_p3 = ~p4_valid
        reg_p4 = p4_valid
        reg_p3 = p3_valid & ~p4_valid
        if boxes.shape[0] and not bool(((cls_p3.to(torch.int8) + cls_p4.to(torch.int8)) == 1).all()):
            raise RuntimeError("Every E32 GT must have exactly one classification owner")
        if bool((reg_p3 & reg_p4).any()):
            raise RuntimeError("E32 regression ownership is not exclusive")

        p3_loss, p3_stats = self._level_loss(
            "P3", p3, boxes, classes, images, cls_p3, reg_p3
        )
        p4_loss, p4_stats = self._level_loss(
            "P4", p4, boxes, classes, images, cls_p4, reg_p4
        )
        # Preserve E18a's layer-balanced aggregation.  Do not pool positives
        # across levels: that would silently reduce P4's weight to its GT share.
        auxiliary = 0.5 * (p3_loss + p4_loss)
        stats = {
            "gt": int(boxes.shape[0]),
            "p3_valid": int(p3_valid.sum().item()),
            "p4_valid": int(p4_valid.sum().item()),
            "old_duplicate_positive": int(p4_valid.sum().item()),
            "classification_p3": int(cls_p3.sum().item()),
            "classification_p4": int(cls_p4.sum().item()),
            "regression_p3": int(reg_p3.sum().item()),
            "regression_p4": int(reg_p4.sum().item()),
            "regression_none": int((~p3_valid).sum().item()),
            "P3": p3_stats,
            "P4": p4_stats,
            "auxiliary_loss": auxiliary.detach(),
        }
        return auxiliary, stats

    def _accumulate(self, stats: dict[str, Any]) -> None:
        self._batches += 1
        for key in (
            "gt",
            "p3_valid",
            "p4_valid",
            "old_duplicate_positive",
            "classification_p3",
            "classification_p4",
            "regression_p3",
            "regression_p4",
            "regression_none",
        ):
            self._online[key] += int(stats[key])
        for level in ("P3", "P4"):
            row = stats[level]
            for key in (
                "classification_rows",
                "classification_unique",
                "classification_collisions",
                "regression_rows",
                "regression_unique",
                "regression_collisions",
                "ignored_cells",
                "total_cells",
                "negative_logits",
            ):
                self._online[f"{level}_{key}"] += int(row.get(key, 0))
        global LATEST_E32_ONLINE_STATS
        LATEST_E32_ONLINE_STATS = self.stats_snapshot()

    def stats_snapshot(self) -> dict[str, Any]:
        data = {key: int(value) for key, value in self._online.items()}
        total = max(1, data.get("gt", 0))
        reg_total = max(1, data.get("regression_p3", 0) + data.get("regression_p4", 0))
        data.update(
            batches=int(self._batches),
            classification_p3_fraction=data.get("classification_p3", 0) / total,
            classification_p4_fraction=data.get("classification_p4", 0) / total,
            regression_p3_fraction_of_owned=data.get("regression_p3", 0) / reg_total,
            regression_p4_fraction_of_owned=data.get("regression_p4", 0) / reg_total,
            regression_none_fraction=data.get("regression_none", 0) / total,
        )
        for level in ("P3", "P4"):
            ignored = data.get(f"{level}_ignored_cells", 0)
            cells = max(1, data.get(f"{level}_total_cells", 0))
            data[f"{level}_ignore_fraction"] = ignored / cells
        return deepcopy(data)

    def __call__(self, preds, batch):
        logits = self.head.take_aux_logits()
        if logits is None and not self.head.training:
            return super().__call__(preds, batch)
        try:
            main, items = super().__call__(preds, batch)
            if logits is None:
                raise RuntimeError("Training E17Detect did not provide E32 auxiliary logits")
            # Enables an exact main-loss/gradient preflight without constructing
            # an auxiliary zero-gradient path.
            if self.gain == 0.0:
                return main, items
            auxiliary, stats = self.auxiliary_loss(logits, batch)
            if not torch.isfinite(auxiliary):
                raise FloatingPointError("Non-finite E32 auxiliary loss")
            self.last_aux_stats = stats
            self._accumulate(stats)
            if self._batches % 1000 == 0:
                snap = self.stats_snapshot()
                LOGGER.info(
                    "E32 SCAS: batches=%d GT=%d cls(P3/P4)=%.3f/%.3f reg-owned(P3/P4)=%.3f/%.3f none=%.3f",
                    snap["batches"],
                    snap.get("gt", 0),
                    snap["classification_p3_fraction"],
                    snap["classification_p4_fraction"],
                    snap["regression_p3_fraction_of_owned"],
                    snap["regression_p4_fraction_of_owned"],
                    snap["regression_none_fraction"],
                )
            return main + self.gain * auxiliary * batch["img"].shape[0], items
        finally:
            self.head.clear_prior_logits()


__all__ = ("LATEST_E32_ONLINE_STATS", "v8E32DetectionLoss")
