"""Error-aware regional zoom distillation for the E14a VisDrone probe.

The module adds no inference layer.  During training, a frozen E1 detector
looks at a small number of object-centred zoom views.  Only class responses
that are more discriminative in the zoom view than in the full image are
distilled into the full-image P2/P3 logits.  Standard YOLO detection loss is
left unchanged and remains the dominant objective.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss


def _number(config: dict[str, Any], name: str, default: float) -> float:
    value = float(config.get(name, default))
    if not math.isfinite(value):
        raise ValueError(f"ER-ZD {name} must be finite, got {value}")
    return value


def _integer(config: dict[str, Any], name: str, default: int) -> int:
    value = int(config.get(name, default))
    if value < 0:
        raise ValueError(f"ER-ZD {name} must be non-negative, got {value}")
    return value


class RegionalZoomTargetBuilder:
    """Build sparse zoom-teacher targets from a preprocessed detection batch."""

    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)
        self.crop_fraction = _number(config, "crop_fraction", 0.5625)
        self.crops_per_batch = _integer(config, "crops_per_batch", 2)
        self.tiny_side_px = _number(config, "tiny_side_px", 32.0)
        self.max_targets_per_crop = _integer(config, "max_targets_per_crop", 48)
        self.anchor_candidates = _integer(config, "anchor_candidates", 16)
        self.levels = tuple(int(x) for x in config.get("levels", (0, 1)))
        self.teacher_patch_radius = tuple(int(x) for x in config.get("teacher_patch_radius", (3, 2)))
        if not 0.25 <= self.crop_fraction <= 0.9:
            raise ValueError("ER-ZD crop_fraction must be in [0.25, 0.9]")
        if not self.levels or len(self.levels) != len(self.teacher_patch_radius):
            raise ValueError("ER-ZD levels and teacher_patch_radius must have the same non-zero length")
        if (
            self.crops_per_batch < 1
            or self.max_targets_per_crop < 1
            or self.anchor_candidates < 1
            or self.tiny_side_px <= 0
        ):
            raise ValueError("ER-ZD crop/target limits and tiny_side_px must be positive")
        self.last_stats: dict[str, Any] = {}

    @staticmethod
    def _raw_features(output):
        if isinstance(output, tuple) and len(output) >= 2 and isinstance(output[1], (list, tuple)):
            return list(output[1])
        if isinstance(output, (list, tuple)) and output and all(torch.is_tensor(x) for x in output):
            return list(output)
        raise RuntimeError("Frozen ER-ZD teacher did not return raw Detect feature maps")

    @staticmethod
    def _window_vector(
        feature: torch.Tensor,
        x: float,
        y: float,
        radius: int,
        class_offset: int,
        class_id: int,
    ) -> torch.Tensor:
        height, width = feature.shape[-2:]
        ix = min(width - 1, max(0, int(math.floor(x * width))))
        iy = min(height - 1, max(0, int(math.floor(y * height))))
        x0, x1 = max(0, ix - radius), min(width, ix + radius + 1)
        y0, y1 = max(0, iy - radius), min(height, iy + radius + 1)
        logits = feature[class_offset:, y0:y1, x0:x1].float()
        if logits.numel() == 0:
            raise RuntimeError("ER-ZD teacher produced an empty response window")
        flattened = logits.flatten(1)
        # All class probabilities come from the same spatial cell: the cell
        # with the strongest correct-class response inside the local window.
        position = flattened[class_id].argmax()
        # Clone outside the teacher inference context so BCE autograd never
        # receives an inference-mode tensor as a saved target.
        return flattened[:, position].sigmoid().detach().clone()

    def _candidate_for_image(
        self,
        image_index: int,
        bboxes: torch.Tensor,
        classes: torch.Tensor,
        batch_index: torch.Tensor,
        height: int,
        width: int,
        crop_h: int,
        crop_w: int,
    ):
        indices = torch.nonzero(batch_index == image_index, as_tuple=False).flatten()
        if not len(indices):
            return None
        boxes = bboxes[indices]
        side = torch.sqrt((boxes[:, 2] * width).clamp_min(1e-6) * (boxes[:, 3] * height).clamp_min(1e-6))
        tiny_local = torch.nonzero(side <= self.tiny_side_px, as_tuple=False).flatten()
        if not len(tiny_local):
            return None

        best = None
        # Dense VisDrone images may contain hundreds of objects.  Bound the
        # CPU geometry search to the smallest targets to avoid a slow Python
        # loop while retaining the cases that benefit most from zooming.
        anchor_order = torch.argsort(side[tiny_local])[: self.anchor_candidates]
        anchor_locals = tiny_local[anchor_order]
        for anchor_local in anchor_locals.tolist():
            cx = float(boxes[anchor_local, 0] * width)
            cy = float(boxes[anchor_local, 1] * height)
            x0 = int(round(cx - crop_w / 2))
            y0 = int(round(cy - crop_h / 2))
            x0 = min(max(0, x0), width - crop_w)
            y0 = min(max(0, y0), height - crop_h)
            # Four-pixel alignment makes the full-image P2 mapping reproducible.
            x0 = min((x0 // 4) * 4, width - crop_w)
            y0 = min((y0 // 4) * 4, height - crop_h)
            x1, y1 = x0 + crop_w, y0 + crop_h
            left = (boxes[:, 0] - boxes[:, 2] / 2) * width
            right = (boxes[:, 0] + boxes[:, 2] / 2) * width
            top = (boxes[:, 1] - boxes[:, 3] / 2) * height
            bottom = (boxes[:, 1] + boxes[:, 3] / 2) * height
            inside = (left >= x0 + 1) & (right <= x1 - 1) & (top >= y0 + 1) & (bottom <= y1 - 1)
            selected_local = torch.nonzero(inside & (side <= self.tiny_side_px), as_tuple=False).flatten()
            if not len(selected_local):
                continue
            difficulty = (self.tiny_side_px / side[selected_local].clamp_min(2.0)).clamp(0.5, 3.0)
            score = float(difficulty.sum())
            proposal = {
                "image_index": image_index,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "target_indices": indices[selected_local],
                "target_sides": side[selected_local],
                "score": score,
                "anchor_local": anchor_local,
                "classes": classes,
            }
            if best is None or (proposal["score"], -anchor_local) > (best["score"], -best["anchor_local"]):
                best = proposal
        return best

    def __call__(self, teacher, batch: dict[str, Any]) -> dict[str, Any]:
        images = batch["img"]
        if images.ndim != 4 or images.shape[-2] != images.shape[-1]:
            raise ValueError(f"ER-ZD expects a square BCHW training batch, got {tuple(images.shape)}")
        # Geometry is tiny compared with image tensors and is faster on CPU;
        # teacher response vectors remain on the accelerator.
        batch_index = batch["batch_idx"].detach().view(-1).to(device="cpu", dtype=torch.long)
        bboxes = batch["bboxes"].detach().view(-1, 4).to(device="cpu", dtype=torch.float32)
        classes = batch["cls"].detach().view(-1).to(device="cpu", dtype=torch.long)
        height, width = int(images.shape[-2]), int(images.shape[-1])
        crop_h = max(32, int(round(height * self.crop_fraction / 32.0)) * 32)
        crop_w = max(32, int(round(width * self.crop_fraction / 32.0)) * 32)
        crop_h, crop_w = min(height, crop_h), min(width, crop_w)

        candidates = []
        for image_index in range(images.shape[0]):
            candidate = self._candidate_for_image(
                image_index, bboxes, classes, batch_index, height, width, crop_h, crop_w
            )
            if candidate is not None:
                candidates.append(candidate)
        candidates.sort(key=lambda row: (-row["score"], row["image_index"]))
        candidates = candidates[: self.crops_per_batch]
        if not candidates:
            self.last_stats = {
                "selected_crops": 0,
                "eligible_targets": 0,
                "zoom_factor": width / crop_w,
            }
            return {"entries": [], "builder_stats": dict(self.last_stats)}

        zoom_images = torch.cat(
            [
                F.interpolate(
                    images[row["image_index"] : row["image_index"] + 1, :, row["y0"] : row["y1"], row["x0"] : row["x1"]],
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
                for row in candidates
            ],
            dim=0,
        )
        teacher.eval()
        with torch.inference_mode():
            raw = self._raw_features(teacher(zoom_images))
        detect = teacher.model[-1]
        class_offset = int(detect.reg_max) * 4
        if len(raw) <= max(self.levels):
            raise RuntimeError(f"ER-ZD teacher has only {len(raw)} detection levels")

        entries = []
        confidences = []
        for crop_index, row in enumerate(candidates):
            target_indices = row["target_indices"]
            target_sides = row["target_sides"]
            order = torch.argsort(target_sides)[: self.max_targets_per_crop]
            for local_position in order.tolist():
                target_index = int(target_indices[local_position])
                x, y = (float(v) for v in bboxes[target_index, :2])
                zx = (x * width - row["x0"]) / crop_w
                zy = (y * height - row["y0"]) / crop_h
                if not (0.0 <= zx <= 1.0 and 0.0 <= zy <= 1.0):
                    continue
                class_id = int(classes[target_index])
                teacher_probabilities = []
                for level, radius in zip(self.levels, self.teacher_patch_radius):
                    vector = self._window_vector(
                        raw[level][crop_index], zx, zy, radius, class_offset, class_id
                    )
                    teacher_probabilities.append(vector)
                confidences.extend(vector[class_id].detach() for vector in teacher_probabilities)
                entries.append(
                    {
                        "image_index": row["image_index"],
                        "class_id": class_id,
                        "full_xy": (x, y),
                        "side_px": float(target_sides[local_position]),
                        "teacher_probabilities": teacher_probabilities,
                    }
                )

        self.last_stats = {
            "selected_crops": len(candidates),
            "eligible_targets": len(entries),
            "zoom_factor": width / crop_w,
            "crop_shape": [crop_h, crop_w],
            "teacher_target_conf_mean": (
                float(torch.stack(confidences).mean()) if confidences else None
            ),
        }
        del zoom_images, raw
        return {"entries": entries, "builder_stats": dict(self.last_stats)}


class v8ERZDDetectionLoss(v8DetectionLoss):
    """Standard v8 detection loss plus sparse, teacher-superior class distillation."""

    def __init__(self, model):
        super().__init__(model)
        config = model.yaml.get("erzd", {})
        if not isinstance(config, dict) or not config.get("enabled", False):
            raise ValueError("v8ERZDDetectionLoss requires yaml.erzd.enabled=true")
        self.levels = tuple(int(x) for x in config.get("levels", (0, 1)))
        self.student_patch_radius = tuple(int(x) for x in config.get("student_patch_radius", (1, 1)))
        if len(self.levels) != len(self.student_patch_radius):
            raise ValueError("ER-ZD levels and student_patch_radius lengths differ")
        self.min_teacher_conf = _number(config, "min_teacher_conf", 0.08)
        self.margin = _number(config, "teacher_margin", 0.03)
        self.wrong_class_weight = _number(config, "wrong_class_weight", 0.5)
        self.tiny_side_px = _number(config, "tiny_side_px", 32.0)
        self.last_stats: dict[str, Any] = {}

    @staticmethod
    def _student_window(
        feature: torch.Tensor,
        image_index: int,
        xy,
        radius: int,
        class_offset: int,
        class_id: int,
    ):
        height, width = feature.shape[-2:]
        x, y = xy
        ix = min(width - 1, max(0, int(math.floor(float(x) * width))))
        iy = min(height - 1, max(0, int(math.floor(float(y) * height))))
        x0, x1 = max(0, ix - radius), min(width, ix + radius + 1)
        y0, y1 = max(0, iy - radius), min(height, iy + radius + 1)
        flattened = feature[image_index, class_offset:, y0:y1, x0:x1].float().flatten(1)
        position = flattened[class_id].detach().argmax()
        return flattened[:, position]

    def _distillation(self, feats, payload):
        zero = sum(feature.sum() * 0.0 for feature in feats[:1])
        if not isinstance(payload, dict) or not payload.get("entries"):
            return zero, {"candidate_pairs": 0, "active_pairs": 0, "raw_loss": 0.0}
        class_offset = self.reg_max * 4
        student_vectors = []
        teacher_vectors = []
        class_ids = []
        sides = []
        for entry in payload["entries"]:
            class_id = int(entry["class_id"])
            if not 0 <= class_id < self.nc:
                raise ValueError(f"ER-ZD target class outside [0,{self.nc}): {class_id}")
            for position, (level, radius) in enumerate(zip(self.levels, self.student_patch_radius)):
                student_logits = self._student_window(
                    feats[level],
                    int(entry["image_index"]),
                    entry["full_xy"],
                    radius,
                    class_offset,
                    class_id,
                )
                teacher_probs = entry["teacher_probabilities"][position].to(
                    device=student_logits.device, dtype=torch.float32
                )
                if teacher_probs.numel() != self.nc:
                    raise ValueError("ER-ZD teacher class-vector width differs from the detector")
                student_vectors.append(student_logits)
                teacher_vectors.append(teacher_probs)
                class_ids.append(class_id)
                sides.append(float(entry["side_px"]))
        candidate_pairs = len(student_vectors)
        if not candidate_pairs:
            return zero, {"candidate_pairs": 0, "active_pairs": 0, "raw_loss": 0.0}

        student_logits = torch.stack(student_vectors)
        teacher_probs = torch.stack(teacher_vectors)
        classes = torch.tensor(class_ids, device=student_logits.device, dtype=torch.long)
        side_tensor = torch.tensor(sides, device=student_logits.device, dtype=torch.float32)
        rows = torch.arange(candidate_pairs, device=student_logits.device)
        student_probs = student_logits.detach().sigmoid()
        wrong_scores = student_probs.clone()
        wrong_scores.scatter_(1, classes[:, None], -1.0)
        wrong_classes = wrong_scores.argmax(1)
        teacher_pos = teacher_probs[rows, classes]
        teacher_wrong = teacher_probs[rows, wrong_classes]
        student_pos = student_probs[rows, classes]
        student_wrong = student_probs[rows, wrong_classes]
        advantage = (teacher_pos - teacher_wrong) - (student_pos - student_wrong)
        active = (teacher_pos >= self.min_teacher_conf) & (advantage > self.margin)

        # Targets are one-sided: ER-ZD cannot lower the correct-class response
        # or raise the student's strongest competing class.
        positive_target = torch.maximum(teacher_pos, student_pos).detach()
        wrong_target = torch.minimum(teacher_wrong, student_wrong).detach()
        positive_loss = F.binary_cross_entropy_with_logits(
            student_logits[rows, classes], positive_target, reduction="none"
        )
        wrong_loss = F.binary_cross_entropy_with_logits(
            student_logits[rows, wrong_classes], wrong_target, reduction="none"
        )
        smallness = torch.sqrt(self.tiny_side_px / side_tensor.clamp_min(2.0)).clamp(0.5, 2.0)
        weights = smallness * advantage.detach().clamp(0.05, 1.0) * active.to(torch.float32)
        distill = ((positive_loss + self.wrong_class_weight * wrong_loss) * weights).sum() / weights.sum().clamp_min(1e-6)
        active_pairs = int(active.sum().detach())
        stats = {
            "candidate_pairs": candidate_pairs,
            "active_pairs": active_pairs,
            "raw_loss": float(distill.detach()),
            "teacher_advantage_mean": (
                float(advantage[active].mean().detach()) if active_pairs else None
            ),
        }
        return distill, stats

    def __call__(self, preds, batch):
        base_total, base_items = super().__call__(preds, batch)
        feats = preds[1] if isinstance(preds, tuple) else preds
        payload = batch.get("_erzd")
        distill, stats = self._distillation(feats, payload)
        loss_weight = float(payload.get("loss_weight", 0.0)) if isinstance(payload, dict) else 0.0
        if loss_weight < 0 or not math.isfinite(loss_weight):
            raise ValueError(f"Invalid ER-ZD loss weight: {loss_weight}")
        batch_size = feats[0].shape[0]
        scaled = distill * loss_weight
        items = base_items.clone()
        items[1] += scaled.detach()
        stats.update(loss_weight=loss_weight, scaled_loss=float(scaled.detach()))
        if isinstance(payload, dict):
            stats["builder_stats"] = payload.get("builder_stats", {})
        self.last_stats = stats
        return base_total + scaled * batch_size, items


__all__ = ("RegionalZoomTargetBuilder", "v8ERZDDetectionLoss")
