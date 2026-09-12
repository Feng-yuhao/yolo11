#!/usr/bin/env python3
"""D16a grouped out-of-fold cross-view reliability calibration probe.

The frozen E1 detector and the frozen D15a 1.5-crop router are unchanged.
For every selected local candidate, a small residual MLP predicts whether the
candidate is geometrically and semantically reliable using only inference-time
full/local evidence.  Validation labels supervise other sequence groups only:
the prediction for an image is produced by a calibrator that never saw labels
from that image's sequence group.

This is a decision probe, not a final test-set claim.  It does not modify
Ultralytics, the detector checkpoint, the router, or any detector feature map.
It never shuts down the instance.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import evaluate_e1_e5a_slicing as core


ROOT = Path("/root/autodl-tmp/yolo11_project")
REPORTS = ROOT / "comparison_reports"
WEIGHT = ROOT / "runs/e1_yolo11s_p2add_img1024_seed1/weights/best.pt"
D15B_GLOB = "DIAG_D15B_e1_correction_fusion_headroom_*/metrics.json"

SCRIPT_REVISION = "d16a_grouped_nested_oof_cvrc_v1"
EXPECTED_CORE_REVISION = "fixed_2x2_slicing_upper_bound_v2"
PRIMARY_BUDGET = 1.50
ROUTER_CONF = 0.05
ROUTER_SMALL_EFFECTIVE_SIDE = 32.0
ROUTER_UNCERTAINTY_WEIGHT = 0.5
TARGET_IOU = 0.50
SMALL_AREA = 32.0 ** 2
OUTER_FOLDS = 5
INNER_FRACTION = 0.20
EPOCHS = 14
BATCH_SIZE = 4096
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
NEGATIVE_RATIO = 5
MIN_NEGATIVES_PER_IMAGE = 64
ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)
REFERENCE_TOLERANCE = 0.002
MAP_GATE = 0.002
APS_GATE = 0.003
APM_FLOOR = -0.005

FEATURE_NAMES = (
    "local_logit", "local_conf", "log_area_ratio", "effective_side_over_32",
    "log_aspect", "center_x", "center_y", "crop_border_min",
    "crop_border_x", "crop_border_y", "log1p_tile_proxy", "tile_rank",
    "full_same_iou", "full_same_conf_at_iou", "full_same_max_conf",
    "full_any_iou", "full_any_conf_at_iou", "full_any_class_agree",
    "local_minus_full_same_conf", "log_area_ratio_to_full_same",
    "peer_same_iou", "peer_same_conf", "peer_same_count",
    "tile_0", "tile_1", "tile_2", "tile_3",
    "class_0", "class_1", "class_2", "class_3", "class_4",
    "class_5", "class_6", "class_7", "class_8", "class_9",
)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def save(report: dict) -> None:
    report["updated_at"] = now()
    out = Path(report["report_dir"])
    core.atomic_json(out / "metrics.json", report)
    core.atomic_text(out / "metrics.txt", report_text(report))


def report_text(report: dict) -> str:
    lines = [
        "D16a grouped nested out-of-fold cross-view reliability calibration",
        f"status: {report.get('status')}",
        f"revision: {SCRIPT_REVISION}",
        f"started_at: {report.get('started_at')}",
        f"finished_at: {report.get('finished_at')}",
        "",
        "No validation image is scored by a calibrator trained on its sequence group.",
        "This is an OOF decision probe; final claims still require an untouched test set.",
        "E1, D15a routing, Ultralytics and all detector weights remain frozen.",
        "",
        "[modes]",
    ]
    for name, row in report.get("modes", {}).items():
        overall = row.get("overall", {})
        lines.append(
            f"{name}: mAP50_95={overall.get('mAP50_95')}, "
            f"mAP50={overall.get('mAP50')}, mAP75={overall.get('mAP75')}, "
            f"AP_small={overall.get('AP_small')}, AR_small={overall.get('AR_small')}, "
            f"AP_medium={overall.get('AP_medium')}, AP_large={overall.get('AP_large')}"
        )
    lines.extend([
        "", "[decision]",
        json.dumps(report.get("decision"), ensure_ascii=False, indent=2),
        "", "[folds]",
        json.dumps(report.get("folds"), ensure_ascii=False, indent=2),
        "", "[full_metadata]",
        json.dumps(report, ensure_ascii=False, indent=2, default=core.json_default),
    ])
    return "\n".join(lines) + "\n"


def latest_d15b() -> tuple[Path, dict]:
    rows = []
    for path in sorted(REPORTS.glob(D15B_GLOB)):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        decision = value.get("decisions", {})
        modes = value.get("models", {}).get("e1", {}).get("modes", {})
        if (
            value.get("status") == "completed"
            and value.get("is_full_diagnostic") is True
            and decision.get("selected_branch") == "cross_view_local_candidate_verifier"
            and "proxy_b1p50_standard" in modes
        ):
            rows.append((path, value))
    if not rows:
        raise FileNotFoundError("No completed full D15b verifier-headroom report found")
    return rows[-1]


def annotation_index(gt: dict) -> dict[str, dict]:
    path_by_id = {int(row["id"]): str(row["file_name"]) for row in gt["images"]}
    values = {path: [] for path in path_by_id.values()}
    for ann in gt["annotations"]:
        x, y, width, height = map(float, ann["bbox"])
        values[path_by_id[int(ann["image_id"])]].append({
            "id": int(ann["id"]),
            "category_id": int(ann["category_id"]),
            "box": core.np.asarray([x, y, x + width, y + height], dtype=core.np.float32),
            "area": float(ann["area"]),
        })
    result = {}
    for path, annotations in values.items():
        by_class = {}
        for ann in annotations:
            by_class.setdefault(ann["category_id"], []).append(ann)
        result[path] = {"rows": annotations, "by_class": by_class}
    return result


def image_meta(gt: dict) -> dict[str, dict]:
    return {
        str(row["file_name"]): {
            "width": int(row["width"]), "height": int(row["height"]),
            "image_id": int(row["id"]),
        }
        for row in gt["images"]
    }


def assigned_tile(cx: float, cy: float, width: int, height: int) -> int:
    return (0 if cy < height / 2 else 1) * 2 + (0 if cx < width / 2 else 1)


def proxy_scores(detections, meta: dict) -> tuple[list[float], list[int]]:
    scores = [0.0] * 4
    counts = [0] * 4
    resize_scale = core.IMGSZ / max(meta["width"], meta["height"])
    for x1, y1, x2, y2, confidence, _cls in detections.tolist():
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        effective_side = math.sqrt(width * height) * resize_scale
        if confidence < ROUTER_CONF or effective_side >= ROUTER_SMALL_EFFECTIVE_SIDE:
            continue
        tile = assigned_tile(
            (x1 + x2) / 2, (y1 + y2) / 2, meta["width"], meta["height"]
        )
        evidence = math.sqrt(max(confidence, 0.0))
        deficit = max(0.0, 1.0 - effective_side / ROUTER_SMALL_EFFECTIVE_SIDE)
        uncertainty = 4.0 * confidence * (1.0 - confidence)
        scores[tile] += evidence * (1.0 + deficit) * (
            1.0 + ROUTER_UNCERTAINTY_WEIGHT * uncertainty
        )
        counts[tile] += 1
    return scores, counts


def build_proxy_rows(images: list[Path], full: dict, meta: dict) -> list[dict]:
    rows = []
    for path in images:
        key = str(path)
        scores, counts = proxy_scores(full[key], meta[key])
        order = sorted(range(4), key=lambda index: (-scores[index], index))
        rows.append({
            "image": key, "tile_scores": scores, "candidate_counts": counts,
            "ranked_tiles": order,
            "ranked_scores": [scores[index] for index in order],
        })
    return rows


def allocate_budget(rows: list[dict], budget: float) -> tuple[dict[str, list[int]], dict]:
    requested = int(round(budget * len(rows)))
    marginals = []
    for row in rows:
        for rank in (0, 1):
            marginals.append((float(row["ranked_scores"][rank]), rank, row["image"]))
    marginals.sort(key=lambda item: (-item[0], item[1], item[2]))
    ranks = {}
    for _score, rank, key in marginals[:requested]:
        ranks.setdefault(key, set()).add(rank)
    selections = {}
    distribution = {"0": 0, "1": 0, "2": 0}
    for row in rows:
        chosen = ranks.get(row["image"], set())
        if 1 in chosen and 0 not in chosen:
            raise RuntimeError("D15a rank-prefix constraint was violated")
        count = len(chosen)
        selections[row["image"]] = row["ranked_tiles"][:count]
        distribution[str(count)] += 1
    return selections, {
        "requested": requested,
        "actual_crops_per_image": sum(map(len, selections.values())) / len(rows),
        "images_by_crop_count": distribution,
        "last_selected_score": marginals[requested - 1][0],
        "first_rejected_score": marginals[requested][0],
    }


def nms_with_source(tensors: list, tile_ids: list[int]):
    if not tensors:
        return core.empty_detections(), core.np.empty((0,), dtype=core.np.int64)
    data = core.torch.cat(tensors, dim=0)
    sources = core.np.concatenate([
        core.np.full((len(tensor),), tile, dtype=core.np.int64)
        for tensor, tile in zip(tensors, tile_ids)
    ])
    finite = core.torch.isfinite(data).all(dim=1)
    data = data[finite]
    sources = sources[finite.numpy()]
    if data.numel() == 0:
        return core.empty_detections(), core.np.empty((0,), dtype=core.np.int64)
    keep = core.batched_nms(data[:, :4], data[:, 4], data[:, 5].long(), core.MERGE_NMS_IOU)
    keep = keep[:core.MAX_DET]
    return data[keep].contiguous(), sources[keep.numpy()]


def pairwise_iou(one, two):
    if len(one) == 0 or len(two) == 0:
        return core.np.zeros((len(one), len(two)), dtype=core.np.float32)
    lt = core.np.maximum(one[:, None, :2], two[None, :, :2])
    rb = core.np.minimum(one[:, None, 2:], two[None, :, 2:])
    inter = core.np.prod(core.np.maximum(0.0, rb - lt), axis=2)
    area_one = core.np.maximum(0.0, one[:, 2] - one[:, 0]) * core.np.maximum(0.0, one[:, 3] - one[:, 1])
    area_two = core.np.maximum(0.0, two[:, 2] - two[:, 0]) * core.np.maximum(0.0, two[:, 3] - two[:, 1])
    return inter / core.np.maximum(area_one[:, None] + area_two[None, :] - inter, 1e-12)


def feature_matrix(local, sources, full, tiles: list[dict], proxy_row: dict, meta: dict):
    data = local.detach().float().cpu().numpy()
    full_data = full.detach().float().cpu().numpy()
    count = len(data)
    if count == 0:
        return core.np.empty((0, len(FEATURE_NAMES)), dtype=core.np.float32), core.np.empty((0,), dtype=bool)
    boxes, conf, classes = data[:, :4], data[:, 4], data[:, 5].astype(core.np.int64)
    width = core.np.maximum(1e-3, boxes[:, 2] - boxes[:, 0])
    height = core.np.maximum(1e-3, boxes[:, 3] - boxes[:, 1])
    area = width * height
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    image_area = float(meta["width"] * meta["height"])
    resize_scale = core.IMGSZ / max(meta["width"], meta["height"])
    logit = core.np.log(core.np.clip(conf, 1e-5, 1 - 1e-5) / core.np.clip(1 - conf, 1e-5, 1))

    border_x = core.np.zeros(count, dtype=core.np.float32)
    border_y = core.np.zeros(count, dtype=core.np.float32)
    tile_proxy = core.np.zeros(count, dtype=core.np.float32)
    tile_rank = core.np.zeros(count, dtype=core.np.float32)
    tile_onehot = core.np.zeros((count, 4), dtype=core.np.float32)
    rank_map = {tile: rank for rank, tile in enumerate(proxy_row["ranked_tiles"])}
    for index, tile_id in enumerate(sources.tolist()):
        tile = tiles[tile_id]
        dx = min(boxes[index, 0] - tile["x0"], tile["x1"] - boxes[index, 2])
        dy = min(boxes[index, 1] - tile["y0"], tile["y1"] - boxes[index, 3])
        border_x[index] = core.np.clip(dx / max(1.0, tile["x1"] - tile["x0"]), -0.1, 0.5)
        border_y[index] = core.np.clip(dy / max(1.0, tile["y1"] - tile["y0"]), -0.1, 0.5)
        tile_proxy[index] = math.log1p(max(0.0, proxy_row["tile_scores"][tile_id]))
        tile_rank[index] = rank_map[tile_id] / 3.0
        tile_onehot[index, tile_id] = 1.0

    full_same_iou = core.np.zeros(count, dtype=core.np.float32)
    full_same_conf_iou = core.np.zeros(count, dtype=core.np.float32)
    full_same_max_conf = core.np.zeros(count, dtype=core.np.float32)
    full_any_iou = core.np.zeros(count, dtype=core.np.float32)
    full_any_conf_iou = core.np.zeros(count, dtype=core.np.float32)
    full_any_agree = core.np.zeros(count, dtype=core.np.float32)
    area_ratio_full = core.np.zeros(count, dtype=core.np.float32)
    if len(full_data):
        full_boxes = full_data[:, :4]
        full_iou = pairwise_iou(boxes, full_boxes)
        full_area = core.np.maximum(1e-3, full_boxes[:, 2] - full_boxes[:, 0]) * core.np.maximum(1e-3, full_boxes[:, 3] - full_boxes[:, 1])
        for index in range(count):
            any_index = int(core.np.argmax(full_iou[index]))
            full_any_iou[index] = full_iou[index, any_index]
            full_any_conf_iou[index] = full_data[any_index, 4]
            full_any_agree[index] = float(int(full_data[any_index, 5]) == classes[index])
            same = core.np.flatnonzero(full_data[:, 5].astype(core.np.int64) == classes[index])
            if len(same):
                same_index = int(same[int(core.np.argmax(full_iou[index, same]))])
                full_same_iou[index] = full_iou[index, same_index]
                full_same_conf_iou[index] = full_data[same_index, 4]
                full_same_max_conf[index] = float(core.np.max(full_data[same, 4]))
                area_ratio_full[index] = math.log(max(1e-6, area[index] / full_area[same_index]))

    peer_iou = core.np.zeros(count, dtype=core.np.float32)
    peer_conf = core.np.zeros(count, dtype=core.np.float32)
    peer_count = core.np.zeros(count, dtype=core.np.float32)
    if count > 1:
        local_iou = pairwise_iou(boxes, boxes)
        core.np.fill_diagonal(local_iou, -1.0)
        for index in range(count):
            same = core.np.flatnonzero(classes == classes[index])
            same = same[same != index]
            if len(same):
                peer = int(same[int(core.np.argmax(local_iou[index, same]))])
                peer_iou[index] = max(0.0, local_iou[index, peer])
                peer_conf[index] = conf[peer]
                peer_count[index] = min(10.0, float(core.np.sum(local_iou[index, same] >= 0.5))) / 10.0

    class_onehot = core.np.zeros((count, 10), dtype=core.np.float32)
    class_onehot[core.np.arange(count), classes] = 1.0
    matrix = core.np.column_stack([
        logit, conf, core.np.log(core.np.maximum(area / image_area, 1e-9)),
        core.np.sqrt(area) * resize_scale / 32.0,
        core.np.log(core.np.maximum(width / height, 1e-6)),
        cx / meta["width"], cy / meta["height"],
        core.np.minimum(border_x, border_y), border_x, border_y,
        tile_proxy, tile_rank, full_same_iou, full_same_conf_iou,
        full_same_max_conf, full_any_iou, full_any_conf_iou, full_any_agree,
        conf - full_same_conf_iou, area_ratio_full,
        peer_iou, peer_conf, peer_count, tile_onehot, class_onehot,
    ]).astype(core.np.float32)
    if matrix.shape[1] != len(FEATURE_NAMES) or not core.np.isfinite(matrix).all():
        raise RuntimeError(f"Invalid CVRC feature matrix: {matrix.shape}")
    return matrix, area < SMALL_AREA


def candidate_targets(local, annotations: dict):
    data = local.detach().float().cpu().numpy()
    targets = core.np.zeros(len(data), dtype=core.np.float32)
    positive_small = core.np.zeros(len(data), dtype=bool)
    if len(data) == 0:
        return targets, positive_small
    classes = data[:, 5].astype(core.np.int64) + 1
    for category in range(1, 11):
        detections = core.np.flatnonzero(classes == category)
        ground = annotations["by_class"].get(category, ())
        if not len(detections) or not ground:
            continue
        boxes = core.np.stack([row["box"] for row in ground])
        overlaps = pairwise_iou(data[detections, :4], boxes)
        pairs = []
        for di, gi in core.np.argwhere(overlaps >= TARGET_IOU):
            index = int(detections[int(di)])
            pairs.append((float(overlaps[di, gi]), float(data[index, 4]), index, int(gi)))
        pairs.sort(key=lambda row: (-row[0], -row[1], row[2], row[3]))
        used_detection, used_ground = set(), set()
        for overlap, _confidence, index, gi in pairs:
            if index in used_detection or gi in used_ground:
                continue
            used_detection.add(index)
            used_ground.add(gi)
            targets[index] = overlap
            positive_small[index] = ground[gi]["area"] < SMALL_AREA
    return targets, positive_small


def collect_candidates(model, images, full, selections, proxy_by_path, meta, annotations, report):
    records = []
    started = time.perf_counter()
    core.torch.cuda.reset_peak_memory_stats(0)
    for image_index, path in enumerate(images, 1):
        key = str(path)
        image = core.cv2.imread(key, core.cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV failed to read {path}")
        tiles = core.make_tiles(image)
        chosen = selections[key]
        tensors, tile_ids = [], []
        if chosen:
            results = core.model_predict(model, [tiles[index]["image"] for index in chosen], len(chosen))
            for result, tile_id in zip(results, chosen):
                mapped = core.map_tile_detections(result, tiles[tile_id])
                if len(mapped):
                    tensors.append(mapped)
                    tile_ids.append(tile_id)
            del results
        local, sources = nms_with_source(tensors, tile_ids)
        features, candidate_small = feature_matrix(
            local, sources, full[key], tiles, proxy_by_path[key], meta[key]
        )
        targets, positive_small = candidate_targets(local, annotations[key])
        records.append({
            "path": key, "local": local, "features": features, "targets": targets,
            "candidate_small": candidate_small, "positive_small": positive_small,
            "classes": local[:, 5].numpy().astype(core.np.int64) if len(local) else core.np.empty((0,), dtype=core.np.int64),
        })
        del image, tiles, tensors
        if image_index == 1 or image_index % 50 == 0 or image_index == len(images):
            print(f"D16a local candidate extraction: {image_index}/{len(images)}", flush=True)
            report["progress"] = {"stage": "candidate_extraction", "images": image_index}
            save(report)
    return records, {
        "seconds": time.perf_counter() - started,
        "peak_memory": core.memory_peak(),
        "candidates": sum(len(row["targets"]) for row in records),
        "positives": sum(int(core.np.sum(row["targets"] > 0)) for row in records),
    }


def sequence_group(path: str, smoke: bool) -> str:
    stem = Path(path).stem
    if smoke:
        return stem
    return stem.split("_")[0] if "_" in stem else stem


def balanced_group_folds(records: list[dict], folds: int, smoke: bool):
    group_sizes = {}
    for row in records:
        group = sequence_group(row["path"], smoke)
        group_sizes[group] = group_sizes.get(group, 0) + 1
    groups = sorted(group_sizes)
    if len(groups) < 2:
        raise RuntimeError("At least two independent groups are required for OOF calibration")
    folds = min(folds, len(groups))
    ordered = sorted(
        groups,
        key=lambda value: (
            -group_sizes[value],
            hashlib.sha256(("outer:" + value).encode()).hexdigest(),
        ),
    )
    loads = [0] * folds
    group_counts = [0] * folds
    mapping = {}
    for group in ordered:
        fold = min(range(folds), key=lambda index: (loads[index], group_counts[index], index))
        mapping[group] = fold
        loads[fold] += group_sizes[group]
        group_counts[fold] += 1
    return mapping, folds


def sample_record(row: dict):
    target = row["targets"]
    positive = core.np.flatnonzero(target > 0)
    negative = core.np.flatnonzero(target == 0)
    wanted = min(len(negative), max(MIN_NEGATIVES_PER_IMAGE, NEGATIVE_RATIO * max(1, len(positive))))
    if wanted < len(negative):
        hard_count = wanted // 2
        conf = row["local"][:, 4].numpy()
        hard = negative[core.np.argsort(-conf[negative], kind="stable")[:hard_count]]
        remaining = core.np.setdiff1d(negative, hard, assume_unique=False)
        seed = int(hashlib.sha256(row["path"].encode()).hexdigest()[:8], 16)
        rng = core.np.random.default_rng(seed)
        random = rng.choice(remaining, size=wanted - hard_count, replace=False)
        negative = core.np.concatenate([hard, random])
    return core.np.concatenate([positive, negative]).astype(core.np.int64)


def concatenate_records(records: list[dict], sampled: bool):
    xs, ys, small, classes, base = [], [], [], [], []
    for row in records:
        indices = sample_record(row) if sampled else core.np.arange(len(row["targets"]))
        if not len(indices):
            continue
        xs.append(row["features"][indices])
        ys.append(row["targets"][indices])
        small.append(row["candidate_small"][indices])
        classes.append(row["classes"][indices])
        base.append(row["local"][:, 4].numpy()[indices])
    if not xs:
        raise RuntimeError("No CVRC candidates were collected")
    return tuple(core.np.concatenate(values, axis=0) for values in (xs, ys, small, classes, base))


def average_precision(labels, scores) -> float:
    labels = core.np.asarray(labels, dtype=bool)
    if not core.np.any(labels):
        return 0.0
    order = core.np.argsort(-core.np.asarray(scores), kind="stable")
    ranked = labels[order]
    precision = core.np.cumsum(ranked) / core.np.arange(1, len(ranked) + 1)
    return float(core.np.sum(precision * ranked) / core.np.sum(ranked))


def candidate_metrics(labels, scores, small, classes) -> dict:
    positive = labels > 0
    overall = average_precision(positive, scores)
    small_mask = core.np.asarray(small, dtype=bool)
    small_ap = average_precision(positive[small_mask], scores[small_mask]) if core.np.any(small_mask) else overall
    class_aps = []
    for category in range(10):
        mask = classes == category
        if core.np.any(positive[mask]):
            class_aps.append(average_precision(positive[mask], scores[mask]))
    macro = float(core.np.mean(class_aps)) if class_aps else overall
    return {
        "AP_candidate": overall, "AP_candidate_small": small_ap,
        "AP_candidate_class_macro": macro,
        "objective": 0.30 * overall + 0.60 * small_ap + 0.10 * macro,
        "candidates": int(len(labels)), "positives": int(core.np.sum(positive)),
    }


def blend_scores(base, corrected_logits, alpha: float):
    base = core.np.clip(base, 1e-5, 1 - 1e-5)
    base_logits = core.np.log(base / (1 - base))
    logits = (1.0 - alpha) * base_logits + alpha * corrected_logits
    return 1.0 / (1.0 + core.np.exp(-core.np.clip(logits, -20, 20)))


def make_model(input_dim: int):
    nn = core.torch.nn

    class ResidualCalibrator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64), nn.SiLU(), nn.Dropout(0.10),
                nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, features, base_logits):
            return base_logits + self.net(features).squeeze(1)

    return ResidualCalibrator()


def predict_logits(model, features, mean, std):
    values = core.np.clip((features - mean) / std, -8.0, 8.0).astype(core.np.float32)
    result = []
    model.eval()
    with core.torch.no_grad():
        for start in range(0, len(values), BATCH_SIZE):
            x = core.torch.from_numpy(values[start:start + BATCH_SIZE]).cuda()
            base = x[:, 0] * float(std[0]) + float(mean[0])
            result.append(model(x, base).float().cpu().numpy())
    return core.np.concatenate(result) if result else core.np.empty((0,), dtype=core.np.float32)


def train_fold(fold: int, fit_rows, inner_rows, outer_rows, out: Path) -> tuple[dict, dict[str, object]]:
    x_fit, y_fit, small_fit, cls_fit, base_fit = concatenate_records(fit_rows, sampled=True)
    x_inner, y_inner, small_inner, cls_inner, base_inner = concatenate_records(inner_rows, sampled=False)
    mean = x_fit.mean(axis=0).astype(core.np.float32)
    std = x_fit.std(axis=0).astype(core.np.float32)
    std[std < 1e-5] = 1.0
    x_fit = core.np.clip((x_fit - mean) / std, -8.0, 8.0).astype(core.np.float32)
    core.torch.manual_seed(20260912 + fold)
    core.torch.cuda.manual_seed_all(20260912 + fold)
    model = make_model(len(FEATURE_NAMES)).cuda()
    optimizer = core.torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    positives = max(1, int(core.np.sum(y_fit > 0)))
    negatives = max(1, len(y_fit) - positives)
    positive_weight = min(8.0, negatives / positives)
    best = None
    generator = core.torch.Generator(device="cpu")
    generator.manual_seed(20260912 + fold)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        permutation = core.torch.randperm(len(x_fit), generator=generator)
        losses = []
        for start in range(0, len(permutation), BATCH_SIZE):
            indices = permutation[start:start + BATCH_SIZE].numpy()
            x = core.torch.from_numpy(x_fit[indices]).cuda()
            target = core.torch.from_numpy(y_fit[indices]).cuda()
            base_logits = x[:, 0] * float(std[0]) + float(mean[0])
            logits = model(x, base_logits)
            weight = core.torch.ones_like(target)
            positive_mask = target > 0
            weight[positive_mask] = positive_weight
            small_positive = core.torch.from_numpy(small_fit[indices]).cuda() & positive_mask
            weight[small_positive] *= 1.5
            loss = core.torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, weight=weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            core.torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        corrected = predict_logits(model, x_inner, mean, std)
        trials = []
        for alpha in ALPHAS:
            metrics = candidate_metrics(
                y_inner, blend_scores(base_inner, corrected, alpha),
                small_inner, cls_inner,
            )
            trials.append({"alpha": alpha, **metrics})
        chosen = max(trials, key=lambda row: (row["objective"], -row["alpha"]))
        row = {"epoch": epoch, "loss": float(core.np.mean(losses)), "trials": trials, "best_alpha": chosen["alpha"], "objective": chosen["objective"]}
        if best is None or row["objective"] > best["row"]["objective"]:
            best = {"row": row, "state": copy.deepcopy(model.state_dict())}
        print(f"D16a fold {fold + 1}: epoch {epoch}/{EPOCHS}, loss={row['loss']:.5f}, inner_objective={row['objective']:.5f}, alpha={row['best_alpha']}", flush=True)
    model.load_state_dict(best["state"])
    alpha = float(best["row"]["best_alpha"])
    score_map = {}
    outer_candidate_rows = []
    for row in outer_rows:
        corrected = predict_logits(model, row["features"], mean, std)
        base = row["local"][:, 4].numpy()
        score_map[row["path"]] = blend_scores(base, corrected, alpha).astype(core.np.float32)
        metrics = candidate_metrics(row["targets"], score_map[row["path"]], row["candidate_small"], row["classes"])
        outer_candidate_rows.append(metrics)
    artifact = {
        "script_revision": SCRIPT_REVISION,
        "feature_names": FEATURE_NAMES,
        "mean": mean, "std": std, "alpha": alpha,
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    core.torch.save(artifact, out / f"cvrc_fold{fold + 1}.pt")
    fold_report = {
        "fold": fold, "fit_images": len(fit_rows), "inner_images": len(inner_rows),
        "outer_images": len(outer_rows), "fit_candidates_sampled": len(y_fit),
        "fit_positives": positives, "positive_weight": positive_weight,
        "selected_epoch": best["row"]["epoch"], "selected_alpha": alpha,
        "inner_selection": best["row"],
        "outer_candidate_objective_mean": float(core.np.mean([row["objective"] for row in outer_candidate_rows])),
    }
    del model, optimizer, x_fit, y_fit
    core.clear_gpu()
    return fold_report, score_map


def evaluate_predictions(name, predictions, images, path_ids, gt, names):
    print(f"D16a evaluating {name} on CPU.", flush=True)
    detections = core.predictions_to_coco(predictions, images, path_ids)
    metrics = core.coco_metrics(gt, detections, names)
    metrics["fixed_operating_point"] = core.fixed_operating_point(gt, detections, names)
    metrics["prediction_count"] = len(detections)
    return metrics


def compose_predictions(images, full, records_by_path, score_map=None):
    predictions = {}
    for path in images:
        key = str(path)
        local = records_by_path[key]["local"].clone()
        if score_map is not None and len(local):
            scores = score_map[key]
            if len(scores) != len(local) or not core.np.isfinite(scores).all():
                raise RuntimeError(f"Invalid OOF scores for {key}")
            local[:, 4] = core.torch.from_numpy(scores)
        predictions[key] = core.nms_detections(core.torch.cat([full[key], local], dim=0))
    return predictions


def delta(one: dict, zero: dict) -> dict:
    keys = ("mAP50_95", "mAP50", "mAP75", "AR_all", "AP_small", "AR_small", "AP_medium", "AR_medium", "AP_large", "AR_large")
    return {key: one[key] - zero[key] for key in keys}


def save_summary_csv(report: dict) -> None:
    path = Path(report["report_dir"]) / "summary.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "mAP50_95", "mAP50", "mAP75", "AP_small", "AR_small", "AP_medium", "AP_large"])
        for name, row in report.get("modes", {}).items():
            metrics = row["overall"]
            writer.writerow([name] + [metrics[key] for key in ("mAP50_95", "mAP50", "mAP75", "AP_small", "AR_small", "AP_medium", "AP_large")])


def preflight(report: dict, check_only: bool):
    if core.SCRIPT_REVISION != EXPECTED_CORE_REVISION:
        raise RuntimeError(f"Unexpected slicing core revision: {core.SCRIPT_REVISION}")
    core.preflight(report, ["e1"], check_only)
    path, reference = latest_d15b()
    report["d15b_reference"] = {
        "path": str(path), "revision": reference.get("script_revision"),
        "standard": reference["models"]["e1"]["modes"]["proxy_b1p50_standard"]["overall"],
        "fusion_oracle": reference["models"]["e1"]["modes"]["proxy_b1p50_oracle_validity"]["overall"],
    }
    model = core.YOLO(str(WEIGHT))
    expected_names = {index: name for index, name in enumerate(core.EXPECTED_CLASSES)}
    structure = core.assert_model(model, expected_names, "e1")
    before = core.sha256(WEIGHT)
    calibrator = make_model(len(FEATURE_NAMES)).cuda()
    x = core.torch.randn(32, len(FEATURE_NAMES), device="cuda")
    target = core.torch.rand(32, device="cuda")
    logits = calibrator(x, x[:, 0])
    loss = core.torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    if not all(parameter.grad is not None and core.torch.isfinite(parameter.grad).all() for parameter in calibrator.parameters()):
        raise RuntimeError("Synthetic CVRC gradient check failed")
    if core.sha256(WEIGHT) != before:
        raise RuntimeError("E1 checkpoint changed during preflight")
    report["preflight"].update({
        "d15b_found": True, "e1_sha256": before, "e1_structure": structure,
        "feature_dimension": len(FEATURE_NAMES), "synthetic_gradient_check": True,
    })
    del model, calibrator, x, target, logits, loss
    core.clear_gpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke2", action="store_true", help="Run a small two-fold pipeline test; metrics cannot drive decisions.")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = "CHECK_D16A" if args.check_only else "SMOKE2_D16A" if args.smoke2 else "DIAG_D16A"
    out = REPORTS / f"{prefix}_e1_cvrc_oof_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "initializing", "purpose": "grouped nested OOF cross-view local-candidate calibration",
        "script_revision": SCRIPT_REVISION, "started_at": now(), "report_dir": str(out),
        "is_full_diagnostic": not args.check_only and not args.smoke2,
        "protocol": {
            "detector": "frozen E1 YOLO11s-P2", "router": "frozen D15a proxy budget 1.5",
            "target": "one-to-one same-class local candidate IoU>=0.50; soft target is IoU",
            "outer_split": "sequence-grouped five-fold OOF",
            "inner_selection": "group-held-out epoch and alpha selection inside each outer fold",
            "features": FEATURE_NAMES, "epochs": EPOCHS, "alphas": ALPHAS,
            "validation_image_label_leakage": False,
            "training_performed": "calibrator_only", "detector_training": False,
            "checkpoint_updates": False, "ultralytics_modified": False,
            "automatic_shutdown": False,
        },
        "modes": {}, "folds": [],
    }
    save(report)
    started = time.perf_counter()
    model = None
    try:
        preflight(report, args.check_only)
        report["status"] = "preflight_passed"
        save(report)
        if args.check_only:
            report.update(status="check_passed", finished_at=now(), total_seconds=time.perf_counter() - started)
            save(report)
            print("D16a preflight/model/gradient/reference checks passed. No prediction or shutdown.", flush=True)
            return
        limit = 12 if args.smoke2 else 0
        images, path_ids, gt, names = core.prepare_dataset(report, limit)
        annotations = annotation_index(gt)
        meta = image_meta(gt)
        model = core.YOLO(str(WEIGHT))
        before_hash = core.sha256(WEIGHT)
        core.assert_model(model, names, "e1")
        full, full_profile = core.predict_full(model, images, report, "D16a E1")
        proxy_rows = build_proxy_rows(images, full, meta)
        proxy_by_path = {row["image"]: row for row in proxy_rows}
        selections, budget_meta = allocate_budget(proxy_rows, PRIMARY_BUDGET)
        report["router"] = budget_meta
        records, candidate_profile = collect_candidates(
            model, images, full, selections, proxy_by_path, meta, annotations, report
        )
        report["candidate_profile"] = candidate_profile
        records_by_path = {row["path"]: row for row in records}
        group_map, fold_count = balanced_group_folds(records, 2 if args.smoke2 else OUTER_FOLDS, args.smoke2)
        report["grouping"] = {
            "fold_count": fold_count, "groups": len(group_map),
            "group_to_fold": group_map,
            "smoke_uses_each_image_as_group": bool(args.smoke2),
        }
        score_map = {}
        for fold in range(fold_count):
            outer = [
                row for row in records
                if group_map[sequence_group(row["path"], args.smoke2)] == fold
            ]
            train_groups = sorted({sequence_group(row["path"], args.smoke2) for row in records if group_map[sequence_group(row["path"], args.smoke2)] != fold})
            ordered_inner = sorted(train_groups, key=lambda value: hashlib.sha256((f"inner:{fold}:" + value).encode()).hexdigest())
            inner_count = max(1, int(round(INNER_FRACTION * len(train_groups))))
            inner_groups = set(ordered_inner[:inner_count])
            inner = [row for row in records if sequence_group(row["path"], args.smoke2) in inner_groups]
            fit = [row for row in records if group_map[sequence_group(row["path"], args.smoke2)] != fold and sequence_group(row["path"], args.smoke2) not in inner_groups]
            if not fit or not inner or not outer:
                raise RuntimeError(f"Invalid nested split for fold {fold}: fit={len(fit)}, inner={len(inner)}, outer={len(outer)}")
            fold_report, fold_scores = train_fold(fold, fit, inner, outer, out)
            report["folds"].append(fold_report)
            overlap = set(score_map).intersection(fold_scores)
            if overlap:
                raise RuntimeError(f"OOF image scored more than once: {sorted(overlap)[:3]}")
            score_map.update(fold_scores)
            report["progress"] = {"stage": "oof_training", "fold": fold + 1, "folds": fold_count}
            save(report)
        if set(score_map) != {str(path) for path in images}:
            raise RuntimeError("OOF scoring did not cover every image exactly once")
        standard_predictions = compose_predictions(images, full, records_by_path)
        calibrated_predictions = compose_predictions(images, full, records_by_path, score_map)
        report["modes"]["d15a_proxy_b1p50_standard"] = evaluate_predictions(
            "d15a_proxy_b1p50_standard", standard_predictions, images, path_ids, gt, names
        )
        report["modes"]["d16a_cvrc_grouped_oof"] = evaluate_predictions(
            "d16a_cvrc_grouped_oof", calibrated_predictions, images, path_ids, gt, names
        )
        report["modes"]["d15a_proxy_b1p50_standard"]["profile"] = {
            **full_profile, "local_crops_per_image": PRIMARY_BUDGET,
            "forward_image_multiplier": 1.0 + PRIMARY_BUDGET,
        }
        report["modes"]["d16a_cvrc_grouped_oof"]["profile"] = {
            **full_profile, "local_crops_per_image": PRIMARY_BUDGET,
            "forward_image_multiplier": 1.0 + PRIMARY_BUDGET,
            "calibrator_parameters_per_fold": sum(parameter.numel() for parameter in make_model(len(FEATURE_NAMES)).parameters()),
        }
        standard = report["modes"]["d15a_proxy_b1p50_standard"]["overall"]
        calibrated = report["modes"]["d16a_cvrc_grouped_oof"]["overall"]
        gain = delta(calibrated, standard)
        if args.smoke2:
            report["decision"] = {"disabled": "Smoke metrics cannot drive a research decision."}
        else:
            reference = report["d15b_reference"]["standard"]
            parity = {key: standard[key] - reference[key] for key in ("mAP50_95", "AP_small", "AR_small", "AP_medium", "AP_large")}
            parity_pass = all(abs(value) <= REFERENCE_TOLERANCE for value in parity.values())
            passed = bool(parity_pass and gain["mAP50_95"] >= MAP_GATE and gain["AP_small"] >= APS_GATE and gain["AP_medium"] >= APM_FLOOR)
            report["decision"] = {
                "d15a_reproduction": {"delta": parity, "absolute_tolerance": REFERENCE_TOLERANCE, "passed": parity_pass},
                "cvrc_minus_d15a": gain,
                "gate": {"mAP50_95": MAP_GATE, "AP_small": APS_GATE, "AP_medium_minimum": APM_FLOOR},
                "passed": passed,
                "selected_branch": "train_deployable_cvrc_on_train_then_test" if passed else "stop_cvrc_and_keep_d15a",
                "claim_boundary": "OOF prevents same-sequence label leakage, but final publication claims require an untouched test set.",
            }
        after_hash = core.sha256(WEIGHT)
        if after_hash != before_hash:
            raise RuntimeError("Frozen E1 checkpoint changed during D16a")
        report["checkpoint_unchanged"] = True
        report.update(status="completed", finished_at=now(), total_seconds=time.perf_counter() - started)
        save_summary_csv(report)
        shutil.copy2(Path(__file__), out / Path(__file__).name)
        save(report)
        print(f"D16a completed. Report: {out / 'metrics.txt'}", flush=True)
    except BaseException:
        report.update(status="failed_or_interrupted", finished_at=now(), total_seconds=time.perf_counter() - started, error=traceback.format_exc())
        save(report)
        core.atomic_text(out / "error.log", report["error"])
        print(f"D16a failure saved: {out}", file=sys.stderr, flush=True)
        raise
    finally:
        if model is not None:
            del model
        core.clear_gpu()
        try:
            core.atomic_json(out / "shutdown_status.json", {
                "time": now(), "enabled": False,
                "status": "disabled_by_user_policy", "reason": report.get("status"),
            })
            save(report)
        except Exception:
            pass
        print("Automatic shutdown is disabled; the instance remains running.", flush=True)


if __name__ == "__main__":
    main()
