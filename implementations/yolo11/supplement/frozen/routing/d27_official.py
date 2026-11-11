#!/usr/bin/env python3
"""D27: zero-training audit using the official VisDrone DET evaluation semantics.

This script deliberately reuses saved ``area_predictions.json`` files.  It does
not load a model, train, modify a checkpoint, or shut down the machine.

The primary metric is a clean-room Python port of the public VisDrone2019 DET
Matlab toolkit v1.0.4:

* detections are globally limited to maxDets before class filtering;
* class AP is integrated over IoU 0.50:0.05:0.95;
* detections covered >= 50% by class-0 ignore regions are removed;
* score-zero GT boxes use the toolkit's ignore matching rule;
* AP is aggregated with the same class-availability weighting as calcAccuracy.m.

The script also reports how many pre-filter false positives overlap class-0
ignore regions or class-11 ``others`` boxes.  The latter is diagnostic only:
the public Matlab v1.0.4 source raster-filters class-0 regions explicitly, while
class 11 is excluded from the ten evaluated categories.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


REVISION = "d27_official_visdrone_det_matlab104_port_v1"
DEFAULT_PROJECT = Path("/root/autodl-tmp/yolo11_project")
DEFAULT_DATASET = Path("/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val")
EXPECTED_IMAGES = 548
CLASS_NAMES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}
IOU_THRESHOLDS = np.arange(0.50, 0.951, 0.05, dtype=np.float64)
MAX_DETS = (1, 10, 100, 500)
CONF_AUDIT = (0.05, 0.10, 0.25)

MODEL_SPECS = {
    "e0": "baseline_yolo11s_img1024_seed1_*",
    "e1": "e1_yolo11s_p2add_img1024_seed1_*",
    "e10a": "e10a_yolo11s_p2_dssg_img1024_seed1_*",
    "e16a": "e16a_yolo11s_p2_tsdr_img1024_seed1_*",
    "e21a": "e21a_yolo11s_p2_rsg_dssg_img1024_seed1_*",
    "e26a": "e26a_yolo11s_p2_terp_img1024_seed1_*",
}


@dataclass
class ImageRecord:
    image_id: int
    stem: str
    image_path: Path
    annotation_path: Path
    width: int
    height: int
    raw_gt: np.ndarray  # [x,y,w,h,score,class,truncation,occlusion]
    eval_gt: np.ndarray  # [x,y,w,h,ignore,class]
    ignore_rects: np.ndarray
    others_rects: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def matlab_round_positive(value: float) -> int:
    """Match Matlab round() for the non-negative coordinates used here."""
    return int(math.floor(float(value) + 0.5))


def union_area_inclusive(rectangles: list[tuple[int, int, int, int]]) -> int:
    """Union area of inclusive integer rectangles (x1, y1, x2, y2)."""
    rectangles = [r for r in rectangles if r[2] >= r[0] and r[3] >= r[1]]
    if not rectangles:
        return 0
    xs = sorted({r[0] for r in rectangles} | {r[2] + 1 for r in rectangles})
    total = 0
    for xa, xb in zip(xs[:-1], xs[1:]):
        intervals = [(y1, y2 + 1) for x1, y1, x2, y2 in rectangles if x1 < xb and x2 + 1 > xa]
        if not intervals:
            continue
        intervals.sort()
        covered = 0
        start, end = intervals[0]
        for nxt_start, nxt_end in intervals[1:]:
            if nxt_start > end:
                covered += end - start
                start, end = nxt_start, nxt_end
            else:
                end = max(end, nxt_end)
        covered += end - start
        total += (xb - xa) * covered
    return int(total)


def official_region_coverage(
    box: Iterable[float], regions: np.ndarray, width: int, height: int
) -> float:
    """Port the integral-image coverage test in dropObjectsInIgr.m without a mask.

    The public Matlab code rasterizes an ignore rectangle with inclusive colon
    ranges, but queries its integral image as ``I(y,x)+I(y+h,x+w)-...``.  That
    query covers rows ``y+1:y+h`` and columns ``x+1:x+w``: exactly ``h*w``
    samples before image-boundary clipping.  Keeping this slightly unusual
    convention is necessary for protocol compatibility.
    """
    if regions.size == 0:
        return 0.0
    x, y, w, h = (matlab_round_positive(v) for v in box)
    x = max(1, min(width, x))
    y = max(1, min(height, y))
    w = max(1, w)
    h = max(1, h)
    query_x1 = x + 1
    query_y1 = y + 1
    query_x2 = min(width, x + w)
    query_y2 = min(height, y + h)
    if query_x2 < query_x1 or query_y2 < query_y1:
        return 0.0
    intersections: list[tuple[int, int, int, int]] = []
    for rx, ry, rw, rh in regions[:, :4]:
        # Original VisDrone GT is integer-valued.  The Matlab toolkit applies
        # max(1, box) and uses inclusive colon ranges x:x+w and y:y+h.
        ix1 = max(1, matlab_round_positive(rx))
        iy1 = max(1, matlab_round_positive(ry))
        ix2 = min(width, ix1 + max(1, matlab_round_positive(rw)))
        iy2 = min(height, iy1 + max(1, matlab_round_positive(rh)))
        ax1, ay1 = max(query_x1, ix1), max(query_y1, iy1)
        ax2, ay2 = min(query_x2, ix2), min(query_y2, iy2)
        if ax2 >= ax1 and ay2 >= ay1:
            intersections.append((ax1, ay1, ax2, ay2))
    return union_area_inclusive(intersections) / float(w * h)


def parse_annotation(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        if fields and fields[-1] == "":
            fields.pop()
        if len(fields) < 8:
            raise ValueError(f"{path}:{line_no}: expected at least 8 comma-separated fields")
        row = [float(item) for item in fields[:8]]
        if not all(math.isfinite(item) for item in row):
            raise ValueError(f"{path}:{line_no}: non-finite annotation")
        if row[2] <= 0 or row[3] <= 0 or int(row[5]) not in range(12):
            raise ValueError(f"{path}:{line_no}: invalid box or class: {row}")
        rows.append(row)
    return np.asarray(rows, dtype=np.float64).reshape(-1, 8)


def drop_gt_in_class0_regions(
    raw_gt: np.ndarray, ignore_rects: np.ndarray, width: int, height: int
) -> np.ndarray:
    kept: list[list[float]] = []
    for row in raw_gt:
        cls = int(row[5])
        if cls == 0:
            continue
        if official_region_coverage(row[:4], ignore_rects, width, height) >= 0.5:
            continue
        ignore_flag = 1.0 if row[4] == 0 else 0.0
        kept.append([row[0], row[1], row[2], row[3], ignore_flag, float(cls)])
    return np.asarray(kept, dtype=np.float64).reshape(-1, 6)


def load_dataset(dataset_root: Path) -> tuple[list[ImageRecord], dict[str, object]]:
    annotation_dir = dataset_root / "annotations"
    image_dir = dataset_root / "images"
    if not annotation_dir.is_dir() or not image_dir.is_dir():
        raise FileNotFoundError(f"Expected images/ and annotations/ under {dataset_root}")
    annotation_paths = sorted(annotation_dir.glob("*.txt"))
    if len(annotation_paths) != EXPECTED_IMAGES:
        raise ValueError(f"Expected {EXPECTED_IMAGES} validation annotations, found {len(annotation_paths)}")

    digest = hashlib.sha256()
    records: list[ImageRecord] = []
    class_counts = {str(key): 0 for key in range(12)}
    score_zero_counts = {str(key): 0 for key in range(12)}
    total_dropped_by_region = 0
    for image_id, annotation_path in enumerate(annotation_paths, 1):
        stem = annotation_path.stem
        candidates = [image_dir / f"{stem}{suffix}" for suffix in (".jpg", ".JPG", ".jpeg", ".png")]
        image_path = next((item for item in candidates if item.is_file()), None)
        if image_path is None:
            raise FileNotFoundError(f"Image for {annotation_path} not found")
        with Image.open(image_path) as image:
            width, height = image.size
        raw_gt = parse_annotation(annotation_path)
        for row in raw_gt:
            cls = int(row[5])
            class_counts[str(cls)] += 1
            if row[4] == 0:
                score_zero_counts[str(cls)] += 1
        ignore_rects = raw_gt[raw_gt[:, 5] == 0, :4].copy()
        others_rects = raw_gt[raw_gt[:, 5] == 11, :4].copy()
        eval_gt = drop_gt_in_class0_regions(raw_gt, ignore_rects, width, height)
        total_dropped_by_region += int(np.count_nonzero(raw_gt[:, 5] != 0) - len(eval_gt))
        content = annotation_path.read_bytes()
        digest.update(annotation_path.name.encode("utf-8") + b"\0" + content + b"\0")
        digest.update(f"{width}x{height}".encode("ascii") + b"\0")
        records.append(
            ImageRecord(
                image_id=image_id,
                stem=stem,
                image_path=image_path,
                annotation_path=annotation_path,
                width=width,
                height=height,
                raw_gt=raw_gt,
                eval_gt=eval_gt,
                ignore_rects=ignore_rects,
                others_rects=others_rects,
            )
        )
    summary = {
        "dataset_root": str(dataset_root),
        "images": len(records),
        "annotation_dimension_sha256": digest.hexdigest(),
        "raw_class_counts": class_counts,
        "score_zero_counts": score_zero_counts,
        "class0_ignore_regions": class_counts["0"],
        "class11_others": class_counts["11"],
        "gt_rows_dropped_by_class0_region": total_dropped_by_region,
        "evaluated_gt_rows": int(sum(np.count_nonzero((r.eval_gt[:, 5] >= 1) & (r.eval_gt[:, 5] <= 10)) for r in records)),
    }
    return records, summary


def find_latest_report(project: Path, pattern: str) -> Path:
    report_root = project / "comparison_reports"
    candidates = [item for item in report_root.glob(pattern) if item.is_dir() and (item / "area_predictions.json").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No report with area_predictions.json matched {report_root / pattern}")
    return max(candidates, key=lambda item: (item.stat().st_mtime_ns, item.name))


def resolve_models(project: Path, names: list[str]) -> dict[str, Path]:
    unknown = sorted(set(names) - set(MODEL_SPECS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}; valid={sorted(MODEL_SPECS)}")
    return {name: find_latest_report(project, MODEL_SPECS[name]) for name in names}


def load_predictions(path: Path, image_count: int) -> list[np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list: {path}")
    grouped: list[list[list[float]]] = [[] for _ in range(image_count)]
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Prediction {index} is not an object")
        image_id = int(item["image_id"])
        category_id = int(item["category_id"])
        bbox = [float(value) for value in item["bbox"]]
        score = float(item["score"])
        if image_id < 1 or image_id > image_count or category_id not in CLASS_NAMES:
            raise ValueError(f"Prediction {index}: invalid image/category id")
        # A clipped prediction can have a zero width/height.  The public
        # Matlab loader does not reject it: dropObjectsInIgr uses max(1,round)
        # only for the ignore query, while compOas later leaves it unmatched.
        # Preserve that behavior instead of silently deleting a false positive.
        if len(bbox) != 4 or bbox[2] < 0 or bbox[3] < 0 or not 0 <= score <= 1:
            raise ValueError(f"Prediction {index}: invalid bbox/score")
        if not all(math.isfinite(value) for value in bbox + [score]):
            raise ValueError(f"Prediction {index}: non-finite value")
        grouped[image_id - 1].append(bbox + [score, float(category_id), float(index)])
    result: list[np.ndarray] = []
    for rows in grouped:
        array = np.asarray(rows, dtype=np.float64).reshape(-1, 7)
        if len(array):
            order = np.argsort(-array[:, 4], kind="mergesort")
            array = array[order]
        result.append(array)
    return result


def overlap_matrix(dt_boxes: np.ndarray, gt_boxes: np.ndarray, gt_ignore: np.ndarray) -> np.ndarray:
    if len(dt_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(dt_boxes), len(gt_boxes)), dtype=np.float64)
    dt_xy2 = dt_boxes[:, :2] + dt_boxes[:, 2:4]
    gt_xy2 = gt_boxes[:, :2] + gt_boxes[:, 2:4]
    inter_w = np.maximum(
        0.0,
        np.minimum(dt_xy2[:, None, 0], gt_xy2[None, :, 0])
        - np.maximum(dt_boxes[:, None, 0], gt_boxes[None, :, 0]),
    )
    inter_h = np.maximum(
        0.0,
        np.minimum(dt_xy2[:, None, 1], gt_xy2[None, :, 1])
        - np.maximum(dt_boxes[:, None, 1], gt_boxes[None, :, 1]),
    )
    inter = inter_w * inter_h
    dt_area = dt_boxes[:, 2] * dt_boxes[:, 3]
    gt_area = gt_boxes[:, 2] * gt_boxes[:, 3]
    union = dt_area[:, None] + gt_area[None, :] - inter
    denominator = np.where(gt_ignore[None, :], dt_area[:, None], union)
    return np.divide(inter, denominator, out=np.zeros_like(inter), where=denominator > 0)


def prepare_match(gt0: np.ndarray, dt0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort as evalRes.m and precompute overlap; returns source orders too."""
    gt0 = np.asarray(gt0, dtype=np.float64).reshape(-1, 5)
    dt0 = np.asarray(dt0, dtype=np.float64).reshape(-1, 5)
    dt_order = np.argsort(-dt0[:, 4], kind="mergesort") if len(dt0) else np.zeros(0, dtype=np.int64)
    gt_order = np.argsort(gt0[:, 4], kind="mergesort") if len(gt0) else np.zeros(0, dtype=np.int64)
    dt = dt0[dt_order]
    gt = gt0[gt_order]
    gt_state = -gt[:, 4].astype(np.int8)
    oa = overlap_matrix(dt[:, :4], gt[:, :4], gt_state == -1)
    return gt, dt, gt_state, oa, dt_order


def match_prepared(gt_state_initial: np.ndarray, oa: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    gt_state = gt_state_initial.copy()
    dt_state = np.zeros(oa.shape[0], dtype=np.int8)
    for det_index in range(oa.shape[0]):
        best_overlap = float(threshold)
        best_gt = -1
        best_match = 0
        for gt_index in range(oa.shape[1]):
            match = int(gt_state[gt_index])
            if match == 1:
                continue
            if best_match != 0 and match == -1:
                break
            value = float(oa[det_index, gt_index])
            if value < best_overlap:
                continue
            best_overlap = value
            best_gt = gt_index
            best_match = 1 if match == 0 else -1
        if best_match == -1:
            dt_state[det_index] = -1
        elif best_match == 1:
            gt_state[best_gt] = 1
            dt_state[det_index] = 1
    return gt_state, dt_state


def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    changed = np.nonzero(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[changed] - mrec[changed - 1]) * mpre[changed]))


def official_evaluate(records: list[ImageRecord], detections: list[np.ndarray]) -> dict[str, object]:
    ap = np.zeros((10, len(IOU_THRESHOLDS)), dtype=np.float64)
    ar = np.zeros((10, len(IOU_THRESHOLDS), len(MAX_DETS)), dtype=np.float64)
    availability = np.zeros(10, dtype=np.int64)
    for cls in range(1, 11):
        availability[cls - 1] = sum(int(np.any(record.eval_gt[:, 5] == cls)) for record in records)
        print(f"Official evaluation class {cls}/10 ({CLASS_NAMES[cls]})", flush=True)
        for max_index, max_det in enumerate(MAX_DETS):
            gt_matches: list[list[np.ndarray]] = [[] for _ in IOU_THRESHOLDS]
            det_scores: list[list[np.ndarray]] = [[] for _ in IOU_THRESHOLDS]
            det_matches: list[list[np.ndarray]] = [[] for _ in IOU_THRESHOLDS]
            for record, image_detections in zip(records, detections):
                gt_class = record.eval_gt[record.eval_gt[:, 5] == cls]
                gt0 = gt_class[:, :5]
                globally_limited = image_detections[: min(len(image_detections), max_det)]
                dt_class = globally_limited[globally_limited[:, 5] == cls]
                gt, dt, gt_state, oa, _ = prepare_match(gt0, dt_class[:, :5])
                del gt
                for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
                    gt_match, dt_match = match_prepared(gt_state, oa, float(threshold))
                    gt_matches[threshold_index].append(gt_match)
                    det_scores[threshold_index].append(dt[:, 4])
                    det_matches[threshold_index].append(dt_match)
            for threshold_index in range(len(IOU_THRESHOLDS)):
                all_gt = np.concatenate(gt_matches[threshold_index]) if gt_matches[threshold_index] else np.zeros(0)
                all_scores = np.concatenate(det_scores[threshold_index]) if det_scores[threshold_index] else np.zeros(0)
                all_matches = np.concatenate(det_matches[threshold_index]) if det_matches[threshold_index] else np.zeros(0)
                order = np.argsort(-all_scores, kind="mergesort") if len(all_scores) else np.zeros(0, dtype=np.int64)
                ranked = all_matches[order]
                tp = np.cumsum(ranked == 1)
                fp = np.cumsum(ranked == 0)
                recall = tp / max(1, len(all_gt))
                precision = tp / np.maximum(1, tp + fp)
                ar[cls - 1, threshold_index, max_index] = (float(np.max(recall)) if len(recall) else 0.0) * 100.0
                if max_det == 500:
                    ap[cls - 1, threshold_index] = voc_ap(recall, precision) * 100.0

    if not np.any(availability):
        raise RuntimeError("No evaluated classes are present")
    weights = availability.astype(np.float64)
    weights /= weights.sum()
    weighted_ap_threshold = np.sum(ap * weights[:, None], axis=0)
    weighted_ar = np.sum(ar * weights[:, None, None], axis=0)
    per_class = {}
    for cls in range(1, 11):
        per_class[str(cls)] = {
            "name": CLASS_NAMES[cls],
            "images_with_class_weight": int(availability[cls - 1]),
            "AP50_95_percent": float(np.mean(ap[cls - 1])),
            "AP50_percent": float(ap[cls - 1, 0]),
            "AP75_percent": float(ap[cls - 1, 5]),
        }
    return {
        "AP50_95_percent": float(np.mean(weighted_ap_threshold)),
        "AP50_percent": float(weighted_ap_threshold[0]),
        "AP75_percent": float(weighted_ap_threshold[5]),
        "AR1_percent": float(np.mean(weighted_ar[:, 0])),
        "AR10_percent": float(np.mean(weighted_ar[:, 1])),
        "AR100_percent": float(np.mean(weighted_ar[:, 2])),
        "AR500_percent": float(np.mean(weighted_ar[:, 3])),
        "class_image_weights": {str(cls): int(availability[cls - 1]) for cls in range(1, 11)},
        "per_class": per_class,
        "aggregation_note": "Exact calcAccuracy.m class-availability weighting, which repeats a class once per image containing it.",
    }


def filter_detections_and_audit(
    records: list[ImageRecord], raw: list[np.ndarray]
) -> tuple[list[np.ndarray], dict[str, object]]:
    filtered: list[np.ndarray] = []
    removed_by_ignore = 0
    overlap_others = 0
    total = 0
    per_class = {str(cls): {"total": 0, "removed_by_class0": 0, "overlap_class11": 0} for cls in range(1, 11)}
    flags_by_image: list[np.ndarray] = []
    for record, image_detections in zip(records, raw):
        keep = np.ones(len(image_detections), dtype=bool)
        flags = np.zeros((len(image_detections), 2), dtype=bool)
        for index, row in enumerate(image_detections):
            cls = str(int(row[5]))
            per_class[cls]["total"] += 1
            ignore_hit = official_region_coverage(row[:4], record.ignore_rects, record.width, record.height) >= 0.5
            others_hit = official_region_coverage(row[:4], record.others_rects, record.width, record.height) >= 0.5
            flags[index] = (ignore_hit, others_hit)
            if ignore_hit:
                keep[index] = False
                removed_by_ignore += 1
                per_class[cls]["removed_by_class0"] += 1
            if others_hit:
                overlap_others += 1
                per_class[cls]["overlap_class11"] += 1
        total += len(image_detections)
        filtered.append(image_detections[keep, :6].copy())
        flags_by_image.append(flags)

    fp_rows: list[dict[str, object]] = []
    for confidence in CONF_AUDIT:
        counters = {
            "all": {"fp": 0, "fp_ignore": 0, "fp_others": 0, "tp": 0, "neutral": 0},
            "small": {"fp": 0, "fp_ignore": 0, "fp_others": 0, "tp": 0, "neutral": 0},
        }
        for record, image_detections, flags in zip(records, raw, flags_by_image):
            globally_limited = image_detections[: min(len(image_detections), 500)]
            globally_flags = flags[: len(globally_limited)]
            for cls in range(1, 11):
                selected = np.nonzero((globally_limited[:, 5] == cls) & (globally_limited[:, 4] >= confidence))[0]
                dt_class = globally_limited[selected]
                flag_class = globally_flags[selected]
                gt_class = record.eval_gt[record.eval_gt[:, 5] == cls]
                _, dt, gt_state, oa, dt_order = prepare_match(gt_class[:, :5], dt_class[:, :5])
                del dt
                _, matches = match_prepared(gt_state, oa, 0.5)
                sorted_detections = dt_class[dt_order]
                sorted_flags = flag_class[dt_order]
                for det_row, det_flags, match in zip(sorted_detections, sorted_flags, matches):
                    groups = ["all"]
                    if det_row[2] * det_row[3] < 32.0**2:
                        groups.append("small")
                    for group in groups:
                        if match == 1:
                            counters[group]["tp"] += 1
                        elif match == -1:
                            counters[group]["neutral"] += 1
                        else:
                            counters[group]["fp"] += 1
                            counters[group]["fp_ignore"] += int(det_flags[0])
                            counters[group]["fp_others"] += int(det_flags[1])
        for group, values in counters.items():
            fp = values["fp"]
            fp_rows.append(
                {
                    "confidence": confidence,
                    "area_group": group,
                    **values,
                    "fp_class0_ignore_fraction": values["fp_ignore"] / max(1, fp),
                    "fp_class11_others_fraction": values["fp_others"] / max(1, fp),
                }
            )
    audit = {
        "raw_prediction_count": total,
        "official_prediction_count_after_class0_filter": int(sum(len(item) for item in filtered)),
        "removed_by_class0_ignore": removed_by_ignore,
        "removed_by_class0_fraction": removed_by_ignore / max(1, total),
        "overlap_class11_others": overlap_others,
        "overlap_class11_fraction": overlap_others / max(1, total),
        "per_class": per_class,
        "fp_audit": fp_rows,
        "fp_audit_note": "Greedy IoU=0.50 class-wise matching before class-0 raster filtering; ignored score-zero GT matches are neutral.",
    }
    return filtered, audit


def load_current_metrics(report_dir: Path) -> dict[str, object]:
    metrics_path = report_dir / "metrics.json"
    if not metrics_path.is_file():
        return {}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "overall": payload.get("overall"),
        "area_metrics": payload.get("area_metrics"),
        "metrics_json": str(metrics_path),
        "metrics_json_sha256": sha256(metrics_path),
    }


def run_unit_tests() -> dict[str, object]:
    # Union must not double-count overlapping ignored rectangles.
    assert union_area_inclusive([(1, 1, 3, 3), (2, 2, 4, 4)]) == 14
    # Integral-image convention returns h*w, not the (w+1)*(h+1) size of the
    # rasterized source region.
    coverage = official_region_coverage(
        [1, 1, 10, 10], np.asarray([[1, 1, 10, 10]], dtype=np.float64), 100, 100
    )
    assert abs(coverage - 1.0) < 1e-12, coverage
    # Ordinary TP.
    gt0 = np.asarray([[0, 0, 10, 10, 0]], dtype=np.float64)
    dt0 = np.asarray([[0, 0, 10, 10, 0.9]], dtype=np.float64)
    _, _, state, oa, _ = prepare_match(gt0, dt0)
    gt_match, dt_match = match_prepared(state, oa, 0.5)
    assert gt_match.tolist() == [1] and dt_match.tolist() == [1]
    # Ignore match uses intersection / detection area and is neutral.
    gt0 = np.asarray([[0, 0, 20, 20, 1]], dtype=np.float64)
    dt0 = np.asarray([[5, 5, 5, 5, 0.8]], dtype=np.float64)
    _, _, state, oa, _ = prepare_match(gt0, dt0)
    _, dt_match = match_prepared(state, oa, 0.5)
    assert dt_match.tolist() == [-1]
    # A normal match is preferred over a later ignore match.
    gt0 = np.asarray([[0, 0, 10, 10, 0], [0, 0, 20, 20, 1]], dtype=np.float64)
    dt0 = np.asarray([[0, 0, 10, 10, 0.7]], dtype=np.float64)
    _, _, state, oa, _ = prepare_match(gt0, dt0)
    _, dt_match = match_prepared(state, oa, 0.5)
    assert dt_match.tolist() == [1]
    # Perfect one-item precision-recall integrates to one.
    assert abs(voc_ap(np.asarray([1.0]), np.asarray([1.0])) - 1.0) < 1e-12
    # End-to-end official aggregation: a single perfect class-1 detection must
    # produce 100 AP/AR while classes absent from the fixture receive zero
    # availability weight.  This catches regressions in maxDet ordering,
    # class filtering, matching, and calcAccuracy-style aggregation together.
    fixture = ImageRecord(
        image_id=1,
        stem="synthetic",
        image_path=Path("synthetic.jpg"),
        annotation_path=Path("synthetic.txt"),
        width=100,
        height=100,
        raw_gt=np.asarray([[10, 10, 10, 10, 1, 1, 0, 0]], dtype=np.float64),
        eval_gt=np.asarray([[10, 10, 10, 10, 0, 1]], dtype=np.float64),
        ignore_rects=np.zeros((0, 4), dtype=np.float64),
        others_rects=np.zeros((0, 4), dtype=np.float64),
    )
    perfect = np.asarray([[10, 10, 10, 10, 0.9, 1, 0]], dtype=np.float64)
    official = official_evaluate([fixture], [perfect])
    for key in ("AP50_95_percent", "AP50_percent", "AP75_percent", "AR500_percent"):
        assert abs(float(official[key]) - 100.0) < 1e-9, (key, official[key])
    return {
        "union_area": "passed",
        "integral_region_coverage": "passed",
        "ordinary_match": "passed",
        "ignore_match": "passed",
        "normal_preferred": "passed",
        "voc_ap": "passed",
        "official_end_to_end": "passed",
    }


def preflight(project: Path, dataset: Path, model_names: list[str]) -> tuple[list[ImageRecord], dict[str, object], dict[str, Path]]:
    unit = run_unit_tests()
    if not project.is_dir():
        raise FileNotFoundError(project)
    records, dataset_summary = load_dataset(dataset)
    reports = resolve_models(project, model_names)
    source_summary = {}
    for name, report_dir in reports.items():
        prediction_path = report_dir / "area_predictions.json"
        ground_truth_path = report_dir / "area_ground_truth.json"
        if not ground_truth_path.is_file():
            raise FileNotFoundError(ground_truth_path)
        # Cheap structural validation without retaining the complete JSON.
        with prediction_path.open("r", encoding="utf-8") as handle:
            first_nonspace = next((character for character in iter(lambda: handle.read(1), "") if not character.isspace()), "")
        if first_nonspace != "[":
            raise ValueError(f"Prediction JSON does not start with a list: {prediction_path}")
        # The predictions contain only numeric image ids.  Verify that those
        # ids mean the same sorted VisDrone files as this audit assumes.
        saved_gt = json.loads(ground_truth_path.read_text(encoding="utf-8"))
        saved_images = saved_gt.get("images") if isinstance(saved_gt, dict) else None
        if not isinstance(saved_images, list) or len(saved_images) != len(records):
            raise ValueError(f"Unexpected image table in {ground_truth_path}")
        for record, saved in zip(records, saved_images):
            if (
                int(saved.get("id", -1)) != record.image_id
                or Path(str(saved.get("file_name", ""))).stem != record.stem
                or int(saved.get("width", -1)) != record.width
                or int(saved.get("height", -1)) != record.height
            ):
                raise ValueError(
                    f"Prediction image-id mapping mismatch for {name}: "
                    f"record={record.image_id}/{record.stem}/{record.width}x{record.height}, saved={saved}"
                )
        del saved_gt, saved_images
        source_summary[name] = {
            "report_dir": str(report_dir),
            "prediction_path": str(prediction_path),
            "prediction_bytes": prediction_path.stat().st_size,
            "prediction_sha256": sha256(prediction_path),
            "ground_truth_path": str(ground_truth_path),
            "ground_truth_sha256": sha256(ground_truth_path),
            "image_id_mapping": "passed",
        }
    return records, {"unit_tests": unit, "dataset": dataset_summary, "sources": source_summary}, reports


def fp_row(audit: dict[str, object], confidence: float, group: str) -> dict[str, object]:
    for row in audit["fp_audit"]:
        if abs(float(row["confidence"]) - confidence) < 1e-9 and row["area_group"] == group:
            return row
    raise KeyError((confidence, group))


def rank_models(results: dict[str, object], source: str) -> list[str]:
    if source == "official":
        key = lambda name: float(results[name]["official"]["AP50_95_percent"])
    elif source == "current_overall":
        key = lambda name: float(results[name]["current_metrics"]["overall"]["mAP50_95"])
    else:
        key = lambda name: float(results[name]["current_metrics"]["area_metrics"]["AP_all"])
    available = []
    for name in results:
        try:
            key(name)
        except (KeyError, TypeError):
            continue
        available.append(name)
    return sorted(available, key=key, reverse=True)


def save_report(report_dir: Path, report: dict[str, object]) -> None:
    report_dir.mkdir(parents=True, exist_ok=False)
    write_json(report_dir / "metrics.json", report)
    write_json(report_dir / "dataset_summary.json", report["preflight"]["dataset"])
    write_json(report_dir / "decision.json", report["decision"])

    with (report_dir / "model_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "official_AP50_95_percent",
                "official_AP50_percent",
                "official_AP75_percent",
                "official_AR500_percent",
                "current_ultralytics_mAP50_95_percent",
                "current_coco_area_AP_all_percent",
                "raw_predictions",
                "removed_by_class0",
                "removed_fraction",
                "small_fp_ignore_fraction_conf010",
                "small_fp_others_fraction_conf010",
            ],
        )
        writer.writeheader()
        for name, item in report["models"].items():
            current = item.get("current_metrics", {})
            overall = current.get("overall") or {}
            area = current.get("area_metrics") or {}
            audit = item["ignore_audit"]
            fp = fp_row(audit, 0.10, "small")
            writer.writerow(
                {
                    "model": name,
                    "official_AP50_95_percent": item["official"]["AP50_95_percent"],
                    "official_AP50_percent": item["official"]["AP50_percent"],
                    "official_AP75_percent": item["official"]["AP75_percent"],
                    "official_AR500_percent": item["official"]["AR500_percent"],
                    "current_ultralytics_mAP50_95_percent": float(overall.get("mAP50_95", float("nan"))) * 100,
                    "current_coco_area_AP_all_percent": float(area.get("AP_all", float("nan"))) * 100,
                    "raw_predictions": audit["raw_prediction_count"],
                    "removed_by_class0": audit["removed_by_class0_ignore"],
                    "removed_fraction": audit["removed_by_class0_fraction"],
                    "small_fp_ignore_fraction_conf010": fp["fp_class0_ignore_fraction"],
                    "small_fp_others_fraction_conf010": fp["fp_class11_others_fraction"],
                }
            )

    with (report_dir / "official_per_class.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = ["model", "class_id", "class_name", "images_with_class_weight", "AP50_95_percent", "AP50_percent", "AP75_percent"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, item in report["models"].items():
            for cls, row in item["official"]["per_class"].items():
                writer.writerow({"model": name, "class_id": cls, "class_name": row["name"], **{key: row[key] for key in fieldnames[3:]}})

    with (report_dir / "ignore_fp_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [
            "model",
            "confidence",
            "area_group",
            "tp",
            "fp",
            "neutral",
            "fp_ignore",
            "fp_others",
            "fp_class0_ignore_fraction",
            "fp_class11_others_fraction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, item in report["models"].items():
            for row in item["ignore_audit"]["fp_audit"]:
                writer.writerow({"model": name, **{key: row[key] for key in fieldnames[1:]}})

    lines = [
        "D27 Official VisDrone DET protocol audit",
        f"status: {report['status']}",
        f"revision: {REVISION}",
        "ZERO TRAINING. Existing saved predictions only. Automatic shutdown disabled.",
        "Primary AP uses a Python port of public VisDrone DET Matlab toolkit v1.0.4.",
        "Do not compare the absolute official AP directly with converted-label COCO area AP.",
        "",
        "[official model ranking]",
    ]
    for rank, name in enumerate(report["decision"]["official_ranking"], 1):
        item = report["models"][name]
        official = item["official"]
        fp = fp_row(item["ignore_audit"], 0.10, "small")
        lines.append(
            f"{rank}. {name}: AP={official['AP50_95_percent']:.4f}% "
            f"AP50={official['AP50_percent']:.4f}% AP75={official['AP75_percent']:.4f}% "
            f"AR500={official['AR500_percent']:.4f}% "
            f"small-FP-ignore@0.10={100*float(fp['fp_class0_ignore_fraction']):.3f}%"
        )
    lines.extend(
        [
            "",
            "[decision]",
            f"ranking_changed_vs_current_ultralytics: {report['decision']['ranking_changed_vs_current_ultralytics']}",
            f"ranking_changed_vs_current_coco_area: {report['decision']['ranking_changed_vs_current_coco_area']}",
            f"e1_small_fp_ignore_fraction_conf010: {report['decision']['e1_small_fp_ignore_fraction_conf010']:.6f}",
            f"next_step: {report['decision']['next_step']}",
            f"rationale: {report['decision']['rationale']}",
            "",
            "Scientific boundary:",
            "- The primary port follows the public Matlab v1.0.4 implementation, not a claim of server bit identity.",
            "- class-11 overlap is reported separately and is not silently inserted into the primary metric.",
            "- this audit does not establish that ignore-aware training will improve AP.",
        ]
    )
    (report_dir / "metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--models", default=",".join(MODEL_SPECS), help="Comma-separated subset of e0,e1,e10a,e16a,e21a,e26a")
    parser.add_argument("--check-only", action="store_true", help="Run unit tests and validate dataset/report sources, then exit")
    parser.add_argument("--self-test-only", action="store_true", help="Run dependency-free synthetic evaluator tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test_only:
        print(json.dumps(run_unit_tests(), indent=2))
        print("D27 synthetic evaluator tests passed.")
        return
    model_names = [item.strip().lower() for item in args.models.split(",") if item.strip()]
    records, preflight_data, reports = preflight(args.project.resolve(), args.dataset.resolve(), model_names)
    if args.check_only:
        print(f"D27 check passed: {len(records)} images, models={','.join(model_names)}")
        for name, item in preflight_data["sources"].items():
            print(f"  {name}: {item['prediction_path']} ({item['prediction_bytes']} bytes)")
        print("No evaluation, training, checkpoint change, or shutdown was performed.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_dir = args.project.resolve() / "comparison_reports" / f"DIAG_D27_OFFICIAL_VISDRONE_{stamp}"
    started = time.perf_counter()
    report: dict[str, object] = {
        "status": "running",
        "revision": REVISION,
        "started_at": datetime.now().astimezone().isoformat(),
        "report_dir": str(report_dir),
        "automatic_shutdown": False,
        "zero_training": True,
        "protocol": {
            "source": "VisDrone/VisDrone2018-DET-toolkit public Matlab v1.0.4 semantics",
            "source_urls": [
                "https://github.com/VisDrone/VisDrone2018-DET-toolkit",
                "https://github.com/VisDrone/VisDrone2018-DET-toolkit/blob/master/utils/calcAccuracy.m",
                "https://github.com/VisDrone/VisDrone2018-DET-toolkit/blob/master/utils/dropObjectsInIgr.m",
                "https://github.com/VisDrone/VisDrone2018-DET-toolkit/blob/master/utils/evalRes.m",
            ],
            "iou_thresholds": IOU_THRESHOLDS.tolist(),
            "max_dets": list(MAX_DETS),
            "primary_ignore_filter": "class-0 raster coverage >= 0.5, as public dropObjectsInIgr.m",
            "class11_policy": "not an evaluated class; overlap is separately audited and not added to primary class-0 raster filter",
        },
        "preflight": preflight_data,
        "models": {},
    }
    try:
        for model_index, (name, report_source) in enumerate(reports.items(), 1):
            prediction_path = report_source / "area_predictions.json"
            print(f"[{model_index}/{len(reports)}] Loading {name}: {prediction_path}", flush=True)
            raw = load_predictions(prediction_path, len(records))
            print(f"[{model_index}/{len(reports)}] Filtering official ignore regions and auditing FP: {name}", flush=True)
            filtered, ignore_audit = filter_detections_and_audit(records, raw)
            print(f"[{model_index}/{len(reports)}] Running official metric port: {name}", flush=True)
            official = official_evaluate(records, filtered)
            report["models"][name] = {
                "source_report": str(report_source),
                "prediction_path": str(prediction_path),
                "prediction_sha256": sha256(prediction_path),
                "current_metrics": load_current_metrics(report_source),
                "ignore_audit": ignore_audit,
                "official": official,
            }
            del raw, filtered
            gc.collect()

        official_ranking = rank_models(report["models"], "official")
        overall_ranking = rank_models(report["models"], "current_overall")
        area_ranking = rank_models(report["models"], "current_area")
        e1_fp = fp_row(report["models"]["e1"]["ignore_audit"], 0.10, "small") if "e1" in report["models"] else None
        e1_fraction = float(e1_fp["fp_class0_ignore_fraction"]) if e1_fp else float("nan")
        recommend_ignore_test = bool(e1_fp and e1_fraction >= 0.10)
        next_step = (
            "DESIGN_ONE_E1_IGNORE_AWARE_CAUSAL_RUN_BEFORE_NEW_ARCHITECTURE"
            if recommend_ignore_test
            else "PROTOCOL_CORRECTED_BASELINE_THEN_NATIVE_RESOLUTION_INFORMATION_ROUTE"
        )
        rationale = (
            "At least 10% of E1 small pre-filter false positives at conf=0.10 fall in class-0 ignore regions; "
            "first test whether preventing these regions from acting as negatives changes the full-view baseline."
            if recommend_ignore_test
            else "E1 class-0 ignore overlap is below the preregistered 10% screening threshold; do not spend a formal run on ignore masking alone."
        )
        report["decision"] = {
            "official_ranking": official_ranking,
            "current_ultralytics_ranking": overall_ranking,
            "current_coco_area_ranking": area_ranking,
            "ranking_changed_vs_current_ultralytics": official_ranking != overall_ranking,
            "ranking_changed_vs_current_coco_area": official_ranking != area_ranking,
            "e1_small_fp_ignore_fraction_conf010": e1_fraction,
            "ignore_aware_causal_test_threshold": 0.10,
            "ignore_aware_causal_test_recommended": recommend_ignore_test,
            "next_step": next_step,
            "rationale": rationale,
            "no_new_architecture_was_trained": True,
        }
        report["status"] = "completed"
        report["finished_at"] = datetime.now().astimezone().isoformat()
        report["elapsed_seconds"] = time.perf_counter() - started
        save_report(report_dir, report)
        print(f"D27 completed. Report: {report_dir / 'metrics.txt'}")
        print("No training was performed. Automatic shutdown is disabled.")
    except Exception as error:
        report_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "failed",
            "revision": REVISION,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "finished_at": datetime.now().astimezone().isoformat(),
            "automatic_shutdown": False,
        }
        write_json(report_dir / "failure.json", failure)
        print(f"D27 failure saved: {report_dir}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
