#!/usr/bin/env python3
"""Frozen-weight slicing diagnostic for E1 YOLO11s-P2 and E5a DySample.

This script does not train or alter either checkpoint.  It compares, under one
custom COCO-style evaluator:

1. full-image prediction;
2. fixed 2x2 overlapping sliced prediction;
3. the class-aware NMS union of full and sliced predictions.

The diagnostic intentionally uses the converted ten-class YOLO labels so its
area metrics are comparable with the existing experiment reports.  It is not
the official VisDrone ignore-region evaluation protocol.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path("/root/autodl-tmp/yolo11_project")
DATA = Path("/root/autodl-tmp/project/VisDrone2019/VisDrone2019.yaml")
RUNS = ROOT / "runs"
REPORTS = ROOT / "comparison_reports"

WEIGHTS = {
    "e1": RUNS / "e1_yolo11s_p2add_img1024_seed1/weights/best.pt",
    "e5a": RUNS / "e5a_yolo11s_p2_dysample_neck_img1024_seed1/weights/best.pt",
}
REFERENCE_REPORT_GLOBS = {
    "e1": "e1_yolo11s_p2add_img1024_seed1_*/metrics.json",
    "e5a": "e5a_yolo11s_p2_dysample_neck_img1024_seed1_*/metrics.json",
}

EXPECTED_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor",
]
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

IMGSZ = 1024
PREDICT_BATCH = 4
CONF = 0.001
MODEL_NMS_IOU = 0.70
MERGE_NMS_IOU = 0.60
MAX_DET = 500
GRID = 2
OVERLAP = 0.20
EDGE_MARGIN_FRACTION = 0.005
FIXED_PR_CONF = 0.25
FIXED_PR_IOU = 0.50
SCRIPT_REVISION = "fixed_2x2_slicing_upper_bound_v2"
REFERENCE_PARITY_TOLERANCE = 0.002

np = None
torch = None
cv2 = None
ultralytics = None
YOLO = None
Image = None
check_det_dataset = None
img2label_paths = None
COCO = None
COCOeval = None
batched_nms = None


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot JSON serialize {type(value)!r}")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dependencies() -> None:
    global np, torch, cv2, ultralytics, YOLO, Image
    global check_det_dataset, img2label_paths, COCO, COCOeval, batched_nms
    import numpy as _np
    import torch as _torch
    import cv2 as _cv2
    import ultralytics as _ultralytics
    from PIL import Image as _Image
    from pycocotools.coco import COCO as _COCO
    from pycocotools.cocoeval import COCOeval as _COCOeval
    from torchvision.ops import batched_nms as _batched_nms
    from ultralytics import YOLO as _YOLO
    from ultralytics.data.utils import check_det_dataset as _check_det_dataset
    from ultralytics.data.utils import img2label_paths as _img2label_paths

    np, torch, cv2 = _np, _torch, _cv2
    ultralytics, YOLO, Image = _ultralytics, _YOLO, _Image
    check_det_dataset, img2label_paths = _check_det_dataset, _img2label_paths
    COCO, COCOeval, batched_nms = _COCO, _COCOeval, _batched_nms


def clear_gpu() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def memory_peak() -> dict:
    return {
        "allocated_GiB": torch.cuda.max_memory_allocated(0) / 1024**3,
        "reserved_GiB": torch.cuda.max_memory_reserved(0) / 1024**3,
    }


def normalize_names(names) -> dict:
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(k): str(v) for k, v in names.items()}


def prepare_dataset(report: dict, limit: int = 0):
    dataset = check_det_dataset(str(DATA), autodownload=False)
    names = normalize_names(dataset["names"])
    if [names.get(k) for k in range(10)] != EXPECTED_CLASSES or len(names) != 10:
        raise ValueError("Dataset class order differs from the agreed VisDrone ten classes.")
    source = dataset["val"]
    if not isinstance(source, str) or not Path(source).is_dir():
        raise ValueError("Validation split must resolve to one image directory.")
    all_images = sorted(
        p.resolve() for p in Path(source).rglob("*")
        if p.is_file() and p.suffix.lower() in SUFFIXES
    )
    if len(all_images) != 548:
        raise ValueError(f"Expected 548 validation images, found {len(all_images)}.")
    images = all_images[:limit] if limit else all_images

    gt = {
        "info": {"description": "YOLO-converted VisDrone diagnostic ground truth"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": k + 1, "name": v} for k, v in names.items()],
    }
    path_ids = {}
    class_counts = {k: 0 for k in names}
    digest = hashlib.sha256()
    labels = img2label_paths([str(p) for p in images])
    duplicates = 0
    for image_id, (image_path, label_path) in enumerate(zip(images, labels), 1):
        label_path = Path(label_path)
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing validation label: {label_path}")
        with Image.open(image_path) as image:
            width, height = image.size
            if image.getexif().get(274, 1) not in (None, 1):
                raise ValueError(f"EXIF-rotated image requires protocol review: {image_path}")
        content = label_path.read_bytes()
        digest.update(str(image_path).encode() + b"\0" + content + b"\0")
        path_ids[str(image_path)] = image_id
        gt["images"].append({
            "id": image_id, "file_name": str(image_path),
            "width": width, "height": height,
        })
        seen = set()
        for line in content.decode("utf-8-sig").splitlines():
            if not line.strip():
                continue
            fields = tuple(map(float, line.split()))
            if len(fields) != 5 or not all(math.isfinite(x) for x in fields):
                raise ValueError(f"Invalid detection label: {label_path}")
            cls, xc, yc, bw, bh = fields
            if cls != int(cls) or int(cls) not in names:
                raise ValueError(f"Invalid class ID in {label_path}")
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                raise ValueError(f"Invalid normalized box in {label_path}")
            if fields in seen:
                duplicates += 1
                continue
            seen.add(fields)
            w, h = bw * width, bh * height
            gt["annotations"].append({
                "id": len(gt["annotations"]) + 1,
                "image_id": image_id,
                "category_id": int(cls) + 1,
                "bbox": [xc * width - w / 2, yc * height - h / 2, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            class_counts[int(cls)] += 1
    if not limit and len(gt["annotations"]) != 38759:
        raise ValueError(f"Expected 38759 validation labels, found {len(gt['annotations'])}.")
    report["dataset"] = {
        "available_val_images": len(all_images),
        "evaluated_images": len(images),
        "evaluated_instances": len(gt["annotations"]),
        "validation_label_sha256": digest.hexdigest(),
        "duplicate_labels_removed": duplicates,
        "class_counts": class_counts,
    }
    atomic_json(Path(report["report_dir"]) / "ground_truth.json", gt)
    return images, path_ids, gt, names


def axis_tiles(length: int) -> tuple[int, list[int]]:
    if length <= 1:
        raise ValueError(f"Invalid image dimension: {length}")
    denominator = GRID - (GRID - 1) * OVERLAP
    tile = min(length, int(math.ceil(length / denominator)))
    starts = [int(round(i * (length - tile) / (GRID - 1))) for i in range(GRID)]
    starts[0], starts[-1] = 0, length - tile
    if starts != sorted(set(starts)):
        raise RuntimeError(f"Degenerate tile starts for length {length}: {starts}")
    return tile, starts


def make_tiles(image) -> list[dict]:
    height, width = image.shape[:2]
    tile_w, xs = axis_tiles(width)
    tile_h, ys = axis_tiles(height)
    tiles = []
    for y0 in ys:
        for x0 in xs:
            x1, y1 = x0 + tile_w, y0 + tile_h
            crop = image[y0:y1, x0:x1]
            if crop.shape[0] != tile_h or crop.shape[1] != tile_w:
                raise RuntimeError("Tile geometry does not cover the requested crop.")
            tiles.append({
                "image": crop,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "full_w": width, "full_h": height,
            })
    if len(tiles) != GRID * GRID:
        raise RuntimeError(f"Expected {GRID * GRID} tiles, produced {len(tiles)}")
    return tiles


def empty_detections():
    return torch.empty((0, 6), dtype=torch.float32)


def result_tensor(result):
    if result.boxes is None or result.boxes.data.numel() == 0:
        return empty_detections()
    return result.boxes.data.detach().float().cpu().clone()


def nms_detections(data, iou: float = MERGE_NMS_IOU, max_det: int = MAX_DET):
    if data is None or data.numel() == 0:
        return empty_detections()
    data = data[torch.isfinite(data).all(dim=1)]
    if data.numel() == 0:
        return empty_detections()
    boxes = data[:, :4]
    scores = data[:, 4]
    classes = data[:, 5].long()
    keep = batched_nms(boxes, scores, classes, iou)[:max_det]
    return data[keep].contiguous()


def map_tile_detections(result, tile: dict):
    data = result_tensor(result)
    if data.numel() == 0:
        return data
    crop_h, crop_w = tile["image"].shape[:2]
    margin = max(2.0, min(crop_w, crop_h) * EDGE_MARGIN_FRACTION)
    internal_left = tile["x0"] > 0
    internal_top = tile["y0"] > 0
    internal_right = tile["x1"] < tile["full_w"]
    internal_bottom = tile["y1"] < tile["full_h"]
    keep = torch.ones(len(data), dtype=torch.bool)
    if internal_left:
        keep &= data[:, 0] > margin
    if internal_top:
        keep &= data[:, 1] > margin
    if internal_right:
        keep &= data[:, 2] < crop_w - margin
    if internal_bottom:
        keep &= data[:, 3] < crop_h - margin
    data = data[keep]
    if data.numel():
        data[:, [0, 2]] += float(tile["x0"])
        data[:, [1, 3]] += float(tile["y0"])
    return data


def model_predict(model, sources, batch: int):
    return model.predict(
        source=sources,
        imgsz=IMGSZ,
        batch=batch,
        device=0,
        conf=CONF,
        iou=MODEL_NMS_IOU,
        max_det=MAX_DET,
        half=False,
        augment=False,
        verbose=False,
        save=False,
        stream=False,
    )


def predict_full(model, images: list[Path], report: dict, model_name: str):
    predictions = {}
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for start in range(0, len(images), PREDICT_BATCH):
        chunk = images[start:start + PREDICT_BATCH]
        results = model_predict(model, [str(p) for p in chunk], len(chunk))
        if len(results) != len(chunk):
            raise RuntimeError("Full-image prediction result count mismatch.")
        for path, result in zip(chunk, results):
            predictions[str(path)] = result_tensor(result)
        del results
        done = min(start + PREDICT_BATCH, len(images))
        if start == 0 or done % 100 == 0 or done == len(images):
            print(f"{model_name} full prediction: {done}/{len(images)}", flush=True)
            report["progress"] = {"model": model_name, "stage": "full", "images": done}
            save_report(report)
    torch.cuda.synchronize()
    return predictions, {
        "seconds": time.perf_counter() - started,
        "peak_memory": memory_peak(),
        "forward_images": len(images),
    }


def predict_sliced(model, images: list[Path], report: dict, model_name: str):
    predictions = {}
    geometry_example = None
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for index, path in enumerate(images, 1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV failed to read {path}")
        tiles = make_tiles(image)
        if geometry_example is None:
            geometry_example = [
                {k: v for k, v in tile.items() if k != "image"} for tile in tiles
            ]
        results = model_predict(model, [tile["image"] for tile in tiles], len(tiles))
        if len(results) != len(tiles):
            raise RuntimeError("Sliced prediction result count mismatch.")
        mapped = [map_tile_detections(result, tile) for result, tile in zip(results, tiles)]
        candidates = torch.cat(mapped, dim=0) if mapped else empty_detections()
        predictions[str(path)] = nms_detections(candidates)
        del results, mapped, candidates, image, tiles
        if index == 1 or index % 50 == 0 or index == len(images):
            print(f"{model_name} sliced prediction: {index}/{len(images)}", flush=True)
            report["progress"] = {"model": model_name, "stage": "sliced", "images": index}
            save_report(report)
    torch.cuda.synchronize()
    return predictions, {
        "seconds": time.perf_counter() - started,
        "peak_memory": memory_peak(),
        "forward_images": len(images) * GRID * GRID,
        "tiles_per_image": GRID * GRID,
        "example_geometry": geometry_example,
    }


def combine_predictions(full: dict, sliced: dict, images: list[Path]):
    combined = {}
    started = time.perf_counter()
    for path in images:
        key = str(path)
        candidates = torch.cat([full[key], sliced[key]], dim=0)
        combined[key] = nms_detections(candidates)
    return combined, {"seconds": time.perf_counter() - started}


def predictions_to_coco(predictions: dict, images: list[Path], path_ids: dict) -> list[dict]:
    rows = []
    for path in images:
        data = predictions[str(path)]
        image_id = path_ids[str(path)]
        for x1, y1, x2, y2, score, cls in data.tolist():
            rows.append({
                "image_id": image_id,
                "category_id": int(cls) + 1,
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": float(score),
            })
    return rows


def valid_mean(values):
    values = values[values > -1]
    return float(values.mean()) if values.size else None


def coco_metrics(gt: dict, detections: list[dict], names: dict) -> dict:
    gt_coco = COCO()
    gt_coco.dataset = gt
    gt_coco.createIndex()
    if detections:
        dt_coco = gt_coco.loadRes(detections)
    else:
        dt_coco = COCO()
        dt_coco.dataset = {
            "images": gt["images"], "categories": gt["categories"], "annotations": []
        }
        dt_coco.createIndex()
    evaluator = COCOeval(gt_coco, dt_coco, "bbox")
    evaluator.params.maxDets = [1, 10, MAX_DET]
    evaluator.evaluate()
    evaluator.accumulate()
    params = evaluator.params
    max_index = params.maxDets.index(MAX_DET)
    area_all = params.areaRngLbl.index("all")
    threshold_50 = int(np.argmin(np.abs(params.iouThrs - 0.50)))
    threshold_75 = int(np.argmin(np.abs(params.iouThrs - 0.75)))
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    overall = {
        "mAP50_95": valid_mean(precision[:, :, :, area_all, max_index]),
        "mAP50": valid_mean(precision[threshold_50, :, :, area_all, max_index]),
        "mAP75": valid_mean(precision[threshold_75, :, :, area_all, max_index]),
        "AR_all": valid_mean(recall[:, :, area_all, max_index]),
    }
    for label in ("small", "medium", "large"):
        area_index = params.areaRngLbl.index(label)
        overall[f"AP_{label}"] = valid_mean(precision[:, :, :, area_index, max_index])
        overall[f"AR_{label}"] = valid_mean(recall[:, :, area_index, max_index])
    per_class = {}
    for class_index, category_id in enumerate(params.catIds):
        name = names[int(category_id) - 1]
        row = {
            "AP50_95": valid_mean(precision[:, :, class_index, area_all, max_index]),
            "AP50": valid_mean(precision[threshold_50, :, class_index, area_all, max_index]),
            "AP75": valid_mean(precision[threshold_75, :, class_index, area_all, max_index]),
        }
        for label in ("small", "medium", "large"):
            area_index = params.areaRngLbl.index(label)
            row[f"AP_{label}"] = valid_mean(
                precision[:, :, class_index, area_index, max_index]
            )
        per_class[name] = row
    return {"overall": overall, "per_class": per_class}


def xywh_to_xyxy(boxes):
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    result = boxes.copy()
    result[:, 2] = boxes[:, 0] + boxes[:, 2]
    result[:, 3] = boxes[:, 1] + boxes[:, 3]
    return result


def box_iou_numpy(one, many):
    if len(many) == 0:
        return np.empty((0,), dtype=np.float32)
    left_top = np.maximum(one[:2], many[:, :2])
    right_bottom = np.minimum(one[2:], many[:, 2:])
    intersection = np.prod(np.maximum(0.0, right_bottom - left_top), axis=1)
    area_one = max(0.0, one[2] - one[0]) * max(0.0, one[3] - one[1])
    area_many = np.maximum(0.0, many[:, 2] - many[:, 0]) * np.maximum(
        0.0, many[:, 3] - many[:, 1]
    )
    return intersection / np.maximum(area_one + area_many - intersection, 1e-12)


def fixed_operating_point(gt: dict, detections: list[dict], names: dict) -> dict:
    gt_groups = defaultdict(list)
    pred_groups = defaultdict(list)
    for ann in gt["annotations"]:
        gt_groups[(ann["image_id"], ann["category_id"])].append(ann["bbox"])
    for pred in detections:
        if pred["score"] >= FIXED_PR_CONF:
            pred_groups[(pred["image_id"], pred["category_id"])].append(pred)
    totals = {category_id: {"tp": 0, "fp": 0, "fn": 0} for category_id in range(1, 11)}
    image_ids = [row["id"] for row in gt["images"]]
    for image_id in image_ids:
        for category_id in range(1, 11):
            ground = xywh_to_xyxy(gt_groups.get((image_id, category_id), []))
            predictions = sorted(
                pred_groups.get((image_id, category_id), []),
                key=lambda row: row["score"], reverse=True,
            )
            used = np.zeros(len(ground), dtype=bool)
            tp = 0
            for prediction in predictions:
                box = xywh_to_xyxy([prediction["bbox"]])[0]
                ious = box_iou_numpy(box, ground)
                if len(ious):
                    ious[used] = -1
                    best = int(np.argmax(ious))
                    if ious[best] >= FIXED_PR_IOU:
                        used[best] = True
                        tp += 1
            totals[category_id]["tp"] += tp
            totals[category_id]["fp"] += len(predictions) - tp
            totals[category_id]["fn"] += len(ground) - tp

    def summarize(counts):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        return {
            **counts,
            "Precision": tp / (tp + fp) if tp + fp else 0.0,
            "Recall": tp / (tp + fn) if tp + fn else 0.0,
        }

    merged = {key: sum(row[key] for row in totals.values()) for key in ("tp", "fp", "fn")}
    return {
        "definition": f"micro matching at conf>={FIXED_PR_CONF}, IoU>={FIXED_PR_IOU}",
        "overall": summarize(merged),
        "per_class": {names[k - 1]: summarize(v) for k, v in totals.items()},
    }


def evaluate_mode(
    model_name: str,
    mode_name: str,
    predictions: dict,
    images: list[Path],
    path_ids: dict,
    gt: dict,
    names: dict,
    out: Path,
):
    print(f"Evaluating {model_name}/{mode_name} on CPU.", flush=True)
    detections = predictions_to_coco(predictions, images, path_ids)
    atomic_json(out / f"{model_name}_{mode_name}_predictions.json", detections)
    metrics = coco_metrics(gt, detections, names)
    metrics["fixed_operating_point"] = fixed_operating_point(gt, detections, names)
    metrics["prediction_count"] = len(detections)
    return metrics


def assert_model(model, names: dict, model_name: str) -> dict:
    model_names = normalize_names(model.names)
    if model_names != names:
        raise ValueError(f"{model_name} checkpoint class names differ from the dataset.")
    strides = [int(round(float(value))) for value in model.model.stride.cpu().tolist()]
    if strides != [4, 8, 16, 32]:
        raise ValueError(f"{model_name} expected strides [4,8,16,32], got {strides}")
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    return {"strides": strides, "parameters_loaded": parameter_count}


def load_reference(model_name: str) -> dict | None:
    matches = sorted(REPORTS.glob(REFERENCE_REPORT_GLOBS[model_name]))
    completed = []
    for path in matches:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if value.get("status") == "completed":
            completed.append((path, value))
    if not completed:
        return None
    path, value = completed[-1]
    return {
        "path": str(path),
        "reported_overall": value.get("overall"),
        "reported_area_metrics": value.get("area_metrics"),
    }


def model_diagnostic(
    model_name: str,
    weight: Path,
    images: list[Path],
    path_ids: dict,
    gt: dict,
    names: dict,
    report: dict,
):
    print(f"\nLoading frozen {model_name} checkpoint: {weight}", flush=True)
    weight_sha256_before = sha256(weight)
    model = YOLO(str(weight))
    structure = assert_model(model, names, model_name)
    report["models"].setdefault(model_name, {}).update(
        weight=str(weight), sha256=weight_sha256_before, structure=structure,
        reference_report=load_reference(model_name),
    )
    save_report(report)

    full, full_profile = predict_full(model, images, report, model_name)
    sliced, sliced_profile = predict_sliced(model, images, report, model_name)
    combined, merge_profile = combine_predictions(full, sliced, images)

    out = Path(report["report_dir"])
    mode_predictions = {"full": full, "sliced": sliced, "combined": combined}
    mode_profiles = {
        "full": full_profile,
        "sliced": sliced_profile,
        "combined": {
            "seconds": (
                full_profile["seconds"] + sliced_profile["seconds"] + merge_profile["seconds"]
            ),
            "merge_only_seconds": merge_profile["seconds"],
            "forward_images": full_profile["forward_images"] + sliced_profile["forward_images"],
            "peak_memory": {
                key: max(full_profile["peak_memory"][key], sliced_profile["peak_memory"][key])
                for key in ("allocated_GiB", "reserved_GiB")
            },
        },
    }
    metrics = {}
    for mode_name, predictions in mode_predictions.items():
        metrics[mode_name] = evaluate_mode(
            model_name, mode_name, predictions, images, path_ids, gt, names, out
        )
        metrics[mode_name]["profile"] = mode_profiles[mode_name]
        report["models"][model_name]["modes"] = metrics
        save_report(report)
    del model, full, sliced, combined, mode_predictions
    clear_gpu()
    weight_sha256_after = sha256(weight)
    if weight_sha256_after != weight_sha256_before:
        raise RuntimeError(f"Frozen checkpoint changed during evaluation: {weight}")
    report["models"][model_name]["checkpoint_unchanged"] = True
    return metrics


def build_decisions(report: dict) -> dict:
    decisions = {}
    for model_name, model_row in report["models"].items():
        modes = model_row.get("modes")
        if not modes:
            continue
        full = modes["full"]["overall"]
        combined = modes["combined"]["overall"]
        delta_ap = combined["mAP50_95"] - full["mAP50_95"]
        delta_small = combined["AP_small"] - full["AP_small"]
        reference = model_row.get("reference_report") or {}
        reference_area = reference.get("reported_area_metrics") or {}
        reference_pairs = {
            "mAP50_95": "AP_all",
            "AP_small": "AP_small",
            "AR_small": "AR_small",
            "AP_medium": "AP_medium",
            "AP_large": "AP_large",
        }
        reference_deltas = {
            current_key: full[current_key] - reference_area[reference_key]
            for current_key, reference_key in reference_pairs.items()
            if current_key in full and reference_key in reference_area
        }
        parity_available = len(reference_deltas) == len(reference_pairs)
        parity_passed = parity_available and all(
            abs(value) <= REFERENCE_PARITY_TOLERANCE
            for value in reference_deltas.values()
        )
        zoom_gate_passed = bool(delta_ap >= 0.01 and delta_small >= 0.02)
        decisions[model_name] = {
            "full_vs_original_area_report": {
                "reference_path": reference.get("path"),
                "current_minus_reference": reference_deltas,
                "absolute_tolerance": REFERENCE_PARITY_TOLERANCE,
                "available": parity_available,
                "passed": parity_passed,
            },
            "combined_minus_full": {
                "mAP50_95": delta_ap,
                "AP_small": delta_small,
                "mAP50": combined["mAP50"] - full["mAP50"],
                "mAP75": combined["mAP75"] - full["mAP75"],
            },
            "passes_zoom_gate": zoom_gate_passed,
            "decision_is_valid": parity_passed,
            "pursue_adaptive_zoom": bool(parity_passed and zoom_gate_passed),
            "gate": {"mAP50_95_min_gain": 0.01, "AP_small_min_gain": 0.02},
        }
    if "e1" in report["models"] and "e5a" in report["models"]:
        cross = {}
        for mode in ("full", "sliced", "combined"):
            one = report["models"]["e1"]["modes"][mode]["overall"]
            five = report["models"]["e5a"]["modes"][mode]["overall"]
            cross[mode] = {
                key: five[key] - one[key]
                for key in ("mAP50_95", "mAP50", "mAP75", "AP_small", "AR_small")
            }
        decisions["e5a_minus_e1"] = cross
    return decisions


def report_text(report: dict) -> str:
    lines = [
        "E1/E5a fixed-slicing diagnostic",
        f"status: {report.get('status')}",
        f"revision: {SCRIPT_REVISION}",
        f"started_at: {report.get('started_at')}",
        f"finished_at: {report.get('finished_at')}",
        "",
        "All AP/AR values are 0-1. Multiply by 100 for percentages.",
        "This is converted-YOLO-label COCO-style evaluation, not official VisDrone evaluation.",
        "",
    ]
    for model_name, model_row in report.get("models", {}).items():
        lines.append(f"[{model_name}]")
        lines.append(f"weight: {model_row.get('weight')}")
        for mode_name, row in model_row.get("modes", {}).items():
            overall = row.get("overall", {})
            lines.append(
                f"{mode_name}: "
                + ", ".join(f"{key}={overall.get(key)}" for key in (
                    "mAP50_95", "mAP50", "mAP75", "AP_small", "AR_small",
                    "AP_medium", "AP_large",
                ))
            )
        lines.append("")
    lines.append("[decisions]")
    lines.append(json.dumps(report.get("decisions"), ensure_ascii=False, indent=2))
    lines += ["", "[full_metadata]", json.dumps(report, ensure_ascii=False, indent=2, default=json_default)]
    return "\n".join(lines) + "\n"


def save_csv(report: dict) -> None:
    out = Path(report["report_dir"])
    rows = []
    for model_name, model_row in report.get("models", {}).items():
        for mode_name, value in model_row.get("modes", {}).items():
            fixed = value["fixed_operating_point"]["overall"]
            rows.append({
                "model": model_name,
                "mode": mode_name,
                **value["overall"],
                "Precision_conf025_iou05": fixed["Precision"],
                "Recall_conf025_iou05": fixed["Recall"],
                "prediction_count": value["prediction_count"],
                "prediction_seconds": value["profile"].get("seconds"),
                "forward_images": value["profile"].get("forward_images"),
            })
    if rows:
        with (out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    class_rows = []
    for model_name, model_row in report.get("models", {}).items():
        for mode_name, value in model_row.get("modes", {}).items():
            for class_name, metrics in value["per_class"].items():
                fixed = value["fixed_operating_point"]["per_class"][class_name]
                class_rows.append({
                    "model": model_name, "mode": mode_name, "class": class_name,
                    **metrics,
                    "Precision_conf025_iou05": fixed["Precision"],
                    "Recall_conf025_iou05": fixed["Recall"],
                })
    if class_rows:
        with (out / "per_class.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
            writer.writeheader()
            writer.writerows(class_rows)


def save_report(report: dict) -> None:
    report["updated_at"] = now()
    out = Path(report["report_dir"])
    atomic_json(out / "metrics.json", report)
    atomic_text(out / "metrics.txt", report_text(report))


def request_shutdown(report_dir: Path, enabled: bool, reason: str) -> None:
    marker = report_dir / "shutdown_status.json"
    state = {
        "time": now(), "enabled": enabled, "reason": reason,
        "command": ["/bin/bash", "-c", "/usr/bin/shutdown"],
        "status": "disabled" if not enabled else "requesting",
    }
    atomic_json(marker, state)
    if not enabled:
        print("Auto shutdown disabled. Turn off the instance manually if idle.", flush=True)
        return
    if platform.system() != "Linux" or not Path("/root/autodl-tmp").is_dir():
        state["status"] = "refused_non_autodl_path"
        atomic_json(marker, state)
        return
    if not Path("/usr/bin/shutdown").is_file():
        state["status"] = "shutdown_command_missing"
        atomic_json(marker, state)
        return
    if hasattr(os, "sync"):
        os.sync()
    try:
        completed = subprocess.run(
            state["command"], capture_output=True, text=True, timeout=30, check=False
        )
        state.update(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            status="command_returned_unverified" if completed.returncode == 0 else "request_failed",
        )
    except Exception as exc:
        state.update(status="request_exception", error=str(exc))
    atomic_json(marker, state)
    if hasattr(os, "sync"):
        os.sync()


def selected_models(value: str) -> list[str]:
    return ["e1", "e5a"] if value == "both" else [value]


def preflight(report: dict, model_names: list[str], check_only: bool) -> None:
    load_dependencies()
    if ultralytics.__version__ != "8.3.0":
        raise RuntimeError(f"Expected Ultralytics 8.3.0, got {ultralytics.__version__}")
    if torch.__version__ != "2.5.1+cu121":
        raise RuntimeError(f"Expected torch 2.5.1+cu121, got {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run this diagnostic in GPU mode.")
    if not DATA.is_file():
        raise FileNotFoundError(DATA)
    for model_name in model_names:
        if not WEIGHTS[model_name].is_file():
            raise FileNotFoundError(WEIGHTS[model_name])
    report["environment"] = {
        "python": platform.python_version(),
        "ultralytics": ultralytics.__version__,
        "ultralytics_source": ultralytics.__file__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
    }
    report["preflight"] = {"passed": True, "check_only": check_only}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", choices=("e1", "e5a", "both"), default="both")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--smoke-images", type=int, default=0, metavar="N",
        help="Evaluate only the first N images for pipeline testing; metrics are not decision results.",
    )
    parser.add_argument("--shutdown", action="store_true", help="Request AutoDL shutdown on exit.")
    args = parser.parse_args()
    if args.smoke_images < 0 or args.smoke_images > 548:
        parser.error("--smoke-images must be between 0 and 548")

    chosen = selected_models(args.models)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = "CHECK_SLICING" if args.check_only else "SMOKE_SLICING" if args.smoke_images else "DIAG_SLICING"
    out = REPORTS / f"{prefix}_{args.models}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "initializing",
        "purpose": "frozen E1/E5a fixed-slicing upper-bound diagnostic",
        "script_revision": SCRIPT_REVISION,
        "started_at": now(),
        "report_dir": str(out),
        "models_requested": chosen,
        "is_full_diagnostic": not args.check_only and not args.smoke_images,
        "protocol": {
            "data": str(DATA), "split": "val", "imgsz": IMGSZ,
            "prediction_batch": PREDICT_BATCH, "conf": CONF,
            "model_nms_iou": MODEL_NMS_IOU, "merge_nms_iou": MERGE_NMS_IOU,
            "max_det": MAX_DET, "grid": f"{GRID}x{GRID}", "overlap": OVERLAP,
            "edge_margin_fraction": EDGE_MARGIN_FRACTION,
            "fixed_pr_conf": FIXED_PR_CONF, "fixed_pr_iou": FIXED_PR_IOU,
            "reference_parity_absolute_tolerance": REFERENCE_PARITY_TOLERANCE,
            "notes": [
                "Frozen checkpoint evaluation only; no training and no weight updates.",
                "COCO-style metrics use converted ten-class YOLO labels and original-image area ranges.",
                "This is not official VisDrone ignore-region evaluation.",
                "Full, sliced and combined modes are compared inside this evaluator; do not mix with Ultralytics AP.",
                "The 2x2 fixed slicing result is an upper-bound diagnostic, not the proposed adaptive method.",
            ],
        },
        "models": {},
    }
    save_report(report)
    success = False
    started = time.perf_counter()
    try:
        preflight(report, chosen, args.check_only)
        dataset = prepare_dataset(report, args.smoke_images)
        images, path_ids, gt, names = dataset
        for model_name in chosen:
            model = YOLO(str(WEIGHTS[model_name]))
            report["models"][model_name] = {
                "weight": str(WEIGHTS[model_name]),
                "sha256": sha256(WEIGHTS[model_name]),
                "structure": assert_model(model, names, model_name),
                "reference_report": load_reference(model_name),
            }
            del model
            clear_gpu()
        report["status"] = "preflight_passed"
        save_report(report)
        if args.check_only:
            success = True
            report.update(status="check_passed", finished_at=now())
            print("Preflight passed. No prediction, training, or shutdown.", flush=True)
            return

        for model_name in chosen:
            report["status"] = f"predicting_{model_name}"
            save_report(report)
            report["models"][model_name]["modes"] = model_diagnostic(
                model_name, WEIGHTS[model_name], images, path_ids, gt, names, report
            )
            save_report(report)
        report["decisions"] = build_decisions(report) if not args.smoke_images else {
            "disabled": "Smoke-image subset metrics must not drive a research decision."
        }
        report.update(
            status="completed", finished_at=now(), total_seconds=time.perf_counter() - started
        )
        save_csv(report)
        save_report(report)
        shutil.copy2(Path(__file__), out / Path(__file__).name)
        success = True
        print(f"\nCompleted. Report: {out / 'metrics.txt'}", flush=True)
    except BaseException:
        report.update(
            status="failed_or_interrupted", finished_at=now(),
            total_seconds=time.perf_counter() - started,
            error=traceback.format_exc(),
        )
        save_report(report)
        atomic_text(out / "error.log", report["error"])
        print(f"Failure saved: {out}", file=sys.stderr, flush=True)
        raise
    finally:
        try:
            save_report(report)
        except Exception:
            pass
        enabled = bool(args.shutdown and not args.check_only and not args.smoke_images)
        try:
            request_shutdown(out, enabled, report.get("status", "unknown"))
        except Exception:
            if success:
                raise


if __name__ == "__main__":
    main()
