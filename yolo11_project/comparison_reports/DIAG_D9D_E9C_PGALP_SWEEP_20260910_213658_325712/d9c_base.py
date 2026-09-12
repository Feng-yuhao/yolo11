#!/usr/bin/env python3
"""D9c: zero-training causal and routing audit for the E9c AGPGALP probe.

The audit never updates model parameters.  It evaluates the same checkpoint
under controlled inference-time interventions and records where PGALP actually
applies smoothing.  Automatic instance shutdown is deliberately absent.
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
import sys
import time
import traceback
import types
from collections import defaultdict
from datetime import datetime
from pathlib import Path


PROJECT = Path("/root/autodl-tmp/yolo11_project")
SOURCE = Path("/root/autodl-tmp/yolo11_src")
DATA = Path("/root/autodl-tmp/project/VisDrone2019/VisDrone2019.yaml")
WEIGHTS = PROJECT / "runs/e9c_probe_yolo11s_p2_agpgalp_img1024_seed1/weights/best.pt"
E9B_REPORT = (
    PROJECT
    / "runs/e9c_probe_yolo11s_p2_agpgalp_img1024_seed1/complete_metrics.json"
)
CORE_SCRIPT = PROJECT / "yolo11s_p2_img1024_seed1.py"
PGALP_SOURCE = SOURCE / "ultralytics/nn/modules/ag_pgalp.py"

IMGSZ = 1024
BATCH = 4
CONF = 0.001
IOU = 0.7
MAX_DET = 500
HIST_BINS = 200
SCRIPT_REVISION = "d9c_agpgalp_causal_routing_audit_v1"
DEFAULT_MODES = ("normal", "identity", "p3_zero", "p3_shuffle", "force3", "force5", "force7")
VALID_MODES = set(DEFAULT_MODES)


def now():
    return datetime.now().isoformat(timespec="seconds")


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def clear_gpu(torch):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_modes(text):
    modes = tuple(item.strip() for item in text.split(",") if item.strip())
    unknown = sorted(set(modes) - VALID_MODES)
    if unknown:
        raise ValueError(f"Unknown diagnostic modes: {unknown}; valid={sorted(VALID_MODES)}")
    if not modes:
        raise ValueError("At least one diagnostic mode is required")
    if len(set(modes)) != len(modes):
        raise ValueError("Diagnostic modes must not be repeated")
    return modes


def load_dependencies():
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    import numpy as np
    import torch
    import torch.nn.functional as F
    import ultralytics
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from ultralytics import YOLO

    import yolo11s_p2_img1024_seed1 as core

    core.load_dependencies()
    return np, torch, F, ultralytics, COCO, COCOeval, YOLO, core


def find_pgalp(network):
    found = [module for module in network.modules() if module.__class__.__name__ == "AGPGALP"]
    if len(found) != 1:
        raise RuntimeError(f"Expected exactly one AGPGALP module, found {len(found)}")
    module = found[0]
    if tuple(module.kernels) != (3, 5, 7) or tuple(module.sigmas) != (0.8, 1.2, 1.8):
        raise RuntimeError("Checkpoint AGPGALP filter bank differs from E9c")
    if module.p2_channels != 128 or module.p3_channels != 128:
        raise RuntimeError("Checkpoint AGPGALP channels differ from E9c")
    return module


class PGALPController:
    """Patch one loaded PGALP instance without changing its parameters."""

    def __init__(self, module, mode, capture=False):
        self.module = module
        self.mode = mode
        self.capture = bool(capture)
        self.captures = []
        self.original_forward = module.forward
        module.forward = types.MethodType(lambda _, inputs: self.forward(inputs), module)

    def forward(self, inputs):
        torch = sys.modules["torch"]
        p2, p3 = inputs
        if self.mode == "identity":
            return p2

        route_p3 = p3
        if self.mode == "p3_zero":
            route_p3 = torch.zeros_like(p3)
        elif self.mode == "p3_shuffle":
            if p3.shape[0] > 1:
                route_p3 = torch.roll(p3, shifts=1, dims=0)
            else:
                route_p3 = torch.roll(
                    p3,
                    shifts=(max(1, p3.shape[-2] // 2), max(1, p3.shape[-1] // 2)),
                    dims=(-2, -1),
                )

        candidates = [self.module._blur(p2, index) for index in range(len(self.module.kernels))]
        reliability, scale_weights = self.module.routing(p2, route_p3, candidates[0])
        if self.mode.startswith("force"):
            kernel = int(self.mode.replace("force", ""))
            selected = self.module.kernels.index(kernel)
            smooth = candidates[selected]
        else:
            smooth = torch.zeros_like(p2)
            for index, candidate in enumerate(candidates):
                smooth = smooth + scale_weights[:, index : index + 1] * candidate
        output = p2 + self.module.layer_scale * (1.0 - reliability) * (smooth - p2)

        if self.capture:
            with torch.no_grad():
                entropy = -(scale_weights.clamp_min(1e-12) * scale_weights.clamp_min(1e-12).log()).sum(1)
                entropy = entropy / math.log(scale_weights.shape[1])
                self.captures.append(
                    {
                        "reliability": reliability[:, 0].detach().float().cpu(),
                        "weights": scale_weights.detach().float().cpu(),
                        "entropy": entropy.detach().float().cpu(),
                        "change": (output - p2).abs().mean(1).detach().float().cpu(),
                        "feature": p2.abs().mean(1).detach().float().cpu(),
                        "shape": list(p2.shape),
                    }
                )
        return output

    def consume(self, expected_batch):
        matches = [item for item in self.captures if item["shape"][0] == expected_batch]
        self.captures.clear()
        if not matches:
            raise RuntimeError(f"No PGALP capture matched prediction batch {expected_batch}")
        return matches[-1]


class RegionStats:
    def __init__(self, torch):
        self.torch = torch
        self.count = 0
        self.rel_sum = 0.0
        self.rel_sq_sum = 0.0
        self.rel_hist = torch.zeros(HIST_BINS, dtype=torch.float64)
        self.weight_sum = torch.zeros(3, dtype=torch.float64)
        self.winner_count = torch.zeros(3, dtype=torch.int64)
        self.entropy_sum = 0.0
        self.change_sum = 0.0
        self.feature_sum = 0.0

    def update(self, reliability, weights, entropy, change, feature, mask):
        values = reliability[mask]
        count = int(values.numel())
        if not count:
            return
        self.count += count
        values64 = values.double()
        self.rel_sum += float(values64.sum())
        self.rel_sq_sum += float(values64.square().sum())
        self.rel_hist += self.torch.histc(values.float(), bins=HIST_BINS, min=0.0, max=1.0).double()
        selected_weights = weights[:, mask]
        self.weight_sum += selected_weights.double().sum(1)
        winners = selected_weights.argmax(0)
        self.winner_count += self.torch.bincount(winners, minlength=3).cpu()
        self.entropy_sum += float(entropy[mask].double().sum())
        self.change_sum += float(change[mask].double().sum())
        self.feature_sum += float(feature[mask].double().sum())

    def quantile(self, probability):
        if not self.count:
            return None
        threshold = probability * self.count
        cumulative = self.rel_hist.cumsum(0)
        index = int((cumulative >= threshold).nonzero(as_tuple=False)[0])
        return (index + 0.5) / HIST_BINS

    def export(self):
        if not self.count:
            return {"pixels": 0}
        mean = self.rel_sum / self.count
        variance = max(0.0, self.rel_sq_sum / self.count - mean * mean)
        return {
            "pixels": self.count,
            "reliability_mean": mean,
            "reliability_std": math.sqrt(variance),
            "reliability_q05": self.quantile(0.05),
            "reliability_q50": self.quantile(0.50),
            "reliability_q95": self.quantile(0.95),
            "scale_weight_mean_3_5_7": [float(v / self.count) for v in self.weight_sum],
            "scale_winner_fraction_3_5_7": [float(v / self.count) for v in self.winner_count],
            "normalized_scale_entropy_mean": self.entropy_sum / self.count,
            "mean_absolute_feature_change": self.change_sum / self.count,
            "change_to_feature_l1_ratio": self.change_sum / max(self.feature_sum, 1e-12),
        }


class RoutingAudit:
    REGION_NAMES = ("all", "foreground_all", "small", "medium", "large", "near_object", "background")

    def __init__(self, torch, F, annotation_map, image_map):
        self.torch = torch
        self.F = F
        self.annotation_map = annotation_map
        self.image_map = image_map
        self.regions = {name: RegionStats(torch) for name in self.REGION_NAMES}
        self.images = 0

    @staticmethod
    def paint(mask, x1, y1, x2, y2):
        height, width = mask.shape
        left = max(0, min(width - 1, int(math.floor(x1))))
        top = max(0, min(height - 1, int(math.floor(y1))))
        right = max(left + 1, min(width, int(math.ceil(x2))))
        bottom = max(top + 1, min(height, int(math.ceil(y2))))
        mask[top:bottom, left:right] = True

    def masks(self, image_id, feature_height, feature_width):
        info = self.image_map[image_id]
        original_width, original_height = info["width"], info["height"]
        input_height, input_width = feature_height * 4, feature_width * 4
        gain = min(input_height / original_height, input_width / original_width)
        resized_width = round(original_width * gain)
        resized_height = round(original_height * gain)
        pad_x = (input_width - resized_width) / 2.0
        pad_y = (input_height - resized_height) / 2.0
        stride_x = input_width / feature_width
        stride_y = input_height / feature_height

        masks = {
            "foreground_all": self.torch.zeros((feature_height, feature_width), dtype=self.torch.bool),
            "small": self.torch.zeros((feature_height, feature_width), dtype=self.torch.bool),
            "medium": self.torch.zeros((feature_height, feature_width), dtype=self.torch.bool),
            "large": self.torch.zeros((feature_height, feature_width), dtype=self.torch.bool),
        }
        valid = self.torch.zeros((feature_height, feature_width), dtype=self.torch.bool)
        self.paint(
            valid,
            pad_x / stride_x,
            pad_y / stride_y,
            (pad_x + resized_width) / stride_x,
            (pad_y + resized_height) / stride_y,
        )
        for annotation in self.annotation_map[image_id]:
            x, y, width, height = annotation["bbox"]
            x1 = (x * gain + pad_x) / stride_x
            y1 = (y * gain + pad_y) / stride_y
            x2 = ((x + width) * gain + pad_x) / stride_x
            y2 = ((y + height) * gain + pad_y) / stride_y
            area = float(annotation["area"])
            area_name = "small" if area < 32.0**2 else "medium" if area < 96.0**2 else "large"
            self.paint(masks["foreground_all"], x1, y1, x2, y2)
            self.paint(masks[area_name], x1, y1, x2, y2)

        foreground = masks["foreground_all"]
        expanded = self.F.max_pool2d(
            foreground[None, None].float(), kernel_size=9, stride=1, padding=4
        )[0, 0].bool()
        masks["near_object"] = expanded & ~foreground & valid
        masks["background"] = ~expanded & valid
        masks["all"] = valid
        return masks

    def update(self, capture, image_ids):
        reliability = capture["reliability"]
        weights = capture["weights"]
        entropy = capture["entropy"]
        change = capture["change"]
        feature = capture["feature"]
        if reliability.shape[0] != len(image_ids):
            raise RuntimeError("Routing capture/result batch mismatch")
        for index, image_id in enumerate(image_ids):
            height, width = reliability.shape[-2:]
            masks = self.masks(image_id, height, width)
            for name, mask in masks.items():
                self.regions[name].update(
                    reliability[index], weights[index], entropy[index], change[index], feature[index], mask
                )
            self.images += 1

    @staticmethod
    def auc(positive_hist, negative_hist):
        positives = float(positive_hist.sum())
        negatives = float(negative_hist.sum())
        if positives == 0 or negatives == 0:
            return None
        negative_before = negative_hist.cumsum(0) - negative_hist
        favorable = (positive_hist * (negative_before + 0.5 * negative_hist)).sum()
        return float(favorable / (positives * negatives))

    def export(self):
        exported = {name: stats.export() for name, stats in self.regions.items()}
        small = self.regions["small"]
        foreground = self.regions["foreground_all"]
        background = self.regions["background"]
        return {
            "images": self.images,
            "regions": exported,
            "small_vs_background_reliability_auc_approx": self.auc(small.rel_hist, background.rel_hist),
            "all_foreground_vs_background_reliability_auc_approx": self.auc(
                foreground.rel_hist, background.rel_hist
            ),
            "small_minus_background_reliability_mean": (
                exported["small"].get("reliability_mean", 0.0)
                - exported["background"].get("reliability_mean", 0.0)
            ),
            "foreground_minus_background_reliability_mean": (
                exported["foreground_all"].get("reliability_mean", 0.0)
                - exported["background"].get("reliability_mean", 0.0)
            ),
        }


def subset_ground_truth(gt, selected_images):
    ids = {int(item["id"]) for item in selected_images}
    return {
        "info": gt.get("info", {}),
        "licenses": gt.get("licenses", []),
        "images": [item for item in gt["images"] if int(item["id"]) in ids],
        "annotations": [item for item in gt["annotations"] if int(item["image_id"]) in ids],
        "categories": gt["categories"],
    }


def mean_valid(values, np):
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else None


def evaluate_coco(gt, detections, COCO, COCOeval, np):
    gt_coco = COCO()
    gt_coco.dataset = gt
    gt_coco.createIndex()
    if detections:
        dt_coco = gt_coco.loadRes(detections)
    else:
        dt_coco = COCO()
        dt_coco.dataset = {"images": gt["images"], "categories": gt["categories"], "annotations": []}
        dt_coco.createIndex()
    evaluator = COCOeval(gt_coco, dt_coco, "bbox")
    evaluator.params.maxDets = [1, 10, MAX_DET]
    evaluator.evaluate()
    evaluator.accumulate()
    params = evaluator.params
    max_index = params.maxDets.index(MAX_DET)
    result = {}
    for area_name in ("all", "small", "medium", "large"):
        area_index = params.areaRngLbl.index(area_name)
        precision = evaluator.eval["precision"][:, :, :, area_index, max_index]
        recall = evaluator.eval["recall"][:, :, area_index, max_index]
        result[f"AP_{area_name}"] = mean_valid(precision, np)
        result[f"AR_{area_name}"] = mean_valid(recall, np)
    all_index = params.areaRngLbl.index("all")
    for label, threshold in (("AP50", 0.50), ("AP75", 0.75)):
        threshold_index = int(np.abs(params.iouThrs - threshold).argmin())
        values = evaluator.eval["precision"][threshold_index, :, :, all_index, max_index]
        result[label] = mean_valid(values, np)
    result["prediction_count"] = len(detections)
    result["max_det"] = MAX_DET
    return result


def evaluate_mode(
    mode,
    weights_path,
    images,
    path_ids,
    gt,
    names,
    dependencies,
    output,
    collect_routing,
):
    np, torch, F, _, COCO, COCOeval, YOLO, _ = dependencies
    clear_gpu(torch)
    model = YOLO(str(weights_path))
    if model.names != names:
        raise RuntimeError("E9b checkpoint classes do not match VisDrone dataset")
    module = find_pgalp(model.model)
    controller = PGALPController(module, mode=mode, capture=collect_routing)

    annotation_map = defaultdict(list)
    for annotation in gt["annotations"]:
        annotation_map[int(annotation["image_id"])].append(annotation)
    image_map = {int(item["id"]): item for item in gt["images"]}
    audit = RoutingAudit(torch, F, annotation_map, image_map) if collect_routing else None

    detections = []
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for start in range(0, len(images), BATCH):
        chunk = images[start : start + BATCH]
        results = model.predict(
            source=[str(path) for path in chunk],
            imgsz=IMGSZ,
            batch=len(chunk),
            device=0,
            conf=CONF,
            iou=IOU,
            max_det=MAX_DET,
            half=False,
            augment=False,
            verbose=False,
            save=False,
            stream=False,
        )
        result_ids = []
        for result in results:
            image_id = path_ids[str(Path(result.path).resolve())]
            result_ids.append(image_id)
            if result.boxes is not None:
                for x1, y1, x2, y2, score, cls in result.boxes.data.cpu().tolist():
                    detections.append(
                        {
                            "image_id": image_id,
                            "category_id": int(cls) + 1,
                            "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                            "score": score,
                        }
                    )
        if audit is not None:
            audit.update(controller.consume(len(results)), result_ids)
        del results
        done = min(start + BATCH, len(images))
        if start == 0 or done % 50 == 0 or done == len(images):
            print(f"{mode} prediction: {done}/{len(images)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = {
        "allocated_GiB": torch.cuda.max_memory_allocated(0) / 1024**3,
        "reserved_GiB": torch.cuda.max_memory_reserved(0) / 1024**3,
    }
    del model
    clear_gpu(torch)

    print(f"Evaluating {mode} detections on CPU.", flush=True)
    metrics = evaluate_coco(gt, detections, COCO, COCOeval, np)
    metrics.update(seconds=elapsed, peak_memory=peak)
    write_json(output / f"{mode}_metrics.json", metrics)
    return metrics, audit.export() if audit is not None else None


def delta(left, right, keys):
    return {key: finite(left[key] - right[key]) for key in keys if left.get(key) is not None and right.get(key) is not None}


def make_decision(report):
    modes = report["modes"]
    normal = modes.get("normal")
    routing = report.get("routing_audit")
    decisions = []
    if normal and "identity" in modes:
        gain_all = normal["AP_all"] - modes["identity"]["AP_all"]
        gain_small = normal["AP_small"] - modes["identity"]["AP_small"]
        if abs(gain_all) < 0.001 and abs(gain_small) < 0.002:
            decisions.append(
                "Learned-vs-identity effect is weak; PGALP may mainly alter the training trajectory rather than contribute at inference."
            )
        else:
            decisions.append("PGALP has a measurable direct inference-time contribution in its co-adapted E9b checkpoint.")
    if normal and "p3_zero" in modes and "p3_shuffle" in modes:
        p3_effect = max(
            abs(normal["AP_all"] - modes["p3_zero"]["AP_all"]),
            abs(normal["AP_all"] - modes["p3_shuffle"]["AP_all"]),
        )
        if p3_effect < 0.001:
            decisions.append("P3 intervention changes AP_all by <0.1 pp; the router probably underuses P3 guidance.")
        else:
            decisions.append("P3 intervention measurably changes AP_all; P3 guidance is being used.")
    if routing:
        gap = routing["small_minus_background_reliability_mean"]
        entropy = routing["regions"]["all"].get("normalized_scale_entropy_mean")
        small_change = routing["regions"]["small"].get("change_to_feature_l1_ratio")
        background_change = routing["regions"]["background"].get("change_to_feature_l1_ratio")
        if gap < 0.05:
            decisions.append("Small-object reliability exceeds background by <0.05; foreground protection is weak.")
        else:
            decisions.append("Small-object reliability is spatially separated from background.")
        if entropy is not None and entropy > 0.95:
            decisions.append("Scale entropy >0.95; 3/5/7 selection is close to diffuse/uniform.")
        if small_change is not None and background_change is not None:
            if small_change >= 0.8 * background_change:
                decisions.append("Small-object features receive nearly as much smoothing as background; target protection is insufficient.")
            else:
                decisions.append("Small-object features are modified materially less than background.")
    return decisions


def write_text_report(report, path):
    lines = [
        "D9c AGPGALP zero-training diagnostic",
        f"status: {report['status']}",
        f"revision: {SCRIPT_REVISION}",
        f"weights: {report.get('weights', {}).get('path')}",
        f"images: {report['images']}",
        "All AP/AR values are fractions; multiply by 100 for percentage points.",
        "Identity is an inference ablation of the co-adapted E9b checkpoint, not the frozen E1 model.",
        "",
        "Causal mode metrics:",
        "mode, AP_all, AP50, AP75, AP_small, AP_medium, AP_large, predictions",
    ]
    for mode, values in report.get("modes", {}).items():
        lines.append(
            ", ".join(
                str(item)
                for item in (
                    mode,
                    values.get("AP_all"),
                    values.get("AP50"),
                    values.get("AP75"),
                    values.get("AP_small"),
                    values.get("AP_medium"),
                    values.get("AP_large"),
                    values.get("prediction_count"),
                )
            )
        )
    if report.get("routing_audit"):
        routing = report["routing_audit"]
        lines += [
            "",
            "Routing summary:",
            f"small-background reliability gap: {routing['small_minus_background_reliability_mean']}",
            f"foreground-background reliability gap: {routing['foreground_minus_background_reliability_mean']}",
            f"small-vs-background reliability AUROC (histogram approximation): {routing['small_vs_background_reliability_auc_approx']}",
            f"all-foreground-vs-background AUROC: {routing['all_foreground_vs_background_reliability_auc_approx']}",
        ]
        for name, values in routing["regions"].items():
            lines.append(
                f"{name}: R={values.get('reliability_mean')}, Rstd={values.get('reliability_std')}, "
                f"W357={values.get('scale_weight_mean_3_5_7')}, H={values.get('normalized_scale_entropy_mean')}, "
                f"change_ratio={values.get('change_to_feature_l1_ratio')}"
            )
    lines += ["", "Automatic shutdown: disabled", "", "Diagnostic interpretation:"]
    lines.extend(f"- {item}" for item in report.get("decision_signals", []))
    lines += ["", "Full JSON metadata is in metrics.json."]
    atomic_text(path, "\n".join(lines) + "\n")


def write_csv(report, path):
    fields = ["mode", "AP_all", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR_all", "prediction_count", "seconds"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode, values in report.get("modes", {}).items():
            writer.writerow({"mode": mode, **{key: values.get(key) for key in fields[1:]}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--smoke-images", type=int, default=0, help="Use only the first N images to test the diagnostic flow")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    modes = parse_modes(args.modes)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = "CHECK" if args.check_only else "SMOKE" if args.smoke_images else "DIAG"
    output = PROJECT / "comparison_reports" / f"{prefix}_D9B_E9B_PGALP_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "starting",
        "purpose": "D9c zero-training causal and routing audit of E9c AGPGALP",
        "revision": SCRIPT_REVISION,
        "started_at": now(),
        "report_dir": str(output),
        "images": 0,
        "automatic_shutdown": False,
    }
    write_json(output / "metrics.json", report)

    try:
        dependencies = load_dependencies()
        _, torch, _, ultralytics, _, _, YOLO, core = dependencies
        weights_path = args.weights.resolve()
        for required in (weights_path, DATA, CORE_SCRIPT, PGALP_SOURCE):
            if not required.is_file():
                raise FileNotFoundError(required)
        if ultralytics.__version__ != "8.3.0":
            raise RuntimeError(f"Expected Ultralytics 8.3.0, got {ultralytics.__version__}")
        if Path(ultralytics.__file__).resolve().parent != (SOURCE / "ultralytics").resolve():
            raise RuntimeError(f"Ultralytics is not loaded from {SOURCE}")
        if not args.check_only and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the full D9b diagnostic")

        report["environment"] = {
            "python": platform.python_version(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        report["weights"] = {"path": str(weights_path), "sha256": sha256(weights_path)}
        report["protocol"] = {
            "data": str(DATA),
            "imgsz": IMGSZ,
            "prediction_batch": BATCH,
            "conf": CONF,
            "iou": IOU,
            "max_det": MAX_DET,
            "half": False,
            "modes": list(modes),
            "note": "COCO area evaluation on converted YOLO labels; not official VisDrone ignore-region evaluation.",
        }
        shutil.copy2(Path(__file__), output / Path(__file__).name)
        shutil.copy2(PGALP_SOURCE, output / "pgalp.py")

        check_model = YOLO(str(weights_path))
        module = find_pgalp(check_model.model)
        report["learned_state"] = {
            "layer_scale_mean": float(module.layer_scale.detach().float().mean()),
            "layer_scale_std": float(module.layer_scale.detach().float().std()),
            "layer_scale_min": float(module.layer_scale.detach().float().min()),
            "layer_scale_max": float(module.layer_scale.detach().float().max()),
            "router_output_weight_l2": float(module.router[-1].weight.detach().float().norm()),
            "router_output_bias": module.router[-1].bias.detach().float().cpu().tolist(),
        }
        names = check_model.names
        del check_model
        clear_gpu(torch)
        if args.check_only:
            report.update(status="checked", finished_at=now())
            write_json(output / "metrics.json", report)
            write_text_report(report, output / "metrics.txt")
            print(f"D9b check passed. No prediction, no training, no shutdown. Report: {output}", flush=True)
            return

        dataset_stub = {"report_dir": str(output), "dataset": {}}
        images, path_ids, gt, prepared_names, _ = core.prepare_dataset(dataset_stub)
        if prepared_names != names:
            raise RuntimeError("Prepared dataset names differ from checkpoint names")
        report["dataset"] = dataset_stub["dataset"]

        if args.smoke_images:
            if not 1 <= args.smoke_images <= len(images):
                raise ValueError(f"--smoke-images must be in [1,{len(images)}]")
            images = images[: args.smoke_images]
            selected_ids = {path_ids[str(path.resolve())] for path in images}
            gt = subset_ground_truth(gt, [item for item in gt["images"] if int(item["id"]) in selected_ids])
        report["images"] = len(images)
        report["status"] = "evaluating"
        write_json(output / "metrics.json", report)

        mode_metrics = {}
        routing = None
        for mode in modes:
            print(f"\nD9b mode: {mode}", flush=True)
            values, mode_routing = evaluate_mode(
                mode,
                weights_path,
                images,
                path_ids,
                gt,
                names,
                dependencies,
                output,
                collect_routing=(mode == "normal"),
            )
            mode_metrics[mode] = values
            if mode_routing is not None:
                routing = mode_routing
            report["modes"] = mode_metrics
            if routing is not None:
                report["routing_audit"] = routing
            write_json(output / "metrics.json", report)

        keys = ("AP_all", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR_all")
        if "normal" in mode_metrics:
            report["deltas_from_normal"] = {
                mode: delta(values, mode_metrics["normal"], keys)
                for mode, values in mode_metrics.items()
                if mode != "normal"
            }
        if E9B_REPORT.is_file() and "normal" in mode_metrics and not args.smoke_images:
            frozen = json.loads(E9B_REPORT.read_text(encoding="utf-8"))
            reference = frozen["area_metrics"]
            report["normal_reproduction_vs_frozen_e9c"] = {
                key: mode_metrics["normal"].get(key) - reference.get(key)
                for key in ("AP_all", "AP_small", "AP_medium", "AP_large")
            }
        report["decision_signals"] = make_decision(report)
        report.update(status="completed", finished_at=now())
        write_json(output / "metrics.json", report)
        write_text_report(report, output / "metrics.txt")
        write_csv(report, output / "summary.csv")
        print(f"\nD9b completed. Report: {output / 'metrics.txt'}", flush=True)
        print("No training was performed. Automatic shutdown is disabled.", flush=True)
    except BaseException as error:
        report.update(status="failed", finished_at=now(), error=repr(error), traceback=traceback.format_exc())
        write_json(output / "metrics.json", report)
        write_text_report(report, output / "metrics.txt")
        print(f"D9b failure saved: {output}", flush=True)
        print("Automatic shutdown is disabled.", flush=True)
        raise


if __name__ == "__main__":
    main()

