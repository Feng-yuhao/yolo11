#!/usr/bin/env python3
"""E32a: E18a inference graph with Scale-Consistent Auxiliary Supervision.

Modes:
  --check-only              CPU structure/loss/TAL checks + static train audit
  --smoke2                  standard two-epoch CUDA smoke
  --probe90                 200-epoch schedule, graceful stop after epoch 90
  --resume200 CHECKPOINT    resume the epoch-90 optimizer checkpoint to epoch 200
  --full200                 full 200 epochs from the same initialization
  --eval-only BEST_PT       standard frozen evaluation

Automatic shutdown is disabled in every mode.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = Path("/root/autodl-tmp/yolo11_project")
SOURCE = Path("/root/autodl-tmp/yolo11_src")
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _locate_e18_runner() -> Path:
    direct = ROOT / "yolo11s_p2_dssg_fusion_ready_img1024_seed1.py"
    if direct.is_file():
        return direct
    candidates = sorted(
        ROOT.glob(
            "comparison_reports/e18a_yolo11s_p2_dssg_fusion_ready_img1024_seed1_*/"
            "yolo11s_p2_dssg_fusion_ready_img1024_seed1.py"
        )
    )
    if not candidates:
        raise FileNotFoundError("Cannot find the audited E18a runner")
    return candidates[-1]


E18_RUNNER = _locate_e18_runner()
if str(E18_RUNNER.parent) not in sys.path:
    sys.path.insert(0, str(E18_RUNNER.parent))
_spec = importlib.util.spec_from_file_location("_e32_parent_e18", E18_RUNNER)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot import E18a runner: {E18_RUNNER}")
e18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e18)
core = e18.core

EXPERIMENT = "e32a_yolo11s_p2_dssg_fusion_ready_scas_img1024_seed1"
REVISION = "e32a_scas_unique_task_owner_all_gt_ignore_level_balanced_probe90_gatefix_v2"
MODEL_YAML = HERE / "yolo11s-p2-e32a-scas.yaml"
LOSS_SOURCE = HERE / "e32_scale_consistent_loss.py"
SMOKE_GATE = ROOT / "comparison_reports/e32a_scas_smoke_passed.json"
D27_SCRIPT = ROOT / "diagnose_d27_official_visdrone.py"
VAL_ROOT = Path("/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val")
PROBE_STOP_EPOCH = 90
FORMAL_EPOCHS = 200

_CORE_EVALUATE = core.evaluate


def _expected_e32_config():
    return {
        "enabled": True,
        "classification_owner": "p4_if_valid_else_p3",
        "regression_owner": "coarsest_valid_else_none",
        "negative_ignore": "all_gt_both_levels_3x3",
        "classification_collision": "unique_image_class_cell",
        "regression_collision": "largest_box_per_image_cell",
        "level_aggregation": "equal_level_mean",
    }


def validate_p2_config(cfg, original_cfg):
    e18.validate_p2_config(cfg, original_cfg)
    if cfg.get("e32_aux") != _expected_e32_config():
        raise ValueError(f"E32 configuration differs from preregistration: {cfg.get('e32_aux')!r}")
    expected_e17 = {
        "enabled": True,
        "gain": 0.1,
        "min_feature_side": 1.5,
        "negative_weight": 0.1,
        "box_weight": 0.25,
    }
    if cfg.get("e17_aux") != expected_e17:
        raise ValueError("E32 must retain every E18 auxiliary scalar unchanged")


@contextlib.contextmanager
def _temporary_e32_criterion():
    from e32_scale_consistent_loss import v8E32DetectionLoss
    from ultralytics.nn.tasks import DetectionModel

    original = DetectionModel.init_criterion

    def patched(model):
        cfg = model.yaml.get("e32_aux", {})
        if isinstance(cfg, dict) and cfg.get("enabled", False):
            return v8E32DetectionLoss(model)
        return original(model)

    DetectionModel.init_criterion = patched
    try:
        yield
    finally:
        DetectionModel.init_criterion = original


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def _dataset_train_images() -> Path:
    from ultralytics.utils import yaml_load

    data = yaml_load(core.DATA)
    base = Path(data.get("path", Path(core.DATA).parent))
    if not base.is_absolute():
        base = Path(core.DATA).parent / base
    train = Path(data["train"])
    return train if train.is_absolute() else base / train


def _mask_cell_count(points: list[tuple[int, int]], height: int, width: int) -> int:
    cells: set[int] = set()
    for x, y in points:
        for dy in (-1, 0, 1):
            yy = min(height - 1, max(0, y + dy))
            for dx in (-1, 0, 1):
                xx = min(width - 1, max(0, x + dx))
                cells.add(yy * width + xx)
    return len(cells)


def static_train_routing_audit() -> dict:
    """Scan unaugmented train labels; online augmented stats are logged separately."""
    from PIL import Image

    images_dir = _dataset_train_images().resolve()
    labels_dir = images_dir.parent / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Training images/labels unavailable: {images_dir} / {labels_dir}")
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if len(images) != 6471:
        raise RuntimeError(f"Expected 6471 train images, found {len(images)}")

    counts = {
        "images": len(images),
        "gt": 0,
        "duplicate_labels_removed": 0,
        "p3_valid": 0,
        "p4_valid": 0,
        "old_duplicate_positive": 0,
        "classification_p3": 0,
        "classification_p4": 0,
        "regression_p3": 0,
        "regression_p4": 0,
        "regression_none": 0,
        "P3_current_ignored_cells": 0,
        "P3_all_gt_ignored_cells": 0,
        "P3_total_cells": 0,
        "P4_current_ignored_cells": 0,
        "P4_all_gt_ignored_cells": 0,
        "P4_total_cells": 0,
        "P3_classification_collisions": 0,
        "P4_classification_collisions": 0,
        "P3_regression_collisions": 0,
        "P4_regression_collisions": 0,
    }
    grids = {"P3": (128, 128), "P4": (64, 64)}

    for index, image_path in enumerate(images, 1):
        label_path = labels_dir / (image_path.stem + ".txt")
        with Image.open(image_path) as image:
            width, height = image.size
        ratio = min(core.IMGSZ / height, core.IMGSZ / width)
        resized_w, resized_h = round(width * ratio), round(height * ratio)
        pad_x, pad_y = (core.IMGSZ - resized_w) / 2.0, (core.IMGSZ - resized_h) / 2.0

        rows: list[tuple[int, float, float, float, float]] = []
        seen = set()
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                values = line.split()
                if len(values) != 5:
                    raise ValueError(f"Invalid YOLO label row: {label_path}: {line!r}")
                cls = int(float(values[0]))
                x, y, w, h = (float(value) for value in values[1:])
                row = (cls, x, y, w, h)
                if row in seen:
                    counts["duplicate_labels_removed"] += 1
                    continue
                seen.add(row)
                if not (0 <= cls < 10 and w > 0 and h > 0):
                    raise ValueError(f"Invalid YOLO label values: {label_path}: {row}")
                rows.append(row)

        level_points = {"P3": [], "P4": []}
        current_points = {"P3": [], "P4": []}
        cls_keys = {"P3": [], "P4": []}
        reg_keys = {"P3": [], "P4": []}
        for cls, xn, yn, wn, hn in rows:
            bx = (xn * width * ratio + pad_x) / core.IMGSZ
            by = (yn * height * ratio + pad_y) / core.IMGSZ
            bw = wn * width * ratio / core.IMGSZ
            bh = hn * height * ratio / core.IMGSZ
            valid = {}
            cells = {}
            for level, (gh, gw) in grids.items():
                cx = min(gw - 1, max(0, math.floor(bx * gw)))
                cy = min(gh - 1, max(0, math.floor(by * gh)))
                cells[level] = (cx, cy)
                level_points[level].append((cx, cy))
                valid[level] = min(bw * gw, bh * gh) >= 1.5
                if valid[level]:
                    current_points[level].append((cx, cy))

            p3_valid, p4_valid = valid["P3"], valid["P4"]
            if p4_valid and not p3_valid:
                raise RuntimeError("Static audit found impossible P4-valid/P3-invalid target")
            cls_level = "P4" if p4_valid else "P3"
            reg_level = "P4" if p4_valid else "P3" if p3_valid else None
            cx, cy = cells[cls_level]
            cls_keys[cls_level].append((cls, cy, cx))
            if reg_level:
                cx, cy = cells[reg_level]
                reg_keys[reg_level].append((cy, cx))

            counts["gt"] += 1
            counts["p3_valid"] += int(p3_valid)
            counts["p4_valid"] += int(p4_valid)
            counts["old_duplicate_positive"] += int(p4_valid)
            counts[f"classification_{cls_level.lower()}"] += 1
            if reg_level:
                counts[f"regression_{reg_level.lower()}"] += 1
            else:
                counts["regression_none"] += 1

        for level, (gh, gw) in grids.items():
            counts[f"{level}_current_ignored_cells"] += _mask_cell_count(current_points[level], gh, gw)
            counts[f"{level}_all_gt_ignored_cells"] += _mask_cell_count(level_points[level], gh, gw)
            counts[f"{level}_total_cells"] += gh * gw
            counts[f"{level}_classification_collisions"] += len(cls_keys[level]) - len(set(cls_keys[level]))
            counts[f"{level}_regression_collisions"] += len(reg_keys[level]) - len(set(reg_keys[level]))
        if index % 1000 == 0:
            print(f"E32 static train audit: {index}/{len(images)}", flush=True)

    total = max(1, counts["gt"])
    reg_owned = max(1, counts["regression_p3"] + counts["regression_p4"])
    result = dict(counts)
    result.update(
        classification_p3_fraction=counts["classification_p3"] / total,
        classification_p4_fraction=counts["classification_p4"] / total,
        regression_p3_fraction_of_owned=counts["regression_p3"] / reg_owned,
        regression_p4_fraction_of_owned=counts["regression_p4"] / reg_owned,
        regression_none_fraction=counts["regression_none"] / total,
        old_duplicate_positive_fraction=counts["old_duplicate_positive"] / total,
        note="Static 1024 letterbox estimate; epoch-1 online augmented routing is recorded separately.",
    )
    for level in ("P3", "P4"):
        cells = max(1, counts[f"{level}_total_cells"])
        result[f"{level}_current_ignore_fraction"] = counts[f"{level}_current_ignored_cells"] / cells
        result[f"{level}_all_gt_ignore_fraction"] = counts[f"{level}_all_gt_ignored_cells"] / cells
    if result["classification_p3_fraction"] < 0.10 or result["classification_p4_fraction"] < 0.10:
        raise RuntimeError(f"Static classification routing collapses one level: {result}")
    if result["regression_p3_fraction_of_owned"] < 0.10 or result["regression_p4_fraction_of_owned"] < 0.10:
        raise RuntimeError(f"Static regression routing collapses one level: {result}")
    return result


def _synthetic_batch(torch):
    return {
        "img": torch.rand(2, 3, 128, 128),
        "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
        "cls": torch.tensor([[1.0], [3.0], [8.0]]),
        # For synthetic auxiliary grids P3=16, P4=8:
        # row0: no regression owner; row1: P3; row2: P4.
        "bboxes": torch.tensor(
            [[0.20, 0.20, 0.05, 0.05], [0.62, 0.62, 0.10, 0.10], [0.45, 0.45, 0.25, 0.25]],
            dtype=torch.float32,
        ),
    }


def _gradient_map(model):
    return {name: None if p.grad is None else p.grad.detach().clone() for name, p in model.named_parameters()}


def _assert_gradient_equal(reference, candidate):
    import torch

    if reference.keys() != candidate.keys():
        raise AssertionError("Gradient parameter names differ")
    maximum = 0.0
    for name in reference:
        left, right = reference[name], candidate[name]
        if (left is None) != (right is None):
            raise AssertionError(f"AUX-OFF gradient presence differs at {name}")
        if left is not None:
            delta = float((left - right).abs().max())
            maximum = max(maximum, delta)
            if not torch.equal(left, right):
                raise AssertionError(f"AUX-OFF gradient differs at {name}: {delta}")
    return maximum


def _e32_loss_preflight(source_model):
    import torch
    from e32_scale_consistent_loss import v8E32DetectionLoss
    from ultralytics.utils.loss import v8DetectionLoss

    args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    batch = _synthetic_batch(torch)
    probe = copy.deepcopy(source_model).cpu().train()
    probe.args = args
    criterion = v8E32DetectionLoss(probe)
    logits = (torch.randn(2, 14, 16, 16, requires_grad=True), torch.randn(2, 14, 8, 8, requires_grad=True))
    auxiliary, stats = criterion.auxiliary_loss(logits, batch)
    expected = {
        "classification_p3": 2,
        "classification_p4": 1,
        "regression_p3": 1,
        "regression_p4": 1,
        "regression_none": 1,
    }
    for key, value in expected.items():
        if stats[key] != value:
            raise AssertionError(f"Synthetic E32 routing mismatch for {key}: {stats[key]} != {value}")
    auxiliary.backward()
    if not all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in logits):
        raise AssertionError("E32 auxiliary loss did not backpropagate finite P3/P4 gradients")

    # With gain forced to zero, predictions, main loss and every gradient must
    # be exactly the base E18/main-YOLO result.
    reference = copy.deepcopy(source_model).cpu().train()
    candidate = copy.deepcopy(source_model).cpu().train()
    reference.args = args
    candidate.args = args
    base = v8DetectionLoss(reference)
    e32 = v8E32DetectionLoss(candidate)
    e32.gain = 0.0
    reference.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    pred_ref = reference(batch["img"])
    pred_e32 = candidate(batch["img"])
    if len(pred_ref) != len(pred_e32) or any(not torch.equal(a, b) for a, b in zip(pred_ref, pred_e32)):
        raise AssertionError("E32 inference tensors differ from E18 at identical initialization")
    base_total, base_items = base(pred_ref, batch)
    e32_total, e32_items = e32(pred_e32, batch)
    if not torch.equal(base_total, e32_total) or not torch.equal(base_items, e32_items):
        raise AssertionError("E32 aux-gain-zero loss differs from the unchanged main loss")
    base_total.backward()
    e32_total.backward()
    gradient_delta = _assert_gradient_equal(_gradient_map(reference), _gradient_map(candidate))
    return {
        "synthetic_routing": stats,
        "auxiliary_loss": float(auxiliary.detach()),
        "aux_gain_zero_total_equal": True,
        "aux_gain_zero_items_equal": True,
        "aux_gain_zero_gradient_max_abs_difference": gradient_delta,
    }


def p2_preflight(report, eval_only=False):
    old_yaml = e18.MODEL_YAML
    e18.MODEL_YAML = MODEL_YAML
    try:
        with _temporary_e32_criterion():
            e18.p2_preflight(report, eval_only=eval_only)
    finally:
        e18.MODEL_YAML = old_yaml

    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils.torch_utils import init_seeds

    cfg = yaml_model_load(str(MODEL_YAML))
    validate_p2_config(cfg, None)
    init_seeds(core.SEED, deterministic=True)
    model = DetectionModel(cfg, nc=10, verbose=False)
    e18.assert_p2_model(model)
    transfer = e18.load_p2_pretrained(model)
    loss_checks = _e32_loss_preflight(model)
    static_audit = static_train_routing_audit()
    report["p2_script_revision"] = REVISION
    report["e32_scas"] = {
        "config": _expected_e32_config(),
        "loss_source": str(LOSS_SOURCE),
        "loss_sha256": _sha256(LOSS_SOURCE),
        "parent_e18_runner": str(E18_RUNNER),
        "initialization": transfer,
        "inference_topology_identical_to_e18": True,
        "new_inference_parameters": 0,
        "new_inference_flops": 0,
        "level_aggregation": "0.5 * P3 + 0.5 * P4 (unchanged from E18)",
        "loss_preflight": loss_checks,
        "static_train_routing_audit": static_audit,
        "scientific_boundary": (
            "E32 changes three coherent auxiliary-target policies together: unique task-specific ownership, "
            "all-GT non-owner negative ignore, and unrepresentable same-cell target coalescing. It does not "
            "prove that any one component alone causes a metric change."
        ),
    }
    _json(Path(report["report_dir"]) / "e32_static_train_routing_audit.json", static_audit)
    shutil.copy2(LOSS_SOURCE, Path(report["report_dir"]) / LOSS_SOURCE.name)
    core.save_report(report)
    del model
    gc.collect()
    print("E32a SCAS structure, exact-main-loss, routing, train-audit and TAL checks passed.", flush=True)


def _archive_online_stats(report):
    from e32_scale_consistent_loss import LATEST_E32_ONLINE_STATS

    stats = copy.deepcopy(LATEST_E32_ONLINE_STATS)
    if stats:
        report.setdefault("e32_scas", {})["online_augmented_routing"] = stats
        _json(Path(report["report_dir"]) / "e32_online_augmented_routing.json", stats)
        core.save_report(report)


def train(report, smoke=False):
    with _temporary_e32_criterion():
        weights = e18.train(report, smoke=smoke)
    _archive_online_stats(report)
    run_dir = Path(report["run_dir"])
    shutil.copy2(LOSS_SOURCE, run_dir / LOSS_SOURCE.name)
    return weights


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _e18_official_reference():
    candidates = sorted(ROOT.glob("comparison_reports/DIAG_D30_EXISTING_OFFICIAL_SWEEP_*/metrics.json"))
    for path in reversed(candidates):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            row = data["models"]["e18a"]["official"]
            return row, str(path)
        except Exception:
            continue
    raise FileNotFoundError("D30 E18a official reference metrics are required")


def _official_evaluation(report):
    if not D27_SCRIPT.is_file() or not VAL_ROOT.is_dir():
        raise FileNotFoundError("D27 evaluator or original VisDrone val annotations are missing")
    d27 = _load_module(D27_SCRIPT, "_e32_d27")
    records, dataset_summary = d27.load_dataset(VAL_ROOT)
    prediction_path = Path(report["report_dir"]) / "area_predictions.json"
    raw = d27.load_predictions(prediction_path, len(records))
    filtered, ignore_audit = d27.filter_detections_and_audit(records, raw)
    official = d27.official_evaluate(records, filtered)
    reference, reference_path = _e18_official_reference()
    keys = ("AP50_95_percent", "AP50_percent", "AP75_percent", "AR500_percent")
    deltas = {key: float(official[key]) - float(reference[key]) for key in keys}
    result = {
        "evaluator_revision": getattr(d27, "REVISION", "unknown"),
        "dataset": dataset_summary,
        "ignore_audit": ignore_audit,
        "official": official,
        "E18a_reference": reference,
        "E18a_reference_path": reference_path,
        "delta_vs_E18a_pp": deltas,
    }
    report["official_visdrone"] = result
    _json(Path(report["report_dir"]) / "e32_official_visdrone.json", result)
    return result


def _read_map_curve(path: Path, limit: int | None = None) -> list[float]:
    values = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            clean = {key.strip(): value.strip() for key, value in row.items() if key is not None}
            values.append(float(clean["metrics/mAP50-95(B)"]))
    return values[:limit] if limit else values


def _probe_gate(report):
    official = report["official_visdrone"]
    delta = official["delta_vs_E18a_pp"]
    run_curve = _read_map_curve(Path(report["report_dir"]) / "results.csv")
    e18_results = sorted(
        ROOT.glob("comparison_reports/e18a_yolo11s_p2_dssg_fusion_ready_img1024_seed1_*/results.csv")
    )[-1]
    parent_curve = _read_map_curve(e18_results, PROBE_STOP_EPOCH)
    if len(run_curve) != PROBE_STOP_EPOCH or len(parent_curve) < PROBE_STOP_EPOCH:
        raise RuntimeError("E32/E18 90-epoch curves are incomplete")
    last10_delta_pp = 100.0 * (sum(run_curve[-10:]) / 10.0 - sum(parent_curve[-10:]) / 10.0)
    best_epoch = int(max(range(len(run_curve)), key=run_curve.__getitem__) + 1)
    ap_delta = float(delta["AP50_95_percent"])
    ap50_ok = float(delta["AP50_percent"]) >= 0.0
    ap75_ok = float(delta["AP75_percent"]) >= 0.0
    if ap_delta >= 0.25 and ap50_ok and ap75_ok:
        decision = "CONTINUE_E32_90_TO_200"
    elif ap_delta <= 0.10 or not (ap50_ok and ap75_ok):
        decision = "CLOSE_E32_KEEP_E18"
    elif last10_delta_pp >= 0.10 and best_epoch >= 75:
        decision = "CONTINUE_E32_90_TO_200_BORDERLINE"
    else:
        decision = "CLOSE_E32_KEEP_E18"
    gate = {
        "decision": decision,
        "official_delta_vs_E18a_pp": delta,
        "epoch81_90_ultralytics_map_delta_pp": last10_delta_pp,
        "best_epoch_by_ultralytics_map": best_epoch,
        "continue_rule": "Official AP >= +0.25pp with AP50/AP75 non-decreasing; borderline also needs last10 >= +0.10pp and best epoch >=75.",
    }
    report["e32_probe90_gate"] = gate
    _json(Path(report["report_dir"]) / "e32_probe90_gate.json", gate)
    return gate


def _final_gate(report):
    official = report["official_visdrone"]
    delta = official["delta_vs_E18a_pp"]
    references = sorted(
        ROOT.glob("comparison_reports/e18a_yolo11s_p2_dssg_fusion_ready_img1024_seed1_*/metrics.json")
    )
    if not references:
        raise FileNotFoundError("E18a formal custom-area metrics are missing")
    parent = json.loads(references[-1].read_text(encoding="utf-8"))
    area_delta_pp = {
        name: 100.0 * (
            float(report["area_metrics"][name]) - float(parent["area_metrics"][name])
        )
        for name in ("AP_small", "AP_medium", "AP_large")
    }
    passed = (
        float(delta["AP50_95_percent"]) >= 0.25
        and float(delta["AP50_percent"]) >= 0.0
        and float(delta["AP75_percent"]) >= 0.0
        and min(area_delta_pp.values()) >= -0.50
    )
    gate = {
        "passed": bool(passed),
        "decision": "RETAIN_E32_START_MULTI_SEED" if passed else "CLOSE_E32_KEEP_E18",
        "official_delta_vs_E18a_pp": delta,
        "custom_area_delta_vs_E18a_pp": area_delta_pp,
        "requirements": {
            "Official_AP_pp": ">= +0.25",
            "Official_AP50_pp": ">= 0",
            "Official_AP75_pp": ">= 0",
            "each_custom_area_AP_pp": ">= -0.50",
        },
    }
    report["e32_final200_gate"] = gate
    _json(Path(report["report_dir"]) / "e32_final200_gate.json", gate)
    return gate


def evaluate(report, weights, dataset):
    _CORE_EVALUATE(report, weights, dataset)
    if report.get("purpose", "").startswith("SMOKE2"):
        report["official_visdrone"] = {"status": "skipped_for_smoke"}
    else:
        _official_evaluation(report)
        completed = int((report.get("training") or {}).get("completed_epochs") or 0)
        if report.get("is_formal") and completed >= FORMAL_EPOCHS:
            _final_gate(report)
    core.save_report(report)


def _base_report(prefix: str, is_formal: bool):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = ROOT / "comparison_reports" / f"{prefix}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "initializing",
        "purpose": prefix,
        "experiment": EXPERIMENT,
        "seed": core.SEED,
        "seed_count": 1,
        "multi_seed_std": None,
        "started_at": core.now(),
        "report_dir": str(out),
        "is_formal": is_formal,
        "evaluation_protocol": {
            "split": "val",
            "imgsz": core.IMGSZ,
            "batch": core.BATCH,
            "area_batch": core.AREA_BATCH,
            "conf": core.CONF,
            "iou": core.IOU,
            "max_det": core.MAX_DET,
            "half": False,
            # This block must remain byte-for-byte equivalent (after JSON
            # serialization) to the frozen core report used by --smoke2.
            # D27 official-ignore results are stored separately under
            # report["official_visdrone"].  Mixing the D27 description into
            # this field made the valid smoke signature impossible to reuse
            # for --probe90.
            "protocol_id": "visdrone_yolo_labels_ultra830_fp32_area_separate_predict_v1",
            "area_definition": "COCO default original-image bbox area ranges, IoU 0.50:0.95",
            "notes": [
                "COCO area evaluation on converted VisDrone labels, not official VisDrone ignore regions.",
                "Ultralytics AP and COCO AP_all are separate evaluators/prediction passes; do not mix them.",
                "Batch val may use rectangular padding; area predict uses each 4-image chunk padding.",
                "P/R are Ultralytics selected F1 operating point metrics, not P/R at conf=0.001.",
                "Training built-in validation may use FP16 and double batch; final report is standalone FP32.",
            ],
        },
    }
    core.save_report(report)
    return report, out


def _copy_training_artifacts(report, run_dir: Path):
    out = Path(report["report_dir"])
    for filename in ("results.csv", "args.yaml"):
        source = run_dir / filename
        if source.is_file():
            shutil.copy2(source, out / filename)
    shutil.copy2(LOSS_SOURCE, run_dir / LOSS_SOURCE.name)
    shutil.copy2(LOSS_SOURCE, out / LOSS_SOURCE.name)


def train_probe90(report):
    import torch
    from p2_tal_chunked import chunked_assignment

    params = core.training_params()
    params.update(epochs=FORMAL_EPOCHS, name=EXPERIMENT + "_probe90")
    report["train_params"] = copy.deepcopy(params)
    report["training"] = {
        "completed_epochs": 0,
        "formal_schedule_epochs": FORMAL_EPOCHS,
        "planned_probe_stop_epoch": PROBE_STOP_EPOCH,
        "best_checkpoint_epoch": None,
        "best_epoch_by_map50_95": None,
        "memory_note": "200-epoch scheduler; optimizer-bearing snapshot copied before final_eval at epoch 90.",
    }
    best_map = -math.inf
    previous_epoch = 0
    started = time.perf_counter()
    core.clear_gpu()
    torch.cuda.reset_peak_memory_stats(0)
    model = core.YOLO(str(MODEL_YAML), task="detect")

    def setup(trainer):
        report["run_dir"] = str(Path(trainer.save_dir).resolve())
        report["training"]["effective_amp"] = bool(trainer.amp)
        report["training"]["parameters_before_fusion"] = sum(p.numel() for p in trainer.model.parameters())
        shutil.copy2(Path(trainer.save_dir) / "args.yaml", Path(report["report_dir"]) / "train_args.yaml")
        core.save_report(report)

    def saved(trainer):
        if trainer.fitness == trainer.best_fitness:
            report["training"]["best_checkpoint_epoch"] = int(trainer.epoch) + 1
        epoch = int(trainer.epoch) + 1
        if epoch == PROBE_STOP_EPOCH:
            checkpoint = Path(trainer.wdir) / f"probe_resume_epoch{PROBE_STOP_EPOCH:03d}.pt"
            shutil.copy2(Path(trainer.last), checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            optimizer_present = payload.get("optimizer") is not None
            if int(payload.get("epoch", -1)) != PROBE_STOP_EPOCH - 1 or not optimizer_present:
                raise RuntimeError("Probe resume snapshot lacks epoch/optimizer state")
            report["training"].update(
                probe_resume_checkpoint=str(checkpoint),
                probe_resume_checkpoint_sha256=_sha256(checkpoint),
                probe_resume_optimizer_present=True,
                probe_snapshot_taken_before_final_eval=True,
            )
            core.save_report(report)
            trainer.stop = True

    def epoch_end(trainer):
        nonlocal best_map, previous_epoch
        epoch = int(trainer.epoch) + 1
        if epoch <= previous_epoch:
            return
        previous_epoch = epoch
        score = core.value_or_none(trainer.metrics.get("metrics/mAP50-95(B)", float("nan")))
        if score is not None and score > best_map:
            best_map = score
            report["training"]["best_epoch_by_map50_95"] = epoch
        report["training"].update(
            completed_epochs=epoch,
            elapsed_seconds=time.perf_counter() - started,
            peak_memory=core.memory_peak(),
        )
        criterion = getattr(trainer.model, "criterion", None)
        if criterion is not None and hasattr(criterion, "stats_snapshot"):
            report.setdefault("e32_scas", {})["online_augmented_routing"] = criterion.stats_snapshot()
        core.save_report(report)

    model.add_callback("on_train_start", setup)
    model.add_callback("on_model_save", saved)
    model.add_callback("on_fit_epoch_end", epoch_end)
    report["status"] = "training_probe90"
    core.save_report(report)
    matching_stats = {}
    try:
        with _temporary_e32_criterion(), chunked_assignment() as matching_stats:
            model.train(trainer=core.make_p2_trainer(report), **params)
        torch.cuda.synchronize()
        run_dir = Path(model.trainer.save_dir).resolve()
        report["run_dir"] = str(run_dir)
        report["training"]["seconds_including_builtin_final_validation"] = time.perf_counter() - started
        if report["training"]["completed_epochs"] != PROBE_STOP_EPOCH:
            raise RuntimeError(f"Probe stopped at {report['training']['completed_epochs']}, expected {PROBE_STOP_EPOCH}")
        weights = run_dir / "weights/best.pt"
        if not weights.is_file():
            raise FileNotFoundError(weights)
        _copy_training_artifacts(report, run_dir)
        report["status"] = "probe90_training_completed"
        _archive_online_stats(report)
        core.save_report(report)
        return weights
    finally:
        report["training"]["matching_execution"] = dict(matching_stats)
        report["training"]["elapsed_seconds"] = time.perf_counter() - started
        report["training"]["peak_memory"] = core.memory_peak()
        core.save_report(report)
        del model
        core.clear_gpu()


def _run_probe90():
    report, out = _base_report("PROBE90_" + EXPERIMENT, is_formal=False)
    started = time.perf_counter()
    try:
        dataset = core.preflight(report, core.PRETRAINED, require_gpu=True)
        p2_preflight(report, eval_only=False)
        core.verify_smoke_gate(report)
        weights = train_probe90(report)
        evaluate(report, weights, dataset)
        core.export_csv(report)
        gate = _probe_gate(report)
        report.update(status="completed_probe", finished_at=core.now(), total_seconds=time.perf_counter() - started)
        core.save_report(report)
        run_dir = Path(report["run_dir"])
        for name in ("metrics.txt", "metrics.json", "per_class.csv", "seed_metrics.csv"):
            if (out / name).is_file():
                shutil.copy2(out / name, run_dir / ("probe90_" + name))
        print(f"\nE32a 90-epoch probe completed. Decision: {gate['decision']}", flush=True)
        print(f"Report: {out / 'metrics.txt'}", flush=True)
        print(f"Resume checkpoint: {report['training']['probe_resume_checkpoint']}", flush=True)
    except BaseException:
        report.update(status="failed_or_interrupted", finished_at=core.now(), error=traceback.format_exc())
        core.save_report(report)
        core.atomic_text(out / "error.log", report["error"])
        raise
    finally:
        print("Automatic shutdown disabled; turn off the instance manually if idle.", flush=True)


def _verify_resume_checkpoint(path: Path):
    import torch

    path = path.resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("epoch", -1)) != PROBE_STOP_EPOCH - 1:
        raise ValueError(f"Expected epoch-90 snapshot (stored epoch 89), got {payload.get('epoch')}")
    if payload.get("optimizer") is None:
        raise ValueError("Resume checkpoint optimizer was stripped")
    args = payload.get("train_args") or {}
    if int(args.get("epochs", -1)) != FORMAL_EPOCHS:
        raise ValueError(f"Resume checkpoint was not created with a 200-epoch schedule: {args.get('epochs')}")
    return path


def _run_resume200(checkpoint: Path):
    import torch
    from p2_tal_chunked import chunked_assignment

    checkpoint = _verify_resume_checkpoint(checkpoint)
    report, out = _base_report("RESUME200_" + EXPERIMENT, is_formal=True)
    started = time.perf_counter()
    try:
        dataset = core.preflight(report, checkpoint, require_gpu=True)
        p2_preflight(report, eval_only=True)
        report["training"] = {
            "resume_checkpoint": str(checkpoint),
            "resume_checkpoint_sha256": _sha256(checkpoint),
            "start_epoch": PROBE_STOP_EPOCH + 1,
            "target_epoch": FORMAL_EPOCHS,
        }
        model = core.YOLO(str(checkpoint), task="detect")

        def setup(trainer):
            report["run_dir"] = str(Path(trainer.save_dir).resolve())
            report["training"]["effective_amp"] = bool(trainer.amp)
            core.save_report(report)

        def epoch_end(trainer):
            epoch = int(trainer.epoch) + 1
            report["training"].update(
                completed_epochs=epoch,
                elapsed_seconds=time.perf_counter() - started,
                peak_memory=core.memory_peak(),
            )
            criterion = getattr(trainer.model, "criterion", None)
            if criterion is not None and hasattr(criterion, "stats_snapshot"):
                report.setdefault("e32_scas", {})["resume_online_augmented_routing"] = criterion.stats_snapshot()
            core.save_report(report)

        model.add_callback("on_train_start", setup)
        model.add_callback("on_fit_epoch_end", epoch_end)
        report["status"] = "resuming_to_200"
        core.save_report(report)
        matching_stats = {}
        with _temporary_e32_criterion(), chunked_assignment() as matching_stats:
            model.train(resume=str(checkpoint), device=0)
        torch.cuda.synchronize()
        if int(report["training"].get("completed_epochs") or 0) != FORMAL_EPOCHS:
            raise RuntimeError(
                f"Resume ended at epoch {report['training'].get('completed_epochs')}, expected {FORMAL_EPOCHS}"
            )
        run_dir = Path(model.trainer.save_dir).resolve()
        weights = run_dir / "weights/best.pt"
        if not weights.is_file():
            raise FileNotFoundError(weights)
        report["training"]["matching_execution"] = dict(matching_stats)
        _copy_training_artifacts(report, run_dir)
        evaluate(report, weights, dataset)
        core.export_csv(report)
        report.update(status="completed", finished_at=core.now(), total_seconds=time.perf_counter() - started)
        core.save_report(report)
        with (ROOT / "comparison_reports/ALL_EXPERIMENTS_COMPARISON.txt").open("a", encoding="utf-8") as handle:
            handle.write("\n" + "=" * 72 + "\n" + core.report_text(report))
        print(f"\nE32a resumed training completed through epoch 200. Report: {out / 'metrics.txt'}", flush=True)
    except BaseException:
        report.update(status="failed_or_interrupted", finished_at=core.now(), error=traceback.format_exc())
        core.save_report(report)
        core.atomic_text(out / "error.log", report["error"])
        raise
    finally:
        print("Automatic shutdown disabled; turn off the instance manually if idle.", flush=True)


def install_overrides():
    if not MODEL_YAML.is_file() or not LOSS_SOURCE.is_file():
        raise FileNotFoundError("E32 YAML/loss is missing")
    core.EXPERIMENT = EXPERIMENT
    core.SCRIPT_REVISION = REVISION
    core.MODEL_YAML = MODEL_YAML
    core.SMOKE_GATE = SMOKE_GATE
    core.MATCHING_HELPER = e18.MATCHING_HELPER
    core.AUTO_SHUTDOWN = False
    core.SHUTDOWN_ON_FAILURE = False
    core.__file__ = str(Path(__file__).resolve())
    core.load_p2_pretrained = e18.load_p2_pretrained
    core.assert_p2_model = e18.assert_p2_model
    core.validate_p2_config = validate_p2_config
    core.p2_preflight = p2_preflight
    core.train = train
    core.evaluate = evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--smoke2", action="store_true")
    modes.add_argument("--probe90", action="store_true")
    modes.add_argument("--resume200", type=Path, metavar="PROBE_RESUME_PT")
    modes.add_argument("--full200", action="store_true")
    modes.add_argument("--eval-only", type=Path, metavar="BEST_PT")
    args = parser.parse_args()
    install_overrides()
    if args.probe90:
        _run_probe90()
        return
    if args.resume200:
        _run_resume200(args.resume200)
        return

    forwarded = [sys.argv[0]]
    if args.check_only:
        forwarded.append("--check-only")
    elif args.smoke2:
        forwarded.append("--smoke2")
    elif args.eval_only:
        forwarded.extend(("--eval-only", str(args.eval_only)))
    # --full200 forwards no mode flag to the frozen core main.
    sys.argv = forwarded
    core.main()


if __name__ == "__main__":
    main()
