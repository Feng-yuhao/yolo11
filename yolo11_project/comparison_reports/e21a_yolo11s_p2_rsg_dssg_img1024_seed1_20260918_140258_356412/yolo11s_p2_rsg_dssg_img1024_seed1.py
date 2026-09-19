#!/usr/bin/env python3
"""E21a: E10a DSSG + training-only Relative Small-aware Gate guidance.

Formal protocol remains E10a: VisDrone, imgsz=1024, batch=8, seed=1,
200 epochs, TAL2, original yolo11s.pt. No E10a best.pt continuation.
Inference graph/parameters/GFLOPs are unchanged.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/root/autodl-tmp/yolo11_project")
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _locate_e10a_runner() -> Path:
    direct = ROOT / "yolo11s_p2_dssg_img1024_seed1.py"
    if direct.is_file():
        return direct
    candidates = sorted(ROOT.glob(
        "comparison_reports/e10a_yolo11s_p2_dssg_img1024_seed1_*/"
        "yolo11s_p2_dssg_img1024_seed1.py"
    ))
    if not candidates:
        raise FileNotFoundError("Cannot find audited E10a runner")
    return candidates[-1]


E10A_RUNNER = _locate_e10a_runner()
if str(E10A_RUNNER.parent) not in sys.path:
    sys.path.insert(0, str(E10A_RUNNER.parent))
spec = importlib.util.spec_from_file_location("_e21_parent_e10a", E10A_RUNNER)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load E10a runner: {E10A_RUNNER}")
e10a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e10a)
core = e10a.core

EXPERIMENT = "e21a_yolo11s_p2_rsg_dssg_img1024_seed1"
SCRIPT_REVISION = "e21a_rsg_dssg_relative_gate_seed1_200e_v3_pickle_safe_importfix"
MODEL_YAML = HERE / "yolo11s-p2-rsg-dssg.yaml"
RSG_SOURCE = HERE / "rsg_loss.py"
SMOKE_GATE = ROOT / "comparison_reports/e21a_rsg_dssg_smoke_passed.json"
E10A_FORMAL_GLOB = "comparison_reports/e10a_yolo11s_p2_dssg_img1024_seed1_*/metrics.json"


def _expected_rsg():
    return {
        "enabled": True,
        "gain": 0.02,
        "warmup_epochs": 20.0,
        "batches_per_epoch": 809,
        "max_aux_ratio": 0.05,
        "tiny_equivalent_side": 32.0,
        "margin": 0.25,
        "sigma_scale": 0.35,
        "sigma_min": 0.75,
        "sigma_max": 2.5,
        "radius": 6,
        "log_interval": 1000,
    }


def validate_p2_config(cfg, original_cfg):
    e10a.validate_p2_config(cfg, original_cfg)
    if cfg.get("rsg") != _expected_rsg():
        raise ValueError(f"E21a rsg config differs: {cfg.get('rsg')!r}")
    if math.ceil(6471 / core.BATCH) != _expected_rsg()["batches_per_epoch"]:
        raise RuntimeError("Train image count/batch no longer matches RSG warmup schedule")


def _rsg_unit_check(model):
    import torch
    from rsg_loss import RSGSettings, relative_small_gate_loss

    detail = model.model[27]
    settings = RSGSettings.from_yaml(model.yaml["rsg"])
    settings.validate()
    captured = {}

    def capture(_module, _inputs, output):
        captured["logits"] = output

    handle = detail.spatial_gate.net[-1].register_forward_hook(capture)
    try:
        detail.train()
        torch.manual_seed(2101)
        p2 = torch.randn(2, 128, 64, 64, requires_grad=True)
        p3 = torch.randn(2, 128, 32, 32, requires_grad=True)
        _ = detail([p2, p3])
        logits = captured.get("logits")
        if logits is None or logits.shape != (2, 1, 64, 64):
            raise RuntimeError("RSG hook did not capture P2 pre-sigmoid gate logits")

        batch = {
            "img": torch.zeros(2, 3, 256, 256),
            "bboxes": torch.tensor([
                [0.30, 0.35, 0.05, 0.06],
                [0.65, 0.60, 0.25, 0.20],
            ], dtype=torch.float32),
            "batch_idx": torch.tensor([0, 1], dtype=torch.long),
        }
        loss, stats = relative_small_gate_loss(logits, batch, settings)
        if not torch.isfinite(loss) or stats.get("small_objects") != 1:
            raise RuntimeError(f"RSG unit loss/selection failed: {stats}")
        for p in detail.spatial_gate.parameters():
            p.grad = None
        loss.backward()
        grad = detail.spatial_gate.net[-1].weight.grad
        if grad is None or not torch.isfinite(grad).all() or float(grad.abs().sum()) <= 0:
            raise RuntimeError("RSG does not deliver finite nonzero gradient")
        return {
            "captured_logits_shape": list(logits.shape),
            "selected_small_objects": stats["small_objects"],
            "aux_loss": float(loss.detach()),
            "final_gate_conv_grad_abs_sum": float(grad.abs().sum()),
            "training_definition": "equivalent side <=32 px in current augmented input",
            "no_background_negative": True,
            "medium_large_ignored": True,
        }
    finally:
        handle.remove()


def p2_preflight(report, eval_only=False):
    old_yaml = e10a.MODEL_YAML
    e10a.MODEL_YAML = MODEL_YAML
    try:
        e10a.p2_preflight(report, eval_only=eval_only)
    finally:
        e10a.MODEL_YAML = old_yaml

    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils.torch_utils import init_seeds

    cfg = yaml_model_load(str(MODEL_YAML))
    validate_p2_config(cfg, None)
    init_seeds(core.SEED, deterministic=True)
    model = DetectionModel(cfg, nc=10, verbose=False)
    e10a.assert_p2_model(model)
    transfer = e10a.load_p2_pretrained(model)

    report["p2_script_revision"] = SCRIPT_REVISION
    report["rsg"] = {
        "config": _expected_rsg(),
        "loss_source": str(RSG_SOURCE),
        "loss_source_sha256": core.sha256(RSG_SOURCE),
        "parent_e10a_runner": str(E10A_RUNNER),
        "inference_topology_identical_to_e10a": True,
        "new_inference_parameters": 0,
        "new_inference_flops": 0,
        "initialization": transfer,
        "unit_check": _rsg_unit_check(model),
        "claim_boundary": (
            "Training-only relative small-aware guidance of the existing E10a P2 spatial gate; "
            "not foreground-mask supervision; no background negatives; Detect/TAL/inference unchanged."
        ),
    }
    shutil.copy2(RSG_SOURCE, Path(report["report_dir"]) / RSG_SOURCE.name)
    core.save_report(report)
    del model
    print("E21a RSG checks passed.", flush=True)


@contextlib.contextmanager
def _temporary_rsg_criterion():
    from ultralytics.nn.tasks import DetectionModel
    from rsg_loss import v8RSGDetectionLoss

    original = DetectionModel.init_criterion

    def patched(self):
        cfg = self.yaml.get("rsg", {})
        if isinstance(cfg, dict) and cfg.get("enabled", False):
            return v8RSGDetectionLoss(self)
        return original(self)

    DetectionModel.init_criterion = patched
    try:
        yield
    finally:
        DetectionModel.init_criterion = original


def train(report, smoke=False):
    with _temporary_rsg_criterion():
        weights = e10a._CORE_TRAIN(report, smoke=smoke)
    run_dir = Path(report["run_dir"])
    for path in (RSG_SOURCE, e10a.DSSG_SOURCE, e10a.REFERENCE_E1_YAML, e10a.MATCHING_HELPER):
        if path.is_file():
            shutil.copy2(path, run_dir / path.name)
    return weights


def _formal_e10a_reference():
    candidates = []
    for path in ROOT.glob(E10A_FORMAL_GLOB):
        if "SMOKE" in str(path).upper():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("status") == "completed":
            candidates.append((path, data))
    return sorted(candidates, key=lambda x: str(x[0]))[-1] if candidates else None


def evaluate(report, weights, dataset):
    e10a._CORE_EVALUATE(report, weights, dataset)
    reference = _formal_e10a_reference()
    if reference is None:
        report["comparison_to_frozen_E10a"] = {
            "status": "unavailable",
            "note": "No completed formal E10a metrics.json found",
        }
    else:
        path, ref = reference
        overall_names = ("Precision", "Recall", "mAP50", "mAP75", "mAP50_95")
        area_names = ("AP_all", "AP_small", "AP_medium", "AP_large")
        d_overall = {k: float(report["overall"][k]) - float(ref["overall"][k]) for k in overall_names}
        d_area = {k: float(report["area_metrics"][k]) - float(ref["area_metrics"][k]) for k in area_names}
        hard_pass = (
            d_area["AP_small"] >= 0.003
            and d_overall["mAP50_95"] >= 0.0
            and d_area["AP_medium"] >= -0.003
            and d_area["AP_large"] >= -0.005
        )
        strong_pass = hard_pass and d_overall["mAP50_95"] >= 0.0015
        report["comparison_to_frozen_E10a"] = {
            "status": "computed",
            "reference": str(path),
            "deltas": {"overall": d_overall, "area_metrics": d_area},
            "gate": {
                "hard_requirements": {
                    "AP_small_min_delta": 0.003,
                    "mAP50_95_min_delta": 0.0,
                    "AP_medium_min_delta": -0.003,
                    "AP_large_min_delta": -0.005,
                },
                "strong_target": {"mAP50_95_min_delta": 0.0015},
                "passed": bool(hard_pass),
                "strong_passed": bool(strong_pass),
                "failure_policy": "If AP_small < +0.30 pp, stop this supervision line.",
            },
        }

    trained = core.YOLO(str(weights)).model.float()
    e10a.assert_p2_model(trained)
    deep, detail = trained.model[26], trained.model[27]
    report["dssg_learned_state"] = {
        "P3_layer_scale_mean": float(deep.layer_scale.detach().mean()),
        "P2_layer_scale_mean": float(detail.layer_scale.detach().mean()),
        "channel_output_weight_l2": float(deep.channel_gate[1].weight.detach().norm()),
        "P3_spatial_output_weight_l2": float(deep.spatial_gate.net[-1].weight.detach().norm()),
        "P2_spatial_output_weight_l2": float(detail.spatial_gate.net[-1].weight.detach().norm()),
        "note": "Parameter magnitude is descriptive only; it is not causal proof.",
    }
    del trained
    core.save_report(report)


def install_overrides():
    if not MODEL_YAML.is_file() or not RSG_SOURCE.is_file():
        raise FileNotFoundError("E21a companion YAML/rsg_loss.py missing")

    core.EXPERIMENT = EXPERIMENT
    core.SCRIPT_REVISION = SCRIPT_REVISION
    core.MODEL_YAML = MODEL_YAML
    core.SMOKE_GATE = SMOKE_GATE
    core.MATCHING_HELPER = e10a.MATCHING_HELPER
    core.AUTO_SHUTDOWN = False
    core.SHUTDOWN_ON_FAILURE = False
    core.__file__ = str(Path(__file__).resolve())

    core.transfer_key = e10a.transfer_key
    core.plan_transfer = e10a.plan_transfer
    core.load_p2_pretrained = e10a.load_p2_pretrained
    core.assert_p2_model = e10a.assert_p2_model
    core.validate_p2_config = validate_p2_config
    core.p2_preflight = p2_preflight
    core.train = train
    core.evaluate = evaluate


if __name__ == "__main__":
    install_overrides()
    core.main()
