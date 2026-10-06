#!/usr/bin/env python3
"""E23a — E10a + training-only gradient-isolated P2-small regression quality.

Parent: E10a topology and deterministic initialization.
Formal protocol: VisDrone, imgsz=1024, batch=8, seed=1, 200 epochs, TAL2.

The only formal intervention is extra CIoU+DFL gradient for TAL-assigned SMALL
P2 positives.  The extra gradient is restricted to E10a's P2-exclusive modules:
model[25], model[27], Detect.cv2[0].

No E10a best.pt continuation.  Initialization remains original yolo11s.pt plus
the same deterministic E1 transfer used by E10a.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import math
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = Path("/root/autodl-tmp/yolo11_project")
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _locate_e10a_runner() -> Path:
    direct = ROOT / "yolo11s_p2_dssg_img1024_seed1.py"
    if direct.is_file():
        return direct
    candidates = sorted(
        ROOT.glob(
            "comparison_reports/e10a_yolo11s_p2_dssg_img1024_seed1_*/"
            "yolo11s_p2_dssg_img1024_seed1.py"
        )
    )
    if not candidates:
        raise FileNotFoundError("Cannot find audited E10a runner")
    return candidates[-1]


E10A_RUNNER = _locate_e10a_runner()
if str(E10A_RUNNER.parent) not in sys.path:
    sys.path.insert(0, str(E10A_RUNNER.parent))
spec = importlib.util.spec_from_file_location("_e23_parent_e10a", E10A_RUNNER)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load E10a runner: {E10A_RUNNER}")
e10a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e10a)
core = e10a.core

EXPERIMENT = "e23a_yolo11s_p2_dssg_regq_img1024_seed1"
SCRIPT_REVISION = "e23a_p2_regq_gradisolated_v1_5_ema_validation_safe"
MODEL_YAML = HERE / "yolo11s-p2-e23a-regq.yaml"
LOSS_SOURCE = HERE / "e23a_p2_regq_loss.py"
SMOKE_GATE = ROOT / "comparison_reports/e23a_p2_regq_smoke_passed.json"
E10A_FORMAL_GLOB = "comparison_reports/e10a_yolo11s_p2_dssg_img1024_seed1_*/metrics.json"

_PARENT_SMOKE_SIGNATURE = core.smoke_signature


def _expected_regq():
    return {
        "enabled": True,
        "extra_gain": 0.50,
        "warmup_epochs": 20.0,
        "batches_per_epoch": 809,
        "small_equivalent_side": 32.0,
        "log_interval": 1000,
    }


def validate_p2_config(cfg, original_cfg):
    e10a.validate_p2_config(cfg, original_cfg)
    if cfg.get("p2_regq") != _expected_regq():
        raise ValueError(f"E23a p2_regq config differs: {cfg.get('p2_regq')!r}")
    if math.ceil(6471 / core.BATCH) != _expected_regq()["batches_per_epoch"]:
        raise RuntimeError("Train image count/batch no longer matches E23a warmup schedule")


def _attach_preflight_args(model):
    """Attach the same formal trainer hyperparameters to a direct preflight model.

    DetectionTrainer normally sets `model.args = self.args` before criterion
    creation.  A model constructed directly inside --check-only has no `.args`.
    """
    if not hasattr(model, "args"):
        model.args = SimpleNamespace(**core.training_params())
    for name, expected in (("box", 7.5), ("cls", 0.5), ("dfl", 1.5)):
        actual = float(getattr(model.args, name))
        if abs(actual - expected) > 1e-12:
            raise RuntimeError(
                f"E23a preflight loss hyperparameter mismatch: "
                f"{name}={actual}, expected={expected}"
            )
    return model.args


def _grad_sum(grads):
    import torch
    return float(sum(g.detach().abs().sum().cpu() for g in grads if g is not None))


def _loss_unit_and_isolation_check(model):
    import torch
    from ultralytics.utils.loss import v8DetectionLoss

    # Direct --check-only models have no trainer-attached `.args`.
    _attach_preflight_args(model)
    from e23a_p2_regq_loss import (
        v8P2RegQDetectionLoss,
        module_gradient_abs_sum,
    )

    model.train()
    torch.manual_seed(2301)
    img = torch.randn(2, 3, 256, 256)
    batch = {
        "img": img,
        "batch_idx": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "cls": torch.tensor([[0], [3], [2], [7]], dtype=torch.float32),
        # current augmented-image sizes:
        # 0: ~13x15 px small; 1: ~64x51 medium
        # 2: ~18x15 px small; 3: ~77x64 medium
        "bboxes": torch.tensor([
            [0.35, 0.35, 0.050, 0.060],
            [0.70, 0.65, 0.250, 0.200],
            [0.55, 0.45, 0.070, 0.060],
            [0.25, 0.70, 0.300, 0.250],
        ], dtype=torch.float32),
    }

    # Gain=0 must numerically reduce to standard v8DetectionLoss on same predictions.
    preds = model(img)
    base = v8DetectionLoss(model)
    custom0 = v8P2RegQDetectionLoss(model)
    custom0.settings = replace(custom0.settings, extra_gain=0.0, warmup_epochs=0.0)
    # Post-init zero gain is intentional here: it probes exact base-path equivalence.
    total_b, items_b = base(preds, batch)
    total_0, items_0 = custom0(preds, batch)
    loss_abs = float((total_b.detach() - total_0.detach()).abs())
    items_abs = float((items_b.detach() - items_0.detach()).abs().max())
    if loss_abs > 1e-5 or items_abs > 1e-6:
        raise RuntimeError(
            f"E23a near-zero-gain base equivalence failed: total={loss_abs}, items={items_abs}"
        )

    # Fresh graph for gradient-isolation audit at full gain.
    # IMPORTANT: audit the actual incremental objective using the EXACT base
    # subgraph that the custom criterion used:
    #     delta_total = custom_total - custom_base_total
    # This avoids false leakage from floating cancellation between two separately
    # constructed-but-equivalent base loss graphs.
    preds = model(img)
    crit = v8P2RegQDetectionLoss(model)
    crit.settings = replace(crit.settings, warmup_epochs=0.0)
    crit.audit_capture = True
    total, items = crit(preds, batch)
    proxy = crit._last_proxy_graph
    base_total_full = crit._last_base_total_graph
    stats = crit._last_stats
    if proxy is None or base_total_full is None or stats is None:
        raise RuntimeError("E23a audit did not capture auxiliary proxy/base graph")
    if stats["p2_small_positive_anchors"] <= 0:
        raise RuntimeError(f"E23a unit batch produced no P2-small positives: {stats}")
    if not torch.isfinite(total) or not torch.isfinite(proxy) or not torch.isfinite(base_total_full):
        raise RuntimeError("E23a unit loss/proxy/base is non-finite")

    # Float32 addition is not exactly invertible: (base + proxy) - base can differ
    # from proxy by one or a few ULPs.  Audit the construction with scale-aware
    # allclose instead of demanding exact scalar subtraction.
    expected_total = base_total_full.detach() + proxy.detach()
    forward_identity_abs = float((total.detach() - expected_total).abs())
    forward_scale = max(
        1.0,
        float(total.detach().abs()),
        float(base_total_full.detach().abs()),
        float(proxy.detach().abs()),
    )
    forward_tol = 8.0 * torch.finfo(total.dtype).eps * forward_scale
    if forward_identity_abs > forward_tol:
        raise RuntimeError(
            "E23a total != base + isolated proxy beyond FP tolerance: "
            f"abs={forward_identity_abs}, tol={forward_tol}, scale={forward_scale}"
        )

    # The actual extra graph is `proxy`; audit its dependency graph directly.
    # `total` is constructed immediately above as `base_total + proxy`.
    allowed = {
        "model25_p2_fuse": module_gradient_abs_sum(model.model[25], proxy),
        "model27_semantic_detail_gate": module_gradient_abs_sum(model.model[27], proxy),
        "detect_cv2_0_p2_box": module_gradient_abs_sum(model.model[28].cv2[0], proxy),
    }
    forbidden = {
        "model26_p3_semantic_guide": module_gradient_abs_sum(model.model[26], proxy),
        "detect_cv2_1_p3_box": module_gradient_abs_sum(model.model[28].cv2[1], proxy),
        "detect_cv2_2_p4_box": module_gradient_abs_sum(model.model[28].cv2[2], proxy),
        "detect_cv2_3_p5_box": module_gradient_abs_sum(model.model[28].cv2[3], proxy),
        "detect_cv3_0_p2_cls": module_gradient_abs_sum(model.model[28].cv3[0], proxy),
        "detect_cv3_1_p3_cls": module_gradient_abs_sum(model.model[28].cv3[1], proxy),
        "detect_cv3_2_p4_cls": module_gradient_abs_sum(model.model[28].cv3[2], proxy),
        "detect_cv3_3_p5_cls": module_gradient_abs_sum(model.model[28].cv3[3], proxy),
        "backbone_layer2": module_gradient_abs_sum(model.model[2], proxy),
    }
    if not all(v > 0.0 and math.isfinite(v) for v in allowed.values()):
        raise RuntimeError(f"E23a allowed auxiliary gradient missing/nonfinite: {allowed}")
    if any(abs(v) > 1e-12 for v in forbidden.values()):
        raise RuntimeError(f"E23a auxiliary gradient leaked outside allowed P2 modules: {forbidden}")

    # Regression test for the exact Ultralytics in-training validation pattern:
    # ModelEMA has requires_grad=False on every parameter and validator sets eval().
    # The custom criterion must instantiate and compute STANDARD loss only.
    saved_requires_grad = [p.requires_grad for p in model.parameters()]
    was_training = model.training
    try:
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        with torch.inference_mode():
            eval_preds = model(img)
            eval_crit = v8P2RegQDetectionLoss(model)
            if eval_crit.allowed_params:
                raise RuntimeError(
                    "E23a EMA/eval regression test unexpectedly found trainable allowed params"
                )
            eval_total, eval_items = eval_crit(eval_preds, batch)
        if not torch.isfinite(eval_total) or not torch.isfinite(eval_items).all():
            raise RuntimeError("E23a EMA/eval validation loss is non-finite")
        ema_eval_check = {
            "all_parameters_requires_grad_false": True,
            "model_training": False,
            "allowed_param_count": 0,
            "finite_validation_total": float(eval_total.detach().float().cpu()),
            "finite_validation_items": [
                float(x) for x in eval_items.detach().float().cpu().tolist()
            ],
            "auxiliary_inactive": True,
        }
    finally:
        for p, flag in zip(model.parameters(), saved_requires_grad):
            p.requires_grad_(flag)
        model.train(was_training)

    return {
        "near_zero_gain_base_total_abs_delta": loss_abs,
        "near_zero_gain_base_items_max_abs_delta": items_abs,
        "full_gain_total_minus_base_plus_proxy_abs_value": forward_identity_abs,
        "full_gain_forward_fp_tolerance": forward_tol,
        "selected_P2_small_positive_anchors": stats["p2_small_positive_anchors"],
        "allowed_aux_gradient_abs_sum": allowed,
        "forbidden_aux_gradient_abs_sum": forbidden,
        "extra_gradient_scope": [
            "model[25] fused P2 C3k2",
            "model[27] SemanticDetailGate",
            "model[28].cv2[0] P2 regression branch",
        ],
        "classification_aux_gradient": "exactly zero in unit audit",
        "P3_P4_P5_aux_gradient": "exactly zero in unit audit",
        "shared_P3_semantic_guide_aux_gradient": "exactly zero in unit audit",
        "ema_eval_validation_path": ema_eval_check,
        "small_definition": "equivalent side <=32 px in current augmented input",
        "assignment_changed": False,
        "inference_topology_changed": False,
    }


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
    _attach_preflight_args(model)

    report["p2_script_revision"] = SCRIPT_REVISION
    report["e23a_regression_quality"] = {
        "config": _expected_regq(),
        "loss_source": str(LOSS_SOURCE),
        "loss_source_sha256": core.sha256(LOSS_SOURCE),
        "parent_e10a_runner": str(E10A_RUNNER),
        "inference_topology_identical_to_e10a": True,
        "new_inference_parameters": 0,
        "new_inference_flops": 0,
        "initialization": transfer,
        "gradient_isolation": (
            "ordinary E10a loss unchanged everywhere; only EXTRA P2-small CIoU+DFL "
            "gradient is restricted to model[25], model[27], Detect.cv2[0]"
        ),
        "preflight_loss_args": {
            "box": float(model.args.box),
            "cls": float(model.args.cls),
            "dfl": float(model.args.dfl),
            "note": (
                "Direct DetectionModel construction has no args; preflight attaches "
                "core.training_params() exactly. Real DetectionTrainer attaches self.args "
                "before criterion creation."
            ),
        },
        "unit_check": _loss_unit_and_isolation_check(model),
        "claim_boundary": (
            "Tests whether stronger regression learning on E10a's P2-exclusive path can "
            "improve high-IoU small raw-box geometry. It does not modify TAL, classification, "
            "P3/P4/P5, inference topology, or add a new feature module."
        ),
    }
    shutil.copy2(LOSS_SOURCE, Path(report["report_dir"]) / LOSS_SOURCE.name)
    core.save_report(report)
    del model
    print("E23a P2-RegQ structure/loss/gradient-isolation checks passed.", flush=True)


@contextlib.contextmanager
def _temporary_regq_criterion():
    from ultralytics.nn.tasks import DetectionModel
    from e23a_p2_regq_loss import v8P2RegQDetectionLoss, reset_runtime_audit

    original = DetectionModel.init_criterion

    def patched(self):
        cfg = self.yaml.get("p2_regq", {})
        if isinstance(cfg, dict) and cfg.get("enabled", False):
            return v8P2RegQDetectionLoss(self)
        return original(self)

    reset_runtime_audit()
    DetectionModel.init_criterion = patched
    try:
        yield
    finally:
        DetectionModel.init_criterion = original


def train(report, smoke=False):
    from e23a_p2_regq_loss import runtime_audit_snapshot

    try:
        with _temporary_regq_criterion():
            weights = e10a._CORE_TRAIN(report, smoke=smoke)
    finally:
        # Preserve E23a-specific runtime evidence even if a later validator,
        # evaluator, or callback raises.  This makes smoke failures diagnosable.
        report["e23a_regression_quality"]["training_runtime_audit"] = runtime_audit_snapshot()
        core.save_report(report)

    run_dir = Path(report["run_dir"])
    for path in (
        LOSS_SOURCE,
        e10a.DSSG_SOURCE,
        e10a.REFERENCE_E1_YAML,
        e10a.MATCHING_HELPER,
    ):
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
        d_overall = {
            k: float(report["overall"][k]) - float(ref["overall"][k])
            for k in overall_names
        }
        d_area = {
            k: float(report["area_metrics"][k]) - float(ref["area_metrics"][k])
            for k in area_names
        }

        # Pre-registered screening/resource gates.  These are not significance tests.
        gates = {
            "AP_small_delta_ge_0.005": d_area["AP_small"] >= 0.005,
            "mAP50_95_delta_ge_0.0015": d_overall["mAP50_95"] >= 0.0015,
            "AP_medium_delta_ge_minus_0.002": d_area["AP_medium"] >= -0.002,
            "AP_large_delta_ge_minus_0.005": d_area["AP_large"] >= -0.005,
            "AP_all_not_lower": d_area["AP_all"] >= 0.0,
        }
        report["comparison_to_frozen_E10a"] = {
            "status": "computed",
            "reference": str(path),
            "deltas": {"overall": d_overall, "area_metrics": d_area},
            "screening_gate": {
                "rules": {
                    "AP_small_min_delta": 0.005,
                    "mAP50_95_min_delta": 0.0015,
                    "AP_medium_min_delta": -0.002,
                    "AP_large_min_delta": -0.005,
                    "AP_all_min_delta": 0.0,
                },
                "checks": gates,
                "passed": bool(all(gates.values())),
                "note": (
                    "Passing metric gates is necessary but NOT sufficient. "
                    "Run diagnose_e23a_regq_mechanism.py and require raw P2 geometry improvement."
                ),
            },
        }

    trained = core.YOLO(str(weights)).model.float()
    e10a.assert_p2_model(trained)
    report["e23a_regression_quality"]["posttrain_topology_audit"] = {
        "parameters": sum(p.numel() for p in trained.parameters()),
        "strides": [float(x) for x in trained.stride.cpu().tolist()],
        "detect_inputs": list(trained.model[-1].f),
        "detect_class": trained.model[-1].__class__.__name__,
    }
    del trained
    core.save_report(report)



def smoke_signature(report):
    """Parent smoke signature + E23a loss implementation/config identity."""
    parent = _PARENT_SMOKE_SIGNATURE(report)
    payload = {
        "parent_signature": parent,
        "loss_source_sha256": core.sha256(LOSS_SOURCE),
        "regq_config": _expected_regq(),
        "runner_revision": SCRIPT_REVISION,
    }
    import hashlib
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def install_overrides():
    if not MODEL_YAML.is_file() or not LOSS_SOURCE.is_file():
        raise FileNotFoundError("E23a companion YAML/loss source missing")

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
    core.smoke_signature = smoke_signature


if __name__ == "__main__":
    install_overrides()
    core.main()
