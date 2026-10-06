#!/usr/bin/env python3
"""E25a: E10a DSSG + direct-gradient-isolated E21a RSG.

Formal protocol: VisDrone, imgsz=1024, batch=8, seed=1, 200 epochs,
unchanged TAL2 and original yolo11s.pt initialization.  Inference is E10a.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = Path("/root/autodl-tmp/yolo11_project")
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


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
spec = importlib.util.spec_from_file_location("_e25_parent_e10a", E10A_RUNNER)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load E10a runner: {E10A_RUNNER}")
e10a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e10a)
core = e10a.core

EXPERIMENT = "e25a_yolo11s_p2_gi_rsg_dssg_img1024_seed1"
SCRIPT_REVISION = "e25a_gradient_isolated_rsg_strict_single_variable_seed1_200e_v1"
MODEL_YAML = HERE / "yolo11s-p2-gi-rsg-dssg.yaml"
GI_RSG_SOURCE = HERE / "gi_rsg_loss.py"
SMOKE_GATE = ROOT / "comparison_reports/e25a_gi_rsg_dssg_smoke_passed.json"


def _expected_gi_rsg():
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
    if cfg.get("gi_rsg") != _expected_gi_rsg():
        raise ValueError(f"E25a GI-RSG config differs from preregistration: {cfg.get('gi_rsg')!r}")
    if "rsg" in cfg:
        raise ValueError("E25a YAML must not enable the original E21a criterion")
    if math.ceil(6471 / core.BATCH) != _expected_gi_rsg()["batches_per_epoch"]:
        raise RuntimeError("Train image count/batch no longer matches the frozen RSG warmup schedule")


def _synthetic_batch(torch, batch_size=2, size=128):
    return {
        "img": torch.rand(batch_size, 3, size, size),
        "batch_idx": torch.tensor([0, 0, 1], dtype=torch.float32),
        "cls": torch.tensor([[0.0], [3.0], [9.0]], dtype=torch.float32),
        "bboxes": torch.tensor([
            [0.30, 0.35, 0.08, 0.10],
            [0.65, 0.60, 0.28, 0.24],
            [0.48, 0.52, 0.12, 0.10],
        ], dtype=torch.float32),
    }


def _flatten_tensors(value):
    if hasattr(value, "shape"):
        return [value]
    if isinstance(value, (tuple, list)):
        rows = []
        for item in value:
            rows.extend(_flatten_tensors(item))
        return rows
    if isinstance(value, dict):
        rows = []
        for key in sorted(value):
            rows.extend(_flatten_tensors(value[key]))
        return rows
    raise TypeError(f"Unsupported prediction container: {type(value)!r}")


def _max_tensor_difference(left, right):
    import torch

    left_rows, right_rows = _flatten_tensors(left), _flatten_tensors(right)
    if len(left_rows) != len(right_rows):
        raise AssertionError("Prediction tensor counts differ")
    maximum = 0.0
    for a, b in zip(left_rows, right_rows):
        if a.shape != b.shape:
            raise AssertionError(f"Prediction shape differs: {a.shape} vs {b.shape}")
        maximum = max(maximum, float((a.detach() - b.detach()).abs().max()))
        if not torch.allclose(a.detach(), b.detach(), rtol=0.0, atol=0.0):
            raise AssertionError(f"Prediction values differ; max abs={maximum}")
    return maximum


def _gradient_map(model):
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def _compare_gradient_maps(reference, candidate):
    import torch

    if reference.keys() != candidate.keys():
        raise AssertionError("Gradient parameter names differ")
    maximum = 0.0
    for name in reference:
        a, b = reference[name], candidate[name]
        if (a is None) != (b is None):
            raise AssertionError(f"AUX-OFF gradient presence differs at {name}")
        if a is not None:
            difference = float((a - b).abs().max())
            maximum = max(maximum, difference)
            if not torch.allclose(a, b, rtol=0.0, atol=0.0):
                raise AssertionError(f"AUX-OFF gradient differs at {name}: {difference}")
    return maximum


def _bn_state(module):
    return {
        name: value.detach().clone()
        for name, value in module.named_buffers()
        if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }


def _same_state(a, b):
    import torch

    return a.keys() == b.keys() and all(torch.equal(a[name], b[name]) for name in a)


def _gi_rsg_causal_preflight(source_model):
    """Bounded CPU checks for the preregistered direct-gradient intervention."""
    import torch
    from gi_rsg_loss import v8GIRSGDetectionLoss
    from ultralytics.utils.loss import v8DetectionLoss

    torch.manual_seed(2501)
    batch = _synthetic_batch(torch)
    args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    # A: with the auxiliary objective disabled, E25a must be exactly E10a.
    reference = copy.deepcopy(source_model).cpu().train()
    candidate = copy.deepcopy(source_model).cpu().train()
    reference.args = args
    candidate.args = args
    base_criterion = v8DetectionLoss(reference)
    gi_criterion = v8GIRSGDetectionLoss(candidate)
    gi_criterion.aux_enabled = False
    reference.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    pred_reference = reference(batch["img"])
    pred_candidate = candidate(batch["img"])
    prediction_difference = _max_tensor_difference(pred_reference, pred_candidate)
    base_total, base_items = base_criterion(pred_reference, batch)
    gi_total, gi_items = gi_criterion(pred_candidate, batch)
    if not torch.equal(base_total.detach(), gi_total.detach()):
        raise AssertionError("AUX-OFF total loss differs from E10a")
    if not torch.equal(base_items.detach(), gi_items.detach()):
        raise AssertionError("AUX-OFF box/cls/DFL items differ from E10a")
    base_total.backward()
    gi_total.backward()
    gradient_difference = _compare_gradient_maps(_gradient_map(reference), _gradient_map(candidate))
    gi_criterion.close()

    # B/C/D: aux-only gradients, descriptor detachment and BN buffer isolation.
    isolated = copy.deepcopy(source_model).cpu().train()
    isolated.args = args
    criterion = v8GIRSGDetectionLoss(isolated)
    criterion._training_batches_seen = int(
        round(criterion.settings.warmup_epochs * criterion.settings.batches_per_epoch)
    )
    isolated.zero_grad(set_to_none=True)
    gate = isolated.model[27].spatial_gate
    normal_logits = {}

    def capture_normal_logits(_module, _inputs, output):
        normal_logits["value"] = output.detach().clone()

    logits_handle = gate.net[-1].register_forward_hook(capture_normal_logits)
    before_normal = _bn_state(gate.net)
    try:
        _ = isolated(batch["img"])
    finally:
        logits_handle.remove()
    descriptor = criterion._latest_descriptor
    if descriptor is None or not descriptor.requires_grad:
        raise AssertionError("Normal forward did not capture a differentiable five-channel descriptor")
    if "value" not in normal_logits:
        raise AssertionError("Normal forward did not expose pre-sigmoid gate logits")
    descriptor.retain_grad()
    after_normal = _bn_state(gate.net)
    tracked = [name for name in before_normal if name.endswith("num_batches_tracked")]
    if not tracked or any(
        int(after_normal[name].item()) != int(before_normal[name].item()) + 1 for name in tracked
    ):
        raise AssertionError("Normal spatial-gate forward did not update real BN exactly once")

    from gi_rsg_loss import isolated_gate_logits

    recomputed_logits = isolated_gate_logits(gate.net, descriptor)
    if not torch.equal(normal_logits["value"], recomputed_logits.detach()):
        difference = float((normal_logits["value"] - recomputed_logits.detach()).abs().max())
        raise AssertionError(f"Isolated recompute changed E21a gate logits: max abs={difference}")
    if not _same_state(after_normal, _bn_state(gate.net)):
        raise AssertionError("Logit-equivalence recompute modified real BN buffers")
    del recomputed_logits

    aux = criterion.auxiliary_loss(batch, torch.tensor(10.0))
    after_aux = _bn_state(gate.net)
    if not _same_state(after_normal, after_aux):
        raise AssertionError("Auxiliary functional call modified the real spatial-gate BN buffers")
    aux.backward()
    if descriptor.grad is not None and float(descriptor.grad.abs().max()) != 0.0:
        raise AssertionError("GI-RSG auxiliary loss leaked gradient into the captured descriptor")

    allowed_prefix = "model.27.spatial_gate."
    nonzero, outside = [], []
    for name, parameter in isolated.named_parameters():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            continue
        if float(parameter.grad.abs().sum()) == 0.0:
            continue
        nonzero.append(name)
        if not name.startswith(allowed_prefix):
            outside.append(name)
    if not nonzero:
        raise AssertionError("GI-RSG auxiliary objective produced no nonzero parameter gradient")
    if outside:
        raise AssertionError(f"GI-RSG auxiliary gradient escaped spatial gate: {outside}")
    criterion.close()

    del reference, candidate, isolated
    return {
        "A_aux_off": {
            "prediction_max_abs_difference": prediction_difference,
            "total_loss_equal": True,
            "loss_items_equal": True,
            "gradient_max_abs_difference": gradient_difference,
        },
        "B_aux_only": {
            "allowed_prefix": allowed_prefix,
            "nonzero_gradient_parameters": nonzero,
            "outside_allowed_scope": outside,
        },
        "C_descriptor_detach": {
            "descriptor_shape": list(descriptor.shape),
            "descriptor_requires_grad_before_detach": True,
            "descriptor_aux_gradient_is_none_or_zero": True,
        },
        "D_bn_buffers": {
            "normal_forward_incremented_num_batches_once": True,
            "aux_recompute_left_real_buffers_unchanged": True,
            "functional_call_training_mode": True,
            "recomputed_logits_exactly_equal_normal_E21a_logits": True,
        },
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
    audit = _gi_rsg_causal_preflight(model)

    report["p2_script_revision"] = SCRIPT_REVISION
    report["gi_rsg"] = {
        "config": _expected_gi_rsg(),
        "loss_source": str(GI_RSG_SOURCE),
        "loss_source_sha256": core.sha256(GI_RSG_SOURCE),
        "parent_e10a_runner": str(E10A_RUNNER),
        "initialization": transfer,
        "inference_topology_identical_to_e10a": True,
        "new_inference_parameters": 0,
        "new_inference_flops": 0,
        "single_intervention": (
            "E21a descriptor is detached and the same P2 spatial-gate parameters are called "
            "with cloned BN buffers; direct auxiliary gradients are restricted to that gate."
        ),
        "causal_preflight": audit,
        "scientific_boundary": (
            "Direct-gradient isolation only. Updated gate parameters can indirectly change later "
            "normal detection-loss trajectories; this is not complete causal compartmentalization."
        ),
    }
    shutil.copy2(GI_RSG_SOURCE, Path(report["report_dir"]) / GI_RSG_SOURCE.name)
    core.save_report(report)
    del model
    print("E25a GI-RSG strict causal preflight passed.", flush=True)


@contextlib.contextmanager
def _temporary_gi_rsg_criterion():
    from gi_rsg_loss import v8GIRSGDetectionLoss
    from ultralytics.nn.tasks import DetectionModel

    original = DetectionModel.init_criterion

    def patched(self):
        cfg = self.yaml.get("gi_rsg", {})
        if isinstance(cfg, dict) and cfg.get("enabled", False):
            return v8GIRSGDetectionLoss(self)
        return original(self)

    DetectionModel.init_criterion = patched
    try:
        yield
    finally:
        DetectionModel.init_criterion = original


def train(report, smoke=False):
    with _temporary_gi_rsg_criterion():
        weights = e10a._CORE_TRAIN(report, smoke=smoke)
    run_dir = Path(report["run_dir"])
    for path in (GI_RSG_SOURCE, e10a.DSSG_SOURCE, e10a.REFERENCE_E1_YAML, e10a.MATCHING_HELPER):
        if path.is_file():
            shutil.copy2(path, run_dir / path.name)
    return weights


def _formal_reference(pattern):
    candidates = []
    for path in ROOT.glob(pattern):
        if "SMOKE" in str(path).upper():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("status") == "completed":
            candidates.append((path, data))
    return sorted(candidates, key=lambda row: str(row[0]))[-1] if candidates else None


def evaluate(report, weights, dataset):
    e10a._CORE_EVALUATE(report, weights, dataset)
    references = {
        "E10a": _formal_reference(
            "comparison_reports/e10a_yolo11s_p2_dssg_img1024_seed1_*/metrics.json"
        ),
        "E21a": _formal_reference(
            "comparison_reports/e21a_yolo11s_p2_rsg_dssg_img1024_seed1_*/metrics.json"
        ),
    }
    comparisons = {}
    for label, reference in references.items():
        if reference is None:
            comparisons[label] = {"status": "unavailable"}
            continue
        path, data = reference
        comparisons[label] = {
            "status": "computed",
            "reference": str(path),
            "deltas": {
                "mAP50": float(report["overall"]["mAP50"]) - float(data["overall"]["mAP50"]),
                "mAP75": float(report["overall"]["mAP75"]) - float(data["overall"]["mAP75"]),
                "mAP50_95": float(report["overall"]["mAP50_95"]) - float(data["overall"]["mAP50_95"]),
                "AP_small": float(report["area_metrics"]["AP_small"]) - float(data["area_metrics"]["AP_small"]),
                "AP_medium": float(report["area_metrics"]["AP_medium"]) - float(data["area_metrics"]["AP_medium"]),
                "AP_large": float(report["area_metrics"]["AP_large"]) - float(data["area_metrics"]["AP_large"]),
            },
        }

    observed = {
        "mAP50_95": float(report["overall"]["mAP50_95"]),
        "AP75": float(report["overall"]["mAP75"]),
        "AP_small": float(report["area_metrics"]["AP_small"]),
        "AP_medium": float(report["area_metrics"]["AP_medium"]),
        "AP_large": float(report["area_metrics"]["AP_large"]),
    }
    thresholds = {
        "mAP50_95": 0.3435,
        "AP75": 0.3580,
        "AP_small": 0.2310,
        "AP_medium": 0.4270,
        "AP_large": 0.56135,
    }
    passed = all(observed[name] >= threshold for name, threshold in thresholds.items())
    report["comparison_to_E10a_and_E21a"] = comparisons
    report["e25a_preregistered_gate"] = {
        "observed": observed,
        "minimums": thresholds,
        "passed": bool(passed),
        "decision": "retain_GI_RSG" if passed else "close_RSG_line_no_E25b_or_E25c",
        "scope": "Single-seed screening gate, not a statistical-significance claim.",
    }

    trained = core.YOLO(str(weights)).model.float()
    e10a.assert_p2_model(trained)
    gate = trained.model[27].spatial_gate
    report["gi_rsg_learned_state"] = {
        "P2_spatial_output_weight_l2": float(gate.net[-1].weight.detach().norm()),
        "P2_spatial_output_bias": float(gate.net[-1].bias.detach().mean()),
        "note": "Descriptive state only; parameter magnitude is not causal evidence.",
    }
    del trained
    core.save_report(report)


def install_overrides():
    if not MODEL_YAML.is_file() or not GI_RSG_SOURCE.is_file():
        raise FileNotFoundError("E25a companion YAML/gi_rsg_loss.py missing")

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
