#!/usr/bin/env python3
"""E11a: 30-epoch CBSG gate probe initialized from the frozen E10a best.pt.

Only the P3/P2 spatial-gate parameters at layers 26/27 are optimized. The
backbone, original neck, P2 branch, semantic projections and Detect head are
frozen. This is a screening probe, not a formal 200-epoch comparison.
Automatic shutdown is disabled on every path.
"""
from __future__ import annotations

import copy
import gc
import json
import math
import os
import shutil
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
if not os.environ.get("OMP_NUM_THREADS", "").isdigit() or int(os.environ.get("OMP_NUM_THREADS", "0")) < 1:
    os.environ["OMP_NUM_THREADS"] = "8"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import p2_tal_chunked_vgpu32 as fast_tal
sys.modules["p2_tal_chunked"] = fast_tal
import yolo11s_p2_dssg_img1024_seed1 as e10a


core = e10a.core
EXPERIMENT = "e11a_probe_yolo11s_p2_cbsg_img1024_seed1"
SCRIPT_REVISION = "cbsg_e10a_gate_only_probe30_v2"
MODEL_YAML = HERE / "yolo11s-p2-cbsg-probe.yaml"
E10A_YAML = HERE / "yolo11s-p2-dssg.yaml"
E10A_BEST = core.ROOT / "runs/e10a_yolo11s_p2_dssg_img1024_seed1/weights/best.pt"
E10A_BEST_SHA256 = "8c92316c138ea01bdbe3eacf3d275c9be15d96231c13c65818811bcc3cd2d512"
MODULE_SOURCE = core.SOURCE / "ultralytics/nn/modules/cbsg.py"
LOSS_SOURCE = core.SOURCE / "ultralytics/utils/cbsg_loss.py"
MATCHING_HELPER = Path(fast_tal.__file__).resolve()
SMOKE_GATE = core.ROOT / "comparison_reports/e11a_cbsg_probe_smoke_passed.json"
PROBE_EPOCHS = 30
FROZEN_LAYERS = list(range(26)) + [28]
TRAINABLE_LAYERS = [26, 27]

_BASE_TRAIN = e10a._CORE_TRAIN
_BASE_EVALUATE = e10a._CORE_EVALUATE
_BASE_TRAINING_PARAMS = core.training_params


def install_criterion_override():
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.cbsg_loss import v8CBSGDetectionLoss

    if not hasattr(DetectionModel, "_e11a_original_init_criterion"):
        DetectionModel._e11a_original_init_criterion = DetectionModel.init_criterion

    def init_criterion(model):
        config = model.yaml.get("cbsg", {})
        if isinstance(config, dict) and config.get("enabled", False):
            return v8CBSGDetectionLoss(model)
        return model._e11a_original_init_criterion()

    DetectionModel.init_criterion = init_criterion


def assert_probe_model(model):
    from ultralytics.nn.modules import CBSGDeepSemanticGuide, CBSGDetailGate

    if len(model.model) != 29 or model.yaml.get("scale") != "s":
        raise ValueError("Expected 29-layer s-scale E11a architecture")
    if not isinstance(model.model[26], CBSGDeepSemanticGuide):
        raise ValueError(f"Layer 26 is {type(model.model[26]).__name__}")
    if not isinstance(model.model[27], CBSGDetailGate):
        raise ValueError(f"Layer 27 is {type(model.model[27]).__name__}")
    if model.model[28].f != [27, 26, 19, 22] or model.model[28].nc != 10:
        raise ValueError("E11a Detect inputs/classes differ from E10a")
    strides = [float(value) for value in model.stride.detach().cpu().tolist()]
    if strides != [4.0, 8.0, 16.0, 32.0]:
        raise ValueError(f"Unexpected strides: {strides}")


def architecture_record():
    return {
        "yaml": str(MODEL_YAML), "yaml_sha256": core.sha256(MODEL_YAML),
        "reference": "frozen E10a best.pt", "reference_checkpoint": str(E10A_BEST),
        "reference_sha256": core.sha256(E10A_BEST),
        "unique_change": (
            "E10a P3/P2 sigmoid gates receive learnable local-background contrast; "
            "legacy P4/P5 pooled channel output is fixed at neutral 0.5; training-only "
            "scale-weighted target-centre gate supervision"
        ),
        "unchanged": [
            "YOLO11s backbone and original neck", "E1 P2 branch", "all nearest upsampling",
            "E10a P4/P3/P2 semantic projections and residual scales", "Detect head and TAL2",
        ],
        "probe": {
            "epochs": PROBE_EPOCHS, "frozen_layers": FROZEN_LAYERS,
            "trainable_layers": TRAINABLE_LAYERS,
            "fine_grained_trainable": "only layer26/27 spatial_gate parameters",
            "lr0": 0.001, "lrf": 0.1,
        },
        "automatic_shutdown": False,
    }


def validate_config(cfg, _original_cfg):
    from ultralytics.utils import yaml_load

    reference = yaml_load(E10A_YAML)
    if cfg.get("nc") != reference.get("nc") or cfg.get("scales") != reference.get("scales"):
        raise ValueError("E11a nc/scales differ from E10a")
    if cfg.get("backbone") != reference.get("backbone"):
        raise ValueError("E11a backbone differs from E10a")
    if cfg["head"][:15] != reference["head"][:15] or cfg["head"][-1] != reference["head"][-1]:
        raise ValueError("E11a changed paths outside DSSG layers 26/27")
    if cfg["head"][15][2] != "CBSGDeepSemanticGuide" or cfg["head"][16][2] != "CBSGDetailGate":
        raise ValueError("E11a YAML does not contain the audited CBSG replacements")
    expected_cbsg = {
        "enabled": True, "p3_loss_weight": 0.02, "p2_loss_weight": 0.04,
        "focal_gamma": 2.0, "negative_balance": 0.25, "small_boost": 1.5,
        "p3_dilation": 3, "p2_dilation": 5,
    }
    if cfg.get("cbsg") != expected_cbsg:
        raise ValueError(f"Unexpected cbsg loss configuration: {cfg.get('cbsg')}")


def load_e10a_state(target):
    if not E10A_BEST.is_file() or core.sha256(E10A_BEST) != E10A_BEST_SHA256:
        raise RuntimeError("Frozen E10a best.pt is missing or changed")
    source = core.YOLO(str(E10A_BEST)).model.float()
    if source.model[26].__class__.__name__ != "DeepSemanticGuide" or source.model[27].__class__.__name__ != "SemanticDetailGate":
        raise ValueError("Reference checkpoint is not the audited E10a model")
    source_state = source.state_dict()
    result = target.load_state_dict(source_state, strict=False)
    expected_missing = {
        "model.26.spatial_gate.contrast_logit", "model.26.spatial_gate.log_temperature",
        "model.27.spatial_gate.contrast_logit", "model.27.spatial_gate.log_temperature",
    }
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise RuntimeError(f"Unexpected E10a->E11a transfer result: {result}")
    target_state = target.state_dict()
    unequal = [key for key, value in source_state.items() if key not in target_state or not core.torch.equal(value.cpu(), target_state[key].cpu())]
    if unequal:
        raise RuntimeError(f"E10a tensor transfer mismatch: {unequal[:5]}")
    audit = {
        "source": str(E10A_BEST), "source_sha256": E10A_BEST_SHA256,
        "loaded_tensors": len(source_state), "verified_tensor_equality": True,
        "new_parameters": sorted(expected_missing),
        "new_parameter_initialization": {"contrast_logit": -8.0, "log_temperature": 0.0},
    }
    del source, source_state, target_state
    gc.collect()
    return audit


def load_p2_pretrained(target):
    return load_e10a_state(target)


def training_params():
    params = _BASE_TRAINING_PARAMS()
    params.update(
        epochs=PROBE_EPOCHS, name=EXPERIMENT, freeze=FROZEN_LAYERS,
        lr0=0.001, lrf=0.1, warmup_epochs=1.0, close_mosaic=5,
    )
    return params


def freeze_gate_only(model):
    for index in TRAINABLE_LAYERS:
        for name, parameter in model.model[index].named_parameters():
            parameter.requires_grad = name.startswith("spatial_gate.")


def make_probe_trainer(report):
    import torch.nn as nn
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.nn.tasks import DetectionModel

    class ProbeTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if weights is not None:
                raise RuntimeError("E11a explicitly transfers the frozen E10a best checkpoint")
            model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose)
            assert_probe_model(model)
            report["pretrained_transfer"] = load_e10a_state(model)
            core.write_json(Path(report["report_dir"]) / "pretrained_transfer.json", report["pretrained_transfer"])
            core.save_report(report)
            return model

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            # Ultralytics applies layer-level freeze before building the
            # optimizer. Enforce the finer spatial-gate-only boundary here.
            freeze_gate_only(self.model)
            for index in FROZEN_LAYERS:
                for module in self.model.model[index].modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
            for index in TRAINABLE_LAYERS:
                for name, module in self.model.model[index].named_modules():
                    if isinstance(module, nn.BatchNorm2d) and not name.startswith("spatial_gate."):
                        module.eval()
            return batch

    return ProbeTrainer


def initial_equivalence(target):
    source = core.YOLO(str(E10A_BEST)).model.float().eval()
    target.eval()
    core.torch.manual_seed(24680)
    sample = core.torch.randn(1, 3, 256, 256)
    with core.torch.inference_mode():
        source_output = source(sample)[0]
        target_output = target(sample)[0]
    difference = (source_output.float() - target_output.float()).abs()
    result = {
        "mean_abs_prediction_delta": float(difference.mean()),
        "max_abs_prediction_delta": float(difference.max()),
        "note": "Small nonzero delta is expected from fixed 0.5 channel output and near-zero contrast gain.",
    }
    if result["mean_abs_prediction_delta"] > 0.001 or result["max_abs_prediction_delta"] > 0.1:
        raise RuntimeError(f"E11a does not start sufficiently close to E10a: {result}")
    del source, sample, source_output, target_output, difference
    gc.collect()
    return result


def loss_gradient_check(model):
    freeze_gate_only(model)
    model.train()
    batch = {
        "img": core.torch.zeros(2, 3, 256, 256),
        "batch_idx": core.torch.tensor([0, 0, 1]),
        "cls": core.torch.tensor([[0.0], [3.0], [1.0]]),
        "bboxes": core.torch.tensor([
            [0.25, 0.25, 0.035, 0.030], [0.70, 0.65, 0.12, 0.10], [0.55, 0.40, 0.05, 0.04]
        ]),
    }
    total, items = model(batch)
    if items.numel() != 3 or not core.torch.isfinite(total):
        raise RuntimeError(f"Expected standard three finite loss items, got {items}")
    total.backward()
    gradient_rows = {}
    for index in TRAINABLE_LAYERS:
        gradients = []
        for name, parameter in model.model[index].named_parameters():
            if parameter.requires_grad:
                if parameter.grad is None or not core.torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"Missing/nonfinite CBSG gradient: layer{index}.{name}")
                gradients.append(parameter.grad.detach().abs().sum())
        total_gradient = float(core.torch.stack(gradients).sum()) if gradients else 0.0
        if total_gradient <= 0:
            raise RuntimeError(f"Zero CBSG gradient at layer {index}")
        gradient_rows[f"layer_{index}"] = total_gradient
    stats = {key: float(value) for key, value in model.criterion.last_gate_stats.items()}
    result = {"loss_items": [float(value) for value in items], "total": float(total.detach()), "gradient_abs_sums": gradient_rows, **stats}
    model.zero_grad(set_to_none=True)
    for index in TRAINABLE_LAYERS:
        model.model[index].spatial_gate.clear_gate_logits()
    return result


def p2_preflight(report, eval_only=False):
    del eval_only
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils import yaml_load

    print(f"E11a CBSG probe revision: {SCRIPT_REVISION}", flush=True)
    required = (MODEL_YAML, E10A_YAML, E10A_BEST, MODULE_SOURCE, LOSS_SOURCE, MATCHING_HELPER)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    cfg = yaml_model_load(str(MODEL_YAML))
    validate_config(cfg, yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml"))
    model = DetectionModel(cfg, nc=10, verbose=False)
    assert_probe_model(model)
    report["is_formal"] = False
    report["is_probe"] = True
    report["p2_architecture"] = architecture_record()
    report["e11a_probe"] = {
        "epochs": PROBE_EPOCHS, "start_checkpoint": str(E10A_BEST),
        "trainable_scope": "only P3/P2 spatial gates", "automatic_shutdown": False,
        "success_screen": "versus E10a: AP_small >= +0.0015 and mAP50_95 not below -0.0010",
    }
    report["preflight_transfer"] = load_e10a_state(model)
    report["initial_equivalence"] = initial_equivalence(model)
    report["loss_gradient_check"] = loss_gradient_check(model)
    model.eval()
    with core.torch.inference_mode():
        _, raw = model(core.torch.zeros(1, 3, 256, 256))
    sizes = [list(value.shape[-2:]) for value in raw]
    if sizes != [[64, 64], [32, 32], [16, 16], [8, 8]]:
        raise RuntimeError(f"Unexpected feature shapes: {sizes}")
    report["CPU_forward_256_feature_shapes"] = sizes
    out = Path(report["report_dir"])
    for path in required:
        shutil.copy2(path, out / path.name)
    core.save_report(report)
    del model, raw
    gc.collect()

    baseline_cfg = yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml")
    baseline_cfg.update(scale="s", nc=10)
    report["matching_equivalence"] = fast_tal.verify_equivalence(copy.deepcopy(cfg), device="cpu", baseline_cfg=baseline_cfg)
    fast_tal.assert_patch_restoration()
    core.write_json(out / "tal_equivalence.json", report["matching_equivalence"])
    core.save_report(report)
    print("E11a structure/E10a-transfer/CBSG-loss-gradient/TAL2 checks passed.", flush=True)


def train(report, smoke=False):
    weights = _BASE_TRAIN(report, smoke=smoke)
    run_dir = Path(report["run_dir"])
    for path in (MODULE_SOURCE, LOSS_SOURCE, MATCHING_HELPER, E10A_YAML):
        shutil.copy2(path, run_dir / path.name)
    report["e11a_probe"]["gate_parameter_count"] = sum(
        parameter.numel()
        for index in TRAINABLE_LAYERS
        for name, parameter in core.YOLO(str(weights)).model.model[index].named_parameters()
        if name.startswith("spatial_gate.")
    )
    core.save_report(report)
    return weights


def completed_reference(pattern):
    candidates = [path for path in (core.ROOT / "comparison_reports").glob(pattern) if "SMOKE" not in str(path).upper()]
    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime)
    for path in reversed(candidates):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "completed":
            return path, value
    return None


def metric_deltas(current, reference):
    sections = {
        "overall": ("Precision", "Recall", "mAP50", "mAP75", "mAP50_95"),
        "area_metrics": ("AP_all", "AP_small", "AP_medium", "AP_large"),
    }
    return {
        section: {name: float(current[section][name]) - float(reference[section][name]) for name in names}
        for section, names in sections.items()
    }


def evaluate(report, weights, dataset):
    _BASE_EVALUATE(report, weights, dataset)
    comparisons = {}
    references = {
        "E10a": "e10a_yolo11s_p2_dssg_img1024_seed1_*/metrics.json",
        "E1": "e1_yolo11s_p2add_img1024_seed1_*/metrics.json",
    }
    for label, pattern in references.items():
        reference = completed_reference(pattern)
        if reference is None:
            comparisons[label] = {"status": "unavailable", "pattern": pattern}
            continue
        path, values = reference
        comparisons[label] = {"status": "computed", "path": str(path), "deltas": metric_deltas(report, values)}
    report["probe_comparisons"] = comparisons
    if comparisons.get("E10a", {}).get("status") == "computed":
        delta = comparisons["E10a"]["deltas"]
        report["e11a_probe"]["screen_result"] = {
            "passed": delta["area_metrics"]["AP_small"] >= 0.0015 and delta["overall"]["mAP50_95"] >= -0.0010,
            "AP_small_required": 0.0015,
            "mAP50_95_floor": -0.0010,
            "observed_AP_small_delta": delta["area_metrics"]["AP_small"],
            "observed_mAP50_95_delta": delta["overall"]["mAP50_95"],
            "note": "Passing authorizes a fresh 200-epoch formal run; this probe is not the final paper result.",
        }
    trained = core.YOLO(str(weights)).model.float()
    assert_probe_model(trained)
    report["cbsg_learned_state"] = {}
    for label, index in (("P3", 26), ("P2", 27)):
        gate = trained.model[index].spatial_gate
        report["cbsg_learned_state"][label] = {
            "contrast_logit": float(gate.contrast_logit.detach()),
            "contrast_gain": float(gate.contrast_gain().detach()),
            "temperature": float(gate.temperature().detach()),
            "gate_output_weight_l2": float(gate.net[-1].weight.detach().norm()),
        }
    core.save_report(report)
    del trained
    core.clear_gpu()


def install_overrides():
    # First establish E10a's audited runner behavior, then narrow it to E11a.
    e10a.install_overrides()
    install_criterion_override()
    core.EXPERIMENT = EXPERIMENT
    core.SCRIPT_REVISION = SCRIPT_REVISION
    core.MODEL_YAML = MODEL_YAML
    core.PRETRAINED = E10A_BEST
    core.PRETRAINED_SHA256 = E10A_BEST_SHA256
    core.MATCHING_HELPER = MATCHING_HELPER
    core.SMOKE_GATE = SMOKE_GATE
    core.EPOCHS = PROBE_EPOCHS
    core.AUTO_SHUTDOWN = False
    core.SHUTDOWN_ON_FAILURE = False
    core.__file__ = str(Path(__file__).resolve())
    core.training_params = training_params
    core.make_p2_trainer = make_probe_trainer
    core.load_p2_pretrained = load_p2_pretrained
    core.assert_p2_model = assert_probe_model
    core.validate_p2_config = validate_config
    core.p2_preflight = p2_preflight
    core.train = train
    core.evaluate = evaluate


if __name__ == "__main__":
    install_overrides()
    core.main()
