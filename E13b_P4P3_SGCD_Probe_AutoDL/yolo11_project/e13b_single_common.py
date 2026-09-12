"""E13b single-path probe, paired against the already completed C13 control.

E13b starts from the exact same converged E10a checkpoint and optimizes the
same P2/P3 classification towers as C13. Its only extra trainable component
is one P4->P3 classification-exclusive semantic adapter. E10a's existing
P3/P2 spatial gates remain unchanged; no new P3->P2 adapter is present.
"""
from __future__ import annotations

import copy
import gc
import json
import math
import os
import shutil
import sys
import types
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
E10A_BEST = core.ROOT / "runs/e10a_yolo11s_p2_dssg_img1024_seed1/weights/best.pt"
E10A_BEST_SHA256 = "8c92316c138ea01bdbe3eacf3d275c9be15d96231c13c65818811bcc3cd2d512"
E10A_YAML = HERE / "yolo11s-p2-dssg.yaml"
SINGLE_YAML = HERE / "yolo11s-p2-p4p3-sgcd-probe.yaml"
DSSG_SOURCE = core.SOURCE / "ultralytics/nn/modules/dssg.py"
SINGLE_SOURCE = core.SOURCE / "ultralytics/nn/modules/p4p3_sgcd_head.py"
MATCHING_HELPER = Path(fast_tal.__file__).resolve()
PROBE_EPOCHS = 30
FROZEN_COMPLETE_LAYERS = list(range(28))

_BASE_TRAIN = e10a._CORE_TRAIN
_BASE_EVALUATE = e10a._CORE_EVALUATE
_BASE_TRAINING_PARAMS = core.training_params


def settings(variant):
    if variant == "control":
        return {
            "experiment": "c13_control_e10a_clsft30_seed1",
            "revision": "paired_e10a_p2p3_classification_finetune_control_v1",
            "yaml": E10A_YAML,
            "gate": core.ROOT / "comparison_reports/c13_control_smoke_passed.json",
        }
    if variant == "single":
        return {
            "experiment": "e13b_probe_yolo11s_p2_p4p3_sgcd_img1024_seed1",
            "revision": "p4p3_only_sgcd_paired_probe30_v1",
            "yaml": SINGLE_YAML,
            "gate": core.ROOT / "comparison_reports/e13b_p4p3_sgcd_probe_smoke_passed.json",
        }
    raise ValueError(f"Unknown paired variant: {variant}")


def trainable_parameter(name, variant):
    class_tower = name.startswith("model.28.cv3.0.") or name.startswith("model.28.cv3.1.")
    adapter = variant == "single" and name.startswith("model.28.semantic_adapter.")
    return class_tower or adapter


def apply_trainable_scope(model, variant):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = trainable_parameter(name, variant)


def freeze_untrained_batchnorm(model):
    import torch.nn as nn

    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) and not (
            name.startswith("model.28.cv3.0.") or name.startswith("model.28.cv3.1.")
        ):
            module.eval()


def check_model(model, variant):
    from ultralytics.nn.modules import DeepSemanticGuide, SemanticDetailGate

    if len(model.model) != 29 or model.yaml.get("scale") != "s":
        raise ValueError("Expected a 29-layer s-scale graph")
    if not isinstance(model.model[26], DeepSemanticGuide) or model.model[26].f != [16, 19, 22]:
        raise ValueError("Layer 26 must remain the E10a DeepSemanticGuide")
    if not isinstance(model.model[27], SemanticDetailGate) or model.model[27].f != [25, 26]:
        raise ValueError("Layer 27 must remain the E10a SemanticDetailGate")
    detect = model.model[28]
    expected_head = "Detect" if variant == "control" else "P4P3SGCDDetect"
    if detect.__class__.__name__ != expected_head:
        raise ValueError(f"Expected {expected_head}, got {detect.__class__.__name__}")
    if detect.f != [27, 26, 19, 22] or detect.nc != 10:
        raise ValueError("Detect inputs/classes changed")
    if [float(value) for value in model.stride.cpu().tolist()] != [4.0, 8.0, 16.0, 32.0]:
        raise ValueError(f"Unexpected strides: {model.stride}")
    for index in (11, 14, 23):
        layer = model.model[index]
        if layer.__class__.__name__ != "Upsample" or layer.mode != "nearest" or layer.scale_factor != 2.0:
            raise ValueError(f"Layer {index} must remain nearest-neighbour 2x upsampling")
    forbidden = {layer.__class__.__name__ for layer in model.model} & {
        "PGALP", "P3SDE", "DySample", "LocalDiscriminativeSemanticGuide"
    }
    if forbidden:
        raise ValueError(f"Forbidden branch present: {sorted(forbidden)}")
    if variant == "single":
        adapter = detect.semantic_adapter
        widths = (adapter.target_channels, adapter.source_channels, adapter.hidden_channels)
        if widths != (128, 256, 32):
            raise ValueError(f"Unexpected P4->P3 SGCD runtime widths: {widths}")


def validate_config(cfg, variant):
    from ultralytics.utils import yaml_load

    reference = yaml_load(E10A_YAML)
    if cfg.get("backbone") != reference.get("backbone"):
        raise ValueError("Backbone differs from E10a")
    if cfg.get("nc") != reference.get("nc") or cfg.get("scales") != reference.get("scales"):
        raise ValueError("Dataset/model scaling differs from E10a")
    expected_head = copy.deepcopy(reference["head"])
    if variant == "single":
        expected_head[-1] = [[27, 26, 19, 22], 1, "P4P3SGCDDetect", ["nc", 0.25, 0.0625, 0.0]]
    if cfg.get("head") != expected_head:
        raise ValueError("Paired probe changed a graph component outside the final Detect class")


def load_e10a_state(target, variant):
    if not E10A_BEST.is_file() or core.sha256(E10A_BEST) != E10A_BEST_SHA256:
        raise RuntimeError("Frozen E10a best.pt is missing or changed")
    source = core.YOLO(str(E10A_BEST)).model.float()
    if source.model[28].__class__.__name__ != "Detect":
        raise ValueError("Reference checkpoint is not the audited E10a model")
    source_state, target_state = source.state_dict(), target.state_dict()
    mapping = {}
    for key, value in source_state.items():
        if key not in target_state or tuple(value.shape) != tuple(target_state[key].shape):
            raise RuntimeError(f"E10a tensor cannot transfer exactly: {key}")
        mapping[key] = value
    result = target.load_state_dict(mapping, strict=False)
    expected_missing = {
        key for key in target_state if variant == "single" and key.startswith("model.28.semantic_adapter.")
    }
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise RuntimeError(f"Unexpected E10a transfer result: {result}")
    actual = target.state_dict()
    unequal = [key for key, value in mapping.items() if not core.torch.equal(value.cpu(), actual[key].cpu())]
    if unequal:
        raise RuntimeError(f"Transferred tensor mismatch: {unequal[:5]}")
    audit = {
        "source": str(E10A_BEST),
        "source_sha256": E10A_BEST_SHA256,
        "loaded_tensors": len(mapping),
        "verified_tensor_equality": True,
        "new_tensors": sorted(expected_missing),
        "initialization": "E10a-exact; the P4->P3 SGCD residual gain is zero" if variant == "single" else "E10a-exact",
    }
    del source, source_state, target_state, actual, mapping
    gc.collect()
    return audit


def initial_equivalence(target, variant):
    source = core.YOLO(str(E10A_BEST)).model.float().eval()
    target.eval()
    core.torch.manual_seed(131313)
    sample = core.torch.randn(1, 3, 256, 256)
    with core.torch.inference_mode():
        source_pred, source_raw = source(sample)
        target_pred, target_raw = target(sample)
    pred_delta = (source_pred.float() - target_pred.float()).abs()
    raw_max = max(
        float((left.float() - right.float()).abs().max())
        for left, right in zip(source_raw, target_raw)
    )
    result = {
        "prediction_mean_abs_delta": float(pred_delta.mean()),
        "prediction_max_abs_delta": float(pred_delta.max()),
        "raw_output_max_abs_delta": raw_max,
    }
    if result["prediction_max_abs_delta"] > 1e-6 or raw_max > 1e-6:
        raise RuntimeError(f"Probe does not initialize exactly as E10a: {result}")

    if variant == "single":
        detect = target.model[28]
        with core.torch.no_grad():
            detect.semantic_adapter.gain.fill_(0.01)
        with core.torch.inference_mode():
            _, opened_raw = target(sample)
        box_delta = max(
            float((left[:, : detect.reg_max * 4] - right[:, : detect.reg_max * 4]).abs().max())
            for left, right in zip(source_raw, opened_raw)
        )
        class_delta = max(
            float((left[:, detect.reg_max * 4 :] - right[:, detect.reg_max * 4 :]).abs().max())
            for left, right in zip(source_raw, opened_raw)
        )
        if box_delta != 0.0 or class_delta <= 0.0:
            raise RuntimeError(f"P4->P3 SGCD path isolation failed: box={box_delta}, cls={class_delta}")
        result["opened_gain_check"] = {
            "box_raw_max_abs_delta": box_delta,
            "class_raw_max_abs_delta": class_delta,
        }
        with core.torch.no_grad():
            detect.semantic_adapter.gain.zero_()
    del source, sample, source_pred, source_raw, target_pred, target_raw
    gc.collect()
    return result


def gradient_check(model, variant):
    apply_trainable_scope(model, variant)
    model.train()
    if not hasattr(model, "args"):
        model.args = types.SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    freeze_untrained_batchnorm(model)
    batch = {
        "img": core.torch.rand(2, 3, 256, 256),
        "batch_idx": core.torch.tensor([0, 0, 1]),
        "cls": core.torch.tensor([[0.0], [3.0], [1.0]]),
        "bboxes": core.torch.tensor([
            [0.25, 0.25, 0.035, 0.030],
            [0.70, 0.65, 0.12, 0.10],
            [0.55, 0.40, 0.05, 0.04],
        ]),
    }
    total, items = model(batch)
    total.backward()
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    frozen_with_grad = [
        name for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if frozen_with_grad:
        raise RuntimeError(f"Frozen tensors received gradients: {frozen_with_grad[:5]}")
    class_grad = sum(
        float(parameter.grad.detach().abs().sum())
        for name, parameter in trainable.items()
        if ".cv3." in name and parameter.grad is not None
    )
    if class_grad <= 0:
        raise RuntimeError("P2/P3 classification towers received no gradient")
    result = {
        "loss_items": [float(value) for value in items],
        "trainable_parameters": sum(parameter.numel() for parameter in trainable.values()),
        "trainable_tensor_names": sorted(trainable),
        "classification_gradient_abs_sum": class_grad,
        "regression_tensors_trainable": any(".cv2." in name for name in trainable),
    }
    if result["regression_tensors_trainable"]:
        raise RuntimeError("Regression path must remain frozen")
    if variant == "single":
        gain_grad = sum(
            float(parameter.grad.detach().abs().sum())
            for name, parameter in trainable.items()
            if name.endswith(".gain") and parameter.grad is not None
        )
        if gain_grad <= 0:
            raise RuntimeError("Zero-initialized P4->P3 SGCD gain received no gradient")
        result["adapter_gain_gradient_abs_sum"] = gain_grad
    model.zero_grad(set_to_none=True)
    return result


def training_params(variant):
    params = _BASE_TRAINING_PARAMS()
    params.update(
        epochs=PROBE_EPOCHS,
        name=settings(variant)["experiment"],
        freeze=FROZEN_COMPLETE_LAYERS,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.1,
        warmup_epochs=1.0,
        close_mosaic=5,
    )
    return params


def make_pair_trainer(report, variant):
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.nn.tasks import DetectionModel

    class PairedClassificationTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if weights is not None:
                raise RuntimeError("Paired probe performs its own exact E10a transfer")
            model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose)
            check_model(model, variant)
            report["pretrained_transfer"] = load_e10a_state(model, variant)
            apply_trainable_scope(model, variant)
            core.write_json(Path(report["report_dir"]) / "pretrained_transfer.json", report["pretrained_transfer"])
            core.save_report(report)
            return model

        def build_optimizer(self, model, *args, **kwargs):
            # Ultralytics re-enables all parameters in layer 28 after get_model().
            # Re-apply the exact paired scope immediately before optimizer creation.
            apply_trainable_scope(model, variant)
            optimizer = super().build_optimizer(model, *args, **kwargs)
            # Ultralytics 8.3.0 registers every model tensor in optimizer groups,
            # including requires_grad=False tensors. Such tensors have no gradient
            # and optimizer.step() skips them. Audit requires_grad here instead of
            # incorrectly requiring optimizer groups to exclude frozen tensors.
            trainable_names = {
                name for name, parameter in model.named_parameters() if parameter.requires_grad
            }
            unexpected = sorted(name for name in trainable_names if not trainable_parameter(name, variant))
            if not trainable_names or unexpected:
                raise RuntimeError(f"Unexpected trainable parameter scope: {unexpected[:5]}")
            if any(".cv2." in name for name in trainable_names):
                raise RuntimeError("Regression cv2 entered the trainable scope")
            return optimizer

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            apply_trainable_scope(self.model, variant)
            freeze_untrained_batchnorm(self.model)
            return batch

    return PairedClassificationTrainer


def completed_reference(pattern):
    candidates = sorted(
        (path for path in (core.ROOT / "comparison_reports").glob(pattern) if "SMOKE" not in str(path).upper()),
        key=lambda path: path.stat().st_mtime,
    )
    for path in reversed(candidates):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") == "completed":
            return path, data
    return None


def metric_deltas(current, reference):
    sections = {
        "overall": ("Precision", "Recall", "mAP50", "mAP75", "mAP50_95"),
        "area_metrics": ("AP_all", "AR_all", "AP_small", "AR_small", "AP_medium", "AP_large"),
    }
    return {
        section: {name: float(current[section][name]) - float(reference[section][name]) for name in names}
        for section, names in sections.items()
    }


def p2_preflight(report, variant):
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils import yaml_load

    cfg_path = settings(variant)["yaml"]
    required = [cfg_path, E10A_YAML, E10A_BEST, DSSG_SOURCE, MATCHING_HELPER]
    if variant == "single":
        required.append(SINGLE_SOURCE)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if core.sha256(E10A_BEST) != E10A_BEST_SHA256:
        raise RuntimeError("Frozen E10a best.pt differs from the audited checkpoint")
    cfg = yaml_model_load(str(cfg_path))
    validate_config(cfg, variant)
    model = DetectionModel(cfg, nc=10, verbose=False)
    check_model(model, variant)
    report["is_formal"] = False
    report["is_probe"] = True
    report["p2_script_revision"] = settings(variant)["revision"]
    report["p2_architecture"] = {
        "experimental_hierarchy": {"E0": "original YOLO11s baseline", "E1": "YOLO11s + P2 unified base"},
        "probe_initialization": "same frozen E10a best.pt and P2/P3 cv3 scope as completed C13; E1 remains the base",
        "variant": variant,
        "yaml": str(cfg_path),
        "yaml_sha256": core.sha256(cfg_path),
        "unchanged": ["E10a checkpoint tensors", "P2/P3/P4/P5 box regression", "nearest upsampling", "TAL2"],
        "trainable_scope": "P2/P3 cv3 classification towers" + (" plus one P4->P3 SGCD adapter" if variant == "single" else " only"),
        "pair_rule": "E13b minus the completed C13 is the causal single-adapter delta",
        "automatic_shutdown": False,
    }
    report["preflight_transfer"] = load_e10a_state(model, variant)
    report["initial_equivalence"] = initial_equivalence(model, variant)
    # initial_equivalence opens gains on its temporary model; restore exact state.
    report["preflight_transfer_after_equivalence"] = load_e10a_state(model, variant)
    report["gradient_check"] = gradient_check(model, variant)
    out = Path(report["report_dir"])
    for path in required:
        shutil.copy2(path, out / path.name)
    core.save_report(report)
    del model
    gc.collect()

    baseline_cfg = yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml")
    baseline_cfg.update(scale="s", nc=10)
    print("Checking strict TAL2 equivalence on CPU; CUDA batch8/1024 is checked by --smoke2.", flush=True)
    report["matching_equivalence"] = fast_tal.verify_equivalence(
        copy.deepcopy(cfg), device="cpu", baseline_cfg=baseline_cfg
    )
    fast_tal.assert_patch_restoration()
    core.write_json(out / "tal_equivalence.json", report["matching_equivalence"])
    core.save_report(report)
    print(f"{variant} structure/transfer/isolation/gradient/TAL2 checks passed.", flush=True)


def train(report, smoke, variant):
    weights = _BASE_TRAIN(report, smoke=smoke)
    run_dir = Path(report["run_dir"])
    for path in (DSSG_SOURCE, MATCHING_HELPER, E10A_YAML):
        shutil.copy2(path, run_dir / path.name)
    if variant == "single":
        shutil.copy2(SINGLE_SOURCE, run_dir / SINGLE_SOURCE.name)
    return weights


def evaluate(report, weights, dataset, variant):
    _BASE_EVALUATE(report, weights, dataset)
    references = {
        "E1_base": "e1_yolo11s_p2add_img1024_seed1_*/metrics.json",
        "E10a_candidate": "e10a_yolo11s_p2_dssg_img1024_seed1_*/metrics.json",
    }
    if variant == "single":
        references["C13_paired_control"] = "c13_control_e10a_clsft30_seed1_*/metrics.json"
    comparisons = {}
    for label, pattern in references.items():
        reference = completed_reference(pattern)
        if reference is None:
            comparisons[label] = {"status": "unavailable", "pattern": pattern}
        else:
            path, values = reference
            comparisons[label] = {
                "status": "computed", "path": str(path), "deltas": metric_deltas(report, values)
            }
    report["probe_comparisons"] = comparisons
    if variant == "single" and comparisons.get("C13_paired_control", {}).get("status") == "computed":
        delta = comparisons["C13_paired_control"]["deltas"]
        report["paired_screen"] = {
            "passed": bool(
                delta["area_metrics"]["AP_small"] >= 0.0015
                and delta["overall"]["mAP50_95"] >= 0.0010
                and delta["overall"]["mAP75"] >= 0.0
            ),
            "required": {"AP_small": 0.0015, "mAP50_95": 0.0010, "mAP75_floor": 0.0},
            "observed": {
                "AP_small": delta["area_metrics"]["AP_small"],
                "mAP50_95": delta["overall"]["mAP50_95"],
                "mAP75": delta["overall"]["mAP75"],
            },
            "decision": "A pass retains P4->P3 for refinement; a failure ends direct SGCD injection without a 200-epoch run.",
        }
    trained = core.YOLO(str(weights)).model.float()
    check_model(trained, variant)
    apply_trainable_scope(trained, variant)
    report["trained_scope"] = {
        "trainable_parameters": sum(p.numel() for p in trained.parameters() if p.requires_grad),
        "regression_trainable": any(
            p.requires_grad for name, p in trained.named_parameters() if ".cv2." in name
        ),
    }
    if variant == "single":
        report["sgcd_learned_state"] = {
            "P4_to_P3_gain": {
                "mean": float(trained.model[28].semantic_adapter.gain.detach().mean()),
                "std": float(trained.model[28].semantic_adapter.gain.detach().std()),
                "l2": float(trained.model[28].semantic_adapter.gain.detach().norm()),
            },
            "note": "Nonzero gains prove use, not accuracy.",
        }
    core.save_report(report)
    del trained
    core.clear_gpu()


def run(variant, runner_file):
    cfg = settings(variant)
    e10a.install_overrides()
    core.EXPERIMENT = cfg["experiment"]
    core.SCRIPT_REVISION = cfg["revision"]
    core.MODEL_YAML = cfg["yaml"]
    core.PRETRAINED = E10A_BEST
    core.PRETRAINED_SHA256 = E10A_BEST_SHA256
    core.MATCHING_HELPER = MATCHING_HELPER
    core.SMOKE_GATE = cfg["gate"]
    core.EPOCHS = PROBE_EPOCHS
    core.AUTO_SHUTDOWN = False
    core.SHUTDOWN_ON_FAILURE = False
    core.__file__ = str(Path(runner_file).resolve())
    core.training_params = lambda: training_params(variant)
    core.make_p2_trainer = lambda report: make_pair_trainer(report, variant)
    core.load_p2_pretrained = lambda target: load_e10a_state(target, variant)
    core.assert_p2_model = lambda model: check_model(model, variant)
    core.validate_p2_config = lambda cfg_value, _original: validate_config(cfg_value, variant)
    core.p2_preflight = lambda report, eval_only=False: p2_preflight(report, variant)
    core.train = lambda report, smoke=False: train(report, smoke, variant)
    core.evaluate = lambda report, weights, dataset: evaluate(report, weights, dataset, variant)
    core.main()
