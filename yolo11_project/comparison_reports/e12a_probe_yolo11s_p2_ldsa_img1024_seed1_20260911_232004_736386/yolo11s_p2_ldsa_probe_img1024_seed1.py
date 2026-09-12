#!/usr/bin/env python3
"""E12a: 30-epoch LDSA probe initialized from the frozen E10a best.pt.

D12 found that E10a primarily reduces classification/background errors while
leaving localization unchanged.  E12a therefore keeps both causally useful
E10a spatial gates and replaces only the inactive global pooled-channel path
with local P3-query/P4-semantic attention.  Only the new attention parameters
are optimized; all pre-existing E10a tensors and BatchNorm statistics remain
frozen.  This is a screening probe, not a formal 200-epoch result.

Automatic shutdown is disabled on check, smoke, probe, evaluation and failure.
"""
from __future__ import annotations

import copy
import gc
import json
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
EXPERIMENT = "e12a_probe_yolo11s_p2_ldsa_img1024_seed1"
SCRIPT_REVISION = "ldsa_local_p3_query_p4_semantic_probe30_v1"
MODEL_YAML = HERE / "yolo11s-p2-ldsa-probe.yaml"
E10A_YAML = HERE / "yolo11s-p2-dssg.yaml"
E10A_BEST = core.ROOT / "runs/e10a_yolo11s_p2_dssg_img1024_seed1/weights/best.pt"
E10A_BEST_SHA256 = "8c92316c138ea01bdbe3eacf3d275c9be15d96231c13c65818811bcc3cd2d512"
MODULE_SOURCE = core.SOURCE / "ultralytics/nn/modules/ldsa.py"
E10A_MODULE_SOURCE = core.SOURCE / "ultralytics/nn/modules/dssg.py"
MATCHING_HELPER = Path(fast_tal.__file__).resolve()
SMOKE_GATE = core.ROOT / "comparison_reports/e12a_ldsa_probe_smoke_passed.json"
PROBE_EPOCHS = 30
FROZEN_LAYERS = list(range(26)) + [27, 28]
TRAINABLE_LAYER = 26
TRAINABLE_PREFIX = "local_attention."

_BASE_TRAIN = e10a._CORE_TRAIN
_BASE_EVALUATE = e10a._CORE_EVALUATE
_BASE_TRAINING_PARAMS = core.training_params


def assert_probe_model(model):
    from ultralytics.nn.modules import LocalDiscriminativeSemanticGuide, SemanticDetailGate

    if len(model.model) != 29 or model.yaml.get("scale") != "s":
        raise ValueError("Expected 29-layer s-scale E12a architecture")
    deep, detail, detect = model.model[26], model.model[27], model.model[28]
    if not isinstance(deep, LocalDiscriminativeSemanticGuide) or deep.f != [16, 19]:
        raise ValueError("Layer 26 must be LocalDiscriminativeSemanticGuide(P3,P4)")
    if (deep.p3_channels, deep.p4_channels, deep.attention_channels) != (128, 256, 64):
        raise ValueError("Unexpected E12a runtime channel widths")
    if deep.heads != 4 or deep.kernel_size != 5:
        raise ValueError("E12a must use four heads and a 5x5 local neighbourhood")
    if not isinstance(detail, SemanticDetailGate) or detail.f != [25, 26]:
        raise ValueError("Layer 27 must remain the E10a SemanticDetailGate")

def _finish_assert_probe_model(model):
    """Checks separated to keep error messages precise."""
    detect = model.model[28]
    if detect.f != [27, 26, 19, 22] or detect.nc != 10:
        raise ValueError("E12a Detect inputs/classes differ from E10a")
    strides = [float(value) for value in model.stride.detach().cpu().tolist()]
    if strides != [4.0, 8.0, 16.0, 32.0]:
        raise ValueError(f"Unexpected strides: {strides}")
    for index in (11, 14, 23):
        layer = model.model[index]
        if layer.__class__.__name__ != "Upsample" or layer.mode != "nearest" or layer.scale_factor != 2.0:
            raise ValueError(f"Layer {index} must remain nearest-neighbour 2x upsampling")
    if model.model[10].__class__.__name__ != "C2PSA":
        raise ValueError("Original C2PSA changed unexpectedly")
    forbidden = {layer.__class__.__name__ for layer in model.model} & {
        "PGALP", "P3SDE", "DySample", "CBSGDeepSemanticGuide", "CBSGDetailGate"
    }
    if forbidden:
        raise ValueError(f"E12a contains forbidden experimental modules: {sorted(forbidden)}")


def check_probe_model(model):
    assert_probe_model(model)
    _finish_assert_probe_model(model)


def architecture_record():
    return {
        "yaml": str(MODEL_YAML),
        "yaml_sha256": core.sha256(MODEL_YAML),
        "reference": "frozen E10a best.pt",
        "reference_checkpoint": str(E10A_BEST),
        "reference_sha256": core.sha256(E10A_BEST),
        "diagnostic_basis": {
            "D10": "P4/P5 pooled channel gate was causally neutral; P3/P2 spatial gates were useful",
            "D12": "classification/background errors dominated localization and were reduced by E10a",
        },
        "unique_change": (
            "Replace the inactive P4/P5 pooled channel path with four-head 5x5 local "
            "cross-scale attention at P4 resolution: pooled P3 detail is query and projected "
            "P4 semantics are key/value. No additional spatial gate is added."
        ),
        "unchanged": [
            "YOLO11s backbone and original neck",
            "E1 P2 branch",
            "all nearest-neighbour upsampling",
            "E10a P3 and P2 spatial gates",
            "E10a semantic projections/refinement and residual layer scales",
            "Detect head, standard loss and TAL2",
        ],
        "probe": {
            "epochs": PROBE_EPOCHS,
            "frozen_layers": FROZEN_LAYERS,
            "trainable_layer": TRAINABLE_LAYER,
            "fine_grained_trainable": TRAINABLE_PREFIX,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.1,
        },
        "automatic_shutdown": False,
    }


def validate_config(cfg, _original_cfg):
    from ultralytics.utils import yaml_load

    reference = yaml_load(E10A_YAML)
    if cfg.get("nc") != reference.get("nc") or cfg.get("scales") != reference.get("scales"):
        raise ValueError("E12a nc/scales differ from E10a")
    if cfg.get("backbone") != reference.get("backbone"):
        raise ValueError("E12a backbone differs from E10a")
    expected_head = copy.deepcopy(reference["head"])
    expected_head[15] = [
        [16, 19], 1, "LocalDiscriminativeSemanticGuide", [0.5, 4, 5, 0.0, 0.05]
    ]
    if cfg.get("head") != expected_head:
        raise ValueError("E12a changed a head path outside the audited layer-26 replacement")
    if "cbsg" in cfg:
        raise ValueError("E12a must use the standard detection loss, not CBSG supervision")


def load_e10a_state(target):
    if not E10A_BEST.is_file() or core.sha256(E10A_BEST) != E10A_BEST_SHA256:
        raise RuntimeError("Frozen E10a best.pt is missing or changed")
    source = core.YOLO(str(E10A_BEST)).model.float()
    if source.model[26].__class__.__name__ != "DeepSemanticGuide":
        raise ValueError("Reference checkpoint is not the audited E10a model")
    if source.model[27].__class__.__name__ != "SemanticDetailGate":
        raise ValueError("Reference E10a P2 gate is invalid")

    source_state = source.state_dict()
    target_state = target.state_dict()
    removed_prefixes = (
        "model.26.p4_descriptor.",
        "model.26.p5_descriptor.",
        "model.26.channel_gate.",
    )
    mapping, removed = {}, []
    for key, value in source_state.items():
        if key in target_state:
            if tuple(value.shape) != tuple(target_state[key].shape):
                raise RuntimeError(f"Shared E10a/E12a tensor shape differs: {key}")
            mapping[key] = value
        elif key.startswith(removed_prefixes):
            removed.append(key)
        else:
            raise RuntimeError(f"Unexpected E10a-only tensor: {key}")

    result = target.load_state_dict(mapping, strict=False)
    expected_missing = {key for key in target_state if key.startswith("model.26.local_attention.")}
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise RuntimeError(f"Unexpected E10a->E12a transfer result: {result}")
    actual = target.state_dict()
    unequal = [key for key, value in mapping.items() if not core.torch.equal(value.cpu(), actual[key].cpu())]
    if unequal:
        raise RuntimeError(f"E10a tensor transfer mismatch: {unequal[:5]}")
    audit = {
        "source": str(E10A_BEST),
        "source_sha256": E10A_BEST_SHA256,
        "loaded_tensors": len(mapping),
        "verified_tensor_equality": True,
        "removed_inactive_E10a_tensors": sorted(removed),
        "new_attention_tensors": sorted(expected_missing),
        "new_attention_initialization": "local-attention residual gain is exactly zero",
    }
    del source, source_state, target_state, actual, mapping
    gc.collect()
    return audit


def load_p2_pretrained(target):
    return load_e10a_state(target)


def training_params():
    params = _BASE_TRAINING_PARAMS()
    params.update(
        epochs=PROBE_EPOCHS,
        name=EXPERIMENT,
        freeze=FROZEN_LAYERS,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.1,
        warmup_epochs=1.0,
        close_mosaic=5,
    )
    return params


def freeze_attention_only(model):
    for index, layer in enumerate(model.model):
        for name, parameter in layer.named_parameters():
            parameter.requires_grad = index == TRAINABLE_LAYER and name.startswith(TRAINABLE_PREFIX)


def set_frozen_batchnorm_eval(model):
    import torch.nn as nn

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def make_probe_trainer(report):
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.nn.tasks import DetectionModel

    class ProbeTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if weights is not None:
                raise RuntimeError("E12a explicitly transfers the frozen E10a checkpoint")
            model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose)
            check_probe_model(model)
            report["pretrained_transfer"] = load_e10a_state(model)
            freeze_attention_only(model)
            core.write_json(Path(report["report_dir"]) / "pretrained_transfer.json", report["pretrained_transfer"])
            core.save_report(report)
            return model

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            freeze_attention_only(self.model)
            set_frozen_batchnorm_eval(self.model)
            return batch

    return ProbeTrainer


def initial_equivalence(target):
    source = core.YOLO(str(E10A_BEST)).model.float().eval()
    target.eval()
    core.torch.manual_seed(121212)
    sample = core.torch.randn(1, 3, 256, 256)
    with core.torch.inference_mode():
        full_output = source(sample)[0]

        def neutral_channel_weights(module, p4, p5):
            del p5
            return p4.new_full((p4.shape[0], module.p3_channels, 1, 1), 0.5)

        source.model[26].channel_weights = types.MethodType(neutral_channel_weights, source.model[26])
        neutral_output = source(sample)[0]
        target_output = target(sample)[0]

    neutral_delta = (neutral_output.float() - target_output.float()).abs()
    full_delta = (full_output.float() - target_output.float()).abs()
    result = {
        "target_vs_D10_neutral_channel": {
            "mean_abs_prediction_delta": float(neutral_delta.mean()),
            "max_abs_prediction_delta": float(neutral_delta.max()),
        },
        "target_vs_full_E10a": {
            "mean_abs_prediction_delta": float(full_delta.mean()),
            "max_abs_prediction_delta": float(full_delta.max()),
        },
        "note": (
            "Zero attention gain makes E12a exactly the D10-tested neutral-channel graph; "
            "the full-E10a delta only reflects removal of the causally inactive channel output."
        ),
    }
    if result["target_vs_D10_neutral_channel"]["max_abs_prediction_delta"] > 1e-5:
        raise RuntimeError(f"E12a zero-gain initialization is not neutral: {result}")
    del source, sample, full_output, neutral_output, target_output, neutral_delta, full_delta
    gc.collect()
    return result


def loss_gradient_check(model):
    freeze_attention_only(model)
    model.train()
    if not hasattr(model, "args"):
        model.args = types.SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    set_frozen_batchnorm_eval(model)
    batch = {
        "img": core.torch.zeros(2, 3, 256, 256),
        "batch_idx": core.torch.tensor([0, 0, 1]),
        "cls": core.torch.tensor([[0.0], [3.0], [1.0]]),
        "bboxes": core.torch.tensor([
            [0.25, 0.25, 0.035, 0.030],
            [0.70, 0.65, 0.12, 0.10],
            [0.55, 0.40, 0.05, 0.04],
        ]),
    }
    total1, items1 = model(batch)
    if items1.numel() != 3 or not core.torch.isfinite(total1):
        raise RuntimeError(f"Expected three finite standard loss items, got {items1}")
    total1.backward()
    attention = model.model[26].local_attention
    gain_gradient = attention.gain.grad
    if gain_gradient is None or not core.torch.isfinite(gain_gradient).all() or float(gain_gradient.abs().sum()) <= 0:
        raise RuntimeError("Zero/nonfinite LDSA gain gradient on the first backward")
    first_gain_gradient = float(gain_gradient.detach().abs().sum())

    # A bounded synthetic first update opens the exact-zero ReZero branch.
    with core.torch.no_grad():
        normalized = gain_gradient / gain_gradient.abs().mean().clamp_min(1e-12)
        attention.gain.add_(-0.01 * normalized)
    model.zero_grad(set_to_none=True)
    total2, items2 = model(batch)
    total2.backward()
    feature_gradient = 0.0
    for name, parameter in attention.named_parameters():
        if name == "gain":
            continue
        if parameter.grad is None or not core.torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Missing/nonfinite LDSA gradient after gain opens: {name}")
        feature_gradient += float(parameter.grad.detach().abs().sum())
    if feature_gradient <= 0:
        raise RuntimeError("LDSA q/k/v/output parameters receive zero gradient after branch opening")
    result = {
        "standard_loss_items_step1": [float(value) for value in items1],
        "standard_loss_items_step2": [float(value) for value in items2],
        "first_gain_gradient_abs_sum": first_gain_gradient,
        "second_attention_feature_gradient_abs_sum": feature_gradient,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    model.zero_grad(set_to_none=True)
    return result


def p2_preflight(report, eval_only=False):
    del eval_only
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils import yaml_load

    print(f"E12a LDSA probe revision: {SCRIPT_REVISION}", flush=True)
    required = (
        MODEL_YAML, E10A_YAML, E10A_BEST, MODULE_SOURCE,
        E10A_MODULE_SOURCE, MATCHING_HELPER,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if core.sha256(E10A_BEST) != E10A_BEST_SHA256:
        raise RuntimeError("Frozen E10a best.pt differs from the audited checkpoint")
    cfg = yaml_model_load(str(MODEL_YAML))
    validate_config(cfg, yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml"))
    model = DetectionModel(cfg, nc=10, verbose=False)
    check_probe_model(model)

    report["is_formal"] = False
    report["is_probe"] = True
    report["p2_script_revision"] = SCRIPT_REVISION
    report["p2_architecture"] = architecture_record()
    report["e12a_probe"] = {
        "epochs": PROBE_EPOCHS,
        "start_checkpoint": str(E10A_BEST),
        "trainable_scope": "only layer26.local_attention parameters",
        "standard_detection_loss": True,
        "automatic_shutdown": False,
        "success_screen": "versus E10a: AP_small >= +0.0015 and mAP50_95 >= -0.0005",
    }
    report["preflight_transfer"] = load_e10a_state(model)
    report["initial_equivalence"] = initial_equivalence(model)
    model.eval()
    with core.torch.inference_mode():
        _, raw = model(core.torch.zeros(1, 3, 256, 256))
    sizes = [list(value.shape[-2:]) for value in raw]
    if sizes != [[64, 64], [32, 32], [16, 16], [8, 8]]:
        raise RuntimeError(f"Unexpected feature shapes: {sizes}")
    report["CPU_forward_256_feature_shapes"] = sizes
    report["loss_gradient_check"] = loss_gradient_check(model)

    out = Path(report["report_dir"])
    for path in required:
        shutil.copy2(path, out / path.name)
    core.save_report(report)
    del model, raw
    gc.collect()

    baseline_cfg = yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml")
    baseline_cfg.update(scale="s", nc=10)
    report["matching_equivalence"] = fast_tal.verify_equivalence(
        copy.deepcopy(cfg), device="cpu", baseline_cfg=baseline_cfg
    )
    fast_tal.assert_patch_restoration()
    core.write_json(out / "tal_equivalence.json", report["matching_equivalence"])
    core.save_report(report)
    print("E12a structure/E10a-transfer/neutral-init/gradient/TAL2 checks passed.", flush=True)


def train(report, smoke=False):
    weights = _BASE_TRAIN(report, smoke=smoke)
    run_dir = Path(report["run_dir"])
    for path in (MODULE_SOURCE, E10A_MODULE_SOURCE, MATCHING_HELPER, E10A_YAML):
        shutil.copy2(path, run_dir / path.name)
    trained = core.YOLO(str(weights)).model.float()
    check_probe_model(trained)
    report["e12a_probe"]["attention_parameter_count"] = sum(
        parameter.numel() for parameter in trained.model[26].local_attention.parameters()
    )
    core.save_report(report)
    del trained
    core.clear_gpu()
    return weights


def completed_reference(pattern):
    candidates = [
        path for path in (core.ROOT / "comparison_reports").glob(pattern)
        if "SMOKE" not in str(path).upper()
    ]
    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime)
    for path in reversed(candidates):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "completed":
            return path, value
    return None


def metric_deltas(current, reference):
    sections = {
        "overall": ("Precision", "Recall", "mAP50", "mAP75", "mAP50_95"),
        "area_metrics": ("AP_all", "AR_all", "AP_small", "AR_small", "AP_medium", "AP_large"),
    }
    return {
        section: {
            name: float(current[section][name]) - float(reference[section][name])
            for name in names
        }
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
        comparisons[label] = {
            "status": "computed",
            "path": str(path),
            "deltas": metric_deltas(report, values),
        }
    report["probe_comparisons"] = comparisons
    if comparisons.get("E10a", {}).get("status") == "computed":
        delta = comparisons["E10a"]["deltas"]
        report["e12a_probe"]["screen_result"] = {
            "passed": bool(
                delta["area_metrics"]["AP_small"] >= 0.0015
                and delta["overall"]["mAP50_95"] >= -0.0005
            ),
            "AP_small_required": 0.0015,
            "mAP50_95_floor": -0.0005,
            "observed_AP_small_delta": delta["area_metrics"]["AP_small"],
            "observed_mAP50_95_delta": delta["overall"]["mAP50_95"],
            "note": "Passing authorizes a fresh, fair 200-epoch formal E12a run.",
        }

    trained = core.YOLO(str(weights)).model.float()
    check_probe_model(trained)
    deep = trained.model[26]
    attention = deep.local_attention
    report["ldsa_learned_state"] = {
        "attention_gain": {
            "mean": float(attention.gain.detach().mean()),
            "std": float(attention.gain.detach().std()),
            "minimum": float(attention.gain.detach().min()),
            "maximum": float(attention.gain.detach().max()),
            "l2": float(attention.gain.detach().norm()),
        },
        "relative_bias_l2": float(attention.relative_bias.detach().norm()),
        "query_weight_l2": float(attention.query.weight.detach().norm()),
        "key_weight_l2": float(attention.key.weight.detach().norm()),
        "value_weight_l2": float(attention.value.weight.detach().norm()),
        "output_weight_l2": float(attention.output.weight.detach().norm()),
        "retained_P3_spatial_gate_l2": float(deep.spatial_gate.net[-1].weight.detach().norm()),
        "retained_P2_spatial_gate_l2": float(trained.model[27].spatial_gate.net[-1].weight.detach().norm()),
        "note": "Nonzero values prove branch use, not accuracy or causal benefit.",
    }
    core.save_report(report)
    del trained
    core.clear_gpu()


def install_overrides():
    e10a.install_overrides()
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
    core.assert_p2_model = check_probe_model
    core.validate_p2_config = validate_config
    core.p2_preflight = p2_preflight
    core.train = train
    core.evaluate = evaluate


if __name__ == "__main__":
    install_overrides()
    core.main()
