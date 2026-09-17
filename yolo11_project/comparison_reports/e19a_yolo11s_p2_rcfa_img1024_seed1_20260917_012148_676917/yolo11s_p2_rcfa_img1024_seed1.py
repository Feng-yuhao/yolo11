#!/usr/bin/env python3
"""E19a: E17a controlled replacement with reliability-conditioned local alignment.

Formal protocol is unchanged from E17a: YOLO11s, VisDrone, 1024, batch 8,
seed 1, deterministic SGD, AMP, 200 epochs and TAL2 matching chunks of two.
Automatic shutdown is disabled on every execution path.
"""

from __future__ import annotations

import copy
import gc
import math
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if not os.environ.get("OMP_NUM_THREADS", "").isdigit() or int(os.environ.get("OMP_NUM_THREADS", "0")) < 1:
    os.environ["OMP_NUM_THREADS"] = "8"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import p2_tal_chunked_vgpu32 as fast_tal
except ImportError as exc:
    raise ImportError("E19a requires p2_tal_chunked_vgpu32.py beside this file") from exc
sys.modules["p2_tal_chunked"] = fast_tal

try:
    import yolo11s_p2_img1024_seed1 as core
except ImportError as exc:
    raise ImportError("E19a requires the frozen E1 runner beside this file") from exc


EXPERIMENT = "e19a_yolo11s_p2_rcfa_img1024_seed1"
REVISION = "e19a_rcfa_e17_control_seed1_200e_tal2_v1"
MODEL_YAML = HERE / "yolo11s-p2-rcfa.yaml"
REFERENCE_E1_YAML = HERE / "yolo11s-p2-add.yaml"
REFERENCE_E17_YAML = HERE / "yolo11s-p2-fusion-ready.yaml"
RCFA_SOURCE = core.SOURCE / "ultralytics/nn/modules/e19_rcfa.py"
E17_MODULE = core.SOURCE / "ultralytics/nn/modules/e17_fusion.py"
E17_LOSS = core.SOURCE / "ultralytics/utils/e17_fusion_loss.py"
MATCHING_HELPER = Path(fast_tal.__file__).resolve()
SMOKE_GATE = core.ROOT / "comparison_reports/e19a_rcfa_smoke_passed.json"

_E1_LOAD = core.load_p2_pretrained
_E1_ASSERT = core.assert_p2_model
_E1_TRAIN = core.train

# Exact semantic mapping from the deterministic E1 initialization into E19a.
E1_TO_E19 = {
    **{index: index for index in range(11)},
    11: 11,
    12: 13,
    13: 14,
    14: 15,
    15: 17,
    16: 18,
    17: 19,
    18: 20,
    19: 21,
    20: 22,
    21: 23,
    22: 24,
    23: 25,
    24: 26,
    25: 27,
    26: 28,
}


def _target_key(key):
    parts = key.split(".")
    if len(parts) < 3 or parts[0] != "model" or not parts[1].isdigit():
        raise ValueError(f"Unrecognized E1 state tensor: {key}")
    source_index = int(parts[1])
    if source_index not in E1_TO_E19:
        raise ValueError(f"No E1->E19 semantic layer mapping for: {key}")
    parts[1] = str(E1_TO_E19[source_index])
    return ".".join(parts)


def _reference():
    """Rebuild the same deterministic E1 initialization used by E17a."""
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils.torch_utils import init_seeds

    init_seeds(core.SEED + 1, deterministic=True)
    cfg = yaml_model_load(str(REFERENCE_E1_YAML))
    cfg["scale"] = "s"
    reference = DetectionModel(cfg, nc=10, verbose=False)
    _E1_ASSERT(reference)
    original_audit = _E1_LOAD(reference)
    cpu_rng = core.torch.random.get_rng_state()
    cuda_rng = core.torch.cuda.get_rng_state_all() if core.torch.cuda.is_available() else None
    return reference, original_audit, cpu_rng, cuda_rng


def load_p2_pretrained(target):
    """Map every E1 tensor exactly and leave only RCFA/auxiliary tensors new."""
    reference, original_audit, cpu_rng, cuda_rng = _reference()
    target_state = target.state_dict()
    mapped = {}
    mapping = []
    for key, value in reference.state_dict().items():
        destination = _target_key(key)
        if destination not in target_state or value.shape != target_state[destination].shape:
            raise ValueError(f"E1 semantic transfer mismatch: {key} -> {destination}")
        if destination in mapped:
            raise ValueError(f"Duplicate transfer target: {destination}")
        mapped[destination] = value
        mapping.append(dict(source=key, target=destination, shape=list(value.shape)))

    missing = sorted(set(target_state) - set(mapped))
    allowed_prefixes = ("model.12.", "model.16.", "model.28.aux_head.")
    unexpected = [key for key in missing if not key.startswith(allowed_prefixes)]
    if unexpected:
        raise ValueError(f"Unexpected uninitialized E19a target tensors: {unexpected}")
    if not missing:
        raise ValueError("E19a unexpectedly has no new tensors")

    result = target.load_state_dict(mapped, strict=False)
    if sorted(result.missing_keys) != missing or result.unexpected_keys:
        raise ValueError(
            f"Incomplete E1 migration: missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    core.torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        core.torch.cuda.set_rng_state_all(cuda_rng)

    actual = target.state_dict()
    for key, value in mapped.items():
        if not core.torch.equal(actual[key].cpu(), value.cpu()):
            raise RuntimeError(f"Transferred E1 tensor changed: {key}")

    exact_groups = {
        "e1_p2": [key for key in mapped if key.startswith("model.27.")],
        "e1_detect": [
            key for key in mapped
            if key.startswith("model.28.") and not key.startswith("model.28.aux_head.")
        ],
    }
    if not all(exact_groups.values()):
        raise RuntimeError(f"E19a exact-transfer group unexpectedly empty: {exact_groups}")

    audit = dict(
        source=(
            "original 80-class yolo11s.pt transferred into deterministic E1, then "
            "every E1 tensor mapped exactly into E19a"
        ),
        source_audit=original_audit,
        loaded_tensor_count=len(mapped),
        new_tensor_keys=missing,
        exact_tensor_equality=True,
        exact_groups={name: len(keys) for name, keys in exact_groups.items()},
        tensor_mapping=mapping,
        reference_e1_yaml_sha256=core.sha256(REFERENCE_E1_YAML),
        note=(
            "No experiment best.pt is a parent. Only layers 12/16 RCFA and "
            "model.28.aux_head are newly initialized."
        ),
    )
    del reference
    return audit


def assert_p2_model(model):
    """Validate topology only; never enforce trainable initialization values."""
    if len(model.model) != 29 or model.yaml.get("scale") != "s":
        raise ValueError("Expected the 29-layer s-scale E19a architecture")
    head = model.model[28]
    if head.__class__.__name__ != "E17Detect" or head.f != [27, 18, 21, 24, 12, 16] or head.nc != 10:
        raise ValueError("E19a must retain E17Detect with four main plus two auxiliary sources")
    if list(model.stride.cpu().tolist()) != [4.0, 8.0, 16.0, 32.0]:
        raise ValueError(f"E19a detection strides changed: {model.stride}")

    expected_channels = {12: (512, 256), 16: (256, 256)}
    for index, inputs in ((12, [11, 6]), (16, [15, 4])):
        adapter = model.model[index]
        if adapter.__class__.__name__ != "RCFAAdapter" or adapter.f != inputs:
            raise ValueError(f"Incorrect E19a RCFA adapter at layer {index}")
        if tuple(adapter.channels) != expected_channels[index] or adapter.hidden != 32:
            raise ValueError(f"Incorrect E19a RCFA runtime channels at layer {index}")
        if adapter.center_bias.shape != (1, 9, 1, 1):
            raise ValueError(f"Incorrect E19a center prior shape at layer {index}")

    for index in (11, 15, 25):
        layer = model.model[index]
        if layer.__class__.__name__ != "Upsample" or layer.mode != "nearest" or layer.scale_factor != 2.0:
            raise ValueError(f"Upsample {index} changed from frozen nearest-neighbor 2x")

    names = {layer.__class__.__name__ for layer in model.model}
    forbidden_tokens = (
        "FusionReadyAdapter", "DeepSemanticGuide", "SemanticDetailGate", "DySample",
        "PGALP", "P3SDE", "P4P3SpatialGuide", "TSDR",
    )
    forbidden = sorted(name for name in names if any(token in name for token in forbidden_tokens))
    if forbidden:
        raise ValueError(f"E19a contains unplanned modules: {forbidden}")


def validate_p2_config(cfg, original_cfg):
    del original_cfg
    from ultralytics.utils import yaml_load

    e17 = yaml_load(REFERENCE_E17_YAML)
    expected_head = copy.deepcopy(e17["head"])
    expected_head[1] = [[11, 6], 1, "RCFAAdapter", [32, 0.05, 4.0]]
    expected_head[5] = [[15, 4], 1, "RCFAAdapter", [32, 0.05, 4.0]]
    checks = [
        ("backbone", core.canonical_layers(cfg.get("backbone", [])), core.canonical_layers(e17["backbone"])),
        ("head", core.canonical_layers(cfg.get("head", [])), core.canonical_layers(expected_head)),
        ("scale", cfg.get("scale"), "s"),
        ("nc", cfg.get("nc"), 10),
        ("s_scaling", cfg.get("scales", {}).get("s"), e17["scales"]["s"]),
        ("e17_aux", cfg.get("e17_aux"), e17.get("e17_aux")),
    ]
    differences = [(name, actual, expected) for name, actual, expected in checks if actual != expected]
    if differences:
        details = "; ".join(
            f"{name}: expected={expected!r}, actual={actual!r}" for name, actual, expected in differences
        )
        raise ValueError("E19a YAML is not the strict E17a adapter replacement. " + details)


def _has_finite_gradient(module):
    gradients = [
        parameter.grad for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    return bool(gradients) and all(core.torch.isfinite(gradient).all() for gradient in gradients)


def _rcfa_unit_check(adapter):
    top_channels, lateral_channels = adapter.channels
    top = core.torch.randn(2, top_channels, 16, 20, requires_grad=True)
    lateral = core.torch.randn(2, lateral_channels, 16, 20, requires_grad=True)
    output = adapter([top, lateral])
    update, reliability, weights = adapter.routing([top, lateral])
    if output.shape != lateral.shape or update.shape != lateral.shape:
        raise RuntimeError("E19a RCFA shape-preservation check failed")
    if not all(core.torch.isfinite(tensor).all() for tensor in (output, update, reliability, weights)):
        raise RuntimeError("E19a RCFA produced a non-finite tensor")
    if not core.torch.allclose(weights.sum(1), core.torch.ones_like(weights[:, 0]), atol=1e-6, rtol=0):
        raise RuntimeError("E19a shift weights do not sum to one")

    expected_center = math.exp(adapter.center_prior_value) / (math.exp(adapter.center_prior_value) + 8.0)
    center_region = weights[:, adapter.center_index, 1:-1, 1:-1]
    center_mean = float(center_region.mean())
    reliability_mean = float(reliability.mean())
    if abs(center_mean - expected_center) > 1e-6:
        raise RuntimeError(f"E19a center-prior initialization changed: {center_mean} vs {expected_center}")
    if abs(reliability_mean - 0.5) > 1e-6:
        raise RuntimeError(f"E19a reliability initialization changed: {reliability_mean}")

    # A corner impulse must never reappear at the opposite corner (no cyclic roll).
    impulse = core.torch.zeros(1, 1, 3, 3)
    impulse[..., 0, 0] = 1.0
    shifted = list(adapter._shift_views(impulse))
    if len(shifted) != 9 or any(float(view[..., -1, -1].abs().max()) != 0.0 for view in shifted):
        raise RuntimeError("E19a local shift candidates wrap across image boundaries")

    output.float().square().mean().backward()
    if not _has_finite_gradient(adapter):
        raise RuntimeError("E19a RCFA backward/gradient check failed")
    return dict(
        channels=list(adapter.channels),
        hidden=adapter.hidden,
        center_prior=adapter.center_prior_value,
        expected_center_weight=expected_center,
        measured_center_weight=center_mean,
        measured_reliability=reliability_mean,
        parameter_count=sum(parameter.numel() for parameter in adapter.parameters()),
        bounded_local_candidates=9,
        cyclic_wrap=False,
    )


def _loss_mode_regression(model):
    """Exercise train/backward -> eval loss -> train/backward on one criterion."""
    original_args = getattr(model, "args", None)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    batch = dict(
        img=core.torch.rand(2, 3, 128, 128),
        batch_idx=core.torch.tensor([0.0, 0.0, 1.0, 1.0]),
        cls=core.torch.tensor([[3.0], [0.0], [4.0], [8.0]]),
        bboxes=core.torch.tensor([
            [0.30, 0.40, 0.20, 0.20],
            [0.72, 0.68, 0.08, 0.09],
            [0.60, 0.60, 0.30, 0.24],
            [0.22, 0.25, 0.14, 0.12],
        ]),
    )

    model.train()
    model.zero_grad(set_to_none=True)
    train1, items1 = model(batch)
    if not core.torch.isfinite(train1) or len(items1) != 3:
        raise RuntimeError("E19a first training loss is invalid")
    train1.backward()
    for index in (12, 16):
        if not _has_finite_gradient(model.model[index]):
            raise RuntimeError(f"E19a adapter {index} did not receive finite gradients")
    if not _has_finite_gradient(model.model[28].aux_head):
        raise RuntimeError("E19a auxiliary head did not receive finite gradients")
    if model.model[28]._aux_logits is not None:
        raise RuntimeError("E19a auxiliary logits were not cleared after training loss")

    model.eval()
    with core.torch.no_grad():
        preds = model(batch["img"])
        val_total, val_items = model.criterion(preds, batch)
    if not core.torch.isfinite(val_total) or len(val_items) != 3:
        raise RuntimeError("E19a eval-mode validation loss is invalid")
    if model.model[28]._aux_logits is not None:
        raise RuntimeError("E19a eval mode retained auxiliary logits")

    model.train()
    model.zero_grad(set_to_none=True)
    train2, items2 = model(batch)
    if not core.torch.isfinite(train2) or len(items2) != 3:
        raise RuntimeError("E19a second training loss is invalid after validation")
    train2.backward()
    for index in (12, 16):
        if not _has_finite_gradient(model.model[index]):
            raise RuntimeError(f"E19a adapter {index} lost gradients after eval->train transition")
    if not _has_finite_gradient(model.model[28].aux_head):
        raise RuntimeError("E19a auxiliary head lost gradients after eval->train transition")

    if original_args is not None:
        model.args = original_args
    return dict(
        train_loss_before_eval=float(train1.detach()),
        validation_loss=float(val_total.detach()),
        train_loss_after_eval=float(train2.detach()),
        criterion=model.criterion.__class__.__name__,
        transition="train/backward -> eval/no_grad loss -> train/backward",
    )


def p2_preflight(report, eval_only=False):
    del eval_only
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils import yaml_load
    from ultralytics.utils.torch_utils import init_seeds

    required = (
        MODEL_YAML, REFERENCE_E1_YAML, REFERENCE_E17_YAML, RCFA_SOURCE,
        E17_MODULE, E17_LOSS, MATCHING_HELPER,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if Path(core.ultralytics.__file__).resolve().parent != (core.SOURCE / "ultralytics").resolve():
        raise RuntimeError("Wrong editable Ultralytics source; activate yolo11 conda environment")
    if core.sha256(core.PRETRAINED) != core.PRETRAINED_SHA256:
        raise RuntimeError("Original yolo11s.pt changed")

    cfg = yaml_model_load(str(MODEL_YAML))
    validate_p2_config(cfg, yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml"))
    out = Path(report["report_dir"])
    for path in required:
        shutil.copy2(path, out / path.name)

    report["p2_script_revision"] = REVISION
    report["p2_architecture"] = dict(
        yaml_sha256=core.sha256(MODEL_YAML),
        reference_e1_yaml_sha256=core.sha256(REFERENCE_E1_YAML),
        reference_e17_yaml_sha256=core.sha256(REFERENCE_E17_YAML),
        original_weights_sha256=core.sha256(core.PRETRAINED),
        rcfa_sha256=core.sha256(RCFA_SOURCE),
        e17_module_sha256=core.sha256(E17_MODULE),
        e17_loss_sha256=core.sha256(E17_LOSS),
        tasks_sha256=core.sha256(core.SOURCE / "ultralytics/nn/tasks.py"),
        variant=(
            "E19a = E17a controlled replacement: both FusionReadyAdapters become "
            "reliability-conditioned bounded 3x3 correlation alignment adapters"
        ),
        detect_strides=[4, 8, 16, 32],
        upsampling="nearest unchanged at layers 11/15/25",
        alignment="masked zero-border sliced 3x3 candidates; no grid_sample/DCN/replicate-pad/torch.roll",
        auxiliary_gain=cfg["e17_aux"]["gain"],
        tal2=fast_tal.implementation_info(),
        claim_boundary=(
            "One E17a adapter-replacement experiment only; no E10a/E18a DSSG, E16a TSDR, "
            "DySample, PGALP, P3SDE or slicing"
        ),
    )

    init_seeds(core.SEED, deterministic=True)
    model = DetectionModel(cfg, nc=10, verbose=False)
    assert_p2_model(model)
    report["preflight_transfer"] = load_p2_pretrained(model)

    # Initialization assertions are intentionally scoped to this fresh model.
    expected_center = math.exp(4.0) / (math.exp(4.0) + 8.0)
    initialization = []
    for index in (12, 16):
        adapter = model.model[index]
        if abs(float(adapter.layer_scale.detach()) - 0.05) > 1e-7:
            raise ValueError(f"RCFA layer_scale initialization changed at layer {index}")
        initialization.append(_rcfa_unit_check(adapter))
    if any(abs(item["measured_center_weight"] - expected_center) > 1e-6 for item in initialization):
        raise RuntimeError("E19a center-prior initialization check failed")
    report["p2_architecture"]["fresh_initialization_checks"] = initialization

    captured = {}

    def capture_head_inputs(module, args):
        del module
        features = args[0]
        captured["channels"] = [int(tensor.shape[1]) for tensor in features]
        captured["grids"] = [list(tensor.shape[-2:]) for tensor in features]

    hook = model.model[28].register_forward_pre_hook(capture_head_inputs)
    model.eval()
    with core.torch.no_grad():
        output, raw = model(core.torch.zeros(1, 3, 256, 256))
    hook.remove()
    expected_channels = [128, 128, 256, 512, 256, 256]
    sizes = [list(tensor.shape[-2:]) for tensor in raw]
    if captured.get("channels") != expected_channels:
        raise RuntimeError(f"E19a head channels changed: {captured.get('channels')}")
    if sizes != [[64, 64], [32, 32], [16, 16], [8, 8]] or not core.torch.isfinite(output).all():
        raise RuntimeError(f"Incorrect/nonfinite E19a inference features: {sizes}")
    report["p2_architecture"]["head_source_channels"] = captured["channels"]
    report["p2_architecture"]["head_source_grids_256"] = captured["grids"]
    report["p2_architecture"]["cpu_feature_grids"] = sizes
    report["p2_architecture"]["parameters_including_training_only_aux"] = sum(
        parameter.numel() for parameter in model.parameters()
    )

    report["loss_mode_regression"] = _loss_mode_regression(model)
    core.save_report(report)
    del model, raw, output
    gc.collect()
    print("E19a topology, E1 transfer, RCFA routing and E17 loss-mode checks passed.", flush=True)

    original_cfg = yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml")
    original_cfg.update(scale="s", nc=10)
    report["matching_equivalence"] = fast_tal.verify_equivalence(
        copy.deepcopy(cfg), device="cpu", baseline_cfg=original_cfg
    )
    fast_tal.assert_patch_restoration()
    core.write_json(out / "tal_equivalence.json", report["matching_equivalence"])
    core.save_report(report)
    print("TAL2 equivalence passed; run --smoke2 for CUDA batch8/1024 feasibility.", flush=True)


def train(report, smoke=False):
    weights = _E1_TRAIN(report, smoke=smoke)
    run_dir = Path(report["run_dir"])
    for path in (RCFA_SOURCE, E17_MODULE, E17_LOSS, REFERENCE_E1_YAML, REFERENCE_E17_YAML):
        shutil.copy2(path, run_dir / path.name)
    return weights


def install_overrides():
    core.EXPERIMENT = EXPERIMENT
    core.SCRIPT_REVISION = REVISION
    core.MODEL_YAML = MODEL_YAML
    core.SMOKE_GATE = SMOKE_GATE
    core.MATCHING_HELPER = MATCHING_HELPER
    core.AUTO_SHUTDOWN = False
    core.SHUTDOWN_ON_FAILURE = False
    core.__file__ = str(Path(__file__).resolve())
    core.load_p2_pretrained = load_p2_pretrained
    core.assert_p2_model = assert_p2_model
    core.validate_p2_config = validate_p2_config
    core.p2_preflight = p2_preflight
    core.train = train


if __name__ == "__main__":
    install_overrides()
    core.main()
