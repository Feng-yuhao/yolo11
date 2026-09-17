#!/usr/bin/env python3
"""E16a: E10a + tiny-supervised, task-decoupled P1 detail routing."""

from __future__ import annotations

import copy
import gc
import hashlib
import inspect
import json
import math
import os
import shutil
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
if not os.environ.get("OMP_NUM_THREADS", "").isdigit() or int(os.environ.get("OMP_NUM_THREADS", "0")) < 1:
    os.environ["OMP_NUM_THREADS"] = "8"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import p2_tal_chunked_vgpu32 as fast_tal
except ImportError as exc:
    raise ImportError("E16a requires p2_tal_chunked_vgpu32.py beside this file") from exc
sys.modules["p2_tal_chunked"] = fast_tal

try:
    import yolo11s_p2_img1024_seed1 as core
except ImportError as exc:
    raise ImportError("E16a requires the frozen E1 runner beside this file") from exc

EXPERIMENT = "e16a_yolo11s_p2_tsdr_img1024_seed1"
SCRIPT_REVISION = "e16a_tsdr_formal_v1"
MODEL_YAML = HERE / "yolo11s-p2-tsdr.yaml"
REFERENCE_E1_YAML = core.ROOT / "yolo11s-p2-add.yaml"
AUTH_E10A_DIR = core.ROOT / "comparison_reports/e10a_yolo11s_p2_dssg_img1024_seed1_20260910_222554_227195"
AUTH_E10A_YAML = AUTH_E10A_DIR / "yolo11s-p2-dssg.yaml"
EMBEDDED_E10A_YAML = HERE / "e10a_reference_yolo11s-p2-dssg.yaml"
DSSG_SOURCE = core.SOURCE / "ultralytics/nn/modules/dssg.py"
TSDR_SOURCE = core.SOURCE / "ultralytics/nn/modules/tsdr.py"
TSDR_LOSS_SOURCE = core.SOURCE / "ultralytics/utils/tsdr_loss.py"
MATCHING_HELPER = Path(fast_tal.__file__).resolve()
SMOKE_GATE = core.ROOT / "comparison_reports/e16a_tsdr_tal2_smoke_passed.json"
CORE_RUNNER = HERE / "yolo11s_p2_img1024_seed1.py"
FROZEN_E10A_METRICS = HERE / "e10a_reference_metrics_seed1_200e.json"

_CORE_TRAIN = core.train
_CORE_EVALUATE = core.evaluate
_E1_TRANSFER_KEY = core.transfer_key
_E1_PLAN_TRANSFER = core.plan_transfer
_E1_LOAD_PRETRAINED = core.load_p2_pretrained
_E1_ASSERT_MODEL = core.assert_p2_model


def _normalized_sha256(path):
    """SHA256 after normalizing CRLF/CR newlines to LF."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _smoke_code_hashes():
    files = {
        "yolo11s_p2_tsdr_img1024_seed1.py": Path(__file__).resolve(),
        "yolo11s_p2_img1024_seed1.py": CORE_RUNNER,
        "p2_tal_chunked_vgpu32.py": MATCHING_HELPER,
        "ultralytics/nn/tasks.py": core.SOURCE / "ultralytics/nn/tasks.py",
        "ultralytics/nn/modules/dssg.py": DSSG_SOURCE,
        "ultralytics/nn/modules/tsdr.py": TSDR_SOURCE,
        "ultralytics/utils/tsdr_loss.py": TSDR_LOSS_SOURCE,
        "ultralytics/utils/loss.py": core.SOURCE / "ultralytics/utils/loss.py",
        "ultralytics/utils/tal.py": core.SOURCE / "ultralytics/utils/tal.py",
    }
    missing = [str(path) for path in files.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Smoke-signature source files missing: {missing}")
    return {name: _normalized_sha256(path) for name, path in files.items()}


def _tensor_dict_hash(items):
    digest = hashlib.sha256()
    for key, value in sorted(items.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _state_prefix(state, prefixes):
    return {k: v for k, v in state.items() if any(k.startswith(prefix) for prefix in prefixes)}


def _tsdr_hashes(module):
    state = module.state_dict()
    return {
        "all_trainable": _tensor_dict_hash({k: v for k, v in state.items() if k != "gaussian3"}),
        "projection": _tensor_dict_hash({k: v for k, v in state.items() if k.startswith("detail_project.")}),
        "gate": _tensor_dict_hash({k: v for k, v in state.items() if k.startswith("gate_net.")}),
        "raw_gain": _tensor_dict_hash({"raw_gain": module.raw_gain.detach()}),
    }


def transfer_key(source_key):
    parts = source_key.split(".")
    if len(parts) < 3 or parts[0] != "model" or not parts[1].isdigit():
        raise ValueError(f"Unexpected source key: {source_key}")
    index = int(parts[1])
    if 0 <= index <= 22:
        return source_key
    if index != 23:
        raise ValueError(f"Expected original YOLO11s Detect at layer 23: {source_key}")
    parts[1] = "28"
    if parts[2] == "dfl":
        return ".".join(parts)
    if parts[2] in ("cv2", "cv3") and len(parts) > 4 and parts[3] in ("0", "1", "2"):
        parts[3] = str(int(parts[3]) + 1)
        return ".".join(parts)
    raise ValueError(f"Unknown Detect state key: {source_key}")


def plan_transfer(source_state, target_state):
    mapped, skipped, mapping = {}, [], []
    for key, tensor in source_state.items():
        target_key = transfer_key(key)
        if target_key not in target_state:
            raise ValueError(f"Missing target tensor: {key} -> {target_key}")
        target = target_state[target_key]
        if tuple(tensor.shape) != tuple(target.shape):
            expected = (
                core.class_output_key(key, 23, ("0", "1", "2"))
                and tensor.shape[0] == 80
                and target.shape[0] == 10
                and tuple(tensor.shape[1:]) == tuple(target.shape[1:])
            )
            if not expected:
                raise ValueError(
                    f"Unexpected shape mismatch: {key} -> {target_key}: "
                    f"{tuple(tensor.shape)} != {tuple(target.shape)}"
                )
            skipped.append(dict(source=key, target=target_key, reason="80 COCO classes -> 10 VisDrone classes"))
            continue
        if target_key in mapped:
            raise ValueError(f"Duplicate target mapping: {target_key}")
        mapped[target_key] = tensor
        mapping.append(dict(source=key, target=target_key, shape=list(tensor.shape)))

    missing = sorted(set(target_state) - set(mapped))
    for key in missing:
        allowed = (
            key.startswith("model.25.")
            or key.startswith("model.26.")
            or key.startswith("model.27.")
            or key.startswith("model.28.")
        )
        if not allowed:
            raise ValueError(f"Unexpected uninitialized tensor: {key}")
    if len(skipped) != 6:
        raise ValueError(f"Expected six class-output skips, got {len(skipped)}")
    return mapped, dict(
        rule="original layers 0..22 unchanged; Detect 23->28 and branches 0..2->1..3",
        source="original 80-class yolo11s.pt; never an experiment best.pt",
        loaded_tensors=len(mapped),
        skipped_class_outputs=skipped,
        new_or_reinitialized_target_tensors=missing,
        tensor_mapping=mapping,
    )


def _remap_e1_reference(reference_state, detect_layer):
    fair = {}
    for key, value in reference_state.items():
        if key.startswith("model.25."):
            fair[key] = value
        elif key.startswith("model.26."):
            fair[key.replace("model.26.", f"model.{detect_layer}.", 1)] = value
    return fair


def _load_frozen_e1_reference():
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils.torch_utils import init_seeds

    init_seeds(core.SEED + 1, deterministic=True)
    cfg = yaml_model_load(str(REFERENCE_E1_YAML))
    cfg["scale"] = "s"
    reference = DetectionModel(cfg, nc=10, verbose=False)
    _E1_ASSERT_MODEL(reference)
    current_transfer, current_plan = core.transfer_key, core.plan_transfer
    try:
        core.transfer_key = _E1_TRANSFER_KEY
        core.plan_transfer = _E1_PLAN_TRANSFER
        _E1_LOAD_PRETRAINED(reference)
    finally:
        core.transfer_key = current_transfer
        core.plan_transfer = current_plan
    fair_cpu_rng = core.torch.random.get_rng_state()
    fair_cuda_rng = core.torch.cuda.get_rng_state_all() if core.torch.cuda.is_available() else None
    return reference, fair_cpu_rng, fair_cuda_rng


def _original_mapping_for(target, detect_layer):
    original = core.YOLO(str(core.PRETRAINED)).model.float()
    source_state, target_state = original.state_dict(), target.state_dict()
    mapped, skipped, mapping = {}, [], []
    for key, tensor in source_state.items():
        parts = key.split(".")
        index = int(parts[1])
        if 0 <= index <= 22:
            target_key = key
        elif index == 23:
            parts[1] = str(detect_layer)
            if parts[2] == "dfl":
                target_key = ".".join(parts)
            elif parts[2] in ("cv2", "cv3") and len(parts) > 4 and parts[3] in ("0", "1", "2"):
                parts[3] = str(int(parts[3]) + 1)
                target_key = ".".join(parts)
            else:
                raise ValueError(f"Unknown source Detect key: {key}")
        else:
            raise ValueError(f"Unexpected original layer index in {key}")
        if target_key not in target_state:
            raise ValueError(f"Missing target key {target_key}")
        if tuple(tensor.shape) != tuple(target_state[target_key].shape):
            expected = (
                core.class_output_key(key, 23, ("0", "1", "2"))
                and tensor.shape[0] == 80
                and target_state[target_key].shape[0] == 10
                and tuple(tensor.shape[1:]) == tuple(target_state[target_key].shape[1:])
            )
            if not expected:
                raise ValueError(f"Unexpected source/target shape mismatch: {key} -> {target_key}")
            skipped.append((key, target_key))
            continue
        mapped[target_key] = tensor
        mapping.append((key, target_key))
    del original
    return mapped, skipped, mapping


def _load_e10a_fair(target):
    mapped, skipped, mapping = _original_mapping_for(target, 28)
    result = target.load_state_dict(mapped, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"E10a reference original transfer unexpected keys: {result.unexpected_keys}")
    reference, fair_cpu_rng, fair_cuda_rng = _load_frozen_e1_reference()
    fair_state = _remap_e1_reference(reference.state_dict(), 28)
    target.load_state_dict(fair_state, strict=False)
    core.torch.random.set_rng_state(fair_cpu_rng)
    if fair_cuda_rng is not None:
        core.torch.cuda.set_rng_state_all(fair_cuda_rng)
    del reference
    return dict(original_loaded=len(mapped), original_skipped=len(skipped), e1_tensor_count=len(fair_state), mapping=mapping)


def load_p2_pretrained(target):
    original = core.YOLO(str(core.PRETRAINED)).model.float()
    if len(original.model) != 24 or list(original.stride.cpu().tolist()) != [8.0, 16.0, 32.0]:
        raise ValueError("Pretrained checkpoint is not original P3/P4/P5 YOLO11s")
    for index in range(23):
        candidate = target.model[index]
        if type(original.model[index]) is not type(candidate):
            raise ValueError(f"Unexpected architecture change at original layer {index}")
        if original.model[index].f != candidate.f:
            raise ValueError(f"Layer source changed unexpectedly at original layer {index}")

    mapping, audit = plan_transfer(original.state_dict(), target.state_dict())

    # PyTorch BatchNorm backward-compatibility intentionally does not report a missing
    # ``num_batches_tracked`` buffer: BatchNorm._load_from_state_dict inserts the
    # current buffer value when checkpoint metadata predates that buffer.  The transfer
    # plan is derived from target.state_dict(), so its "new" list does include these
    # buffers.  Audit them separately instead of treating PyTorch's compatibility
    # behavior as an initialization failure.
    allowed_missing = set(audit["new_or_reinitialized_target_tensors"])
    compatibility_bn_counters = {
        key for key in allowed_missing if key.endswith(".num_batches_tracked")
    }
    expected_reported_missing = allowed_missing - compatibility_bn_counters
    bn_counter_before = {
        key: target.state_dict()[key].detach().cpu().clone()
        for key in compatibility_bn_counters
    }

    result = target.load_state_dict(mapping, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected loaded keys: {result.unexpected_keys}")

    actual_missing = set(result.missing_keys)
    if actual_missing != expected_reported_missing:
        raise RuntimeError(
            "Original-transfer missing-key audit mismatch after normalizing PyTorch "
            "BatchNorm num_batches_tracked compatibility: "
            f"expected_reported={sorted(expected_reported_missing)}, "
            f"actual_reported={sorted(actual_missing)}, "
            f"compatibility_bn_counters={sorted(compatibility_bn_counters)}"
        )

    post_original_state = target.state_dict()
    changed_bn_counters = [
        key
        for key, before in bn_counter_before.items()
        if not core.torch.equal(post_original_state[key].detach().cpu(), before)
    ]
    if changed_bn_counters:
        raise RuntimeError(
            "Original-transfer unexpectedly changed compatibility BatchNorm counters: "
            f"{changed_bn_counters}"
        )

    reference, fair_cpu_rng, fair_cuda_rng = _load_frozen_e1_reference()
    fair_state = _remap_e1_reference(reference.state_dict(), 28)
    result2 = target.load_state_dict(fair_state, strict=False)
    if result2.unexpected_keys:
        raise ValueError(f"Unexpected E1-loaded keys: {result2.unexpected_keys}")
    core.torch.random.set_rng_state(fair_cpu_rng)
    if fair_cuda_rng is not None:
        core.torch.cuda.set_rng_state_all(fair_cuda_rng)

    actual = target.state_dict()
    if any(not core.torch.equal(actual[key].cpu(), value.cpu()) for key, value in mapping.items()):
        raise RuntimeError("Original pretrained tensor equality check failed")
    if any(not core.torch.equal(actual[key].cpu(), value.cpu()) for key, value in fair_state.items()):
        raise RuntimeError("Frozen E1 P2/Detect tensor equality check failed")

    audit.update(
        verified_tensor_equality=True,
        original_transfer_missing_keys_reported=sorted(actual_missing),
        original_transfer_missing_keys_expected_after_bn_compat=sorted(expected_reported_missing),
        original_transfer_bn_num_batches_tracked_compatibility=sorted(compatibility_bn_counters),
        original_transfer_bn_num_batches_tracked_preserved=True,
        original_transfer_unexpected_keys=list(result.unexpected_keys),
        frozen_e1_random_tensor_count=len(fair_state),
        frozen_e1_reference_yaml=str(REFERENCE_E1_YAML),
        frozen_e1_reference_yaml_sha256=core.sha256(REFERENCE_E1_YAML),
        initialization_note=(
            "Original layers use yolo11s.pt. P2 and complete four-scale Detect are copied from deterministic E1. "
            "Original layers, E1 P2/Detect towers, and DSSG use the same deterministic initialization as E10a. "
            "Only the TSDR high-pass projector, tiny router, and bounded regression residual are new."
        ),
    )
    del mapping, fair_state, reference, original
    return audit


def assert_p2_model(model):
    if len(model.model) != 29 or model.yaml.get("scale") != "s":
        raise ValueError("Expected 29-layer s-scale E16a")
    head = model.model[28]
    if head.nc != 10 or head.f != [0, 27, 26, 19, 22]:
        raise ValueError("Wrong E16a TSDRDetect inputs")
    if list(model.stride.cpu().tolist()) != [4.0, 8.0, 16.0, 32.0]:
        raise ValueError(f"Wrong strides: {model.stride}")
    for index in (11, 14, 23):
        layer = model.model[index]
        if layer.__class__.__name__ != "Upsample" or layer.mode != "nearest" or layer.scale_factor != 2.0:
            raise ValueError(f"Layer {index} must remain nearest 2x")
    deep, detail = model.model[26], model.model[27]
    if deep.__class__.__name__ != "DeepSemanticGuide" or deep.f != [16, 19, 22]:
        raise ValueError("Layer 26 must remain E10a DeepSemanticGuide")
    if (deep.p3_channels, deep.p4_channels, deep.p5_channels, deep.hidden) != (128, 256, 512, 32):
        raise ValueError("DeepSemanticGuide runtime channels differ from E10a")
    if detail.__class__.__name__ != "SemanticDetailGate" or detail.f != [25, 26]:
        raise ValueError("Layer 27 must remain E10a SemanticDetailGate")
    if (detail.p2_channels, detail.p3_channels) != (128, 128):
        raise ValueError("SemanticDetailGate runtime channels differ from E10a")
    if head.__class__.__name__ != "TSDRDetect":
        raise ValueError("Layer 28 must be TSDRDetect")
    if head.input_channels != (32, 128, 128, 256, 512):
        raise ValueError(f"TSDRDetect runtime channels differ: {head.input_channels}")
    if head.nl != 4 or head.gate_hidden != 16 or abs(head.max_gain - 0.25) > 1e-12:
        raise ValueError("TSDRDetect protocol differs")
    tsdr_cfg = model.yaml.get("tsdr", {})
    if tsdr_cfg != {"enabled": True, "gate_loss_gain": 0.05, "tiny_equivalent_side": 32.0}:
        raise ValueError(f"TSDR loss protocol differs: {tsdr_cfg}")
    if model.model[10].__class__.__name__ != "C2PSA":
        raise ValueError("Original C2PSA changed")
    forbidden = {layer.__class__.__name__ for layer in model.model} & {"PGALP", "P3SDE", "DySample"}
    if forbidden:
        raise ValueError(f"Forbidden modules present: {sorted(forbidden)}")


def _e10a_reference_yaml():
    return AUTH_E10A_YAML if AUTH_E10A_YAML.is_file() else EMBEDDED_E10A_YAML


def validate_p2_config(cfg, original_cfg):
    del original_cfg
    from ultralytics.utils import yaml_load

    e10a_cfg = yaml_load(_e10a_reference_yaml())
    expected_head = copy.deepcopy(e10a_cfg["head"][:-1])
    expected_head.append([[0, 27, 26, 19, 22], 1, "TSDRDetect", ["nc", 16, 0.25]])
    checks = [
        ("backbone", core.canonical_layers(cfg.get("backbone", [])), core.canonical_layers(e10a_cfg["backbone"])),
        ("head", core.canonical_layers(cfg.get("head", [])), core.canonical_layers(expected_head)),
        ("scale", cfg.get("scale"), "s"),
        ("nc", cfg.get("nc"), 10),
        ("s_scaling", cfg.get("scales", {}).get("s"), e10a_cfg["scales"]["s"]),
        (
            "tsdr",
            cfg.get("tsdr"),
            {"enabled": True, "gate_loss_gain": 0.05, "tiny_equivalent_side": 32.0},
        ),
    ]
    differences = [(name, actual, expected) for name, actual, expected in checks if actual != expected]
    if differences:
        details = "; ".join(f"{name}: expected={expected!r}, actual={actual!r}" for name, actual, expected in differences)
        raise ValueError("E16a YAML is not the audited TSDR replacement of E10a Detect. " + details)


def _tsdr_two_step_check(module):
    torch = core.torch
    torch.manual_seed(1501)
    p1 = torch.randn(2, 32, 128, 160)
    p2 = torch.randn(2, 128, 64, 80)
    p3 = torch.randn(2, 128, 32, 40)
    p4 = torch.randn(2, 256, 16, 20)
    p5 = torch.randn(2, 512, 8, 10)
    module = copy.deepcopy(module).train()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.05)

    before = module.raw_gain.detach().clone()
    out0 = module([p1, p2, p3, p4, p5])
    with torch.no_grad():
        p2_box_reference = module.cv2[0](p2)
        p2_cls_reference = module.cv3[0](p2)
    identity_error = float((out0[0][:, : module.reg_max * 4].detach() - p2_box_reference).abs().max())
    classification_error = float((out0[0][:, module.reg_max * 4 :].detach() - p2_cls_reference).abs().max())
    if identity_error > 1e-7:
        raise RuntimeError(f"TSDR P2 box identity initialization failed: {identity_error}")
    if classification_error > 1e-7:
        raise RuntimeError(f"TSDR P2 classification isolation failed: {classification_error}")

    optimizer.zero_grad(set_to_none=True)
    logits0 = module._tsdr_gate_logits
    (sum(x.float().square().mean() for x in out0) + 0.05 * torch.nn.functional.softplus(-logits0).mean()).backward()
    raw_grad1 = module.raw_gain.grad
    if raw_grad1 is None or not torch.isfinite(raw_grad1).all() or float(raw_grad1.abs().sum()) <= 0:
        raise RuntimeError("TSDR raw_gain first-step gradient failed")
    optimizer.step()
    gain_update = float((module.raw_gain.detach() - before).abs().max())
    if gain_update <= 0:
        raise RuntimeError("TSDR raw_gain did not update")

    optimizer.zero_grad(set_to_none=True)
    out1 = module([p1, p2, p3, p4, p5])
    logits1 = module._tsdr_gate_logits
    (sum(x.float().square().mean() for x in out1) + 0.05 * torch.nn.functional.softplus(-logits1).mean()).backward()

    def group_grad(prefix):
        grads = [p.grad for name, p in module.named_parameters() if name.startswith(prefix) and p.requires_grad]
        finite = bool(grads) and all(g is not None and torch.isfinite(g).all() for g in grads)
        nonzero = sum(float(g.abs().sum()) for g in grads if g is not None) if grads else 0.0
        return finite, nonzero

    proj_finite, proj_grad = group_grad("detail_project.")
    gate_finite, gate_grad = group_grad("gate_net.")
    if not proj_finite or proj_grad <= 0:
        raise RuntimeError(f"TSDR projection step-2 gradient failed: {proj_finite}/{proj_grad}")
    if not gate_finite or gate_grad <= 0:
        raise RuntimeError(f"TSDR gate step-2 gradient failed: {gate_finite}/{gate_grad}")
    raw_grad2 = module.raw_gain.grad
    if raw_grad2 is None or not torch.isfinite(raw_grad2).all() or float(raw_grad2.abs().sum()) <= 0:
        raise RuntimeError("TSDR raw_gain step-2 gradient failed")

    with torch.no_grad():
        phase = module.pixel_unshuffle(module._high_pass(p1))
    if phase.shape[-2:] != p2.shape[-2:]:
        raise RuntimeError("PixelUnshuffle/P2 geometry failed")
    return {
        "identity_max_abs_error": identity_error,
        "p2_classification_max_abs_error": classification_error,
        "step1_raw_gain_update_max_abs": gain_update,
        "step2_projection_grad_abs_sum": proj_grad,
        "step2_gate_grad_abs_sum": gate_grad,
        "step2_raw_gain_grad_abs_sum": float(raw_grad2.abs().sum()),
        "pixel_unshuffle_shape": list(phase.shape),
        "p2_shape": list(p2.shape),
    }


def _compare_e10a_initialization(e16a):
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils.torch_utils import init_seeds

    init_seeds(core.SEED, deterministic=True)
    cfg = yaml_model_load(str(_e10a_reference_yaml()))
    cfg["scale"] = "s"
    e10a = DetectionModel(cfg, nc=10, verbose=False)
    if len(e10a.model) != 29:
        raise RuntimeError("E10a reference must have 29 layers")
    _load_e10a_fair(e10a)

    e10a_state, e16a_state = e10a.state_dict(), e16a.state_dict()
    compared, mismatches, max_error = 0, [], 0.0
    for key, value in e10a_state.items():
        parts = key.split(".")
        if len(parts) < 2 or parts[0] != "model" or not parts[1].isdigit() or int(parts[1]) > 27:
            continue
        if key not in e16a_state or tuple(value.shape) != tuple(e16a_state[key].shape):
            mismatches.append(key)
            continue
        candidate = e16a_state[key]
        if not core.torch.equal(value.cpu(), candidate.cpu()):
            max_error = max(max_error, float((value.cpu() - candidate.cpu()).abs().max()))
            mismatches.append(key)
        compared += 1
    if mismatches:
        raise RuntimeError(f"E10a layers0..27 init mismatch: {mismatches[:10]}, max_error={max_error}")
    head_compared = 0
    for key, value in e10a_state.items():
        if not key.startswith(("model.28.cv2.", "model.28.cv3.", "model.28.dfl.")):
            continue
        if key not in e16a_state or not core.torch.equal(value.cpu(), e16a_state[key].cpu()):
            mismatches.append(key)
        head_compared += 1
    if mismatches:
        raise RuntimeError(f"E10a Detect-tower init mismatch: {mismatches[:10]}")
    del e10a
    return {
        "layers_0_27_tensor_count": compared,
        "detect_tower_tensor_count": head_compared,
        "exact_elementwise_equal": True,
        "max_abs_error": max_error,
    }


def _model_initial_hashes(model):
    state = model.state_dict()
    e10a_state = _state_prefix(state, tuple(f"model.{i}." for i in range(28)))
    return {"e10a_layers_0_27": _tensor_dict_hash(e10a_state), "tsdr": _tsdr_hashes(model.model[28])}


def p2_preflight(report, eval_only=False):
    del eval_only
    from ultralytics.nn.modules import DeepSemanticGuide, SemanticDetailGate, TSDRDetect  # noqa: F401
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils import yaml_load
    from ultralytics.utils.torch_utils import init_seeds

    report["p2_script_revision"] = SCRIPT_REVISION
    print(f"E16a script revision: {SCRIPT_REVISION}", flush=True)
    required = (
        MODEL_YAML, REFERENCE_E1_YAML, DSSG_SOURCE, TSDR_SOURCE, TSDR_LOSS_SOURCE, MATCHING_HELPER,
        EMBEDDED_E10A_YAML, CORE_RUNNER, FROZEN_E10A_METRICS,
        core.SOURCE / "ultralytics/nn/tasks.py",
        core.SOURCE / "ultralytics/utils/loss.py",
        core.SOURCE / "ultralytics/utils/tal.py",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if Path(core.ultralytics.__file__).resolve().parent != (core.SOURCE / "ultralytics").resolve():
        raise RuntimeError("Wrong Ultralytics source")
    if not core.PRETRAINED.is_file() or core.sha256(core.PRETRAINED) != core.PRETRAINED_SHA256:
        raise RuntimeError("Original yolo11s.pt missing or changed")

    cfg = yaml_model_load(str(MODEL_YAML))
    original_cfg = yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml")
    validate_p2_config(cfg, original_cfg)

    smoke_signature_source = inspect.getsource(core.smoke_signature)
    if "p2_architecture" not in smoke_signature_source:
        raise RuntimeError(
            "Base smoke_signature no longer binds report['p2_architecture']; "
            "E16a cannot guarantee code-hash smoke invalidation."
        )
    out = Path(report["report_dir"])
    for path in required:
        shutil.copy2(path, out / path.name)

    report["p2_architecture"] = {
        "yaml": str(MODEL_YAML),
        "yaml_sha256": core.sha256(MODEL_YAML),
        "e10a_reference_yaml": str(_e10a_reference_yaml()),
        "strides": [4, 8, 16, 32],
        "variant": "complete E10a plus TSDRDetect at layer28",
        "tsdr": {
            "layer": 28,
            "inputs": [0, 27, 26, 19, 22],
            "runtime_channels": [32, 128, 128, 256, 512],
            "gate_hidden": 16,
            "max_gain": 0.25,
            "detail": "fixed Gaussian3 high-pass then PixelUnshuffle(2)",
            "classification_input": "P2_sem exactly",
            "regression_input": "P2_sem + bounded_gain * tiny_gate * P1_detail",
            "gate_supervision": "training-only <=32px equivalent-side 3x3 center neighborhood",
            "gate_loss_gain": 0.05,
            "tal_assignment_changed": False,
        },
        "tal2": fast_tal.implementation_info(),
        "normalized_sha256": _smoke_code_hashes(),
        "smoke_signature_architecture_binding_verified": True,
        "smoke_signature_note": (
            "core.smoke_signature source was checked to bind p2_architecture; changing any "
            "normalized_sha256 entry invalidates the saved smoke2 gate and forces smoke2 to be rerun."
        ),
        "claim_boundary": "Gaussian high-pass, PixelUnshuffle and decoupled heads alone are not claimed as novel.",
    }

    init_seeds(core.SEED, deterministic=True)
    model = DetectionModel(cfg, nc=10, verbose=False)
    assert_p2_model(model)
    report["preflight_transfer"] = load_p2_pretrained(model)
    report["p2_architecture"]["e10a_initialization_equality"] = _compare_e10a_initialization(model)
    report["p2_architecture"]["tsdr_unit_check"] = _tsdr_two_step_check(model.model[28])
    report["initial_state_hashes"] = _model_initial_hashes(model)

    model.eval()
    with core.torch.inference_mode():
        _, raw = model(core.torch.zeros(1, 3, 256, 256))
    sizes = [list(tensor.shape[-2:]) for tensor in raw]
    if sizes != [[64, 64], [32, 32], [16, 16], [8, 8]]:
        raise RuntimeError(f"Unexpected feature sizes: {sizes}")
    report["p2_architecture"]["CPU_forward_256_feature_shapes"] = sizes
    report["p2_architecture"]["preflight_parameters"] = sum(p.numel() for p in model.parameters())
    report["p2_architecture"]["tsdr_head_parameters"] = sum(p.numel() for p in model.model[28].parameters())
    core.save_report(report)
    del model, raw
    gc.collect()

    baseline_cfg = copy.deepcopy(original_cfg)
    baseline_cfg.update(scale="s", nc=10)
    report["status"] = "checking_matching_equivalence"
    core.save_report(report)
    try:
        report["matching_equivalence"] = fast_tal.verify_equivalence(copy.deepcopy(cfg), device="cpu", baseline_cfg=baseline_cfg)
        fast_tal.assert_patch_restoration()
        core.write_json(out / "tal_equivalence.json", report["matching_equivalence"])
    except BaseException:
        core.write_json(out / "tal_equivalence.json", {"status": "failed", "error": traceback.format_exc()})
        raise
    finally:
        core.clear_gpu()
    core.save_report(report)
    print("E16a check-only architecture/init/two-step-gradient/CPU-forward/TAL2 gates passed.", flush=True)


def train(report, smoke=False):
    weights = _CORE_TRAIN(report, smoke=smoke)
    run_dir = Path(report["run_dir"])
    for path in (DSSG_SOURCE, TSDR_SOURCE, TSDR_LOSS_SOURCE, REFERENCE_E1_YAML, MATCHING_HELPER, MODEL_YAML):
        shutil.copy2(path, run_dir / path.name)
    return weights


def _completed_e1_reference():
    """E1 is informational only; E10a never uses discovery/globbing."""
    completed = []
    for path in sorted((core.ROOT / "comparison_reports").glob("e1_yolo11s_p2add_img1024_seed1_*/metrics.json")):
        if "SMOKE" in str(path).upper():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("status") == "completed":
            completed.append((path, data))
    return completed[-1] if completed else None


def _load_frozen_e10a_reference(current_report):
    if not FROZEN_E10A_METRICS.is_file():
        raise FileNotFoundError(f"Frozen E10a reference is missing: {FROZEN_E10A_METRICS}")
    reference = json.loads(FROZEN_E10A_METRICS.read_text(encoding="utf-8"))

    errors = []
    if reference.get("status") != "completed":
        errors.append(f"status={reference.get('status')!r}")
    if reference.get("experiment") != "e10a_yolo11s_p2_dssg_img1024_seed1":
        errors.append(f"experiment={reference.get('experiment')!r}")
    if reference.get("purpose") != "e10a_yolo11s_p2_dssg_img1024_seed1":
        errors.append(f"purpose={reference.get('purpose')!r}")
    if reference.get("seed") != 1:
        errors.append(f"seed={reference.get('seed')!r}")
    if reference.get("is_formal") is not True:
        errors.append(f"is_formal={reference.get('is_formal')!r}")

    training = reference.get("training", {})
    train_params = reference.get("train_params", {})
    if training.get("completed_epochs") != 200:
        errors.append(f"training.completed_epochs={training.get('completed_epochs')!r}")
    expected_train = {"epochs": 200, "imgsz": 1024, "batch": 8, "seed": 1}
    for key, expected in expected_train.items():
        if train_params.get(key) != expected:
            errors.append(f"train_params.{key}={train_params.get(key)!r}, expected={expected!r}")

    ref_protocol = reference.get("evaluation_protocol")
    current_protocol = current_report.get("evaluation_protocol")
    if not isinstance(ref_protocol, dict):
        errors.append("evaluation_protocol missing/not dict")
    else:
        if ref_protocol.get("imgsz") != 1024:
            errors.append(f"evaluation_protocol.imgsz={ref_protocol.get('imgsz')!r}")
        if ref_protocol.get("batch") != 8:
            errors.append(f"evaluation_protocol.batch={ref_protocol.get('batch')!r}")
        if ref_protocol != current_protocol:
            errors.append("evaluation_protocol differs from current E16a formal evaluation protocol")

    weights = reference.get("evaluation_weights", {})
    expected_weight_sha = "8c92316c138ea01bdbe3eacf3d275c9be15d96231c13c65818811bcc3cd2d512"
    if weights.get("sha256") != expected_weight_sha:
        errors.append(f"evaluation_weights.sha256={weights.get('sha256')!r}")

    for section, required_keys in {
        "overall": ("Precision", "Recall", "mAP50", "mAP75", "mAP50_95"),
        "area_metrics": ("AP_small", "AP_medium", "AP_large", "AR_small", "AR_medium", "AR_large"),
    }.items():
        values = reference.get(section)
        if not isinstance(values, dict):
            errors.append(f"{section} missing/not dict")
        else:
            missing = [key for key in required_keys if key not in values]
            if missing:
                errors.append(f"{section} missing keys {missing}")

    if errors:
        raise RuntimeError(
            "Frozen E10a reference validation failed; result comparison is aborted. "
            + "; ".join(errors)
        )
    return FROZEN_E10A_METRICS, reference


def _nonfinite_paths(value, prefix="report"):
    """Return JSON paths containing NaN/Inf without mutating model/metric values."""
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(_nonfinite_paths(child, f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_nonfinite_paths(child, f"{prefix}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        found.append((prefix, repr(value)))
    return found


def _sanitize_optional_diagnostics(value, prefix="diagnostic", nonfinite=None):
    """Replace only optional diagnostic NaN/Inf with None and record their exact paths.

    Accuracy metrics and success-gate inputs are never passed through this sanitizer.
    """
    if nonfinite is None:
        nonfinite = []
    if isinstance(value, dict):
        return {
            key: _sanitize_optional_diagnostics(child, f"{prefix}.{key}", nonfinite)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _sanitize_optional_diagnostics(child, f"{prefix}[{index}]", nonfinite)
            for index, child in enumerate(value)
        ]
    if isinstance(value, tuple):
        return [
            _sanitize_optional_diagnostics(child, f"{prefix}[{index}]", nonfinite)
            for index, child in enumerate(value)
        ]
    if isinstance(value, float) and not math.isfinite(value):
        nonfinite.append({"path": prefix, "value": repr(value)})
        return None
    return value


def _require_finite_number(value, path):
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Non-finite required E16a value at {path}: {number!r}")
    return number


def _diagnose_tsdr(wrapper, image_paths, max_images=32):
    """Collect actual-forward gate and residual health without re-entering the head."""
    module = wrapper.model.model[28]
    collected = []

    def hook(mod, args, output):
        del args, output
        if mod._diagnostic is None:
            raise RuntimeError("TSDR diagnostic flag produced no actual-forward statistics")
        collected.append(dict(mod._diagnostic))

    module.collect_diagnostics = True
    handle = module.register_forward_hook(hook)
    try:
        selected = [str(p) for p in image_paths[:max_images]]
        if selected:
            wrapper.predict(
                source=selected, imgsz=core.IMGSZ, batch=min(core.AREA_BATCH, len(selected)),
                device=0, conf=core.CONF, iou=core.IOU, max_det=core.MAX_DET,
                half=False, augment=False, verbose=False, save=False, stream=False,
            )
    finally:
        handle.remove()
        module.collect_diagnostics = False
        module._diagnostic = None

    if not collected:
        raise RuntimeError("TSDR inference-path diagnostic captured no values")
    fields = ("gate_mean", "gate_std", "detail_rms", "residual_rms", "p2_rms")
    averages = {field: sum(row[field] for row in collected) / len(collected) for field in fields}
    ratio = averages["residual_rms"] / max(averages["p2_rms"], 1e-12)
    return {
        "images_requested": min(max_images, len(image_paths)),
        "forward_batches": len(collected),
        "diagnostic_mode": "actual_forward_instrumentation_no_reentry",
        "gate_mean": averages["gate_mean"],
        "gate_std": averages["gate_std"],
        "gate_min": min(row["gate_min"] for row in collected),
        "gate_max": max(row["gate_max"] for row in collected),
        "detail_rms": averages["detail_rms"],
        "regression_residual_rms": averages["residual_rms"],
        "p2_sem_rms": averages["p2_rms"],
        "regression_residual_rms_over_p2_sem_rms": ratio,
    }


def evaluate(report, weights, dataset):
    _CORE_EVALUATE(report, weights, dataset)

    images = dataset[0]
    wrapper = core.YOLO(str(weights))
    trained = wrapper.model.float()
    assert_p2_model(trained)
    tsdr = trained.model[28]

    state_hashes = _model_initial_hashes(trained)
    initial = report.get("initial_state_hashes", {})
    update_audit = {
        "raw_gain_changed": state_hashes["tsdr"]["raw_gain"] != initial.get("tsdr", {}).get("raw_gain"),
        "projection_changed": state_hashes["tsdr"]["projection"] != initial.get("tsdr", {}).get("projection"),
        "gate_changed": state_hashes["tsdr"]["gate"] != initial.get("tsdr", {}).get("gate"),
        "e10a_layers_0_27_changed": state_hashes["e10a_layers_0_27"] != initial.get("e10a_layers_0_27"),
    }
    if not all(update_audit.values()):
        raise RuntimeError(f"Learned-state update gate failed: {update_audit}")

    raw = tsdr.raw_gain.detach().float().cpu()
    if not core.torch.isfinite(raw).all():
        raise RuntimeError("TSDR raw_gain contains NaN/Inf after training")
    raw_gain_stats = {
        "mean": _require_finite_number(raw.mean(), "tsdr_learned_state.raw_gain.mean"),
        "std": _require_finite_number(raw.std(unbiased=False), "tsdr_learned_state.raw_gain.std"),
        "minimum": _require_finite_number(raw.min(), "tsdr_learned_state.raw_gain.minimum"),
        "maximum": _require_finite_number(raw.max(), "tsdr_learned_state.raw_gain.maximum"),
        "mean_abs": _require_finite_number(raw.abs().mean(), "tsdr_learned_state.raw_gain.mean_abs"),
        "bounded_gain_mean_abs": _require_finite_number(
            (tsdr.max_gain * core.torch.tanh(raw)).abs().mean(),
            "tsdr_learned_state.raw_gain.bounded_gain_mean_abs",
        ),
    }
    learned_state = {
        "head_parameter_count": sum(p.numel() for p in tsdr.parameters()),
        "raw_gain": raw_gain_stats,
        "update_audit": update_audit,
        **_diagnose_tsdr(wrapper, images, max_images=32),
    }
    optional_nonfinite = []
    learned_state = _sanitize_optional_diagnostics(
        learned_state, "tsdr_learned_state", optional_nonfinite
    )
    learned_state["nonfinite_optional_diagnostics_sanitized"] = optional_nonfinite
    report["tsdr_learned_state"] = learned_state

    e1 = _completed_e1_reference()
    e10a = _load_frozen_e10a_reference(report)
    report["frozen_e10a_reference_audit"] = {
        "path": str(e10a[0]),
        "sha256": core.sha256(e10a[0]),
        "seed": e10a[1]["seed"],
        "completed_epochs": e10a[1]["training"]["completed_epochs"],
        "imgsz": e10a[1]["train_params"]["imgsz"],
        "batch": e10a[1]["train_params"]["batch"],
        "evaluation_protocol_exact_match": True,
        "selection": "pinned packaged reference; no glob/newest-result selection",
    }
    report["comparison_to_references"] = {}
    for name, ref in {"E1": e1, "E10a": e10a}.items():
        if ref is None:
            report["comparison_to_references"][name] = {"status": "unavailable"}
            continue
        path, data = ref
        deltas = {}
        for metric, section in (
            ("Precision", "overall"), ("Recall", "overall"),
            ("mAP50", "overall"), ("mAP75", "overall"), ("mAP50_95", "overall"),
            ("AP_small", "area_metrics"), ("AP_medium", "area_metrics"), ("AP_large", "area_metrics"),
            ("AR_small", "area_metrics"), ("AR_medium", "area_metrics"), ("AR_large", "area_metrics"),
        ):
            current_value = _require_finite_number(
                report[section][metric], f"current.{section}.{metric}"
            )
            reference_value = _require_finite_number(
                data[section][metric], f"{name}.{section}.{metric}"
            )
            deltas[metric] = _require_finite_number(
                current_value - reference_value, f"comparison_to_references.{name}.deltas.{metric}"
            )
        report["comparison_to_references"][name] = {"status": "computed", "reference": str(path), "deltas": deltas}

    if e10a is None:
        report["success_gate"] = {"passed": False, "status": "no_completed_E10a_reference"}
    else:
        d = report["comparison_to_references"]["E10a"]["deltas"]
        passed = bool(
            d["mAP50_95"] >= 0.002 and d["AP_small"] >= 0.005
            and d["Precision"] >= -0.002
            and d["AP_medium"] >= -0.003 and d["AP_large"] >= -0.005
            and update_audit["raw_gain_changed"] and update_audit["projection_changed"] and update_audit["gate_changed"]
        )
        report["success_gate"] = {
            "requirements": {
                "mAP50_95_min_delta": 0.002, "AP_small_min_delta": 0.005,
                "Precision_min_delta": -0.002,
                "AP_medium_min_delta": -0.003, "AP_large_min_delta": -0.005,
                "batch8_img1024_stable": True, "raw_gain_and_gate_updated": True,
            },
            "delta_vs_E10a_seed1": d,
            "module_update_audit": update_audit,
            "passed": passed,
            "failure_policy": "report only; no automatic tuning, extension, or E16b",
        }

    remaining_nonfinite = _nonfinite_paths(report)
    if remaining_nonfinite:
        raise RuntimeError(
            "E16a report contains non-finite values before JSON save: "
            + "; ".join(f"{path}={value}" for path, value in remaining_nonfinite[:20])
        )
    core.save_report(report)
    del wrapper, trained


def install_overrides():
    core.EXPERIMENT = EXPERIMENT
    core.SCRIPT_REVISION = SCRIPT_REVISION
    core.MODEL_YAML = MODEL_YAML
    core.SMOKE_GATE = SMOKE_GATE
    core.MATCHING_HELPER = MATCHING_HELPER
    core.AUTO_SHUTDOWN = False
    core.SHUTDOWN_ON_FAILURE = False
    core.__file__ = str(Path(__file__).resolve())
    core.transfer_key = transfer_key
    core.plan_transfer = plan_transfer
    core.load_p2_pretrained = load_p2_pretrained
    core.assert_p2_model = assert_p2_model
    core.validate_p2_config = validate_p2_config
    core.p2_preflight = p2_preflight
    core.train = train
    core.evaluate = evaluate


if __name__ == "__main__":
    install_overrides()
    core.main()
