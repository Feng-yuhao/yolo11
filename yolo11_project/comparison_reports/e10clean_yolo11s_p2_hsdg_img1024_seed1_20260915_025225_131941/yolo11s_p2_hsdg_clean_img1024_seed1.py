#!/usr/bin/env python3
"""E10-clean: E1 plus clean hierarchical spatial semantic-detail guidance.

Formal protocol: YOLO11s, VisDrone, imgsz 1024, batch 8, seed 1,
200 epochs. The D10-inactive P5/sparse-pooling/channel branch is removed;
only P4->P3 and P3->P2 spatially gated residual guidance remains. Automatic
shutdown is disabled for check, smoke, formal training and failure paths.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
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
    raise ImportError("E10-clean requires p2_tal_chunked_vgpu32.py beside this file") from exc
sys.modules["p2_tal_chunked"] = fast_tal

try:
    import yolo11s_p2_img1024_seed1 as core
except ImportError as exc:
    raise ImportError("E10-clean requires the frozen E1 runner beside this file") from exc


EXPERIMENT = "e10clean_yolo11s_p2_hsdg_img1024_seed1"
SCRIPT_REVISION = "hsdg_clean_e1_insert_seed1_200e_tal2_v1"
MODEL_YAML = HERE / "yolo11s-p2-hsdg-clean.yaml"
REFERENCE_E1_YAML = core.ROOT / "yolo11s-p2-add.yaml"
HSDG_SOURCE = core.SOURCE / "ultralytics/nn/modules/hsdg.py"
MATCHING_HELPER = Path(fast_tal.__file__).resolve()
SMOKE_GATE = core.ROOT / "comparison_reports/e10clean_hsdg_tal2_smoke_passed.json"

_CORE_TRAIN = core.train
_CORE_EVALUATE = core.evaluate
_E1_TRANSFER_KEY = core.transfer_key
_E1_PLAN_TRANSFER = core.plan_transfer
_E1_LOAD_PRETRAINED = core.load_p2_pretrained
_E1_ASSERT_MODEL = core.assert_p2_model


def transfer_key(source_key):
    """Map original 3-head YOLO11s tensors into E10-clean by identity."""
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
            or key.startswith("model.28.cv2.0.")
            or key.startswith("model.28.cv3.0.")
            or core.class_output_key(key, 28, ("1", "2", "3"))
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


def _remap_e1_reference(reference_state):
    fair = {}
    for key, value in reference_state.items():
        if key.startswith("model.25."):
            fair[key] = value
        elif key.startswith("model.26."):
            fair[key.replace("model.26.", "model.28.", 1)] = value
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
    result = target.load_state_dict(mapping, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected loaded keys: {result.unexpected_keys}")

    reference, fair_cpu_rng, fair_cuda_rng = _load_frozen_e1_reference()
    fair_state = _remap_e1_reference(reference.state_dict())
    target.load_state_dict(fair_state, strict=False)
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
        frozen_e1_random_tensor_count=len(fair_state),
        frozen_e1_reference_yaml=str(REFERENCE_E1_YAML),
        frozen_e1_reference_yaml_sha256=core.sha256(REFERENCE_E1_YAML),
        initialization_note=(
            "All unchanged tensors use original yolo11s.pt. P2-C3k2 and the complete Detect "
            "initialization are copied tensor-for-tensor from deterministic E1. Only the two "
            "clean HSDG stages are new."
        ),
    )
    del mapping, fair_state, reference, original
    return audit


def assert_p2_model(model):
    if len(model.model) != 29 or model.yaml.get("scale") != "s":
        raise ValueError("Expected the 29-layer s-scale E10-clean architecture")
    if model.model[-1].nc != 10 or model.model[-1].f != [27, 26, 19, 22]:
        raise ValueError("Wrong E10-clean P2/P3/P4/P5 Detect inputs")
    if list(model.stride.cpu().tolist()) != [4.0, 8.0, 16.0, 32.0]:
        raise ValueError(f"Wrong strides: {model.stride}")
    for index in (11, 14, 23):
        layer = model.model[index]
        if layer.__class__.__name__ != "Upsample" or layer.mode != "nearest" or layer.scale_factor != 2.0:
            raise ValueError(f"Layer {index} must remain nearest-neighbor 2x upsampling")
    p4p3 = model.model[26]
    p3p2 = model.model[27]
    if p4p3.__class__.__name__ != "P4P3SpatialGuide" or p4p3.f != [16, 19]:
        raise ValueError("Layer 26 must be P4P3SpatialGuide fed only by P3/P4")
    if (p4p3.p3_channels, p4p3.p4_channels) != (128, 256):
        raise ValueError("P4P3SpatialGuide runtime channels differ from the audited design")
    if p3p2.__class__.__name__ != "P3P2DetailGuide" or p3p2.f != [25, 26]:
        raise ValueError("Layer 27 must be P3P2DetailGuide fed by P2/P3_sem")
    if (p3p2.p2_channels, p3p2.p3_channels) != (128, 128):
        raise ValueError("P3P2DetailGuide runtime channels differ from the audited design")
    forbidden_attributes = ("p5_descriptor", "p4_descriptor", "channel_gate", "pool")
    if any(hasattr(p4p3, name) for name in forbidden_attributes):
        raise ValueError("E10-clean accidentally retained an inactive E10a descriptor/channel branch")
    if model.model[10].__class__.__name__ != "C2PSA":
        raise ValueError("Original C2PSA must remain unchanged")
    forbidden = {layer.__class__.__name__ for layer in model.model} & {"PGALP", "P3SDE", "DySample"}
    if forbidden:
        raise ValueError(f"E10-clean contains forbidden experimental modules: {sorted(forbidden)}")


def validate_p2_config(cfg, original_cfg):
    del original_cfg
    from ultralytics.utils import yaml_load

    e1_cfg = yaml_load(REFERENCE_E1_YAML)
    expected_head = copy.deepcopy(e1_cfg["head"][:-1])
    expected_head.append([[16, 19], 1, "P4P3SpatialGuide", [0.05, 16]])
    expected_head.append([[25, 26], 1, "P3P2DetailGuide", [0.05, 16]])
    expected_head.append([[27, 26, 19, 22], 1, "Detect", ["nc"]])
    checks = [
        ("backbone", core.canonical_layers(cfg.get("backbone", [])), core.canonical_layers(e1_cfg["backbone"])),
        ("head", core.canonical_layers(cfg.get("head", [])), core.canonical_layers(expected_head)),
        ("scale", cfg.get("scale"), "s"),
        ("nc", cfg.get("nc"), 10),
        ("s_scaling", cfg.get("scales", {}).get("s"), e1_cfg["scales"]["s"]),
    ]
    differences = [(name, actual, expected) for name, actual, expected in checks if actual != expected]
    if differences:
        details = "; ".join(f"{name}: expected={expected!r}, actual={actual!r}" for name, actual, expected in differences)
        raise ValueError("E10-clean YAML is not the audited HSDG-only addition to E1. " + details)


def _all_finite_gradients(module):
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    return gradients and all(gradient is not None and core.torch.isfinite(gradient).all() for gradient in gradients)


def _module_unit_check(p4p3, p3p2):
    p2 = core.torch.randn(2, 128, 64, 80, requires_grad=True)
    p3 = core.torch.randn(2, 128, 32, 40, requires_grad=True)
    p4 = core.torch.randn(2, 256, 16, 20, requires_grad=True)
    p3_sem = p4p3([p3, p4])
    p2_sem = p3p2([p2, p3_sem])
    if p3_sem.shape != p3.shape or p2_sem.shape != p2.shape:
        raise RuntimeError("HSDG shape-preservation check failed")
    if not core.torch.isfinite(p3_sem).all() or not core.torch.isfinite(p2_sem).all():
        raise RuntimeError("HSDG finite-output check failed")
    p3_change = float((p3_sem.detach() - p3.detach()).abs().max())
    p2_change = float((p2_sem.detach() - p2.detach()).abs().max())
    _, p3_spatial = p4p3.routing(p3.detach(), p4.detach())
    _, p2_spatial = p3p2.routing(p2.detach(), p3_sem.detach())
    if float((p3_spatial - 0.5).abs().max()) > 1e-6 or float((p2_spatial - 0.5).abs().max()) > 1e-6:
        raise RuntimeError("HSDG spatial initialization is not neutral")
    p2_sem.float().square().mean().backward()
    if not _all_finite_gradients(p4p3) or not _all_finite_gradients(p3p2):
        raise RuntimeError("HSDG backward/gradient check failed")
    if not 0.0 < p3_change < 0.5 or not 0.0 < p2_change < 0.5:
        raise RuntimeError(f"HSDG residual initialization is unstable: {p3_change}, {p2_change}")
    return dict(
        input_shapes=[list(p2.shape), list(p3.shape), list(p4.shape)],
        output_shapes=[list(p2_sem.shape), list(p3_sem.shape)],
        initial_max_abs_residual=dict(P3=p3_change, P2=p2_change),
        initial_P3_spatial_mean=float(p3_spatial.mean()),
        initial_P2_spatial_mean=float(p2_spatial.mean()),
        parameter_count=sum(parameter.numel() for parameter in p4p3.parameters())
        + sum(parameter.numel() for parameter in p3p2.parameters()),
        removed_branches=["P5 descriptor", "P4/P5 sparse Top-K pooling", "channel gate"],
        layer_scale_initial_mean=dict(
            P3=float(p4p3.layer_scale.detach().mean()),
            P2=float(p3p2.layer_scale.detach().mean()),
        ),
    )


def p2_preflight(report, eval_only=False):
    del eval_only
    from ultralytics.nn.modules import P4P3SpatialGuide, P3P2DetailGuide  # noqa: F401
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils import yaml_load
    from ultralytics.utils.torch_utils import init_seeds

    report["p2_script_revision"] = SCRIPT_REVISION
    print(f"E10-clean HSDG script revision: {SCRIPT_REVISION}", flush=True)
    for path in (MODEL_YAML, REFERENCE_E1_YAML, HSDG_SOURCE, MATCHING_HELPER):
        if not path.is_file():
            raise FileNotFoundError(path)
    if Path(core.ultralytics.__file__).resolve().parent != (core.SOURCE / "ultralytics").resolve():
        raise RuntimeError("Wrong Ultralytics source; activate editable yolo11 environment")
    if not core.PRETRAINED.is_file() or core.sha256(core.PRETRAINED) != core.PRETRAINED_SHA256:
        raise RuntimeError("Original yolo11s.pt missing or changed")

    cfg = yaml_model_load(str(MODEL_YAML))
    original_cfg = yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml")
    validate_p2_config(cfg, original_cfg)
    code_files = [
        "nn/tasks.py", "nn/modules/hsdg.py", "nn/modules/head.py", "nn/modules/conv.py",
        "engine/model.py", "engine/trainer.py", "models/yolo/detect/train.py", "utils/tal.py", "utils/loss.py",
    ]
    hashes = {
        name: hashlib.sha256((core.SOURCE / "ultralytics" / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for name in code_files
    }
    report["p2_architecture"] = dict(
        yaml=str(MODEL_YAML),
        yaml_sha256=core.sha256(MODEL_YAML),
        reference_e1_yaml=str(REFERENCE_E1_YAML),
        reference_e1_yaml_sha256=core.sha256(REFERENCE_E1_YAML),
        original_weights_sha256=core.sha256(core.PRETRAINED),
        strides=[4, 8, 16, 32],
        variant=(
            "E10-clean = frozen E1 P2 baseline + P4P3SpatialGuide at layer 26 and "
            "P3P2DetailGuide at layer 27. Projected P4 semantics are selected at P3, "
            "then enhanced P3 semantics are detail-gated into fused P2."
        ),
        hsdg=dict(
            p4p3_layer=26,
            p4p3_inputs=[16, 19],
            p3p2_layer=27,
            p3p2_inputs=[25, 26],
            runtime_channels=dict(P2=128, P3=128, P4=256),
            gate_hidden=16,
            layer_scale_init=0.05,
            uses_ground_truth=False,
            removed_after_d10=["P5 descriptor", "sparse Top-K pooling", "channel gate"],
            upsampling="nearest alignment only; original layers 11/14/23 unchanged",
        ),
        tal2=fast_tal.implementation_info(),
        source_code_sha256=hashes,
        fairness=(
            "All formal hyperparameters equal E1. Same original yolo11s.pt and deterministic "
            "E1 P2/Detect initialization; only the two clean HSDG stages are new."
        ),
        claim_boundary=(
            "This is a causal consolidation, not a new module stack. It removes the "
            "D10-inactive E10a channel path and retains only two spatial guidance paths."
        ),
    )
    out = Path(report["report_dir"])
    for path in (MODEL_YAML, REFERENCE_E1_YAML, HSDG_SOURCE, MATCHING_HELPER):
        shutil.copy2(path, out / path.name)

    init_seeds(core.SEED, deterministic=True)
    model = DetectionModel(cfg, nc=10, verbose=False)
    assert_p2_model(model)
    report["preflight_transfer"] = load_p2_pretrained(model)
    report["p2_architecture"]["module_unit_check"] = _module_unit_check(
        model.model[26], model.model[27]
    )
    model.eval()
    with core.torch.inference_mode():
        _, raw = model(core.torch.zeros(1, 3, 256, 256))
    sizes = [list(tensor.shape[-2:]) for tensor in raw]
    if sizes != [[64, 64], [32, 32], [16, 16], [8, 8]]:
        raise RuntimeError(f"Unexpected feature shapes: {sizes}")
    report["p2_architecture"]["CPU_forward_256_feature_shapes"] = sizes
    report["p2_architecture"]["preflight_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    core.save_report(report)
    del model, raw
    gc.collect()
    print("E10-clean structure/transfer/HSDG/CPU forward checks passed. Strides: 4,8,16,32.", flush=True)

    baseline_cfg = copy.deepcopy(original_cfg)
    baseline_cfg.update(scale="s", nc=10)
    print("Checking strict TAL2 equivalence on CPU; CUDA batch8/1024 is checked by --smoke2.", flush=True)
    report["status"] = "checking_matching_equivalence"
    core.save_report(report)
    try:
        report["matching_equivalence"] = fast_tal.verify_equivalence(
            copy.deepcopy(cfg), device="cpu", baseline_cfg=baseline_cfg
        )
        fast_tal.assert_patch_restoration()
        core.write_json(out / "tal_equivalence.json", report["matching_equivalence"])
    except BaseException:
        core.write_json(out / "tal_equivalence.json", dict(status="failed", device="cpu", error=traceback.format_exc()))
        raise
    finally:
        core.clear_gpu()
    core.save_report(report)
    print("TAL2 equivalence passed. This is not yet the 1024px memory smoke.", flush=True)


def train(report, smoke=False):
    weights = _CORE_TRAIN(report, smoke=smoke)
    run_dir = Path(report["run_dir"])
    for path in (HSDG_SOURCE, REFERENCE_E1_YAML, MATCHING_HELPER):
        shutil.copy2(path, run_dir / path.name)
    return weights


def _frozen_e1_report():
    candidates = sorted(
        path for path in (core.ROOT / "comparison_reports").glob(
            "e1_yolo11s_p2add_img1024_seed1_*/metrics.json"
        ) if "SMOKE" not in str(path).upper()
    )
    if len(candidates) != 1:
        return None
    data = json.loads(candidates[0].read_text(encoding="utf-8"))
    return (candidates[0], data) if data.get("status") == "completed" else None


def _frozen_e10a_report():
    candidates = sorted(
        path for path in (core.ROOT / "comparison_reports").glob(
            "e10a_yolo11s_p2_dssg_img1024_seed1_*/metrics.json"
        ) if "SMOKE" not in str(path).upper()
    )
    if len(candidates) != 1:
        return None
    data = json.loads(candidates[0].read_text(encoding="utf-8"))
    return (candidates[0], data) if data.get("status") == "completed" else None


def _metric_deltas(report, reference):
    sections = {
        "overall": ("Precision", "Recall", "mAP50", "mAP75", "mAP50_95"),
        "area_metrics": ("AP_all", "AP_small", "AP_medium", "AP_large"),
    }
    return {
        section: {
            name: float(report[section][name]) - float(reference[section][name])
            for name in names
        }
        for section, names in sections.items()
    }


def evaluate(report, weights, dataset):
    _CORE_EVALUATE(report, weights, dataset)
    frozen = _frozen_e1_report()
    if frozen is None:
        report["comparison_to_frozen_E1"] = dict(
            status="unavailable", note="Expected exactly one completed formal E1 metrics.json"
        )
    else:
        path, reference = frozen
        comparison = dict(status="computed", reference=str(path), deltas=_metric_deltas(report, reference))
        deltas = comparison["deltas"]
        comparison["gate"] = dict(
            thresholds=dict(mAP50_95=0.002, AP_small=0.0, AP_medium_min=0.0),
            passed=bool(
                deltas["overall"]["mAP50_95"] >= 0.002
                and deltas["area_metrics"]["AP_small"] >= 0.0
                and deltas["area_metrics"]["AP_medium"] >= 0.0
            ),
        )
        report["comparison_to_frozen_E1"] = comparison

    frozen_e10a = _frozen_e10a_report()
    if frozen_e10a is None:
        report["comparison_to_frozen_E10a"] = dict(
            status="unavailable", note="Expected exactly one completed formal E10a metrics.json"
        )
    else:
        path, reference = frozen_e10a
        deltas = _metric_deltas(report, reference)
        fewer_parameters = int(report["model_profile"]["parameters"]) < int(reference["model_profile"]["parameters"])
        fewer_flops = float(report["model_profile"]["GFLOPs"]) < float(reference["model_profile"]["GFLOPs"])
        report["comparison_to_frozen_E10a"] = dict(
            status="computed",
            reference=str(path),
            deltas=deltas,
            consolidation_gate=dict(
                thresholds=dict(mAP50_95_min=-0.001, AP_small_min=-0.001, AP_medium_min=-0.002),
                fewer_parameters=fewer_parameters,
                fewer_GFLOPs=fewer_flops,
                passed=bool(
                    deltas["overall"]["mAP50_95"] >= -0.001
                    and deltas["area_metrics"]["AP_small"] >= -0.001
                    and deltas["area_metrics"]["AP_medium"] >= -0.002
                    and fewer_parameters
                    and fewer_flops
                ),
            ),
        )

    trained = core.YOLO(str(weights)).model.float()
    assert_p2_model(trained)
    p4p3, p3p2 = trained.model[26], trained.model[27]
    report["hsdg_learned_state"] = dict(
        P3_layer_scale=dict(
            mean=float(p4p3.layer_scale.detach().mean()),
            std=float(p4p3.layer_scale.detach().std()),
            minimum=float(p4p3.layer_scale.detach().min()),
            maximum=float(p4p3.layer_scale.detach().max()),
        ),
        P2_layer_scale=dict(
            mean=float(p3p2.layer_scale.detach().mean()),
            std=float(p3p2.layer_scale.detach().std()),
            minimum=float(p3p2.layer_scale.detach().min()),
            maximum=float(p3p2.layer_scale.detach().max()),
        ),
        P3_spatial_output_weight_l2=float(p4p3.spatial_gate.net[-1].weight.detach().norm()),
        P2_spatial_output_weight_l2=float(p3p2.spatial_gate.net[-1].weight.detach().norm()),
        note="Nonzero learned gate/scale parameters show module use, not accuracy or causal proof.",
    )
    del trained


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
