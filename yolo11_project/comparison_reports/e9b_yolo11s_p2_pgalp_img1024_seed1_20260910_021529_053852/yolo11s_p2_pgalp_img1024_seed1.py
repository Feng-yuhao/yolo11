#!/usr/bin/env python3
"""E9b: frozen E1 P2 baseline plus P3-guided adaptive low-pass filtering.

Formal protocol: YOLO11s, VisDrone, imgsz 1024, batch 8, seed 1,
200 epochs.  E9b contains no DySample and no P3SDE.  Automatic shutdown is
disabled for check, smoke, formal training, evaluation and failure paths.
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
    raise ImportError("E9b requires p2_tal_chunked_vgpu32.py beside this file") from exc
sys.modules["p2_tal_chunked"] = fast_tal

try:
    import yolo11s_p2_img1024_seed1 as core
except ImportError as exc:
    raise ImportError("E9b requires the frozen E1 runner beside this file") from exc


EXPERIMENT = "e9b_yolo11s_p2_pgalp_img1024_seed1"
SCRIPT_REVISION = "pgalp_e1_insert_e9b_tal2_v1"
MODEL_YAML = HERE / "yolo11s-p2-pgalp.yaml"
REFERENCE_E1_YAML = core.ROOT / "yolo11s-p2-add.yaml"
PGALP_SOURCE = core.SOURCE / "ultralytics/nn/modules/pgalp.py"
MATCHING_HELPER = Path(fast_tal.__file__).resolve()
SMOKE_GATE = core.ROOT / "comparison_reports/e9b_pgalp_tal2_smoke_passed.json"

_CORE_TRAIN = core.train
_CORE_EVALUATE = core.evaluate
_E1_TRANSFER_KEY = core.transfer_key
_E1_PLAN_TRANSFER = core.plan_transfer
_E1_LOAD_PRETRAINED = core.load_p2_pretrained
_E1_ASSERT_MODEL = core.assert_p2_model


def transfer_key(source_key):
    """Map original 3-head YOLO11s tensors into E9b by semantic identity."""
    parts = source_key.split(".")
    if len(parts) < 3 or parts[0] != "model" or not parts[1].isdigit():
        raise ValueError(f"Unexpected source key: {source_key}")
    index = int(parts[1])
    if 0 <= index <= 22:
        return source_key
    if index != 23:
        raise ValueError(f"Expected original YOLO11s Detect at layer 23: {source_key}")
    parts[1] = "27"
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
            key.startswith("model.25.")          # frozen E1 P2 fusion block
            or key.startswith("model.26.")       # new PGALP
            or key.startswith("model.27.cv2.0.") # new P2 regression branch
            or key.startswith("model.27.cv3.0.") # new P2 classification branch
            or core.class_output_key(key, 27, ("1", "2", "3"))
        )
        if not allowed:
            raise ValueError(f"Unexpected uninitialized tensor: {key}")
    if len(skipped) != 6:
        raise ValueError(f"Expected six class-output skips, got {len(skipped)}")
    return mapped, dict(
        rule="original layers 0..22 unchanged; Detect 23->27 and branches 0..2->1..3",
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
            fair[key.replace("model.26.", "model.27.", 1)] = value
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
            "initialization are copied tensor-for-tensor from deterministic E1. Only PGALP is new."
        ),
    )
    del mapping, fair_state, reference, original
    return audit


def assert_p2_model(model):
    if len(model.model) != 28 or model.yaml.get("scale") != "s":
        raise ValueError("Expected the 28-layer s-scale E9b architecture")
    if model.model[-1].nc != 10 or model.model[-1].f != [26, 16, 19, 22]:
        raise ValueError("Wrong E9b P2/P3/P4/P5 Detect inputs")
    if list(model.stride.cpu().tolist()) != [4.0, 8.0, 16.0, 32.0]:
        raise ValueError(f"Wrong strides: {model.stride}")
    for index in (11, 14, 23):
        layer = model.model[index]
        if layer.__class__.__name__ != "Upsample" or layer.mode != "nearest" or layer.scale_factor != 2.0:
            raise ValueError(f"Layer {index} must remain nearest-neighbor 2x upsampling")
    module = model.model[26]
    if module.__class__.__name__ != "PGALP" or module.f != [25, 16]:
        raise ValueError("Layer 26 must be PGALP fed by fused P2 and raw P3")
    if module.kernels != (3, 5, 7) or module.sigmas != (0.8, 1.2, 1.8):
        raise ValueError("PGALP filters differ from the audited E9b design")
    if (module.p2_channels, module.p3_channels, module.hidden) != (128, 128, 32):
        raise ValueError("PGALP runtime channels differ from the audited E9b design")
    if model.model[10].__class__.__name__ != "C2PSA":
        raise ValueError("Original C2PSA must remain unchanged")


def validate_p2_config(cfg, original_cfg):
    del original_cfg
    from ultralytics.utils import yaml_load

    e1_cfg = yaml_load(REFERENCE_E1_YAML)
    expected_head = copy.deepcopy(e1_cfg["head"][:-1])
    expected_head.append([[25, 16], 1, "PGALP", [0.25, [3, 5, 7], [0.8, 1.2, 1.8], 0.1, 0.05]])
    expected_head.append([[26, 16, 19, 22], 1, "Detect", ["nc"]])
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
        raise ValueError("E9b YAML is not the audited one-module addition to E1. " + details)


def _module_unit_check(module):
    p2 = core.torch.randn(2, 128, 32, 40, requires_grad=True)
    p3 = core.torch.randn(2, 128, 16, 20, requires_grad=True)
    y = module([p2, p3])
    if y.shape != p2.shape or not core.torch.isfinite(y).all():
        raise RuntimeError("PGALP forward/shape check failed")
    initial_change = float((y.detach() - p2.detach()).abs().max())
    low = module._blur(p2.detach(), 0)
    reliability, weights = module.routing(p2.detach(), p3.detach(), low)
    expected_weight = 1.0 / len(module.kernels)
    if abs(float(reliability.mean()) - 0.5) > 1e-6:
        raise RuntimeError("PGALP neutral reliability initialization failed")
    if float((weights - expected_weight).abs().max()) > 1e-6:
        raise RuntimeError("PGALP uniform scale initialization failed")
    y.float().square().mean().backward()
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    if not gradients or any(gradient is None or not core.torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("PGALP backward/gradient check failed")
    if not 0.0 < initial_change < 0.5:
        raise RuntimeError(f"PGALP is not a stable residual at initialization: {initial_change}")
    return dict(
        input_shapes=[list(p2.shape), list(p3.shape)], output_shape=list(y.shape),
        initial_max_abs_residual=initial_change,
        initial_reliability_mean=float(reliability.mean()),
        initial_scale_weights_mean=[float(weights[:, i].mean()) for i in range(weights.shape[1])],
        parameter_count=sum(parameter.numel() for parameter in module.parameters()),
        trainable_parameter_count=sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad),
        fixed_gaussian_buffer_count=sum(buffer.numel() for buffer in module.buffers()),
        layer_scale_initial_mean=float(module.layer_scale.detach().mean()),
    )


def p2_preflight(report, eval_only=False):
    from ultralytics.nn.modules import PGALP  # noqa: F401
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils import yaml_load
    from ultralytics.utils.torch_utils import init_seeds

    report["p2_script_revision"] = SCRIPT_REVISION
    print(f"E9b PGALP script revision: {SCRIPT_REVISION}", flush=True)
    for path in (MODEL_YAML, REFERENCE_E1_YAML, PGALP_SOURCE, MATCHING_HELPER):
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
        "nn/tasks.py", "nn/modules/pgalp.py", "nn/modules/head.py", "nn/modules/conv.py",
        "engine/model.py", "engine/trainer.py", "models/yolo/detect/train.py", "utils/tal.py", "utils/loss.py",
    ]
    hashes = {
        name: hashlib.sha256((core.SOURCE / "ultralytics" / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for name in code_files
    }
    report["p2_architecture"] = dict(
        yaml=str(MODEL_YAML), yaml_sha256=core.sha256(MODEL_YAML),
        reference_e1_yaml=str(REFERENCE_E1_YAML), reference_e1_yaml_sha256=core.sha256(REFERENCE_E1_YAML),
        original_weights_sha256=core.sha256(core.PRETRAINED), strides=[4, 8, 16, 32],
        variant=(
            "E9b = frozen E1 P2 baseline + PGALP at layer 26. Raw P3 guides a spatial foreground "
            "reliability map and per-location 3/5/7 Gaussian selection for fused P2."
        ),
        pgalp=dict(
            layer=26, inputs=[25, 16], runtime_channels=[128, 128], hidden=32,
            kernels=[3, 5, 7], sigmas=[0.8, 1.2, 1.8], layer_scale_init=0.1,
            reliability_range=[0.05, 0.95], uses_ground_truth=False,
            output="shape-preserving cleaned P2; P3/P4/P5 detection paths unchanged",
        ),
        tal2=fast_tal.implementation_info(), source_code_sha256=hashes,
        fairness=(
            "All formal hyperparameters equal E1. Same original yolo11s.pt and deterministic E1 "
            "P2/Detect initialization; only PGALP parameters are new."
        ),
        claim_boundary="E9b tests PGALP independently on E1. It contains no DySample and no P3SDE.",
    )
    out = Path(report["report_dir"])
    for path in (MODEL_YAML, REFERENCE_E1_YAML, PGALP_SOURCE, MATCHING_HELPER):
        shutil.copy2(path, out / path.name)

    init_seeds(core.SEED, deterministic=True)
    model = DetectionModel(cfg, nc=10, verbose=False)
    assert_p2_model(model)
    report["preflight_transfer"] = load_p2_pretrained(model)
    report["p2_architecture"]["module_unit_check"] = _module_unit_check(model.model[26])
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
    print("E9b structure/transfer/PGALP/CPU forward checks passed. Strides: 4,8,16,32.", flush=True)

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
    for path in (PGALP_SOURCE, REFERENCE_E1_YAML, MATCHING_HELPER):
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


def evaluate(report, weights, dataset):
    _CORE_EVALUATE(report, weights, dataset)
    # THOP counts registered Conv2d layers but not functional depthwise Gaussian
    # convolutions. Report their exact MAC-derived cost separately and add it to
    # a corrected total without changing the legacy comparable GFLOPs field.
    p2_size = core.IMGSZ // 4
    gaussian_gflops = (
        2.0 * p2_size * p2_size * 128 * sum(kernel * kernel for kernel in (3, 5, 7)) / 1.0e9
    )
    report["model_profile"]["functional_gaussian_GFLOPs"] = gaussian_gflops
    reported_gflops = report["model_profile"].get("GFLOPs")
    report["model_profile"]["GFLOPs_corrected"] = (
        float(reported_gflops) + gaussian_gflops if reported_gflops is not None else None
    )
    report["model_profile"]["GFLOPs_note"] = (
        "GFLOPs is the legacy THOP value; GFLOPs_corrected adds the three functional "
        "depthwise Gaussian filters at P2 resolution."
    )
    frozen = _frozen_e1_report()
    if frozen is None:
        report["comparison_to_frozen_E1"] = dict(
            status="unavailable", note="Expected exactly one completed formal E1 metrics.json"
        )
    else:
        path, reference = frozen
        sections = {
            "overall": ("Precision", "Recall", "mAP50", "mAP75", "mAP50_95"),
            "area_metrics": ("AP_all", "AP_small", "AP_medium", "AP_large"),
        }
        comparison = dict(status="computed", reference=str(path), deltas={})
        for section, names in sections.items():
            comparison["deltas"][section] = {
                name: float(report[section][name]) - float(reference[section][name]) for name in names
            }
        report["comparison_to_frozen_E1"] = comparison

    trained = core.YOLO(str(weights)).model.float()
    assert_p2_model(trained)
    module = trained.model[26]
    report["pgalp_learned_state"] = dict(
        layer_scale_mean=float(module.layer_scale.detach().mean()),
        layer_scale_std=float(module.layer_scale.detach().std()),
        layer_scale_min=float(module.layer_scale.detach().min()),
        layer_scale_max=float(module.layer_scale.detach().max()),
        router_output_weight_l2=float(module.router[-1].weight.detach().norm()),
        router_output_bias=[float(value) for value in module.router[-1].bias.detach()],
        note="Nonzero learned routing/scale values show use, but are not an accuracy metric.",
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
