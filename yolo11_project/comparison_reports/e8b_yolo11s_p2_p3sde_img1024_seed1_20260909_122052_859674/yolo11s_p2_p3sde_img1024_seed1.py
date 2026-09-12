#!/usr/bin/env python3
"""E8b-clean: frozen E1 P2 baseline plus one P3 detail-semantic enhancer.

Formal protocol is unchanged: YOLO11s, VisDrone, 1024, batch 8, seed 1,
200 epochs.  TAL2 changes only the image chunk used during positive matching;
the already validated full forward batch, GT padding and loss normalization
remain unchanged.
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

# Load the validated chunk-2 helper, then redirect the name imported lazily by
# the frozen E1 runner.  p2_tal_chunked_vgpu32 itself first imports the
# audited chunk-1 helper, so its equivalence implementation remains available.
try:
    import p2_tal_chunked_vgpu32 as fast_tal
except ImportError as exc:
    raise ImportError("E8b-clean requires p2_tal_chunked_vgpu32.py beside this file") from exc
sys.modules["p2_tal_chunked"] = fast_tal

try:
    import yolo11s_p2_img1024_seed1 as core
except ImportError as exc:
    raise ImportError("E8b-clean requires the frozen E1 runner beside this file") from exc


EXPERIMENT = "e8b_yolo11s_p2_p3sde_img1024_seed1"
SCRIPT_REVISION = "p3sde_clean_e1_insert_e8b_tal2_v1"
MODEL_YAML = HERE / "yolo11s-p2-p3sde.yaml"
REFERENCE_E1_YAML = core.ROOT / "yolo11s-p2-add.yaml"
P3SDE_SOURCE = core.SOURCE / "ultralytics/nn/modules/p3sde.py"
MATCHING_HELPER = Path(fast_tal.__file__).resolve()
SMOKE_GATE = core.ROOT / "comparison_reports/e8b_p3sde_clean_tal2_smoke_passed.json"

_CORE_TRAIN = core.train
_CORE_EVALUATE = core.evaluate
_E1_TRANSFER_KEY = core.transfer_key
_E1_PLAN_TRANSFER = core.plan_transfer
_E1_LOAD_PRETRAINED = core.load_p2_pretrained
_E1_ASSERT_MODEL = core.assert_p2_model


def transfer_key(source_key):
    """Map original YOLO11s weights into E8b-clean by semantic layer identity."""
    parts = source_key.split(".")
    if len(parts) < 3 or parts[0] != "model" or not parts[1].isdigit():
        raise ValueError(f"Unexpected source key: {source_key}")
    index = int(parts[1])
    if 0 <= index <= 16:
        return source_key
    if 17 <= index <= 22:
        parts[1] = str(index + 1)  # P3SDE is inserted at target layer 17.
        return ".".join(parts)
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
            key.startswith("model.17.")              # new P3SDE
            or key.startswith("model.26.")           # E1 P2 fusion block
            or key.startswith("model.27.cv2.0.")     # new P2 regression branch
            or key.startswith("model.27.cv3.0.")     # new P2 classification branch
            or core.class_output_key(key, 27, ("1", "2", "3"))
        )
        if not allowed:
            raise ValueError(f"Unexpected uninitialized tensor: {key}")
    if len(skipped) != 6:
        raise ValueError(f"Expected six class-output skips, got {len(skipped)}")
    return mapped, dict(
        rule=("original 0..16 unchanged; original 17..22 -> 18..23; "
              "Detect 23->27 and branches 0..2->1..3"),
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
            fair[key.replace("model.25.", "model.26.", 1)] = value
        elif key.startswith("model.26."):
            fair[key.replace("model.26.", "model.27.", 1)] = value
    return fair


def _load_frozen_e1_reference():
    """Rebuild E1's deterministic random P2/Detect initialization."""
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
        target_index = index if index <= 16 else index + 1
        candidate = target.model[target_index]
        if type(original.model[index]) is not type(candidate):
            raise ValueError(f"Unexpected architecture change at original layer {index}")
        if index == 17:
            if candidate.f != 16:
                raise ValueError("Target bottom-up P4 route must read raw P3 layer 16")
        elif original.model[index].f != candidate.f:
            raise ValueError(f"Layer source changed unexpectedly at original layer {index}")

    mapping, audit = plan_transfer(original.state_dict(), target.state_dict())
    result = target.load_state_dict(mapping, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected loaded keys: {result.unexpected_keys}")

    # Recreate the exact E1 P2/Detect initialization and restore its post-build
    # RNG state, matching the audited E1 fairness protocol.
    reference, fair_cpu_rng, fair_cuda_rng = _load_frozen_e1_reference()
    fair_state = _remap_e1_reference(reference.state_dict())
    target.load_state_dict(fair_state, strict=False)
    core.torch.random.set_rng_state(fair_cpu_rng)
    if fair_cuda_rng is not None:
        core.torch.cuda.set_rng_state_all(fair_cuda_rng)

    actual = target.state_dict()
    if any(not core.torch.equal(actual[k].cpu(), v.cpu()) for k, v in mapping.items()):
        raise RuntimeError("Original pretrained tensor equality check failed")
    if any(not core.torch.equal(actual[k].cpu(), v.cpu()) for k, v in fair_state.items()):
        raise RuntimeError("Frozen E1 P2/Detect tensor equality check failed")
    audit.update(
        verified_tensor_equality=True,
        frozen_e1_random_tensor_count=len(fair_state),
        frozen_e1_reference_yaml=str(REFERENCE_E1_YAML),
        frozen_e1_reference_yaml_sha256=core.sha256(REFERENCE_E1_YAML),
        initialization_note=(
            "All unchanged YOLO11s tensors come from original yolo11s.pt. P2-C3k2 and complete "
            "Detect initialization are copied tensor-for-tensor from the deterministic E1 reference. "
            "Only P3SDE is additionally initialized; no DySample exists in this model."
        ),
    )
    del mapping, fair_state, reference, original
    return audit


def assert_p2_model(model):
    if len(model.model) != 28 or model.yaml.get("scale") != "s":
        raise ValueError("Expected the 28-layer s-scale E8b-clean architecture")
    if model.model[-1].nc != 10 or model.model[-1].f != [26, 17, 20, 23]:
        raise ValueError("Wrong E8b-clean P2/P3/P4/P5 Detect inputs")
    if list(model.stride.cpu().tolist()) != [4.0, 8.0, 16.0, 32.0]:
        raise ValueError(f"Wrong strides: {model.stride}")
    for index in (11, 14):
        layer = model.model[index]
        if layer.__class__.__name__ != "Upsample" or layer.mode != "nearest" or layer.scale_factor != 2.0:
            raise ValueError(f"Layer {index} must remain nearest-neighbor 2x upsampling")
    enhancer = model.model[17]
    if enhancer.__class__.__name__ != "P3SDE":
        raise ValueError("Layer 17 must be registered P3SDE")
    if (enhancer.channels, enhancer.hidden, enhancer.context_kernel, enhancer.context_dilation) != (128, 128, 5, 2):
        raise ValueError("P3SDE configuration differs from audited E8b-clean")
    if model.model[18].f != 16:
        raise ValueError("P4/P5 bottom-up path must remain sourced from raw P3")
    if model.model[24].__class__.__name__ != "Upsample" or model.model[24].f != 17:
        raise ValueError("P3->P2 must remain nearest and read enhanced P3")
    if model.model[10].__class__.__name__ != "C2PSA":
        raise ValueError("Original C2PSA must remain unchanged")


def validate_p2_config(cfg, original_cfg):
    del original_cfg  # Comparison is against the frozen E1 YAML.
    from ultralytics.utils import yaml_load

    e1_cfg = yaml_load(REFERENCE_E1_YAML)
    expected_head = copy.deepcopy(e1_cfg["head"][:6])
    expected_head.append([-1, 1, "P3SDE", [1.0, 5, 2, 0.001]])
    tail = copy.deepcopy(e1_cfg["head"][6:])
    tail[0][0] = 16
    tail[6][0] = 17
    tail[-1][0] = [26, 17, 20, 23]
    expected_head.extend(tail)
    checks = [
        ("backbone", core.canonical_layers(cfg.get("backbone", [])), core.canonical_layers(e1_cfg["backbone"])),
        ("head", core.canonical_layers(cfg.get("head", [])), core.canonical_layers(expected_head)),
        ("scale", cfg.get("scale"), "s"),
        ("nc", cfg.get("nc"), 10),
        ("s_scaling", cfg.get("scales", {}).get("s"), e1_cfg["scales"]["s"]),
    ]
    differences = [(name, actual, expected) for name, actual, expected in checks if actual != expected]
    if differences:
        details = "; ".join(f"{n}: expected={e!r}, actual={a!r}" for n, a, e in differences)
        raise ValueError("E8b-clean YAML is not the audited one-module addition to E1. " + details)


def _module_unit_check(enhancer):
    x = core.torch.randn(2, enhancer.channels, 24, 32, requires_grad=True)
    y = enhancer(x)
    if y.shape != x.shape or not core.torch.isfinite(y).all():
        raise RuntimeError("P3SDE forward/shape check failed")
    initial_max_change = float((y.detach() - x.detach()).abs().max())
    y.float().square().mean().backward()
    gradients = [p.grad for p in enhancer.parameters() if p.requires_grad]
    if not gradients or any(g is None or not core.torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("P3SDE backward/gradient check failed")
    if not (0.0 < initial_max_change < 0.1):
        raise RuntimeError(f"P3SDE is not identity-like at initialization: {initial_max_change}")
    return dict(
        input_shape=list(x.shape), output_shape=list(y.shape),
        initial_max_abs_residual=initial_max_change,
        parameter_count=sum(p.numel() for p in enhancer.parameters()),
        layer_scale_initial_mean=float(enhancer.layer_scale.detach().mean()),
    )


def p2_preflight(report, eval_only=False):
    from ultralytics.nn.modules import P3SDE  # noqa: F401
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils import yaml_load
    from ultralytics.utils.torch_utils import init_seeds

    report["p2_script_revision"] = SCRIPT_REVISION
    print(f"E8b-clean P3SDE script revision: {SCRIPT_REVISION}", flush=True)
    for path in (MODEL_YAML, REFERENCE_E1_YAML, P3SDE_SOURCE, MATCHING_HELPER):
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
        "nn/tasks.py", "nn/modules/p3sde.py",
        "nn/modules/head.py", "nn/modules/conv.py", "engine/model.py",
        "engine/trainer.py", "models/yolo/detect/train.py", "utils/tal.py", "utils/loss.py",
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
            "E8b-clean = frozen E1 P2 baseline + one shape-preserving P3SDE at layer 17. "
            "All upsampling remains nearest. Enhanced P3 feeds only P3 Detect and P3->P2; "
            "P4/P5 retain raw P3 input."
        ),
        p3sde=dict(
            layer=17, runtime_channels=128, expansion=1.0, context_kernel=5,
            context_dilation=2, layer_scale_init=0.001,
            mechanisms=["local detail branch", "dilated context branch", "spatial-channel branch selection"],
        ),
        tal2=fast_tal.implementation_info(),
        source_code_sha256=hashes,
        fairness=(
            "All formal hyperparameters equal E1. Same original yolo11s.pt source and deterministic "
            "E1 P2/Detect initialization; only P3SDE parameters are additionally new."
        ),
        claim_boundary=(
            "E8b-clean tests P3 detail-semantic enhancement on E1 only. It contains no DySample."
        ),
    )
    out = Path(report["report_dir"])
    for path in (MODEL_YAML, REFERENCE_E1_YAML, P3SDE_SOURCE, MATCHING_HELPER):
        shutil.copy2(path, out / path.name)

    init_seeds(core.SEED, deterministic=True)
    model = DetectionModel(cfg, nc=10, verbose=False)
    assert_p2_model(model)
    report["preflight_transfer"] = load_p2_pretrained(model)
    report["p2_architecture"]["module_unit_check"] = _module_unit_check(model.model[17])
    model.eval()
    with core.torch.inference_mode():
        _, raw = model(core.torch.zeros(1, 3, 256, 256))
    sizes = [list(t.shape[-2:]) for t in raw]
    if sizes != [[64, 64], [32, 32], [16, 16], [8, 8]]:
        raise RuntimeError(f"Unexpected feature shapes: {sizes}")
    report["p2_architecture"]["CPU_forward_256_feature_shapes"] = sizes
    report["p2_architecture"]["preflight_parameters"] = sum(p.numel() for p in model.parameters())
    core.save_report(report)
    del model, raw
    gc.collect()
    print("E8b-clean structure/transfer/P3SDE/CPU forward checks passed. Strides: 4,8,16,32.", flush=True)

    baseline_cfg = copy.deepcopy(original_cfg)
    baseline_cfg.update(scale="s", nc=10)
    device = "cpu"
    print("Checking strict TAL2 equivalence on CPU; CUDA batch8/1024 is checked by --smoke2.", flush=True)
    report["status"] = "checking_matching_equivalence"
    core.save_report(report)
    try:
        report["matching_equivalence"] = fast_tal.verify_equivalence(
            copy.deepcopy(cfg), device=device, baseline_cfg=baseline_cfg
        )
        fast_tal.assert_patch_restoration()
        core.write_json(out / "tal_equivalence.json", report["matching_equivalence"])
    except BaseException:
        core.write_json(out / "tal_equivalence.json", dict(status="failed", device=device, error=traceback.format_exc()))
        raise
    finally:
        core.clear_gpu()
    core.save_report(report)
    print("TAL2 equivalence passed. This is not yet the 1024px memory smoke.", flush=True)


def train(report, smoke=False):
    weights = _CORE_TRAIN(report, smoke=smoke)
    run_dir = Path(report["run_dir"])
    for path in (P3SDE_SOURCE, REFERENCE_E1_YAML, MATCHING_HELPER):
        shutil.copy2(path, run_dir / path.name)
    return weights


def _frozen_e1_report():
    candidates = sorted(
        p for p in (core.ROOT / "comparison_reports").glob(
            "e1_yolo11s_p2add_img1024_seed1_*/metrics.json"
        )
        if "SMOKE" not in str(p).upper()
    )
    if len(candidates) != 1:
        return None
    data = json.loads(candidates[0].read_text(encoding="utf-8"))
    if data.get("status") != "completed":
        return None
    return candidates[0], data


def evaluate(report, weights, dataset):
    _CORE_EVALUATE(report, weights, dataset)
    frozen = _frozen_e1_report()
    if frozen is None:
        report["comparison_to_frozen_E1"] = dict(
            status="unavailable", note="Expected exactly one completed formal E1 metrics.json"
        )
        return
    path, reference = frozen
    keys = {
        "overall": ("Precision", "Recall", "mAP50", "mAP75", "mAP50_95"),
        "area_metrics": ("AP_all", "AP_small", "AP_medium", "AP_large"),
    }
    comparison = dict(status="computed", reference=str(path), deltas={})
    for section, names in keys.items():
        comparison["deltas"][section] = {
            name: float(report[section][name]) - float(reference[section][name]) for name in names
        }
    report["comparison_to_frozen_E1"] = comparison


def install_overrides():
    core.EXPERIMENT = EXPERIMENT
    core.SCRIPT_REVISION = SCRIPT_REVISION
    core.MODEL_YAML = MODEL_YAML
    core.SMOKE_GATE = SMOKE_GATE
    core.MATCHING_HELPER = MATCHING_HELPER
    # User policy: never shut down the AutoDL instance automatically, including
    # formal success, failure and interruption paths.
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
