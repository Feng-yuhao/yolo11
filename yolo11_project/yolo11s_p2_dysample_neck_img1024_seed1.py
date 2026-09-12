#!/usr/bin/env python3
"""E5a existing-method ablation: E1 YOLO11s-P2 plus standard Neck DySample.

This reuses the frozen E1 train/evaluate/report core. Only the two original
P5->P4 and P4->P3 nearest-neighbor upsamplers become standard ICCV-2023
DySample (lp, groups=4, no dyscope). The E1 P3->P2 path remains untouched.
"""

from __future__ import annotations

import copy
import gc
import hashlib
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
    import yolo11s_p2_img1024_seed1 as core
except ImportError as exc:
    raise ImportError(
        "E5a requires frozen yolo11s_p2_img1024_seed1.py beside this file."
    ) from exc


EXPERIMENT = "e5a_yolo11s_p2_dysample_neck_img1024_seed1"
SCRIPT_REVISION = "standard_dysample_neck_e5a_v2_cpu_equivalence"
MODEL_YAML = HERE / "yolo11s-p2-dysample-neck.yaml"
REFERENCE_E1_YAML = core.ROOT / "yolo11s-p2-add.yaml"
DYSAMPLE_SOURCE = core.SOURCE / "ultralytics/nn/modules/dysample.py"
SMOKE_GATE = core.ROOT / "comparison_reports/e5a_dysample_smoke_passed.json"

_E1_TRANSFER_KEY = core.transfer_key
_E1_PLAN_TRANSFER = core.plan_transfer
_E1_LOAD_PRETRAINED = core.load_p2_pretrained
_E1_ASSERT_MODEL = core.assert_p2_model
_CORE_TRAIN = core.train


def transfer_key(source_key):
    """Original layers 0..22 keep indices; Detect 23 moves to E1 layer 26."""
    return _E1_TRANSFER_KEY(source_key)


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
            skipped.append(
                dict(source=key, target=target_key, reason="80 COCO classes -> 10 VisDrone classes")
            )
            continue
        if target_key in mapped:
            raise ValueError(f"Duplicate target mapping: {target_key}")
        mapped[target_key] = tensor
        mapping.append(dict(source=key, target=target_key, shape=list(tensor.shape)))

    missing = sorted(set(target_state) - set(mapped))
    for key in missing:
        allowed = (
            key.startswith("model.11.")
            or key.startswith("model.14.")
            or key.startswith("model.25.")
            or key.startswith("model.26.cv2.0.")
            or key.startswith("model.26.cv3.0.")
            or core.class_output_key(key, 26, ("1", "2", "3"))
        )
        if not allowed:
            raise ValueError(f"Unexpected uninitialized tensor: {key}")
    if len(skipped) != 6:
        raise ValueError(f"Expected six COCO-to-VisDrone class-output skips, got {len(skipped)}")
    return mapped, dict(
        rule="identity layers 0..22; Detect 23->26; old branches 0..2->new 1..3",
        source="original 80-class yolo11s.pt; never an experiment best.pt",
        loaded_tensors=len(mapped),
        skipped_class_outputs=skipped,
        new_or_reinitialized_target_tensors=missing,
        tensor_mapping=mapping,
    )


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
        if index in (11, 14):
            if original.model[index].__class__.__name__ != "Upsample":
                raise ValueError(f"Original layer {index} is not nearest Upsample")
            if target.model[index].__class__.__name__ != "DySample":
                raise ValueError(f"Target layer {index} is not DySample")
        elif type(original.model[index]) is not type(target.model[index]):
            raise ValueError(f"Unexpected architecture change at original layer {index}")
        if original.model[index].f != target.model[index].f:
            raise ValueError(f"Layer source changed at {index}")

    mapping, audit = plan_transfer(original.state_dict(), target.state_dict())
    result = target.load_state_dict(mapping, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected loaded keys: {result.unexpected_keys}")

    # DySample is constructed before the P2 branch. Recopy E1's deterministic
    # P2 C3k2 and complete Detect state to remove the extra-RNG confound.
    reference, fair_cpu_rng, fair_cuda_rng = _load_frozen_e1_reference()
    reference_state = reference.state_dict()
    fair_state = {
        key: tensor
        for key, tensor in reference_state.items()
        if key.startswith(("model.25.", "model.26."))
    }
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
            "Only DySample tensors are new. Shared tensors come from yolo11s.pt; E1 P2-C3k2 "
            "and complete Detect tensors are copied from the deterministic E1 seed-1 reference."
        ),
    )
    del mapping, fair_state, reference_state, reference, original
    return audit


def assert_p2_model(model):
    if len(model.model) != 27 or model.yaml.get("scale") != "s":
        raise ValueError("Expected the 27-layer s-scale E5a architecture")
    if model.model[-1].nc != 10 or model.model[-1].f != [25, 16, 19, 22]:
        raise ValueError("Expected frozen E1 P2/P3/P4/P5 Detect inputs")
    if list(model.stride.cpu().tolist()) != [4.0, 8.0, 16.0, 32.0]:
        raise ValueError(f"Wrong strides: {model.stride}")
    for index, channels in ((11, 512), (14, 256)):
        layer = model.model[index]
        if layer.__class__.__name__ != "DySample":
            raise ValueError(f"Layer {index} must be registered DySample")
        if (layer.channels, layer.scale, layer.style, layer.groups, layer.dyscope) != (
            channels, 2, "lp", 4, False
        ):
            raise ValueError(f"Layer {index} differs from standard E5a DySample configuration")
    if model.model[23].__class__.__name__ != "Upsample":
        raise ValueError("E1 P3->P2 nearest upsampling must remain unchanged")
    if model.model[10].__class__.__name__ != "C2PSA":
        raise ValueError("Original C2PSA must remain unchanged")


def validate_p2_config(cfg, original_cfg):
    expected_head = copy.deepcopy(original_cfg["head"][:12])
    for relative_index in (0, 3):
        expected_head[relative_index] = [-1, 1, "DySample", [2, "lp", 4, False]]
    expected_head.extend([
        [16, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 2], 1, "Concat", [1]],
        [-1, 2, "C3k2", [256, False]],
        [[25, 16, 19, 22], 1, "Detect", ["nc"]],
    ])
    checks = [
        ("backbone", core.canonical_layers(cfg.get("backbone", [])), core.canonical_layers(original_cfg["backbone"])),
        ("head", core.canonical_layers(cfg.get("head", [])), core.canonical_layers(expected_head)),
        ("scale", cfg.get("scale"), "s"),
        ("nc", cfg.get("nc"), 10),
        ("s_scaling", cfg.get("scales", {}).get("s"), original_cfg["scales"]["s"]),
    ]
    differences = [(name, actual, expected) for name, actual, expected in checks if actual != expected]
    if differences:
        details = "; ".join(f"{n}: expected={e!r}, actual={a!r}" for n, a, e in differences)
        raise ValueError("E5a YAML does not match the audited two-DySample-only design. " + details)


def p2_preflight(report, eval_only=False):
    from ultralytics.nn.modules import DySample
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils import yaml_load
    from ultralytics.utils.torch_utils import init_seeds

    report["p2_script_revision"] = SCRIPT_REVISION
    print(f"E5a DySample script revision: {SCRIPT_REVISION}", flush=True)
    for path in (MODEL_YAML, REFERENCE_E1_YAML, DYSAMPLE_SOURCE):
        if not path.is_file():
            raise FileNotFoundError(path)
    if Path(core.ultralytics.__file__).resolve().parent != (core.SOURCE / "ultralytics").resolve():
        raise RuntimeError("Wrong Ultralytics import location; activate editable yolo11 environment")
    if not core.PRETRAINED.is_file() or core.sha256(core.PRETRAINED) != core.PRETRAINED_SHA256:
        raise RuntimeError("Original yolo11s.pt missing/changed")

    cfg = yaml_model_load(str(MODEL_YAML))
    original_cfg = yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml")
    validate_p2_config(cfg, original_cfg)
    code_files = [
        "nn/tasks.py", "nn/modules/dysample.py", "nn/modules/head.py", "nn/modules/conv.py",
        "engine/model.py", "engine/trainer.py", "models/yolo/detect/train.py",
        "utils/tal.py", "utils/loss.py",
    ]
    hashes = {
        name: hashlib.sha256((core.SOURCE / "ultralytics" / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for name in code_files
    }
    report["p2_architecture"] = dict(
        yaml=str(MODEL_YAML), yaml_sha256=core.sha256(MODEL_YAML),
        original_weights_sha256=core.sha256(core.PRETRAINED), strides=[4, 8, 16, 32],
        variant=(
            "E5a existing-method control: standard DySample replaces only original neck layers 11 and 14; "
            "P3->P2 nearest upsampling and all E1 P2 structure remain unchanged"
        ),
        dysample=dict(layers=[11, 14], scale=2, style="lp", groups=4, dyscope=False,
                      reference="Liu et al., ICCV 2023; github.com/tiny-smart/dysample"),
        novelty_claim="none; E5a is a required existing-method mechanism control",
        original_C2PSA_unchanged=True,
        profile_note="THOP may not count grid_sample interpolation arithmetic; always report measured latency too.",
        determinism_note=(
            "PyTorch 2.5.1 CUDA grid_sample backward has no deterministic implementation. "
            "Full TAL loss/gradient equivalence is therefore checked on CPU; smoke2 verifies real CUDA training. "
            "Accuracy claims require multi-seed confirmation if E5a is retained."
        ),
        source_code_sha256=hashes,
        fairness="Frozen E1 P2-C3k2 and complete Detect initialization copied tensor-for-tensor.",
    )
    out = Path(report["report_dir"])
    for path in (MODEL_YAML, DYSAMPLE_SOURCE, REFERENCE_E1_YAML):
        shutil.copy2(path, out / path.name)

    init_seeds(core.SEED, deterministic=True)
    model = DetectionModel(cfg, nc=10, verbose=False)
    assert_p2_model(model)
    report["preflight_transfer"] = load_p2_pretrained(model)

    # Direct module tests: shape, gradients, finite values and zero-offset
    # agreement with align_corners=False bilinear interpolation.
    test_rows = []
    for index, channels, height, width in ((11, 512, 8, 12), (14, 256, 16, 24)):
        layer = model.model[index]
        x = core.torch.randn(2, channels, height, width, requires_grad=True)
        y = layer(x)
        if y.shape != (2, channels, 2 * height, 2 * width) or not core.torch.isfinite(y).all():
            raise RuntimeError(f"DySample layer {index} forward failed")
        y.float().square().mean().backward()
        if layer.offset.weight.grad is None or not core.torch.isfinite(layer.offset.weight.grad).all():
            raise RuntimeError(f"DySample layer {index} backward failed")
        with core.torch.no_grad():
            saved_weight, saved_bias = layer.offset.weight.clone(), layer.offset.bias.clone()
            layer.offset.weight.zero_(); layer.offset.bias.zero_()
            zero_offset = layer(x.detach())
            bilinear = core.torch.nn.functional.interpolate(
                x.detach(), scale_factor=2, mode="bilinear", align_corners=False
            )
            max_error = float((zero_offset - bilinear).abs().max())
            layer.offset.weight.copy_(saved_weight); layer.offset.bias.copy_(saved_bias)
        if max_error > 2e-5:
            raise RuntimeError(f"DySample zero-offset geometry differs from bilinear: {max_error}")
        test_rows.append(dict(layer=index, input_shape=list(x.shape), output_shape=list(y.shape),
                              zero_offset_vs_bilinear_max_abs_error=max_error))
    report["p2_architecture"]["module_unit_checks"] = test_rows

    model.eval()
    with core.torch.inference_mode():
        _, raw = model(core.torch.zeros(1, 3, 256, 256))
    sizes = [list(t.shape[-2:]) for t in raw]
    if sizes != [[64, 64], [32, 32], [16, 16], [8, 8]]:
        raise RuntimeError(f"Unexpected feature shapes: {sizes}")
    report["p2_architecture"]["CPU_forward_256_feature_shapes"] = sizes
    report["p2_architecture"]["preflight_parameters"] = sum(p.numel() for p in model.parameters())
    core.save_report(report)
    del model, raw, x, y, zero_offset, bilinear
    gc.collect()
    print("E5a structure/transfer/DySample/CPU forward checks passed. Strides: 4,8,16,32.", flush=True)

    if not core.MATCHING_HELPER.is_file():
        raise FileNotFoundError(core.MATCHING_HELPER)
    from p2_tal_chunked import assert_patch_restoration, implementation_info, verify_equivalence
    report["p2_architecture"]["matching_implementation"] = implementation_info()
    shutil.copy2(core.MATCHING_HELPER, out / core.MATCHING_HELPER.name)
    baseline_cfg = copy.deepcopy(original_cfg)
    baseline_cfg.update(scale="s", nc=10)
    # CUDA grid_sample backward is nondeterministic in pinned torch 2.5.1 and
    # produces false failures when two mathematically identical models are
    # backpropagated sequentially. CPU grid_sample is deterministic, so the
    # strict full-loss/gradient/update equivalence gate belongs on CPU. The
    # following smoke2 gate separately exercises real CUDA batch8/1024 training.
    device = "cpu"
    print("Checking strict TAL loss/gradient equivalence on CPU (deterministic grid_sample).", flush=True)
    report["status"] = "checking_matching_equivalence"
    core.save_report(report)
    try:
        report["matching_equivalence"] = verify_equivalence(
            copy.deepcopy(cfg), device=device, baseline_cfg=baseline_cfg
        )
        assert_patch_restoration()
        core.write_json(out / "tal_equivalence.json", report["matching_equivalence"])
    except BaseException:
        core.write_json(out / "tal_equivalence.json", dict(
            status="failed", device=device, error=traceback.format_exc()))
        raise
    finally:
        core.clear_gpu()
    core.save_report(report)
    print("TAL equivalence checks passed. This is not yet a 1024px memory test.", flush=True)


def train(report, smoke=False):
    weights = _CORE_TRAIN(report, smoke=smoke)
    run_dir = Path(report["run_dir"])
    shutil.copy2(DYSAMPLE_SOURCE, run_dir / DYSAMPLE_SOURCE.name)
    shutil.copy2(REFERENCE_E1_YAML, run_dir / "reference_yolo11s-p2-add.yaml")
    return weights


def install_overrides():
    core.EXPERIMENT = EXPERIMENT
    core.SCRIPT_REVISION = SCRIPT_REVISION
    core.MODEL_YAML = MODEL_YAML
    core.SMOKE_GATE = SMOKE_GATE
    core.__file__ = str(Path(__file__).resolve())
    core.transfer_key = transfer_key
    core.plan_transfer = plan_transfer
    core.load_p2_pretrained = load_p2_pretrained
    core.assert_p2_model = assert_p2_model
    core.validate_p2_config = validate_p2_config
    core.p2_preflight = p2_preflight
    core.train = train


if __name__ == "__main__":
    install_overrides()
    core.main()
