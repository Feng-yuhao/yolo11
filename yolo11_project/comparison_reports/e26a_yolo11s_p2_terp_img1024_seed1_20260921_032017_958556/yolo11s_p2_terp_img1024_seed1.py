#!/usr/bin/env python3
"""E26a TERPHead: E1 plus opposite-direction task-exclusive residual routing.

Formal protocol is exactly E1: VisDrone, imgsz=1024, batch=8, seed=1,
200 epochs and the same image-wise TAL2 implementation.  Automatic shutdown
is disabled on every path.
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
    raise ImportError("E26a requires p2_tal_chunked_vgpu32.py beside the E1 runner") from exc
sys.modules["p2_tal_chunked"] = fast_tal

try:
    import yolo11s_p2_img1024_seed1 as core
except ImportError as exc:
    raise ImportError("E26a requires the frozen E1 runner in yolo11_project") from exc


EXPERIMENT = "e26a_yolo11s_p2_terp_img1024_seed1"
REVISION = "e26a_terp_e1_strict_identity_task_exclusive_reverse_pyramid_seed1_200e_v1"
MODEL_YAML = HERE / "yolo11s-p2-terp.yaml"
REFERENCE_E1_YAML = HERE / "e26a_reference_yolo11s-p2-add.yaml"
TERP_SOURCE = core.SOURCE / "ultralytics/nn/modules/terp.py"
MATCHING_HELPER = Path(fast_tal.__file__).resolve()
SMOKE_GATE = core.ROOT / "comparison_reports/e26a_terp_smoke_passed.json"
MAX_GAIN = 0.2
MAX_PARAMETERS = 10_500_000
MAX_GFLOPS_1024 = 96.0

_E1_LOAD = core.load_p2_pretrained
_E1_ASSERT = core.assert_p2_model
_E1_TRAIN = core.train


def _reference():
    """Rebuild the deterministic E1 training initialization from original yolo11s.pt."""
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
    """Copy every E1 tensor exactly and leave only TERP route tensors new."""
    reference, original_audit, cpu_rng, cuda_rng = _reference()
    target_state = target.state_dict()
    mapped = {}
    for key, value in reference.state_dict().items():
        if key not in target_state or tuple(target_state[key].shape) != tuple(value.shape):
            raise ValueError(f"E1->E26a common tensor mismatch: {key}")
        mapped[key] = value

    missing = sorted(set(target_state) - set(mapped))
    allowed_prefixes = (
        "model.26.cls_aligners.",
        "model.26.reg_aligners.",
        "model.26.cls_routes.",
        "model.26.reg_routes.",
    )
    unexpected = [key for key in missing if not key.startswith(allowed_prefixes)]
    if unexpected or not missing:
        raise ValueError(f"Unexpected E26a-only tensors: unexpected={unexpected}, all={missing}")
    result = target.load_state_dict(mapped, strict=False)
    # PyTorch intentionally omits BatchNorm num_batches_tracked buffers from
    # IncompatibleKeys.missing_keys for compatibility with older checkpoints.
    # They are nevertheless present in target_state and correctly remain at
    # the new TERP module's own initialization.  Compare the reported set
    # against the same compatibility-filtered expectation.
    expected_reported_missing = sorted(
        key for key in missing if not key.endswith(".num_batches_tracked")
    )
    if sorted(result.missing_keys) != expected_reported_missing or result.unexpected_keys:
        raise ValueError(
            "Incomplete strict E1 migration: "
            f"expected_reported_missing={expected_reported_missing}, actual={result}"
        )

    # Match the deterministic E1 loader's post-transfer RNG boundary.
    core.torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        core.torch.cuda.set_rng_state_all(cuda_rng)
    changed = [
        key for key, value in mapped.items()
        if not core.torch.equal(target.state_dict()[key].detach().cpu(), value.detach().cpu())
    ]
    if changed:
        raise RuntimeError(f"Transferred E1 tensors changed: {changed[:5]}")
    audit = dict(
        source="deterministic E1 initialized from original 80-class yolo11s.pt; never an experimental best.pt",
        source_audit=original_audit,
        loaded_tensor_count=len(mapped),
        new_tensor_count=len(missing),
        new_tensor_keys=missing,
        exact_common_tensor_equality=True,
        reference_e1_yaml_sha256=core.sha256(REFERENCE_E1_YAML),
        note="Only TERP alignment, gate and alpha tensors are new; E1 backbone/neck/P2/Detect towers are exact.",
    )
    del reference
    return audit


def assert_p2_model(model):
    if len(model.model) != 27 or model.yaml.get("scale") != "s":
        raise ValueError("E26a requires the 27-layer s-scale E1 graph")
    head = model.model[26]
    if head.__class__.__name__ != "TERPHead":
        raise ValueError(f"Expected TERPHead, found {head.__class__.__name__}")
    if head.f != [0, 2, 4, 6, 25, 16, 19, 22] or head.nc != 10 or head.nl != 4:
        raise ValueError(f"Wrong E26a head contract: f={head.f}, nc={head.nc}, nl={head.nl}")
    if tuple(head.source_channels) != (32, 128, 256, 256):
        raise ValueError(f"Wrong shallow route channels: {head.source_channels}")
    if tuple(head.detect_channels) != (128, 128, 256, 512):
        raise ValueError(f"Wrong Detect pyramid channels: {head.detect_channels}")
    if len(head.cls_routes) != 3 or len(head.reg_routes) != 4:
        raise ValueError("E26a must contain exactly three classification and four regression residual routes")
    if any(abs(route.max_gain - MAX_GAIN) > 1.0e-12 for route in list(head.cls_routes) + list(head.reg_routes)):
        raise ValueError("TERP bounded gain changed from 0.2")
    if list(model.stride.detach().cpu().tolist()) != [4.0, 8.0, 16.0, 32.0]:
        raise ValueError(f"E26a strides changed: {model.stride}")
    if model.model[10].__class__.__name__ != "C2PSA":
        raise ValueError("Original YOLO11 C2PSA changed")
    for index in (11, 14, 23):
        layer = model.model[index]
        if layer.__class__.__name__ != "Upsample" or layer.mode != "nearest" or layer.scale_factor != 2.0:
            raise ValueError(f"Frozen nearest upsample changed at layer {index}")


def validate_p2_config(cfg, original_cfg):
    del original_cfg
    from ultralytics.utils import yaml_load

    e1 = yaml_load(REFERENCE_E1_YAML)
    expected_head = copy.deepcopy(e1["head"])
    expected_head[-1] = [[0, 2, 4, 6, 25, 16, 19, 22], 1, "TERPHead", ["nc"]]
    checks = [
        ("backbone", core.canonical_layers(cfg.get("backbone", [])), core.canonical_layers(e1["backbone"])),
        ("head", core.canonical_layers(cfg.get("head", [])), core.canonical_layers(expected_head)),
        ("scale", cfg.get("scale"), "s"),
        ("nc", cfg.get("nc"), 10),
        ("s_scaling", cfg.get("scales", {}).get("s"), e1["scales"]["s"]),
    ]
    differences = [(name, actual, expected) for name, actual, expected in checks if actual != expected]
    if differences:
        raise ValueError("E26a differs from preregistered E1+TERP: " + "; ".join(
            f"{name}: expected={expected!r}, actual={actual!r}" for name, actual, expected in differences
        ))


def _synthetic_batch():
    return dict(
        img=core.torch.rand(2, 3, 128, 128),
        batch_idx=core.torch.tensor([0.0, 0.0, 1.0, 1.0]),
        cls=core.torch.tensor([[0.0], [3.0], [5.0], [9.0]]),
        bboxes=core.torch.tensor([
            [0.25, 0.30, 0.08, 0.09],
            [0.68, 0.62, 0.24, 0.20],
            [0.46, 0.52, 0.12, 0.10],
            [0.70, 0.28, 0.34, 0.30],
        ]),
    )


def _flatten(value):
    if core.torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        rows = []
        for item in value:
            rows.extend(_flatten(item))
        return rows
    if isinstance(value, dict):
        rows = []
        for key in sorted(value):
            rows.extend(_flatten(value[key]))
        return rows
    raise TypeError(type(value))


def _exact_prediction_and_gradient_audit(model):
    """At alpha=0, predictions, loss and every E1 parameter gradient must be exact."""
    from ultralytics.utils.loss import v8DetectionLoss

    core.torch.manual_seed(2601)
    reference, _, _, _ = _reference()
    candidate = copy.deepcopy(model).cpu()
    reference = reference.cpu()
    args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    reference.args = args
    candidate.args = args
    reference.train()
    candidate.train()
    batch = _synthetic_batch()
    reference.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    pred_reference = reference(batch["img"])
    pred_candidate = candidate(batch["img"])
    flat_reference, flat_candidate = _flatten(pred_reference), _flatten(pred_candidate)
    if len(flat_reference) != len(flat_candidate):
        raise RuntimeError("E1/E26a raw prediction tensor counts differ")
    prediction_max_abs = 0.0
    for left, right in zip(flat_reference, flat_candidate):
        if left.shape != right.shape:
            raise RuntimeError(f"E1/E26a raw shape mismatch: {left.shape} vs {right.shape}")
        prediction_max_abs = max(prediction_max_abs, float((left.detach() - right.detach()).abs().max()))
        if not core.torch.equal(left.detach(), right.detach()):
            raise RuntimeError(f"E26a alpha=0 is not exact E1; prediction max abs={prediction_max_abs}")

    ref_criterion = v8DetectionLoss(reference)
    candidate_criterion = v8DetectionLoss(candidate)
    ref_total, ref_items = ref_criterion(pred_reference, batch)
    candidate_total, candidate_items = candidate_criterion(pred_candidate, batch)
    if not core.torch.equal(ref_total.detach(), candidate_total.detach()):
        raise RuntimeError("E26a alpha=0 total loss differs from E1")
    if not core.torch.equal(ref_items.detach(), candidate_items.detach()):
        raise RuntimeError("E26a alpha=0 box/cls/DFL items differ from E1")
    ref_total.backward()
    candidate_total.backward()

    reference_grads = {name: parameter.grad for name, parameter in reference.named_parameters()}
    candidate_parameters = dict(candidate.named_parameters())
    gradient_max_abs = 0.0
    for name, left in reference_grads.items():
        right = candidate_parameters[name].grad
        if (left is None) != (right is None):
            raise RuntimeError(f"E1 common gradient presence differs at {name}")
        if left is not None:
            difference = float((left - right).abs().max())
            gradient_max_abs = max(gradient_max_abs, difference)
            if not core.torch.equal(left, right):
                raise RuntimeError(f"E1 common gradient differs at {name}: {difference}")

    head = candidate.model[26]
    real_alpha = {}
    for name, route in zip(head.route_names, list(head.cls_routes) + list(head.reg_routes)):
        grad = route.alpha.grad
        real_alpha[name] = None if grad is None else float(grad.detach())
        if grad is not None and not bool(core.torch.isfinite(grad)):
            raise RuntimeError(f"Non-finite real-loss alpha gradient at {name}")
    for group_name, group in (("cls_aligners", head.cls_aligners), ("reg_aligners", head.reg_aligners)):
        for name, parameter in group.named_parameters():
            if parameter.grad is not None and float(parameter.grad.abs().max()) != 0.0:
                raise RuntimeError(f"alpha=0 leaked real-loss gradient into {group_name}.{name}")
    for group_name, group in (("cls_routes", head.cls_routes), ("reg_routes", head.reg_routes)):
        for name, parameter in group.named_parameters():
            if name.endswith("alpha"):
                continue
            if parameter.grad is not None and float(parameter.grad.abs().max()) != 0.0:
                raise RuntimeError(f"alpha=0 leaked real-loss gradient into {group_name}.{name}")

    del reference, candidate
    gc.collect()
    return dict(
        prediction_max_abs_difference=prediction_max_abs,
        total_loss_exact=True,
        loss_items_exact=True,
        common_gradient_max_abs_difference=gradient_max_abs,
        real_detection_loss_alpha_gradients=real_alpha,
        branch_parameters_blocked_at_alpha_zero=True,
    )


def _nonzero_finite_gradient(parameter) -> bool:
    return (
        parameter.grad is not None
        and bool(core.torch.isfinite(parameter.grad).all())
        and float(parameter.grad.detach().abs().sum()) > 0.0
    )


def _staged_route_gradient_audit(model):
    """Deterministic two-stage connectivity check independent of TAL positive distribution."""
    core.torch.manual_seed(2602)
    head = copy.deepcopy(model.model[26]).cpu().train()
    source_shapes = ((32, 128), (128, 64), (256, 32), (256, 16))
    detect_shapes = ((128, 64), (128, 32), (256, 16), (512, 8))
    backbone = [core.torch.randn(2, channels, side, side) for channels, side in source_shapes]
    base = [core.torch.randn(2, channels, side, side) for channels, side in detect_shapes]

    def objective(reg_inputs, cls_inputs):
        value = core.torch.zeros((), dtype=reg_inputs[0].dtype)
        for index, tensor in enumerate(reg_inputs):
            value = value + (index + 1.0) * tensor.square().mean()
        for index, tensor in enumerate(cls_inputs):
            value = value + (index + 1.5) * tensor.square().mean()
        return value

    head.zero_grad(set_to_none=True)
    reg0, cls0 = head.prepare_task_inputs(backbone + base)
    for output, target in zip(reg0, base):
        if not core.torch.equal(output.detach(), target):
            raise RuntimeError("Regression route is not exact identity at alpha=0")
    for output, target in zip(cls0, base):
        if not core.torch.equal(output.detach(), target):
            raise RuntimeError("Classification route is not exact identity at alpha=0")
    objective(reg0, cls0).backward()

    routes = list(head.cls_routes) + list(head.reg_routes)
    step1_alpha = {}
    for name, route in zip(head.route_names, routes):
        if not _nonzero_finite_gradient(route.alpha):
            raise RuntimeError(f"Stage-1 alpha did not receive a finite nonzero gradient: {name}")
        step1_alpha[name] = float(route.alpha.grad.detach())
    branch_groups = (head.cls_aligners, head.reg_aligners)
    for group in branch_groups:
        for parameter in group.parameters():
            if parameter.grad is not None and float(parameter.grad.abs().max()) != 0.0:
                raise RuntimeError("Stage-1 alpha=0 did not block an alignment parameter")
    for route in routes:
        for parameter_name, parameter in route.named_parameters():
            if parameter_name == "alpha":
                continue
            if parameter.grad is not None and float(parameter.grad.abs().max()) != 0.0:
                raise RuntimeError(f"Stage-1 alpha=0 did not block gate parameter {parameter_name}")

    # Move only alpha away from zero by a bounded deterministic sign step.
    with core.torch.no_grad():
        for route in routes:
            route.alpha.add_(-0.05 * route.alpha.grad.sign())
    head.zero_grad(set_to_none=True)
    reg1, cls1 = head.prepare_task_inputs(backbone + base)
    objective(reg1, cls1).backward()

    # Require every logical path and every weight-bearing operation to unlock.
    # A particular BN/gate bias may legitimately sum to zero on a symmetric
    # synthetic tensor, so biases are required to be finite when present but
    # are not individually used as connectivity gates.
    failed = []
    for index, aligner in enumerate(head.cls_aligners):
        if not _nonzero_finite_gradient(aligner.project.conv.weight):
            failed.append(f"cls_aligners.{index}.project.conv.weight")
    for index, aligner in enumerate(head.reg_aligners):
        if not _nonzero_finite_gradient(aligner.down.weight):
            failed.append(f"reg_aligners.{index}.down.weight")
        if not _nonzero_finite_gradient(aligner.project.conv.weight):
            failed.append(f"reg_aligners.{index}.project.conv.weight")
    for index, route in enumerate(routes):
        for name, parameter in (
            ("alpha", route.alpha),
            ("gate_dw.weight", route.gate_dw.weight),
            ("gate_out.weight", route.gate_out.weight),
        ):
            if not _nonzero_finite_gradient(parameter):
                failed.append(f"route.{index}.{name}")
    for name, parameter in head.named_parameters():
        if parameter.grad is not None and not bool(core.torch.isfinite(parameter.grad).all()):
            failed.append(f"nonfinite:{name}")
    if failed:
        raise RuntimeError(f"Stage-2 route parameters did not unlock: {failed}")
    return dict(
        initial_outputs_exactly_equal_E1_bases=True,
        stage1_alpha_gradients=step1_alpha,
        stage1_alignment_and_gate_gradients_zero=True,
        alpha_sign_step=0.05,
        stage2_all_weight_bearing_route_operations_nonzero_finite=True,
        stage2_all_observed_gradients_finite=True,
    )


def _initialization_audit(head):
    expected = core.torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
    ).div(16.0)
    for index, aligner in enumerate(head.reg_aligners):
        kernels = aligner.down.weight.detach().cpu()
        if not core.torch.equal(kernels, expected.view(1, 1, 3, 3).repeat(kernels.shape[0], 1, 1, 1)):
            raise RuntimeError(f"Regression aligner {index} lost its exact binomial initialization")
        if not aligner.down.weight.requires_grad:
            raise RuntimeError(f"Regression aligner {index} depthwise kernel is frozen")
    routes = list(head.cls_routes) + list(head.reg_routes)
    if any(float(route.alpha.detach()) != 0.0 or float(route.beta.detach()) != 0.0 for route in routes):
        raise RuntimeError("TERP alpha/beta must start at exact zero")
    gate_max = max(float(route.gate_out.weight.detach().abs().max()) for route in routes)
    gate_nonzero = all(float(route.gate_out.weight.detach().abs().sum()) > 0.0 for route in routes)
    if not gate_nonzero or gate_max >= 0.02:
        raise RuntimeError(f"Gate output initialization is not tiny-nonzero: max={gate_max}")
    return dict(
        learnable_AA_kernel="exact 3x3 binomial /16 at initialization; requires_grad=True",
        all_alpha_exact_zero=True,
        all_beta_exact_zero=True,
        gate_output_weight_tiny_nonzero=True,
        gate_output_max_abs=gate_max,
        gate_bias_zero=True,
    )


def p2_preflight(report, eval_only=False):
    del eval_only
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils import yaml_load
    from ultralytics.utils.torch_utils import get_flops, init_seeds

    required = (MODEL_YAML, REFERENCE_E1_YAML, TERP_SOURCE, MATCHING_HELPER)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if Path(core.ultralytics.__file__).resolve().parent != (core.SOURCE / "ultralytics").resolve():
        raise RuntimeError("Wrong editable Ultralytics source; activate conda environment yolo11")
    if core.sha256(core.PRETRAINED) != core.PRETRAINED_SHA256:
        raise RuntimeError("Original yolo11s.pt is missing or changed")

    cfg = yaml_model_load(str(MODEL_YAML))
    validate_p2_config(cfg, yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml"))
    out = Path(report["report_dir"])
    for path in required:
        shutil.copy2(path, out / path.name)

    init_seeds(core.SEED + 1, deterministic=True)
    model = DetectionModel(cfg, nc=10, verbose=False)
    assert_p2_model(model)
    report["preflight_transfer"] = load_p2_pretrained(model)
    head = model.model[26]
    initialization = _initialization_audit(head)
    identity = _exact_prediction_and_gradient_audit(model)
    staged = _staged_route_gradient_audit(model)

    model.eval()
    with core.torch.inference_mode():
        prediction, raw = model(core.torch.zeros(1, 3, 256, 256))
    grids = [list(tensor.shape[-2:]) for tensor in raw]
    channels = [int(tensor.shape[1]) for tensor in raw]
    if grids != [[64, 64], [32, 32], [16, 16], [8, 8]] or channels != [74, 74, 74, 74]:
        raise RuntimeError(f"E26a Detect output contract changed: grids={grids}, channels={channels}")
    if prediction.shape[1] != 14 or not core.torch.isfinite(prediction).all():
        raise RuntimeError(f"E26a inference contract is invalid: {tuple(prediction.shape)}")

    parameters = sum(parameter.numel() for parameter in model.parameters())
    gflops = float(get_flops(model, imgsz=core.IMGSZ))
    if parameters > MAX_PARAMETERS:
        raise RuntimeError(f"E26a parameter gate failed: {parameters} > {MAX_PARAMETERS}")
    if not math.isfinite(gflops) or gflops <= 0.0:
        raise RuntimeError("E26a GFLOPs could not be measured; install/repair thop before training")
    if gflops > MAX_GFLOPS_1024:
        raise RuntimeError(f"E26a compute gate failed: {gflops:.4f} > {MAX_GFLOPS_1024}")

    report["p2_script_revision"] = REVISION
    report["p2_architecture"] = dict(
        yaml_sha256=core.sha256(MODEL_YAML),
        reference_e1_yaml_sha256=core.sha256(REFERENCE_E1_YAML),
        original_weights_sha256=core.sha256(core.PRETRAINED),
        terp_source_sha256=core.sha256(TERP_SOURCE),
        tasks_sha256=core.sha256(core.SOURCE / "ultralytics/nn/tasks.py"),
        variant="E1 plus non-recursive task-exclusive reverse-pyramid residual routing inside Detect input preparation",
        parent="E1 only; no experimental checkpoint",
        shared_base="P2/P3/P4/P5 remain shared by cv2 and cv3",
        classification_sources="raw adjacent P3->P2, P4->P3, P5->P4; no recursive semantic cascade",
        regression_sources="raw B1->P2, B2->P3, B3->P4, B4->P5",
        regression_alignment="trainable raw depthwise stride-2 conv with exact binomial initialization, then linear 1x1+BN",
        gate="parameter-free functional GroupNorm descriptor; DW3x3+SiLU+tiny-nonzero 1x1+sigmoid",
        residual_gain="0.2*tanh(alpha), alpha initialized exactly zero per route",
        claim_boundary="Only added cross-scale residual sources are task exclusive; base pyramid features are still shared",
        unchanged="E1 neck, cv2/cv3 towers, reg_max=16, DFL/decode, TAL2, CIoU/DFL/BCE, NMS and training protocol",
        no_modules="No DSSG, TSDR, FusionReady, RSG, SDQC, slicing, distillation, DySample, PGALP or loss change",
        detect_strides=[4, 8, 16, 32],
        raw_feature_grids=grids,
        raw_output_channels=channels,
        inference_channels=int(prediction.shape[1]),
        parameters=parameters,
        GFLOPs_1024=gflops,
        parameter_gate=MAX_PARAMETERS,
        GFLOPs_gate_1024=MAX_GFLOPS_1024,
        initialization_audit=initialization,
        strict_E1_identity_audit=identity,
        staged_gradient_audit=staged,
        tal2=fast_tal.implementation_info(),
    )
    core.save_report(report)
    del model, raw, prediction
    gc.collect()
    print(
        f"E26a topology, exact E1 identity, staged gradients and profile passed: "
        f"params={parameters}, GFLOPs@1024={gflops:.3f}.",
        flush=True,
    )

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
    for path in (TERP_SOURCE, REFERENCE_E1_YAML):
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
