#!/usr/bin/env python3
"""E34a formal seed2: cross-stage persistent P2, paired with existing E1 seed2.

Use --check-only, then --smoke2, then no mode flag for the unchanged 200-epoch
schedule. Smoke is a safety test and never serves as an accuracy decision.
No automatic shutdown is permitted.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
import yolo11s_p2_img1024_seed1 as core


HERE = Path(__file__).resolve().parent
EXPERIMENT = "e34a_persistent_p2_e1_img1024_seed2"
REVISION = "e34a_persistent_p2_stage_feedback_seed2_v1"
MODEL_YAML = HERE / "yolo11s-p2-persistent-e34a.yaml"
E1_YAML = HERE / "yolo11s-p2-add.yaml"
MODULE_SOURCE = core.SOURCE / "ultralytics/nn/modules/persistent_p2.py"
E1_YAML_NORMALIZED_SHA256 = "98df605986f71295216f2abc549f5f0e381a7a39a6cc81b0419ab118cb29135b"
E1_TO_E34 = {
    **{i: i for i in range(5)},
    5: 7, 6: 8, 7: 10, 8: 11, 9: 12, 10: 13,
    11: 14, 12: 15, 13: 16, 14: 17, 15: 18, 16: 19,
    17: 20, 18: 21, 19: 22, 20: 23, 21: 24, 22: 25,
    23: 26, 24: 27, 25: 28, 26: 29,
}
NEW_LAYERS = (5, 6, 9)
_E1_LOAD = core.load_p2_pretrained
_E1_ASSERT = core.assert_p2_model
_E1_VERIFY_SMOKE = core.verify_smoke_gate


def e1_yaml_digest():
    return hashlib.sha256(E1_YAML.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def reference_e1():
    """Recreate the paired E1 initialization, not its trained best checkpoint."""
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils.torch_utils import init_seeds

    if e1_yaml_digest() != E1_YAML_NORMALIZED_SHA256:
        raise RuntimeError("Frozen E1 YAML differs from the paired-seed2 reference")
    init_seeds(2, deterministic=True)
    cfg = yaml_model_load(str(E1_YAML))
    cfg["scale"] = "s"
    reference = DetectionModel(cfg, nc=10, verbose=False)
    _E1_ASSERT(reference)
    original_audit = _E1_LOAD(reference)
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return reference, original_audit, cpu_rng, cuda_rng


def load_p2_pretrained(target):
    """Every original E1 tensor is copied exactly; only layers 5/6/9 are new."""
    reference, original_audit, cpu_rng, cuda_rng = reference_e1()
    target_state = target.state_dict()
    mapped = {}
    for key, value in reference.state_dict().items():
        parts = key.split(".", 2)
        new_key = key if len(parts) != 3 or parts[0] != "model" else f"model.{E1_TO_E34[int(parts[1])]}.{parts[2]}"
        if new_key not in target_state or tuple(target_state[new_key].shape) != tuple(value.shape):
            raise RuntimeError(f"E1->E34a transfer mismatch: {key} -> {new_key}")
        if new_key in mapped:
            raise RuntimeError(f"Duplicate transfer destination: {new_key}")
        mapped[new_key] = value
    missing = sorted(set(target_state) - set(mapped))
    expected_prefixes = tuple(f"model.{i}." for i in NEW_LAYERS)
    if not missing or any(not key.startswith(expected_prefixes) for key in missing):
        raise RuntimeError(f"Missing tensors are not exclusively new E34a modules: {missing}")
    result = target.load_state_dict(mapped, strict=False)
    expected_reported = sorted(key for key in missing if not key.endswith(".num_batches_tracked"))
    if sorted(result.missing_keys) != expected_reported or result.unexpected_keys:
        raise RuntimeError(f"Incomplete E1 transfer: {result}")
    if any(not torch.equal(target.state_dict()[key].cpu(), value.cpu()) for key, value in mapped.items()):
        raise RuntimeError("One or more transferred tensors changed")
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    del reference
    return dict(source="deterministic E1-seed2 initialization from original yolo11s.pt",
                e1_transfer=original_audit, layer_map=E1_TO_E34,
                loaded_tensor_count=len(mapped), new_tensor_count=len(missing),
                exact_common_tensor_equality=True,
                initial_prediction_equality_with_E1=False)


def assert_p2_model(model):
    if len(model.model) != 30 or model.yaml.get("scale") != "s":
        raise RuntimeError("E34a requires exactly 30 layers at s scale")
    expected = {
        5: ("PersistentP2Stage1", [2, 4]),
        6: ("PersistentP2Feedback", [4, 5]),
        9: ("PersistentP2Stage2", [5, 8]),
        29: ("Detect", [28, 19, 22, 25]),
    }
    for index, (name, sources) in expected.items():
        layer = model.model[index]
        if layer.__class__.__name__ != name or layer.f != sources:
            raise RuntimeError(f"E34a layer {index} topology changed: {type(layer).__name__}, {layer.f}")
    if model.model[13].__class__.__name__ != "C2PSA" or model.model[29].nc != 10:
        raise RuntimeError("Original deep block / four-scale 10-class Detect changed")
    if list(model.stride.detach().cpu().tolist()) != [4., 8., 16., 32.]:
        raise RuntimeError(f"Wrong Detect strides: {model.stride}")
    for index in NEW_LAYERS:
        layer = model.model[index]
        actual_hidden = (layer.reduce[0] if index == 6 else layer.local.reduce[0]).out_channels
        if layer.gain != 0.1 or actual_hidden != 32:
            raise RuntimeError(f"E34a layer {index} hidden width/gain drift")


def validate_p2_config(cfg, original_cfg):
    from ultralytics.utils import yaml_load

    e1 = yaml_load(E1_YAML)
    original_backbone = copy.deepcopy(e1["backbone"])
    expected_backbone = (
        original_backbone[:5]
        + [
            [[2, 4], 1, "PersistentP2Stage1", [32, 0.1]],
            [[4, 5], 1, "PersistentP2Feedback", [32, 0.1]],
            [6, 1, "Conv", [512, 3, 2]],
            [-1, 2, "C3k2", [512, True]],
            [[5, 8], 1, "PersistentP2Stage2", [32, 0.1]],
            [8, 1, "Conv", [1024, 3, 2]],
        ]
        + original_backbone[8:]
    )
    expected_head = [
        [13, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 8], 1, "Concat", [1]],
        [-1, 2, "C3k2", [512, False]],
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 4], 1, "Concat", [1]],
        [-1, 2, "C3k2", [256, False]],
        [-1, 1, "Conv", [256, 3, 2]],
        [[-1, 16], 1, "Concat", [1]],
        [-1, 2, "C3k2", [512, False]],
        [-1, 1, "Conv", [512, 3, 2]],
        [[-1, 13], 1, "Concat", [1]],
        [-1, 2, "C3k2", [1024, True]],
        [19, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 9], 1, "Concat", [1]],
        [-1, 2, "C3k2", [256, False]],
        [[28, 19, 22, 25], 1, "Detect", ["nc"]],
    ]
    drift = {
        "backbone": (cfg.get("backbone"), expected_backbone),
        "head": (core.canonical_layers(cfg.get("head")), core.canonical_layers(expected_head)),
        "scale": (cfg.get("scale"), "s"),
        "nc": (cfg.get("nc"), 10),
        "s_scaling": (cfg.get("scales", {}).get("s"), original_cfg["scales"]["s"]),
    }
    drift = {key: value for key, value in drift.items() if value[0] != value[1]}
    if drift:
        raise RuntimeError(f"E34a architecture drift: {drift}")


def _active_path_and_gradient_audit(model):
    """A zero-initialized or disconnected new path is a hard preflight failure."""
    from ultralytics.utils.loss import v8DetectionLoss

    model.cpu().eval()
    torch.manual_seed(3402)
    image = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        _, original_raw = model(image)

        def disable_feedback(_module, inputs, _output):
            return inputs[0][0]

        with model.model[6].register_forward_hook(disable_feedback):
            _, no_feedback_raw = model(image)
        feedback_max = max(float((a - b).abs().max()) for a, b in zip(original_raw, no_feedback_raw))

        def disable_stage2(_module, inputs, _output):
            return inputs[0][0]

        with model.model[9].register_forward_hook(disable_stage2):
            _, no_stage2_raw = model(image)
        stage2_max = float((original_raw[0] - no_stage2_raw[0]).abs().max())
    if not (feedback_max > 0.0 and stage2_max > 0.0):
        raise RuntimeError(f"A required E34a edge is inactive at initialization: {feedback_max}, {stage2_max}")

    model.train()
    model.zero_grad(set_to_none=True)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    torch.manual_seed(3403)
    batch = dict(
        img=torch.rand(2, 3, 128, 128),
        batch_idx=torch.tensor([0., 0., 1., 1.]),
        cls=torch.tensor([[0.], [3.], [5.], [9.]]),
        bboxes=torch.tensor([[.25, .30, .08, .09], [.68, .62, .24, .20],
                             [.46, .52, .12, .10], [.70, .28, .34, .30]]),
    )
    raw = model(batch["img"])
    loss, items = v8DetectionLoss(model)(raw, batch)
    if not torch.isfinite(loss) or not torch.isfinite(items).all():
        raise RuntimeError("E34a loss is not finite")
    loss.backward()
    probes = {
        "stage1_local": model.model[5].local.reduce[0].weight,
        "stage1_semantic": model.model[5].semantic.project[0].weight,
        "early_feedback": model.model[6].project[0].weight,
        "stage2_local": model.model[9].local.reduce[0].weight,
        "stage2_semantic": model.model[9].semantic.project[0].weight,
        # Classification BCE supplies gradients even if random TAL assignment
        # happens to assign no positives to the P2 regression tower here.
        "P2_detect_classification": model.model[29].cv3[0][-1].weight,
    }
    gradients = {}
    for name, parameter in probes.items():
        grad = parameter.grad
        if grad is None or not torch.isfinite(grad).all() or float(grad.norm()) <= 0:
            raise RuntimeError(f"E34a real detection gradient missing: {name}")
        gradients[name] = float(grad.norm())
    model.zero_grad(set_to_none=True)
    return dict(feedback_raw_max_abs=feedback_max, stage2_p2_raw_max_abs=stage2_max,
                total_loss=float(loss.detach()), gradient_norms=gradients)


def p2_preflight(report, eval_only=False):
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils import yaml_load
    from ultralytics.utils.torch_utils import get_flops, init_seeds

    if not MODEL_YAML.is_file() or not E1_YAML.is_file() or not MODULE_SOURCE.is_file():
        raise FileNotFoundError("E34a YAML/module/E1 reference missing; run installer")
    if e1_yaml_digest() != E1_YAML_NORMALIZED_SHA256:
        raise RuntimeError("E1 reference YAML changed")
    if not core.PRETRAINED.is_file() or core.sha256(core.PRETRAINED) != core.PRETRAINED_SHA256:
        raise RuntimeError("Original yolo11s.pt missing or changed")
    if Path(core.ultralytics.__file__).resolve().parent != (core.SOURCE / "ultralytics").resolve():
        raise RuntimeError("Wrong Ultralytics source; activate the yolo11 environment")
    cfg = yaml_model_load(str(MODEL_YAML))
    original = yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml")
    validate_p2_config(cfg, original)
    source_hashes = {name: core.sha256(core.SOURCE / "ultralytics" / name) for name in (
        "nn/tasks.py", "nn/modules/__init__.py", "nn/modules/head.py", "nn/modules/conv.py",
        "utils/loss.py", "utils/tal.py", "engine/trainer.py")}
    report["p2_script_revision"] = REVISION
    report["p2_architecture"] = dict(
        yaml=str(MODEL_YAML), yaml_sha256=core.sha256(MODEL_YAML),
        e1_yaml_normalized_sha256=e1_yaml_digest(), module_sha256=core.sha256(MODULE_SOURCE),
        original_weights_sha256=core.sha256(core.PRETRAINED), source_code_sha256=source_hashes,
        variant="Mandatory P2 state across B3/B4; one early H2->deep-backbone feedback; E1 neck/Detect/loss/TAL",
        hidden_width=32, fixed_gain=0.1, original_loss_tal_detect=True,
        initialization_note="All compatible E1-initialized tensors copied exactly; new paths active, predictions not E1-identical",
    )
    out = Path(report["report_dir"])
    for source in (MODEL_YAML, E1_YAML, MODULE_SOURCE):
        shutil.copy2(source, out / source.name)
    init_seeds(core.SEED, deterministic=True)
    model = DetectionModel(cfg, nc=10, verbose=False)
    assert_p2_model(model)
    report["preflight_transfer"] = load_p2_pretrained(model)
    model.eval()
    with torch.no_grad():
        _, raw = model(torch.zeros(1, 3, 256, 256))
    sizes = [list(x.shape[-2:]) for x in raw]
    if sizes != [[64, 64], [32, 32], [16, 16], [8, 8]]:
        raise RuntimeError(f"E34a Detect feature shapes changed: {sizes}")
    report["p2_architecture"]["CPU_forward_256_feature_shapes"] = sizes
    parameters = sum(x.numel() for x in model.parameters())
    model.eval()
    flops = float(get_flops(model, imgsz=1024))
    if not 0 < flops <= 100.0 or parameters > 11_000_000:
        raise RuntimeError(f"E34a resource budget exceeded/unavailable: params={parameters}, GFLOPs={flops}")
    report["p2_architecture"].update(preflight_parameters=parameters, preflight_GFLOPs_1024=flops)
    report["active_path_audit"] = _active_path_and_gradient_audit(model)
    del model, raw
    gc.collect()
    if not core.MATCHING_HELPER.is_file():
        raise FileNotFoundError(core.MATCHING_HELPER)
    from p2_tal_chunked import implementation_info, verify_equivalence, assert_patch_restoration
    report["p2_architecture"]["matching_implementation"] = implementation_info()
    shutil.copy2(core.MATCHING_HELPER, out / core.MATCHING_HELPER.name)
    baseline_cfg = copy.deepcopy(original)
    baseline_cfg.update(scale="s", nc=10)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    report["matching_equivalence"] = verify_equivalence(copy.deepcopy(cfg), device=device, baseline_cfg=baseline_cfg)
    assert_patch_restoration()
    core.write_json(out / "tal_equivalence.json", report["matching_equivalence"])
    core.save_report(report)
    core.clear_gpu()
    print("E34a topology, exact shared transfer, active path, gradients, resource and TAL checks passed.", flush=True)


def verify_smoke_gate(report):
    _E1_VERIFY_SMOKE(report)
    gate = report["smoke_gate"]
    smoke = json.loads((Path(gate["report_dir"]) / "metrics.json").read_text(encoding="utf-8"))
    training = smoke.get("training") or {}
    peak = training.get("peak_memory") or {}
    reserved = peak.get("reserved_GiB")
    if (smoke.get("status") != "completed" or training.get("completed_epochs") != 2
            or reserved is None or not 0 < float(reserved) <= 29.0):
        raise RuntimeError(f"E34a CUDA smoke did not leave >=3 GiB reserved-memory headroom: {peak}")
    report["smoke_memory_gate"] = dict(reserved_GiB=float(reserved), limit_GiB=29.0)


def install_e34a():
    core.SEED = 2
    core.EXPERIMENT = EXPERIMENT
    core.SCRIPT_REVISION = REVISION
    core.MODEL_YAML = MODEL_YAML
    core.SMOKE_GATE = core.ROOT / "comparison_reports/e34a_seed2_smoke_passed.json"
    core.AUTO_SHUTDOWN = False
    core.SHUTDOWN_ON_FAILURE = False
    core.__file__ = str(Path(__file__).resolve())
    core.load_p2_pretrained = load_p2_pretrained
    core.assert_p2_model = assert_p2_model
    core.validate_p2_config = validate_p2_config
    core.p2_preflight = p2_preflight
    core.verify_smoke_gate = verify_smoke_gate
    required = dict(epochs=200, imgsz=1024, batch=8, workers=8, seed=2,
                    deterministic=True, optimizer="SGD", lr0=0.01, lrf=0.01,
                    momentum=0.937, weight_decay=0.0005, amp=True, cos_lr=True,
                    close_mosaic=10, nbs=64, cache=False, rect=False, multi_scale=False)
    params = core.training_params()
    drift = {key: (params.get(key), value) for key, value in required.items() if params.get(key) != value}
    if drift:
        raise RuntimeError(f"E1 seed2 training protocol drift: {drift}")


if __name__ == "__main__":
    install_e34a()
    core.main()
