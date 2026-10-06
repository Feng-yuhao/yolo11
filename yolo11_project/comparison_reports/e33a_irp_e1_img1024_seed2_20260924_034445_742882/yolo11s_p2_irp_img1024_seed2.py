#!/usr/bin/env python3
"""E33a IRP: strict E1-seed2 protocol, with one iterative P2/P3/PAN pass.

No E10/E18 weights, custom loss, training schedule, or automatic shutdown.
Run --check-only, then --smoke2, then formal training with no mode flag.
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
EXPERIMENT = "e33a_irp_e1_img1024_seed2"
REVISION = "e33a_irp_e1_seed2_exact_transfer_v1"
MODEL_YAML = HERE / "yolo11s-p2-irp.yaml"
E1_YAML = HERE / "yolo11s-p2-add.yaml"
E1_YAML_NORMALIZED_SHA256 = "98df605986f71295216f2abc549f5f0e381a7a39a6cc81b0419ab118cb29135b"
MODULE_SOURCE = core.SOURCE / "ultralytics/nn/modules/irp.py"
MAP = {**{i: i for i in range(17)}, 17: 21, 18: 22, 19: 23,
       20: 24, 21: 25, 22: 26, 23: 17, 24: 18, 25: 19, 26: 28}
_E1_LOAD = core.load_p2_pretrained
_E1_ASSERT = core.assert_p2_model


def e1_yaml_digest():
    # The local Windows checkout uses CRLF; the AutoDL checkout may use LF.
    return hashlib.sha256(E1_YAML.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def reference_e1():
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils.torch_utils import init_seeds

    if e1_yaml_digest() != E1_YAML_NORMALIZED_SHA256:
        raise RuntimeError("Frozen E1 YAML differs from the paired-seed2 reference")
    init_seeds(core.SEED + 1, deterministic=True)
    cfg = yaml_model_load(str(E1_YAML))
    cfg["scale"] = "s"
    reference = DetectionModel(cfg, nc=10, verbose=False)
    _E1_ASSERT(reference)
    audit = _E1_LOAD(reference)
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return reference, audit, cpu_rng, cuda_rng


def load_p2_pretrained(target):
    """Transfer all and only pre-existing E1 tensors by audited layer map."""
    reference, original_audit, cpu_rng, cuda_rng = reference_e1()
    target_state = target.state_dict()
    mapped = {}
    for key, value in reference.state_dict().items():
        parts = key.split(".", 2)
        if len(parts) != 3 or parts[0] != "model":
            if key not in target_state or tuple(target_state[key].shape) != tuple(value.shape):
                raise RuntimeError(f"Unmapped E1 global tensor: {key}")
            new_key = key
        else:
            new_key = f"model.{MAP[int(parts[1])]}.{parts[2]}"
        if new_key not in target_state or tuple(target_state[new_key].shape) != tuple(value.shape):
            raise RuntimeError(f"E1->E33a key/shape mismatch: {key} -> {new_key}")
        if new_key in mapped:
            raise RuntimeError(f"Duplicate E1->E33a mapping: {new_key}")
        mapped[new_key] = value
    missing = sorted(set(target_state) - set(mapped))
    if not missing or any(not x.startswith(("model.20.", "model.27.")) for x in missing):
        raise RuntimeError(f"Unaccounted E33a-only tensors: {missing}")
    result = target.load_state_dict(mapped, strict=False)
    expected_reported = sorted(x for x in missing if not x.endswith(".num_batches_tracked"))
    if sorted(result.missing_keys) != expected_reported or result.unexpected_keys:
        raise RuntimeError(f"E1 migration was not strict: {result}")
    if any(not torch.equal(target.state_dict()[key].cpu(), value.cpu()) for key, value in mapped.items()):
        raise RuntimeError("Transferred E1 tensors differ after loading")
    # Match the random-state boundary of the deterministic E1 loader.
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    del reference
    return dict(source="deterministic E1 initialization from original yolo11s.pt",
                e1_transfer=original_audit, layer_map=MAP,
                loaded_tensor_count=len(mapped), new_tensor_count=len(missing),
                exact_common_tensor_equality=True)


def assert_p2_model(model):
    if len(model.model) != 29 or model.yaml.get("scale") != "s":
        raise RuntimeError("E33a requires the 29-layer s-scale graph")
    if model.model[20].__class__.__name__ != "IRPUpdate":
        raise RuntimeError("Missing P2->P3 update at layer 20")
    if model.model[27].__class__.__name__ != "IRPReturn":
        raise RuntimeError("Missing P3-delta->P2 return at layer 27")
    head = model.model[28]
    if head.__class__.__name__ != "Detect" or head.f != [27, 20, 23, 26] or head.nc != 10:
        raise RuntimeError("E33a must retain the standard four-level Detect")
    if list(model.stride.detach().cpu().tolist()) != [4., 8., 16., 32.]:
        raise RuntimeError(f"E33a Detect strides changed: {model.stride}")
    for old, new in MAP.items():
        if old in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
            continue
        if old == 26 and model.model[new].__class__.__name__ != "Detect":
            raise RuntimeError("Original Detect tower changed")


def validate_p2_config(cfg, original_cfg):
    from ultralytics.utils import yaml_load

    e1 = yaml_load(E1_YAML)
    expected = [
        *copy.deepcopy(e1["head"][:6]),
        [16, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 2], 1, "Concat", [1]],
        [-1, 2, "C3k2", [256, False]],
        [[16, 19], 1, "IRPUpdate", []],
        [20, 1, "Conv", [256, 3, 2]],
        [[-1, 13], 1, "Concat", [1]],
        [-1, 2, "C3k2", [512, False]],
        [-1, 1, "Conv", [512, 3, 2]],
        [[-1, 10], 1, "Concat", [1]],
        [-1, 2, "C3k2", [1024, True]],
        [[19, 20, 16], 1, "IRPReturn", []],
        [[27, 20, 23, 26], 1, "Detect", ["nc"]],
    ]
    checks = {
        "backbone": (cfg.get("backbone"), e1["backbone"]),
        "head": (core.canonical_layers(cfg.get("head")), core.canonical_layers(expected)),
        "scale": (cfg.get("scale"), "s"),
        "nc": (cfg.get("nc"), 10),
        "s_scaling": (cfg.get("scales", {}).get("s"), original_cfg["scales"]["s"]),
    }
    drift = {name: (actual, required) for name, (actual, required) in checks.items() if actual != required}
    if drift:
        raise RuntimeError(f"E33a architecture drift: {drift}")


def _identity_audit(candidate):
    from ultralytics.utils.loss import v8DetectionLoss

    reference, _, _, _ = reference_e1()
    reference.cpu().train()
    candidate.cpu().train()
    args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    reference.args = args
    candidate.args = args
    torch.manual_seed(3302)
    batch = dict(
        img=torch.rand(2, 3, 128, 128),
        batch_idx=torch.tensor([0., 0., 1., 1.]),
        cls=torch.tensor([[0.], [3.], [5.], [9.]]),
        bboxes=torch.tensor([[.25, .30, .08, .09], [.68, .62, .24, .20],
                             [.46, .52, .12, .10], [.70, .28, .34, .30]]),
    )
    reference.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    base_raw = reference(batch["img"])
    new_raw = candidate(batch["img"])
    if len(base_raw) != len(new_raw) or any(not torch.equal(a, b) for a, b in zip(base_raw, new_raw)):
        raise RuntimeError("E33a initialization does not reproduce E1 raw predictions exactly")
    base_loss, base_items = v8DetectionLoss(reference)(base_raw, batch)
    new_loss, new_items = v8DetectionLoss(candidate)(new_raw, batch)
    if not torch.equal(base_loss, new_loss) or not torch.equal(base_items, new_items):
        raise RuntimeError("E33a initialization does not reproduce E1 loss exactly")
    new_loss.backward()
    if candidate.model[20].out.weight.grad is None or not torch.isfinite(candidate.model[20].out.weight.grad).all():
        raise RuntimeError("P2->P3 update has no finite detection gradient")
    return dict(raw_prediction_exact=True, total_loss_exact=True, loss_items_exact=True,
                update_output_gradient_norm=float(candidate.model[20].out.weight.grad.norm()))


def _two_step_new_path_audit(model):
    """An initial zero residual must not permanently block either new path."""
    update = copy.deepcopy(model.model[20]).train()
    returned = copy.deepcopy(model.model[27]).train()
    update.zero_grad(set_to_none=True)
    returned.zero_grad(set_to_none=True)
    torch.manual_seed(3303)
    p3 = torch.randn(2, 128, 8, 8)
    p2 = torch.randn(2, 128, 16, 16)
    old_p3 = p3.clone()
    first_p3 = update([p3, p2])
    first_p2 = returned([p2, first_p3, old_p3])
    if not torch.equal(first_p3, p3) or not torch.equal(first_p2, p2):
        raise RuntimeError("IRP modules are not exact identities at initialisation")
    first_p3.square().mean().backward()
    grad = update.out.weight.grad
    if grad is None or not torch.isfinite(grad).all() or float(grad.norm()) <= 0:
        raise RuntimeError("IRPUpdate final projection is blocked at step one")
    with torch.no_grad():
        update.out.weight.add_(-0.01 * grad)
    update.zero_grad(set_to_none=True)
    returned.zero_grad(set_to_none=True)
    second_p3 = update([p3, p2])
    second_p2 = returned([p2, second_p3, old_p3])
    if torch.equal(second_p3, old_p3):
        raise RuntimeError("IRPUpdate stayed zero after its first gradient step")
    second_p2.square().mean().backward()
    return_grad = returned.project[-1].weight.grad
    if return_grad is None or not torch.isfinite(return_grad).all() or float(return_grad.norm()) <= 0:
        raise RuntimeError("IRPReturn did not receive gradient after the P3 update activated")
    return dict(initial_identities_exact=True, update_step1_grad_norm=float(grad.norm()),
                return_step2_grad_norm=float(return_grad.norm()))


def p2_preflight(report, eval_only=False):
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils import yaml_load
    from ultralytics.utils.torch_utils import init_seeds

    if not MODEL_YAML.is_file() or not E1_YAML.is_file() or not MODULE_SOURCE.is_file():
        raise FileNotFoundError("E33a files or frozen E1 reference are missing; run installer")
    if e1_yaml_digest() != E1_YAML_NORMALIZED_SHA256:
        raise RuntimeError("E1 reference YAML changed; do not train")
    if not core.PRETRAINED.is_file() or core.sha256(core.PRETRAINED) != core.PRETRAINED_SHA256:
        raise RuntimeError("Original yolo11s.pt missing or changed")
    if Path(core.ultralytics.__file__).resolve().parent != (core.SOURCE / "ultralytics").resolve():
        raise RuntimeError("Wrong Ultralytics source; activate yolo11 environment")
    cfg = yaml_model_load(str(MODEL_YAML))
    original = yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml")
    validate_p2_config(cfg, original)
    hashes = {name: core.sha256(core.SOURCE / "ultralytics" / name) for name in (
        "nn/tasks.py", "nn/modules/block.py", "nn/modules/head.py", "nn/modules/conv.py",
        "engine/model.py", "engine/trainer.py", "models/yolo/detect/train.py",
        "utils/tal.py", "utils/loss.py")}
    report["p2_script_revision"] = REVISION
    report["p2_architecture"] = dict(
        yaml=str(MODEL_YAML), yaml_sha256=core.sha256(MODEL_YAML),
        e1_yaml_normalized_sha256=e1_yaml_digest(), module_sha256=core.sha256(MODULE_SOURCE),
        original_weights_sha256=core.sha256(core.PRETRAINED),
        source_code_sha256=hashes, strides=[4, 8, 16, 32],
        variant="E1 seed2 plus P2->P3 re-aggregation, original PAN P4/P5, P3-delta->P2 return",
        original_loss_tal_detect=True, initialization_note="Exact deterministic E1 transfer; only IRP layers new",
    )
    out = Path(report["report_dir"])
    for source in (MODEL_YAML, E1_YAML, MODULE_SOURCE):
        shutil.copy2(source, out / source.name)
    init_seeds(core.SEED, deterministic=True)
    model = DetectionModel(cfg, nc=10, verbose=False)
    assert_p2_model(model)
    report["preflight_transfer"] = load_p2_pretrained(model)
    report["initial_identity"] = _identity_audit(model)
    report["new_path_gradient_audit"] = _two_step_new_path_audit(model)
    model.eval()
    with torch.no_grad():
        _, raw = model(torch.zeros(1, 3, 256, 256))
    sizes = [list(x.shape[-2:]) for x in raw]
    if sizes != [[64, 64], [32, 32], [16, 16], [8, 8]]:
        raise RuntimeError(f"Wrong E33a Detect feature shapes: {sizes}")
    report["p2_architecture"]["CPU_forward_256_feature_shapes"] = sizes
    report["p2_architecture"]["preflight_parameters"] = sum(x.numel() for x in model.parameters())
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
    print("E33a E1-transfer, initial identity, loss and TAL checks passed.", flush=True)


def install_e33a():
    core.SEED = 2
    core.EXPERIMENT = EXPERIMENT
    core.SCRIPT_REVISION = REVISION
    core.MODEL_YAML = MODEL_YAML
    core.SMOKE_GATE = core.ROOT / "comparison_reports/e33a_seed2_smoke_passed.json"
    core.AUTO_SHUTDOWN = False
    core.SHUTDOWN_ON_FAILURE = False
    core.__file__ = str(Path(__file__).resolve())
    core.load_p2_pretrained = load_p2_pretrained
    core.assert_p2_model = assert_p2_model
    core.validate_p2_config = validate_p2_config
    core.p2_preflight = p2_preflight
    params = core.training_params()
    required = dict(epochs=200, imgsz=1024, batch=8, workers=8, seed=2,
                    deterministic=True, optimizer="SGD", lr0=0.01, lrf=0.01,
                    momentum=0.937, weight_decay=0.0005, amp=True, cos_lr=True,
                    close_mosaic=10, nbs=64, cache=False, rect=False, multi_scale=False)
    drift = {k: (params.get(k), v) for k, v in required.items() if params.get(k) != v}
    if drift:
        raise RuntimeError(f"E1 seed2 training protocol drift: {drift}")


if __name__ == "__main__":
    install_e33a()
    core.main()
