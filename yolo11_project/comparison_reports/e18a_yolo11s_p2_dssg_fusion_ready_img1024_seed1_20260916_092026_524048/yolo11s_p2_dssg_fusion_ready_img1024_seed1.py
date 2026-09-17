#!/usr/bin/env python3
"""E18a: strict E17a pre-fusion adaptation + original E10a DSSG on frozen E1.

Formal protocol is unchanged from E1/E10a/E17a: YOLO11s, VisDrone, imgsz 1024,
batch 8, seed 1, deterministic SGD, AMP, cosine LR, 200 epochs, TAL2 chunk=2.
No automatic shutdown is used on check, smoke, formal, evaluation or failure paths.
"""

from __future__ import annotations

import copy
import gc
import hashlib
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
    raise ImportError("E18a requires p2_tal_chunked_vgpu32.py beside this file") from exc
sys.modules["p2_tal_chunked"] = fast_tal

try:
    import yolo11s_p2_img1024_seed1 as core
except ImportError as exc:
    raise ImportError("E18a requires the frozen E1 runner beside this file") from exc


EXPERIMENT = "e18a_yolo11s_p2_dssg_fusion_ready_img1024_seed1"
REVISION = "e18a_e10a_e17a_strict_combo_seed1_200e_tal2_v4_state_transfer_fix"
MODEL_YAML = HERE / "yolo11s-p2-dssg-fusion-ready.yaml"
REFERENCE_E1_YAML = HERE / "yolo11s-p2-add.yaml"
REFERENCE_E17_YAML = HERE / "yolo11s-p2-fusion-ready.yaml"
E17_MODULE = core.SOURCE / "ultralytics/nn/modules/e17_fusion.py"
E17_LOSS = core.SOURCE / "ultralytics/utils/e17_fusion_loss.py"
DSSG_SOURCE = core.SOURCE / "ultralytics/nn/modules/dssg.py"
MATCHING_HELPER = Path(fast_tal.__file__).resolve()
SMOKE_GATE = core.ROOT / "comparison_reports/e18a_dssg_fusion_ready_smoke_passed.json"

# Frozen source fingerprints from E1/E17a/E10a authorities.
EXPECTED_E1_YAML_SHA256 = "98df605986f71295216f2abc549f5f0e381a7a39a6cc81b0419ab118cb29135b"
EXPECTED_E17_YAML_SHA256 = "4bbd5843d2a8d418becbeb1bbd4747ee38fcbcdb3dd71469586fd2b3a3e19825"
EXPECTED_E17_MODULE_SHA256 = "84e266272b93c76dbf9d30c0e40d2cdd8e0d6d38fec00a7a7fd76c079b6c64ff"
EXPECTED_E17_LOSS_SHA256 = "4c4620ffaabffc83c243f04b71d80414f7629c88ea411dc2a8fec1f6f92976f0"
EXPECTED_DSSG_SHA256 = "a0c26c5d0d3cfae7428f2211ce2ee37ec7cbda9193cfbb4a391ad38235b652fc"

_E1_LOAD = core.load_p2_pretrained
_E1_ASSERT = core.assert_p2_model
_E1_TRAIN = core.train

# Semantic identity mapping from the deterministic E1 model into E18a.
# E18a new layers are 12,16 (E17 adapters), 28,29 (E10a DSSG) and 30.aux_head.
E1_TO_E18 = {
    **{i: i for i in range(11)},
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
    26: 30,
}


def _target_key(key):
    parts = key.split(".")
    if len(parts) < 3 or parts[0] != "model" or not parts[1].isdigit():
        raise ValueError(f"Unrecognized E1 state tensor: {key}")
    source_index = int(parts[1])
    if source_index not in E1_TO_E18:
        raise ValueError(f"No E1->E18 semantic layer mapping for: {key}")
    parts[1] = str(E1_TO_E18[source_index])
    return ".".join(parts)


def _reference():
    """Build the same deterministic E1 initialization used by the E17a fair transfer."""
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
    """Copy every E1 tensor exactly; initialize only the explicitly new E18a tensors."""
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
    allowed_prefixes = (
        "model.12.",
        "model.16.",
        "model.28.",
        "model.29.",
        "model.30.aux_head.",
    )
    unexpected = [key for key in missing if not key.startswith(allowed_prefixes)]
    if unexpected:
        raise ValueError(f"Unexpected uninitialized E18a target tensors: {unexpected}")
    if not missing:
        raise ValueError("E18a unexpectedly has no new tensors")

    result = target.load_state_dict(mapped, strict=False)
    # PyTorch deliberately suppresses missing BatchNorm ``num_batches_tracked``
    # buffers from ``IncompatibleKeys.missing_keys`` for backward compatibility.
    # They are still present in target.state_dict() and are legitimately new only
    # inside E18a's explicitly allowed modules. Compare against the exact set that
    # PyTorch is expected to report rather than against the raw target-state diff.
    expected_reported_missing = sorted(
        key for key in missing if not key.endswith("num_batches_tracked")
    )
    reported_missing = sorted(result.missing_keys)
    if reported_missing != expected_reported_missing or result.unexpected_keys:
        raise ValueError(
            "Incomplete/incorrect E1 state migration: "
            f"reported_missing={reported_missing}, "
            f"expected_reported_missing={expected_reported_missing}, "
            f"suppressed_bn_counters={sorted(set(missing) - set(expected_reported_missing))}, "
            f"unexpected={result.unexpected_keys}"
        )

    # Preserve the deterministic RNG state after constructing the E1 reference so
    # E18a's new tensors are compared under the same fair initialization procedure.
    core.torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        core.torch.cuda.set_rng_state_all(cuda_rng)

    actual = target.state_dict()
    for key, value in mapped.items():
        if not core.torch.equal(actual[key].cpu(), value.cpu()):
            raise RuntimeError(f"Transferred E1 tensor changed: {key}")

    # Explicitly prove that E1 P2 C3k2 and the complete four-scale Detect state are exact.
    exact_groups = {
        "e1_p2": [key for key in mapped if key.startswith("model.27.")],
        "e1_detect": [key for key in mapped if key.startswith("model.30.") and not key.startswith("model.30.aux_head.")],
    }
    if not all(exact_groups.values()):
        raise RuntimeError(f"E18a exact E1 transfer group unexpectedly empty: {exact_groups}")

    audit = dict(
        source="original 80-class yolo11s.pt transferred into deterministic E1, then every E1 tensor mapped exactly into E18a",
        source_audit=original_audit,
        loaded_tensor_count=len(mapped),
        new_tensor_keys=missing,
        exact_tensor_equality=True,
        exact_groups={name: len(keys) for name, keys in exact_groups.items()},
        tensor_mapping=mapping,
        reference_e1_yaml_sha256=core.sha256(REFERENCE_E1_YAML),
        note=(
            "No experiment best.pt is a parent. E1 backbone/neck/P2/four-scale Detect tensors are exact. "
            "Only layers 12,16,28,29 and model.30.aux_head are newly initialized."
        ),
    )
    del reference
    return audit


def _canonical_text_sha256(path):
    """Hash UTF-8 text after normalizing CRLF/CR to LF; content otherwise stays exact."""
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"Frozen text source is not UTF-8: {path}") from exc
    canonical = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _assert_frozen_sources():
    checks = {
        REFERENCE_E1_YAML: EXPECTED_E1_YAML_SHA256,
        REFERENCE_E17_YAML: EXPECTED_E17_YAML_SHA256,
        E17_MODULE: EXPECTED_E17_MODULE_SHA256,
        E17_LOSS: EXPECTED_E17_LOSS_SHA256,
        DSSG_SOURCE: EXPECTED_DSSG_SHA256,
    }
    mismatches = []
    for path, expected in checks.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = core.sha256(path)
        canonical = _canonical_text_sha256(path)
        if raw != expected and canonical != expected:
            mismatches.append((str(path), expected, raw, canonical))
    if mismatches:
        raise RuntimeError(
            "Frozen E10a/E17a source mismatch after line-ending normalization: "
            f"{mismatches}"
        )


def assert_p2_model(model):
    if len(model.model) != 31 or model.yaml.get("scale") != "s":
        raise ValueError("Expected the 31-layer s-scale E18a architecture")

    # E17a pre-fusion adapters unchanged.
    for index, inputs in ((12, [11, 6]), (16, [15, 4])):
        adapter = model.model[index]
        if adapter.__class__.__name__ != "FusionReadyAdapter" or adapter.f != inputs:
            raise ValueError(f"Incorrect E18a pre-fusion adapter at layer {index}")
        if adapter.layer_scale.numel() != 1 or not core.torch.isfinite(adapter.layer_scale.detach()).all():
            raise ValueError(f"FusionReadyAdapter layer_scale must be one finite scalar at layer {index}")

    # All original E1/E17a upsampling remains nearest.
    for index in (11, 15, 25):
        layer = model.model[index]
        if layer.__class__.__name__ != "Upsample" or layer.mode != "nearest" or layer.scale_factor != 2.0:
            raise ValueError(f"Upsample {index} changed from frozen nearest-neighbor 2x")

    deep = model.model[28]
    detail = model.model[29]
    if deep.__class__.__name__ != "DeepSemanticGuide" or deep.f != [18, 21, 24]:
        raise ValueError("Layer 28 must be original E10a DeepSemanticGuide(P3,P4,P5)")
    if (deep.p3_channels, deep.p4_channels, deep.p5_channels, deep.hidden) != (128, 256, 512, 32):
        raise ValueError("DeepSemanticGuide runtime channels differ from E10a")
    if abs(deep.topk_ratio - 0.0625) > 1e-12 or abs(deep.sparse_mix - 0.5) > 1e-12:
        raise ValueError("DeepSemanticGuide sparse settings differ from E10a")
    if tuple(deep.layer_scale.shape) != (1, 128, 1, 1) or not core.torch.isfinite(deep.layer_scale.detach()).all():
        raise ValueError("DeepSemanticGuide layer_scale must remain finite with shape [1,128,1,1]")

    if detail.__class__.__name__ != "SemanticDetailGate" or detail.f != [27, 28]:
        raise ValueError("Layer 29 must be original E10a SemanticDetailGate(P2,P3_sem)")
    if (detail.p2_channels, detail.p3_channels) != (128, 128):
        raise ValueError("SemanticDetailGate runtime channels differ from E10a")
    if tuple(detail.layer_scale.shape) != (1, 128, 1, 1) or not core.torch.isfinite(detail.layer_scale.detach()).all():
        raise ValueError("SemanticDetailGate layer_scale must remain finite with shape [1,128,1,1]")

    head = model.model[30]
    expected_sources = [29, 28, 21, 24, 12, 16]
    if head.__class__.__name__ != "E17Detect" or head.f != expected_sources or head.nc != 10:
        raise ValueError("E18a must use the repaired E17Detect with four main + two auxiliary sources")
    if list(model.stride.cpu().tolist()) != [4.0, 8.0, 16.0, 32.0]:
        raise ValueError(f"E18a detection strides changed: {model.stride}")

    class_names = {layer.__class__.__name__ for layer in model.model}
    forbidden_tokens = ("DySample", "PGALP", "P3SDE", "P4P3SpatialGuide", "TSDR")
    forbidden = sorted(name for name in class_names if any(token in name for token in forbidden_tokens))
    if forbidden:
        raise ValueError(f"E18a contains unplanned modules: {forbidden}")


def validate_p2_config(cfg, original_cfg):
    del original_cfg
    from ultralytics.utils import yaml_load

    e17 = yaml_load(REFERENCE_E17_YAML)
    expected_head = copy.deepcopy(e17["head"][:-1])
    # E18a specification spells the frozen P2 Concat source explicitly as layer 25
    # instead of E17a's equivalent -1 notation. No computation changes.
    expected_head[26 - 11][0] = [25, 2]
    expected_head.extend((
        [[18, 21, 24], 1, "DeepSemanticGuide", [0.25, 0.0625, 0.5, 0.05]],
        [[27, 28], 1, "SemanticDetailGate", [0.05]],
        [[29, 28, 21, 24, 12, 16], 1, "E17Detect", ["nc"]],
    ))
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
        details = "; ".join(f"{name}: expected={expected!r}, actual={actual!r}" for name, actual, expected in differences)
        raise ValueError("E18a YAML is not the strict E17a + original E10a combination. " + details)


def _has_finite_gradient(module):
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad and parameter.grad is not None]
    return bool(gradients) and all(core.torch.isfinite(gradient).all() for gradient in gradients)


def _assert_new_module_initialization(model):
    """Initialization-only checks; never call this on a trained checkpoint."""
    for index in (12, 16):
        value = float(model.model[index].layer_scale.detach())
        if abs(value - 0.05) > 1e-7:
            raise ValueError(f"FusionReadyAdapter residual initialization changed at layer {index}: {value}")
    deep = model.model[28]
    detail = model.model[29]
    deep_scale = deep.layer_scale.detach()
    detail_scale = detail.layer_scale.detach()
    if float((deep_scale - 0.05).abs().max()) > 1e-7:
        raise ValueError("DeepSemanticGuide residual initialization differs from E10a")
    if float((detail_scale - 0.05).abs().max()) > 1e-7:
        raise ValueError("SemanticDetailGate residual initialization differs from E10a")


def _dssg_unit_check(deep, detail):
    p2 = core.torch.randn(2, 128, 64, 80, requires_grad=True)
    p3 = core.torch.randn(2, 128, 32, 40, requires_grad=True)
    p4 = core.torch.randn(2, 256, 16, 20, requires_grad=True)
    p5 = core.torch.randn(2, 512, 8, 10, requires_grad=True)
    p3_sem = deep([p3, p4, p5])
    p2_sem = detail([p2, p3_sem])
    if p3_sem.shape != p3.shape or p2_sem.shape != p2.shape:
        raise RuntimeError("E18a DSSG shape-preservation check failed")
    if not core.torch.isfinite(p3_sem).all() or not core.torch.isfinite(p2_sem).all():
        raise RuntimeError("E18a DSSG finite-output check failed")
    _, channel, p3_spatial = deep.routing(p3.detach(), p4.detach(), p5.detach())
    _, p2_spatial = detail.routing(p2.detach(), p3_sem.detach())
    if float((channel - 0.5).abs().max()) > 1e-6:
        raise RuntimeError("DeepSemanticGuide channel initialization is not neutral")
    if float((p3_spatial - 0.5).abs().max()) > 1e-6 or float((p2_spatial - 0.5).abs().max()) > 1e-6:
        raise RuntimeError("DSSG spatial initialization is not neutral")
    p2_sem.float().square().mean().backward()
    if not _has_finite_gradient(deep) or not _has_finite_gradient(detail):
        raise RuntimeError("DSSG backward/gradient check failed")
    return dict(
        parameter_count=sum(p.numel() for p in deep.parameters()) + sum(p.numel() for p in detail.parameters()),
        channel_mean=float(channel.mean()),
        p3_spatial_mean=float(p3_spatial.mean()),
        p2_spatial_mean=float(p2_spatial.mean()),
    )


def _loss_mode_regression(model):
    """Cover train/backward -> eval validation loss -> train/backward on one model instance."""
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
        raise RuntimeError("E18a first training loss is invalid")
    train1.backward()
    for index in (12, 16, 28, 29):
        if not _has_finite_gradient(model.model[index]):
            raise RuntimeError(f"E18a new module {index} did not receive finite training gradients")
    if not _has_finite_gradient(model.model[30].aux_head):
        raise RuntimeError("E18a shared auxiliary head did not receive finite gradients")
    if model.model[30]._aux_logits is not None:
        raise RuntimeError("E18a auxiliary logits were not cleared after training loss")

    # Important: criterion was initialized in normal training mode above. Validation
    # intentionally uses no_grad(), not inference_mode(), then calls the same criterion.
    model.eval()
    with core.torch.no_grad():
        preds = model(batch["img"])
        val_total, val_items = model.criterion(preds, batch)
    if not core.torch.isfinite(val_total) or len(val_items) != 3:
        raise RuntimeError("E18a eval-mode validation loss is invalid")
    if model.model[30]._aux_logits is not None:
        raise RuntimeError("E18a eval mode unexpectedly retained auxiliary logits")

    # Switch back to training and prove backward still works after validation.
    model.train()
    model.zero_grad(set_to_none=True)
    train2, items2 = model(batch)
    if not core.torch.isfinite(train2) or len(items2) != 3:
        raise RuntimeError("E18a second training loss is invalid after validation")
    train2.backward()
    for index in (12, 16, 28, 29):
        if not _has_finite_gradient(model.model[index]):
            raise RuntimeError(f"E18a module {index} lost gradients after eval->train transition")
    if not _has_finite_gradient(model.model[30].aux_head):
        raise RuntimeError("E18a auxiliary head lost gradients after eval->train transition")

    if original_args is not None:
        model.args = original_args
    return dict(
        train_loss_before_eval=float(train1.detach()),
        validation_loss=float(val_total.detach()),
        train_loss_after_eval=float(train2.detach()),
        criterion=model.criterion.__class__.__name__,
        mode_transition="train/backward -> eval/no_grad validation loss -> train/backward",
    )


def p2_preflight(report, eval_only=False):
    del eval_only
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils import yaml_load
    from ultralytics.utils.torch_utils import init_seeds

    for path in (
        MODEL_YAML, REFERENCE_E1_YAML, REFERENCE_E17_YAML,
        E17_MODULE, E17_LOSS, DSSG_SOURCE, MATCHING_HELPER,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if Path(core.ultralytics.__file__).resolve().parent != (core.SOURCE / "ultralytics").resolve():
        raise RuntimeError("Wrong editable Ultralytics source; activate yolo11 conda environment")
    if core.sha256(core.PRETRAINED) != core.PRETRAINED_SHA256:
        raise RuntimeError("Original yolo11s.pt changed")
    _assert_frozen_sources()

    cfg = yaml_model_load(str(MODEL_YAML))
    validate_p2_config(cfg, yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml"))
    out = Path(report["report_dir"])
    for path in (
        MODEL_YAML, REFERENCE_E1_YAML, REFERENCE_E17_YAML,
        E17_MODULE, E17_LOSS, DSSG_SOURCE, MATCHING_HELPER,
    ):
        shutil.copy2(path, out / path.name)

    report["p2_script_revision"] = REVISION
    report["p2_architecture"] = dict(
        yaml_sha256=core.sha256(MODEL_YAML),
        reference_e1_yaml_sha256=core.sha256(REFERENCE_E1_YAML),
        reference_e17_yaml_sha256=core.sha256(REFERENCE_E17_YAML),
        original_weights_sha256=core.sha256(core.PRETRAINED),
        e17_module_sha256=core.sha256(E17_MODULE),
        e17_loss_sha256=core.sha256(E17_LOSS),
        dssg_sha256=core.sha256(DSSG_SOURCE),
        tasks_sha256=core.sha256(core.SOURCE / "ultralytics/nn/tasks.py"),
        variant=(
            "E18a = frozen E1 + unchanged E17a pre-Concat P4/P3 FusionReadyAdapters and "
            "training-only scale-valid auxiliary supervision + unchanged E10a DeepSemanticGuide/SemanticDetailGate"
        ),
        detect_strides=[4, 8, 16, 32],
        main_detect_channels=[128, 128, 256, 512],
        auxiliary_lateral_channels=[256, 256],
        upsampling="nearest unchanged at layers 11/15/25; DSSG nearest internal alignment unchanged",
        auxiliary_gain=cfg["e17_aux"]["gain"],
        tal2=fast_tal.implementation_info(),
        claim_boundary=(
            "One strict E10a+E17a compatibility experiment only; no E10clean, DySample, slicing, PGALP, P3SDE or E16a TSDR."
        ),
    )

    init_seeds(core.SEED, deterministic=True)
    model = DetectionModel(cfg, nc=10, verbose=False)
    assert_p2_model(model)
    report["preflight_transfer"] = load_p2_pretrained(model)
    _assert_new_module_initialization(model)
    report["p2_architecture"]["initialization_check"] = {
        "adapter_layer_scale": [float(model.model[i].layer_scale.detach()) for i in (12, 16)],
        "dssg_layer_scale_mean": [
            float(model.model[28].layer_scale.detach().mean()),
            float(model.model[29].layer_scale.detach().mean()),
        ],
        "scope": "fresh deterministic E1->E18a initialization only; not enforced on trained checkpoints",
    }
    report["p2_architecture"]["dssg_unit_check"] = _dssg_unit_check(model.model[28], model.model[29])

    captured = {}
    def capture_head_inputs(module, args):
        del module
        features = args[0]
        captured["channels"] = [int(t.shape[1]) for t in features]
        captured["grids"] = [list(t.shape[-2:]) for t in features]

    hook = model.model[30].register_forward_pre_hook(capture_head_inputs)
    model.eval()
    with core.torch.no_grad():
        output, raw = model(core.torch.zeros(1, 3, 256, 256))
    hook.remove()
    expected_channels = [128, 128, 256, 512, 256, 256]
    if captured.get("channels") != expected_channels:
        raise RuntimeError(f"E18a head source channels changed: {captured.get('channels')}")
    sizes = [list(tensor.shape[-2:]) for tensor in raw]
    if sizes != [[64, 64], [32, 32], [16, 16], [8, 8]] or not core.torch.isfinite(output).all():
        raise RuntimeError(f"Incorrect/nonfinite E18a inference features: {sizes}")
    report["p2_architecture"]["head_source_channels"] = captured["channels"]
    report["p2_architecture"]["head_source_grids_256"] = captured["grids"]
    report["p2_architecture"]["cpu_feature_grids"] = sizes
    report["p2_architecture"]["parameters_including_training_only_aux"] = sum(p.numel() for p in model.parameters())

    # Explicitly regression-test the repaired E17 auxiliary-loss mode transitions on
    # the combined model before any GPU time is spent.
    report["loss_mode_regression"] = _loss_mode_regression(model)
    core.save_report(report)
    del model, raw, output
    gc.collect()
    print("E18a topology, frozen-source, exact-E1-transfer, DSSG and loss-mode checks passed.", flush=True)

    original_cfg = yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml")
    original_cfg.update(scale="s", nc=10)
    report["matching_equivalence"] = fast_tal.verify_equivalence(
        copy.deepcopy(cfg), device="cpu", baseline_cfg=original_cfg
    )
    fast_tal.assert_patch_restoration()
    core.write_json(out / "tal_equivalence.json", report["matching_equivalence"])
    core.save_report(report)
    print("TAL2 equivalence passed; run --smoke2 for real CUDA batch8/1024 memory/training/full evaluation.", flush=True)


def train(report, smoke=False):
    weights = _E1_TRAIN(report, smoke=smoke)
    run_dir = Path(report["run_dir"])
    for path in (E17_MODULE, E17_LOSS, DSSG_SOURCE, REFERENCE_E1_YAML, REFERENCE_E17_YAML):
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
