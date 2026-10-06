#!/usr/bin/env python3
"""E20b: strict E16a TSDR plus scale-adaptive DFL quality calibration."""

from __future__ import annotations

import copy
import gc
import json
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
    raise ImportError("E20b requires p2_tal_chunked_vgpu32.py beside this file") from exc
sys.modules["p2_tal_chunked"] = fast_tal

try:
    import yolo11s_p2_img1024_seed1 as core
except ImportError as exc:
    raise ImportError("E20b requires the frozen E1 runner beside this file") from exc


EXPERIMENT = "e20b_yolo11s_p2_tsdr_sdqc_img1024_seed1"
REVISION = "e20b_strict_e16a_plus_sdqc_seed1_200e_tal2_v1"
MODEL_YAML = HERE / "yolo11s-p2-tsdr-sdqc.yaml"
REFERENCE_E16_YAML = HERE / "e20b_reference_yolo11s-p2-tsdr.yaml"
REFERENCE_E1_YAML = HERE / "e20b_reference_yolo11s-p2-add.yaml"
MATCHING_HELPER = Path(fast_tal.__file__).resolve()
SMOKE_GATE = core.ROOT / "comparison_reports/e20b_tsdr_sdqc_smoke_passed.json"
MODULE_SOURCE = core.SOURCE / "ultralytics/nn/modules/e20b_tsdr_sdqc.py"
LOSS_SOURCE = core.SOURCE / "ultralytics/utils/e20b_tsdr_sdqc_loss.py"
SDQC_SOURCE = core.SOURCE / "ultralytics/nn/modules/sdqc.py"
SDQC_LOSS = core.SOURCE / "ultralytics/utils/sdqc_loss.py"
DSSG_SOURCE = core.SOURCE / "ultralytics/nn/modules/dssg.py"
TSDR_SOURCE = core.SOURCE / "ultralytics/nn/modules/tsdr.py"
TSDR_LOSS = core.SOURCE / "ultralytics/utils/tsdr_loss.py"
E16_METRICS = HERE / "e20b_reference_e16a_metrics_seed1_200e.json"

_CORE_TRAIN = core.train
_CORE_EVALUATE = core.evaluate
_E1_LOAD = core.load_p2_pretrained
_E1_ASSERT = core.assert_p2_model


def _build_e1_reference():
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils.torch_utils import init_seeds

    init_seeds(core.SEED + 1, deterministic=True)
    cfg = yaml_model_load(str(REFERENCE_E1_YAML))
    cfg["scale"] = "s"
    model = DetectionModel(cfg, nc=10, verbose=False)
    _E1_ASSERT(model)
    audit = _E1_LOAD(model)
    cpu_rng = core.torch.random.get_rng_state()
    cuda_rng = core.torch.cuda.get_rng_state_all() if core.torch.cuda.is_available() else None
    return model, audit, cpu_rng, cuda_rng


def _e1_target_key(key):
    if key.startswith("model.26."):
        return key.replace("model.26.", "model.28.", 1)
    return key


def load_p2_pretrained(target):
    """Map all E1 tensors exactly; E16a and SDQC tensors remain deterministic-new."""
    reference, source_audit, cpu_rng, cuda_rng = _build_e1_reference()
    target_state = target.state_dict()
    mapped = {}
    mapping = []
    for key, value in reference.state_dict().items():
        destination = _e1_target_key(key)
        if destination not in target_state or target_state[destination].shape != value.shape:
            raise ValueError(f"E1->E20b semantic transfer mismatch: {key} -> {destination}")
        mapped[destination] = value
        mapping.append(dict(source=key, target=destination, shape=list(value.shape)))
    missing = sorted(set(target_state) - set(mapped))
    allowed = ("model.26.", "model.27.", "model.28.detail_project.", "model.28.gate_net.",
               "model.28.raw_gain", "model.28.gaussian3", "model.28.quality_predictor.",
               "model.28.quality_gamma_logit")
    unexpected = [key for key in missing if not key.startswith(allowed)]
    if unexpected or not missing:
        raise ValueError(f"Unexpected E20b uninitialized tensors: unexpected={unexpected}, all={missing}")
    # PyTorch's BatchNorm backward-compatibility hook keeps a target module's
    # num_batches_tracked buffer but intentionally omits that key from the
    # returned missing_keys list.  Audit those buffers separately instead of
    # treating the compatibility behavior as a failed semantic transfer.
    compatibility_bn_counters = {key for key in missing if key.endswith(".num_batches_tracked")}
    expected_reported_missing = set(missing) - compatibility_bn_counters
    bn_counter_before = {
        key: target_state[key].detach().cpu().clone() for key in compatibility_bn_counters
    }
    result = target.load_state_dict(mapped, strict=False)
    if set(result.missing_keys) != expected_reported_missing or result.unexpected_keys:
        raise RuntimeError(f"E20b transfer audit failed: {result}")
    core.torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        core.torch.cuda.set_rng_state_all(cuda_rng)
    actual = target.state_dict()
    changed_bn_counters = [
        key for key, value in bn_counter_before.items() if not core.torch.equal(actual[key].cpu(), value)
    ]
    if changed_bn_counters:
        raise RuntimeError(f"BatchNorm compatibility counters changed: {changed_bn_counters[:5]}")
    changed = [key for key, value in mapped.items() if not core.torch.equal(actual[key].cpu(), value.cpu())]
    if changed:
        raise RuntimeError(f"E1 tensor equality failed: {changed[:5]}")
    del reference
    return dict(
        source="original yolo11s.pt -> deterministic E1 -> semantic map into E20b",
        source_audit=source_audit,
        loaded_tensor_count=len(mapped), new_tensor_keys=missing,
        exact_tensor_equality=True, tensor_mapping=mapping,
        note="E10a DSSG, E16a TSDR and SDQC are deterministic-new; every E1 tensor is exact.",
    )


def assert_p2_model(model):
    if len(model.model) != 29 or model.yaml.get("scale") != "s":
        raise ValueError("Expected the 29-layer s-scale E20b graph")
    deep, detail, head = model.model[26], model.model[27], model.model[28]
    if deep.__class__.__name__ != "DeepSemanticGuide" or deep.f != [16, 19, 22]:
        raise ValueError("Layer26 must be the unchanged E10a DeepSemanticGuide")
    if detail.__class__.__name__ != "SemanticDetailGate" or detail.f != [25, 26]:
        raise ValueError("Layer27 must be the unchanged E10a SemanticDetailGate")
    if head.__class__.__name__ != "TSDRSDQCDetect" or head.f != [0, 27, 26, 19, 22]:
        raise ValueError("Layer28 must be TSDRSDQCDetect with E16a inputs")
    if head.input_channels != (32, 128, 128, 256, 512) or head.gate_hidden != 16:
        raise ValueError("E20b runtime channels or TSDR gate changed")
    if abs(head.max_gain - 0.25) > 1e-12 or head.no != head.nc + head.reg_max * 4 + 1:
        raise ValueError("E20b gain/raw-output contract changed")
    if list(model.stride.cpu().tolist()) != [4.0, 8.0, 16.0, 32.0]:
        raise ValueError(f"E20b strides changed: {model.stride}")
    for index in (11, 14, 23):
        layer = model.model[index]
        if layer.__class__.__name__ != "Upsample" or layer.mode != "nearest" or layer.scale_factor != 2.0:
            raise ValueError(f"Nearest upsample changed at layer {index}")


def validate_p2_config(cfg, original_cfg):
    del original_cfg
    from ultralytics.utils import yaml_load

    e16 = yaml_load(REFERENCE_E16_YAML)
    expected_head = copy.deepcopy(e16["head"])
    expected_head[-1][2] = "TSDRSDQCDetect"
    checks = [
        ("backbone", core.canonical_layers(cfg.get("backbone", [])), core.canonical_layers(e16["backbone"])),
        ("head", core.canonical_layers(cfg.get("head", [])), core.canonical_layers(expected_head)),
        ("scale", cfg.get("scale"), "s"), ("nc", cfg.get("nc"), 10),
        ("tsdr", cfg.get("tsdr"), e16.get("tsdr")),
        ("sdqc", cfg.get("sdqc"), dict(enabled=True, quality_gain=0.25, nwd_constant=12.8,
                                         scale_center=32.0, scale_temperature=8.0)),
        ("combined", cfg.get("e20b_tsdr_sdqc"), {"enabled": True}),
    ]
    differences = [(name, actual, expected) for name, actual, expected in checks if actual != expected]
    if differences:
        raise ValueError("E20b is not the strict E16a+SDQC design: " + "; ".join(
            f"{name}: expected={expected!r}, actual={actual!r}" for name, actual, expected in differences
        ))


def _compare_e16_initialization(target):
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils.torch_utils import init_seeds

    init_seeds(core.SEED + 1, deterministic=True)
    cfg = yaml_model_load(str(REFERENCE_E16_YAML))
    cfg["scale"] = "s"
    reference = DetectionModel(cfg, nc=10, verbose=False)
    load_p2_pretrained(reference)
    target_state, reference_state = target.state_dict(), reference.state_dict()
    missing = [key for key in reference_state if key not in target_state]
    unequal = [key for key, value in reference_state.items()
               if key in target_state and not core.torch.equal(target_state[key].cpu(), value.cpu())]
    if missing or unequal:
        raise RuntimeError(f"E16a common initialization differs: missing={missing[:5]}, unequal={unequal[:5]}")
    count = len(reference_state)
    del reference
    return dict(common_tensor_count=count, exact=True,
                note="All E16a tensors at deterministic initialization are identical; only SDQC predictor and gamma are additional.")


def _has_gradient(module):
    gradients = [parameter.grad for parameter in module.parameters()
                 if parameter.requires_grad and parameter.grad is not None]
    return bool(gradients) and all(core.torch.isfinite(value).all() for value in gradients)


def _two_step_loss_check(model):
    original_args = getattr(model, "args", None)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    batch = dict(
        img=core.torch.rand(2, 3, 128, 128),
        batch_idx=core.torch.tensor([0.0, 0.0, 1.0, 1.0]),
        cls=core.torch.tensor([[3.0], [0.0], [4.0], [8.0]]),
        bboxes=core.torch.tensor([[.30,.40,.20,.20],[.72,.68,.08,.09],[.60,.60,.30,.24],[.22,.25,.14,.12]]),
    )
    optimizer = core.torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.0)
    rows = []
    head = model.model[28]
    for step in range(2):
        model.train(); optimizer.zero_grad(set_to_none=True)
        total, items = model(batch)
        if not core.torch.isfinite(total) or len(items) != 3:
            raise RuntimeError(f"E20b invalid training loss at step {step+1}")
        total.backward()
        if head.raw_gain.grad is None or not core.torch.isfinite(head.raw_gain.grad).all():
            raise RuntimeError("E20b raw_gain gradient failed")
        if not _has_gradient(head.gate_net) or not _has_gradient(head.quality_predictor):
            raise RuntimeError("E20b TSDR gate or SDQC predictor gradient failed")
        if head.quality_gamma_logit.grad is None or not core.torch.isfinite(head.quality_gamma_logit.grad):
            raise RuntimeError("E20b gamma gradient failed")
        rows.append(dict(step=step+1, total=float(total.detach()),
                         diagnostics=dict(model.criterion.last_diagnostics)))
        optimizer.step()
    if not _has_gradient(head.detail_project):
        raise RuntimeError("E20b detail projector did not receive gradient after raw_gain update")
    model.eval()
    with core.torch.no_grad():
        preds = model(batch["img"])
        val_total, val_items = model.criterion(preds, batch)
    if not core.torch.isfinite(val_total) or len(val_items) != 3:
        raise RuntimeError("E20b validation-mode loss failed")
    if original_args is not None:
        model.args = original_args
    return dict(steps=rows, validation_loss=float(val_total), criterion=model.criterion.__class__.__name__)


def p2_preflight(report, eval_only=False):
    del eval_only
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils import yaml_load
    from ultralytics.utils.torch_utils import init_seeds

    required = (MODEL_YAML, REFERENCE_E16_YAML, REFERENCE_E1_YAML, E16_METRICS, MATCHING_HELPER,
                MODULE_SOURCE, LOSS_SOURCE, SDQC_SOURCE, SDQC_LOSS, DSSG_SOURCE, TSDR_SOURCE, TSDR_LOSS)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if Path(core.ultralytics.__file__).resolve().parent != (core.SOURCE / "ultralytics").resolve():
        raise RuntimeError("Wrong editable Ultralytics source")
    if core.sha256(core.PRETRAINED) != core.PRETRAINED_SHA256:
        raise RuntimeError("Original yolo11s.pt is missing or changed")
    cfg = yaml_model_load(str(MODEL_YAML))
    validate_p2_config(cfg, yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml"))
    out = Path(report["report_dir"])
    for path in required:
        shutil.copy2(path, out / path.name)
    report["p2_script_revision"] = REVISION
    report["p2_architecture"] = dict(
        yaml_sha256=core.sha256(MODEL_YAML), reference_e16_yaml_sha256=core.sha256(REFERENCE_E16_YAML),
        original_weights_sha256=core.sha256(core.PRETRAINED), module_sha256=core.sha256(MODULE_SOURCE),
        loss_sha256=core.sha256(LOSS_SOURCE), strides=[4,8,16,32],
        variant="strict E16a plus one detached-DFL SDQC calibrator",
        unchanged="E10a DSSG + E16a P1 high-pass/TSDR/tiny supervision + TAL2",
        added="shared 17->32->1 DFL quality predictor and bounded gamma only",
        claim_boundary="No E17/FusionReady, RCFA, DySample, PGALP, slicing or altered assignment",
        tal2=fast_tal.implementation_info(),
    )
    init_seeds(core.SEED + 1, deterministic=True)
    model = DetectionModel(cfg, nc=10, verbose=False)
    assert_p2_model(model)
    report["preflight_transfer"] = load_p2_pretrained(model)
    report["e16_initialization_equality"] = _compare_e16_initialization(model)
    model.eval()
    with core.torch.no_grad():
        prediction, raw = model(core.torch.zeros(1, 3, 256, 256))
    if [list(t.shape[-2:]) for t in raw] != [[64,64],[32,32],[16,16],[8,8]]:
        raise RuntimeError("E20b feature grid changed")
    if [int(t.shape[1]) for t in raw] != [75,75,75,75] or prediction.shape[1] != 14:
        raise RuntimeError("E20b raw/inference output contract changed")
    report["loss_mode_regression"] = _two_step_loss_check(model)
    report["p2_architecture"].update(parameters=sum(p.numel() for p in model.parameters()),
                                     initial_gamma=float(model.model[28].quality_gamma.detach()))
    core.save_report(report)
    del model, prediction, raw
    gc.collect()
    print("E20b topology, E16a initialization equality, TSDR+SDQC gradients and eval-loss passed.", flush=True)
    baseline = yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml")
    baseline.update(scale="s", nc=10)
    report["matching_equivalence"] = fast_tal.verify_equivalence(copy.deepcopy(cfg), device="cpu", baseline_cfg=baseline)
    fast_tal.assert_patch_restoration()
    core.write_json(out / "tal_equivalence.json", report["matching_equivalence"])
    core.save_report(report)
    print("TAL2 equivalence passed; run --smoke2 for CUDA batch8/1024 feasibility.", flush=True)


def train(report, smoke=False):
    weights = _CORE_TRAIN(report, smoke=smoke)
    run_dir = Path(report["run_dir"])
    for path in (MODULE_SOURCE, LOSS_SOURCE, SDQC_SOURCE, SDQC_LOSS, DSSG_SOURCE, TSDR_SOURCE, TSDR_LOSS,
                REFERENCE_E16_YAML, REFERENCE_E1_YAML, E16_METRICS):
        shutil.copy2(path, run_dir / path.name)
    return weights


def evaluate(report, weights, dataset):
    _CORE_EVALUATE(report, weights, dataset)
    if not E16_METRICS.is_file():
        report["comparison_to_e16a"] = {"status": "unavailable", "expected": str(E16_METRICS)}
        core.save_report(report)
        return
    reference = json.loads(E16_METRICS.read_text(encoding="utf-8"))
    current = {**report.get("overall", {}), **report.get("area_metrics", {})}
    baseline = {**reference.get("overall", {}), **reference.get("area_metrics", {})}
    keys = ("Precision", "Recall", "mAP50", "mAP75", "mAP50_95", "AP_small", "AP_medium", "AP_large")
    delta = {key: current[key] - baseline[key] for key in keys if key in current and key in baseline}
    report["comparison_to_e16a"] = dict(
        status="computed", reference=str(E16_METRICS), delta=delta,
        success_requirements=dict(mAP50_95=0.004, mAP75=0.004, AP_small=0.004,
                                  AP_medium_min=0.0, AP_large_min=-0.005),
        passed=(delta.get("mAP50_95", -1) >= 0.004 and delta.get("mAP75", -1) >= 0.004
                and delta.get("AP_small", -1) >= 0.004 and delta.get("AP_medium", -1) >= 0.0
                and delta.get("AP_large", -1) >= -0.005),
    )
    core.save_report(report)


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
    core.evaluate = evaluate


if __name__ == "__main__":
    install_overrides()
    core.main()
