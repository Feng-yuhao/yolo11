#!/usr/bin/env python3
"""E17a full 200-epoch E1 pre-fusion adapter plus training-only auxiliary supervision."""

from __future__ import annotations

import copy
import gc
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if not os.environ.get("OMP_NUM_THREADS", "").isdigit() or int(os.environ.get("OMP_NUM_THREADS", "0")) < 1:
    os.environ["OMP_NUM_THREADS"] = "8"
sys.path.insert(0, str(HERE))
import p2_tal_chunked_vgpu32 as fast_tal
sys.modules["p2_tal_chunked"] = fast_tal
import yolo11s_p2_img1024_seed1 as core

EXPERIMENT = "e17a_yolo11s_p2_fusion_ready_img1024_seed1"
REVISION = "e17a_prefusion_adapter_aux_seed1_200e_tal2_v1"
MODEL_YAML = HERE / "yolo11s-p2-fusion-ready.yaml"
REFERENCE_YAML = HERE / "yolo11s-p2-add.yaml"
MODULE_SOURCE = core.SOURCE / "ultralytics/nn/modules/e17_fusion.py"
LOSS_SOURCE = core.SOURCE / "ultralytics/utils/e17_fusion_loss.py"
MATCHING_HELPER = Path(fast_tal.__file__).resolve()
SMOKE_GATE = core.ROOT / "comparison_reports/e17a_fusion_ready_smoke_passed.json"
_E1_LOAD = core.load_p2_pretrained
_E1_ASSERT = core.assert_p2_model
_E1_TRAIN = core.train

E1_TO_E17 = {**{i: i for i in range(11)}, 11: 11, 12: 13, 13: 14,
             14: 15, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21,
             20: 22, 21: 23, 22: 24, 23: 25, 24: 26, 25: 27, 26: 28}


def _target_key(key):
    parts = key.split(".")
    if len(parts) < 3 or parts[0] != "model" or not parts[1].isdigit():
        raise ValueError(f"Unrecognized E1 state tensor: {key}")
    parts[1] = str(E1_TO_E17[int(parts[1])])
    return ".".join(parts)


def _reference():
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils.torch_utils import init_seeds

    init_seeds(core.SEED + 1, deterministic=True)
    cfg = yaml_model_load(str(REFERENCE_YAML))
    cfg["scale"] = "s"
    reference = DetectionModel(cfg, nc=10, verbose=False)
    _E1_ASSERT(reference)
    original_audit = _E1_LOAD(reference)
    cpu_rng = core.torch.random.get_rng_state()
    cuda_rng = core.torch.cuda.get_rng_state_all() if core.torch.cuda.is_available() else None
    return reference, original_audit, cpu_rng, cuda_rng


def load_p2_pretrained(target):
    reference, original_audit, cpu_rng, cuda_rng = _reference()
    target_state = target.state_dict()
    mapped = {}
    for key, value in reference.state_dict().items():
        destination = _target_key(key)
        if destination not in target_state or value.shape != target_state[destination].shape:
            raise ValueError(f"E1 semantic transfer mismatch: {key} -> {destination}")
        if destination in mapped:
            raise ValueError(f"Duplicate transfer target: {destination}")
        mapped[destination] = value
    missing = sorted(set(target_state) - set(mapped))
    if any(not key.startswith(("model.12.", "model.16.", "model.28.aux_head.")) for key in missing):
        raise ValueError(f"Unexpected uninitialized target tensors: {missing}")
    result = target.load_state_dict(mapped, strict=False)
    if sorted(result.missing_keys) != missing or result.unexpected_keys:
        raise ValueError("Incomplete/incorrect E1 state migration")
    core.torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        core.torch.cuda.set_rng_state_all(cuda_rng)
    if any(not core.torch.equal(target.state_dict()[key].cpu(), value.cpu()) for key, value in mapped.items()):
        raise RuntimeError("Transferred tensors are not exactly equal to E1 initialization")
    del reference
    return dict(
        source="original 80-class yolo11s.pt transferred into deterministic E1, then E1 tensor initialization mapped into E17a",
        source_audit=original_audit, loaded_tensor_count=len(mapped),
        new_tensor_keys=missing, exact_tensor_equality=True,
        reference_e1_yaml_sha256=core.sha256(REFERENCE_YAML),
        note="No best.pt is used as parent; only adapters and shared auxiliary head are new."
    )


def assert_p2_model(model):
    if len(model.model) != 29 or model.yaml.get("scale") != "s":
        raise ValueError("Expected E17a 29-layer YOLO11s topology")
    head = model.model[28]
    if head.__class__.__name__ != "E17Detect" or head.f != [27, 18, 21, 24, 12, 16] or head.nc != 10:
        raise ValueError("E17a must detect from frozen four PAN heads plus two training-only lateral sources")
    if list(model.stride.cpu().tolist()) != [4., 8., 16., 32.]:
        raise ValueError("E17a four detection strides must remain 4/8/16/32")
    for index, inputs in ((12, [11, 6]), (16, [15, 4])):
        adapter = model.model[index]
        if adapter.__class__.__name__ != "FusionReadyAdapter" or adapter.f != inputs:
            raise ValueError(f"Incorrect pre-fusion adapter at layer {index}")
    for index in (11, 15, 25):
        layer = model.model[index]
        if layer.__class__.__name__ != "Upsample" or layer.mode != "nearest" or layer.scale_factor != 2.0:
            raise ValueError(f"Upsample {index} changed from frozen nearest")
    forbidden = {layer.__class__.__name__ for layer in model.model} & {"DySample", "PGALP", "P3SDE", "P4P3SpatialGuide"}
    if forbidden:
        raise ValueError(f"E17a includes unplanned modules: {forbidden}")


def validate_p2_config(cfg, original_cfg):
    del original_cfg
    from ultralytics.utils import yaml_load

    e1 = yaml_load(REFERENCE_YAML)
    expected = []
    for old_index, (source, repeats, module, arguments) in enumerate(e1["head"], start=11):
        if old_index == 12:
            expected.extend((
                [[11, 6], 1, "FusionReadyAdapter", [32, 0.05]],
                [[11, 12], 1, "Concat", [1]],
            ))
            continue
        if old_index == 15:
            expected.extend((
                [[15, 4], 1, "FusionReadyAdapter", [32, 0.05]],
                [[15, 16], 1, "Concat", [1]],
            ))
            continue
        if old_index == 26:
            expected.append([[27, 18, 21, 24, 12, 16], 1, "E17Detect", ["nc"]])
            continue
        remap = lambda x: E1_TO_E17[x] if isinstance(x, int) and x >= 11 else x
        moved = [remap(x) for x in source] if isinstance(source, list) else remap(source)
        expected.append([moved, repeats, module, arguments])
    if core.canonical_layers(cfg["backbone"]) != core.canonical_layers(e1["backbone"]):
        raise ValueError("E17a backbone differs from E1")
    if core.canonical_layers(cfg["head"]) != core.canonical_layers(expected):
        raise ValueError("E17a head is not exactly the two-adapter E1 topology")
    if cfg.get("scale") != "s" or cfg.get("nc") != 10 or cfg["scales"]["s"] != e1["scales"]["s"]:
        raise ValueError("E17a scale or class count differs from E1")
    if not cfg.get("e17_aux", {}).get("enabled"):
        raise ValueError("E17a training-only auxiliary supervision is disabled")


def p2_preflight(report, eval_only=False):
    del eval_only
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils import yaml_load
    from ultralytics.utils.torch_utils import init_seeds

    for path in (MODEL_YAML, REFERENCE_YAML, MODULE_SOURCE, LOSS_SOURCE, MATCHING_HELPER):
        if not path.is_file():
            raise FileNotFoundError(path)
    if Path(core.ultralytics.__file__).resolve().parent != (core.SOURCE / "ultralytics").resolve():
        raise RuntimeError("Wrong editable Ultralytics source; activate yolo11 conda environment")
    if core.sha256(core.PRETRAINED) != core.PRETRAINED_SHA256:
        raise RuntimeError("Original yolo11s.pt changed")
    cfg = yaml_model_load(str(MODEL_YAML))
    validate_p2_config(cfg, yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml"))
    out = Path(report["report_dir"])
    for path in (MODEL_YAML, REFERENCE_YAML, MODULE_SOURCE, LOSS_SOURCE, MATCHING_HELPER):
        shutil.copy2(path, out / path.name)
    report["p2_script_revision"] = REVISION
    report["p2_architecture"] = dict(
        yaml_sha256=core.sha256(MODEL_YAML), original_weights_sha256=core.sha256(core.PRETRAINED),
        module_sha256=core.sha256(MODULE_SOURCE), loss_sha256=core.sha256(LOSS_SOURCE),
        tasks_sha256=core.sha256(core.SOURCE / "ultralytics/nn/tasks.py"),
        variant="E1 plus shared near-identity pre-Concat lateral adapters P4/P3 and training-only scale-valid aux supervision",
        detect_strides=[4, 8, 16, 32], upsampling="nearest unchanged",
        tal2=fast_tal.implementation_info(),
        auxiliary_gain=cfg["e17_aux"]["gain"],
    )
    init_seeds(core.SEED, deterministic=True)
    model = DetectionModel(cfg, nc=10, verbose=False)
    assert_p2_model(model)
    report["preflight_transfer"] = load_p2_pretrained(model)
    for index in (12, 16):
        if abs(float(model.model[index].layer_scale.detach()) - 0.05) > 1e-7:
            raise ValueError("Adapter residual initialization changed")
    model.eval()
    with core.torch.inference_mode():
        output, raw = model(core.torch.zeros(1, 3, 256, 256))
    sizes = [list(tensor.shape[-2:]) for tensor in raw]
    if sizes != [[64, 64], [32, 32], [16, 16], [8, 8]] or not core.torch.isfinite(output).all():
        raise RuntimeError(f"Incorrect/nonfinite E17a inference features: {sizes}")
    report["p2_architecture"]["cpu_feature_grids"] = sizes
    report["p2_architecture"]["parameters_including_training_only_aux"] = sum(p.numel() for p in model.parameters())
    core.save_report(report)
    del model, raw, output
    gc.collect()
    print("E17a structure/E1 exact initialization/inference checks passed.", flush=True)

    original_cfg = yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml")
    original_cfg.update(scale="s", nc=10)
    report["matching_equivalence"] = fast_tal.verify_equivalence(copy.deepcopy(cfg), device="cpu", baseline_cfg=original_cfg)
    fast_tal.assert_patch_restoration()
    core.write_json(out / "tal_equivalence.json", report["matching_equivalence"])
    core.save_report(report)
    print("TAL2 equivalence and real auxiliary loss/gradient checks passed; run --smoke2 for CUDA batch8/1024.", flush=True)


def train(report, smoke=False):
    weights = _E1_TRAIN(report, smoke=smoke)
    run_dir = Path(report["run_dir"])
    for path in (MODULE_SOURCE, LOSS_SOURCE, REFERENCE_YAML):
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
