#!/usr/bin/env python3
"""E9c: 30-epoch assignment-guided PGALP correction probe.

Starts from the frozen E9b best checkpoint, freezes every layer except PGALP
layer 26, and adds a reliability loss built from the current P2 TAL positive
mask.  The inference graph is unchanged from E9b.  No mode shuts down AutoDL.
"""

from __future__ import annotations

import copy
import gc
import json
import os
import shutil
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
if not os.environ.get("OMP_NUM_THREADS", "").isdigit() or int(os.environ.get("OMP_NUM_THREADS", "0")) < 1:
    os.environ["OMP_NUM_THREADS"] = "8"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import p2_tal_chunked_vgpu32 as fast_tal
sys.modules["p2_tal_chunked"] = fast_tal
import yolo11s_p2_img1024_seed1 as core


EXPERIMENT = "e9c_probe_yolo11s_p2_agpgalp_img1024_seed1"
SCRIPT_REVISION = "agpgalp_tal_positive_reliability_probe_v1"
MODEL_YAML = HERE / "yolo11s-p2-agpgalp.yaml"
E9B_BEST = core.ROOT / "runs/e9b_yolo11s_p2_pgalp_img1024_seed1/weights/best.pt"
E9B_METRICS = core.ROOT / "runs/e9b_yolo11s_p2_pgalp_img1024_seed1/complete_metrics.json"
E1_METRICS = core.ROOT / "runs/e1_yolo11s_p2add_img1024_seed1/complete_metrics.json"
MODULE_SOURCE = core.SOURCE / "ultralytics/nn/modules/ag_pgalp.py"
LOSS_SOURCE = core.SOURCE / "ultralytics/utils/ag_pgalp_loss.py"
MATCHING_HELPER = Path(fast_tal.__file__).resolve()
SMOKE_GATE = core.ROOT / "comparison_reports/e9c_agpgalp_probe_smoke_passed.json"
PROBE_EPOCHS = 30
TRAINABLE_LAYER = 26
FROZEN_LAYERS = list(range(26)) + [27]

_CORE_TRAIN = core.train
_CORE_EVALUATE = core.evaluate


def architecture_record():
    return dict(
        yaml=str(MODEL_YAML),
        yaml_sha256=core.sha256(MODEL_YAML),
        reference="frozen E9b best.pt",
        reference_checkpoint=str(E9B_BEST),
        reference_checkpoint_sha256=core.sha256(E9B_BEST),
        variant=(
            "E9c probe = E9b inference graph plus training-only reliability supervision "
            "from the current 3x3-dilated P2 TAL-positive mask."
        ),
        strides=[4, 8, 16, 32],
        trainable_layer=TRAINABLE_LAYER,
        frozen_layers=FROZEN_LAYERS,
        filters=dict(kernels=[3, 5, 7], sigmas=[0.8, 1.2, 1.8]),
        reliability_loss=dict(weight=0.05, focal_gamma=2.0, dilation=3),
        inference_graph_change_vs_e9b=False,
        tal2=fast_tal.implementation_info(),
    )

def assert_p2_model(model):
    if len(model.model) != 28 or model.yaml.get("scale") != "s":
        raise ValueError("Expected 28-layer s-scale E9c architecture")
    if list(model.stride.cpu().tolist()) != [4.0, 8.0, 16.0, 32.0]:
        raise ValueError(f"Wrong strides: {model.stride}")
    if model.model[-1].f != [26, 16, 19, 22] or model.model[-1].nc != 10:
        raise ValueError("Wrong P2/P3/P4/P5 Detect inputs")
    for index in (11, 14, 23):
        layer = model.model[index]
        if layer.__class__.__name__ != "Upsample" or layer.mode != "nearest" or layer.scale_factor != 2.0:
            raise ValueError(f"Layer {index} must remain nearest-neighbor 2x")
    module = model.model[TRAINABLE_LAYER]
    if module.__class__.__name__ != "AGPGALP" or module.f != [25, 16]:
        raise ValueError("Layer 26 must be AGPGALP fed by fused P2 and raw P3")
    if tuple(module.kernels) != (3, 5, 7) or tuple(module.sigmas) != (0.8, 1.2, 1.8):
        raise ValueError("E9b filter bank changed unexpectedly")
    if (module.p2_channels, module.p3_channels, module.hidden) != (128, 128, 32):
        raise ValueError("AGPGALP channels differ from E9b")


def validate_p2_config(cfg, _original_cfg):
    from ultralytics.utils import yaml_load

    e9b_cfg = yaml_load(HERE / "yolo11s-p2-pgalp.yaml")
    expected = copy.deepcopy(e9b_cfg)
    expected["head"][-2][2] = "AGPGALP"
    expected["scale"] = "s"
    expected["ag_pgalp"] = dict(enabled=True, loss_weight=0.05, focal_gamma=2.0, dilation=3)
    for key in ("backbone", "head", "scale", "nc", "scales", "ag_pgalp"):
        actual = core.canonical_layers(cfg[key]) if key in ("backbone", "head") else cfg[key]
        wanted = core.canonical_layers(expected[key]) if key in ("backbone", "head") else expected[key]
        if actual != wanted:
            raise ValueError(f"E9c config mismatch at {key}: expected={wanted!r}, actual={actual!r}")


def load_e9b_state(target):
    if not E9B_BEST.is_file():
        raise FileNotFoundError(E9B_BEST)
    source = core.YOLO(str(E9B_BEST)).model.float()
    if source.model[26].__class__.__name__ != "PGALP":
        raise ValueError("E9b checkpoint does not contain PGALP at layer 26")
    source_state = source.state_dict()
    result = target.load_state_dict(source_state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Strict E9b transfer failed: {result}")
    target_state = target.state_dict()
    unequal = [k for k, v in source_state.items() if not core.torch.equal(v.cpu(), target_state[k].cpu())]
    if unequal:
        raise RuntimeError(f"E9b tensor equality failed: {unequal[:3]}")
    audit = dict(
        source=str(E9B_BEST), source_sha256=core.sha256(E9B_BEST),
        loaded_tensors=len(source_state), strict=True, verified_tensor_equality=True,
        note="AGPGALP has the same persistent state and inference equation as E9b PGALP.",
    )
    del source, source_state, target_state
    return audit


def load_p2_pretrained(target):
    return load_e9b_state(target)


def training_params():
    params = core._E9C_ORIGINAL_TRAINING_PARAMS()
    params.update(
        epochs=PROBE_EPOCHS,
        name=EXPERIMENT,
        freeze=FROZEN_LAYERS,
        lr0=0.002,
        lrf=0.1,
        warmup_epochs=1.0,
        close_mosaic=5,
    )
    return params


def make_probe_trainer(report):
    from copy import copy as shallow_copy
    from ultralytics.models import yolo
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.nn.tasks import DetectionModel

    class ProbeTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if weights is not None:
                raise RuntimeError("E9c explicitly transfers the frozen E9b best checkpoint")
            model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose)
            assert_p2_model(model)
            report["pretrained_transfer"] = load_e9b_state(model)
            core.write_json(Path(report["report_dir"]) / "pretrained_transfer.json", report["pretrained_transfer"])
            core.save_report(report)
            return model

        def get_validator(self):
            self.loss_names = "box_loss", "cls_loss", "dfl_loss", "reliability_loss"
            return yolo.detect.DetectionValidator(
                self.test_loader, save_dir=self.save_dir, args=shallow_copy(self.args), _callbacks=self.callbacks
            )

        def preprocess_batch(self, batch):
            # Freeze means parameters only in Ultralytics. Also lock BatchNorm
            # running statistics outside layer 26 so the probe changes no hidden
            # state elsewhere in the E9b checkpoint.
            import torch.nn as nn

            batch = super().preprocess_batch(batch)
            for index in FROZEN_LAYERS:
                for submodule in self.model.model[index].modules():
                    if isinstance(submodule, nn.BatchNorm2d):
                        submodule.eval()
            return batch

    return ProbeTrainer


def _loss_gradient_check(model):
    model.train()
    batch = {
        "img": core.torch.zeros(2, 3, 256, 256),
        "batch_idx": core.torch.tensor([0, 1]),
        "cls": core.torch.tensor([[0.0], [3.0]]),
        "bboxes": core.torch.tensor([[0.25, 0.25, 0.04, 0.04], [0.70, 0.65, 0.06, 0.05]]),
    }
    total, items = model(batch)
    if items.numel() != 4 or not core.torch.isfinite(total):
        raise RuntimeError(f"Expected four finite loss items, got {items}")
    total.backward()
    module = model.model[26]
    gradients = [p.grad for p in module.parameters() if p.requires_grad]
    if not gradients or any(g is None or not core.torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("AGPGALP loss/backward check failed")
    stats = model.criterion.last_reliability_stats
    result = {k: float(v) for k, v in stats.items()}
    result["loss_items"] = [float(x) for x in items]
    model.zero_grad(set_to_none=True)
    module.clear_reliability_logits()
    return result


def p2_preflight(report, eval_only=False):
    from ultralytics.nn.modules import AGPGALP  # noqa: F401
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils import yaml_load

    print(f"E9c AGPGALP probe revision: {SCRIPT_REVISION}", flush=True)
    required = (MODEL_YAML, HERE / "yolo11s-p2-pgalp.yaml", E9B_BEST, MODULE_SOURCE, LOSS_SOURCE, MATCHING_HELPER)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    cfg = yaml_model_load(str(MODEL_YAML))
    validate_p2_config(cfg, yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml"))
    model = DetectionModel(cfg, nc=10, verbose=False)
    assert_p2_model(model)
    report["is_formal"] = False
    report["is_probe"] = True
    report["p2_architecture"] = architecture_record()
    report["e9c_probe"] = dict(
        is_probe=True, epochs=PROBE_EPOCHS, start_checkpoint=str(E9B_BEST),
        trainable_layer=TRAINABLE_LAYER, frozen_layers=FROZEN_LAYERS,
        unique_change="3x3-dilated current P2 TAL-positive reliability supervision",
        inference_graph_change_vs_e9b=False, loss_weight=0.05, focal_gamma=2.0,
        automatic_shutdown=False,
    )
    report["preflight_transfer"] = load_e9b_state(model)
    report["loss_gradient_check"] = _loss_gradient_check(model)
    model.eval()
    with core.torch.inference_mode():
        _, raw = model(core.torch.zeros(1, 3, 256, 256))
    sizes = [list(x.shape[-2:]) for x in raw]
    if sizes != [[64, 64], [32, 32], [16, 16], [8, 8]]:
        raise RuntimeError(f"Unexpected feature shapes: {sizes}")
    report["CPU_forward_256_feature_shapes"] = sizes
    out = Path(report["report_dir"])
    for path in required:
        shutil.copy2(path, out / path.name)
    core.save_report(report)
    del model, raw
    gc.collect()

    baseline_cfg = yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml")
    baseline_cfg.update(scale="s", nc=10)
    report["matching_equivalence"] = fast_tal.verify_equivalence(copy.deepcopy(cfg), device="cpu", baseline_cfg=baseline_cfg)
    fast_tal.assert_patch_restoration()
    core.write_json(out / "tal_equivalence.json", report["matching_equivalence"])
    core.save_report(report)
    print("E9c structure/E9b-transfer/loss-gradient/TAL2 checks passed.", flush=True)


def train(report, smoke=False):
    weights = _CORE_TRAIN(report, smoke=smoke)
    run_dir = Path(report["run_dir"])
    for path in (MODULE_SOURCE, LOSS_SOURCE, MATCHING_HELPER):
        shutil.copy2(path, run_dir / path.name)
    report["e9c_probe"]["actual_trainable_parameters"] = sum(
        p.numel() for p in core.YOLO(str(weights)).model.model[26].parameters()
    )
    core.save_report(report)
    return weights


def _load_metrics(path):
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if value.get("status") == "completed" else None


def evaluate(report, weights, dataset):
    _CORE_EVALUATE(report, weights, dataset)
    comparisons = {}
    for label, path in (("E9b", E9B_METRICS), ("E1", E1_METRICS)):
        reference = _load_metrics(path)
        if reference is None:
            comparisons[label] = {"status": "unavailable", "path": str(path)}
            continue
        comparisons[label] = {
            "status": "computed", "path": str(path),
            "mAP50_95_delta": float(report["overall"]["mAP50_95"]) - float(reference["overall"]["mAP50_95"]),
            "AP_small_delta": float(report["area_metrics"]["AP_small"]) - float(reference["area_metrics"]["AP_small"]),
            "AP_medium_delta": float(report["area_metrics"]["AP_medium"]) - float(reference["area_metrics"]["AP_medium"]),
        }
    report["comparison_to_frozen"] = comparisons
    trained = core.YOLO(str(weights)).model.float()
    assert_p2_model(trained)
    module = trained.model[26]
    report["agpgalp_learned_state"] = {
        "layer_scale_mean": float(module.layer_scale.detach().mean()),
        "router_output_bias": [float(x) for x in module.router[-1].bias.detach()],
        "next_decision": "Run diagnose_e9c_agpgalp.py; retain only if routing direction and AP-small improve.",
    }
    del trained


def install_overrides():
    core._E9C_ORIGINAL_TRAINING_PARAMS = core.training_params
    core.EXPERIMENT = EXPERIMENT
    core.SCRIPT_REVISION = SCRIPT_REVISION
    core.MODEL_YAML = MODEL_YAML
    core.MATCHING_HELPER = MATCHING_HELPER
    core.SMOKE_GATE = SMOKE_GATE
    core.EPOCHS = PROBE_EPOCHS
    core.AUTO_SHUTDOWN = False
    core.SHUTDOWN_ON_FAILURE = False
    core.__file__ = str(Path(__file__).resolve())
    core.training_params = training_params
    core.make_p2_trainer = make_probe_trainer
    core.load_p2_pretrained = load_p2_pretrained
    core.assert_p2_model = assert_p2_model
    core.validate_p2_config = validate_p2_config
    core.p2_preflight = p2_preflight
    core.train = train
    core.evaluate = evaluate


if __name__ == "__main__":
    install_overrides()
    core.main()




