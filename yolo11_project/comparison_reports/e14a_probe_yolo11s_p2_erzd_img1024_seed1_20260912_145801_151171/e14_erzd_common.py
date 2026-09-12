"""Shared, paired E14a ER-ZD and C14 continuation-control experiment code."""
from __future__ import annotations

import copy
import gc
import json
import os
import shutil
import sys
import types
from pathlib import Path


HERE = Path(__file__).resolve().parent
if not os.environ.get("OMP_NUM_THREADS", "").isdigit() or int(os.environ.get("OMP_NUM_THREADS", "0")) < 1:
    os.environ["OMP_NUM_THREADS"] = "8"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch

import p2_tal_chunked_vgpu32 as fast_tal

sys.modules["p2_tal_chunked"] = fast_tal
import yolo11s_p2_img1024_seed1 as core


E1_BEST = core.ROOT / "runs/e1_yolo11s_p2add_img1024_seed1/weights/best.pt"
E1_BEST_SHA256 = "d1bbacf76cb838f8f441ad08fc56bf1d7fe377a57d28dcffc6e00c886af7c371"
E1_YAML = HERE / "yolo11s-p2-add.yaml"
E14A_YAML = HERE / "yolo11s-p2-erzd-probe.yaml"
ERZD_SOURCE = core.SOURCE / "ultralytics/utils/erzd_loss.py"
MATCHING_HELPER = Path(fast_tal.__file__).resolve()
COMMON_SOURCE = Path(__file__).resolve()
PROBE_EPOCHS = 30

_BASE_TRAIN = core.train
_BASE_EVALUATE = core.evaluate
_BASE_TRAINING_PARAMS = core.training_params


def settings(variant: str):
    if variant == "erzd":
        return {
            "experiment": "e14a_probe_yolo11s_p2_erzd_img1024_seed1",
            "revision": "e1_error_aware_regional_zoom_distillation_probe30_v1",
            "yaml": E14A_YAML,
            "gate": core.ROOT / "comparison_reports/e14a_erzd_probe_smoke_passed.json",
        }
    if variant == "control":
        return {
            "experiment": "c14_control_e1_continue30_seed1",
            "revision": "e1_plain_matched_continue30_control_v1",
            "yaml": E1_YAML,
            "gate": core.ROOT / "comparison_reports/c14_e1_continue30_smoke_passed.json",
        }
    raise ValueError(f"Unknown E14 variant: {variant}")


def _yaml_config(path: Path):
    from ultralytics.nn.tasks import yaml_model_load

    return yaml_model_load(str(path))


def check_model(model, variant: str):
    if len(model.model) != 27 or model.yaml.get("scale") != "s":
        raise ValueError("E14 must keep the 27-layer E1 s-scale graph")
    detect = model.model[26]
    if detect.__class__.__name__ != "Detect" or detect.f != [25, 16, 19, 22] or detect.nc != 10:
        raise ValueError("E14 changed the E1 P2/P3/P4/P5 Detect graph")
    if [float(x) for x in model.stride.detach().cpu().tolist()] != [4.0, 8.0, 16.0, 32.0]:
        raise ValueError(f"Unexpected strides: {model.stride}")
    for index in (11, 14, 23):
        layer = model.model[index]
        if layer.__class__.__name__ != "Upsample" or layer.mode != "nearest" or layer.scale_factor != 2.0:
            raise ValueError(f"Layer {index} must remain nearest-neighbour 2x upsampling")
    forbidden = {
        "DySample", "P3SDE", "PGALP", "DeepSemanticGuide", "SemanticDetailGate",
        "SGCDDetect", "P4P3SGCDDetect", "LocalDiscriminativeSemanticGuide",
    }
    present = {module.__class__.__name__ for module in model.modules()} & forbidden
    if present:
        raise ValueError(f"E14 unexpectedly contains a retired experimental module: {sorted(present)}")
    erzd = model.yaml.get("erzd", {})
    enabled = isinstance(erzd, dict) and erzd.get("enabled", False)
    if enabled != (variant == "erzd"):
        raise ValueError(f"ER-ZD criterion flag does not match variant={variant}")


def validate_config(config: dict, variant: str):
    from ultralytics.utils import yaml_load

    reference = yaml_load(E1_YAML)
    if config.get("backbone") != reference.get("backbone") or config.get("head") != reference.get("head"):
        raise ValueError("E14 architecture differs from the frozen E1 YAML")
    if config.get("nc") != 10 or config.get("scales") != reference.get("scales"):
        raise ValueError("E14 changed classes or compound scaling")
    erzd = config.get("erzd")
    if variant == "control" and erzd is not None:
        raise ValueError("C14 must use the unmodified E1 YAML without ER-ZD")
    if variant == "erzd":
        expected = {
            "enabled": True,
            "crop_fraction": 0.5625,
            "crops_per_batch": 2,
            "tiny_side_px": 32.0,
            "max_targets_per_crop": 48,
            "anchor_candidates": 16,
            "levels": [0, 1],
            "teacher_patch_radius": [3, 2],
            "student_patch_radius": [1, 1],
            "max_loss_weight": 0.25,
            "ramp_epochs": 5,
            "min_teacher_conf": 0.08,
            "teacher_margin": 0.03,
            "wrong_class_weight": 0.5,
        }
        if erzd != expected:
            raise ValueError(f"E14a ER-ZD configuration changed: {erzd}")


def load_e1_state(target, variant: str):
    from ultralytics import YOLO

    if not E1_BEST.is_file() or core.sha256(E1_BEST) != E1_BEST_SHA256:
        raise RuntimeError("Frozen E1 best.pt is missing or changed")
    source = YOLO(str(E1_BEST)).model.float()
    check_model(source, "control")
    source_state = source.state_dict()
    target_state = target.state_dict()
    key_difference = sorted(set(source_state) ^ set(target_state))
    if key_difference:
        raise RuntimeError(f"E14 and E1 state keys differ: {key_difference[:5]}")
    mismatched = [
        key for key in source_state if tuple(source_state[key].shape) != tuple(target_state[key].shape)
    ]
    if mismatched:
        raise RuntimeError(f"E14 and E1 state shapes differ: {mismatched[:5]}")
    result = target.load_state_dict(source_state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Strict E1 transfer failed: {result}")
    actual = target.state_dict()
    unequal = [key for key in source_state if not torch.equal(source_state[key].cpu(), actual[key].cpu())]
    if unequal:
        raise RuntimeError(f"E1 tensor equality failed: {unequal[:5]}")
    audit = {
        "source": str(E1_BEST),
        "source_sha256": E1_BEST_SHA256,
        "loaded_tensors": len(source_state),
        "verified_tensor_equality": True,
        "new_model_parameters": 0,
        "initialization": "exact frozen E1 best.pt; ER-ZD has no inference parameters",
    }
    del source, source_state, target_state, actual
    gc.collect()
    return audit


def prediction_equivalence(target):
    from ultralytics import YOLO

    source = YOLO(str(E1_BEST)).model.float().eval()
    target.eval()
    torch.manual_seed(1414)
    sample = torch.randn(1, 3, 256, 256)
    with torch.inference_mode():
        source_pred, source_raw = source(sample)
        target_pred, target_raw = target(sample)
    pred_max = float((source_pred.float() - target_pred.float()).abs().max())
    raw_max = max(float((a.float() - b.float()).abs().max()) for a, b in zip(source_raw, target_raw))
    if pred_max != 0.0 or raw_max != 0.0:
        raise RuntimeError(f"E14 does not initialize prediction-identically to E1: {pred_max}, {raw_max}")
    del source, sample, source_pred, source_raw, target_pred, target_raw
    gc.collect()
    return {"prediction_max_abs_delta": pred_max, "raw_output_max_abs_delta": raw_max}


def _synthetic_batch(size=256):
    return {
        "img": torch.rand(2, 3, size, size),
        "batch_idx": torch.tensor([0, 0, 1, 1]),
        "cls": torch.tensor([[0.0], [3.0], [1.0], [9.0]]),
        "bboxes": torch.tensor(
            [
                [0.30, 0.35, 0.030, 0.025],
                [0.68, 0.62, 0.110, 0.090],
                [0.42, 0.55, 0.028, 0.032],
                [0.72, 0.30, 0.060, 0.045],
            ]
        ),
    }


def gradient_and_builder_check(model, variant: str):
    model.args = types.SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    model.train()
    batch = _synthetic_batch()
    builder_result = None
    if variant == "erzd":
        from ultralytics import YOLO
        from ultralytics.utils.erzd_loss import RegionalZoomTargetBuilder

        teacher = YOLO(str(E1_BEST)).model.float().eval()
        builder = RegionalZoomTargetBuilder(model.yaml["erzd"])
        builder_payload = builder(teacher, batch)
        if not builder_payload["entries"] or builder.last_stats.get("selected_crops", 0) < 1:
            raise RuntimeError(f"ER-ZD crop/target builder produced no target: {builder.last_stats}")
        builder_result = dict(builder.last_stats)
        del teacher, builder_payload

        high = torch.full((10,), 0.01)
        high[0] = 0.99
        batch["_erzd"] = {
            "loss_weight": 0.25,
            "builder_stats": {"synthetic": True},
            "entries": [
                {
                    "image_index": 0,
                    "class_id": 0,
                    "full_xy": (0.30, 0.35),
                    "side_px": 8.0,
                    "teacher_probabilities": [high.clone(), high.clone()],
                }
            ],
        }
    total, items = model(batch)
    if items.numel() != 3 or not torch.isfinite(total):
        raise RuntimeError(f"E14 loss check failed: total={total}, items={items}")
    total.backward()
    gradient = sum(float(p.grad.detach().abs().sum()) for p in model.parameters() if p.grad is not None)
    if gradient <= 0:
        raise RuntimeError("E14 model received no gradient")
    criterion_stats = {}
    if variant == "erzd":
        criterion_stats = dict(getattr(model.criterion, "last_stats", {}))
        if criterion_stats.get("active_pairs", 0) < 1 or criterion_stats.get("scaled_loss", 0.0) <= 0:
            raise RuntimeError(f"ER-ZD auxiliary gradient path is inactive: {criterion_stats}")
    model.zero_grad(set_to_none=True)
    return {
        "loss_items": [float(x) for x in items],
        "gradient_abs_sum": gradient,
        "builder": builder_result,
        "criterion": criterion_stats,
    }


def training_params(variant: str):
    params = _BASE_TRAINING_PARAMS()
    params.update(
        epochs=PROBE_EPOCHS,
        name=settings(variant)["experiment"],
        optimizer="SGD",
        lr0=0.001,
        lrf=0.1,
        warmup_epochs=1.0,
        close_mosaic=5,
        pretrained=False,
        freeze=0,
    )
    return params


def make_probe_trainer(report: dict, variant: str):
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.erzd_loss import RegionalZoomTargetBuilder

    class E14PairedTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if weights is not None:
                raise RuntimeError("E14 performs its own exact E1 checkpoint transfer")
            model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose)
            check_model(model, variant)
            report["pretrained_transfer"] = load_e1_state(model, variant)
            core.write_json(Path(report["report_dir"]) / "pretrained_transfer.json", report["pretrained_transfer"])
            core.save_report(report)
            return model

        def _ensure_erzd_teacher(self):
            if getattr(self, "_erzd_teacher", None) is None:
                teacher = YOLO(str(E1_BEST)).model.float().to(self.device).eval()
                check_model(teacher, "control")
                for parameter in teacher.parameters():
                    parameter.requires_grad_(False)
                self._erzd_teacher = teacher
                self._erzd_builder = RegionalZoomTargetBuilder(self.model.yaml["erzd"])
                self._erzd_totals = {
                    "batches": 0,
                    "selected_crops": 0,
                    "eligible_targets": 0,
                    "criterion_active_pairs": 0,
                }
                report["erzd_training"] = {
                    "teacher": str(E1_BEST),
                    "teacher_sha256": E1_BEST_SHA256,
                    "teacher_frozen": True,
                    "student_initialization": "same exact E1 best.pt",
                    "inference_change": "none",
                    "totals": dict(self._erzd_totals),
                }

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            if variant != "erzd":
                return batch
            self._ensure_erzd_teacher()
            previous = getattr(getattr(self.model, "criterion", None), "last_stats", None)
            if isinstance(previous, dict):
                self._erzd_totals["criterion_active_pairs"] += int(previous.get("active_pairs", 0))
            payload = self._erzd_builder(self._erzd_teacher, batch)
            config = self.model.yaml["erzd"]
            ramp_epochs = max(1, int(config["ramp_epochs"]))
            payload["loss_weight"] = float(config["max_loss_weight"]) * min(1.0, (int(self.epoch) + 1) / ramp_epochs)
            batch["_erzd"] = payload
            stats = payload["builder_stats"]
            self._erzd_totals["batches"] += 1
            self._erzd_totals["selected_crops"] += int(stats.get("selected_crops", 0))
            self._erzd_totals["eligible_targets"] += int(stats.get("eligible_targets", 0))
            report["erzd_training"]["current_epoch"] = int(self.epoch) + 1
            report["erzd_training"]["current_loss_weight"] = payload["loss_weight"]
            report["erzd_training"]["last_builder_stats"] = stats
            report["erzd_training"]["totals"] = dict(self._erzd_totals)
            return batch

    return E14PairedTrainer


def _completed_reference(pattern: str):
    candidates = sorted(
        (
            path
            for path in (core.ROOT / "comparison_reports").glob(pattern)
            if "SMOKE" not in str(path).upper()
        ),
        key=lambda path: path.stat().st_mtime,
    )
    for path in reversed(candidates):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("status") == "completed":
            return path, data
    return None


def _metric_deltas(current: dict, reference: dict):
    sections = {
        "overall": ("Precision", "Recall", "mAP50", "mAP75", "mAP50_95"),
        "area_metrics": ("AP_all", "AR_all", "AP_small", "AR_small", "AP_medium", "AP_large"),
    }
    return {
        section: {
            name: float(current[section][name]) - float(reference[section][name])
            for name in names
        }
        for section, names in sections.items()
    }


def p2_preflight(report: dict, variant: str):
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import yaml_load

    cfg_path = settings(variant)["yaml"]
    required = [cfg_path, E1_YAML, E1_BEST, MATCHING_HELPER, COMMON_SOURCE, ERZD_SOURCE]
    for path in required:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if core.sha256(E1_BEST) != E1_BEST_SHA256:
        raise RuntimeError("Frozen E1 checkpoint differs from the audited file")
    cfg = _yaml_config(cfg_path)
    validate_config(cfg, variant)
    model = DetectionModel(cfg, nc=10, verbose=False)
    check_model(model, variant)
    report.update(
        is_formal=False,
        is_probe=True,
        p2_script_revision=settings(variant)["revision"],
        p2_architecture={
            "experimental_hierarchy": {
                "E0": "original YOLO11s baseline",
                "E1": "YOLO11s + P2 frozen scientific base",
                "E14a": "E1 plus training-only ER-ZD; inference graph unchanged",
            },
            "variant": variant,
            "graph": "exact E1 27-layer P2/P3/P4/P5 Detect",
            "initialization": "exact same frozen E1 best.pt for E14a and C14",
            "trainable_scope": (
                "all E1 student parameters plus a frozen E1 zoom teacher"
                if variant == "erzd"
                else "all E1 student parameters; no teacher"
            ),
            "single_change": (
                "teacher-superior P2/P3 class-response distillation from 1.78x regional zoom views"
                if variant == "erzd"
                else "none; matched ordinary 30-epoch continuation control"
            ),
            "not_changed": [
                "inference architecture", "nearest upsampling", "box/DFL loss", "TAL2",
                "imgsz=1024", "batch=8", "dataset", "paired fine-tuning schedule and augmentations",
            ],
            "yaml": str(cfg_path),
            "yaml_sha256": core.sha256(cfg_path),
            "common_sha256": core.sha256(COMMON_SOURCE),
            "erzd_source_sha256": core.sha256(ERZD_SOURCE) if variant == "erzd" else None,
            "automatic_shutdown": False,
        },
    )
    report["preflight_transfer"] = load_e1_state(model, variant)
    report["initial_prediction_equivalence"] = prediction_equivalence(model)
    report["gradient_and_builder_check"] = gradient_and_builder_check(model, variant)
    out = Path(report["report_dir"])
    for path in required:
        shutil.copy2(path, out / Path(path).name)
    core.save_report(report)
    del model
    gc.collect()

    baseline_cfg = yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml")
    baseline_cfg.update(scale="s", nc=10)
    print("Checking strict TAL2 equivalence on CPU; CUDA batch8/1024 is checked by --smoke2.", flush=True)
    report["matching_equivalence"] = fast_tal.verify_equivalence(
        copy.deepcopy(cfg), device="cpu", baseline_cfg=baseline_cfg
    )
    fast_tal.assert_patch_restoration()
    core.write_json(out / "tal_equivalence.json", report["matching_equivalence"])
    core.save_report(report)
    print(f"{variant} E1-transfer/graph/loss-gradient/TAL2 checks passed.", flush=True)


def train(report: dict, smoke: bool, variant: str):
    weights = _BASE_TRAIN(report, smoke=smoke)
    run_dir = Path(report["run_dir"])
    for path in (COMMON_SOURCE, MATCHING_HELPER, E1_YAML):
        shutil.copy2(path, run_dir / Path(path).name)
    if variant == "erzd":
        for path in (E14A_YAML, ERZD_SOURCE):
            shutil.copy2(path, run_dir / Path(path).name)
    return weights


def evaluate(report: dict, weights: Path, dataset, variant: str):
    _BASE_EVALUATE(report, weights, dataset)
    references = {"E1_frozen_base": "e1_yolo11s_p2add_img1024_seed1_*/metrics.json"}
    if variant == "erzd":
        references["C14_matched_continue30"] = "c14_control_e1_continue30_seed1_*/metrics.json"
    comparisons = {}
    for label, pattern in references.items():
        found = _completed_reference(pattern)
        if found is None:
            comparisons[label] = {"status": "unavailable", "pattern": pattern}
            continue
        path, values = found
        comparisons[label] = {
            "status": "computed",
            "path": str(path),
            "deltas": _metric_deltas(report, values),
        }
    report["probe_comparisons"] = comparisons
    if variant == "erzd":
        paired = comparisons.get("C14_matched_continue30", {})
        if paired.get("status") == "computed":
            delta = paired["deltas"]
            observed = {
                "mAP50_95": delta["overall"]["mAP50_95"],
                "AP_small": delta["area_metrics"]["AP_small"],
                "AP_medium": delta["area_metrics"]["AP_medium"],
            }
            report["paired_screen"] = {
                "status": "passed" if (
                    observed["mAP50_95"] >= 0.002
                    and observed["AP_small"] >= 0.005
                    and observed["AP_medium"] >= -0.005
                ) else "failed",
                "required": {"mAP50_95": 0.002, "AP_small": 0.005, "AP_medium_floor": -0.005},
                "observed": observed,
                "comparison": "E14a minus same-checkpoint, same-optimizer, same-30-epoch C14",
            }
        else:
            report["paired_screen"] = {
                "status": "pending_control",
                "decision": "Run C14 only if E14a shows a useful direct signal; do not claim a causal gain before C14.",
            }
    core.save_report(report)


def run(variant: str, runner_file: str):
    cfg = settings(variant)
    core.EXPERIMENT = cfg["experiment"]
    core.SCRIPT_REVISION = cfg["revision"]
    core.MODEL_YAML = cfg["yaml"]
    core.PRETRAINED = E1_BEST
    core.PRETRAINED_SHA256 = E1_BEST_SHA256
    core.MATCHING_HELPER = MATCHING_HELPER
    core.SMOKE_GATE = cfg["gate"]
    core.EPOCHS = PROBE_EPOCHS
    core.AUTO_SHUTDOWN = False
    core.SHUTDOWN_ON_FAILURE = False
    core.__file__ = str(Path(runner_file).resolve())
    core.training_params = lambda: training_params(variant)
    core.make_p2_trainer = lambda report: make_probe_trainer(report, variant)
    core.load_p2_pretrained = lambda target: load_e1_state(target, variant)
    core.assert_p2_model = lambda model: check_model(model, variant)
    core.validate_p2_config = lambda config, _original: validate_config(config, variant)
    core.p2_preflight = lambda report, eval_only=False: p2_preflight(report, variant)
    core.train = lambda report, smoke=False: train(report, smoke, variant)
    core.evaluate = lambda report, weights, dataset: evaluate(report, weights, dataset, variant)
    core.main()
