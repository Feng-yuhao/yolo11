#!/usr/bin/env python3
"""E35b: E1 local Bottlenecks + zero-start PKI deltas, seed1, 200 epochs.

--check-only -> --smoke2 -> formal (no mode flag). Automatic shutdown disabled.
Not a complete PKINet reproduction or a claim of architectural originality.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "e35b_runtime"
if not (RUNTIME / "install_receipt.json").is_file():
    raise RuntimeError("Run install_e35b_local_pki.py first. No shared files will be patched.")
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(RUNTIME / "source"))

import torch
import e1_640_core as core
from e35b_checks import PREFIXES, LEAVES, assert_architecture, validate_config, gradient_audit, serialization_audit
from e35b_checks import original_to_target, is_new_key, initial_equivalence_audit

EXPERIMENT = "e35b_local_pki_e1_img640_seed1"
REVISION = "e35b_local_pki_640_seed1_v1"
CFG = RUNTIME / "yolo11s-p2-local-pki.yaml"
E1_CFG = RUNTIME / "e1_reference.yaml"
MODULE = RUNTIME / "source/ultralytics/nn/modules/e35b_local_pki.py"
_BASE_LOAD = core.load_p2_pretrained
_BASE_ASSERT = core.assert_p2_model
_BASE_SAVE = core.save_report
_BASE_TRAIN = core.train
_LAST_REPORT = None


def normalized_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def reference_e1():
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils import RANK
    from ultralytics.utils.torch_utils import init_seeds
    if RANK != -1:
        raise RuntimeError("This audited package is single-process/single-GPU, not DDP")
    # BaseTrainer uses seed + 1 + RANK. With RANK=-1 this is SEED, not SEED+1.
    init_seeds(core.SEED + 1 + RANK, deterministic=True)
    cfg = yaml_model_load(str(E1_CFG))
    cfg["scale"] = "s"
    reference = DetectionModel(copy.deepcopy(cfg), nc=10, verbose=False)
    _BASE_ASSERT(reference)
    transfer = _BASE_LOAD(reference)
    return reference, transfer


def load_pretrained(target):
    from ultralytics.nn.modules.e35b_local_pki import E35LocalPKI
    reference, audit = reference_e1()
    source = reference.state_dict()
    destination = target.state_dict()
    shared = {original_to_target(k): v for k, v in source.items()}
    actual_shared = {k for k in destination if not is_new_key(k)}
    if set(shared) != actual_shared:
        raise RuntimeError(f"Unaccounted shared tensors: {set(shared) ^ actual_shared}")
    for name, value in shared.items():
        if destination[name].shape != value.shape:
            raise RuntimeError(f"Common tensor shape changed: {name}")
    # Load a COMPLETE target dictionary: do not waive all missing_keys. This also
    # avoids PyTorch's version-dependent handling of BN num_batches_tracked.
    merged = dict(destination)
    merged.update(shared)
    target.load_state_dict(merged, strict=True)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(35000 + core.SEED)
        for name in LEAVES:
            block = target.get_submodule(name)
            if type(block) is not E35LocalPKI:
                raise RuntimeError(f"Wrong replacement: {name}")
            block.reset_new_parameters()
    if any(not torch.equal(target.state_dict()[k].cpu(), v.cpu()) for k, v in shared.items()):
        raise RuntimeError("Common E1 tensors differ after transfer")
    result = dict(reference_seed=core.SEED, trainer_initialization_seed=core.SEED,
                  source="original yolo11s.pt -> deterministic E1 initialization; NOT E1 best.pt",
                  original_e1_transfer=audit, shared_tensor_count=len(shared),
                  common_tensors_exact=True, initial_prediction_equal_to_e1=True,
                  initial_equivalence_pending_preflight=True,
                  preserved_local_prefixes=list(PREFIXES),
                  discarded_e1_keys=[],
                  new_pki_keys=sorted(k for k in destination if is_new_key(k)),
                  new_pki_rng_seed=35000 + core.SEED,
                  rng_policy="new block reset in fork_rng; restores post-E1-transfer torch RNG")
    del reference
    return result


def save_report(report):
    global _LAST_REPORT
    _LAST_REPORT = report
    _BASE_SAVE(report)


def train(report, smoke=False):
    if not smoke and (HERE / "runs" / EXPERIMENT).exists():
        raise RuntimeError("E35b formal run directory already exists. Refusing accidental duplicate/restart. "
                           "Inspect the existing run; use --eval-only for a completed checkpoint.")
    return _BASE_TRAIN(report, smoke=smoke)


def make_trainer(report):
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.nn.tasks import DetectionModel

    class E35bTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if weights is not None:
                raise RuntimeError("No resume/fine-tuning in this from-original-weights experiment")
            model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose)
            assert_architecture(model)
            report["pretrained_transfer"] = load_pretrained(model)
            core.write_json(Path(report["report_dir"]) / "pretrained_transfer.json", report["pretrained_transfer"])
            save_report(report)
            return model

        def build_optimizer(self, model, *args, **kwargs):
            optimizer = super().build_optimizer(model, *args, **kwargs)
            expected = {id(p) for n, p in model.named_parameters() if is_new_key(n)}
            actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
            if not expected or any(actual.count(p) != 1 for p in expected):
                raise RuntimeError("PKI parameters missing/duplicated in the REAL training optimizer")
            report["pki_optimizer_audit"] = dict(passed=True, parameter_tensors=len(expected))
            save_report(report)
            return optimizer

    return E35bTrainer


def verify_snapshot():
    receipt = json.loads((RUNTIME / "install_receipt.json").read_text(encoding="utf-8"))
    for relative, expected in receipt["runtime_hashes"].items():
        if normalized_hash(RUNTIME / relative) != expected:
            raise RuntimeError(f"Frozen runtime changed: {relative}. Do not bypass the smoke gate.")
    manifest = json.loads((RUNTIME / "package_manifest.json").read_text(encoding="utf-8"))
    for name in ("yolo11s_p2_local_pki_img640_seed1.py", "evaluate_e35b_640_official.py", "run_e35b_local_pki_seed1.sh"):
        if normalized_hash(HERE / name) != manifest["package_hashes"]["payload/" + name]:
            raise RuntimeError(f"Installed entry changed: {name}")
    return receipt


def preflight(report, eval_only=False):
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils import yaml_load
    from ultralytics.utils.torch_utils import get_flops
    from p2_tal_chunked import verify_equivalence, assert_patch_restoration, implementation_info
    receipt = verify_snapshot()
    if Path(core.ultralytics.__file__).resolve().parent != (core.SOURCE / "ultralytics").resolve():
        raise RuntimeError("Wrong Ultralytics import; expected private e35b_runtime/source")
    if core.sha256(core.PRETRAINED) != core.PRETRAINED_SHA256:
        raise RuntimeError("Original yolo11s.pt missing/changed")
    cfg = yaml_model_load(str(CFG))
    cfg["scale"] = "s"
    validate_config(cfg, yaml_load(E1_CFG))
    out = Path(report["report_dir"])
    for source in (CFG, E1_CFG, MODULE, RUNTIME / "source/ultralytics/nn/modules/e35a_pki.py",
                   RUNTIME / "e1_640_core.py", RUNTIME / "e35b_checks.py",
                   RUNTIME / "install_receipt.json", RUNTIME / "package_manifest.json", core.MATCHING_HELPER):
        shutil.copy2(source, out / source.name)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = DetectionModel(copy.deepcopy(cfg), nc=10, verbose=False)
    architecture = assert_architecture(model)
    report["preflight_transfer"] = load_pretrained(model)
    reference, _ = reference_e1()
    report["initial_e1_equivalence"] = initial_equivalence_audit(model, reference)
    report["pki_gradient_audit"] = gradient_audit(model, reference)
    report["pki_serialization_audit"] = serialization_audit(model, out)
    model.eval()
    reference.eval()
    base_params = sum(p.numel() for p in reference.parameters())
    new_params = sum(p.numel() for p in model.parameters())
    base_flops, new_flops = get_flops(reference, imgsz=640), get_flops(model, imgsz=640)
    if not (base_flops > 0 and new_flops > 0):
        raise RuntimeError("FLOPs profiling failed; install baseline's ultralytics-thop, do not assume zero cost")
    if new_params > base_params * 1.15 or new_flops > base_flops * 1.15:
        raise RuntimeError(f"PKI exceeds fixed +15% budget: params {new_params}/{base_params}, FLOPs {new_flops}/{base_flops}")
    report["p2_script_revision"] = REVISION
    report["p2_architecture"] = dict(
        **architecture, variant="E1 original local Bottlenecks + zero-init 1x1 projection of unchanged PKI delta",
        yaml_sha256=normalized_hash(CFG), module_sha256=normalized_hash(MODULE),
        pki_core_sha256=normalized_hash(RUNTIME / "source/ultralytics/nn/modules/e35a_pki.py"),
        frozen_core_sha256=normalized_hash(RUNTIME / "e1_640_core.py"),
        snapshot_receipt_sha256=normalized_hash(RUNTIME / "install_receipt.json"),
        source_code_sha256={name: normalized_hash(core.SOURCE / "ultralytics" / name)
                           for name in ("nn/tasks.py", "nn/modules/block.py", "nn/modules/head.py",
                                        "utils/loss.py", "utils/tal.py", "engine/trainer.py")},
        preflight_parameters=new_params, e1_parameters=base_params,
        preflight_GFLOPs_640=new_flops, e1_GFLOPs_640=base_flops,
        original_loss_and_detect=True, matching_implementation=implementation_info(),
        shared_source_was_modified=receipt["shared_source_was_modified"],
        initialization_note="ALL E1 tensors exact, including preserved local Bottlenecks. Initial outputs/loss/gradients audited against E1.")
    del model, reference
    gc.collect()
    original = yaml_load(core.SOURCE / "ultralytics/cfg/models/11/yolo11.yaml")
    original.update(scale="s", nc=10)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    report["matching_equivalence"] = verify_equivalence(copy.deepcopy(cfg), device=device, baseline_cfg=original)
    assert_patch_restoration()
    core.write_json(out / "tal_equivalence.json", report["matching_equivalence"])
    save_report(report)
    core.clear_gpu()
    print("E35 structure, exact shared transfer, PKI gradients/updates, checkpoint/fuse and TAL checks passed.", flush=True)


def configure():
    core.ROOT = HERE
    core.SOURCE = RUNTIME / "source"
    core.PRETRAINED = HERE / "weights/yolo11s.pt"
    core.MODEL_YAML = CFG
    core.MATCHING_HELPER = RUNTIME / "p2_tal_chunked.py"
    core.EXPERIMENT = EXPERIMENT
    core.SCRIPT_REVISION = REVISION
    core.SEED = 1
    core.SMOKE_GATE = HERE / "comparison_reports/e35b_local_pki_640_seed1_smoke_passed.json"
    core.AUTO_SHUTDOWN = core.SHUTDOWN_ON_FAILURE = False
    core.__file__ = str(Path(__file__).resolve())
    core.assert_p2_model = assert_architecture
    core.load_p2_pretrained = load_pretrained
    core.p2_preflight = preflight
    core.make_p2_trainer = make_trainer
    core.train = train
    core.save_report = save_report
    manifest = json.loads((RUNTIME / "package_manifest.json").read_text(encoding="utf-8"))
    params = core.training_params()
    ignored = {"data", "project", "name"}
    expected = {k: v for k, v in manifest["baseline_train_params"].items() if k not in ignored}
    actual = {k: v for k, v in params.items() if k not in ignored}
    if expected != actual:
        raise RuntimeError(f"Training protocol differs from E1-640 seed1: {expected.items() ^ actual.items()}")


if __name__ == "__main__":
    configure()
    core.main()
    if _LAST_REPORT and _LAST_REPORT.get("status") == "completed" and _LAST_REPORT.get("is_formal"):
        print("Training/evaluation complete. Scoring saved predictions with frozen D27 (no retraining).", flush=True)
        subprocess.run([sys.executable, str(HERE / "evaluate_e35b_640_official.py"),
                        "--candidate-report", _LAST_REPORT["report_dir"]], check=True)
