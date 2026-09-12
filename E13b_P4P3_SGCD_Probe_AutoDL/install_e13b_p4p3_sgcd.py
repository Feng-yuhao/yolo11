#!/usr/bin/env python3
"""Install and verify E13b without touching existing experiment artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import py_compile
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
PROJECT = Path("/root/autodl-tmp/yolo11_project")
SOURCE = Path("/root/autodl-tmp/yolo11_src")
MODULES = SOURCE / "ultralytics/nn/modules"
TASKS = SOURCE / "ultralytics/nn/tasks.py"
INIT = MODULES / "__init__.py"
RECORDS = PROJECT / "environment_records"
E10A_BEST = PROJECT / "runs/e10a_yolo11s_p2_dssg_img1024_seed1/weights/best.pt"
E10A_SHA256 = "8c92316c138ea01bdbe3eacf3d275c9be15d96231c13c65818811bcc3cd2d512"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def insert_after(path, anchors, insertion, already):
    text = path.read_text(encoding="utf-8")
    if already in text:
        return False
    for anchor in anchors:
        if text.count(anchor) == 1:
            path.write_text(text.replace(anchor, anchor + insertion, 1), encoding="utf-8", newline="\n")
            return True
    raise RuntimeError(f"No unique registration anchor found in {path}")


def register_parser(path):
    text = path.read_text(encoding="utf-8")
    if "elif m in {Detect, P4P3SGCDDetect," in text:
        return False
    anchor = "elif m in {Detect,"
    if text.count(anchor) != 1:
        raise RuntimeError(f"Expected one Detect parser anchor in {path}, found {text.count(anchor)}")
    path.write_text(
        text.replace(anchor, "elif m in {Detect, P4P3SGCDDetect,", 1),
        encoding="utf-8",
        newline="\n",
    )
    return True


def main():
    required = (
        PROJECT / "yolo11s_p2_img1024_seed1.py",
        PROJECT / "yolo11s_p2_dssg_img1024_seed1.py",
        PROJECT / "yolo11s-p2-dssg.yaml",
        PROJECT / "p2_tal_chunked_vgpu32.py",
        MODULES / "dssg.py",
        TASKS,
        INIT,
        E10A_BEST,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    actual = sha256(E10A_BEST)
    if actual != E10A_SHA256:
        raise RuntimeError(f"Frozen E10a best.pt changed: expected={E10A_SHA256}, actual={actual}")

    copies = {
        PACKAGE / "yolo11_src/ultralytics/nn/modules/p4p3_sgcd_head.py": MODULES / "p4p3_sgcd_head.py",
        PACKAGE / "yolo11_project/yolo11s-p2-p4p3-sgcd-probe.yaml": PROJECT / "yolo11s-p2-p4p3-sgcd-probe.yaml",
        PACKAGE / "yolo11_project/e13b_single_common.py": PROJECT / "e13b_single_common.py",
        PACKAGE / "yolo11_project/yolo11s_p2_p4p3_sgcd_probe_img1024_seed1.py": PROJECT / "yolo11s_p2_p4p3_sgcd_probe_img1024_seed1.py",
        PACKAGE / "yolo11_project/run_e13b_p4p3_sgcd_probe.sh": PROJECT / "run_e13b_p4p3_sgcd_probe.sh",
    }
    for source, destination in copies.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for destination in copies.values():
        if destination.suffix == ".sh":
            destination.chmod(0o755)

    changes = {}
    changes["init_import"] = insert_after(
        INIT,
        (
            "from .sgcd_head import ClassificationSemanticAdapter, SGCDDetect\n",
            "from .dssg import DeepSemanticGuide, SemanticDetailGate\n",
        ),
        "from .p4p3_sgcd_head import P4P3ClassificationSemanticAdapter, P4P3SGCDDetect\n",
        "from .p4p3_sgcd_head import P4P3ClassificationSemanticAdapter, P4P3SGCDDetect",
    )
    changes["init_export"] = insert_after(
        INIT,
        ('    "SGCDDetect",\n', '    "SemanticDetailGate",\n'),
        '    "P4P3ClassificationSemanticAdapter",\n    "P4P3SGCDDetect",\n',
        '    "P4P3SGCDDetect",',
    )
    changes["tasks_import"] = insert_after(
        TASKS,
        ("    SGCDDetect,\n", "    SemanticDetailGate,\n"),
        "    P4P3SGCDDetect,\n",
        "    P4P3SGCDDetect,",
    )
    changes["tasks_parser"] = register_parser(TASKS)

    for path in (*copies.values(), INIT, TASKS):
        if path.suffix == ".py":
            py_compile.compile(str(path), doraise=True)

    check_code = f'''import torch, ultralytics
from pathlib import Path
from types import SimpleNamespace
source_root = Path({str(SOURCE)!r}).resolve()
if Path(ultralytics.__file__).resolve().parent != source_root / "ultralytics":
    raise RuntimeError(f"wrong ultralytics source: {{ultralytics.__file__}}")
if ultralytics.__version__ != "8.3.0":
    raise RuntimeError(f"wrong ultralytics version: {{ultralytics.__version__}}")
from ultralytics import YOLO
from ultralytics.nn.modules import P4P3SGCDDetect
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
cfg = yaml_model_load({str(PROJECT / 'yolo11s-p2-p4p3-sgcd-probe.yaml')!r})
cfg["scale"] = "s"
target = DetectionModel(cfg, nc=10, verbose=False)
target.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
detect = target.model[28]
if not isinstance(detect, P4P3SGCDDetect):
    raise RuntimeError("P4P3SGCDDetect registration failed")
if hasattr(detect, "class_adapters"):
    raise RuntimeError("E13b unexpectedly constructed the old two-adapter head")
source = YOLO({str(E10A_BEST)!r}).model.float()
source_state, target_state = source.state_dict(), target.state_dict()
mapping = {{key: value for key, value in source_state.items() if key in target_state}}
if len(mapping) != len(source_state):
    missing_source = sorted(set(source_state) - set(mapping))
    raise RuntimeError(f"not every E10a tensor maps into E13b: {{missing_source[:5]}}")
result = target.load_state_dict(mapping, strict=False)
expected_missing = {{key for key in target_state if key.startswith("model.28.semantic_adapter.")}}
if set(result.missing_keys) != expected_missing or result.unexpected_keys:
    raise RuntimeError(f"bad E10a transfer: {{result}}")
source.eval(); target.eval(); torch.manual_seed(1313)
sample = torch.randn(1,3,256,256)
with torch.inference_mode():
    source_pred, source_raw = source(sample)
    target_pred, target_raw = target(sample)
if not torch.equal(source_pred, target_pred) or any(not torch.equal(a,b) for a,b in zip(source_raw,target_raw)):
    raise RuntimeError("zero-gain E13b is not exactly prediction-identical to E10a")
with torch.no_grad():
    detect.semantic_adapter.gain.fill_(0.01)
with torch.inference_mode():
    _, opened_raw = target(sample)
reg = detect.reg_max * 4
for index, (reference, opened) in enumerate(zip(source_raw, opened_raw)):
    if not torch.equal(reference[:, :reg], opened[:, :reg]):
        raise RuntimeError(f"regression logits changed at level {{index}}")
    cls_delta = float((reference[:, reg:] - opened[:, reg:]).abs().max())
    if (index == 1 and cls_delta <= 0.0) or (index != 1 and cls_delta != 0.0):
        raise RuntimeError(f"classification isolation failed at level {{index}}: {{cls_delta}}")
with torch.no_grad():
    detect.semantic_adapter.gain.zero_()
target.train()
for name, parameter in target.named_parameters():
    parameter.requires_grad = (
        name.startswith("model.28.cv3.0.") or name.startswith("model.28.cv3.1.")
        or name.startswith("model.28.semantic_adapter.")
    )
batch = {{
    "img": torch.rand(2,3,256,256),
    "batch_idx": torch.tensor([0,1]),
    "cls": torch.tensor([[0.0],[1.0]]),
    "bboxes": torch.tensor([[0.3,0.3,0.04,0.04],[0.7,0.6,0.08,0.06]]),
}}
total, items = target(batch)
if items.numel() != 3 or not torch.isfinite(total):
    raise RuntimeError(f"bad standard loss: {{items}}")
total.backward()
gain_grad = float(detect.semantic_adapter.gain.grad.abs().sum())
if gain_grad <= 0:
    raise RuntimeError("P4->P3 residual gain received no gradient")
if any(p.requires_grad for name,p in target.named_parameters() if ".cv2." in name):
    raise RuntimeError("regression path is unexpectedly trainable")
print("E13b build, E10a transfer, exact equivalence, P3-only classification isolation and gradient checks passed")
'''
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["OMP_NUM_THREADS"] = "8"
    completed = subprocess.run(
        [sys.executable, "-c", check_code],
        text=True,
        capture_output=True,
        env=environment,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("E13b installation check failed:\n" + completed.stdout + completed.stderr)
    print(completed.stdout.strip())

    RECORDS.mkdir(parents=True, exist_ok=True)
    record_path = RECORDS / "e13b_p4p3_sgcd_registration.json"
    record = {
        "status": "installed_and_checked",
        "time": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "source": str(SOURCE),
        "project": str(PROJECT),
        "experimental_base": "E1 YOLO11s+P2",
        "probe_initialization": {"name": "E10a candidate", "path": str(E10A_BEST), "sha256": E10A_SHA256},
        "paired_control": "existing c13_control_e10a_clsft30_seed1; not rerun",
        "single_change": "one P4->P3 classification semantic adapter; no new P3->P2 adapter",
        "changes": changes,
        "files": {str(path): sha256(path) for path in copies.values()},
        "existing_reports_removed": False,
        "existing_checkpoints_modified": False,
        "automatic_shutdown": False,
        "next": "check-only -> smoke2 -> 30-epoch E13b probe",
    }
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Installation record:", record_path)
    print("No existing source module, experiment report or checkpoint was deleted.")
    print("Next: check-only -> smoke2 -> 30-epoch E13b. Automatic shutdown is disabled.")


if __name__ == "__main__":
    main()
