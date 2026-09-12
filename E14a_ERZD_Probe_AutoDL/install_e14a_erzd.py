#!/usr/bin/env python3
"""Install E14a ER-ZD files and verify the loss without replacing old experiments."""
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
TASKS = SOURCE / "ultralytics/nn/tasks.py"
UTILS = SOURCE / "ultralytics/utils"
RECORDS = PROJECT / "environment_records"
E1_BEST = PROJECT / "runs/e1_yolo11s_p2add_img1024_seed1/weights/best.pt"
E1_SHA256 = "d1bbacf76cb838f8f441ad08fc56bf1d7fe377a57d28dcffc6e00c886af7c371"
E1_YAML = PROJECT / "yolo11s-p2-add.yaml"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patch_tasks(text: str):
    changed = {"erzd_import": False, "criterion_route": False}
    import_line = "from ultralytics.utils.erzd_loss import v8ERZDDetectionLoss\n"
    if import_line not in text:
        anchors = (
            "from ultralytics.utils.tiny_prior_loss import v8TinyPriorDetectionLoss\n",
            "from ultralytics.utils.ops import make_divisible\n",
        )
        for anchor in anchors:
            if text.count(anchor) == 1:
                text = text.replace(anchor, anchor + import_line, 1)
                changed["erzd_import"] = True
                break
        else:
            raise RuntimeError("No unique tasks.py import anchor for ER-ZD")

    detection_start = text.find("class DetectionModel(")
    detection_end = text.find("\nclass OBBModel(", detection_start)
    if detection_start < 0 or detection_end < 0:
        raise RuntimeError("Could not isolate DetectionModel in tasks.py")
    section = text[detection_start:detection_end]
    route = (
        "        erzd_config = self.yaml.get(\"erzd\", {})\n"
        "        if isinstance(erzd_config, dict) and erzd_config.get(\"enabled\", False):\n"
        "            return v8ERZDDetectionLoss(self)\n"
    )
    if route not in section:
        anchor = "        tiny_prior = self.yaml.get(\"tiny_prior\", {})\n"
        if section.count(anchor) != 1:
            raise RuntimeError("No unique DetectionModel criterion anchor for ER-ZD")
        section = section.replace(anchor, route + anchor, 1)
        text = text[:detection_start] + section + text[detection_end:]
        changed["criterion_route"] = True
    return text, changed


def main():
    required = (
        PROJECT / "yolo11s_p2_img1024_seed1.py",
        PROJECT / "p2_tal_chunked_vgpu32.py",
        E1_YAML,
        E1_BEST,
        TASKS,
        PACKAGE / "yolo11_src/ultralytics/utils/erzd_loss.py",
        PACKAGE / "yolo11_project/e14_erzd_common.py",
        PACKAGE / "yolo11_project/yolo11s-p2-erzd-probe.yaml",
        PACKAGE / "yolo11_project/yolo11s_p2_erzd_probe_img1024_seed1.py",
        PACKAGE / "yolo11_project/c14_control_e1_continue30_seed1.py",
        PACKAGE / "yolo11_project/run_e14a_erzd_probe.sh",
        PACKAGE / "yolo11_project/run_c14_e1_continue30.sh",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    actual_e1 = sha256(E1_BEST)
    if actual_e1 != E1_SHA256:
        raise RuntimeError(f"Frozen E1 best.pt changed: expected={E1_SHA256}, actual={actual_e1}")

    copies = {
        PACKAGE / "yolo11_src/ultralytics/utils/erzd_loss.py": UTILS / "erzd_loss.py",
        PACKAGE / "yolo11_project/e14_erzd_common.py": PROJECT / "e14_erzd_common.py",
        PACKAGE / "yolo11_project/yolo11s-p2-erzd-probe.yaml": PROJECT / "yolo11s-p2-erzd-probe.yaml",
        PACKAGE / "yolo11_project/yolo11s_p2_erzd_probe_img1024_seed1.py": PROJECT / "yolo11s_p2_erzd_probe_img1024_seed1.py",
        PACKAGE / "yolo11_project/c14_control_e1_continue30_seed1.py": PROJECT / "c14_control_e1_continue30_seed1.py",
        PACKAGE / "yolo11_project/run_e14a_erzd_probe.sh": PROJECT / "run_e14a_erzd_probe.sh",
        PACKAGE / "yolo11_project/run_c14_e1_continue30.sh": PROJECT / "run_c14_e1_continue30.sh",
    }
    for source, destination in copies.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if destination.suffix == ".sh":
            destination.chmod(0o755)

    original_tasks = TASKS.read_text(encoding="utf-8")
    before_hash = sha256(TASKS)
    patched_tasks, changes = patch_tasks(original_tasks)
    if patched_tasks != original_tasks:
        TASKS.write_text(patched_tasks, encoding="utf-8", newline="\n")

    try:
        for path in (*copies.values(), TASKS):
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
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils.erzd_loss import v8ERZDDetectionLoss
cfg = yaml_model_load({str(PROJECT / 'yolo11s-p2-erzd-probe.yaml')!r})
cfg["scale"] = "s"
target = DetectionModel(cfg, nc=10, verbose=False)
source = YOLO({str(E1_BEST)!r}).model.float().eval()
if len(target.model) != 27 or target.model[-1].f != [25,16,19,22]:
    raise RuntimeError("E14a graph is not the exact E1 P2 graph")
if set(source.state_dict()) != set(target.state_dict()):
    raise RuntimeError("E14a introduced inference state tensors")
target.load_state_dict(source.state_dict(), strict=True)
target.eval(); torch.manual_seed(1414)
sample = torch.randn(1,3,256,256)
with torch.inference_mode():
    a, ar = source(sample)
    b, br = target(sample)
if not torch.equal(a,b) or any(not torch.equal(x,y) for x,y in zip(ar,br)):
    raise RuntimeError("E14a is not prediction-identical to E1 at initialization")
target.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
target.train()
teacher = torch.full((10,), 0.01); teacher[0] = 0.99
batch = {{
    "img": torch.rand(2,3,256,256),
    "batch_idx": torch.tensor([0,1]),
    "cls": torch.tensor([[0.0],[1.0]]),
    "bboxes": torch.tensor([[0.3,0.3,0.03,0.03],[0.7,0.6,0.07,0.06]]),
    "_erzd": {{
        "loss_weight": 0.25,
        "builder_stats": {{"installer_synthetic": True}},
        "entries": [{{
            "image_index": 0,
            "class_id": 0,
            "full_xy": (0.3,0.3),
            "side_px": 8.0,
            "teacher_probabilities": [teacher.clone(), teacher.clone()],
        }}],
    }},
}}
total, items = target(batch)
if items.numel() != 3 or not torch.isfinite(total):
    raise RuntimeError(f"bad E14a loss: {{total}}, {{items}}")
if not isinstance(target.criterion, v8ERZDDetectionLoss):
    raise RuntimeError("E14a did not select v8ERZDDetectionLoss")
total.backward()
stats = target.criterion.last_stats
if stats.get("active_pairs",0) < 1 or stats.get("scaled_loss",0.0) <= 0:
    raise RuntimeError(f"ER-ZD gradient route inactive: {{stats}}")
print("E14a build, exact E1 transfer, inference equivalence, loss and backward checks passed")
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
            raise RuntimeError("E14a installation check failed:\n" + completed.stdout + completed.stderr)
        print(completed.stdout.strip())
    except Exception:
        if TASKS.read_text(encoding="utf-8") != original_tasks:
            TASKS.write_text(original_tasks, encoding="utf-8", newline="\n")
        raise

    RECORDS.mkdir(parents=True, exist_ok=True)
    record_path = RECORDS / "e14a_erzd_registration.json"
    record = {
        "status": "installed_and_checked",
        "time": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "source": str(SOURCE),
        "project": str(PROJECT),
        "scientific_base": "E1 YOLO11s+P2 at imgsz=1024, batch=8, seed=1",
        "student_and_teacher_checkpoint": {"path": str(E1_BEST), "sha256": E1_SHA256},
        "single_change": "training-only teacher-superior P2/P3 class distillation from object-centred 1.78x zoom views",
        "inference_architecture_change": False,
        "paired_control": "C14 uses the same E1 checkpoint and same 30-epoch continuation without ER-ZD",
        "tasks_before_sha256": before_hash,
        "tasks_after_sha256": sha256(TASKS),
        "changes": changes,
        "installed_files": {str(path): sha256(path) for path in copies.values()},
        "existing_reports_removed": False,
        "existing_checkpoints_modified": False,
        "automatic_shutdown": False,
        "next": "E14a check-only -> smoke2 -> 30 epochs; run C14 only if E14a has a positive direct signal",
    }
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Installation record:", record_path)
    print("No existing report, checkpoint, run directory or old experiment file was deleted.")
    print("Next: E14a check-only -> smoke2 -> 30-epoch probe. Automatic shutdown is disabled.")


if __name__ == "__main__":
    main()
