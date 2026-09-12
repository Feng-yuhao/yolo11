#!/usr/bin/env python3
"""Install D16a project-only files without modifying Ultralytics or E1."""

from __future__ import annotations

import hashlib
import json
import os
import py_compile
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
SOURCE = PACKAGE / "yolo11_project"
TARGET = Path("/root/autodl-tmp/yolo11_project")
CORE = TARGET / "evaluate_e1_e5a_slicing.py"
WEIGHT = TARGET / "runs/e1_yolo11s_p2add_img1024_seed1/weights/best.pt"
FILES = ("diagnose_d16a_cvrc_oof.py", "run_d16a_cvrc_oof.sh")
REQUIRED_CORE_TOKENS = (
    'SCRIPT_REVISION = "fixed_2x2_slicing_upper_bound_v2"',
    "def make_tiles(", "def map_tile_detections(", "def model_predict(",
    "def coco_metrics(", "def fixed_operating_point(",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def latest_completed_d15b() -> Path:
    completed = []
    for path in sorted((TARGET / "comparison_reports").glob("DIAG_D15B_e1_correction_fusion_headroom_*/metrics.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            value.get("status") == "completed"
            and value.get("is_full_diagnostic") is True
            and value.get("decisions", {}).get("selected_branch") == "cross_view_local_candidate_verifier"
        ):
            completed.append(path)
    if not completed:
        raise RuntimeError("D16a requires a completed full D15b report selecting the verifier branch")
    return completed[-1]


def main() -> None:
    if not TARGET.is_dir():
        raise FileNotFoundError(TARGET)
    if not CORE.is_file() or not WEIGHT.is_file():
        raise FileNotFoundError("Missing slicing core or frozen E1 checkpoint")
    content = CORE.read_text(encoding="utf-8")
    missing = [token for token in REQUIRED_CORE_TOKENS if token not in content]
    if missing:
        raise RuntimeError(f"Slicing core differs from the audited version: {missing}")
    d15b = latest_completed_d15b()
    protected_before = {str(CORE): sha256(CORE), str(WEIGHT): sha256(WEIGHT)}
    installed = {}
    for name in FILES:
        source = SOURCE / name
        target = TARGET / name
        if not source.is_file():
            raise FileNotFoundError(source)
        existed = target.is_file()
        if existed and sha256(source) != sha256(target):
            backup = target.with_name(target.name + ".pre_d16a_backup")
            shutil.copy2(target, backup)
        shutil.copy2(source, target)
        installed[name] = {"sha256": sha256(target), "replaced": existed}
    py_compile.compile(str(TARGET / FILES[0]), doraise=True)
    completed = subprocess.run(["bash", "-n", str(TARGET / FILES[1])], capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError("Shell syntax check failed:\n" + completed.stdout + completed.stderr)
    protected_after = {str(CORE): sha256(CORE), str(WEIGHT): sha256(WEIGHT)}
    if protected_after != protected_before:
        raise RuntimeError("Installer changed the core or E1 checkpoint")
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "experiment": "D16a grouped nested OOF CVRC probe",
        "d15b_reference": str(d15b), "installed": installed,
        "protected": protected_after, "ultralytics_modified": False,
        "detector_checkpoint_modified": False, "automatic_shutdown": False,
    }
    destination = TARGET / "environment_records/d16a_cvrc_oof_registration.json"
    atomic_json(destination, record)
    print("D16a files installed and syntax-checked.")
    print(f"D15b reference: {d15b}")
    print(f"Installation record: {destination}")
    print("Next: --check-only, --smoke2, then run_d16a_cvrc_oof.sh.")
    print("Automatic shutdown is disabled.")


if __name__ == "__main__":
    main()
