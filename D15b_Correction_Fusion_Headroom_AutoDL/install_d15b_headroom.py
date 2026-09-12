#!/usr/bin/env python3
"""Install D15b project-only files without changing Ultralytics or E1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
SOURCE_PROJECT = PACKAGE / "yolo11_project"
TARGET = Path("/root/autodl-tmp/yolo11_project")
CORE = TARGET / "evaluate_e1_e5a_slicing.py"
WEIGHT = TARGET / "runs/e1_yolo11s_p2add_img1024_seed1/weights/best.pt"
FILES = (
    "diagnose_d15b_correction_fusion_headroom.py",
    "run_d15b_correction_fusion_headroom.sh",
)
REQUIRED_CORE_TOKENS = (
    'SCRIPT_REVISION = "fixed_2x2_slicing_upper_bound_v2"',
    "def make_tiles(",
    "def map_tile_detections(",
    "def model_predict(",
    "def coco_metrics(",
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
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def install_file(source: Path, target: Path, force: bool) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    existed = target.exists()
    if existed:
        if sha256(source) == sha256(target):
            return "already_identical"
        if not force:
            raise RuntimeError(
                f"Refusing to overwrite a different file: {target}\n"
                "Inspect it or rerun with --force."
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return "replaced" if existed else "installed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not TARGET.is_dir():
        raise FileNotFoundError(TARGET)
    if not CORE.is_file():
        raise FileNotFoundError(CORE)
    if not WEIGHT.is_file():
        raise FileNotFoundError(WEIGHT)
    core_text = CORE.read_text(encoding="utf-8")
    missing = [token for token in REQUIRED_CORE_TOKENS if token not in core_text]
    if missing:
        raise RuntimeError(f"Slicing core is incompatible; missing tokens: {missing}")
    d15a_reports = sorted((TARGET / "comparison_reports").glob(
        "DIAG_D15A_e1_dynamic_zoom_budget_*/metrics.json"
    ))
    if not d15a_reports:
        raise FileNotFoundError("D15b requires the completed D15a report on the server")

    protected_before = {str(CORE): sha256(CORE), str(WEIGHT): sha256(WEIGHT)}
    actions = {}
    for name in FILES:
        actions[name] = install_file(SOURCE_PROJECT / name, TARGET / name, args.force)

    py_compile.compile(str(TARGET / FILES[0]), doraise=True)
    completed = subprocess.run(
        ["bash", "-n", str(TARGET / FILES[1])],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"bash -n failed:\n{completed.stdout}{completed.stderr}")
    protected_after = {str(CORE): sha256(CORE), str(WEIGHT): sha256(WEIGHT)}
    if protected_after != protected_before:
        raise RuntimeError("Installer unexpectedly changed the core or E1 checkpoint")

    record = {
        "experiment": "D15b correction-selection and fusion-headroom diagnostic",
        "installed_at": datetime.now().isoformat(timespec="seconds"),
        "package": str(PACKAGE),
        "target": str(TARGET),
        "actions": actions,
        "installed_sha256": {name: sha256(TARGET / name) for name in FILES},
        "protected_files_unchanged": protected_after,
        "ultralytics_modified": False,
        "checkpoint_modified": False,
        "training_performed": False,
        "automatic_shutdown": False,
    }
    record_path = TARGET / "environment_records/d15b_headroom_installation.json"
    atomic_json(record_path, record)
    print("D15b installation/check passed.")
    print(f"Installation record: {record_path}")
    print("Ultralytics and E1 were not modified.")
    print("Next: --check-only, --smoke-images 8, then the full diagnostic.")
    print("Automatic shutdown is disabled.")


if __name__ == "__main__":
    main()
