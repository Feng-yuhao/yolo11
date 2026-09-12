#!/usr/bin/env python3
"""Install D16b project files while protecting E1, D16a and Ultralytics."""

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
FILES = ("diagnose_d16b_onesided_cvrc.py", "run_d16b_onesided_cvrc.sh")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def find_d16a() -> Path:
    rows = []
    for path in sorted((TARGET / "comparison_reports").glob("DIAG_D16A_e1_cvrc_oof_*/metrics.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        files = [path.parent / "diagnose_d16a_cvrc_oof.py"] + [path.parent / f"cvrc_fold{i}.pt" for i in range(1, 6)]
        if value.get("status") == "completed" and len(value.get("folds", [])) == 5 and all(item.is_file() for item in files):
            rows.append(path)
    if not rows:
        raise RuntimeError("Completed D16a report and five fold artifacts are required")
    return rows[-1]


def main() -> None:
    if not TARGET.is_dir() or not CORE.is_file() or not WEIGHT.is_file():
        raise FileNotFoundError("Missing yolo11_project, slicing core, or E1 checkpoint")
    d16a = find_d16a()
    protected_files = [CORE, WEIGHT, d16a, d16a.parent / "diagnose_d16a_cvrc_oof.py"] + [d16a.parent / f"cvrc_fold{i}.pt" for i in range(1, 6)]
    before = {str(path): sha256(path) for path in protected_files}
    installed = {}
    for name in FILES:
        source, target = SOURCE / name, TARGET / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target)
        installed[name] = sha256(target)
    py_compile.compile(str(TARGET / FILES[0]), doraise=True)
    checked = subprocess.run(["bash", "-n", str(TARGET / FILES[1])], capture_output=True, text=True)
    if checked.returncode:
        raise RuntimeError("Shell syntax failed:\n" + checked.stdout + checked.stderr)
    after = {str(path): sha256(path) for path in protected_files}
    if before != after:
        raise RuntimeError("Installer changed a protected E1/D16a/core file")
    record = TARGET / "environment_records/d16b_onesided_cvrc_installation.json"
    atomic_json(record, {
        "time": datetime.now().isoformat(timespec="seconds"),
        "experiment": "D16b frozen one-sided CVRC",
        "d16a_reference": str(d16a), "installed": installed,
        "protected_files_unchanged": after, "training_performed": False,
        "ultralytics_modified": False, "automatic_shutdown": False,
    })
    print("D16b installed and syntax-checked.")
    print(f"D16a reference: {d16a}")
    print(f"Installation record: {record}")
    print("Next: --check-only, --smoke2, then run_d16b_onesided_cvrc.sh.")
    print("No training and no automatic shutdown.")


if __name__ == "__main__":
    main()
