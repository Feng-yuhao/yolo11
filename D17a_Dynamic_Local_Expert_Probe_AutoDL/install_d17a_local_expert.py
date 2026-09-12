#!/usr/bin/env python3
"""Install D17a project files without modifying Ultralytics or frozen results."""

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
ROUTE = TARGET / "diagnose_d16a_cvrc_oof.py"
E1 = TARGET / "runs/e1_yolo11s_p2add_img1024_seed1/weights/best.pt"
FILES = ("yolo11s_d17a_dynamic_local_expert_probe.py", "run_d17a_local_expert_probe.sh")


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


def completed(pattern: str, branch: str | None = None) -> Path:
    rows = []
    for path in sorted((TARGET / "comparison_reports").glob(pattern)):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        selected = value.get("decision", value.get("decisions", {})).get("selected_branch")
        if value.get("status") == "completed" and (branch is None or selected == branch):
            rows.append(path)
    if not rows:
        raise RuntimeError(f"Required completed report not found: {pattern}")
    return rows[-1]


def main() -> None:
    for path in (TARGET, CORE, ROUTE, E1):
        if not path.exists():
            raise FileNotFoundError(path)
    d15b = completed("DIAG_D15B_e1_correction_fusion_headroom_*/metrics.json")
    d16b = completed("DIAG_D16B_e1_onesided_cvrc_*/metrics.json", "stop_cvrc_keep_d15a")
    protected = [CORE, ROUTE, E1, d15b, d16b]
    before = {str(path): sha256(path) for path in protected}
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
    after = {str(path): sha256(path) for path in protected}
    if before != after:
        raise RuntimeError("D17a installer changed a protected file")
    record = TARGET / "environment_records/d17a_local_expert_installation.json"
    atomic_json(record, {
        "time": datetime.now().isoformat(timespec="seconds"),
        "experiment": "D17a dynamic local expert probe30",
        "references": {"d15b": str(d15b), "d16b": str(d16b)},
        "installed": installed, "protected_files_unchanged": after,
        "ultralytics_modified": False, "automatic_shutdown": False,
    })
    print("D17a installed and syntax-checked.")
    print(f"Installation record: {record}")
    print("Next: --check-only, --smoke2, then run_d17a_local_expert_probe.sh.")
    print("Automatic shutdown is disabled.")


if __name__ == "__main__":
    main()
