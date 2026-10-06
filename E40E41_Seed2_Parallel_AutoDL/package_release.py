#!/usr/bin/env python3
"""Create integrity manifests and one uploadable AutoDL archive."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
COMMON = (
    "checks.py", "d27_official.py", "d38_common.py", "e1_640_core.py",
    "e1_reference.yaml", "evaluate.py", "LICENSE-Ultralytics.txt",
    "p2_tal_chunked.py", "smoke_memory.py",
)
FILES = {
    "e40": COMMON + ("run_e40.py", "run_e40.sh", "spatial_task_head.py"),
    "e41": COMMON + ("run_e41.py", "run_e41.sh", "channel_task_head.py",
                     "queue_after_e40.py"),
}


def normalized(path: Path) -> bytes:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    if path.suffix == ".sh" and not data.endswith(b"\n"):
        data += b"\n"
    return data


def main() -> None:
    manifests = {}
    for name, files in FILES.items():
        manifest = {}
        for filename in files:
            path = ROOT / name / filename
            if not path.is_file():
                raise RuntimeError("Missing required source file: " + str(path))
            manifest[filename] = sha256(normalized(path)).hexdigest()
        manifests[name] = manifest
        (ROOT / name / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    control_files = ("run_pair.py", "run_pair.sh", "README.md")
    controller_manifest = {
        filename: sha256(normalized(ROOT / filename)).hexdigest()
        for filename in control_files
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(controller_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    archive = ROOT.parent / (ROOT.name + ".zip")
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=7) as output:
        for name, files in FILES.items():
            for filename in (*files, "manifest.json"):
                path = ROOT / name / filename
                output.writestr(f"{ROOT.name}/{name}/{filename}", normalized(path))
        for filename in (*control_files, "manifest.json"):
            path = ROOT / filename
            output.writestr(f"{ROOT.name}/{filename}", normalized(path))
    with ZipFile(archive) as output:
        names = set(output.namelist())
        for name, files in FILES.items():
            manifest = json.loads(output.read(f"{ROOT.name}/{name}/manifest.json"))
            for filename in files:
                actual = sha256(output.read(f"{ROOT.name}/{name}/{filename}")).hexdigest()
                if manifest[filename] != actual:
                    raise RuntimeError(f"Archived hash mismatch: {name}/{filename}")
        controller = json.loads(output.read(f"{ROOT.name}/manifest.json"))
        for filename in control_files:
            if controller[filename] != sha256(output.read(f"{ROOT.name}/{filename}")).hexdigest():
                raise RuntimeError("Archived controller hash mismatch: " + filename)
        expected = sum(len(files) + 1 for files in FILES.values()) + len(control_files) + 1
        if len(names) != expected:
            raise RuntimeError("Unexpected archive content count")
    print(f"READY {archive} ({archive.stat().st_size} bytes; {len(names)} files)")


if __name__ == "__main__":
    main()
