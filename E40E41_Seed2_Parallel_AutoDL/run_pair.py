#!/usr/bin/env python3
"""Run the two independent 640/seed2 models on one GPU; never alter seed1 runs."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parent
NAMES = ("e40", "e41")
EXPERIMENTS = {
    "e40": "e40a_spatial_task_head_img640_seed2_200e",
    "e41": "e41a_channel_depth_task_head_img640_seed2_200e",
}
ENTRY = {name: ROOT / name / f"run_{name}.sh" for name in NAMES}
BASE = "e1_yolo11s_p2add_img640_seed2_20260926_172907_911324"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def say(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def package_check() -> None:
    path = ROOT / "manifest.json"
    if not path.is_file():
        raise RuntimeError("Missing seed2 pair release manifest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        source = ROOT / relative
        if not source.is_file():
            raise RuntimeError("Incomplete seed2 pair release: " + relative)
        actual = hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise RuntimeError("Changed seed2 pair release file: " + relative)


def gpu_state() -> dict:
    memory = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.free,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15, check=True,
    )
    rows = [row.strip() for row in memory.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("Expected exactly one GPU: " + repr(rows))
    values = [int(x.strip()) for x in rows[0].split(",")]
    if len(values) != 4:
        raise RuntimeError("Invalid nvidia-smi GPU row: " + rows[0])
    compute = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15, check=True,
    )
    pids = [int(x.strip()) for x in compute.stdout.splitlines() if x.strip().isdigit()]
    return dict(total_MiB=values[0], free_MiB=values[1], used_MiB=values[2],
                utilization_percent=values[3], compute_pids=pids)


def baseline_check(project: Path) -> None:
    path = project / "comparison_reports" / BASE / "metrics.json"
    if not path.is_file():
        raise RuntimeError("Missing E1 seed2 baseline report: " + str(path))
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("status") != "completed" or report.get("seed") != 2 or
            report.get("training", {}).get("completed_epochs") != 200):
        raise RuntimeError("E1 seed2 baseline is not a completed 200-epoch run")
    parameters = report["train_params"]
    if (parameters["imgsz"], parameters["batch"], parameters["epochs"]) != (640, 16, 200):
        raise RuntimeError("E1 seed2 baseline protocol differs")
    best = project / "runs/e1_yolo11s_p2add_img640_seed2/weights/best.pt"
    if not best.is_file():
        raise RuntimeError("Missing E1 seed2 best.pt: " + str(best))
    for name in NAMES:
        if not ENTRY[name].is_file():
            raise RuntimeError("Missing package entry: " + str(ENTRY[name]))


def run_one(name: str, mode: str) -> int:
    say(f"START {name} {mode}")
    result = subprocess.run(["bash", str(ENTRY[name]), mode], check=False)
    say(f"END {name} {mode}: exit={result.returncode}")
    return result.returncode


def progress(project: Path, name: str) -> str:
    prefix = name.upper() + "S2_RUN_"
    root = project / "comparison_reports"
    reports = sorted((p for p in root.glob(prefix + "*") if p.is_dir()), reverse=True)
    for path in reports:
        report = path / "training.json"
        if report.is_file():
            try:
                value = json.loads(report.read_text(encoding="utf-8"))
                return f"{value.get('completed_epochs', 0)}/200 status={value.get('status')}"
            except (OSError, json.JSONDecodeError):
                return "writing training.json"
    return "starting"


def launch_pair(mode: str, project: Path) -> int:
    log_dir = project / "logs" / ("pair_e40_e41_seed2_" + mode + "_" + timestamp())
    log_dir.mkdir(parents=True, exist_ok=False)
    say(f"{mode.upper()} logs: {log_dir}")
    children = {}
    handles = []
    try:
        for name in NAMES:
            handle = (log_dir / f"{name}_{mode}.log").open("w", encoding="utf-8", buffering=1)
            handles.append(handle)
            environment = os.environ.copy()
            if mode == 'run':
                environment['E40E41_SEED2_PAIR'] = '1'
            child = subprocess.Popen(["bash", str(ENTRY[name]), mode], stdout=handle,
                                     stderr=subprocess.STDOUT, start_new_session=True, env=environment)
            children[name] = child
            say(f"{name}: PID={child.pid}; log={log_dir / f'{name}_{mode}.log'}")
        state = dict(mode=mode, started_at=datetime.now().isoformat(),
                     logs={name: str(log_dir / f"{name}_{mode}.log") for name in NAMES},
                     pids={name: child.pid for name, child in children.items()}, exit_codes={})
        write_json(log_dir / "state.json", state)
        while True:
            state["exit_codes"] = {name: child.poll() for name, child in children.items()}
            try:
                state["gpu"] = gpu_state()
            except Exception as exc:
                state["gpu_error"] = str(exc)
            state["updated_at"] = datetime.now().isoformat()
            if mode == "run":
                state["epochs"] = {name: progress(project, name) for name in NAMES}
            write_json(log_dir / "state.json", state)
            say(" | ".join(f"{name}: {state.get('epochs', {}).get(name, mode)} exit={state['exit_codes'][name]}"
                           for name in NAMES) + f" | GPU free={state.get('gpu', {}).get('free_MiB', '?')} MiB")
            if all(code is not None for code in state["exit_codes"].values()):
                state["finished_at"] = datetime.now().isoformat()
                write_json(log_dir / "state.json", state)
                break
            time.sleep(60 if mode == "run" else 5)
        for name, code in state["exit_codes"].items():
            say(f"FINISHED {name} {mode}: exit={code}; log={state['logs'][name]}")
        return 0 if all(code == 0 for code in state["exit_codes"].values()) else 1
    finally:
        for handle in handles:
            handle.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("check", "smoke", "verify", "run"))
    parser.add_argument("--project", type=Path, default=Path("/root/autodl-tmp/yolo11_project"))
    args = parser.parse_args()
    package_check()
    project = args.project.resolve()
    baseline_check(project)
    lock = (ROOT / ".seed2_pair_execution.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another E40/E41 seed2 pair controller is already active") from exc
    if args.mode in ("check", "verify"):
        mode = "check" if args.mode == "check" else "verify-gates"
        codes = {name: run_one(name, mode) for name in NAMES}
        return 0 if all(code == 0 for code in codes.values()) else 1
    if args.mode == "smoke":
        # Each model's smoke entry validates its own seed2 CUDA check receipt.
        state = gpu_state()
        if state["free_MiB"] < 27 * 1024 or state["compute_pids"]:
            raise RuntimeError("Concurrent smoke requires an idle GPU with >=27 GiB free: " + str(state))
        return launch_pair("smoke", project)
    for name in NAMES:
        if run_one(name, "verify-gates") != 0:
            raise RuntimeError(name + " seed2 check/smoke receipts missing or stale")
        run_dir = project / "runs" / EXPERIMENTS[name]
        if run_dir.exists():
            raise RuntimeError("Existing run directory; refusing overwrite: " + str(run_dir))
    state = gpu_state()
    if state["free_MiB"] < 28 * 1024 or state["compute_pids"]:
        raise RuntimeError("Concurrent formal requires an idle GPU with >=28 GiB free: " + str(state))
    return launch_pair("run", project)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        say("FAILED: " + repr(exc))
        raise
