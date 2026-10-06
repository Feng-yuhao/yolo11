#!/usr/bin/env python3
"""E10a original DSSG, strict seed-2 replication (not E10clean)."""

import json
from pathlib import Path

import yolo11s_p2_dssg_img1024_seed1 as variant


EXPERIMENT = "e10a_yolo11s_p2_dssg_img1024_seed2"
REVISION = "e10a_original_dssg_strict_seed2_200e_v1"


def seed2_e1_reference():
    root = variant.core.ROOT / "comparison_reports"
    candidates = []
    for path in root.glob("e1_yolo11s_p2add_img1024_seed2_*/metrics.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") == "completed" and int(data.get("training", {}).get("completed_epochs", -1)) == 200:
            candidates.append((path, data))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one completed E1 seed2 reference, found {len(candidates)}")
    return candidates[0]


def install_seed2():
    seed2_e1_reference()  # Fail before any training if the paired baseline is absent.
    variant.EXPERIMENT = EXPERIMENT
    variant.SCRIPT_REVISION = REVISION
    variant.SMOKE_GATE = variant.core.ROOT / "comparison_reports/e10a_dssg_seed2_smoke_passed.json"
    variant._frozen_e1_report = seed2_e1_reference
    variant.install_overrides()

    core = variant.core
    core.SEED = 2
    core.EXPERIMENT = EXPERIMENT
    core.SCRIPT_REVISION = REVISION
    core.SMOKE_GATE = variant.SMOKE_GATE
    core.AUTO_SHUTDOWN = False
    core.SHUTDOWN_ON_FAILURE = False
    core.__file__ = str(Path(__file__).resolve())

    expected = {
        "epochs": 200, "imgsz": 1024, "batch": 8, "workers": 8,
        "seed": 2, "deterministic": True, "optimizer": "SGD",
        "lr0": 0.01, "lrf": 0.01, "momentum": 0.937,
        "weight_decay": 0.0005, "amp": True, "cos_lr": True,
        "close_mosaic": 10, "nbs": 64, "cache": False,
        "rect": False, "multi_scale": False, "resume": False,
        "exist_ok": False, "name": EXPERIMENT,
    }
    params = core.training_params()
    drift = {key: (params.get(key), value) for key, value in expected.items() if params.get(key) != value}
    if drift:
        raise RuntimeError(f"E10a seed2 training protocol drift: {drift}")


if __name__ == "__main__":
    install_seed2()
    variant.core.main()
