#!/usr/bin/env python3
"""Strict seed-2 replication of the frozen E1 P2 experiment."""

from pathlib import Path

import yolo11s_p2_img1024_seed1 as core


SEED = 2
EXPERIMENT = "e1_yolo11s_p2add_img1024_seed2"
REVISION = "e1_p2_strict_seed2_200e_tal2_v1"


def install_seed2() -> None:
    """Change only the random seed and output identity; preserve every train setting."""
    core.SEED = SEED
    core.EXPERIMENT = EXPERIMENT
    core.SCRIPT_REVISION = REVISION
    core.SMOKE_GATE = core.ROOT / "comparison_reports/e1_seed2_smoke_passed.json"
    core.AUTO_SHUTDOWN = False
    core.SHUTDOWN_ON_FAILURE = False
    core.__file__ = str(Path(__file__).resolve())

    params = core.training_params()
    required = {
        "epochs": 200,
        "imgsz": 1024,
        "batch": 8,
        "workers": 8,
        "seed": 2,
        "deterministic": True,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "amp": True,
        "cos_lr": True,
        "close_mosaic": 10,
        "nbs": 64,
        "cache": False,
        "rect": False,
        "multi_scale": False,
    }
    changed = {key: (params.get(key), expected) for key, expected in required.items() if params.get(key) != expected}
    if changed:
        raise RuntimeError(f"E1 seed2 protocol drift: {changed}")


if __name__ == "__main__":
    install_seed2()
    core.main()
