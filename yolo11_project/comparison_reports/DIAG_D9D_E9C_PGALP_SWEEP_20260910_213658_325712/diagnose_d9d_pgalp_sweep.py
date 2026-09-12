#!/usr/bin/env python3
"""D9d: zero-training PGALP scale/strength sweep on the frozen E9c best.pt.

The corrected E9c reliability map is preserved. Only the inference-time blur
policy and a scalar multiplier on the learned residual are intervened on.
No model parameter is updated and automatic shutdown is never requested.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import shutil
import sys
import time
import traceback
import types
from datetime import datetime
from pathlib import Path

import d9c_base as base


PROJECT = Path("/root/autodl-tmp/yolo11_project")
SOURCE = Path("/root/autodl-tmp/yolo11_src")
WEIGHTS = PROJECT / "runs/e9c_probe_yolo11s_p2_agpgalp_img1024_seed1/weights/best.pt"
E9B_METRICS = PROJECT / "runs/e9b_yolo11s_p2_pgalp_img1024_seed1/complete_metrics.json"
SCRIPT_REVISION = "d9d_e9c_fixed_reliability_scale_strength_sweep_v1"
NEW_MODES = (
    "dynamic_s025", "dynamic_s050", "dynamic_s075",
    "f3_s025", "f3_s050", "f3_s075",
    "f5_s025", "f5_s050", "f5_s075",
    "mix35_s025", "mix35_s050", "mix35_s075", "mix35_s100",
)
REFERENCE_MODE_MAP = {
    "identity_s000": "identity",
    "dynamic_s100": "normal",
    "f3_s100": "force3",
    "f5_s100": "force5",
    "f7_s100": "force7",
}
MODE_PATTERN = re.compile(r"^(dynamic|f3|f5|f7|mix35)_s(\d{3})$")


def now():
    return datetime.now().isoformat(timespec="seconds")


def parse_mode(mode):
    match = MODE_PATTERN.fullmatch(mode)
    if match is None:
        raise ValueError(f"Invalid D9d mode: {mode}")
    policy, digits = match.groups()
    multiplier = int(digits) / 100.0
    if multiplier <= 0 or multiplier > 1:
        raise ValueError(f"Strength must be in (0,1], got {mode}")
    return policy, multiplier


class SweepController:
    """Intervene on blur scale and residual strength without changing weights."""

    def __init__(self, module, mode, capture=False):
        if capture:
            raise ValueError("D9d reuses the completed D9c routing audit; capture must be false")
        self.module = module
        self.mode = mode
        self.policy, self.multiplier = parse_mode(mode)
        module.forward = types.MethodType(lambda _, inputs: self.forward(inputs), module)

    def forward(self, inputs):
        torch = sys.modules["torch"]
        p2, p3 = inputs
        candidates = [self.module._blur(p2, index) for index in range(len(self.module.kernels))]
        reliability, weights = self.module.routing(p2, p3, candidates[0])
        if self.policy == "dynamic":
            smooth = torch.zeros_like(p2)
            for index, candidate in enumerate(candidates):
                smooth = smooth + weights[:, index : index + 1] * candidate
        elif self.policy == "f3":
            smooth = candidates[self.module.kernels.index(3)]
        elif self.policy == "f5":
            smooth = candidates[self.module.kernels.index(5)]
        elif self.policy == "f7":
            smooth = candidates[self.module.kernels.index(7)]
        elif self.policy == "mix35":
            smooth = 0.5 * candidates[self.module.kernels.index(3)] + 0.5 * candidates[self.module.kernels.index(5)]
        else:
            raise AssertionError(self.policy)
        return p2 + self.multiplier * self.module.layer_scale * (1.0 - reliability) * (smooth - p2)


def find_d9c_reference(weight_sha):
    candidates = []
    for metrics_path in (PROJECT / "comparison_reports").glob("*/metrics.json"):
        try:
            value = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            value.get("status") == "completed"
            and value.get("revision") == "d9c_agpgalp_causal_routing_audit_v1"
            and value.get("weights", {}).get("sha256") == weight_sha
            and all(name in value.get("modes", {}) for name in REFERENCE_MODE_MAP.values())
        ):
            candidates.append((metrics_path.stat().st_mtime, metrics_path, value))
    if not candidates:
        raise FileNotFoundError("Completed D9c report for the current E9c checkpoint was not found")
    _, path, value = sorted(candidates)[-1]
    return path, value


def metric_delta(left, right):
    keys = ("AP_all", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR_all")
    return {key: float(left[key]) - float(right[key]) for key in keys}


def pareto_front(rows):
    front = []
    for name, values in rows.items():
        dominated = any(
            other != name
            and candidate["AP_all"] >= values["AP_all"]
            and candidate["AP_small"] >= values["AP_small"]
            and (candidate["AP_all"] > values["AP_all"] or candidate["AP_small"] > values["AP_small"])
            for other, candidate in rows.items()
        )
        if not dominated:
            front.append(name)
    return sorted(front)


def write_summary(report, path):
    lines = [
        "D9d E9c PGALP zero-training scale/strength sweep",
        f"status: {report['status']}",
        f"revision: {SCRIPT_REVISION}",
        f"weights: {report.get('weights', {}).get('path')}",
        f"images: {report.get('images')}",
        "AP/AR are fractions; multiply by 100 for percentage points.",
        "Reliability routing is frozen from E9c; only blur policy and residual multiplier change.",
        "",
        "mode, source, policy, strength, AP_all, AP_small, AP_medium, AP_large, delta_E9b_AP_all, delta_E9b_AP_small",
    ]
    for mode, row in report.get("combined_modes", {}).items():
        lines.append(
            ", ".join(str(row.get(key)) for key in (
                "mode", "source", "policy", "strength", "AP_all", "AP_small", "AP_medium", "AP_large",
                "delta_E9b_AP_all", "delta_E9b_AP_small",
            ))
        )
    decision = report.get("decision", {})
    lines += [
        "",
        f"best evaluated AP-small: {decision.get('best_new_ap_small')}",
        f"best evaluated AP-all: {decision.get('best_new_ap_all')}",
        f"Pareto modes: {decision.get('pareto_new_modes')}",
        f"recommended candidate: {decision.get('recommended_candidate')}",
        f"recommendation reason: {decision.get('reason')}",
        "",
        "No training was performed. Automatic shutdown is disabled.",
    ]
    base.atomic_text(path, "\n".join(lines) + "\n")


def write_csv(report, path):
    fields = [
        "mode", "source", "policy", "strength", "AP_all", "AP50", "AP75", "AP_small", "AP_medium",
        "AP_large", "AR_all", "delta_E9b_AP_all", "delta_E9b_AP_small", "seconds",
    ]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report.get("combined_modes", {}).values():
            writer.writerow({key: row.get(key) for key in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--smoke-images", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = "CHECK" if args.check_only else "SMOKE" if args.smoke_images else "DIAG"
    output = PROJECT / "comparison_reports" / f"{prefix}_D9D_E9C_PGALP_SWEEP_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "starting",
        "purpose": "D9d zero-training scale/strength diagnosis of corrected E9c PGALP",
        "revision": SCRIPT_REVISION,
        "started_at": now(),
        "report_dir": str(output),
        "automatic_shutdown": False,
        "new_modes": list(NEW_MODES),
    }
    base.write_json(output / "metrics.json", report)

    try:
        dependencies = base.load_dependencies()
        _, torch, _, ultralytics, _, _, YOLO, core = dependencies
        weights_path = args.weights.resolve()
        for path in (weights_path, E9B_METRICS, SOURCE / "ultralytics/nn/modules/ag_pgalp.py"):
            if not path.is_file():
                raise FileNotFoundError(path)
        if Path(ultralytics.__file__).resolve().parent != (SOURCE / "ultralytics").resolve():
            raise RuntimeError(f"Wrong Ultralytics source: {ultralytics.__file__}")
        if ultralytics.__version__ != "8.3.0":
            raise RuntimeError(f"Expected Ultralytics 8.3.0, got {ultralytics.__version__}")
        if not args.check_only and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for D9d prediction")

        weight_sha = base.sha256(weights_path)
        reference_path, d9c = find_d9c_reference(weight_sha)
        e9b = json.loads(E9B_METRICS.read_text(encoding="utf-8"))
        if e9b.get("status") != "completed":
            raise RuntimeError("Frozen E9b formal metrics are incomplete")
        e9b_area = e9b["area_metrics"]
        report["weights"] = {"path": str(weights_path), "sha256": weight_sha}
        report["d9c_reference"] = str(reference_path)
        report["e9b_reference"] = str(E9B_METRICS)
        report["environment"] = {
            "python": platform.python_version(), "ultralytics": ultralytics.__version__, "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        check_model = YOLO(str(weights_path))
        module = base.find_pgalp(check_model.model)
        if module.__class__.__name__ != "AGPGALP":
            raise RuntimeError("D9d requires the E9c AGPGALP checkpoint")
        for mode in NEW_MODES:
            parse_mode(mode)
        names = check_model.names
        del check_model
        base.clear_gpu(torch)
        shutil.copy2(Path(__file__), output / Path(__file__).name)
        shutil.copy2(Path(base.__file__), output / Path(base.__file__).name)
        if args.check_only:
            report.update(status="checked", finished_at=now())
            base.write_json(output / "metrics.json", report)
            write_summary(report, output / "metrics.txt")
            print(f"D9d check passed. No prediction, training, or shutdown. Report: {output}", flush=True)
            return

        stub = {"report_dir": str(output), "dataset": {}}
        images, path_ids, gt, prepared_names, _ = core.prepare_dataset(stub)
        if prepared_names != names:
            raise RuntimeError("Dataset/checkpoint class names differ")
        if args.smoke_images:
            if not 1 <= args.smoke_images <= len(images):
                raise ValueError(f"--smoke-images must be in [1,{len(images)}]")
            images = images[: args.smoke_images]
            selected = {path_ids[str(path.resolve())] for path in images}
            gt = base.subset_ground_truth(gt, [item for item in gt["images"] if int(item["id"]) in selected])
        report["images"] = len(images)
        report["dataset"] = stub["dataset"]
        report["protocol"] = {
            "imgsz": base.IMGSZ, "prediction_batch": base.BATCH, "conf": base.CONF, "iou": base.IOU,
            "max_det": base.MAX_DET, "half": False,
            "note": "Same COCO-area protocol as D9c; this is not official VisDrone ignore-region evaluation.",
        }
        report["status"] = "evaluating"
        base.write_json(output / "metrics.json", report)

        base.PGALPController = SweepController
        evaluated = {}
        for mode in NEW_MODES:
            print(f"\nD9d mode: {mode}", flush=True)
            values, _ = base.evaluate_mode(
                mode, weights_path, images, path_ids, gt, names, dependencies, output, collect_routing=False
            )
            evaluated[mode] = values
            report["evaluated_modes"] = evaluated
            base.write_json(output / "metrics.json", report)

        if args.smoke_images:
            report["combined_modes"] = {
                mode: {
                    "mode": mode, "source": "D9d_smoke_subset",
                    "policy": parse_mode(mode)[0], "strength": parse_mode(mode)[1], **values,
                }
                for mode, values in evaluated.items()
            }
            report["decision"] = {
                "recommended_candidate": None,
                "reason": "Smoke subset validates execution only; its AP is not compared with full-validation references.",
            }
            report.update(status="smoke_completed", finished_at=now())
            base.write_json(output / "metrics.json", report)
            write_summary(report, output / "metrics.txt")
            write_csv(report, output / "summary.csv")
            print(f"\nD9d smoke completed. Report: {output / 'metrics.txt'}", flush=True)
            print("No training was performed. Automatic shutdown is disabled.", flush=True)
            return

        references = {}
        for new_name, old_name in REFERENCE_MODE_MAP.items():
            references[new_name] = d9c["modes"][old_name]
        combined = {}
        for source_name, rows in (("D9d_new", evaluated), ("D9c_reused", references)):
            for mode, values in rows.items():
                if mode.startswith("identity"):
                    policy, strength = "identity", 0.0
                else:
                    policy, strength = parse_mode(mode)
                row = {"mode": mode, "source": source_name, "policy": policy, "strength": strength, **values}
                row["delta_E9b_AP_all"] = float(values["AP_all"]) - float(e9b_area["AP_all"])
                row["delta_E9b_AP_small"] = float(values["AP_small"]) - float(e9b_area["AP_small"])
                combined[mode] = row
        report["reference_modes"] = references
        report["combined_modes"] = combined
        report["deltas_from_e9c_normal"] = {
            mode: metric_delta(values, references["dynamic_s100"]) for mode, values in evaluated.items()
        }

        best_small = max(evaluated, key=lambda name: (evaluated[name]["AP_small"], evaluated[name]["AP_all"]))
        best_all = max(evaluated, key=lambda name: (evaluated[name]["AP_all"], evaluated[name]["AP_small"]))
        eligible = [
            name for name, values in evaluated.items()
            if values["AP_all"] >= e9b_area["AP_all"] - 0.0005
            and values["AP_small"] > e9b_area["AP_small"]
        ]
        if eligible:
            recommended = max(eligible, key=lambda name: (evaluated[name]["AP_small"], evaluated[name]["AP_all"]))
            reason = "Candidate beats frozen E9b AP-small while keeping AP-all within 0.05 percentage points."
        else:
            recommended = None
            reason = "No zero-training intervention meets the E9b gate; do not start E9d training from this sweep alone."
        report["decision"] = {
            "best_new_ap_small": best_small,
            "best_new_ap_all": best_all,
            "pareto_new_modes": pareto_front(evaluated),
            "recommended_candidate": recommended,
            "reason": reason,
            "gate": "AP_small > E9b and AP_all >= E9b - 0.0005",
        }
        report["status"] = "completed"
        report["finished_at"] = now()
        base.write_json(output / "metrics.json", report)
        write_summary(report, output / "metrics.txt")
        write_csv(report, output / "summary.csv")
        print(f"\nD9d completed. Report: {output / 'metrics.txt'}", flush=True)
        print("No training was performed. Automatic shutdown is disabled.", flush=True)
    except BaseException as error:
        report.update(status="failed", finished_at=now(), error=repr(error), traceback=traceback.format_exc())
        base.write_json(output / "metrics.json", report)
        write_summary(report, output / "metrics.txt")
        print(f"D9d failure saved: {output}", flush=True)
        print("Automatic shutdown is disabled.", flush=True)
        raise


if __name__ == "__main__":
    main()
