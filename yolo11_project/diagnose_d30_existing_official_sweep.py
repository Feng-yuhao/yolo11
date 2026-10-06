#!/usr/bin/env python3
"""
D30: Existing-model Official VisDrone sweep.

Purpose
-------
Re-evaluate already-completed YOLO11/VisDrone experiments under exactly the
same official-ignore-aligned evaluator used by D27.

Hard boundaries
---------------
- ZERO TRAINING.
- ZERO CHECKPOINT MODIFICATION.
- Reuse existing formal ``area_predictions.json`` only.
- Import D27 evaluator functions directly; do not reimplement matching.
- Existing CLOSED routes are NOT reopened by a ranking change.
- Missing prediction files are reported, not regenerated automatically.
- E27a is intentionally excluded because its formal training was interrupted.

Primary metrics
---------------
Official VisDrone DET Matlab-toolkit-v1.0.4-port semantics from D27:
AP@[.50:.95], AP50, AP75, AR500.

Outputs
-------
comparison_reports/DIAG_D30_EXISTING_OFFICIAL_SWEEP_<timestamp>/
    metrics.json
    metrics.txt
    model_summary.csv
    official_per_class.csv
    reproduction_gate.json
    missing_sources.json
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import sys
import time
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REVISION = "d30_existing_models_official_sweep_reuse_d27_v1"
EXPECTED_D27_REVISION = "d27_official_visdrone_det_matlab104_port_v1"

DEFAULT_PROJECT = Path("/root/autodl-tmp/yolo11_project")
DEFAULT_DATASET = Path("/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val")
D27_SCRIPT_NAME = "diagnose_d27_official_visdrone.py"

# Tight patterns: formal report directories only.  Smoke/probe directories are
# deliberately excluded unless the experiment itself was a formal named run.
MODEL_SPECS = OrderedDict(
    [
        ("e0", "baseline_yolo11s_img1024_seed1_*"),
        ("e1", "e1_yolo11s_p2add_img1024_seed1_*"),
        ("e5a", "e5a_yolo11s_p2_dysample_neck_img1024_seed1_*"),
        ("e8b", "e8b_yolo11s_p2_p3sde_img1024_seed1_*"),
        ("e9b", "e9b_yolo11s_p2_pgalp_img1024_seed1_*"),
        ("e10a", "e10a_yolo11s_p2_dssg_img1024_seed1_*"),
        ("e10clean", "e10clean_yolo11s_p2_hsdg_img1024_seed1_*"),
        ("e15a", "e15a_yolo11s_p2_dssg_p1sdi_img1024_seed1_*"),
        ("e16a", "e16a_yolo11s_p2_tsdr_img1024_seed1_*"),
        ("e17a", "e17a_yolo11s_p2_fusion_ready_img1024_seed1_*"),
        ("e18a", "e18a_yolo11s_p2_dssg_fusion_ready_img1024_seed1_*"),
        ("e19a", "e19a_yolo11s_p2_rcfa_img1024_seed1_*"),
        ("e20a", "e20a_yolo11s_p2_sdqc_img1024_seed1_*"),
        ("e20b", "e20b_yolo11s_p2_tsdr_sdqc_img1024_seed1_*"),
        ("e21a", "e21a_yolo11s_p2_rsg_dssg_img1024_seed1_*"),
        ("e23a", "e23a_yolo11s_p2_dssg_regq_img1024_seed1_*"),
        ("e25a", "e25a_yolo11s_p2_gi_rsg_dssg_img1024_seed1_*"),
        ("e26a", "e26a_yolo11s_p2_terp_img1024_seed1_*"),
    ]
)

# This is historical route state, not a score verdict.
HISTORICAL_STATUS = {
    "e0": "REFERENCE",
    "e1": "ACTIVE_BASELINE",
    "e5a": "CLOSED",
    "e8b": "CLOSED",
    "e9b": "CLOSED",
    "e10a": "SUPPORTING_RESULT",
    "e10clean": "CLOSED",
    "e15a": "CLOSED",
    "e16a": "CLOSED",
    "e17a": "CLOSED",
    "e18a": "CLOSED",
    "e19a": "CLOSED",
    "e20a": "CLOSED",
    "e20b": "CLOSED",
    "e21a": "CLOSED",
    "e23a": "CLOSED",
    "e25a": "CLOSED",
    "e26a": "CLOSED",
}

HISTORICAL_NOTE = {
    "e0": "Original YOLO11s reference.",
    "e1": "Only active structural baseline.",
    "e5a": "DySample route closed.",
    "e8b": "P3SDE route closed.",
    "e9b": "PGALP route closed.",
    "e10a": "DSSG positive but small historical gain; supporting result.",
    "e10clean": "DSSG/HSDG derivative closed.",
    "e15a": "P1SDI: P1 late detail injection gave negligible small-object gain.",
    "e16a": "TSDR: localization-oriented mechanism did not improve small-object geometry.",
    "e17a": "FusionReady route closed for further derivatives; retained for existing-model score audit.",
    "e18a": "DSSG+FusionReady retained for existing-model score audit; no automatic reopening.",
    "e19a": "RCFA route closed.",
    "e20a": "SDQC route closed.",
    "e20b": "TSDR+SDQC route closed.",
    "e21a": "RSG permanently closed: preregistered scale-balance failure.",
    "e23a": "Regression-quality strengthening route closed.",
    "e25a": "Gradient-Isolated RSG repair failed; permanently closed.",
    "e26a": "TERP formal run ineffective; closed.",
}

# Models already evaluated in D27.  D30 must reproduce them exactly before its
# newly added rankings are trusted.
D27_REPRO_MODELS = ("e0", "e1", "e10a", "e16a", "e21a", "e26a")
REPRO_TOLERANCE_PERCENT = 1e-6


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def load_d27_module(project: Path):
    path = project / D27_SCRIPT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing D27 evaluator: {path}")
    spec = importlib.util.spec_from_file_location("d27_official_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import D27 evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if getattr(module, "REVISION", None) != EXPECTED_D27_REVISION:
        raise RuntimeError(
            f"D27 revision changed: expected={EXPECTED_D27_REVISION}, "
            f"found={getattr(module, 'REVISION', None)}"
        )
    return module, path


def find_latest_completed_d27(project: Path) -> Path:
    root = project / "comparison_reports"
    candidates = []
    for item in root.glob("DIAG_D27_OFFICIAL_VISDRONE_*"):
        metrics = item / "metrics.json"
        if not item.is_dir() or not metrics.is_file():
            continue
        try:
            payload = json.loads(metrics.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            payload.get("status") == "completed"
            and payload.get("revision") == EXPECTED_D27_REVISION
        ):
            candidates.append(item)
    if not candidates:
        raise FileNotFoundError(
            "No completed D27 report found under comparison_reports/"
        )
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.name))


def find_latest_report(project: Path, pattern: str) -> Path | None:
    root = project / "comparison_reports"
    candidates = []
    for item in root.glob(pattern):
        if not item.is_dir():
            continue
        # D30 intentionally requires formal saved predictions and the image-id
        # table used by D27's preflight.
        if not (item / "area_predictions.json").is_file():
            continue
        if not (item / "area_ground_truth.json").is_file():
            continue
        if not (item / "metrics.json").is_file():
            continue
        candidates.append(item)
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.name))


def load_old_metrics(report_dir: Path) -> dict[str, Any]:
    path = report_dir / "metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "overall": payload.get("overall") or {},
        "area_metrics": payload.get("area_metrics") or {},
        "model_profile": payload.get("model_profile") or {},
        "training": {
            "completed_epochs": (payload.get("training") or {}).get("completed_epochs"),
            "best_checkpoint_epoch": (payload.get("training") or {}).get("best_checkpoint_epoch"),
        },
        "metrics_json": str(path),
    }


def validate_image_mapping(
    d27,
    records,
    ground_truth_path: Path,
    model_name: str,
) -> None:
    saved_gt = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    saved_images = saved_gt.get("images") if isinstance(saved_gt, dict) else None
    if not isinstance(saved_images, list) or len(saved_images) != len(records):
        raise ValueError(f"Unexpected image table in {ground_truth_path}")
    for record, saved in zip(records, saved_images):
        if (
            int(saved.get("id", -1)) != record.image_id
            or Path(str(saved.get("file_name", ""))).stem != record.stem
            or int(saved.get("width", -1)) != record.width
            or int(saved.get("height", -1)) != record.height
        ):
            raise ValueError(
                f"Prediction image-id mapping mismatch for {model_name}: "
                f"record={record.image_id}/{record.stem}/{record.width}x{record.height}, "
                f"saved={saved}"
            )


def score_band(delta_e1_pp: float) -> str:
    # Numeric screen only.  It does NOT reopen a CLOSED route.
    if delta_e1_pp >= 0.50:
        return "GE_0.50_PP"
    if delta_e1_pp >= 0.35:
        return "0.35_TO_0.50_PP"
    if delta_e1_pp >= 0.20:
        return "0.20_TO_0.35_PP"
    if delta_e1_pp >= 0.00:
        return "0_TO_0.20_PP"
    return "BELOW_E1"


def resolve_sources(project: Path, selected: list[str]) -> tuple[dict[str, Path], dict[str, str]]:
    found: dict[str, Path] = {}
    missing: dict[str, str] = {}
    for name in selected:
        report = find_latest_report(project, MODEL_SPECS[name])
        if report is None:
            missing[name] = (
                f"No formal report matching {MODEL_SPECS[name]} with "
                "area_predictions.json + area_ground_truth.json + metrics.json"
            )
        else:
            found[name] = report
    return found, missing


def load_d27_reference(d27_dir: Path) -> dict[str, Any]:
    path = d27_dir / "metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise ValueError(f"D27 reference is not completed: {path}")
    return payload


def preflight(
    project: Path,
    dataset: Path,
    selected: list[str],
) -> tuple[Any, Any, list, dict[str, Any], dict[str, Path], dict[str, str], dict[str, Any]]:
    d27, d27_script = load_d27_module(project)
    unit = d27.run_unit_tests()

    records, dataset_summary = d27.load_dataset(dataset)
    if dataset_summary.get("images") != d27.EXPECTED_IMAGES:
        raise ValueError(f"Unexpected dataset size: {dataset_summary.get('images')}")

    d27_dir = find_latest_completed_d27(project)
    d27_reference = load_d27_reference(d27_dir)

    # Strong dataset guard: D30 must use the same annotation+dimension mapping.
    old_dataset = (d27_reference.get("preflight") or {}).get("dataset") or {}
    old_hash = old_dataset.get("annotation_dimension_sha256")
    new_hash = dataset_summary.get("annotation_dimension_sha256")
    if old_hash != new_hash:
        raise RuntimeError(
            "Dataset mapping differs from D27: "
            f"D27={old_hash}, D30={new_hash}"
        )

    sources, missing = resolve_sources(project, selected)
    source_summary = {}

    for name, report_dir in sources.items():
        pred = report_dir / "area_predictions.json"
        gt = report_dir / "area_ground_truth.json"

        # Cheap JSON shape check.
        with pred.open("r", encoding="utf-8") as handle:
            first_nonspace = next(
                (
                    ch
                    for ch in iter(lambda: handle.read(1), "")
                    if not ch.isspace()
                ),
                "",
            )
        if first_nonspace != "[":
            raise ValueError(f"Prediction JSON does not start with a list: {pred}")

        validate_image_mapping(d27, records, gt, name)

        source_summary[name] = {
            "report_dir": str(report_dir),
            "prediction_path": str(pred),
            "prediction_bytes": pred.stat().st_size,
            "prediction_sha256": d27.sha256(pred),
            "ground_truth_path": str(gt),
            "ground_truth_sha256": d27.sha256(gt),
            "image_id_mapping": "passed",
        }

    # Before any expensive evaluation, verify that the six old D27 sources are
    # byte-identical whenever they are part of the D30 request.
    source_reproduction = {}
    old_sources = (d27_reference.get("preflight") or {}).get("sources") or {}
    for name in D27_REPRO_MODELS:
        if name not in source_summary:
            continue
        old = old_sources.get(name) or {}
        cur = source_summary[name]
        pred_ok = cur["prediction_sha256"] == old.get("prediction_sha256")
        gt_ok = cur["ground_truth_sha256"] == old.get("ground_truth_sha256")
        source_reproduction[name] = {
            "prediction_sha256_match": pred_ok,
            "ground_truth_sha256_match": gt_ok,
            "passed": bool(pred_ok and gt_ok),
        }
        if not pred_ok or not gt_ok:
            raise RuntimeError(
                f"D27 source hash reproduction failed for {name}: "
                f"{source_reproduction[name]}"
            )

    preflight_data = {
        "unit_tests": unit,
        "dataset": dataset_summary,
        "d27_script": str(d27_script),
        "d27_script_sha256": d27.sha256(d27_script),
        "d27_reference_report": str(d27_dir),
        "d27_reference_revision": d27_reference.get("revision"),
        "source_reproduction": source_reproduction,
        "sources": source_summary,
    }
    return d27, d27_reference, records, preflight_data, sources, missing, dataset_summary


def evaluate_all(d27, records, sources: dict[str, Path]) -> dict[str, Any]:
    models = {}
    total = len(sources)
    for idx, (name, report_dir) in enumerate(sources.items(), 1):
        pred = report_dir / "area_predictions.json"
        print(f"[{idx}/{total}] {name}: load {pred}", flush=True)
        raw = d27.load_predictions(pred, len(records))

        print(f"[{idx}/{total}] {name}: D27 class-0 ignore filter + audit", flush=True)
        filtered, ignore_audit = d27.filter_detections_and_audit(records, raw)

        print(f"[{idx}/{total}] {name}: official evaluator", flush=True)
        official = d27.official_evaluate(records, filtered)

        models[name] = {
            "source_report": str(report_dir),
            "prediction_path": str(pred),
            "prediction_sha256": d27.sha256(pred),
            "historical_status": HISTORICAL_STATUS.get(name, "UNKNOWN"),
            "historical_note": HISTORICAL_NOTE.get(name, ""),
            "old_metrics": load_old_metrics(report_dir),
            "ignore_audit": ignore_audit,
            "official": official,
        }

        del raw, filtered
        gc.collect()
    return models


def check_metric_reproduction(
    models: dict[str, Any],
    d27_reference: dict[str, Any],
) -> dict[str, Any]:
    gate = {
        "tolerance_percent": REPRO_TOLERANCE_PERCENT,
        "models": {},
        "passed": True,
    }
    old_models = d27_reference.get("models") or {}
    keys = ("AP50_95_percent", "AP50_percent", "AP75_percent", "AR500_percent")

    for name in D27_REPRO_MODELS:
        if name not in models:
            continue
        if name not in old_models:
            gate["models"][name] = {
                "passed": False,
                "error": "missing from historical D27 metrics",
            }
            gate["passed"] = False
            continue

        cur = models[name]["official"]
        old = old_models[name]["official"]
        diffs = {key: float(cur[key]) - float(old[key]) for key in keys}
        passed = all(abs(v) <= REPRO_TOLERANCE_PERCENT for v in diffs.values())
        gate["models"][name] = {
            "differences_percent": diffs,
            "passed": passed,
        }
        gate["passed"] = bool(gate["passed"] and passed)

    return gate


def add_deltas(models: dict[str, Any]) -> None:
    if "e1" not in models:
        raise RuntimeError("E1 must be present in D30 for delta computation.")
    e1 = float(models["e1"]["official"]["AP50_95_percent"])
    e0 = (
        float(models["e0"]["official"]["AP50_95_percent"])
        if "e0" in models
        else math.nan
    )
    for name, row in models.items():
        ap = float(row["official"]["AP50_95_percent"])
        delta_e1 = ap - e1
        row["delta_vs_e1_pp"] = delta_e1
        row["delta_vs_e0_pp"] = ap - e0 if math.isfinite(e0) else None
        row["score_band_vs_e1"] = score_band(delta_e1)


def rank(models: dict[str, Any]) -> list[str]:
    return sorted(
        models,
        key=lambda name: float(models[name]["official"]["AP50_95_percent"]),
        reverse=True,
    )


def safe_percent_fraction(value: Any) -> float | None:
    try:
        return float(value) * 100.0
    except (TypeError, ValueError):
        return None


def write_outputs(
    report_dir: Path,
    report: dict[str, Any],
    reproduction_gate: dict[str, Any],
    missing: dict[str, str],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=False)

    write_json(report_dir / "metrics.json", report)
    write_json(report_dir / "reproduction_gate.json", reproduction_gate)
    write_json(report_dir / "missing_sources.json", missing)

    fields = [
        "rank",
        "model",
        "historical_status",
        "official_AP50_95_percent",
        "delta_vs_e1_pp",
        "delta_vs_e0_pp",
        "official_AP50_percent",
        "official_AP75_percent",
        "official_AR500_percent",
        "score_band_vs_e1",
        "old_ultralytics_mAP50_95_percent",
        "old_custom_AP_all_percent",
        "old_custom_AP_small_percent",
        "old_custom_AP_medium_percent",
        "old_custom_AP_large_percent",
        "parameters",
        "GFLOPs",
        "completed_epochs",
        "source_report",
    ]
    ranking = report["ranking"]
    with (report_dir / "model_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for position, name in enumerate(ranking, 1):
            row = report["models"][name]
            old = row.get("old_metrics") or {}
            overall = old.get("overall") or {}
            area = old.get("area_metrics") or {}
            profile = old.get("model_profile") or {}
            training = old.get("training") or {}
            writer.writerow(
                {
                    "rank": position,
                    "model": name,
                    "historical_status": row["historical_status"],
                    "official_AP50_95_percent": row["official"]["AP50_95_percent"],
                    "delta_vs_e1_pp": row["delta_vs_e1_pp"],
                    "delta_vs_e0_pp": row["delta_vs_e0_pp"],
                    "official_AP50_percent": row["official"]["AP50_percent"],
                    "official_AP75_percent": row["official"]["AP75_percent"],
                    "official_AR500_percent": row["official"]["AR500_percent"],
                    "score_band_vs_e1": row["score_band_vs_e1"],
                    "old_ultralytics_mAP50_95_percent": safe_percent_fraction(
                        overall.get("mAP50_95")
                    ),
                    "old_custom_AP_all_percent": safe_percent_fraction(area.get("AP_all")),
                    "old_custom_AP_small_percent": safe_percent_fraction(area.get("AP_small")),
                    "old_custom_AP_medium_percent": safe_percent_fraction(area.get("AP_medium")),
                    "old_custom_AP_large_percent": safe_percent_fraction(area.get("AP_large")),
                    "parameters": profile.get("parameters"),
                    "GFLOPs": profile.get("GFLOPs"),
                    "completed_epochs": training.get("completed_epochs"),
                    "source_report": row["source_report"],
                }
            )

    with (report_dir / "official_per_class.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fields_pc = [
            "model",
            "class_id",
            "class_name",
            "images_with_class_weight",
            "AP50_95_percent",
            "AP50_percent",
            "AP75_percent",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields_pc)
        writer.writeheader()
        for name in ranking:
            for cls, row in report["models"][name]["official"]["per_class"].items():
                writer.writerow(
                    {
                        "model": name,
                        "class_id": cls,
                        "class_name": row["name"],
                        "images_with_class_weight": row["images_with_class_weight"],
                        "AP50_95_percent": row["AP50_95_percent"],
                        "AP50_percent": row["AP50_percent"],
                        "AP75_percent": row["AP75_percent"],
                    }
                )

    lines = [
        "D30 Existing-Model Official VisDrone Sweep",
        f"status: {report['status']}",
        f"revision: {REVISION}",
        f"D27 evaluator revision: {EXPECTED_D27_REVISION}",
        "ZERO TRAINING. ZERO CHECKPOINT MODIFICATION. Existing formal predictions only.",
        "",
        f"reproduction_gate_passed: {reproduction_gate['passed']}",
        f"evaluated_models: {len(report['models'])}",
        f"missing_models: {len(missing)}",
        "",
        "[official ranking]",
    ]
    for pos, name in enumerate(ranking, 1):
        row = report["models"][name]
        o = row["official"]
        lines.append(
            f"{pos:02d}. {name:<9} "
            f"AP={o['AP50_95_percent']:.4f}% "
            f"dE1={row['delta_vs_e1_pp']:+.4f}pp "
            f"AP50={o['AP50_percent']:.4f}% "
            f"AP75={o['AP75_percent']:.4f}% "
            f"AR500={o['AR500_percent']:.4f}% "
            f"status={row['historical_status']} "
            f"band={row['score_band_vs_e1']}"
        )

    lines += [
        "",
        "[missing formal saved predictions]",
    ]
    if missing:
        for name, reason in missing.items():
            lines.append(f"- {name}: {reason}")
    else:
        lines.append("- none")

    lines += [
        "",
        "[hard boundary]",
        "- Ranking changes do NOT automatically reopen CLOSED routes.",
        "- E21a/E25a remain closed regardless of ranking.",
        "- E27a is excluded because the formal run was interrupted.",
        "- D30 does not regenerate missing predictions and does not train.",
        "- Old custom AP-small/medium/large remain auxiliary diagnostics, not official VisDrone scale metrics.",
        "",
        "Use model_summary.csv for the unified comparison table.",
    ]

    (report_dir / "metrics.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--models",
        default=",".join(MODEL_SPECS.keys()),
        help="Comma-separated model subset. Default: all D30 historical formal models.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate D27 evaluator, dataset, source reports and D27 source hashes; no evaluation.",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail if any requested model lacks formal saved predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    dataset = args.dataset.resolve()

    requested = [x.strip().lower() for x in args.models.split(",") if x.strip()]
    unknown = sorted(set(requested) - set(MODEL_SPECS))
    if unknown:
        raise ValueError(f"Unknown D30 model names: {unknown}")
    # Preserve MODEL_SPECS order even if --models comes in another order.
    selected = [name for name in MODEL_SPECS if name in set(requested)]

    started = time.perf_counter()
    report_dir = (
        project
        / "comparison_reports"
        / f"DIAG_D30_EXISTING_OFFICIAL_SWEEP_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )

    try:
        (
            d27,
            d27_reference,
            records,
            preflight_data,
            sources,
            missing,
            dataset_summary,
        ) = preflight(project, dataset, selected)

        if args.strict_missing and missing:
            raise RuntimeError(f"Missing requested formal prediction sources: {missing}")

        print("D30 preflight passed.", flush=True)
        print(f"  D27 evaluator: {preflight_data['d27_script']}", flush=True)
        print(f"  dataset images: {len(records)}", flush=True)
        print(f"  available models ({len(sources)}): {','.join(sources)}", flush=True)
        print(f"  missing models ({len(missing)}): {','.join(missing) if missing else 'none'}", flush=True)

        if "e1" not in sources:
            raise RuntimeError("E1 source is required for D30 delta-vs-E1 reporting.")

        if args.check_only:
            print("\n[CHECK-ONLY] Sources")
            for name, row in preflight_data["sources"].items():
                print(
                    f"  {name:<9} {row['prediction_bytes']:>10} bytes "
                    f"{row['prediction_sha256'][:12]}... "
                    f"{row['report_dir']}"
                )
            if missing:
                print("\n[CHECK-ONLY] Missing")
                for name, reason in missing.items():
                    print(f"  {name}: {reason}")
            print(
                "\nNo official evaluation, inference, training, checkpoint change, or shutdown was performed."
            )
            return

        models = evaluate_all(d27, records, sources)
        reproduction_gate = check_metric_reproduction(models, d27_reference)
        if not reproduction_gate["passed"]:
            # Save a failure package because continuing to interpret new models
            # would be unsafe if the six old D27 numbers no longer reproduce.
            report_dir.mkdir(parents=True, exist_ok=False)
            write_json(report_dir / "reproduction_gate.json", reproduction_gate)
            write_json(report_dir / "missing_sources.json", missing)
            raise RuntimeError(
                f"D27 metric reproduction gate failed; see {report_dir / 'reproduction_gate.json'}"
            )

        add_deltas(models)
        ranking = rank(models)

        report = {
            "status": "completed",
            "revision": REVISION,
            "started_at": datetime.now().astimezone().isoformat(),
            "finished_at": datetime.now().astimezone().isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "zero_training": True,
            "zero_inference": True,
            "zero_checkpoint_modification": True,
            "automatic_shutdown": False,
            "project": str(project),
            "dataset": str(dataset),
            "protocol": {
                "evaluator_imported_from": preflight_data["d27_script"],
                "d27_revision": EXPECTED_D27_REVISION,
                "official_metric": "VisDrone DET Matlab toolkit v1.0.4 semantics via D27 clean-room Python port",
                "primary_metrics": ["AP50_95_percent", "AP50_percent", "AP75_percent", "AR500_percent"],
                "max_det": 500,
                "class0_ignore_filter": "same D27 implementation",
                "class11_policy": "same D27 implementation: audit only, not added to primary class-0 raster filter",
            },
            "preflight": preflight_data,
            "missing_sources": missing,
            "reproduction_gate": reproduction_gate,
            "models": models,
            "ranking": ranking,
            "decision_boundary": {
                "ranking_does_not_reopen_closed_routes": True,
                "e21a_e25a_remain_closed": True,
                "e27a_excluded_incomplete_formal_run": True,
                "missing_predictions_are_not_regenerated": True,
                "score_band_is_numeric_screen_only": True,
            },
        }

        write_outputs(report_dir, report, reproduction_gate, missing)

        print("\nD30 completed.", flush=True)
        print(f"Report: {report_dir / 'metrics.txt'}", flush=True)
        print(f"CSV:    {report_dir / 'model_summary.csv'}", flush=True)
        print("\nTop official results:", flush=True)
        for pos, name in enumerate(ranking[:10], 1):
            row = models[name]
            print(
                f"{pos:02d}. {name:<9} "
                f"AP={row['official']['AP50_95_percent']:.4f}% "
                f"dE1={row['delta_vs_e1_pp']:+.4f}pp "
                f"AP50={row['official']['AP50_percent']:.4f}% "
                f"AP75={row['official']['AP75_percent']:.4f}% "
                f"status={row['historical_status']}",
                flush=True,
            )
        print("\nNo training or inference was performed. Automatic shutdown is disabled.", flush=True)

    except Exception:
        # If a normal report directory has not yet been created, save a compact
        # failure record so the run is auditable.
        if not report_dir.exists():
            report_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "failed",
            "revision": REVISION,
            "error": traceback.format_exc().splitlines()[-1],
            "traceback": traceback.format_exc(),
            "finished_at": datetime.now().astimezone().isoformat(),
            "zero_training": True,
            "zero_inference": True,
            "automatic_shutdown": False,
        }
        write_json(report_dir / "failure.json", failure)
        print(f"D30 failed; failure record: {report_dir / 'failure.json'}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
