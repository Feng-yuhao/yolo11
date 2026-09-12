#!/usr/bin/env python3
"""D16b one-sided CVRC correction using frozen D16a fold artifacts.

D16a allowed local scores to move up or down and produced many high-score false
positives. D16b performs no training. It reuses each grouped-OOF calibrator and
its inner-selected alpha, but applies only the negative residual:

    new_logit = base_logit + alpha * min(0, corrected_logit - base_logit)

Thus no local score can increase. E1, D15a routing, full-view detections,
Ultralytics, and all saved checkpoints remain unchanged. No shutdown occurs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import evaluate_e1_e5a_slicing as core


ROOT = Path("/root/autodl-tmp/yolo11_project")
REPORTS = ROOT / "comparison_reports"
WEIGHT = ROOT / "runs/e1_yolo11s_p2add_img1024_seed1/weights/best.pt"
D16A_GLOB = "DIAG_D16A_e1_cvrc_oof_*/metrics.json"
SCRIPT_REVISION = "d16b_frozen_oof_negative_residual_only_v1"
REFERENCE_TOLERANCE = 0.002
MAP_GATE = 0.002
APS_GATE = 0.003
TP_LOSS_LIMIT = 0.01
FP_REDUCTION_REQUIRED = 1


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def latest_d16a() -> tuple[Path, dict]:
    completed = []
    for path in sorted(REPORTS.glob(D16A_GLOB)):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        report_dir = path.parent
        artifacts = [report_dir / f"cvrc_fold{fold}.pt" for fold in range(1, 6)]
        script = report_dir / "diagnose_d16a_cvrc_oof.py"
        if (
            value.get("status") == "completed"
            and value.get("is_full_diagnostic") is True
            and len(value.get("folds", [])) == 5
            and script.is_file()
            and all(item.is_file() for item in artifacts)
        ):
            completed.append((path, value))
    if not completed:
        raise FileNotFoundError("No complete D16a OOF report with five fold artifacts found")
    return completed[-1]


def load_d16a(report_path: Path):
    script = report_path.parent / "diagnose_d16a_cvrc_oof.py"
    spec = importlib.util.spec_from_file_location("d16a_frozen_core", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen D16a script: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.SCRIPT_REVISION != "d16a_grouped_nested_oof_cvrc_v1":
        raise RuntimeError(f"Unexpected D16a revision: {module.SCRIPT_REVISION}")
    return module


def save(report: dict) -> None:
    report["updated_at"] = now()
    out = Path(report["report_dir"])
    core.atomic_json(out / "metrics.json", report)
    core.atomic_text(out / "metrics.txt", report_text(report))


def report_text(report: dict) -> str:
    lines = [
        "D16b frozen one-sided cross-view candidate suppression",
        f"status: {report.get('status')}",
        f"revision: {SCRIPT_REVISION}",
        f"started_at: {report.get('started_at')}",
        f"finished_at: {report.get('finished_at')}",
        "",
        "No training. No local score may exceed its original D15a score.",
        "E1, D15a, D16a fold artifacts and Ultralytics remain unchanged.",
        "",
        "[modes]",
    ]
    for name, row in report.get("modes", {}).items():
        metrics = row.get("overall", {})
        fixed = row.get("fixed_operating_point", {}).get("overall", {})
        lines.append(
            f"{name}: mAP50_95={metrics.get('mAP50_95')}, "
            f"AP_small={metrics.get('AP_small')}, AR_small={metrics.get('AR_small')}, "
            f"AP_medium={metrics.get('AP_medium')}, TP={fixed.get('tp')}, "
            f"FP={fixed.get('fp')}, Precision={fixed.get('Precision')}"
        )
    lines.extend([
        "", "[decision]",
        json.dumps(report.get("decision"), ensure_ascii=False, indent=2),
        "", "[score_audit]",
        json.dumps(report.get("score_audit"), ensure_ascii=False, indent=2),
        "", "[full_metadata]",
        json.dumps(report, ensure_ascii=False, indent=2, default=core.json_default),
    ])
    return "\n".join(lines) + "\n"


def artifact_path(reference_path: Path, fold: int) -> Path:
    return reference_path.parent / f"cvrc_fold{fold + 1}.pt"


def load_fold_model(d16a, reference_path: Path, fold: int):
    artifact = core.torch.load(
        artifact_path(reference_path, fold), map_location="cpu", weights_only=False
    )
    if tuple(artifact.get("feature_names", ())) != tuple(d16a.FEATURE_NAMES):
        raise RuntimeError(f"Fold {fold} feature schema mismatch")
    model = d16a.make_model(len(d16a.FEATURE_NAMES))
    model.load_state_dict(artifact["model_state"], strict=True)
    model = model.cuda().eval()
    alpha = float(artifact["alpha"])
    if not 0.0 <= alpha <= 1.0:
        raise RuntimeError(f"Fold {fold} has invalid alpha {alpha}")
    return model, artifact, alpha


def one_sided_scores(d16a, model, artifact, alpha: float, row: dict):
    base = row["local"][:, 4].numpy()
    if not len(base):
        return base.astype(core.np.float32), {
            "candidates": 0, "suppressed": 0, "raised": 0,
            "above_025_before": 0, "above_025_after": 0,
        }
    corrected = d16a.predict_logits(
        model, row["features"], artifact["mean"], artifact["std"]
    )
    clipped = core.np.clip(base, 1e-5, 1 - 1e-5)
    base_logits = core.np.log(clipped / (1.0 - clipped))
    residual = corrected - base_logits
    new_logits = base_logits + alpha * core.np.minimum(residual, 0.0)
    scores = 1.0 / (1.0 + core.np.exp(-core.np.clip(new_logits, -20, 20)))
    if core.np.any(scores > base + 1e-6):
        raise RuntimeError("One-sided invariant failed: a local score increased")
    return scores.astype(core.np.float32), {
        "candidates": int(len(base)),
        "suppressed": int(core.np.sum(scores < base - 1e-6)),
        "raised": int(core.np.sum(scores > base + 1e-6)),
        "above_025_before": int(core.np.sum(base >= 0.25)),
        "above_025_after": int(core.np.sum(scores >= 0.25)),
        "mean_before": float(core.np.mean(base)),
        "mean_after": float(core.np.mean(scores)),
        "mean_negative_residual": float(core.np.mean(core.np.minimum(residual, 0.0))),
    }


def sum_audit(rows: list[dict]) -> dict:
    additive = ("candidates", "suppressed", "raised", "above_025_before", "above_025_after")
    result = {key: sum(row[key] for row in rows) for key in additive}
    candidates = max(1, result["candidates"])
    result["suppressed_fraction"] = result["suppressed"] / candidates
    result["high_score_removed"] = result["above_025_before"] - result["above_025_after"]
    return result


def metric_delta(one: dict, zero: dict) -> dict:
    keys = (
        "mAP50_95", "mAP50", "mAP75", "AR_all", "AP_small", "AR_small",
        "AP_medium", "AR_medium", "AP_large", "AR_large",
    )
    return {key: one[key] - zero[key] for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke2", action="store_true", help="Run on 12 images; metrics cannot drive decisions")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = "CHECK_D16B" if args.check_only else "SMOKE2_D16B" if args.smoke2 else "DIAG_D16B"
    out = REPORTS / f"{prefix}_e1_onesided_cvrc_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "initializing", "purpose": "frozen D16a negative-residual-only score correction",
        "script_revision": SCRIPT_REVISION, "started_at": now(), "report_dir": str(out),
        "is_full_diagnostic": not args.check_only and not args.smoke2,
        "protocol": {
            "formula": "new_logit = base_logit + fold_alpha * min(0, corrected_logit-base_logit)",
            "local_scores_can_increase": False, "training_performed": False,
            "e1_frozen": True, "d15a_router_frozen": True,
            "d16a_fold_artifacts_frozen": True, "ultralytics_modified": False,
            "automatic_shutdown": False,
        },
        "modes": {},
    }
    save(report)
    started = time.perf_counter()
    detector = None
    try:
        reference_path, reference = latest_d16a()
        d16a = load_d16a(reference_path)
        report["d16a_reference"] = {
            "path": str(reference_path), "sha256": core.sha256(reference_path),
            "decision": reference.get("decision"), "grouping": reference.get("grouping"),
        }
        d16a.preflight(report, args.check_only)
        for fold in range(5):
            calibrator, artifact, alpha = load_fold_model(d16a, reference_path, fold)
            x = core.torch.zeros((2, len(d16a.FEATURE_NAMES)), dtype=core.torch.float32)
            row = {
                "local": core.torch.tensor([[0, 0, 1, 1, 0.25, 0], [0, 0, 1, 1, 0.75, 1]], dtype=core.torch.float32),
                "features": x.numpy(),
            }
            scores, audit = one_sided_scores(d16a, calibrator, artifact, alpha, row)
            if audit["raised"] != 0 or core.np.any(scores > row["local"][:, 4].numpy() + 1e-6):
                raise RuntimeError(f"Fold {fold} one-sided synthetic check failed")
            del calibrator, artifact
        core.clear_gpu()
        report["status"] = "preflight_passed"
        save(report)
        if args.check_only:
            report.update(status="check_passed", finished_at=now(), total_seconds=time.perf_counter() - started)
            save(report)
            print("D16b artifacts/formula/invariant checks passed. No prediction, training, or shutdown.", flush=True)
            return

        limit = 12 if args.smoke2 else 0
        images, path_ids, gt, names = core.prepare_dataset(report, limit)
        annotations = d16a.annotation_index(gt)
        meta = d16a.image_meta(gt)
        detector = core.YOLO(str(WEIGHT))
        before_hash = core.sha256(WEIGHT)
        core.assert_model(detector, names, "e1")
        full, full_profile = core.predict_full(detector, images, report, "D16b E1")
        proxy_rows = d16a.build_proxy_rows(images, full, meta)
        proxy_by_path = {row["image"]: row for row in proxy_rows}
        selections, budget_meta = d16a.allocate_budget(proxy_rows, d16a.PRIMARY_BUDGET)
        records, candidate_profile = d16a.collect_candidates(
            detector, images, full, selections, proxy_by_path, meta, annotations, report
        )
        records_by_path = {row["path"]: row for row in records}
        report["router"] = budget_meta
        report["candidate_profile"] = candidate_profile

        if args.smoke2:
            group_map, _folds = d16a.balanced_group_folds(records, 2, True)
            fold_for_path = {
                row["path"]: group_map[d16a.sequence_group(row["path"], True)]
                for row in records
            }
        else:
            reference_groups = reference.get("grouping", {}).get("group_to_fold", {})
            if not reference_groups:
                raise RuntimeError("D16a grouping map is missing")
            fold_for_path = {}
            for row in records:
                group = d16a.sequence_group(row["path"], False)
                if group not in reference_groups:
                    raise RuntimeError(f"Sequence group absent from D16a reference: {group}")
                fold_for_path[row["path"]] = int(reference_groups[group])

        score_map, audit_rows = {}, []
        for fold in sorted(set(fold_for_path.values())):
            calibrator, artifact, alpha = load_fold_model(d16a, reference_path, fold)
            for row in records:
                if fold_for_path[row["path"]] != fold:
                    continue
                scores, audit = one_sided_scores(d16a, calibrator, artifact, alpha, row)
                score_map[row["path"]] = scores
                audit_rows.append({"path": row["path"], "fold": fold, "alpha": alpha, **audit})
            del calibrator, artifact
            core.clear_gpu()
        if set(score_map) != {str(path) for path in images}:
            raise RuntimeError("One-sided scoring did not cover every image exactly once")
        report["score_audit"] = sum_audit(audit_rows)
        core.atomic_json(out / "score_audit.json", {
            "summary": report["score_audit"], "per_image": audit_rows,
        })

        standard_predictions = d16a.compose_predictions(images, full, records_by_path)
        corrected_predictions = d16a.compose_predictions(images, full, records_by_path, score_map)
        report["modes"]["d15a_proxy_b1p50_standard"] = d16a.evaluate_predictions(
            "d15a_proxy_b1p50_standard", standard_predictions, images, path_ids, gt, names
        )
        report["modes"]["d16b_onesided_cvrc"] = d16a.evaluate_predictions(
            "d16b_onesided_cvrc", corrected_predictions, images, path_ids, gt, names
        )
        standard = report["modes"]["d15a_proxy_b1p50_standard"]
        corrected = report["modes"]["d16b_onesided_cvrc"]
        gain = metric_delta(corrected["overall"], standard["overall"])
        fixed_zero = standard["fixed_operating_point"]["overall"]
        fixed_one = corrected["fixed_operating_point"]["overall"]
        tp_loss_fraction = max(0, fixed_zero["tp"] - fixed_one["tp"]) / max(1, fixed_zero["tp"])
        fp_reduction = fixed_zero["fp"] - fixed_one["fp"]
        if args.smoke2:
            report["decision"] = {"disabled": "Smoke metrics cannot drive research decisions"}
        else:
            d16a_standard = reference["modes"]["d15a_proxy_b1p50_standard"]["overall"]
            parity = {
                key: standard["overall"][key] - d16a_standard[key]
                for key in ("mAP50_95", "AP_small", "AR_small", "AP_medium", "AP_large")
            }
            parity_pass = all(abs(value) <= REFERENCE_TOLERANCE for value in parity.values())
            passed = bool(
                parity_pass and gain["mAP50_95"] >= MAP_GATE
                and gain["AP_small"] >= APS_GATE
                and tp_loss_fraction <= TP_LOSS_LIMIT
                and fp_reduction >= FP_REDUCTION_REQUIRED
            )
            report["decision"] = {
                "d15a_reproduction": {"delta": parity, "tolerance": REFERENCE_TOLERANCE, "passed": parity_pass},
                "d16b_minus_d15a": gain,
                "fixed_operating_point": {
                    "tp_before": fixed_zero["tp"], "tp_after": fixed_one["tp"],
                    "tp_loss_fraction": tp_loss_fraction,
                    "fp_before": fixed_zero["fp"], "fp_after": fixed_one["fp"],
                    "fp_reduction": fp_reduction,
                },
                "gate": {
                    "mAP50_95": MAP_GATE, "AP_small": APS_GATE,
                    "tp_loss_fraction_max": TP_LOSS_LIMIT,
                    "fp_reduction_min": FP_REDUCTION_REQUIRED,
                },
                "passed": passed,
                "selected_branch": "formalize_onesided_cvrc" if passed else "stop_cvrc_keep_d15a",
            }
        if core.sha256(WEIGHT) != before_hash:
            raise RuntimeError("Frozen E1 checkpoint changed during D16b")
        report["checkpoint_unchanged"] = True
        report["profile"] = {
            "full_prediction": full_profile, "total_seconds": time.perf_counter() - started,
            "training_performed": False,
        }
        report.update(status="completed", finished_at=now(), total_seconds=time.perf_counter() - started)
        shutil.copy2(Path(__file__), out / Path(__file__).name)
        save(report)
        print(f"D16b completed. Report: {out / 'metrics.txt'}", flush=True)
    except BaseException:
        report.update(status="failed_or_interrupted", finished_at=now(), total_seconds=time.perf_counter() - started, error=traceback.format_exc())
        save(report)
        core.atomic_text(out / "error.log", report["error"])
        print(f"D16b failure saved: {out}", file=sys.stderr, flush=True)
        raise
    finally:
        if detector is not None:
            del detector
        core.clear_gpu()
        try:
            core.atomic_json(out / "shutdown_status.json", {
                "time": now(), "enabled": False,
                "status": "disabled_by_user_policy", "reason": report.get("status"),
            })
            save(report)
        except Exception:
            pass
        print("Automatic shutdown is disabled; the instance remains running.", flush=True)


if __name__ == "__main__":
    main()
