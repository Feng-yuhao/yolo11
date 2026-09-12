#!/usr/bin/env python3
"""D15a frozen-E1 dynamic regional-zoom budget diagnostic.

This is a falsifiable decision experiment, not a new detector training run and
not yet a publishable method.  It answers one question before more GPU money is
spent: can prediction-only, image-wise allocation of 0/1/2 local crops retain
the known Top-2/four-tile small-object gain at a lower average crop budget?

All four local crops are inferred once and cached in CPU memory.  Every routing
policy is then evaluated from exactly the same detections, so policy comparisons
contain no repeated-inference noise.  Ground-truth labels are used only by the
COCO evaluator, never by the router or its budget threshold.
"""

from __future__ import annotations

import argparse
import gc
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

D1_GLOB = "DIAG_SLICING_both_*/metrics.json"
D2_GLOB = "DIAG_D2_e1_adaptive_budget_*/metrics.json"

ROUTER_CONF = 0.05
ROUTER_SMALL_EFFECTIVE_SIDE = 32.0
ROUTER_UNCERTAINTY_WEIGHT = 0.5
DYNAMIC_BUDGETS = (0.50, 0.75, 1.00, 1.25, 1.50, 1.75)
PRIMARY_BUDGET_MODE = "adaptive_b1p50"
SAME_BUDGET_MODE = "adaptive_b1p00"
REFERENCE_TOLERANCE = 0.002

SCRIPT_REVISION = "d15a_frozen_e1_cached_marginal_budget_v1"
EXPECTED_CORE_REVISION = "fixed_2x2_slicing_upper_bound_v2"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def budget_name(value: float) -> str:
    return f"adaptive_b{value:.2f}".replace(".", "p")


def save(report: dict) -> None:
    report["updated_at"] = now()
    out = Path(report["report_dir"])
    core.atomic_json(out / "metrics.json", report)
    core.atomic_text(out / "metrics.txt", report_text(report))


def report_text(report: dict) -> str:
    lines = [
        "D15a frozen-E1 dynamic regional-zoom budget diagnostic",
        f"status: {report.get('status')}",
        f"revision: {SCRIPT_REVISION}",
        f"started_at: {report.get('started_at')}",
        f"finished_at: {report.get('finished_at')}",
        "",
        "All AP/AR values are 0-1; multiply by 100 for percentage points.",
        "Routing uses frozen E1 predictions only. Ground truth is evaluator-only.",
        "Thresholds are calibrated only to crop count on this split, not to AP.",
        "For a paper, calibrate thresholds on train and freeze them before val/test.",
        "This uses converted YOLO labels, not official VisDrone ignore handling.",
        "",
        "[modes]",
    ]
    modes = report.get("models", {}).get("e1", {}).get("modes", {})
    for mode, row in modes.items():
        overall = row.get("overall", {})
        profile = row.get("profile", {})
        lines.append(
            f"{mode}: crops/image={profile.get('local_crops_per_image')}, "
            f"forward_multiplier={profile.get('forward_image_multiplier')}, "
            + ", ".join(
                f"{key}={overall.get(key)}"
                for key in (
                    "mAP50_95", "mAP50", "mAP75", "AP_small", "AR_small",
                    "AP_medium", "AP_large",
                )
            )
        )
    lines.extend([
        "",
        "[decisions]",
        json.dumps(report.get("decisions"), ensure_ascii=False, indent=2),
        "",
        "[full_metadata]",
        json.dumps(report, ensure_ascii=False, indent=2, default=core.json_default),
    ])
    return "\n".join(lines) + "\n"


def latest_completed(pattern: str, required_path: tuple[str, ...]) -> tuple[Path, dict] | None:
    completed = []
    for path in sorted(REPORTS.glob(pattern)):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cursor = value
        for key in required_path:
            if not isinstance(cursor, dict) or key not in cursor:
                cursor = None
                break
            cursor = cursor[key]
        if value.get("status") == "completed" and cursor is not None:
            completed.append((path, value))
    return completed[-1] if completed else None


def load_references() -> dict:
    references = {}
    d1 = latest_completed(D1_GLOB, ("models", "e1", "modes", "combined"))
    d2 = latest_completed(D2_GLOB, ("models", "e1", "modes", "proxy_top2"))
    if d1:
        path, value = d1
        references["d1"] = {
            "path": str(path),
            "script_revision": value.get("script_revision"),
            "full": value["models"]["e1"]["modes"]["full"]["overall"],
            "fixed_top4": value["models"]["e1"]["modes"]["combined"]["overall"],
        }
    if d2:
        path, value = d2
        modes = value["models"]["e1"]["modes"]
        references["d2"] = {
            "path": str(path),
            "script_revision": value.get("script_revision"),
            "full": modes["full"]["overall"],
            "fixed_top1": modes["proxy_top1"]["overall"],
            "fixed_top2": modes["proxy_top2"]["overall"],
        }
    return references


def image_meta(gt: dict) -> dict[str, dict]:
    return {
        str(row["file_name"]): {
            "image_id": int(row["id"]),
            "width": int(row["width"]),
            "height": int(row["height"]),
        }
        for row in gt["images"]
    }


def assigned_tile(cx: float, cy: float, width: int, height: int) -> int:
    column = 0 if cx < width / 2 else 1
    row = 0 if cy < height / 2 else 1
    return row * 2 + column


def proxy_scores(detections, meta: dict) -> tuple[list[float], list[int]]:
    """Frozen D2 proxy: small/uncertain full detections vote for one quadrant."""
    scores = [0.0] * 4
    counts = [0] * 4
    resize_scale = core.IMGSZ / max(meta["width"], meta["height"])
    for x1, y1, x2, y2, confidence, _cls in detections.tolist():
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        effective_side = math.sqrt(width * height) * resize_scale
        if confidence < ROUTER_CONF or effective_side >= ROUTER_SMALL_EFFECTIVE_SIDE:
            continue
        tile = assigned_tile(
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            meta["width"],
            meta["height"],
        )
        evidence = math.sqrt(max(confidence, 0.0))
        scale_deficit = max(0.0, 1.0 - effective_side / ROUTER_SMALL_EFFECTIVE_SIDE)
        uncertainty = 4.0 * confidence * (1.0 - confidence)
        scores[tile] += evidence * (1.0 + scale_deficit) * (
            1.0 + ROUTER_UNCERTAINTY_WEIGHT * uncertainty
        )
        counts[tile] += 1
    return scores, counts


def build_score_rows(images: list[Path], full: dict, meta_by_path: dict) -> list[dict]:
    rows = []
    for path in images:
        key = str(path)
        scores, counts = proxy_scores(full[key], meta_by_path[key])
        order = sorted(range(4), key=lambda index: (-scores[index], index))
        ranked = [scores[index] for index in order]
        rows.append({
            "image": key,
            "tile_scores": scores,
            "candidate_counts": counts,
            "ranked_tiles": order,
            "ranked_scores": ranked,
            "total_score": float(sum(scores)),
            "second_to_first_ratio": (
                float(ranked[1] / ranked[0]) if ranked[0] > 0 else 0.0
            ),
        })
    return rows


def fixed_selections(rows: list[dict], count: int) -> dict[str, list[int]]:
    if count == 4:
        return {row["image"]: [0, 1, 2, 3] for row in rows}
    return {row["image"]: row["ranked_tiles"][:count] for row in rows}


def marginal_budget_selections(
    rows: list[dict], budget: float
) -> tuple[dict[str, list[int]], dict]:
    """Allocate a global crop budget using only rank-1/rank-2 proxy evidence.

    Each image contributes two marginal candidates.  Because score(rank 1) is
    never below score(rank 2), globally taking the largest marginals preserves
    the prefix constraint: a second crop cannot be selected without the first.
    The resulting cutoff is a deployment threshold that must later be calibrated
    on train, rather than on validation labels.
    """
    requested = int(round(budget * len(rows)))
    marginals = []
    for row in rows:
        for rank in (0, 1):
            marginals.append({
                "score": float(row["ranked_scores"][rank]),
                "rank": rank,
                "image": row["image"],
                "tile": int(row["ranked_tiles"][rank]),
            })
    marginals.sort(key=lambda item: (-item["score"], item["rank"], item["image"]))
    chosen = marginals[:requested]
    chosen_ranks: dict[str, set[int]] = {}
    for item in chosen:
        chosen_ranks.setdefault(item["image"], set()).add(item["rank"])
    by_image = {row["image"]: row for row in rows}
    selections = {}
    distribution = {"0": 0, "1": 0, "2": 0}
    for row in rows:
        ranks = chosen_ranks.get(row["image"], set())
        if 1 in ranks and 0 not in ranks:
            raise RuntimeError("Marginal allocation violated rank-prefix constraint")
        count = len(ranks)
        selections[row["image"]] = by_image[row["image"]]["ranked_tiles"][:count]
        distribution[str(count)] += 1
    actual = sum(len(value) for value in selections.values())
    if actual != requested:
        raise RuntimeError(f"Requested {requested} local crops, allocated {actual}")
    calibration = {
        "target_crops_per_image": budget,
        "target_total_crops": requested,
        "actual_total_crops": actual,
        "actual_crops_per_image": actual / len(rows),
        "images_by_crop_count": distribution,
        "last_selected_marginal_score": chosen[-1]["score"] if chosen else None,
        "first_rejected_marginal_score": (
            marginals[requested]["score"] if requested < len(marginals) else None
        ),
        "uses_ground_truth": False,
        "calibration_warning": (
            "The cutoff only targets compute on this validation prediction distribution. "
            "Before a paper result, estimate the cutoff on train and freeze it."
        ),
    }
    return selections, calibration


def build_routes(rows: list[dict]) -> tuple[dict, dict]:
    routes = {
        "fixed_top1": fixed_selections(rows, 1),
        "fixed_top2": fixed_selections(rows, 2),
        "fixed_top4": fixed_selections(rows, 4),
    }
    metadata = {
        "fixed_top1": {"actual_crops_per_image": 1.0, "uses_ground_truth": False},
        "fixed_top2": {"actual_crops_per_image": 2.0, "uses_ground_truth": False},
        "fixed_top4": {"actual_crops_per_image": 4.0, "uses_ground_truth": False},
    }
    for budget in DYNAMIC_BUDGETS:
        name = budget_name(budget)
        routes[name], metadata[name] = marginal_budget_selections(rows, budget)
    return routes, metadata


def predict_all_tiles(model, images: list[Path], report: dict):
    """Infer each of four tiles exactly once and retain mapped CPU tensors."""
    cache = {}
    geometry_example = None
    core.torch.cuda.reset_peak_memory_stats(0)
    core.torch.cuda.synchronize()
    started = time.perf_counter()
    for image_index, path in enumerate(images, 1):
        image = core.cv2.imread(str(path), core.cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV failed to read {path}")
        tiles = core.make_tiles(image)
        if geometry_example is None:
            geometry_example = [
                {key: value for key, value in tile.items() if key != "image"}
                for tile in tiles
            ]
        results = core.model_predict(
            model, [tile["image"] for tile in tiles], len(tiles)
        )
        if len(results) != 4:
            raise RuntimeError("Expected four cached tile predictions per image")
        cache[str(path)] = [
            core.map_tile_detections(result, tile)
            for result, tile in zip(results, tiles)
        ]
        del results, tiles, image
        if image_index == 1 or image_index % 50 == 0 or image_index == len(images):
            print(f"D15a cached local tiles: {image_index}/{len(images)}", flush=True)
            report["progress"] = {"stage": "cache_all_four_tiles", "images": image_index}
            save(report)
    core.torch.cuda.synchronize()
    return cache, {
        "seconds": time.perf_counter() - started,
        "peak_memory": core.memory_peak(),
        "forward_images": 4 * len(images),
        "tiles_per_image": 4,
        "example_geometry": geometry_example,
    }


def compose_route(
    images: list[Path], full: dict, tile_cache: dict, selections: dict
) -> tuple[dict, dict]:
    combined = {}
    started = time.perf_counter()
    total_crops = 0
    counts = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}
    for path in images:
        key = str(path)
        selected = selections[key]
        total_crops += len(selected)
        counts[str(len(selected))] += 1
        mapped = [tile_cache[key][index] for index in selected]
        if mapped:
            local = core.nms_detections(core.torch.cat(mapped, dim=0))
        else:
            local = core.empty_detections()
        candidates = core.torch.cat([full[key], local], dim=0)
        combined[key] = core.nms_detections(candidates)
        del mapped, local, candidates
    return combined, {
        "merge_seconds": time.perf_counter() - started,
        "local_forward_images": total_crops,
        "local_crops_per_image": total_crops / len(images),
        "images_by_crop_count": counts,
    }


def evaluate_compact(
    mode_name: str,
    predictions: dict,
    images: list[Path],
    path_ids: dict,
    gt: dict,
    names: dict,
) -> dict:
    print(f"D15a evaluating {mode_name} on CPU.", flush=True)
    detections = core.predictions_to_coco(predictions, images, path_ids)
    metrics = core.coco_metrics(gt, detections, names)
    metrics["fixed_operating_point"] = core.fixed_operating_point(gt, detections, names)
    metrics["prediction_count"] = len(detections)
    return metrics


def metric_delta(one: dict, zero: dict) -> dict:
    keys = (
        "mAP50_95", "mAP50", "mAP75", "AR_all", "AP_small", "AR_small",
        "AP_medium", "AR_medium", "AP_large", "AR_large",
    )
    return {key: one[key] - zero[key] for key in keys}


def parity_checks(report: dict) -> dict:
    modes = report["models"]["e1"]["modes"]
    references = report.get("references", {})
    checks = {}
    mapping = []
    if "d2" in references:
        mapping.extend([
            ("full", "d2", "full"),
            ("fixed_top1", "d2", "fixed_top1"),
            ("fixed_top2", "d2", "fixed_top2"),
        ])
    if "d1" in references:
        mapping.append(("fixed_top4", "d1", "fixed_top4"))
    keys = ("mAP50_95", "AP_small", "AR_small", "AP_medium", "AP_large")
    for current_name, ref_group, ref_name in mapping:
        current = modes[current_name]["overall"]
        reference = references[ref_group][ref_name]
        delta = {key: current[key] - reference[key] for key in keys}
        checks[current_name] = {
            "reference": f"{ref_group}/{ref_name}",
            "current_minus_reference": delta,
            "absolute_tolerance": REFERENCE_TOLERANCE,
            "passed": all(abs(value) <= REFERENCE_TOLERANCE for value in delta.values()),
        }
    required = {"full", "fixed_top1", "fixed_top2", "fixed_top4"}
    return {
        "checks": checks,
        "all_required_available": set(checks) == required,
        "passed": set(checks) == required and all(row["passed"] for row in checks.values()),
    }


def build_decisions(report: dict) -> dict:
    modes = report["models"]["e1"]["modes"]
    full = modes["full"]["overall"]
    gains = {
        name: metric_delta(row["overall"], full)
        for name, row in modes.items() if name != "full"
    }
    parity = parity_checks(report)
    top2 = gains["fixed_top2"]
    primary = gains[PRIMARY_BUDGET_MODE]
    same_budget_delta = metric_delta(
        modes[SAME_BUDGET_MODE]["overall"], modes["fixed_top1"]["overall"]
    )

    def retained(key: str) -> float | None:
        return primary[key] / top2[key] if top2[key] > 0 else None

    retained_top2 = {
        key: retained(key) for key in ("mAP50_95", "AP_small", "AR_small")
    }
    regional_zoom_causal = bool(
        top2["mAP50_95"] >= 0.020
        and top2["AP_small"] >= 0.030
        and top2["AR_small"] >= 0.035
    )
    redistribution_helpful = bool(
        same_budget_delta["mAP50_95"] >= 0.001
        and same_budget_delta["AP_small"] >= 0.002
        and same_budget_delta["AR_small"] >= -0.002
    )
    primary_efficient = bool(
        modes[PRIMARY_BUDGET_MODE]["profile"]["local_crops_per_image"] <= 1.500001
        and primary["mAP50_95"] >= 0.020
        and primary["AP_small"] >= 0.030
        and primary["AR_small"] >= 0.030
        and retained_top2["mAP50_95"] is not None
        and retained_top2["AP_small"] is not None
        and retained_top2["mAP50_95"] >= 0.85
        and retained_top2["AP_small"] >= 0.85
        and primary["AP_medium"] >= -0.005
    )
    proceed = bool(
        parity["passed"]
        and regional_zoom_causal
        and (redistribution_helpful or primary_efficient)
    )
    if not parity["passed"]:
        action = "STOP: reproduce D1/D2 anchors before interpreting any dynamic route."
    elif not regional_zoom_causal:
        action = "STOP this route: the previously observed regional-zoom gain did not reproduce."
    elif proceed:
        action = (
            "Proceed to E15a: train a lightweight correction-utility router on the train split, "
            "freeze its crop threshold there, then evaluate once on validation. Do not alter E1."
        )
    else:
        action = (
            "Keep regional zoom as a system upper bound, but do not train a router yet. "
            "The current prediction proxy did not justify dynamic allocation; redesign its "
            "utility target from matched FN recovery minus new FP and compute cost."
        )
    return {
        "reference_reproduction": parity,
        "combined_minus_full": gains,
        "fixed_top2_causal_gate": {
            "requirements": {
                "mAP50_95_gain": 0.020,
                "AP_small_gain": 0.030,
                "AR_small_gain": 0.035,
            },
            "passed": regional_zoom_causal,
        },
        "same_mean_budget_test": {
            "comparison": f"{SAME_BUDGET_MODE} minus fixed_top1",
            "delta": same_budget_delta,
            "requirements": {
                "mAP50_95": 0.001,
                "AP_small": 0.002,
                "AR_small": -0.002,
            },
            "passed": redistribution_helpful,
        },
        "primary_efficiency_test": {
            "mode": PRIMARY_BUDGET_MODE,
            "fraction_of_fixed_top2_gain": retained_top2,
            "requirements": {
                "max_crops_per_image": 1.5,
                "mAP50_95_gain": 0.020,
                "AP_small_gain": 0.030,
                "AR_small_gain": 0.030,
                "retain_mAP_and_AP_small_fraction": 0.85,
                "AP_medium_min_delta": -0.005,
            },
            "passed": primary_efficient,
        },
        "proceed_to_e15a_trainable_utility_router": proceed,
        "recommended_next_action": action,
        "claim_boundary": (
            "D15a is a decision/efficiency diagnostic. It is not itself the paper innovation."
        ),
    }


def route_profile(
    full_profile: dict,
    tile_profile: dict,
    compose_profile: dict,
    image_count: int,
) -> dict:
    local_count = compose_profile["local_forward_images"]
    fraction_of_cached_tiles = local_count / max(1, tile_profile["forward_images"])
    estimated_local = tile_profile["seconds"] * fraction_of_cached_tiles
    return {
        **compose_profile,
        "seconds": full_profile["seconds"] + estimated_local + compose_profile["merge_seconds"],
        "estimated_full_seconds": full_profile["seconds"],
        "estimated_local_seconds_from_cache_rate": estimated_local,
        "timing_is_estimate": True,
        "timing_note": (
            "The diagnostic inferred four tiles per image once. Deployment time is estimated "
            "from equivalent forward-image count; batching/IO can change real latency."
        ),
        "forward_images": image_count + local_count,
        "forward_image_multiplier": (image_count + local_count) / image_count,
        "diagnostic_actual_forward_images": (
            full_profile["forward_images"] + tile_profile["forward_images"]
        ),
        "peak_memory": {
            key: max(full_profile["peak_memory"][key], tile_profile["peak_memory"][key])
            for key in ("allocated_GiB", "reserved_GiB")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--smoke-images", type=int, default=0, metavar="N",
        help="Pipeline test on the first N images; its metrics cannot drive decisions.",
    )
    args = parser.parse_args()
    if args.smoke_images < 0 or args.smoke_images > 548:
        parser.error("--smoke-images must be between 0 and 548")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = "CHECK_D15A" if args.check_only else "SMOKE_D15A" if args.smoke_images else "DIAG_D15A"
    out = REPORTS / f"{prefix}_e1_dynamic_zoom_budget_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "initializing",
        "purpose": "frozen E1 dynamic regional-zoom precision-compute decision diagnostic",
        "script_revision": SCRIPT_REVISION,
        "started_at": now(),
        "report_dir": str(out),
        "is_full_diagnostic": not args.check_only and not args.smoke_images,
        "protocol": {
            "weight": str(WEIGHT),
            "data": str(core.DATA),
            "imgsz": core.IMGSZ,
            "grid": "2x2 with 20% overlap",
            "router": {
                "confidence_floor": ROUTER_CONF,
                "effective_small_side_px": ROUTER_SMALL_EFFECTIVE_SIDE,
                "uncertainty_weight": ROUTER_UNCERTAINTY_WEIGHT,
                "uses_ground_truth": False,
            },
            "dynamic_budgets_crops_per_image": list(DYNAMIC_BUDGETS),
            "primary_mode_predeclared": PRIMARY_BUDGET_MODE,
            "prediction_json_saved": False,
            "training_performed": False,
            "checkpoint_updates": False,
            "automatic_shutdown": False,
            "notes": [
                "All four local crops are inferred once, then reused by every policy.",
                "Fixed Top-1, Top-2 and all-four policies are strict reproduction anchors.",
                "Dynamic allocation ranks the first two marginal proxy scores globally.",
                "No AP or label is used to set a crop-count threshold.",
                "Validation crop-count calibration is diagnostic; final thresholds belong on train.",
                "Metrics use converted YOLO labels, not official VisDrone ignored-region evaluation.",
            ],
        },
        "models": {"e1": {"modes": {}}},
    }
    save(report)
    started = time.perf_counter()
    success = False
    model = None
    try:
        if core.SCRIPT_REVISION != EXPECTED_CORE_REVISION:
            raise RuntimeError(f"Unexpected slicing core revision: {core.SCRIPT_REVISION}")
        core.preflight(report, ["e1"], args.check_only)
        report["references"] = load_references()
        images, path_ids, gt, names = core.prepare_dataset(report, args.smoke_images)
        model = core.YOLO(str(WEIGHT))
        before_hash = core.sha256(WEIGHT)
        report["models"]["e1"].update({
            "weight": str(WEIGHT),
            "sha256": before_hash,
            "structure": core.assert_model(model, names, "e1"),
        })
        report["status"] = "preflight_passed"
        save(report)
        if args.check_only:
            report.update(status="check_passed", finished_at=now())
            success = True
            print("D15a preflight passed. No prediction, training, or shutdown.", flush=True)
            return

        full, full_profile = core.predict_full(model, images, report, "D15a E1")
        full_metrics = evaluate_compact("full", full, images, path_ids, gt, names)
        full_metrics["profile"] = {
            **full_profile,
            "local_forward_images": 0,
            "local_crops_per_image": 0.0,
            "forward_image_multiplier": 1.0,
            "diagnostic_actual_forward_images": len(images),
            "timing_is_estimate": False,
        }
        report["models"]["e1"]["modes"]["full"] = full_metrics
        save(report)

        score_rows = build_score_rows(images, full, image_meta(gt))
        routes, routing_metadata = build_routes(score_rows)
        audit = {
            "script_revision": SCRIPT_REVISION,
            "uses_ground_truth": False,
            "route_metadata": routing_metadata,
            "per_image": score_rows,
        }
        core.atomic_json(out / "routing_audit.json", audit)
        report["routing"] = {
            "metadata": routing_metadata,
            "audit_path": str(out / "routing_audit.json"),
        }
        save(report)

        report["status"] = "caching_local_predictions"
        save(report)
        tile_cache, tile_profile = predict_all_tiles(model, images, report)
        report["diagnostic_cache_profile"] = tile_profile
        save(report)

        route_order = [
            "fixed_top1", "fixed_top2", "fixed_top4",
            *(budget_name(value) for value in DYNAMIC_BUDGETS),
        ]
        for route_name in route_order:
            report["status"] = f"evaluating_{route_name}"
            save(report)
            predictions, compose_profile = compose_route(
                images, full, tile_cache, routes[route_name]
            )
            metrics = evaluate_compact(
                route_name, predictions, images, path_ids, gt, names
            )
            metrics["profile"] = route_profile(
                full_profile, tile_profile, compose_profile, len(images)
            )
            report["models"]["e1"]["modes"][route_name] = metrics
            del predictions
            gc.collect()
            save(report)

        after_hash = core.sha256(WEIGHT)
        if after_hash != before_hash:
            raise RuntimeError("Frozen E1 checkpoint changed during D15a")
        report["models"]["e1"]["checkpoint_unchanged"] = True
        if args.smoke_images:
            report["decisions"] = {
                "disabled": "Smoke-subset metrics must not drive research decisions."
            }
        else:
            report["decisions"] = build_decisions(report)
        report.update(
            status="completed",
            finished_at=now(),
            total_seconds=time.perf_counter() - started,
        )
        core.save_csv(report)
        shutil.copy2(Path(__file__), out / Path(__file__).name)
        save(report)
        success = True
        print(f"D15a completed. Report: {out / 'metrics.txt'}", flush=True)
    except BaseException:
        report.update(
            status="failed_or_interrupted",
            finished_at=now(),
            total_seconds=time.perf_counter() - started,
            error=traceback.format_exc(),
        )
        save(report)
        core.atomic_text(out / "error.log", report["error"])
        print(f"D15a failure saved: {out}", file=sys.stderr, flush=True)
        raise
    finally:
        if model is not None:
            del model
        core.clear_gpu()
        try:
            core.atomic_json(out / "shutdown_status.json", {
                "time": now(),
                "enabled": False,
                "status": "disabled_by_user_policy",
                "reason": report.get("status", "unknown"),
            })
            save(report)
        except Exception:
            if success:
                raise
        print("Automatic shutdown is disabled; the instance remains running.", flush=True)


if __name__ == "__main__":
    main()
